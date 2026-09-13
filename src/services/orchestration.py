"""
Step 10: production orchestration for the two unattended jobs the
prediction-history system needs -- generating/persisting today's board
(run_prediction_cycle) and settling pending predictions
(run_settlement_cycle). This module owns NO prediction math and NO
grading rules; it only sequences existing Step 8/9 building blocks
(build_daily_prediction_board, persist_prediction_run,
settle_pending_predictions) and adds the operational judgment calls a
human running the manual scripts used to make by hand:

  - is the schema even migrated? (fail fast, don't let a raw SQL error
    surface from inside persist_prediction_run)
  - did the Odds API call / model load actually fail, or is there
    genuinely nothing to do today? (build_daily_prediction_board
    deliberately conflates both into "return an empty board" -- see its
    docstring -- which is correct for the UI's resilience but wrong for
    automation, which must tell FAILED apart from NO_WORK)
  - is each individual player's game still pregame? (the Edge Board is
    pregame-only; automation must not persist a "pregame" snapshot for a
    game that has already tipped off, without shutting down every OTHER
    still-pregame game that night)

See CycleStatus below for the status vocabulary, and the Step 10 report
for the full lifecycle write-up.
"""

from __future__ import annotations

import dataclasses
import re
import time
from datetime import datetime, timezone
from enum import Enum
from zoneinfo import ZoneInfo

from src.services.migrations import is_schema_ready
from src.services.prediction_repository import (
    QUALIFIED_EDGE_THRESHOLD_DEFAULT,
    persist_prediction_run,
)
from src.services.prediction_result import PredictionStatus
from src.services.settlement_service import (
    _is_final_status,
    _is_postponed_or_canceled,
    settle_pending_predictions,
)

DEFAULT_TIMEZONE = (
    "America/Chicago"  # matches scripts/pregame_pipeline.py's existing convention
)

_DSN_CREDENTIALS_PATTERN = re.compile(r"://[^/\s@]+:[^/\s@]+@")


def _redact_secrets(text) -> str:
    """
    Defensively strips DATABASE_URL (if configured) and any generic
    scheme://user:password@ credential pattern out of a string before
    it is ever put into a PredictionCycleResult/SettlementCycleResult
    message -- see Step 10 Phase 7 ("do not log API keys, DATABASE_URL
    credentials, secret values"). Every exception surfaced by this
    module is passed through this first. Testing against the real
    psycopg driver (Step 10's Phase 0 validation) showed its own
    connection-failure messages do not echo the DSN, but this does not
    rely on that staying true for psycopg or any future driver.
    """
    text = str(text)
    import os

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        text = text.replace(database_url, "[REDACTED]")
    return _DSN_CREDENTIALS_PATTERN.sub("://[REDACTED]@", text)


class CycleStatus(str, Enum):
    """
    The four states a scheduler needs to be able to tell apart (Step 10
    Phase 8). Exit-code mapping (see scripts/persist_prediction_board.py
    and scripts/settle_predictions.py): SUCCESS/NO_WORK/PARTIAL -> 0,
    FAILED -> 1. NO_WORK and PARTIAL are both "the job ran correctly" --
    only FAILED should page anyone.
    """

    SUCCESS = "SUCCESS"
    NO_WORK = "NO_WORK"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Per-game status classification (Phase 4: pregame cutoff safety)
# ---------------------------------------------------------------------------

_PREGAME_CLOCK_PATTERN = re.compile(
    r"\d{1,2}:\d{2}\s*(a\.?m\.?|p\.?m\.?)\s*et", re.IGNORECASE
)


def classify_game_status(status_text) -> str:
    """
    Classifies a ScheduledGame.game_status_text (equivalently,
    PredictionResult.game_status) into one of PREGAME, LIVE, FINAL,
    POSTPONED_CANCELED, UNKNOWN.

    Reuses settlement_service's existing, already-tested FINAL/
    POSTPONED-CANCELED text classification rather than duplicating that
    keyword logic. PREGAME is detected from the NBA stats API's own
    scheduled-tipoff text convention -- a clock time ending in "ET"
    (e.g. "7:30 pm ET"; always Eastern Time regardless of viewer
    timezone, confirmed via a live provider call during Step 10's
    validation). Anything else (a live quarter/clock string, "Halftime",
    etc.) is classified LIVE rather than guessed as PREGAME --
    deliberately conservative, since misclassifying a live game as
    pregame would risk persisting a stale "pregame" snapshot for a game
    that has already started. UNKNOWN (empty/missing status text) is
    treated identically to LIVE by every caller in this module: NOT
    eligible for a new pregame snapshot, since pregame state cannot be
    confirmed.
    """
    if not status_text:
        return "UNKNOWN"
    if _is_postponed_or_canceled(status_text):
        return "POSTPONED_CANCELED"
    if _is_final_status(status_text):
        return "FINAL"
    if _PREGAME_CLOCK_PATTERN.search(str(status_text)):
        return "PREGAME"
    return "LIVE"


def _is_eligible_for_persistence(result) -> bool:
    """Non-OK predictions (unmatched/unavailable/etc.) are always kept
    -- Step 9 already records those for run-health tracking, and their
    status has nothing to do with game timing. An OK prediction is kept
    only if its own game is still PREGAME."""
    if result.status != PredictionStatus.OK:
        return True
    return classify_game_status(result.game_status) == "PREGAME"


# ---------------------------------------------------------------------------
# Prediction cycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PredictionCycleResult:
    status: CycleStatus
    message: str
    reason_code: str | None = None
    props_discovered: int = 0
    predictions_generated: int = 0
    persisted_count: int = 0
    excluded_live_count: int = 0
    excluded_final_count: int = 0
    excluded_postponed_or_canceled_count: int = 0
    unmatched_count: int = 0
    unavailable_count: int = 0
    provider_calls: int = 0
    run_id: int | None = None
    run_created: bool | None = None
    duration_seconds: float = 0.0


def _fetch_props_with_call_count(props_fetcher, api_key, bookmaker_key):
    """Wraps whatever the real fetcher is (default:
    shared_app.fetch_all_today_player_props) so ONE cycle can report how
    many real Odds-API HTTP calls it made (Step 10's quota-telemetry
    requirement), without modifying shared_app.py's fetch code at all --
    it patches only `requests.get` as seen through src.shared_app's own
    module reference, for the duration of this one call."""
    from unittest import mock

    from src import shared_app as shared_app_module

    call_count = {"n": 0}
    real_get = shared_app_module.requests.get

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return real_get(*args, **kwargs)

    with mock.patch.object(shared_app_module.requests, "get", counting_get):
        df = props_fetcher(api_key, bookmaker_key)
    return df, call_count["n"]


def _fetch_props_with_retry(
    props_fetcher,
    api_key,
    bookmaker_key,
    *,
    max_attempts=2,
    retry_delay_seconds=3,
    sleep_func=time.sleep,
):
    """
    The ONE new retry Step 10 adds (Phase 9): a single bounded retry
    around the top-level props fetch, because
    shared_app.fetch_all_today_player_props raises straight through on a
    failed /events call (unlike its own per-event odds calls, which
    already skip-and-continue on failure -- see that function's
    docstring). Deliberately narrow: this does not wrap
    build_daily_prediction_board, the database, or settlement -- see the
    Step 10 report's retry-policy section for why each of those is
    handled differently (fail fast, or already safe-to-rerun via the
    next scheduled cycle).

    Returns (df, call_count, error). error is None on success.
    """
    last_error = None
    total_calls = 0
    for attempt in range(1, max_attempts + 1):
        try:
            df, calls = _fetch_props_with_call_count(
                props_fetcher, api_key, bookmaker_key
            )
            return df, total_calls + calls, None
        except Exception as exc:  # noqa: BLE001 -- classified by the caller, not raised
            last_error = exc
            total_calls += 1  # the failed attempt still made at least one real call
            if attempt < max_attempts:
                sleep_func(retry_delay_seconds)
    return None, total_calls, last_error


def run_prediction_cycle(
    *,
    api_key=None,
    bookmaker_key=None,
    provider=None,
    model=None,
    conn=None,
    props_fetcher=None,
    game_date=None,
    edge_threshold: float = QUALIFIED_EDGE_THRESHOLD_DEFAULT,
    max_fetch_attempts: int = 2,
    retry_delay_seconds: int = 3,
    sleep_func=time.sleep,
) -> PredictionCycleResult:
    """
    The production entry point for "generate and persist today's
    board" -- what scripts/persist_prediction_board.py calls, and what
    a scheduled workflow runs. Reuses build_daily_prediction_board() and
    persist_prediction_run() unchanged; adds only: schema-readiness
    pre-check, FAILED-vs-NO_WORK disambiguation for the props fetch and
    model load, per-game pregame-cutoff filtering before persistence,
    and structured, classified results.

    `conn`, `provider`, `model`, `props_fetcher` are all injectable so
    this can be tested deterministically with no live network (see
    tests/test_orchestration.py) -- production callers (the scripts)
    leave them as None and get the real database/provider/model/odds
    fetch.
    """
    start = time.monotonic()
    import os

    from src import shared_app
    from src.data.basketball.provider import get_basketball_provider
    from src.services.prediction_service import build_daily_prediction_board

    api_key = api_key if api_key is not None else os.environ.get("ODDS_API_KEY")
    bookmaker_key = bookmaker_key or shared_app.BOOKMAKER_KEY
    provider = provider or get_basketball_provider()
    real_props_fetcher = props_fetcher or shared_app.fetch_all_today_player_props

    if not api_key:
        return PredictionCycleResult(
            status=CycleStatus.FAILED,
            message="ODDS_API_KEY is not configured.",
            reason_code="MISSING_ODDS_API_KEY",
            duration_seconds=time.monotonic() - start,
        )

    owns_conn = conn is None
    if conn is None:
        try:
            from src.services.db_connection import get_prediction_db_connection

            conn = get_prediction_db_connection()
        except Exception as exc:  # noqa: BLE001 -- classified, not raised
            return PredictionCycleResult(
                status=CycleStatus.FAILED,
                message=f"Database unavailable: {_redact_secrets(exc)}",
                reason_code="DATABASE_UNAVAILABLE",
                duration_seconds=time.monotonic() - start,
            )

    try:
        if not is_schema_ready(conn):
            return PredictionCycleResult(
                status=CycleStatus.FAILED,
                message="Prediction-history schema is not migrated. Run "
                "scripts/apply_prediction_history_migrations.py before enabling automation.",
                reason_code="SCHEMA_NOT_READY",
                duration_seconds=time.monotonic() - start,
            )

        try:
            resolved_model = model if model is not None else shared_app.load_model()
        except Exception as exc:  # noqa: BLE001 -- classified, not raised
            return PredictionCycleResult(
                status=CycleStatus.FAILED,
                message=f"Model unavailable: {_redact_secrets(exc)}",
                reason_code="MODEL_UNAVAILABLE",
                duration_seconds=time.monotonic() - start,
            )

        props_df, provider_calls, fetch_error = _fetch_props_with_retry(
            real_props_fetcher,
            api_key,
            bookmaker_key,
            max_attempts=max_fetch_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleep_func=sleep_func,
        )
        if fetch_error is not None:
            return PredictionCycleResult(
                status=CycleStatus.FAILED,
                message=f"Odds API unavailable after {max_fetch_attempts} attempt(s): {_redact_secrets(fetch_error)}",
                reason_code="ODDS_API_UNAVAILABLE",
                provider_calls=provider_calls,
                duration_seconds=time.monotonic() - start,
            )

        tz = ZoneInfo(DEFAULT_TIMEZONE)
        resolved_game_date = game_date or datetime.now(tz).strftime("%m/%d/%Y")
        try:
            todays_games = provider.get_todays_scoreboard(resolved_game_date)
        except Exception:  # noqa: BLE001 -- schedule lookup is best-effort context, never fatal
            todays_games = []

        board = build_daily_prediction_board(
            api_key=api_key,
            bookmaker_key=bookmaker_key,
            provider=provider,
            model=resolved_model,
            props_fetcher=lambda *_: props_df,
            sleep_func=sleep_func,
        )

        if board.props_discovered == 0:
            reason = "NO_GAMES" if not todays_games else "NO_PROPS_YET"
            message = (
                "No NBA games scheduled today."
                if reason == "NO_GAMES"
                else "Games are scheduled today, but the sportsbook has not posted player-points props yet."
            )
            return PredictionCycleResult(
                status=CycleStatus.NO_WORK,
                message=message,
                reason_code=reason,
                provider_calls=provider_calls,
                duration_seconds=time.monotonic() - start,
            )

        excluded_live = sum(
            1
            for r in board.predictions
            if r.status == PredictionStatus.OK
            and classify_game_status(r.game_status) == "LIVE"
        )
        excluded_final = sum(
            1
            for r in board.predictions
            if r.status == PredictionStatus.OK
            and classify_game_status(r.game_status) == "FINAL"
        )
        excluded_postponed = sum(
            1
            for r in board.predictions
            if r.status == PredictionStatus.OK
            and classify_game_status(r.game_status) == "POSTPONED_CANCELED"
        )
        eligible = tuple(
            r for r in board.predictions if _is_eligible_for_persistence(r)
        )
        eligible_ok_count = sum(1 for r in eligible if r.status == PredictionStatus.OK)

        if eligible_ok_count == 0 and board.predictions_generated > 0:
            # Every OK prediction existed but was filtered out purely for
            # game-timing reasons (already live/final/postponed) -- a
            # timing artifact, not a failure. Persisting a run here would
            # write a misleading zero-prediction row. Report NO_WORK
            # instead of touching the database at all.
            return PredictionCycleResult(
                status=CycleStatus.NO_WORK,
                message="All of today's pregame-eligible games have already started or finished; "
                "nothing new to persist this cycle.",
                reason_code="ALL_GAMES_LIVE_OR_FINAL",
                props_discovered=board.props_discovered,
                predictions_generated=0,
                excluded_live_count=excluded_live,
                excluded_final_count=excluded_final,
                excluded_postponed_or_canceled_count=excluded_postponed,
                unmatched_count=board.unmatched_count,
                unavailable_count=board.unavailable_count,
                provider_calls=provider_calls,
                duration_seconds=time.monotonic() - start,
            )

        persistable_board = dataclasses.replace(
            board,
            predictions=eligible,
            predictions_generated=eligible_ok_count,
        )

        try:
            persist_result = persist_prediction_run(
                persistable_board, conn, edge_threshold=edge_threshold
            )
        except Exception as exc:  # noqa: BLE001 -- classified, not raised
            return PredictionCycleResult(
                status=CycleStatus.FAILED,
                message=f"Failed to persist prediction run: {_redact_secrets(exc)}",
                reason_code="PERSIST_FAILED",
                props_discovered=board.props_discovered,
                predictions_generated=eligible_ok_count,
                provider_calls=provider_calls,
                duration_seconds=time.monotonic() - start,
            )

        has_issues = board.unmatched_count > 0 or board.unavailable_count > 0
        status = CycleStatus.PARTIAL if has_issues else CycleStatus.SUCCESS
        message = f"Persisted {eligible_ok_count} pregame prediction(s)" + (
            " with some unmatched/unavailable players." if has_issues else "."
        )

        return PredictionCycleResult(
            status=status,
            message=message,
            reason_code=None,
            props_discovered=board.props_discovered,
            predictions_generated=eligible_ok_count,
            persisted_count=len(eligible),
            excluded_live_count=excluded_live,
            excluded_final_count=excluded_final,
            excluded_postponed_or_canceled_count=excluded_postponed,
            unmatched_count=board.unmatched_count,
            unavailable_count=board.unavailable_count,
            provider_calls=provider_calls,
            run_id=persist_result["run_id"],
            run_created=persist_result["created"],
            duration_seconds=time.monotonic() - start,
        )
    finally:
        if owns_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Settlement cycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SettlementCycleResult:
    status: CycleStatus
    message: str
    reason_code: str | None = None
    pending_checked: int = 0
    settled: int = 0
    still_pending: int = 0
    no_action: int = 0
    unavailable: int = 0
    errors: int = 0
    scoreboard_calls: int = 0
    gamelog_calls: int = 0
    duration_seconds: float = 0.0


class _CachingSettlementProvider:
    """
    Wraps a real BasketballDataProvider so ONE settlement cycle reuses
    scoreboard/gamelog responses across snapshots that share a game_date
    or (player_id, season) -- Step 10 Phase 5's "group/dedupe external
    requests where appropriate". Without this, N pending snapshots for
    the same game_date each trigger their own get_todays_scoreboard call
    inside settlement_service.settle_prediction (which settles one
    snapshot at a time and has no cross-snapshot memory of its own by
    design -- see that module's docstring). Caches only for the lifetime
    of one run_settlement_cycle() call; never used across cycles.
    """

    def __init__(self, provider):
        self._provider = provider
        self._scoreboard_cache = {}
        self._gamelog_cache = {}
        self.scoreboard_calls = 0
        self.gamelog_calls = 0

    def get_todays_scoreboard(self, game_date=None):
        if game_date not in self._scoreboard_cache:
            self.scoreboard_calls += 1
            self._scoreboard_cache[game_date] = self._provider.get_todays_scoreboard(
                game_date
            )
        return self._scoreboard_cache[game_date]

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        key = (player_id, season)
        if key not in self._gamelog_cache:
            self.gamelog_calls += 1
            self._gamelog_cache[key] = self._provider.get_player_game_logs(
                player_id, season, player_name=player_name
            )
        return self._gamelog_cache[key]


def run_settlement_cycle(
    *, conn=None, provider=None, limit=None
) -> SettlementCycleResult:
    """
    The production entry point for "settle whatever can be settled
    right now" -- what scripts/settle_predictions.py calls, and what a
    scheduled workflow runs. Reuses settle_pending_predictions()
    unchanged; adds only schema-readiness pre-checking, request
    deduping via _CachingSettlementProvider, and status classification.

    Safe to run repeatedly at any cadence -- settle_pending_predictions
    is itself idempotent (see Step 9), and a snapshot left PENDING here
    is simply picked up again on the next scheduled run.
    """
    start = time.monotonic()

    owns_conn = conn is None
    if conn is None:
        try:
            from src.services.db_connection import get_prediction_db_connection

            conn = get_prediction_db_connection()
        except Exception as exc:  # noqa: BLE001 -- classified, not raised
            return SettlementCycleResult(
                status=CycleStatus.FAILED,
                message=f"Database unavailable: {_redact_secrets(exc)}",
                reason_code="DATABASE_UNAVAILABLE",
                duration_seconds=time.monotonic() - start,
            )

    try:
        if not is_schema_ready(conn):
            return SettlementCycleResult(
                status=CycleStatus.FAILED,
                message="Prediction-history schema is not migrated. Run "
                "scripts/apply_prediction_history_migrations.py before enabling automation.",
                reason_code="SCHEMA_NOT_READY",
                duration_seconds=time.monotonic() - start,
            )

        if provider is None:
            from src.data.basketball.provider import get_basketball_provider

            provider = get_basketball_provider()
        caching_provider = _CachingSettlementProvider(provider)

        try:
            counts = settle_pending_predictions(
                conn, provider=caching_provider, limit=limit
            )
        except Exception as exc:  # noqa: BLE001 -- classified, not raised
            return SettlementCycleResult(
                status=CycleStatus.FAILED,
                message=f"Settlement cycle failed: {_redact_secrets(exc)}",
                reason_code="SETTLEMENT_FAILED",
                duration_seconds=time.monotonic() - start,
            )

        if counts["pending_checked"] == 0:
            return SettlementCycleResult(
                status=CycleStatus.NO_WORK,
                message="No pending predictions to settle right now.",
                reason_code="NO_PENDING_PREDICTIONS",
                scoreboard_calls=caching_provider.scoreboard_calls,
                gamelog_calls=caching_provider.gamelog_calls,
                duration_seconds=time.monotonic() - start,
            )

        status = CycleStatus.PARTIAL if counts["errors"] > 0 else CycleStatus.SUCCESS
        message = (
            f"Settled {counts['settled']}, still pending {counts['still_pending']}, "
            f"no_action {counts['no_action']}, unavailable {counts['unavailable']}, "
            f"errors {counts['errors']}."
        )

        return SettlementCycleResult(
            status=status,
            message=message,
            reason_code=None,
            pending_checked=counts["pending_checked"],
            settled=counts["settled"],
            still_pending=counts["still_pending"],
            no_action=counts["no_action"],
            unavailable=counts["unavailable"],
            errors=counts["errors"],
            scoreboard_calls=caching_provider.scoreboard_calls,
            gamelog_calls=caching_provider.gamelog_calls,
            duration_seconds=time.monotonic() - start,
        )
    finally:
        if owns_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Freshness classification (Phase 13 -- used by the UI, not automation)
# ---------------------------------------------------------------------------


class BoardFreshness(str, Enum):
    FRESH = "FRESH"
    WAITING_FOR_PROPS = "WAITING_FOR_PROPS"
    NO_GAMES = "NO_GAMES"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


def compute_board_freshness(
    *,
    latest_run: dict | None,
    games_today: bool,
    now_utc: datetime | None = None,
    stale_after_hours: float = 6.0,
) -> tuple:
    """
    Classifies today's data freshness for the UI (Phase 13) -- never
    treats "no games today" as a failure. `latest_run` is the dict
    returned by prediction_repository.get_latest_run (or None if
    nothing has ever been persisted). Returns (BoardFreshness, message).

    This is read-only, UI-facing classification -- it never triggers
    automation and never appears in scheduler exit codes.
    """
    now_utc = now_utc or datetime.now(timezone.utc)

    if not games_today:
        return BoardFreshness.NO_GAMES, "No NBA games are scheduled today."

    if latest_run is None:
        return (
            BoardFreshness.WAITING_FOR_PROPS,
            "Games are scheduled today; waiting for the first board of the day.",
        )

    generated_at = latest_run.get("generated_at_utc")
    try:
        generated_dt = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if generated_dt.tzinfo is None:
            generated_dt = generated_dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return (
            BoardFreshness.UNKNOWN,
            "Could not determine when the latest board was generated.",
        )

    age_hours = (now_utc - generated_dt).total_seconds() / 3600.0
    if age_hours <= stale_after_hours:
        return (
            BoardFreshness.FRESH,
            f"Latest board generated {age_hours:.1f} hour(s) ago.",
        )
    return (
        BoardFreshness.STALE,
        f"Latest board is {age_hours:.1f} hour(s) old -- automation may not be running.",
    )
