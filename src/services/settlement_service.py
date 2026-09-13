"""
Settlement: turning a pending prediction_snapshots row into a graded
prediction_outcomes row, once (and only once) the real game is over and
the actual result is known.

Reuses BasketballDataProvider.get_player_game_logs (Step 6/7) to find
the completed game's actual points, and
BasketballDataProvider.get_todays_scoreboard (Step 7) -- called with an
explicit past date, which that method already supports -- to confirm a
game has actually finished before treating a missing gamelog row as a
DNP rather than "hasn't happened yet". No provider interface changes
were needed; both capabilities already existed.

Identity discipline: a snapshot is settled ONLY when it carries BOTH
player_id and game_id. If either is missing, the outcome is recorded as
UNAVAILABLE with an explicit reason -- never guessed from player name
or date-text matching (unlike the pre-existing Google-Sheets-based
results pipeline in src/results_pipeline.py, which does exactly that;
this module is a deliberately stricter, separate system -- see the
Step 9 report).
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.data.basketball.errors import ProviderError
from src.data.basketball.provider import get_basketball_provider
from src.services.prediction_repository import (
    get_outcome_for_snapshot,
    get_pending_snapshots,
    insert_outcome,
)

logger = logging.getLogger(__name__)

SOURCE = "settlement_service"


def _is_final_status(status_text) -> bool:
    return bool(status_text) and "final" in str(status_text).lower()


def _is_postponed_or_canceled(status_text) -> bool:
    if not status_text:
        return False
    text = str(status_text).lower()
    return any(token in text for token in ("ppd", "postponed", "cancel"))


def _season_for_date(game_date_str: str) -> str:
    """nba_api season-string convention ("YYYY-YY") from a
    "%m/%d/%Y"-formatted game date (the same format
    BasketballDataProvider.get_todays_scoreboard uses -- see
    src/data/basketball/models.py::ScheduledGame.game_date)."""
    dt = datetime.strptime(game_date_str, "%m/%d/%Y")  # noqa: DTZ007 -- a calendar date, not a timestamp; tz-naive by design
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def grade_points_prediction(
    direction: str, actual_points: float, sportsbook_line: float
) -> str:
    """
    OVER:  actual > line -> WIN, actual < line -> LOSS, actual == line -> PUSH
    UNDER: actual < line -> WIN, actual > line -> LOSS, actual == line -> PUSH
    NEUTRAL: NO_ACTION (never graded win/loss/push)
    """
    if direction == "NEUTRAL":
        return "NO_ACTION"
    if actual_points == sportsbook_line:
        return "PUSH"
    if direction == "OVER":
        return "WIN" if actual_points > sportsbook_line else "LOSS"
    if direction == "UNDER":
        return "WIN" if actual_points < sportsbook_line else "LOSS"
    raise ValueError(f"Unknown prediction direction: {direction!r}")


def settle_prediction(conn, snapshot: dict, *, provider=None):
    """
    Attempts to settle ONE pending snapshot (a dict as returned by
    get_pending_snapshots/get_snapshots_for_run). Returns the outcome
    dict if one was written (or already existed), or None if the game
    isn't final yet and the snapshot should remain PENDING (no outcome
    row is written for "still pending" -- a snapshot with no outcome
    row IS the PENDING state; see get_pending_snapshots).

    Never raises for expected, structural cases (postponed/canceled
    game, DNP, missing identity) -- each produces an explicit outcome
    status instead. Idempotent: if this snapshot already has an outcome,
    that existing row is returned unchanged.
    """
    provider = provider or get_basketball_provider()

    existing = get_outcome_for_snapshot(conn, snapshot["id"])
    if existing is not None:
        return existing

    direction = snapshot.get("direction")
    if direction is None or direction == "NEUTRAL":
        return insert_outcome(
            conn,
            prediction_snapshot_id=snapshot["id"],
            actual_points=None,
            result_status="NO_ACTION",
            game_status=snapshot.get("game_status"),
            source=SOURCE,
        )

    player_id = snapshot.get("player_id")
    game_id = snapshot.get("game_id")
    game_date = snapshot.get("game_date")
    if not player_id or not game_id:
        return insert_outcome(
            conn,
            prediction_snapshot_id=snapshot["id"],
            actual_points=None,
            result_status="UNAVAILABLE",
            game_status=snapshot.get("game_status"),
            source=SOURCE,
        )

    game_status_text = None
    if game_date:
        try:
            games = provider.get_todays_scoreboard(game_date)
        except ProviderError as exc:
            logger.debug("Scoreboard lookup failed for %s: %s", game_date, exc)
            return None  # transient -- leave PENDING, try again next run
        matched_game = next((g for g in games if g.game_id == game_id), None)
        if matched_game is not None:
            game_status_text = matched_game.game_status_text

    if _is_postponed_or_canceled(game_status_text):
        return insert_outcome(
            conn,
            prediction_snapshot_id=snapshot["id"],
            actual_points=None,
            result_status="NO_ACTION",
            game_status=game_status_text,
            source=SOURCE,
        )

    if game_status_text and not _is_final_status(game_status_text):
        return None  # game hasn't finished yet -- remains PENDING

    if not game_date:
        # No game_date captured at prediction time -- we cannot even
        # check whether the game is over. Do not guess; report it.
        return insert_outcome(
            conn,
            prediction_snapshot_id=snapshot["id"],
            actual_points=None,
            result_status="UNAVAILABLE",
            game_status=game_status_text,
            source=SOURCE,
        )

    try:
        season = _season_for_date(game_date)
        logs = provider.get_player_game_logs(player_id, season)
    except (ProviderError, ValueError) as exc:
        logger.debug("Gamelog lookup failed for player_id=%s: %s", player_id, exc)
        return None  # transient/unexpected -- leave PENDING, try again next run

    matched_log = next((g for g in logs if g.game_id == game_id), None)

    if matched_log is None:
        if game_status_text and _is_final_status(game_status_text):
            # Game is over, but no box-score row for this player in it:
            # they did not play (DNP/inactive).
            return insert_outcome(
                conn,
                prediction_snapshot_id=snapshot["id"],
                actual_points=None,
                result_status="NO_ACTION",
                game_status=game_status_text,
                source=SOURCE,
            )
        return None  # game status unknown/not confirmed final -- remains PENDING

    actual_points = matched_log.points
    sportsbook_line = snapshot.get("sportsbook_line")
    if actual_points is None or sportsbook_line is None:
        return insert_outcome(
            conn,
            prediction_snapshot_id=snapshot["id"],
            actual_points=actual_points,
            result_status="UNAVAILABLE",
            game_status=game_status_text,
            source=SOURCE,
        )

    result_status = grade_points_prediction(
        direction, float(actual_points), float(sportsbook_line)
    )
    return insert_outcome(
        conn,
        prediction_snapshot_id=snapshot["id"],
        actual_points=actual_points,
        result_status=result_status,
        game_status=game_status_text or "Final",
        source=SOURCE,
    )


def settle_pending_predictions(conn, *, provider=None, limit=None) -> dict:
    """
    Settles as many pending snapshots as it safely can, one at a time.
    One snapshot's failure never stops the others (caught, logged,
    counted, loop continues). Safe to call repeatedly -- already-settled
    snapshots are never revisited (excluded by get_pending_snapshots
    once they have an outcome row), and re-running while some games are
    still in progress simply leaves those snapshots pending again.
    """
    provider = provider or get_basketball_provider()
    pending = get_pending_snapshots(conn, limit=limit)

    settled = 0
    still_pending = 0
    no_action = 0
    unavailable = 0
    errors = 0

    for snapshot in pending:
        try:
            outcome = settle_prediction(conn, snapshot, provider=provider)
        except Exception as exc:  # noqa: BLE001 -- one bad snapshot must never abort the batch
            errors += 1
            logger.warning(
                "Settlement failed for snapshot_id=%s: %s: %s",
                snapshot.get("id"),
                type(exc).__name__,
                exc,
            )
            continue

        if outcome is None:
            still_pending += 1
        elif outcome["result_status"] == "NO_ACTION":
            no_action += 1
        elif outcome["result_status"] == "UNAVAILABLE":
            unavailable += 1
        else:
            settled += 1

    return {
        "pending_checked": len(pending),
        "settled": settled,
        "still_pending": still_pending,
        "no_action": no_action,
        "unavailable": unavailable,
        "errors": errors,
    }
