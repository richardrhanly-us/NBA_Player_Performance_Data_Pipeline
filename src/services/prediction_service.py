"""
Product-level prediction services: one function that predicts a single
player (predict_player), and one that builds today's full board of
points-prop predictions across the current sportsbook slate
(build_daily_prediction_board). Both return PredictionResult objects
(see prediction_result.py) -- neither is Streamlit-coupled, so both are
directly testable and usable from a future batch job.

Both reuse the SAME core logic apps/publicapp.py::build_prediction
already runs today (via src/shared_app.py's functions) -- this module
does not reimplement feature engineering, model inference, or live
adjustment; it orchestrates the existing pieces and packages the result
as a typed, product-level object instead of an ad hoc dict/DataFrame.

Pregame vs live: predict_player() applies the existing live-adjustment
logic by default (apply_live_adjustment=True), matching
apps/publicapp.py::build_prediction's current single-player behavior
exactly. build_daily_prediction_board() calls predict_player() with
apply_live_adjustment=False for every player -- this matches
src/shared_app.py::get_top_plays_today_df's EXISTING behavior (it never
called get_live_player_stats at all), not a new restriction invented by
this module. See the Step 8 report for the full rationale.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src import shared_app
from src.data.basketball.errors import ProviderError
from src.data.basketball.models import ScheduledGame
from src.data.basketball.provider import BasketballDataProvider, get_basketball_provider
from src.features.build_features import build_player_feature_row
from src.services.prediction_result import (
    PredictionResult,
    PredictionStatus,
    compute_edge_and_direction,
)

logger = logging.getLogger(__name__)

MODEL_PATH = "models/points_regression.pkl"


def _model_version_identifier(model_path: str = MODEL_PATH) -> str:
    """
    A cheap, honest "what model produced this" identifier: the artifact
    file's name and last-modified time. Deliberately NOT a content hash
    (hashing a ~20MB file per prediction would be wasteful) and
    deliberately NOT a fabricated semantic version (none exists for this
    artifact today) -- this changes if and only if the deployed file
    actually changes.
    """
    try:
        mtime = os.path.getmtime(model_path)
        mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        return f"{os.path.basename(model_path)}@{mtime_iso}"
    except OSError:
        return os.path.basename(model_path)


def _resolve_player(
    player_name: str,
    actual_name_to_id: dict,
    normalized_to_actual: dict,
    *,
    use_fuzzy_name_resolution: bool,
):
    """
    Two EXISTING, unchanged matching strategies, selected explicitly
    rather than merged into one "smarter" matcher:

    - use_fuzzy_name_resolution=False: apps/publicapp.py::build_prediction's
      exact strategy (normalize, look up, fall back to the raw input
      name if not found -- which then simply fails to resolve a
      player_id). Appropriate for the manual-search UI, where the input
      is normally chosen from an exact autocomplete list already.
    - use_fuzzy_name_resolution=True: src/shared_app.py::resolve_player_name,
      the EXISTING sportsbook-name-matching strategy
      get_top_plays_today_df already uses (exact normalized match, then
      a first+last-name fallback, then a final exact scan). Used for
      board-building, where sportsbook prop names are free text.

    Returns (actual_name, player_id) or (None, None) if unresolved.
    Never invents a third, fuzzier strategy.
    """
    if use_fuzzy_name_resolution:
        actual_name = shared_app.resolve_player_name(player_name, normalized_to_actual)
        if actual_name is None:
            return None, None
    else:
        normalized = shared_app.normalize_name(player_name)
        actual_name = normalized_to_actual.get(normalized, player_name)

    player_id = actual_name_to_id.get(actual_name)
    if not player_id:
        return None, None
    return actual_name, player_id


def _match_todays_game(
    team_id: int | None, todays_games: Sequence[ScheduledGame] | None
) -> ScheduledGame | None:
    if team_id is None or not todays_games:
        return None
    for game in todays_games:
        if game.home_team_id == team_id or game.away_team_id == team_id:
            return game
    return None


def _latest_game_date(gamelog_df: pd.DataFrame) -> str | None:
    try:
        parsed = pd.to_datetime(gamelog_df["GAME_DATE"], errors="coerce")
        latest = parsed.max()
        if pd.isna(latest):
            return None
        return str(latest.date())
    except Exception:  # noqa: BLE001 -- freshness metadata must never break a prediction
        return None


def predict_player(
    player_name: str,
    *,
    sportsbook_line: float | None = None,
    bookmaker: str | None = None,
    season: str | None = None,
    provider: BasketballDataProvider | None = None,
    model=None,
    model_version: str | None = None,
    active_players: tuple | None = None,
    gamelog_cache: dict | None = None,
    apply_live_adjustment: bool = True,
    use_fuzzy_name_resolution: bool = False,
    todays_games: Sequence[ScheduledGame] | None = None,
    matchup: str | None = None,
) -> PredictionResult:
    """
    Produces one player's PredictionResult, reusing the exact pipeline
    apps/publicapp.py::build_prediction already runs (model load,
    active-player resolution, recent gamelog, legacy feature row, model
    predict, optional live adjustment) -- see this module's docstring
    for the pregame/live default and the two name-resolution strategies.

    All of `model`/`active_players`/`gamelog_cache` are accepted so a
    batch caller (build_daily_prediction_board) can load/resolve them
    ONCE and pass them into every per-player call, rather than this
    function reaching for them itself each time. When omitted (the
    single-player/UI case), this function resolves them itself via
    src/shared_app.py's own (Streamlit-)cached loaders.
    """
    provider = provider or get_basketball_provider()
    season = season or shared_app.CURRENT_SEASON
    generated_at = datetime.now(timezone.utc).isoformat()
    resolved_model_version = model_version or _model_version_identifier()

    def _unavailable(status, *, player_id=None, reason=None) -> PredictionResult:
        return PredictionResult(
            player_name=player_name,
            player_id=player_id,
            status=status,
            sportsbook_line=sportsbook_line,
            bookmaker=bookmaker,
            matchup=matchup,
            model_version=resolved_model_version,
            generated_at_utc=generated_at,
            reason=reason,
        )

    if model is None:
        try:
            model = shared_app.load_model()
        except Exception as exc:  # noqa: BLE001 -- model-load failure is a real, reportable status
            return _unavailable(
                PredictionStatus.MODEL_UNAVAILABLE,
                reason=f"Could not load prediction model: {exc}",
            )

    if active_players is None:
        try:
            active_players = shared_app.load_active_players(_provider=provider)
        except ProviderError as exc:
            return _unavailable(
                PredictionStatus.PROVIDER_UNAVAILABLE,
                reason=str(exc),
            )
    actual_name_to_id, normalized_to_actual = active_players

    actual_name, player_id = _resolve_player(
        player_name,
        actual_name_to_id,
        normalized_to_actual,
        use_fuzzy_name_resolution=use_fuzzy_name_resolution,
    )
    if player_id is None:
        return _unavailable(
            PredictionStatus.UNMATCHED,
            reason=f"Could not confidently match '{player_name}' to an active player.",
        )

    # NOTE: shared_app.get_player_gamelog_df already catches ProviderError
    # internally (Step 7 behavior, preserved) and returns an empty
    # DataFrame rather than raising -- so a gamelog-fetch failure surfaces
    # below as MISSING_HISTORY ("Player gamelog unavailable."), exactly
    # matching pre-Step-8 build_prediction's behavior, not as
    # PROVIDER_UNAVAILABLE. The except clause here is a defensive guard
    # in case that inner function's behavior ever changes; it is not
    # expected to fire today.
    try:
        if gamelog_cache is not None and player_id in gamelog_cache:
            gamelog_df = gamelog_cache[player_id]
        else:
            gamelog_df = shared_app.get_player_gamelog_df(
                player_id, season, _provider=provider
            )
            if gamelog_cache is not None:
                gamelog_cache[player_id] = gamelog_df
    except ProviderError as exc:
        return _unavailable(
            PredictionStatus.PROVIDER_UNAVAILABLE,
            player_id=player_id,
            reason=str(exc),
        )

    if gamelog_df is None or gamelog_df.empty:
        return _unavailable(
            PredictionStatus.MISSING_HISTORY,
            player_id=player_id,
            # Exact wording match with apps/publicapp.py's pre-Step-8
            # build_prediction error message -- see the Step 8 report.
            reason="Player gamelog unavailable.",
        )

    X = build_player_feature_row(gamelog_df, actual_name, sportsbook_line)
    if X is None or X.empty:
        return _unavailable(
            PredictionStatus.MISSING_HISTORY,
            player_id=player_id,
            reason="Not enough games to build features.",
        )

    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if model_feature_names:
        missing_features = [c for c in model_feature_names if c not in X.columns]
        if missing_features:
            return _unavailable(
                PredictionStatus.ERROR,
                player_id=player_id,
                reason=f"Model feature mismatch. Missing: {', '.join(missing_features)}",
            )
        X = X.reindex(columns=model_feature_names)

    model_projection = float(model.predict(X)[0])

    if apply_live_adjustment:
        try:
            live_stats = shared_app.get_live_player_stats(
                actual_name, provider=provider
            )
        except Exception:  # noqa: BLE001 -- matches build_prediction's own live-stats fallback
            live_stats = None
        model_projection = shared_app.get_live_adjusted_projection(
            model_projection, live_stats
        )

    team_abbreviation = None
    team_id = None
    try:
        details = shared_app.get_player_details(player_id, _provider=provider)
        if details is not None:
            team_abbreviation = details.team_abbreviation
            team_id = details.team_id
    except Exception as exc:  # noqa: BLE001 -- display metadata must never break a prediction
        logger.debug(
            "Could not fetch player details for player_id=%s: %s", player_id, exc
        )

    matched_game = _match_todays_game(team_id, todays_games)
    game_id = matched_game.game_id if matched_game is not None else None
    game_date = matched_game.game_date if matched_game is not None else None
    game_status = matched_game.game_status_text if matched_game is not None else None

    edge, direction = compute_edge_and_direction(model_projection, sportsbook_line)
    status = (
        PredictionStatus.OK
        if sportsbook_line is not None
        else PredictionStatus.MISSING_LINE
    )

    return PredictionResult(
        player_name=actual_name,
        player_id=player_id,
        team_abbreviation=team_abbreviation,
        matchup=matchup,
        game_id=game_id,
        game_date=game_date,
        game_status=game_status,
        model_projection=model_projection,
        sportsbook_line=sportsbook_line,
        edge=edge,
        direction=direction,
        bookmaker=bookmaker,
        model_version=resolved_model_version,
        generated_at_utc=generated_at,
        latest_game_date=_latest_game_date(gamelog_df),
        status=status,
        reason=None
        if status == PredictionStatus.OK
        else "No sportsbook line available for this player.",
    )


@dataclass(frozen=True)
class PredictionBoard:
    """
    Today's full board: every prop the sportsbook offered (deduped by
    player), each paired with a PredictionResult -- OK, or a specific
    unavailable/unmatched status, never silently dropped. Summary counts
    are the useful, at-a-glance health metrics the Step 8 brief asked
    for ("props discovered, players matched, predictions generated,
    unavailable, unmatched").
    """

    generated_at_utc: str
    bookmaker: str | None
    model_version: str
    predictions: tuple
    props_discovered: int
    players_matched: int
    predictions_generated: int
    unmatched_count: int
    unavailable_count: int

    def qualified(self, edge_threshold: float) -> tuple:
        """Predictions whose |edge| meets `edge_threshold` -- for
        highlighting, NOT for filtering the board (see the Step 8
        report: the board shows every prediction; qualified predictions
        are only flagged, never hidden)."""
        return tuple(
            r
            for r in self.predictions
            if r.edge is not None and abs(r.edge) >= edge_threshold
        )


def _empty_board(
    generated_at_utc: str, bookmaker: str | None, model_version: str
) -> PredictionBoard:
    return PredictionBoard(
        generated_at_utc=generated_at_utc,
        bookmaker=bookmaker,
        model_version=model_version,
        predictions=(),
        props_discovered=0,
        players_matched=0,
        predictions_generated=0,
        unmatched_count=0,
        unavailable_count=0,
    )


_UNAVAILABLE_STATUSES = (
    PredictionStatus.PROVIDER_UNAVAILABLE,
    PredictionStatus.MODEL_UNAVAILABLE,
    PredictionStatus.ERROR,
)


def build_daily_prediction_board(
    api_key: str,
    *,
    bookmaker_key: str | None = None,
    provider: BasketballDataProvider | None = None,
    model=None,
    season: str | None = None,
    props_fetcher=None,
    request_delay: float = 0.5,
    sleep_func=time.sleep,
) -> PredictionBoard:
    """
    Today's full points-prop board: fetch today's props once, resolve
    the active-player roster once, load the model once, fetch today's
    schedule once, then predict_player() once per unique player --
    reusing every one of those upfront results (see predict_player's
    `model`/`active_players`/`gamelog_cache`/`todays_games` parameters)
    so a 100-player board never repeats an already-answered external
    call. Pregame-only (apply_live_adjustment=False on every call) --
    matching src/shared_app.py::get_top_plays_today_df's existing,
    unchanged behavior, not a new restriction (see this module's
    docstring).

    One bad player never aborts the board: predict_player() already
    returns a specific unavailable/unmatched PredictionResult instead of
    raising for per-player problems; only a total model-load or
    props-fetch failure produces an empty board.

    `props_fetcher` defaults to src/shared_app.py::fetch_all_today_player_props
    and is deliberately injectable so tests never need real network/API
    access. `request_delay`/`sleep_func` reproduce
    get_top_plays_today_df's existing polite pacing between players
    doing a real (cache-miss) gamelog fetch -- tests pass
    sleep_func=lambda *_: None to run instantly.
    """
    provider = provider or get_basketball_provider()
    bookmaker_key = bookmaker_key or shared_app.BOOKMAKER_KEY
    season = season or shared_app.CURRENT_SEASON
    props_fetcher = props_fetcher or shared_app.fetch_all_today_player_props
    generated_at = datetime.now(timezone.utc).isoformat()
    model_version = _model_version_identifier()

    if model is None:
        try:
            model = shared_app.load_model()
        except Exception:  # noqa: BLE001 -- no model, no board -- report empty, don't crash the UI
            return _empty_board(generated_at, bookmaker_key, model_version)

    try:
        props_df = props_fetcher(api_key, bookmaker_key)
    except Exception:  # noqa: BLE001 -- sportsbook fetch failure -- report empty, don't crash the UI
        props_df = pd.DataFrame()

    if props_df is None or props_df.empty:
        return _empty_board(generated_at, bookmaker_key, model_version)

    props_df = props_df.copy()
    props_df["normalized_name"] = props_df["player_name_raw"].apply(
        shared_app.normalize_name
    )
    props_df = props_df.drop_duplicates(subset=["normalized_name"]).reset_index(
        drop=True
    )
    props_discovered = len(props_df)

    active_players = shared_app.load_active_players(_provider=provider)

    try:
        todays_games = shared_app.get_scoreboard_for_date(_provider=provider)
    except Exception:  # noqa: BLE001 -- schedule lookup is best-effort context, never fatal
        todays_games = []

    gamelog_cache: dict = {}
    predictions = []
    players_matched = 0
    unmatched_count = 0
    unavailable_count = 0

    rows = list(props_df.itertuples(index=False))
    for i, row in enumerate(rows):
        raw_name = row.player_name_raw
        matchup = f"{getattr(row, 'away_team', '')} @ {getattr(row, 'home_team', '')}"
        line = shared_app.safe_float(getattr(row, "line", None))
        row_bookmaker = str(getattr(row, "bookmaker_key", "") or bookmaker_key)

        result = predict_player(
            raw_name,
            sportsbook_line=line,
            bookmaker=row_bookmaker,
            season=season,
            provider=provider,
            model=model,
            model_version=model_version,
            active_players=active_players,
            gamelog_cache=gamelog_cache,
            apply_live_adjustment=False,
            use_fuzzy_name_resolution=True,
            todays_games=todays_games,
            matchup=matchup,
        )
        predictions.append(result)

        if result.status == PredictionStatus.UNMATCHED:
            unmatched_count += 1
        elif result.status in _UNAVAILABLE_STATUSES:
            unavailable_count += 1
        else:
            players_matched += 1

        if request_delay and i < len(rows) - 1:
            sleep_func(request_delay)

    predictions.sort(
        key=lambda r: abs(r.edge) if r.edge is not None else -1.0, reverse=True
    )
    predictions_generated = sum(
        1 for r in predictions if r.status == PredictionStatus.OK
    )

    return PredictionBoard(
        generated_at_utc=generated_at,
        bookmaker=bookmaker_key,
        model_version=model_version,
        predictions=tuple(predictions),
        props_discovered=props_discovered,
        players_matched=players_matched,
        predictions_generated=predictions_generated,
        unmatched_count=unmatched_count,
        unavailable_count=unavailable_count,
    )
