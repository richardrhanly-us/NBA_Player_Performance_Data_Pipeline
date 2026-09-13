"""
The NBA development provider: implements BasketballDataProvider by
wrapping training.data.nba_client (the existing, already-tested
stats.nba.com retry/fetch layer) and training.data.storage's existing
raw normalization, then converting the result into canonical models via
src.data.basketball.normalization.

This is the CURRENT, ACTIVE development provider. It is not a permanent
commitment to stats.nba.com -- see src/data/basketball/__init__.py.

NOTE ON THE training/ DEPENDENCY: this is the one module under src/ that
imports from training/ (training.data.nba_client, training.data.storage),
rather than the reverse (the convention everywhere else in this
repository -- training/ depends on src/, never the other way around).
This is a deliberate, narrow exception: training/data/nba_client.py and
training/data/storage.py already contain the correct, already-tested
NBA-specific fetch/retry/parsing logic, and physically relocating that
code into src/ during this refactor was judged unnecessary risk for a
step whose #1 constraint is zero behavior change (see the Step 6
report's dependency-map/limitations sections). A future licensed
provider implementation would be self-contained under
src/data/basketball/providers/ and would not need this pattern.
"""

from __future__ import annotations

import random
import time

from src.data.basketball.errors import ProviderUnavailableError
from src.data.basketball.models import Player, PlayerGameLog
from src.data.basketball.normalization import (
    dataframe_to_player_game_logs,
    dataframe_to_players,
)
from training.data import nba_client, storage


class NBAApiProvider:
    """Current development BasketballDataProvider, backed by
    stats.nba.com via nba_api (see training/data/nba_client.py)."""

    def __init__(self, *, sleep_func=time.sleep, rand_func=random.random):
        self._sleep_func = sleep_func
        self._rand_func = rand_func

    def get_season_roster(self, season: str) -> list[Player]:
        try:
            roster_df = nba_client.fetch_season_roster(
                season, sleep_func=self._sleep_func, rand_func=self._rand_func
            )
        except nba_client.NbaApiError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        return dataframe_to_players(roster_df)

    def get_player_game_logs(
        self, player_id: int, season: str, *, player_name: str | None = None
    ) -> list[PlayerGameLog]:
        try:
            raw_df = nba_client.fetch_player_gamelog(
                player_id,
                season,
                sleep_func=self._sleep_func,
                rand_func=self._rand_func,
            )
        except nba_client.NbaApiError as exc:
            raise ProviderUnavailableError(str(exc)) from exc

        # storage.normalize_raw_gamelog is the existing, already-tested
        # NBA-specific parsing step (Player_ID/Game_ID rename, MATCHUP
        # splitting, dtype coercion) -- unchanged by this refactor. Its
        # output is already shaped like RAW_GAMELOG_COLUMNS, which is
        # exactly what dataframe_to_player_game_logs expects.
        normalized_df = storage.normalize_raw_gamelog(
            raw_df, season=season, player_id=player_id, player_name=player_name or ""
        )
        return dataframe_to_player_game_logs(normalized_df)
