"""
The basketball-data provider contract: the ONLY interface downstream
collection/training/live-inference code should need in order to obtain
basketball data, regardless of which external source backs it.

Methods here are derived strictly from current repository usage. Do NOT
add get_injuries()/get_projected_lineups()/get_schedule()-for-a-future-
date/etc. here until real code actually needs to consume them -- see
src/data/basketball/__init__.py for the fuller rationale. Keeping this
contract minimal is deliberate, not an oversight.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.data.basketball.models import (
    LiveBoxScore,
    Player,
    PlayerDetails,
    PlayerGameLog,
    ScheduledGame,
)


@runtime_checkable
class BasketballDataProvider(Protocol):
    # ---- Step 6: historical collection / training -------------------
    def get_season_roster(self, season: str) -> list[Player]:
        """Every player who appeared in at least one game in `season`."""
        ...

    def get_player_game_logs(
        self, player_id: int, season: str, *, player_name: str | None = None
    ) -> list[PlayerGameLog]:
        """
        One player's game-by-game box score log for `season`. Used for
        BOTH historical collection (a full past season) and the live
        app's "recent games" feature input (the current season to date)
        -- the same capability serves both, since nba_api's PlayerGameLog
        endpoint already returns everything played so far when `season`
        is the current season. See src/shared_app.py::get_player_gamelog_df.

        `player_name` is accepted (optional) because the current NBA
        source's game-log response does not itself carry a display
        name -- a caller that already has it (e.g. from
        get_season_roster()/get_active_players()) passes it through so
        PlayerGameLog.player_name can be populated. A future provider
        whose game-log response already includes a name is free to
        ignore this argument.
        """
        ...

    # ---- Step 7: live application -------------------------------------
    def get_active_players(self) -> list[Player]:
        """Every player currently on an active NBA roster -- used for
        the live app's player search/autocomplete. A different concept
        from get_season_roster(): "active right now" vs "played at all
        in a given past/current season"."""
        ...

    def get_player_details(self, player_id: int) -> PlayerDetails | None:
        """Current team/position lookup for one player. Returns None if
        the player can't be found or the lookup fails -- matches the
        pre-Step-7 behavior of silently falling back to an empty result
        rather than raising, since this is display/context metadata,
        not something prediction correctness depends on."""
        ...

    def get_todays_scoreboard(
        self, game_date: str | None = None
    ) -> list[ScheduledGame]:
        """
        Every game scheduled on `game_date` (nba_api's own
        "%m/%d/%Y"-formatted string; None defaults to the local system's
        current date, matching the pre-Step-7 default exactly -- see
        NBAApiProvider.get_todays_scoreboard's docstring for why that
        specific, slightly quirky default is preserved rather than
        "fixed"). Returns an empty list if there are no games or the
        lookup fails.
        """
        ...

    def get_live_box_score(self, game_id: str) -> LiveBoxScore | None:
        """One game's current live box score (per-player stat lines,
        period, game clock). Returns None if the game isn't live yet, no
        live data is available, or the lookup fails."""
        ...


_active_provider: BasketballDataProvider | None = None


def get_basketball_provider() -> BasketballDataProvider:
    """
    Returns the process-wide active basketball-data provider.

    Currently always the NBA development provider
    (src.data.basketball.providers.nba_api_provider.NBAApiProvider) --
    there is no commercial provider to select between yet, and this
    function deliberately exposes no configuration suggesting one
    exists (see this package's __init__.py).

    Collectors/services should still prefer accepting a `provider`
    parameter (defaulting to this function's result) over calling this
    at import time or deep in a call stack, so tests can inject a fake
    provider via a plain function argument instead of patching global
    state. See training/data/collect_gamelogs.py and src/shared_app.py.
    """
    global _active_provider
    if _active_provider is None:
        from src.data.basketball.providers.nba_api_provider import NBAApiProvider

        _active_provider = NBAApiProvider()
    return _active_provider
