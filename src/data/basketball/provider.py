"""
The basketball-data provider contract: the ONLY interface downstream
collection/training code should need in order to obtain roster and
game-log data, regardless of which external source backs it.

Methods here are derived strictly from current repository usage --
training.data.collect_gamelogs's two real capabilities, historically
implemented as direct nba_api calls: enumerate a season's roster, then
fetch one player's game log for a season. Do NOT add
get_injuries()/get_projected_lineups()/get_schedule() etc. here until
real code actually needs to consume them -- see
src/data/basketball/__init__.py for the fuller rationale. Keeping this
contract minimal is deliberate, not an oversight.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.data.basketball.models import Player, PlayerGameLog


@runtime_checkable
class BasketballDataProvider(Protocol):
    def get_season_roster(self, season: str) -> list[Player]:
        """Every player who appeared in at least one game in `season`."""
        ...

    def get_player_game_logs(
        self, player_id: int, season: str, *, player_name: str | None = None
    ) -> list[PlayerGameLog]:
        """
        One player's game-by-game box score log for `season`.

        `player_name` is accepted (optional) because the current NBA
        source's game-log response does not itself carry a display
        name -- the caller (collect_gamelogs.py) already has it from
        get_season_roster() and passes it through so
        PlayerGameLog.player_name can be populated. A future provider
        whose game-log response already includes a name is free to
        ignore this argument.
        """
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
    state. See training/data/collect_gamelogs.py.
    """
    global _active_provider
    if _active_provider is None:
        from src.data.basketball.providers.nba_api_provider import NBAApiProvider

        _active_provider = NBAApiProvider()
    return _active_provider
