"""
Canonical internal basketball-data models.

These are OUR application's data contract -- named and typed for what
this project actually needs, not for any one vendor's response shape. A
provider (see provider.py) is responsible for translating its own raw
payload into these objects; nothing downstream of a provider should need
to know whether that payload came from nba_api, a licensed commercial
feed, or anything else.

Only two objects exist here because only two are justified by current
repository usage -- training/data/collect_gamelogs.py (the sole current
consumer of a provider) needs exactly: a roster entry to enumerate
players, and one player's one-game box-score log. Do not add
speculative fields/objects for capabilities (injuries, lineups,
schedule, ...) the project does not yet consume -- see this package's
__init__.py for the fuller rationale.

A NOTE ON player_id: it is deliberately typed and treated as an opaque
identifier scoped to whichever provider produced it -- NOT assumed to be
a universal, cross-provider identity. Today it is always an NBA
stats.nba.com player ID (nba_api's own PLAYER_ID), because NBAApiProvider
is the only provider that exists. The rest of this codebase (the raw
Parquet store, the V1 feature panel, the trained model) currently keys
everything on this same NBA ID throughout -- that is real migration debt
for a future licensed provider, whose IDs will differ. This module does
not attempt to fix that now (no ID-mapping table, no internal
canonical-identity concept -- out of scope for Step 6, see the Step 6
report's identifier-strategy section); the smallest useful seam it does
provide is simply not hard-coding "player_id is an NBA ID" into the
canonical contract's name or type, so a future mapping layer (translating
a provider's player_id into a stable internal identity) has somewhere
sensible to attach without renaming this field.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Player:
    """One entry from a season roster -- exactly what
    collect_gamelogs.py's player-enumeration loop needs, no more."""

    player_id: int
    player_name: str


@dataclass(frozen=True)
class PlayerGameLog:
    """
    One player's box-score-style statistics for one game.

    Field names and shape mirror training.data.storage.RAW_GAMELOG_COLUMNS
    (the existing, already-justified raw historical-data contract) in
    our own snake_case naming, decoupled from any vendor's raw column
    casing (e.g. nba_api's inconsistent Player_ID/Game_ID naming).

    A field being Optional reflects that the current NBA source can
    leave it missing/unparseable for a given row (see
    training.data.storage.normalize_raw_gamelog) -- it is NOT an
    invitation to invent data the source doesn't provide.
    """

    season: str
    season_id: str | None
    player_id: int | None
    player_name: str
    game_id: str | None
    game_date: datetime | None
    matchup: str
    team_abbreviation: str | None
    opponent_abbreviation: str | None
    is_home: int | None
    wl: str | None
    minutes: str | None
    fgm: float | None
    fga: float | None
    fg_pct: float | None
    fg3m: float | None
    fg3a: float | None
    fg3_pct: float | None
    ftm: float | None
    fta: float | None
    ft_pct: float | None
    oreb: float | None
    dreb: float | None
    reb: float | None
    ast: float | None
    stl: float | None
    blk: float | None
    tov: float | None
    pf: float | None
    points: float | None
    plus_minus: float | None
    video_available: float | None
