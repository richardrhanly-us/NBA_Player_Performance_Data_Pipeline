"""
Canonical internal basketball-data models.

These are OUR application's data contract -- named and typed for what
this project actually needs, not for any one vendor's response shape. A
provider (see provider.py) is responsible for translating its own raw
payload into these objects; nothing downstream of a provider should need
to know whether that payload came from nba_api, a licensed commercial
feed, or anything else.

Step 6 added two objects, justified by training/data/collect_gamelogs.py
(historical collection): a roster entry (Player) and one player's
one-game box score (PlayerGameLog). Step 7 added four more, justified by
src/shared_app.py's live prediction path: PlayerDetails (current team/
position lookup), ScheduledGame (today's schedule), LivePlayerStatLine
and LiveBoxScore (in-game stat line). Every field on every model here is
something the application genuinely reads today -- do not add
speculative fields/objects for capabilities (injuries, lineups,
projected minutes, ...) the project does not yet consume -- see this
package's __init__.py for the fuller rationale.

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
from typing import Any


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


@dataclass(frozen=True)
class PlayerDetails:
    """
    Current-team/position lookup for one player -- exactly the 4 fields
    src/shared_app.py's live path reads from nba_api's CommonPlayerInfo
    (a payload with dozens of other columns nothing in this app uses).
    team_id is used internally (to match a player to today's scheduled
    game); team_name/team_abbreviation/position are display metadata.
    """

    player_id: int
    team_id: int | None
    team_name: str | None
    team_abbreviation: str | None
    position: str | None


@dataclass(frozen=True)
class ScheduledGame:
    """
    One game on a given date's schedule -- exactly the fields
    src/shared_app.py's live path reads from nba_api's ScoreboardV2 game
    header to find which game a team is playing today and whether it's
    live. `game_date` is the same "%m/%d/%Y"-formatted string the app has
    always used to request a schedule (see get_todays_scoreboard) --
    preserved verbatim rather than parsed into a date object, since nothing
    downstream needs more than the original string.
    """

    game_id: str
    game_date: str
    home_team_id: int | None
    away_team_id: int | None
    game_status_text: str | None


@dataclass(frozen=True)
class LivePlayerStatLine:
    """
    One player's row within a LiveBoxScore. `points`/`minutes` are kept
    as `Any`, deliberately unconverted -- the live NBA payload's own
    dynamic/inconsistent typing (points as int or str; minutes as an
    ISO-8601-ish duration string) is exactly what
    src/shared_app.py's existing post-processing
    (parse_game_clock_to_minutes-style parsing, str()/float() coercions
    at the point of use) already expects and handles; normalizing here
    would risk silently changing that existing behavior.
    """

    player_id: int | None
    first_name: str
    last_name: str
    points: Any
    minutes: Any


@dataclass(frozen=True)
class LiveBoxScore:
    """
    One in-progress-or-final game's live box score -- exactly what
    src/shared_app.py's get_live_player_stats reads from nba_api's live
    BoxScore endpoint: the game clock/period, and every player's live
    stat line (to find the one player being predicted for).
    """

    game_id: str
    period: int | None
    game_clock: str | None
    players: tuple[LivePlayerStatLine, ...]
