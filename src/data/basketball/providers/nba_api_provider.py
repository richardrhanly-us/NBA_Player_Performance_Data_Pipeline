"""
The NBA development provider: implements BasketballDataProvider by
wrapping training.data.nba_client (the existing, already-tested
stats.nba.com retry/fetch layer) and training.data.storage's existing
raw normalization, then converting the result into canonical models via
src.data.basketball.normalization. Step 7 added the live-application
capabilities (get_active_players, get_player_details,
get_todays_scoreboard, get_live_box_score), each wrapping the exact
nba_api endpoint src/shared_app.py used directly before this refactor.

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
step whose #1 constraint is zero behavior change (see the Step 6/7
reports' dependency-map/limitations sections). A future licensed
provider implementation would be self-contained under
src/data/basketball/providers/ and would not need this pattern.

NOTE ON RETRY BEHAVIOR: get_season_roster/get_player_game_logs go
through training.data.nba_client.call_with_retries (5 attempts,
exponential backoff) -- this was already true for historical collection
in Step 6, and Step 7 reuses get_player_game_logs unchanged for the live
app's "recent games" fetch too (see the Protocol's docstring). The four
new live-only methods below (get_active_players, get_player_details,
get_todays_scoreboard, get_live_box_score) are each a single attempt
with a broad try/except-return-None/empty fallback -- exactly matching
what src/shared_app.py did directly before this refactor for each of
them (none of those four had any retry loop pre-Step-7 either).
"""

from __future__ import annotations

import random
import time
from datetime import datetime

import pandas as pd

from src.data.basketball.errors import ProviderUnavailableError
from src.data.basketball.models import (
    LiveBoxScore,
    LivePlayerStatLine,
    Player,
    PlayerDetails,
    PlayerGameLog,
    ScheduledGame,
)
from src.data.basketball.normalization import (
    dataframe_to_player_game_logs,
    dataframe_to_players,
)
from training.data import nba_client, storage


def _int_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


class NBAApiProvider:
    """Current development BasketballDataProvider, backed by
    stats.nba.com / nba_api (see training/data/nba_client.py)."""

    def __init__(self, *, sleep_func=time.sleep, rand_func=random.random):
        self._sleep_func = sleep_func
        self._rand_func = rand_func

    # ---- Step 6: historical collection / training -----------------------

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

    # ---- Step 7: live application -----------------------------------------

    def get_active_players(self) -> list[Player]:
        """
        nba_api.stats.static.players is a package-bundled static lookup,
        NOT a network call -- no retry/timeout logic applies here, same
        as the pre-Step-7 shared_app.py implementation.
        """
        from nba_api.stats.static import players as static_players

        active = static_players.get_active_players()
        return [
            Player(player_id=int(p["id"]), player_name=str(p["full_name"]).strip())
            for p in active
        ]

    def get_player_details(self, player_id: int) -> PlayerDetails | None:
        from nba_api.stats.endpoints import commonplayerinfo

        try:
            info_df = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id, timeout=12
            ).get_data_frames()[0]
        except Exception:  # noqa: BLE001 -- matches pre-Step-7 behavior exactly:
            # any failure (network, malformed response, unknown player) falls
            # back to "no details available" rather than propagating, since
            # this is display/context metadata the app has always tolerated
            # losing gracefully.
            return None

        if info_df is None or info_df.empty:
            return None

        row = info_df.iloc[0]
        return PlayerDetails(
            player_id=int(player_id),
            team_id=_int_or_none(row.get("TEAM_ID"))
            if "TEAM_ID" in info_df.columns
            else None,
            team_name=_str_or_none(row.get("TEAM_NAME"))
            if "TEAM_NAME" in info_df.columns
            else None,
            team_abbreviation=_str_or_none(row.get("TEAM_ABBREVIATION"))
            if "TEAM_ABBREVIATION" in info_df.columns
            else None,
            position=_str_or_none(row.get("POSITION"))
            if "POSITION" in info_df.columns
            else None,
        )

    def get_todays_scoreboard(
        self, game_date: str | None = None
    ) -> list[ScheduledGame]:
        """
        game_date defaults to the LOCAL SYSTEM's naive current date
        (datetime.now(), no timezone) -- this exactly matches the
        pre-Step-7 shared_app.get_scoreboard_for_date default. It is a
        pre-existing quirk (the module's only real caller,
        get_live_player_stats, always passes an explicit US/Eastern date
        string instead of relying on this default) preserved verbatim,
        not "fixed", per the Step 7 zero-behavior-change constraint.
        """
        from nba_api.stats.endpoints import scoreboardv2

        if game_date is None:
            game_date = datetime.now().strftime("%m/%d/%Y")

        try:
            frames = scoreboardv2.ScoreboardV2(
                game_date=game_date,
                # nba_api's own default (DayOffset.default) is the string
                # "0", not the int 0 -- matching it keeps behavior identical.
                day_offset="0",
                league_id="00",
                timeout=12,
            ).get_data_frames()
        except Exception:  # noqa: BLE001 -- matches pre-Step-7 behavior:
            # any failure means "no schedule available", not a raised error.
            return []

        # Pre-Step-7 required >= 2 frames before treating the scoreboard as
        # usable (a defensive check against ScoreboardV2's normal
        # multi-frame shape), even though only frame [0] is ever read.
        # Preserved exactly: fewer than 2 frames -> no games.
        if not frames or len(frames) < 2:
            return []

        game_header = frames[0]
        if game_header is None or game_header.empty:
            return []

        games = []
        for row in game_header.to_dict(orient="records"):
            games.append(
                ScheduledGame(
                    game_id=str(row.get("GAME_ID")),
                    game_date=game_date,
                    home_team_id=_int_or_none(row.get("HOME_TEAM_ID")),
                    away_team_id=_int_or_none(row.get("VISITOR_TEAM_ID")),
                    game_status_text=_str_or_none(row.get("GAME_STATUS_TEXT")),
                )
            )
        return games

    def get_live_box_score(self, game_id: str) -> LiveBoxScore | None:
        try:
            from nba_api.live.nba.endpoints import boxscore as live_boxscore
        except Exception:  # noqa: BLE001 -- matches pre-Step-7 behavior:
            # the live endpoints module is optional/best-effort.
            return None

        try:
            data = live_boxscore.BoxScore(game_id=game_id).get_dict()
        except Exception:  # noqa: BLE001 -- matches pre-Step-7 behavior:
            # any live-fetch failure means "no live data", not a raised error.
            return None

        game_data = data.get("game", {}) if isinstance(data, dict) else {}
        players_raw = []
        players_raw.extend(game_data.get("homeTeam", {}).get("players", []))
        players_raw.extend(game_data.get("awayTeam", {}).get("players", []))

        player_lines = tuple(
            LivePlayerStatLine(
                player_id=_int_or_none(p.get("personId")),
                first_name=str(p.get("firstName", "") or "").strip(),
                last_name=str(p.get("familyName", "") or "").strip(),
                # Deliberately unconverted -- see LivePlayerStatLine's
                # docstring: shared_app.py's existing post-processing
                # already handles the live payload's own dynamic typing.
                points=(p.get("statistics", {}) or {}).get("points", 0),
                minutes=(p.get("statistics", {}) or {}).get("minutes", "0"),
            )
            for p in players_raw
        )

        return LiveBoxScore(
            game_id=str(game_id),
            period=_int_or_none(game_data.get("period")),
            game_clock=game_data.get("gameClock"),
            players=player_lines,
        )
