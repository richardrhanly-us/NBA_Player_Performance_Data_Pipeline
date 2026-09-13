"""
Tests proving src/shared_app.py's live-application functions
(load_active_players, get_player_details, get_player_gamelog_df,
get_scoreboard_for_date, get_live_player_stats) operate correctly against
a fake, offline, in-memory BasketballDataProvider -- no NBA.com, no
network access, no sportsbook API access.

This is itself the proof that these functions depend only on
src.data.basketball.provider.BasketballDataProvider, not on any NBA
endpoint class or object.
"""

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
from src.shared_app import (
    get_live_player_stats,
    get_player_details,
    get_player_gamelog_df,
    get_scoreboard_for_date,
    load_active_players,
)


class FakeLiveProvider:
    """Minimal in-memory BasketballDataProvider covering every live
    capability, each independently configurable/failable per test."""

    def __init__(
        self,
        *,
        active_players=None,
        player_details=None,
        game_logs=None,
        scoreboard=None,
        box_score=None,
        raise_on_gamelogs=False,
    ):
        self._active_players = active_players or []
        self._player_details = player_details or {}
        self._game_logs = game_logs or {}
        self._scoreboard = scoreboard if scoreboard is not None else []
        self._box_score = box_score
        self._raise_on_gamelogs = raise_on_gamelogs

    def get_season_roster(self, season):
        raise NotImplementedError  # not used by the live path

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        if self._raise_on_gamelogs:
            raise ProviderUnavailableError("simulated provider outage")
        return self._game_logs.get(player_id, [])

    def get_active_players(self):
        return list(self._active_players)

    def get_player_details(self, player_id):
        return self._player_details.get(player_id)

    def get_todays_scoreboard(self, game_date=None):
        return list(self._scoreboard)

    def get_live_box_score(self, game_id):
        return self._box_score


def _game_log(player_id, points, game_id="G1"):
    return PlayerGameLog(
        season="2025-26",
        season_id="22025",
        player_id=player_id,
        player_name="Test Player",
        game_id=game_id,
        game_date=pd.Timestamp("2026-01-15"),
        matchup="LAL vs DEN",
        team_abbreviation="LAL",
        opponent_abbreviation="DEN",
        is_home=1,
        wl="W",
        minutes="35",
        fgm=10.0,
        fga=18.0,
        fg_pct=0.55,
        fg3m=2.0,
        fg3a=5.0,
        fg3_pct=0.4,
        ftm=4.0,
        fta=5.0,
        ft_pct=0.8,
        oreb=1.0,
        dreb=6.0,
        reb=7.0,
        ast=8.0,
        stl=1.0,
        blk=1.0,
        tov=3.0,
        pf=2.0,
        points=float(points),
        plus_minus=10.0,
        video_available=1.0,
    )


# ---- player lookup ------------------------------------------------------


def test_load_active_players_maps_provider_players_correctly():
    provider = FakeLiveProvider(
        active_players=[
            Player(player_id=2544, player_name="LeBron James"),
            Player(player_id=201939, player_name="Stephen Curry"),
        ]
    )
    actual_name_to_id, normalized_to_actual = load_active_players(_provider=provider)

    assert actual_name_to_id["LeBron James"] == 2544
    assert actual_name_to_id["Stephen Curry"] == 201939
    assert normalized_to_actual["lebron james"] == "LeBron James"


def test_load_active_players_empty_roster_returns_empty_dicts():
    provider = FakeLiveProvider(active_players=[])
    actual_name_to_id, normalized_to_actual = load_active_players(_provider=provider)
    assert actual_name_to_id == {}
    assert normalized_to_actual == {}


# ---- player metadata ------------------------------------------------------


def test_get_player_details_returns_provider_result():
    details = PlayerDetails(
        player_id=2544,
        team_id=1610612747,
        team_name="Lakers",
        team_abbreviation="LAL",
        position="F",
    )
    provider = FakeLiveProvider(player_details={2544: details})
    result = get_player_details(2544, _provider=provider)
    assert result == details


def test_get_player_details_player_not_found_returns_none():
    provider = FakeLiveProvider(player_details={})
    result = get_player_details(999999, _provider=provider)
    assert result is None


def test_get_player_details_missing_optional_metadata_preserved_as_none():
    details = PlayerDetails(
        player_id=1, team_id=None, team_name=None, team_abbreviation=None, position=None
    )
    provider = FakeLiveProvider(player_details={1: details})
    result = get_player_details(1, _provider=provider)
    assert result.team_name is None
    assert result.position is None


# ---- recent gamelog -------------------------------------------------------


def test_get_player_gamelog_df_returns_dataframe_built_from_provider_records():
    provider = FakeLiveProvider(
        game_logs={2544: [_game_log(2544, 30, "G1"), _game_log(2544, 25, "G2")]}
    )
    df = get_player_gamelog_df(2544, "2025-26", _provider=provider)

    assert not df.empty
    assert list(df["PTS"]) == [30.0, 25.0]
    assert "MATCHUP" in df.columns
    assert "MIN" in df.columns


def test_get_player_gamelog_df_provider_unavailable_returns_empty_dataframe():
    provider = FakeLiveProvider(raise_on_gamelogs=True)
    df = get_player_gamelog_df(2544, "2025-26", _provider=provider)
    assert df.empty


def test_get_player_gamelog_df_no_games_returns_empty_dataframe():
    provider = FakeLiveProvider(game_logs={2544: []})
    df = get_player_gamelog_df(2544, "2025-26", _provider=provider)
    assert df.empty


# ---- schedule / scoreboard -------------------------------------------------


def test_get_scoreboard_for_date_maps_provider_games():
    game = ScheduledGame(
        game_id="G1",
        game_date="01/15/2026",
        home_team_id=1,
        away_team_id=2,
        game_status_text="7:30 pm ET",
    )
    provider = FakeLiveProvider(scoreboard=[game])
    result = get_scoreboard_for_date("01/15/2026", _provider=provider)
    assert result == [game]


def test_get_scoreboard_for_date_no_games_today_returns_empty_list():
    provider = FakeLiveProvider(scoreboard=[])
    result = get_scoreboard_for_date("01/15/2026", _provider=provider)
    assert result == []


# ---- live player stats: full scenario coverage -----------------------------


def test_get_live_player_stats_player_not_found_returns_none():
    provider = FakeLiveProvider(active_players=[])
    result = get_live_player_stats("Unknown Player", provider=provider)
    assert result is None


def test_get_live_player_stats_no_games_today_returns_none():
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={
            2544: PlayerDetails(
                player_id=2544,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        scoreboard=[],
    )
    result = get_live_player_stats("LeBron James", provider=provider)
    assert result is None


def test_get_live_player_stats_pregame_returns_none_when_no_live_box_score():
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={
            2544: PlayerDetails(
                player_id=2544,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="today",
                home_team_id=1610612747,
                away_team_id=999,
                game_status_text="7:30 pm ET",
            )
        ],
        box_score=None,  # game hasn't tipped off -- no live box score yet
    )
    result = get_live_player_stats("LeBron James", provider=provider)
    assert result is None


def test_get_live_player_stats_live_game_returns_expected_dict_shape():
    line = LivePlayerStatLine(
        player_id=2544,
        first_name="LeBron",
        last_name="James",
        points=24,
        minutes="30:00",
    )
    box_score = LiveBoxScore(
        game_id="G1", period=3, game_clock="PT5M00.00S", players=(line,)
    )
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={
            2544: PlayerDetails(
                player_id=2544,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="today",
                home_team_id=1610612747,
                away_team_id=999,
                game_status_text="Live",
            )
        ],
        box_score=box_score,
    )
    result = get_live_player_stats("LeBron James", provider=provider)

    assert result is not None
    assert set(result.keys()) == {
        "points",
        "minutes",
        "game_status",
        "period",
        "game_clock",
        "game_minutes_remaining",
    }
    assert result["points"] == 24
    assert result["minutes"] == "30:00"
    assert result["period"] == 3


def test_get_live_player_stats_final_game_status_preserved():
    line = LivePlayerStatLine(
        player_id=2544,
        first_name="LeBron",
        last_name="James",
        points=40,
        minutes="38:00",
    )
    box_score = LiveBoxScore(
        game_id="G1", period=4, game_clock="PT0M00.00S", players=(line,)
    )
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={
            2544: PlayerDetails(
                player_id=2544,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="today",
                home_team_id=1610612747,
                away_team_id=999,
                game_status_text="Final",
            )
        ],
        box_score=box_score,
    )
    result = get_live_player_stats("LeBron James", provider=provider)
    assert result["game_status"] == "Final"
    assert result["points"] == 40


def test_get_live_player_stats_missing_player_in_box_score_returns_none():
    other_line = LivePlayerStatLine(
        player_id=999,
        first_name="Someone",
        last_name="Else",
        points=10,
        minutes="20:00",
    )
    box_score = LiveBoxScore(
        game_id="G1", period=2, game_clock="PT6M00.00S", players=(other_line,)
    )
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={
            2544: PlayerDetails(
                player_id=2544,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="today",
                home_team_id=1610612747,
                away_team_id=999,
                game_status_text="Live",
            )
        ],
        box_score=box_score,
    )
    result = get_live_player_stats("LeBron James", provider=provider)
    assert result is None


def test_get_live_player_stats_matches_by_name_when_person_id_missing():
    """personId absent/mismatched -- falls back to first+last name match,
    exactly like the pre-Step-7 implementation."""
    line = LivePlayerStatLine(
        player_id=None,
        first_name="LeBron",
        last_name="James",
        points=15,
        minutes="18:00",
    )
    box_score = LiveBoxScore(
        game_id="G1", period=2, game_clock="PT4M00.00S", players=(line,)
    )
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={
            2544: PlayerDetails(
                player_id=2544,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="today",
                home_team_id=1610612747,
                away_team_id=999,
                game_status_text="Live",
            )
        ],
        box_score=box_score,
    )
    result = get_live_player_stats("LeBron James", provider=provider)
    assert result is not None
    assert result["points"] == 15


def test_get_live_player_stats_provider_unavailable_for_details_returns_none():
    provider = FakeLiveProvider(
        active_players=[Player(player_id=2544, player_name="LeBron James")],
        player_details={},  # get_player_details returns None -- "provider has no data"
    )
    result = get_live_player_stats("LeBron James", provider=provider)
    assert result is None
