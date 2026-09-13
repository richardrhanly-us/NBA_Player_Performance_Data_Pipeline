"""
Tests for the Step 7 live-application capabilities added to
NBAApiProvider (get_active_players, get_player_details,
get_todays_scoreboard, get_live_box_score). No network access -- nba_api
endpoint classes are mocked.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.basketball.models import Player
from src.data.basketball.provider import BasketballDataProvider
from src.data.basketball.providers.nba_api_provider import NBAApiProvider


def test_nba_api_provider_still_satisfies_the_expanded_protocol():
    provider = NBAApiProvider()
    assert isinstance(provider, BasketballDataProvider)
    for method in (
        "get_season_roster",
        "get_player_game_logs",
        "get_active_players",
        "get_player_details",
        "get_todays_scoreboard",
        "get_live_box_score",
    ):
        assert hasattr(provider, method)


def test_get_active_players_wraps_static_lookup():
    fake_active = [
        {"id": 2544, "full_name": "LeBron James"},
        {"id": 201939, "full_name": "Stephen Curry"},
    ]
    with patch(
        "nba_api.stats.static.players.get_active_players", return_value=fake_active
    ):
        provider = NBAApiProvider()
        result = provider.get_active_players()

    assert result == [
        Player(player_id=2544, player_name="LeBron James"),
        Player(player_id=201939, player_name="Stephen Curry"),
    ]


def test_get_active_players_does_not_hit_network_or_retry():
    """Static player list is package-bundled data, not a network call --
    confirm no retry/sleep machinery is invoked."""
    with patch(
        "nba_api.stats.static.players.get_active_players", return_value=[]
    ) as mock_call:
        NBAApiProvider().get_active_players()
    mock_call.assert_called_once()


def _player_info_df(**overrides):
    row = {
        "PERSON_ID": 2544,
        "TEAM_ID": 1610612747,
        "TEAM_NAME": "Lakers",
        "TEAM_ABBREVIATION": "LAL",
        "POSITION": "F",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_get_player_details_maps_fields_correctly():
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [_player_info_df()]
    with patch(
        "nba_api.stats.endpoints.commonplayerinfo.CommonPlayerInfo",
        return_value=fake_response,
    ):
        result = NBAApiProvider().get_player_details(2544)

    assert result is not None
    assert result.player_id == 2544
    assert result.team_id == 1610612747
    assert result.team_name == "Lakers"
    assert result.team_abbreviation == "LAL"
    assert result.position == "F"


def test_get_player_details_returns_none_on_empty_response():
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [pd.DataFrame()]
    with patch(
        "nba_api.stats.endpoints.commonplayerinfo.CommonPlayerInfo",
        return_value=fake_response,
    ):
        result = NBAApiProvider().get_player_details(999999)
    assert result is None


def test_get_player_details_returns_none_on_failure_not_raise():
    """Matches pre-Step-7 shared_app.py behavior: any failure (not just
    "not found") silently falls back to None."""
    with patch(
        "nba_api.stats.endpoints.commonplayerinfo.CommonPlayerInfo",
        side_effect=ConnectionError("down"),
    ):
        result = NBAApiProvider().get_player_details(2544)
    assert result is None


def test_get_player_details_handles_missing_optional_columns():
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [
        pd.DataFrame([{"PERSON_ID": 2544, "TEAM_ID": 1610612747}])
    ]
    with patch(
        "nba_api.stats.endpoints.commonplayerinfo.CommonPlayerInfo",
        return_value=fake_response,
    ):
        result = NBAApiProvider().get_player_details(2544)

    assert result.team_id == 1610612747
    assert result.team_name is None
    assert result.team_abbreviation is None
    assert result.position is None


def _scoreboard_frames(rows):
    game_header = pd.DataFrame(rows)
    line_score = pd.DataFrame({"placeholder": [1]})  # a 2nd frame, unused
    return [game_header, line_score]


def test_get_todays_scoreboard_maps_games_correctly():
    frames = _scoreboard_frames(
        [
            {
                "GAME_ID": "0022500123",
                "HOME_TEAM_ID": 1610612747,
                "VISITOR_TEAM_ID": 1610612738,
                "GAME_STATUS_TEXT": "7:30 pm ET",
            }
        ]
    )
    with patch("nba_api.stats.endpoints.scoreboardv2.ScoreboardV2") as mock_cls:
        mock_cls.return_value.get_data_frames.return_value = frames
        games = NBAApiProvider().get_todays_scoreboard("01/15/2026")

    assert len(games) == 1
    game = games[0]
    assert game.game_id == "0022500123"
    assert game.game_date == "01/15/2026"
    assert game.home_team_id == 1610612747
    assert game.away_team_id == 1610612738
    assert game.game_status_text == "7:30 pm ET"


def test_get_todays_scoreboard_returns_empty_list_on_no_games():
    with patch("nba_api.stats.endpoints.scoreboardv2.ScoreboardV2") as mock_cls:
        mock_cls.return_value.get_data_frames.return_value = _scoreboard_frames([])
        games = NBAApiProvider().get_todays_scoreboard("01/15/2026")
    assert games == []


def test_get_todays_scoreboard_requires_at_least_two_frames():
    """Preserves the pre-Step-7 defensive check exactly."""
    with patch("nba_api.stats.endpoints.scoreboardv2.ScoreboardV2") as mock_cls:
        mock_cls.return_value.get_data_frames.return_value = [
            pd.DataFrame([{"GAME_ID": "X"}])
        ]
        games = NBAApiProvider().get_todays_scoreboard("01/15/2026")
    assert games == []


def test_get_todays_scoreboard_returns_empty_list_on_failure():
    with patch(
        "nba_api.stats.endpoints.scoreboardv2.ScoreboardV2",
        side_effect=ConnectionError("down"),
    ):
        games = NBAApiProvider().get_todays_scoreboard("01/15/2026")
    assert games == []


def _live_box_dict(players):
    return {
        "game": {
            "period": 3,
            "gameClock": "PT5M12.00S",
            "homeTeam": {"players": players[:1]},
            "awayTeam": {"players": players[1:]},
        }
    }


def test_get_live_box_score_maps_players_and_clock_correctly():
    players = [
        {
            "personId": 2544,
            "firstName": "LeBron",
            "familyName": "James",
            "statistics": {"points": 24, "minutes": "PT30M00.00S"},
        },
        {
            "personId": 201939,
            "firstName": "Stephen",
            "familyName": "Curry",
            "statistics": {"points": 18, "minutes": "PT28M00.00S"},
        },
    ]
    fake_live = MagicMock()
    fake_live.get_dict.return_value = _live_box_dict(players)
    with patch("nba_api.live.nba.endpoints.boxscore.BoxScore", return_value=fake_live):
        box_score = NBAApiProvider().get_live_box_score("0022500123")

    assert box_score is not None
    assert box_score.game_id == "0022500123"
    assert box_score.period == 3
    assert box_score.game_clock == "PT5M12.00S"
    assert len(box_score.players) == 2
    lebron = next(p for p in box_score.players if p.player_id == 2544)
    assert lebron.first_name == "LeBron"
    assert lebron.last_name == "James"
    assert lebron.points == 24
    assert lebron.minutes == "PT30M00.00S"


def test_get_live_box_score_returns_none_on_failure():
    with patch(
        "nba_api.live.nba.endpoints.boxscore.BoxScore",
        side_effect=ConnectionError("down"),
    ):
        result = NBAApiProvider().get_live_box_score("0022500123")
    assert result is None


def test_get_live_box_score_defaults_missing_stats_like_pre_step7():
    players = [
        {"personId": 1, "firstName": "A", "familyName": "B"}
    ]  # no "statistics" key
    fake_live = MagicMock()
    fake_live.get_dict.return_value = _live_box_dict(players + [{}])
    with patch("nba_api.live.nba.endpoints.boxscore.BoxScore", return_value=fake_live):
        box_score = NBAApiProvider().get_live_box_score("G1")

    line = box_score.players[0]
    assert line.points == 0
    assert line.minutes == "0"
