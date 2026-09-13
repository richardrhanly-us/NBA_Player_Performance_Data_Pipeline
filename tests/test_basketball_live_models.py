"""
Tests for the Step 7 live-application canonical models
(PlayerDetails, ScheduledGame, LivePlayerStatLine, LiveBoxScore). No
network access.
"""

from dataclasses import FrozenInstanceError

import pytest

from src.data.basketball.models import (
    LiveBoxScore,
    LivePlayerStatLine,
    PlayerDetails,
)


def test_player_details_holds_expected_fields():
    details = PlayerDetails(
        player_id=2544,
        team_id=1610612747,
        team_name="Los Angeles Lakers",
        team_abbreviation="LAL",
        position="F",
    )
    assert details.player_id == 2544
    assert details.team_id == 1610612747
    assert details.team_name == "Los Angeles Lakers"
    assert details.team_abbreviation == "LAL"
    assert details.position == "F"


def test_player_details_is_immutable():
    details = PlayerDetails(
        player_id=1, team_id=None, team_name=None, team_abbreviation=None, position=None
    )
    with pytest.raises(FrozenInstanceError):
        details.team_name = "X"


def test_player_details_allows_missing_fields_as_none():
    details = PlayerDetails(
        player_id=1, team_id=None, team_name=None, team_abbreviation=None, position=None
    )
    assert details.team_id is None
    assert details.team_name is None


def test_scheduled_game_holds_expected_fields():
    from src.data.basketball.models import ScheduledGame

    game = ScheduledGame(
        game_id="0022500123",
        game_date="01/15/2026",
        home_team_id=1610612747,
        away_team_id=1610612738,
        game_status_text="7:30 pm ET",
    )
    assert game.game_id == "0022500123"
    assert game.game_date == "01/15/2026"
    assert game.home_team_id == 1610612747
    assert game.away_team_id == 1610612738
    assert game.game_status_text == "7:30 pm ET"


def test_scheduled_game_is_immutable():
    from src.data.basketball.models import ScheduledGame

    game = ScheduledGame(
        game_id="G1",
        game_date="01/01/2026",
        home_team_id=1,
        away_team_id=2,
        game_status_text=None,
    )
    with pytest.raises(FrozenInstanceError):
        game.game_id = "G2"


def test_live_player_stat_line_preserves_raw_points_and_minutes_unconverted():
    line = LivePlayerStatLine(
        player_id=2544,
        first_name="LeBron",
        last_name="James",
        points="12",
        minutes="PT8M32.00S",
    )
    assert line.points == "12"
    assert line.minutes == "PT8M32.00S"


def test_live_box_score_holds_period_clock_and_players():
    line = LivePlayerStatLine(
        player_id=2544,
        first_name="LeBron",
        last_name="James",
        points=12,
        minutes="8:32",
    )
    box_score = LiveBoxScore(
        game_id="0022500123", period=2, game_clock="PT8M32.00S", players=(line,)
    )
    assert box_score.game_id == "0022500123"
    assert box_score.period == 2
    assert box_score.game_clock == "PT8M32.00S"
    assert box_score.players == (line,)


def test_live_box_score_is_immutable():
    box_score = LiveBoxScore(game_id="G1", period=1, game_clock=None, players=())
    with pytest.raises(FrozenInstanceError):
        box_score.period = 2
