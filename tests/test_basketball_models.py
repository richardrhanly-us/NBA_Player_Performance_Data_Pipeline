"""
Tests for src/data/basketball/models.py: canonical records have
deterministic, well-defined field/type behavior. No network access.
"""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from src.data.basketball.models import Player, PlayerGameLog


def test_player_is_immutable_and_holds_expected_fields():
    player = Player(player_id=2544, player_name="LeBron James")
    assert player.player_id == 2544
    assert player.player_name == "LeBron James"
    with pytest.raises(FrozenInstanceError):
        player.player_id = 999


def test_player_game_log_holds_all_v1_relevant_fields():
    log = PlayerGameLog(
        season="2023-24",
        season_id="22023",
        player_id=2544,
        player_name="LeBron James",
        game_id="0022300001",
        game_date=pd.Timestamp("2023-10-24"),
        matchup="LAL vs DEN",
        team_abbreviation="LAL",
        opponent_abbreviation="DEN",
        is_home=1,
        wl="W",
        minutes="38",
        fgm=10,
        fga=20,
        fg_pct=0.5,
        fg3m=2,
        fg3a=5,
        fg3_pct=0.4,
        ftm=4,
        fta=5,
        ft_pct=0.8,
        oreb=1,
        dreb=6,
        reb=7,
        ast=8,
        stl=1,
        blk=1,
        tov=3,
        pf=2,
        points=26,
        plus_minus=10,
        video_available=1,
    )
    assert log.season == "2023-24"
    assert log.player_id == 2544
    assert log.points == 26
    assert log.team_abbreviation == "LAL"
    assert log.opponent_abbreviation == "DEN"


def test_player_game_log_is_immutable():
    log = PlayerGameLog(
        season="2023-24",
        season_id=None,
        player_id=1,
        player_name="X",
        game_id="G1",
        game_date=None,
        matchup="",
        team_abbreviation=None,
        opponent_abbreviation=None,
        is_home=None,
        wl=None,
        minutes=None,
        fgm=None,
        fga=None,
        fg_pct=None,
        fg3m=None,
        fg3a=None,
        fg3_pct=None,
        ftm=None,
        fta=None,
        ft_pct=None,
        oreb=None,
        dreb=None,
        reb=None,
        ast=None,
        stl=None,
        blk=None,
        tov=None,
        pf=None,
        points=None,
        plus_minus=None,
        video_available=None,
    )
    with pytest.raises(FrozenInstanceError):
        log.points = 100


def test_player_game_log_allows_missing_values_as_none():
    """Optional fields being None reflects real source gaps -- must not
    raise or silently coerce to a fabricated value."""
    log = PlayerGameLog(
        season="2023-24",
        season_id=None,
        player_id=1,
        player_name="X",
        game_id="G1",
        game_date=None,
        matchup="",
        team_abbreviation=None,
        opponent_abbreviation=None,
        is_home=None,
        wl=None,
        minutes=None,
        fgm=None,
        fga=None,
        fg_pct=None,
        fg3m=None,
        fg3a=None,
        fg3_pct=None,
        ftm=None,
        fta=None,
        ft_pct=None,
        oreb=None,
        dreb=None,
        reb=None,
        ast=None,
        stl=None,
        blk=None,
        tov=None,
        pf=None,
        points=None,
        plus_minus=None,
        video_available=None,
    )
    assert log.points is None
    assert log.game_date is None
