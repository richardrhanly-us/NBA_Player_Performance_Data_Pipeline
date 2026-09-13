"""
Tests for src/data/basketball/normalization.py: the provider-payload <->
canonical-record boundary. No network access.
"""

import numpy as np
import pandas as pd

from src.data.basketball.models import Player, PlayerGameLog
from src.data.basketball.normalization import (
    dataframe_to_player_game_logs,
    dataframe_to_players,
)


def _normalized_gamelog_row(**overrides):
    row = {
        "SEASON": "2023-24",
        "SEASON_ID": "22023",
        "PLAYER_ID": 2544,
        "PLAYER_NAME": "LeBron James",
        "GAME_ID": "0022300001",
        "GAME_DATE": pd.Timestamp("2023-10-24"),
        "MATCHUP": "LAL vs DEN",
        "TEAM_ABBREVIATION": "LAL",
        "OPPONENT_ABBREVIATION": "DEN",
        "IS_HOME": 1,
        "WL": "W",
        "MIN": "38",
        "FGM": 10.0,
        "FGA": 20.0,
        "FG_PCT": 0.5,
        "FG3M": 2.0,
        "FG3A": 5.0,
        "FG3_PCT": 0.4,
        "FTM": 4.0,
        "FTA": 5.0,
        "FT_PCT": 0.8,
        "OREB": 1.0,
        "DREB": 6.0,
        "REB": 7.0,
        "AST": 8.0,
        "STL": 1.0,
        "BLK": 1.0,
        "TOV": 3.0,
        "PF": 2.0,
        "PTS": 26.0,
        "PLUS_MINUS": 10.0,
        "VIDEO_AVAILABLE": 1.0,
    }
    row.update(overrides)
    return row


def test_dataframe_to_player_game_logs_maps_fields_correctly():
    df = pd.DataFrame([_normalized_gamelog_row()])
    records = dataframe_to_player_game_logs(df)

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, PlayerGameLog)
    assert record.season == "2023-24"
    assert record.player_id == 2544
    assert record.player_name == "LeBron James"
    assert record.game_id == "0022300001"
    assert record.team_abbreviation == "LAL"
    assert record.opponent_abbreviation == "DEN"
    assert record.is_home == 1
    assert record.points == 26.0
    assert record.fga == 20.0


def test_dataframe_to_player_game_logs_is_independent_of_column_order():
    df = pd.DataFrame([_normalized_gamelog_row()])
    shuffled = df[list(reversed(df.columns))]

    records_original = dataframe_to_player_game_logs(df)
    records_shuffled = dataframe_to_player_game_logs(shuffled)

    assert records_original == records_shuffled


def test_dataframe_to_player_game_logs_preserves_row_order():
    df = pd.DataFrame(
        [
            _normalized_gamelog_row(GAME_ID="G1", PTS=10.0),
            _normalized_gamelog_row(GAME_ID="G2", PTS=20.0),
            _normalized_gamelog_row(GAME_ID="G3", PTS=30.0),
        ]
    )
    records = dataframe_to_player_game_logs(df)
    assert [r.game_id for r in records] == ["G1", "G2", "G3"]
    assert [r.points for r in records] == [10.0, 20.0, 30.0]


def test_dataframe_to_player_game_logs_handles_missing_values_as_none_not_fabricated():
    df = pd.DataFrame(
        [_normalized_gamelog_row(FG3A=np.nan, FG3M=np.nan, FG3_PCT=pd.NA)]
    )
    records = dataframe_to_player_game_logs(df)
    record = records[0]
    assert record.fg3a is None
    assert record.fg3m is None
    assert record.fg3_pct is None
    # Unrelated fields are unaffected by another field's missingness.
    assert record.points == 26.0


def test_dataframe_to_player_game_logs_empty_input_returns_empty_list():
    assert dataframe_to_player_game_logs(pd.DataFrame()) == []


def test_dataframe_to_player_game_logs_is_deterministic():
    df = pd.DataFrame(
        [_normalized_gamelog_row(GAME_ID="G1"), _normalized_gamelog_row(GAME_ID="G2")]
    )
    assert dataframe_to_player_game_logs(df) == dataframe_to_player_game_logs(df)


def test_dataframe_to_players_maps_fields_and_ignores_extra_columns():
    df = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "PLAYER_NAME": ["Alpha", "Beta"],
            "GP": [10, 20],  # extra column, must be ignored
            "SOME_OTHER_STAT": [1.0, 2.0],
        }
    )
    players = dataframe_to_players(df)
    assert players == [
        Player(player_id=1, player_name="Alpha"),
        Player(player_id=2, player_name="Beta"),
    ]


def test_dataframe_to_players_is_independent_of_column_order():
    df = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alpha"], "GP": [10]})
    shuffled = df[["GP", "PLAYER_NAME", "PLAYER_ID"]]
    assert dataframe_to_players(df) == dataframe_to_players(shuffled)


def test_dataframe_to_players_empty_input_returns_empty_list():
    assert dataframe_to_players(pd.DataFrame()) == []
