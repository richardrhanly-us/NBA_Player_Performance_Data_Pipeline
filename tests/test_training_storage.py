"""
Storage-layer tests: raw schema shape, dedup, save/load round trip, the
resumability check, and season-level concatenation. No network access.
"""

import pandas as pd
import pytest

from training import config
from training.data import storage


@pytest.fixture(autouse=True)
def _isolated_raw_dir(tmp_path, monkeypatch):
    """Every test in this file writes under a throwaway tmp_path, never the
    real training/data/raw/."""
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")


def _raw_playergamelog_df(rows):
    """
    Build a DataFrame shaped exactly like nba_api's real PlayerGameLog
    output (native column names/casing, including the Player_ID/Game_ID
    inconsistency), from a list of simple per-row dicts.
    """
    columns = [
        "SEASON_ID",
        "Player_ID",
        "Game_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "PLUS_MINUS",
        "VIDEO_AVAILABLE",
    ]
    defaults = {
        "SEASON_ID": "22023",
        "MIN": "30",
        "FGM": 8,
        "FGA": 16,
        "FG_PCT": 0.5,
        "FG3M": 2,
        "FG3A": 5,
        "FG3_PCT": 0.4,
        "FTM": 4,
        "FTA": 5,
        "FT_PCT": 0.8,
        "OREB": 1,
        "DREB": 5,
        "REB": 6,
        "AST": 4,
        "STL": 1,
        "BLK": 0,
        "TOV": 2,
        "PF": 2,
        "PTS": 22,
        "PLUS_MINUS": 5,
        "VIDEO_AVAILABLE": 1,
    }
    data = []
    for row in rows:
        merged = {**defaults, **row}
        data.append([merged[c] for c in columns])
    return pd.DataFrame(data, columns=columns)


def test_normalize_raw_gamelog_produces_expected_schema_and_order():
    raw = _raw_playergamelog_df(
        [
            {
                "Player_ID": 2544,
                "Game_ID": "0022300001",
                "GAME_DATE": "2023-10-24",
                "MATCHUP": "LAL vs DEN",
                "WL": "W",
                "MIN": "38",
            },
        ]
    )

    result = storage.normalize_raw_gamelog(
        raw, season="2023-24", player_id=2544, player_name="LeBron James"
    )

    assert list(result.columns) == list(storage.RAW_GAMELOG_COLUMNS)
    assert len(result) == 1

    row = result.iloc[0]
    assert row["SEASON"] == "2023-24"
    assert row["PLAYER_ID"] == 2544
    assert row["PLAYER_NAME"] == "LeBron James"
    assert row["GAME_ID"] == "0022300001"
    assert row["TEAM_ABBREVIATION"] == "LAL"
    assert row["OPPONENT_ABBREVIATION"] == "DEN"
    assert row["IS_HOME"] == 1
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])


def test_normalize_raw_gamelog_derives_away_game_correctly():
    raw = _raw_playergamelog_df(
        [
            {
                "Player_ID": 2544,
                "Game_ID": "0022300002",
                "GAME_DATE": "2023-10-26",
                "MATCHUP": "LAL @ PHX",
                "WL": "L",
                "MIN": "35",
            },
        ]
    )

    result = storage.normalize_raw_gamelog(
        raw, season="2023-24", player_id=2544, player_name="LeBron James"
    )

    row = result.iloc[0]
    assert row["TEAM_ABBREVIATION"] == "LAL"
    assert row["OPPONENT_ABBREVIATION"] == "PHX"
    assert row["IS_HOME"] == 0


def test_normalize_raw_gamelog_deduplicates_by_player_and_game_id():
    raw = _raw_playergamelog_df(
        [
            {
                "Player_ID": 2544,
                "Game_ID": "0022300001",
                "GAME_DATE": "2023-10-24",
                "MATCHUP": "LAL vs DEN",
                "WL": "W",
            },
            {
                "Player_ID": 2544,
                "Game_ID": "0022300001",
                "GAME_DATE": "2023-10-24",
                "MATCHUP": "LAL vs DEN",
                "WL": "W",
            },
            {
                "Player_ID": 2544,
                "Game_ID": "0022300002",
                "GAME_DATE": "2023-10-26",
                "MATCHUP": "LAL @ PHX",
                "WL": "L",
            },
        ]
    )

    result = storage.normalize_raw_gamelog(
        raw, season="2023-24", player_id=2544, player_name="LeBron James"
    )

    assert len(result) == 2
    assert sorted(result["GAME_ID"]) == ["0022300001", "0022300002"]


def test_deduplicate_keeps_first_occurrence_of_duplicate_key():
    df = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 2],
            "GAME_ID": ["A", "A", "B"],
            "PTS": [
                10,
                999,
                20,
            ],  # duplicate row has a different PTS to prove "first" wins
        }
    )

    result = storage.deduplicate(df)

    assert len(result) == 2
    row_a = result[result["GAME_ID"] == "A"].iloc[0]
    assert row_a["PTS"] == 10


def test_save_and_load_player_gamelog_round_trip_preserves_types(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")

    raw = _raw_playergamelog_df(
        [
            {
                "Player_ID": 2544,
                "Game_ID": "0022300001",
                "GAME_DATE": "2023-10-24",
                "MATCHUP": "LAL vs DEN",
                "WL": "W",
            },
        ]
    )
    normalized = storage.normalize_raw_gamelog(
        raw, season="2023-24", player_id=2544, player_name="LeBron James"
    )

    path = storage.save_player_gamelog(normalized, season="2023-24", player_id=2544)
    assert path.exists()

    loaded = storage.load_player_gamelog("2023-24", 2544)

    assert list(loaded.columns) == list(storage.RAW_GAMELOG_COLUMNS)
    assert loaded["PLAYER_ID"].iloc[0] == 2544
    assert loaded["GAME_ID"].iloc[0] == "0022300001"
    assert pd.api.types.is_datetime64_any_dtype(loaded["GAME_DATE"])


def test_is_player_collected_false_when_no_file_true_after_save():
    assert storage.is_player_collected("2023-24", 999) is False

    raw = _raw_playergamelog_df(
        [
            {
                "Player_ID": 999,
                "Game_ID": "0022300099",
                "GAME_DATE": "2023-11-01",
                "MATCHUP": "BOS vs MIA",
                "WL": "W",
            },
        ]
    )
    normalized = storage.normalize_raw_gamelog(
        raw, season="2023-24", player_id=999, player_name="Test Player"
    )
    storage.save_player_gamelog(normalized, season="2023-24", player_id=999)

    assert storage.is_player_collected("2023-24", 999) is True


def test_is_player_collected_false_for_corrupt_file():
    path = storage.player_file_path("2023-24", 555)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real parquet file")

    assert storage.is_player_collected("2023-24", 555) is False


def test_load_season_gamelogs_concatenates_and_dedupes_across_players():
    for player_id, game_id in [(1, "A"), (2, "B")]:
        raw = _raw_playergamelog_df(
            [
                {
                    "Player_ID": player_id,
                    "Game_ID": game_id,
                    "GAME_DATE": "2023-10-24",
                    "MATCHUP": "BOS vs MIA",
                    "WL": "W",
                },
            ]
        )
        normalized = storage.normalize_raw_gamelog(
            raw,
            season="2023-24",
            player_id=player_id,
            player_name=f"Player {player_id}",
        )
        storage.save_player_gamelog(normalized, season="2023-24", player_id=player_id)

    combined = storage.load_season_gamelogs("2023-24")

    assert len(combined) == 2
    assert set(combined["PLAYER_ID"]) == {1, 2}


def test_load_season_gamelogs_empty_when_nothing_collected():
    combined = storage.load_season_gamelogs("2099-00")
    assert combined.empty
    assert list(combined.columns) == list(storage.RAW_GAMELOG_COLUMNS)


def test_manifest_round_trip():
    manifest = storage.load_manifest("2023-24")
    assert manifest["completed_player_ids"] == []

    manifest["completed_player_ids"] = [1, 2, 3]
    manifest["rows_collected"] = 150
    storage.save_manifest("2023-24", manifest)

    reloaded = storage.load_manifest("2023-24")
    assert reloaded["completed_player_ids"] == [1, 2, 3]
    assert reloaded["rows_collected"] == 150


def test_manifest_missing_file_returns_empty_defaults():
    manifest = storage.load_manifest("2099-00")
    assert manifest["season"] == "2099-00"
    assert manifest["players_succeeded"] == 0
    assert manifest["failed_players"] == {}


def test_gamelogs_to_dataframe_round_trips_normalize_raw_gamelog_exactly():
    """
    Step 6 provider-refactor safety net: normalize_raw_gamelog (the
    pre-refactor path) and dataframe_to_player_game_logs ->
    gamelogs_to_dataframe (the new provider/canonical-record path) must
    produce byte-identical output for the same underlying raw data --
    both funnel through the same _finalize_gamelog_frame tail.
    """
    from src.data.basketball.normalization import dataframe_to_player_game_logs

    raw = _raw_playergamelog_df(
        [
            {
                "Player_ID": 2544,
                "Game_ID": "0022300001",
                "GAME_DATE": "2023-10-24",
                "MATCHUP": "LAL vs DEN",
                "WL": "W",
                "MIN": "38",
            },
            {
                "Player_ID": 2544,
                "Game_ID": "0022300002",
                "GAME_DATE": "2023-10-26",
                "MATCHUP": "LAL @ PHX",
                "WL": "L",
                "MIN": "35",
            },
        ]
    )

    old_path = storage.normalize_raw_gamelog(
        raw, season="2023-24", player_id=2544, player_name="LeBron James"
    )

    records = dataframe_to_player_game_logs(old_path)
    new_path = storage.gamelogs_to_dataframe(records)

    pd.testing.assert_frame_equal(old_path, new_path)


def test_gamelogs_to_dataframe_empty_input_matches_normalize_raw_gamelog_empty_output():
    empty_via_normalize = storage.normalize_raw_gamelog(
        pd.DataFrame(), season="2023-24", player_id=1, player_name="X"
    )
    empty_via_records = storage.gamelogs_to_dataframe([])
    pd.testing.assert_frame_equal(empty_via_normalize, empty_via_records)
