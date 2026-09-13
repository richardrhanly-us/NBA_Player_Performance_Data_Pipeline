"""
Tests for training/dataset.py: end-to-end build/persist round trip, no
(PLAYER_ID, GAME_ID) duplicates, no sportsbook columns, and CLI wiring.
All run against small in-memory fixtures written to tmp_path -- no live
NBA API calls, no dependency on the real training/data/raw/ contents.
"""

import pandas as pd
import pytest

from training import config, dataset
from training.data import storage


def _raw_row(season, game_id, date, team, opponent, is_home, player_id, pts):
    matchup = f"{team} vs {opponent}" if is_home else f"{team} @ {opponent}"
    return {
        "SEASON": season,
        "SEASON_ID": "2" + season[:4],
        "PLAYER_ID": player_id,
        "PLAYER_NAME": f"Player {player_id}",
        "GAME_ID": game_id,
        "GAME_DATE": pd.Timestamp(date),
        "MATCHUP": matchup,
        "TEAM_ABBREVIATION": team,
        "OPPONENT_ABBREVIATION": opponent,
        "IS_HOME": 1 if is_home else 0,
        "WL": "W",
        "MIN": 30,
        "FGM": 5,
        "FGA": 10,
        "FG_PCT": 0.5,
        "FG3M": 1,
        "FG3A": 3,
        "FG3_PCT": 0.3,
        "FTM": 2,
        "FTA": 2,
        "FT_PCT": 1.0,
        "OREB": 1,
        "DREB": 4,
        "REB": 5,
        "AST": 3,
        "STL": 1,
        "BLK": 0,
        "TOV": 2,
        "PF": 2,
        "PTS": pts,
        "PLUS_MINUS": pts - 10,
        "VIDEO_AVAILABLE": 1,
    }


@pytest.fixture(autouse=True)
def _isolated_raw_and_processed_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        dataset, "DEFAULT_OUTPUT_PATH", tmp_path / "processed" / "v1_panel.parquet"
    )


def _seed_raw_season(season, n_games=8):
    rows = []
    for i in range(n_games):
        date = pd.Timestamp("2023-10-24") + pd.Timedelta(days=2 * i)
        game_id = f"{season}-G{i}"
        rows.append(
            _raw_row(season, game_id, date, "AAA", "BBB", i % 2 == 0, 101, 10 + i)
        )
        rows.append(
            _raw_row(season, game_id, date, "BBB", "AAA", i % 2 == 1, 201, 12 + i)
        )
    df = pd.DataFrame(rows)
    for player_id in (101, 201):
        player_rows = df[df["PLAYER_ID"] == player_id]
        storage.save_player_gamelog(player_rows, season=season, player_id=player_id)


def test_build_dataset_produces_expected_row_count_and_no_duplicates():
    _seed_raw_season("2023-24", n_games=8)
    _seed_raw_season("2024-25", n_games=6)

    panel, report = dataset.build_dataset(seasons=["2023-24", "2024-25"])

    assert len(panel) == (8 + 6) * 2  # 2 players per game
    assert panel.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum() == 0
    assert report["duplicate_key_count"] == 0
    assert report["raw_rows_total"] == len(panel)


def test_build_dataset_assigns_splits_by_season():
    _seed_raw_season("2023-24", n_games=4)
    _seed_raw_season("2025-26", n_games=4)

    panel, _ = dataset.build_dataset(seasons=["2023-24", "2025-26"])

    assert set(panel.loc[panel["SEASON"] == "2023-24", "SPLIT"]) == {"train"}
    assert set(panel.loc[panel["SEASON"] == "2025-26", "SPLIT"]).issubset(
        {"validation", "test"}
    )


def test_no_sportsbook_columns_in_the_persisted_panel():
    _seed_raw_season("2023-24", n_games=4)
    panel, _ = dataset.build_dataset(seasons=["2023-24"])
    forbidden = {"closing_line", "sportsbook_line", "sportsbook", "odds", "edge"}
    assert forbidden.isdisjoint(set(panel.columns))


def test_save_and_reload_round_trip_preserves_dtypes_and_identifiers(tmp_path):
    _seed_raw_season("2023-24", n_games=4)
    panel, _ = dataset.build_dataset(seasons=["2023-24"])

    output_path = tmp_path / "output" / "panel.parquet"
    dataset.save_dataset(panel, output_path=output_path)
    assert output_path.exists()

    reloaded = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(reloaded, panel)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["GAME_DATE"])
    assert reloaded["PLAYER_ID"].notna().all()
    assert reloaded["GAME_ID"].notna().all()


def test_save_dataset_refuses_to_overwrite_without_the_flag(tmp_path):
    _seed_raw_season("2023-24", n_games=4)
    panel, _ = dataset.build_dataset(seasons=["2023-24"])

    output_path = tmp_path / "output" / "panel.parquet"
    dataset.save_dataset(panel, output_path=output_path)

    with pytest.raises(FileExistsError):
        dataset.save_dataset(panel, output_path=output_path, overwrite=False)

    # overwrite=True must succeed without raising.
    dataset.save_dataset(panel, output_path=output_path, overwrite=True)


def test_build_dataset_reports_cold_start_and_eligibility_counts():
    _seed_raw_season("2023-24", n_games=8)
    _, report = dataset.build_dataset(seasons=["2023-24"])

    assert (
        report["training_eligible_rows"] + report["rows_excluded_insufficient_history"]
        == report["processed_rows"]
    )
    assert sum(report["cold_start_histogram"].values()) == report["processed_rows"]


def test_cli_build_arg_parser_defaults_and_options():
    parser = dataset.build_arg_parser()
    args = parser.parse_args(["build"])
    assert args.season is None
    assert args.output is None
    assert args.overwrite is False

    args = parser.parse_args(
        [
            "build",
            "--season",
            "2023-24",
            "2024-25",
            "--output",
            "custom.parquet",
            "--overwrite",
        ]
    )
    assert args.season == ["2023-24", "2024-25"]
    assert args.output == "custom.parquet"
    assert args.overwrite is True


def test_main_with_no_subcommand_behaves_like_build(tmp_path, monkeypatch):
    _seed_raw_season("2023-24", n_games=8)
    output_path = tmp_path / "cli_output.parquet"
    monkeypatch.setattr(dataset, "DEFAULT_OUTPUT_PATH", output_path)

    exit_code = dataset.main([])

    assert exit_code == 0
    assert output_path.exists()
