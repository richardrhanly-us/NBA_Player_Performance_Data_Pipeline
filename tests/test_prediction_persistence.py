"""
Tests for src/services/persistence.py's minimal local board-snapshot
writer. No network access.
"""

import json

from src.services.persistence import save_board_snapshot
from src.services.prediction_result import (
    PredictionDirection,
    PredictionResult,
    PredictionStatus,
)
from src.services.prediction_service import PredictionBoard


def _sample_board():
    result = PredictionResult(
        player_name="Test Player",
        status=PredictionStatus.OK,
        player_id=1,
        team_abbreviation="LAL",
        model_projection=22.5,
        sportsbook_line=20.0,
        edge=2.5,
        direction=PredictionDirection.OVER,
        bookmaker="draftkings",
        model_version="points_regression.pkl@2026-01-01T00:00:00+00:00",
        generated_at_utc="2026-01-15T12:00:00+00:00",
        latest_game_date="2026-01-10",
    )
    return PredictionBoard(
        generated_at_utc="2026-01-15T120000Z",
        bookmaker="draftkings",
        model_version="points_regression.pkl@2026-01-01T00:00:00+00:00",
        predictions=(result,),
        props_discovered=1,
        players_matched=1,
        predictions_generated=1,
        unmatched_count=0,
        unavailable_count=0,
    )


def test_save_board_snapshot_writes_a_readable_json_file(tmp_path):
    board = _sample_board()
    path = save_board_snapshot(board, output_dir=tmp_path)

    assert path.exists()
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["props_discovered"] == 1
    assert payload["predictions_generated"] == 1
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["player_name"] == "Test Player"
    assert payload["predictions"][0]["status"] == "ok"
    assert payload["predictions"][0]["direction"] == "OVER"
    assert payload["predictions"][0]["edge"] == 2.5


def test_save_board_snapshot_creates_output_dir_if_missing(tmp_path):
    nested_dir = tmp_path / "nested" / "snapshots"
    board = _sample_board()
    path = save_board_snapshot(board, output_dir=nested_dir)
    assert path.exists()
    assert path.parent == nested_dir


def test_save_board_snapshot_empty_board_round_trips(tmp_path):
    board = PredictionBoard(
        generated_at_utc="2026-01-15T120000Z",
        bookmaker="draftkings",
        model_version="v1",
        predictions=(),
        props_discovered=0,
        players_matched=0,
        predictions_generated=0,
        unmatched_count=0,
        unavailable_count=0,
    )
    path = save_board_snapshot(board, output_dir=tmp_path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["predictions"] == []
