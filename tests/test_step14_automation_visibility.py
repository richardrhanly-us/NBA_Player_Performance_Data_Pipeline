"""
Step 14 Phase 6: operational visibility for prediction/settlement
automation -- get_latest_run_by_status() and
get_latest_settlement_activity() in src/services/prediction_repository.py.
"""

from src.services.prediction_repository import (
    get_latest_run_by_status,
    get_latest_settlement_activity,
    get_snapshots_for_run,
    insert_outcome,
    persist_prediction_run,
)
from src.services.prediction_result import (
    PredictionDirection,
    PredictionResult,
    PredictionStatus,
)
from src.services.prediction_service import PredictionBoard


def _result(name, **overrides):
    defaults = {
        "player_name": name,
        "status": PredictionStatus.OK,
        "player_id": hash(name) % 100000,
        "model_projection": 22.5,
        "sportsbook_line": 20.0,
        "edge": 2.5,
        "direction": PredictionDirection.OVER,
        "bookmaker": "draftkings",
        "model_version": "v1",
        "generated_at_utc": "2026-01-15T12:00:00+00:00",
        "game_id": "G1",
        "game_date": "01/15/2026",
    }
    defaults.update(overrides)
    return PredictionResult(**defaults)


def _board(predictions, **overrides):
    defaults = {
        "generated_at_utc": "2026-01-15T12:00:00+00:00",
        "bookmaker": "draftkings",
        "model_version": "v1",
        "predictions": tuple(predictions),
        "props_discovered": len(predictions),
        "players_matched": len(predictions),
        "predictions_generated": sum(1 for p in predictions if p.status == PredictionStatus.OK),
        "unmatched_count": sum(1 for p in predictions if p.status == PredictionStatus.UNMATCHED),
        "unavailable_count": 0,
    }
    defaults.update(overrides)
    return PredictionBoard(**defaults)


def test_no_runs_returns_none_for_any_status(db_conn):
    assert get_latest_run_by_status(db_conn, "SUCCESS") is None
    assert get_latest_run_by_status(db_conn, "FAILED") is None


def test_latest_run_by_status_distinguishes_success_from_failed(db_conn):
    success_board = _board(
        [_result("Player A")], generated_at_utc="2026-01-15T12:00:00+00:00"
    )
    persist_prediction_run(success_board, db_conn)

    # A FAILED run per _run_status_for(): props discovered but zero
    # predictions actually generated.
    failing_result = _result("Player B", status=PredictionStatus.UNMATCHED)
    failed_run_board = _board(
        [failing_result],
        generated_at_utc="2026-01-15T14:00:00+00:00",
        predictions_generated=0,
        unmatched_count=1,
    )
    persist_prediction_run(failed_run_board, db_conn)

    latest_success = get_latest_run_by_status(db_conn, "SUCCESS")
    assert latest_success is not None
    assert latest_success["generated_at_utc"] == "2026-01-15T12:00:00+00:00"

    latest_failed = get_latest_run_by_status(db_conn, "FAILED")
    assert latest_failed is not None
    assert latest_failed["generated_at_utc"] == "2026-01-15T14:00:00+00:00"


def test_settlement_activity_is_none_before_any_settlement(db_conn):
    assert get_latest_settlement_activity(db_conn) is None


def test_settlement_activity_reflects_most_recent_settled_at(db_conn):
    board = _board([_result("Player A"), _result("Player B", player_id=999)])
    persist_prediction_run(board, db_conn)
    snapshots = get_snapshots_for_run(db_conn, 1)

    insert_outcome(
        db_conn,
        prediction_snapshot_id=snapshots[0]["id"],
        actual_points=25.0,
        result_status="WIN",
        source="test",
        settled_at="2026-01-16T08:00:00+00:00",
    )
    insert_outcome(
        db_conn,
        prediction_snapshot_id=snapshots[1]["id"],
        actual_points=18.0,
        result_status="LOSS",
        source="test",
        settled_at="2026-01-16T09:30:00+00:00",
    )

    latest = get_latest_settlement_activity(db_conn)
    assert latest == "2026-01-16T09:30:00+00:00"
