"""
Step 9 performance-metrics tests (test list items 18-23). Uses the
in-memory SQLite `db_conn` fixture, writing snapshots/outcomes directly
via the repository so these tests are independent of the settlement
service itself (already covered in test_settlement_service.py).
"""

from src.services.performance_service import (
    compute_edge_bucket_breakdown,
    compute_performance_summary,
    compute_qualified_vs_unqualified_breakdown,
)
from src.services.prediction_repository import insert_outcome, persist_prediction_run
from src.services.prediction_result import (
    PredictionDirection,
    PredictionResult,
    PredictionStatus,
)
from src.services.prediction_service import PredictionBoard


def _graded_board(rows, run_suffix=""):
    """rows: list of (player_id, projection, line, edge, direction) tuples."""
    predictions = []
    for i, (player_id, projection, line, edge, direction) in enumerate(rows):
        predictions.append(
            PredictionResult(
                player_name=f"Player {player_id}",
                status=PredictionStatus.OK,
                player_id=player_id,
                model_projection=projection,
                sportsbook_line=line,
                edge=edge,
                direction=direction,
                bookmaker="draftkings",
                model_version="v1",
                generated_at_utc=f"2026-01-{15 + i:02d}T12:00:00+00:00{run_suffix}",
                game_id=f"G{player_id}{run_suffix}",
                game_date="01/15/2026",
            )
        )
    return PredictionBoard(
        generated_at_utc=predictions[0].generated_at_utc,
        bookmaker="draftkings",
        model_version="v1",
        predictions=tuple(predictions),
        props_discovered=len(predictions),
        players_matched=len(predictions),
        predictions_generated=len(predictions),
        unmatched_count=0,
        unavailable_count=0,
    )


def _settle_all(db_conn, run_id, results_by_player_id):
    """results_by_player_id: {player_id: (actual_points, result_status)}"""
    from src.services.prediction_repository import get_snapshots_for_run

    for snapshot in get_snapshots_for_run(db_conn, run_id):
        actual_points, result_status = results_by_player_id[snapshot["player_id"]]
        insert_outcome(
            db_conn,
            prediction_snapshot_id=snapshot["id"],
            actual_points=actual_points,
            result_status=result_status,
            game_status="Final",
            source="test",
        )


def test_pushes_excluded_from_win_rate_denominator(db_conn):
    board = _graded_board(
        [
            (1, 22.0, 20.0, 2.0, PredictionDirection.OVER),  # WIN
            (2, 20.0, 20.0, 0.0, PredictionDirection.NEUTRAL),  # will settle PUSH
        ]
    )
    run = persist_prediction_run(board, db_conn)
    _settle_all(db_conn, run["run_id"], {1: (25.0, "WIN"), 2: (20.0, "PUSH")})

    summary = compute_performance_summary(db_conn)
    assert summary.graded == 2
    assert summary.wins == 1
    assert summary.pushes == 1
    assert summary.win_rate == 1.0  # 1 win / (1 win + 0 losses), push excluded


def test_pending_predictions_excluded_from_graded_record(db_conn):
    board = _graded_board([(1, 22.0, 20.0, 2.0, PredictionDirection.OVER)])
    persist_prediction_run(board, db_conn)
    # deliberately never settled

    summary = compute_performance_summary(db_conn)
    assert summary.graded == 0
    assert summary.win_rate is None


def test_qualified_statistics_calculated_separately(db_conn):
    board = _graded_board(
        [
            (1, 22.0, 20.0, 2.0, PredictionDirection.OVER),  # below threshold (3.0)
            (2, 28.0, 20.0, 8.0, PredictionDirection.OVER),  # qualified
        ]
    )
    run = persist_prediction_run(board, db_conn)
    _settle_all(db_conn, run["run_id"], {1: (25.0, "WIN"), 2: (10.0, "LOSS")})

    summary = compute_performance_summary(db_conn)
    assert summary.graded == 2
    assert summary.qualified_graded == 1
    assert summary.qualified_losses == 1
    assert summary.qualified_win_rate == 0.0
    assert summary.win_rate == 0.5  # overall: 1 win, 1 loss


def test_over_under_filtering(db_conn):
    board = _graded_board(
        [
            (1, 22.0, 20.0, 2.0, PredictionDirection.OVER),
            (2, 18.0, 20.0, -2.0, PredictionDirection.UNDER),
        ]
    )
    run = persist_prediction_run(board, db_conn)
    _settle_all(db_conn, run["run_id"], {1: (25.0, "WIN"), 2: (15.0, "WIN")})

    over_summary = compute_performance_summary(db_conn, direction="OVER")
    under_summary = compute_performance_summary(db_conn, direction="UNDER")
    assert over_summary.graded == 1
    assert under_summary.graded == 1


def test_date_range_filtering(db_conn):
    board = _graded_board([(1, 22.0, 20.0, 2.0, PredictionDirection.OVER)])
    run = persist_prediction_run(board, db_conn)
    _settle_all(db_conn, run["run_id"], {1: (25.0, "WIN")})

    in_range = compute_performance_summary(
        db_conn,
        start_date="2026-01-01T00:00:00+00:00",
        end_date="2026-12-31T00:00:00+00:00",
    )
    out_of_range = compute_performance_summary(
        db_conn,
        start_date="2027-01-01T00:00:00+00:00",
        end_date="2027-12-31T00:00:00+00:00",
    )
    assert in_range.graded == 1
    assert out_of_range.graded == 0


def test_edge_bucket_calculation(db_conn):
    board = _graded_board(
        [
            (1, 20.5, 20.0, 0.5, PredictionDirection.OVER),  # bucket 0-1
            (2, 22.5, 20.0, 2.5, PredictionDirection.OVER),  # bucket 2-3
            (3, 29.0, 20.0, 9.0, PredictionDirection.OVER),  # bucket 4+
        ]
    )
    run = persist_prediction_run(board, db_conn)
    _settle_all(
        db_conn, run["run_id"], {1: (25.0, "WIN"), 2: (10.0, "LOSS"), 3: (30.0, "WIN")}
    )

    buckets = compute_edge_bucket_breakdown(db_conn)
    by_label = {b["bucket"]: b for b in buckets}
    assert by_label["0-1"]["n"] == 1
    assert by_label["2-3"]["n"] == 1
    assert by_label["4+"]["n"] == 1
    assert by_label["1-2"]["n"] == 0


def test_qualified_vs_unqualified_clean_comparison(db_conn):
    board = _graded_board(
        [
            (1, 22.0, 20.0, 2.0, PredictionDirection.OVER),  # below 3.0
            (2, 28.0, 20.0, 8.0, PredictionDirection.OVER),  # >= 3.0
        ]
    )
    run = persist_prediction_run(board, db_conn)
    _settle_all(db_conn, run["run_id"], {1: (25.0, "WIN"), 2: (10.0, "LOSS")})

    breakdown = compute_qualified_vs_unqualified_breakdown(db_conn)
    assert breakdown["edge_threshold"] == 3.0
    assert breakdown["below_threshold"]["n"] == 1
    assert breakdown["below_threshold"]["win_rate"] == 1.0
    assert breakdown["at_or_above_threshold"]["n"] == 1
    assert breakdown["at_or_above_threshold"]["win_rate"] == 0.0


def test_no_action_and_unavailable_excluded_from_win_loss_counts_but_visible(db_conn):
    board = _graded_board([(1, 22.0, 20.0, 2.0, PredictionDirection.OVER)])
    run = persist_prediction_run(board, db_conn)
    _settle_all(db_conn, run["run_id"], {1: (None, "NO_ACTION")})

    summary = compute_performance_summary(db_conn)
    assert summary.graded == 0
    assert summary.no_action_count == 1
