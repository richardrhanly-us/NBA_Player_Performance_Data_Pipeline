"""
Step 9 schema/persistence tests (test list items 1-6): prediction runs
and snapshots persist correctly, immutably, and idempotently. Uses the
in-memory SQLite `db_conn` fixture (tests/conftest.py) -- real SQL
constraint enforcement, no live database, no network access.
"""

import sqlite3

import pytest

from src.services.prediction_repository import (
    compute_board_idempotency_key,
    get_pending_snapshots,
    get_run_by_idempotency_key,
    get_snapshots_for_run,
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
        "predictions_generated": sum(
            1 for p in predictions if p.status == PredictionStatus.OK
        ),
        "unmatched_count": sum(
            1 for p in predictions if p.status == PredictionStatus.UNMATCHED
        ),
        "unavailable_count": 0,
    }
    defaults.update(overrides)
    return PredictionBoard(**defaults)


# ---- 1. prediction run persists --------------------------------------------


def test_prediction_run_persists(db_conn):
    board = _board([_result("Player A")])
    result = persist_prediction_run(board, db_conn)
    assert result["created"] is True
    assert result["run_id"] is not None

    run = get_run_by_idempotency_key(db_conn, compute_board_idempotency_key(board))
    assert run is not None
    assert run["model_version"] == "v1"
    assert run["bookmaker"] == "draftkings"
    assert run["props_discovered"] == 1


# ---- 2. multiple snapshots persist under one run ---------------------------


def test_multiple_snapshots_persist_under_one_run(db_conn):
    board = _board(
        [
            _result("Player A", player_id=1),
            _result("Player B", player_id=2),
            _result("Player C", player_id=3),
        ]
    )
    result = persist_prediction_run(board, db_conn)

    snapshots = get_snapshots_for_run(db_conn, result["run_id"])
    assert len(snapshots) == 3
    assert {s["player_name"] for s in snapshots} == {"Player A", "Player B", "Player C"}
    assert all(s["prediction_run_id"] == result["run_id"] for s in snapshots)


# ---- 3. snapshot prediction fields are immutable through normal service API -


def test_no_update_statement_exists_for_snapshot_fields(db_conn):
    """
    Structural guard: the repository module's normal API has no function
    that updates prediction_snapshots' own fields. Persisting the same
    logical prediction again (even with different values) must produce
    a NEW row (new run + new snapshot), never an edit to the old one.
    """
    board_v1 = _board(
        [_result("Player A", model_projection=20.0, sportsbook_line=18.0, edge=2.0)]
    )
    persist_prediction_run(board_v1, db_conn)

    board_v2 = _board(
        [_result("Player A", model_projection=25.0, sportsbook_line=18.0, edge=7.0)]
    )
    persist_prediction_run(board_v2, db_conn)

    cur = db_conn.execute(
        "SELECT model_projection, edge FROM prediction_snapshots ORDER BY id"
    )
    rows = cur.fetchall()
    assert len(rows) == 2  # both preserved, neither overwritten
    assert rows[0] == (20.0, 2.0)
    assert rows[1] == (25.0, 7.0)


def test_prediction_repository_module_has_no_update_function():
    import src.services.prediction_repository as repo_module

    public_functions = [name for name in dir(repo_module) if not name.startswith("_")]
    assert not any("update" in name.lower() for name in public_functions)


# ---- 4. duplicate retry does not create duplicate run/snapshots -----------


def test_duplicate_retry_does_not_create_duplicate_run(db_conn):
    board = _board([_result("Player A"), _result("Player B", player_id=2)])

    first = persist_prediction_run(board, db_conn)
    second = persist_prediction_run(
        board, db_conn
    )  # simulated retry, identical content

    assert first["created"] is True
    assert second["created"] is False
    assert first["run_id"] == second["run_id"]

    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 1
    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_snapshots")
    assert cur.fetchone()[0] == 2  # not 4


# ---- 5. genuinely later run can store a changed line -----------------------


def test_later_run_with_changed_line_creates_a_new_run(db_conn):
    board_v1 = _board([_result("Player A", sportsbook_line=20.0, edge=2.5)])
    board_v2 = _board(
        [_result("Player A", sportsbook_line=23.5, edge=-1.0)]
    )  # line moved

    first = persist_prediction_run(board_v1, db_conn)
    second = persist_prediction_run(board_v2, db_conn)

    assert second["created"] is True
    assert first["run_id"] != second["run_id"]

    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 2

    # The original 2:00pm-style snapshot is untouched by the later run.
    original_snapshots = get_snapshots_for_run(db_conn, first["run_id"])
    assert original_snapshots[0]["sportsbook_line"] == 20.0


# ---- 6. transaction failure does not create a misleading partial run ------


def test_transaction_failure_leaves_no_partial_run(db_conn):
    """
    A snapshot insert that violates a constraint mid-run (simulated here
    with a NOT NULL violation on the second player) must roll back the
    ENTIRE run -- not leave a prediction_runs row with only the first
    snapshot attached.
    """
    good = _result("Player A")
    # model_version is NOT NULL in the schema; force a violation to
    # simulate an unexpected failure partway through persisting a run.
    bad = _result("Player B", player_id=2, model_version=None)
    board = _board([good, bad])

    with pytest.raises(sqlite3.IntegrityError):
        persist_prediction_run(board, db_conn)

    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 0
    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_snapshots")
    assert cur.fetchone()[0] == 0


def test_get_pending_snapshots_only_returns_ok_status_without_outcome(db_conn):
    board = _board(
        [
            _result("Player A", player_id=1),
            _result(
                "Player B",
                player_id=2,
                status=PredictionStatus.UNMATCHED,
                model_projection=None,
                sportsbook_line=None,
                edge=None,
                direction=None,
            ),
        ]
    )
    persist_prediction_run(board, db_conn)

    pending = get_pending_snapshots(db_conn)
    assert len(pending) == 1
    assert pending[0]["player_name"] == "Player A"
