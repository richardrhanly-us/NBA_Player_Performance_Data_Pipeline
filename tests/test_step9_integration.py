"""
Step 9 integration tests (test list items 24-30 -- the ones not already
covered by test_prediction_repository.py/test_settlement_service.py/
test_performance_service.py). No live network dependency.
"""

from src.services.prediction_repository import get_snapshots_for_run
from src.services.prediction_service import build_daily_prediction_board
from tests.test_prediction_service_board import _board_provider, _prop_row, _props_df
from tests.test_prediction_service_single import _FakeModel

# ---- 24. daily board can persist successfully ------------------------------


def test_daily_board_persists_successfully(db_conn):
    from src.services.prediction_repository import persist_prediction_run

    provider = _board_provider({1: ("Player One", 10), 2: ("Player Two", 10)})
    props_df = _props_df([_prop_row("Player One", 15.0), _prop_row("Player Two", 25.0)])

    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    result = persist_prediction_run(board, db_conn)
    assert result["created"] is True

    snapshots = get_snapshots_for_run(db_conn, result["run_id"])
    assert len(snapshots) == 2


# ---- 25. ordinary Streamlit rerun does not create new history -------------


def test_identical_rerun_of_the_same_board_does_not_create_new_history(db_conn):
    """
    Simulates an ordinary Streamlit rerun: the SAME already-built board
    object (as Streamlit's own cache_data would return on a rerun within
    its TTL) is "submitted" for persistence again. This must not create
    a second run -- only an explicit, content-different board would.
    """
    from src.services.prediction_repository import persist_prediction_run

    provider = _board_provider({1: ("Player One", 10)})
    props_df = _props_df([_prop_row("Player One", 15.0)])

    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    first = persist_prediction_run(board, db_conn)
    # Simulate 5 more "reruns" all handed the identical cached board object.
    for _ in range(5):
        again = persist_prediction_run(board, db_conn)
        assert again["run_id"] == first["run_id"]
        assert again["created"] is False

    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 1


def test_persist_is_a_separate_explicit_call_not_bundled_into_board_build():
    """Structural guard: build_daily_prediction_board() itself never
    touches the database -- persistence is a deliberate, separate call
    (persist_prediction_run), matching the "explicit action" design the
    Step 9 brief asked for."""
    import inspect

    import src.services.prediction_service as service_module

    source = inspect.getsource(service_module.build_daily_prediction_board)
    assert "persist_prediction_run" not in source
    assert "conn" not in source


# ---- 26 (Edge Board still works) / 27 (manual prediction still works) are
# already exercised end-to-end by test_prediction_service_board.py and
# test_prediction_service_parity.py respectively -- re-running the full
# suite (see the Step 9 report) is the actual proof; nothing Step-9-
# specific changed either code path, confirmed by the git diff.


# ---- 30. model artifacts unchanged (a second confirmation, in addition to
# tests/test_step7_static_guard.py's registry-pointer check) -------------


def test_model_registry_and_legacy_model_path_unreferenced_by_new_modules():
    import pathlib

    step9_modules = [
        "src/services/db_connection.py",
        "src/services/migrations.py",
        "src/services/schema_sqlite.py",
        "src/services/prediction_repository.py",
        "src/services/settlement_service.py",
        "src/services/performance_service.py",
    ]
    forbidden = ("points_regression.pkl", "CURRENT_V1", "MODEL_REGISTRY_DIR")
    for rel_path in step9_modules:
        text = pathlib.Path(rel_path).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{rel_path} unexpectedly references {token!r}"
