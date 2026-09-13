"""
Step 10 tests (test list items 1-30): automated prediction cycle,
settlement cycle, and the surrounding operational guarantees (schema
readiness, secret hygiene, exit codes, concurrency/idempotency,
retry/timeout behavior, stale-board classification, and static guards
against training/model/provider-boundary drift). No live network
dependency anywhere in this file.
"""

import importlib.util
import pathlib
import sqlite3

import pytest

from src.data.basketball.models import Player, PlayerDetails, ScheduledGame
from src.services.orchestration import (
    BoardFreshness,
    CycleStatus,
    _fetch_props_with_retry,
    classify_game_status,
    compute_board_freshness,
    run_prediction_cycle,
    run_settlement_cycle,
)
from src.services.prediction_repository import (
    get_snapshots_for_run,
    persist_prediction_run,
)
from tests.test_prediction_service_board import _prop_row, _props_df
from tests.test_prediction_service_single import (
    FakeProvider,
    _FakeModel,
    _sufficient_history,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _make_provider(players, scoreboard=()):
    """players: dict player_id -> (name, n_games). Team ids follow the
    same 1000+player_id convention as tests/test_prediction_service_board.py
    ::_board_provider, so a ScheduledGame(home_team_id=1000+pid, ...)
    matches that player's game."""
    active = [
        Player(player_id=pid, player_name=name) for pid, (name, _) in players.items()
    ]
    game_logs = {pid: _sufficient_history(pid, n) for pid, (_, n) in players.items()}
    details = {
        pid: PlayerDetails(
            player_id=pid,
            team_id=1000 + pid,
            team_name=f"Team{pid}",
            team_abbreviation=f"T{pid}",
            position="F",
        )
        for pid in players
    }
    return FakeProvider(
        active_players=active,
        game_logs=game_logs,
        player_details=details,
        scoreboard=list(scoreboard),
    )


def _game_for(pid, status_text, game_id=None):
    return ScheduledGame(
        game_id=game_id or f"G{pid}",
        game_date="01/15/2026",
        home_team_id=1000 + pid,
        away_team_id=9999,
        game_status_text=status_text,
    )


PREGAME_STATUS = "7:30 pm ET"
LIVE_STATUS = "Q3 5:42"
FINAL_STATUS = "Final"
POSTPONED_STATUS = "Postponed"


# ---- AUTOMATION -------------------------------------------------------------


def test_01_successful_prediction_cycle(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )
    props_df = _props_df([_prop_row("Player One", 15.0)])

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.SUCCESS
    assert result.persisted_count == 1
    assert result.run_id is not None
    assert len(get_snapshots_for_run(db_conn, result.run_id)) == 1


def test_02_no_games_cycle_is_no_work(db_conn):
    provider = _make_provider({}, scoreboard=[])
    empty_props = _props_df([])

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(),
        props_fetcher=lambda *_: empty_props,
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.NO_WORK
    assert result.reason_code == "NO_GAMES"


def test_03_no_props_yet_is_no_work_not_failed(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )
    empty_props = _props_df([])

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(),
        props_fetcher=lambda *_: empty_props,
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.NO_WORK
    assert result.reason_code == "NO_PROPS_YET"


def test_04_partial_player_failures_persist_the_successful_ones(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )
    # "Nobody Matched" has no active-player entry at all -> UNMATCHED.
    props_df = _props_df(
        [_prop_row("Player One", 15.0), _prop_row("Nobody Matched", 10.0)]
    )

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.PARTIAL
    assert result.unmatched_count == 1
    assert result.persisted_count == 2  # the OK snapshot AND the recorded UNMATCHED one
    snapshots = get_snapshots_for_run(db_conn, result.run_id)
    assert any(
        s["player_name"] == "Player One" and s["prediction_status"] == "ok"
        for s in snapshots
    )


def test_05_database_unavailable_is_failed(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    result = run_prediction_cycle(api_key="fake-key")

    assert result.status == CycleStatus.FAILED
    assert result.reason_code == "DATABASE_UNAVAILABLE"


def test_06_model_unavailable_is_failed(db_conn, monkeypatch):
    import src.shared_app as shared_app_module

    def _raise_model_load():
        raise RuntimeError("model.pkl missing")

    monkeypatch.setattr(shared_app_module, "load_model", _raise_model_load)
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=None,
        props_fetcher=lambda *_: _props_df([_prop_row("Player One", 15.0)]),
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.FAILED
    assert result.reason_code == "MODEL_UNAVAILABLE"


def test_07_duplicate_invocation_does_not_duplicate_history(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )
    props_df = _props_df([_prop_row("Player One", 15.0)])

    def cycle():
        return run_prediction_cycle(
            api_key="fake-key",
            conn=db_conn,
            provider=provider,
            model=_FakeModel(fixed_prediction=20.0),
            props_fetcher=lambda *_: props_df,
            sleep_func=lambda *_: None,
        )

    first = cycle()
    second = cycle()

    assert first.run_id == second.run_id
    assert second.run_created is False
    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 1


def test_08_changed_line_creates_a_new_immutable_run(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )

    first = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: _props_df([_prop_row("Player One", 15.0)]),
        sleep_func=lambda *_: None,
    )
    second = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: _props_df([_prop_row("Player One", 18.5)]),
        sleep_func=lambda *_: None,
    )

    assert second.run_created is True
    assert first.run_id != second.run_id
    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 2


def test_09_pregame_game_included(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, PREGAME_STATUS)]
    )

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: _props_df([_prop_row("Player One", 15.0)]),
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.SUCCESS
    assert result.excluded_live_count == 0
    assert result.persisted_count == 1


def test_10_live_game_excluded_from_new_pregame_prediction(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, LIVE_STATUS)]
    )

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: _props_df([_prop_row("Player One", 15.0)]),
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.NO_WORK
    assert result.reason_code == "ALL_GAMES_LIVE_OR_FINAL"
    assert result.excluded_live_count == 1
    assert result.persisted_count == 0
    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_runs")
    assert cur.fetchone()[0] == 0  # nothing written for a pure timing exclusion


def test_11_final_game_excluded(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10)}, scoreboard=[_game_for(1, FINAL_STATUS)]
    )

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: _props_df([_prop_row("Player One", 15.0)]),
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.NO_WORK
    assert result.excluded_final_count == 1


def test_12_later_pregame_game_still_processed_when_earlier_game_is_live(db_conn):
    provider = _make_provider(
        {1: ("Player One", 10), 2: ("Player Two", 10)},
        scoreboard=[_game_for(1, LIVE_STATUS), _game_for(2, PREGAME_STATUS)],
    )
    props_df = _props_df([_prop_row("Player One", 15.0), _prop_row("Player Two", 22.0)])

    result = run_prediction_cycle(
        api_key="fake-key",
        conn=db_conn,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert result.status == CycleStatus.SUCCESS
    assert result.excluded_live_count == 1
    assert result.persisted_count == 1
    snapshots = get_snapshots_for_run(db_conn, result.run_id)
    assert len(snapshots) == 1
    assert snapshots[0]["player_name"] == "Player Two"


# ---- SETTLEMENT -------------------------------------------------------------


def _persist_snapshot(
    db_conn,
    player_id=1,
    player_name="Player A",
    line=20.0,
    edge=2.5,
    direction_value="OVER",
):
    from src.services.prediction_result import (
        PredictionDirection,
        PredictionResult,
        PredictionStatus,
    )
    from src.services.prediction_service import PredictionBoard

    direction = PredictionDirection(direction_value) if direction_value else None
    result = PredictionResult(
        player_name=player_name,
        status=PredictionStatus.OK,
        player_id=player_id,
        model_projection=line + edge,
        sportsbook_line=line,
        edge=edge,
        direction=direction,
        bookmaker="draftkings",
        model_version="v1",
        generated_at_utc="2026-01-15T12:00:00+00:00",
        game_id="G1",
        game_date="01/15/2026",
    )
    board = PredictionBoard(
        generated_at_utc="2026-01-15T12:00:00+00:00",
        bookmaker="draftkings",
        model_version="v1",
        predictions=(result,),
        props_discovered=1,
        players_matched=1,
        predictions_generated=1,
        unmatched_count=0,
        unavailable_count=0,
    )
    run = persist_prediction_run(board, db_conn)
    return get_snapshots_for_run(db_conn, run["run_id"])[0]


class _FakeSettlementProvider:
    def __init__(
        self, *, scoreboard=None, gamelogs_by_player=None, raise_for_player_ids=()
    ):
        self._scoreboard = scoreboard or []
        self._gamelogs_by_player = gamelogs_by_player or {}
        self._raise_for_player_ids = set(raise_for_player_ids)

    def get_todays_scoreboard(self, game_date=None):
        return list(self._scoreboard)

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        if player_id in self._raise_for_player_ids:
            raise RuntimeError("boom -- simulated unexpected failure")
        return list(self._gamelogs_by_player.get(player_id, []))


def _points_log(player_id, points, game_id="G1"):
    import pandas as pd

    from src.data.basketball.models import PlayerGameLog

    return PlayerGameLog(
        season="2025-26",
        season_id="22025",
        player_id=player_id,
        player_name="Player A",
        game_id=game_id,
        game_date=pd.Timestamp("2026-01-15"),
        matchup="LAL vs DEN",
        team_abbreviation="LAL",
        opponent_abbreviation="DEN",
        is_home=1,
        wl="W",
        minutes=30,
        fgm=8,
        fga=16,
        fg_pct=0.5,
        fg3m=2,
        fg3a=5,
        fg3_pct=0.4,
        ftm=4,
        fta=5,
        ft_pct=0.8,
        oreb=1,
        dreb=5,
        reb=6,
        ast=4,
        stl=1,
        blk=0,
        tov=2,
        pf=2,
        points=points,
        plus_minus=5,
        video_available=1,
    )


def test_13_successful_settlement_cycle(db_conn):
    _persist_snapshot(db_conn, player_id=1)
    provider = _FakeSettlementProvider(
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="01/15/2026",
                home_team_id=1,
                away_team_id=2,
                game_status_text="Final",
            )
        ],
        gamelogs_by_player={1: [_points_log(1, 25)]},
    )

    result = run_settlement_cycle(conn=db_conn, provider=provider)

    assert result.status == CycleStatus.SUCCESS
    assert result.settled == 1


def test_14_no_pending_predictions_is_no_work(db_conn):
    result = run_settlement_cycle(conn=db_conn, provider=_FakeSettlementProvider())

    assert result.status == CycleStatus.NO_WORK
    assert result.reason_code == "NO_PENDING_PREDICTIONS"


def test_15_unfinished_games_remain_pending(db_conn):
    _persist_snapshot(db_conn, player_id=1)
    provider = _FakeSettlementProvider(
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="01/15/2026",
                home_team_id=1,
                away_team_id=2,
                game_status_text=PREGAME_STATUS,
            )
        ],
    )

    result = run_settlement_cycle(conn=db_conn, provider=provider)

    assert result.status == CycleStatus.SUCCESS  # nothing wrong -- just not final yet
    assert result.still_pending == 1
    assert result.settled == 0


def test_16_repeated_settlement_is_idempotent(db_conn):
    _persist_snapshot(db_conn, player_id=1)
    provider = _FakeSettlementProvider(
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="01/15/2026",
                home_team_id=1,
                away_team_id=2,
                game_status_text="Final",
            )
        ],
        gamelogs_by_player={1: [_points_log(1, 25)]},
    )

    first = run_settlement_cycle(conn=db_conn, provider=provider)
    second = run_settlement_cycle(conn=db_conn, provider=provider)

    assert first.settled == 1
    assert second.status == CycleStatus.NO_WORK  # already settled, nothing pending
    cur = db_conn.execute("SELECT COUNT(*) FROM prediction_outcomes")
    assert cur.fetchone()[0] == 1


def test_17_partial_settlement_failure_classification(db_conn):
    _persist_snapshot(db_conn, player_id=1, player_name="Good Player")
    _persist_snapshot(db_conn, player_id=2, player_name="Bad Player")
    provider = _FakeSettlementProvider(
        scoreboard=[
            ScheduledGame(
                game_id="G1",
                game_date="01/15/2026",
                home_team_id=1,
                away_team_id=2,
                game_status_text="Final",
            )
        ],
        gamelogs_by_player={1: [_points_log(1, 25)]},
        raise_for_player_ids={2},
    )

    result = run_settlement_cycle(conn=db_conn, provider=provider)

    assert result.status == CycleStatus.PARTIAL
    assert result.settled == 1
    assert result.errors == 1


def test_18_provider_unavailable_does_not_crash_the_cycle(db_conn):
    _persist_snapshot(db_conn, player_id=1)

    class _AlwaysBrokenProvider:
        def get_todays_scoreboard(self, game_date=None):
            raise RuntimeError("provider completely down")

        def get_player_game_logs(self, player_id, season, *, player_name=None):
            raise RuntimeError("provider completely down")

    result = run_settlement_cycle(conn=db_conn, provider=_AlwaysBrokenProvider())

    assert result.status == CycleStatus.PARTIAL
    assert result.errors == 1


# ---- OPERATIONS --------------------------------------------------------------


def test_19_schema_readiness_missing_migration_fails_clearly():
    bare_conn = sqlite3.connect(
        ":memory:"
    )  # no create_sqlite_prediction_schema applied
    try:
        prediction_result = run_prediction_cycle(
            api_key="fake-key", conn=bare_conn, provider=_make_provider({})
        )
        settlement_result = run_settlement_cycle(conn=bare_conn)
    finally:
        bare_conn.close()

    assert prediction_result.status == CycleStatus.FAILED
    assert prediction_result.reason_code == "SCHEMA_NOT_READY"
    assert settlement_result.status == CycleStatus.FAILED
    assert settlement_result.reason_code == "SCHEMA_NOT_READY"


def test_20_secrets_never_emitted_in_failure_messages(monkeypatch):
    """No real connection attempt here -- psycopg.connect is patched to
    fail instantly with a DSN-shaped error (as some drivers can produce
    on auth/config failures), proving the failure message src/services/
    orchestration.py builds never echoes it, without waiting on a real
    OS-level TCP timeout against an unreachable host."""
    import psycopg

    fake_secret = "SUPERSECRETPASSWORD12345"
    fake_dsn = f"postgresql://baduser:{fake_secret}@127.0.0.1:1/nosuchdb"
    monkeypatch.setenv("DATABASE_URL", fake_dsn)

    def _raise_with_dsn_in_message(*args, **kwargs):
        raise psycopg.OperationalError(f"connection to server failed: {fake_dsn}")

    monkeypatch.setattr(psycopg, "connect", _raise_with_dsn_in_message)

    result = run_prediction_cycle(api_key="fake-key")

    assert result.status == CycleStatus.FAILED
    assert fake_secret not in result.message
    assert fake_secret not in (result.reason_code or "")


def _load_script_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_21_exit_code_mapping():
    from src.services.orchestration import PredictionCycleResult

    ppb = _load_script_module("ppb_exit_code_test", "persist_prediction_board.py")

    for status in (CycleStatus.SUCCESS, CycleStatus.NO_WORK, CycleStatus.PARTIAL):
        canned = PredictionCycleResult(status=status, message="canned")
        ppb.run_prediction_cycle = lambda **_: canned  # noqa: B023 -- intentional per-iteration override
        assert ppb.main() == 0

    ppb.run_prediction_cycle = lambda **_: PredictionCycleResult(
        status=CycleStatus.FAILED, message="canned"
    )
    assert ppb.main() == 1


def test_22_concurrency_idempotency_db_level_guard(db_conn):
    """Even bypassing the application's check-then-insert idempotency
    logic entirely, the database's own UNIQUE constraint on
    idempotency_key is the last line of defense against a true
    concurrent double-run slipping past it (see the Step 10 report's
    concurrency section -- the workflow-level `concurrency:` group is
    the first line of defense; this is the second)."""
    db_conn.execute(
        "INSERT INTO prediction_runs (idempotency_key, generated_at_utc, bookmaker, model_version, "
        "run_status, props_discovered, players_matched, predictions_generated, unmatched_count, unavailable_count) "
        "VALUES ('dupe-key', '2026-01-15T12:00:00+00:00', 'draftkings', 'v1', 'SUCCESS', 1, 1, 1, 0, 0)"
    )
    db_conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO prediction_runs (idempotency_key, generated_at_utc, bookmaker, model_version, "
            "run_status, props_discovered, players_matched, predictions_generated, unmatched_count, unavailable_count) "
            "VALUES ('dupe-key', '2026-01-15T13:00:00+00:00', 'draftkings', 'v1', 'SUCCESS', 1, 1, 1, 0, 0)"
        )


def test_23_fetch_retry_succeeds_on_second_attempt():
    calls = {"n": 0}
    sleeps = []

    def flaky_fetcher(api_key, bookmaker_key):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient network blip")
        import pandas as pd

        return pd.DataFrame(
            [{"player_name_raw": "X", "line": 10.0, "bookmaker_key": "draftkings"}]
        )

    df, _call_count, error = _fetch_props_with_retry(
        flaky_fetcher,
        "key",
        "draftkings",
        max_attempts=2,
        retry_delay_seconds=5,
        sleep_func=sleeps.append,
    )

    assert error is None
    assert df is not None
    assert sleeps == [5]


def test_23b_fetch_retry_gives_up_after_max_attempts():
    def always_fails(api_key, bookmaker_key):
        raise RuntimeError("permanently down")

    sleeps = []
    df, _call_count, error = _fetch_props_with_retry(
        always_fails,
        "key",
        "draftkings",
        max_attempts=2,
        retry_delay_seconds=1,
        sleep_func=sleeps.append,
    )

    assert df is None
    assert error is not None
    assert sleeps == [1]  # slept once between the two attempts, never after the last


def test_24_offseason_no_slate_is_healthy_end_to_end(db_conn):
    ppb = _load_script_module("ppb_offseason_test", "persist_prediction_board.py")
    from src.services.orchestration import PredictionCycleResult

    ppb.run_prediction_cycle = lambda **_: PredictionCycleResult(
        status=CycleStatus.NO_WORK,
        message="No NBA games scheduled today.",
        reason_code="NO_GAMES",
    )

    assert ppb.main() == 0


def test_25_stale_board_classifier():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)

    status, _ = compute_board_freshness(latest_run=None, games_today=False, now_utc=now)
    assert status == BoardFreshness.NO_GAMES

    status, _ = compute_board_freshness(latest_run=None, games_today=True, now_utc=now)
    assert status == BoardFreshness.WAITING_FOR_PROPS

    recent_run = {"generated_at_utc": (now - timedelta(hours=1)).isoformat()}
    status, _ = compute_board_freshness(
        latest_run=recent_run, games_today=True, now_utc=now, stale_after_hours=6.0
    )
    assert status == BoardFreshness.FRESH

    old_run = {"generated_at_utc": (now - timedelta(hours=10)).isoformat()}
    status, _ = compute_board_freshness(
        latest_run=old_run, games_today=True, now_utc=now, stale_after_hours=6.0
    )
    assert status == BoardFreshness.STALE


def test_26_ui_never_imports_orchestration_entry_points():
    """Structural guard: apps/publicapp.py must stay read-only with
    respect to automation -- it may READ persisted history (Step 9) but
    must never call run_prediction_cycle/run_settlement_cycle itself
    (see the Step 10 hard requirement: 'Do not let Streamlit UI code
    own this orchestration')."""
    text = (REPO_ROOT / "apps" / "publicapp.py").read_text(encoding="utf-8")
    assert "run_prediction_cycle" not in text
    assert "run_settlement_cycle" not in text


def test_28_model_registry_and_legacy_model_path_unreferenced_by_step10_modules():
    step10_modules = [
        "src/services/orchestration.py",
        "src/services/automation_config.py",
    ]
    forbidden = ("points_regression.pkl", "CURRENT_V1", "MODEL_REGISTRY_DIR")
    for rel_path in step10_modules:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{rel_path} unexpectedly references {token!r}"


def test_29_no_training_code_touched():
    step10_modules = [
        "src/services/orchestration.py",
        "src/services/automation_config.py",
        "scripts/persist_prediction_board.py",
        "scripts/settle_predictions.py",
        "scripts/check_schema_readiness.py",
    ]
    for rel_path in step10_modules:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "import training" not in text
        assert "from training" not in text


def test_30_workflow_yaml_is_valid():
    yaml = pytest.importorskip("yaml")

    for name in ("prediction-cycle.yml", "settlement-cycle.yml"):
        path = REPO_ROOT / ".github" / "workflows" / name
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        assert "jobs" in doc
        # PyYAML parses the bare `on:` key as boolean True (YAML 1.1) --
        # this is a real gotcha for GitHub Actions files; assert on
        # whichever key form appears rather than assuming.
        assert "on" in doc or True in doc
        job = next(iter(doc["jobs"].values()))
        assert job["timeout-minutes"] > 0
        assert "concurrency" in doc
        assert doc["permissions"] == {"contents": "read"}


# ---- Extra: classify_game_status direct coverage ----------------------------


def test_classify_game_status_all_branches():
    assert classify_game_status(PREGAME_STATUS) == "PREGAME"
    assert classify_game_status(LIVE_STATUS) == "LIVE"
    assert classify_game_status(FINAL_STATUS) == "FINAL"
    assert classify_game_status(POSTPONED_STATUS) == "POSTPONED_CANCELED"
    assert classify_game_status(None) == "UNKNOWN"
    assert classify_game_status("") == "UNKNOWN"
