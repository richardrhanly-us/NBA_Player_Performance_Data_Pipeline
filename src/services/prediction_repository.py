"""
The prediction-history repository: persists a PredictionBoard as one
immutable prediction_runs row + immutable prediction_snapshots rows, and
provides the read paths settlement/performance services need.

Every function here accepts an explicit `conn` (a DB-API 2.0 connection
-- either a real psycopg/Postgres connection, via
src/services/db_connection.py, or a sqlite3 connection built from
src/services/schema_sqlite.py in tests). SQL is written once, with `?`
placeholders, and translated to `%s` for psycopg connections -- see
_ph()/_execute() below -- so the exact same code path is what tests
exercise against SQLite and what production exercises against Postgres.

INSERT-ONLY discipline: there is no UPDATE statement anywhere in this
module for prediction_snapshots' own columns (projection, line, edge,
direction, model_version, ...). A prediction, once written, is never
edited. A new/changed board produces a new prediction_runs row and new
prediction_snapshots rows -- see persist_prediction_run()'s idempotency
handling for the one legitimate exception (returning an ALREADY-
persisted run's id when the exact same board content is submitted
again, rather than writing duplicate rows).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from src.services.prediction_result import PredictionStatus

QUALIFIED_EDGE_THRESHOLD_DEFAULT = (
    3.0  # matches shared_app.EDGE_THRESHOLD; see note on qualified()
)


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg")


def _ph(conn) -> str:
    return "%s" if _is_postgres(conn) else "?"


def _execute(conn, sql: str, params=()):
    placeholder = _ph(conn)
    adapted_sql = sql.replace("?", placeholder) if placeholder != "?" else sql
    cur = conn.cursor()
    cur.execute(adapted_sql, params)
    return cur


def _insert_and_get_id(conn, sql: str, params: tuple):
    """
    Executes an INSERT and returns the new row's `id`. For Postgres,
    appends `RETURNING id` and fetches it (the idiomatic, unambiguous
    way -- not session-global `lastval()`); for SQLite, uses the
    cursor's own `lastrowid` (set for any INTEGER PRIMARY KEY
    AUTOINCREMENT insert).
    """
    if _is_postgres(conn):
        cur = _execute(conn, sql + " RETURNING id", params)
        return cur.fetchone()[0]
    cur = _execute(conn, sql, params)
    return cur.lastrowid


def _row_to_dict(cur, row):
    columns = [d[0] for d in cur.description]
    return dict(zip(columns, row))


def compute_board_idempotency_key(board) -> str:
    """
    A deterministic hash of a board's actual content (bookmaker, model
    version, and every player's identity/line/projection/status) -- NOT
    its generated_at_utc timestamp. Submitting the exact same board
    twice (e.g. a retry after a network blip) produces the SAME key, so
    persist_prediction_run() below reuses the existing run instead of
    writing a duplicate. A genuinely different board -- a line moved, a
    new prop appeared, a projection changed -- produces a different key
    and a new, distinct run, exactly as it should.
    """
    parts = [str(board.bookmaker or ""), str(board.model_version or "")]
    for r in sorted(
        board.predictions,
        key=lambda p: (p.player_id if p.player_id is not None else -1, p.player_name),
    ):
        parts.append(
            "|".join(
                str(x)
                for x in (
                    r.player_id,
                    r.player_name,
                    r.sportsbook_line,
                    r.model_projection,
                    r.status.value,
                )
            )
        )
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_run_by_idempotency_key(conn, idempotency_key: str):
    cur = _execute(
        conn,
        "SELECT * FROM prediction_runs WHERE idempotency_key = ?",
        (idempotency_key,),
    )
    row = cur.fetchone()
    return _row_to_dict(cur, row) if row is not None else None


def _run_status_for(board) -> str:
    if board.props_discovered == 0:
        return "SUCCESS"  # nothing to do today is a legitimate, successful (empty) run
    if board.predictions_generated == 0:
        return "FAILED"
    if board.unmatched_count > 0 or board.unavailable_count > 0:
        return "PARTIAL"
    return "SUCCESS"


def persist_prediction_run(
    board, conn, *, edge_threshold: float = QUALIFIED_EDGE_THRESHOLD_DEFAULT
):
    """
    Persists one PredictionBoard: one prediction_runs row, and one
    immutable prediction_snapshots row per PredictionResult (including
    non-OK ones -- unmatched/unavailable players are recorded too, so
    the run's own health counts are reconstructible later from the
    snapshots themselves, not just the summary columns).

    Idempotent by content (see compute_board_idempotency_key): if a run
    with the same idempotency_key already exists, this returns that
    existing run's id and `created=False` WITHOUT inserting anything
    new. Otherwise the run + all snapshots are written, and if anything
    fails partway through, the whole insert is rolled back (conn.rollback())
    so a failed write never leaves a misleading partial run behind.

    `qualified` is computed here (edge_threshold, defaulting to the
    existing EDGE_THRESHOLD=3.0 policy value) rather than stored on
    PredictionResult itself -- it is a reporting/threshold concern, not
    a core prediction field, and keeping it out of PredictionResult
    avoids touching Step 8's contract for a Step 9 concern.

    Returns {"run_id": ..., "created": bool}.
    """
    idempotency_key = compute_board_idempotency_key(board)
    existing = get_run_by_idempotency_key(conn, idempotency_key)
    if existing is not None:
        return {"run_id": existing["id"], "created": False}

    run_status = _run_status_for(board)

    try:
        run_id = _insert_and_get_id(
            conn,
            """
            INSERT INTO prediction_runs (
                idempotency_key, generated_at_utc, bookmaker, model_version,
                run_status, props_discovered, players_matched,
                predictions_generated, unmatched_count, unavailable_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                board.generated_at_utc,
                board.bookmaker,
                board.model_version,
                run_status,
                board.props_discovered,
                board.players_matched,
                board.predictions_generated,
                board.unmatched_count,
                board.unavailable_count,
            ),
        )

        for r in board.predictions:
            qualified = bool(r.edge is not None and abs(r.edge) >= edge_threshold)
            direction_value = r.direction.value if r.direction is not None else None
            _execute(
                conn,
                """
                INSERT INTO prediction_snapshots (
                    prediction_run_id, player_id, player_name, team_abbreviation,
                    matchup, game_id, game_date, game_status, bookmaker,
                    sportsbook_line, model_projection, edge, direction, qualified,
                    prediction_status, reason, model_version, generated_at_utc,
                    latest_game_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    r.player_id,
                    r.player_name,
                    r.team_abbreviation,
                    r.matchup,
                    r.game_id,
                    r.game_date,
                    r.game_status,
                    r.bookmaker,
                    r.sportsbook_line,
                    r.model_projection,
                    r.edge,
                    direction_value,
                    qualified,
                    r.status.value,
                    r.reason,
                    r.model_version,
                    r.generated_at_utc,
                    r.latest_game_date,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {"run_id": run_id, "created": True}


def get_latest_run(conn):
    """
    Step 10: the most recently persisted prediction_runs row (by
    generated_at_utc), or None if nothing has ever been persisted. Used
    by the UI's freshness indicator (src/services/orchestration.py
    ::compute_board_freshness) -- read-only, no automation implications.
    """
    cur = _execute(
        conn, "SELECT * FROM prediction_runs ORDER BY generated_at_utc DESC LIMIT 1"
    )
    row = cur.fetchone()
    return _row_to_dict(cur, row) if row is not None else None


def get_snapshots_for_run(conn, run_id: int) -> list:
    cur = _execute(
        conn,
        "SELECT * FROM prediction_snapshots WHERE prediction_run_id = ?",
        (run_id,),
    )
    return [_row_to_dict(cur, row) for row in cur.fetchall()]


def get_pending_snapshots(conn, limit=None) -> list:
    """
    Snapshots with a real (status=OK) projection that do not yet have a
    prediction_outcomes row -- i.e. genuinely awaiting settlement.
    Non-OK snapshots (unmatched/unavailable/missing line/etc.) are never
    "pending settlement" -- they never had a gradable prediction.
    """
    sql = """
        SELECT s.* FROM prediction_snapshots s
        LEFT JOIN prediction_outcomes o ON o.prediction_snapshot_id = s.id
        WHERE o.id IS NULL AND s.prediction_status = ?
        ORDER BY s.id
    """
    cur = _execute(conn, sql, (PredictionStatus.OK.value,))
    rows = [_row_to_dict(cur, row) for row in cur.fetchall()]
    return rows[:limit] if limit is not None else rows


def get_outcome_for_snapshot(conn, snapshot_id: int):
    cur = _execute(
        conn,
        "SELECT * FROM prediction_outcomes WHERE prediction_snapshot_id = ?",
        (snapshot_id,),
    )
    row = cur.fetchone()
    return _row_to_dict(cur, row) if row is not None else None


def insert_outcome(
    conn,
    *,
    prediction_snapshot_id: int,
    actual_points,
    result_status: str,
    game_status=None,
    source: str,
    settled_at=None,
) -> dict:
    """
    Idempotent by prediction_snapshot_id: if an outcome already exists
    for this snapshot (checked at the application level, backed by the
    schema's UNIQUE constraint as a defensive backstop against races),
    this returns the EXISTING row unchanged rather than inserting a
    second one or updating the first -- outcomes, like snapshots, are
    never edited once written.
    """
    existing = get_outcome_for_snapshot(conn, prediction_snapshot_id)
    if existing is not None:
        return existing

    settled_at = settled_at or datetime.now(timezone.utc).isoformat()
    try:
        _insert_and_get_id(
            conn,
            """
            INSERT INTO prediction_outcomes (
                prediction_snapshot_id, actual_points, result_status,
                game_status, source, settled_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_snapshot_id,
                actual_points,
                result_status,
                game_status,
                source,
                settled_at,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return get_outcome_for_snapshot(conn, prediction_snapshot_id)


def get_settled_predictions(
    conn,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    direction: str | None = None,
    qualified_only: bool = False,
) -> list:
    """Joins prediction_snapshots + prediction_outcomes for every
    settled (non-PENDING) prediction, with optional filters -- the read
    path the performance service builds summaries from."""
    sql = """
        SELECT s.*, o.actual_points AS outcome_actual_points,
               o.result_status AS outcome_result_status,
               o.settled_at AS outcome_settled_at
        FROM prediction_snapshots s
        JOIN prediction_outcomes o ON o.prediction_snapshot_id = s.id
        WHERE o.result_status != ?
    """
    params = ["PENDING"]
    if start_date is not None:
        sql += " AND s.generated_at_utc >= ?"
        params.append(start_date)
    if end_date is not None:
        sql += " AND s.generated_at_utc <= ?"
        params.append(end_date)
    if direction is not None:
        sql += " AND s.direction = ?"
        params.append(direction)
    if qualified_only:
        sql += " AND s.qualified = ?"
        params.append(True if _is_postgres(conn) else 1)

    sql += " ORDER BY s.generated_at_utc DESC"
    cur = _execute(conn, sql, tuple(params))
    return [_row_to_dict(cur, row) for row in cur.fetchall()]
