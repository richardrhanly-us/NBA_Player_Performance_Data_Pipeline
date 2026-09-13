"""
A minimal migration runner for the prediction-history schema.

There is no existing migration framework anywhere in this repository
(verified: no alembic, no .sql files, no migrations/ directory existed
before Step 9) -- so rather than adopt a heavy new dependency for one
small, additive schema, this is a small, dependency-free (stdlib +
whatever DB-API 2.0 connection is passed in) runner: read versioned .sql
files from migrations/ in filename order, apply any not yet recorded in
a schema_migrations table, and record them as applied.

This module is dialect-agnostic -- it splits each file into individual
statements (on top-level semicolons, skipping blank/comment-only lines)
and executes them one at a time, which works identically against both
psycopg (Postgres) and sqlite3 connections without relying on either
driver's own multi-statement execution quirks.

The real migrations/0001_create_prediction_history.sql targets Postgres
(the project's existing DATABASE_URL/psycopg convention -- see
src/services/db_connection.py); tests exercise this runner's own
apply/track/head logic against small, self-contained SQLite-compatible
fixture migrations, not that file, since SQLite does not understand
Postgres-only syntax like BIGSERIAL/now(). Schema-content correctness
for the real migration is validated separately (see
src/services/schema_sqlite.py and the Step 9 report's migration-
validation section).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

_VERSION_PATTERN = re.compile(r"^(\d+)_")


def _migration_files(migrations_dir: Path) -> list:
    return sorted(Path(migrations_dir).glob("*.sql"), key=lambda p: p.name)


def _version_for(path: Path) -> str:
    match = _VERSION_PATTERN.match(path.stem)
    return match.group(1) if match else path.stem


def _split_statements(sql: str) -> list:
    """Splits a SQL script into individual statements on top-level
    semicolons, dropping full-line `--` comments and blank statements.
    Simple and sufficient for this project's own migration files (no
    semicolons inside string literals or dollar-quoted blocks)."""
    lines = [
        line
        for line in sql.splitlines()
        if not line.strip().startswith("--") and line.strip()
    ]
    cleaned = "\n".join(lines)
    statements = [s.strip() for s in cleaned.split(";")]
    return [s for s in statements if s]


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg")


def _ensure_schema_migrations_table(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_applied_versions(conn) -> set:
    _ensure_schema_migrations_table(conn)
    cur = conn.cursor()
    cur.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def get_current_head(conn):
    """Returns the highest applied migration version (as an int, since
    versions are zero-padded numeric prefixes like '0001'), or None if
    no migrations have been applied yet."""
    applied = get_applied_versions(conn)
    if not applied:
        return None
    return max(int(v) for v in applied)


REQUIRED_PREDICTION_HISTORY_TABLES = (
    "prediction_runs",
    "prediction_snapshots",
    "prediction_outcomes",
)


def is_schema_ready(conn, required_tables=REQUIRED_PREDICTION_HISTORY_TABLES) -> bool:
    """
    Step 10: a lightweight, READ-ONLY readiness check -- true if every
    table in `required_tables` already exists, false otherwise (never
    raises, and never creates anything, unlike get_applied_versions()
    above which bootstraps schema_migrations as a side effect). Used by
    automation entry points (src/services/orchestration.py) to fail
    fast with a clear, actionable message when
    migrations/0001_create_prediction_history.sql has not been applied
    yet, instead of letting a raw "relation does not exist" SQL error
    surface from deep inside persist_prediction_run/
    settle_pending_predictions. Deliberately does NOT apply migrations
    itself -- schema changes stay an explicit, human-run operation (see
    scripts/apply_prediction_history_migrations.py).
    """
    cur = conn.cursor()
    try:
        if _is_postgres(conn):
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                (list(required_tables),),
            )
        else:
            placeholders = ",".join("?" * len(required_tables))
            cur.execute(
                f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
                tuple(required_tables),
            )
        found = {row[0] for row in cur.fetchall()}
    except Exception:  # noqa: BLE001 -- any failure here means "not ready"
        return False
    return set(required_tables).issubset(found)


def apply_migrations(conn, migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> list:
    """
    Applies every .sql file in `migrations_dir` not already recorded in
    schema_migrations, in filename order, each in its own transaction.
    Returns the list of newly-applied version strings (empty if nothing
    new). Safe to call repeatedly -- already-applied migrations are
    skipped, never re-run.
    """
    _ensure_schema_migrations_table(conn)
    applied = get_applied_versions(conn)
    newly_applied = []
    placeholder = "%s" if _is_postgres(conn) else "?"

    for path in _migration_files(migrations_dir):
        version = _version_for(path)
        if version in applied:
            continue

        sql = path.read_text(encoding="utf-8")
        cur = conn.cursor()
        for statement in _split_statements(sql):
            cur.execute(statement)
        cur.execute(
            f"INSERT INTO schema_migrations (version, applied_at) VALUES ({placeholder}, {placeholder})",
            (version, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        newly_applied.append(version)

    return newly_applied
