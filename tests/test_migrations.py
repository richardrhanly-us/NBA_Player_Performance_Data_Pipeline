"""
Tests for src/services/migrations.py's runner mechanics (apply, track,
head), using small, self-contained SQLite-compatible fixture migrations
-- NOT the real migrations/0001_create_prediction_history.sql, which
targets Postgres-only syntax (BIGSERIAL, now()) that SQLite does not
understand. See src/services/schema_sqlite.py for how that real
migration's CONTENT is validated instead. No live database access.
"""

import sqlite3

from src.services.migrations import (
    apply_migrations,
    get_applied_versions,
    get_current_head,
)


def _write_migration(tmp_path, name, sql):
    path = tmp_path / name
    path.write_text(sql, encoding="utf-8")
    return path


def test_apply_migrations_creates_tables_and_records_version(tmp_path):
    _write_migration(
        tmp_path,
        "0001_create_widgets.sql",
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT);",
    )
    conn = sqlite3.connect(":memory:")

    applied = apply_migrations(conn, migrations_dir=tmp_path)
    assert applied == ["0001"]
    assert get_applied_versions(conn) == {"0001"}

    conn.execute("INSERT INTO widgets (name) VALUES ('a')")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM widgets").fetchone()[0] == 1


def test_apply_migrations_is_idempotent_no_reapply(tmp_path):
    _write_migration(
        tmp_path,
        "0001_create_widgets.sql",
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY);",
    )
    conn = sqlite3.connect(":memory:")

    first = apply_migrations(conn, migrations_dir=tmp_path)
    second = apply_migrations(conn, migrations_dir=tmp_path)

    assert first == ["0001"]
    assert second == []  # already applied -- not re-run


def test_apply_migrations_applies_in_order_and_tracks_head(tmp_path):
    _write_migration(
        tmp_path,
        "0001_create_widgets.sql",
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY);",
    )
    _write_migration(
        tmp_path,
        "0002_create_gadgets.sql",
        "CREATE TABLE gadgets (id INTEGER PRIMARY KEY);",
    )
    conn = sqlite3.connect(":memory:")

    assert get_current_head(conn) is None

    applied = apply_migrations(conn, migrations_dir=tmp_path)
    assert applied == ["0001", "0002"]
    assert get_current_head(conn) == 2


def test_apply_migrations_only_applies_new_ones_when_rerun_after_adding_one(tmp_path):
    _write_migration(
        tmp_path,
        "0001_create_widgets.sql",
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY);",
    )
    conn = sqlite3.connect(":memory:")
    apply_migrations(conn, migrations_dir=tmp_path)

    _write_migration(
        tmp_path,
        "0002_create_gadgets.sql",
        "CREATE TABLE gadgets (id INTEGER PRIMARY KEY);",
    )
    second_run = apply_migrations(conn, migrations_dir=tmp_path)

    assert second_run == ["0002"]
    assert get_current_head(conn) == 2


def test_multi_statement_migration_file_applies_all_statements(tmp_path):
    sql = """
    -- a comment line, should be ignored
    CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT);
    CREATE INDEX idx_widgets_name ON widgets(name);
    """
    _write_migration(tmp_path, "0001_multi.sql", sql)
    conn = sqlite3.connect(":memory:")
    apply_migrations(conn, migrations_dir=tmp_path)

    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_widgets_name'"
    )
    assert cur.fetchone() is not None


def test_real_migration_file_exists_and_is_nonempty():
    from pathlib import Path

    from src.services.migrations import DEFAULT_MIGRATIONS_DIR

    files = sorted(Path(DEFAULT_MIGRATIONS_DIR).glob("*.sql"))
    assert len(files) >= 1
    assert files[0].name.startswith("0001")
    assert len(files[0].read_text(encoding="utf-8")) > 0
