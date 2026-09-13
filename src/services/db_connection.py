"""
Database connection helper for the Step 9 prediction-history/settlement
persistence layer.

Deliberately separate from the pre-existing src/db.py: that module
exists but is not actually imported by any current call site (verified
by repo-wide search -- only scripts/pregame_pipeline.py's OWN inline
duplicate of get_db_connection()/insert_line_snapshot() is used in
production today). Rather than touch that unrelated, pre-existing
module, this file provides the same minimal connection pattern
(psycopg + DATABASE_URL) for the NEW prediction-history tables this
step adds, without disturbing line_snapshots or anything that reads it.

psycopg (v3) is used here to match src/db.py's existing import
convention. It was not previously declared in requirements.txt (a
pre-existing gap -- it is already used in production via
scripts/pregame_pipeline.py's own psycopg import); this step adds it
properly since new code here genuinely depends on it.
"""

from __future__ import annotations

import os


def get_prediction_db_connection():
    """
    Returns a new psycopg connection using the DATABASE_URL environment
    variable -- the same variable src/db.py and
    scripts/pregame_pipeline.py already use for the existing Postgres/Neon
    database. Raises a clear error if it isn't configured, rather than
    failing with an opaque KeyError deep inside psycopg.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set -- the prediction-history persistence "
            "layer has no database to connect to."
        )
    import psycopg

    return psycopg.connect(database_url)


def is_postgres_connection(conn) -> bool:
    """
    True for a real psycopg connection, False for the SQLite connections
    tests use instead (see src/services/schema_sqlite.py). Used only to
    pick the right placeholder style for the small number of SQL
    statements that need it -- see prediction_repository._execute.
    """
    return type(conn).__module__.startswith("psycopg")
