"""
Step 9: applies migrations/*.sql (currently just
0001_create_prediction_history.sql) to whatever database DATABASE_URL
points at, via src/services/migrations.py::apply_migrations.

This is a MANUAL, human-invoked script -- it is never called
automatically by the Streamlit app or by any of the other scheduled
scripts in this directory. Point DATABASE_URL at a disposable/dev
database to validate the migration, or at the real production database
only when a human has deliberately decided to run this. Prints which
migrations were applied (an empty list means the schema was already up
to date) and the resulting head version.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.db_connection import get_prediction_db_connection
from src.services.migrations import apply_migrations, get_current_head


def log(msg):
    print(msg, flush=True)


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL not found in environment -- refusing to guess a target database"
        )

    log(
        f"[MIGRATE] Applying migrations to: {database_url.split('@')[-1]}"
    )  # never log credentials

    conn = get_prediction_db_connection()
    try:
        applied = apply_migrations(conn)
        head = get_current_head(conn)
    finally:
        conn.close()

    if applied:
        log(f"[MIGRATE] Applied: {applied}")
    else:
        log("[MIGRATE] Schema already up to date -- nothing applied.")
    log(f"[MIGRATE] Current head version: {head}")


if __name__ == "__main__":
    main()
