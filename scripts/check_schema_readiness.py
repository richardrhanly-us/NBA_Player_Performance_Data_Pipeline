"""
Step 10: manual/CI pre-flight check -- "is the prediction-history
schema migrated on whatever DATABASE_URL currently points at?" Prints a
clear yes/no instead of a raw SQL exception, and exits 0 (ready) or 1
(not ready) so it can gate a workflow step or be run by hand during
incident recovery.

This performs NO writes and applies NO migrations -- see
scripts/apply_prediction_history_migrations.py for that (a separate,
deliberately manual operation).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.db_connection import get_prediction_db_connection
from src.services.migrations import REQUIRED_PREDICTION_HISTORY_TABLES, is_schema_ready


def log(msg):
    print(msg, flush=True)


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        log("[SCHEMA CHECK] DATABASE_URL is not set.")
        return 1

    conn = get_prediction_db_connection()
    try:
        ready = is_schema_ready(conn)
    finally:
        conn.close()

    if ready:
        log(
            f"[SCHEMA CHECK] READY -- all required tables present: {REQUIRED_PREDICTION_HISTORY_TABLES}"
        )
        return 0

    log(
        "[SCHEMA CHECK] NOT READY -- one or more of "
        f"{REQUIRED_PREDICTION_HISTORY_TABLES} is missing. "
        "Run scripts/apply_prediction_history_migrations.py against this database first."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
