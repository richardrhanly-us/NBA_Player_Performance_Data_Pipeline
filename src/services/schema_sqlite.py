"""
A SQLite-compatible mirror of migrations/0001_create_prediction_history.sql,
used ONLY by the automated test suite (no live Postgres/Neon access is
available in this environment or in CI -- see the Step 9 report's
migration-validation section). This lets tests exercise real SQL
constraint enforcement (UNIQUE, CHECK, FOREIGN KEY) end-to-end, fully
offline and deterministically, via Python's stdlib sqlite3 module.

Kept deliberately side-by-side and reviewed together with the real
Postgres migration -- table/column names and constraints are logically
identical; only dialect-specific syntax differs:

    BIGSERIAL PRIMARY KEY        -> INTEGER PRIMARY KEY AUTOINCREMENT
    TIMESTAMPTZ ... DEFAULT now() -> TEXT ... DEFAULT (datetime('now'))
    (everything else -- NUMERIC, BOOLEAN, TEXT, CHECK, UNIQUE, FOREIGN KEY
    ON DELETE CASCADE -- is understood by SQLite as-is)

If this schema and the real migration ever diverge, the tests here stop
being a meaningful proxy for the production schema -- keep them in sync
whenever one changes.
"""

from __future__ import annotations

SQLITE_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS prediction_runs (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key         TEXT NOT NULL UNIQUE,
        generated_at_utc        TEXT NOT NULL,
        bookmaker               TEXT,
        model_version           TEXT NOT NULL,
        run_status               TEXT NOT NULL CHECK (run_status IN ('SUCCESS', 'PARTIAL', 'FAILED')),
        props_discovered         INTEGER NOT NULL DEFAULT 0,
        players_matched          INTEGER NOT NULL DEFAULT 0,
        predictions_generated    INTEGER NOT NULL DEFAULT 0,
        unmatched_count           INTEGER NOT NULL DEFAULT 0,
        unavailable_count         INTEGER NOT NULL DEFAULT 0,
        created_at                 TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_snapshots (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_run_id       INTEGER NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
        player_id                INTEGER,
        player_name               TEXT NOT NULL,
        team_abbreviation          TEXT,
        matchup                    TEXT,
        game_id                    TEXT,
        game_date                  TEXT,
        game_status                 TEXT,
        bookmaker                   TEXT,
        sportsbook_line              NUMERIC,
        model_projection             NUMERIC,
        edge                          NUMERIC,
        direction                     TEXT CHECK (direction IN ('OVER', 'UNDER', 'NEUTRAL')),
        qualified                      BOOLEAN NOT NULL DEFAULT 0,
        prediction_status               TEXT NOT NULL,
        reason                           TEXT,
        model_version                    TEXT NOT NULL,
        generated_at_utc                  TEXT NOT NULL,
        latest_game_date                   TEXT,
        created_at                          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_run_id ON prediction_snapshots(prediction_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_player_id ON prediction_snapshots(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_game_id ON prediction_snapshots(game_id)",
    """
    CREATE TABLE IF NOT EXISTS prediction_outcomes (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_snapshot_id      INTEGER NOT NULL UNIQUE REFERENCES prediction_snapshots(id) ON DELETE CASCADE,
        actual_points                NUMERIC,
        result_status                 TEXT NOT NULL CHECK (
            result_status IN ('PENDING', 'WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'UNAVAILABLE')
        ),
        game_status                    TEXT,
        source                          TEXT NOT NULL,
        settled_at                       TEXT,
        created_at                        TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_status ON prediction_outcomes(result_status)",
)


def create_sqlite_prediction_schema(conn) -> None:
    """Applies SQLITE_SCHEMA_STATEMENTS to a sqlite3 connection and
    enables foreign-key enforcement (off by default in SQLite)."""
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    for statement in SQLITE_SCHEMA_STATEMENTS:
        cur.execute(statement)
    conn.commit()


# SQLite-compatible mirror of migrations/0002_create_accounts_and_entitlements.sql
# PLUS migrations/0003_add_billing_fields_and_stripe_events.sql (Step 13) --
# same rationale as SQLITE_SCHEMA_STATEMENTS above -- no live Postgres/Neon
# access here or in CI. Kept side-by-side and reviewed together with the
# real migrations; only dialect-specific syntax differs (BIGSERIAL ->
# INTEGER PRIMARY KEY AUTOINCREMENT, TIMESTAMPTZ ... DEFAULT now() -> TEXT
# ... DEFAULT (datetime('now'))). This mirrors the CUMULATIVE end state of
# both migrations (not one file per migration) -- see create_sqlite_accounts_schema's
# docstring. If this schema and the real migrations ever diverge, keep
# them in sync whenever one changes.
ACCOUNTS_SQLITE_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS users (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        email               TEXT NOT NULL,
        display_name        TEXT,
        auth_provider       TEXT NOT NULL,
        auth_subject        TEXT NOT NULL,
        is_active           BOOLEAN NOT NULL DEFAULT 1,
        stripe_customer_id  TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (auth_provider, auth_subject)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_stripe_customer_id ON users(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS subscriptions (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id                     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        provider                    TEXT NOT NULL DEFAULT 'none',
        provider_customer_id        TEXT,
        provider_subscription_id    TEXT,
        plan_key                    TEXT NOT NULL,
        status                      TEXT NOT NULL CHECK (
            status IN ('active', 'trialing', 'past_due', 'canceled', 'incomplete', 'none')
        ),
        current_period_start        TEXT,
        current_period_end          TEXT,
        cancel_at_period_end        BOOLEAN NOT NULL DEFAULT 0,
        stripe_price_id             TEXT,
        last_synced_at              TEXT,
        created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at                  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_subscriptions_provider_subscription_id ON subscriptions(provider_subscription_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_subscriptions_provider_subscription_unique ON subscriptions(provider, provider_subscription_id) WHERE provider_subscription_id IS NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS entitlement_overrides (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        override_tier       TEXT NOT NULL CHECK (override_tier IN ('FREE', 'PRO', 'ADMIN')),
        enabled             BOOLEAN NOT NULL DEFAULT 1,
        reason              TEXT,
        expires_at          TEXT,
        created_by          TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entitlement_overrides_user_id ON entitlement_overrides(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_entitlement_overrides_user_enabled ON entitlement_overrides(user_id, enabled)",
    """
    CREATE TABLE IF NOT EXISTS stripe_events (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        stripe_event_id     TEXT NOT NULL UNIQUE,
        event_type          TEXT NOT NULL,
        received_at         TEXT NOT NULL DEFAULT (datetime('now')),
        processed_at        TEXT,
        processing_status   TEXT NOT NULL DEFAULT 'received' CHECK (
            processing_status IN ('received', 'processed', 'failed')
        ),
        error_message       TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_stripe_events_status ON stripe_events(processing_status)",
)


def create_sqlite_accounts_schema(conn) -> None:
    """Applies ACCOUNTS_SQLITE_SCHEMA_STATEMENTS to a sqlite3 connection
    (in addition to, or independent of, create_sqlite_prediction_schema)
    and enables foreign-key enforcement."""
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    for statement in ACCOUNTS_SQLITE_SCHEMA_STATEMENTS:
        cur.execute(statement)
    conn.commit()
