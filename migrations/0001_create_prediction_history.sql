-- Step 9: prediction history, settlement, and performance tracking.
--
-- This applies to the SAME Postgres/Neon database that
-- line_snapshots already lives in (see src/db.py,
-- scripts/pregame_pipeline.py) -- these tables are purely additive;
-- line_snapshots and every other existing table are untouched.
--
-- Applied via src/services/migrations.py (see that module for the
-- runner). NOT applied automatically -- see the Step 9 report's
-- migration-validation section for exactly where/how this was
-- exercised (a disposable, in-process SQLite mirror of this schema;
-- no production database was reachable or touched in this
-- environment).
--
-- Design notes:
--   * prediction_snapshots rows are INSERT-ONLY from application code
--     (see src/services/prediction_repository.py -- there is no
--     UPDATE statement anywhere in that module's normal API for
--     projection/line/edge/direction/model_version). A later line
--     move or model change produces a NEW prediction_runs row and NEW
--     prediction_snapshots rows, never an edit to existing ones.
--   * prediction_outcomes is a SEPARATE, append-only table (not columns
--     bolted onto prediction_snapshots), so "what we predicted" and
--     "what actually happened" are always distinguishable, and a
--     snapshot's own columns never change after insertion.
--   * idempotency_key on prediction_runs is a deterministic hash of the
--     run's actual content (bookmaker, model_version, every player's
--     line/projection) -- see
--     prediction_repository.compute_board_idempotency_key. Submitting
--     the exact same board twice (a retry) reuses the same run;
--     a genuinely different board (a line moved, a new player
--     appeared) gets a new key and a new run.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    id                      BIGSERIAL PRIMARY KEY,
    idempotency_key         TEXT NOT NULL UNIQUE,
    generated_at_utc        TIMESTAMPTZ NOT NULL,
    bookmaker               TEXT,
    model_version           TEXT NOT NULL,
    run_status              TEXT NOT NULL CHECK (run_status IN ('SUCCESS', 'PARTIAL', 'FAILED')),
    props_discovered        INTEGER NOT NULL DEFAULT 0,
    players_matched         INTEGER NOT NULL DEFAULT 0,
    predictions_generated   INTEGER NOT NULL DEFAULT 0,
    unmatched_count         INTEGER NOT NULL DEFAULT 0,
    unavailable_count       INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id                      BIGSERIAL PRIMARY KEY,
    prediction_run_id       BIGINT NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
    player_id               BIGINT,
    player_name             TEXT NOT NULL,
    team_abbreviation       TEXT,
    matchup                 TEXT,
    game_id                 TEXT,
    game_date               TEXT,
    game_status             TEXT,
    bookmaker                TEXT,
    sportsbook_line          NUMERIC,
    model_projection         NUMERIC,
    edge                     NUMERIC,
    direction                TEXT CHECK (direction IN ('OVER', 'UNDER', 'NEUTRAL')),
    qualified                 BOOLEAN NOT NULL DEFAULT FALSE,
    prediction_status         TEXT NOT NULL,
    reason                    TEXT,
    model_version             TEXT NOT NULL,
    generated_at_utc          TIMESTAMPTZ NOT NULL,
    latest_game_date          TEXT,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_run_id
    ON prediction_snapshots(prediction_run_id);
CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_player_id
    ON prediction_snapshots(player_id);
CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_game_id
    ON prediction_snapshots(game_id);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id                          BIGSERIAL PRIMARY KEY,
    prediction_snapshot_id      BIGINT NOT NULL UNIQUE REFERENCES prediction_snapshots(id) ON DELETE CASCADE,
    actual_points               NUMERIC,
    result_status                TEXT NOT NULL CHECK (
        result_status IN ('PENDING', 'WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'UNAVAILABLE')
    ),
    game_status                  TEXT,
    source                       TEXT NOT NULL,
    settled_at                   TIMESTAMPTZ,
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_status
    ON prediction_outcomes(result_status);
