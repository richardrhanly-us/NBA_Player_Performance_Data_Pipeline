-- Step 11: accounts, subscriptions, and entitlement overrides.
--
-- Purely additive -- applies to the SAME Postgres/Neon database as
-- prediction_runs/prediction_snapshots/prediction_outcomes
-- (migrations/0001_create_prediction_history.sql). No existing table is
-- touched, altered, or dropped.
--
-- Applied the same way as 0001: via src/services/migrations.py, run
-- manually with scripts/apply_prediction_history_migrations.py (that
-- script applies every migrations/*.sql not yet recorded in
-- schema_migrations -- no new script is needed for this file).
--
-- Design notes:
--   * `users.auth_subject` is the identity anchor -- the id assigned by
--     whichever auth provider authenticated the session (Supabase user
--     id, or a deterministic "dev:<email>" subject from the local dev
--     fallback provider). UNIQUE(auth_provider, auth_subject) is the
--     real identity key; email is NOT unique on its own, since it is
--     never trusted as identity proof by itself (a user could in
--     principle exist under more than one provider).
--   * `subscriptions` is reserved for real billing-provider-synced
--     state (Step 12+). Nothing in Step 11 writes to it in production;
--     admin-granted PRO access goes through `entitlement_overrides`
--     instead, so manual grants and real billing sync never collide.
--   * `entitlement_overrides` rows are never deleted -- "revoke" sets
--     enabled=FALSE so the audit trail (who granted what, when, why)
--     stays intact. At most one ENABLED override per user is expected
--     (the repository disables any prior enabled override before
--     inserting a new one), but this is an application-level
--     invariant, not a DB constraint, to keep the schema simple.

CREATE TABLE IF NOT EXISTS users (
    id                  BIGSERIAL PRIMARY KEY,
    email               TEXT NOT NULL,
    display_name        TEXT,
    auth_provider       TEXT NOT NULL,
    auth_subject        TEXT NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (auth_provider, auth_subject)
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS subscriptions (
    id                          BIGSERIAL PRIMARY KEY,
    user_id                     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider                    TEXT NOT NULL DEFAULT 'none',
    provider_customer_id        TEXT,
    provider_subscription_id    TEXT,
    plan_key                    TEXT NOT NULL,
    status                      TEXT NOT NULL CHECK (
        status IN ('active', 'trialing', 'past_due', 'canceled', 'incomplete', 'none')
    ),
    current_period_start        TIMESTAMPTZ,
    current_period_end          TIMESTAMPTZ,
    cancel_at_period_end        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_provider_subscription_id
    ON subscriptions(provider_subscription_id);

CREATE TABLE IF NOT EXISTS entitlement_overrides (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    override_tier       TEXT NOT NULL CHECK (override_tier IN ('FREE', 'PRO', 'ADMIN')),
    enabled             BOOLEAN NOT NULL DEFAULT TRUE,
    reason              TEXT,
    expires_at          TIMESTAMPTZ,
    created_by          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entitlement_overrides_user_id ON entitlement_overrides(user_id);
CREATE INDEX IF NOT EXISTS idx_entitlement_overrides_user_enabled
    ON entitlement_overrides(user_id, enabled);
