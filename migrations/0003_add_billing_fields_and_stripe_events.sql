-- Step 13: Stripe billing integration -- additive schema changes only.
--
-- Applies to the SAME Postgres/Neon database as migrations 0001/0002.
-- No existing table is dropped or destructively altered; every change
-- here is a new nullable column, a new table, or a new index.
--
-- Design notes:
--   * users.stripe_customer_id is the stable local-user <-> Stripe
--     Customer mapping, set once (see
--     accounts_repository.set_user_stripe_customer_id, which only ever
--     sets it when currently NULL) the first time a user starts
--     checkout, and reused for every subsequent checkout/portal call --
--     this is what prevents duplicate Stripe customers. It is looked up
--     by webhook handling to resolve which local user a Stripe event
--     belongs to; email is NEVER used for this mapping (see the Step 13
--     report's security-review section).
--   * subscriptions.stripe_price_id / last_synced_at are diagnostic/
--     policy fields -- stripe_price_id records which Price was active
--     (useful once more than one paid plan exists), last_synced_at
--     records when this row was last written by a webhook, shown in
--     the admin Users & Access tab.
--   * The partial unique index on subscriptions(provider,
--     provider_subscription_id) makes webhook-driven upserts idempotent
--     by construction: the same Stripe subscription id can only ever
--     correspond to one row (see
--     accounts_repository.sync_subscription_from_stripe).
--   * stripe_events is the webhook idempotency ledger (Step 13 Phase
--     14): a UNIQUE stripe_event_id means Stripe's automatic retries of
--     the same event become a no-op rather than double-processing, and
--     processing_status/error_message make a failed event diagnosable
--     without needing external log access.

ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_stripe_customer_id
    ON users(stripe_customer_id)
    WHERE stripe_customer_id IS NOT NULL;

ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS stripe_price_id TEXT;
ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ;

CREATE UNIQUE INDEX IF NOT EXISTS idx_subscriptions_provider_subscription_unique
    ON subscriptions(provider, provider_subscription_id)
    WHERE provider_subscription_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS stripe_events (
    id                  BIGSERIAL PRIMARY KEY,
    stripe_event_id     TEXT NOT NULL UNIQUE,
    event_type          TEXT NOT NULL,
    received_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at        TIMESTAMPTZ,
    processing_status   TEXT NOT NULL DEFAULT 'received' CHECK (
        processing_status IN ('received', 'processed', 'failed')
    ),
    error_message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_stripe_events_status ON stripe_events(processing_status);
