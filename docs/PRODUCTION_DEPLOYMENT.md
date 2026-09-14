# Production Deployment Guide

This document is the exact setup order for deploying the NBA Player Performance Data Pipeline to production, and the environment variables each service needs. It assumes Steps 8–14 of the productization roadmap are complete (accounts/entitlements, Supabase auth, Stripe billing, readiness/observability).

Everything here separates **TEST MODE** (Stripe test-mode keys, safe to break) from **PRODUCTION MODE** (Stripe live-mode keys, real money) explicitly. Do not skip the test-mode verification step.

## Service inventory

| Service | What it is | Stateful? |
|---|---|---|
| Public Streamlit app (`apps/publicapp.py`) | Edge Board, prediction history, account/billing UI | Stateless (reads Postgres/Supabase/Stripe) |
| Admin Streamlit app (`apps/adminapp.py`) | Internal ops dashboard, Users & Access | Stateless |
| `webhook_service` (FastAPI) | Receives/verifies Stripe webhooks | Stateless |
| GitHub Actions workflows | Prediction cycle + settlement automation | Stateless (writes to Postgres) |
| Neon/PostgreSQL | Prediction history, accounts, subscriptions, stripe_events | **Stateful** — the one durable datastore |
| Supabase Auth | Identity provider | Stateful (managed by Supabase) |
| Stripe | Billing provider | Stateful (managed by Stripe) |

Only Postgres is *this project's* stateful system. Supabase and Stripe are managed third parties with their own durability guarantees.

## Setup order

### 1. Provision Neon/Postgres
Create the database. Note the connection string — this is `DATABASE_URL`, needed by every service below.

### 2. Run migrations
```
DATABASE_URL=<production-url> python scripts/apply_prediction_history_migrations.py
```
This applies `migrations/0001` through `0003` (idempotently — safe to re-run). See [Migration Policy](#migration-policy) below for why this is always a manual, explicit step.

### 3. Configure Supabase (TEST MODE first)
Create a Supabase project, enable email/password auth. Note the Project URL and anon key.

### 4. Bootstrap the first ADMIN
Sign in once through the public app (creates a `users` row), then:
```
DATABASE_URL=<production-url> python scripts/create_admin_user.py \
    --auth-provider supabase --auth-subject <supabase-user-uuid> --email you@example.com
```
Or, before Supabase is fully wired up, use the dev-mode form (`--auth-provider dev`) — see `scripts/create_admin_user.py`'s docstring.

### 5. Configure Stripe products/prices (TEST MODE)
In the Stripe Dashboard (test mode), create one recurring Price for PRO. Note the Price ID (`price_...`).

### 6. Deploy `webhook_service`
Deploy `webhook_service/main.py` (Render, Railway, Fly.io, a small container — anywhere that runs a long-lived Python HTTP process reachable over HTTPS). This is a **separate deployment target** from the Streamlit apps — Streamlit cannot host a webhook endpoint that receives a raw request body for signature verification (see the Step 13 report's Phase 8 audit).
```
uvicorn webhook_service.main:app --host 0.0.0.0 --port 8000
```

### 7. Configure the Stripe webhook endpoint (TEST MODE)
In the Stripe Dashboard, add a webhook endpoint pointing at `https://<webhook-host>/stripe/webhook`, subscribed to:
`checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_succeeded`, `invoice.payment_failed`. Copy the signing secret.

### 8. Configure Streamlit secrets
Set the environment variables/secrets listed in [Environment Variables](#environment-variables) below on both Streamlit apps.

### 9. Configure GitHub Actions secrets/variables
`DATABASE_URL`, `ODDS_API_KEY` as repository secrets; `PREDICTION_AUTOMATION_ENABLED` as a repository **variable** (the automation kill switch — see the Step 10 report).

### 10. Validate readiness
```
python scripts/production_smoke_test.py --deep
```
Must print `RESULT: READY` for `public_app`, `admin_app`, and `webhook_service` before proceeding. `automation` readiness requires `ODDS_API_KEY`.

### 11. Run smoke tests
Sign in through the public app (dev or Supabase), confirm the Edge Board loads, confirm the admin app's Overview tab shows automation health without error.

### 12. Enable prediction automation
Set the `PREDICTION_AUTOMATION_ENABLED` GitHub Actions repository variable to `true`. Automation runs on the existing cron schedule (see `.github/workflows/prediction-cycle.yml`).

### 13. Verify webhook sync in Stripe TEST MODE
Run a full test-mode checkout (see [Local/Test-Mode Setup](#local-test-mode-setup)) and confirm:
- the webhook is received (`webhook_service` logs `webhook.processed`)
- the admin app's Users & Access → Billing Health section shows the event as processed
- the test user's tier becomes PRO on the next page load

### 14. Move to LIVE-MODE billing only after test-mode verification
Only after step 13 passes: switch `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, and `STRIPE_PRO_PRICE_ID` to their live-mode equivalents, and add a new live-mode webhook endpoint in the Stripe Dashboard (test-mode and live-mode webhooks are configured separately). Re-run `production_smoke_test.py --deep`.

## Environment variables

### Public app
```
DATABASE_URL
SUPABASE_URL, SUPABASE_ANON_KEY          # or DEV_AUTH_ENABLED=true for local dev only
STRIPE_SECRET_KEY, STRIPE_PRO_PRICE_ID,  # optional — billing degrades gracefully if absent
STRIPE_SUCCESS_URL, STRIPE_CANCEL_URL, STRIPE_PORTAL_RETURN_URL
ODDS_API_KEY                              # optional — Edge Board degrades gracefully if absent
APP_ENV=production
```

### Admin app
```
DATABASE_URL
SUPABASE_URL, SUPABASE_ANON_KEY
LEGACY_ADMIN_KEY_ENABLED=true             # bootstrap only — disable after step 4 above
ADMIN_KEY                                 # the bootstrap key value, if LEGACY_ADMIN_KEY_ENABLED
APP_ENV=production
```

### webhook_service
```
DATABASE_URL
STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET
STRIPE_PRO_PRICE_ID
```

### GitHub Actions (automation)
```
DATABASE_URL            # secret
ODDS_API_KEY            # secret
PREDICTION_AUTOMATION_ENABLED   # variable, "true" to enable
```

Never set `DEV_AUTH_ENABLED=true` or `LEGACY_ADMIN_KEY_ENABLED=true` (beyond initial bootstrap) in production — `python scripts/production_smoke_test.py` fails readiness if dev auth is actually reachable under `APP_ENV=production`.

## Backup / recovery

Neon already provides point-in-time recovery and automated backups for the Postgres database — this project does not build a second backup system on top of that. What matters is *using* Neon's existing capability and knowing what depends on it.

**What must be backed up (Neon PITR covers this):**
- `users`, `subscriptions`, `entitlement_overrides` — account/entitlement state. **Not reconstructable from anywhere else** if lost; this is the only source of truth for who has PRO access and why.
- `prediction_runs`, `prediction_snapshots`, `prediction_outcomes` — prediction history. Snapshots are **not reconstructable** once the underlying sportsbook lines/model version have moved on; treat this as permanent history.
- `stripe_events` — the webhook idempotency ledger. **Partially reconstructable**: Stripe retains its own event log and can re-deliver/re-list events on request, but the local `processing_status`/`error_message` diagnostic trail is not recoverable from Stripe itself.

**What can be reconstructed from Stripe directly** (if the local copy is lost but the DB itself is fine — e.g. a specific row was bad, not a full DB loss): current subscription status, customer id, price id, period dates — via `StripeBillingProvider.sync_subscription()` or the Stripe Dashboard/API directly. Stripe is the authoritative billing record; the local `subscriptions` table is a synchronized cache of it (see the Step 13 report's architecture).

**What can be reconstructed from prediction sources:** nothing meaningfully — a day's sportsbook lines are not retrievable after the fact from the Odds API, so a lost `prediction_snapshots` row for a past day is gone. This is why PITR matters more for this table than for `subscriptions`.

**What cannot be reconstructed at all:** `users.stripe_customer_id`/`auth_subject` mappings pre-dating a restore, and any admin-granted `entitlement_overrides` history (grant/revoke audit trail).

**Recovery order** (full database loss scenario):
1. Restore the Postgres database from Neon's most recent backup/PITR point.
2. Run `python scripts/check_schema_readiness.py` to confirm the restored schema matches what the deployed code expects.
3. For each user with an active Stripe subscription as of the restore point, run `sync_subscription()` (or wait for the next webhook/renewal cycle) to reconcile any billing state that changed between the backup point and the incident.
4. Re-verify admin access: confirm at least one ADMIN account is present in the restored data before disabling any break-glass (legacy key) access.
5. Run `python scripts/production_smoke_test.py --deep` to confirm the restored system is ready.

**Restore validation checklist:**
- [ ] Schema readiness check passes
- [ ] A known test user's tier resolves correctly (matches pre-incident expectation)
- [ ] At least one ADMIN account is present and can sign in
- [ ] `stripe_events` table is present (even if some recent entries are missing — Stripe re-delivery will backfill via retries)
- [ ] `production_smoke_test.py --deep` reports READY

## Migration policy

- Migrations are **always run explicitly by a human**, before deploying the services that depend on the new schema (step 2 above) — never automatically from Streamlit page startup or `webhook_service` startup.
- No service auto-applies migrations. `src/services/readiness.py`'s deep database check only *detects* whether the schema is ready (`is_schema_ready`) — it never applies anything.
- Deployment order is always: **migrate → verify readiness → deploy/restart services**.
