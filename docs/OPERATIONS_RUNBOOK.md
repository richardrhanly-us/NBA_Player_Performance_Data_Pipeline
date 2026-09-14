# Operations Runbook

Practical, command-oriented reference for diagnosing and remediating production issues. See `docs/PRODUCTION_DEPLOYMENT.md` for initial setup and `docs/PAID_BETA_CHECKLIST.md` for the launch gate.

## Service inventory & health checks

| Service | Liveness | Readiness |
|---|---|---|
| `webhook_service` | `GET /health` → `{"status": "ok"}` | `GET /ready` → 200 if ready, 503 otherwise, JSON body lists every check |
| Public/Admin Streamlit apps | App loads at all | No dedicated endpoint — run `python scripts/production_smoke_test.py` |
| Automation (GitHub Actions) | Workflow run visible in the Actions tab | `python scripts/production_smoke_test.py` (checks the `automation` process) |
| Database | — | `python scripts/check_schema_readiness.py` |

```
curl https://<webhook-host>/health
curl https://<webhook-host>/ready
python scripts/production_smoke_test.py --deep
```

## Diagnostic commands

```
# Full readiness report for all four process types (read-only)
python scripts/production_smoke_test.py --deep

# Is the DB schema migrated?
DATABASE_URL=<url> python scripts/check_schema_readiness.py

# Apply any pending migrations (idempotent)
DATABASE_URL=<url> python scripts/apply_prediction_history_migrations.py
```

**Logs to inspect:**
- `webhook_service` stdout: one JSON line per event, `event` field one of `webhook.processed`, `webhook.duplicate`, `webhook.failed`, `webhook.rejected`.
- Streamlit app stdout: `nba_pipeline.auth` logger (`auth.sign_in_failed`, `auth.session_expired`, `auth.session_refreshed`), `nba_pipeline.billing` logger (same webhook events, emitted from `StripeBillingProvider` regardless of which process handled it), `nba_pipeline.admin_ops` logger (`automation.db_unavailable`).
- GitHub Actions: the workflow's own job log + step summary (`prediction-cycle.yml`, `settlement-cycle.yml`).

None of these logs ever contain a password, access/refresh token, Supabase key, Stripe secret/webhook key, or a full webhook payload — see `src/services/observability.py`.

## Common failure symptoms → checks

| Symptom | Check |
|---|---|
| Edge Board shows "temporarily unavailable" | `ODDS_API_KEY` set? Odds API status? |
| Prediction history section shows "unavailable" | `DATABASE_URL` set? `check_schema_readiness.py` |
| Sign-in fails for everyone | Supabase status page; `SUPABASE_URL`/`SUPABASE_ANON_KEY` correct? |
| Admin app won't load past login | Is there a real ADMIN user? `LEGACY_ADMIN_KEY_ENABLED` + `ADMIN_KEY` set for break-glass access |
| User paid but still shows FREE | See [Billing Incident: customer paid but still FREE](#customer-paid-but-still-free) |
| Board hasn't updated in hours | Admin app → Overview → Prediction Automation Health → Freshness/Latest run |
| GitHub Actions workflow failing | Actions tab → workflow run → job log; re-run via `workflow_dispatch` |

## Billing incidents

### webhook_service down
Stripe retries failed webhook deliveries automatically (with backoff) for several days. Once `webhook_service` is back up, retried events arrive and are processed idempotently (`stripe_events` table, unique `stripe_event_id`) — no manual replay needed for anything within Stripe's retry window. For anything older, use the Stripe Dashboard's "resend" action on the specific event.

### Stripe events delayed
Check `GET /ready` on `webhook_service` and the admin app's Billing Health section (`last_processed_at`). If events are queued in Stripe but not arriving, check the endpoint URL/secret in the Stripe Dashboard match this deployment.

### Duplicate events
No action needed — `try_claim_stripe_event()` makes duplicate delivery a no-op by construction (see the Step 13 report).

### Incorrect local subscription state / customer paid but still FREE
1. Admin app → Users & Access → select the user → check "Billing" panel (provider, Stripe customer present, subscription status).
2. If no Stripe customer is mapped, or the subscription row looks stale, the fix is a **read from Stripe, not a manual DB edit**: use `StripeBillingProvider.sync_subscription(user_id=...)` (a small Python REPL/script using `get_billing_provider()`, or extend the smoke test invocation for one-off use) to re-pull the subscription from Stripe and write it via the same repository path a webhook would use.
3. If genuinely urgent and Stripe access confirms the customer is paid, use the admin **Grant Temporary/Permanent PRO override** button (Users & Access tab) as an immediate stopgap — this is independent of Stripe and never requires a raw DB edit.

### Customer canceled but still PRO
Same diagnostic path as above. A canceled Stripe subscription should arrive as `customer.subscription.deleted` and flip local status to `canceled` (→ FREE). If it hasn't synced, `sync_subscription()` as above. Never hand-edit the `subscriptions` table.

### Stripe outage
Checkout/portal creation will fail with a generic "temporarily unavailable" message (`BillingProvider` catches the failure — see `src/services/billing_session.py`). Authentication and prediction features are unaffected (billing is decoupled by design). No action beyond waiting for Stripe's status to recover.

### DB outage during webhook handling
`webhook_service` returns 500, which tells Stripe to retry. Once the DB recovers, the retried delivery is processed normally.

## Auth incidents

### Supabase outage
Sign-in/sign-up fail with a generic message; **existing sessions already in `st.session_state` keep working** until their access token expires (≈1 hour) or the periodic revalidation call fails (every ~60s, only for sessions already past the revalidation window — see the Step 12 report). Prediction/Edge Board features are unaffected for anonymous/already-signed-in users.

### Disabled account
Handled automatically — `get_current_user()` force-signs-out a disabled user on its very next check (see the Step 11/12 reports). No manual action needed beyond disabling the account via Users & Access.

### Compromised account
1. Admin app → Users & Access → select the user → **Disable Account**. Takes effect on that user's next request, everywhere.
2. If they had an ADMIN override, it is *not* automatically revoked by disabling the account (disabling forces logout regardless of tier) — but explicitly **Revoke Override** as well for a clean state before ever reactivating.

### Legacy admin-key emergency use
Only works when `LEGACY_ADMIN_KEY_ENABLED=true` and `ADMIN_KEY` is set. Every action taken this way is tagged `source="legacy_admin_key"` in the Admin Logs, distinct from real authenticated-admin actions — check the Logs tab after an emergency session to review exactly what was done.

### Turning the legacy key off
Set `LEGACY_ADMIN_KEY_ENABLED=false` (or unset it) and restart/redeploy the admin app. Any already-open legacy session is also killed on its next check (see Step 12's kill-switch behavior) — no separate revocation step needed.

### Rotating Supabase keys
1. Generate the new anon key in the Supabase dashboard.
2. Update `SUPABASE_ANON_KEY` on both Streamlit apps and redeploy.
3. Existing sessions are unaffected (the anon key is used for new sign-ins/token refresh, not embedded in an issued session) — no forced logout needed.

## Automation incidents

### Prediction cycle failed
Admin app → Overview → "Latest failed run" (Step 14 addition). Cross-reference with the GitHub Actions job log for the same timestamp. Re-run manually:
```
gh workflow run prediction-cycle.yml
```
(or the Actions tab → "Run workflow", which works regardless of the `PREDICTION_AUTOMATION_ENABLED` kill switch — `workflow_dispatch` always executes).

### Settlement cycle failed / stale
Admin app → Overview → "Last settlement activity". Re-run:
```
gh workflow run settlement-cycle.yml
```

### Stale board
Check "Freshness" in the admin Overview tab. If `STALE`, first check whether games are scheduled today (`WAITING_FOR_PROPS`/`NO_GAMES` are expected-normal, not stale) — see `src/services/orchestration.py::compute_board_freshness`.

### Odds API / NBA provider unavailable
Both degrade independently and are visible via the Edge Board's own "temporarily unavailable" messaging and the automation run's `unmatched_count`/`unavailable_count` fields on the relevant `prediction_runs` row.

### Automation kill switch
```
# Disable immediately (GitHub repo Settings → Secrets and variables → Actions → Variables)
PREDICTION_AUTOMATION_ENABLED=false
```
Manual `workflow_dispatch` runs still work regardless of this switch (for controlled testing/recovery).

## Rollback guidance

- **Streamlit apps**: redeploy the previous known-good commit/tag. No DB rollback needed for a pure code revert (migrations are additive-only, so an older code version still works against a newer schema).
- **webhook_service**: same — redeploy the previous version. Stripe will retry any events that failed during the bad window.
- **Migrations**: this project has no down-migrations by design (every migration so far is additive-only — see `migrations/000{1,2,3}_*.sql`'s docstrings). A schema rollback is not expected to ever be necessary; if one genuinely is, write a new forward migration that reverses the specific change rather than editing history.

## Escalation / remediation flow

1. Check the relevant health/readiness endpoint or `production_smoke_test.py --deep`.
2. Check the relevant service's structured logs for the specific `event` name.
3. Check the admin app's operational sections (Overview → automation health, Users & Access → Billing Health).
4. Apply the specific remediation above.
5. Re-run `production_smoke_test.py --deep` to confirm recovery.
