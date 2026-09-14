# Paid Beta Launch Checklist

Concise, checkbox-style gate before charging real users. See `docs/PRODUCTION_DEPLOYMENT.md` for setup detail and `docs/OPERATIONS_RUNBOOK.md` for incident procedures.

## 🚫 DO NOT LAUNCH IF

- [ ] `python scripts/production_smoke_test.py --deep` does not print `RESULT: READY` for `public_app`, `admin_app`, and `webhook_service`
- [ ] `DEV_AUTH_ENABLED` is reachable in production (smoke test's `dev_auth_not_in_production` check fails)
- [ ] No real ADMIN account exists (only the legacy key works)
- [ ] Stripe webhook has not been verified end-to-end in **test mode**
- [ ] `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` are mismatched mode (one live, one test)
- [ ] Migrations have not been applied to the production database

---

## INFRASTRUCTURE
- [ ] Neon/Postgres provisioned, `DATABASE_URL` set on every service
- [ ] `webhook_service` deployed and reachable over HTTPS from the public internet
- [ ] `python scripts/check_schema_readiness.py` passes

## AUTH
- [ ] Supabase project created, `SUPABASE_URL`/`SUPABASE_ANON_KEY` set
- [ ] `APP_ENV=production` set on both Streamlit apps
- [ ] `DEV_AUTH_ENABLED` unset (or explicitly refused by the production hint)
- [ ] A real ADMIN account exists (`scripts/create_admin_user.py` run once)
- [ ] `LEGACY_ADMIN_KEY_ENABLED` set back to `false` after bootstrap

## BILLING
- [ ] Stripe Price created for PRO (test mode verified, then live mode)
- [ ] `STRIPE_SECRET_KEY`, `STRIPE_PRO_PRICE_ID`, `STRIPE_SUCCESS_URL`, `STRIPE_CANCEL_URL`, `STRIPE_PORTAL_RETURN_URL` set
- [ ] Full test-mode checkout → webhook → PRO entitlement verified end-to-end
- [ ] Customer Portal session tested for a real test-mode customer
- [ ] Live-mode webhook endpoint configured separately from test-mode

## DATABASE
- [ ] Migrations `0001`–`0003` applied
- [ ] Neon backup/point-in-time-recovery enabled (see backup/recovery notes below)

## WEBHOOKS
- [ ] `GET /ready` on `webhook_service` returns 200
- [ ] `STRIPE_WEBHOOK_SECRET` matches the endpoint configured in the Stripe Dashboard
- [ ] `stripe trigger checkout.session.completed` (test mode) processed successfully and visible in admin Billing Health

## AUTOMATION
- [ ] `ODDS_API_KEY` configured as a GitHub Actions secret
- [ ] `PREDICTION_AUTOMATION_ENABLED` reviewed (on only when ready for live automation)
- [ ] A manual `workflow_dispatch` run of `prediction-cycle.yml` succeeds

## SECURITY
- [ ] No secrets appear in `production_smoke_test.py` output
- [ ] No secrets appear in `webhook_service` `/ready` output
- [ ] Legacy admin key disabled (or explicitly, temporarily accepted with a plan to disable)
- [ ] `python -m pytest -q` passes in full

## PUBLIC APP
- [ ] Edge Board loads for an anonymous visitor (truncated view)
- [ ] Sign-up → sign-in → Upgrade to PRO flow works end-to-end
- [ ] "Manage Billing" opens the Stripe Customer Portal for a PRO user

## ADMIN APP
- [ ] Non-admin cannot reach the admin app's tabs (denied at the gate)
- [ ] Users & Access shows accurate tier/subscription/override state
- [ ] Billing Health section shows recent Stripe events

## TESTING
- [ ] Full test suite passes: `python -m pytest -q`
- [ ] `python scripts/production_smoke_test.py --deep` passes

## MONITORING
- [ ] `webhook_service` logs are being collected somewhere reviewable
- [ ] Someone is watching GitHub Actions workflow run status

## BACKUP
- [ ] Neon backup/PITR confirmed enabled (see `docs/PRODUCTION_DEPLOYMENT.md`'s backup section, to be read alongside this checklist)
- [ ] Restore validation checklist reviewed at least once (not necessarily executed against production)

## ROLLBACK
- [ ] Previous known-good deployment tag/commit identified for both Streamlit apps and `webhook_service`
- [ ] Team knows the automation kill switch (`PREDICTION_AUTOMATION_ENABLED=false`) and how to flip it
