# Paid Beta Launch-Candidate Validation

**Validation date:** 2026-09-14
**Git commit tested (base):** `ead1586` (HEAD, `main`, matches `origin/main`) — Step 15 changes below are validated on top of this commit but not committed.
**Environment class:** local development (Windows, no live Postgres/Supabase/Stripe/GitHub network access from this environment).

## What this validation covered

- Full offline automated test suite (unit, integration, static-guard, and new live-execution acceptance tests).
- Real, headless execution of `apps/publicapp.py` and `apps/adminapp.py` via Streamlit's `AppTest` API (not just static analysis) under multiple realistic configurations: unconfigured, dev-auth-only, unreachable database, ADMIN-override-granted, legacy-key bootstrap.
- `webhook_service`'s `/health` and `/ready` endpoints via `TestClient`, healthy and unhealthy.
- `scripts/production_smoke_test.py` run locally.
- A focused secret-leak audit (grep sweep + one empirical `requests` exception-message test) that found and fixed a real, previously-unknown leak vector.
- A read-only production database schema inspection (user-executed, results reported back and recorded below).
- Code-level review of `docs/PRODUCTION_DEPLOYMENT.md`, `docs/OPERATIONS_RUNBOOK.md`, `docs/PAID_BETA_CHECKLIST.md` against the actual current codebase (paths, commands, endpoints, admin controls).

## What this validation did NOT cover (requires the user's own external access)

- A real Stripe **test-mode** checkout → webhook → entitlement round trip against a deployed `webhook_service`.
- A real Supabase sign-in against a live Supabase project.
- A real GitHub Actions `workflow_dispatch` run.
- Any check against the actual production/beta Neon database beyond the one read-only schema inspection the user ran and reported (see below).

## Database investigation (resumed per explicit user instruction)

A leftover debug script (`repair_db.py`, added during a prior Supabase-debugging session and never removed) contained `ALTER TABLE public.subscriptions SET SCHEMA legacy`. Before any further DB-dependent validation, the user ran a read-only inspection against the actual dev/beta database and reported:

- `public.subscriptions` — **confirmed present**
- `legacy.subscriptions` — **confirmed absent**
- No other tables found in a `legacy` schema
- **Conclusion: no database repair needed.** The database is consistent with what the application code expects (unqualified `subscriptions` references resolving via the default `search_path`).
- DB-dependent validation was resumed only after this was confirmed.

`repair_db.py` was not executed as part of this validation and is now removed from the repository (see Defects Found/Fixed).

## Launch blockers found in code

**None remaining.** Two real defects were found and fixed (see below); all fixes are verified by new regression tests and the full suite is green.

## Known limitations (not blockers, but real)

- No secure cross-browser-refresh session persistence (Streamlit platform limitation, documented since Step 12 — unchanged).
- `past_due` Stripe subscriptions get no grace period before losing PRO (deliberate policy from Step 13, unchanged).
- Several pre-existing `st.error(f"...{e}")` call sites for Google Sheets/cache-refresh failures were reviewed and judged low-risk (their exception sources don't carry credentials) — not modified, per "smallest possible change" scope.
- `production_smoke_test.py --deep` cannot report READY from this sandboxed environment (no real secrets available here) — this is the expected, correct behavior of an intentionally unconfigured environment, not a code defect. It must be re-run in the actual deployment target.

## Final recommendation

**GO for paid beta**, contingent entirely on the external deployment/configuration steps in `docs/PAID_BETA_CHECKLIST.md` (Stripe test-mode webhook verification, Supabase production config, GitHub Actions secrets, admin bootstrap, Neon backup confirmation) — none of which are code-level gaps. See the Step 15 report's full blocker/checklist breakdown for exactly what remains and who must do it.
