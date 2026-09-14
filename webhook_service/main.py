"""
Step 13: a minimal, independently deployable HTTP service whose ONLY job
is to receive and verify Stripe webhooks and hand them to the same
StripeBillingProvider/accounts_repository/db_connection code the
Streamlit apps use. No subscription/entitlement logic is duplicated here
-- see src/services/billing_provider.py::StripeBillingProvider.handle_webhook_event
for all of it.

Why this exists instead of a route inside apps/publicapp.py or
apps/adminapp.py (Step 13 Phase 8 audit finding): Streamlit serves only
its own app protocol on a single route -- there is no supported way to
register an arbitrary POST endpoint that receives the raw, unparsed
request body Stripe's signature verification requires (Streamlit's own
HTTP surface is not meant to be extended with custom routes, and doing
so through unsupported internals would be fragile and unsafe for a
security-critical endpoint). A separate, tiny FastAPI process is the
correct, low-complexity answer -- not a second application, just this
one route (plus a health check).

Run locally:
    uvicorn webhook_service.main:app --reload --port 8000

Then, with the Stripe CLI (test mode):
    stripe listen --forward-to localhost:8000/stripe/webhook
    stripe trigger checkout.session.completed

Deploy: anywhere that can run a small long-lived Python HTTP process
reachable by Stripe over HTTPS (Render, Railway, Fly.io, a single
container, a small VM, ...) -- see the Step 13 report for exact setup.
This process needs the SAME environment variables as the Streamlit apps
for billing/DB config (STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET,
STRIPE_PRO_PRICE_ID, STRIPE_SUCCESS_URL, STRIPE_CANCEL_URL,
STRIPE_PORTAL_RETURN_URL, DATABASE_URL) -- it shares billing_config.py
and db_connection.py with them, not a separate configuration surface.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from src.services import readiness
from src.services.billing_provider import (
    BillingNotConfiguredError,
    WebhookVerificationError,
    get_billing_provider,
)
from src.services.observability import get_logger, log_event

_logger = get_logger("webhook_service")

app = FastAPI(title="NBA Player Performance Pipeline -- Billing Webhook")


@app.get("/health")
def health() -> dict:
    """Liveness only -- 'is this process running'. Never touches config,
    the database, or Stripe. See /ready for an actual readiness check."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """
    Readiness -- config valid, DB reachable, required schema present,
    Stripe config present (Step 14 Phase 3). Deliberately does NOT call
    Stripe itself (there is nothing meaningful to check there beyond
    "is the secret set", which is a config check, not a network call) --
    only a real Postgres connection + schema check, which is the one
    check in src/services/readiness.py that isn't free. Returns 200 when
    ready, 503 otherwise, with a minimal machine-readable body that never
    includes secrets or raw exception text.
    """
    report = readiness.check_webhook_service_readiness(deep=True)
    status_code = 200 if report.is_ready else 503
    return JSONResponse(status_code=status_code, content=report.to_dict())


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> Response:
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    provider = get_billing_provider()

    try:
        result = provider.handle_webhook_event(payload=payload, signature=signature)
    except WebhookVerificationError:
        # Signature/payload could not be verified -- reject outright.
        # Never echo the payload/signature back, and never log the
        # webhook secret or payload contents (see
        # tests/test_step13_static_guards.py's token/secret-logging guards).
        log_event(_logger, "webhook.rejected", severity="warning", reason="invalid_signature")
        return Response(status_code=400, content="Invalid signature")
    except BillingNotConfiguredError:
        # Deployed without STRIPE_WEBHOOK_SECRET set -- a configuration
        # problem to fix, not something Stripe should retry forever.
        log_event(_logger, "webhook.rejected", severity="error", reason="not_configured")
        return Response(status_code=503, content="Webhook not configured")
    except Exception:
        # Any unexpected failure: never leak internals to the caller or
        # the logs. Returning 500 tells Stripe to retry, which is the
        # safe default for a transient failure (e.g. the database was
        # briefly down).
        log_event(_logger, "webhook.rejected", severity="error", reason="unexpected_failure")
        return Response(status_code=500, content="Internal error")

    log_event(
        _logger,
        "webhook.handled",
        severity="info" if result.get("status") == "processed" else "warning",
        stripe_event_id=result.get("event_id"),
        event_type=result.get("event_type"),
        status=result.get("status"),
    )
    return Response(status_code=200, content=json.dumps(result), media_type="application/json")
