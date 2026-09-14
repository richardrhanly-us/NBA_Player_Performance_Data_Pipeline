"""
Step 13: the one place billing-related environment variables are read.
src/services/billing_provider.py calls these functions instead of
touching os.environ directly -- see tests/test_step13_static_guards.py
for the guard that enforces this. Mirrors src/services/auth_config.py's
role for authentication.

No Streamlit import here on purpose -- pure/testable, and this module is
imported by the standalone webhook_service (a plain FastAPI process,
not a Streamlit app) as well as by the Streamlit apps.

Environment variables (see the Step 13 report for full documentation):

    STRIPE_SECRET_KEY          -- server-side Stripe API key (sk_...).
                                   NEVER sent to the browser.
    STRIPE_PUBLISHABLE_KEY     -- optional; unused by the current
                                   redirect-based Checkout/Portal flow
                                   (no Stripe.js is loaded), kept for
                                   documentation/future use only.
    STRIPE_WEBHOOK_SECRET      -- used to verify inbound webhook
                                   signatures (whsec_...). Required only
                                   by webhook_service, not by checkout/
                                   portal creation.
    STRIPE_PRO_PRICE_ID        -- the one Price ID this app will ever
                                   pass to Stripe Checkout. The browser
                                   can never choose a different one --
                                   see billing_provider.py.
    STRIPE_SUCCESS_URL         -- where Stripe redirects after a
                                   completed Checkout (e.g.
                                   "https://app.example.com/?checkout=success").
    STRIPE_CANCEL_URL          -- where Stripe redirects after a
                                   canceled Checkout.
    STRIPE_PORTAL_RETURN_URL   -- where the Stripe Customer Portal
                                   returns the user.
"""

from __future__ import annotations

import os


def stripe_secret_key() -> str | None:
    return os.environ.get("STRIPE_SECRET_KEY") or None


def stripe_publishable_key() -> str | None:
    return os.environ.get("STRIPE_PUBLISHABLE_KEY") or None


def stripe_webhook_secret() -> str | None:
    return os.environ.get("STRIPE_WEBHOOK_SECRET") or None


def stripe_pro_price_id() -> str | None:
    return os.environ.get("STRIPE_PRO_PRICE_ID") or None


def stripe_success_url() -> str | None:
    return os.environ.get("STRIPE_SUCCESS_URL") or None


def stripe_cancel_url() -> str | None:
    return os.environ.get("STRIPE_CANCEL_URL") or None


def stripe_portal_return_url() -> str | None:
    return os.environ.get("STRIPE_PORTAL_RETURN_URL") or None


def is_stripe_configured() -> bool:
    """True once enough configuration exists for checkout/portal
    creation. Deliberately all-or-nothing rather than a partial state --
    a half-configured deployment should behave like billing is entirely
    unavailable (NullBillingProvider), not partially broken."""
    return all(
        (
            stripe_secret_key(),
            stripe_pro_price_id(),
            stripe_success_url(),
            stripe_cancel_url(),
            stripe_portal_return_url(),
        )
    )


def is_webhook_configured() -> bool:
    """The webhook secret is checked separately (and lazily, inside
    StripeBillingProvider.handle_webhook_event) so a deployment can set
    up checkout before the webhook endpoint exists."""
    return bool(stripe_webhook_secret())
