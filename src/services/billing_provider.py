"""
Step 11/13: the payment-ready boundary, now with a real Stripe
implementation behind it.

`BillingProvider` is the interface Streamlit pages and the standalone
webhook_service call. `stripe` is imported ONLY in this module -- see
tests/test_step13_static_guards.py, which fails the build if a `stripe`
import ever appears in apps/*.py or src/services/entitlement_service.py.
EntitlementService never imports this module, and this module never
imports EntitlementService's opposite direction dependency -- the data
flow is one-way: Stripe webhook -> this module -> accounts_repository's
subscriptions table -> EntitlementService reads local state (see
src/services/entitlement_service.py::compute_effective_tier). Stripe is
never read directly from page rendering.

`NullBillingProvider` is used whenever Stripe isn't configured
(src/services/billing_config.py::is_stripe_configured() is False) --
billing actions are refused with a clear, typed error; authentication
and prediction features are entirely unaffected (see get_billing_provider()).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from src.domain.accounts import AccessTier
from src.services import accounts_repository, billing_config, entitlement_service


class BillingNotConfiguredError(RuntimeError):
    """Raised by NullBillingProvider for any operation that requires a
    real billing provider, and by StripeBillingProvider for an action
    that needs configuration it doesn't have (e.g. no portal return URL
    set yet). Distinct from a generic RuntimeError so callers (and
    tests) can tell "billing isn't enabled/fully set up" apart from an
    actual failure of a configured provider."""


class BillingActionDeniedError(RuntimeError):
    """Raised when a billing action is refused for authorization
    reasons that are true regardless of Stripe's own state -- an
    anonymous/disabled/already-PRO-or-ADMIN account, or an unsupported
    plan key. This is server-side enforcement, independent of (and in
    addition to) whatever the calling page chooses to show or hide."""


class WebhookVerificationError(RuntimeError):
    """Raised when an inbound webhook's signature cannot be verified
    (bad/missing signature, malformed payload). The HTTP layer
    (webhook_service) maps this to a 4xx response so Stripe does not
    treat it as a transient failure to retry indefinitely."""


class BillingProvider(ABC):
    @abstractmethod
    def get_subscription_status(self, *, user_id: int) -> Optional[dict[str, Any]]:
        """Returns the provider's current view of this user's
        subscription, or None if it has none. Must not raise for "no
        subscription" -- only for a genuine provider-side failure."""

    @abstractmethod
    def sync_subscription(self, *, user_id: int) -> None:
        """Pulls the provider's current subscription state for this user
        and writes it via accounts_repository.sync_subscription_from_stripe."""

    @abstractmethod
    def create_checkout_session(self, *, user_id: int, plan_key: str) -> str:
        """Returns a URL the user is redirected to to purchase `plan_key`."""

    @abstractmethod
    def create_customer_portal_session(self, *, user_id: int) -> str:
        """Returns a URL the user is redirected to to manage their
        existing subscription (cancel, update payment method, ...)."""

    @abstractmethod
    def handle_webhook_event(self, *, payload: bytes, signature: str) -> dict:
        """Verifies and processes one inbound billing-provider webhook
        event. Returns a small status dict for diagnostics (never
        secrets); raises WebhookVerificationError for a bad signature."""


class NullBillingProvider(BillingProvider):
    """Used whenever Stripe isn't configured. Billing is decoupled and
    optional -- this makes that explicit rather than leaving the seam
    unimplemented."""

    def get_subscription_status(self, *, user_id: int) -> Optional[dict[str, Any]]:
        return None

    def sync_subscription(self, *, user_id: int) -> None:
        return None

    def create_checkout_session(self, *, user_id: int, plan_key: str) -> str:
        raise BillingNotConfiguredError(
            "Billing is not yet enabled for this deployment -- "
            "create_checkout_session() has no real provider configured."
        )

    def create_customer_portal_session(self, *, user_id: int) -> str:
        raise BillingNotConfiguredError(
            "Billing is not yet enabled for this deployment -- "
            "create_customer_portal_session() has no real provider configured."
        )

    def handle_webhook_event(self, *, payload: bytes, signature: str) -> dict:
        raise BillingNotConfiguredError(
            "Billing is not yet enabled for this deployment -- "
            "handle_webhook_event() has no real provider configured."
        )


# ---------------------------------------------------------------------------
# Stripe subscription-status -> local status policy (Step 13 Phase 9).
#
# EntitlementService only ever grants PRO for local status "active" or
# "trialing" (see entitlement_service._PAID_STATUSES) -- everything below
# is a normalization into the small vocabulary the subscriptions table's
# CHECK constraint already allows ('active', 'trialing', 'past_due',
# 'canceled', 'incomplete', 'none'), decided deliberately, once, here:
#
#   Stripe status         -> local status   -> grants PRO?
#   active                -> active         -> yes
#   trialing              -> trialing       -> yes (trials are supported)
#   past_due              -> past_due       -> NO -- deliberate: no grace
#                                               period in this step. A
#                                               missed payment loses PRO
#                                               immediately rather than
#                                               silently extending access;
#                                               softening this later (e.g.
#                                               an N-day grace window) is
#                                               a policy change for a
#                                               future step, not this one.
#   unpaid                -> past_due       -> no  (same bucket as above)
#   canceled              -> canceled       -> no
#   incomplete            -> incomplete     -> no  (payment never completed)
#   incomplete_expired    -> canceled       -> no  (never activated, now dead)
#   paused                -> canceled       -> no  (pausing isn't a
#                                               separately-supported
#                                               entitlement state in this
#                                               step)
# ---------------------------------------------------------------------------
_STRIPE_STATUS_TO_LOCAL_STATUS = {
    "active": "active",
    "trialing": "trialing",
    "past_due": "past_due",
    "unpaid": "past_due",
    "canceled": "canceled",
    "incomplete": "incomplete",
    "incomplete_expired": "canceled",
    "paused": "canceled",
}


def normalize_stripe_subscription_status(stripe_status: str | None) -> str:
    """Maps a raw Stripe subscription status string onto this app's own
    local status vocabulary -- see the policy table above. Unknown/future
    Stripe statuses fail closed to "canceled" (no PRO) rather than
    raising, since a webhook must never crash on an unrecognized-but-
    otherwise-valid event."""
    if not stripe_status:
        return "none"
    return _STRIPE_STATUS_TO_LOCAL_STATUS.get(stripe_status, "canceled")


def _epoch_to_iso(epoch) -> str | None:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


class StripeBillingProvider(BillingProvider):
    """
    Real Stripe integration. Uses Stripe's classic module-level API
    (`stripe.api_key = ...`, `stripe.checkout.Session.create(...)`, ...)
    -- no Stripe.js, no client-side keys: Checkout/Portal are plain
    server-created, browser-redirected sessions, so STRIPE_SECRET_KEY
    and STRIPE_WEBHOOK_SECRET never need to reach the browser at all.

    Every method that touches the database acquires its own connection
    (via src/services/db_connection.py, same pattern as
    src/services/auth_session.py) -- callers pass only a user_id, never
    a connection, matching the existing BillingProvider interface.
    """

    def __init__(
        self,
        *,
        secret_key: str,
        pro_price_id: str,
        success_url: str,
        cancel_url: str,
        portal_return_url: str,
        webhook_secret: str | None,
        timeout: float = 15.0,
    ):
        import stripe

        self._stripe = stripe
        self._stripe.api_key = secret_key
        self._pro_price_id = pro_price_id
        self._success_url = success_url
        self._cancel_url = cancel_url
        self._portal_return_url = portal_return_url
        self._webhook_secret = webhook_secret
        self._timeout = timeout

    def _get_db_connection(self):
        from src.services.db_connection import get_prediction_db_connection

        return get_prediction_db_connection()

    def _price_id_for_plan(self, plan_key: str) -> str:
        """The ONLY place a Price ID is chosen -- always the one this
        provider instance was constructed with (itself sourced from
        billing_config at construction time, see get_billing_provider()),
        never from a caller-supplied value. A client can request
        plan_key="pro" (a symbolic name our own code controls), never a
        literal Stripe Price ID."""
        if plan_key != "pro":
            raise BillingActionDeniedError(f"Unsupported plan: {plan_key!r}.")
        if not self._pro_price_id:
            raise BillingNotConfiguredError(f"No Price ID configured for plan {plan_key!r}.")
        return self._pro_price_id

    def _get_or_create_customer_id(self, conn, user_row: dict) -> str:
        existing = user_row.get("stripe_customer_id")
        if existing:
            return existing
        customer = self._stripe.Customer.create(
            email=user_row["email"],  # Stripe's own record-keeping only -- never used to look a user back up
            metadata={"local_user_id": str(user_row["id"])},
        )
        accounts_repository.set_user_stripe_customer_id(conn, user_row["id"], customer.id)
        # Re-read in case a concurrent call already won the race and set
        # a different (earlier) customer id -- set_user_stripe_customer_id
        # is a no-op once already set, so the DB is the source of truth.
        refreshed = accounts_repository.get_user_by_id(conn, user_row["id"])
        return refreshed.get("stripe_customer_id") or customer.id

    def _require_actionable_account(self, conn, user_id: int) -> dict:
        user_row = accounts_repository.get_user_by_id(conn, user_id)
        if user_row is None or not user_row.get("is_active", True):
            raise BillingActionDeniedError("This account cannot start checkout.")
        return user_row

    def create_checkout_session(self, *, user_id: int, plan_key: str) -> str:
        price_id = self._price_id_for_plan(plan_key)
        conn = self._get_db_connection()
        try:
            user_row = self._require_actionable_account(conn, user_id)

            subscription = accounts_repository.get_latest_subscription(conn, user_id)
            override = accounts_repository.get_active_override(conn, user_id)
            tier = entitlement_service.compute_effective_tier(
                user=user_row, subscription=subscription, active_override=override
            )
            if tier != AccessTier.FREE:
                raise BillingActionDeniedError(
                    "This account already has PRO access and does not need checkout."
                )

            customer_id = self._get_or_create_customer_id(conn, user_row)
            session = self._stripe.checkout.Session.create(
                mode="subscription",
                customer=customer_id,
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=self._success_url,
                cancel_url=self._cancel_url,
                client_reference_id=str(user_row["id"]),
                metadata={"local_user_id": str(user_row["id"])},
                subscription_data={"metadata": {"local_user_id": str(user_row["id"])}},
            )
            return session.url
        finally:
            conn.close()

    def create_customer_portal_session(self, *, user_id: int) -> str:
        conn = self._get_db_connection()
        try:
            user_row = self._require_actionable_account(conn, user_id)
            customer_id = user_row.get("stripe_customer_id")
            if not customer_id:
                raise BillingActionDeniedError("This account has no billing history yet.")
            session = self._stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=self._portal_return_url,
            )
            return session.url
        finally:
            conn.close()

    def get_subscription_status(self, *, user_id: int) -> Optional[dict[str, Any]]:
        """Diagnostic-only read (e.g. an admin "force resync" action) --
        never used to decide entitlements directly; entitlements always
        come from the local subscriptions table."""
        conn = self._get_db_connection()
        try:
            return accounts_repository.get_latest_subscription(conn, user_id)
        finally:
            conn.close()

    def sync_subscription(self, *, user_id: int) -> None:
        conn = self._get_db_connection()
        try:
            existing = accounts_repository.get_latest_subscription(conn, user_id)
            if existing is None or not existing.get("provider_subscription_id"):
                return
            subscription_obj = self._stripe.Subscription.retrieve(
                existing["provider_subscription_id"]
            )
            self._sync_subscription_object(conn, subscription_obj)
        finally:
            conn.close()

    # -- webhook handling ---------------------------------------------------

    def handle_webhook_event(self, *, payload: bytes, signature: str) -> dict:
        if not self._webhook_secret:
            raise BillingNotConfiguredError("STRIPE_WEBHOOK_SECRET is not configured.")

        try:
            event = self._stripe.Webhook.construct_event(
                payload, signature, self._webhook_secret
            )
        except ValueError as e:
            raise WebhookVerificationError("Invalid webhook payload.") from e
        except self._stripe.error.SignatureVerificationError as e:
            raise WebhookVerificationError("Invalid webhook signature.") from e

        event_id = event["id"]
        event_type = event["type"]

        conn = self._get_db_connection()
        try:
            claimed = accounts_repository.try_claim_stripe_event(conn, event_id, event_type)
            if not claimed:
                return {"status": "duplicate", "event_id": event_id, "event_type": event_type}

            try:
                self._dispatch_event(conn, event_type, event["data"]["object"])
                accounts_repository.mark_stripe_event_processed(conn, event_id)
                return {"status": "processed", "event_id": event_id, "event_type": event_type}
            except Exception as e:  # noqa: BLE001 -- captured for diagnosis, never re-raised with secrets
                accounts_repository.mark_stripe_event_failed(conn, event_id, str(e))
                return {"status": "failed", "event_id": event_id, "event_type": event_type}
        finally:
            conn.close()

    def _resolve_user_id(self, conn, *, customer_id: str | None, metadata: dict | None) -> int | None:
        """Identity resolution priority: stored Stripe customer id
        mapping first (the stable, persisted link), metadata's
        local_user_id second (present on the session/subscription this
        app itself created) -- email is NEVER used to resolve a webhook
        event to a local user."""
        if customer_id:
            user_row = accounts_repository.get_user_by_stripe_customer_id(conn, customer_id)
            if user_row is not None:
                return user_row["id"]
        if metadata and metadata.get("local_user_id"):
            try:
                return int(metadata["local_user_id"])
            except (TypeError, ValueError):
                return None
        return None

    def _sync_subscription_object(self, conn, subscription_obj) -> None:
        customer_id = subscription_obj.get("customer")
        metadata = subscription_obj.get("metadata") or {}
        user_id = self._resolve_user_id(conn, customer_id=customer_id, metadata=metadata)
        if user_id is None:
            raise RuntimeError(
                "Could not resolve a local user for this Stripe subscription event."
            )

        if customer_id:
            accounts_repository.set_user_stripe_customer_id(conn, user_id, customer_id)

        price_id = None
        items = (subscription_obj.get("items") or {}).get("data") or []
        if items:
            price_id = (items[0].get("price") or {}).get("id")

        accounts_repository.sync_subscription_from_stripe(
            conn,
            user_id=user_id,
            provider_customer_id=customer_id,
            provider_subscription_id=subscription_obj["id"],
            plan_key="pro",
            status=normalize_stripe_subscription_status(subscription_obj.get("status")),
            current_period_start=_epoch_to_iso(subscription_obj.get("current_period_start")),
            current_period_end=_epoch_to_iso(subscription_obj.get("current_period_end")),
            cancel_at_period_end=bool(subscription_obj.get("cancel_at_period_end")),
            stripe_price_id=price_id,
        )

    def _handle_checkout_completed(self, conn, session_obj) -> None:
        """checkout.session.completed fires immediately on return from
        Checkout -- fetching the full Subscription lets PRO show up as
        soon as possible without waiting for customer.subscription.created
        to arrive separately. If this fails (or that event never
        arrives, e.g. delivery reordering), customer.subscription.*
        events still independently keep local state correct -- no
        entitlement logic is duplicated here, only re-triggered."""
        subscription_id = session_obj.get("subscription")
        if not subscription_id:
            return  # not a subscription-mode checkout (shouldn't happen for plan_key="pro")
        subscription_obj = self._stripe.Subscription.retrieve(subscription_id)
        self._sync_subscription_object(conn, subscription_obj)

    def _handle_invoice_event(self, conn, invoice_obj) -> None:
        """Re-syncs the associated subscription from Stripe -- covers
        renewal period-date advances and payment-failure status changes
        without re-implementing status mapping here (reuses
        _sync_subscription_object, so there is exactly one place that
        interprets a Stripe subscription's fields)."""
        subscription_id = invoice_obj.get("subscription")
        if not subscription_id:
            return
        subscription_obj = self._stripe.Subscription.retrieve(subscription_id)
        self._sync_subscription_object(conn, subscription_obj)

    def _dispatch_event(self, conn, event_type: str, obj: dict) -> None:
        if event_type == "checkout.session.completed":
            self._handle_checkout_completed(conn, obj)
        elif event_type in (
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
            "customer.subscription.paused",
            "customer.subscription.resumed",
        ):
            self._sync_subscription_object(conn, obj)
        elif event_type in ("invoice.payment_succeeded", "invoice.payment_failed"):
            self._handle_invoice_event(conn, obj)
        # Any other event type is safely ignored -- not an error, just
        # nothing this app needs to react to.


def get_billing_provider() -> BillingProvider:
    """The one factory every caller uses. Stripe configured (see
    billing_config.is_stripe_configured()) -> StripeBillingProvider;
    otherwise -> NullBillingProvider. Billing absence never affects
    authentication or prediction features -- only billing actions."""
    if billing_config.is_stripe_configured():
        return StripeBillingProvider(
            secret_key=billing_config.stripe_secret_key(),
            pro_price_id=billing_config.stripe_pro_price_id(),
            success_url=billing_config.stripe_success_url(),
            cancel_url=billing_config.stripe_cancel_url(),
            portal_return_url=billing_config.stripe_portal_return_url(),
            webhook_secret=billing_config.stripe_webhook_secret(),
        )
    return NullBillingProvider()
