"""
Step 11: the payment-ready boundary.

`BillingProvider` is the interface any future real billing integration
(Stripe first, most likely) implements. Nothing in this module imports
Stripe or any provider SDK -- see the Step 11 report's architectural-
guards section (tests/test_step11_static_guards.py), which fails the
build if a `stripe` import ever appears anywhere near entitlement logic.

`NullBillingProvider` is what the app actually uses today: billing is
NOT live in Step 11. It answers "no subscription info available" for
reads and refuses writes with a clear, typed error rather than silently
no-op'ing something a caller might mistake for success.

get_billing_provider() is the one seam a future step wires a real
provider into -- entitlement_service.py and accounts_repository.py never
call this directly; only a future billing-sync job would.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BillingNotConfiguredError(RuntimeError):
    """Raised by NullBillingProvider for any operation that requires a
    real billing provider. Distinct from a generic RuntimeError so
    callers (and tests) can tell "billing isn't enabled yet" apart from
    an actual failure of a configured provider."""


class BillingProvider(ABC):
    @abstractmethod
    def get_subscription_status(self, *, user_id: int) -> Optional[dict[str, Any]]:
        """Returns the provider's current view of this user's
        subscription, or None if it has none. Must not raise for "no
        subscription" -- only for a genuine provider-side failure."""

    @abstractmethod
    def sync_subscription(self, *, user_id: int) -> None:
        """Pulls the provider's current subscription state for this user
        and writes it via accounts_repository.upsert_subscription."""

    @abstractmethod
    def create_checkout_session(self, *, user_id: int, plan_key: str) -> str:
        """Returns a URL the user is redirected to to purchase `plan_key`."""

    @abstractmethod
    def create_customer_portal_session(self, *, user_id: int) -> str:
        """Returns a URL the user is redirected to to manage their
        existing subscription (cancel, update payment method, ...)."""

    @abstractmethod
    def handle_webhook_event(self, *, payload: bytes, signature: str) -> None:
        """Verifies and processes one inbound billing-provider webhook
        event (e.g. subscription.updated)."""


class NullBillingProvider(BillingProvider):
    """The only BillingProvider wired up today. Billing is decoupled and
    optional per Step 11's scope -- this makes that explicit rather than
    leaving the seam unimplemented."""

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

    def handle_webhook_event(self, *, payload: bytes, signature: str) -> None:
        raise BillingNotConfiguredError(
            "Billing is not yet enabled for this deployment -- "
            "handle_webhook_event() has no real provider configured."
        )


def get_billing_provider() -> BillingProvider:
    return NullBillingProvider()
