import pytest

from src.services.billing_provider import (
    BillingNotConfiguredError,
    NullBillingProvider,
    get_billing_provider,
)


def test_get_billing_provider_returns_null_provider():
    assert isinstance(get_billing_provider(), NullBillingProvider)


def test_null_provider_get_subscription_status_returns_none():
    provider = NullBillingProvider()
    assert provider.get_subscription_status(user_id=1) is None


def test_null_provider_sync_subscription_is_a_safe_noop():
    provider = NullBillingProvider()
    assert provider.sync_subscription(user_id=1) is None


def test_null_provider_checkout_session_raises_typed_error():
    provider = NullBillingProvider()
    with pytest.raises(BillingNotConfiguredError):
        provider.create_checkout_session(user_id=1, plan_key="pro")


def test_null_provider_portal_session_raises_typed_error():
    provider = NullBillingProvider()
    with pytest.raises(BillingNotConfiguredError):
        provider.create_customer_portal_session(user_id=1)


def test_null_provider_webhook_raises_typed_error():
    provider = NullBillingProvider()
    with pytest.raises(BillingNotConfiguredError):
        provider.handle_webhook_event(payload=b"{}", signature="sig")
