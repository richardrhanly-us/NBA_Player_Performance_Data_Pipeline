"""
Step 13: StripeBillingProvider tests. The real `stripe` package is
imported (no network calls happen -- every Stripe API method used is
monkeypatched), so these exercise the exact code paths a real
integration would run.
"""

import time
import types

import pytest

from src.domain.accounts import AccessTier
from src.services import accounts_repository as repo
from src.services.billing_provider import (
    BillingActionDeniedError,
    BillingNotConfiguredError,
    NullBillingProvider,
    StripeBillingProvider,
    WebhookVerificationError,
    get_billing_provider,
    normalize_stripe_subscription_status,
)
from src.services import entitlement_service


def _clear_stripe_env(monkeypatch):
    for name in (
        "STRIPE_SECRET_KEY",
        "STRIPE_PUBLISHABLE_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRO_PRICE_ID",
        "STRIPE_SUCCESS_URL",
        "STRIPE_CANCEL_URL",
        "STRIPE_PORTAL_RETURN_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_get_billing_provider_returns_null_when_unconfigured(monkeypatch):
    _clear_stripe_env(monkeypatch)
    assert isinstance(get_billing_provider(), NullBillingProvider)


def test_get_billing_provider_returns_stripe_when_configured(monkeypatch):
    _clear_stripe_env(monkeypatch)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/?checkout=success")
    monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/?checkout=cancel")
    monkeypatch.setenv("STRIPE_PORTAL_RETURN_URL", "https://app.example.com/")
    assert isinstance(get_billing_provider(), StripeBillingProvider)


def test_missing_billing_config_does_not_affect_authentication(monkeypatch, accounts_db_conn):
    """Phase 16: billing failures/absence must only ever degrade billing
    functionality -- authentication must remain fully usable."""
    _clear_stripe_env(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite://in-memory-test")
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection",
        lambda: _NonClosingConn(accounts_db_conn),
    )

    from src.services import auth_session

    error = auth_session.sign_in("person@example.com", "anything")
    assert error is None
    current = auth_session.get_current_user()
    assert current.is_authenticated is True
    assert isinstance(get_billing_provider(), NullBillingProvider)


class _NonClosingConn:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


@pytest.fixture
def provider(monkeypatch, accounts_db_conn):
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection",
        lambda: _NonClosingConn(accounts_db_conn),
    )
    return StripeBillingProvider(
        secret_key="sk_test_x",
        pro_price_id="price_pro",
        success_url="https://app.example.com/?checkout=success",
        cancel_url="https://app.example.com/?checkout=cancel",
        portal_return_url="https://app.example.com/",
        webhook_secret="whsec_test",
    )


@pytest.fixture
def free_user(accounts_db_conn):
    return repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="free@example.com"
    )


# ---------------------------------------------------------------------------
# status normalization policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stripe_status,expected_local_status",
    [
        ("active", "active"),
        ("trialing", "trialing"),
        ("past_due", "past_due"),
        ("unpaid", "past_due"),
        ("canceled", "canceled"),
        ("incomplete", "incomplete"),
        ("incomplete_expired", "canceled"),
        ("paused", "canceled"),
        (None, "none"),
        ("some_future_status_stripe_might_add", "canceled"),
    ],
)
def test_status_normalization_policy(stripe_status, expected_local_status):
    assert normalize_stripe_subscription_status(stripe_status) == expected_local_status


@pytest.mark.parametrize("granting_status", ["active", "trialing"])
def test_granting_statuses_actually_grant_pro_via_entitlement_service(granting_status):
    local_status = normalize_stripe_subscription_status(granting_status)
    tier = entitlement_service.compute_effective_tier(
        user={"is_active": True},
        subscription={"status": local_status, "plan_key": "pro", "current_period_end": None},
        active_override=None,
    )
    assert tier == AccessTier.PRO


@pytest.mark.parametrize(
    "non_granting_status", ["past_due", "unpaid", "canceled", "incomplete", "incomplete_expired", "paused"]
)
def test_non_granting_statuses_do_not_grant_pro(non_granting_status):
    local_status = normalize_stripe_subscription_status(non_granting_status)
    tier = entitlement_service.compute_effective_tier(
        user={"is_active": True},
        subscription={"status": local_status, "plan_key": "pro", "current_period_end": None},
        active_override=None,
    )
    assert tier == AccessTier.FREE


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------

def test_checkout_creates_customer_and_session(provider, accounts_db_conn, free_user, monkeypatch):
    monkeypatch.setattr(
        provider._stripe.Customer, "create", lambda **kw: types.SimpleNamespace(id="cus_new")
    )
    captured = {}

    def fake_session_create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(url="https://checkout.stripe.com/session_abc")

    monkeypatch.setattr(provider._stripe.checkout.Session, "create", fake_session_create)

    url = provider.create_checkout_session(user_id=free_user["id"], plan_key="pro")

    assert url == "https://checkout.stripe.com/session_abc"
    assert captured["line_items"] == [{"price": "price_pro", "quantity": 1}]
    assert captured["customer"] == "cus_new"
    assert captured["mode"] == "subscription"
    assert captured["metadata"]["local_user_id"] == str(free_user["id"])

    reloaded = repo.get_user_by_id(accounts_db_conn, free_user["id"])
    assert reloaded["stripe_customer_id"] == "cus_new"


def test_checkout_does_not_write_subscription_state(provider, accounts_db_conn, free_user, monkeypatch):
    """A successful checkout SESSION creation must never itself grant PRO
    -- only a webhook does that."""
    monkeypatch.setattr(
        provider._stripe.Customer, "create", lambda **kw: types.SimpleNamespace(id="cus_new")
    )
    monkeypatch.setattr(
        provider._stripe.checkout.Session,
        "create",
        lambda **kw: types.SimpleNamespace(url="https://checkout.stripe.com/x"),
    )
    provider.create_checkout_session(user_id=free_user["id"], plan_key="pro")
    assert repo.get_latest_subscription(accounts_db_conn, free_user["id"]) is None


def test_checkout_reuses_existing_customer_no_duplicate(provider, accounts_db_conn, free_user, monkeypatch):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_existing")
    customer_create_calls = []
    monkeypatch.setattr(
        provider._stripe.Customer,
        "create",
        lambda **kw: customer_create_calls.append(kw) or types.SimpleNamespace(id="cus_should_not_be_used"),
    )
    captured = {}
    monkeypatch.setattr(
        provider._stripe.checkout.Session,
        "create",
        lambda **kw: captured.update(kw) or types.SimpleNamespace(url="https://checkout.stripe.com/x"),
    )

    provider.create_checkout_session(user_id=free_user["id"], plan_key="pro")

    assert customer_create_calls == []
    assert captured["customer"] == "cus_existing"


def test_checkout_denied_for_disabled_user(provider, accounts_db_conn, free_user):
    repo.set_user_active(accounts_db_conn, free_user["id"], False)
    with pytest.raises(BillingActionDeniedError):
        provider.create_checkout_session(user_id=free_user["id"], plan_key="pro")


def test_checkout_denied_for_existing_pro_user(provider, accounts_db_conn, free_user):
    repo.create_override(
        accounts_db_conn,
        user_id=free_user["id"],
        override_tier="PRO",
        reason="test",
        created_by="tester",
    )
    with pytest.raises(BillingActionDeniedError):
        provider.create_checkout_session(user_id=free_user["id"], plan_key="pro")


def test_checkout_denied_for_admin_user(provider, accounts_db_conn, free_user):
    repo.create_override(
        accounts_db_conn,
        user_id=free_user["id"],
        override_tier="ADMIN",
        reason="test",
        created_by="tester",
    )
    with pytest.raises(BillingActionDeniedError):
        provider.create_checkout_session(user_id=free_user["id"], plan_key="pro")


def test_checkout_rejects_unknown_plan_key(provider, free_user):
    with pytest.raises(BillingActionDeniedError):
        provider.create_checkout_session(user_id=free_user["id"], plan_key="some-other-plan")


def test_checkout_denied_for_unknown_user(provider):
    with pytest.raises(BillingActionDeniedError):
        provider.create_checkout_session(user_id=999999, plan_key="pro")


# ---------------------------------------------------------------------------
# customer portal
# ---------------------------------------------------------------------------

def test_portal_denied_without_stripe_customer(provider, free_user):
    with pytest.raises(BillingActionDeniedError):
        provider.create_customer_portal_session(user_id=free_user["id"])


def test_portal_creates_session_for_own_customer(provider, accounts_db_conn, free_user, monkeypatch):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_existing")
    captured = {}
    monkeypatch.setattr(
        provider._stripe.billing_portal.Session,
        "create",
        lambda **kw: captured.update(kw) or types.SimpleNamespace(url="https://billing.stripe.com/p/x"),
    )

    url = provider.create_customer_portal_session(user_id=free_user["id"])

    assert url == "https://billing.stripe.com/p/x"
    assert captured["customer"] == "cus_existing"


def test_portal_denied_for_disabled_user(provider, accounts_db_conn, free_user):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_existing")
    repo.set_user_active(accounts_db_conn, free_user["id"], False)
    with pytest.raises(BillingActionDeniedError):
        provider.create_customer_portal_session(user_id=free_user["id"])


# ---------------------------------------------------------------------------
# webhooks
# ---------------------------------------------------------------------------

def _stub_construct_event(provider, monkeypatch, event: dict):
    monkeypatch.setattr(
        provider._stripe.Webhook, "construct_event", lambda payload, sig_header, secret: event
    )


def test_webhook_invalid_signature_is_rejected(provider, monkeypatch):
    def raise_sig_error(payload, sig_header, secret):
        raise provider._stripe.error.SignatureVerificationError("bad signature", sig_header)

    monkeypatch.setattr(provider._stripe.Webhook, "construct_event", raise_sig_error)
    with pytest.raises(WebhookVerificationError):
        provider.handle_webhook_event(payload=b"{}", signature="bad-sig")


def test_webhook_invalid_payload_is_rejected(provider, monkeypatch):
    def raise_value_error(payload, sig_header, secret):
        raise ValueError("malformed payload")

    monkeypatch.setattr(provider._stripe.Webhook, "construct_event", raise_value_error)
    with pytest.raises(WebhookVerificationError):
        provider.handle_webhook_event(payload=b"not json", signature="whatever")


def test_webhook_without_configured_secret_raises_not_configured(accounts_db_conn, monkeypatch):
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection",
        lambda: _NonClosingConn(accounts_db_conn),
    )
    provider = StripeBillingProvider(
        secret_key="sk_test_x",
        pro_price_id="price_pro",
        success_url="https://app.example.com/?checkout=success",
        cancel_url="https://app.example.com/?checkout=cancel",
        portal_return_url="https://app.example.com/",
        webhook_secret=None,
    )
    with pytest.raises(BillingNotConfiguredError):
        provider.handle_webhook_event(payload=b"{}", signature="sig")


def test_webhook_checkout_completed_syncs_subscription_to_pro(
    provider, accounts_db_conn, free_user, monkeypatch
):
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_checkout_1",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_1",
                    "subscription": "sub_abc",
                    "customer": "cus_123",
                    "metadata": {"local_user_id": str(free_user["id"])},
                }
            },
        },
    )
    monkeypatch.setattr(
        provider._stripe.Subscription,
        "retrieve",
        lambda sub_id: {
            "id": "sub_abc",
            "customer": "cus_123",
            "status": "active",
            "current_period_start": time.time() - 3600,
            "current_period_end": time.time() + 30 * 86400,
            "cancel_at_period_end": False,
            "items": {"data": [{"price": {"id": "price_pro"}}]},
            "metadata": {"local_user_id": str(free_user["id"])},
        },
    )

    result = provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    assert result["status"] == "processed"
    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    assert sub["status"] == "active"
    assert sub["provider_subscription_id"] == "sub_abc"
    assert sub["stripe_price_id"] == "price_pro"

    tier = entitlement_service.compute_effective_tier(
        user=repo.get_user_by_id(accounts_db_conn, free_user["id"]),
        subscription=sub,
        active_override=None,
    )
    assert tier == AccessTier.PRO


def test_webhook_resolves_user_via_customer_id_when_metadata_missing(
    provider, accounts_db_conn, free_user, monkeypatch
):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_123")
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_sub_created_1",
            "type": "customer.subscription.created",
            "data": {
                "object": {
                    "id": "sub_abc",
                    "customer": "cus_123",
                    "status": "active",
                    "current_period_start": None,
                    "current_period_end": None,
                    "cancel_at_period_end": False,
                    "items": {"data": []},
                    "metadata": {},
                }
            },
        },
    )

    result = provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    assert result["status"] == "processed"
    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    assert sub["status"] == "active"


def test_webhook_unresolvable_user_is_marked_failed_not_processed(provider, accounts_db_conn, monkeypatch):
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_unresolvable",
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_orphan",
                    "customer": "cus_does_not_exist",
                    "status": "active",
                    "current_period_start": None,
                    "current_period_end": None,
                    "cancel_at_period_end": False,
                    "items": {"data": []},
                    "metadata": {},
                }
            },
        },
    )

    result = provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    assert result["status"] == "failed"
    event_row = repo.get_stripe_event(accounts_db_conn, "evt_unresolvable")
    assert event_row["processing_status"] == "failed"
    assert "resolve" in event_row["error_message"].lower()


def test_webhook_subscription_deleted_downgrades_to_free(provider, accounts_db_conn, free_user, monkeypatch):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_123")
    repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=free_user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_abc",
        plan_key="pro",
        status="active",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_sub_deleted",
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_abc",
                    "customer": "cus_123",
                    "status": "canceled",
                    "current_period_start": None,
                    "current_period_end": None,
                    "cancel_at_period_end": False,
                    "items": {"data": []},
                    "metadata": {},
                }
            },
        },
    )

    provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    assert sub["status"] == "canceled"
    tier = entitlement_service.compute_effective_tier(
        user=repo.get_user_by_id(accounts_db_conn, free_user["id"]), subscription=sub, active_override=None
    )
    assert tier == AccessTier.FREE


def test_webhook_invoice_payment_failed_resyncs_subscription_status(
    provider, accounts_db_conn, free_user, monkeypatch
):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_123")
    repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=free_user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_abc",
        plan_key="pro",
        status="active",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_invoice_failed",
            "type": "invoice.payment_failed",
            "data": {"object": {"id": "in_1", "customer": "cus_123", "subscription": "sub_abc"}},
        },
    )
    monkeypatch.setattr(
        provider._stripe.Subscription,
        "retrieve",
        lambda sub_id: {
            "id": "sub_abc",
            "customer": "cus_123",
            "status": "past_due",
            "current_period_start": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
            "items": {"data": []},
            "metadata": {},
        },
    )

    provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    assert sub["status"] == "past_due"


def test_webhook_invoice_payment_succeeded_resyncs_subscription(
    provider, accounts_db_conn, free_user, monkeypatch
):
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_123")
    repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=free_user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_abc",
        plan_key="pro",
        status="past_due",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_invoice_succeeded",
            "type": "invoice.payment_succeeded",
            "data": {"object": {"id": "in_2", "customer": "cus_123", "subscription": "sub_abc"}},
        },
    )
    monkeypatch.setattr(
        provider._stripe.Subscription,
        "retrieve",
        lambda sub_id: {
            "id": "sub_abc",
            "customer": "cus_123",
            "status": "active",
            "current_period_start": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
            "items": {"data": []},
            "metadata": {},
        },
    )

    provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    assert sub["status"] == "active"


def test_duplicate_webhook_delivery_is_idempotent(provider, accounts_db_conn, free_user, monkeypatch):
    event = {
        "id": "evt_dup_1",
        "type": "customer.subscription.created",
        "data": {
            "object": {
                "id": "sub_abc",
                "customer": "cus_123",
                "status": "active",
                "current_period_start": None,
                "current_period_end": None,
                "cancel_at_period_end": False,
                "items": {"data": []},
                "metadata": {"local_user_id": str(free_user["id"])},
            }
        },
    }
    _stub_construct_event(provider, monkeypatch, event)

    first = provider.handle_webhook_event(payload=b"{}", signature="good-sig")
    second = provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    assert first["status"] == "processed"
    assert second["status"] == "duplicate"

    # Only one subscription row despite two deliveries.
    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    assert sub is not None
    assert sub["provider_subscription_id"] == "sub_abc"


def test_unhandled_event_type_is_processed_as_a_noop(provider, accounts_db_conn, monkeypatch):
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_unhandled",
            "type": "customer.updated",
            "data": {"object": {"id": "cus_123"}},
        },
    )
    result = provider.handle_webhook_event(payload=b"{}", signature="good-sig")
    assert result["status"] == "processed"


def test_admin_override_survives_subscription_cancellation(provider, accounts_db_conn, free_user, monkeypatch):
    """Phase 10 priority: ADMIN, once granted via override, must remain
    ADMIN regardless of what Stripe reports for the underlying
    subscription."""
    repo.create_override(
        accounts_db_conn,
        user_id=free_user["id"],
        override_tier="ADMIN",
        reason="test",
        created_by="tester",
    )
    repo.set_user_stripe_customer_id(accounts_db_conn, free_user["id"], "cus_123")
    _stub_construct_event(
        provider,
        monkeypatch,
        {
            "id": "evt_admin_unaffected",
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_abc",
                    "customer": "cus_123",
                    "status": "canceled",
                    "current_period_start": None,
                    "current_period_end": None,
                    "cancel_at_period_end": False,
                    "items": {"data": []},
                    "metadata": {},
                }
            },
        },
    )

    provider.handle_webhook_event(payload=b"{}", signature="good-sig")

    user_row = repo.get_user_by_id(accounts_db_conn, free_user["id"])
    sub = repo.get_latest_subscription(accounts_db_conn, free_user["id"])
    override = repo.get_active_override(accounts_db_conn, free_user["id"])
    tier = entitlement_service.compute_effective_tier(
        user=user_row, subscription=sub, active_override=override
    )
    assert tier == AccessTier.ADMIN
