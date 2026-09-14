"""
Step 13: HTTP-layer tests for the standalone webhook_service FastAPI
app. get_billing_provider() is monkeypatched to a small fake so these
never touch a real Stripe/DB connection -- they exist to verify the
HTTP status-code contract (400 for a bad signature, 503 for missing
config, 500 for an unexpected failure, 200 + JSON for success),
independent of StripeBillingProvider's own internals (see
tests/test_stripe_billing_provider.py for those).
"""

import pytest
from fastapi.testclient import TestClient

from webhook_service.main import app
import webhook_service.main as webhook_main
from src.services.billing_provider import (
    BillingNotConfiguredError,
    BillingProvider,
    WebhookVerificationError,
)


class _FakeProvider(BillingProvider):
    def __init__(self, *, result=None, error=None):
        self._result = result
        self._error = error

    def get_subscription_status(self, *, user_id):
        return None

    def sync_subscription(self, *, user_id):
        return None

    def create_checkout_session(self, *, user_id, plan_key):
        raise NotImplementedError

    def create_customer_portal_session(self, *, user_id):
        raise NotImplementedError

    def handle_webhook_event(self, *, payload, signature):
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_success_returns_200_and_result(client, monkeypatch):
    monkeypatch.setattr(
        webhook_main,
        "get_billing_provider",
        lambda: _FakeProvider(result={"status": "processed", "event_id": "evt_1"}),
    )
    response = client.post(
        "/stripe/webhook", content=b"{}", headers={"stripe-signature": "sig"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "processed", "event_id": "evt_1"}


def test_webhook_invalid_signature_returns_400(client, monkeypatch):
    monkeypatch.setattr(
        webhook_main,
        "get_billing_provider",
        lambda: _FakeProvider(error=WebhookVerificationError("bad signature")),
    )
    response = client.post(
        "/stripe/webhook", content=b"{}", headers={"stripe-signature": "bad"}
    )
    assert response.status_code == 400


def test_webhook_missing_config_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        webhook_main,
        "get_billing_provider",
        lambda: _FakeProvider(error=BillingNotConfiguredError("no webhook secret")),
    )
    response = client.post(
        "/stripe/webhook", content=b"{}", headers={"stripe-signature": "sig"}
    )
    assert response.status_code == 503


def test_webhook_unexpected_failure_returns_500_without_leaking_details(client, monkeypatch):
    monkeypatch.setattr(
        webhook_main,
        "get_billing_provider",
        lambda: _FakeProvider(error=RuntimeError("db connection string: postgres://secret")),
    )
    response = client.post(
        "/stripe/webhook", content=b"{}", headers={"stripe-signature": "sig"}
    )
    assert response.status_code == 500
    assert "secret" not in response.text
    assert "postgres://" not in response.text
