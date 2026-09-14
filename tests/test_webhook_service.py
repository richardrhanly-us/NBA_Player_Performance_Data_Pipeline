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


def _clear_stripe_and_db_env(monkeypatch):
    for name in (
        "DATABASE_URL",
        "STRIPE_SECRET_KEY",
        "STRIPE_PUBLISHABLE_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRO_PRICE_ID",
        "STRIPE_SUCCESS_URL",
        "STRIPE_CANCEL_URL",
        "STRIPE_PORTAL_RETURN_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_ready_endpoint_returns_503_when_unconfigured(client, monkeypatch):
    _clear_stripe_and_db_env(monkeypatch)
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["process"] == "webhook_service"


def _thread_safe_prediction_db_conn():
    """FastAPI's TestClient runs the app in a separate worker thread, so
    a plain sqlite3 in-memory connection (check_same_thread=True by
    default, like the shared `db_conn` fixture) can't be reused here --
    build a dedicated, thread-safe one instead."""
    import sqlite3

    from src.services.schema_sqlite import create_sqlite_prediction_schema

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    create_sqlite_prediction_schema(conn)
    return conn


def test_ready_endpoint_returns_200_when_healthy(client, monkeypatch):
    _clear_stripe_and_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/?checkout=success")
    monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/?checkout=cancel")
    monkeypatch.setenv("STRIPE_PORTAL_RETURN_URL", "https://app.example.com/")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    conn = _thread_safe_prediction_db_conn()
    monkeypatch.setattr("src.services.db_connection.get_prediction_db_connection", lambda: conn)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True


def test_ready_endpoint_never_leaks_configured_secret_values(client, monkeypatch):
    _clear_stripe_and_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:supersecretpassword@host/db")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_reallysecretvalue")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/?checkout=success")
    monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/?checkout=cancel")
    monkeypatch.setenv("STRIPE_PORTAL_RETURN_URL", "https://app.example.com/")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_reallysecretvalue")
    conn = _thread_safe_prediction_db_conn()
    monkeypatch.setattr("src.services.db_connection.get_prediction_db_connection", lambda: conn)

    response = client.get("/ready")

    assert "supersecretpassword" not in response.text
    assert "sk_test_reallysecretvalue" not in response.text
    assert "whsec_reallysecretvalue" not in response.text
