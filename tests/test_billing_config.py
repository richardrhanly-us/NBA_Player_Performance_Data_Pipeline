from src.services import billing_config


def _clear(monkeypatch):
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


def _set_all(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_123")
    monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/?checkout=success")
    monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/?checkout=cancel")
    monkeypatch.setenv("STRIPE_PORTAL_RETURN_URL", "https://app.example.com/")


def test_not_configured_by_default(monkeypatch):
    _clear(monkeypatch)
    assert billing_config.is_stripe_configured() is False


def test_configured_requires_all_fields(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123")
    assert billing_config.is_stripe_configured() is False
    _set_all(monkeypatch)
    assert billing_config.is_stripe_configured() is True


def test_missing_any_single_field_fails_closed(monkeypatch):
    _clear(monkeypatch)
    _set_all(monkeypatch)
    monkeypatch.delenv("STRIPE_PORTAL_RETURN_URL", raising=False)
    assert billing_config.is_stripe_configured() is False


def test_webhook_configured_independent_of_checkout_config(monkeypatch):
    _clear(monkeypatch)
    _set_all(monkeypatch)
    assert billing_config.is_webhook_configured() is False
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_123")
    assert billing_config.is_webhook_configured() is True


def test_getters_return_none_when_unset(monkeypatch):
    _clear(monkeypatch)
    assert billing_config.stripe_secret_key() is None
    assert billing_config.stripe_publishable_key() is None
    assert billing_config.stripe_webhook_secret() is None
    assert billing_config.stripe_pro_price_id() is None
    assert billing_config.stripe_success_url() is None
    assert billing_config.stripe_cancel_url() is None
    assert billing_config.stripe_portal_return_url() is None
