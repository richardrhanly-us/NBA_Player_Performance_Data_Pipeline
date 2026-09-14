import pytest

from src.services import readiness


def _clear_all(monkeypatch):
    for name in (
        "DATABASE_URL",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "DEV_AUTH_ENABLED",
        "DEV_AUTH_FORCE",
        "ENVIRONMENT",
        "APP_ENV",
        "LEGACY_ADMIN_KEY_ENABLED",
        "ADMIN_KEY",
        "STRIPE_SECRET_KEY",
        "STRIPE_PUBLISHABLE_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRO_PRICE_ID",
        "STRIPE_SUCCESS_URL",
        "STRIPE_CANCEL_URL",
        "STRIPE_PORTAL_RETURN_URL",
        "ODDS_API_KEY",
        "PREDICTION_AUTOMATION_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def _configure_dev_auth(monkeypatch):
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")


def _configure_supabase(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")


def _configure_stripe(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/?checkout=success")
    monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/?checkout=cancel")
    monkeypatch.setenv("STRIPE_PORTAL_RETURN_URL", "https://app.example.com/")


# ---------------------------------------------------------------------------
# public app
# ---------------------------------------------------------------------------

def test_public_app_not_ready_with_nothing_configured(monkeypatch):
    _clear_all(monkeypatch)
    report = readiness.check_public_app_readiness()
    assert report.is_ready is False
    assert report.app_env == "development"


def test_public_app_ready_with_dev_auth_and_database(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_dev_auth(monkeypatch)
    report = readiness.check_public_app_readiness()
    assert report.is_ready is True


def test_public_app_ready_without_stripe_or_odds_configured(monkeypatch):
    """Billing/odds are optional -- their absence must not block readiness."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_supabase(monkeypatch)
    report = readiness.check_public_app_readiness()
    assert report.is_ready is True
    stripe_check = next(c for c in report.checks if c.name == "stripe_checkout_configured")
    assert stripe_check.status.value == "missing"
    assert stripe_check.required is False


def test_dev_auth_active_in_production_fails_readiness(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("DEV_AUTH_FORCE", "true")  # actually active despite prod hint
    report = readiness.check_public_app_readiness()
    assert report.is_ready is False
    check = next(c for c in report.checks if c.name == "dev_auth_not_in_production")
    assert check.status.value == "degraded"


def test_dev_auth_blocked_in_production_does_not_fail_readiness(monkeypatch):
    """can_use_dev_auth() already refuses dev auth here (no FORCE flag)
    -- readiness should reflect that it's genuinely inactive, but the
    app then has NO auth provider at all, so overall readiness still
    fails on auth_provider, just not on the dev-auth check itself."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    report = readiness.check_public_app_readiness()
    dev_check = next(c for c in report.checks if c.name == "dev_auth_not_in_production")
    assert dev_check.status.value == "ok"
    auth_check = next(c for c in report.checks if c.name == "auth_provider")
    assert auth_check.status.value == "missing"
    assert report.is_ready is False


# ---------------------------------------------------------------------------
# admin app
# ---------------------------------------------------------------------------

def test_admin_app_flags_legacy_key_in_production(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_supabase(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LEGACY_ADMIN_KEY_ENABLED", "true")
    report = readiness.check_admin_app_readiness()
    check = next(c for c in report.checks if c.name == "legacy_admin_key_disabled_or_bootstrap_only")
    assert check.status.value == "degraded"
    assert check.required is False
    # Informational only -- must not block readiness by itself.
    assert report.is_ready is True


def test_admin_app_legacy_key_note_ok_when_disabled(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_supabase(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    report = readiness.check_admin_app_readiness()
    check = next(c for c in report.checks if c.name == "legacy_admin_key_disabled_or_bootstrap_only")
    assert check.status.value == "ok"


# ---------------------------------------------------------------------------
# webhook service
# ---------------------------------------------------------------------------

def test_webhook_service_not_ready_without_stripe_config(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    report = readiness.check_webhook_service_readiness()
    assert report.is_ready is False


def test_webhook_service_ready_with_full_config(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_stripe(monkeypatch)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    report = readiness.check_webhook_service_readiness()
    assert report.is_ready is True


def test_webhook_service_shallow_check_never_connects_to_database(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_stripe(monkeypatch)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")

    def _explode():
        raise AssertionError("shallow readiness must not open a DB connection")

    monkeypatch.setattr("src.services.db_connection.get_prediction_db_connection", lambda: _explode())
    report = readiness.check_webhook_service_readiness(deep=False)
    assert report.is_ready is True


def test_webhook_service_deep_check_reports_database_unavailable(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_stripe(monkeypatch)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")

    def _raise():
        raise RuntimeError("connection refused to postgres://user:hunterpassword@host/db")

    monkeypatch.setattr("src.services.db_connection.get_prediction_db_connection", _raise)
    report = readiness.check_webhook_service_readiness(deep=True)
    assert report.is_ready is False
    db_check = next(c for c in report.checks if c.name == "database_connectivity")
    assert db_check.status.value == "degraded"
    assert "hunterpassword" not in db_check.detail
    assert "postgres://" not in db_check.detail


def test_webhook_service_deep_check_reports_schema_missing(monkeypatch, db_conn):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_stripe(monkeypatch)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")

    import sqlite3

    empty_conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection", lambda: empty_conn
    )
    report = readiness.check_webhook_service_readiness(deep=True)
    assert report.is_ready is False
    db_check = next(c for c in report.checks if c.name == "database_connectivity")
    assert db_check.status.value == "degraded"
    assert "migrat" in db_check.detail.lower()


def test_webhook_service_deep_check_ok_with_ready_schema(monkeypatch, db_conn):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    _configure_stripe(monkeypatch)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection", lambda: db_conn
    )
    report = readiness.check_webhook_service_readiness(deep=True)
    assert report.is_ready is True


# ---------------------------------------------------------------------------
# automation
# ---------------------------------------------------------------------------

def test_automation_not_ready_without_odds_api_key(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    report = readiness.check_automation_readiness()
    assert report.is_ready is False


def test_automation_ready_when_configured_even_if_kill_switch_off(monkeypatch):
    """The kill switch being off is a deliberate, safe default -- it
    must not fail readiness (it's informational)."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds-key")
    report = readiness.check_automation_readiness()
    assert report.is_ready is True
    enabled_check = next(c for c in report.checks if c.name == "automation_enabled")
    assert enabled_check.status.value == "degraded"
    assert enabled_check.required is False


# ---------------------------------------------------------------------------
# no secret leakage in the serialized report
# ---------------------------------------------------------------------------

def test_readiness_report_to_dict_never_contains_configured_secret_values(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:supersecretpassword@host/db")
    _configure_stripe(monkeypatch)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_supersecretvalue")
    report = readiness.check_webhook_service_readiness()
    serialized = str(report.to_dict())
    assert "supersecretpassword" not in serialized
    assert "whsec_supersecretvalue" not in serialized
    assert "sk_test_x" not in serialized
