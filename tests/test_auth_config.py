from src.services import auth_config


def _clear(monkeypatch):
    for name in (
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "DEV_AUTH_ENABLED",
        "DEV_AUTH_FORCE",
        "ENVIRONMENT",
        "APP_ENV",
        "LEGACY_ADMIN_KEY_ENABLED",
        "ADMIN_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_supabase_not_configured_by_default(monkeypatch):
    _clear(monkeypatch)
    assert auth_config.is_supabase_configured() is False
    assert auth_config.supabase_url() is None
    assert auth_config.supabase_anon_key() is None


def test_supabase_configured_requires_both_values(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    assert auth_config.is_supabase_configured() is False
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    assert auth_config.is_supabase_configured() is True


def test_dev_auth_disabled_by_default(monkeypatch):
    _clear(monkeypatch)
    assert auth_config.is_dev_auth_enabled() is False
    assert auth_config.can_use_dev_auth() is False


def test_dev_auth_enabled_explicitly(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    assert auth_config.can_use_dev_auth() is True


def test_dev_auth_various_truthy_spellings(monkeypatch):
    _clear(monkeypatch)
    for value in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("DEV_AUTH_ENABLED", value)
        assert auth_config.is_dev_auth_enabled() is True


def test_production_hint_blocks_dev_auth(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert auth_config.is_production_hint_set() is True
    assert auth_config.can_use_dev_auth() is False


def test_production_hint_via_app_env(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")
    assert auth_config.is_production_hint_set() is True


def test_dev_auth_force_overrides_production_hint(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEV_AUTH_FORCE", "true")
    assert auth_config.can_use_dev_auth() is True


def test_non_production_environment_value_does_not_block_dev_auth(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    assert auth_config.can_use_dev_auth() is True


def test_legacy_admin_key_disabled_by_default(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ADMIN_KEY", "some-secret-key")
    assert auth_config.is_legacy_admin_key_enabled() is False


def test_legacy_admin_key_enabled_explicitly(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LEGACY_ADMIN_KEY_ENABLED", "true")
    assert auth_config.is_legacy_admin_key_enabled() is True


def test_admin_key_from_env(monkeypatch):
    _clear(monkeypatch)
    assert auth_config.admin_key_from_env() is None
    monkeypatch.setenv("ADMIN_KEY", "some-secret-key")
    assert auth_config.admin_key_from_env() == "some-secret-key"
