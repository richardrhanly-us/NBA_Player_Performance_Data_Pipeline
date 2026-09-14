import pytest

from src.services.auth_providers import (
    AuthError,
    AuthNotConfiguredError,
    DevAuthProvider,
    EmailConfirmationRequiredError,
    SupabaseAuthProvider,
    compute_dev_auth_subject,
    get_auth_provider,
    is_supabase_configured,
)


def test_dev_provider_accepts_any_password():
    provider = DevAuthProvider()
    result = provider.sign_in("person@example.com", "literally anything")
    assert result.auth_provider == "dev"
    assert result.email == "person@example.com"
    assert result.auth_subject.startswith("dev:")
    assert result.expires_at is None


def test_dev_provider_rejects_empty_or_invalid_email():
    provider = DevAuthProvider()
    with pytest.raises(AuthError):
        provider.sign_in("", "x")
    with pytest.raises(AuthError):
        provider.sign_in("not-an-email", "x")


def test_dev_provider_subject_is_deterministic_and_case_insensitive():
    provider = DevAuthProvider()
    a = provider.sign_in("Person@Example.com", "x")
    b = provider.sign_in("person@example.com", "y")
    assert a.auth_subject == b.auth_subject
    assert compute_dev_auth_subject("person@example.com") == a.auth_subject


def test_dev_provider_different_emails_get_different_subjects():
    provider = DevAuthProvider()
    a = provider.sign_in("one@example.com", "x")
    b = provider.sign_in("two@example.com", "x")
    assert a.auth_subject != b.auth_subject


def test_dev_provider_refuses_password_reset():
    provider = DevAuthProvider()
    with pytest.raises(AuthError):
        provider.request_password_reset("person@example.com")


def test_is_supabase_configured_false_when_env_unset(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert is_supabase_configured() is False


def test_is_supabase_configured_true_when_both_set(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    assert is_supabase_configured() is True


def _clear_auth_env(monkeypatch):
    for name in (
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "DEV_AUTH_ENABLED",
        "DEV_AUTH_FORCE",
        "ENVIRONMENT",
        "APP_ENV",
    ):
        monkeypatch.delenv(name, raising=False)


def test_get_auth_provider_fails_closed_when_nothing_configured(monkeypatch):
    """Step 12: the Step 11 behavior of silently activating DevAuthProvider
    whenever Supabase isn't configured is exactly what this guards
    against -- with nothing configured at all, sign-in must be refused,
    not silently downgraded to email-only auth."""
    _clear_auth_env(monkeypatch)
    with pytest.raises(AuthNotConfiguredError):
        get_auth_provider()


def test_get_auth_provider_uses_dev_when_explicitly_enabled(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    provider = get_auth_provider()
    assert isinstance(provider, DevAuthProvider)


def test_get_auth_provider_refuses_dev_under_production_hint(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(AuthNotConfiguredError):
        get_auth_provider()


def test_get_auth_provider_allows_dev_under_production_hint_when_forced(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEV_AUTH_FORCE", "true")
    provider = get_auth_provider()
    assert isinstance(provider, DevAuthProvider)


def test_get_auth_provider_uses_supabase_when_configured(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    provider = get_auth_provider()
    assert isinstance(provider, SupabaseAuthProvider)


def test_get_auth_provider_prefers_supabase_even_if_dev_also_enabled(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    provider = get_auth_provider()
    assert isinstance(provider, SupabaseAuthProvider)


class _FakeResponse:
    def __init__(self, status_code, json_data=None, content=b"1"):
        self.status_code = status_code
        self._json_data = {} if json_data is None else json_data
        self.content = content

    def json(self):
        return self._json_data


def test_supabase_sign_in_success(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        assert "token?grant_type=password" in url
        return _FakeResponse(
            200,
            {
                "access_token": "tok123",
                "refresh_token": "refresh123",
                "expires_in": 3600,
                "user": {"id": "user-uuid-1", "email": json["email"]},
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    result = provider.sign_in("person@example.com", "hunter2")
    assert result.auth_subject == "user-uuid-1"
    assert result.access_token == "tok123"
    assert result.refresh_token == "refresh123"
    assert result.expires_at is not None
    assert result.auth_provider == "supabase"


def test_supabase_sign_in_rejects_bad_credentials(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(400, {"error": "invalid_grant"})

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AuthError):
        provider.sign_in("person@example.com", "wrong")


def test_supabase_sign_in_handles_server_error(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(503, {"error": "unavailable"})

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AuthError):
        provider.sign_in("person@example.com", "hunter2")


def test_supabase_sign_in_handles_malformed_response(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(200, {"user": {}})

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AuthError):
        provider.sign_in("person@example.com", "hunter2")


def test_supabase_sign_up_pending_email_confirmation(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        assert "/auth/v1/signup" in url
        return _FakeResponse(
            200,
            {"user": {"id": "user-uuid-2", "email": json["email"]}, "access_token": None},
        )

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(EmailConfirmationRequiredError):
        provider.sign_up("person@example.com", "hunter2")


def test_supabase_sign_up_with_immediate_session(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(
            200,
            {
                "access_token": "tok456",
                "refresh_token": "refresh456",
                "expires_in": 3600,
                "user": {"id": "user-uuid-3", "email": json["email"]},
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    result = provider.sign_up("person@example.com", "hunter2")
    assert result.access_token == "tok456"


def test_supabase_refresh_success(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        assert "grant_type=refresh_token" in url
        assert json == {"refresh_token": "old-refresh"}
        return _FakeResponse(
            200,
            {
                "access_token": "new-token",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "user": {"id": "user-uuid-1", "email": "person@example.com"},
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    result = provider.refresh("old-refresh")
    assert result.access_token == "new-token"
    assert result.refresh_token == "new-refresh"


def test_supabase_refresh_failure_raises_auth_error(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(401, {"error": "invalid_grant"})

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AuthError):
        provider.refresh("expired-or-revoked-refresh-token")


def test_supabase_get_user_success(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_get(url, headers, timeout):
        assert url.endswith("/auth/v1/user")
        assert headers["Authorization"] == "Bearer tok123"
        return _FakeResponse(200, {"id": "user-uuid-1", "email": "person@example.com"})

    monkeypatch.setattr("requests.get", fake_get)
    result = provider.get_user("tok123")
    assert result.auth_subject == "user-uuid-1"
    assert result.email == "person@example.com"


def test_supabase_get_user_rejects_invalid_token(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_get(url, headers, timeout):
        return _FakeResponse(401, {"error": "invalid token"})

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(AuthError):
        provider.get_user("revoked-token")


def test_supabase_request_password_reset_calls_recover_endpoint(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")
    called = {}

    def fake_post(url, json, headers, timeout):
        called["url"] = url
        called["json"] = json
        return _FakeResponse(200, {}, content=b"")

    monkeypatch.setattr("requests.post", fake_post)
    provider.request_password_reset("person@example.com")
    assert "/auth/v1/recover" in called["url"]
    assert called["json"] == {"email": "person@example.com"}
