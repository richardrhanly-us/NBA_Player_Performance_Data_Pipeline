import pytest

from src.services.auth_providers import (
    AuthError,
    DevAuthProvider,
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


def test_is_supabase_configured_false_when_env_unset(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert is_supabase_configured() is False


def test_is_supabase_configured_true_when_both_set(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    assert is_supabase_configured() is True


def test_get_auth_provider_falls_back_to_dev_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    provider = get_auth_provider()
    assert isinstance(provider, DevAuthProvider)


def test_get_auth_provider_uses_supabase_when_configured(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    provider = get_auth_provider()
    assert isinstance(provider, SupabaseAuthProvider)


class _FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

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
                "user": {"id": "user-uuid-1", "email": json["email"]},
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    result = provider.sign_in("person@example.com", "hunter2")
    assert result.auth_subject == "user-uuid-1"
    assert result.access_token == "tok123"
    assert result.auth_provider == "supabase"


def test_supabase_sign_in_rejects_bad_credentials(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(400, {"error": "invalid_grant"})

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AuthError):
        provider.sign_in("person@example.com", "wrong")


def test_supabase_sign_in_handles_malformed_response(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(200, {"user": {}})

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AuthError):
        provider.sign_in("person@example.com", "hunter2")
