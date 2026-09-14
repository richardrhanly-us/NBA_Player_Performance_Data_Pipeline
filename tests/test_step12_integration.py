"""
Step 12 integration tests: token expiration/refresh/revalidation, the
dev-auth/legacy-admin-key fail-closed gates, and admin-session-expiry
behavior in src/services/auth_session.py.

A `_FakeSupabaseAuthProvider` (a real SupabaseAuthProvider subclass, so
isinstance checks in auth_session.py still pass) stands in for the
network calls -- this test module never talks to a real Supabase
project, but exercises exactly the same code paths auth_session.py uses
for a real one.
"""

import time

import pytest
import streamlit as st

from src.services import accounts_repository as repo
from src.services import auth_config, auth_session
from src.services.auth_providers import AuthError, AuthResult, SupabaseAuthProvider


class _NonClosingConn:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


class _FakeSupabaseAuthProvider(SupabaseAuthProvider):
    """A SupabaseAuthProvider subclass with no network calls -- lets
    tests control sign_in/refresh/get_user outcomes directly."""

    def __init__(self):
        super().__init__("https://example.supabase.co", "anon-key")
        self.refresh_calls = []
        self.get_user_calls = []
        self.refresh_result = None
        self.refresh_error = None
        self.get_user_result = None
        self.get_user_error = None

    def sign_in(self, email, password):
        return AuthResult(
            auth_provider="supabase",
            auth_subject="supabase-user-1",
            email=email,
            access_token="initial-token",
            refresh_token="initial-refresh",
            expires_at=time.time() + 3600,
        )

    def sign_up(self, email, password):
        return self.sign_in(email, password)

    def sign_out(self, access_token):
        return None

    def refresh(self, refresh_token):
        self.refresh_calls.append(refresh_token)
        if self.refresh_error is not None:
            raise self.refresh_error
        return self.refresh_result

    def get_user(self, access_token):
        self.get_user_calls.append(access_token)
        if self.get_user_error is not None:
            raise self.get_user_error
        return self.get_user_result


@pytest.fixture
def fake_provider(monkeypatch):
    provider = _FakeSupabaseAuthProvider()
    monkeypatch.setattr(auth_session, "get_auth_provider", lambda: provider)
    return provider


@pytest.fixture(autouse=True)
def _clean_session_state():
    st.session_state["_auth_session"] = None
    st.session_state.pop("_legacy_admin_ok", None)
    st.session_state.pop("_auth_session_expired", None)
    st.session_state.pop("_auth_signup_notice", None)
    yield
    st.session_state["_auth_session"] = None
    st.session_state.pop("_legacy_admin_ok", None)
    st.session_state.pop("_auth_session_expired", None)
    st.session_state.pop("_auth_signup_notice", None)


def _wire_database(monkeypatch, conn):
    monkeypatch.setenv("DATABASE_URL", "sqlite://in-memory-test")
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection",
        lambda: _NonClosingConn(conn),
    )


# ---------------------------------------------------------------------------
# Fail-closed dev auth / legacy admin key gates (Phase 2 / Phase 3)
# ---------------------------------------------------------------------------

def test_sign_in_fails_closed_with_no_provider_configured(monkeypatch):
    for name in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "DEV_AUTH_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    error = auth_session.sign_in("person@example.com", "anything")
    assert error is not None
    assert "not configured" in error.lower()
    assert auth_session.get_current_user().is_authenticated is False


def test_legacy_admin_key_disabled_by_default_even_if_key_set(monkeypatch):
    monkeypatch.delenv("LEGACY_ADMIN_KEY_ENABLED", raising=False)
    monkeypatch.setenv("ADMIN_KEY", "super-secret-admin-key")
    assert auth_session._get_legacy_admin_key() is None


def test_legacy_admin_key_available_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("LEGACY_ADMIN_KEY_ENABLED", "true")
    monkeypatch.setenv("ADMIN_KEY", "super-secret-admin-key")
    assert auth_session._get_legacy_admin_key() == "super-secret-admin-key"


def test_established_legacy_session_dies_if_flag_disabled_later(monkeypatch):
    monkeypatch.setenv("LEGACY_ADMIN_KEY_ENABLED", "true")
    monkeypatch.setenv("ADMIN_KEY", "super-secret-admin-key")
    st.session_state["_legacy_admin_ok"] = True

    monkeypatch.setenv("LEGACY_ADMIN_KEY_ENABLED", "false")
    # authorize_admin_or_legacy_key() would fall through to rendering the
    # login expander in a real app; we only assert the fast-path early
    # return no longer fires once the flag is disabled.
    assert not (
        st.session_state.get("_legacy_admin_ok")
        and auth_config.is_legacy_admin_key_enabled()
    )


# ---------------------------------------------------------------------------
# Supabase session validity / refresh / revalidation (Phase 4 / 7)
# ---------------------------------------------------------------------------

def test_fresh_session_does_not_trigger_refresh_or_revalidation(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    current = auth_session.get_current_user()
    assert current.is_authenticated is True
    assert fake_provider.refresh_calls == []
    assert fake_provider.get_user_calls == []


def test_expired_token_with_valid_refresh_is_refreshed(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    # Force expiry.
    session = st.session_state["_auth_session"]
    session["expires_at"] = time.time() - 10
    st.session_state["_auth_session"] = session

    fake_provider.refresh_result = AuthResult(
        auth_provider="supabase",
        auth_subject="supabase-user-1",
        email="person@example.com",
        access_token="refreshed-token",
        refresh_token="refreshed-refresh",
        expires_at=time.time() + 3600,
    )

    current = auth_session.get_current_user()
    assert current.is_authenticated is True
    assert fake_provider.refresh_calls == ["initial-refresh"]
    assert st.session_state["_auth_session"]["access_token"] == "refreshed-token"


def test_expired_token_with_failed_refresh_clears_session(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    session = st.session_state["_auth_session"]
    session["expires_at"] = time.time() - 10
    st.session_state["_auth_session"] = session
    fake_provider.refresh_error = AuthError("refresh token expired")

    current = auth_session.get_current_user()
    assert current.is_authenticated is False
    assert st.session_state.get("_auth_session") is None
    assert st.session_state.get("_auth_session_expired") is True


def test_expired_token_with_no_refresh_token_clears_session(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    session = st.session_state["_auth_session"]
    session["expires_at"] = time.time() - 10
    session["refresh_token"] = None
    st.session_state["_auth_session"] = session

    current = auth_session.get_current_user()
    assert current.is_authenticated is False
    assert fake_provider.refresh_calls == []


def test_stale_but_unexpired_token_is_revalidated(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    session = st.session_state["_auth_session"]
    session["last_verified_at"] = time.time() - 1000  # older than the revalidation window
    st.session_state["_auth_session"] = session

    fake_provider.get_user_result = AuthResult(
        auth_provider="supabase",
        auth_subject="supabase-user-1",
        email="person@example.com",
        access_token="initial-token",
    )

    current = auth_session.get_current_user()
    assert current.is_authenticated is True
    assert fake_provider.get_user_calls == ["initial-token"]


def test_revoked_token_fails_revalidation_and_clears_session(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    session = st.session_state["_auth_session"]
    session["last_verified_at"] = time.time() - 1000
    st.session_state["_auth_session"] = session
    fake_provider.get_user_error = AuthError("token revoked")

    current = auth_session.get_current_user()
    assert current.is_authenticated is False
    assert st.session_state.get("_auth_session") is None
    assert st.session_state.get("_auth_session_expired") is True


def test_revalidation_identity_mismatch_clears_session(monkeypatch, fake_provider, accounts_db_conn):
    """Defense in depth: even if a provider bug returned a different
    subject, never silently swap identity mid-session."""
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")

    session = st.session_state["_auth_session"]
    session["last_verified_at"] = time.time() - 1000
    st.session_state["_auth_session"] = session
    fake_provider.get_user_result = AuthResult(
        auth_provider="supabase",
        auth_subject="a-completely-different-user",
        email="person@example.com",
        access_token="initial-token",
    )

    current = auth_session.get_current_user()
    assert current.is_authenticated is False


def test_recently_verified_token_is_not_revalidated_again(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")
    # last_verified_at was just set by sign_in -- within the window.
    auth_session.get_current_user()
    assert fake_provider.get_user_calls == []


# ---------------------------------------------------------------------------
# Disabled accounts / entitlement integrity under the new session model
# ---------------------------------------------------------------------------

def test_disabled_account_forces_sign_out_even_with_valid_supabase_token(
    monkeypatch, fake_provider, accounts_db_conn
):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")
    auth_session.get_current_user()  # ensures the users row exists
    user_row = repo.get_user_by_auth_subject(accounts_db_conn, "supabase", "supabase-user-1")
    repo.set_user_active(accounts_db_conn, user_row["id"], False)

    current = auth_session.get_current_user()
    assert current.is_authenticated is False
    assert st.session_state.get("_auth_session") is None


def test_email_does_not_grant_admin(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("admin@company.com", "anything")

    current = auth_session.get_current_user()
    assert current.entitlements.tier_label == "FREE"
    assert current.entitlements.can_access_admin is False


def test_pro_override_preserved_across_token_refresh(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("pro-user@example.com", "anything")
    auth_session.get_current_user()  # ensures the users row exists
    user_row = repo.get_user_by_auth_subject(accounts_db_conn, "supabase", "supabase-user-1")
    repo.create_override(
        accounts_db_conn,
        user_id=user_row["id"],
        override_tier="PRO",
        reason="test",
        created_by="tester",
    )

    session = st.session_state["_auth_session"]
    session["expires_at"] = time.time() - 10
    st.session_state["_auth_session"] = session
    fake_provider.refresh_result = AuthResult(
        auth_provider="supabase",
        auth_subject="supabase-user-1",
        email="pro-user@example.com",
        access_token="refreshed-token",
        refresh_token="refreshed-refresh",
        expires_at=time.time() + 3600,
    )

    current = auth_session.get_current_user()
    assert current.entitlements.tier_label == "PRO"


def test_admin_access_lost_when_session_expires_and_refresh_fails(
    monkeypatch, fake_provider, accounts_db_conn
):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("admin-user@example.com", "anything")
    auth_session.get_current_user()  # ensures the users row exists
    user_row = repo.get_user_by_auth_subject(accounts_db_conn, "supabase", "supabase-user-1")
    repo.create_override(
        accounts_db_conn,
        user_id=user_row["id"],
        override_tier="ADMIN",
        reason="test",
        created_by="tester",
    )
    assert auth_session.require_admin() is not None

    session = st.session_state["_auth_session"]
    session["expires_at"] = time.time() - 10
    st.session_state["_auth_session"] = session
    fake_provider.refresh_error = AuthError("refresh token expired")

    assert auth_session.require_admin() is None


# ---------------------------------------------------------------------------
# Logout / secret hygiene (Phase 5 / 10)
# ---------------------------------------------------------------------------

def test_logout_clears_all_sensitive_session_state(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("person@example.com", "anything")
    st.session_state["_legacy_admin_ok"] = True
    st.session_state["_auth_session_expired"] = True

    auth_session.sign_out()

    assert st.session_state.get("_auth_session") is None
    assert st.session_state.get("_legacy_admin_ok") is False
    assert st.session_state.get("_auth_session_expired") is False


def test_password_never_appears_in_session_state(monkeypatch, fake_provider, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    secret_password = "correct-horse-battery-staple-9f2a"
    auth_session.sign_in("person@example.com", secret_password)

    for value in st.session_state.values():
        assert secret_password not in repr(value)


def test_auth_error_message_never_echoes_provider_response_body(monkeypatch):
    provider = SupabaseAuthProvider("https://example.supabase.co", "anon-key")

    class _Resp:
        status_code = 400
        content = b"1"

        def json(self):
            return {"error": "invalid_grant", "secret_debug_value": "sk_live_should_not_leak"}

    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp())
    monkeypatch.setattr(auth_session, "get_auth_provider", lambda: provider)

    error = auth_session.sign_in("person@example.com", "wrong-password")
    assert error is not None
    assert "sk_live_should_not_leak" not in error
