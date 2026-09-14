"""
Step 14 Phase 4: structured logging emitted by auth_session.py never
contains tokens/passwords, and sign-in failures/session-expiry events
are actually observable via the log stream. billing_provider.py's
webhook logging is covered separately in
tests/test_stripe_billing_provider.py; this file covers the auth side.
"""

import time

import pytest
import streamlit as st

from src.services import auth_session
from src.services.auth_providers import AuthError, AuthResult, SupabaseAuthProvider


class _FakeSupabaseAuthProvider(SupabaseAuthProvider):
    def __init__(self):
        super().__init__("https://example.supabase.co", "anon-key")
        self.refresh_error = None

    def sign_in(self, email, password):
        return AuthResult(
            auth_provider="supabase",
            auth_subject="supabase-user-1",
            email=email,
            access_token="initial-token",
            refresh_token="initial-refresh",
            expires_at=time.time() + 3600,
        )

    def sign_out(self, access_token):
        return None

    def refresh(self, refresh_token):
        if self.refresh_error is not None:
            raise self.refresh_error
        raise AuthError("refresh not stubbed for this test")


@pytest.fixture(autouse=True)
def _clean_session_state():
    st.session_state["_auth_session"] = None
    st.session_state.pop("_auth_session_expired", None)
    yield
    st.session_state["_auth_session"] = None
    st.session_state.pop("_auth_session_expired", None)


def test_sign_in_failure_is_logged_without_password(monkeypatch, caplog):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("DEV_AUTH_ENABLED", raising=False)

    secret_password = "hunter2-super-secret"
    with caplog.at_level("WARNING", logger="nba_pipeline.auth"):
        auth_session.sign_in("person@example.com", secret_password)

    failure_records = [r for r in caplog.records if "auth.sign_in_failed" in r.message]
    assert len(failure_records) == 1
    assert secret_password not in failure_records[0].message


def test_sign_in_success_is_logged_without_token(monkeypatch, caplog):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")

    with caplog.at_level("INFO", logger="nba_pipeline.auth"):
        auth_session.sign_in("person@example.com", "anything")

    success_records = [r for r in caplog.records if "auth.sign_in_succeeded" in r.message]
    assert len(success_records) == 1
    assert "dev-session:" not in success_records[0].message


def test_session_expiry_from_failed_refresh_is_logged(monkeypatch, caplog):
    provider = _FakeSupabaseAuthProvider()
    provider.refresh_error = AuthError("refresh token expired")
    monkeypatch.setattr(auth_session, "get_auth_provider", lambda: provider)

    auth_session.sign_in("person@example.com", "anything")
    session = st.session_state["_auth_session"]
    session["expires_at"] = time.time() - 10
    st.session_state["_auth_session"] = session

    with caplog.at_level("WARNING", logger="nba_pipeline.auth"):
        auth_session.get_current_user()

    expiry_records = [r for r in caplog.records if "auth.session_expired" in r.message]
    assert len(expiry_records) == 1
    assert "initial-token" not in expiry_records[0].message
    assert "initial-refresh" not in expiry_records[0].message
