"""
Step 11 integration tests: the auth/session layer (src/services/
auth_session.py) exercised end-to-end against st.session_state and a
real (in-memory SQLite) accounts schema, plus a static check that
apps/adminapp.py's pre-existing operational surface (tabs, functions)
is still present after the Step 11 changes -- this repo's existing test
suite never actually executes apps/*.py as a running Streamlit script
(see tests/test_step7_static_guard.py), so "still import/run" is
verified the same way the rest of this suite already verifies app-page
behavior: statically.

DevAuthProvider is used throughout instead of Supabase -- it needs no
network and is exactly what runs in this test environment (no
SUPABASE_URL/SUPABASE_ANON_KEY configured), which is itself the
"external provider secrets not available" dev-fallback path Step 11
was asked to support.
"""

from pathlib import Path

import pytest
import streamlit as st

from src.services import accounts_repository as repo
from src.services import auth_session


class _NonClosingConn:
    """Wraps a sqlite3 connection so auth_session's `finally: conn.close()`
    doesn't tear down the fixture's in-memory database between calls --
    lets one test exercise get_current_user() multiple times against the
    same accounts_db_conn fixture."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean_auth_session_state(monkeypatch):
    # Never talk to a real Supabase project from this test module.
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    st.session_state["_auth_session"] = None
    st.session_state.pop("_legacy_admin_ok", None)
    yield
    st.session_state["_auth_session"] = None
    st.session_state.pop("_legacy_admin_ok", None)


def _wire_database(monkeypatch, conn):
    monkeypatch.setenv("DATABASE_URL", "sqlite://in-memory-test")
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection",
        lambda: _NonClosingConn(conn),
    )


# ---------------------------------------------------------------------------
# anonymous / no database
# ---------------------------------------------------------------------------

def test_no_session_is_anonymous():
    current = auth_session.get_current_user()
    assert current.is_authenticated is False
    assert current.entitlements.tier is None
    assert current.entitlements.can_access_admin is False


def test_sign_in_without_database_degrades_to_transient_free(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    error = auth_session.sign_in("person@example.com", "anything")
    assert error is None

    current = auth_session.get_current_user()
    assert current.is_authenticated is True
    assert current.user.email == "person@example.com"
    assert current.entitlements.tier_label == "FREE"
    assert current.entitlements.can_access_admin is False


def test_sign_out_clears_session(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    auth_session.sign_in("person@example.com", "anything")
    assert auth_session.get_current_user().is_authenticated is True

    auth_session.sign_out()
    assert auth_session.get_current_user().is_authenticated is False
    assert st.session_state.get("_auth_session") is None


# ---------------------------------------------------------------------------
# with a database: real tier resolution
# ---------------------------------------------------------------------------

def test_new_authenticated_user_defaults_to_free(monkeypatch, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("free-user@example.com", "anything")

    current = auth_session.get_current_user()
    assert current.is_authenticated is True
    assert current.entitlements.tier_label == "FREE"
    assert current.entitlements.can_view_full_edge_board is False


def test_admin_override_grants_admin_access(monkeypatch, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("admin-user@example.com", "anything")
    auth_session.get_current_user()  # ensures the users row exists

    user_row = repo.get_user_by_auth_subject(
        accounts_db_conn, "dev", auth_session.get_current_user().user.auth_subject
    )
    repo.create_override(
        accounts_db_conn,
        user_id=user_row["id"],
        override_tier="ADMIN",
        reason="test",
        created_by="tester",
    )

    current = auth_session.get_current_user()
    assert current.entitlements.can_access_admin is True
    assert current.entitlements.can_manage_users is True


def test_disabled_user_is_forced_out(monkeypatch, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("disabled-user@example.com", "anything")
    user_row = repo.get_user_by_auth_subject(
        accounts_db_conn, "dev", auth_session.get_current_user().user.auth_subject
    )
    repo.set_user_active(accounts_db_conn, user_row["id"], False)

    current = auth_session.get_current_user()
    assert current.is_authenticated is False
    assert st.session_state.get("_auth_session") is None


def test_expired_override_does_not_grant_pro(monkeypatch, accounts_db_conn):
    from datetime import datetime, timedelta, timezone

    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("expired-override@example.com", "anything")
    user_row = repo.get_user_by_auth_subject(
        accounts_db_conn, "dev", auth_session.get_current_user().user.auth_subject
    )
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    repo.create_override(
        accounts_db_conn,
        user_id=user_row["id"],
        override_tier="PRO",
        reason="test",
        created_by="tester",
        expires_at=past,
    )

    current = auth_session.get_current_user()
    assert current.entitlements.tier_label == "FREE"


# ---------------------------------------------------------------------------
# require_admin / audit labels
# ---------------------------------------------------------------------------

def test_require_admin_denies_free_user(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    auth_session.sign_in("free-user@example.com", "anything")
    assert auth_session.require_admin() is None


def test_require_admin_allows_admin(monkeypatch, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("admin-user@example.com", "anything")
    user_row = repo.get_user_by_auth_subject(
        accounts_db_conn, "dev", auth_session.get_current_user().user.auth_subject
    )
    repo.create_override(
        accounts_db_conn,
        user_id=user_row["id"],
        override_tier="ADMIN",
        reason="test",
        created_by="tester",
    )
    current = auth_session.require_admin()
    assert current is not None
    assert current.entitlements.can_access_admin is True


def test_audit_source_label_for_authenticated_admin(monkeypatch, accounts_db_conn):
    _wire_database(monkeypatch, accounts_db_conn)
    auth_session.sign_in("admin-user@example.com", "anything")
    user_row = repo.get_user_by_auth_subject(
        accounts_db_conn, "dev", auth_session.get_current_user().user.auth_subject
    )
    repo.create_override(
        accounts_db_conn,
        user_id=user_row["id"],
        override_tier="ADMIN",
        reason="test",
        created_by="tester",
    )
    current = auth_session.get_current_user()
    assert auth_session.audit_source_label(current) == "authenticated_admin:admin-user@example.com"


def test_audit_source_label_for_anonymous():
    current = auth_session.get_current_user()
    assert auth_session.audit_source_label(current) == "unknown_admin"


# ---------------------------------------------------------------------------
# existing admin operational surface preserved (static check -- this repo
# never executes apps/*.py directly in tests; see module docstring)
# ---------------------------------------------------------------------------

PRE_EXISTING_ADMIN_TAB_LABELS = (
    "Overview",
    "Operations",
    "Logs",
    "Usage",
    "Data Review",
)

PRE_EXISTING_ADMIN_FUNCTIONS = (
    "write_admin_log",
    "get_admin_logs_df",
    "get_usage_logs_df",
    "load_strong_plays_df",
    "get_automation_health",
    "build_sheet1_debug_summary",
)


def test_admin_app_preserves_existing_tabs():
    source = Path("apps/adminapp.py").read_text(encoding="utf-8")
    for label in PRE_EXISTING_ADMIN_TAB_LABELS:
        assert f'"{label}"' in source, f"expected existing admin tab {label!r} to remain"


def test_admin_app_adds_users_and_access_tab():
    source = Path("apps/adminapp.py").read_text(encoding="utf-8")
    assert "Users & Access" in source


def test_admin_app_preserves_existing_functions():
    import ast

    tree = ast.parse(Path("apps/adminapp.py").read_text(encoding="utf-8"))
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for name in PRE_EXISTING_ADMIN_FUNCTIONS:
        assert name in defined, f"expected existing admin function {name} to remain defined"
