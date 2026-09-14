"""
Step 11: the ONLY module that touches st.session_state for
authentication. apps/publicapp.py and apps/adminapp.py call the
functions here (get_current_user, require_admin, render_account_widget,
authorize_admin_or_legacy_key) -- they never read/write auth-related
session_state keys themselves. See the Step 11 report's architectural-
guards section (tests/test_step11_static_guards.py) for what this
centralization is protecting against.

Identity resolution is deliberately NOT cached (no @st.cache_data) --
entitlement-critical reads must reflect an admin's revoke/disable action
on the next rerun, not up to a cache TTL later. The extra DB round trip
per page load is cheap relative to the odds-API/model calls this app
already makes on every render.

Session persistence caveat (Streamlit Community Cloud has no built-in
persistent login): the auth session lives only in st.session_state, so
it resets on a hard refresh/new tab, exactly like every other piece of
this app's session state (selected player, usage session_id, etc.).
Signing in again is a small form, not a lost account -- no data is lost,
only the "currently signed in" flag.
"""

from __future__ import annotations

import os

import streamlit as st

from src.domain.accounts import AccessTier, CurrentUser, User
from src.services import accounts_repository, entitlement_service
from src.services.auth_providers import (
    AuthError,
    DevAuthProvider,
    get_auth_provider,
    is_supabase_configured,
)

_SESSION_KEY = "_auth_session"  # {"auth_provider", "auth_subject", "email", "access_token"}
_LEGACY_ADMIN_KEY = "_legacy_admin_ok"

_ANONYMOUS = CurrentUser(user=None, entitlements=entitlement_service.for_anonymous())


def _get_db_connection():
    """Returns a prediction-history-style DB connection, or None if
    DATABASE_URL isn't configured/reachable. Auth/entitlements degrade
    the same way every other DB-backed section of this app already does
    -- never raises out of this module."""
    if not os.environ.get("DATABASE_URL"):
        return None
    try:
        from src.services.db_connection import get_prediction_db_connection

        return get_prediction_db_connection()
    except Exception:
        return None


def _to_domain_user(user_row: dict) -> User:
    return User(
        id=user_row["id"],
        email=user_row["email"],
        display_name=user_row.get("display_name"),
        auth_provider=user_row["auth_provider"],
        auth_subject=user_row["auth_subject"],
        is_active=bool(user_row["is_active"]),
        created_at=str(user_row.get("created_at")),
    )


def get_current_user() -> CurrentUser:
    """Never raises. Returns the anonymous CurrentUser whenever there is
    no session, the session's user is disabled, or the database is
    unreachable (fail closed to anonymous/FREE, never to PRO/ADMIN)."""
    session = st.session_state.get(_SESSION_KEY)
    if not session:
        return _ANONYMOUS

    conn = _get_db_connection()
    if conn is None:
        # No DB configured (dev/offline) -- an authenticated session
        # still exists, but there is nowhere to look up overrides/
        # subscriptions, so it degrades to a transient FREE account
        # rather than silently granting anything higher.
        return CurrentUser(
            user=User(
                id=-1,
                email=session["email"],
                display_name=None,
                auth_provider=session["auth_provider"],
                auth_subject=session["auth_subject"],
                is_active=True,
                created_at="",
            ),
            entitlements=entitlement_service.for_tier(AccessTier.FREE),
        )

    try:
        user_row = accounts_repository.get_or_create_user_by_auth_subject(
            conn,
            auth_provider=session["auth_provider"],
            auth_subject=session["auth_subject"],
            email=session["email"],
        )
        if not user_row.get("is_active", True):
            # Disabled account: force sign-out rather than downgrading in
            # place -- a disabled user should not appear "logged in".
            st.session_state[_SESSION_KEY] = None
            return _ANONYMOUS

        subscription = accounts_repository.get_latest_subscription(conn, user_row["id"])
        override = accounts_repository.get_active_override(conn, user_row["id"])
        tier = entitlement_service.compute_effective_tier(
            user=user_row, subscription=subscription, active_override=override
        )
        return CurrentUser(
            user=_to_domain_user(user_row), entitlements=entitlement_service.for_tier(tier)
        )
    except Exception:
        return _ANONYMOUS
    finally:
        conn.close()


def sign_in(email: str, password: str) -> str | None:
    """Attempts sign-in; on success stores the session and returns None.
    On failure returns a user-facing error message (never the raw
    provider exception, which could echo request details)."""
    try:
        result = get_auth_provider().sign_in(email, password)
    except AuthError as e:
        return str(e)
    st.session_state[_SESSION_KEY] = {
        "auth_provider": result.auth_provider,
        "auth_subject": result.auth_subject,
        "email": result.email,
        "access_token": result.access_token,
    }
    return None


def sign_up(email: str, password: str) -> str | None:
    try:
        result = get_auth_provider().sign_up(email, password)
    except AuthError as e:
        return str(e)
    st.session_state[_SESSION_KEY] = {
        "auth_provider": result.auth_provider,
        "auth_subject": result.auth_subject,
        "email": result.email,
        "access_token": result.access_token,
    }
    return None


def sign_out() -> None:
    session = st.session_state.get(_SESSION_KEY)
    if session:
        try:
            get_auth_provider().sign_out(session.get("access_token", ""))
        except Exception:
            pass
    st.session_state[_SESSION_KEY] = None


def render_account_widget(container=None) -> CurrentUser:
    """Renders a compact sign-in/account section (sidebar by default)
    and returns the resulting CurrentUser. This is the only UI entry
    point pages should use for login/logout -- keeps auth widgets out of
    page-body layout code."""
    target = container if container is not None else st.sidebar
    current = get_current_user()

    with target:
        st.markdown("#### Account")
        if current.is_authenticated:
            st.caption(f"Signed in as **{current.display_label}**")
            st.caption(f"Tier: **{current.entitlements.tier_label}**")
            if st.button("Sign out", key="account_sign_out"):
                sign_out()
                st.rerun()
        else:
            if not is_supabase_configured():
                st.caption("DEV MODE: any email/password signs in (not secure).")
            tab_in, tab_up = st.tabs(["Sign in", "Sign up"])
            with tab_in:
                email = st.text_input("Email", key="signin_email")
                password = st.text_input("Password", type="password", key="signin_password")
                if st.button("Sign in", key="signin_submit"):
                    error = sign_in(email, password)
                    if error:
                        st.error(error)
                    else:
                        st.rerun()
            with tab_up:
                email_up = st.text_input("Email", key="signup_email")
                password_up = st.text_input(
                    "Password", type="password", key="signup_password"
                )
                if st.button("Create account", key="signup_submit"):
                    error = sign_up(email_up, password_up)
                    if error:
                        st.error(error)
                    else:
                        st.rerun()
    return current


def require_admin() -> CurrentUser | None:
    """For pages that only ever run in an authenticated-admin context.
    Returns the CurrentUser if authorized, else renders an error and
    returns None -- callers should st.stop() when this returns None."""
    current = get_current_user()
    if current.entitlements.can_access_admin:
        return current
    st.error("Admin access required.")
    return None


def _get_legacy_admin_key() -> str | None:
    try:
        key = st.secrets["admin_key"]
    except Exception:
        key = os.environ.get("ADMIN_KEY")
    return key or None


def authorize_admin_or_legacy_key() -> CurrentUser | None:
    """
    apps/adminapp.py's single authorization gate. Preferred path: a real
    authenticated session whose entitlements include can_access_admin
    (granted via an ADMIN entitlement_overrides row -- see
    scripts/create_admin_user.py). Fallback path: the legacy shared
    admin_key (Step 11 decision: kept as a bootstrap/break-glass
    mechanism, see the Step 11 report) -- only offered at all when an
    admin_key is actually configured, and every use is tagged distinctly
    (auth_provider="legacy_key") so it is always distinguishable from a
    real admin in the Admin Logs.

    Does not trust query params, session_state booleans set elsewhere,
    or email alone -- the legacy key must match exactly, and a real
    session must carry entitlements.can_access_admin computed
    server-side by EntitlementService.
    """
    current = get_current_user()
    if current.entitlements.can_access_admin:
        return current

    legacy_key = _get_legacy_admin_key()

    if st.session_state.get(_LEGACY_ADMIN_KEY):
        return CurrentUser(
            user=User(
                id=-1,
                email="(legacy admin key)",
                display_name="Legacy bootstrap admin",
                auth_provider="legacy_key",
                auth_subject="legacy-bootstrap",
                is_active=True,
                created_at="",
            ),
            entitlements=entitlement_service.for_tier(AccessTier.ADMIN),
        )

    with st.expander("Admin Login", expanded=True):
        st.markdown("##### Sign in with an admin account")
        render_account_widget(container=st.container())
        current = get_current_user()
        if current.entitlements.can_access_admin:
            return current

        if legacy_key is not None:
            st.markdown("##### Or use the legacy bootstrap key")
            st.caption(
                "This is a temporary fallback for bootstrapping the first admin "
                "account. Prefer signing in with a real ADMIN account above once "
                "one exists."
            )
            key_input = st.text_input(
                "Enter admin key", type="password", key="admin_key_input"
            )
            if key_input and key_input == legacy_key:
                st.session_state[_LEGACY_ADMIN_KEY] = True
                st.rerun()
            elif key_input:
                st.error("Invalid admin key")

    return None


def audit_source_label(current: CurrentUser) -> str:
    """The `source` value admin pages should pass to write_admin_log()
    for any mutation -- keeps legacy-key actions distinguishable from
    real authenticated-admin actions in the audit trail forever."""
    if current.user is not None and current.user.auth_provider == "legacy_key":
        return "legacy_admin_key"
    if current.user is not None:
        return f"authenticated_admin:{current.user.email}"
    return "unknown_admin"
