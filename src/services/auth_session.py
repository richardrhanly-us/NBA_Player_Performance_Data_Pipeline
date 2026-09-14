"""
Step 11/12: the ONLY module that touches st.session_state for
authentication. apps/publicapp.py and apps/adminapp.py call the
functions here (get_current_user, require_admin, render_account_widget,
authorize_admin_or_legacy_key) -- they never read/write auth-related
session_state keys themselves, and never call src/services/auth_providers.py
directly. See tests/test_step11_static_guards.py and
tests/test_step12_static_guards.py for what this centralization protects
against.

Identity resolution is deliberately NOT cached (no @st.cache_data) --
entitlement-critical reads must reflect an admin's revoke/disable action
on the next rerun, not up to a cache TTL later. The extra DB round trip
per page load is cheap relative to the odds-API/model calls this app
already makes on every render.

Step 12 session model: a Supabase-backed session stores access_token,
refresh_token, and expires_at (unix epoch seconds) alongside identity.
On every get_current_user() call for a "supabase" session:

    1. if the access token is still within its validity window (with a
       30s leeway), and was revalidated against Supabase within the
       last _REVALIDATION_INTERVAL_SECONDS, trust it as-is;
    2. else if it's expired but a refresh_token is present, refresh it
       through Supabase and replace the whole session dict atomically;
    3. else re-validate the still-unexpired-but-stale token against
       Supabase's /auth/v1/user (catches revocation/password changes);
    4. if refresh or revalidation fails, the session is cleared
       entirely and the caller sees an anonymous CurrentUser plus a
       one-shot "session expired" notice on next render.

Dev ("dev") and legacy-key sessions have no real provider token to
validate -- dev sessions are additionally re-checked against
auth_config.can_use_dev_auth() on every call (a dev session dies the
moment dev auth is disabled, even mid-session); legacy-key admin status
lives entirely in its own session_state flag, gated by
auth_config.is_legacy_admin_key_enabled(), and never touches the
database (see authorize_admin_or_legacy_key()'s docstring).

Session persistence caveat (Streamlit Community Cloud has no built-in
persistent login): the auth session lives only in st.session_state, so
it resets on a hard refresh/new tab/browser restart, exactly like every
other piece of this app's session state (selected player, usage
session_id, etc.). Refresh-token handling above buys a *single browser
tab* a longer-lived session past Supabase's short access-token TTL
(typically ~1 hour) -- it does NOT provide persistence across a refresh,
since st.session_state itself is gone by then. See the Step 12 report's
Streamlit-persistence-limitations section for why a cookie-based
approach was deliberately not built here.
"""

from __future__ import annotations

import os
import time

import streamlit as st

from src.domain.accounts import AccessTier, CurrentUser, User
from src.services import accounts_repository, auth_config, entitlement_service
from src.services.auth_providers import (
    AuthError,
    AuthNotConfiguredError,
    AuthResult,
    EmailConfirmationRequiredError,
    SupabaseAuthProvider,
    get_auth_provider,
)
from src.services.observability import get_logger, log_event

_logger = get_logger("auth")

_SESSION_KEY = "_auth_session"
_LEGACY_ADMIN_KEY = "_legacy_admin_ok"
_SESSION_EXPIRED_FLAG = "_auth_session_expired"
_SIGNUP_NOTICE_KEY = "_auth_signup_notice"

# How long a revalidated-against-Supabase identity is trusted before the
# next get_current_user() call re-checks it via /auth/v1/user, even if
# the access token itself hasn't expired yet. Balances "detect a revoked
# session reasonably promptly" against "don't call Supabase on every
# single Streamlit rerun" (widget interactions rerun the whole script).
_REVALIDATION_INTERVAL_SECONDS = 60.0

# Treat a token as expired slightly before its real expiry to absorb
# clock skew and request latency.
_TOKEN_EXPIRY_LEEWAY_SECONDS = 30.0

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


def _session_dict_from_result(result: AuthResult) -> dict:
    return {
        "auth_provider": result.auth_provider,
        "auth_subject": result.auth_subject,
        "email": result.email,
        "access_token": result.access_token,
        "refresh_token": result.refresh_token,
        "expires_at": result.expires_at,
        "last_verified_at": time.time(),
    }


def _ensure_valid_supabase_session(session: dict) -> dict | None:
    """
    Returns an up-to-date session dict for a "supabase"-provider session
    (possibly refreshed), or None if the session must be dropped
    (expired with no usable refresh token, a failed refresh, or a failed
    revalidation). Never mutates `session` in place -- callers replace
    st.session_state[_SESSION_KEY] wholesale with the return value so a
    failed attempt never leaves a torn/partial session behind.
    """
    provider = get_auth_provider()
    if not isinstance(provider, SupabaseAuthProvider):
        # Configuration no longer resolves to Supabase (e.g. secrets
        # were removed) -- there's nowhere to validate this session
        # against, so it can no longer be trusted.
        return None

    now = time.time()
    expires_at = session.get("expires_at")

    if expires_at is not None and now >= (expires_at - _TOKEN_EXPIRY_LEEWAY_SECONDS):
        refresh_token = session.get("refresh_token")
        if not refresh_token:
            log_event(_logger, "auth.session_expired", severity="warning", reason="no_refresh_token")
            return None
        try:
            result = provider.refresh(refresh_token)
        except AuthError:
            log_event(_logger, "auth.session_expired", severity="warning", reason="refresh_failed")
            return None
        if result.auth_subject != session.get("auth_subject"):
            log_event(_logger, "auth.session_expired", severity="error", reason="refresh_identity_mismatch")
            return None  # never trust a mismatched identity
        log_event(_logger, "auth.session_refreshed", auth_subject=result.auth_subject)
        return _session_dict_from_result(result)

    last_verified_at = session.get("last_verified_at") or 0.0
    if now - last_verified_at > _REVALIDATION_INTERVAL_SECONDS:
        try:
            result = provider.get_user(session["access_token"])
        except AuthError:
            log_event(_logger, "auth.session_expired", severity="warning", reason="revalidation_failed")
            return None
        if result.auth_subject != session.get("auth_subject"):
            log_event(_logger, "auth.session_expired", severity="error", reason="revalidation_identity_mismatch")
            return None  # never trust a mismatched identity
        refreshed = dict(session)
        refreshed["email"] = result.email
        refreshed["last_verified_at"] = now
        return refreshed

    return session


def get_current_user() -> CurrentUser:
    """Never raises. Returns the anonymous CurrentUser whenever there is
    no session, the session's provider/config is no longer valid, the
    session's user is disabled, or the database is unreachable (fail
    closed to anonymous/FREE, never to PRO/ADMIN)."""
    session = st.session_state.get(_SESSION_KEY)
    if not session:
        return _ANONYMOUS

    if session["auth_provider"] == "dev" and not auth_config.can_use_dev_auth():
        # Dev auth was disabled after this session was created -- kill
        # it immediately rather than letting it linger.
        st.session_state[_SESSION_KEY] = None
        return _ANONYMOUS

    if session["auth_provider"] == "supabase":
        updated = _ensure_valid_supabase_session(session)
        if updated is None:
            st.session_state[_SESSION_KEY] = None
            st.session_state[_SESSION_EXPIRED_FLAG] = True
            return _ANONYMOUS
        if updated is not session:
            st.session_state[_SESSION_KEY] = updated
        session = updated

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
    provider exception, which could echo request details -- and never
    the password, which never leaves this call in any form)."""
    try:
        result = get_auth_provider().sign_in(email, password)
    except (AuthError, AuthNotConfiguredError) as e:
        log_event(_logger, "auth.sign_in_failed", severity="warning", reason=str(e))
        return str(e)
    st.session_state[_SESSION_KEY] = _session_dict_from_result(result)
    st.session_state[_SESSION_EXPIRED_FLAG] = False
    log_event(_logger, "auth.sign_in_succeeded", auth_provider=result.auth_provider)
    return None


def sign_up(email: str, password: str) -> str | None:
    """Returns None on a fully-signed-in success. On a "check your email
    to confirm" outcome, stores an informational (non-error) notice and
    also returns None -- render_account_widget() distinguishes the two
    by checking whether a session now exists."""
    try:
        result = get_auth_provider().sign_up(email, password)
    except EmailConfirmationRequiredError as e:
        st.session_state[_SIGNUP_NOTICE_KEY] = str(e)
        return None
    except (AuthError, AuthNotConfiguredError) as e:
        return str(e)
    st.session_state[_SESSION_KEY] = _session_dict_from_result(result)
    st.session_state[_SESSION_EXPIRED_FLAG] = False
    return None


def request_password_reset(email: str) -> str:
    """Always returns a generic, non-enumerating message -- never
    reveals whether the email address has an account."""
    generic_message = (
        "If an account exists for that email, a password reset link has been sent."
    )
    try:
        get_auth_provider().request_password_reset(email)
    except (AuthError, AuthNotConfiguredError):
        pass
    return generic_message


def sign_out() -> None:
    """Clears every piece of sensitive session state -- the provider
    session, the legacy-admin-key flag, and any pending notices. A
    legacy-key admin session is cleared by this too, even though it was
    never tied to the same session_state key set used for provider
    sign-in (see authorize_admin_or_legacy_key())."""
    session = st.session_state.get(_SESSION_KEY)
    if session:
        try:
            get_auth_provider().sign_out(session.get("access_token", ""))
        except Exception:
            pass
    st.session_state[_SESSION_KEY] = None
    st.session_state[_LEGACY_ADMIN_KEY] = False
    st.session_state[_SESSION_EXPIRED_FLAG] = False


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
            return current

        if st.session_state.pop(_SESSION_EXPIRED_FLAG, False):
            st.warning("Your session expired. Please sign in again.")

        notice = st.session_state.pop(_SIGNUP_NOTICE_KEY, None)
        if notice:
            st.info(notice)

        if auth_config.is_supabase_configured():
            pass
        elif auth_config.can_use_dev_auth():
            st.caption("DEV MODE: any email/password signs in (not secure).")
        else:
            st.info("Sign-in is not available for this deployment right now.")
            return current

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
            if auth_config.is_supabase_configured():
                with st.expander("Forgot password?"):
                    reset_email = st.text_input("Email", key="reset_password_email")
                    if st.button("Send reset link", key="reset_password_submit"):
                        st.info(request_password_reset(reset_email))
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
    """Only returns a usable key when LEGACY_ADMIN_KEY_ENABLED=true --
    the mere presence of an admin_key secret/env var is no longer
    sufficient (Step 12 hardening; see the Step 12 report's Phase 3)."""
    if not auth_config.is_legacy_admin_key_enabled():
        return None
    try:
        key = st.secrets["admin_key"]
    except Exception:
        key = auth_config.admin_key_from_env()
    return key or None


def authorize_admin_or_legacy_key() -> CurrentUser | None:
    """
    apps/adminapp.py's single authorization gate. Preferred path: a real
    authenticated session whose entitlements include can_access_admin
    (granted via an ADMIN entitlement_overrides row -- see
    scripts/create_admin_user.py). Fallback path: the legacy shared
    admin_key -- offered ONLY when LEGACY_ADMIN_KEY_ENABLED=true AND a
    key is actually configured (Step 12; see _get_legacy_admin_key()).
    Every use is tagged distinctly (auth_provider="legacy_key") so it is
    always distinguishable from a real admin in the Admin Logs, and it
    never creates or mutates any user's permanent ADMIN entitlement --
    the CurrentUser it produces is synthetic and touches no database
    row.

    Does not trust query params, session_state booleans set elsewhere,
    or email alone -- the legacy key must match exactly, and a real
    session must carry entitlements.can_access_admin computed
    server-side by EntitlementService (which itself re-validates the
    Supabase session and rejects disabled accounts on every call -- see
    get_current_user()).
    """
    current = get_current_user()
    if current.entitlements.can_access_admin:
        return current

    legacy_key = _get_legacy_admin_key()

    if st.session_state.get(_LEGACY_ADMIN_KEY) and auth_config.is_legacy_admin_key_enabled():
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
                "one exists, then disable LEGACY_ADMIN_KEY_ENABLED."
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
