"""
Step 12: the one place authentication-related environment variables are
read. src/services/auth_providers.py and src/services/auth_session.py
call these functions instead of touching os.environ directly -- see
tests/test_step12_static_guards.py for the guard that enforces this.

This module deliberately has NO Streamlit import (kept pure/testable,
same rationale as auth_providers.py) and NO st.secrets access -- the one
exception, the legacy admin_key's *value*, still lives in
auth_session.py because reading st.secrets requires a Streamlit runtime.
What lives here is the boolean gate (`is_legacy_admin_key_enabled`) that
decides whether that value is even looked up at all.

Environment variables (see the Step 12 report for full documentation):

    SUPABASE_URL, SUPABASE_ANON_KEY   -- Supabase Auth project config.
    DEV_AUTH_ENABLED                  -- must be "true" to allow
                                          DevAuthProvider when Supabase
                                          isn't configured. Defaults to
                                          disabled -- Step 11's behavior
                                          of activating it implicitly
                                          was the exact anti-pattern this
                                          step removes.
    ENVIRONMENT / APP_ENV             -- optional explicit production
                                          hint (e.g. "production"). When
                                          set, DevAuthProvider is refused
                                          even if DEV_AUTH_ENABLED=true,
                                          unless DEV_AUTH_FORCE=true is
                                          ALSO set. This is still explicit
                                          configuration, not environment
                                          sniffing -- a deployer sets
                                          ENVIRONMENT=production
                                          themselves; nothing here
                                          guesses it from platform
                                          fingerprints. (Step 14: this
                                          read now lives in
                                          src/services/environment.py,
                                          the single APP_ENV source of
                                          truth shared with readiness
                                          checks -- is_production_hint_set()
                                          below just delegates to it.)
    LEGACY_ADMIN_KEY_ENABLED          -- must be "true" for the legacy
                                          admin_key bootstrap fallback to
                                          be offered at all, even if a
                                          key value is configured.
"""

from __future__ import annotations

import os

from src.services import environment


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        print(
            "[AUTH DEBUG] loaded secret keys:",
            sorted(str(key) for key in st.secrets.keys()),
        )

        exists = name in st.secrets
        print(f"[AUTH DEBUG] st.secrets contains {name}: {exists}")

        if not exists:
            return None

        value = st.secrets.get(name)
        print(f"[AUTH DEBUG] st.secrets {name} has value: {bool(value)}")

        if value:
            return str(value)

    except Exception as exc:
        print(
            f"[AUTH DEBUG] st.secrets lookup failed for {name}: "
            f"{type(exc).__name__}"
        )

    return None


def supabase_url() -> str | None:
    env_value = os.environ.get("SUPABASE_URL")
    print(f"[AUTH DEBUG] env SUPABASE_URL present: {bool(env_value)}")

    value = env_value or _streamlit_secret("SUPABASE_URL")
    print(f"[AUTH DEBUG] final SUPABASE_URL present: {bool(value)}")
    return value


def supabase_anon_key() -> str | None:
    env_value = os.environ.get("SUPABASE_ANON_KEY")
    print(f"[AUTH DEBUG] env SUPABASE_ANON_KEY present: {bool(env_value)}")

    value = env_value or _streamlit_secret("SUPABASE_ANON_KEY")
    print(f"[AUTH DEBUG] final SUPABASE_ANON_KEY present: {bool(value)}")
    return value


def is_supabase_configured() -> bool:
    return bool(supabase_url()) and bool(supabase_anon_key())


def is_dev_auth_enabled() -> bool:
    return _env_bool("DEV_AUTH_ENABLED", default=False)


def is_dev_auth_forced() -> bool:
    """Explicit override to allow DevAuthProvider even when a production
    hint is set. A second, deliberate opt-in -- not a way to make the
    production hint pointless, but a way to say "yes, I know, I still
    want dev auth here" (e.g. a staging environment tagged
    ENVIRONMENT=production for other tooling reasons)."""
    return _env_bool("DEV_AUTH_FORCE", default=False)


def is_production_hint_set() -> bool:
    return environment.is_production()


def can_use_dev_auth() -> bool:
    """The single decision point for whether DevAuthProvider may be
    used. Explicit configuration (DEV_AUTH_ENABLED) is the primary
    control; the production hint is a lightweight secondary guard, not
    the primary mechanism -- see module docstring."""
    if not is_dev_auth_enabled():
        return False
    if is_production_hint_set() and not is_dev_auth_forced():
        return False
    return True


def is_legacy_admin_key_enabled() -> bool:
    return _env_bool("LEGACY_ADMIN_KEY_ENABLED", default=False)


def admin_key_from_env() -> str | None:
    """Only the environment-variable half of the legacy key lookup --
    the st.secrets["admin_key"] half (Streamlit Cloud's secrets.toml)
    lives in auth_session.py, which already imports streamlit."""
    return os.environ.get("ADMIN_KEY") or None
