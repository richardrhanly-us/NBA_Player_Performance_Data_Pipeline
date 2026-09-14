"""
Step 14: centralized production-readiness validation, one process type
at a time. src/services/auth_config.py and billing_config.py already
centralize *which* env vars exist; this module centralizes *what a
given process needs to be ready* -- apps/publicapp.py,
apps/adminapp.py, webhook_service/main.py, and scripts/*.py (automation)
each ask one of the check_*_readiness() functions below instead of
re-deriving "am I configured enough to run" by hand.

Every check here is cheap by default (env var presence only). The one
genuinely expensive check -- an actual database connection + schema
check -- is opt-in via `deep=True` on check_webhook_service_readiness()
and is never invoked from Streamlit page rendering (see the Step 14
report's Phase 3/7 sections: "do not add expensive health checks to
every Streamlit rerun"). It exists for webhook_service's /ready endpoint
and scripts/production_smoke_test.py, both of which are called
occasionally (a load balancer probe, a human running a smoke test), not
on every page view.

Never returns or logs a raw secret value or exception message -- every
ReadinessCheck.detail string is a short, fixed, human-written string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from src.services import auth_config, billing_config, environment


class ReadinessStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: ReadinessStatus
    required: bool
    detail: str = ""


@dataclass(frozen=True)
class ReadinessReport:
    process: str
    app_env: str
    checks: tuple[ReadinessCheck, ...]

    @property
    def is_ready(self) -> bool:
        return all(c.status == ReadinessStatus.OK for c in self.checks if c.required)

    def to_dict(self) -> dict:
        return {
            "process": self.process,
            "app_env": self.app_env,
            "ready": self.is_ready,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "required": c.required,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def _check(name, ok, *, required, ok_detail="configured", missing_detail="not configured"):
    return ReadinessCheck(
        name=name,
        status=ReadinessStatus.OK if ok else ReadinessStatus.MISSING,
        required=required,
        detail=ok_detail if ok else missing_detail,
    )


def _database_configured_check(*, required: bool = True) -> ReadinessCheck:
    return _check(
        "database_configured",
        bool(os.environ.get("DATABASE_URL")),
        required=required,
        missing_detail="DATABASE_URL is not set",
    )


def _database_connectivity_check() -> ReadinessCheck:
    """The one expensive check in this module -- a real connection plus
    a schema-readiness query. Never includes the raw exception (which
    could contain the DSN) in the detail string."""
    if not os.environ.get("DATABASE_URL"):
        return ReadinessCheck(
            "database_connectivity", ReadinessStatus.MISSING, True, "DATABASE_URL is not set"
        )
    try:
        from src.services.db_connection import get_prediction_db_connection
        from src.services.migrations import is_schema_ready

        conn = get_prediction_db_connection()
        try:
            schema_ready = is_schema_ready(conn)
        finally:
            conn.close()
    except Exception:
        return ReadinessCheck(
            "database_connectivity", ReadinessStatus.DEGRADED, True, "database unreachable"
        )
    if not schema_ready:
        return ReadinessCheck(
            "database_connectivity",
            ReadinessStatus.DEGRADED,
            True,
            "connected, but required schema is missing (run migrations)",
        )
    return ReadinessCheck(
        "database_connectivity", ReadinessStatus.OK, True, "connected, schema ready"
    )


def _auth_provider_check(*, required: bool = True) -> ReadinessCheck:
    """Whether SOME usable auth provider exists -- Supabase or dev auth
    are both a "ready" answer to this specific question. Whether dev
    auth is appropriate for THIS environment is a separate, dedicated
    check (_dev_auth_not_active_in_production_check) -- conflating the
    two would make a healthy dev deployment report "not ready"."""
    if auth_config.is_supabase_configured():
        return ReadinessCheck("auth_provider", ReadinessStatus.OK, required, "supabase configured")
    if auth_config.can_use_dev_auth():
        return ReadinessCheck(
            "auth_provider", ReadinessStatus.OK, required, "dev auth active"
        )
    return ReadinessCheck(
        "auth_provider", ReadinessStatus.MISSING, required, "no auth provider configured"
    )


def _dev_auth_not_active_in_production_check() -> ReadinessCheck:
    """Phase 2/18: 'production should reject dangerous development
    fallbacks.' auth_config.can_use_dev_auth() already REFUSES dev auth
    under a production hint unless DEV_AUTH_FORCE=true is also set
    (Step 12) -- this check exists to make that fact loudly visible in
    readiness/smoke-test output, and to fail readiness outright if dev
    auth is actually reachable in a production-classified deployment."""
    if not environment.is_production():
        return ReadinessCheck(
            "dev_auth_not_in_production", ReadinessStatus.OK, False, "not a production environment"
        )
    if auth_config.can_use_dev_auth():
        return ReadinessCheck(
            "dev_auth_not_in_production",
            ReadinessStatus.DEGRADED,
            True,
            "DEV AUTH IS ACTIVE IN A PRODUCTION ENVIRONMENT",
        )
    return ReadinessCheck(
        "dev_auth_not_in_production", ReadinessStatus.OK, True, "dev auth is not active"
    )


def _legacy_admin_key_production_note() -> ReadinessCheck:
    """Informational only (required=False) -- a freshly-bootstrapped
    production deployment legitimately needs the legacy key briefly
    (there is no ADMIN account yet to sign in with), so this must not
    hard-fail readiness. It exists purely to surface the fact loudly so
    it doesn't get forgotten -- see the Step 14 report's auth-incident
    section: 'disable LEGACY_ADMIN_KEY_ENABLED after first-admin
    bootstrap.'"""
    if not environment.is_production():
        return ReadinessCheck(
            "legacy_admin_key_disabled_or_bootstrap_only",
            ReadinessStatus.OK,
            False,
            "not a production environment",
        )
    if auth_config.is_legacy_admin_key_enabled():
        return ReadinessCheck(
            "legacy_admin_key_disabled_or_bootstrap_only",
            ReadinessStatus.DEGRADED,
            False,
            "legacy admin key is enabled in production -- disable after first-admin bootstrap",
        )
    return ReadinessCheck(
        "legacy_admin_key_disabled_or_bootstrap_only", ReadinessStatus.OK, False, "disabled"
    )


def _stripe_checkout_config_check(*, required: bool = False) -> ReadinessCheck:
    return _check(
        "stripe_checkout_configured",
        billing_config.is_stripe_configured(),
        required=required,
        missing_detail="billing not configured (checkout/portal disabled -- optional)",
    )


def _stripe_webhook_config_check(*, required: bool = True) -> ReadinessCheck:
    return _check(
        "stripe_webhook_configured",
        billing_config.is_webhook_configured(),
        required=required,
        missing_detail="STRIPE_WEBHOOK_SECRET is not set",
    )


def _odds_api_configured_check(*, required: bool = False) -> ReadinessCheck:
    return _check(
        "odds_api_configured",
        bool(os.environ.get("ODDS_API_KEY")),
        required=required,
        missing_detail="ODDS_API_KEY is not set (Edge Board degrades gracefully)",
    )


def _automation_enabled_check() -> ReadinessCheck:
    from src.services.automation_config import AUTOMATION_ENABLED

    return ReadinessCheck(
        "automation_enabled",
        ReadinessStatus.OK if AUTOMATION_ENABLED else ReadinessStatus.DEGRADED,
        False,
        "enabled" if AUTOMATION_ENABLED else "disabled (kill switch off)",
    )


def check_public_app_readiness() -> ReadinessReport:
    checks = (
        _database_configured_check(required=True),
        _auth_provider_check(required=True),
        _dev_auth_not_active_in_production_check(),
        _stripe_checkout_config_check(required=False),
        _odds_api_configured_check(required=False),
    )
    return ReadinessReport(process="public_app", app_env=environment.get_app_env(), checks=checks)


def check_admin_app_readiness() -> ReadinessReport:
    checks = (
        _database_configured_check(required=True),
        _auth_provider_check(required=True),
        _dev_auth_not_active_in_production_check(),
        _legacy_admin_key_production_note(),
    )
    return ReadinessReport(process="admin_app", app_env=environment.get_app_env(), checks=checks)


def check_webhook_service_readiness(*, deep: bool = False) -> ReadinessReport:
    checks = [
        _database_configured_check(required=True),
        _check(
            "stripe_secret_key_configured",
            bool(billing_config.stripe_secret_key()),
            required=True,
            missing_detail="STRIPE_SECRET_KEY is not set",
        ),
        _stripe_webhook_config_check(required=True),
        _stripe_checkout_config_check(required=False),
    ]
    if deep:
        checks.append(_database_connectivity_check())
    return ReadinessReport(
        process="webhook_service", app_env=environment.get_app_env(), checks=tuple(checks)
    )


def check_automation_readiness() -> ReadinessReport:
    checks = (
        _database_configured_check(required=True),
        _odds_api_configured_check(required=True),
        _automation_enabled_check(),
    )
    return ReadinessReport(process="automation", app_env=environment.get_app_env(), checks=checks)
