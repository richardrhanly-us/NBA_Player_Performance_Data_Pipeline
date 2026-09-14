"""
Step 14: the single source of truth for "is this a production
deployment" -- APP_ENV classification (Phase 2).

Previously (Step 12) this exact ENVIRONMENT/APP_ENV read lived inline in
src/services/auth_config.py, scoped only to the dev-auth production
hint. Step 14 needs the same classification for readiness checks
(src/services/readiness.py) and the production smoke test
(scripts/production_smoke_test.py), so it's centralized here and
auth_config.py now delegates to it -- no other module reads
ENVIRONMENT/APP_ENV directly (see tests/test_step14_static_guards.py).

No Streamlit import here -- pure/testable, usable from the webhook
service and scripts as well as the Streamlit apps.
"""

from __future__ import annotations

import os

PRODUCTION = "production"
DEVELOPMENT = "development"


def get_app_env() -> str:
    """Returns "production" or "development" (the default). Accepts
    APP_ENV or ENVIRONMENT (either spelling), case-insensitively, with
    "production"/"prod" both meaning production."""
    raw = (os.environ.get("APP_ENV") or os.environ.get("ENVIRONMENT") or "").strip().lower()
    return PRODUCTION if raw in ("production", "prod") else DEVELOPMENT


def is_production() -> bool:
    return get_app_env() == PRODUCTION
