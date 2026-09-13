"""
Step 10: centralized, non-secret automation configuration. Every value
here is an operational/scheduling knob (cadence, timezone, the kill
switch) -- NOT a modeling value. EDGE_THRESHOLD stays exactly where it
already lives (src/shared_app.py) and is deliberately NOT duplicated or
re-exported here; automation config and prediction-quality policy are
different concerns (see the Step 10 report's Phase 11 section).

Secrets (ODDS_API_KEY, DATABASE_URL) are never read or stored here --
they stay in the environment/GitHub Actions secrets, read directly by
src/services/db_connection.py and the orchestration entry points, same
as Step 9.

All values are read from the environment with production-safe
defaults, so a fresh checkout with no configuration at all behaves
exactly like "automation is off" -- the kill switch defaults closed.
"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# The kill switch (Phase 18). Defaults to False so merging Step 10's
# workflow files to main does NOT silently start hitting a production
# database/Odds API -- a human must explicitly set
# PREDICTION_AUTOMATION_ENABLED=true (a GitHub Actions repository
# variable, not a secret) to turn it on, and can flip it back to false
# at any time to disable automation instantly without touching code.
AUTOMATION_ENABLED = _env_bool("PREDICTION_AUTOMATION_ENABLED", default=False)

# IANA timezone automation uses for "what date is today" / pregame
# windows -- matches scripts/pregame_pipeline.py's existing convention.
AUTOMATION_TIMEZONE = os.environ.get(
    "PREDICTION_AUTOMATION_TIMEZONE", "America/Chicago"
)

# How stale (hours) a persisted run must be before the UI calls it
# STALE rather than FRESH -- see
# src/services/orchestration.py::compute_board_freshness. This is a
# freshness-display knob, not a generation cadence.
STALE_AFTER_HOURS = _env_float("PREDICTION_STALE_AFTER_HOURS", default=6.0)

# Odds-API fetch retry policy (Step 10 Phase 9) -- see
# src/services/orchestration.py::run_prediction_cycle.
FETCH_MAX_ATTEMPTS = int(_env_float("PREDICTION_FETCH_MAX_ATTEMPTS", default=2))
FETCH_RETRY_DELAY_SECONDS = _env_float(
    "PREDICTION_FETCH_RETRY_DELAY_SECONDS", default=3.0
)

# bookmaker: intentionally NOT duplicated here -- src/shared_app.py's
# existing BOOKMAKER_KEY constant remains the single source of truth
# (orchestration.py imports it directly), same as every other Step
# 8/9 caller.
