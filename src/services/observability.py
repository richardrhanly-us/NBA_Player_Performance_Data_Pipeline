"""
Step 14: structured logging for operationally important services --
webhook_service, billing_provider, auth_session. Standard library
`logging` only (no new dependency).

log_event() emits one JSON line per call with a consistent shape
(timestamp, service, event, severity, plus whatever fields the caller
passes). It defensively redacts any field whose NAME looks sensitive
(see _SENSITIVE_KEY_MARKERS) even though callers should never pass a
secret in the first place -- belt-and-suspenders against a future call
site accidentally doing so. This is the only place log formatting
happens; callers never build their own log strings for these services
(see tests/test_step14_static_guards.py).

Never log: passwords, access/refresh tokens, Supabase anon/service
keys, the Stripe secret key, the Stripe webhook secret, full
Authorization headers, or raw webhook payload bodies. Safe to log:
event names, ids (stripe_event_id, prediction_run_id, user id),
statuses, counts, short error summaries.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_REDACTED = "[REDACTED]"

_SENSITIVE_KEY_MARKERS = (
    "password",
    "token",
    "secret",
    "authorization",
    "api_key",
    "apikey",
    "admin_key",
    "signature",
)


def _redact_fields(fields: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in fields.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS):
            safe[key] = _REDACTED
        else:
            safe[key] = value
    return safe


def get_logger(service: str) -> logging.Logger:
    """One logger per service name, stdout, one JSON object per line --
    plain enough for any log aggregator to ingest without configuration.
    Idempotent: calling this twice for the same service name does not
    attach duplicate handlers."""
    logger = logging.getLogger(f"nba_pipeline.{service}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Deliberately left propagating (the default) -- so pytest's
        # caplog fixture and any host application's root logging
        # configuration still see these records, not just our own
        # stdout handler.
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    severity: str = "info",
    **fields: Any,
) -> None:
    """Emits one structured, redacted JSON log line. `severity` is one
    of the standard logging level names (case-insensitive): debug, info,
    warning, error, critical."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "severity": severity.lower(),
        **_redact_fields(fields),
    }
    line = json.dumps(payload, default=str)
    level = getattr(logging, severity.upper(), logging.INFO)
    logger.log(level, line)
