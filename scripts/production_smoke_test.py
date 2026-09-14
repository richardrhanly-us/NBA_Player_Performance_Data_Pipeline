"""
Step 14: a safe, read-only production-readiness smoke test.

Runs src/services/readiness.py's checks for all four process types
(public app, admin app, webhook service, automation) against whatever
environment this script is run in, and prints a human-readable report.
Exits 0 if every process is ready, 1 otherwise -- suitable as a manual
pre-deploy check or a CI gate.

This script NEVER:
    - creates a Stripe customer, charge, or checkout session
    - mutates any subscription/entitlement_overrides row
    - triggers a prediction cycle or settlement run
    - runs or applies database migrations
    - sends an email

By default it only checks configuration (cheap, no network calls beyond
the one optional deep DB check below). Pass --deep to additionally
attempt a real database connection and schema-readiness check (the one
check in src/services/readiness.py that isn't free) -- still entirely
read-only.

Usage:
    python scripts/production_smoke_test.py
    python scripts/production_smoke_test.py --deep
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services import readiness


def _print_report(report: readiness.ReadinessReport) -> None:
    header = f"[{report.process}] app_env={report.app_env} ready={report.is_ready}"
    print(header, flush=True)
    for check in report.checks:
        marker = "REQUIRED" if check.required else "optional"
        print(f"    - {check.name} [{marker}]: {check.status.value} -- {check.detail}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also attempt a real (read-only) database connection + schema check "
        "for the webhook service. Still makes no writes and never calls Stripe.",
    )
    args = parser.parse_args(argv)

    reports = [
        readiness.check_public_app_readiness(),
        readiness.check_admin_app_readiness(),
        readiness.check_webhook_service_readiness(deep=args.deep),
        readiness.check_automation_readiness(),
    ]

    print("=== Production Readiness Smoke Test (read-only) ===", flush=True)
    for report in reports:
        _print_report(report)
        print("", flush=True)

    all_ready = all(report.is_ready for report in reports)
    print("RESULT: READY" if all_ready else "RESULT: NOT READY", flush=True)
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
