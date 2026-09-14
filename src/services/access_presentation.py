"""
Step 11: server-side truncation helpers for FREE/anonymous views.

Kept separate from apps/publicapp.py so the "how much does a FREE/
anonymous visitor get to see" policy is unit-testable without a
Streamlit runtime, and so it is enforced once here rather than by
whichever widgets happen to be hidden in the UI (hiding a widget is not
authorization -- see the Step 11 report's security-review section).

The board/history data itself is built exactly once regardless of tier
(see build_daily_prediction_board/get_prediction_history_cached in
apps/publicapp.py) -- these functions only ever cut down an
already-built result; they never change what is computed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.domain.accounts import Entitlements
from src.services.entitlement_service import (
    FREE_EDGE_BOARD_ROW_LIMIT,
    FREE_HISTORY_MAX_DAYS,
    FREE_HISTORY_MAX_ROWS,
)


def limit_rows(rows: list, *, allowed_full: bool, free_limit: int) -> tuple[list, bool]:
    """Returns (possibly-truncated rows, was_truncated). Never expands
    or reorders `rows` -- only ever slices the first `free_limit` when
    `allowed_full` is False and there is anything to cut."""
    if allowed_full or len(rows) <= free_limit:
        return rows, False
    return rows[:free_limit], True


def limit_edge_board_rows(rows: list, entitlements: Entitlements) -> tuple[list, bool]:
    return limit_rows(
        rows,
        allowed_full=entitlements.can_view_full_edge_board,
        free_limit=FREE_EDGE_BOARD_ROW_LIMIT,
    )


def limit_history_rows(rows: list, entitlements: Entitlements) -> tuple[list, bool]:
    return limit_rows(
        rows,
        allowed_full=entitlements.can_view_full_prediction_history,
        free_limit=FREE_HISTORY_MAX_ROWS,
    )


def clamp_history_start_date(
    entitlements: Entitlements,
    requested_start_date: str | None,
    *,
    now: datetime | None = None,
) -> str | None:
    """
    Clamps the query's start_date to the FREE history window when the
    caller isn't entitled to full history, regardless of what the UI's
    date-range widget was set to -- this is the actual enforcement point
    (a database query filter), not just a truncated display.
    """
    if entitlements.can_view_full_prediction_history:
        return requested_start_date

    now = now or datetime.now(timezone.utc)
    max_window_start = (now - timedelta(days=FREE_HISTORY_MAX_DAYS)).isoformat()

    if requested_start_date is None:
        return max_window_start
    return max(requested_start_date, max_window_start)
