from datetime import datetime, timezone

from src.services import access_presentation as ap
from src.services import entitlement_service as es
from src.domain.accounts import AccessTier


def test_free_edge_board_is_truncated():
    rows = list(range(20))
    truncated, was_truncated = ap.limit_edge_board_rows(rows, es.for_tier(AccessTier.FREE))
    assert was_truncated is True
    assert len(truncated) == es.FREE_EDGE_BOARD_ROW_LIMIT
    assert truncated == rows[: es.FREE_EDGE_BOARD_ROW_LIMIT]


def test_anonymous_edge_board_is_truncated():
    rows = list(range(20))
    truncated, was_truncated = ap.limit_edge_board_rows(rows, es.for_anonymous())
    assert was_truncated is True
    assert len(truncated) == es.FREE_EDGE_BOARD_ROW_LIMIT


def test_pro_edge_board_is_not_truncated():
    rows = list(range(20))
    truncated, was_truncated = ap.limit_edge_board_rows(rows, es.for_tier(AccessTier.PRO))
    assert was_truncated is False
    assert truncated == rows


def test_admin_edge_board_is_not_truncated():
    rows = list(range(20))
    truncated, was_truncated = ap.limit_edge_board_rows(rows, es.for_tier(AccessTier.ADMIN))
    assert was_truncated is False
    assert truncated == rows


def test_short_board_is_never_marked_truncated():
    rows = [1, 2]
    truncated, was_truncated = ap.limit_edge_board_rows(rows, es.for_tier(AccessTier.FREE))
    assert was_truncated is False
    assert truncated == rows


def test_free_history_rows_are_truncated():
    rows = list(range(50))
    truncated, was_truncated = ap.limit_history_rows(rows, es.for_tier(AccessTier.FREE))
    assert was_truncated is True
    assert len(truncated) == es.FREE_HISTORY_MAX_ROWS


def test_pro_history_rows_are_not_truncated():
    rows = list(range(50))
    truncated, was_truncated = ap.limit_history_rows(rows, es.for_tier(AccessTier.PRO))
    assert was_truncated is False
    assert len(truncated) == 50


def test_pro_history_start_date_is_not_clamped():
    requested = "2020-01-01T00:00:00+00:00"
    result = ap.clamp_history_start_date(es.for_tier(AccessTier.PRO), requested)
    assert result == requested


def test_pro_history_all_time_is_not_clamped():
    result = ap.clamp_history_start_date(es.for_tier(AccessTier.PRO), None)
    assert result is None


def test_free_history_all_time_is_clamped_to_window():
    now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    result = ap.clamp_history_start_date(es.for_tier(AccessTier.FREE), None, now=now)
    assert result is not None
    assert result < now.isoformat()


def test_free_history_requested_start_older_than_window_is_clamped():
    now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    requested = "2020-01-01T00:00:00+00:00"  # far older than the FREE window
    result = ap.clamp_history_start_date(es.for_tier(AccessTier.FREE), requested, now=now)
    assert result != requested
    assert result > requested


def test_free_history_requested_start_within_window_is_kept():
    now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    requested = (now.replace(day=14)).isoformat()  # 1 day back, inside the FREE window
    result = ap.clamp_history_start_date(es.for_tier(AccessTier.FREE), requested, now=now)
    assert result == requested
