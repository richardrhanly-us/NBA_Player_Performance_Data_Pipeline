"""
Step 11: the single centralized place tier/entitlement logic lives.

Nothing outside this module decides what a FREE/PRO/ADMIN/anonymous
session can do -- pages and other services only ever read the resulting
`Entitlements` booleans (see src/domain/accounts.py). This is what
tests/test_step11_static_guards.py enforces by scanning apps/ and src/
for scattered `== "pro"`-style checks.

compute_effective_tier() is intentionally the ONLY function that maps
(user, subscription, override) -> AccessTier. Everything else here is a
pure function of an AccessTier (or "anonymous") to an Entitlements
bundle, so adding a future tier only ever touches _ENTITLEMENTS_BY_TIER.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.domain.accounts import AccessTier, Entitlements

# One row per tier -- the only table this module needs to grow when a
# new tier (e.g. a future "TEAM" tier) is introduced. Anonymous is
# handled separately by for_anonymous() below, since it is not a stored
# AccessTier at all.
_ENTITLEMENTS_BY_TIER: dict[AccessTier, Entitlements] = {
    AccessTier.FREE: Entitlements(
        tier=AccessTier.FREE,
        can_view_full_edge_board=False,
        can_view_full_prediction_history=False,
        can_view_qualified_edges=False,
        can_view_advanced_metrics=False,
        can_export_data=False,
        can_access_admin=False,
        can_manage_users=False,
    ),
    AccessTier.PRO: Entitlements(
        tier=AccessTier.PRO,
        can_view_full_edge_board=True,
        can_view_full_prediction_history=True,
        can_view_qualified_edges=True,
        can_view_advanced_metrics=True,
        can_export_data=True,
        can_access_admin=False,
        can_manage_users=False,
    ),
    AccessTier.ADMIN: Entitlements(
        tier=AccessTier.ADMIN,
        can_view_full_edge_board=True,
        can_view_full_prediction_history=True,
        can_view_qualified_edges=True,
        can_view_advanced_metrics=True,
        can_export_data=True,
        can_access_admin=True,
        can_manage_users=True,
    ),
}

_ANONYMOUS_ENTITLEMENTS = Entitlements(
    tier=None,
    can_view_full_edge_board=False,
    can_view_full_prediction_history=False,
    can_view_qualified_edges=False,
    can_view_advanced_metrics=False,
    can_export_data=False,
    can_access_admin=False,
    can_manage_users=False,
)

# Row counts/windows used to truncate anonymous/FREE views -- centralized
# here (not sprinkled through apps/publicapp.py) since they are part of
# the tier policy, not presentation detail.
FREE_EDGE_BOARD_ROW_LIMIT = 5
FREE_HISTORY_MAX_ROWS = 10
FREE_HISTORY_MAX_DAYS = 7

_PAID_STATUSES = ("active", "trialing")
_PLAN_KEY_TO_TIER = {"pro": AccessTier.PRO}


def for_anonymous() -> Entitlements:
    return _ANONYMOUS_ENTITLEMENTS


def for_tier(tier: AccessTier) -> Entitlements:
    return _ENTITLEMENTS_BY_TIER[tier]


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_effective_tier(
    *,
    user: dict | None,
    subscription: dict | None,
    active_override: dict | None,
    now: datetime | None = None,
) -> AccessTier:
    """
    The one place tier is decided. `user`/`subscription`/`active_override`
    are plain dicts as returned by src/services/accounts_repository.py
    (never ORM/domain objects here, to keep this importable without a
    live DB in tests). `active_override` must already be filtered to
    "enabled and not expired" by the caller (see
    accounts_repository.get_active_override) -- this function does not
    re-check expiry so both real callers and tests share one definition
    of "expired".

    Fails closed: any missing/ambiguous/expired signal resolves to FREE,
    never to PRO or ADMIN.
    """
    now = now or datetime.now(timezone.utc)

    if user is None or not user.get("is_active", False):
        return AccessTier.FREE

    if active_override is not None:
        try:
            return AccessTier(active_override["override_tier"])
        except ValueError:
            pass  # unrecognized override value -- fail closed below

    if subscription is not None:
        status = subscription.get("status")
        period_end = _parse_dt(subscription.get("current_period_end"))
        not_expired = period_end is None or period_end > now
        if status in _PAID_STATUSES and not_expired:
            tier = _PLAN_KEY_TO_TIER.get(str(subscription.get("plan_key", "")).lower())
            if tier is not None:
                return tier

    return AccessTier.FREE
