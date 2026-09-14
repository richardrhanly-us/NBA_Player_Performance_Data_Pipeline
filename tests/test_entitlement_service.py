from datetime import datetime, timedelta, timezone

from src.domain.accounts import AccessTier
from src.services import entitlement_service as es


def test_anonymous_entitlements_are_fully_restricted():
    ent = es.for_anonymous()
    assert ent.tier is None
    assert ent.can_view_full_edge_board is False
    assert ent.can_view_full_prediction_history is False
    assert ent.can_view_qualified_edges is False
    assert ent.can_view_advanced_metrics is False
    assert ent.can_export_data is False
    assert ent.can_access_admin is False
    assert ent.can_manage_users is False


def test_free_entitlements_are_restricted_but_authenticated():
    ent = es.for_tier(AccessTier.FREE)
    assert ent.tier == AccessTier.FREE
    assert ent.can_view_full_edge_board is False
    assert ent.can_export_data is False
    assert ent.can_access_admin is False


def test_pro_entitlements_grant_full_customer_access_but_not_admin():
    ent = es.for_tier(AccessTier.PRO)
    assert ent.can_view_full_edge_board is True
    assert ent.can_view_full_prediction_history is True
    assert ent.can_view_qualified_edges is True
    assert ent.can_view_advanced_metrics is True
    assert ent.can_export_data is True
    assert ent.can_access_admin is False
    assert ent.can_manage_users is False


def test_admin_entitlements_grant_everything():
    ent = es.for_tier(AccessTier.ADMIN)
    assert ent.can_view_full_edge_board is True
    assert ent.can_export_data is True
    assert ent.can_access_admin is True
    assert ent.can_manage_users is True


def test_missing_user_resolves_to_free():
    tier = es.compute_effective_tier(user=None, subscription=None, active_override=None)
    assert tier == AccessTier.FREE


def test_disabled_user_resolves_to_free_even_with_active_subscription():
    user = {"is_active": False}
    subscription = {"status": "active", "plan_key": "pro", "current_period_end": None}
    tier = es.compute_effective_tier(
        user=user, subscription=subscription, active_override=None
    )
    assert tier == AccessTier.FREE


def test_no_subscription_and_no_override_resolves_to_free():
    user = {"is_active": True}
    tier = es.compute_effective_tier(user=user, subscription=None, active_override=None)
    assert tier == AccessTier.FREE


def test_active_pro_subscription_resolves_to_pro():
    user = {"is_active": True}
    subscription = {"status": "active", "plan_key": "pro", "current_period_end": None}
    tier = es.compute_effective_tier(
        user=user, subscription=subscription, active_override=None
    )
    assert tier == AccessTier.PRO


def test_expired_subscription_resolves_to_free():
    user = {"is_active": True}
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    subscription = {"status": "active", "plan_key": "pro", "current_period_end": past}
    tier = es.compute_effective_tier(
        user=user, subscription=subscription, active_override=None
    )
    assert tier == AccessTier.FREE


def test_canceled_subscription_resolves_to_free():
    user = {"is_active": True}
    subscription = {"status": "canceled", "plan_key": "pro", "current_period_end": None}
    tier = es.compute_effective_tier(
        user=user, subscription=subscription, active_override=None
    )
    assert tier == AccessTier.FREE


def test_active_override_grants_temporary_pro():
    user = {"is_active": True}
    override = {"override_tier": "PRO"}
    tier = es.compute_effective_tier(user=user, subscription=None, active_override=override)
    assert tier == AccessTier.PRO


def test_active_override_grants_permanent_admin():
    user = {"is_active": True}
    override = {"override_tier": "ADMIN"}
    tier = es.compute_effective_tier(user=user, subscription=None, active_override=override)
    assert tier == AccessTier.ADMIN


def test_override_takes_priority_over_subscription():
    user = {"is_active": True}
    subscription = {"status": "canceled", "plan_key": "pro", "current_period_end": None}
    override = {"override_tier": "PRO"}
    tier = es.compute_effective_tier(
        user=user, subscription=subscription, active_override=override
    )
    assert tier == AccessTier.PRO


def test_unrecognized_override_value_fails_closed_to_free():
    user = {"is_active": True}
    override = {"override_tier": "NOT_A_REAL_TIER"}
    tier = es.compute_effective_tier(user=user, subscription=None, active_override=override)
    assert tier == AccessTier.FREE
