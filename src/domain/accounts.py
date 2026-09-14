"""
Step 11: the centralized account/entitlement domain model.

Everything here is plain dataclasses/enums with no Streamlit and no
database import -- these are the shapes that
src/services/accounts_repository.py reads/writes and that
src/services/entitlement_service.py computes from. Pages
(apps/publicapp.py, apps/adminapp.py) are expected to depend on
`Entitlements` booleans and `CurrentUser`, never on raw plan strings --
see the Step 11 report's architectural-guards section
(tests/test_step11_static_guards.py) for what this is guarding against.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AccessTier(str, Enum):
    """The tiers a *stored* user account can effectively hold. Anonymous
    (no account at all) is deliberately NOT a member of this enum -- it
    is represented by CurrentUser.user being None, with
    EntitlementService.for_anonymous() supplying its entitlements."""

    FREE = "FREE"
    PRO = "PRO"
    ADMIN = "ADMIN"


@dataclass(frozen=True)
class User:
    id: int
    email: str
    display_name: Optional[str]
    auth_provider: str
    auth_subject: str
    is_active: bool
    created_at: str


@dataclass(frozen=True)
class Subscription:
    id: int
    user_id: int
    provider: str
    provider_customer_id: Optional[str]
    provider_subscription_id: Optional[str]
    plan_key: str
    status: str
    current_period_start: Optional[str]
    current_period_end: Optional[str]
    cancel_at_period_end: bool


@dataclass(frozen=True)
class EntitlementOverride:
    id: int
    user_id: int
    override_tier: AccessTier
    enabled: bool
    reason: Optional[str]
    expires_at: Optional[str]
    created_by: Optional[str]
    created_at: str


@dataclass(frozen=True)
class Entitlements:
    """The single source of truth for "can this session do X" -- pages
    branch on these booleans, never on `tier` directly or on any raw
    plan/role string. See EntitlementService.for_tier()/for_anonymous()
    for the only place these are ever constructed."""

    tier: Optional[AccessTier]  # None means anonymous (no account)
    can_view_full_edge_board: bool
    can_view_full_prediction_history: bool
    can_view_qualified_edges: bool
    can_view_advanced_metrics: bool
    can_export_data: bool
    can_access_admin: bool
    can_manage_users: bool

    @property
    def tier_label(self) -> str:
        return self.tier.value if self.tier is not None else "ANONYMOUS"


@dataclass(frozen=True)
class CurrentUser:
    """The one object pages ask for identity/access questions. `user` is
    None for anonymous visitors; `entitlements` is always populated
    (EntitlementService.for_anonymous() for anonymous visitors) so
    callers never need a None-check before reading a can_* flag."""

    user: Optional[User]
    entitlements: Entitlements

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None

    @property
    def display_label(self) -> str:
        if self.user is None:
            return "Anonymous"
        return self.user.display_name or self.user.email
