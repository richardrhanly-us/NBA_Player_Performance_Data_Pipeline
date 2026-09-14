"""
Step 13: the Streamlit-facing billing glue layer -- mirrors
src/services/auth_session.py's role for authentication. This is the
ONLY module that renders billing UI or calls
src/services/billing_provider.py from a Streamlit context; pages call
render_billing_widget() and nothing else.

No entitlement logic lives here -- it only reads `current_user.entitlements`
(already computed by src/services/auth_session.py::get_current_user())
and, on a button click, delegates to get_billing_provider(). It never
mutates subscriptions/entitlement_overrides directly and never imports
`stripe` (see tests/test_step13_static_guards.py).
"""

from __future__ import annotations

import streamlit as st

from src.domain.accounts import CurrentUser
from src.services.billing_provider import (
    BillingActionDeniedError,
    BillingNotConfiguredError,
    get_billing_provider,
)

_CHECKOUT_QUERY_PARAM = "checkout"


def render_checkout_return_banner() -> None:
    """
    Reads the `?checkout=success|cancel` query param Stripe's own
    success_url/cancel_url redirect back to (see billing_config.py) and
    shows a purely informational message. This NEVER grants PRO or
    touches any session/entitlement state -- forging this query param
    yourself does nothing except show yourself a message; entitlement
    only ever changes once a verified webhook updates the local
    subscriptions table (see StripeBillingProvider.handle_webhook_event),
    which the next rerun's get_current_user() call will reflect
    naturally -- no polling, no client-side trust of Stripe state.
    """
    checkout_result = st.query_params.get(_CHECKOUT_QUERY_PARAM)
    if checkout_result == "success":
        st.success(
            "Payment received -- your PRO subscription is being activated. "
            "This can take a few seconds; refresh if it doesn't appear right away."
        )
    elif checkout_result == "cancel":
        st.info("Checkout was canceled. You can upgrade to PRO anytime.")


def render_billing_widget(current: CurrentUser, container=None) -> None:
    """Renders the plan/upgrade/manage-billing section. Anonymous
    visitors see upgrade messaging with no checkout action (they must
    sign in first -- see src/services/auth_session.py::render_account_widget);
    FREE sees an "Upgrade to PRO" button; PRO sees "Manage Billing";
    ADMIN sees its state without being pushed toward checkout."""
    target = container if container is not None else st.sidebar
    entitlements = current.entitlements

    with target:
        st.markdown("#### Billing")

        if not current.is_authenticated:
            st.caption("Sign in to upgrade to PRO.")
            return

        if entitlements.tier_label == "ADMIN":
            st.caption("Plan: **ADMIN** (full access)")
            return

        if entitlements.tier_label == "PRO":
            st.caption("Plan: **PRO**")
            if st.button("Manage Billing", key="billing_manage_button"):
                _start_portal(current)
            return

        # FREE
        st.caption("Plan: **FREE**")
        if st.button("Upgrade to PRO", key="billing_upgrade_button"):
            _start_checkout(current)


def _start_checkout(current: CurrentUser) -> None:
    try:
        url = get_billing_provider().create_checkout_session(
            user_id=current.user.id, plan_key="pro"
        )
    except BillingNotConfiguredError:
        st.info("Billing is not available for this deployment yet.")
        return
    except BillingActionDeniedError as e:
        st.warning(str(e))
        return
    except Exception:
        st.error("Checkout is temporarily unavailable. Please try again shortly.")
        return
    st.link_button("Continue to Stripe Checkout", url)


def _start_portal(current: CurrentUser) -> None:
    try:
        url = get_billing_provider().create_customer_portal_session(user_id=current.user.id)
    except BillingNotConfiguredError:
        st.info("Billing is not available for this deployment yet.")
        return
    except BillingActionDeniedError as e:
        st.warning(str(e))
        return
    except Exception:
        st.error("The billing portal is temporarily unavailable. Please try again shortly.")
        return
    st.link_button("Open Customer Portal", url)
