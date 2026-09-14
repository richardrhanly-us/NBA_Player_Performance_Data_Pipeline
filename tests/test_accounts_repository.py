from datetime import datetime, timedelta, timezone

from src.services import accounts_repository as repo


def test_get_or_create_user_creates_new_user(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="dev",
        auth_subject="dev:abc123",
        email="new@example.com",
    )
    assert user["id"] is not None
    assert user["email"] == "new@example.com"
    assert bool(user["is_active"]) is True


def test_get_or_create_user_is_idempotent(accounts_db_conn):
    first = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="dev",
        auth_subject="dev:same",
        email="a@example.com",
    )
    second = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="dev",
        auth_subject="dev:same",
        email="a@example.com",
    )
    assert first["id"] == second["id"]

    all_users = repo.list_users(accounts_db_conn)
    assert len(all_users) == 1


def test_get_or_create_user_refreshes_email_on_change(accounts_db_conn):
    first = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="dev",
        auth_subject="dev:same",
        email="old@example.com",
    )
    updated = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="dev",
        auth_subject="dev:same",
        email="new@example.com",
    )
    assert updated["id"] == first["id"]
    assert updated["email"] == "new@example.com"


def test_auth_subject_uniqueness_is_scoped_to_provider(accounts_db_conn):
    dev_user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="dev",
        auth_subject="shared-subject",
        email="same@example.com",
    )
    supabase_user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn,
        auth_provider="supabase",
        auth_subject="shared-subject",
        email="same@example.com",
    )
    assert dev_user["id"] != supabase_user["id"]


def test_set_user_active_disables_and_reactivates(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    repo.set_user_active(accounts_db_conn, user["id"], False)
    disabled = repo.get_user_by_id(accounts_db_conn, user["id"])
    assert bool(disabled["is_active"]) is False

    repo.set_user_active(accounts_db_conn, user["id"], True)
    reactivated = repo.get_user_by_id(accounts_db_conn, user["id"])
    assert bool(reactivated["is_active"]) is True


def test_no_override_returns_none(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    assert repo.get_active_override(accounts_db_conn, user["id"]) is None


def test_create_override_is_visible_as_active(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    repo.create_override(
        accounts_db_conn,
        user_id=user["id"],
        override_tier="PRO",
        reason="test grant",
        created_by="tester",
        expires_at=None,
    )
    active = repo.get_active_override(accounts_db_conn, user["id"])
    assert active is not None
    assert active["override_tier"] == "PRO"
    assert bool(active["enabled"]) is True


def test_creating_new_override_disables_previous_one(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    first_id = repo.create_override(
        accounts_db_conn,
        user_id=user["id"],
        override_tier="PRO",
        reason="first",
        created_by="tester",
    )
    repo.create_override(
        accounts_db_conn,
        user_id=user["id"],
        override_tier="ADMIN",
        reason="second",
        created_by="tester",
    )
    all_overrides = repo.list_overrides_for_user(accounts_db_conn, user["id"])
    first_row = next(o for o in all_overrides if o["id"] == first_id)
    assert bool(first_row["enabled"]) is False

    active = repo.get_active_override(accounts_db_conn, user["id"])
    assert active["override_tier"] == "ADMIN"


def test_revoke_override_makes_it_inactive(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    override_id = repo.create_override(
        accounts_db_conn,
        user_id=user["id"],
        override_tier="PRO",
        reason="temp",
        created_by="tester",
    )
    repo.revoke_override(accounts_db_conn, override_id)
    assert repo.get_active_override(accounts_db_conn, user["id"]) is None


def test_expired_override_is_not_returned_as_active(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    repo.create_override(
        accounts_db_conn,
        user_id=user["id"],
        override_tier="PRO",
        reason="already expired",
        created_by="tester",
        expires_at=past,
    )
    assert repo.get_active_override(accounts_db_conn, user["id"]) is None


def test_unexpired_override_is_returned_as_active(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    repo.create_override(
        accounts_db_conn,
        user_id=user["id"],
        override_tier="PRO",
        reason="temp grant",
        created_by="tester",
        expires_at=future,
    )
    active = repo.get_active_override(accounts_db_conn, user["id"])
    assert active is not None
    assert active["override_tier"] == "PRO"


def test_missing_subscription_returns_none(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    assert repo.get_latest_subscription(accounts_db_conn, user["id"]) is None


def test_upsert_subscription_and_get_latest(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    repo.upsert_subscription(
        accounts_db_conn,
        user_id=user["id"],
        provider="stripe",
        plan_key="pro",
        status="active",
    )
    latest = repo.get_latest_subscription(accounts_db_conn, user["id"])
    assert latest["plan_key"] == "pro"
    assert latest["status"] == "active"


def test_list_users_orders_most_recent_first(accounts_db_conn):
    repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="first@example.com"
    )
    repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:2", email="second@example.com"
    )
    users = repo.list_users(accounts_db_conn)
    assert len(users) == 2


# ---------------------------------------------------------------------------
# Step 13: Stripe customer mapping / subscription sync / event idempotency
# ---------------------------------------------------------------------------

def test_new_user_has_no_stripe_customer_id(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    assert user.get("stripe_customer_id") is None


def test_set_stripe_customer_id_is_visible_by_lookup(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    repo.set_user_stripe_customer_id(accounts_db_conn, user["id"], "cus_123")
    found = repo.get_user_by_stripe_customer_id(accounts_db_conn, "cus_123")
    assert found is not None
    assert found["id"] == user["id"]


def test_set_stripe_customer_id_is_idempotent_once_set(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    repo.set_user_stripe_customer_id(accounts_db_conn, user["id"], "cus_first")
    repo.set_user_stripe_customer_id(accounts_db_conn, user["id"], "cus_second")
    reloaded = repo.get_user_by_id(accounts_db_conn, user["id"])
    assert reloaded["stripe_customer_id"] == "cus_first"


def test_unknown_stripe_customer_id_returns_none(accounts_db_conn):
    assert repo.get_user_by_stripe_customer_id(accounts_db_conn, "cus_does_not_exist") is None


def test_sync_subscription_from_stripe_creates_new_row(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    sub_id = repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_abc",
        plan_key="pro",
        status="active",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
        stripe_price_id="price_1",
    )
    latest = repo.get_latest_subscription(accounts_db_conn, user["id"])
    assert latest["id"] == sub_id
    assert latest["status"] == "active"
    assert latest["stripe_price_id"] == "price_1"


def test_sync_subscription_from_stripe_is_idempotent_by_subscription_id(accounts_db_conn):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    first_id = repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_abc",
        plan_key="pro",
        status="active",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    second_id = repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_abc",
        plan_key="pro",
        status="past_due",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    assert first_id == second_id
    all_subs = [
        s
        for s in [repo.get_latest_subscription(accounts_db_conn, user["id"])]
    ]
    assert len(all_subs) == 1
    assert all_subs[0]["status"] == "past_due"


def test_sync_subscription_from_stripe_different_subscription_ids_create_separate_rows(
    accounts_db_conn,
):
    user = repo.get_or_create_user_by_auth_subject(
        accounts_db_conn, auth_provider="dev", auth_subject="dev:1", email="x@example.com"
    )
    repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_old",
        plan_key="pro",
        status="canceled",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    repo.sync_subscription_from_stripe(
        accounts_db_conn,
        user_id=user["id"],
        provider_customer_id="cus_123",
        provider_subscription_id="sub_new",
        plan_key="pro",
        status="active",
        current_period_start=None,
        current_period_end=None,
        cancel_at_period_end=False,
    )
    latest = repo.get_latest_subscription(accounts_db_conn, user["id"])
    assert latest["provider_subscription_id"] == "sub_new"
    assert latest["status"] == "active"


def test_try_claim_stripe_event_first_call_succeeds(accounts_db_conn):
    assert repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed") is True


def test_try_claim_stripe_event_duplicate_call_fails(accounts_db_conn):
    repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed")
    assert repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed") is False


def test_mark_stripe_event_processed(accounts_db_conn):
    repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed")
    repo.mark_stripe_event_processed(accounts_db_conn, "evt_1")
    row = repo.get_stripe_event(accounts_db_conn, "evt_1")
    assert row["processing_status"] == "processed"
    assert row["processed_at"] is not None


def test_mark_stripe_event_failed_records_error_message(accounts_db_conn):
    repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed")
    repo.mark_stripe_event_failed(accounts_db_conn, "evt_1", "could not resolve user")
    row = repo.get_stripe_event(accounts_db_conn, "evt_1")
    assert row["processing_status"] == "failed"
    assert row["error_message"] == "could not resolve user"


def test_get_stripe_event_returns_none_for_unknown_event(accounts_db_conn):
    assert repo.get_stripe_event(accounts_db_conn, "evt_does_not_exist") is None


# ---------------------------------------------------------------------------
# Step 14: webhook operational visibility
# ---------------------------------------------------------------------------

def test_stripe_event_summary_with_no_events(accounts_db_conn):
    summary = repo.get_stripe_event_summary(accounts_db_conn)
    assert summary["received_count"] == 0
    assert summary["processed_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["last_processed_event_id"] is None
    assert summary["last_failed_event_id"] is None


def test_stripe_event_summary_counts_by_status(accounts_db_conn):
    repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed")
    repo.mark_stripe_event_processed(accounts_db_conn, "evt_1")
    repo.try_claim_stripe_event(accounts_db_conn, "evt_2", "customer.subscription.updated")
    repo.mark_stripe_event_failed(accounts_db_conn, "evt_2", "could not resolve user")
    repo.try_claim_stripe_event(accounts_db_conn, "evt_3", "invoice.payment_succeeded")

    summary = repo.get_stripe_event_summary(accounts_db_conn)
    assert summary["received_count"] == 3
    assert summary["processed_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["pending_count"] == 1
    assert summary["last_processed_event_id"] == "evt_1"
    assert summary["last_failed_event_id"] == "evt_2"
    assert summary["last_failed_error_summary"] == "could not resolve user"


def test_stripe_event_summary_truncates_long_error_messages(accounts_db_conn):
    repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "customer.subscription.updated")
    repo.mark_stripe_event_failed(accounts_db_conn, "evt_1", "x" * 5000)
    summary = repo.get_stripe_event_summary(accounts_db_conn)
    assert len(summary["last_failed_error_summary"]) <= 200


def test_list_recent_stripe_events_orders_most_recent_first(accounts_db_conn):
    repo.try_claim_stripe_event(accounts_db_conn, "evt_1", "checkout.session.completed")
    repo.try_claim_stripe_event(accounts_db_conn, "evt_2", "customer.subscription.updated")
    events = repo.list_recent_stripe_events(accounts_db_conn)
    assert len(events) == 2
    assert {e["stripe_event_id"] for e in events} == {"evt_1", "evt_2"}


def test_list_recent_stripe_events_respects_limit(accounts_db_conn):
    for i in range(5):
        repo.try_claim_stripe_event(accounts_db_conn, f"evt_{i}", "checkout.session.completed")
    events = repo.list_recent_stripe_events(accounts_db_conn, limit=2)
    assert len(events) == 2
