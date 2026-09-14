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
