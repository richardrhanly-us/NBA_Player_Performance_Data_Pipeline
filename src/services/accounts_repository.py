"""
Step 11: the accounts/entitlements repository -- users, subscriptions,
and entitlement_overrides (migrations/0002_create_accounts_and_entitlements.sql).

Follows the same conventions as src/services/prediction_repository.py:
every function takes an explicit `conn` (a real psycopg/Postgres
connection via src/services/db_connection.py, or a sqlite3 connection
built from src/services/schema_sqlite.py::create_sqlite_accounts_schema
in tests), SQL is written once with `?` placeholders and translated to
`%s` for psycopg connections, and rows come back as plain dicts.

This module is the ONLY place that writes to users/subscriptions/
entitlement_overrides. Streamlit pages (apps/adminapp.py) call these
functions rather than executing SQL directly -- see the Step 11 report's
architectural-guards section (tests/test_step11_static_guards.py).
"""

from __future__ import annotations

from datetime import datetime, timezone


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg")


def _ph(conn) -> str:
    return "%s" if _is_postgres(conn) else "?"


def _execute(conn, sql: str, params=()):
    placeholder = _ph(conn)
    adapted_sql = sql.replace("?", placeholder) if placeholder != "?" else sql
    cur = conn.cursor()
    cur.execute(adapted_sql, params)
    return cur


def _insert_and_get_id(conn, sql: str, params: tuple):
    if _is_postgres(conn):
        cur = _execute(conn, sql + " RETURNING id", params)
        return cur.fetchone()[0]
    cur = _execute(conn, sql, params)
    return cur.lastrowid


def _row_to_dict(cur, row):
    if row is None:
        return None
    columns = [d[0] for d in cur.description]
    return dict(zip(columns, row))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

def get_user_by_auth_subject(conn, auth_provider: str, auth_subject: str):
    cur = _execute(
        conn,
        "SELECT * FROM users WHERE auth_provider = ? AND auth_subject = ?",
        (auth_provider, auth_subject),
    )
    return _row_to_dict(cur, cur.fetchone())


def get_user_by_id(conn, user_id: int):
    cur = _execute(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
    return _row_to_dict(cur, cur.fetchone())


def get_or_create_user_by_auth_subject(
    conn,
    *,
    auth_provider: str,
    auth_subject: str,
    email: str,
    display_name: str | None = None,
):
    """
    Idempotent identity resolution for a freshly-authenticated session:
    returns the existing users row for (auth_provider, auth_subject) if
    one exists (refreshing email/display_name if they changed upstream),
    otherwise creates one as FREE-by-default (tier is never stored on
    this row directly -- see EntitlementService; a brand-new account has
    no subscription and no override, which computes to FREE).
    """
    existing = get_user_by_auth_subject(conn, auth_provider, auth_subject)
    now = _now_iso()

    if existing is not None:
        if existing["email"] != email or existing.get("display_name") != display_name:
            _execute(
                conn,
                "UPDATE users SET email = ?, display_name = ?, updated_at = ? WHERE id = ?",
                (email, display_name, now, existing["id"]),
            )
            conn.commit()
            return get_user_by_id(conn, existing["id"])
        return existing

    user_id = _insert_and_get_id(
        conn,
        "INSERT INTO users (email, display_name, auth_provider, auth_subject, "
        "is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (email, display_name, auth_provider, auth_subject, True, now, now),
    )
    conn.commit()
    return get_user_by_id(conn, user_id)


def list_users(conn, *, limit: int = 200):
    cur = _execute(
        conn, "SELECT * FROM users ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def set_user_active(conn, user_id: int, is_active: bool) -> None:
    _execute(
        conn,
        "UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?",
        (is_active, _now_iso(), user_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stripe customer mapping (Step 13) -- the stable local-user <-> Stripe
# Customer identity link. Set ONCE per user (only when currently NULL),
# never overwritten, and never resolved by email -- see
# src/services/billing_provider.py, which is the only writer.
# ---------------------------------------------------------------------------

def get_user_by_stripe_customer_id(conn, stripe_customer_id: str):
    cur = _execute(
        conn, "SELECT * FROM users WHERE stripe_customer_id = ?", (stripe_customer_id,)
    )
    return _row_to_dict(cur, cur.fetchone())


def set_user_stripe_customer_id(conn, user_id: int, stripe_customer_id: str) -> None:
    """Only takes effect if this user has no Stripe customer id yet --
    idempotent by construction, so a retried/duplicate checkout attempt
    can never silently swap a user's mapped customer."""
    _execute(
        conn,
        "UPDATE users SET stripe_customer_id = ?, updated_at = ? "
        "WHERE id = ? AND stripe_customer_id IS NULL",
        (stripe_customer_id, _now_iso(), user_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# subscriptions (reserved for real billing-provider sync -- see
# src/services/billing_provider.py; nothing in Step 11 writes here in
# production, but the repository function is provided so a future
# billing sync has a stable, tested write path from day one)
# ---------------------------------------------------------------------------

def get_latest_subscription(conn, user_id: int):
    cur = _execute(
        conn,
        "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    )
    return _row_to_dict(cur, cur.fetchone())


def upsert_subscription(
    conn,
    *,
    user_id: int,
    provider: str,
    plan_key: str,
    status: str,
    provider_customer_id: str | None = None,
    provider_subscription_id: str | None = None,
    current_period_start: str | None = None,
    current_period_end: str | None = None,
    cancel_at_period_end: bool = False,
):
    now = _now_iso()
    sub_id = _insert_and_get_id(
        conn,
        "INSERT INTO subscriptions (user_id, provider, provider_customer_id, "
        "provider_subscription_id, plan_key, status, current_period_start, "
        "current_period_end, cancel_at_period_end, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            provider,
            provider_customer_id,
            provider_subscription_id,
            plan_key,
            status,
            current_period_start,
            current_period_end,
            cancel_at_period_end,
            now,
            now,
        ),
    )
    conn.commit()
    return sub_id


def sync_subscription_from_stripe(
    conn,
    *,
    user_id: int,
    provider_customer_id: str | None,
    provider_subscription_id: str,
    plan_key: str,
    status: str,
    current_period_start: str | None,
    current_period_end: str | None,
    cancel_at_period_end: bool,
    stripe_price_id: str | None = None,
):
    """
    The webhook-driven write path (Step 13) -- idempotent by
    (provider="stripe", provider_subscription_id): updates the existing
    row in place if one already exists for this Stripe subscription,
    otherwise inserts a new one. This is what makes repeated webhook
    delivery for the same subscription a no-op rather than creating
    duplicate rows (see migrations/0003_..._stripe_events.sql's partial
    unique index, which backs this at the database level too). Returns
    the subscription row's id.
    """
    now = _now_iso()
    cur = _execute(
        conn,
        "SELECT id FROM subscriptions WHERE provider = ? AND provider_subscription_id = ?",
        ("stripe", provider_subscription_id),
    )
    existing = cur.fetchone()

    if existing is not None:
        sub_id = existing[0]
        _execute(
            conn,
            "UPDATE subscriptions SET user_id = ?, provider_customer_id = ?, "
            "plan_key = ?, status = ?, current_period_start = ?, "
            "current_period_end = ?, cancel_at_period_end = ?, stripe_price_id = ?, "
            "last_synced_at = ?, updated_at = ? WHERE id = ?",
            (
                user_id,
                provider_customer_id,
                plan_key,
                status,
                current_period_start,
                current_period_end,
                cancel_at_period_end,
                stripe_price_id,
                now,
                now,
                sub_id,
            ),
        )
        conn.commit()
        return sub_id

    sub_id = _insert_and_get_id(
        conn,
        "INSERT INTO subscriptions (user_id, provider, provider_customer_id, "
        "provider_subscription_id, plan_key, status, current_period_start, "
        "current_period_end, cancel_at_period_end, stripe_price_id, "
        "last_synced_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            "stripe",
            provider_customer_id,
            provider_subscription_id,
            plan_key,
            status,
            current_period_start,
            current_period_end,
            cancel_at_period_end,
            stripe_price_id,
            now,
            now,
            now,
        ),
    )
    conn.commit()
    return sub_id


# ---------------------------------------------------------------------------
# stripe_events -- webhook idempotency ledger (Step 13 Phase 14)
# ---------------------------------------------------------------------------

def try_claim_stripe_event(conn, stripe_event_id: str, event_type: str) -> bool:
    """
    Attempts to record a newly-received Stripe event. Returns True if
    this call is the first to see this event id (i.e. it should be
    processed), False if it was already recorded (a duplicate delivery
    -- Stripe retries aggressively, so this MUST be checked before any
    entitlement-affecting work happens). Relies on stripe_event_id's
    UNIQUE constraint -- a second INSERT for the same id fails, which is
    treated as "already claimed" rather than an error.
    """
    try:
        _execute(
            conn,
            "INSERT INTO stripe_events (stripe_event_id, event_type, received_at, "
            "processing_status) VALUES (?, ?, ?, ?)",
            (stripe_event_id, event_type, _now_iso(), "received"),
        )
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def mark_stripe_event_processed(conn, stripe_event_id: str) -> None:
    _execute(
        conn,
        "UPDATE stripe_events SET processing_status = ?, processed_at = ? "
        "WHERE stripe_event_id = ?",
        ("processed", _now_iso(), stripe_event_id),
    )
    conn.commit()


def mark_stripe_event_failed(conn, stripe_event_id: str, error_message: str) -> None:
    """`error_message` must already be a safe-to-store string -- callers
    (billing_provider.py) never pass a raw exception containing a secret
    (API keys, webhook secrets) here; only a short, descriptive message."""
    _execute(
        conn,
        "UPDATE stripe_events SET processing_status = ?, processed_at = ?, "
        "error_message = ? WHERE stripe_event_id = ?",
        ("failed", _now_iso(), str(error_message)[:2000], stripe_event_id),
    )
    conn.commit()


def get_stripe_event(conn, stripe_event_id: str):
    cur = _execute(
        conn, "SELECT * FROM stripe_events WHERE stripe_event_id = ?", (stripe_event_id,)
    )
    return _row_to_dict(cur, cur.fetchone())


def list_recent_stripe_events(conn, *, limit: int = 50):
    """Step 14 Phase 5: operational visibility into webhook processing --
    never returns raw webhook payloads (those are never stored at all;
    only Stripe's event id/type and this app's own processing outcome
    are persisted -- see migrations/0003_..._stripe_events.sql)."""
    cur = _execute(
        conn,
        "SELECT stripe_event_id, event_type, received_at, processed_at, "
        "processing_status, error_message FROM stripe_events "
        "ORDER BY received_at DESC LIMIT ?",
        (limit,),
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_stripe_event_summary(conn) -> dict:
    """Step 14 Phase 5: 'how many events received/succeeded/failed,
    last successful, last failed' -- one small, cheap read for an
    operational dashboard. Counts are over the full stripe_events
    history (it's an append-only ledger, never pruned by this app)."""
    cur = _execute(
        conn, "SELECT processing_status, COUNT(*) FROM stripe_events GROUP BY processing_status"
    )
    counts = {row[0]: row[1] for row in cur.fetchall()}

    cur = _execute(
        conn,
        "SELECT stripe_event_id, event_type, processed_at FROM stripe_events "
        "WHERE processing_status = ? ORDER BY processed_at DESC LIMIT 1",
        ("processed",),
    )
    last_processed = cur.fetchone()

    cur = _execute(
        conn,
        "SELECT stripe_event_id, event_type, processed_at, error_message FROM stripe_events "
        "WHERE processing_status = ? ORDER BY processed_at DESC LIMIT 1",
        ("failed",),
    )
    last_failed = cur.fetchone()

    return {
        "received_count": sum(counts.values()),
        "processed_count": counts.get("processed", 0),
        "failed_count": counts.get("failed", 0),
        "pending_count": counts.get("received", 0),
        "last_processed_event_id": last_processed[0] if last_processed else None,
        "last_processed_event_type": last_processed[1] if last_processed else None,
        "last_processed_at": last_processed[2] if last_processed else None,
        "last_failed_event_id": last_failed[0] if last_failed else None,
        "last_failed_event_type": last_failed[1] if last_failed else None,
        "last_failed_at": last_failed[2] if last_failed else None,
        "last_failed_error_summary": (
            str(last_failed[3])[:200] if last_failed and last_failed[3] else None
        ),
    }


# ---------------------------------------------------------------------------
# entitlement_overrides
# ---------------------------------------------------------------------------

def get_active_override(conn, user_id: int, *, now: str | None = None):
    """The single enabled, non-expired override for this user (if any).
    expires_at IS NULL means permanent. String comparison works for both
    Postgres TIMESTAMPTZ (cast to text on read by the driver only when
    compared as text -- but here we always compare in Python against ISO
    8601 UTC strings, which sort/compare correctly lexicographically)."""
    now = now or _now_iso()
    cur = _execute(
        conn,
        "SELECT * FROM entitlement_overrides WHERE user_id = ? AND enabled = ? "
        "ORDER BY id DESC LIMIT 1",
        (user_id, True),
    )
    row = _row_to_dict(cur, cur.fetchone())
    if row is None:
        return None
    expires_at = row.get("expires_at")
    if expires_at is not None and str(expires_at) < now:
        return None
    return row


def list_overrides_for_user(conn, user_id: int):
    cur = _execute(
        conn,
        "SELECT * FROM entitlement_overrides WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def create_override(
    conn,
    *,
    user_id: int,
    override_tier: str,
    reason: str | None,
    created_by: str | None,
    expires_at: str | None = None,
):
    """
    Disables any currently-enabled override for this user, then inserts
    the new one -- at most one ENABLED override per user at a time (an
    application-level invariant, see the migration's docstring). Returns
    the new override's id.
    """
    _execute(
        conn,
        "UPDATE entitlement_overrides SET enabled = ? WHERE user_id = ? AND enabled = ?",
        (False, user_id, True),
    )
    override_id = _insert_and_get_id(
        conn,
        "INSERT INTO entitlement_overrides (user_id, override_tier, enabled, "
        "reason, expires_at, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, override_tier, True, reason, expires_at, created_by, _now_iso()),
    )
    conn.commit()
    return override_id


def revoke_override(conn, override_id: int) -> None:
    _execute(
        conn,
        "UPDATE entitlement_overrides SET enabled = ? WHERE id = ?",
        (False, override_id),
    )
    conn.commit()
