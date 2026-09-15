"""
Step 15: real end-to-end acceptance tests for apps/publicapp.py and
apps/adminapp.py, using Streamlit's own headless testing API
(streamlit.testing.v1.AppTest) instead of static source-text analysis.

Previously (Steps 11-14) these two apps were only ever validated by
static guards (AST/text scanning) and by unit-testing the service layer
underneath them -- nothing actually *executed* either script end to end.
AppTest runs the real file through Streamlit's script runner and exposes
the resulting element tree, so these tests catch real runtime defects
(a crash, a rendering failure, an entitlement-gate that doesn't actually
gate) that static analysis cannot.

Fully offline: DEV_AUTH_ENABLED=true (no Supabase network), DATABASE_URL
points at an in-memory SQLite connection (no Postgres network), and
Google Sheets/odds-API calls are simply absent from the environment,
which the apps already handle by degrading gracefully (see the
'[ERROR] ... credentials not found' messages these apps print to stdout
-- pre-existing, intentional behavior, not a Step 15 change).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.services import accounts_repository as repo
from src.services.auth_providers import compute_dev_auth_subject
from src.services.prediction_repository import persist_prediction_run
from src.services.prediction_result import PredictionDirection, PredictionResult, PredictionStatus
from src.services.prediction_service import PredictionBoard
from src.services.schema_sqlite import create_sqlite_accounts_schema, create_sqlite_prediction_schema


_REPO_ROOT = Path(__file__).resolve().parent.parent

ADMIN_TAB_LABELS = ["Overview", "Operations", "Logs", "Usage", "Data Review", "Users & Access"]


@pytest.fixture(autouse=True)
def _clear_streamlit_caches(monkeypatch):
    """st.cache_data's store is process-global, not per-AppTest-instance
    -- without clearing it, get_automation_health() (cached ttl=30s in
    apps/adminapp.py) can silently return a PREVIOUS test's cached
    result. This is not an application defect (the cache is intentional,
    real-world production behavior across concurrent users of one server
    process); it's a test-isolation requirement specific to running many
    independent AppTest sessions back-to-back in one pytest process.

    Also stubs get_scoreboard_for_date() -- both apps call it (a live
    NBA API request, see src/data/basketball/providers/nba_api_provider.py)
    for "are there games today" freshness checks, independent of whether
    the DB call in the same code path succeeds or fails. Stubbed
    unconditionally here so this offline test suite never depends on
    NBA API availability (Step 15 testing policy), regardless of which
    code path a given test happens to exercise.
    """
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr("src.shared_app.get_scoreboard_for_date", lambda: [])
    yield


class _NonClosingConn:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


def _full_schema_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    create_sqlite_accounts_schema(conn)
    create_sqlite_prediction_schema(conn)
    return conn


def _wire_dev_auth(monkeypatch):
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


def _wire_database(monkeypatch, conn):
    monkeypatch.setenv("DATABASE_URL", "sqlite://in-memory-apptest")
    monkeypatch.setattr(
        "src.services.db_connection.get_prediction_db_connection",
        lambda: _NonClosingConn(conn),
    )


def _grant_admin(conn, email: str) -> dict:
    subject = compute_dev_auth_subject(email)
    user = repo.get_or_create_user_by_auth_subject(
        conn, auth_provider="dev", auth_subject=subject, email=email
    )
    repo.create_override(
        conn, user_id=user["id"], override_tier="ADMIN", reason="test", created_by="tester"
    )
    return user


def _sign_in(at: AppTest, email: str, password: str = "anything") -> None:
    at.text_input(key="signin_email").set_value(email)
    at.text_input(key="signin_password").set_value(password)
    at.button(key="signin_submit").click()
    at.run()


# ---------------------------------------------------------------------------
# Public app
# ---------------------------------------------------------------------------

def test_public_app_boots_with_no_backend_configured(monkeypatch):
    """No DATABASE_URL, no ODDS_API_KEY -- the app must render without
    raising, and degrade with clear, safe messages (not silence, not a
    crash)."""
    _wire_dev_auth(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    at = AppTest.from_file(str(_REPO_ROOT / "apps/publicapp.py"), default_timeout=45)
    at.run()

    assert len(at.exception) == 0
    info_text = "\n".join(str(i.value) for i in at.info)
    assert "Edge board unavailable" in info_text
    assert "Prediction history is unavailable" in info_text


def test_public_app_edge_board_degrades_safely_when_odds_api_fails(monkeypatch):
    """Step 15 Phase P finding: the Edge Board's own props fetch
    (shared_app.fetch_all_today_player_props, used by
    build_daily_prediction_board) previously let a raw requests
    exception -- with the ODDS_API_KEY embedded in its message via
    raise_for_status() -- propagate straight to
    `st.info(f"Edge board is temporarily unavailable: {e}")`, visible to
    anonymous public users. Verified end-to-end through the real page
    with the actual fix (src/shared_app.py::fetch_upcoming_nba_events)
    in place."""
    _wire_dev_auth(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "leak-check-odds-key-should-not-appear")

    import requests

    class _FakeResponse:
        status_code = 401
        url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey=leak-check-odds-key-should-not-appear"

        def raise_for_status(self):
            raise requests.HTTPError(f"401 Client Error: rejected for url: {self.url}")

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())

    at = AppTest.from_file(str(_REPO_ROOT / "apps/publicapp.py"), default_timeout=45)
    at.run()

    assert len(at.exception) == 0
    rendered_text = "\n".join(str(i.value) for i in list(at.info) + list(at.warning) + list(at.error))
    assert "leak-check-odds-key-should-not-appear" not in rendered_text


def test_public_app_degrades_safely_when_database_is_unreachable(monkeypatch):
    """Step 14's DSN-leak fix (src/services/db_connection.py), verified
    end-to-end through the real page: a genuinely unreachable database
    (bad host, embedded credentials) must never surface the DSN/password
    to the rendered page."""
    _wire_dev_auth(monkeypatch)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://baduser:leak-check-password@nonexistent-host-xyz123.invalid:5432/db",
    )

    at = AppTest.from_file(str(_REPO_ROOT / "apps/publicapp.py"), default_timeout=45)
    at.run()

    assert len(at.exception) == 0
    rendered_text = "\n".join(str(i.value) for i in list(at.info) + list(at.warning) + list(at.error))
    assert "leak-check-password" not in rendered_text
    assert "nonexistent-host-xyz123" not in rendered_text
    assert "Could not connect to the prediction-history database" in rendered_text


# ---------------------------------------------------------------------------
# Admin app
# ---------------------------------------------------------------------------

def test_admin_app_denies_unauthenticated_visitor(monkeypatch):
    _wire_dev_auth(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LEGACY_ADMIN_KEY_ENABLED", raising=False)

    at = AppTest.from_file(str(_REPO_ROOT / "apps/adminapp.py"), default_timeout=45)
    at.run()

    assert len(at.exception) == 0
    assert [t.label for t in at.tabs if t.label in ADMIN_TAB_LABELS] == []
    full_text = "\n".join(str(m.value) for m in at.markdown)
    assert "Sign in with an admin account" in full_text


def test_admin_app_denies_signed_in_non_admin_user(monkeypatch):
    """A real dev-auth sign-in succeeds (there is no DB, so it's a
    transient FREE session) but the admin tabs must still not appear --
    authorization is a separate, server-side check from authentication."""
    _wire_dev_auth(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    at = AppTest.from_file(str(_REPO_ROOT / "apps/adminapp.py"), default_timeout=45)
    at.run()
    _sign_in(at, "non-admin-user@example.com")

    assert len(at.exception) == 0
    assert [t.label for t in at.tabs if t.label in ADMIN_TAB_LABELS] == []


def test_admin_app_grants_access_for_admin_override_and_shows_all_tabs(monkeypatch):
    conn = _full_schema_conn()
    _wire_dev_auth(monkeypatch)
    _wire_database(monkeypatch, conn)
    _grant_admin(conn, "admin-user@example.com")

    at = AppTest.from_file(str(_REPO_ROOT / "apps/adminapp.py"), default_timeout=45)
    at.run()
    _sign_in(at, "admin-user@example.com")

    assert len(at.exception) == 0
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Operations",
        "Logs",
        "Usage",
        "Data Review",
        "Users & Access",
    ]


def test_admin_app_shows_latest_failed_run_distinct_from_latest_success(monkeypatch):
    """The Step 14 fix: a FAILED run must remain visible even though an
    older SUCCESS run also exists -- this is the specific regression
    Step 15 was asked to re-verify live."""
    conn = _full_schema_conn()
    _wire_dev_auth(monkeypatch)
    _wire_database(monkeypatch, conn)
    _grant_admin(conn, "admin-user@example.com")

    def _result(name, status=PredictionStatus.OK, **overrides):
        defaults = dict(
            player_name=name, status=status, player_id=hash(name) % 100000,
            model_projection=22.5, sportsbook_line=20.0, edge=2.5,
            direction=PredictionDirection.OVER, bookmaker="draftkings",
            model_version="v1", generated_at_utc="2026-01-15T12:00:00+00:00",
            game_id="G1", game_date="01/15/2026",
        )
        defaults.update(overrides)
        return PredictionResult(**defaults)

    def _board(preds, **overrides):
        defaults = dict(
            generated_at_utc="2026-01-15T12:00:00+00:00", bookmaker="draftkings",
            model_version="v1", predictions=tuple(preds), props_discovered=len(preds),
            players_matched=len(preds),
            predictions_generated=sum(1 for p in preds if p.status == PredictionStatus.OK),
            unmatched_count=sum(1 for p in preds if p.status == PredictionStatus.UNMATCHED),
            unavailable_count=0,
        )
        defaults.update(overrides)
        return PredictionBoard(**defaults)

    persist_prediction_run(_board([_result("Player A")]), conn)
    persist_prediction_run(
        _board(
            [_result("Player B", status=PredictionStatus.UNMATCHED)],
            generated_at_utc="2026-01-15T18:00:00+00:00",
            predictions_generated=0,
            unmatched_count=1,
        ),
        conn,
    )

    at = AppTest.from_file(str(_REPO_ROOT / "apps/adminapp.py"), default_timeout=45)
    at.run()
    _sign_in(at, "admin-user@example.com")

    assert len(at.exception) == 0
    full_text = "\n".join(str(m.value) for m in at.markdown)
    assert "Latest failed run" in full_text
    assert "none" not in full_text.split("Latest failed run:")[1][:80].lower()


def test_admin_app_billing_health_section_renders(monkeypatch):
    conn = _full_schema_conn()
    _wire_dev_auth(monkeypatch)
    _wire_database(monkeypatch, conn)
    _grant_admin(conn, "admin-user@example.com")

    repo.try_claim_stripe_event(conn, "evt_1", "checkout.session.completed")
    repo.mark_stripe_event_processed(conn, "evt_1")
    repo.try_claim_stripe_event(conn, "evt_2", "customer.subscription.updated")
    repo.mark_stripe_event_failed(conn, "evt_2", "could not resolve user")

    at = AppTest.from_file(str(_REPO_ROOT / "apps/adminapp.py"), default_timeout=45)
    at.run()
    _sign_in(at, "admin-user@example.com")

    assert len(at.exception) == 0
    full_text = "\n".join(str(m.value) for m in at.markdown)
    assert "Billing Health" in full_text


def test_admin_app_legacy_key_bypasses_gate_without_leaking_secrets_when_db_unreachable(monkeypatch):
    _wire_dev_auth(monkeypatch)
    monkeypatch.setenv("LEGACY_ADMIN_KEY_ENABLED", "true")
    monkeypatch.setenv("ADMIN_KEY", "leak-check-legacy-key")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://baduser:leak-check-password@nonexistent-host-xyz123.invalid:5432/db",
    )

    at = AppTest.from_file(str(_REPO_ROOT / "apps/adminapp.py"), default_timeout=45)
    at.run()
    at.text_input(key="admin_key_input").set_value("leak-check-legacy-key")
    at.run()

    assert len(at.exception) == 0
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Operations",
        "Logs",
        "Usage",
        "Data Review",
        "Users & Access",
    ]
    full_text = "\n".join(str(m.value) for m in at.markdown)
    assert "leak-check-password" not in full_text
    assert "nonexistent-host-xyz123" not in full_text
    assert "leak-check-legacy-key" not in full_text
