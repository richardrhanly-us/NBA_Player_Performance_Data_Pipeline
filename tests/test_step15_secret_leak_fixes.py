"""
Step 15 Phase P finding: requests.Response.raise_for_status() (and
connection-level failures) embed the full request URL -- including
query-string secrets -- in the exception's own message. Verified
empirically against a real HTTP round trip during Step 15's audit:

    requests.get(url, params={"apiKey": "SECRET"}) -> raise_for_status()
    str(exception) == "... for url: https://...?apiKey=SECRET"

src/shared_app.py's odds-API fetchers pass ODDS_API_KEY exactly this
way, and apps/publicapp.py displays the resulting exception's str()
directly to anonymous public users on failure
(`st.warning(f"Could not load sportsbook line: {e}")`). These tests
prove the fix (src/shared_app.py::fetch_upcoming_nba_events /
fetch_player_points_market now catch requests.RequestException and
re-raise a generic, secret-free RuntimeError) actually holds, using a
fake `requests.get` -- no real network access.
"""

from __future__ import annotations

import pytest
import requests

from src import shared_app

_FAKE_API_KEY = "leak-check-odds-api-key-should-never-appear"


class _FakeResponse:
    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: rejected for url: {self.url}",
                response=self,
            )

    def json(self):
        return {}


def test_fetch_upcoming_nba_events_sanitizes_http_error(monkeypatch):
    def fake_get(url, params, timeout):
        full_url = f"{url}?apiKey={params['apiKey']}"
        return _FakeResponse(401, full_url)

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        shared_app.fetch_upcoming_nba_events(_FAKE_API_KEY)

    assert _FAKE_API_KEY not in str(excinfo.value)
    assert "odds data provider" in str(excinfo.value)


def test_fetch_upcoming_nba_events_sanitizes_connection_error(monkeypatch):
    def fake_get(url, params, timeout):
        raise requests.ConnectionError(
            f"Max retries exceeded with url: {url}?apiKey={params['apiKey']}"
        )

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        shared_app.fetch_upcoming_nba_events(_FAKE_API_KEY)

    assert _FAKE_API_KEY not in str(excinfo.value)


def test_fetch_player_points_market_sanitizes_http_error(monkeypatch):
    def fake_get(url, params, timeout):
        full_url = f"{url}?apiKey={params['apiKey']}"
        return _FakeResponse(429, full_url)

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        shared_app.fetch_player_points_market(_FAKE_API_KEY, "evt_1", "draftkings")

    assert _FAKE_API_KEY not in str(excinfo.value)
    assert "odds data provider" in str(excinfo.value)


def test_fetch_all_today_player_props_never_leaks_key_when_events_call_fails(monkeypatch):
    """The end-to-end path a real page load exercises: events fetch
    fails -> fetch_all_today_player_props propagates the (now sanitized)
    error uncaught, exactly matching src/shared_app.py's existing
    documented behavior (only the per-event odds fetch is individually
    swallowed, not the events fetch itself)."""

    def fake_get(url, params, timeout):
        full_url = f"{url}?apiKey={params['apiKey']}"
        return _FakeResponse(401, full_url)

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        shared_app.fetch_all_today_player_props(_FAKE_API_KEY, "draftkings")

    assert _FAKE_API_KEY not in str(excinfo.value)


def test_get_player_points_lines_propagates_sanitized_error_only(monkeypatch):
    """The exact call path behind apps/publicapp.py's
    'Could not load sportsbook line: {e}' message."""
    monkeypatch.setattr(shared_app, "get_odds_api_key", lambda: _FAKE_API_KEY)

    def fake_get(url, params, timeout):
        full_url = f"{url}?apiKey={params['apiKey']}"
        return _FakeResponse(500, full_url)

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        shared_app.get_player_points_lines("LeBron James", "draftkings")

    assert _FAKE_API_KEY not in str(excinfo.value)


def test_get_today_games_still_degrades_gracefully_on_sanitized_error(monkeypatch):
    """get_today_games() already caught Exception broadly and returned
    [] -- confirms the sanitization doesn't change that pre-existing
    graceful-degradation behavior."""
    monkeypatch.setattr(shared_app.get_today_games, "clear", lambda: None, raising=False)

    def fake_get(url, params, timeout):
        full_url = f"{url}?apiKey={params['apiKey']}"
        return _FakeResponse(401, full_url)

    monkeypatch.setattr(requests, "get", fake_get)

    result = shared_app.get_today_games(_FAKE_API_KEY)
    assert result == []
