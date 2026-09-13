"""
nba_client tests: retry/backoff behavior and the two fetch functions, with
the underlying nba_api endpoint classes mocked. No network access.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from training import config
from training.data import nba_client


def _fake_sleep_tracker():
    """Returns (sleep_func, calls) where calls records every delay passed in,
    without actually sleeping -- keeps retry/backoff tests instant."""
    calls = []

    def _sleep(delay):
        calls.append(delay)

    return _sleep, calls


def test_call_with_retries_returns_result_on_first_success():
    sleep_func, calls = _fake_sleep_tracker()
    func = MagicMock(return_value="ok")

    result = nba_client.call_with_retries(
        func, description="test call", sleep_func=sleep_func
    )

    assert result == "ok"
    assert func.call_count == 1
    assert calls == []


def test_call_with_retries_recovers_after_transient_failures():
    sleep_func, calls = _fake_sleep_tracker()
    func = MagicMock(
        side_effect=[ConnectionError("boom"), TimeoutError("boom again"), "ok"]
    )

    result = nba_client.call_with_retries(
        func, description="test call", sleep_func=sleep_func, rand_func=lambda: 0.0
    )

    assert result == "ok"
    assert func.call_count == 3
    # Two failures before the eventual success -> two backoff sleeps.
    assert len(calls) == 2
    # Backoff should not shrink between successive attempts.
    assert calls[1] >= calls[0]


def test_call_with_retries_raises_nba_api_error_after_exhausting_attempts():
    sleep_func, calls = _fake_sleep_tracker()
    func = MagicMock(side_effect=RuntimeError("persistent failure"))

    with pytest.raises(nba_client.NbaApiError) as exc_info:
        nba_client.call_with_retries(
            func, description="test call", sleep_func=sleep_func, rand_func=lambda: 0.0
        )

    assert func.call_count == config.MAX_ATTEMPTS
    assert (
        len(calls) == config.MAX_ATTEMPTS - 1
    )  # no sleep after the final failed attempt
    assert "persistent failure" in str(exc_info.value)


def test_backoff_delay_grows_and_is_capped():
    delays = [
        nba_client._backoff_delay(attempt, rand_func=lambda: 0.0)
        for attempt in range(1, 8)
    ]

    assert delays == sorted(delays)  # non-decreasing
    assert all(d <= config.BACKOFF_MAX_SECONDS for d in delays)
    assert delays[0] == pytest.approx(config.BACKOFF_BASE_SECONDS)


def test_fetch_player_gamelog_returns_mocked_dataframe():
    expected_df = pd.DataFrame(
        {"Player_ID": [2544], "Game_ID": ["0022300001"], "PTS": [30]}
    )
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [expected_df]

    with patch(
        "training.data.nba_client.playergamelog.PlayerGameLog",
        return_value=fake_response,
    ) as mock_cls:
        result = nba_client.fetch_player_gamelog(
            2544, "2023-24", sleep_func=lambda *_: None
        )

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["player_id"] == 2544
    assert kwargs["season"] == "2023-24"
    assert kwargs["headers"] == config.NBA_STATS_HEADERS
    pd.testing.assert_frame_equal(result, expected_df)


def test_fetch_season_roster_filters_to_players_with_games_played():
    roster_df = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "PLAYER_NAME": ["Played A", "Never Played", "Played B"],
            "GP": [72, 0, 41],
        }
    )
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [roster_df]

    with patch(
        "training.data.nba_client.leaguedashplayerstats.LeagueDashPlayerStats",
        return_value=fake_response,
    ) as mock_cls:
        result = nba_client.fetch_season_roster("2023-24", sleep_func=lambda *_: None)

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["season"] == "2023-24"
    assert kwargs["season_type_all_star"] == "Regular Season"
    assert sorted(result["PLAYER_ID"]) == [1, 3]


def test_fetch_season_roster_drops_zero_gp_rows_defensively():
    # LeagueDashPlayerStats should never actually return a GP=0 row, but the
    # filter exists as a defensive check -- verify it does something if it did.
    roster_df = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "PLAYER_NAME": ["Played", "Zero Games"],
            "GP": [10, 0],
        }
    )
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [roster_df]

    with patch(
        "training.data.nba_client.leaguedashplayerstats.LeagueDashPlayerStats",
        return_value=fake_response,
    ):
        result = nba_client.fetch_season_roster("2023-24", sleep_func=lambda *_: None)

    assert list(result["PLAYER_ID"]) == [1]


def test_fetch_player_gamelog_retries_then_raises_nba_api_error():
    with (
        patch(
            "training.data.nba_client.playergamelog.PlayerGameLog",
            side_effect=ConnectionError("network down"),
        ),
        pytest.raises(nba_client.NbaApiError),
    ):
        nba_client.fetch_player_gamelog(
            2544, "2023-24", sleep_func=lambda *_: None, rand_func=lambda: 0.0
        )
