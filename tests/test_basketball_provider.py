"""
Tests for src/data/basketball/provider.py and
src/data/basketball/providers/nba_api_provider.py:
- the NBA development provider structurally satisfies BasketballDataProvider
- get_basketball_provider() returns it by default
- NBAApiProvider correctly wraps training.data.nba_client (mocked) and
  training.data.storage, converting failures into ProviderUnavailableError
No network access -- nba_client's own endpoint classes are mocked.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.basketball.errors import ProviderUnavailableError
from src.data.basketball.models import Player, PlayerGameLog
from src.data.basketball.provider import BasketballDataProvider, get_basketball_provider
from src.data.basketball.providers.nba_api_provider import NBAApiProvider
from training.data import nba_client


def test_nba_api_provider_satisfies_the_protocol():
    provider = NBAApiProvider()
    assert isinstance(provider, BasketballDataProvider)
    assert hasattr(provider, "get_season_roster")
    assert hasattr(provider, "get_player_game_logs")


def test_get_basketball_provider_returns_an_nba_api_provider_by_default():
    provider = get_basketball_provider()
    assert isinstance(provider, NBAApiProvider)


def test_get_basketball_provider_returns_the_same_instance_on_repeat_calls():
    assert get_basketball_provider() is get_basketball_provider()


def _roster_df():
    return pd.DataFrame(
        {"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Alpha", "Beta"], "GP": [10, 20]}
    )


def _raw_gamelog_df(player_id):
    return pd.DataFrame(
        {
            "SEASON_ID": ["22023"],
            "Player_ID": [player_id],
            "Game_ID": ["0022300001"],
            "GAME_DATE": ["2023-10-24"],
            "MATCHUP": ["BOS vs MIA"],
            "WL": ["W"],
            "MIN": ["30"],
            "FGM": [8],
            "FGA": [16],
            "FG_PCT": [0.5],
            "FG3M": [2],
            "FG3A": [5],
            "FG3_PCT": [0.4],
            "FTM": [4],
            "FTA": [5],
            "FT_PCT": [0.8],
            "OREB": [1],
            "DREB": [5],
            "REB": [6],
            "AST": [4],
            "STL": [1],
            "BLK": [0],
            "TOV": [2],
            "PF": [2],
            "PTS": [22],
            "PLUS_MINUS": [5],
            "VIDEO_AVAILABLE": [1],
        }
    )


def test_get_season_roster_wraps_nba_client_and_returns_canonical_players():
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [_roster_df()]

    with patch(
        "training.data.nba_client.leaguedashplayerstats.LeagueDashPlayerStats",
        return_value=fake_response,
    ):
        provider = NBAApiProvider(sleep_func=lambda *_: None)
        result = provider.get_season_roster("2023-24")

    assert result == [
        Player(player_id=1, player_name="Alpha"),
        Player(player_id=2, player_name="Beta"),
    ]


def test_get_player_game_logs_wraps_nba_client_and_returns_canonical_records():
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [_raw_gamelog_df(2544)]

    with patch(
        "training.data.nba_client.playergamelog.PlayerGameLog",
        return_value=fake_response,
    ):
        provider = NBAApiProvider(sleep_func=lambda *_: None)
        result = provider.get_player_game_logs(
            2544, "2023-24", player_name="LeBron James"
        )

    assert len(result) == 1
    record = result[0]
    assert isinstance(record, PlayerGameLog)
    assert record.player_id == 2544
    assert record.player_name == "LeBron James"
    assert record.team_abbreviation == "BOS"
    assert record.opponent_abbreviation == "MIA"
    assert record.is_home == 1
    assert record.points == 22


def test_get_season_roster_raises_provider_unavailable_after_nba_client_exhausts_retries():
    with patch(
        "training.data.nba_client.leaguedashplayerstats.LeagueDashPlayerStats",
        side_effect=ConnectionError("down"),
    ):
        provider = NBAApiProvider(sleep_func=lambda *_: None, rand_func=lambda: 0.0)
        with pytest.raises(ProviderUnavailableError):
            provider.get_season_roster("2023-24")


def test_get_player_game_logs_raises_provider_unavailable_after_nba_client_exhausts_retries():
    with patch(
        "training.data.nba_client.playergamelog.PlayerGameLog",
        side_effect=ConnectionError("down"),
    ):
        provider = NBAApiProvider(sleep_func=lambda *_: None, rand_func=lambda: 0.0)
        with pytest.raises(ProviderUnavailableError):
            provider.get_player_game_logs(2544, "2023-24", player_name="LeBron James")


def test_provider_unavailable_error_chains_the_underlying_nba_api_error():
    with patch(
        "training.data.nba_client.playergamelog.PlayerGameLog",
        side_effect=ConnectionError("down"),
    ):
        provider = NBAApiProvider(sleep_func=lambda *_: None, rand_func=lambda: 0.0)
        with pytest.raises(ProviderUnavailableError) as exc_info:
            provider.get_player_game_logs(2544, "2023-24")

    assert isinstance(exc_info.value.__cause__, nba_client.NbaApiError)
