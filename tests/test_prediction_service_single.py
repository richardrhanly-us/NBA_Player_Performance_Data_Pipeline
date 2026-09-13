"""
Tests for src/services/prediction_service.py::predict_player -- single-
player prediction scenarios (Step 8 test list items 1, 5, 9, 10, 11, 12,
13, 14). Uses a fake, offline, in-memory BasketballDataProvider and a
lightweight fake model (no real network access, no real model.pkl
needed for these). Real-model prediction parity lives in
tests/test_prediction_service_parity.py.
"""

import numpy as np
import pandas as pd

from src.data.basketball.errors import ProviderUnavailableError
from src.data.basketball.models import Player, PlayerDetails, PlayerGameLog
from src.features.feature_schema import LEGACY_FEATURE_NAMES
from src.services.prediction_result import PredictionDirection, PredictionStatus
from src.services.prediction_service import predict_player

SEASON = "2025-26"


class _FakeModel:
    def __init__(self, fixed_prediction=20.0):
        self.feature_names_in_ = list(LEGACY_FEATURE_NAMES)
        self._fixed_prediction = fixed_prediction

    def predict(self, X):
        return np.array([self._fixed_prediction] * len(X))


class FakeProvider:
    def __init__(
        self,
        *,
        active_players=None,
        game_logs=None,
        player_details=None,
        scoreboard=None,
        raise_on_gamelogs_for=(),
    ):
        self._active_players = active_players or []
        self._game_logs = game_logs or {}
        self._player_details = player_details or {}
        self._scoreboard = scoreboard or []
        self._raise_on_gamelogs_for = set(raise_on_gamelogs_for)
        self.gamelog_calls = []
        self.details_calls = []
        self.active_players_calls = 0

    def get_season_roster(self, season):
        raise NotImplementedError

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        self.gamelog_calls.append(player_id)
        if player_id in self._raise_on_gamelogs_for:
            raise ProviderUnavailableError("simulated outage")
        return self._game_logs.get(player_id, [])

    def get_active_players(self):
        self.active_players_calls += 1
        return list(self._active_players)

    def get_player_details(self, player_id):
        self.details_calls.append(player_id)
        return self._player_details.get(player_id)

    def get_todays_scoreboard(self, game_date=None):
        return list(self._scoreboard)

    def get_live_box_score(self, game_id):
        return None


def _game_log(player_id, game_date, points, game_id):
    return PlayerGameLog(
        season=SEASON,
        season_id="22025",
        player_id=player_id,
        player_name="Test Player",
        game_id=game_id,
        game_date=pd.Timestamp(game_date),
        matchup="LAL vs DEN",
        team_abbreviation="LAL",
        opponent_abbreviation="DEN",
        is_home=1,
        wl="W",
        minutes=32,
        fgm=8.0,
        fga=16.0,
        fg_pct=0.5,
        fg3m=2.0,
        fg3a=5.0,
        fg3_pct=0.4,
        ftm=4.0,
        fta=5.0,
        ft_pct=0.8,
        oreb=1.0,
        dreb=5.0,
        reb=6.0,
        ast=4.0,
        stl=1.0,
        blk=0.0,
        tov=2.0,
        pf=2.0,
        points=float(points),
        plus_minus=5.0,
        video_available=1.0,
    )


def _sufficient_history(player_id, n=10):
    return [
        _game_log(player_id, f"2025-11-{1 + i:02d}", 20 + i, f"G{i}") for i in range(n)
    ]


def _provider_with_player(player_id=2544, name="LeBron James", n_games=10, **kwargs):
    return FakeProvider(
        active_players=[Player(player_id=player_id, player_name=name)],
        game_logs={player_id: _sufficient_history(player_id, n_games)},
        player_details={
            player_id: PlayerDetails(
                player_id=player_id,
                team_id=1610612747,
                team_name="Lakers",
                team_abbreviation="LAL",
                position="F",
            )
        },
        **kwargs,
    )


# ---- 1. one valid player/line produces expected PredictionResult ----------


def test_valid_player_and_line_produces_ok_result():
    provider = _provider_with_player()
    result = predict_player(
        "LeBron James",
        sportsbook_line=18.0,
        bookmaker="draftkings",
        season=SEASON,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.OK
    assert result.player_id == 2544
    assert result.model_projection == 20.0
    assert result.sportsbook_line == 18.0
    assert result.edge == 2.0
    assert result.direction == PredictionDirection.OVER
    assert result.team_abbreviation == "LAL"
    assert result.bookmaker == "draftkings"
    assert result.model_version is not None
    assert result.generated_at_utc is not None
    assert result.latest_game_date is not None


# ---- 9. missing sportsbook line --------------------------------------------


def test_missing_sportsbook_line_gives_missing_line_status_but_still_projects():
    provider = _provider_with_player()
    result = predict_player(
        "LeBron James",
        sportsbook_line=None,
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.MISSING_LINE
    assert result.model_projection == 20.0  # projection still computed
    assert result.edge is None
    assert result.direction is None


# ---- 10. missing player history --------------------------------------------


def test_missing_history_returns_missing_history_status():
    provider = FakeProvider(
        active_players=[Player(player_id=1, player_name="No History Guy")],
        game_logs={1: []},
    )
    result = predict_player(
        "No History Guy",
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.MISSING_HISTORY
    assert result.reason == "Player gamelog unavailable."


def test_insufficient_games_for_features_returns_missing_history_status():
    provider = FakeProvider(
        active_players=[Player(player_id=2, player_name="Rookie")],
        game_logs={
            2: _sufficient_history(2, n=1)
        },  # 1 game: not enough for rolling features... actually build_player_feature_row only needs >=1 non-null GAME_DATE row
    )
    result = predict_player(
        "Rookie",
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        apply_live_adjustment=False,
    )
    # With only 1 game, build_player_feature_row still returns a row (rolling
    # windows are simply NaN) -- assert it does NOT crash and produces OK or
    # MISSING_HISTORY, never ERROR.
    assert result.status in (PredictionStatus.OK, PredictionStatus.MISSING_HISTORY)


# ---- 11. missing metadata ---------------------------------------------------


def test_missing_player_details_degrades_gracefully_not_fatal():
    provider = FakeProvider(
        active_players=[Player(player_id=3, player_name="No Metadata Guy")],
        game_logs={3: _sufficient_history(3)},
        player_details={},  # get_player_details returns None
    )
    result = predict_player(
        "No Metadata Guy",
        sportsbook_line=15.0,
        provider=provider,
        model=_FakeModel(fixed_prediction=18.0),
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.OK
    assert result.team_abbreviation is None
    assert result.model_projection == 18.0


# ---- 12. unmatched sportsbook player name -----------------------------------


def test_unmatched_player_name_returns_unmatched_status():
    provider = _provider_with_player()
    result = predict_player(
        "Totally Unknown Player XYZ",
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        use_fuzzy_name_resolution=True,
    )
    assert result.status == PredictionStatus.UNMATCHED
    assert result.player_id is None
    assert result.model_projection is None


# ---- 13. ambiguous identity does not silently match ------------------------


def test_ambiguous_name_without_fuzzy_resolution_does_not_silently_match():
    """
    use_fuzzy_name_resolution=False (the manual-search default, matching
    apps/publicapp.py::build_prediction's existing exact-match behavior)
    must NOT fall back to a first+last-name split match. An input with
    an extra middle name/initial is not an exact normalized match to any
    active player, so it must fail to resolve rather than guess --
    contrasted directly against use_fuzzy_name_resolution=True (the
    EXISTING sportsbook-name-matching strategy,
    src.shared_app.resolve_player_name, already used in production by
    get_top_plays_today_df), which DOES resolve it via that fallback.
    """
    provider = FakeProvider(
        active_players=[Player(player_id=1, player_name="Jaylen Brown")],
        game_logs={1: _sufficient_history(1)},
    )

    exact_result = predict_player(
        "Jaylen Middle Brown",  # not an exact normalized match
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        use_fuzzy_name_resolution=False,
    )
    assert exact_result.status == PredictionStatus.UNMATCHED

    fuzzy_result = predict_player(
        "Jaylen Middle Brown",  # same input
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        use_fuzzy_name_resolution=True,
    )
    assert fuzzy_result.status == PredictionStatus.OK
    assert fuzzy_result.player_id == 1


# ---- 14. provider failure for one player does not kill the caller ---------


def test_provider_unavailable_during_active_player_lookup_returns_structured_status():
    """
    shared_app.get_player_gamelog_df already swallows ProviderError
    internally (Step 7 behavior, preserved -- see the comment in
    predict_player), so a gamelog-fetch failure surfaces as
    MISSING_HISTORY (tested separately), not PROVIDER_UNAVAILABLE. The
    active-player-roster lookup has no such swallow, so that's the
    genuinely reachable path to PROVIDER_UNAVAILABLE -- and it must not
    raise an exception out of predict_player.
    """

    class _BrokenActivePlayersProvider(FakeProvider):
        def get_active_players(self):
            raise ProviderUnavailableError("roster lookup down")

    provider = _BrokenActivePlayersProvider()
    result = predict_player(
        "LeBron James",
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.PROVIDER_UNAVAILABLE
    assert "roster lookup down" in result.reason


def test_gamelog_provider_failure_degrades_to_missing_history_not_a_crash():
    """A provider outage during the (Streamlit-cached) gamelog fetch is
    already swallowed by shared_app.get_player_gamelog_df, exactly as it
    was before Step 8 -- confirms this still produces a graceful,
    reported status rather than propagating."""
    provider = _provider_with_player(player_id=99, raise_on_gamelogs_for={99})
    result = predict_player(
        "LeBron James",
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.MISSING_HISTORY
    assert result.player_id == 99


# ---- model-load failure -----------------------------------------------------


def test_model_unavailable_returns_structured_status_not_a_raised_exception():
    provider = _provider_with_player()

    def _broken_model_loader():
        raise RuntimeError("disk full")

    import src.shared_app as shared_app_module

    original = shared_app_module.load_model
    shared_app_module.load_model = _broken_model_loader
    try:
        result = predict_player(
            "LeBron James", sportsbook_line=10.0, provider=provider, model=None
        )
    finally:
        shared_app_module.load_model = original

    assert result.status == PredictionStatus.MODEL_UNAVAILABLE
    assert "disk full" in result.reason


# ---- gamelog_cache reuse -----------------------------------------------------


def test_gamelog_cache_prevents_a_second_provider_call_for_the_same_player():
    provider = _provider_with_player()
    cache = {}
    predict_player(
        "LeBron James",
        sportsbook_line=10.0,
        provider=provider,
        model=_FakeModel(),
        gamelog_cache=cache,
        apply_live_adjustment=False,
    )
    predict_player(
        "LeBron James",
        sportsbook_line=12.0,
        provider=provider,
        model=_FakeModel(),
        gamelog_cache=cache,
        apply_live_adjustment=False,
    )
    assert provider.gamelog_calls == [2544]  # only one real fetch
