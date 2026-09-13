"""
Step 8 parity test (test list items #6, #20): proves
src.services.prediction_service.predict_player reproduces EXACTLY the
core computation apps/publicapp.py::build_prediction ran inline before
Step 8's refactor (resolve player -> recent gamelog -> legacy feature
row -> model.predict, pre-live-adjustment).

The "old path" below is a faithful, unmodified reconstruction of that
pre-Step-8 inline logic (see git history: commit 9102d5f's
apps/publicapp.py::build_prediction), used ONLY as a test fixture --
apps/publicapp.py itself cannot be imported directly (it is a top-level
Streamlit script), so this is how parity is proven offline.

Uses the ACTUAL deployed legacy model (models/points_regression.pkl, no
retraining) for genuine prediction parity, and a synthetic-but-realistic
gamelog fixture (matching real nba_api response shape) for determinism
and speed. Skips gracefully if the model artifact isn't present.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.basketball.models import Player, PlayerDetails, PlayerGameLog
from src.features.build_features import build_player_feature_row
from src.services.prediction_result import PredictionStatus
from src.services.prediction_service import predict_player

LEGACY_MODEL_PATH = "models/points_regression.pkl"
PLAYER_ID = 2544
PLAYER_NAME = "LeBron James"
SEASON = "2025-26"
SPORTSBOOK_LINE = 24.5


def _game_log(player_id, game_date, points, game_id, opponent):
    return PlayerGameLog(
        season=SEASON,
        season_id="22025",
        player_id=player_id,
        player_name=PLAYER_NAME,
        game_id=game_id,
        game_date=pd.Timestamp(game_date),
        matchup=f"LAL vs {opponent}",
        team_abbreviation="LAL",
        opponent_abbreviation=opponent,
        is_home=1,
        wl="W",
        minutes=float(30 + (points % 5)),
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


def _fixture_gamelogs(n=15):
    opponents = ["DEN", "PHX", "BOS", "MIA", "NYK"]
    return [
        _game_log(
            PLAYER_ID,
            f"2025-11-{1 + i:02d}",
            15 + (i * 3) % 20,
            f"G{i}",
            opponents[i % 5],
        )
        for i in range(n)
    ]


class FakeGamelogProvider:
    """Minimal BasketballDataProvider serving one fixed player's fixture
    gamelog -- used only so predict_player's internal
    shared_app.get_player_gamelog_df(..., _provider=...) call has
    something deterministic and offline to fetch."""

    def __init__(self, game_logs):
        self._game_logs = game_logs

    def get_season_roster(self, season):
        raise NotImplementedError

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        return list(self._game_logs)

    def get_active_players(self):
        return [Player(player_id=PLAYER_ID, player_name=PLAYER_NAME)]

    def get_player_details(self, player_id):
        return PlayerDetails(
            player_id=PLAYER_ID,
            team_id=1610612747,
            team_name="Lakers",
            team_abbreviation="LAL",
            position="F",
        )

    def get_todays_scoreboard(self, game_date=None):
        return []

    def get_live_box_score(self, game_id):
        return None


def _old_path_prediction(model, gamelog_df):
    """
    Faithful reconstruction of pre-Step-8 build_prediction's core logic
    (resolve -> feature row -> reindex -> predict), operating on the
    exact same fixture DataFrame the new path uses. Not "improved" or
    "cleaned up" in any way relative to the original.
    """
    X = build_player_feature_row(gamelog_df, PLAYER_NAME, SPORTSBOOK_LINE)
    assert X is not None and not X.empty

    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if model_feature_names:
        missing_features = [c for c in model_feature_names if c not in X.columns]
        assert not missing_features
        X = X.reindex(columns=model_feature_names)

    return float(model.predict(X)[0]), X


def _new_path_prediction(model):
    """The new service path, using a fake provider so this stays
    offline; provider-level fetching is what changed in Step 6/7 (and is
    separately parity-tested there) -- this test isolates the
    core-computation parity Step 8 is responsible for."""
    provider = FakeGamelogProvider(_fixture_gamelogs())
    result = predict_player(
        PLAYER_NAME,
        sportsbook_line=SPORTSBOOK_LINE,
        season=SEASON,
        provider=provider,
        model=model,
        apply_live_adjustment=False,
    )
    assert result.status == PredictionStatus.OK
    return result


def test_predict_player_matches_old_inline_logic_with_a_fake_model():
    from src.features.feature_schema import LEGACY_FEATURE_NAMES

    class _FakeModel:
        def __init__(self):
            self.feature_names_in_ = list(LEGACY_FEATURE_NAMES)

        def predict(self, X):
            return np.array([21.0] * len(X))

    model = _FakeModel()
    gamelog_df = pd.DataFrame(
        [
            {
                "SEASON": SEASON,
                "SEASON_ID": "22025",
                "PLAYER_ID": PLAYER_ID,
                "PLAYER_NAME": PLAYER_NAME,
                "GAME_ID": g.game_id,
                "GAME_DATE": g.game_date,
                "MATCHUP": g.matchup,
                "TEAM_ABBREVIATION": g.team_abbreviation,
                "OPPONENT_ABBREVIATION": g.opponent_abbreviation,
                "IS_HOME": g.is_home,
                "WL": g.wl,
                "MIN": g.minutes,
                "FGM": g.fgm,
                "FGA": g.fga,
                "FG_PCT": g.fg_pct,
                "FG3M": g.fg3m,
                "FG3A": g.fg3a,
                "FG3_PCT": g.fg3_pct,
                "FTM": g.ftm,
                "FTA": g.fta,
                "FT_PCT": g.ft_pct,
                "OREB": g.oreb,
                "DREB": g.dreb,
                "REB": g.reb,
                "AST": g.ast,
                "STL": g.stl,
                "BLK": g.blk,
                "TOV": g.tov,
                "PF": g.pf,
                "PTS": g.points,
                "PLUS_MINUS": g.plus_minus,
                "VIDEO_AVAILABLE": g.video_available,
            }
            for g in _fixture_gamelogs()
        ]
    )

    old_pred, old_X = _old_path_prediction(model, gamelog_df)
    new_result = _new_path_prediction(model)

    assert list(old_X.columns) == model.feature_names_in_
    assert new_result.model_projection == pytest.approx(old_pred)


@pytest.mark.skipif(
    not os.path.exists(LEGACY_MODEL_PATH),
    reason="models/points_regression.pkl not present in this checkout",
)
def test_predict_player_prediction_parity_with_real_deployed_model():
    """
    Rows compared: 1. Max absolute prediction difference reported via
    assertion -- this is the real, deployed model, unchanged, no
    retraining, exactly what apps/publicapp.py::build_prediction uses.
    """
    import joblib

    model = joblib.load(LEGACY_MODEL_PATH)

    gamelog_df = pd.DataFrame(
        [
            {
                "SEASON": SEASON,
                "SEASON_ID": "22025",
                "PLAYER_ID": PLAYER_ID,
                "PLAYER_NAME": PLAYER_NAME,
                "GAME_ID": g.game_id,
                "GAME_DATE": g.game_date,
                "MATCHUP": g.matchup,
                "TEAM_ABBREVIATION": g.team_abbreviation,
                "OPPONENT_ABBREVIATION": g.opponent_abbreviation,
                "IS_HOME": g.is_home,
                "WL": g.wl,
                "MIN": g.minutes,
                "FGM": g.fgm,
                "FGA": g.fga,
                "FG_PCT": g.fg_pct,
                "FG3M": g.fg3m,
                "FG3A": g.fg3a,
                "FG3_PCT": g.fg3_pct,
                "FTM": g.ftm,
                "FTA": g.fta,
                "FT_PCT": g.ft_pct,
                "OREB": g.oreb,
                "DREB": g.dreb,
                "REB": g.reb,
                "AST": g.ast,
                "STL": g.stl,
                "BLK": g.blk,
                "TOV": g.tov,
                "PF": g.pf,
                "PTS": g.points,
                "PLUS_MINUS": g.plus_minus,
                "VIDEO_AVAILABLE": g.video_available,
            }
            for g in _fixture_gamelogs()
        ]
    )

    old_pred, _ = _old_path_prediction(model, gamelog_df)
    new_result = _new_path_prediction(model)

    max_abs_prediction_diff = abs(old_pred - new_result.model_projection)
    assert max_abs_prediction_diff < 1e-6
