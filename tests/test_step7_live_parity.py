"""
Step 7 live parity tests: prove that swapping src/shared_app.py's live
gamelog fetch from "raw nba_api DataFrame straight into
build_player_feature_row" (the pre-Step-7 implementation) to
"NBAApiProvider -> canonical PlayerGameLog records ->
player_game_logs_to_dataframe -> build_player_feature_row" (the new
provider-boundary implementation) produces IDENTICAL LEGACY feature
values and IDENTICAL predictions from the actual deployed model
(models/points_regression.pkl).

NOTE: the live app (apps/publicapp.py::build_prediction) uses the LEGACY
feature builder (src/features/build_features.py, LEGACY_FEATURE_NAMES)
and the deployed legacy model -- NOT the V1 training pipeline/model.
This test targets exactly what the live path actually runs, rather than
the V1 pipeline the historical/training path uses (see Step 6/7
reports).

Synthetic-but-realistic multi-game fixture (not real player data) so
this runs fast and offline -- no network access. Uses unittest.mock.patch
on nba_client's own endpoint class, exactly like
tests/test_step6_provider_parity.py, so the "new path" genuinely
exercises NBAApiProvider end-to-end.
"""

from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.basketball.normalization import player_game_logs_to_dataframe
from src.data.basketball.providers.nba_api_provider import NBAApiProvider
from src.features.build_features import build_player_feature_row
from src.features.feature_schema import LEGACY_FEATURE_NAMES

PLAYER_ID = 2544
PLAYER_NAME = "LeBron James"
SEASON = "2025-26"
SPORTSBOOK_LINE = 24.5

LEGACY_MODEL_PATH = "models/points_regression.pkl"


def _raw_gamelog_df(n_games=12, seed=0):
    """nba_api-shaped raw PlayerGameLog rows (Player_ID/Game_ID casing),
    matching real nba_api MIN dtype (numeric, verified in Step 6) --
    enough games to populate every rolling-window legacy feature."""
    rng = np.random.RandomState(seed)
    opponents = ["DEN", "PHX", "BOS", "MIA", "NYK", "CHI"]
    rows = []
    for g in range(n_games):
        opponent = opponents[g % len(opponents)]
        is_home = g % 2 == 0
        matchup = f"LAL vs {opponent}" if is_home else f"LAL @ {opponent}"
        pts = float(rng.randint(10, 40))
        rows.append(
            {
                "SEASON_ID": "22025",
                "Player_ID": PLAYER_ID,
                "Game_ID": f"0022500{g:03d}",
                "GAME_DATE": f"2025-11-{1 + g:02d}",
                "MATCHUP": matchup,
                "WL": "W" if pts >= 20 else "L",
                "MIN": float(rng.randint(24, 40)),
                "FGM": float(rng.randint(4, 15)),
                "FGA": float(rng.randint(10, 26)),
                "FG_PCT": 0.48,
                "FG3M": float(rng.randint(0, 5)),
                "FG3A": float(rng.randint(1, 9)),
                "FG3_PCT": 0.36,
                "FTM": float(rng.randint(0, 9)),
                "FTA": float(rng.randint(0, 11)),
                "FT_PCT": 0.78,
                "OREB": float(rng.randint(0, 3)),
                "DREB": float(rng.randint(3, 9)),
                "REB": float(rng.randint(4, 11)),
                "AST": float(rng.randint(2, 11)),
                "STL": float(rng.randint(0, 3)),
                "BLK": float(rng.randint(0, 2)),
                "TOV": float(rng.randint(1, 5)),
                "PF": float(rng.randint(0, 4)),
                "PTS": pts,
                "PLUS_MINUS": float(rng.randint(-15, 15)),
                "VIDEO_AVAILABLE": 1,
            }
        )
    return pd.DataFrame(rows)


def _old_path_features():
    """Pre-Step-7: the raw nba_api DataFrame went straight into
    build_player_feature_row, unmodified."""
    raw_df = _raw_gamelog_df()
    return build_player_feature_row(raw_df, PLAYER_NAME, SPORTSBOOK_LINE)


def _new_path_features():
    """Post-Step-7: NBAApiProvider.get_player_game_logs (mocking only
    nba_api's own endpoint class) -> canonical records ->
    player_game_logs_to_dataframe -> build_player_feature_row."""
    fake_response = MagicMock()
    fake_response.get_data_frames.return_value = [_raw_gamelog_df()]
    with patch(
        "training.data.nba_client.playergamelog.PlayerGameLog",
        return_value=fake_response,
    ):
        provider = NBAApiProvider(sleep_func=lambda *_: None)
        records = provider.get_player_game_logs(PLAYER_ID, SEASON)
    reconstructed_df = player_game_logs_to_dataframe(records)
    return build_player_feature_row(reconstructed_df, PLAYER_NAME, SPORTSBOOK_LINE)


def test_legacy_feature_names_and_order_are_identical():
    old_X = _old_path_features()
    new_X = _new_path_features()
    assert old_X is not None and new_X is not None
    assert list(old_X.columns) == list(new_X.columns)
    assert list(old_X.columns) == list(LEGACY_FEATURE_NAMES)


def test_legacy_feature_values_are_identical():
    old_X = _old_path_features()
    new_X = _new_path_features()

    max_abs_feature_diff = float(
        np.nanmax(
            np.abs(
                old_X[list(LEGACY_FEATURE_NAMES)].to_numpy(dtype=float)
                - new_X[list(LEGACY_FEATURE_NAMES)].to_numpy(dtype=float)
            )
        )
    )
    assert max_abs_feature_diff == 0.0


def test_legacy_feature_dataframes_are_exactly_equal():
    old_X = _old_path_features()
    new_X = _new_path_features()
    pd.testing.assert_frame_equal(old_X, new_X)


@pytest.mark.skipif(
    not __import__("os").path.exists(LEGACY_MODEL_PATH),
    reason="models/points_regression.pkl not present in this checkout",
)
def test_legacy_model_prediction_parity_between_old_and_new_paths():
    """
    Uses the ACTUAL deployed legacy model (no retraining) to predict on
    both paths' feature rows and confirms prediction delta is zero
    (machine precision). Reports rows compared, max abs feature diff,
    max abs prediction diff -- exactly what the Step 7 brief requests.
    """
    old_X = _old_path_features()
    new_X = _new_path_features()

    model = joblib.load(LEGACY_MODEL_PATH)
    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if model_feature_names:
        old_X = old_X.reindex(columns=model_feature_names, fill_value=0)
        new_X = new_X.reindex(columns=model_feature_names, fill_value=0)

    pred_old = model.predict(old_X)
    pred_new = model.predict(new_X)

    max_abs_feature_diff = float(
        np.nanmax(np.abs(old_X.to_numpy(dtype=float) - new_X.to_numpy(dtype=float)))
    )
    max_abs_prediction_diff = float(np.max(np.abs(pred_old - pred_new)))

    assert len(old_X) == len(new_X) == 1  # build_player_feature_row returns one row
    assert max_abs_feature_diff == 0.0
    # Machine-precision tolerance, not exact 0.0: this model has n_jobs=-1
    # (verified: repeated model.predict() calls on the SAME input are
    # bit-identical, so the model itself is deterministic), but two
    # independently-constructed pandas DataFrames feeding the same
    # feature values through joblib's parallel per-tree reduction can
    # legitimately land on a different, still mathematically valid,
    # floating-point summation order -- observed magnitude ~1e-14, ~13
    # orders of magnitude below any prediction this model produces.
    # Asserting exact equality here was flaky (intermittently ~1e-15
    # off); 1e-6 is generously tight while eliminating that flake.
    assert max_abs_prediction_diff < 1e-6
