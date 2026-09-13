"""
Behavior-parity tests for the src/features/build_features.py extraction.

These tests prove that pulling build_player_feature_row() out of
src/shared_app.py and into src/features/build_features.py did not change
its behavior, using a deterministic synthetic gamelog fixture (no live NBA
API calls) and a golden set of expected values captured from the ORIGINAL
implementation before the extraction happened.

To reproduce/re-verify the golden values independently: construct
GAMELOG_ROWS below, run the pre-refactor src/shared_app.py implementation
against it with sportsbook_line=21.5, and compare.
"""

import math

import pandas as pd
import pytest

from src.features.build_features import build_player_feature_row
from src.features.feature_schema import LEGACY_FEATURE_NAMES

PLAYER_NAME = "Test Player"
SPORTSBOOK_LINE = 21.5

# Deterministic synthetic gamelog: one player, 15 games, 3 opponents (BOS's
# perspective), mixed home/away MATCHUP strings, one back-to-back at the end
# (2024-11-19 -> 2024-11-20, 1 day rest), and MIN provided in both "MM:SS"
# and plain-float forms to exercise _parse_minutes_value's branches.
GAMELOG_ROWS = [
    ("2024-10-24", "BOS vs NYK", "34:12", 22, 8, 16, 4, 4, 5, 1, 5, 1, 4, 0, 2, 3),
    ("2024-10-26", "BOS @ MIA", "30:45", 18, 7, 14, 2, 2, 2, 0, 4, 2, 3, 1, 3, 2),
    ("2024-10-28", "BOS vs MIA", 28.5, 15, 6, 13, 1, 2, 3, 2, 3, 1, 2, 0, 1, 1),
    ("2024-10-30", "BOS @ CHI", "36:00", 27, 10, 19, 3, 4, 4, 0, 6, 0, 5, 1, 2, 4),
    ("2024-11-01", "BOS vs CHI", "31:20", 19, 7, 15, 2, 3, 3, 1, 4, 2, 3, 0, 2, 2),
    ("2024-11-02", "BOS @ NYK", "33:10", 24, 9, 18, 3, 3, 4, 0, 5, 1, 6, 1, 1, 3),
    ("2024-11-05", "BOS vs CHI", "29:40", 16, 6, 12, 1, 2, 2, 2, 2, 0, 4, 0, 3, 2),
    ("2024-11-07", "BOS @ MIA", "35:15", 30, 11, 20, 4, 5, 6, 1, 7, 2, 4, 1, 2, 2),
    ("2024-11-09", "BOS vs NYK", "32:00", 21, 8, 17, 2, 3, 3, 0, 4, 1, 5, 0, 2, 3),
    ("2024-11-11", "BOS @ CHI", "27:30", 14, 5, 11, 1, 3, 4, 1, 3, 1, 2, 0, 1, 1),
    ("2024-11-13", "BOS vs MIA", "34:50", 25, 9, 16, 3, 5, 5, 0, 6, 2, 3, 1, 2, 2),
    ("2024-11-14", "BOS @ NYK", "38:20", 33, 12, 22, 5, 6, 7, 1, 8, 0, 6, 1, 3, 4),
    ("2024-11-17", "BOS vs CHI", "30:10", 17, 7, 14, 2, 2, 2, 1, 3, 1, 3, 0, 2, 2),
    ("2024-11-19", "BOS @ MIA", "36:40", 28, 10, 19, 3, 5, 6, 0, 5, 1, 5, 1, 2, 3),
    ("2024-11-20", "BOS vs NYK", "33:00", 20, 8, 15, 2, 3, 3, 1, 4, 1, 4, 0, 2, 2),
]

GAMELOG_COLUMNS = [
    "GAME_DATE", "MATCHUP", "MIN", "PTS", "FGM", "FGA", "FG3A",
    "FTM", "FTA", "OREB", "DREB", "STL", "AST", "BLK", "PF", "TOV",
]

# Captured from the pre-refactor src/shared_app.py::build_player_feature_row()
# running against GAMELOG_ROWS with sportsbook_line=21.5. NaN entries are
# genuine (insufficient same-opponent / 20-game history in this small
# fixture) and are part of the expected, parity-checked output.
EXPECTED_FEATURES = {
    "player_avg_pts": 22.071428571428573,
    "player_avg_pts_sq": 487.14795918367355,
    "season_minutes_avg": 32.740476190476194,
    "predicted_minutes": 33.5,
    "home_game": 1,
    "days_rest": 1.0,
    "is_back_to_back": 1,
    "last3_pts": 26.0,
    "last5_pts": 23.4,
    "last10_pts": 22.7,
    "last20_pts": math.nan,
    "last5_fga": 16.4,
    "last5_fta": 4.8,
    "last5_minutes": 33.5,
    "last5_gmsc": 17.92,
    "last5_usage_proxy": 20.912,
    "minutes_volatility": 4.535354941395923,
    "opp_pts_allowed": math.nan,
    "opp_pts_allowed_last5": math.nan,
    "points_volatility": 7.829431652425353,
    "is_star": 1,
    "closing_line": 21.5,
    "opp_pts_volatility": math.nan,
    "last5_3pa": 2.8,
}

EXPECTED_INT_COLUMNS = {"home_game", "is_back_to_back", "is_star"}

EXPECTED_PREDICTION = 20.22848389926869


@pytest.fixture
def gamelog_df():
    return pd.DataFrame(GAMELOG_ROWS, columns=GAMELOG_COLUMNS)


def _assert_matches_golden(result):
    assert result is not None
    assert list(result.columns) == list(LEGACY_FEATURE_NAMES)
    assert len(result) == 1

    for col in LEGACY_FEATURE_NAMES:
        actual = result[col].iloc[0]
        expected = EXPECTED_FEATURES[col]

        if isinstance(expected, float) and math.isnan(expected):
            assert pd.isna(actual), f"{col}: expected NaN, got {actual!r}"
            continue

        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"{col}: expected {expected!r}, got {actual!r}"
        )

    for col in EXPECTED_INT_COLUMNS:
        assert pd.api.types.is_integer_dtype(result[col]), (
            f"{col}: expected an integer dtype column, got {result[col].dtype}"
        )


def test_build_player_feature_row_matches_golden_columns_and_order(gamelog_df):
    result = build_player_feature_row(gamelog_df, PLAYER_NAME, SPORTSBOOK_LINE)

    assert list(result.columns) == list(LEGACY_FEATURE_NAMES), (
        "Output column set/order no longer matches the legacy schema contract"
    )


def test_build_player_feature_row_matches_golden_values(gamelog_df):
    result = build_player_feature_row(gamelog_df, PLAYER_NAME, SPORTSBOOK_LINE)
    _assert_matches_golden(result)


def test_build_player_feature_row_missing_value_pattern_matches_golden(gamelog_df):
    """
    Explicit check that the *pattern* of missing data (NaN vs populated) is
    unchanged -- this fixture intentionally has too little same-opponent and
    20-game history to populate opp_pts_* / last20_pts, and the extraction
    must preserve that, not silently start filling or dropping them.
    """
    result = build_player_feature_row(gamelog_df, PLAYER_NAME, SPORTSBOOK_LINE)

    expected_nan_cols = {
        col for col, val in EXPECTED_FEATURES.items()
        if isinstance(val, float) and math.isnan(val)
    }
    actual_nan_cols = {col for col in LEGACY_FEATURE_NAMES if pd.isna(result[col].iloc[0])}

    assert actual_nan_cols == expected_nan_cols


def test_shared_app_reexports_the_same_function_object():
    """
    shared_app.py must be a thin importer, not a second implementation.
    Importing it requires the app's full dependency set (gspread, streamlit,
    nba_api, ...), matching this repo's existing convention that anything
    touching shared_app.py runs against requirements.txt fully installed.
    """
    from src import shared_app
    from src.features import build_features

    assert shared_app.build_player_feature_row is build_features.build_player_feature_row


def test_shared_app_import_path_produces_identical_output(gamelog_df):
    from src.shared_app import build_player_feature_row as shared_app_build_row

    result = shared_app_build_row(gamelog_df, PLAYER_NAME, SPORTSBOOK_LINE)
    _assert_matches_golden(result)


def test_prediction_parity_against_legacy_model(gamelog_df):
    """
    Strongest proof of no behavior change: build the feature row through the
    new shared implementation, run it through the unmodified, unretrained
    legacy model artifact, and confirm the prediction matches the value
    captured from the pre-refactor implementation.
    """
    import os

    import joblib

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "points_regression.pkl",
    )
    model = joblib.load(model_path)

    result = build_player_feature_row(gamelog_df, PLAYER_NAME, SPORTSBOOK_LINE)
    X = result.reindex(columns=list(model.feature_names_in_))

    prediction = float(model.predict(X)[0])

    assert prediction == pytest.approx(EXPECTED_PREDICTION, rel=1e-9, abs=1e-9)


def test_build_player_feature_row_returns_none_for_empty_gamelog():
    empty_df = pd.DataFrame(columns=GAMELOG_COLUMNS)
    assert build_player_feature_row(empty_df, PLAYER_NAME, SPORTSBOOK_LINE) is None
