"""
Tests for training/evaluate.py: metrics correctness, residual artifact
schema, segment diagnostics, and simple-baseline scoring. No model
training, no network access.
"""

import numpy as np
import pandas as pd
import pytest

from training import evaluate


def test_compute_regression_metrics_matches_hand_computed_values():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    # residuals (pred - actual): 2, -2, 3, -4
    metrics = evaluate.compute_regression_metrics(y_true, y_pred)

    assert metrics["n"] == 4
    assert metrics["mae"] == pytest.approx((2 + 2 + 3 + 4) / 4)
    assert metrics["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 16) / 4))
    assert metrics["bias_mean_pred_minus_actual"] == pytest.approx((2 - 2 + 3 - 4) / 4)
    assert metrics["actual_mean"] == pytest.approx(25.0)
    assert metrics["predicted_mean"] == pytest.approx(24.75)

    ss_res = 4 + 4 + 9 + 16
    ss_tot = sum((v - 25.0) ** 2 for v in y_true)
    assert metrics["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_compute_regression_metrics_perfect_predictions_give_zero_error_and_r2_one():
    y_true = np.array([5.0, 10.0, 15.0, 20.0])
    metrics = evaluate.compute_regression_metrics(y_true, y_true.copy())

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["bias_mean_pred_minus_actual"] == pytest.approx(0.0)
    assert metrics["prediction_actual_correlation"] == pytest.approx(1.0)


def test_bias_sign_convention_is_predicted_minus_actual():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([15.0, 15.0])  # consistently overpredicts by 5
    metrics = evaluate.compute_regression_metrics(y_true, y_pred)
    assert metrics["bias_mean_pred_minus_actual"] == pytest.approx(5.0)


def _sample_subset_df():
    return pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "PLAYER_NAME": ["A", "B", "C"],
            "GAME_ID": ["G1", "G2", "G3"],
            "GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "SEASON": ["2025-26"] * 3,
            "SPLIT": ["validation"] * 3,
            "TEAM_ABBREVIATION": ["AAA", "BBB", "CCC"],
            "OPPONENT_ABBREVIATION": ["XXX", "YYY", "ZZZ"],
            "home_game": [1, 0, 1],
            "is_back_to_back": [0, 1, 0],
            "PRIOR_GAMES_THIS_SEASON": [10, 20, 5],
            "player_avg_pts": [15.0, 22.0, 8.0],
            "last5_pts": [14.0, 24.0, 9.0],
            "recent_minutes_avg": [30.0, 32.0, 20.0],
        }
    )


def test_build_residuals_df_has_required_identifiers_and_correct_math():
    subset = _sample_subset_df()
    y_true = pd.Series([20.0, 25.0, 10.0])
    y_pred = np.array([18.0, 27.0, 12.0])

    residuals = evaluate.build_residuals_df(subset, y_true, y_pred)

    for col in evaluate.RESIDUAL_IDENTIFIER_COLUMNS:
        assert col in residuals.columns
    assert list(residuals["actual_pts"]) == [20.0, 25.0, 10.0]
    assert list(residuals["predicted_pts"]) == [18.0, 27.0, 12.0]
    assert list(residuals["residual"]) == pytest.approx([-2.0, 2.0, 2.0])
    assert list(residuals["absolute_error"]) == pytest.approx([2.0, 2.0, 2.0])


def test_build_residuals_df_contains_no_sportsbook_columns():
    subset = _sample_subset_df()
    residuals = evaluate.build_residuals_df(
        subset, [20.0, 25.0, 10.0], [18.0, 27.0, 12.0]
    )
    forbidden = {"closing_line", "sportsbook_line", "sportsbook", "odds", "edge"}
    assert forbidden.isdisjoint(set(residuals.columns))


def test_segment_by_scoring_level_uses_pregame_player_avg_pts_not_actual():
    subset = (
        _sample_subset_df()
    )  # player_avg_pts: 15, 22, 8 -> bands 10-19.9 / 20-29.9 / 0-9.9
    residuals = evaluate.build_residuals_df(subset, [5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    segments = evaluate.segment_by_scoring_level(residuals)
    labels_present = {row["segment"] for row in segments}
    assert labels_present == {"0-9.9", "10-19.9", "20-29.9"}
    for row in segments:
        assert row["n"] == 1
        assert row["mae"] == pytest.approx(0.0)


def test_segment_by_prior_games_buckets_correctly():
    subset = _sample_subset_df()  # PRIOR_GAMES_THIS_SEASON: 10, 20, 5
    residuals = evaluate.build_residuals_df(subset, [5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    segments = {
        row["segment"]: row for row in evaluate.segment_by_prior_games(residuals)
    }
    assert segments["5-9"]["n"] == 1
    assert segments["10-19"]["n"] == 1
    assert segments["20+"]["n"] == 1


def test_segment_by_home_away_and_back_to_back():
    subset = _sample_subset_df()
    residuals = evaluate.build_residuals_df(subset, [5.0, 5.0, 5.0], [7.0, 5.0, 3.0])

    home_away = {
        row["segment"]: row for row in evaluate.segment_by_home_away(residuals)
    }
    assert home_away["home"]["n"] == 2  # rows 0 and 2
    assert home_away["away"]["n"] == 1  # row 1

    b2b = {row["segment"]: row for row in evaluate.segment_by_back_to_back(residuals)}
    assert b2b["back_to_back"]["n"] == 1
    assert b2b["not_back_to_back"]["n"] == 2


def test_compute_simple_baseline_metrics_uses_feature_as_prediction():
    df = pd.DataFrame({"player_avg_pts": [10.0, 20.0, 30.0], "PTS": [12.0, 18.0, 33.0]})
    result = evaluate.compute_simple_baseline_metrics(df, "player_avg_pts")

    assert result["feature_used_as_prediction"] == "player_avg_pts"
    assert result["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert result["rows_with_missing_feature_excluded"] == 0


def test_compute_simple_baseline_metrics_excludes_rows_with_missing_feature():
    df = pd.DataFrame({"last10_pts": [10.0, np.nan, 30.0], "PTS": [12.0, 18.0, 33.0]})
    result = evaluate.compute_simple_baseline_metrics(df, "last10_pts")

    assert result["n"] == 2
    assert result["rows_with_missing_feature_excluded"] == 1
