"""
Evaluation utilities for the V1 baseline: regression metrics, row-level
residuals, segment diagnostics, and simple pregame-average baselines.

Pure functions over arrays/DataFrames already produced by training/train.py
-- no model fitting happens here, and nothing here reads or writes files.
"""

import numpy as np
import pandas as pd

RESIDUAL_IDENTIFIER_COLUMNS = (
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "SPLIT",
)

# Optional context columns carried into the residual artifact when present,
# useful for later analysis without needing to rejoin the full panel.
RESIDUAL_CONTEXT_COLUMNS = (
    "TEAM_ABBREVIATION",
    "OPPONENT_ABBREVIATION",
    "home_game",
    "is_back_to_back",
    "PRIOR_GAMES_THIS_SEASON",
    "player_avg_pts",
    "last5_pts",
    "recent_minutes_avg",
)


def compute_regression_metrics(y_true, y_pred) -> dict:
    """
    MAE, RMSE, R^2, bias (mean(pred - actual)), plus the distributional
    detail the report requires: actual/predicted mean+std, residual
    mean+std+percentiles, and the correlation between prediction and
    actual. `bias` is defined explicitly as mean(predicted - actual) --
    positive means the model overpredicts on average.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_pred - y_true
    abs_error = np.abs(residual)

    mae = float(abs_error.mean())
    rmse = float(np.sqrt((residual**2).mean()))
    ss_res = float((residual**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    bias = float(residual.mean())

    # Undefined (and not just "1 row") whenever either series has zero
    # variance -- guard explicitly rather than letting numpy warn and
    # return NaN implicitly.
    if len(y_true) > 1 and y_true.std() > 0 and y_pred.std() > 0:
        correlation = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        correlation = float("nan")

    percentiles = {
        "p5": float(np.percentile(residual, 5)),
        "p25": float(np.percentile(residual, 25)),
        "p50": float(np.percentile(residual, 50)),
        "p75": float(np.percentile(residual, 75)),
        "p95": float(np.percentile(residual, 95)),
    }

    return {
        "n": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias_mean_pred_minus_actual": bias,
        "actual_mean": float(y_true.mean()),
        "actual_std": float(y_true.std()),
        "predicted_mean": float(y_pred.mean()),
        "predicted_std": float(y_pred.std()),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_percentiles": percentiles,
        "prediction_actual_correlation": correlation,
    }


def build_residuals_df(subset_df: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """
    One row per (PLAYER_ID, GAME_ID) with identifiers, optional context,
    and actual_pts/predicted_pts/residual/absolute_error. No sportsbook
    columns -- none exist in the V1 panel to begin with (see
    src/features/v1_schema.py), so none can leak in here either.
    """
    result = pd.DataFrame(index=subset_df.index)
    for col in RESIDUAL_IDENTIFIER_COLUMNS:
        result[col] = subset_df[col].values
    for col in RESIDUAL_CONTEXT_COLUMNS:
        if col in subset_df.columns:
            result[col] = subset_df[col].values

    result["actual_pts"] = np.asarray(y_true, dtype=float)
    result["predicted_pts"] = np.asarray(y_pred, dtype=float)
    result["residual"] = result["predicted_pts"] - result["actual_pts"]
    result["absolute_error"] = result["residual"].abs()

    return result.reset_index(drop=True)


def _bucket_stats(
    residuals_df: pd.DataFrame, bucket_series: pd.Series, bucket_labels=None
) -> list:
    grouped = residuals_df.groupby(bucket_series, observed=True)
    rows = []
    for label, group in grouped:
        rows.append(
            {
                "segment": str(label),
                "n": len(group),
                "mae": float(group["absolute_error"].mean()),
                "bias": float(group["residual"].mean()),
            }
        )
    if bucket_labels is not None:
        order = {str(label): i for i, label in enumerate(bucket_labels)}
        rows.sort(key=lambda r: order.get(r["segment"], len(order)))
    return rows


def segment_by_scoring_level(
    residuals_df: pd.DataFrame, column: str = "player_avg_pts"
) -> list:
    """
    Buckets by a pregame scoring-level proxy (default: player_avg_pts, the
    player's season-to-date average entering the game) rather than that
    game's own actual PTS. Bucketing by the target itself would mix real
    model-quality differences with a pure regression-to-the-mean artifact
    (a model that is otherwise unbiased will still show a negative bias
    on rows where the actual value happened to be unusually high, and a
    positive bias where it was unusually low, regardless of model
    quality). Bucketing by a pregame proxy avoids that.
    """
    bins = [-0.01, 9.9, 19.9, 29.9, float("inf")]
    labels = ["0-9.9", "10-19.9", "20-29.9", "30+"]
    bucketed = pd.cut(residuals_df[column], bins=bins, labels=labels)
    return _bucket_stats(residuals_df, bucketed, bucket_labels=labels)


def segment_by_prior_games(residuals_df: pd.DataFrame) -> list:
    """Buckets by PRIOR_GAMES_THIS_SEASON. Only 5-9/10-19/20+ are possible
    among training-eligible rows (< 5 is excluded from eligibility)."""
    prior = residuals_df["PRIOR_GAMES_THIS_SEASON"]
    bins = [4, 9, 19, float("inf")]
    labels = ["5-9", "10-19", "20+"]
    bucketed = pd.cut(prior, bins=bins, labels=labels)
    return _bucket_stats(residuals_df, bucketed, bucket_labels=labels)


def segment_by_home_away(residuals_df: pd.DataFrame) -> list:
    labels = {1: "home", 0: "away"}
    bucketed = residuals_df["home_game"].map(labels)
    return _bucket_stats(residuals_df, bucketed, bucket_labels=["home", "away"])


def segment_by_back_to_back(residuals_df: pd.DataFrame) -> list:
    labels = {1: "back_to_back", 0: "not_back_to_back"}
    bucketed = residuals_df["is_back_to_back"].map(labels)
    return _bucket_stats(
        residuals_df, bucketed, bucket_labels=["back_to_back", "not_back_to_back"]
    )


def segment_by_season(residuals_df: pd.DataFrame) -> list:
    return _bucket_stats(residuals_df, residuals_df["SEASON"])


def build_segment_diagnostics(residuals_df: pd.DataFrame) -> dict:
    return {
        "by_scoring_level_player_avg_pts": segment_by_scoring_level(residuals_df),
        "by_prior_games_this_season": segment_by_prior_games(residuals_df),
        "by_home_away": segment_by_home_away(residuals_df),
        "by_back_to_back": segment_by_back_to_back(residuals_df),
        "by_season": segment_by_season(residuals_df),
    }


def compute_simple_baseline_metrics(
    df: pd.DataFrame, feature_column: str, target_column: str = "PTS"
) -> dict:
    """
    Treats an existing pregame V1 feature (e.g. player_avg_pts, last5_pts)
    directly as the prediction and scores it with the same metric set as
    the model -- the genuine "is the model actually better than just
    guessing the player's own recent average" check.
    """
    valid = df[[feature_column, target_column]].dropna()
    metrics = compute_regression_metrics(valid[target_column], valid[feature_column])
    metrics["feature_used_as_prediction"] = feature_column
    metrics["rows_with_missing_feature_excluded"] = int(len(df) - len(valid))
    return metrics
