"""
Stage A (temporal-CV) hyperparameter evaluation and Stage B (official
validation) scoring for a single, fixed feature set. RandomForestRegressor
only -- see training/experiments/search_space.py for the predeclared
candidate list this module scores, and training/experiments/cv.py for
fold construction.

Every function here that accepts a DataFrame with a SPLIT column
rejects rows from the frozen test split via
training.experiments.guardrails.reject_test_rows.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features import v1_schema
from training import evaluate
from training.experiments.cv import TemporalFold
from training.experiments.guardrails import reject_test_rows

logger = logging.getLogger(__name__)


def select_feature_subset(df: pd.DataFrame, feature_names) -> pd.DataFrame:
    """
    Selects `feature_names` in their canonical V1_FEATURE_NAMES order --
    never in whatever order `feature_names` happens to be passed in --
    so every candidate/ablation sees a consistent, schema-ordered matrix.
    """
    wanted = set(feature_names)
    ordered = [f for f in v1_schema.V1_FEATURE_NAMES if f in wanted]
    return df[ordered].copy()


def _fit_and_score(
    params: dict, train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_names
):
    X_train = select_feature_subset(train_df, feature_names)
    y_train = train_df[v1_schema.V1_TARGET_COLUMN]
    X_eval = select_feature_subset(eval_df, feature_names)
    y_eval = eval_df[v1_schema.V1_TARGET_COLUMN]

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    metrics = evaluate.compute_regression_metrics(y_eval, y_pred)
    return metrics, model


def evaluate_candidate_cv(
    params: dict,
    cv_folds: list[TemporalFold],
    feature_names=v1_schema.V1_FEATURE_NAMES,
) -> dict:
    """
    Fits `params` on each fold's train_df and scores on that fold's
    val_df (both carved from the official train split by
    training.experiments.cv.build_temporal_cv_folds) -- never the
    official validation or test splits. Returns per-fold metrics plus
    mean/std aggregates across folds.
    """
    for fold in cv_folds:
        reject_test_rows(fold.train_df, "evaluate_candidate_cv (fold train)")
        reject_test_rows(fold.val_df, "evaluate_candidate_cv (fold validation)")

    fold_results = []
    for fold in cv_folds:
        metrics, _ = _fit_and_score(params, fold.train_df, fold.val_df, feature_names)
        fold_results.append({"fold": fold.fold_index, **metrics})

    maes = np.array([r["mae"] for r in fold_results])
    rmses = np.array([r["rmse"] for r in fold_results])
    biases = np.array([r["bias_mean_pred_minus_actual"] for r in fold_results])

    return {
        "fold_results": fold_results,
        "mean_cv_mae": float(maes.mean()),
        "std_cv_mae": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
        "mean_cv_rmse": float(rmses.mean()),
        "mean_cv_bias": float(biases.mean()),
    }


def run_search(
    search_space: list[dict],
    cv_folds: list[TemporalFold],
    feature_names=v1_schema.V1_FEATURE_NAMES,
) -> list[dict]:
    """
    Evaluates every predeclared candidate in `search_space`, in the
    order given (deterministic, no shuffling), against `cv_folds`.
    Returns results ranked by mean_cv_mae ascending -- the primary
    development selection metric (see the Step 5 report for why MAE,
    not R^2/RMSE, drives ranking).
    """
    results = []
    for i, candidate in enumerate(search_space):
        start = time.monotonic()
        cv_result = evaluate_candidate_cv(candidate["params"], cv_folds, feature_names)
        elapsed = time.monotonic() - start
        result = {
            "candidate_id": candidate["candidate_id"],
            "note": candidate["note"],
            "params": candidate["params"],
            **cv_result,
            "fit_seconds": elapsed,
        }
        results.append(result)
        logger.info(
            "[%d/%d] candidate=%s mean_cv_mae=%.4f std_cv_mae=%.4f (%.1fs)",
            i + 1,
            len(search_space),
            candidate["candidate_id"],
            result["mean_cv_mae"],
            result["std_cv_mae"],
            elapsed,
        )
    results.sort(key=lambda r: (r["mean_cv_mae"], r["mean_cv_rmse"]))
    return results


def evaluate_on_official_validation(
    params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names=v1_schema.V1_FEATURE_NAMES,
) -> dict:
    """
    Stage B: fits ONE candidate on the full official train split and
    scores it on the full official validation split. Intended to be
    called only after Stage A (temporal CV) has already narrowed the
    candidate set -- see training/experiments/run.py. Rejects test rows
    exactly like the CV path above, and additionally requires every
    val_df row to actually carry the official "validation" SPLIT label.
    """
    reject_test_rows(train_df, "evaluate_on_official_validation (train)")
    reject_test_rows(val_df, "evaluate_on_official_validation (validation)")
    if "SPLIT" in val_df.columns and (val_df["SPLIT"] != "validation").any():
        raise ValueError(
            "evaluate_on_official_validation: val_df contains rows outside "
            "the official 'validation' split."
        )
    metrics, model = _fit_and_score(params, train_df, val_df, feature_names)
    return {"metrics": metrics, "model": model}
