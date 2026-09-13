"""
Paired per-row error comparison and a deterministic bootstrap confidence
interval for MAE differences, computed only on the official V1
validation split (never test). See training/experiments/run.py for how
this is invoked for the final Step 5 candidate against: the frozen
Step 4 RF, and the player_avg_pts/last5_pts/last10_pts simple baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.experiments.guardrails import reject_test_rows

DEFAULT_N_BOOTSTRAP = 5000
DEFAULT_BOOTSTRAP_SEED = 42


def paired_absolute_error_diff(actual, candidate_pred, baseline_pred) -> np.ndarray:
    """
    Per-row (|candidate error| - |baseline error|). Negative means the
    candidate did better on that row; the mean of this array is exactly
    MAE(candidate) - MAE(baseline) over the same rows.
    """
    actual = np.asarray(actual, dtype=float)
    candidate_pred = np.asarray(candidate_pred, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    return np.abs(candidate_pred - actual) - np.abs(baseline_pred - actual)


def summarize_paired_comparison(diff: np.ndarray, tie_atol: float = 1e-9) -> dict:
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    wins = int(np.sum(diff < -tie_atol))
    losses = int(np.sum(diff > tie_atol))
    ties = n - wins - losses
    return {
        "n": n,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "fraction_candidate_wins": wins / n,
        "fraction_ties": ties / n,
        "fraction_candidate_loses": losses / n,
    }


def bootstrap_mae_diff_ci(
    diff: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """
    Deterministic (fixed-seed) bootstrap CI for the mean paired
    difference, i.e. MAE(candidate) - MAE(baseline). Resamples row
    indices with replacement, `n_bootstrap` times, from a single
    np.random.RandomState(seed) -- the same seed, n_bootstrap, and input
    array always reproduce the same CI. Negative is better for the
    candidate; if the interval spans zero, the difference is not
    distinguishable from noise at this confidence level.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return {
        "point_estimate_mae_diff": float(diff.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci_excludes_zero": bool((lower > 0) or (upper < 0)),
    }


def paired_validation_comparison(
    val_df: pd.DataFrame,
    candidate_pred,
    baselines: dict,
    target_column: str = "PTS",
) -> dict:
    """
    `baselines`: {name: prediction_array}, each already aligned to the
    SAME rows/order as `candidate_pred` and `val_df` -- see
    training/experiments/run.py, which builds every baseline's
    predictions from this exact val_df before calling this function, so
    candidate and baseline are always compared on an identical row set.
    Rejects any val_df containing test rows.
    """
    reject_test_rows(val_df, "paired_validation_comparison")
    actual = val_df[target_column].to_numpy(dtype=float)
    out = {}
    for name, baseline_pred in baselines.items():
        diff = paired_absolute_error_diff(actual, candidate_pred, baseline_pred)
        out[name] = {
            **summarize_paired_comparison(diff),
            "bootstrap_ci": bootstrap_mae_diff_ci(diff),
        }
    return out
