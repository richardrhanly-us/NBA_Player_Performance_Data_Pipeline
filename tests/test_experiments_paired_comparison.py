"""
Tests for training/experiments/paired_comparison.py: paired-error math
correctness, bootstrap CI determinism under a fixed seed, and that the
official-validation comparison uses an identical row set for the
candidate and each baseline. No network access.
"""

import numpy as np
import pandas as pd
import pytest

from training.experiments.paired_comparison import (
    bootstrap_mae_diff_ci,
    paired_absolute_error_diff,
    paired_validation_comparison,
    summarize_paired_comparison,
)


def test_paired_absolute_error_diff_matches_hand_computed_values():
    actual = np.array([10.0, 20.0, 30.0])
    candidate_pred = np.array([12.0, 18.0, 33.0])  # abs errors: 2, 2, 3
    baseline_pred = np.array([15.0, 15.0, 30.0])  # abs errors: 5, 5, 0

    diff = paired_absolute_error_diff(actual, candidate_pred, baseline_pred)
    assert diff == pytest.approx([2 - 5, 2 - 5, 3 - 0])


def test_paired_diff_mean_equals_mae_difference():
    rng = np.random.RandomState(0)
    actual = rng.uniform(0, 30, size=200)
    candidate_pred = actual + rng.normal(0, 3, size=200)
    baseline_pred = actual + rng.normal(0, 5, size=200)

    diff = paired_absolute_error_diff(actual, candidate_pred, baseline_pred)
    candidate_mae = np.mean(np.abs(candidate_pred - actual))
    baseline_mae = np.mean(np.abs(baseline_pred - actual))

    assert diff.mean() == pytest.approx(candidate_mae - baseline_mae)


def test_summarize_paired_comparison_counts_wins_ties_losses_correctly():
    # candidate strictly better on 2 rows, strictly worse on 1, tied on 1
    diff = np.array([-1.0, -2.0, 3.0, 0.0])
    summary = summarize_paired_comparison(diff)

    assert summary["n"] == 4
    assert summary["fraction_candidate_wins"] == pytest.approx(2 / 4)
    assert summary["fraction_candidate_loses"] == pytest.approx(1 / 4)
    assert summary["fraction_ties"] == pytest.approx(1 / 4)
    assert summary["mean_diff"] == pytest.approx(0.0)
    assert summary["median_diff"] == pytest.approx(-0.5)


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed():
    rng = np.random.RandomState(1)
    diff = rng.normal(-0.2, 2.0, size=500)

    ci_a = bootstrap_mae_diff_ci(diff, n_bootstrap=500, seed=42)
    ci_b = bootstrap_mae_diff_ci(diff, n_bootstrap=500, seed=42)

    assert ci_a == ci_b


def test_bootstrap_ci_differs_for_different_seeds():
    rng = np.random.RandomState(1)
    diff = rng.normal(-0.2, 2.0, size=500)

    ci_a = bootstrap_mae_diff_ci(diff, n_bootstrap=200, seed=1)
    ci_b = bootstrap_mae_diff_ci(diff, n_bootstrap=200, seed=2)

    assert ci_a["ci_lower"] != ci_b["ci_lower"] or ci_a["ci_upper"] != ci_b["ci_upper"]


def test_bootstrap_ci_reports_whether_it_excludes_zero():
    # a constant, clearly negative diff: every bootstrap resample's mean
    # is exactly -5.0, so the CI sits entirely below zero regardless of seed.
    diff = np.full(300, -5.0)
    ci = bootstrap_mae_diff_ci(diff, n_bootstrap=200, seed=42)
    assert ci["ci_upper"] < 0
    assert ci["ci_excludes_zero"] is True

    # an all-zero diff: every bootstrap resample's mean is exactly 0.0,
    # so the CI is exactly [0.0, 0.0] regardless of seed -- does not
    # exclude zero (the boundary itself is zero, not strictly beyond it).
    diff_zero = np.zeros(50)
    ci_zero = bootstrap_mae_diff_ci(diff_zero, n_bootstrap=200, seed=42)
    assert ci_zero["ci_lower"] == 0.0
    assert ci_zero["ci_upper"] == 0.0
    assert ci_zero["ci_excludes_zero"] is False


def test_paired_validation_comparison_uses_identical_row_set_for_every_baseline():
    val_df = pd.DataFrame(
        {
            "SPLIT": ["validation"] * 5,
            "PTS": [10.0, 20.0, 15.0, 5.0, 25.0],
        }
    )
    candidate_pred = np.array([11.0, 19.0, 14.0, 6.0, 24.0])
    baselines = {
        "baseline_a": np.array([12.0, 18.0, 16.0, 4.0, 23.0]),
        "baseline_b": np.array([9.0, 21.0, 13.0, 7.0, 26.0]),
    }

    result = paired_validation_comparison(val_df, candidate_pred, baselines)

    # Recompute independently using the same actual/candidate/baseline
    # arrays, in the same row order, and check the reported stats match
    # exactly -- proving no row was dropped/reordered differently
    # between the two baselines' comparisons.
    actual = val_df["PTS"].to_numpy()
    for name, baseline_pred in baselines.items():
        expected_diff = paired_absolute_error_diff(
            actual, candidate_pred, baseline_pred
        )
        expected_summary = summarize_paired_comparison(expected_diff)
        assert result[name]["n"] == expected_summary["n"] == 5
        assert result[name]["mean_diff"] == pytest.approx(expected_summary["mean_diff"])


def test_paired_validation_comparison_rejects_test_split_rows():
    val_df = pd.DataFrame({"SPLIT": ["test"], "PTS": [10.0]})
    with pytest.raises(ValueError, match="test"):
        paired_validation_comparison(val_df, np.array([9.0]), {"b": np.array([8.0])})
