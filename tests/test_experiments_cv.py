"""
Tests for training/experiments/cv.py: temporal CV fold chronology and
train-split-only enforcement. No model fitting, no network access.
"""

import pandas as pd
import pytest

from training.experiments.cv import DEFAULT_N_BLOCKS, build_temporal_cv_folds


def _train_only(panel):
    return panel[panel["SPLIT"] == "train"].reset_index(drop=True)


def test_folds_are_chronological_train_before_validation(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)

    assert len(folds) == DEFAULT_N_BLOCKS - 1
    for fold in folds:
        assert fold.train_date_max < fold.val_date_min
        assert fold.train_df["GAME_DATE"].max() <= fold.train_date_max
        assert fold.val_df["GAME_DATE"].min() >= fold.val_date_min
        assert fold.val_df["GAME_DATE"].max() <= fold.val_date_max


def test_no_date_appears_on_both_sides_of_a_fold(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)

    for fold in folds:
        train_dates = set(fold.train_df["GAME_DATE"])
        val_dates = set(fold.val_df["GAME_DATE"])
        assert train_dates.isdisjoint(val_dates)


def test_folds_use_an_expanding_training_window(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)

    sizes = [fold.n_train for fold in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_build_temporal_cv_folds_rejects_rows_outside_official_train_split(
    experiments_panel,
):
    contaminated = pd.concat(
        [
            _train_only(experiments_panel),
            experiments_panel[experiments_panel["SPLIT"] == "validation"].head(1),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="SPLIT"):
        build_temporal_cv_folds(contaminated)


def test_build_temporal_cv_folds_rejects_test_rows_specifically(experiments_panel):
    contaminated = pd.concat(
        [
            _train_only(experiments_panel),
            experiments_panel[experiments_panel["SPLIT"] == "test"].head(1),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="test"):
        build_temporal_cv_folds(contaminated)


def test_build_temporal_cv_folds_deterministic_given_same_input(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds_a = build_temporal_cv_folds(train_df)
    folds_b = build_temporal_cv_folds(train_df)

    for fa, fb in zip(folds_a, folds_b):
        assert fa.train_date_min == fb.train_date_min
        assert fa.train_date_max == fb.train_date_max
        assert fa.val_date_min == fb.val_date_min
        assert fa.val_date_max == fb.val_date_max
        assert fa.n_train == fb.n_train
        assert fa.n_val == fb.n_val
