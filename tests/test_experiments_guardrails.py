"""
Tests proving the frozen official test split (and, separately, the
official validation split) cannot accidentally participate in Step 5
candidate selection, and that no production registry state is touched.
No network access.
"""

import pandas as pd
import pytest

from training import config, model_registry
from training.experiments import search_space
from training.experiments.cv import build_temporal_cv_folds
from training.experiments.guardrails import reject_test_rows
from training.experiments.tuning import (
    evaluate_candidate_cv,
    evaluate_on_official_validation,
)


def _splits(panel):
    train_df = panel[panel["SPLIT"] == "train"].reset_index(drop=True)
    val_df = panel[panel["SPLIT"] == "validation"].reset_index(drop=True)
    test_df = panel[panel["SPLIT"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def test_reject_test_rows_raises_when_test_rows_present(experiments_panel):
    _, _, test_df = _splits(experiments_panel)
    with pytest.raises(ValueError, match="test"):
        reject_test_rows(test_df, "unit test")


def test_reject_test_rows_passes_for_test_free_frame(experiments_panel):
    train_df, _, _ = _splits(experiments_panel)
    reject_test_rows(train_df, "unit test")  # must not raise


def test_reject_test_rows_ignores_frames_without_a_split_column():
    reject_test_rows(pd.DataFrame({"x": [1, 2, 3]}), "unit test")  # must not raise


def test_official_test_rows_cannot_enter_temporal_cv_fold_construction(
    experiments_panel,
):
    train_df, _, test_df = _splits(experiments_panel)
    contaminated = pd.concat([train_df, test_df.head(2)], ignore_index=True)
    with pytest.raises(ValueError):
        build_temporal_cv_folds(contaminated)


def test_official_test_rows_cannot_enter_candidate_cv_scoring(experiments_panel):
    """Even if a caller managed to build a fold-shaped object from test
    rows (bypassing build_temporal_cv_folds), evaluate_candidate_cv
    itself refuses to fit/score against them."""
    train_df, _, test_df = _splits(experiments_panel)
    fake_fold_train = train_df.head(30)
    fake_fold_val = test_df.head(5)  # test rows smuggled in as the "validation" side

    class _FakeFold:
        fold_index = 1
        train_df = fake_fold_train
        val_df = fake_fold_val

    params = dict(search_space.STEP4_BASELINE_PARAMS, random_state=42, n_jobs=1)
    with pytest.raises(ValueError, match="test"):
        evaluate_candidate_cv(params, [_FakeFold()])


def test_official_test_rows_cannot_enter_official_validation_scoring(experiments_panel):
    train_df, val_df, test_df = _splits(experiments_panel)
    params = dict(search_space.STEP4_BASELINE_PARAMS, random_state=42, n_jobs=1)

    # test rows smuggled in as the "validation" argument
    with pytest.raises(ValueError, match="test"):
        evaluate_on_official_validation(params, train_df, test_df)

    # test rows smuggled into the "train" argument
    contaminated_train = pd.concat([train_df, test_df.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="test"):
        evaluate_on_official_validation(params, contaminated_train, val_df)


def test_official_validation_rows_never_reach_model_fit(experiments_panel, monkeypatch):
    """Spy on RandomForestRegressor.fit to prove validation rows are
    never part of the X passed to .fit() during official-validation
    scoring (they must only ever appear in .predict())."""
    train_df, val_df, _ = _splits(experiments_panel)
    val_game_ids = set(val_df["GAME_ID"])

    from sklearn.ensemble import RandomForestRegressor

    from training.experiments import tuning as tuning_module

    original_fit = RandomForestRegressor.fit
    seen_fit_sizes = []

    def spy_fit(self, X, y, *args, **kwargs):
        seen_fit_sizes.append(len(X))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", spy_fit)

    params = dict(search_space.STEP4_BASELINE_PARAMS, random_state=42, n_jobs=1)
    tuning_module.evaluate_on_official_validation(params, train_df, val_df)

    assert seen_fit_sizes == [len(train_df)]
    assert len(val_game_ids) > 0  # sanity: validation split is non-trivial


def test_no_production_registry_state_is_touched(experiments_panel):
    """Running Step 5 experiment building blocks must never write to
    models/registry/ or move models/registry/CURRENT_V1."""
    real_registry_dir = config.MODEL_REGISTRY_DIR
    pointer_before = None
    if model_registry.current_pointer_path().exists():
        pointer_before = model_registry.get_current_version()

    train_df, val_df, _ = _splits(experiments_panel)
    cv_folds = build_temporal_cv_folds(train_df)
    params = dict(search_space.STEP4_BASELINE_PARAMS, random_state=42, n_jobs=1)
    evaluate_candidate_cv(params, cv_folds[:1])
    evaluate_on_official_validation(params, train_df, val_df)

    assert config.MODEL_REGISTRY_DIR == real_registry_dir  # never repointed
    if pointer_before is not None:
        assert model_registry.get_current_version() == pointer_before
