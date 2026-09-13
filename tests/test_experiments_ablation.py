"""
Tests for training/experiments/feature_groups.py and ablation.py:
feature groups reference valid V1_FEATURE_NAMES, no ablation contains a
duplicate feature name, and the full-feature ablation reproduces the
exact V1 schema order. No network access.
"""

from src.features import v1_schema
from training.experiments.ablation import run_ablation_cv
from training.experiments.cv import build_temporal_cv_folds
from training.experiments.feature_groups import ABLATION_SPECS, FEATURE_GROUPS


def test_feature_groups_exactly_reference_valid_v1_feature_names():
    schema_set = set(v1_schema.V1_FEATURE_NAMES)
    for group_name, features in FEATURE_GROUPS.items():
        for feature in features:
            assert feature in schema_set, (
                f"FEATURE_GROUPS['{group_name}'] references '{feature}', "
                "which is not in V1_FEATURE_NAMES."
            )


def test_feature_groups_exactly_partition_v1_feature_names():
    all_grouped = [f for group in FEATURE_GROUPS.values() for f in group]
    assert set(all_grouped) == set(v1_schema.V1_FEATURE_NAMES)
    assert len(all_grouped) == len(v1_schema.V1_FEATURE_NAMES)


def test_no_duplicate_feature_names_within_any_single_ablation():
    for spec in ABLATION_SPECS:
        features = spec["feature_names"]
        assert len(features) == len(set(features)), (
            f"Ablation '{spec['ablation_id']}' contains a duplicate feature name."
        )


def test_no_duplicate_feature_names_within_any_feature_group():
    for group_name, features in FEATURE_GROUPS.items():
        assert len(features) == len(set(features)), (
            f"FEATURE_GROUPS['{group_name}'] contains a duplicate feature name."
        )


def test_full_feature_ablation_reproduces_full_v1_schema_order():
    full = next(spec for spec in ABLATION_SPECS if spec["ablation_id"] == "A_full_v1")
    assert full["feature_names"] == tuple(v1_schema.V1_FEATURE_NAMES)


def test_every_ablation_is_a_subset_of_the_full_v1_schema():
    schema_set = set(v1_schema.V1_FEATURE_NAMES)
    for spec in ABLATION_SPECS:
        assert set(spec["feature_names"]) <= schema_set


def test_ablation_d_and_f_are_the_same_configuration_by_construction():
    """D (full minus opponent+schedule) and F (core+volume only) are
    documented as equivalent given the four FEATURE_GROUPS exactly
    partition V1_FEATURE_NAMES -- verify that equivalence holds, rather
    than assuming the module docstring's claim is still true after any
    future edit to FEATURE_GROUPS."""
    combined = next(
        spec
        for spec in ABLATION_SPECS
        if spec["ablation_id"] == "D_minus_opponent_and_schedule_F_core_and_volume_only"
    )
    expected = set(FEATURE_GROUPS["core_scoring_history"]) | set(
        FEATURE_GROUPS["volume_role"]
    )
    assert set(combined["feature_names"]) == expected


def test_ablation_specs_have_unique_ids():
    ids = [spec["ablation_id"] for spec in ABLATION_SPECS]
    assert len(ids) == len(set(ids))


def test_run_ablation_cv_scores_every_spec_and_ranks_by_mae(experiments_panel):
    train_df = experiments_panel[experiments_panel["SPLIT"] == "train"].reset_index(
        drop=True
    )
    folds = build_temporal_cv_folds(train_df)
    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 1,
    }

    results = run_ablation_cv(params, folds, ABLATION_SPECS)

    assert len(results) == len(ABLATION_SPECS)
    result_ids = {r["ablation_id"] for r in results}
    assert result_ids == {spec["ablation_id"] for spec in ABLATION_SPECS}
    maes = [r["mean_cv_mae"] for r in results]
    assert maes == sorted(maes)
