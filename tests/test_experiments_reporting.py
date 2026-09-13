"""
Tests for training/experiments/reporting.py: the persisted experiment
metadata contains every required provenance field, and results are
written under training/experiments/results/ -- never under
models/registry/. No network access.
"""

import json

from training.experiments import reporting

REQUIRED_METADATA_FIELDS = (
    "git_commit",
    "panel_path",
    "feature_schema",
    "train_row_count",
    "validation_row_count",
    "test_row_count_excluded_not_used",
    "temporal_cv_fold_boundaries",
    "search_space_size",
    "ablation_count",
    "random_state",
    "sklearn_version",
    "selection_metric",
    "predeclared_meaningful_improvement_mae_threshold",
    "selected_hyperparameters",
    "selected_ablation_id",
    "run_start_utc",
    "run_end_utc",
)


def _dummy_metadata(**overrides):
    from datetime import datetime, timezone

    defaults = {
        "panel_path": "training/data/processed/v1_panel.parquet",
        "train_row_count": 100,
        "validation_row_count": 30,
        "test_row_count_excluded": 20,
        "cv_fold_boundaries": [{"fold": 1}],
        "candidate_count": 18,
        "ablation_count": 5,
        "selected_params": {"n_estimators": 400},
        "selected_ablation_id": "A_full_v1",
        "run_start": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "run_end": datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return reporting.build_experiment_metadata(**defaults)


def test_experiment_metadata_contains_all_required_provenance_fields():
    metadata = _dummy_metadata()
    for field in REQUIRED_METADATA_FIELDS:
        assert field in metadata, f"missing required provenance field: {field}"


def test_experiment_metadata_records_test_row_count_but_not_test_data():
    metadata = _dummy_metadata(test_row_count_excluded=42)
    assert metadata["test_row_count_excluded_not_used"] == 42
    # only a count is recorded -- no test-split metric/result key exists anywhere
    assert not any("test_mae" in k or "test_metric" in k for k in metadata)


def test_meaningful_improvement_threshold_is_predeclared_at_point_zero_five():
    assert reporting.MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD == 0.05


def test_random_state_recorded_as_42():
    metadata = _dummy_metadata()
    assert metadata["random_state"] == 42


def test_save_experiment_results_writes_expected_files_under_results_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reporting, "EXPERIMENTS_ROOT", tmp_path)

    search_results = [{"candidate_id": "step4_baseline", "mean_cv_mae": 4.5}]
    ablation_results = [
        {"ablation_id": "A_full_v1", "mean_cv_mae": 4.5, "n_features": 23}
    ]
    validation_comparison = {"final_candidate": "best_hp_full_features"}
    metadata = _dummy_metadata()

    out_dir = reporting.save_experiment_results(
        "step5_test_run",
        search_results=search_results,
        ablation_results=ablation_results,
        validation_comparison=validation_comparison,
        experiment_metadata=metadata,
    )

    assert out_dir == tmp_path / "step5_test_run"
    assert (out_dir / "search_results.csv").exists()
    assert (out_dir / "search_results.json").exists()
    assert (out_dir / "ablation_results.csv").exists()
    assert (out_dir / "ablation_results.json").exists()
    assert (out_dir / "validation_comparison.json").exists()
    assert (out_dir / "experiment_metadata.json").exists()
    assert not (
        out_dir / "selected_candidate_model.pkl"
    ).exists()  # not persisted unless given

    with open(out_dir / "experiment_metadata.json", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["random_state"] == 42


def test_default_experiments_root_is_separate_from_the_model_registry():
    from training import config

    default_root = reporting.EXPERIMENTS_ROOT
    registry_root = config.MODEL_REGISTRY_DIR

    assert default_root != registry_root
    assert registry_root not in default_root.parents
    assert default_root not in registry_root.parents
    assert "experiments" in default_root.parts
    assert "registry" not in default_root.parts
