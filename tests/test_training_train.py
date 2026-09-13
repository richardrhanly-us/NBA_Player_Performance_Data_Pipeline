"""
Tests for training/train.py: feature contract enforcement, split isolation,
determinism, metadata completeness, artifact round-trip, and native-NaN
handling. Uses a small synthetic panel (not the real 79k-row dataset) so
these run fast and without any dependency on training/data/processed/
existing on disk. No network access.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import v1_schema
from training import config, model_registry, train


def _synthetic_panel(n_train=40, n_val=15, n_test=15, seed=0):
    """
    A small but structurally realistic V1 panel: identifiers, all
    V1_FEATURE_NAMES (with some genuine NaNs in a couple of columns, to
    exercise native NaN handling), PTS, PRIOR_GAMES_THIS_SEASON >= 5 for
    every row (all training-eligible), and SPLIT labels.
    """
    rng = np.random.RandomState(seed)
    rows = []
    specs = [
        ("2023-24", "train", n_train),
        ("2025-26", "validation", n_val),
        ("2025-26", "test", n_test),
    ]
    game_counter = 0
    for season, split, n in specs:
        for i in range(n):
            game_counter += 1
            player_avg = float(rng.uniform(5, 25))
            pts = max(0.0, player_avg + rng.normal(0, 5))
            row = {
                "PLAYER_ID": 100 + (i % 5),
                "PLAYER_NAME": f"Player {100 + (i % 5)}",
                "GAME_ID": f"G{game_counter}",
                "GAME_DATE": pd.Timestamp("2023-10-24")
                + pd.Timedelta(days=game_counter),
                "SEASON": season,
                "TEAM_ABBREVIATION": "AAA",
                "OPPONENT_ABBREVIATION": "BBB",
                "PTS": pts,
                "player_avg_pts": player_avg,
                "player_avg_pts_sq": player_avg**2,
                "season_minutes_avg": float(rng.uniform(15, 35)),
                "recent_minutes_avg": float(rng.uniform(15, 35)),
                "home_game": int(i % 2),
                "days_rest": float(rng.choice([1, 2, 3])),
                "is_back_to_back": int(i % 7 == 0),
                "last3_pts": player_avg + rng.normal(0, 2),
                "last5_pts": player_avg + rng.normal(0, 2),
                # Deliberately sparse -- exercises native NaN handling and
                # the missing-value-rate reality of the real panel.
                "last10_pts": (player_avg + rng.normal(0, 2)) if i % 3 != 0 else np.nan,
                "last20_pts": (player_avg + rng.normal(0, 2)) if i % 4 != 0 else np.nan,
                "last5_fga": float(rng.uniform(5, 20)),
                "last5_fta": float(rng.uniform(1, 8)),
                "last5_3pa": float(rng.uniform(0, 10)),
                "last5_minutes": float(rng.uniform(15, 35)),
                "last5_gmsc": float(rng.uniform(5, 25)),
                "last5_usage_proxy": float(rng.uniform(8, 25)),
                "minutes_volatility": float(rng.uniform(0, 6)),
                "points_volatility": float(rng.uniform(0, 8)),
                "opponent_points_allowed_per_game": float(rng.uniform(105, 120)),
                "opponent_points_allowed_last5": float(rng.uniform(105, 120)),
                "opponent_pace": float(rng.uniform(95, 105)),
                "opponent_defensive_rating": float(rng.uniform(105, 120)),
                "PRIOR_GAMES_THIS_SEASON": 10,
                "TRAINING_ELIGIBLE": 1,
                "SPLIT": split,
            }
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_panel():
    return _synthetic_panel()


@pytest.fixture(autouse=True)
def _fast_model_params(monkeypatch):
    """Fewer trees for test speed -- structurally identical config, just
    smaller, so tests run in milliseconds instead of seconds."""
    monkeypatch.setattr(
        config,
        "V1_BASELINE_MODEL_PARAMS",
        {
            "n_estimators": 20,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "random_state": 42,
        },
    )


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_REGISTRY_DIR", tmp_path / "registry")


def test_native_nan_support_is_verified_without_raising():
    train.verify_native_nan_support()  # must not raise in this environment


def test_validate_feature_contract_passes_for_a_valid_panel(synthetic_panel):
    train.validate_feature_contract(synthetic_panel)  # must not raise


def test_validate_feature_contract_fails_loudly_on_missing_feature(synthetic_panel):
    broken = synthetic_panel.drop(columns=["last5_pts"])
    with pytest.raises(ValueError, match="last5_pts"):
        train.validate_feature_contract(broken)


def test_validate_feature_contract_fails_on_forbidden_sportsbook_column(
    synthetic_panel,
):
    broken = synthetic_panel.copy()
    broken["closing_line"] = 20.5
    with pytest.raises(ValueError, match="closing_line"):
        train.validate_feature_contract(broken)


def test_select_feature_matrix_uses_exact_schema_order_and_ignores_extra_columns(
    synthetic_panel,
):
    df_with_extra = synthetic_panel.copy()
    df_with_extra["sportsbook_line"] = 20.5  # should simply be ignored, not selected
    X = train.select_feature_matrix(df_with_extra)
    assert list(X.columns) == list(v1_schema.V1_FEATURE_NAMES)
    assert "sportsbook_line" not in X.columns


def test_load_eligible_splits_isolates_train_validation_test(synthetic_panel):
    train_df, val_df, test_df = train.load_eligible_splits(synthetic_panel)

    assert set(train_df["SPLIT"]) == {"train"}
    assert set(val_df["SPLIT"]) == {"validation"}
    assert set(test_df["SPLIT"]) == {"test"}
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_panel)


def test_ineligible_rows_are_excluded_from_every_split(synthetic_panel):
    panel = synthetic_panel.copy()
    panel.loc[panel.index[0], "TRAINING_ELIGIBLE"] = 0
    train_df, val_df, test_df = train.load_eligible_splits(panel)
    excluded_id = synthetic_panel.loc[synthetic_panel.index[0], "GAME_ID"]
    assert excluded_id not in set(train_df["GAME_ID"])
    assert excluded_id not in set(val_df["GAME_ID"])
    assert excluded_id not in set(test_df["GAME_ID"])


def test_validation_and_test_rows_are_never_used_in_fit(synthetic_panel, monkeypatch):
    """Directly proves .fit() only ever sees train rows: monkeypatch
    RandomForestRegressor.fit to record what it was called with."""
    seen = {}
    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def spy_fit(self, X, y, *args, **kwargs):
        seen["n_rows"] = len(X)
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", spy_fit)

    train_df, val_df, test_df = train.load_eligible_splits(synthetic_panel)
    train.train_baseline_model(train_df)

    assert seen["n_rows"] == len(train_df)
    assert seen["n_rows"] != len(train_df) + len(val_df) + len(test_df)


def test_predictions_are_deterministic_under_fixed_seed(synthetic_panel):
    train_df, val_df, _ = train.load_eligible_splits(synthetic_panel)

    model_1 = train.train_baseline_model(train_df)
    model_2 = train.train_baseline_model(train_df)

    X_val = train.select_feature_matrix(val_df)
    preds_1 = model_1.predict(X_val)
    preds_2 = model_2.predict(X_val)

    np.testing.assert_array_equal(preds_1, preds_2)


def test_feature_importances_align_with_schema_names_and_order(synthetic_panel):
    train_df, _, _ = train.load_eligible_splits(synthetic_panel)
    model = train.train_baseline_model(train_df)

    assert list(model.feature_names_in_) == list(v1_schema.V1_FEATURE_NAMES)
    assert len(model.feature_importances_) == len(v1_schema.V1_FEATURE_NAMES)


def test_metadata_includes_required_provenance_fields(synthetic_panel, tmp_path):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)

    result = train.run_training(panel_path=panel_path, version="v_test_metadata")
    metadata = result["metadata"]

    for key in (
        "model_type",
        "hyperparameters",
        "target",
        "feature_schema",
        "missing_value_policy",
        "split_boundary",
        "eligible_row_counts",
        "data_date_ranges",
        "git_commit",
        "python_version",
        "sklearn_version",
        "pandas_version",
        "numpy_version",
        "training_start_utc",
        "training_end_utc",
    ):
        assert key in metadata, f"metadata missing required key: {key}"

    assert metadata["target"] == "PTS"
    assert metadata["feature_schema"] == list(v1_schema.V1_FEATURE_NAMES)
    assert metadata["eligible_row_counts"]["train"] > 0


def test_residual_artifact_schema_is_correct(synthetic_panel, tmp_path):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)

    result = train.run_training(panel_path=panel_path, version="v_test_residuals")
    residuals = result["residuals_df"]

    required = (
        "PLAYER_ID",
        "PLAYER_NAME",
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SPLIT",
        "actual_pts",
        "predicted_pts",
        "residual",
        "absolute_error",
    )
    for col in required:
        assert col in residuals.columns
    assert set(residuals["SPLIT"]) == {"validation", "test"}
    assert "train" not in set(residuals["SPLIT"])


def test_artifact_save_and_load_round_trip_reproduces_predictions(
    synthetic_panel, tmp_path
):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)

    result = train.run_training(panel_path=panel_path, version="v_test_roundtrip")

    reloaded_model = model_registry.load_model("v_test_roundtrip")
    _, val_df, _ = train.load_eligible_splits(synthetic_panel)
    X_val = train.select_feature_matrix(val_df)

    original_preds = result["model"].predict(X_val)
    reloaded_preds = reloaded_model.predict(X_val)
    np.testing.assert_array_equal(original_preds, reloaded_preds)

    reloaded_metadata = model_registry.load_metadata("v_test_roundtrip")
    reloaded_metrics = model_registry.load_metrics("v_test_roundtrip")
    reloaded_importances = model_registry.load_feature_importances("v_test_roundtrip")
    reloaded_residuals = model_registry.load_residuals("v_test_roundtrip")

    assert reloaded_metadata["target"] == "PTS"
    assert reloaded_metrics["validation"] is not None
    assert set(reloaded_importances.keys()) == set(v1_schema.V1_FEATURE_NAMES)
    assert len(reloaded_residuals) > 0


def test_current_pointer_is_written_and_points_to_the_trained_version(
    synthetic_panel, tmp_path
):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)

    train.run_training(panel_path=panel_path, version="v_test_pointer")

    assert model_registry.get_current_version() == "v_test_pointer"


def test_run_training_does_not_touch_the_legacy_model_artifact(
    synthetic_panel, tmp_path
):
    legacy_path = config.REPO_ROOT / "models" / "points_regression.pkl"
    assert legacy_path.exists()
    before_mtime = legacy_path.stat().st_mtime
    before_size = legacy_path.stat().st_size

    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)
    train.run_training(panel_path=panel_path, version="v_test_legacy_untouched")

    assert legacy_path.stat().st_mtime == before_mtime
    assert legacy_path.stat().st_size == before_size


def test_simple_baselines_are_computed_for_validation_and_test(
    synthetic_panel, tmp_path
):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)

    result = train.run_training(panel_path=panel_path, version="v_test_baselines")
    baselines = result["metrics"]["simple_baselines"]

    for feature in ("player_avg_pts", "last5_pts", "last10_pts"):
        assert feature in baselines
        assert "validation" in baselines[feature]
        assert "test" in baselines[feature]
        assert "mae" in baselines[feature]["validation"]


def test_save_artifacts_refuses_overwrite_without_flag(synthetic_panel, tmp_path):
    panel_path = tmp_path / "panel.parquet"
    synthetic_panel.to_parquet(panel_path, index=False)

    train.run_training(panel_path=panel_path, version="v_test_overwrite")
    with pytest.raises(FileExistsError):
        train.run_training(
            panel_path=panel_path, version="v_test_overwrite", overwrite=False
        )

    # overwrite=True must succeed.
    train.run_training(
        panel_path=panel_path, version="v_test_overwrite", overwrite=True
    )
