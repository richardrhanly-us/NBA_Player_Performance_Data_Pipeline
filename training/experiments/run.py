"""
Step 5 orchestration: loads the V1 panel, restricts fitting to the
official train split, builds temporal CV folds, runs the predeclared RF
hyperparameter search, runs the predeclared feature ablations, evaluates
a narrowed candidate set on the official validation split, runs paired
comparisons against the frozen Step 4 model and the simple baselines,
and persists everything under training/experiments/results/<run_id>/.

NEVER loads/evaluates the official test split beyond logging its row
count -- see training/experiments/__init__.py and training/splits.py.

Does not write to models/registry/, does not update
models/registry/CURRENT_V1, does not retrain or replace
models/points_regression.pkl.

Usage:
    python -m training.experiments
    python -m training.experiments --panel training/data/processed/v1_panel.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import v1_schema
from training import config, model_registry
from training import train as production_train
from training.experiments import (
    ablation,
    feature_groups,
    reporting,
    search_space,
    tuning,
)
from training.experiments.cv import build_temporal_cv_folds
from training.experiments.paired_comparison import paired_validation_comparison

logger = logging.getLogger(__name__)

# The frozen Step 4 registry entry -- read-only reference for the paired
# comparison and the "did we beat Step 4 by >= 0.05 MAE" check. Never
# written to, never re-pointed via CURRENT_V1.
STEP4_MODEL_VERSION = "v1_baseline_20260913_183045"

MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD = reporting.MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD
BASELINE_FEATURES = production_train.BASELINE_FEATURES


def load_step4_model():
    """Loads the frozen Step 4 model as a read-only comparison baseline.
    Never refits, never overwrites, never touches CURRENT_V1."""
    return model_registry.load_model(STEP4_MODEL_VERSION)


def run_experiment(panel_path=None, run_id=None) -> dict:
    run_start = datetime.now(timezone.utc)

    panel_path = Path(panel_path) if panel_path else config.DEFAULT_PANEL_PATH
    logger.info("Loading processed panel from %s", panel_path)
    panel_df = pd.read_parquet(panel_path)
    logger.info(
        "Panel loaded: %d rows, %d columns", len(panel_df), len(panel_df.columns)
    )

    production_train.validate_feature_contract(panel_df)
    logger.info("Feature contract validated against V1_FEATURE_NAMES.")

    train_df, val_df, test_df = production_train.load_eligible_splits(panel_df)
    test_row_count = len(test_df)
    del test_df  # structurally cannot be used below by accident
    logger.info(
        "Eligible rows -- train: %d, validation: %d, test: %d "
        "(TEST EXCLUDED: row count logged only, DataFrame discarded, "
        "not used anywhere below)",
        len(train_df),
        len(val_df),
        test_row_count,
    )

    # ---------------- Stage A: temporal CV inside train only ----------------
    cv_folds = build_temporal_cv_folds(train_df)
    cv_fold_boundaries = [
        {
            "fold": f.fold_index,
            "train_date_min": str(f.train_date_min),
            "train_date_max": str(f.train_date_max),
            "val_date_min": str(f.val_date_min),
            "val_date_max": str(f.val_date_max),
            "n_train": f.n_train,
            "n_val": f.n_val,
        }
        for f in cv_folds
    ]
    for b in cv_fold_boundaries:
        logger.info("CV fold %s", b)

    logger.info(
        "Running RF hyperparameter search: %d predeclared candidates",
        len(search_space.RF_SEARCH_SPACE),
    )
    search_results = tuning.run_search(search_space.RF_SEARCH_SPACE, cv_folds)
    best_hp_candidate = search_results[0]
    logger.info(
        "Best CV candidate: %s mean_cv_mae=%.4f std_cv_mae=%.4f",
        best_hp_candidate["candidate_id"],
        best_hp_candidate["mean_cv_mae"],
        best_hp_candidate["std_cv_mae"],
    )

    # ---------------- Feature ablation using the best CV hyperparameters ----------------
    logger.info(
        "Running %d predeclared feature ablations", len(feature_groups.ABLATION_SPECS)
    )
    ablation_results = ablation.run_ablation_cv(
        best_hp_candidate["params"], cv_folds, feature_groups.ABLATION_SPECS
    )
    best_ablation = ablation_results[0]
    logger.info(
        "Best CV ablation: %s (n_features=%d) mean_cv_mae=%.4f",
        best_ablation["ablation_id"],
        best_ablation["n_features"],
        best_ablation["mean_cv_mae"],
    )

    # ---------------- Stage B: official validation, narrowed candidates only ----------------
    full_v1_ablation = next(
        a for a in feature_groups.ABLATION_SPECS if a["ablation_id"] == "A_full_v1"
    )
    candidates_for_validation = [
        (
            "best_hp_full_features",
            best_hp_candidate["params"],
            full_v1_ablation["feature_names"],
        ),
    ]
    if best_ablation["ablation_id"] != "A_full_v1":
        best_ablation_spec = next(
            a
            for a in feature_groups.ABLATION_SPECS
            if a["ablation_id"] == best_ablation["ablation_id"]
        )
        candidates_for_validation.append(
            (
                "best_hp_best_ablation",
                best_hp_candidate["params"],
                best_ablation_spec["feature_names"],
            )
        )

    official_validation_results = {}
    for name, params, feature_names in candidates_for_validation:
        result = tuning.evaluate_on_official_validation(
            params, train_df, val_df, feature_names
        )
        official_validation_results[name] = {
            "params": params,
            "feature_names": list(feature_names),
            "metrics": result["metrics"],
            "model": result["model"],
        }
        logger.info(
            "Official validation -- %s: MAE=%.4f RMSE=%.4f R2=%.4f bias=%.4f",
            name,
            result["metrics"]["mae"],
            result["metrics"]["rmse"],
            result["metrics"]["r2"],
            result["metrics"]["bias_mean_pred_minus_actual"],
        )

    final_name = min(
        official_validation_results,
        key=lambda k: official_validation_results[k]["metrics"]["mae"],
    )
    final = official_validation_results[final_name]
    logger.info(
        "Selected Step 5 final candidate: %s (official validation MAE=%.4f)",
        final_name,
        final["metrics"]["mae"],
    )

    # ---------------- Paired comparison on official validation ----------------
    final_model = final["model"]
    final_feature_names = final["feature_names"]
    X_val_final = tuning.select_feature_subset(val_df, final_feature_names)
    final_val_pred = final_model.predict(X_val_final)

    step4_model = load_step4_model()
    X_val_full = tuning.select_feature_subset(val_df, v1_schema.V1_FEATURE_NAMES)
    step4_val_pred = step4_model.predict(X_val_full)

    baselines = {"step4_v1_rf": step4_val_pred}
    for feature in BASELINE_FEATURES:
        baselines[feature] = val_df[feature].to_numpy(dtype=float)

    # player_avg_pts/last5_pts/last10_pts can contain NaN for some rows
    # (see Step 4 missingness report). For each baseline, the paired
    # comparison restricts itself to rows where that baseline has a real
    # value -- exactly like evaluate.compute_simple_baseline_metrics --
    # rather than silently propagating NaN into the diff. Candidate and
    # baseline are always compared on that same restricted row set.
    paired_results = {}
    for name, baseline_pred in baselines.items():
        baseline_pred = np.asarray(baseline_pred, dtype=float)
        valid_mask = ~np.isnan(baseline_pred)
        sub_val_df = val_df.loc[valid_mask]
        comparison = paired_validation_comparison(
            sub_val_df, final_val_pred[valid_mask], {name: baseline_pred[valid_mask]}
        )
        paired_results[name] = comparison[name]
        paired_results[name]["rows_with_missing_baseline_excluded"] = int(
            (~valid_mask).sum()
        )
        logger.info(
            "Paired comparison vs %s: mean_diff=%.4f wins=%.1f%% ci=[%.4f, %.4f]",
            name,
            paired_results[name]["mean_diff"],
            paired_results[name]["fraction_candidate_wins"] * 100,
            paired_results[name]["bootstrap_ci"]["ci_lower"],
            paired_results[name]["bootstrap_ci"]["ci_upper"],
        )

    # ---------------- Feature importances vs Step 4 ----------------
    final_importances = dict(
        sorted(
            zip(
                final_feature_names,
                (float(x) for x in final_model.feature_importances_),
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    step4_importances = model_registry.load_feature_importances(STEP4_MODEL_VERSION)

    # ---------------- Threshold check (predeclared BEFORE results) ----------------
    step4_official_val_mae = model_registry.load_metrics(STEP4_MODEL_VERSION)[
        "validation"
    ]["mae"]
    mae_improvement_over_step4 = step4_official_val_mae - final["metrics"]["mae"]
    meaningful_improvement_achieved = (
        mae_improvement_over_step4 >= MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD
    )
    logger.info(
        "Step 4 official validation MAE=%.4f, Step 5 final=%.4f, "
        "improvement=%.4f (threshold=%.2f, achieved=%s)",
        step4_official_val_mae,
        final["metrics"]["mae"],
        mae_improvement_over_step4,
        MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD,
        meaningful_improvement_achieved,
    )

    run_end = datetime.now(timezone.utc)

    run_id = run_id or reporting.default_run_id()
    selected_ablation_id = (
        "A_full_v1"
        if final_name == "best_hp_full_features"
        else best_ablation["ablation_id"]
    )
    experiment_metadata = reporting.build_experiment_metadata(
        panel_path=panel_path,
        train_row_count=len(train_df),
        validation_row_count=len(val_df),
        test_row_count_excluded=test_row_count,
        cv_fold_boundaries=cv_fold_boundaries,
        candidate_count=len(search_space.RF_SEARCH_SPACE),
        ablation_count=len(feature_groups.ABLATION_SPECS),
        selected_params=final["params"],
        selected_ablation_id=selected_ablation_id,
        run_start=run_start,
        run_end=run_end,
    )
    experiment_metadata["step4_official_validation_mae"] = step4_official_val_mae
    experiment_metadata["final_candidate_name"] = final_name
    experiment_metadata["final_official_validation_mae"] = final["metrics"]["mae"]
    experiment_metadata["mae_improvement_over_step4"] = mae_improvement_over_step4
    experiment_metadata["meaningful_improvement_achieved"] = (
        meaningful_improvement_achieved
    )
    experiment_metadata["final_feature_importances"] = final_importances
    experiment_metadata["step4_feature_importances"] = step4_importances

    validation_comparison_payload = {
        "candidates_evaluated_on_official_validation": {
            name: {
                "params": v["params"],
                "feature_names": v["feature_names"],
                "metrics": v["metrics"],
            }
            for name, v in official_validation_results.items()
        },
        "final_candidate": final_name,
        "paired_comparison": paired_results,
    }

    artifact_dir = reporting.save_experiment_results(
        run_id,
        search_results=search_results,
        ablation_results=ablation_results,
        validation_comparison=validation_comparison_payload,
        experiment_metadata=experiment_metadata,
        selected_model=final_model,
    )

    logger.info("=== Step 5 experiment complete: %s ===", run_id)
    logger.info(
        "Test split was NOT used at any point in this run "
        "(excluded row count logged only: %d). Output: %s",
        test_row_count,
        artifact_dir,
    )

    return {
        "run_id": run_id,
        "artifact_dir": artifact_dir,
        "search_results": search_results,
        "ablation_results": ablation_results,
        "official_validation_results": official_validation_results,
        "paired_results": paired_results,
        "experiment_metadata": experiment_metadata,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.experiments",
        description=(
            "Step 5: RandomForest hyperparameter tuning + feature ablation, "
            "train/validation only. Never evaluates the frozen test split."
        ),
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help=f"Path to the processed V1 panel (default: {config.DEFAULT_PANEL_PATH}).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Experiment run id (default: step5_experiment_<UTC timestamp>).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    result = run_experiment(panel_path=args.panel, run_id=args.run_id)
    logger.info("Results written to %s", result["artifact_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
