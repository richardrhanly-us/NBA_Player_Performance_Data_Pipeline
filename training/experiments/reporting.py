"""
Provenance metadata assembly and lightweight persistence for Step 5
experiment runs. Deliberately separate from training/model_registry.py
(the production-training-side registry): Step 5 writes under
training/experiments/results/<run_id>/, never under models/registry/,
and never touches models/registry/CURRENT_V1.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.features import v1_schema
from training import config

logger = logging.getLogger(__name__)

EXPERIMENTS_ROOT = config.REPO_ROOT / "training" / "experiments" / "results"

# Predeclared BEFORE any Step 5 result was observed -- see the Step 5
# report's "meaningful improvement" section. Not a hard production rule,
# just a bar for calling a result a clearly meaningful improvement
# rather than overselling microscopic movement. Do not change this
# value after seeing results.
MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD = 0.05


def _git_commit_hash() -> str | None:
    """Best-effort provenance capture -- must never block the experiment
    run if git isn't available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.REPO_ROOT,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug(
            "git rev-parse HEAD exited %d; recording no commit hash", result.returncode
        )
    except Exception as exc:  # noqa: BLE001 - metadata capture must never fail the run
        logger.debug(
            "Could not determine git commit hash: %s: %s", type(exc).__name__, exc
        )
    return None


def default_run_id() -> str:
    return "step5_experiment_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_experiment_metadata(
    *,
    panel_path,
    train_row_count,
    validation_row_count,
    test_row_count_excluded,
    cv_fold_boundaries,
    candidate_count,
    ablation_count,
    selected_params,
    selected_ablation_id,
    run_start,
    run_end,
) -> dict:
    return {
        "step": "5",
        "description": (
            "RandomForest hyperparameter tuning + feature-group ablation, "
            "train/validation only. Official test split frozen and unused."
        ),
        "git_commit": _git_commit_hash(),
        "panel_path": str(panel_path),
        "feature_schema": list(v1_schema.V1_FEATURE_NAMES),
        "train_row_count": train_row_count,
        "validation_row_count": validation_row_count,
        "test_row_count_excluded_not_used": test_row_count_excluded,
        "temporal_cv_fold_boundaries": cv_fold_boundaries,
        "search_space_size": candidate_count,
        "ablation_count": ablation_count,
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "selection_metric": (
            "mean_cv_mae (temporal CV, Stage A), then official validation "
            "MAE (Stage B) for the narrowed candidate set"
        ),
        "predeclared_meaningful_improvement_mae_threshold": (
            MEANINGFUL_IMPROVEMENT_MAE_THRESHOLD
        ),
        "selected_hyperparameters": selected_params,
        "selected_ablation_id": selected_ablation_id,
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "run_duration_seconds": (run_end - run_start).total_seconds(),
    }


def save_experiment_results(
    run_id: str,
    *,
    search_results,
    ablation_results,
    validation_comparison,
    experiment_metadata,
    selected_model=None,
) -> Path:
    out_dir = EXPERIMENTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    search_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "fold_results"} for r in search_results]
    )
    search_df.to_csv(out_dir / "search_results.csv", index=False)
    with open(out_dir / "search_results.json", "w", encoding="utf-8") as f:
        json.dump(search_results, f, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "search_results.csv")

    ablation_df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("fold_results", "feature_names")}
            for r in ablation_results
        ]
    )
    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "ablation_results.csv")

    with open(out_dir / "validation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(validation_comparison, f, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "validation_comparison.json")

    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "experiment_metadata.json")

    if selected_model is not None:
        import joblib

        model_path = out_dir / "selected_candidate_model.pkl"
        joblib.dump(selected_model, model_path)
        logger.info("Wrote %s", model_path)

    logger.info("Step 5 experiment artifacts written to %s", out_dir)
    return out_dir
