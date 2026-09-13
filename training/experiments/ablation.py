"""
Stage A/B feature-group ablation evaluation. Reuses
training.experiments.tuning's fit/score machinery, restricted to the
predeclared ABLATION_SPECS in training/experiments/feature_groups.py --
never a brute-force search over arbitrary feature subsets.
"""

from __future__ import annotations

import logging

from training.experiments.cv import TemporalFold
from training.experiments.guardrails import reject_test_rows
from training.experiments.tuning import (
    evaluate_candidate_cv,
    evaluate_on_official_validation,
)

logger = logging.getLogger(__name__)


def run_ablation_cv(
    params: dict, cv_folds: list[TemporalFold], ablation_specs: list[dict]
) -> list[dict]:
    """
    Evaluates every predeclared ablation spec against the same temporal
    CV folds, using ONE fixed hyperparameter configuration (the winner
    of the Stage A hyperparameter search), so ablation results isolate
    the effect of the feature set rather than confounding it with
    hyperparameters changing too. Ranked by mean_cv_mae ascending, same
    as tuning.run_search().
    """
    results = []
    for spec in ablation_specs:
        cv_result = evaluate_candidate_cv(params, cv_folds, spec["feature_names"])
        result = {
            "ablation_id": spec["ablation_id"],
            "description": spec["description"],
            "feature_names": list(spec["feature_names"]),
            "n_features": len(spec["feature_names"]),
            **cv_result,
        }
        results.append(result)
        logger.info(
            "ablation=%s n_features=%d mean_cv_mae=%.4f std_cv_mae=%.4f",
            spec["ablation_id"],
            len(spec["feature_names"]),
            result["mean_cv_mae"],
            result["std_cv_mae"],
        )
    results.sort(key=lambda r: (r["mean_cv_mae"], r["mean_cv_rmse"]))
    return results


def evaluate_ablation_on_official_validation(
    params, train_df, val_df, ablation_spec
) -> dict:
    reject_test_rows(train_df, "evaluate_ablation_on_official_validation (train)")
    reject_test_rows(val_df, "evaluate_ablation_on_official_validation (validation)")
    result = evaluate_on_official_validation(
        params, train_df, val_df, ablation_spec["feature_names"]
    )
    return {
        "ablation_id": ablation_spec["ablation_id"],
        "feature_names": list(ablation_spec["feature_names"]),
        **result,
    }
