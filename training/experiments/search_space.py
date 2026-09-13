"""
Predeclared RandomForestRegressor hyperparameter search space for Step 5
temporal-CV tuning.

This is a deliberately small, hand-curated set of candidates -- NOT the
full Cartesian product of the dimensions below (4 x 4 x 3 x 3 x 2 = 288
combinations, far more than a "modest," interpretable search). The
design is a one-at-a-time sensitivity sweep around the frozen Step 4
configuration (vary exactly one hyperparameter at a time, all others
held at the Step 4 value), plus a handful of combined "corner"
candidates that pair multiple simultaneous changes worth checking
together.

Declared once, here, BEFORE any candidate is evaluated. See
training/experiments/run.py, which iterates this list in the fixed order
below and does not add, remove, or reorder candidates based on results.

Dimensions considered (per the Step 5 brief):
    max_depth:         [6, 10, 14, None]
    min_samples_leaf:  [3, 5, 10, 20]
    min_samples_split: [5, 10, 20]
    max_features:      ["sqrt", 0.5, 1.0]
    n_estimators:      [300, 600]  (in addition to the Step 4 value, 400)

random_state=42 on every candidate -- never varied. n_jobs=-1 on every
candidate too: this only affects fit speed (parallelism across trees),
never the fitted model's structure or predictions (each tree's own
random state is derived from random_state independent of n_jobs), so it
is not treated as a tuned hyperparameter and does not appear in the
dimensions above.
"""

from __future__ import annotations

_FIXED = {"random_state": 42, "n_jobs": -1}

STEP4_BASELINE_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": "sqrt",
}


def _candidate(candidate_id: str, note: str, **overrides) -> dict:
    params = dict(STEP4_BASELINE_PARAMS)
    params.update(overrides)
    params.update(_FIXED)
    return {"candidate_id": candidate_id, "note": note, "params": params}


RF_SEARCH_SPACE: list[dict] = [
    _candidate(
        "step4_baseline",
        "Exact frozen Step 4 configuration -- internal CV reference point.",
    ),
    # --- one-at-a-time: max_depth (baseline value 10 is step4_baseline) ---
    _candidate(
        "max_depth_6", "Shallower trees, all else at Step 4 values.", max_depth=6
    ),
    _candidate(
        "max_depth_14", "Deeper trees, all else at Step 4 values.", max_depth=14
    ),
    _candidate(
        "max_depth_none",
        "Unlimited depth, all else at Step 4 values.",
        max_depth=None,
    ),
    # --- one-at-a-time: min_samples_leaf (baseline value 5 is step4_baseline) ---
    _candidate("min_leaf_3", "Less leaf regularization.", min_samples_leaf=3),
    _candidate("min_leaf_10", "More leaf regularization.", min_samples_leaf=10),
    _candidate("min_leaf_20", "Much more leaf regularization.", min_samples_leaf=20),
    # --- one-at-a-time: min_samples_split (baseline value 10 is step4_baseline) ---
    _candidate("min_split_5", "Less split regularization.", min_samples_split=5),
    _candidate("min_split_20", "More split regularization.", min_samples_split=20),
    # --- one-at-a-time: max_features (baseline value "sqrt" is step4_baseline) ---
    _candidate(
        "max_features_half",
        "Half of features considered per split.",
        max_features=0.5,
    ),
    _candidate(
        "max_features_all",
        "All features considered per split (no per-split subsampling).",
        max_features=1.0,
    ),
    # --- one-at-a-time: n_estimators (baseline value 400 is step4_baseline) ---
    _candidate("n_estimators_300", "Fewer trees.", n_estimators=300),
    _candidate("n_estimators_600", "More trees.", n_estimators=600),
    # --- combined "corner" candidates: plausible joint changes worth checking together ---
    _candidate(
        "corner_deeper_more_trees",
        "Deeper trees + more trees, other regularization unchanged.",
        max_depth=14,
        n_estimators=600,
    ),
    _candidate(
        "corner_shallow_heavily_regularized",
        "Shallower trees + heavier leaf/split regularization.",
        max_depth=6,
        min_samples_leaf=20,
        min_samples_split=20,
    ),
    _candidate(
        "corner_more_features_per_split",
        "All features per split + more trees.",
        max_features=1.0,
        n_estimators=600,
    ),
    _candidate(
        "corner_unlimited_depth_regularized",
        "Unlimited depth but compensated with heavier leaf/split "
        "regularization and fewer trees.",
        max_depth=None,
        min_samples_leaf=10,
        min_samples_split=20,
        n_estimators=300,
    ),
    _candidate(
        "corner_high_capacity",
        "Deeper + more trees + fewer features per split + lighter leaf "
        "regularization (highest-capacity, most-randomized corner in "
        "this search).",
        max_depth=14,
        min_samples_leaf=3,
        min_samples_split=5,
        max_features=0.5,
        n_estimators=600,
    ),
]
