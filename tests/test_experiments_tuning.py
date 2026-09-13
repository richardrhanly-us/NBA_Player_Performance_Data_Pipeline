"""
Tests for training/experiments/tuning.py and search_space.py: the Step 4
baseline configuration is present in the candidate set, the search is
deterministic, and candidate ranking uses MAE as the primary metric. No
network access.
"""

from training.experiments import search_space
from training.experiments.cv import build_temporal_cv_folds
from training.experiments.tuning import evaluate_candidate_cv, run_search


def _train_only(panel):
    return panel[panel["SPLIT"] == "train"].reset_index(drop=True)


def _fast_search_space():
    """A tiny 3-candidate subset of the real search space, with few
    trees, so these tests run in well under a second."""
    fast = []
    for candidate in search_space.RF_SEARCH_SPACE[:3]:
        params = dict(candidate["params"])
        params["n_estimators"] = 10
        params["n_jobs"] = 1
        fast.append({**candidate, "params": params})
    return fast


def test_step4_baseline_configuration_is_present_in_the_search_space():
    ids = [c["candidate_id"] for c in search_space.RF_SEARCH_SPACE]
    assert "step4_baseline" in ids
    baseline = next(
        c for c in search_space.RF_SEARCH_SPACE if c["candidate_id"] == "step4_baseline"
    )
    assert baseline["params"]["n_estimators"] == 400
    assert baseline["params"]["max_depth"] == 10
    assert baseline["params"]["min_samples_leaf"] == 5
    assert baseline["params"]["min_samples_split"] == 10
    assert baseline["params"]["max_features"] == "sqrt"
    assert baseline["params"]["random_state"] == 42


def test_search_space_size_is_within_the_predeclared_15_to_30_range():
    assert 15 <= len(search_space.RF_SEARCH_SPACE) <= 30


def test_search_space_has_no_duplicate_candidate_ids():
    ids = [c["candidate_id"] for c in search_space.RF_SEARCH_SPACE]
    assert len(ids) == len(set(ids))


def test_search_space_every_candidate_uses_fixed_random_state_42():
    for candidate in search_space.RF_SEARCH_SPACE:
        assert candidate["params"]["random_state"] == 42


def test_candidate_search_is_deterministic(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)
    fast_space = _fast_search_space()

    results_a = run_search(fast_space, folds)
    results_b = run_search(fast_space, folds)

    ids_a = [r["candidate_id"] for r in results_a]
    ids_b = [r["candidate_id"] for r in results_b]
    assert ids_a == ids_b
    for ra, rb in zip(results_a, results_b):
        assert ra["mean_cv_mae"] == rb["mean_cv_mae"]


def test_candidate_ranking_uses_mae_as_primary_metric(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)
    fast_space = _fast_search_space()

    results = run_search(fast_space, folds)

    maes = [r["mean_cv_mae"] for r in results]
    assert maes == sorted(maes)  # ranked ascending by mean_cv_mae, nothing else


def test_evaluate_candidate_cv_reports_required_stability_fields(experiments_panel):
    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)
    params = dict(
        search_space.STEP4_BASELINE_PARAMS, random_state=42, n_jobs=1, n_estimators=10
    )

    result = evaluate_candidate_cv(params, folds)

    for key in ("mean_cv_mae", "std_cv_mae", "mean_cv_rmse", "mean_cv_bias"):
        assert key in result
    assert len(result["fold_results"]) == len(folds)


def test_evaluate_candidate_cv_never_sees_official_validation_rows(experiments_panel):
    """The folds passed in are carved from train only, and
    evaluate_candidate_cv never receives the official val_df at all --
    prove no validation GAME_ID leaks into any fold's frames."""
    val_df = experiments_panel[experiments_panel["SPLIT"] == "validation"]
    val_game_ids = set(val_df["GAME_ID"])

    train_df = _train_only(experiments_panel)
    folds = build_temporal_cv_folds(train_df)
    for fold in folds:
        assert val_game_ids.isdisjoint(set(fold.train_df["GAME_ID"]))
        assert val_game_ids.isdisjoint(set(fold.val_df["GAME_ID"]))
