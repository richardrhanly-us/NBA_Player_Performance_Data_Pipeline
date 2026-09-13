"""
Step 5 experimentation layer: controlled RandomForestRegressor
hyperparameter tuning and feature-group ablation, developed against the
official V1 train/validation splits only.

This package is deliberately separate from the production training path
(training/train.py, training/model_registry.py, models/registry/):
nothing here writes to models/registry/, nothing here updates
models/registry/CURRENT_V1, and nothing here retrains or replaces
models/points_regression.pkl. Experiment output lives under
training/experiments/results/<run_id>/ -- see training/experiments/reporting.py.

-------------------------------------------------------------------------
THE OFFICIAL TEST SPLIT IS FROZEN (see training/splits.py) as of the
Step 4 V1 baseline -- its results have already been observed. Every
function in this package that accepts a DataFrame carrying a SPLIT
column rejects rows from the frozen test split
(training.experiments.guardrails.reject_test_rows). The orchestration
entrypoint (training/experiments/run.py) loads the official test split
only long enough to log its row count, then discards the DataFrame
itself -- it is never passed into any tuning, ablation, or comparison
function.
-------------------------------------------------------------------------

Modules:
    guardrails.py         -- reject_test_rows(), used everywhere below
    cv.py                 -- Stage A: expanding-window temporal CV folds
                              built from the official train split only
    search_space.py       -- predeclared RandomForestRegressor
                              hyperparameter candidates (not a full
                              Cartesian product)
    feature_groups.py      -- predeclared, interpretable V1 feature
                              groups and the ablation specs built from them
    tuning.py               -- fits/scores one candidate against CV
                              folds (Stage A) or the official validation
                              split (Stage B)
    ablation.py             -- same, restricted to the predeclared
                              feature-group ablations
    paired_comparison.py    -- per-row paired error stats + deterministic
                              bootstrap CI, official validation split only
    reporting.py            -- provenance metadata + result persistence
    run.py                  -- orchestration entrypoint
                              (`python -m training.experiments`)
"""
