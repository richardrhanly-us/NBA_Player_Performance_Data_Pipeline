"""
Guards that make it structurally hard for Step 5 experiment code to
accidentally fit or score against the frozen official test split. See
training/splits.py for why the test split is frozen as of the Step 4 V1
baseline, and training/experiments/__init__.py for how this guard is
used throughout the package.
"""

from __future__ import annotations

import pandas as pd


def reject_test_rows(df: pd.DataFrame, context: str) -> None:
    """
    Raises ValueError if `df` contains any row whose SPLIT == "test".

    Called at the entry of every Step 5 evaluation/fitting function that
    accepts a DataFrame carrying a SPLIT column (temporal CV fold
    construction, candidate/ablation CV scoring, official-validation
    scoring, paired comparison), so a caller cannot pass a
    test-containing frame into candidate selection even by accident.

    A DataFrame without a SPLIT column is left alone -- this guard only
    fires when it can positively identify test rows; it is not a
    substitute for also restricting inputs to the correct split
    upstream (see training/train.py::load_eligible_splits, reused here).
    """
    if "SPLIT" not in df.columns:
        return
    is_test = df["SPLIT"] == "test"
    if is_test.any():
        raise ValueError(
            f"{context}: received {int(is_test.sum())} row(s) from the "
            "frozen 'test' split. Step 5 experiment code must never fit "
            "or score against test rows -- see training/splits.py and "
            "training/experiments/guardrails.py."
        )
