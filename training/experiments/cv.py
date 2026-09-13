"""
Stage A: expanding-window, date-based temporal cross-validation fold
construction, built ONLY from rows in the official V1 train split
(SPLIT == "train").

Never random K-fold, never a row-level random split. Every row sharing
a GAME_DATE is assigned to exactly one side of a given fold (whichever
date-block that date falls in), and because blocks are built from
sorted, non-overlapping calendar dates, every fold satisfies

    max(fold.train_date_max) < min(fold.val_date_min)

by construction -- this is asserted, not merely assumed, in
build_temporal_cv_folds() below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.experiments.guardrails import reject_test_rows

# n_blocks - 1 folds are produced. 4 blocks -> 3 expanding-window folds,
# a "modest number" per the Step 5 brief. Documented here, not tuned.
DEFAULT_N_BLOCKS = 4


@dataclass(frozen=True, eq=False)
class TemporalFold:
    """One expanding-window fold, already materialized as train/val
    DataFrames (both subsets of the official train split)."""

    fold_index: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_date_min: pd.Timestamp
    train_date_max: pd.Timestamp
    val_date_min: pd.Timestamp
    val_date_max: pd.Timestamp

    @property
    def n_train(self) -> int:
        return len(self.train_df)

    @property
    def n_val(self) -> int:
        return len(self.val_df)


def build_temporal_cv_folds(
    train_df: pd.DataFrame, n_blocks: int = DEFAULT_N_BLOCKS
) -> list[TemporalFold]:
    """
    Splits the official train split into `n_blocks` contiguous,
    date-ordered blocks of (approximately) equal numbers of DISTINCT
    calendar dates -- not rows, so a date with an unusually high or low
    game count does not skew which block it lands in -- then builds
    n_blocks - 1 expanding-window folds:

        fold 1: block[0]      -> train, block[1] -> validation
        fold 2: block[0..1]   -> train, block[2] -> validation
        fold 3: block[0..2]   -> train, block[3] -> validation
        ...

    Boundaries are fully deterministic given `train_df` and `n_blocks`
    (np.array_split on a sorted array of unique dates) -- rerunning
    against the same panel reproduces identical fold boundaries.
    """
    reject_test_rows(train_df, "build_temporal_cv_folds")
    if "SPLIT" in train_df.columns and (train_df["SPLIT"] != "train").any():
        raise ValueError(
            "build_temporal_cv_folds: received a row whose SPLIT != 'train'. "
            "Temporal CV folds must be built only from the official train split."
        )
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2 to produce at least one fold.")

    dates = pd.to_datetime(train_df["GAME_DATE"])
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < n_blocks:
        raise ValueError(
            f"Only {len(unique_dates)} distinct GAME_DATE values available in "
            f"the train split, need at least {n_blocks} to build {n_blocks} blocks."
        )
    date_blocks = np.array_split(unique_dates, n_blocks)

    folds = []
    for i in range(1, n_blocks):
        train_dates = np.concatenate(date_blocks[:i])
        val_dates = date_blocks[i]
        train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates)

        fold = TemporalFold(
            fold_index=i,
            train_df=train_df.loc[train_mask].reset_index(drop=True),
            val_df=train_df.loc[val_mask].reset_index(drop=True),
            train_date_min=pd.Timestamp(train_dates.min()),
            train_date_max=pd.Timestamp(train_dates.max()),
            val_date_min=pd.Timestamp(val_dates.min()),
            val_date_max=pd.Timestamp(val_dates.max()),
        )
        if not (fold.train_date_max < fold.val_date_min):
            raise AssertionError(
                "Internal invariant violated: fold train/validation dates overlap "
                f"(fold {i}: train_date_max={fold.train_date_max}, "
                f"val_date_min={fold.val_date_min})."
            )
        folds.append(fold)

    return folds
