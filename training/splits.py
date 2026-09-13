"""
Chronological train/validation/test split assignment for the V1 processed
panel.

Absolute calendar-date boundaries only -- never a random split, never a
row-count cutoff chosen after looking at model performance (there is no
model yet; this module cannot see one).

Split policy:

    train:      SEASON in TRAIN_SEASONS               (2023-24, 2024-25)
    validation: SEASON == VALIDATION_TEST_SEASON       (2025-26)
                and GAME_DATE <= VALIDATION_TEST_BOUNDARY_DATE
    test:       SEASON == VALIDATION_TEST_SEASON
                and GAME_DATE >  VALIDATION_TEST_BOUNDARY_DATE

VALIDATION_TEST_BOUNDARY_DATE is a fixed, reported constant, not something
recomputed on the fly -- it was derived once, from the real collected
2025-26 raw data (2025-10-21 through 2026-04-12, 26,651 rows), as the
earliest calendar date at which the cumulative row count reaches 60% of
the season:

    2026-02-01 -> 16,065 / 26,651 rows on or before this date (60.3%)
                  10,586 / 26,651 rows after this date        (39.7%)

Chosen purely from the row-count/date distribution, before any model
existed to evaluate -- not tuned. It also happens to land right around the
real NBA All-Star break, which is a reasonable, independently meaningful
place for a season to split in two.

If a future season is added to TRAIN_SEASONS or used to replace
VALIDATION_TEST_SEASON, recompute this constant the same way (see
scripts note in this module) and update it deliberately -- do not leave a
stale boundary in place for a different season's date range.

-------------------------------------------------------------------------
THE TEST SPLIT DEFINED HERE IS FROZEN, AS OF THE STEP 4 V1 BASELINE.
-------------------------------------------------------------------------
Its results (MAE/RMSE/R2/bias on this exact `test` split) have now been
observed by a human. From this point forward, ALL future hyperparameter
tuning, feature selection, model-family comparison, calibration design,
and any other optimization decision must be made using train/validation
data only -- never by looking at, or selecting for, an improvement on
this test split. Evaluating a new candidate on `test` is allowed only as
a final, infrequent check reported alongside the decision that was
already made on train/validation grounds -- never as the basis for that
decision. If the test split ever needs to change (e.g. a new season is
added and VALIDATION_TEST_SEASON / VALIDATION_TEST_BOUNDARY_DATE moves),
that is a deliberate, explicitly-called-out change to this module, not
routine model-development work, and should be treated as resetting this
freeze for a newly-defined test set.
"""

import pandas as pd

TRAIN_SEASONS = ("2023-24", "2024-25")
VALIDATION_TEST_SEASON = "2025-26"

# See module docstring for exactly how this was derived.
VALIDATION_TEST_BOUNDARY_DATE = pd.Timestamp("2026-02-01")

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_UNASSIGNED = "unassigned"


def compute_validation_test_boundary(
    raw_or_panel_df: pd.DataFrame, fraction: float = 0.60
) -> pd.Timestamp:
    """
    Utility to (re)derive a chronological boundary date the same way
    VALIDATION_TEST_BOUNDARY_DATE above was computed: the earliest calendar
    date at which the cumulative row count reaches `fraction` of the given
    season's rows. Not called automatically by assign_splits() -- the
    boundary used for splitting is the fixed constant above, so that it
    cannot silently drift if more rows are added later (e.g. more of the
    season being collected). Use this only to deliberately recompute the
    constant when moving to a new validation/test season.
    """
    dates = pd.to_datetime(raw_or_panel_df["GAME_DATE"])
    by_date = dates.groupby(dates.dt.date).size().sort_index()
    cumulative_fraction = by_date.cumsum() / by_date.sum()
    boundary = cumulative_fraction[cumulative_fraction >= fraction].index[0]
    return pd.Timestamp(boundary)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a SPLIT column ("train" / "validation" / "test") to a copy of the
    given processed panel, using only SEASON and GAME_DATE. Rows whose
    SEASON is not in TRAIN_SEASONS or VALIDATION_TEST_SEASON are labeled
    "unassigned" rather than silently guessed at -- this should not happen
    for a panel built only from training.config.TRAINING_SEASONS, and is
    surfaced rather than hidden if it ever does (e.g. a future season
    added to raw collection before this module's constants are updated).
    """
    result = df.copy()
    game_date = pd.to_datetime(result["GAME_DATE"])

    is_train = result["SEASON"].isin(TRAIN_SEASONS)
    is_validation = (result["SEASON"] == VALIDATION_TEST_SEASON) & (
        game_date <= VALIDATION_TEST_BOUNDARY_DATE
    )
    is_test = (result["SEASON"] == VALIDATION_TEST_SEASON) & (
        game_date > VALIDATION_TEST_BOUNDARY_DATE
    )

    result["SPLIT"] = SPLIT_UNASSIGNED
    result.loc[is_train, "SPLIT"] = SPLIT_TRAIN
    result.loc[is_validation, "SPLIT"] = SPLIT_VALIDATION
    result.loc[is_test, "SPLIT"] = SPLIT_TEST

    return result
