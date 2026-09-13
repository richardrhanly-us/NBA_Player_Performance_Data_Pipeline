"""
Tests for training/splits.py: chronological boundaries, exhaustiveness,
and determinism. No network access, no dependency on the real raw data.
"""

import pandas as pd

from training import splits


def _panel(rows):
    """rows: list of (season, date_str). Other columns aren't needed by
    assign_splits()."""
    return pd.DataFrame(
        {
            "SEASON": [r[0] for r in rows],
            "GAME_DATE": [pd.Timestamp(r[1]) for r in rows],
        }
    )


def test_train_seasons_never_assigned_validation_or_test():
    df = _panel(
        [
            ("2023-24", "2023-11-01"),
            ("2024-25", "2025-03-01"),
        ]
    )
    result = splits.assign_splits(df)
    assert (result["SPLIT"] == splits.SPLIT_TRAIN).all()


def test_validation_test_season_split_by_boundary_date():
    before = splits.VALIDATION_TEST_BOUNDARY_DATE - pd.Timedelta(days=1)
    after = splits.VALIDATION_TEST_BOUNDARY_DATE + pd.Timedelta(days=1)

    df = _panel(
        [
            (splits.VALIDATION_TEST_SEASON, str(before.date())),
            (
                splits.VALIDATION_TEST_SEASON,
                str(splits.VALIDATION_TEST_BOUNDARY_DATE.date()),
            ),
            (splits.VALIDATION_TEST_SEASON, str(after.date())),
        ]
    )
    result = splits.assign_splits(df)

    assert result.loc[0, "SPLIT"] == splits.SPLIT_VALIDATION
    # the boundary date itself is inclusive of validation (<=)
    assert result.loc[1, "SPLIT"] == splits.SPLIT_VALIDATION
    assert result.loc[2, "SPLIT"] == splits.SPLIT_TEST


def test_no_train_date_falls_after_the_validation_test_boundary():
    # Train seasons are entirely separate seasons from the validation/test
    # season, so this is really testing that season membership alone
    # determines train assignment, never date comparison against the
    # 2025-26 boundary.
    df = _panel(
        [("2024-25", "2025-12-31")]
    )  # a date "later" than the boundary, wrong season
    result = splits.assign_splits(df)
    assert result.loc[0, "SPLIT"] == splits.SPLIT_TRAIN


def test_no_validation_row_falls_in_test_and_vice_versa():
    dates = pd.date_range("2025-10-21", "2026-04-12", freq="17D")
    df = _panel([(splits.VALIDATION_TEST_SEASON, str(d.date())) for d in dates])
    result = splits.assign_splits(df)

    validation_dates = pd.to_datetime(
        result.loc[result["SPLIT"] == splits.SPLIT_VALIDATION, "GAME_DATE"]
    )
    test_dates = pd.to_datetime(
        result.loc[result["SPLIT"] == splits.SPLIT_TEST, "GAME_DATE"]
    )

    assert (validation_dates <= splits.VALIDATION_TEST_BOUNDARY_DATE).all()
    assert (test_dates > splits.VALIDATION_TEST_BOUNDARY_DATE).all()
    if len(validation_dates) and len(test_dates):
        assert validation_dates.max() < test_dates.min()


def test_every_configured_season_row_gets_assigned_to_exactly_one_split():
    df = _panel(
        [
            ("2023-24", "2023-11-01"),
            ("2024-25", "2024-12-01"),
            (splits.VALIDATION_TEST_SEASON, "2025-11-01"),
            (splits.VALIDATION_TEST_SEASON, "2026-03-01"),
        ]
    )
    result = splits.assign_splits(df)
    assert (result["SPLIT"] != splits.SPLIT_UNASSIGNED).all()
    assert (
        result["SPLIT"]
        .isin([splits.SPLIT_TRAIN, splits.SPLIT_VALIDATION, splits.SPLIT_TEST])
        .all()
    )


def test_a_season_outside_the_configured_scope_is_left_unassigned_not_guessed():
    df = _panel([("2099-00", "2099-01-01")])
    result = splits.assign_splits(df)
    assert result.loc[0, "SPLIT"] == splits.SPLIT_UNASSIGNED


def test_split_assignment_is_deterministic_across_repeated_calls():
    df = _panel(
        [
            ("2023-24", "2023-11-01"),
            (splits.VALIDATION_TEST_SEASON, "2026-01-15"),
            (splits.VALIDATION_TEST_SEASON, "2026-03-01"),
        ]
    )
    result_1 = splits.assign_splits(df)
    result_2 = splits.assign_splits(df.sample(frac=1.0, random_state=3))

    counts_1 = result_1["SPLIT"].value_counts().sort_index()
    counts_2 = result_2["SPLIT"].value_counts().sort_index()
    pd.testing.assert_series_equal(counts_1, counts_2)


def test_assign_splits_does_not_mutate_the_input_dataframe():
    df = _panel([("2023-24", "2023-11-01")])
    original_columns = list(df.columns)
    splits.assign_splits(df)
    assert list(df.columns) == original_columns  # SPLIT not added to the original
