"""
V1 schema contract tests. V1_FEATURE_NAMES is a hand-maintained constant,
not derived from any generated DataFrame -- these tests lock in the exact
list and its key differences from the legacy schema.
"""

from src.features import v1_schema
from src.features.feature_schema import LEGACY_FEATURE_NAMES


def test_v1_feature_names_has_no_duplicates():
    assert len(v1_schema.V1_FEATURE_NAMES) == len(set(v1_schema.V1_FEATURE_NAMES))


def test_v1_feature_names_is_an_explicit_tuple():
    assert isinstance(v1_schema.V1_FEATURE_NAMES, tuple)
    assert len(v1_schema.V1_FEATURE_NAMES) == 23


def test_v1_removes_closing_line_and_is_star():
    assert "closing_line" not in v1_schema.V1_FEATURE_NAMES
    assert "is_star" not in v1_schema.V1_FEATURE_NAMES
    # both were present in the legacy schema -- confirms this is a removal,
    # not just an omission that happened to never exist.
    assert "closing_line" in LEGACY_FEATURE_NAMES
    assert "is_star" in LEGACY_FEATURE_NAMES


def test_v1_removes_mislabeled_legacy_opponent_features():
    for legacy_name in (
        "opp_pts_allowed",
        "opp_pts_allowed_last5",
        "opp_pts_volatility",
    ):
        assert legacy_name not in v1_schema.V1_FEATURE_NAMES
        assert (
            legacy_name in LEGACY_FEATURE_NAMES
        )  # confirms these existed and were removed


def test_v1_has_renamed_minutes_feature_not_the_misleading_legacy_name():
    assert "predicted_minutes" not in v1_schema.V1_FEATURE_NAMES
    assert "recent_minutes_avg" in v1_schema.V1_FEATURE_NAMES
    assert (
        "predicted_minutes" in LEGACY_FEATURE_NAMES
    )  # the name V1 deliberately dropped


def test_v1_has_real_opponent_team_features():
    for name in (
        "opponent_points_allowed_per_game",
        "opponent_points_allowed_last5",
        "opponent_pace",
        "opponent_defensive_rating",
    ):
        assert name in v1_schema.V1_FEATURE_NAMES


def test_v1_carries_over_unchanged_concepts_from_legacy():
    unchanged = (
        "player_avg_pts",
        "player_avg_pts_sq",
        "season_minutes_avg",
        "home_game",
        "days_rest",
        "is_back_to_back",
        "last3_pts",
        "last5_pts",
        "last10_pts",
        "last20_pts",
        "last5_fga",
        "last5_fta",
        "last5_3pa",
        "last5_minutes",
        "last5_gmsc",
        "last5_usage_proxy",
        "minutes_volatility",
        "points_volatility",
    )
    for name in unchanged:
        assert name in v1_schema.V1_FEATURE_NAMES
        assert name in LEGACY_FEATURE_NAMES


def test_no_sportsbook_or_market_columns_in_v1_feature_names():
    forbidden_substrings = (
        "line",
        "odds",
        "sportsbook",
        "closing",
        "edge",
        "bookmaker",
    )
    for name in v1_schema.V1_FEATURE_NAMES:
        lowered = name.lower()
        for bad in forbidden_substrings:
            assert bad not in lowered, (
                f"{name!r} looks like it could be sportsbook-derived"
            )


def test_identifier_and_target_columns_are_not_duplicated_in_feature_names():
    overlap = set(v1_schema.V1_IDENTIFIER_COLUMNS) & set(v1_schema.V1_FEATURE_NAMES)
    assert overlap == set()
    assert v1_schema.V1_TARGET_COLUMN not in v1_schema.V1_FEATURE_NAMES


def test_min_prior_games_for_eligibility_is_five():
    # Pinned explicitly: this is the threshold every diagnostics/eligibility
    # test and the real-data run in this step's report are computed against.
    assert v1_schema.MIN_PRIOR_GAMES_FOR_ELIGIBILITY == 5
