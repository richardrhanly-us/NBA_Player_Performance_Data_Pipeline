"""
Predeclared, interpretable feature groups over V1_FEATURE_NAMES, and the
small set of feature-ablation specs built from them.

These groups and ablations are declared once, here, BEFORE any
experiment is run, and are not adjusted based on results (see
training/experiments/run.py for how the ablation stage consumes them).
This is deliberately NOT brute-force feature selection across arbitrary
subsets -- five hand-chosen, interpretable configurations, each answering
a specific real question about the V1 feature set (do opponent features
help? does schedule context help? does the squared scoring term help?).
"""

from __future__ import annotations

from src.features import v1_schema

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "core_scoring_history": (
        "player_avg_pts",
        "player_avg_pts_sq",
        "last3_pts",
        "last5_pts",
        "last10_pts",
        "last20_pts",
        "points_volatility",
    ),
    "volume_role": (
        "season_minutes_avg",
        "recent_minutes_avg",
        "last5_minutes",
        "minutes_volatility",
        "last5_fga",
        "last5_fta",
        "last5_3pa",
        "last5_usage_proxy",
        "last5_gmsc",
    ),
    "schedule_location": (
        "home_game",
        "days_rest",
        "is_back_to_back",
    ),
    "opponent_context": (
        "opponent_points_allowed_per_game",
        "opponent_points_allowed_last5",
        "opponent_pace",
        "opponent_defensive_rating",
    ),
}


def _validate_feature_groups() -> None:
    all_grouped = [f for group in FEATURE_GROUPS.values() for f in group]
    if len(all_grouped) != len(set(all_grouped)):
        raise AssertionError("FEATURE_GROUPS contains a duplicate feature name.")
    grouped_set = set(all_grouped)
    schema_set = set(v1_schema.V1_FEATURE_NAMES)
    if grouped_set != schema_set:
        raise AssertionError(
            "FEATURE_GROUPS does not exactly partition V1_FEATURE_NAMES. "
            f"Missing from groups: {schema_set - grouped_set}. "
            f"In groups but not in schema: {grouped_set - schema_set}."
        )


# Runs at import time: a schema/group mismatch (e.g. a future V1_FEATURE_NAMES
# edit that forgets to update FEATURE_GROUPS) fails immediately and loudly,
# not silently at ablation time.
_validate_feature_groups()


def _features_excluding(*group_names: str) -> tuple[str, ...]:
    excluded = {f for name in group_names for f in FEATURE_GROUPS[name]}
    return tuple(f for f in v1_schema.V1_FEATURE_NAMES if f not in excluded)


def _features_only(*group_names: str) -> tuple[str, ...]:
    included = {f for name in group_names for f in FEATURE_GROUPS[name]}
    return tuple(f for f in v1_schema.V1_FEATURE_NAMES if f in included)


# Five predeclared ablations. D and F are deliberately the same
# configuration: removing "opponent_context" and "schedule_location"
# leaves exactly "core_scoring_history" + "volume_role", since the four
# FEATURE_GROUPS above exactly partition all 23 V1 features. Computed
# once, reported under both labels, rather than run twice for an
# identical result.
ABLATION_SPECS: list[dict] = [
    {
        "ablation_id": "A_full_v1",
        "description": "Full V1 feature set (all 23 features) -- reference point.",
        "feature_names": tuple(v1_schema.V1_FEATURE_NAMES),
    },
    {
        "ablation_id": "B_minus_opponent",
        "description": "Full V1 minus opponent_context (4 features removed).",
        "feature_names": _features_excluding("opponent_context"),
    },
    {
        "ablation_id": "C_minus_schedule",
        "description": "Full V1 minus schedule_location (3 features removed).",
        "feature_names": _features_excluding("schedule_location"),
    },
    {
        "ablation_id": "D_minus_opponent_and_schedule_F_core_and_volume_only",
        "description": (
            "Full V1 minus opponent_context and schedule_location -- "
            "equivalent to core_scoring_history + volume_role only, since "
            "the four FEATURE_GROUPS exactly partition V1_FEATURE_NAMES."
        ),
        "feature_names": _features_only("core_scoring_history", "volume_role"),
    },
    {
        "ablation_id": "E_minus_player_avg_pts_sq",
        "description": "Full V1 minus the single feature player_avg_pts_sq.",
        "feature_names": tuple(
            f for f in v1_schema.V1_FEATURE_NAMES if f != "player_avg_pts_sq"
        ),
    },
]
