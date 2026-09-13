"""
Authoritative feature schema for the currently deployed ("legacy") points
prediction model at models/points_regression.pkl.

This is a deliberate, hand-maintained code contract -- it is NOT derived at
runtime from the model artifact. LEGACY_FEATURE_NAMES must match
`model.feature_names_in_` exactly, including order. tests/test_feature_schema.py
enforces this against the actual artifact so any drift between this file and
the deployed model fails loudly instead of silently.

Do not reorder, rename, add to, or remove from this list without retraining
(or re-validating against) the model it describes.
"""

LEGACY_FEATURE_NAMES = (
    "player_avg_pts",
    "player_avg_pts_sq",
    "season_minutes_avg",
    "predicted_minutes",
    "home_game",
    "days_rest",
    "is_back_to_back",
    "last3_pts",
    "last5_pts",
    "last10_pts",
    "last20_pts",
    "last5_fga",
    "last5_fta",
    "last5_minutes",
    "last5_gmsc",
    "last5_usage_proxy",
    "minutes_volatility",
    "opp_pts_allowed",
    "opp_pts_allowed_last5",
    "points_volatility",
    "is_star",
    "closing_line",
    "opp_pts_volatility",
    "last5_3pa",
)

# Subset of LEGACY_FEATURE_NAMES that must be non-null for a feature row to be
# usable at all (rows failing this are dropped). Extracted unchanged from the
# original `core_required` list inside build_player_feature_row().
LEGACY_CORE_REQUIRED_FEATURES = (
    "player_avg_pts",
    "player_avg_pts_sq",
    "season_minutes_avg",
    "predicted_minutes",
    "home_game",
    "days_rest",
    "is_back_to_back",
    "closing_line",
)
