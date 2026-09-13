"""
Authoritative feature schema for the V1 (retrained) points-prediction model.

This is a separate, hand-maintained code contract from
src/features/feature_schema.py::LEGACY_FEATURE_NAMES, which describes the
currently deployed model and must remain untouched. V1_FEATURE_NAMES
describes the NEW model this training pipeline is building toward -- it is
not derived at runtime from any generated DataFrame, and it must be updated
deliberately (with tests re-verified) whenever the feature set changes.

Every feature listed here is documented below with its exact formula and
built by src/features/build_v1_features.py, which is the single place that
formula is implemented.

Differences from LEGACY_FEATURE_NAMES:

REMOVED:
  - closing_line       -- the sportsbook line is evaluation/backtesting data,
                           never a model input. Feeding the line being
                           evaluated into the model and then reporting
                           "edge" against that same line is circular.
  - is_star             -- a redundant hand-carved threshold
                           (player_avg_pts >= 20) on a feature the model
                           already has continuously (player_avg_pts,
                           player_avg_pts_sq). A tree-based model gains
                           nothing from also being handed the same
                           information as a binary flag.
  - opp_pts_allowed,
    opp_pts_allowed_last5,
    opp_pts_volatility  -- these were never a team-defense metric. They were
                           a rolling average of THIS PLAYER'S OWN points,
                           grouped by opponent -- sparse (a player faces a
                           given opponent only 2-4x/season) and conflates
                           "how this player personally has scored against
                           this opponent" with "how good this opponent's
                           defense is". Replaced below with real team-level
                           opponent features built from a reconstructed
                           team-game panel.

RENAMED:
  - predicted_minutes -> recent_minutes_avg
                           The legacy name implied a minutes projection
                           model. The actual formula
                           (last5_minutes.combine_first(season_minutes_avg))
                           is a historical average of ACTUAL past minutes,
                           not a projection of anything. V1 keeps the exact
                           same formula under an honest name; a real
                           minutes-projection model is future work, not V1.

ADDED (real team/opponent context, see build_v1_features.py for the full
team-game reconstruction this depends on):
  - opponent_points_allowed_per_game
  - opponent_points_allowed_last5
  - opponent_pace
  - opponent_defensive_rating

Everything else (rolling scoring windows, shot/FT volume, GmSc, the usage
proxy, volatility, rest/back-to-back/home-game context) is carried over
from the legacy schema with its formula unchanged, rebuilt from the raw
historical panel with explicit leakage-safe (shift(1)-based) logic instead
of the legacy single-row-at-a-time inference function.
"""

V1_FEATURE_NAMES = (
    # Baseline scoring ability (expanding, season-to-date; resets each season)
    "player_avg_pts",
    "player_avg_pts_sq",
    # Playing time
    "season_minutes_avg",
    "recent_minutes_avg",
    # Game context
    "home_game",
    "days_rest",
    "is_back_to_back",
    # Recent scoring form (rolling windows over prior games, season-scoped)
    "last3_pts",
    "last5_pts",
    "last10_pts",
    "last20_pts",
    # Recent shot/FT/3PT volume
    "last5_fga",
    "last5_fta",
    "last5_3pa",
    "last5_minutes",
    # Recent overall performance / involvement
    "last5_gmsc",
    "last5_usage_proxy",
    # Consistency
    "minutes_volatility",
    "points_volatility",
    # Opponent/team context (see build_v1_features.py: build_team_game_panel
    # and attach_opponent_features for exactly how these are computed and
    # why they are leakage-safe)
    "opponent_points_allowed_per_game",
    "opponent_points_allowed_last5",
    "opponent_pace",
    "opponent_defensive_rating",
)

# Minimum number of *prior games in the same season* a player must have
# before a row is considered eligible for V1 training. Rows below this
# threshold stay in the processed panel (nothing is silently dropped) but
# are flagged TRAINING_ELIGIBLE=0 -- see build_v1_features.py and
# training/diagnostics.py for the row-count impact of this choice at
# several candidate thresholds.
#
# 5 is chosen because it is the point at which every last5_* feature
# (the most heavily-used feature family, per the legacy model's own
# feature_importances_) is populated from real prior-game data rather
# than being entirely NaN.
MIN_PRIOR_GAMES_FOR_ELIGIBILITY = 5

# Identifier / provenance / target columns carried alongside the feature
# columns in the processed panel. Not part of the model's input feature
# set -- listed separately so V1_FEATURE_NAMES stays exactly the model's
# input contract.
V1_IDENTIFIER_COLUMNS = (
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "TEAM_ABBREVIATION",
    "OPPONENT_ABBREVIATION",
)

V1_TARGET_COLUMN = "PTS"

V1_META_COLUMNS = (
    "PRIOR_GAMES_THIS_SEASON",
    "TRAINING_ELIGIBLE",
    "SPLIT",
)
