"""
Leakage-safe historical feature construction for the V1 model.

This module builds a full historical feature PANEL (every eligible
player-game row, not just "the latest row" the way the legacy
build_player_feature_row() does for live inference). Nothing here is used
by the deployed app; it feeds the (not-yet-built) training pipeline only.

Pipeline, in order:

1. build_team_game_panel(raw_df)
   Player rows -> one row per (SEASON, TEAM_ABBREVIATION, GAME_ID), with
   that team's own score, the opponent's score (points allowed), and
   box-score-estimated possessions/pace/defensive rating for that single
   game. See that function's docstring for exactly how, and why a stable
   TEAM_ID isn't needed here.

2. attach_team_pregame_rolling(team_game_df)
   Adds shift(1)-based rolling versions of those single-game numbers,
   grouped by (SEASON, TEAM_ABBREVIATION) -- i.e. "this team's own rolling
   defensive context entering each of its games", using only that team's
   own prior games this season.

3. build_player_feature_panel(raw_df)
   Player rows -> one row per (PLAYER_ID, GAME_ID), with every V1 player-
   history feature computed via shift(1)-based rolling/expanding windows
   grouped by (PLAYER_ID, SEASON) -- see the "Season-boundary policy"
   section below.

4. attach_opponent_features(player_df, team_game_rolling)
   Joins each player row to its opponent's PREGAME rolling row for that
   exact GAME_ID -- the opponent features a player row gets are computed
   entirely from the opponent's games strictly before that date.

5. build_v1_feature_panel(raw_df)
   Runs the above end to end and returns the final panel: identifiers,
   target (PTS), every column in src.features.v1_schema.V1_FEATURE_NAMES,
   and cold-start bookkeeping (PRIOR_GAMES_THIS_SEASON, TRAINING_ELIGIBLE).
   Does not assign train/validation/test splits -- that's training/splits.py.

Season-boundary policy (deliberate, V1-wide):
    ALL player rolling/expanding features -- including short windows like
    last5_pts -- reset at each season boundary (grouped by
    (PLAYER_ID, SEASON), not just PLAYER_ID). A player's first game of a
    new season always has every rolling/expanding feature NaN, exactly
    like their first game ever. Team/opponent rolling features use the
    same policy, grouped by (SEASON, TEAM_ABBREVIATION).

    This is a simplification, not an oversight: carrying a "last5_pts"
    window across a ~4-5 month offseason gap raises its own semantic
    questions (does a game from the prior season, played on a different
    roster/health context, really belong in "recent form"?) that are out
    of scope for V1. Revisit later if warranted.

days_rest / is_back_to_back policy:
    days_rest = days since the player's previous game *in the same
    season* (grouped by (PLAYER_ID, SEASON), consistent with the policy
    above). A player's first game of a season has days_rest = NaN (there
    is no in-season prior game to measure from) -- NOT filled with an
    arbitrary sentinel value the way the legacy feature does
    (`.fillna(3)`). is_back_to_back = 1 only when days_rest == 1; NaN
    days_rest naturally yields is_back_to_back = 0 (a player cannot be
    proven to be on a back-to-back without a qualifying in-season prior
    game).
"""

import pandas as pd

from src.features import v1_schema

RAW_NUMERIC_COLUMNS = (
    "PTS",
    "MIN",
    "FGM",
    "FGA",
    "FTA",
    "FTM",
    "FG3A",
    "OREB",
    "DREB",
    "STL",
    "AST",
    "BLK",
    "PF",
    "TOV",
)


def compute_game_score(df: pd.DataFrame) -> pd.Series:
    """
    Standard (Hollinger) Game Score, from raw box-score columns:

        GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM)
               + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK
               - 0.4*PF - TOV

    Identical formula to the legacy build_player_feature_row() and the
    original training notebook -- unchanged on purpose, just centralized
    here instead of being reimplemented ad hoc.
    """
    return (
        df["PTS"]
        + 0.4 * df["FGM"]
        - 0.7 * df["FGA"]
        - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"]
        + 0.3 * df["DREB"]
        + df["STL"]
        + 0.7 * df["AST"]
        + 0.7 * df["BLK"]
        - 0.4 * df["PF"]
        - df["TOV"]
    )


def compute_usage_proxy(df: pd.DataFrame) -> pd.Series:
    """
    usage_proxy = FGA + 0.44*FTA + TOV

    This is NOT true NBA usage percentage (real usage% needs team-level
    minutes/FGA/FTA/TOV context: a player's share of their team's
    possessions while on the floor). It is a simple, unscaled proxy for
    offensive involvement -- the same formula the legacy model uses,
    carried over unchanged and named accordingly (last5_usage_proxy, not
    last5_usage_pct).
    """
    return df["FGA"] + 0.44 * df["FTA"] + df["TOV"]


def _sort_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical chronological order for player-history feature construction:
    by player, by season, by date, tie-broken by GAME_ID. Applied
    unconditionally before any rolling computation so that feature values
    never depend on the order rows arrived in (the "shuffled-input
    determinism" requirement).
    """
    return df.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE", "GAME_ID"]).reset_index(
        drop=True
    )


def build_team_game_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs one row per (SEASON, TEAM_ABBREVIATION, GAME_ID) from the
    player-level raw panel: that team's own score, the opponent's score
    (i.e. points allowed), and box-score-estimated possessions/pace/
    defensive rating for that single game.

    No stable TEAM_ID exists in the raw store (see
    training/data/storage.py::RAW_GAMELOG_COLUMNS) -- only
    TEAM_ABBREVIATION, parsed per-row from that game's own MATCHUP string.
    This is safe to use as the team key here because it is verified
    (empirically, against the real 2023-24/2024-25/2025-26 raw data; see
    tests/test_build_v1_features.py) that every GAME_ID maps to exactly 2
    distinct TEAM_ABBREVIATION values with exactly one flagged IS_HOME,
    within a single season. Abbreviation reuse across different
    *franchises* in different seasons (e.g. historical relocations) is not
    a concern here because GAME_ID is itself season-scoped and every join
    in this module is keyed on (SEASON, TEAM_ABBREVIATION, GAME_ID), never
    TEAM_ABBREVIATION alone.

    Possession estimate uses the standard box-score approximation
    (Oliver's simplified formula), since no play-by-play data is
    available in the raw store:

        possessions_est = FGA - OREB + TOV + 0.44*FTA

    This is a genuine, well-established estimate, not an invented metric
    -- but it IS an estimate (typically within a possession or two of a
    play-by-play-derived count), and every feature built from it
    downstream is named/documented as such.
    """
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    team_game = df.groupby(
        ["SEASON", "TEAM_ABBREVIATION", "GAME_ID"], as_index=False
    ).agg(
        GAME_DATE=("GAME_DATE", "first"),
        OPPONENT_ABBREVIATION=("OPPONENT_ABBREVIATION", "first"),
        IS_HOME=("IS_HOME", "max"),
        PTS_SCORED=("PTS", "sum"),
        FGA=("FGA", "sum"),
        OREB=("OREB", "sum"),
        FTA=("FTA", "sum"),
        TOV=("TOV", "sum"),
        TEAM_MINUTES=("MIN", "sum"),
    )

    team_game["POSSESSIONS_EST"] = (
        team_game["FGA"]
        - team_game["OREB"]
        + team_game["TOV"]
        + 0.44 * team_game["FTA"]
    )

    opponent_lookup = team_game[
        ["SEASON", "TEAM_ABBREVIATION", "GAME_ID", "PTS_SCORED", "POSSESSIONS_EST"]
    ].rename(
        columns={
            "TEAM_ABBREVIATION": "OPPONENT_ABBREVIATION",
            "PTS_SCORED": "PTS_ALLOWED",
            "POSSESSIONS_EST": "OPPONENT_POSSESSIONS_EST",
        }
    )

    team_game = team_game.merge(
        opponent_lookup, on=["SEASON", "OPPONENT_ABBREVIATION", "GAME_ID"], how="left"
    )

    # A single game has one true pace/possession count -- both teams share
    # it. Averaging each team's own box-score possession estimate gives a
    # single, symmetric per-game estimate that both of that game's two
    # team-game rows use identically. Pace AND defensive rating are both
    # defined against this shared estimate, never a team's own one-sided
    # offensive possession count.
    team_game["GAME_POSSESSIONS_EST"] = (
        team_game["POSSESSIONS_EST"] + team_game["OPPONENT_POSSESSIONS_EST"]
    ) / 2.0
    team_game["PACE_EST"] = (
        48.0 * team_game["GAME_POSSESSIONS_EST"] / (team_game["TEAM_MINUTES"] / 5.0)
    )
    # Defensive rating = points allowed per 100 of the GAME's possessions,
    # not per 100 of the defending team's own offensive possession
    # estimate (those are two different, only loosely related quantities).
    team_game["DEF_RATING_EST"] = (
        100.0 * team_game["PTS_ALLOWED"] / team_game["GAME_POSSESSIONS_EST"]
    )

    team_game = team_game.sort_values(
        ["SEASON", "TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)
    return team_game


def attach_team_pregame_rolling(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds shift(1)-based rolling versions of each team-game's single-game
    stats, grouped by (SEASON, TEAM_ABBREVIATION): this team's own points
    allowed, pace, and defensive rating averaged over its games strictly
    before this one. These are what a PLAYER's opponent_* features are
    populated from (see attach_opponent_features) -- by construction they
    can never include the game they are eventually attached to, or any
    later game.
    """
    df = team_game_df.sort_values(
        ["SEASON", "TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)
    grouped = df.groupby(["SEASON", "TEAM_ABBREVIATION"])

    df["ROLL_PTS_ALLOWED_PER_GAME"] = grouped["PTS_ALLOWED"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["ROLL_PTS_ALLOWED_LAST5"] = grouped["PTS_ALLOWED"].transform(
        lambda s: s.shift(1).rolling(5).mean()
    )
    df["ROLL_PACE"] = grouped["PACE_EST"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["ROLL_DEF_RATING"] = grouped["DEF_RATING_EST"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return df


def attach_opponent_features(
    player_df: pd.DataFrame, team_game_rolling: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins each player row to its opponent's pregame rolling row for that
    exact GAME_ID, on (SEASON, OPPONENT_ABBREVIATION, GAME_ID). Because
    team_game_rolling's rolling columns are themselves shift(1)-based
    (see attach_team_pregame_rolling), the values attached here reflect
    only the opponent's games strictly before the target game -- never the
    target game itself and never anything after it.
    """
    opponent_features = team_game_rolling[
        [
            "SEASON",
            "TEAM_ABBREVIATION",
            "GAME_ID",
            "ROLL_PTS_ALLOWED_PER_GAME",
            "ROLL_PTS_ALLOWED_LAST5",
            "ROLL_PACE",
            "ROLL_DEF_RATING",
        ]
    ].rename(
        columns={
            "TEAM_ABBREVIATION": "OPPONENT_ABBREVIATION",
            "ROLL_PTS_ALLOWED_PER_GAME": "opponent_points_allowed_per_game",
            "ROLL_PTS_ALLOWED_LAST5": "opponent_points_allowed_last5",
            "ROLL_PACE": "opponent_pace",
            "ROLL_DEF_RATING": "opponent_defensive_rating",
        }
    )

    return player_df.merge(
        opponent_features, on=["SEASON", "OPPONENT_ABBREVIATION", "GAME_ID"], how="left"
    )


def build_player_feature_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (PLAYER_ID, GAME_ID), with every V1 player-history feature
    (everything in V1_FEATURE_NAMES except the four opponent_* columns)
    computed via shift(1)-based rolling/expanding windows grouped by
    (PLAYER_ID, SEASON). Does not attach opponent features or assign
    eligibility/splits -- see build_v1_feature_panel for the full pipeline.
    """
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    for col in RAW_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON"])
    df = _sort_player_history(df)

    df["GMSC"] = compute_game_score(df)
    df["USAGE_PROXY"] = compute_usage_proxy(df)

    grouped = df.groupby(["PLAYER_ID", "SEASON"])

    df["player_avg_pts"] = grouped["PTS"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["player_avg_pts_sq"] = df["player_avg_pts"] ** 2
    df["season_minutes_avg"] = grouped["MIN"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["last5_minutes"] = grouped["MIN"].transform(
        lambda s: s.shift(1).rolling(5).mean()
    )
    df["recent_minutes_avg"] = df["last5_minutes"].combine_first(
        df["season_minutes_avg"]
    )

    df["last3_pts"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["last5_pts"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["last10_pts"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(10).mean())
    df["last20_pts"] = grouped["PTS"].transform(lambda s: s.shift(1).rolling(20).mean())

    df["last5_fga"] = grouped["FGA"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["last5_fta"] = grouped["FTA"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["last5_3pa"] = grouped["FG3A"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["last5_gmsc"] = grouped["GMSC"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["last5_usage_proxy"] = grouped["USAGE_PROXY"].transform(
        lambda s: s.shift(1).rolling(5).mean()
    )

    df["minutes_volatility"] = grouped["MIN"].transform(
        lambda s: s.shift(1).rolling(5).std()
    )
    df["points_volatility"] = grouped["PTS"].transform(
        lambda s: s.shift(1).rolling(5).std()
    )

    df["home_game"] = pd.to_numeric(df["IS_HOME"], errors="coerce").astype("Int64")

    df["days_rest"] = grouped["GAME_DATE"].transform(lambda s: s.diff().dt.days)
    df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)

    # 0-indexed count of this player's prior rows within the same season --
    # doubles as the cold-start eligibility signal (see v1_schema.py).
    df["PRIOR_GAMES_THIS_SEASON"] = grouped.cumcount()

    return df


def build_v1_feature_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: raw player-game rows (any number of concatenated
    seasons; season-scoped grouping throughout means this is equivalent to
    processing each season separately) -> the final V1 processed panel.

    Returns identifiers, PTS (target), every V1_FEATURE_NAMES column, and
    cold-start bookkeeping columns. Does not assign train/validation/test
    splits -- see training/splits.py.
    """
    raw_df = raw_df.copy()

    player_df = build_player_feature_panel(raw_df)
    team_game = build_team_game_panel(raw_df)
    team_game_rolling = attach_team_pregame_rolling(team_game)
    player_df = attach_opponent_features(player_df, team_game_rolling)

    player_df["TRAINING_ELIGIBLE"] = (
        player_df["PRIOR_GAMES_THIS_SEASON"]
        >= v1_schema.MIN_PRIOR_GAMES_FOR_ELIGIBILITY
    ).astype(int)

    final_columns = (
        list(v1_schema.V1_IDENTIFIER_COLUMNS)
        + [v1_schema.V1_TARGET_COLUMN]
        + list(v1_schema.V1_FEATURE_NAMES)
        + ["PRIOR_GAMES_THIS_SEASON", "TRAINING_ELIGIBLE"]
    )
    for col in final_columns:
        if col not in player_df.columns:
            player_df[col] = pd.NA

    result = player_df[final_columns].copy()
    result = result.sort_values(
        ["SEASON", "GAME_DATE", "PLAYER_ID", "GAME_ID"]
    ).reset_index(drop=True)
    return result
