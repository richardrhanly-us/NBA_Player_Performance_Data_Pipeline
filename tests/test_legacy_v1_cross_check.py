"""
Cross-checks V1's rebuilt historical features against the legacy
build_player_feature_row() for the concepts V1 intentionally left
unchanged (player_avg_pts, last3/5/10_pts, season_minutes_avg, days_rest,
home_game). V1 is NOT expected to match legacy on closing_line, is_star,
predicted_minutes, or opp_pts_* -- those were deliberately removed/
renamed/redefined (see src/features/v1_schema.py).

Uses a synthetic, deterministic fixture (not the real persisted raw data,
which may not exist in every environment/CI) shaped like
training.data.storage.RAW_GAMELOG_COLUMNS -- exactly what both
build_player_feature_row() and build_v1_feature_panel() consume.

This was also spot-checked against real 2023-24 data during this step's
development (LeBron James' full 71-game season, and a 3-game low-volume
player) with an exact match on every unchanged concept in both cases; see
this step's audit report for those numbers. The synthetic fixture here
locks that finding in as a repeatable, offline test.
"""

import pandas as pd
import pytest

from src.features import build_v1_features
from src.features.build_features import build_player_feature_row

RAW_COLUMNS = [
    "SEASON",
    "SEASON_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "TEAM_ABBREVIATION",
    "OPPONENT_ABBREVIATION",
    "IS_HOME",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
]

UNCHANGED_CONCEPTS = (
    "player_avg_pts",
    "last3_pts",
    "last5_pts",
    "last10_pts",
    "season_minutes_avg",
    "days_rest",
    "home_game",
)


def _row(game_id, date, is_home, pts, min_):
    team, opponent = ("AAA", "BBB")
    matchup = f"{team} vs {opponent}" if is_home else f"{team} @ {opponent}"
    return {
        "SEASON": "2023-24",
        "SEASON_ID": "22023",
        "PLAYER_ID": 101,
        "PLAYER_NAME": "Test Player",
        "GAME_ID": game_id,
        "GAME_DATE": pd.Timestamp(date),
        "MATCHUP": matchup,
        "TEAM_ABBREVIATION": team,
        "OPPONENT_ABBREVIATION": opponent,
        "IS_HOME": 1 if is_home else 0,
        "WL": "W",
        "MIN": min_,
        "FGM": 5,
        "FGA": 10,
        "FG_PCT": 0.5,
        "FG3M": 1,
        "FG3A": 3,
        "FG3_PCT": 0.3,
        "FTM": 2,
        "FTA": 2,
        "FT_PCT": 1.0,
        "OREB": 1,
        "DREB": 4,
        "REB": 5,
        "AST": 3,
        "STL": 1,
        "BLK": 0,
        "TOV": 2,
        "PF": 2,
        "PTS": pts,
        "PLUS_MINUS": pts - 10,
        "VIDEO_AVAILABLE": 1,
    }


def _compare_last_row(raw_df):
    legacy = build_player_feature_row(
        raw_df.copy(), "Test Player", sportsbook_line=20.0
    )
    v1_panel = build_v1_features.build_v1_feature_panel(raw_df)
    v1_last = v1_panel.sort_values("GAME_DATE").iloc[-1]

    for col in UNCHANGED_CONCEPTS:
        legacy_val = legacy[col].iloc[0]
        v1_val = v1_last[col]
        if pd.isna(legacy_val):
            assert pd.isna(v1_val), f"{col}: legacy=NaN but v1={v1_val!r}"
        else:
            assert v1_val == pytest.approx(legacy_val), (
                f"{col}: legacy={legacy_val!r} v1={v1_val!r}"
            )


def test_unchanged_concepts_match_legacy_on_a_long_history():
    rows = []
    dates_and_pts_min = [
        ("2023-10-24", True, 20, 30),
        ("2023-10-26", False, 22, 32),
        ("2023-10-28", True, 18, 28),
        ("2023-10-30", False, 25, 34),
        ("2023-11-01", True, 30, 36),
        ("2023-11-03", False, 15, 26),
        ("2023-11-05", True, 28, 33),
        ("2023-11-07", False, 24, 31),
    ]
    for i, (date, is_home, pts, min_) in enumerate(dates_and_pts_min):
        rows.append(_row(f"G{i}", date, is_home, pts, min_))
    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)

    _compare_last_row(raw_df)


def test_unchanged_concepts_match_legacy_on_a_very_short_history():
    rows = [
        _row("G0", "2023-10-24", True, 12, 20),
        _row("G1", "2023-10-26", False, 8, 18),
        _row("G2", "2023-10-28", True, 15, 22),
    ]
    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)

    _compare_last_row(raw_df)


def test_v1_intentionally_diverges_on_removed_and_redefined_features():
    """Documents (not just implies) that V1 does NOT try to match legacy on
    everything -- closing_line/is_star/predicted_minutes are gone, and the
    opponent features are a different, real signal, not the legacy
    player-vs-opponent proxy."""
    from src.features.feature_schema import LEGACY_FEATURE_NAMES
    from src.features.v1_schema import V1_FEATURE_NAMES

    legacy_only = set(LEGACY_FEATURE_NAMES) - set(V1_FEATURE_NAMES)
    assert legacy_only == {
        "closing_line",
        "is_star",
        "predicted_minutes",
        "opp_pts_allowed",
        "opp_pts_allowed_last5",
        "opp_pts_volatility",
    }
