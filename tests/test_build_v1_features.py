"""
Leakage-safety and correctness tests for src/features/build_v1_features.py.

All fixtures are small, deterministic, hand-constructed DataFrames shaped
exactly like training.data.storage.RAW_GAMELOG_COLUMNS (i.e. what
storage.load_season_gamelogs() actually returns) -- no live NBA API calls
anywhere in this file.
"""

import math

import pandas as pd
import pytest

from src.features import build_v1_features, v1_schema

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


def _row(
    season,
    game_id,
    date,
    team,
    opponent,
    is_home,
    player_id,
    player_name,
    pts,
    min_=30,
    fgm=5,
    fga=10,
    fta=2,
    ftm=2,
    oreb=1,
    dreb=4,
    stl=1,
    ast=3,
    blk=0,
    pf=2,
    tov=2,
    fg3a=3,
):
    matchup = f"{team} vs {opponent}" if is_home else f"{team} @ {opponent}"
    return {
        "SEASON": season,
        "SEASON_ID": "2" + season[:4],
        "PLAYER_ID": player_id,
        "PLAYER_NAME": player_name,
        "GAME_ID": game_id,
        "GAME_DATE": pd.Timestamp(date),
        "MATCHUP": matchup,
        "TEAM_ABBREVIATION": team,
        "OPPONENT_ABBREVIATION": opponent,
        "IS_HOME": 1 if is_home else 0,
        "WL": "W",
        "MIN": min_,
        "FGM": fgm,
        "FGA": fga,
        "FG_PCT": fgm / fga if fga else 0.0,
        "FG3M": 1,
        "FG3A": fg3a,
        "FG3_PCT": 0.3,
        "FTM": ftm,
        "FTA": fta,
        "FT_PCT": ftm / fta if fta else 0.0,
        "OREB": oreb,
        "DREB": dreb,
        "REB": oreb + dreb,
        "AST": ast,
        "STL": stl,
        "BLK": blk,
        "TOV": tov,
        "PF": pf,
        "PTS": pts,
        "PLUS_MINUS": pts - 10,
        "VIDEO_AVAILABLE": 1,
    }


# ---------------------------------------------------------------------------
# Fixture A: 4 teams, one player each, 6 games per team (2023-24), plus a
# short 2024-25 continuation for player 101 to test season-reset behavior.
# Round-robin-ish schedule so every team has both home and away games.
# ---------------------------------------------------------------------------

_SEASON_A = "2023-24"
_SCHEDULE_A = [
    # (game_id, date, home_team, away_team)
    ("G01", "2023-10-24", "AAA", "BBB"),
    ("G02", "2023-10-26", "CCC", "DDD"),
    ("G03", "2023-10-28", "CCC", "AAA"),
    ("G04", "2023-10-30", "DDD", "BBB"),
    ("G05", "2023-11-01", "AAA", "DDD"),
    ("G06", "2023-11-03", "BBB", "CCC"),
    ("G07", "2023-11-05", "BBB", "AAA"),
    ("G08", "2023-11-07", "DDD", "CCC"),
    ("G09", "2023-11-09", "AAA", "CCC"),
    ("G10", "2023-11-11", "DDD", "BBB"),
    ("G11", "2023-11-13", "DDD", "AAA"),
    ("G12", "2023-11-15", "BBB", "CCC"),
]

_TEAM_PLAYER = {
    "AAA": (101, "Player A"),
    "BBB": (201, "Player B"),
    "CCC": (301, "Player C"),
    "DDD": (401, "Player D"),
}

# Deterministic, distinct points per (team, game index for that team) so
# rolling means can be hand-verified.
_TEAM_PTS_SEQUENCE = {
    "AAA": [20, 22, 24, 18, 26, 30],
    "BBB": [15, 17, 19, 21, 23, 25],
    "CCC": [10, 12, 14, 16, 18, 20],
    "DDD": [30, 28, 26, 24, 22, 20],
}


def _build_fixture_a(mutate_last_game_pts=None):
    """
    mutate_last_game_pts: optional {team: new_pts_for_that_team_final_game}
    used by the future-row-leakage test to change a FUTURE game's stats.
    """
    team_game_counters = {t: 0 for t in _TEAM_PLAYER}
    rows = []

    for game_id, date, home_team, away_team in _SCHEDULE_A:
        for team, opponent, is_home in (
            (home_team, away_team, True),
            (away_team, home_team, False),
        ):
            idx = team_game_counters[team]
            team_game_counters[team] += 1
            pts = _TEAM_PTS_SEQUENCE[team][idx]
            if (
                mutate_last_game_pts
                and team in mutate_last_game_pts
                and idx == len(_TEAM_PTS_SEQUENCE[team]) - 1
            ):
                pts = mutate_last_game_pts[team]
            player_id, player_name = _TEAM_PLAYER[team]
            rows.append(
                _row(
                    _SEASON_A,
                    game_id,
                    date,
                    team,
                    opponent,
                    is_home,
                    player_id,
                    player_name,
                    pts,
                )
            )

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return df


def _build_fixture_a_with_second_season():
    """Fixture A plus two more games for player 101 (team AAA) in 2024-25,
    to test that rolling/expanding features reset at the season boundary."""
    df = _build_fixture_a()
    extra = pd.DataFrame(
        [
            _row(
                "2024-25", "H01", "2024-10-22", "AAA", "BBB", True, 101, "Player A", 40
            ),
            _row(
                "2024-25", "H02", "2024-10-24", "AAA", "CCC", False, 101, "Player A", 42
            ),
        ],
        columns=RAW_COLUMNS,
    )
    return pd.concat([df, extra], ignore_index=True)


@pytest.fixture
def fixture_a():
    return _build_fixture_a()


# ---------------------------------------------------------------------------
# Basic shape / determinism
# ---------------------------------------------------------------------------


def test_panel_has_one_row_per_input_row_and_expected_columns(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    assert len(panel) == len(fixture_a)
    expected_columns = (
        list(v1_schema.V1_IDENTIFIER_COLUMNS)
        + [v1_schema.V1_TARGET_COLUMN]
        + list(v1_schema.V1_FEATURE_NAMES)
        + ["PRIOR_GAMES_THIS_SEASON", "TRAINING_ELIGIBLE"]
    )
    assert list(panel.columns) == expected_columns


def test_panel_has_no_player_game_id_duplicates(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    assert panel.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum() == 0


def test_no_sportsbook_columns_appear_anywhere_in_the_panel(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    forbidden = {"closing_line", "sportsbook_line", "sportsbook", "odds", "edge"}
    assert forbidden.isdisjoint(set(panel.columns))


def test_shuffled_input_row_order_produces_identical_output(fixture_a):
    shuffled = fixture_a.sample(frac=1.0, random_state=7).reset_index(drop=True)

    panel_original = build_v1_features.build_v1_feature_panel(fixture_a)
    panel_shuffled = build_v1_features.build_v1_feature_panel(shuffled)

    pd.testing.assert_frame_equal(panel_original, panel_shuffled)


def test_running_the_builder_twice_on_unchanged_input_is_identical(fixture_a):
    panel_1 = build_v1_features.build_v1_feature_panel(fixture_a)
    panel_2 = build_v1_features.build_v1_feature_panel(fixture_a.copy())
    pd.testing.assert_frame_equal(panel_1, panel_2)


# ---------------------------------------------------------------------------
# Rolling-window correctness (hand-verified against _TEAM_PTS_SEQUENCE)
# ---------------------------------------------------------------------------


def test_rolling_scoring_windows_match_hand_computed_values(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    aaa = (
        panel[panel["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    # AAA's points sequence in _TEAM_PTS_SEQUENCE is [20, 22, 24, 18, 26, 30].

    # Row 0 (first game): no prior games -> everything rolling is NaN.
    assert math.isnan(aaa.loc[0, "player_avg_pts"])
    assert math.isnan(aaa.loc[0, "last3_pts"])

    # Row 2 (3rd game): last3_pts uses games 0-1 only (shift(1).rolling(3),
    # only 2 prior values exist so far -> still NaN, rolling(3) needs 3).
    assert math.isnan(aaa.loc[2, "last3_pts"])

    # Row 3 (4th game): last3_pts = mean(pts[0], pts[1], pts[2]) = mean(20,22,24)
    assert aaa.loc[3, "last3_pts"] == pytest.approx((20 + 22 + 24) / 3)
    # player_avg_pts at row 3 = mean of all 3 prior games (same value here
    # since only 3 prior games exist)
    assert aaa.loc[3, "player_avg_pts"] == pytest.approx((20 + 22 + 24) / 3)
    assert aaa.loc[3, "player_avg_pts_sq"] == pytest.approx(((20 + 22 + 24) / 3) ** 2)

    # Row 5 (6th, last game): last5_pts = mean(pts[0:5]) = mean(20,22,24,18,26)
    assert aaa.loc[5, "last5_pts"] == pytest.approx((20 + 22 + 24 + 18 + 26) / 5)
    # player_avg_pts at row 5 = mean of all 5 prior games (expanding)
    assert aaa.loc[5, "player_avg_pts"] == pytest.approx((20 + 22 + 24 + 18 + 26) / 5)


def test_last5_minutes_and_recent_minutes_avg_use_shifted_history(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    aaa = (
        panel[panel["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    # All rows in fixture A use MIN=30 -- last5_minutes at row 5 should be
    # exactly 30 (mean of five 30s), and recent_minutes_avg should match.
    assert aaa.loc[5, "last5_minutes"] == pytest.approx(30.0)
    assert aaa.loc[5, "recent_minutes_avg"] == pytest.approx(30.0)
    # Row 0 has no prior games: season_minutes_avg is NaN, so
    # recent_minutes_avg (which falls back to it) is also NaN.
    assert math.isnan(aaa.loc[0, "recent_minutes_avg"])


# ---------------------------------------------------------------------------
# Current-row and future-row leakage
# ---------------------------------------------------------------------------


def test_target_games_own_pts_does_not_influence_its_own_features(fixture_a):
    panel_original = build_v1_features.build_v1_feature_panel(fixture_a)

    mutated = fixture_a.copy()
    # Change AAA's LAST game's own PTS drastically. Its own row's rolling
    # features must be computed from *prior* games only, so its own
    # last5_pts/player_avg_pts must be unaffected by this change (only the
    # raw PTS/target column for that row changes).
    mask = (mutated["TEAM_ABBREVIATION"] == "AAA") & (mutated["GAME_ID"] == "G11")
    assert mask.sum() == 1
    mutated.loc[mask, "PTS"] = 999
    mutated.loc[mask, "MIN"] = 48

    panel_mutated = build_v1_features.build_v1_feature_panel(mutated)

    row_original = panel_original[
        (panel_original["TEAM_ABBREVIATION"] == "AAA")
        & (panel_original["GAME_ID"] == "G11")
    ].iloc[0]
    row_mutated = panel_mutated[
        (panel_mutated["TEAM_ABBREVIATION"] == "AAA")
        & (panel_mutated["GAME_ID"] == "G11")
    ].iloc[0]

    # G11 is AAA's 6th game -- well past cold-start, so none of these should
    # be NaN in either version, and they must match exactly despite the
    # mutation to this same row's own PTS/MIN.
    for col in (
        "last3_pts",
        "last5_pts",
        "player_avg_pts",
        "last5_minutes",
        "recent_minutes_avg",
    ):
        assert not math.isnan(row_original[col])
        assert row_original[col] == pytest.approx(row_mutated[col])

    # The target itself, of course, did change.
    assert row_mutated["PTS"] == 999


def test_future_games_stats_do_not_leak_into_earlier_player_rows(fixture_a):
    panel_before = build_v1_features.build_v1_feature_panel(fixture_a)

    mutated = fixture_a.copy()
    # Drastically change AAA's FINAL game (G11) -- a game strictly AFTER
    # every other AAA row in this fixture.
    mask = (mutated["TEAM_ABBREVIATION"] == "AAA") & (mutated["GAME_ID"] == "G11")
    mutated.loc[mask, ["PTS", "MIN", "FGA", "FTA", "TOV", "OREB", "DREB"]] = [
        999,
        48,
        50,
        30,
        20,
        20,
        20,
    ]

    panel_after = build_v1_features.build_v1_feature_panel(mutated)

    earlier_before = panel_before[
        (panel_before["TEAM_ABBREVIATION"] == "AAA")
        & (panel_before["GAME_ID"] != "G11")
    ]
    earlier_after = panel_after[
        (panel_after["TEAM_ABBREVIATION"] == "AAA") & (panel_after["GAME_ID"] != "G11")
    ]

    pd.testing.assert_frame_equal(
        earlier_before.reset_index(drop=True), earlier_after.reset_index(drop=True)
    )


def test_chronological_sorting_is_enforced_even_when_rows_arrive_out_of_order():
    # Deliberately construct a tiny 3-game history for one player with rows
    # in REVERSE chronological order as input.
    rows = [
        _row("2023-24", "G3", "2023-11-01", "AAA", "BBB", True, 101, "P", 30),
        _row("2023-24", "G1", "2023-10-24", "AAA", "BBB", True, 101, "P", 10),
        _row("2023-24", "G2", "2023-10-26", "AAA", "BBB", True, 101, "P", 20),
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)

    panel = build_v1_features.build_v1_feature_panel(df)
    panel = panel.sort_values("GAME_DATE").reset_index(drop=True)

    # Game 3 (last chronologically) must show player_avg_pts computed from
    # games 1 and 2 (10, 20) = 15, regardless of input row order.
    game3 = panel[panel["GAME_ID"] == "G3"].iloc[0]
    assert game3["player_avg_pts"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# GmSc / usage proxy correctness
# ---------------------------------------------------------------------------


def test_game_score_formula_is_exact():
    df = pd.DataFrame(
        [
            {
                "PTS": 30,
                "FGM": 10,
                "FGA": 20,
                "FTA": 6,
                "FTM": 5,
                "OREB": 2,
                "DREB": 6,
                "STL": 2,
                "AST": 5,
                "BLK": 1,
                "PF": 3,
                "TOV": 4,
            }
        ]
    )
    expected = (
        30
        + 0.4 * 10
        - 0.7 * 20
        - 0.4 * (6 - 5)
        + 0.7 * 2
        + 0.3 * 6
        + 2
        + 0.7 * 5
        + 0.7 * 1
        - 0.4 * 3
        - 4
    )
    result = build_v1_features.compute_game_score(df)
    assert result.iloc[0] == pytest.approx(expected)


def test_usage_proxy_formula_is_exact():
    df = pd.DataFrame([{"FGA": 18, "FTA": 5, "TOV": 3}])
    expected = 18 + 0.44 * 5 + 3
    result = build_v1_features.compute_usage_proxy(df)
    assert result.iloc[0] == pytest.approx(expected)


def test_last5_gmsc_uses_shifted_gmsc_history(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    aaa = (
        panel[panel["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    # Every fixture A row shares identical box-score inputs except PTS, so
    # GmSc for each game = pts + (constant offset from the other stats).
    # Verify last5_gmsc at row 5 equals the mean of the first 5 games' GmSc.
    from src.features.build_v1_features import compute_game_score

    raw_for_aaa = (
        fixture_a[fixture_a["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )
    gmsc_values = compute_game_score(raw_for_aaa)
    expected_last5_gmsc = gmsc_values.iloc[0:5].mean()

    assert aaa.loc[5, "last5_gmsc"] == pytest.approx(expected_last5_gmsc)


# ---------------------------------------------------------------------------
# days_rest / is_back_to_back correctness
# ---------------------------------------------------------------------------


def test_days_rest_is_nan_on_first_game_of_a_season(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    aaa = (
        panel[panel["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )
    assert math.isnan(aaa.loc[0, "days_rest"])
    assert aaa.loc[0, "is_back_to_back"] == 0


def test_days_rest_matches_calendar_gap_between_consecutive_games(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    aaa = (
        panel[panel["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    # AAA's games: 10-24, 10-28, 11-01, 11-05, 11-09, 11-13
    expected_gaps = [None, 4, 4, 4, 4, 4]
    for i, expected in enumerate(expected_gaps):
        if expected is None:
            assert math.isnan(aaa.loc[i, "days_rest"])
        else:
            assert aaa.loc[i, "days_rest"] == expected


def test_is_back_to_back_flags_exactly_one_day_gaps():
    rows = [
        _row("2023-24", "G1", "2023-10-24", "AAA", "BBB", True, 101, "P", 10),
        _row(
            "2023-24", "G2", "2023-10-25", "AAA", "BBB", True, 101, "P", 12
        ),  # 1 day later
        _row(
            "2023-24", "G3", "2023-10-28", "AAA", "BBB", True, 101, "P", 14
        ),  # 3 days later
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    panel = (
        build_v1_features.build_v1_feature_panel(df)
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    assert panel.loc[1, "days_rest"] == 1
    assert panel.loc[1, "is_back_to_back"] == 1
    assert panel.loc[2, "days_rest"] == 3
    assert panel.loc[2, "is_back_to_back"] == 0


# ---------------------------------------------------------------------------
# Season-reset behavior
# ---------------------------------------------------------------------------


def test_player_rolling_features_reset_at_season_boundary():
    df = _build_fixture_a_with_second_season()
    panel = build_v1_features.build_v1_feature_panel(df)

    season_2_rows = panel[
        (panel["PLAYER_ID"] == 101) & (panel["SEASON"] == "2024-25")
    ].sort_values("GAME_DATE")
    first_2025_row = season_2_rows.iloc[0]

    # Despite player 101 having 6 games of history in 2023-24, the first
    # 2024-25 row must show fully cold-start rolling features -- no
    # carryover across the season boundary.
    assert math.isnan(first_2025_row["player_avg_pts"])
    assert math.isnan(first_2025_row["last3_pts"])
    assert math.isnan(first_2025_row["last5_pts"])
    assert math.isnan(first_2025_row["days_rest"])
    assert first_2025_row["PRIOR_GAMES_THIS_SEASON"] == 0

    second_2025_row = season_2_rows.iloc[1]
    # The second 2024-25 game DOES see the first 2024-25 game as history --
    # season reset applies at the boundary, not to the whole player.
    assert second_2025_row["player_avg_pts"] == pytest.approx(40.0)
    assert second_2025_row["PRIOR_GAMES_THIS_SEASON"] == 1


def test_season_reset_does_not_affect_other_players_or_seasons():
    df = _build_fixture_a_with_second_season()
    panel = build_v1_features.build_v1_feature_panel(df)

    # BBB's 2023-24 data must be completely unaffected by AAA's 2024-25 rows
    # existing in the same input frame.
    original_panel = build_v1_features.build_v1_feature_panel(_build_fixture_a())
    bbb_from_full = panel[panel["TEAM_ABBREVIATION"] == "BBB"].reset_index(drop=True)
    bbb_from_original = original_panel[
        original_panel["TEAM_ABBREVIATION"] == "BBB"
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(bbb_from_full, bbb_from_original)


# ---------------------------------------------------------------------------
# Cold-start / eligibility
# ---------------------------------------------------------------------------


def test_training_eligible_flag_matches_min_prior_games_threshold(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    aaa = (
        panel[panel["TEAM_ABBREVIATION"] == "AAA"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    for i in range(len(aaa)):
        expected_eligible = 1 if i >= v1_schema.MIN_PRIOR_GAMES_FOR_ELIGIBILITY else 0
        assert aaa.loc[i, "TRAINING_ELIGIBLE"] == expected_eligible
        assert aaa.loc[i, "PRIOR_GAMES_THIS_SEASON"] == i


def test_cold_start_rows_are_kept_in_the_panel_not_dropped(fixture_a):
    panel = build_v1_features.build_v1_feature_panel(fixture_a)
    ineligible = panel[panel["TRAINING_ELIGIBLE"] == 0]
    assert len(ineligible) > 0  # rows below the threshold genuinely exist
    # ...and are still present with identifiers/target intact, not stripped.
    assert ineligible["PTS"].notna().all()
    assert ineligible["PLAYER_ID"].notna().all()


# ---------------------------------------------------------------------------
# Team-game aggregation uniqueness
# ---------------------------------------------------------------------------


def test_team_game_panel_is_unique_on_team_and_game_id(fixture_a):
    team_game = build_v1_features.build_team_game_panel(fixture_a)
    assert (
        team_game.duplicated(subset=["SEASON", "TEAM_ABBREVIATION", "GAME_ID"]).sum()
        == 0
    )
    # 12 games x 2 teams per game = 24 team-game rows.
    assert len(team_game) == 24


def test_team_game_panel_sums_multiple_players_on_the_same_team_game():
    rows = [
        _row("2023-24", "G1", "2023-10-24", "AAA", "BBB", True, 101, "P1", 20, fga=10),
        _row("2023-24", "G1", "2023-10-24", "AAA", "BBB", True, 102, "P2", 15, fga=8),
        _row("2023-24", "G1", "2023-10-24", "BBB", "AAA", False, 201, "P3", 25, fga=12),
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    team_game = build_v1_features.build_team_game_panel(df)

    aaa_row = team_game[team_game["TEAM_ABBREVIATION"] == "AAA"].iloc[0]
    assert aaa_row["PTS_SCORED"] == 35  # 20 + 15, summed across both AAA players
    assert aaa_row["FGA"] == 18  # 10 + 8
    assert aaa_row["PTS_ALLOWED"] == 25  # BBB's single-player score


def test_every_game_has_exactly_two_teams_and_one_home_team(fixture_a):
    team_game = build_v1_features.build_team_game_panel(fixture_a)
    teams_per_game = team_game.groupby("GAME_ID")["TEAM_ABBREVIATION"].nunique()
    assert (teams_per_game == 2).all()
    home_per_game = team_game.groupby("GAME_ID")["IS_HOME"].sum()
    assert (home_per_game == 1).all()


# ---------------------------------------------------------------------------
# Symmetric game-level possession estimate (pace / defensive rating denominator)
# ---------------------------------------------------------------------------


def _build_single_game_fixture(aaa_fga=20, aaa_oreb=3, aaa_fta=6, aaa_tov=4):
    rows = [
        _row(
            "2023-24",
            "G1",
            "2023-10-24",
            "AAA",
            "BBB",
            True,
            101,
            "P1",
            30,
            fga=aaa_fga,
            oreb=aaa_oreb,
            fta=aaa_fta,
            tov=aaa_tov,
        ),
        _row(
            "2023-24",
            "G1",
            "2023-10-24",
            "BBB",
            "AAA",
            False,
            201,
            "P2",
            25,
            fga=18,
            oreb=2,
            fta=8,
            tov=5,
        ),
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_both_teams_in_a_game_share_the_same_game_possessions_est():
    df = _build_single_game_fixture()
    team_game = build_v1_features.build_team_game_panel(df)

    aaa_poss = team_game[team_game["TEAM_ABBREVIATION"] == "AAA"][
        "POSSESSIONS_EST"
    ].iloc[0]
    bbb_poss = team_game[team_game["TEAM_ABBREVIATION"] == "BBB"][
        "POSSESSIONS_EST"
    ].iloc[0]
    expected_game_poss = (aaa_poss + bbb_poss) / 2.0

    aaa_row = team_game[team_game["TEAM_ABBREVIATION"] == "AAA"].iloc[0]
    bbb_row = team_game[team_game["TEAM_ABBREVIATION"] == "BBB"].iloc[0]

    assert aaa_row["GAME_POSSESSIONS_EST"] == pytest.approx(expected_game_poss)
    assert bbb_row["GAME_POSSESSIONS_EST"] == pytest.approx(expected_game_poss)
    # Not just approximately equal -- the two rows of one game must share
    # the exact same possession estimate, since a game has one pace.
    assert aaa_row["GAME_POSSESSIONS_EST"] == bbb_row["GAME_POSSESSIONS_EST"]
    # And it must be the symmetric average, not either team's own estimate
    # alone (own_poss_est != opponent_poss_est in this fixture by design).
    assert aaa_row["POSSESSIONS_EST"] != bbb_row["POSSESSIONS_EST"]


def test_defensive_rating_uses_the_shared_game_possessions_est_not_own_possessions():
    df = _build_single_game_fixture()
    team_game = build_v1_features.build_team_game_panel(df)

    aaa_row = team_game[team_game["TEAM_ABBREVIATION"] == "AAA"].iloc[0]
    bbb_row = team_game[team_game["TEAM_ABBREVIATION"] == "BBB"].iloc[0]

    expected_aaa_def_rating = (
        100.0 * aaa_row["PTS_ALLOWED"] / aaa_row["GAME_POSSESSIONS_EST"]
    )
    expected_bbb_def_rating = (
        100.0 * bbb_row["PTS_ALLOWED"] / bbb_row["GAME_POSSESSIONS_EST"]
    )

    assert aaa_row["DEF_RATING_EST"] == pytest.approx(expected_aaa_def_rating)
    assert bbb_row["DEF_RATING_EST"] == pytest.approx(expected_bbb_def_rating)

    # The bug this guards against: using the DEFENDING team's own
    # (offensive) possession estimate as the denominator instead of the
    # shared game estimate. Assert the two are actually different in this
    # fixture, so this test would fail if that regression were reintroduced.
    wrong_aaa_def_rating = 100.0 * aaa_row["PTS_ALLOWED"] / aaa_row["POSSESSIONS_EST"]
    assert aaa_row["DEF_RATING_EST"] != pytest.approx(wrong_aaa_def_rating)


def test_changing_one_teams_box_score_inputs_shifts_the_shared_possession_estimate_for_both_rows():
    baseline = build_v1_features.build_team_game_panel(_build_single_game_fixture())
    changed = build_v1_features.build_team_game_panel(
        _build_single_game_fixture(aaa_fga=40, aaa_oreb=1, aaa_fta=10, aaa_tov=8)
    )

    baseline_game_poss = baseline["GAME_POSSESSIONS_EST"].iloc[0]
    changed_game_poss = changed["GAME_POSSESSIONS_EST"].iloc[0]
    assert changed_game_poss != pytest.approx(baseline_game_poss)

    # Changed only AAA's own box-score inputs -- BBB's row must show the
    # SAME shifted value, since it's one shared per-game estimate.
    for team in ("AAA", "BBB"):
        baseline_val = baseline[baseline["TEAM_ABBREVIATION"] == team][
            "GAME_POSSESSIONS_EST"
        ].iloc[0]
        changed_val = changed[changed["TEAM_ABBREVIATION"] == team][
            "GAME_POSSESSIONS_EST"
        ].iloc[0]
        assert baseline_val == pytest.approx(baseline_game_poss)
        assert changed_val == pytest.approx(changed_game_poss)


# ---------------------------------------------------------------------------
# Opponent-feature leakage safety (dedicated Game A/B/C/D fixture)
# ---------------------------------------------------------------------------


def _build_opponent_leakage_fixture(future_game_pts=(50, 50)):
    """
    OPP plays: Game A (vs Z1), Game B (vs Z2), Target Game C (vs TEAM_Y --
    the game we're testing opponent features for), Future Game D (vs Z3,
    strictly after C). TEAM_Y's player row for Game C is the one under test.
    """
    rows = [
        # Game A: OPP vs Z1 -- OPP scores 100, allows 90
        _row("2023-24", "GA", "2023-10-24", "OPP", "Z1", True, 901, "OppPlayer", 100),
        _row("2023-24", "GA", "2023-10-24", "Z1", "OPP", False, 902, "Z1Player", 90),
        # Game B: OPP @ Z2 -- OPP scores 95, allows 85
        _row("2023-24", "GB", "2023-10-26", "OPP", "Z2", False, 901, "OppPlayer", 95),
        _row("2023-24", "GB", "2023-10-26", "Z2", "OPP", True, 903, "Z2Player", 85),
        # Game C (TARGET): TEAM_Y vs OPP -- TEAM_Y's player is what we assert on
        _row(
            "2023-24",
            "GC",
            "2023-10-28",
            "TEAM_Y",
            "OPP",
            True,
            500,
            "TargetPlayer",
            22,
        ),
        _row(
            "2023-24", "GC", "2023-10-28", "OPP", "TEAM_Y", False, 901, "OppPlayer", 88
        ),
        # Game D (FUTURE, after C): OPP vs Z3
        _row(
            "2023-24",
            "GD",
            "2023-10-30",
            "OPP",
            "Z3",
            True,
            901,
            "OppPlayer",
            future_game_pts[0],
        ),
        _row(
            "2023-24",
            "GD",
            "2023-10-30",
            "Z3",
            "OPP",
            False,
            904,
            "Z3Player",
            future_game_pts[1],
        ),
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_opponent_features_for_target_game_are_unaffected_by_a_later_game():
    fixture_normal = _build_opponent_leakage_fixture(future_game_pts=(50, 50))
    fixture_drastic = _build_opponent_leakage_fixture(future_game_pts=(200, 3))

    panel_normal = build_v1_features.build_v1_feature_panel(fixture_normal)
    panel_drastic = build_v1_features.build_v1_feature_panel(fixture_drastic)

    target_normal = panel_normal[panel_normal["PLAYER_ID"] == 500].iloc[0]
    target_drastic = panel_drastic[panel_drastic["PLAYER_ID"] == 500].iloc[0]

    for col in (
        "opponent_points_allowed_per_game",
        "opponent_points_allowed_last5",
        "opponent_pace",
        "opponent_defensive_rating",
    ):
        assert target_normal[col] == pytest.approx(target_drastic[col], nan_ok=True)

    # Sanity: the target player's opponent (OPP) faced 2 prior games (A, B)
    # allowing 90 and 85 points -- opponent_points_allowed_per_game entering
    # Game C should be exactly their mean, using ONLY those two games.
    assert target_normal["opponent_points_allowed_per_game"] == pytest.approx(
        (90 + 85) / 2
    )


def test_opponent_features_only_use_opponents_prior_games_not_current_or_future():
    fixture = _build_opponent_leakage_fixture()
    team_game = build_v1_features.build_team_game_panel(fixture)
    rolling = build_v1_features.attach_team_pregame_rolling(team_game)

    opp_rows = (
        rolling[rolling["TEAM_ABBREVIATION"] == "OPP"]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )
    # Row 0 (Game A): no prior OPP games at all.
    assert math.isnan(opp_rows.loc[0, "ROLL_PTS_ALLOWED_PER_GAME"])
    # Row 1 (Game B): only Game A (allowed 90) is prior.
    assert opp_rows.loc[1, "ROLL_PTS_ALLOWED_PER_GAME"] == pytest.approx(90.0)
    # Row 2 (Game C, the target): only Games A and B are prior.
    assert opp_rows.loc[2, "ROLL_PTS_ALLOWED_PER_GAME"] == pytest.approx((90 + 85) / 2)
    # Row 3 (Game D, future): now includes Game C's 22 points allowed too --
    # proving the rolling value DOES change going forward, just never
    # backward into Game C's own attached features.
    assert opp_rows.loc[3, "ROLL_PTS_ALLOWED_PER_GAME"] == pytest.approx(
        (90 + 85 + 22) / 3
    )
