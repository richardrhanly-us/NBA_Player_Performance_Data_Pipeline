"""
Shared pytest fixtures: the synthetic V1-shaped panel used by the
Step 5 (training/experiments/) test suite, and (Step 9) an in-memory
SQLite connection with the prediction-history schema applied, used by
the prediction-repository/settlement/performance test suite in place of
a live Postgres/Neon database (none is reachable in this environment or
in CI -- see the Step 9 report's migration-validation section).
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.services.schema_sqlite import (
    create_sqlite_accounts_schema,
    create_sqlite_prediction_schema,
)


@pytest.fixture
def db_conn():
    """A fresh, isolated in-memory SQLite connection with the
    prediction-history schema (prediction_runs/prediction_snapshots/
    prediction_outcomes) already applied -- real SQL constraint
    enforcement (UNIQUE, CHECK, FOREIGN KEY), fully offline."""
    conn = sqlite3.connect(":memory:")
    create_sqlite_prediction_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def accounts_db_conn():
    """A fresh, isolated in-memory SQLite connection with the Step 11
    accounts schema (users/subscriptions/entitlement_overrides) already
    applied -- same rationale as db_conn above (no live Postgres/Neon
    access here or in CI)."""
    conn = sqlite3.connect(":memory:")
    create_sqlite_accounts_schema(conn)
    yield conn
    conn.close()


def _make_row(rng, game_counter, season, split, game_date, i):
    player_avg = float(rng.uniform(5, 25))
    pts = max(0.0, player_avg + rng.normal(0, 5))
    return {
        "PLAYER_ID": 100 + (i % 5),
        "PLAYER_NAME": f"Player {100 + (i % 5)}",
        "GAME_ID": f"G{game_counter}",
        "GAME_DATE": game_date,
        "SEASON": season,
        "TEAM_ABBREVIATION": "AAA",
        "OPPONENT_ABBREVIATION": "BBB",
        "PTS": pts,
        "player_avg_pts": player_avg,
        "player_avg_pts_sq": player_avg**2,
        "season_minutes_avg": float(rng.uniform(15, 35)),
        "recent_minutes_avg": float(rng.uniform(15, 35)),
        "home_game": int(i % 2),
        "days_rest": float(rng.choice([1, 2, 3])),
        "is_back_to_back": int(i % 7 == 0),
        "last3_pts": player_avg + rng.normal(0, 2),
        "last5_pts": player_avg + rng.normal(0, 2),
        # Deliberately sparse -- exercises native NaN handling, same as
        # the real panel's last10_pts/last20_pts missingness.
        "last10_pts": (player_avg + rng.normal(0, 2)) if i % 3 != 0 else np.nan,
        "last20_pts": (player_avg + rng.normal(0, 2)) if i % 4 != 0 else np.nan,
        "last5_fga": float(rng.uniform(5, 20)),
        "last5_fta": float(rng.uniform(1, 8)),
        "last5_3pa": float(rng.uniform(0, 10)),
        "last5_minutes": float(rng.uniform(15, 35)),
        "last5_gmsc": float(rng.uniform(5, 25)),
        "last5_usage_proxy": float(rng.uniform(8, 25)),
        "minutes_volatility": float(rng.uniform(0, 6)),
        "points_volatility": float(rng.uniform(0, 8)),
        "opponent_points_allowed_per_game": float(rng.uniform(105, 120)),
        "opponent_points_allowed_last5": float(rng.uniform(105, 120)),
        "opponent_pace": float(rng.uniform(95, 105)),
        "opponent_defensive_rating": float(rng.uniform(105, 120)),
        "PRIOR_GAMES_THIS_SEASON": 10,
        "TRAINING_ELIGIBLE": 1,
        "SPLIT": split,
    }


def make_experiments_synthetic_panel(
    n_train_dates=24, rows_per_train_date=6, n_val=60, n_test=30, seed=0
):
    """
    A small but structurally realistic V1 panel spanning three official
    splits: `n_train_dates` distinct calendar dates in "train" (enough
    to build 4-block expanding-window CV folds), plus "validation" and
    "test" rows. Not the real 79k-row panel -- fast, deterministic,
    network-free.
    """
    rng = np.random.RandomState(seed)
    rows = []
    game_counter = 0

    base_date = pd.Timestamp("2023-10-24")
    for d in range(n_train_dates):
        game_date = base_date + pd.Timedelta(days=d)
        for i in range(rows_per_train_date):
            game_counter += 1
            rows.append(_make_row(rng, game_counter, "2023-24", "train", game_date, i))

    val_base = pd.Timestamp("2025-10-30")
    for i in range(n_val):
        game_counter += 1
        game_date = val_base + pd.Timedelta(days=i // 3)
        rows.append(_make_row(rng, game_counter, "2025-26", "validation", game_date, i))

    test_base = pd.Timestamp("2026-02-05")
    for i in range(n_test):
        game_counter += 1
        game_date = test_base + pd.Timedelta(days=i // 3)
        rows.append(_make_row(rng, game_counter, "2025-26", "test", game_date, i))

    return pd.DataFrame(rows)


@pytest.fixture
def experiments_panel_factory():
    return make_experiments_synthetic_panel


@pytest.fixture
def experiments_panel():
    return make_experiments_synthetic_panel()
