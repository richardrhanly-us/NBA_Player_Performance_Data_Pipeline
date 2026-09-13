"""
Step 6 parity tests: prove that swapping the historical collector's data
path from "raw nba_api DataFrame straight into storage.normalize_raw_gamelog"
(the pre-Step-6 implementation) to "NBAApiProvider -> canonical
PlayerGameLog records -> storage.gamelogs_to_dataframe" (the new
provider-boundary implementation) produces IDENTICAL V1 features and
IDENTICAL predictions from the existing frozen model.

Synthetic-but-realistic multi-player, multi-game fixture (not the real
79k-row panel) so this runs fast and offline -- no network access, no
dependency on training/data/raw/ existing on disk. Uses
unittest.mock.patch on nba_client's own endpoint classes, exactly like
tests/test_basketball_provider.py, so the "new path" genuinely exercises
NBAApiProvider end-to-end rather than skipping it.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.basketball.providers.nba_api_provider import NBAApiProvider
from src.features import build_v1_features, v1_schema
from training import model_registry
from training.data import storage

_CURRENT_VERSION = model_registry.get_current_version()
_FROZEN_MODEL_AVAILABLE = (
    _CURRENT_VERSION is not None
    and (model_registry.version_dir(_CURRENT_VERSION) / "model.pkl").exists()
)

SEASON = "2023-24"

_PLAYERS = {
    2544: "LeBron James",
    201939: "Stephen Curry",
    1629029: "Luka Doncic",
}

_TEAM_FOR_PLAYER = {2544: "LAL", 201939: "GSW", 1629029: "DAL"}
_OPPONENTS = ["DEN", "PHX", "BOS", "MIA", "NYK", "CHI", "MIL", "ATL"]


def _raw_gamelog_df(player_id, n_games=12, seed=0):
    """Nba_api-shaped raw PlayerGameLog rows (Player_ID/Game_ID casing) for
    one player across n_games, alternating home/away against a rotating
    set of opponents, with enough games to clear
    MIN_PRIOR_GAMES_FOR_ELIGIBILITY and exercise rolling windows."""
    rng = np.random.RandomState(seed + player_id)
    team = _TEAM_FOR_PLAYER[player_id]
    rows = []
    for g in range(n_games):
        opponent = _OPPONENTS[(player_id + g) % len(_OPPONENTS)]
        is_home = g % 2 == 0
        matchup = f"{team} vs {opponent}" if is_home else f"{team} @ {opponent}"
        pts = float(rng.randint(10, 35))
        rows.append(
            {
                "SEASON_ID": "22023",
                "Player_ID": player_id,
                "Game_ID": f"00223{player_id}{g:03d}",
                "GAME_DATE": f"2023-10-{24 + g:02d}"
                if g < 7
                else f"2023-11-{g - 6:02d}",
                "MATCHUP": matchup,
                "WL": "W" if pts > 20 else "L",
                # Real nba_api PlayerGameLog responses return MIN as a
                # numeric type (verified against real collected raw data,
                # e.g. training/data/raw/2023-24/players/2544.parquet ->
                # MIN dtype int64) -- match that here rather than a
                # string, which build_v1_features.build_team_game_panel
                # sums and divides directly.
                "MIN": float(rng.randint(20, 40)),
                "FGM": float(rng.randint(3, 14)),
                "FGA": float(rng.randint(8, 25)),
                "FG_PCT": 0.45,
                "FG3M": float(rng.randint(0, 5)),
                "FG3A": float(rng.randint(1, 10)),
                "FG3_PCT": 0.35,
                "FTM": float(rng.randint(0, 8)),
                "FTA": float(rng.randint(0, 10)),
                "FT_PCT": 0.8,
                "OREB": float(rng.randint(0, 3)),
                "DREB": float(rng.randint(2, 8)),
                "REB": float(rng.randint(3, 10)),
                "AST": float(rng.randint(1, 10)),
                "STL": float(rng.randint(0, 3)),
                "BLK": float(rng.randint(0, 2)),
                "TOV": float(rng.randint(0, 5)),
                "PF": float(rng.randint(0, 4)),
                "PTS": pts,
                "PLUS_MINUS": float(rng.randint(-15, 15)),
                "VIDEO_AVAILABLE": 1,
            }
        )
    return pd.DataFrame(rows)


def _build_old_path_panel():
    frames = []
    for player_id, player_name in _PLAYERS.items():
        raw = _raw_gamelog_df(player_id)
        normalized = storage.normalize_raw_gamelog(
            raw, season=SEASON, player_id=player_id, player_name=player_name
        )
        frames.append(normalized)
    combined = pd.concat(frames, ignore_index=True)
    return build_v1_features.build_v1_feature_panel(combined)


def _build_new_path_panel():
    """Exercises NBAApiProvider end-to-end (mocking only nba_api's own
    endpoint class, exactly like a real fetch would return), then the
    same storage.gamelogs_to_dataframe reconstruction collect_gamelogs.py
    now uses."""
    frames = []
    for player_id, player_name in _PLAYERS.items():
        fake_response = MagicMock()
        fake_response.get_data_frames.return_value = [_raw_gamelog_df(player_id)]
        with patch(
            "training.data.nba_client.playergamelog.PlayerGameLog",
            return_value=fake_response,
        ):
            provider = NBAApiProvider(sleep_func=lambda *_: None)
            records = provider.get_player_game_logs(
                player_id, SEASON, player_name=player_name
            )
        frames.append(storage.gamelogs_to_dataframe(records))
    combined = pd.concat(frames, ignore_index=True)
    return build_v1_features.build_v1_feature_panel(combined)


def test_v1_feature_panel_is_identical_between_old_and_new_paths():
    old_panel = _build_old_path_panel()
    new_panel = _build_new_path_panel()

    pd.testing.assert_frame_equal(old_panel, new_panel)


def test_v1_feature_panel_has_same_eligible_rows_and_identity_columns():
    old_panel = _build_old_path_panel()
    new_panel = _build_new_path_panel()

    assert len(old_panel) == len(new_panel)
    assert old_panel["TRAINING_ELIGIBLE"].sum() == new_panel["TRAINING_ELIGIBLE"].sum()
    assert list(old_panel["PLAYER_ID"]) == list(new_panel["PLAYER_ID"])
    assert list(old_panel["GAME_ID"]) == list(new_panel["GAME_ID"])
    assert list(old_panel["GAME_DATE"]) == list(new_panel["GAME_DATE"])


def test_v1_feature_values_match_within_floating_point_tolerance():
    """Belt-and-suspenders: even though assert_frame_equal above already
    proves exact equality, explicitly check every V1 feature column with
    an explicit numeric tolerance, as the Step 6 brief requests."""
    old_panel = _build_old_path_panel()
    new_panel = _build_new_path_panel()

    for col in v1_schema.V1_FEATURE_NAMES:
        old_values = old_panel[col].to_numpy(dtype=float)
        new_values = new_panel[col].to_numpy(dtype=float)
        np.testing.assert_allclose(
            old_values, new_values, rtol=0, atol=1e-9, equal_nan=True
        )


@pytest.mark.skipif(
    not _FROZEN_MODEL_AVAILABLE,
    reason=(
        "models/registry/<CURRENT_V1>/model.pkl is gitignored (large, "
        "regenerable via `python -m training.train`) and not present in "
        "this checkout -- prediction parity needs it locally."
    ),
)
def test_model_prediction_parity_between_old_and_new_paths():
    """
    Uses the existing frozen Step 4 model (no retraining) to predict on
    the eligible rows of both panels and confirms prediction delta is
    zero (machine precision). Reports the same three figures the Step 6
    brief asks for via assertions: rows compared, max abs feature diff,
    max abs prediction diff.
    """
    old_panel = _build_old_path_panel()
    new_panel = _build_new_path_panel()

    old_eligible = old_panel[old_panel["TRAINING_ELIGIBLE"] == 1].reset_index(drop=True)
    new_eligible = new_panel[new_panel["TRAINING_ELIGIBLE"] == 1].reset_index(drop=True)
    assert len(old_eligible) > 0  # sanity: the fixture must produce eligible rows

    version = model_registry.get_current_version()
    model = model_registry.load_model(version)

    X_old = old_eligible[list(v1_schema.V1_FEATURE_NAMES)]
    X_new = new_eligible[list(v1_schema.V1_FEATURE_NAMES)]

    max_abs_feature_diff = float(
        np.nanmax(np.abs(X_old.to_numpy(dtype=float) - X_new.to_numpy(dtype=float)))
    )
    pred_old = model.predict(X_old)
    pred_new = model.predict(X_new)
    max_abs_prediction_diff = float(np.max(np.abs(pred_old - pred_new)))

    assert max_abs_feature_diff == 0.0
    assert max_abs_prediction_diff == 0.0
    assert len(X_old) == len(X_new)
