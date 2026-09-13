"""
End-to-end orchestration tests for collect_season()/run_collection(), with
both nba_client fetch functions mocked. No network access. Storage is
redirected to a tmp_path for every test so nothing touches the real
training/data/raw/.
"""

import pandas as pd
import pytest

from training import config
from training.data import collect_gamelogs, nba_client, storage


@pytest.fixture(autouse=True)
def _isolated_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")


def _roster_df(player_ids):
    return pd.DataFrame(
        {
            "PLAYER_ID": player_ids,
            "PLAYER_NAME": [f"Player {pid}" for pid in player_ids],
            "GP": [10] * len(player_ids),
        }
    )


def _gamelog_df(player_id, n_games=3):
    return pd.DataFrame(
        {
            "SEASON_ID": ["22023"] * n_games,
            "Player_ID": [player_id] * n_games,
            "Game_ID": [f"00223000{player_id}{g}" for g in range(n_games)],
            "GAME_DATE": [f"2023-10-2{g}" for g in range(n_games)],
            "MATCHUP": ["BOS vs MIA"] * n_games,
            "WL": ["W"] * n_games,
            "MIN": ["30"] * n_games,
            "FGM": [8] * n_games,
            "FGA": [16] * n_games,
            "FG_PCT": [0.5] * n_games,
            "FG3M": [2] * n_games,
            "FG3A": [5] * n_games,
            "FG3_PCT": [0.4] * n_games,
            "FTM": [4] * n_games,
            "FTA": [5] * n_games,
            "FT_PCT": [0.8] * n_games,
            "OREB": [1] * n_games,
            "DREB": [5] * n_games,
            "REB": [6] * n_games,
            "AST": [4] * n_games,
            "STL": [1] * n_games,
            "BLK": [0] * n_games,
            "TOV": [2] * n_games,
            "PF": [2] * n_games,
            "PTS": [22] * n_games,
            "PLUS_MINUS": [5] * n_games,
            "VIDEO_AVAILABLE": [1] * n_games,
        }
    )


def test_collect_season_happy_path(monkeypatch):
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2, 3])
    )
    fetch_mock_calls = []

    def fake_fetch_gamelog(player_id, season, **_):
        fetch_mock_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    summary = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert summary["players_succeeded"] == 3
    assert summary["players_failed"] == 0
    assert summary["rows_collected"] == 9  # 3 players x 3 games
    assert sorted(fetch_mock_calls) == [1, 2, 3]

    for player_id in (1, 2, 3):
        assert storage.is_player_collected("2023-24", player_id)

    manifest = storage.load_manifest("2023-24")
    assert manifest["players_succeeded"] == 3
    assert manifest["start_time"] is not None
    assert manifest["end_time"] is not None


def test_collect_season_resumes_without_refetching_completed_players(monkeypatch):
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2, 3])
    )
    fetch_calls = []

    def fake_fetch_gamelog(player_id, season, **_):
        fetch_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )
    assert len(fetch_calls) == 3

    # Second, independent run against the same (already-populated) storage.
    fetch_calls.clear()
    summary = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert fetch_calls == []  # nothing was refetched
    assert summary["players_attempted"] == 0
    assert summary["total_completed_all_time"] == 3


def test_collect_season_records_failure_without_corrupting_completed_data(monkeypatch):
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2, 3])
    )

    def fake_fetch_gamelog(player_id, season, **_):
        if player_id == 2:
            raise nba_client.NbaApiError("simulated persistent failure for player 2")
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    summary = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert summary["players_succeeded"] == 2
    assert summary["players_failed"] == 1
    assert storage.is_player_collected("2023-24", 1)
    assert storage.is_player_collected("2023-24", 3)
    assert not storage.is_player_collected("2023-24", 2)

    manifest = storage.load_manifest("2023-24")
    assert "2" in manifest["failed_players"]
    assert manifest["failed_players"]["2"]["error"]

    # Rerun: only the previously-failed player should be retried.
    fetch_calls = []

    def fake_fetch_gamelog_retry(player_id, season, **_):
        fetch_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog_retry)
    summary2 = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert fetch_calls == [2]
    assert summary2["players_succeeded"] == 1
    assert summary2["players_failed"] == 0
    assert storage.is_player_collected("2023-24", 2)

    manifest2 = storage.load_manifest("2023-24")
    assert manifest2["failed_players"] == {}
    assert manifest2["players_succeeded"] == 3


def test_collect_season_refetches_when_manifest_says_done_but_file_is_missing(
    monkeypatch,
):
    """
    Regression test: a stale manifest entry alone must never certify a
    player as collected. If the manifest claims player 1 is complete but no
    Parquet file exists for them, is_player_collected() (not the manifest)
    is authoritative, and the player must be fetched again.
    """
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2])
    )
    fetch_calls = []

    def fake_fetch_gamelog(player_id, season, **_):
        fetch_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    # Hand-craft a stale manifest: player 1 is claimed complete, but its
    # Parquet file was never actually written.
    manifest = storage.load_manifest("2023-24")
    manifest["completed_player_ids"] = [1]
    manifest["player_row_counts"] = {"1": 3}
    manifest["players_succeeded"] = 1
    storage.save_manifest("2023-24", manifest)
    assert not storage.is_player_collected("2023-24", 1)  # file genuinely absent

    summary = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert 1 in fetch_calls  # refetched despite the stale manifest claim
    assert sorted(fetch_calls) == [1, 2]
    assert summary["players_succeeded"] == 2
    assert storage.is_player_collected("2023-24", 1)

    # The repair must be reflected correctly in the manifest afterward.
    final_manifest = storage.load_manifest("2023-24")
    assert final_manifest["completed_player_ids"] == [1, 2]
    assert final_manifest["failed_players"] == {}
    assert (
        final_manifest["player_row_counts"]["1"] == 3
    )  # real row count from the fetch


def test_collect_season_refetches_when_manifest_says_done_but_file_is_corrupt(
    monkeypatch,
):
    """
    Regression test: same as above, but the manifest-claimed player's file
    exists on disk and is unreadable (corrupt). is_player_collected() must
    catch this and force a refetch rather than trusting the manifest.
    """
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2])
    )
    fetch_calls = []

    def fake_fetch_gamelog(player_id, season, **_):
        fetch_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    # Corrupt file on disk for player 1, plus a manifest that (wrongly)
    # claims player 1 is already complete.
    path = storage.player_file_path("2023-24", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real parquet file")

    manifest = storage.load_manifest("2023-24")
    manifest["completed_player_ids"] = [1]
    manifest["player_row_counts"] = {"1": 3}
    manifest["players_succeeded"] = 1
    storage.save_manifest("2023-24", manifest)
    assert not storage.is_player_collected("2023-24", 1)  # corrupt, not usable

    summary = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert 1 in fetch_calls  # refetched despite the stale manifest claim
    assert sorted(fetch_calls) == [1, 2]
    assert summary["players_succeeded"] == 2
    assert storage.is_player_collected("2023-24", 1)

    final_manifest = storage.load_manifest("2023-24")
    assert final_manifest["completed_player_ids"] == [1, 2]
    assert final_manifest["failed_players"] == {}
    assert final_manifest["player_row_counts"]["1"] == 3


def test_collect_season_removes_stale_completed_id_if_repair_fetch_also_fails(
    monkeypatch,
):
    """
    If a stale-"completed" player's file is missing/corrupt AND the repair
    fetch also fails, the manifest must not end up claiming they're both
    completed and failed -- completed_player_ids must drop them.
    """
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1])
    )

    def always_fails(player_id, season, **_):
        raise nba_client.NbaApiError("still down")

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", always_fails)

    manifest = storage.load_manifest("2023-24")
    manifest["completed_player_ids"] = [1]
    manifest["player_row_counts"] = {"1": 3}
    manifest["players_succeeded"] = 1
    storage.save_manifest("2023-24", manifest)

    summary = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )

    assert summary["players_failed"] == 1
    assert not storage.is_player_collected("2023-24", 1)

    final_manifest = storage.load_manifest("2023-24")
    assert final_manifest["completed_player_ids"] == []
    assert "1" not in final_manifest["player_row_counts"]
    assert "1" in final_manifest["failed_players"]


def test_collect_season_respects_max_players_and_is_resumable_in_chunks(monkeypatch):
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2, 3])
    )
    fetch_calls = []

    def fake_fetch_gamelog(player_id, season, **_):
        fetch_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    summary1 = collect_gamelogs.collect_season(
        "2023-24", max_players=1, sleep_func=lambda *_: None, request_delay=0
    )
    assert summary1["players_succeeded"] == 1
    assert len(fetch_calls) == 1

    # Same call again, still capped at 1 new player -- should pick up the next one.
    summary2 = collect_gamelogs.collect_season(
        "2023-24", max_players=1, sleep_func=lambda *_: None, request_delay=0
    )
    assert summary2["players_succeeded"] == 1
    assert len(fetch_calls) == 2

    # Finish the rest with no cap.
    summary3 = collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )
    assert summary3["players_succeeded"] == 1
    assert len(fetch_calls) == 3
    assert summary3["total_completed_all_time"] == 3


def test_collect_season_force_player_refetches_despite_being_completed(monkeypatch):
    monkeypatch.setattr(
        nba_client, "fetch_season_roster", lambda season, **_: _roster_df([1, 2])
    )
    fetch_calls = []

    def fake_fetch_gamelog(player_id, season, **_):
        fetch_calls.append(player_id)
        return _gamelog_df(player_id)

    monkeypatch.setattr(nba_client, "fetch_player_gamelog", fake_fetch_gamelog)

    collect_gamelogs.collect_season(
        "2023-24", sleep_func=lambda *_: None, request_delay=0
    )
    assert fetch_calls == [1, 2]

    fetch_calls.clear()
    summary = collect_gamelogs.collect_season(
        "2023-24", force_player_ids=[1], sleep_func=lambda *_: None, request_delay=0
    )

    assert fetch_calls == [1]
    assert summary["players_succeeded"] == 1


def test_run_collection_continues_to_next_season_if_roster_fetch_fails(monkeypatch):
    def fake_roster(season, **_):
        if season == "2023-24":
            raise nba_client.NbaApiError("roster fetch down")
        return _roster_df([1])

    monkeypatch.setattr(nba_client, "fetch_season_roster", fake_roster)
    monkeypatch.setattr(
        nba_client,
        "fetch_player_gamelog",
        lambda player_id, season, **_: _gamelog_df(player_id),
    )

    summaries = collect_gamelogs.run_collection(
        seasons=["2023-24", "2024-25"], sleep_func=lambda *_: None, request_delay=0
    )

    assert summaries[0]["season"] == "2023-24"
    assert "error" in summaries[0]
    assert summaries[1]["season"] == "2024-25"
    assert summaries[1]["players_succeeded"] == 1


def test_build_arg_parser_defaults_and_options():
    parser = collect_gamelogs.build_arg_parser()

    args = parser.parse_args([])
    assert args.season is None
    assert args.max_players is None
    assert args.force_player == ()
    assert args.no_resume is False

    args = parser.parse_args(
        [
            "--season",
            "2025-26",
            "--max-players",
            "10",
            "--force-player",
            "2544",
            "1629029",
            "--no-resume",
        ]
    )
    assert args.season == ["2025-26"]
    assert args.max_players == 10
    assert args.force_player == [2544, 1629029]
    assert args.no_resume is True
