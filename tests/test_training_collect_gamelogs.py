"""
End-to-end orchestration tests for collect_season()/run_collection(),
against a fake, offline, in-memory BasketballDataProvider (no nba_client,
no nba_api, no network access). Storage is redirected to a tmp_path for
every test so nothing touches the real training/data/raw/.

Using a fake provider (rather than mocking nba_client, as this file did
before the Step 6 provider refactor) is itself the proof that
collect_season()/run_collection() depend only on the
src.data.basketball.provider.BasketballDataProvider contract, not on any
NBA-specific object -- see src/data/basketball/__init__.py.
"""

import pandas as pd
import pytest

from src.data.basketball.errors import ProviderUnavailableError
from src.data.basketball.models import Player, PlayerGameLog
from training import config
from training.data import collect_gamelogs, storage


@pytest.fixture(autouse=True)
def _isolated_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")


class FakeBasketballDataProvider:
    """
    Minimal in-memory BasketballDataProvider for tests: a fixed roster
    per season, plus a per-player callable that returns either a list of
    PlayerGameLog records or raises ProviderUnavailableError -- the same
    two outcomes a real provider can produce.
    """

    def __init__(self, *, rosters=None, gamelog_fn=None):
        self._rosters = rosters or {}
        self._gamelog_fn = gamelog_fn or (lambda player_id, season: [])
        self.game_log_calls = []
        self.roster_calls = []

    def get_season_roster(self, season):
        self.roster_calls.append(season)
        roster = self._rosters.get(season)
        if roster is None:
            raise ProviderUnavailableError(f"no fake roster configured for {season}")
        return list(roster)

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        self.game_log_calls.append(player_id)
        return self._gamelog_fn(player_id, season)


def _roster(player_ids):
    return [Player(player_id=pid, player_name=f"Player {pid}") for pid in player_ids]


def _game_logs(player_id, n_games=3, season="2023-24"):
    return [
        PlayerGameLog(
            season=season,
            season_id="22023",
            player_id=player_id,
            player_name=f"Player {player_id}",
            game_id=f"00223000{player_id}{g}",
            game_date=pd.Timestamp(f"2023-10-2{g}"),
            matchup="BOS vs MIA",
            team_abbreviation="BOS",
            opponent_abbreviation="MIA",
            is_home=1,
            wl="W",
            minutes="30",
            fgm=8,
            fga=16,
            fg_pct=0.5,
            fg3m=2,
            fg3a=5,
            fg3_pct=0.4,
            ftm=4,
            fta=5,
            ft_pct=0.8,
            oreb=1,
            dreb=5,
            reb=6,
            ast=4,
            stl=1,
            blk=0,
            tov=2,
            pf=2,
            points=22,
            plus_minus=5,
            video_available=1,
        )
        for g in range(n_games)
    ]


def test_collect_season_happy_path():
    calls = []

    def gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2, 3])}, gamelog_fn=gamelog_fn
    )

    summary = collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )

    assert summary["players_succeeded"] == 3
    assert summary["players_failed"] == 0
    assert summary["rows_collected"] == 9  # 3 players x 3 games
    assert sorted(calls) == [1, 2, 3]

    for player_id in (1, 2, 3):
        assert storage.is_player_collected("2023-24", player_id)

    manifest = storage.load_manifest("2023-24")
    assert manifest["players_succeeded"] == 3
    assert manifest["start_time"] is not None
    assert manifest["end_time"] is not None


def test_collect_season_resumes_without_refetching_completed_players():
    calls = []

    def gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2, 3])}, gamelog_fn=gamelog_fn
    )

    collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )
    assert len(calls) == 3

    # Second, independent run against the same (already-populated) storage.
    calls.clear()
    summary = collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )

    assert calls == []  # nothing was refetched
    assert summary["players_attempted"] == 0
    assert summary["total_completed_all_time"] == 3


def test_collect_season_records_failure_without_corrupting_completed_data():
    def gamelog_fn(player_id, season):
        if player_id == 2:
            raise ProviderUnavailableError("simulated persistent failure for player 2")
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2, 3])}, gamelog_fn=gamelog_fn
    )

    summary = collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
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
    calls = []

    def retry_gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    retry_provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2, 3])}, gamelog_fn=retry_gamelog_fn
    )
    summary2 = collect_gamelogs.collect_season(
        "2023-24", provider=retry_provider, sleep_func=lambda *_: None, request_delay=0
    )

    assert calls == [2]
    assert summary2["players_succeeded"] == 1
    assert summary2["players_failed"] == 0
    assert storage.is_player_collected("2023-24", 2)

    manifest2 = storage.load_manifest("2023-24")
    assert manifest2["failed_players"] == {}
    assert manifest2["players_succeeded"] == 3


def test_collect_season_refetches_when_manifest_says_done_but_file_is_missing():
    """
    Regression test: a stale manifest entry alone must never certify a
    player as collected. If the manifest claims player 1 is complete but no
    Parquet file exists for them, is_player_collected() (not the manifest)
    is authoritative, and the player must be fetched again.
    """
    calls = []

    def gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2])}, gamelog_fn=gamelog_fn
    )

    # Hand-craft a stale manifest: player 1 is claimed complete, but its
    # Parquet file was never actually written.
    manifest = storage.load_manifest("2023-24")
    manifest["completed_player_ids"] = [1]
    manifest["player_row_counts"] = {"1": 3}
    manifest["players_succeeded"] = 1
    storage.save_manifest("2023-24", manifest)
    assert not storage.is_player_collected("2023-24", 1)  # file genuinely absent

    summary = collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )

    assert 1 in calls  # refetched despite the stale manifest claim
    assert sorted(calls) == [1, 2]
    assert summary["players_succeeded"] == 2
    assert storage.is_player_collected("2023-24", 1)

    # The repair must be reflected correctly in the manifest afterward.
    final_manifest = storage.load_manifest("2023-24")
    assert final_manifest["completed_player_ids"] == [1, 2]
    assert final_manifest["failed_players"] == {}
    assert (
        final_manifest["player_row_counts"]["1"] == 3
    )  # real row count from the fetch


def test_collect_season_refetches_when_manifest_says_done_but_file_is_corrupt():
    """
    Regression test: same as above, but the manifest-claimed player's file
    exists on disk and is unreadable (corrupt). is_player_collected() must
    catch this and force a refetch rather than trusting the manifest.
    """
    calls = []

    def gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2])}, gamelog_fn=gamelog_fn
    )

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
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )

    assert 1 in calls  # refetched despite the stale manifest claim
    assert sorted(calls) == [1, 2]
    assert summary["players_succeeded"] == 2
    assert storage.is_player_collected("2023-24", 1)

    final_manifest = storage.load_manifest("2023-24")
    assert final_manifest["completed_player_ids"] == [1, 2]
    assert final_manifest["failed_players"] == {}
    assert final_manifest["player_row_counts"]["1"] == 3


def test_collect_season_removes_stale_completed_id_if_repair_fetch_also_fails():
    """
    If a stale-"completed" player's file is missing/corrupt AND the repair
    fetch also fails, the manifest must not end up claiming they're both
    completed and failed -- completed_player_ids must drop them.
    """

    def always_fails(player_id, season):
        raise ProviderUnavailableError("still down")

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1])}, gamelog_fn=always_fails
    )

    manifest = storage.load_manifest("2023-24")
    manifest["completed_player_ids"] = [1]
    manifest["player_row_counts"] = {"1": 3}
    manifest["players_succeeded"] = 1
    storage.save_manifest("2023-24", manifest)

    summary = collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )

    assert summary["players_failed"] == 1
    assert not storage.is_player_collected("2023-24", 1)

    final_manifest = storage.load_manifest("2023-24")
    assert final_manifest["completed_player_ids"] == []
    assert "1" not in final_manifest["player_row_counts"]
    assert "1" in final_manifest["failed_players"]


def test_collect_season_respects_max_players_and_is_resumable_in_chunks():
    calls = []

    def gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2, 3])}, gamelog_fn=gamelog_fn
    )

    summary1 = collect_gamelogs.collect_season(
        "2023-24",
        provider=provider,
        max_players=1,
        sleep_func=lambda *_: None,
        request_delay=0,
    )
    assert summary1["players_succeeded"] == 1
    assert len(calls) == 1

    # Same call again, still capped at 1 new player -- should pick up the next one.
    summary2 = collect_gamelogs.collect_season(
        "2023-24",
        provider=provider,
        max_players=1,
        sleep_func=lambda *_: None,
        request_delay=0,
    )
    assert summary2["players_succeeded"] == 1
    assert len(calls) == 2

    # Finish the rest with no cap.
    summary3 = collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )
    assert summary3["players_succeeded"] == 1
    assert len(calls) == 3
    assert summary3["total_completed_all_time"] == 3


def test_collect_season_force_player_refetches_despite_being_completed():
    calls = []

    def gamelog_fn(player_id, season):
        calls.append(player_id)
        return _game_logs(player_id)

    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1, 2])}, gamelog_fn=gamelog_fn
    )

    collect_gamelogs.collect_season(
        "2023-24", provider=provider, sleep_func=lambda *_: None, request_delay=0
    )
    assert calls == [1, 2]

    calls.clear()
    summary = collect_gamelogs.collect_season(
        "2023-24",
        provider=provider,
        force_player_ids=[1],
        sleep_func=lambda *_: None,
        request_delay=0,
    )

    assert calls == [1]
    assert summary["players_succeeded"] == 1


def test_run_collection_continues_to_next_season_if_roster_fetch_fails():
    provider = FakeBasketballDataProvider(
        rosters={"2024-25": _roster([1])},  # deliberately missing "2023-24"
        gamelog_fn=lambda player_id, season: _game_logs(player_id),
    )

    summaries = collect_gamelogs.run_collection(
        seasons=["2023-24", "2024-25"],
        provider=provider,
        sleep_func=lambda *_: None,
        request_delay=0,
    )

    assert summaries[0]["season"] == "2023-24"
    assert "error" in summaries[0]
    assert summaries[1]["season"] == "2024-25"
    assert summaries[1]["players_succeeded"] == 1


def test_run_collection_reuses_one_provider_instance_across_seasons():
    provider = FakeBasketballDataProvider(
        rosters={"2023-24": _roster([1]), "2024-25": _roster([2])},
        gamelog_fn=lambda player_id, season: _game_logs(player_id),
    )

    collect_gamelogs.run_collection(
        seasons=["2023-24", "2024-25"],
        provider=provider,
        sleep_func=lambda *_: None,
        request_delay=0,
    )

    assert provider.roster_calls == ["2023-24", "2024-25"]
    assert sorted(provider.game_log_calls) == [1, 2]


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
