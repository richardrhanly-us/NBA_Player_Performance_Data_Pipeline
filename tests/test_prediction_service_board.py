"""
Tests for src/services/prediction_service.py::build_daily_prediction_board
(Step 8 test list items 7, 8, 15, 16, 17, 18, 19), plus PredictionBoard's
EDGE_THRESHOLD-based `qualified()` helper (#5). Fake, offline sportsbook
props DataFrame + fake provider -- no real network access.
"""

import pandas as pd

from src.data.basketball.models import Player, PlayerDetails
from src.services.prediction_service import build_daily_prediction_board
from tests.test_prediction_service_single import (
    FakeProvider,
    _FakeModel,
    _sufficient_history,
)

SEASON = "2025-26"


def _props_df(rows):
    """Matches shared_app.fetch_all_today_player_props's real output
    columns exactly."""
    return pd.DataFrame(rows)


def _prop_row(
    name, line, home_team="Lakers", away_team="Nuggets", bookmaker_key="draftkings"
):
    return {
        "player_name_raw": name,
        "line": line,
        "bookmaker": bookmaker_key.title(),
        "bookmaker_key": bookmaker_key,
        "last_update": "2026-01-15T00:00:00Z",
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": "2026-01-15T20:00:00Z",
        "over_price": -110,
        "under_price": -110,
    }


def _board_provider(players):
    """players: dict player_id -> (name, n_games, projection-friendly history)."""
    active = [
        Player(player_id=pid, player_name=name) for pid, (name, _) in players.items()
    ]
    game_logs = {pid: _sufficient_history(pid, n) for pid, (_, n) in players.items()}
    details = {
        pid: PlayerDetails(
            player_id=pid,
            team_id=1000 + pid,
            team_name=f"Team{pid}",
            team_abbreviation=f"T{pid}",
            position="F",
        )
        for pid in players
    }
    return FakeProvider(
        active_players=active, game_logs=game_logs, player_details=details
    )


# ---- 7. multiple players produce a board -----------------------------------


def test_multiple_players_produce_a_board_with_all_predictions():
    provider = _board_provider(
        {1: ("Player One", 10), 2: ("Player Two", 10), 3: ("Player Three", 10)}
    )
    props_df = _props_df(
        [
            _prop_row("Player One", 15.0),
            _prop_row("Player Two", 25.0),
            _prop_row("Player Three", 20.0),
        ]
    )

    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert board.props_discovered == 3
    assert board.players_matched == 3
    assert board.predictions_generated == 3
    assert len(board.predictions) == 3
    assert {p.player_name for p in board.predictions} == {
        "Player One",
        "Player Two",
        "Player Three",
    }


# ---- 8. board ordering -------------------------------------------------


def test_board_is_ordered_by_absolute_edge_descending():
    provider = _board_provider(
        {1: ("Player One", 10), 2: ("Player Two", 10), 3: ("Player Three", 10)}
    )
    # projection fixed at 20.0 -> edges: One=+5, Two=-10, Three=+1
    props_df = _props_df(
        [
            _prop_row("Player One", 15.0),
            _prop_row("Player Two", 30.0),
            _prop_row("Player Three", 19.0),
        ]
    )

    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    abs_edges = [abs(p.edge) for p in board.predictions]
    assert abs_edges == sorted(abs_edges, reverse=True)
    assert board.predictions[0].player_name == "Player Two"  # |edge|=10, largest


def test_qualified_highlights_without_hiding_other_predictions():
    provider = _board_provider({1: ("Player One", 10), 2: ("Player Two", 10)})
    # One: edge=+1 (below threshold), Two: edge=+5 (above threshold)
    props_df = _props_df([_prop_row("Player One", 19.0), _prop_row("Player Two", 15.0)])

    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert len(board.predictions) == 2  # both present -- nothing hidden
    qualified = board.qualified(edge_threshold=3.0)
    assert len(qualified) == 1
    assert qualified[0].player_name == "Player Two"


# ---- 15. no games today / 16. no props today --------------------------


def test_no_props_today_returns_empty_board_not_an_error():
    provider = _board_provider({})
    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(),
        props_fetcher=lambda *_: pd.DataFrame(),
        sleep_func=lambda *_: None,
    )
    assert board.predictions == ()
    assert board.props_discovered == 0
    assert board.players_matched == 0


def test_props_fetch_failure_returns_empty_board_not_an_exception():
    def _broken_fetcher(*_args):
        raise RuntimeError("odds API down")

    provider = _board_provider({})
    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(),
        props_fetcher=_broken_fetcher,
        sleep_func=lambda *_: None,
    )
    assert board.predictions == ()


# ---- 17. model loaded once for batch -----------------------------------


def test_model_is_not_reloaded_per_player_when_preloaded():
    """When a model instance is explicitly passed in, build_daily_prediction_board
    must never call shared_app.load_model() itself -- verified by making
    load_model raise if called."""
    import src.shared_app as shared_app_module

    def _explode():
        raise AssertionError(
            "load_model() should not be called when model= is provided"
        )

    original = shared_app_module.load_model
    shared_app_module.load_model = _explode
    try:
        provider = _board_provider({1: ("Player One", 10), 2: ("Player Two", 10)})
        props_df = _props_df(
            [_prop_row("Player One", 15.0), _prop_row("Player Two", 25.0)]
        )
        board = build_daily_prediction_board(
            api_key="fake-key",
            provider=provider,
            model=_FakeModel(),
            props_fetcher=lambda *_: props_df,
            sleep_func=lambda *_: None,
        )
    finally:
        shared_app_module.load_model = original

    assert board.predictions_generated == 2


# ---- 18. duplicate external calls avoided where expected ----------------


def test_active_players_and_gamelog_fetched_once_per_unique_player_only():
    provider = _board_provider({1: ("Player One", 10), 2: ("Player Two", 10)})
    props_df = _props_df(
        [
            _prop_row("Player One", 15.0),
            _prop_row("Player Two", 25.0),
            _prop_row("Player One", 15.5),  # duplicate raw name for the same player
        ]
    )

    build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    # get_active_players is called exactly once for the whole board build.
    assert provider.active_players_calls == 1
    # get_player_game_logs is called exactly once per unique player_id --
    # never twice for the same player even with a duplicate prop row.
    assert sorted(provider.gamelog_calls) == [1, 2]


def test_unmatched_and_unavailable_counts_are_reported():
    provider = _board_provider({1: ("Player One", 10)})
    props_df = _props_df(
        [_prop_row("Player One", 15.0), _prop_row("Totally Unknown Player", 10.0)]
    )

    board = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert board.props_discovered == 2
    assert board.players_matched == 1
    assert board.unmatched_count == 1
    assert board.predictions_generated == 1


# ---- 19. deterministic rebuild (underlies safe caching) -----------------


def test_board_build_is_deterministic_given_identical_inputs():
    provider = _board_provider({1: ("Player One", 10), 2: ("Player Two", 10)})
    props_df = _props_df([_prop_row("Player One", 15.0), _prop_row("Player Two", 25.0)])

    board_a = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )
    board_b = build_daily_prediction_board(
        api_key="fake-key",
        provider=provider,
        model=_FakeModel(fixed_prediction=20.0),
        props_fetcher=lambda *_: props_df,
        sleep_func=lambda *_: None,
    )

    assert [
        (p.player_name, p.model_projection, p.edge) for p in board_a.predictions
    ] == [(p.player_name, p.model_projection, p.edge) for p in board_b.predictions]


def test_pregame_only_board_never_applies_live_adjustment():
    """build_daily_prediction_board must never call get_live_player_stats
    -- confirmed by making it raise if called."""
    import src.shared_app as shared_app_module

    def _explode(*_args, **_kwargs):
        raise AssertionError("get_live_player_stats should not be called by the board")

    original = shared_app_module.get_live_player_stats
    shared_app_module.get_live_player_stats = _explode
    try:
        provider = _board_provider({1: ("Player One", 10)})
        props_df = _props_df([_prop_row("Player One", 15.0)])
        board = build_daily_prediction_board(
            api_key="fake-key",
            provider=provider,
            model=_FakeModel(),
            props_fetcher=lambda *_: props_df,
            sleep_func=lambda *_: None,
        )
    finally:
        shared_app_module.get_live_player_stats = original

    assert board.predictions_generated == 1
