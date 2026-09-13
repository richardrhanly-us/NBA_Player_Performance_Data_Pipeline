"""
Step 9 settlement tests (test list items 7-17). Uses the in-memory
SQLite `db_conn` fixture plus a fake BasketballDataProvider -- no real
network access, no live game data required.
"""

import pandas as pd

from src.data.basketball.errors import ProviderUnavailableError
from src.data.basketball.models import PlayerGameLog, ScheduledGame
from src.services.prediction_repository import (
    get_outcome_for_snapshot,
    persist_prediction_run,
)
from src.services.prediction_result import (
    PredictionDirection,
    PredictionResult,
    PredictionStatus,
)
from src.services.prediction_service import PredictionBoard
from src.services.settlement_service import (
    grade_points_prediction,
    settle_pending_predictions,
    settle_prediction,
)

PLAYER_ID = 1
GAME_ID = "G1"
GAME_DATE = "01/15/2026"
SEASON = "2025-26"


class FakeSettlementProvider:
    def __init__(
        self,
        *,
        scoreboard=None,
        gamelogs=None,
        raise_on_scoreboard=False,
        raise_on_gamelogs=False,
    ):
        self._scoreboard = scoreboard if scoreboard is not None else []
        self._gamelogs = gamelogs if gamelogs is not None else []
        self._raise_on_scoreboard = raise_on_scoreboard
        self._raise_on_gamelogs = raise_on_gamelogs

    def get_todays_scoreboard(self, game_date=None):
        if self._raise_on_scoreboard:
            raise ProviderUnavailableError("scoreboard down")
        return list(self._scoreboard)

    def get_player_game_logs(self, player_id, season, *, player_name=None):
        if self._raise_on_gamelogs:
            raise ProviderUnavailableError("gamelog down")
        return list(self._gamelogs)


def _game_log(points, game_id=GAME_ID, player_id=PLAYER_ID):
    return PlayerGameLog(
        season=SEASON,
        season_id="22025",
        player_id=player_id,
        player_name="Player A",
        game_id=game_id,
        game_date=pd.Timestamp("2026-01-15"),
        matchup="LAL vs DEN",
        team_abbreviation="LAL",
        opponent_abbreviation="DEN",
        is_home=1,
        wl="W",
        minutes=30,
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
        points=points,
        plus_minus=5,
        video_available=1,
    )


def _final_game(game_id=GAME_ID, status="Final"):
    return ScheduledGame(
        game_id=game_id,
        game_date=GAME_DATE,
        home_team_id=1,
        away_team_id=2,
        game_status_text=status,
    )


def _persist_and_get_snapshot(db_conn, **overrides):
    defaults = {
        "player_name": "Player A",
        "status": PredictionStatus.OK,
        "player_id": PLAYER_ID,
        "model_projection": 22.5,
        "sportsbook_line": 20.0,
        "edge": 2.5,
        "direction": PredictionDirection.OVER,
        "bookmaker": "draftkings",
        "model_version": "v1",
        "generated_at_utc": "2026-01-15T12:00:00+00:00",
        "game_id": GAME_ID,
        "game_date": GAME_DATE,
    }
    defaults.update(overrides)
    result = PredictionResult(**defaults)
    board = PredictionBoard(
        generated_at_utc=defaults["generated_at_utc"],
        bookmaker="draftkings",
        model_version="v1",
        predictions=(result,),
        props_discovered=1,
        players_matched=1,
        predictions_generated=1,
        unmatched_count=0,
        unavailable_count=0,
    )
    run = persist_prediction_run(board, db_conn)

    from src.services.prediction_repository import get_snapshots_for_run

    snapshots = get_snapshots_for_run(db_conn, run["run_id"])
    return snapshots[0]


# ---- grading math (7-12) ---------------------------------------------------


def test_over_win():
    assert grade_points_prediction("OVER", 25.0, 20.0) == "WIN"


def test_over_loss():
    assert grade_points_prediction("OVER", 15.0, 20.0) == "LOSS"


def test_under_win():
    assert grade_points_prediction("UNDER", 15.0, 20.0) == "WIN"


def test_under_loss():
    assert grade_points_prediction("UNDER", 25.0, 20.0) == "LOSS"


def test_push():
    assert grade_points_prediction("OVER", 20.0, 20.0) == "PUSH"
    assert grade_points_prediction("UNDER", 20.0, 20.0) == "PUSH"


def test_neutral_is_no_action():
    assert grade_points_prediction("NEUTRAL", 20.0, 20.0) == "NO_ACTION"


# ---- settlement service behavior (13-17) -----------------------------------


def test_unfinished_game_remains_pending(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn)
    provider = FakeSettlementProvider(
        scoreboard=[
            ScheduledGame(
                game_id=GAME_ID,
                game_date=GAME_DATE,
                home_team_id=1,
                away_team_id=2,
                game_status_text="7:30 pm ET",
            )
        ],
        gamelogs=[],
    )
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome is None
    assert get_outcome_for_snapshot(db_conn, snapshot["id"]) is None


def test_dnp_handled_explicitly_as_no_action(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn)
    provider = FakeSettlementProvider(
        scoreboard=[_final_game()], gamelogs=[]
    )  # final, but no box score row
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome["result_status"] == "NO_ACTION"


def test_postponed_game_handled_as_no_action(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn)
    provider = FakeSettlementProvider(
        scoreboard=[_final_game(status="Postponed")], gamelogs=[]
    )
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome["result_status"] == "NO_ACTION"


def test_missing_game_identity_not_guessed(db_conn):
    """No game_id captured at prediction time -- must report UNAVAILABLE,
    never silently settle via name/date matching."""
    snapshot = _persist_and_get_snapshot(db_conn, game_id=None, game_date=None)
    provider = FakeSettlementProvider()
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome["result_status"] == "UNAVAILABLE"


def test_missing_player_id_not_guessed(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn, player_id=None)
    provider = FakeSettlementProvider()
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome["result_status"] == "UNAVAILABLE"


def test_successful_settlement_win(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn)  # OVER, line=20.0
    provider = FakeSettlementProvider(
        scoreboard=[_final_game()], gamelogs=[_game_log(25.0)]
    )
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome["result_status"] == "WIN"
    assert outcome["actual_points"] == 25.0


def test_repeated_settlement_is_idempotent(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn)
    provider = FakeSettlementProvider(
        scoreboard=[_final_game()], gamelogs=[_game_log(25.0)]
    )
    first = settle_prediction(db_conn, snapshot, provider=provider)

    # Second attempt, even with DIFFERENT (wrong) data, must not change anything.
    provider2 = FakeSettlementProvider(
        scoreboard=[_final_game()], gamelogs=[_game_log(1.0)]
    )
    second = settle_prediction(db_conn, snapshot, provider=provider2)

    assert first["id"] == second["id"]
    assert second["actual_points"] == 25.0  # unchanged, not overwritten with 1.0

    cur = db_conn.execute(
        "SELECT COUNT(*) FROM prediction_outcomes WHERE prediction_snapshot_id = ?",
        (snapshot["id"],),
    )
    assert cur.fetchone()[0] == 1


def test_one_settlement_failure_does_not_corrupt_others(db_conn):
    good_snapshot = _persist_and_get_snapshot(
        db_conn, player_id=1, player_name="Good Player"
    )

    board2 = PredictionBoard(
        generated_at_utc="2026-01-15T12:05:00+00:00",
        bookmaker="draftkings",
        model_version="v1",
        predictions=(
            PredictionResult(
                player_name="Bad Player",
                status=PredictionStatus.OK,
                player_id=2,
                model_projection=10.0,
                sportsbook_line=8.0,
                edge=2.0,
                direction=PredictionDirection.OVER,
                bookmaker="draftkings",
                model_version="v1",
                generated_at_utc="2026-01-15T12:05:00+00:00",
                game_id="G2",
                game_date=GAME_DATE,
            ),
        ),
        props_discovered=1,
        players_matched=1,
        predictions_generated=1,
        unmatched_count=0,
        unavailable_count=0,
    )
    persist_prediction_run(board2, db_conn)

    class _PartlyBrokenProvider(FakeSettlementProvider):
        def get_player_game_logs(self, player_id, season, *, player_name=None):
            if player_id == 2:
                raise RuntimeError("boom -- unexpected failure for this one player")
            return super().get_player_game_logs(
                player_id, season, player_name=player_name
            )

    provider = _PartlyBrokenProvider(
        scoreboard=[_final_game(), _final_game(game_id="G2")],
        gamelogs=[_game_log(25.0, game_id=GAME_ID, player_id=1)],
    )

    result = settle_pending_predictions(db_conn, provider=provider)
    assert result["errors"] == 1
    assert result["settled"] == 1  # the good one still got settled

    good_outcome = get_outcome_for_snapshot(db_conn, good_snapshot["id"])
    assert good_outcome is not None
    assert good_outcome["result_status"] == "WIN"


def test_settle_pending_predictions_reports_counts(db_conn):
    _persist_and_get_snapshot(db_conn)
    provider = FakeSettlementProvider(
        scoreboard=[_final_game()], gamelogs=[_game_log(25.0)]
    )
    result = settle_pending_predictions(db_conn, provider=provider)
    assert result == {
        "pending_checked": 1,
        "settled": 1,
        "still_pending": 0,
        "no_action": 0,
        "unavailable": 0,
        "errors": 0,
    }


def test_provider_unavailable_during_scoreboard_check_leaves_pending(db_conn):
    snapshot = _persist_and_get_snapshot(db_conn)
    provider = FakeSettlementProvider(raise_on_scoreboard=True)
    outcome = settle_prediction(db_conn, snapshot, provider=provider)
    assert outcome is None
    assert get_outcome_for_snapshot(db_conn, snapshot["id"]) is None
