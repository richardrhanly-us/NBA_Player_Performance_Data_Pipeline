"""
Prediction persistence: a DESIGN for the eventual production shape,
plus a minimal, local, non-production implementation used only to prove
the Step 8 board end-to-end (smoke test / manual verification) -- NOT a
production database migration.

-------------------------------------------------------------------------
RECOMMENDED PRODUCTION SHAPE (design only -- not implemented as a DB
migration in Step 8)
-------------------------------------------------------------------------
Two tables/collections, not one, because a "run" (one board-build) and
a "snapshot" (one player's prediction within that run) have different
lifecycles and query needs:

    prediction_runs
        run_id            (PK)
        generated_at_utc
        bookmaker
        model_version
        props_discovered
        players_matched
        predictions_generated
        unmatched_count
        unavailable_count

    prediction_snapshots
        snapshot_id        (PK)
        run_id             (FK -> prediction_runs.run_id)
        player_id
        player_name
        team_abbreviation
        matchup
        game_id
        game_status
        model_projection
        sportsbook_line
        edge
        direction
        bookmaker
        model_version
        generated_at_utc
        latest_game_date
        status
        reason

Why two tables: `prediction_runs` answers "when did we build a board,
against what model/bookmaker, how healthy was it" without scanning
every player row; `prediction_snapshots` is the actual historical
record queried per-player later (e.g. "what did we predict for this
player on this date").

IMMUTABILITY REQUIREMENT: a prediction_snapshots row must never be
updated in place once written. If the sportsbook line moves, the model
changes, or features change, that produces a NEW run_id and NEW
snapshot rows -- never an edit to an existing one. This is why
PredictionResult itself carries its own generated_at_utc/model_version/
sportsbook_line at the moment of prediction: a snapshot row is a direct,
one-to-one serialization of one PredictionResult, so "what did the user
see" is reconstructible verbatim from storage, forever, regardless of
what the model or the market does afterward. Actual game outcomes
(needed for historical accuracy reporting) belong in a SEPARATE,
append-only outcomes table joined by (player_id, game_id) -- never
written into an existing snapshot row.

Choosing an actual production database (Postgres/Neon, given this
project's existing Sheets/Neon usage elsewhere, is a reasonable
candidate) and running a real migration is future work, deliberately
deferred -- see the Step 8 report's persistence-recommendation section.

-------------------------------------------------------------------------
STEP 8 MINIMAL IMPLEMENTATION (this module)
-------------------------------------------------------------------------
save_board_snapshot() below writes a PredictionBoard to a local JSON
file -- one file per run, named by run timestamp, under a gitignored
directory (data/prediction_snapshots/ by default). This exists ONLY to
let Step 8's real-data smoke test and manual verification produce and
inspect a durable artifact; it is not wired into any scheduled job, not
read by any app code, and is not a substitute for the schema above.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

DEFAULT_SNAPSHOT_DIR = Path("data") / "prediction_snapshots"


def _serialize_prediction(result) -> dict:
    payload = asdict(result)
    if payload.get("direction") is not None:
        payload["direction"] = result.direction.value
    payload["status"] = result.status.value
    return payload


def save_board_snapshot(board, output_dir: Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    """
    Writes one PredictionBoard as a single local JSON file (run metadata
    + every PredictionResult serialized verbatim). Returns the path
    written. Local/manual use only -- see this module's docstring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_ts = board.generated_at_utc.replace(":", "").replace("+00:00", "Z")
    path = output_dir / f"board_{safe_ts}.json"

    payload = {
        "generated_at_utc": board.generated_at_utc,
        "bookmaker": board.bookmaker,
        "model_version": board.model_version,
        "props_discovered": board.props_discovered,
        "players_matched": board.players_matched,
        "predictions_generated": board.predictions_generated,
        "unmatched_count": board.unmatched_count,
        "unavailable_count": board.unavailable_count,
        "predictions": [_serialize_prediction(r) for r in board.predictions],
    }

    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp_path.replace(path)
    return path
