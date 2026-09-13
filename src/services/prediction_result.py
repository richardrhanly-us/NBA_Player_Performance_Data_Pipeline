"""
The canonical, product-level result of one points-prop prediction.

This is OUR contract -- named and shaped for what the product actually
displays/needs, not a mirror of the sportsbook payload or the raw model
output. Every field here is something the current pipeline can populate
reliably today; nothing is included merely because it sounds useful (no
win-probability, no expected-value, no injury/lineup context -- none of
that exists in this pipeline yet).

Edge semantics (see PredictionDirection below) are fixed and simple:

    edge = model_projection - sportsbook_line
    edge > 0  -> OVER
    edge < 0  -> UNDER
    edge == 0 -> NEUTRAL

This is a raw regression distance, not a probability or a betting
recommendation -- see build_prediction_result()'s docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PredictionStatus(str, Enum):
    """
    Why a PredictionResult does or does not carry a usable projection.
    OK is the only status where model_projection/edge/direction are
    expected to be populated -- every other status means the pipeline
    could not confidently produce (or pair) a projection for this
    player, and reason explains why. Never silently guessed.
    """

    OK = "ok"
    UNMATCHED = (
        "unmatched"  # sportsbook player name could not be resolved to a player_id
    )
    MISSING_HISTORY = "missing_history"  # no/insufficient gamelog to build features
    MISSING_LINE = "missing_line"  # no sportsbook line available for this player
    PROVIDER_UNAVAILABLE = "provider_unavailable"  # BasketballDataProvider raised
    MODEL_UNAVAILABLE = "model_unavailable"  # the model artifact could not be loaded
    ERROR = "error"  # anything else unexpected -- see `reason`


class PredictionDirection(str, Enum):
    OVER = "OVER"
    UNDER = "UNDER"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class PredictionResult:
    """One player's points-prop prediction, paired with (at most) one
    sportsbook line, at one point in time."""

    player_name: str
    status: PredictionStatus

    player_id: int | None = None
    team_abbreviation: str | None = None
    # Best-effort matchup context from the sportsbook event itself
    # (e.g. "LAL @ DEN") -- the same home_team/away_team data
    # fetch_all_today_player_props already returns. Not derived from a
    # play-by-play/roster source.
    matchup: str | None = None
    # From BasketballDataProvider.get_todays_scoreboard(), matched by the
    # player's team_id when that lookup succeeds -- best-effort, always
    # allowed to be None.
    game_id: str | None = None
    game_status: str | None = None

    model_projection: float | None = None
    sportsbook_line: float | None = None
    edge: float | None = None
    direction: PredictionDirection | None = None
    bookmaker: str | None = None

    model_version: str | None = None
    generated_at_utc: str | None = None
    # The most recent GAME_DATE in the gamelog data the projection was
    # built from -- i.e. how fresh the underlying history is, not when
    # the prediction was computed (see generated_at_utc for that).
    latest_game_date: str | None = None

    reason: str | None = None


def compute_edge_and_direction(
    model_projection: float | None, sportsbook_line: float | None
) -> tuple:
    """
    edge = model_projection - sportsbook_line. Returns (edge, direction)
    -- (None, None) if either input is missing (no line to compare
    against is MISSING_LINE, not a zero edge).
    """
    if model_projection is None or sportsbook_line is None:
        return None, None
    edge = float(model_projection) - float(sportsbook_line)
    if edge > 0:
        direction = PredictionDirection.OVER
    elif edge < 0:
        direction = PredictionDirection.UNDER
    else:
        direction = PredictionDirection.NEUTRAL
    return edge, direction
