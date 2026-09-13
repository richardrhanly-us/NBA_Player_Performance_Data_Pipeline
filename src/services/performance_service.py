"""
Historical performance summaries computed from settled predictions
(prediction_snapshots joined with prediction_outcomes -- see
prediction_repository.get_settled_predictions).

Pushes are excluded from every win-rate denominator (a push is neither a
win nor a loss). Pending predictions never appear here at all --
get_settled_predictions only returns rows with a non-PENDING outcome.
NO_ACTION/UNAVAILABLE outcomes are excluded from win/loss/push counting
entirely (they were never gradable), but are visible for transparency
via PerformanceSummary.no_action_count/unavailable_count.

Edge buckets (0-1, 1-2, 2-3, 3-4, 4+) and the qualified/unqualified
(</>= EDGE_THRESHOLD) comparison are reporting only -- nothing here
selects, tunes, or changes EDGE_THRESHOLD itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services.prediction_repository import (
    QUALIFIED_EDGE_THRESHOLD_DEFAULT,
    get_settled_predictions,
)

GRADED_STATUSES = ("WIN", "LOSS", "PUSH")

EDGE_BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, float("inf")))


@dataclass(frozen=True)
class PerformanceSummary:
    graded: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None  # excludes pushes from the denominator
    qualified_graded: int
    qualified_wins: int
    qualified_losses: int
    qualified_pushes: int
    qualified_win_rate: float | None
    avg_model_projection: float | None
    avg_sportsbook_line: float | None
    avg_abs_edge: float | None
    no_action_count: int
    unavailable_count: int


def _graded_rows(rows):
    return [r for r in rows if r.get("outcome_result_status") in GRADED_STATUSES]


def _win_rate(rows):
    wins = sum(1 for r in rows if r["outcome_result_status"] == "WIN")
    losses = sum(1 for r in rows if r["outcome_result_status"] == "LOSS")
    decided = wins + losses
    return (wins / decided) if decided > 0 else None


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def compute_performance_summary(
    conn,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    direction: str | None = None,
    qualified_only: bool = False,
    edge_threshold: float = QUALIFIED_EDGE_THRESHOLD_DEFAULT,
) -> PerformanceSummary:
    rows = get_settled_predictions(
        conn,
        start_date=start_date,
        end_date=end_date,
        direction=direction,
        qualified_only=qualified_only,
    )

    graded = _graded_rows(rows)
    wins = sum(1 for r in graded if r["outcome_result_status"] == "WIN")
    losses = sum(1 for r in graded if r["outcome_result_status"] == "LOSS")
    pushes = sum(1 for r in graded if r["outcome_result_status"] == "PUSH")

    qualified_rows = [r for r in graded if bool(r.get("qualified"))]
    q_wins = sum(1 for r in qualified_rows if r["outcome_result_status"] == "WIN")
    q_losses = sum(1 for r in qualified_rows if r["outcome_result_status"] == "LOSS")
    q_pushes = sum(1 for r in qualified_rows if r["outcome_result_status"] == "PUSH")

    no_action_count = sum(
        1 for r in rows if r.get("outcome_result_status") == "NO_ACTION"
    )
    unavailable_count = sum(
        1 for r in rows if r.get("outcome_result_status") == "UNAVAILABLE"
    )

    return PerformanceSummary(
        graded=len(graded),
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=_win_rate(graded),
        qualified_graded=len(qualified_rows),
        qualified_wins=q_wins,
        qualified_losses=q_losses,
        qualified_pushes=q_pushes,
        qualified_win_rate=_win_rate(qualified_rows),
        avg_model_projection=_mean([r.get("model_projection") for r in graded]),
        avg_sportsbook_line=_mean([r.get("sportsbook_line") for r in graded]),
        avg_abs_edge=_mean(
            [abs(r["edge"]) for r in graded if r.get("edge") is not None]
        ),
        no_action_count=no_action_count,
        unavailable_count=unavailable_count,
    )


def compute_edge_bucket_breakdown(conn, **filters) -> list:
    """Win/loss/push/win-rate for each of the predeclared edge buckets
    (0-1, 1-2, 2-3, 3-4, 4+), reporting only -- see module docstring."""
    rows = get_settled_predictions(conn, **filters)
    graded = [r for r in _graded_rows(rows) if r.get("edge") is not None]

    breakdown = []
    for lo, hi in EDGE_BUCKETS:
        bucket_rows = [r for r in graded if lo <= abs(r["edge"]) < hi]
        wins = sum(1 for r in bucket_rows if r["outcome_result_status"] == "WIN")
        losses = sum(1 for r in bucket_rows if r["outcome_result_status"] == "LOSS")
        pushes = sum(1 for r in bucket_rows if r["outcome_result_status"] == "PUSH")
        label = f"{lo:g}-{hi:g}" if hi != float("inf") else f"{lo:g}+"
        breakdown.append(
            {
                "bucket": label,
                "n": len(bucket_rows),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": _win_rate(bucket_rows),
            }
        )
    return breakdown


def compute_qualified_vs_unqualified_breakdown(
    conn, *, edge_threshold: float = QUALIFIED_EDGE_THRESHOLD_DEFAULT, **filters
) -> dict:
    """
    A clean `< edge_threshold` vs `>= edge_threshold` comparison,
    reporting only. This does NOT tune/select edge_threshold based on
    the results -- it always uses the existing, externally-fixed policy
    value (EDGE_THRESHOLD = 3.0 by default), passed in explicitly.
    """
    rows = get_settled_predictions(conn, **filters)
    graded = [r for r in _graded_rows(rows) if r.get("edge") is not None]

    below = [r for r in graded if abs(r["edge"]) < edge_threshold]
    at_or_above = [r for r in graded if abs(r["edge"]) >= edge_threshold]

    def _summarize(subset):
        wins = sum(1 for r in subset if r["outcome_result_status"] == "WIN")
        losses = sum(1 for r in subset if r["outcome_result_status"] == "LOSS")
        pushes = sum(1 for r in subset if r["outcome_result_status"] == "PUSH")
        return {
            "n": len(subset),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": _win_rate(subset),
        }

    return {
        "edge_threshold": edge_threshold,
        "below_threshold": _summarize(below),
        "at_or_above_threshold": _summarize(at_or_above),
    }
