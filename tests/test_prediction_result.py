"""
Tests for src/services/prediction_result.py: edge/direction semantics
(#2/#3/#4 in the Step 8 test list) and basic PredictionResult behavior.
No network access.
"""

import pytest

from src.services.prediction_result import (
    PredictionDirection,
    PredictionResult,
    PredictionStatus,
    compute_edge_and_direction,
)


def test_positive_edge_is_over():
    edge, direction = compute_edge_and_direction(23.8, 21.5)
    assert edge == pytest.approx(2.3)
    assert direction == PredictionDirection.OVER


def test_negative_edge_is_under():
    edge, direction = compute_edge_and_direction(20.0, 24.0)
    assert edge == -4.0
    assert direction == PredictionDirection.UNDER


def test_zero_edge_is_neutral():
    edge, direction = compute_edge_and_direction(20.0, 20.0)
    assert edge == 0.0
    assert direction == PredictionDirection.NEUTRAL


def test_missing_projection_or_line_gives_no_edge():
    assert compute_edge_and_direction(None, 20.0) == (None, None)
    assert compute_edge_and_direction(20.0, None) == (None, None)
    assert compute_edge_and_direction(None, None) == (None, None)


def test_edge_formula_is_projection_minus_line_not_reversed():
    edge, direction = compute_edge_and_direction(30.0, 10.0)
    assert edge == 20.0  # not -20.0
    assert direction == PredictionDirection.OVER


def test_prediction_result_defaults_are_all_none_except_name_and_status():
    result = PredictionResult(
        player_name="Jalen Williams", status=PredictionStatus.UNMATCHED
    )
    assert result.player_name == "Jalen Williams"
    assert result.status == PredictionStatus.UNMATCHED
    assert result.player_id is None
    assert result.model_projection is None
    assert result.edge is None
    assert result.direction is None


def test_prediction_result_is_immutable():
    from dataclasses import FrozenInstanceError

    result = PredictionResult(player_name="X", status=PredictionStatus.OK)
    with pytest.raises(FrozenInstanceError):
        result.model_projection = 99.0
