"""
Contract test: the explicit LEGACY_FEATURE_NAMES list in
src/features/feature_schema.py must match the feature list the deployed
model at models/points_regression.pkl was actually trained on -- in name
and in order.

This is intentionally NOT a test that derives the expected list from the
model itself; the schema module is a hand-maintained contract, and this
test's job is to catch drift between that contract and the real artifact.
"""

import os

import joblib
import pytest

from src.features.feature_schema import LEGACY_FEATURE_NAMES

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "points_regression.pkl",
)


@pytest.fixture(scope="module")
def legacy_model():
    return joblib.load(MODEL_PATH)


def test_legacy_feature_names_match_deployed_model_exactly(legacy_model):
    deployed_features = list(legacy_model.feature_names_in_)

    assert deployed_features == list(LEGACY_FEATURE_NAMES), (
        "LEGACY_FEATURE_NAMES in src/features/feature_schema.py no longer "
        "matches models/points_regression.pkl's feature_names_in_. This "
        "means either the schema file drifted or a different model was "
        "deployed without updating the schema contract.\n"
        f"deployed model features : {deployed_features}\n"
        f"LEGACY_FEATURE_NAMES    : {list(LEGACY_FEATURE_NAMES)}"
    )


def test_legacy_feature_names_count_is_24(legacy_model):
    # Pinned to the specific, known shape of the current deployed artifact.
    assert legacy_model.n_features_in_ == 24
    assert len(LEGACY_FEATURE_NAMES) == 24


def test_schema_contract_actually_detects_a_mismatch(legacy_model):
    """
    Proves the contract check in test 1 is not a tautology: a deliberately
    wrong feature list (bad name, and separately, wrong order) must fail
    the same comparison this module relies on.
    """
    deployed_features = list(legacy_model.feature_names_in_)

    wrong_name = list(LEGACY_FEATURE_NAMES)
    wrong_name[0] = "totally_not_a_real_feature"
    assert deployed_features != wrong_name

    wrong_order = list(LEGACY_FEATURE_NAMES)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    assert deployed_features != wrong_order
