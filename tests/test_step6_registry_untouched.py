"""
Step 6 regression guards: the provider refactor (src/data/basketball/,
the collect_gamelogs.py migration) must never touch the model registry,
the CURRENT_V1 pointer, or the deployed legacy model artifact -- this
step is architecture-only, not a retrain/redeploy.
"""

from training import config, model_registry


def test_current_v1_pointer_is_unchanged_by_importing_the_new_package():
    """Merely importing everything Step 6 added must not write anything
    to the model registry -- there is no import-time side effect that
    touches it."""
    before = (
        model_registry.get_current_version()
        if model_registry.current_pointer_path().exists()
        else None
    )

    import src.data.basketball
    import src.data.basketball.models
    import src.data.basketball.normalization
    import src.data.basketball.provider
    import src.data.basketball.providers.nba_api_provider  # noqa: F401

    after = (
        model_registry.get_current_version()
        if model_registry.current_pointer_path().exists()
        else None
    )
    assert before == after


def test_legacy_model_path_is_not_referenced_by_the_new_provider_package():
    """Static guard: nothing under src/data/basketball/ should mention
    the deployed legacy model path or the training model registry --
    those are separate, untouched concerns."""
    import pathlib

    package_dir = pathlib.Path("src/data/basketball")
    forbidden = ("points_regression.pkl", "CURRENT_V1", "MODEL_REGISTRY_DIR")
    for path in package_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} unexpectedly references {token!r}"


def test_model_registry_dir_config_value_is_unchanged():
    assert config.MODEL_REGISTRY_DIR == config.REPO_ROOT / "models" / "registry"
