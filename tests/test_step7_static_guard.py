"""
Step 7 static architecture guard: migrated runtime modules
(src/shared_app.py, apps/publicapp.py, apps/adminapp.py) must not import
nba_api directly -- NBA-specific access must stay behind
src/data/basketball/providers/nba_api_provider.py. This guards the
provider boundary; it does NOT ban the nba_api dependency entirely (the
provider implementation, training/data/nba_client.py, and the legacy
notebook are all expected/allowed to import it).

Also confirms the provider refactor never touches the model registry or
the production model artifact.
"""

import ast
from pathlib import Path

from training import config, model_registry

MIGRATED_RUNTIME_MODULES = (
    "src/shared_app.py",
    "apps/publicapp.py",
    "apps/adminapp.py",
    # Step 8: the new prediction service layer must hold to the same
    # provider-boundary guard as the app modules it serves.
    "src/services/prediction_service.py",
    "src/services/prediction_result.py",
    "src/services/persistence.py",
)

ALLOWED_NBA_API_LOCATIONS = (
    "src/data/basketball/providers/nba_api_provider.py",
    "training/data/nba_client.py",
)


def _imported_top_level_modules(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_migrated_runtime_modules_do_not_import_nba_api_directly():
    for rel_path in MIGRATED_RUNTIME_MODULES:
        path = Path(rel_path)
        assert path.exists(), f"expected {rel_path} to exist"
        modules = _imported_top_level_modules(path)
        assert "nba_api" not in modules, (
            f"{rel_path} imports nba_api directly -- NBA-specific access "
            "must stay behind src/data/basketball/providers/nba_api_provider.py"
        )


def test_allowed_nba_api_locations_still_import_it():
    """Sanity check the guard isn't accidentally too broad -- these
    modules ARE expected to import nba_api."""
    for rel_path in ALLOWED_NBA_API_LOCATIONS:
        path = Path(rel_path)
        assert path.exists(), f"expected {rel_path} to exist"
        modules = _imported_top_level_modules(path)
        assert "nba_api" in modules, (
            f"{rel_path} was expected to import nba_api (it's an allowed, "
            "documented location) but does not -- update ALLOWED_NBA_API_LOCATIONS "
            "if this file's role changed"
        )


def test_no_new_nba_api_import_locations_beyond_the_documented_set():
    """Repo-wide guard: every .py file that imports nba_api must be one
    of the documented, allowed locations, or the legacy notebook/tests."""
    repo_root = Path(".")
    allowed = set(ALLOWED_NBA_API_LOCATIONS)
    unexpected = []

    for path in repo_root.rglob("*.py"):
        rel = str(path.as_posix())
        if rel.startswith((".venv/", "tests/")) or "__pycache__" in rel:
            continue
        if rel in allowed:
            continue
        try:
            modules = _imported_top_level_modules(path)
        except SyntaxError:
            continue
        if "nba_api" in modules:
            unexpected.append(rel)

    assert unexpected == [], f"Unexpected nba_api import(s) found in: {unexpected}"


def test_current_v1_pointer_unchanged_after_importing_shared_app():
    before = (
        model_registry.get_current_version()
        if model_registry.current_pointer_path().exists()
        else None
    )

    import src.shared_app  # noqa: F401

    after = (
        model_registry.get_current_version()
        if model_registry.current_pointer_path().exists()
        else None
    )
    assert before == after


def test_model_registry_dir_unchanged():
    assert config.MODEL_REGISTRY_DIR == config.REPO_ROOT / "models" / "registry"
