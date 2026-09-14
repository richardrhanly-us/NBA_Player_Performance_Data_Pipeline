"""
Step 14 static architecture/security guards -- paid-beta readiness,
observability, and deployment safety. Mirrors the style of
tests/test_step11_static_guards.py through test_step13_static_guards.py.
"""

import ast
from pathlib import Path


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


def test_readiness_module_never_imports_stripe_sdk_directly():
    """Phase 2/18: readiness checks read billing_config (presence-only),
    never the stripe SDK itself -- a readiness probe must never make a
    live Stripe API call."""
    modules = _imported_top_level_modules(Path("src/services/readiness.py"))
    assert "stripe" not in modules


def test_readiness_module_never_imports_requests():
    """No network calls of its own -- the one expensive check it has
    (database connectivity) goes through db_connection.py, and
    everything else is a config presence check."""
    modules = _imported_top_level_modules(Path("src/services/readiness.py"))
    assert "requests" not in modules


def test_webhook_service_only_exposes_the_documented_routes():
    """Phase 3/18: no debug endpoints. The service must expose exactly
    /health, /ready, and /stripe/webhook -- nothing else."""
    source = Path("webhook_service/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in ("get", "post", "put", "delete", "patch")
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    routes.append(decorator.args[0].value)
    assert sorted(routes) == sorted(["/health", "/ready", "/stripe/webhook"])


def test_get_prediction_db_connection_never_lets_a_raw_connection_error_propagate():
    """Phase 8/18: a raw psycopg connection failure can embed the DSN
    (including the password) -- get_prediction_db_connection() must
    catch and re-raise as a generic RuntimeError so every caller across
    the codebase automatically gets a safe message."""
    source = Path("src/services/db_connection.py").read_text(encoding="utf-8")
    func_start = source.index("def get_prediction_db_connection()")
    # Skip past the function's own docstring (which, documenting this
    # very guarantee, itself mentions "psycopg.connect(" in prose) --
    # only the real code below it matters for ordering.
    func_body = source[func_start:]
    code_start = func_body.index('"""', func_body.index('"""') + 3) + 3
    func_code = func_body[code_start:]

    try_index = func_code.index("try:")
    connect_index = func_code.index("psycopg.connect(", try_index)
    except_index = func_code.index("except Exception as e:", connect_index)
    raise_index = func_code.index("raise RuntimeError", except_index)

    assert try_index < connect_index < except_index < raise_index, (
        "expected psycopg.connect() to be wrapped in a try/except that re-raises "
        "a generic RuntimeError, so a raw connection error (which can embed the "
        "DSN/password) never propagates to a caller that displays str(exception)"
    )


def test_production_smoke_test_is_read_only():
    """Phase 17/18: the smoke test must never import stripe or the
    accounts repository (which is where every subscription/override
    mutation lives) -- it can only ever read via src.services.readiness."""
    modules = _imported_top_level_modules(Path("scripts/production_smoke_test.py"))
    assert "stripe" not in modules
    source = Path("scripts/production_smoke_test.py").read_text(encoding="utf-8")
    assert "accounts_repository" not in source
    assert "apply_migrations" not in source


def test_observability_module_has_no_third_party_dependency():
    """Structured logging uses only the Python standard library -- no
    new dependency required for this."""
    modules = _imported_top_level_modules(Path("src/services/observability.py"))
    stdlib_only = {"json", "logging", "sys", "datetime", "typing", "__future__"}
    assert modules.issubset(stdlib_only)


def test_docs_directory_has_the_required_operational_documents():
    """Phase 14/15/16: the three required operational documents exist."""
    for rel_path in (
        "docs/PRODUCTION_DEPLOYMENT.md",
        "docs/OPERATIONS_RUNBOOK.md",
        "docs/PAID_BETA_CHECKLIST.md",
    ):
        assert Path(rel_path).exists(), f"expected {rel_path} to exist"
