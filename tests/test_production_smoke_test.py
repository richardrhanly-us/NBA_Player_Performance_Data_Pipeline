"""
Step 14 Phase 17/19: the production smoke test script is read-only and
returns the right exit code for both the "nothing configured" and
"everything configured" cases. Imported as a module (not subprocessed)
so these run fast and can monkeypatch environment variables directly.
"""

import ast
from pathlib import Path

import scripts.production_smoke_test as smoke_test


def _clear_all(monkeypatch):
    for name in (
        "DATABASE_URL",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "DEV_AUTH_ENABLED",
        "STRIPE_SECRET_KEY",
        "STRIPE_PUBLISHABLE_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRO_PRICE_ID",
        "STRIPE_SUCCESS_URL",
        "STRIPE_CANCEL_URL",
        "STRIPE_PORTAL_RETURN_URL",
        "ODDS_API_KEY",
        "PREDICTION_AUTOMATION_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def test_smoke_test_exits_nonzero_with_nothing_configured(monkeypatch, capsys):
    _clear_all(monkeypatch)
    exit_code = smoke_test.main(argv=[])
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "RESULT: NOT READY" in output


def test_smoke_test_exits_zero_when_fully_configured(monkeypatch, capsys):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("STRIPE_SUCCESS_URL", "https://app.example.com/?checkout=success")
    monkeypatch.setenv("STRIPE_CANCEL_URL", "https://app.example.com/?checkout=cancel")
    monkeypatch.setenv("STRIPE_PORTAL_RETURN_URL", "https://app.example.com/")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds-key")

    exit_code = smoke_test.main(argv=[])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "RESULT: READY" in output


def test_smoke_test_output_never_leaks_configured_secrets(monkeypatch, capsys):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:supersecretpassword@host/db")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_reallysecretvalue")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_reallysecretvalue")

    smoke_test.main(argv=[])

    output = capsys.readouterr().out
    assert "supersecretpassword" not in output
    assert "sk_test_reallysecretvalue" not in output
    assert "whsec_reallysecretvalue" not in output


def test_smoke_test_never_imports_stripe_or_writes_to_repositories():
    """Guards the 'never create charges/customers/mutate subscriptions'
    contract at the import level -- this script should only ever import
    src.services.readiness, never stripe or accounts_repository."""
    tree = ast.parse(Path("scripts/production_smoke_test.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert "stripe" not in modules
    source = Path("scripts/production_smoke_test.py").read_text(encoding="utf-8")
    assert "accounts_repository" not in source
