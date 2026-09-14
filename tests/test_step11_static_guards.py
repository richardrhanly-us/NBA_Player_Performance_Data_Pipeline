"""
Step 11 static architecture guards -- accounts/entitlements/billing.

Mirrors the style of tests/test_step7_static_guard.py (AST-based import
scanning) plus a few plain source-text checks for patterns that are hard
to characterize as a single import (raw SQL in page code, session_state
keys leaking outside the one module allowed to touch them, scattered
tier-string comparisons instead of centralized Entitlements booleans).

These are architecture guards, not full static analysis -- they exist to
catch the *shape* of a regression (a new file quietly importing stripe,
a new admin button executing SQL directly) without needing to re-derive
this reasoning by hand every time Step 12+ touches these files.
"""

import ast
from pathlib import Path

APP_FILES = ("apps/publicapp.py", "apps/adminapp.py")

# Every file allowed to know about tier/entitlement internals directly.
# Everything else must go through Entitlements booleans / CurrentUser.
ENTITLEMENT_INTERNALS = (
    "src/domain/accounts.py",
    "src/services/entitlement_service.py",
    "src/services/access_presentation.py",
    "src/services/accounts_repository.py",
    "src/services/auth_session.py",
    "scripts/create_admin_user.py",
)

# The only module allowed to read/write auth-related st.session_state
# keys, per src/services/auth_session.py's module docstring.
AUTH_SESSION_OWNER = "src/services/auth_session.py"
AUTH_SESSION_KEYS = (
    "_auth_session",
    "_legacy_admin_ok",
    # Step 12 additions -- session-expiry/signup-notice one-shot flags.
    "_auth_session_expired",
    "_auth_signup_notice",
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


def _all_repo_py_files():
    repo_root = Path(".")
    for path in repo_root.rglob("*.py"):
        rel = str(path.as_posix())
        if rel.startswith((".venv/", "__pycache__/")) or "__pycache__" in rel:
            continue
        yield path


STRIPE_IMPORT_ALLOWED_LOCATIONS = ("src/services/billing_provider.py",)


def test_no_stripe_import_outside_the_billing_provider_module():
    """Phase 8 (Step 11) / Step 13: entitlement logic must never depend
    on Stripe-specific objects. Step 13 added a real StripeBillingProvider
    -- src/services/billing_provider.py is now the ONE sanctioned
    location for `import stripe` (see tests/test_step13_static_guards.py
    for the apps/entitlement_service-specific half of this guard);
    everywhere else, a `stripe` import is still an error."""
    offenders = []
    for path in _all_repo_py_files():
        rel = str(path.as_posix())
        if rel in STRIPE_IMPORT_ALLOWED_LOCATIONS or rel.startswith("tests/"):
            continue
        try:
            modules = _imported_top_level_modules(path)
        except SyntaxError:
            continue
        if "stripe" in modules:
            offenders.append(rel)
    assert offenders == [], f"Unexpected `stripe` import(s) found in: {offenders}"


def test_billing_provider_module_imports_stripe_and_nothing_else_unexpected():
    """Step 13: billing_provider.py is now the sanctioned Stripe
    integration layer -- it DOES import stripe (a change from Step 11,
    when billing was entirely unimplemented). It still makes no direct
    `requests` network calls of its own (Stripe's SDK handles that)."""
    modules = _imported_top_level_modules(Path("src/services/billing_provider.py"))
    assert "stripe" in modules
    assert "requests" not in modules


def test_app_pages_do_not_execute_raw_sql_mutations():
    """Phase 7: 'All write operations must go through service/repository
    functions. Do not put raw SQL mutations directly inside Streamlit
    page code.' This does NOT ban .execute() outright -- Step 10's
    get_automation_health() already runs a read-only SELECT directly in
    apps/adminapp.py, predating Step 11 and out of scope to move. What
    must never appear in apps/*.py is a raw SQL *mutation* against the
    accounts tables -- grant/revoke override, disable/reactivate user
    must go through src/services/accounts_repository.py instead. Scoped
    to the accounts tables specifically (not a blanket "UPDATE"/"INSERT"
    keyword ban) since apps/*.py legitimately contains unrelated English
    words like "Update" (button labels) and Google Sheets' own
    .update()/.append_row() calls, which are a pre-existing, separate
    data layer (see the Step 11 report's audit)."""
    accounts_tables = ("USERS", "SUBSCRIPTIONS", "ENTITLEMENT_OVERRIDES", "STRIPE_EVENTS")
    mutation_keywords = ("INSERT INTO", "UPDATE", "DELETE FROM")
    offenders = []
    for rel_path in APP_FILES:
        path = Path(rel_path)
        assert path.exists(), f"expected {rel_path} to exist"
        source = path.read_text(encoding="utf-8").upper()
        for keyword in mutation_keywords:
            for table in accounts_tables:
                if f"{keyword} {table}" in source:
                    offenders.append((rel_path, f"{keyword} {table}"))
    assert offenders == [], (
        f"Raw SQL mutation(s) against accounts tables found directly in app page "
        f"code: {offenders} -- writes must go through "
        "src/services/accounts_repository.py"
    )


def test_auth_session_state_keys_are_only_touched_in_auth_session_module():
    """Guards against auth logic being spread back out across Streamlit
    pages -- every other file must go through
    src/services/auth_session.py's functions instead of reading/writing
    these session_state keys itself."""
    offenders = []
    for path in _all_repo_py_files():
        rel = path.as_posix()
        if rel == AUTH_SESSION_OWNER or rel.startswith("tests/"):
            continue
        source = path.read_text(encoding="utf-8")
        for key in AUTH_SESSION_KEYS:
            if key in source:
                offenders.append((rel, key))
    assert offenders == [], (
        f"Auth session_state key(s) referenced outside {AUTH_SESSION_OWNER}: {offenders}"
    )


def test_admin_key_secret_is_only_read_in_auth_session_module():
    """The legacy admin_key bootstrap fallback must have exactly one read
    site -- apps/adminapp.py must not compare against it directly (that
    was the pre-Step-11 pattern; see the Step 11 report's admin-key
    migration decision)."""
    offenders = []
    for rel_path in APP_FILES:
        source = Path(rel_path).read_text(encoding="utf-8")
        if 'secrets["admin_key"]' in source or "secrets['admin_key']" in source:
            offenders.append(rel_path)
    assert offenders == [], (
        f"admin_key read directly in app page code (should go through "
        f"src/services/auth_session.py::authorize_admin_or_legacy_key): {offenders}"
    )


def test_entitlement_tier_literals_are_not_scattered_across_app_pages():
    """Guards against `if user.plan == "pro"`-style checks creeping back
    into page code -- apps/*.py must branch on Entitlements' can_* booleans
    (or CurrentUser), never on a raw tier/plan string literal."""
    tier_literal_patterns = (
        '== "FREE"', "== 'FREE'",
        '== "PRO"', "== 'PRO'",
        '== "ADMIN"', "== 'ADMIN'",
        '.plan ==', '.plan_key ==',
    )
    offenders = []
    for rel_path in APP_FILES:
        source = Path(rel_path).read_text(encoding="utf-8")
        for pattern in tier_literal_patterns:
            if pattern in source:
                offenders.append((rel_path, pattern))
    assert offenders == [], f"Scattered tier-literal comparison(s) found: {offenders}"


def test_entitlement_internals_list_is_itself_consistent():
    """Sanity check the guard isn't silently pointing at deleted files."""
    for rel_path in ENTITLEMENT_INTERNALS:
        assert Path(rel_path).exists(), f"expected {rel_path} to exist"


def test_public_app_never_writes_accounts_or_subscription_state():
    """Phase 6/11: 'no public UI mutation of subscription state' /
    'public display code performing subscription writes'. The public app
    may only ever READ entitlements (via CurrentUser, produced by
    src/services/auth_session.py's sign-in/sign-up, which only ever
    creates/reads a user's own identity row) -- it must never import the
    accounts repository directly, since every write to subscriptions/
    entitlement_overrides is an admin-only action in apps/adminapp.py.

    Step 13 update: publicapp.py DOES now import billing_provider (to
    call create_checkout_session()/create_customer_portal_session() --
    the sanctioned abstraction, see src/services/billing_provider.py).
    That is not a repository bypass: billing_provider.py is the only
    thing that ever writes to subscriptions, and it does so only in
    response to a verified Stripe webhook, never from a Checkout/Portal
    call itself (see tests/test_stripe_billing_provider.py::
    test_checkout_does_not_write_subscription_state). What must still
    never happen is publicapp.py reaching around that abstraction
    straight into accounts_repository."""
    source = Path("apps/publicapp.py").read_text(encoding="utf-8")
    assert "accounts_repository" not in source


def test_admin_authorization_gate_runs_unconditionally_before_any_tab():
    """Phase 7: 'Navigation visibility is not authorization. A non-admin
    who knows the admin URL must still be denied.' Guards against the
    authorization check being moved so it only runs for some tabs, or
    after tab content is already built -- it must appear, and call
    st.stop() on failure, before the Users & Access tab (the last tab)
    is defined."""
    source = Path("apps/adminapp.py").read_text(encoding="utf-8")
    auth_call_index = source.index("authorize_admin_or_legacy_key()")
    stop_index = source.index("st.stop()", auth_call_index)
    users_tab_index = source.index("with users_tab:")

    assert auth_call_index < stop_index < users_tab_index, (
        "expected authorize_admin_or_legacy_key() followed by st.stop() to appear "
        "before the Users & Access tab body"
    )
