"""
Step 13 static architecture/security guards -- Stripe billing
integration. Mirrors the style of tests/test_step11_static_guards.py and
tests/test_step12_static_guards.py: guards against the *shape* of a
regression (a page importing stripe directly, a hardcoded/overridable
Price ID, webhook processing without signature verification) rather
than full static analysis.
"""

import ast
from pathlib import Path

APP_FILES = ("apps/publicapp.py", "apps/adminapp.py")

BILLING_ENV_VAR_NAMES = (
    "STRIPE_SECRET_KEY",
    "STRIPE_PUBLISHABLE_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "STRIPE_PRO_PRICE_ID",
    "STRIPE_SUCCESS_URL",
    "STRIPE_CANCEL_URL",
    "STRIPE_PORTAL_RETURN_URL",
)

ENV_VAR_ALLOWED_FILES = ("src/services/billing_config.py",)


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


def test_apps_never_import_stripe_directly():
    """Phase 2/18: 'Do NOT import Stripe directly from apps/publicapp.py
    or apps/adminapp.py.' Stripe-specific code belongs in
    src/services/billing_provider.py only."""
    offenders = []
    for rel_path in APP_FILES:
        modules = _imported_top_level_modules(Path(rel_path))
        if "stripe" in modules:
            offenders.append(rel_path)
    assert offenders == [], f"Direct `stripe` import found in: {offenders}"


def test_entitlement_service_never_imports_stripe():
    """Phase 2/10/18: EntitlementService must never depend on Stripe --
    it only ever reads normalized local subscription/override state."""
    modules = _imported_top_level_modules(Path("src/services/entitlement_service.py"))
    assert "stripe" not in modules


def test_entitlement_service_never_imports_billing_provider():
    """Phase 10: the data flow is one-way (webhook -> subscriptions
    table -> EntitlementService) -- entitlement_service.py must not
    import billing_provider.py at all, in either direction."""
    modules = _imported_top_level_modules(Path("src/services/entitlement_service.py"))
    assert "billing_provider" not in modules


def test_apps_never_reference_billing_secrets_or_arbitrary_price_ids():
    """Phase 3/5/18: billing secrets must never reach page code, and the
    Price ID passed to Stripe must always come from billing_config, never
    from a caller-suppliable value -- apps/*.py should only ever pass the
    symbolic plan_key="pro" literal, never a raw price_... id or a
    STRIPE_* secret."""
    forbidden_substrings = (
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRO_PRICE_ID",
        "stripe.api_key",
    )
    offenders = []
    for rel_path in APP_FILES:
        source = Path(rel_path).read_text(encoding="utf-8")
        for pattern in forbidden_substrings:
            if pattern in source:
                offenders.append((rel_path, pattern))
    assert offenders == [], f"Billing secret/price-id reference found in app pages: {offenders}"


def test_billing_env_vars_are_only_read_in_billing_config_module():
    """Phase 3/18: no ad-hoc os.environ reads for billing configuration
    scattered across modules -- everything goes through
    src/services/billing_config.py (mirrors auth_config.py's role)."""
    offenders = []
    for path in _all_repo_py_files():
        rel = path.as_posix()
        if rel in ENV_VAR_ALLOWED_FILES or rel.startswith("tests/"):
            continue
        source = path.read_text(encoding="utf-8")
        for var_name in BILLING_ENV_VAR_NAMES:
            if f'"{var_name}"' in source or f"'{var_name}'" in source:
                offenders.append((rel, var_name))
    assert offenders == [], (
        f"Billing env var(s) referenced outside src/services/billing_config.py: {offenders}"
    )


def _stripe_provider_webhook_method_body() -> str:
    """Returns just StripeBillingProvider.handle_webhook_event()'s source
    -- there are two OTHER `def handle_webhook_event(` in this file (the
    ABC's abstract declaration and NullBillingProvider's stub), so the
    search must be scoped to start after `class StripeBillingProvider`."""
    source = Path("src/services/billing_provider.py").read_text(encoding="utf-8")
    class_start = source.index("class StripeBillingProvider")
    method_start = source.index("def handle_webhook_event(", class_start)
    next_method = source.index("\n    def ", method_start + 1)
    return source[method_start:next_method]


def test_webhook_signature_is_verified_before_any_database_write():
    """Phase 7/18: 'never trust webhook payload without signature
    verification.' Guards against a future edit accidentally moving a DB
    write before construct_event() in handle_webhook_event()."""
    method_body = _stripe_provider_webhook_method_body()

    construct_event_index = method_body.index("construct_event")
    first_db_write_index = min(
        idx
        for idx in (
            method_body.find("try_claim_stripe_event"),
            method_body.find("_get_db_connection"),
        )
        if idx != -1
    )
    assert construct_event_index < first_db_write_index, (
        "expected Webhook.construct_event() (signature verification) to run before "
        "any database connection/write in handle_webhook_event()"
    )


def test_stripe_events_are_claimed_before_dispatch():
    """Phase 7/14/18: idempotency check must happen before any
    entitlement-affecting dispatch -- guards against a future edit
    reordering try_claim_stripe_event() after _dispatch_event()."""
    method_body = _stripe_provider_webhook_method_body()

    claim_index = method_body.index("try_claim_stripe_event")
    dispatch_index = method_body.index("_dispatch_event")
    assert claim_index < dispatch_index


def test_billing_provider_never_uses_email_to_resolve_stripe_customers():
    """Phase 4/15/18: 'Never use email alone to associate a Stripe event
    with a user.' Guards against a future edit adding a
    Customer.list(email=...)/Customer.search(...) lookup path -- identity
    resolution must stay confined to stripe_customer_id / metadata.local_user_id
    (see StripeBillingProvider._resolve_user_id)."""
    source = Path("src/services/billing_provider.py").read_text(encoding="utf-8")
    assert "Customer.list(" not in source
    assert "Customer.search(" not in source


def test_checkout_return_banner_never_mutates_state():
    """Phase 12/18: 'A successful redirect back from Stripe does NOT
    itself grant PRO' / 'PRO grant from checkout return query params'
    must never happen. render_checkout_return_banner() must be a pure
    read-and-display function -- no repository/session writes at all."""
    source = Path("src/services/billing_session.py").read_text(encoding="utf-8")
    func_start = source.index("def render_checkout_return_banner(")
    func_body = source[func_start : source.index("\ndef ", func_start + 1)]

    forbidden = (
        "accounts_repository",
        "create_override",
        "set_user_active",
        "sync_subscription",
        "st.session_state",
        "_auth_session",
    )
    offenders = [pattern for pattern in forbidden if pattern in func_body]
    assert offenders == [], (
        f"render_checkout_return_banner() references state-mutating pattern(s): {offenders}"
    )


def test_billing_session_module_never_imports_stripe_or_accounts_repository():
    """The Streamlit-facing billing glue layer only ever calls
    get_billing_provider() -- it must not import stripe directly nor
    reach around into accounts_repository."""
    modules = _imported_top_level_modules(Path("src/services/billing_session.py"))
    assert "stripe" not in modules
    assert "accounts_repository" not in modules


def test_webhook_service_does_not_duplicate_subscription_logic():
    """Phase 8/18: the standalone webhook_service must delegate entirely
    to StripeBillingProvider.handle_webhook_event() -- it must not import
    stripe or accounts_repository itself, which would indicate
    subscription logic leaking into the HTTP layer."""
    modules = _imported_top_level_modules(Path("webhook_service/main.py"))
    assert "stripe" not in modules
    assert "accounts_repository" not in modules
