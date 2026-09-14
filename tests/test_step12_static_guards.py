"""
Step 12 static architecture/security guards -- production auth hardening.

Mirrors the style of tests/test_step7_static_guard.py and
tests/test_step11_static_guards.py: these are guards against the *shape*
of a regression (dev auth quietly becoming an implicit fallback again, a
page reading a raw token, an env var read scattered outside the one
config module) rather than full static analysis.
"""

from pathlib import Path

APP_FILES = ("apps/publicapp.py", "apps/adminapp.py")

AUTH_ENV_VAR_NAMES = (
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "DEV_AUTH_ENABLED",
    "DEV_AUTH_FORCE",
    "ENVIRONMENT",
    "APP_ENV",
    "LEGACY_ADMIN_KEY_ENABLED",
)

# auth_session.py is allowed one narrow exception: ADMIN_KEY / admin_key
# secrets access, documented in its own module docstring as the one
# place st.secrets is read (auth_config.py has no Streamlit import).
# Step 14: ENVIRONMENT/APP_ENV moved to src/services/environment.py (the
# shared APP_ENV classification also used by readiness checks) --
# auth_config.py now delegates to it rather than reading those two
# itself, so environment.py is allowed too.
ENV_VAR_ALLOWED_FILES = ("src/services/auth_config.py", "src/services/environment.py")

TOKEN_FIELD_NAMES = ("access_token", "refresh_token")


def _all_repo_py_files():
    repo_root = Path(".")
    for path in repo_root.rglob("*.py"):
        rel = str(path.as_posix())
        if rel.startswith((".venv/", "__pycache__/")) or "__pycache__" in rel:
            continue
        yield path


def test_dev_auth_is_not_an_implicit_fallback():
    """Phase 2/13: get_auth_provider() must gate DevAuthProvider behind
    can_use_dev_auth() and raise AuthNotConfiguredError otherwise -- the
    Step 11 behavior (silently using DevAuthProvider whenever Supabase
    wasn't configured) is exactly what this guards against."""
    source = Path("src/services/auth_providers.py").read_text(encoding="utf-8")
    func_start = source.index("def get_auth_provider()")
    func_body = source[func_start:]

    can_use_index = func_body.index("can_use_dev_auth")
    dev_provider_index = func_body.index("return DevAuthProvider()")
    not_configured_index = func_body.index("raise AuthNotConfiguredError")

    assert can_use_index < dev_provider_index, (
        "expected can_use_dev_auth() to be checked before returning DevAuthProvider()"
    )
    assert dev_provider_index < not_configured_index, (
        "expected AuthNotConfiguredError to be the final fallback, after the "
        "DevAuthProvider branch"
    )


def test_app_pages_never_reference_raw_auth_tokens():
    """Phase 4/13: token handling must stay inside auth_session.py/
    auth_providers.py -- apps/*.py must never read/store an access_token
    or refresh_token itself."""
    offenders = []
    for rel_path in APP_FILES:
        source = Path(rel_path).read_text(encoding="utf-8")
        for field in TOKEN_FIELD_NAMES:
            if field in source:
                offenders.append((rel_path, field))
    assert offenders == [], f"Raw auth token field(s) referenced in app pages: {offenders}"


def test_app_pages_never_call_supabase_directly():
    """Phase 13: no direct Supabase/GoTrue calls from Streamlit page
    code -- every provider call must go through
    src/services/auth_providers.py via src/services/auth_session.py."""
    offenders = []
    for rel_path in APP_FILES:
        source = Path(rel_path).read_text(encoding="utf-8").lower()
        if "supabase" in source or "/auth/v1/" in source:
            offenders.append(rel_path)
    assert offenders == [], f"Direct Supabase reference(s) found in app pages: {offenders}"


def test_no_email_based_admin_checks():
    """Phase 6/13: ADMIN must never be inferred from an email address or
    domain -- only from local entitlement_overrides/subscriptions state
    (see src/services/entitlement_service.py::compute_effective_tier)."""
    suspicious_patterns = (
        "email.endswith(",
        "email.split(\"@\")",
        "email.split('@')",
        '"@admin"',
        "'@admin'",
    )
    files_to_check = (
        "src/services/entitlement_service.py",
        "src/services/auth_session.py",
        "src/services/accounts_repository.py",
        "apps/publicapp.py",
        "apps/adminapp.py",
    )
    offenders = []
    for rel_path in files_to_check:
        source = Path(rel_path).read_text(encoding="utf-8")
        for pattern in suspicious_patterns:
            if pattern in source:
                offenders.append((rel_path, pattern))
    assert offenders == [], f"Email-based admin-check pattern(s) found: {offenders}"


def test_tokens_are_never_logged():
    """Phase 10/13: token values must never reach a print/log/audit-log
    call -- scans every non-test .py file for a line that mentions a
    token field name alongside a logging-shaped call."""
    logging_markers = ("print(", "write_admin_log(", "logging.", "st.write(")
    offenders = []
    for path in _all_repo_py_files():
        rel = path.as_posix()
        if rel.startswith("tests/"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(field in line for field in TOKEN_FIELD_NAMES) and any(
                marker in line for marker in logging_markers
            ):
                offenders.append((rel, lineno, line.strip()))
    assert offenders == [], f"Possible token logging found: {offenders}"


def test_passwords_are_never_logged():
    """Phase 5/10/13: a password must never be passed to a print/log/
    audit-log call -- passwords go to the identity provider and nowhere
    else (see AuthProvider's docstring)."""
    logging_markers = ("print(", "write_admin_log(", "logging.", "st.write(")
    offenders = []
    for path in _all_repo_py_files():
        rel = path.as_posix()
        if rel.startswith("tests/"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "password" in line.lower() and any(marker in line for marker in logging_markers):
                offenders.append((rel, lineno, line.strip()))
    assert offenders == [], f"Possible password logging found: {offenders}"


def test_legacy_admin_key_session_flag_is_config_gated():
    """Phase 3/13: an established legacy-key session must not be honored
    just because the session_state flag is set -- it must also be
    checked against is_legacy_admin_key_enabled() so flipping the
    feature off kills any session already using it (mirrors the
    dev-auth kill switch)."""
    source = Path("src/services/auth_session.py").read_text(encoding="utf-8")
    assert "is_legacy_admin_key_enabled()" in source
    # The specific fast-path check must combine both conditions.
    assert (
        'st.session_state.get(_LEGACY_ADMIN_KEY) and auth_config.is_legacy_admin_key_enabled()'
        in source
    )


def test_auth_env_vars_are_only_read_in_auth_config_module():
    """Phase 11/13: no ad-hoc os.environ reads for auth configuration
    scattered across modules -- everything goes through
    src/services/auth_config.py."""
    offenders = []
    for path in _all_repo_py_files():
        rel = path.as_posix()
        if rel in ENV_VAR_ALLOWED_FILES or rel.startswith("tests/"):
            continue
        source = path.read_text(encoding="utf-8")
        for var_name in AUTH_ENV_VAR_NAMES:
            if f'"{var_name}"' in source or f"'{var_name}'" in source:
                offenders.append((rel, var_name))
    assert offenders == [], (
        f"Auth env var(s) referenced outside src/services/auth_config.py: {offenders}"
    )


def test_admin_key_env_var_only_read_via_auth_config_helper():
    """ADMIN_KEY's value is still fetched from auth_session.py (it also
    checks st.secrets, which requires Streamlit), but ONLY via
    auth_config.admin_key_from_env() -- never a second, separate
    os.environ.get("ADMIN_KEY") call anywhere else."""
    offenders = []
    for path in _all_repo_py_files():
        rel = path.as_posix()
        if rel == "src/services/auth_config.py" or rel.startswith("tests/"):
            continue
        source = path.read_text(encoding="utf-8")
        if 'os.environ.get("ADMIN_KEY")' in source or "os.environ.get('ADMIN_KEY')" in source:
            offenders.append(rel)
    assert offenders == [], f"Direct ADMIN_KEY env read found outside auth_config.py: {offenders}"
