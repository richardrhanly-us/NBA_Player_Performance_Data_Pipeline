"""
Step 11/12: authentication provider abstraction.

No Streamlit import here on purpose -- this module is pure request/
response logic against an identity provider's REST API (or, for
DevAuthProvider, no network at all), so it is fully unit-testable
without a Streamlit runtime. src/services/auth_session.py is the thin
layer above this that wires a provider's result into
st.session_state/CurrentUser, including token validation/refresh.

Chosen provider: Supabase Auth (GoTrue), called directly over its REST
API via `requests` (already a project dependency) -- no new SDK, no JWT/
crypto dependency. Password storage/verification/email flows are all
handled by Supabase, never by this codebase (see AuthProvider's
docstring for why: "Do NOT build insecure custom password storage").

Step 12 hardening: DevAuthProvider is no longer an implicit fallback.
get_auth_provider() now raises AuthNotConfiguredError when Supabase
isn't configured and dev auth hasn't been explicitly enabled (see
src/services/auth_config.py::can_use_dev_auth) -- silently activating
email-only authentication in a misconfigured deployment was the exact
anti-pattern Step 12 exists to remove. DevAuthProvider still can never
itself grant ADMIN: ADMIN only ever comes from entitlement_overrides/
users state in the database (see src/services/entitlement_service.py),
never from which provider authenticated a session.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.services import auth_config


class AuthError(RuntimeError):
    """Raised for any sign-in/sign-up/refresh/revalidation failure --
    invalid credentials, provider unreachable, malformed response, a
    revoked/expired token, etc. Callers show a generic, non-specific
    message; the original exception message is safe to log (it never
    contains a token or password) but is never shown verbatim to
    end users for provider-side failures (status code only)."""


class AuthNotConfiguredError(RuntimeError):
    """Raised when no usable authentication provider is available at
    all -- Supabase isn't configured AND dev auth isn't explicitly
    enabled (or is refused by the production hint). Distinct from
    AuthError (a configured provider that rejected a request) so
    callers/tests can tell "nothing is configured" apart from "you typed
    the wrong password"."""


class EmailConfirmationRequiredError(RuntimeError):
    """Raised by sign_up() when the identity provider created the
    account but did not issue a session -- i.e. the Supabase project
    requires email confirmation before first sign-in. This is a SUCCESS
    outcome for account creation, not a failure; auth_session.py
    surfaces it as an informational notice, never as an error."""


@dataclass(frozen=True)
class AuthResult:
    auth_provider: str
    auth_subject: str
    email: str
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None  # unix epoch seconds (UTC), or None = does not expire


class AuthProvider(ABC):
    """Do NOT build insecure custom password storage -- every
    implementation of this interface delegates credential verification
    to an external identity provider (or, for DevAuthProvider, performs
    no credential verification at all and is gated to development use
    only)."""

    name: str

    @abstractmethod
    def sign_in(self, email: str, password: str) -> AuthResult: ...

    @abstractmethod
    def sign_up(self, email: str, password: str) -> AuthResult: ...

    @abstractmethod
    def sign_out(self, access_token: str) -> None: ...

    def refresh(self, refresh_token: str) -> AuthResult:
        """Exchanges a refresh token for a new access token. Only
        SupabaseAuthProvider implements this -- dev/legacy sessions never
        expire in the first place (see auth_session.py, which never
        calls this for a non-"supabase" session)."""
        raise NotImplementedError(f"{self.name} does not support token refresh.")

    def get_user(self, access_token: str) -> AuthResult:
        """Re-fetches the authoritative identity for a still-live access
        token, used for periodic provider-identity revalidation (Step 12
        Phase 4). Raises AuthError if the token is no longer valid."""
        raise NotImplementedError(f"{self.name} does not support identity revalidation.")

    def request_password_reset(self, email: str) -> None:
        """Triggers the provider's own password-reset email flow. Never
        implemented locally -- see AuthProvider's docstring."""
        raise NotImplementedError(f"{self.name} does not support password reset.")


class SupabaseAuthProvider(AuthProvider):
    name = "supabase"

    def __init__(self, base_url: str, anon_key: str, *, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._anon_key = anon_key
        self._timeout = timeout

    def _headers(self, *, bearer: str | None = None) -> dict:
        return {
            "apikey": self._anon_key,
            "Authorization": f"Bearer {bearer or self._anon_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, json_body: dict, *, bearer: str | None = None) -> dict:
        import requests

        try:
            response = requests.post(
                f"{self._base_url}{path}",
                json=json_body,
                headers=self._headers(bearer=bearer),
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise AuthError(f"Could not reach the identity provider: {e}") from e

        return self._parse_response(response)

    def _get(self, path: str, *, bearer: str) -> dict:
        import requests

        try:
            response = requests.get(
                f"{self._base_url}{path}",
                headers=self._headers(bearer=bearer),
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise AuthError(f"Could not reach the identity provider: {e}") from e

        return self._parse_response(response)

    def _parse_response(self, response) -> dict:
        if response.status_code >= 500:
            raise AuthError(
                f"The identity provider is temporarily unavailable (status "
                f"{response.status_code})."
            )
        if response.status_code >= 400:
            raise AuthError(
                f"The identity provider rejected this request (status "
                f"{response.status_code})."
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as e:
            raise AuthError("Identity provider returned an unreadable response.") from e

    def sign_in(self, email: str, password: str) -> AuthResult:
        data = self._post(
            "/auth/v1/token?grant_type=password",
            {"email": email, "password": password},
        )
        return self._to_result(data, require_session=True)

    def sign_up(self, email: str, password: str) -> AuthResult:
        data = self._post("/auth/v1/signup", {"email": email, "password": password})
        return self._to_result(data, require_session=False)

    def sign_out(self, access_token: str) -> None:
        import requests

        try:
            requests.post(
                f"{self._base_url}/auth/v1/logout",
                headers=self._headers(bearer=access_token),
                timeout=self._timeout,
            )
        except requests.RequestException:
            pass  # best-effort -- the local session is cleared regardless

    def refresh(self, refresh_token: str) -> AuthResult:
        data = self._post(
            "/auth/v1/token?grant_type=refresh_token",
            {"refresh_token": refresh_token},
        )
        return self._to_result(data, require_session=True)

    def get_user(self, access_token: str) -> AuthResult:
        data = self._get("/auth/v1/user", bearer=access_token)
        subject = data.get("id")
        email = data.get("email")
        if not subject or not email:
            raise AuthError("Session is no longer valid.")
        return AuthResult(
            auth_provider=self.name,
            auth_subject=str(subject),
            email=str(email),
            access_token=access_token,
        )

    def request_password_reset(self, email: str) -> None:
        self._post("/auth/v1/recover", {"email": email})

    def _to_result(self, data: dict, *, require_session: bool) -> AuthResult:
        user = data.get("user") or {}
        subject = user.get("id")
        email = user.get("email")
        access_token = data.get("access_token")

        if not access_token and not require_session:
            if subject and email:
                raise EmailConfirmationRequiredError(
                    "Account created. Check your email to confirm it, then sign in."
                )
            raise AuthError(
                "Sign-up failed: the identity provider response was missing "
                "required fields."
            )

        if not subject or not email or not access_token:
            raise AuthError(
                "Sign-in failed: the identity provider response was missing "
                "required fields."
            )

        refresh_token = data.get("refresh_token")
        expires_at = data.get("expires_at")
        if expires_at is None:
            expires_in = data.get("expires_in")
            if expires_in is not None:
                try:
                    expires_at = time.time() + float(expires_in)
                except (TypeError, ValueError):
                    expires_at = None
        else:
            try:
                expires_at = float(expires_at)
            except (TypeError, ValueError):
                expires_at = None

        return AuthResult(
            auth_provider=self.name,
            auth_subject=str(subject),
            email=str(email),
            access_token=str(access_token),
            refresh_token=str(refresh_token) if refresh_token else None,
            expires_at=expires_at,
        )


class DevAuthProvider(AuthProvider):
    """
    Development-only fallback -- Step 12: available ONLY when
    src/services/auth_config.py::can_use_dev_auth() returns True (i.e.
    DEV_AUTH_ENABLED=true, and no unaddressed production hint). Accepts
    any non-empty email and ANY password (not checked at all);
    auth_subject is a deterministic hash of the email so the same email
    always resolves to the same account across a dev session.

    This is intentionally unsafe as a production identity check -- it
    must never run when real auth is configured (or dev auth is not
    explicitly enabled), and it never grants ADMIN by itself (see
    module docstring). Sessions never expire (expires_at=None) --
    there's no real token to expire, and auth_session.py additionally
    re-checks can_use_dev_auth() on every request so a dev session dies
    the moment dev auth is disabled, even mid-session.
    """

    name = "dev"

    def sign_in(self, email: str, password: str) -> AuthResult:
        return self._result_for(email)

    def sign_up(self, email: str, password: str) -> AuthResult:
        return self._result_for(email)

    def sign_out(self, access_token: str) -> None:
        return None

    def request_password_reset(self, email: str) -> None:
        raise AuthError("Password reset is not available in dev mode.")

    def _result_for(self, email: str) -> AuthResult:
        email = (email or "").strip().lower()
        if not email or "@" not in email:
            raise AuthError("Enter a valid email address.")
        subject = compute_dev_auth_subject(email)
        return AuthResult(
            auth_provider=self.name,
            auth_subject=subject,
            email=email,
            access_token=f"dev-session:{subject}",
            refresh_token=None,
            expires_at=None,
        )


def compute_dev_auth_subject(email: str) -> str:
    """The deterministic auth_subject DevAuthProvider assigns to a given
    email -- exposed so scripts/create_admin_user.py can bootstrap an
    ADMIN override for a dev-mode account without having to sign in
    first."""
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:32]
    return f"dev:{digest}"


def is_supabase_configured() -> bool:
    return auth_config.is_supabase_configured()


def get_auth_provider() -> AuthProvider:
    """
    The one factory every caller uses -- never construct a provider
    directly outside this function (or tests). Fail-closed (Step 12):

        Supabase configured                      -> SupabaseAuthProvider
        Supabase absent + dev auth explicitly on  -> DevAuthProvider
        Supabase absent + dev auth off/unset      -> AuthNotConfiguredError

    Never silently falls back to email-only authentication.
    """
    url = auth_config.supabase_url()
    anon_key = auth_config.supabase_anon_key()

    if url and anon_key:
        return SupabaseAuthProvider(
            base_url=url,
            anon_key=anon_key,
        )
    if auth_config.can_use_dev_auth():
        return DevAuthProvider()
    raise AuthNotConfiguredError(
        "Authentication is not configured for this deployment. Set SUPABASE_URL "
        "and SUPABASE_ANON_KEY, or set DEV_AUTH_ENABLED=true for local development."
    )
