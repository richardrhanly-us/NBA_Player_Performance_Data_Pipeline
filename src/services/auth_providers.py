"""
Step 11: authentication provider abstraction.

No Streamlit import here on purpose -- this module is pure request/
response logic against an identity provider's REST API (or, for
DevAuthProvider, no network at all), so it is fully unit-testable
without a Streamlit runtime. src/services/auth_session.py is the thin
layer above this that wires a provider's result into
st.session_state/CurrentUser.

Chosen provider: Supabase Auth (GoTrue), called directly over its REST
API via `requests` (already a project dependency) -- no new SDK, no JWT/
crypto dependency. Password storage/verification/email flows are all
handled by Supabase, never by this codebase (see AuthProvider's
docstring for why: "Do NOT build insecure custom password storage").

Dev fallback: if SUPABASE_URL/SUPABASE_ANON_KEY are not configured
(e.g. local development, or before the external Supabase project has
been created), get_auth_provider() returns DevAuthProvider instead --
email-only, no password check, clearly unsuitable for production. It can
issue a session for ANY tier a stored user already has, but it can never
itself grant ADMIN: ADMIN only ever comes from
entitlement_overrides/users state in the database (see
src/services/entitlement_service.py), never from which provider
authenticated a session. This is what keeps the dev fallback "safe" per
the Step 11 brief.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


class AuthError(RuntimeError):
    """Raised for any sign-in/sign-up failure -- invalid credentials,
    provider unreachable, malformed response, etc. Callers show a
    generic "sign-in failed" message; the original exception message is
    logged, never a raw provider response (which could echo back
    request details)."""


@dataclass(frozen=True)
class AuthResult:
    auth_provider: str
    auth_subject: str
    email: str
    access_token: str


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


class SupabaseAuthProvider(AuthProvider):
    name = "supabase"

    def __init__(self, base_url: str, anon_key: str, *, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._anon_key = anon_key
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "apikey": self._anon_key,
            "Authorization": f"Bearer {self._anon_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, json_body: dict) -> dict:
        import requests

        try:
            response = requests.post(
                f"{self._base_url}{path}",
                json=json_body,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise AuthError(f"Could not reach the identity provider: {e}") from e

        if response.status_code >= 400:
            raise AuthError(
                "Sign-in failed: the identity provider rejected this request "
                f"(status {response.status_code})."
            )

        try:
            return response.json()
        except ValueError as e:
            raise AuthError("Identity provider returned an unreadable response.") from e

    def sign_in(self, email: str, password: str) -> AuthResult:
        data = self._post(
            "/auth/v1/token?grant_type=password",
            {"email": email, "password": password},
        )
        return self._to_result(data)

    def sign_up(self, email: str, password: str) -> AuthResult:
        data = self._post("/auth/v1/signup", {"email": email, "password": password})
        return self._to_result(data)

    def sign_out(self, access_token: str) -> None:
        import requests

        try:
            requests.post(
                f"{self._base_url}/auth/v1/logout",
                headers={**self._headers(), "Authorization": f"Bearer {access_token}"},
                timeout=self._timeout,
            )
        except requests.RequestException:
            pass  # best-effort -- the local session is cleared regardless

    def _to_result(self, data: dict) -> AuthResult:
        user = data.get("user") or {}
        subject = user.get("id")
        email = user.get("email")
        access_token = data.get("access_token")
        if not subject or not email or not access_token:
            raise AuthError(
                "Sign-in failed: the identity provider response was missing "
                "required fields."
            )
        return AuthResult(
            auth_provider=self.name,
            auth_subject=str(subject),
            email=str(email),
            access_token=str(access_token),
        )


class DevAuthProvider(AuthProvider):
    """
    Development-only fallback -- active automatically whenever
    SUPABASE_URL/SUPABASE_ANON_KEY are not configured (see
    get_auth_provider() below). Accepts any non-empty email and ANY
    password (not checked at all); auth_subject is a deterministic hash
    of the email so the same email always resolves to the same account
    across a dev session.

    This is intentionally unsafe as a production identity check -- it
    must never run when real auth is configured, and it never grants
    ADMIN by itself (see module docstring).
    """

    name = "dev"

    def sign_in(self, email: str, password: str) -> AuthResult:
        return self._result_for(email)

    def sign_up(self, email: str, password: str) -> AuthResult:
        return self._result_for(email)

    def sign_out(self, access_token: str) -> None:
        return None

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
        )


def compute_dev_auth_subject(email: str) -> str:
    """The deterministic auth_subject DevAuthProvider assigns to a given
    email -- exposed so scripts/create_admin_user.py can bootstrap an
    ADMIN override for a dev-mode account without having to sign in
    first."""
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:32]
    return f"dev:{digest}"


def is_supabase_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL")) and bool(
        os.environ.get("SUPABASE_ANON_KEY")
    )


def get_auth_provider() -> AuthProvider:
    """The one factory every caller uses -- never construct a provider
    directly outside this function (or tests)."""
    if is_supabase_configured():
        return SupabaseAuthProvider(
            base_url=os.environ["SUPABASE_URL"],
            anon_key=os.environ["SUPABASE_ANON_KEY"],
        )
    return DevAuthProvider()
