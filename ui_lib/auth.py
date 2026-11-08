"""احراز هویت ساده‌ی UI (Issue #12) — برای حالت BIND_PUBLIC=1.

Application-level password gate for the Streamlit UI. Used whenever a
credential is configured or public exposure is requested (``BIND_PUBLIC=1``);
plain localhost deployments without a credential stay anonymous.

Design notes:
- Streamlit 1.66's native ``st.login`` is OIDC-provider-based and requires an
  external identity provider plus registered redirect URLs — unsuitable for
  an offline research prototype, so this minimal password gate is used for
  the single shared operator credential (documented decision; see the
  network-isolation section of reports/PRIVACY_AND_SECURITY.md).
- The credential comes from ``APP_AUTH_PASSWORD`` or Streamlit secrets
  (``[auth] password``). It is never hardcoded, never logged, never stored
  in session state — only an ``auth_ok`` flag and a role label ("operator",
  deliberately not a per-user identity).
- Comparison is constant-time (``hmac.compare_digest``).
- ``require_auth()`` must be called at the very top of EVERY page (before
  any content-rendering call) so unauthenticated visitors never receive
  protected content.
- Logout clears all auth-related session keys.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

from security_config import (
    AUTH_PASSWORD_VAR,
    AUTH_SECRETS_KEY,
    AUTH_SECRETS_SECTION,
    SecurityConfigError,
    validate_deployment_config,
)

AUTH_SESSION_KEY = "auth_ok"
AUTH_USER_KEY = "auth_user"
AUTH_USER_LABEL = "operator"  # a role label, not a per-user identity


def get_auth_password() -> str | None:
    """Resolve the UI credential from env or Streamlit secrets.

    Returns None when neither source provides one. The value is never
    logged or embedded in exceptions.
    """
    raw = os.environ.get(AUTH_PASSWORD_VAR, "")
    if raw.strip():
        return raw
    try:
        secret_value = st.secrets.get(AUTH_SECRETS_SECTION, {}).get(
            AUTH_SECRETS_KEY
        )
    except Exception:
        return None  # no/garbled secrets.toml — treat as absent
    if secret_value and str(secret_value).strip():
        return str(secret_value)
    return None


def check_password(candidate: object, expected: object) -> bool:
    """Constant-time credential comparison (never leaks via timing)."""
    if not isinstance(candidate, (str, bytes)) or not isinstance(expected, (str, bytes)):
        return False
    if isinstance(candidate, str) or isinstance(expected, str):
        candidate = str(candidate).encode("utf-8")
        expected = str(expected).encode("utf-8")
    return hmac.compare_digest(candidate, expected)


def _is_authenticated(session) -> bool:
    """Pure check usable with the real or a fake session-state mapping."""
    return bool(session.get(AUTH_SESSION_KEY))


def logout(session) -> None:
    """Clear every auth-related session key."""
    for key in (AUTH_SESSION_KEY, AUTH_USER_KEY):
        if key in session:
            del session[key]


def require_auth() -> bool:
    """Authentication gate for one page run. True = proceed with content.

    Order of operations per call:
    1. Validate deployment security (fail closed on BIND_PUBLIC misuse —
       the error is shown, content is never rendered).
    2. Anonymous localhost deployments (no credential, not public): return
       True immediately.
    3. Otherwise render login/logout only, and ``st.stop()`` until a valid
       credential is entered.
    """
    try:
        security = validate_deployment_config(
            password_lookup=get_auth_password
        )
    except SecurityConfigError as exc:
        st.error(f"**Deployment security check failed** — {exc}")
        st.stop()
        return False  # unreachable; keeps type checkers honest

    if not security.auth_required:
        return True  # localhost-only, no credential configured

    if _is_authenticated(st.session_state):
        if st.sidebar.button("Logout", key="auth_logout"):
            logout(st.session_state)
            st.rerun()
        st.sidebar.caption(f"Signed in as: {AUTH_USER_LABEL}")
        return True

    # Not authenticated: render ONLY the login form, then stop.
    st.title("Autism Assistant — Sign in")
    st.caption(
        "This deployment requires a credential. Authentication protects "
        "access to the UI; it does not provide per-user data separation."
    )
    with st.form("auth_login_form"):
        candidate = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        expected = get_auth_password()
        if expected is not None and check_password(candidate, expected):
            st.session_state[AUTH_SESSION_KEY] = True
            st.session_state[AUTH_USER_KEY] = AUTH_USER_LABEL
            st.rerun()
        st.error("Incorrect password.")
    st.stop()
    return False  # unreachable
