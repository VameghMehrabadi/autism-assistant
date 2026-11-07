"""اعتبارسنجی پیکربندی امنیتی استقرار (Issue #12).

Centralized deployment-security configuration for the Streamlit UI:

- ``BIND_PUBLIC`` — network-exposure switch. Absent/falsy (default) means
  **localhost-only** deployment with no authentication requirement. ``1`` /
  ``true`` / ``yes`` / ``on`` explicitly request public exposure and REQUIRE
  a real credential. Unrecognized values are invalid and fail closed.
- ``APP_AUTH_PASSWORD`` — the single shared UI credential (never logged,
  never echoed, never stored). May alternatively come from Streamlit secrets
  (``.streamlit/secrets.toml`` → ``[auth] password``) via the injected
  ``password_lookup`` hook.

Fail-closed rules (never silently downgrade):
- ``BIND_PUBLIC=1`` without any credential → refuse (SecurityConfigError).
- ``BIND_PUBLIC=1`` with a credential shorter than ``MIN_PUBLIC_PASSWORD_LEN``
  → refuse (too weak for public exposure).
- Invalid ``BIND_PUBLIC`` values → refuse in every mode.
- XSRF is NOT authentication and a reverse-proxy claim is NOT proof of
  authentication; neither relaxes these rules.

This module deliberately imports neither streamlit nor network libraries so
it stays pure, testable, and usable from CLI startup paths.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

BIND_PUBLIC_VAR = "BIND_PUBLIC"
AUTH_PASSWORD_VAR = "APP_AUTH_PASSWORD"
AUTH_SECRETS_SECTION = "auth"
AUTH_SECRETS_KEY = "password"

# Values accepted for BIND_PUBLIC (case-insensitive). Anything else that is
# non-empty is invalid — fail closed rather than guessing.
BIND_PUBLIC_TRUTHY = frozenset({"1", "true", "yes", "on"})
BIND_PUBLIC_FALSY = frozenset({"0", "false", "no", "off", "none", ""})

MIN_PUBLIC_PASSWORD_LEN = 8


class SecurityConfigError(RuntimeError):
    """Invalid or incomplete deployment-security configuration."""


@dataclass(frozen=True)
class DeploymentSecurity:
    """Result of validation — carries NO secret material by design."""

    bind_public: bool
    auth_required: bool
    password_source: str  # "env" | "secrets" | "none"


def parse_bind_public(raw: str | None) -> bool:
    """Parse ``BIND_PUBLIC``; raise on values outside the known set."""
    value = (raw or "").strip().lower()
    if value in BIND_PUBLIC_FALSY:
        return False
    if value in BIND_PUBLIC_TRUTHY:
        return True
    raise SecurityConfigError(
        f"Invalid {BIND_PUBLIC_VAR}={raw!r}. Valid values: "
        f"{sorted(BIND_PUBLIC_TRUTHY)} (public exposure, requires "
        f"{AUTH_PASSWORD_VAR}) or {sorted(BIND_PUBLIC_FALSY)} (default, "
        "localhost-only). Refusing to start with an unrecognized value."
    )


def validate_deployment_config(
    env: Mapping[str, str] | None = None,
    *,
    password_lookup: Callable[[], object] | None = None,
) -> DeploymentSecurity:
    """Validate the deployment-security settings; fail closed on problems.

    ``password_lookup`` is an optional hook returning a password from an
    alternative source (e.g. Streamlit secrets) or ``None``. The password
    value itself is never included in the result or in any error message.
    """
    env = os.environ if env is None else env
    bind_public = parse_bind_public(env.get(BIND_PUBLIC_VAR))

    password: str | None = None
    password_source = "none"
    raw_password = env.get(AUTH_PASSWORD_VAR, "")
    if raw_password.strip():
        password = raw_password
        password_source = "env"
    elif password_lookup is not None:
        try:
            looked_up = password_lookup()
        except Exception:
            looked_up = None  # missing/corrupt secrets — treated as absent
        if looked_up and str(looked_up).strip():
            password = str(looked_up)
            password_source = "secrets"

    auth_required = bind_public or password is not None

    if bind_public and password is None:
        raise SecurityConfigError(
            f"Refusing to start: {BIND_PUBLIC_VAR}=1 (public network "
            f"exposure) requires a real credential — set "
            f"{AUTH_PASSWORD_VAR} or add "
            f"[{AUTH_SECRETS_SECTION}] {AUTH_SECRETS_KEY} to "
            ".streamlit/secrets.toml. Unauthenticated public exposure is "
            "never started (XSRF protection is not authentication)."
        )
    if bind_public and password is not None and len(password) < MIN_PUBLIC_PASSWORD_LEN:
        raise SecurityConfigError(
            f"Refusing to start: the {AUTH_PASSWORD_VAR} credential is "
            f"shorter than {MIN_PUBLIC_PASSWORD_LEN} characters — too weak "
            f"for public exposure ({BIND_PUBLIC_VAR}=1). Choose a longer "
            "passphrase; the value itself is not logged."
        )
    return DeploymentSecurity(
        bind_public=bind_public,
        auth_required=auth_required,
        password_source=password_source,
    )
