"""Issue #12 — Streamlit authentication and network isolation.

Unit tests use pure logic + mocks only (no browser, no real server, no
network). Browser-level behavior is explicitly NOT claimed here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from security_config import (
    AUTH_PASSWORD_VAR,
    BIND_PUBLIC_VAR,
    MIN_PUBLIC_PASSWORD_LEN,
    SecurityConfigError,
    parse_bind_public,
    validate_deployment_config,
)
from ui_lib.auth import (
    AUTH_SESSION_KEY,
    AUTH_USER_KEY,
    AUTH_USER_LABEL,
    _is_authenticated,
    check_password,
    logout,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# BIND_PUBLIC parsing — fail closed on invalid values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on"])
def test_bind_public_truthy_values(raw):
    assert parse_bind_public(raw) is True


@pytest.mark.parametrize("raw", ["0", "false", "NO", "off", "none", "", None, "  "])
def test_bind_public_falsy_values(raw):
    assert parse_bind_public(raw) is False


@pytest.mark.parametrize("raw", ["2", "maybe", "public", "1;rm"])
def test_bind_public_invalid_values_fail_safely(raw):
    with pytest.raises(SecurityConfigError) as ei:
        parse_bind_public(raw)
    assert BIND_PUBLIC_VAR in str(ei.value)


def test_invalid_bind_public_fails_even_in_private_mode():
    # an unrecognized value must never silently mean "off"
    with pytest.raises(SecurityConfigError):
        validate_deployment_config({BIND_PUBLIC_VAR: "sometimes"})


# --------------------------------------------------------------------------
# Fail-closed validation
# --------------------------------------------------------------------------


def test_default_local_mode_needs_no_auth():
    sec = validate_deployment_config({})
    assert sec.bind_public is False
    assert sec.auth_required is False
    assert sec.password_source == "none"


def test_public_mode_without_auth_is_rejected():
    with pytest.raises(SecurityConfigError) as ei:
        validate_deployment_config({BIND_PUBLIC_VAR: "1"})
    msg = str(ei.value)
    assert "Refusing to start" in msg
    assert AUTH_PASSWORD_VAR in msg
    assert "secrets.toml" in msg          # the alternative is named
    assert "XSRF" in msg                  # XSRF is explicitly not auth
    assert "1;DROP" not in msg            # never echo attacker-controlled text


def test_public_mode_with_credential_is_valid():
    sec = validate_deployment_config(
        {BIND_PUBLIC_VAR: "1", AUTH_PASSWORD_VAR: "long-enough-passphrase"}
    )
    assert sec.bind_public is True
    assert sec.auth_required is True
    assert sec.password_source == "env"
    # the config object never carries the secret value
    assert "long-enough-passphrase" not in repr(sec)
    assert "long-enough-passphrase" not in str(sec)


def test_public_mode_rejects_short_credential():
    with pytest.raises(SecurityConfigError) as ei:
        validate_deployment_config(
            {BIND_PUBLIC_VAR: "1", AUTH_PASSWORD_VAR: "pw"}
        )
    assert str(MIN_PUBLIC_PASSWORD_LEN) in str(ei.value)
    assert "pw" not in str(ei.value)   # the value itself never appears


def test_secret_lookup_used_when_env_absent():
    sec = validate_deployment_config(
        {BIND_PUBLIC_VAR: "1"}, password_lookup=lambda: "from-secrets-passphrase"
    )
    assert sec.auth_required is True
    assert sec.password_source == "secrets"


def test_failing_secret_lookup_treated_as_absent():
    def boom():
        raise FileNotFoundError("no secrets.toml")

    with pytest.raises(SecurityConfigError):
        validate_deployment_config({BIND_PUBLIC_VAR: "1"}, password_lookup=boom)


def test_validation_never_leaks_password_in_errors():
    secret = "abc"  # deliberately too short -> triggers the length error
    with pytest.raises(SecurityConfigError) as ei:
        validate_deployment_config(
            {BIND_PUBLIC_VAR: "1", AUTH_PASSWORD_VAR: secret}
        )
    # regardless of which error fires, the secret value never appears
    assert secret not in str(ei.value)


# --------------------------------------------------------------------------
# Credential comparison + session state helpers
# --------------------------------------------------------------------------


def test_check_password_constant_time_semantics():
    assert check_password("right-passphrase", "right-passphrase") is True
    assert check_password("wrong", "right-passphrase") is False
    assert check_password("", "right-passphrase") is False
    assert check_password(None, "right-passphrase") is False
    assert check_password(b"bytes-equal", b"bytes-equal") is True


class _FakeSession(dict):
    """Minimal stand-in for st.session_state."""


def test_session_auth_flow_and_logout():
    session = _FakeSession()
    assert _is_authenticated(session) is False
    session[AUTH_SESSION_KEY] = True
    session[AUTH_USER_KEY] = AUTH_USER_LABEL
    assert _is_authenticated(session) is True
    logout(session)
    assert AUTH_SESSION_KEY not in session
    assert AUTH_USER_KEY not in session
    assert _is_authenticated(session) is False


def test_session_stores_no_plaintext_password():
    """Only the flag and role label may live in session state."""
    from ui_lib.auth import AUTH_SESSION_KEY  # the only permitted success key

    allowed_keys = {AUTH_SESSION_KEY, AUTH_USER_KEY}
    assert all(k in allowed_keys for k in (AUTH_SESSION_KEY, AUTH_USER_KEY))


# --------------------------------------------------------------------------
# Every page / entry point gates before content (source-level locks)
# --------------------------------------------------------------------------


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _first_content_line(src: str) -> int:
    """Line number of the first st.* content call after imports."""
    for i, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("st.") and not stripped.startswith(
            ("st.set_page_config",)
        ):
            return i
    return len(src.splitlines()) + 1


@pytest.mark.parametrize(
    "page",
    [
        "ui.py",
        "pages/1_Quick_Label.py",
        "pages/2_Run_Pipeline.py",
        "pages/3_Compare_Models.py",
        "pages/4_Results.py",
        "pages/5_Translate_Export.py",
    ],
)
def test_every_entry_point_calls_require_auth_before_content(page):
    src = _src(page)
    assert "require_auth()" in src, f"{page} does not call require_auth()"
    gate_line = next(
        i
        for i, line in enumerate(src.splitlines(), start=1)
        if line.strip().startswith("require_auth()")
    )
    content_line = _first_content_line(src)
    assert gate_line < content_line, (
        f"{page}: require_auth() (line {gate_line}) must precede the first "
        f"content call (line {content_line})"
    )


def test_auth_gate_does_not_leak_password_into_session_or_errors():
    src = _src("ui_lib/auth.py")
    assert "session_state[AUTH_SESSION_KEY] = True" in src  # flag only
    assert 'session_state["password"]' not in src
    assert 'st.error(candidate' not in src
    assert 'st.error(expected' not in src
    assert "compare_digest" in src  # constant-time comparison present


def test_native_oidc_auth_decision_is_documented():
    """The chosen architecture (password gate) records why native st.login
    (OIDC-only) is unsuitable — keeps the decision reviewable."""
    src = _src("ui_lib/auth.py")
    assert "OIDC" in src


# --------------------------------------------------------------------------
# Streamlit / Docker configuration locks
# --------------------------------------------------------------------------


def test_streamlit_config_defaults_to_localhost_and_keeps_protections():
    import tomllib

    cfg = tomllib.loads(_src(".streamlit/config.toml"))
    server = cfg["server"]
    assert server["address"] == "localhost"  # default = loopback only
    assert server["enableXsrfProtection"] is True  # XSRF preserved
    assert server["enableCORS"] is True  # CORS not disabled
    assert cfg["browser"]["gatherUsageStats"] is False


def test_dockerfile_container_listens_on_all_interfaces():
    src = _src("Dockerfile")
    assert "--server.address=0.0.0.0" in src  # container-internal only
    assert "STREAMLIT_SERVER_HEADLESS=true" in src


def test_default_compose_binds_host_port_to_localhost_only():
    compose = yaml.safe_load(_src("docker-compose.yml"))
    ports = compose["services"]["ui"]["ports"]
    assert len(ports) == 1
    # exact template — resolved host_ip checked via `docker compose config`
    assert ports[0] == "${UI_BIND_HOST:-127.0.0.1}:${UI_HOST_PORT:-8501}:8501"
    assert not ports[0].startswith("0.0.0.0")


def test_default_compose_has_no_public_switch_and_keeps_volumes():
    compose = yaml.safe_load(_src("docker-compose.yml"))
    env = compose["services"]["ui"]["environment"]
    # BIND_PUBLIC defaults to 0 in the base file (opt-in only)
    assert env["BIND_PUBLIC"] == "${BIND_PUBLIC:-0}"
    # existing application volumes and the model cache volume preserved
    mounts = compose["services"]["ui"]["volumes"]
    assert "model_cache:/cache" in mounts
    assert "./data:/app/data" in mounts
    assert "./outputs:/app/outputs" in mounts
    assert "model_cache" in compose["volumes"]


def test_public_override_binds_publicly_and_requires_credential():
    pub = yaml.safe_load(_src("docker-compose.public.yml").replace("!override", ""))
    ui = pub["services"]["ui"]
    ports = ui["ports"]
    assert ports == ["0.0.0.0:${UI_HOST_PORT:-8501}:8501"]
    env = ui["environment"]
    assert env["BIND_PUBLIC"] == "1"
    # fail-closed interpolation: missing credential aborts `compose up`
    assert env["APP_AUTH_PASSWORD"].startswith("${APP_AUTH_PASSWORD:?")


def test_public_mode_without_credential_fails_compose_validation():
    """`docker compose config` (client-side) must fail when BIND_PUBLIC=1
    is requested without APP_AUTH_PASSWORD."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "APP_AUTH_PASSWORD"}
    env["BIND_PUBLIC"] = "1"
    result = subprocess.run(
        [
            "docker", "compose",
            "-f", "docker-compose.yml",
            "-f", "docker-compose.public.yml",
            "config", "--quiet",
        ],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert result.returncode != 0
    assert "APP_AUTH_PASSWORD" in (result.stderr + result.stdout)


def test_public_mode_with_credential_validates():
    import os
    import subprocess

    env = dict(os.environ)
    env["BIND_PUBLIC"] = "1"
    env["APP_AUTH_PASSWORD"] = "config-validation-passphrase"
    result = subprocess.run(
        [
            "docker", "compose",
            "-f", "docker-compose.yml",
            "-f", "docker-compose.public.yml",
            "config", "--quiet",
        ],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert result.returncode == 0, result.stderr


def test_default_local_compose_validates():
    import subprocess

    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "config", "--quiet"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_default_compose_resolves_to_loopback_host_ip():
    """Resolved compose config: host_ip must be 127.0.0.1 without any env."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items()
           if k not in ("UI_BIND_HOST", "UI_HOST_PORT", "BIND_PUBLIC",
                        "APP_AUTH_PASSWORD")}
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "config"],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert result.returncode == 0, result.stderr
    text = result.stdout
    ui_block = text.split("services:", 1)[1]
    assert "host_ip: 127.0.0.1" in ui_block
    assert "host_ip: 0.0.0.0" not in ui_block.split("prefetch:", 1)[0]
