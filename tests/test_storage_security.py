"""Issue #11 — storage security modes (encrypted-host-storage architecture).

Synthetic data only; no real patient conversations, labels, or identifiable
information. IMPORTANT: unit tests here validate CONFIGURATION and PATH
MAPPING only — they cannot and do not prove that any physical disk is
encrypted. At-rest protection is an operator-declared OS-level prerequisite
by design (see storage_security.py and reports/PRIVACY_AND_SECURITY.md).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import config
from storage_security import (
    ACCEPTED_DECLARATIONS,
    DECLARATION_VAR,
    MODE_PATIENT_DATA,
    MODE_RESEARCH,
    MODE_VAR,
    PHI_NOT_SUITABLE_NOTE,
    StorageSecurityError,
    protected_paths,
    security_summary,
    storage_security_exit_code,
    validate_storage_security,
    warning_message,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Mode parsing — secure default, fail closed on invalid values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", [None, "", "  ", "research", "RESEARCH", "Research"]
)
def test_research_mode_is_the_default(raw):
    assert validate_storage_security({MODE_VAR: raw}).mode == MODE_RESEARCH


def test_patient_data_mode_accepted():
    sec = validate_storage_security(
        {MODE_VAR: "patient-data", DECLARATION_VAR: "bitlocker"}
    )
    assert sec.mode == MODE_PATIENT_DATA
    assert sec.declaration == "bitlocker"
    assert sec.declaration_source == "env"


@pytest.mark.parametrize("raw", ["patient", "PUBLIC", "2", "patient data"])
def test_invalid_mode_values_fail_closed(raw):
    with pytest.raises(StorageSecurityError) as ei:
        validate_storage_security({MODE_VAR: raw})
    assert MODE_VAR in str(ei.value)
    assert MODE_PATIENT_DATA in str(ei.value)
    assert MODE_RESEARCH in str(ei.value)


# --------------------------------------------------------------------------
# Patient-data mode — operator declaration required, fail closed
# --------------------------------------------------------------------------


def test_patient_data_without_declaration_is_refused():
    with pytest.raises(StorageSecurityError) as ei:
        validate_storage_security({MODE_VAR: "patient-data"})
    msg = str(ei.value)
    assert "Refusing to start" in msg
    assert DECLARATION_VAR in msg
    assert "operator-declared prerequisite" in msg
    assert "cannot verify" in msg


def test_patient_data_never_silently_downgrades():
    """The error must not be a warning — patient-data mode cannot continue."""
    with pytest.raises(StorageSecurityError):
        validate_storage_security({MODE_VAR: "patient-data"})


@pytest.mark.parametrize("decl", sorted(ACCEPTED_DECLARATIONS))
def test_accepted_declarations(decl):
    sec = validate_storage_security(
        {MODE_VAR: "patient-data", DECLARATION_VAR: decl}
    )
    assert sec.declaration == decl


def test_declaration_case_insensitive():
    sec = validate_storage_security(
        {MODE_VAR: "patient-data", DECLARATION_VAR: "BitLocker"}
    )
    assert sec.declaration == "bitlocker"


def test_invalid_declaration_rejected_in_any_mode():
    with pytest.raises(StorageSecurityError) as ei:
        validate_storage_security({DECLARATION_VAR: "trust-me"})
    # invalid declarations are not secrets — echoed for debuggability
    # (consistent with invalid BIND_PUBLIC handling)
    assert DECLARATION_VAR in str(ei.value)
    assert sorted(ACCEPTED_DECLARATIONS)[0] in str(ei.value)
    assert "cannot verify" in str(ei.value)


def test_research_mode_with_declaration_is_valid():
    sec = validate_storage_security({DECLARATION_VAR: "luks"})
    assert sec.mode == MODE_RESEARCH
    assert sec.declaration == "luks"


def test_protection_is_never_application_verified():
    """Design invariant: the app cannot verify OS-level encryption."""
    for env in ({}, {MODE_VAR: "patient-data", DECLARATION_VAR: "luks"}):
        assert validate_storage_security(env).application_verified is False


# --------------------------------------------------------------------------
# Storage-path inventory — coverage of every write surface
# --------------------------------------------------------------------------


def test_protected_paths_cover_all_write_roots():
    paths = protected_paths()
    values = list(paths.values())
    # every managed write root is represented
    assert str(config.DATA_DIR) in values
    assert str(config.OUTPUT_DIR) in values
    assert str(config.OUTPUT_DIR / "logs") in values
    assert str(config.EMBED_CACHE_DIR) in values
    assert str(config.CACHE_DIR) in values
    # and every listed path sits under one of the three managed roots
    managed = [config.DATA_DIR, config.OUTPUT_DIR, config.CACHE_DIR]
    for value in values:
        p = Path(value)
        assert any(
            p == root or root in p.parents for root in managed
        ), f"path outside managed roots: {value}"


def test_path_inventory_documents_hf_cache_dependency():
    """The HF datasets cache follows HF_HOME — documented, not hidden."""
    assert "HF_HOME" in list(protected_paths())[-1]


# --------------------------------------------------------------------------
# Warning / summary text — honest labeling
# --------------------------------------------------------------------------


def test_research_warning_labels_unencrypted_and_phi_unsuitability():
    msg = warning_message(
        validate_storage_security({MODE_VAR: "research"})
    )
    assert "unencrypted" in msg
    assert PHI_NOT_SUITABLE_NOTE in msg
    assert "BitLocker" in msg and "LUKS" in msg and "FileVault" in msg
    assert "Git exclusion" in msg  # gitignore != encryption


def test_patient_data_warning_says_declared_not_verified():
    msg = warning_message(
        validate_storage_security(
            {MODE_VAR: "patient-data", DECLARATION_VAR: "luks"}
        )
    )
    assert "OPERATOR DECLARATION" in msg
    assert "not an application-verified property" in msg
    assert "cannot detect disk encryption" in msg
    assert "not HIPAA compliance" in msg
    assert "no multi-user data isolation" in msg
    assert "luks" in msg


def test_summary_lines_are_single_line_and_prefixed():
    for env in (
        {},
        {MODE_VAR: "patient-data", DECLARATION_VAR: "filevault"},
    ):
        line = security_summary(validate_storage_security(env))
        assert line.startswith("[storage]")
        assert "\n" not in line
        assert "data security mode" in line


# --------------------------------------------------------------------------
# CLI helper — announce + fail closed with exit code 2
# --------------------------------------------------------------------------


def test_exit_code_zero_and_announcement_in_research_mode(capsys):
    rc = storage_security_exit_code()
    assert rc == 0
    out = capsys.readouterr().out
    assert "[storage] data security mode: research" in out
    assert "unencrypted" in out


def test_exit_code_zero_and_announcement_in_patient_mode(
    capsys, monkeypatch
):
    monkeypatch.setenv(MODE_VAR, "patient-data")
    monkeypatch.setenv(DECLARATION_VAR, "bitlocker")
    rc = storage_security_exit_code()
    assert rc == 0
    out = capsys.readouterr().out
    assert "operator-declared protection: bitlocker" in out
    assert "NOT verified" in out


def test_exit_code_two_on_invalid_config(capsys, monkeypatch):
    monkeypatch.setenv(MODE_VAR, "bogus")
    rc = storage_security_exit_code()
    assert rc == 2
    err = capsys.readouterr().err
    assert "CONFIG ERROR" in err
    assert "bogus" in err  # mode value is not a secret — echoed is fine


# --------------------------------------------------------------------------
# Integration locks — every entry point surfaces the mode
# --------------------------------------------------------------------------


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "rel",
    [
        "main.py",
        "compare_runs.py",
        "export_for_translation.py",
        "translate_dataset.py",
        "translation_qa.py",
    ],
)
def test_every_cli_entry_point_announces_storage_mode(rel):
    src = _src(rel)
    assert "storage_security_exit_code()" in src, f"{rel} bypasses mode check"


def test_ui_home_shows_full_banner_and_pages_show_compact_caption():
    home = _src("ui.py")
    assert "render_storage_security_banner()" in home
    for rel in (
        "pages/1_Quick_Label.py",
        "pages/2_Run_Pipeline.py",
        "pages/3_Compare_Models.py",
        "pages/4_Results.py",
        "pages/5_Translate_Export.py",
    ):
        src = _src(rel)
        assert "render_storage_security_banner(compact=True)" in src, rel


def test_banner_fails_closed_on_invalid_configuration():
    src = _src("ui_lib/widgets.py")
    banner = src.split("def render_storage_security_banner")[1]
    assert "StorageSecurityError" in banner
    assert "st.stop()" in banner  # no content on invalid config


# --------------------------------------------------------------------------
# Docker — mode passed consistently into ui and pipeline services
# --------------------------------------------------------------------------


def test_compose_passes_storage_mode_to_ui_and_pipeline():
    compose = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    for svc in ("ui", "pipeline"):
        env = compose["services"][svc]["environment"]
        assert env["DATA_STORAGE_MODE"] == "${DATA_STORAGE_MODE:-}"
        assert env["STORAGE_PROTECTION_DECLARATION"] == (
            "${STORAGE_PROTECTION_DECLARATION:-}"
        )


def test_compose_has_no_plaintext_key_material():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # only interpolation references — never literal credential values
    assert "APP_AUTH_PASSWORD: '" not in text
    assert "APP_AUTH_PASSWORD=\"" not in text
    assert "BEGIN PRIVATE KEY" not in text
