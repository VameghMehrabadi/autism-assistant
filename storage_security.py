"""حالت امنیت ذخیره‌سازی (Issue #11) — رمزنگاری at-rest با محافظت سطح OS.

Storage-security mode for data and output encryption at rest.

**Architectural decision (recorded):** protection is provided by
**operator-managed encrypted host storage** (BitLocker / LUKS / FileVault /
an encrypted volume holding ``data/``, ``outputs/`` and the caches). This
covers every file the application writes — datasets, labels, translation
outputs, QA review files, gap/comparison reports, charts, logs, temp/backup
files, and the embedding cache — because they all live under
``config.DATA_DIR``, ``config.OUTPUT_DIR`` or ``config.CACHE_DIR`` and the
OS encrypts beneath the filesystem transparently.

Application-level crypto was evaluated and deliberately **not** implemented:
covering every write path (incremental JSONL checkpoints with resume,
torn-output repair, numpy caches, charts, sanitized logs) with authenticated
encryption safely would be a storage-architecture rewrite, and a partial
layer would falsely imply PHI-grade protection. No fake crypto layer is
added to satisfy tests (the issue explicitly forbids that for OS-only
protection).

**Consequence:** the application CANNOT verify disk encryption — there is no
reliable cross-platform way to do so, and none is attempted. In
``patient-data`` mode the operator *declares* the protection mechanism; that
declaration is an operator-declared prerequisite, never an
application-verified property. Modes:

- ``research`` (default): current workflow; UI/CLI warn that files may be
  stored unencrypted and that the mode is unsuitable for PHI.
- ``patient-data``: requires ``STORAGE_PROTECTION_DECLARATION`` naming the
  mechanism; fails closed when missing or invalid; never silently
  downgrades.

This module is pure (no streamlit import) so CLI paths and tests can use it.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

MODE_VAR = "DATA_STORAGE_MODE"
DECLARATION_VAR = "STORAGE_PROTECTION_DECLARATION"

MODE_RESEARCH = "research"
MODE_PATIENT_DATA = "patient-data"

# Operator-declarable protection mechanisms. This is a *declaration*, not a
# detection result — the application never probes disk state.
ACCEPTED_DECLARATIONS = frozenset(
    {"bitlocker", "luks", "filevault", "encrypted-volume"}
)

PHI_NOT_SUITABLE_NOTE = (
    "Unsuitable for real patient-identifiable data or PHI unless the "
    "operator separately establishes and documents equivalent storage "
    "protection."
)


class StorageSecurityError(RuntimeError):
    """Invalid or incomplete storage-security configuration (fail closed)."""


@dataclass(frozen=True)
class StorageSecurity:
    """Validated storage-security configuration (no secret material)."""

    mode: str                      # MODE_RESEARCH | MODE_PATIENT_DATA
    declaration: str | None        # declared mechanism, patient-data mode
    declaration_source: str        # "env" | "none"

    @property
    def application_verified(self) -> bool:
        """Always False: at-rest protection here is operator-managed."""
        return False


def parse_storage_mode(raw: str | None) -> str:
    """Parse ``DATA_STORAGE_MODE``; invalid values fail closed."""
    value = (raw or "").strip().lower()
    if value in ("", MODE_RESEARCH):
        return MODE_RESEARCH
    if value == MODE_PATIENT_DATA:
        return MODE_PATIENT_DATA
    raise StorageSecurityError(
        f"Invalid {MODE_VAR}={raw!r}. Valid values: {MODE_RESEARCH!r} "
        f"(default — local research workflow, files may be stored "
        f"unencrypted) or {MODE_PATIENT_DATA!r} (requires "
        f"{DECLARATION_VAR} naming the encrypted-storage mechanism). "
        "Refusing to start with an unrecognized value."
    )


def validate_storage_security(
    env: Mapping[str, str] | None = None,
) -> StorageSecurity:
    """Validate the storage-security configuration; fail closed on problems.

    ``patient-data`` mode without a valid
    ``STORAGE_PROTECTION_DECLARATION`` raises — it never silently continues
    or downgrades to research mode.
    """
    env = os.environ if env is None else env
    mode = parse_storage_mode(env.get(MODE_VAR))

    declaration: str | None = None
    declaration_source = "none"
    raw_declaration = (env.get(DECLARATION_VAR) or "").strip().lower()
    if raw_declaration:
        if raw_declaration not in ACCEPTED_DECLARATIONS:
            raise StorageSecurityError(
                f"Invalid {DECLARATION_VAR}={env.get(DECLARATION_VAR)!r}. "
                f"Accepted declarations: {sorted(ACCEPTED_DECLARATIONS)}. "
                "The declaration states which mechanism protects the "
                "storage volume (e.g. BitLocker/LUKS/FileVault); the "
                "application cannot verify it."
            )
        declaration = raw_declaration
        declaration_source = "env"

    if mode == MODE_PATIENT_DATA and declaration is None:
        raise StorageSecurityError(
            f"Refusing to start: {MODE_VAR}={MODE_PATIENT_DATA} requires "
            f"{DECLARATION_VAR} naming the protection mechanism "
            f"({sorted(ACCEPTED_DECLARATIONS)}). Patient-data mode without "
            "verified-at-rest storage protection must not run silently — "
            "the declaration is an operator-declared prerequisite the "
            "application cannot verify."
        )
    return StorageSecurity(
        mode=mode,
        declaration=declaration,
        declaration_source=declaration_source,
    )


def protected_paths() -> dict[str, str]:
    """Inventory of storage locations that MUST sit on the protected volume.

    Every file the pipeline writes lands under one of these roots (verified
    against the code's write operations); the operator's encryption must
    cover the physical volumes backing them.
    """
    import config

    return {
        "data (raw datasets, exports, translation outputs, QA review files)":
            str(config.DATA_DIR),
        "outputs (labeled.csv, scores, gap/comparison reports, charts)":
            str(config.OUTPUT_DIR),
        "logs (privacy-sanitized JSON lines)":
            str(config.OUTPUT_DIR / "logs"),
        "embedding cache (vectors keyed by hash — no text)":
            str(config.EMBED_CACHE_DIR),
        "model/dataset caches (model weights are public; HF datasets cache "
        "follows HF_HOME/HF_HUB_CACHE when set)":
            str(config.CACHE_DIR),
    }


def security_summary(security: StorageSecurity) -> str:
    """One-line effective-mode summary for CLI output (no secrets involved)."""
    if security.mode == MODE_PATIENT_DATA:
        return (
            f"[storage] data security mode: {MODE_PATIENT_DATA} "
            f"(operator-declared protection: {security.declaration} — "
            "declared, NOT verified by this application; requires the "
            "volumes behind data/, outputs/ and the caches to be encrypted)"
        )
    return (
        f"[storage] data security mode: {MODE_RESEARCH} (default) — files "
        "are stored unencrypted unless the operator protects the underlying "
        f"volume. {PHI_NOT_SUITABLE_NOTE}"
    )


def warning_message(security: StorageSecurity) -> str:
    """Detailed UI/CLI warning text for the effective mode."""
    if security.mode == MODE_PATIENT_DATA:
        return (
            f"**Storage security mode: patient-data** — operator-declared "
            f"protection mechanism: `{security.declaration}`. This is an "
            "OPERATOR DECLARATION, not an application-verified property: "
            "the application cannot detect disk encryption. The volumes "
            "behind `data/`, `outputs/`, the embedding cache and the "
            "model/dataset caches must actually be encrypted (see "
            "reports/PRIVACY_AND_SECURITY.md). Locking/unlocking the "
            "device, OS-level key escrow and backups are the operator's "
            "responsibility. This is still not HIPAA compliance or "
            "clinical certification, and there is no multi-user data "
            "isolation."
        )
    return (
        "**Storage security mode: research (default)** — files under "
        "`data/`, `outputs/` and the caches are written **unencrypted** "
        "unless the operator protects the underlying volume with "
        "OS-level encryption (BitLocker/LUKS/FileVault). Git exclusion "
        "rules are not encryption. "
        f"{PHI_NOT_SUITABLE_NOTE}"
    )


def storage_security_exit_code() -> int:
    """CLI helper: validate + announce the effective mode.

    Returns 0 after printing the one-line summary, or 2 after printing an
    actionable error — callers ``return`` this value to fail closed.
    """
    try:
        security = validate_storage_security()
    except StorageSecurityError as exc:
        print(f"[storage] CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    print(security_summary(security))
    return 0
