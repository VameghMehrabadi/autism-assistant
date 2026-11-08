"""نگهبان مجوز داده (Issue #13) — مرز انتشار عمومی داده‌ی مشتق MentalChat16K.

Single centralized licensing guard used by **every** export/publication path
that copies MentalChat16K-derived records (raw export, Persian translation,
QA review files). One policy, one message, no per-script inconsistencies.

Policy (assessed 2026-11-07 — see ``reports/PRIVACY_AND_SECURITY.md``):
- The upstream dataset carries an ``MIT`` license tag on Hugging Face and an
  MIT LICENSE file in the official repository, but neither source scopes the
  license for the interview-derived component (paraphrased PISCES clinical-
  trial transcripts), and no source explicitly authorizes translated or
  derivative redistribution. Status: **unclear**.
- Private/local processing (anywhere inside the project workspace:
  ``data/``, ``outputs/``, cache, runs) is never blocked.
- A **public distribution destination** (any path outside the project
  workspace) receives MentalChat16K-derived records only when the caller
  explicitly passes ``acknowledge_license=True`` (CLI: ``--acknowledge-license``).
  The acknowledgment records the user's acceptance of the unresolved risk —
  it does NOT grant legal permission.

The private/public boundary uses the project root (``config.ROOT``) from the
actual export architecture — not ad-hoc directory-name heuristics.
"""
from __future__ import annotations

from pathlib import Path

import config

DATASET_NAME = "ShenLab/MentalChat16K"
DATASET_URL = "https://huggingface.co/datasets/ShenLab/MentalChat16K"
UPSTREAM_REPO_URL = "https://github.com/PennShenLab/MentalChat16K"
UPSTREAM_REVISION = "5f60cd380cfc58f0f12d44892bed41ee3670a70a"
ASSESSMENT_REPORT = "reports/PRIVACY_AND_SECURITY.md"
OVERRIDE_FLAG = "--acknowledge-license"

# Licensing status of upstream-derived data. One of:
#   "confirmed" — authoritative sources establish the needed permission
#   "unclear"   — sources do not establish it (current state, 2026-11-07)
#   "not_found" — no applicable statement found
# Flip to "confirmed" ONLY with a dated, sourced re-assessment in
# reports/PRIVACY_AND_SECURITY.md (e.g. upstream publishes an explicit data
# license covering derivatives, or written permission is obtained).
LICENSE_STATUS = "unclear"


class LicenseGuardError(PermissionError):
    """Refusal to copy restricted/unclear-license data to a public destination."""


def override_flag_help() -> str:
    """argparse help text for the shared ``--acknowledge-license`` flag."""
    return (
        "explicitly acknowledge the UNRESOLVED licensing status of "
        "MentalChat16K-derived data when exporting OUTSIDE the project "
        "workspace (records your acknowledgment of the risk; it does NOT "
        f"grant legal permission — see {ASSESSMENT_REPORT})"
    )


def _private_roots() -> tuple[Path, ...]:
    """The project's actual managed workspace roots (private/local zone).

    All default export destinations (``data/``, ``outputs/``, caches) live
    under one of these; listing them explicitly follows the real export
    architecture instead of ad-hoc directory-name heuristics. In production
    they all sit under ``config.ROOT``; tests may re-anchor ``ROOT`` to a
    per-test workspace while the static dirs stay real.
    """
    return (
        config.ROOT,
        config.DATA_DIR,
        config.OUTPUT_DIR,
        config.CACHE_DIR,
    )


def classify_destination(path: str | Path) -> str:
    """``"private"`` inside the project workspace, ``"public"`` outside it.

    The workspace (project root plus the managed ``data/``/``outputs/``/
    cache roots) is the private/local processing zone used by every default
    export path. Any destination that resolves outside all of them is
    treated as a distribution location (another folder, drive, share, or
    repository checkout).
    """
    target = Path(path).resolve()
    for root in _private_roots():
        try:
            target.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return "private"
    return "public"


def refusal_message(out_path: str | Path, kind: str) -> str:
    """Why the export was refused, and what the options are."""
    return (
        f"[license-guard] REFUSED: not writing {kind} to the public "
        f"destination '{out_path}'.\n"
        f"  Upstream dataset: {DATASET_NAME} ({DATASET_URL}) — revision "
        f"{UPSTREAM_REVISION}.\n"
        f"  Licensing status: '{LICENSE_STATUS}'. The upstream MIT tag does "
        "not, by itself, establish permission to redistribute derivatives "
        "(the dataset mixes synthetic conversations with paraphrased "
        "clinical-trial interview transcripts; no source explicitly "
        "authorizes translated/derivative redistribution).\n"
        f"  Assessment: {ASSESSMENT_REPORT}.\n"
        f"  Options: (a) keep the export inside the project workspace "
        "(private/local processing is never blocked — e.g. "
        f"{config.DATA_DIR}), or (b) re-run with {OVERRIDE_FLAG} to "
        "acknowledge the unresolved licensing risk explicitly."
    )


def acknowledgment_warning(kind: str, out_path: str | Path) -> str:
    """Prominent warning printed when the user overrides the guard.

    Deliberately does NOT claim the flag grants legal permission.
    """
    return (
        f"!!! LICENSE ACKNOWLEDGMENT — {kind} -> '{out_path}' !!!\n"
        f"  You are acknowledging, on your own responsibility, that the "
        f"licensing status of {DATASET_NAME}-derived data is UNRESOLVED "
        f"(status '{LICENSE_STATUS}', assessment: {ASSESSMENT_REPORT}).\n"
        f"  {OVERRIDE_FLAG} records your acknowledgment of this risk. "
        "It does NOT grant legal permission and does not clear the data "
        "for redistribution or commercial use. You are responsible for "
        "obtaining any required permission before publishing or sharing "
        "these records."
    )


def assert_export_allowed(
    out_path: str | Path,
    *,
    kind: str = "MentalChat16K-derived records",
    acknowledge_license: bool = False,
) -> str:
    """Central guard for every export/publication path.

    Returns the destination classification ("private" | "public"). Raises
    ``LicenseGuardError`` for a public destination while the license status
    is unclear, unless ``acknowledge_license`` is set (then the prominent
    acknowledgment warning is printed to stdout and the export proceeds).
    Private destinations always pass — ordinary local processing stays
    functional.
    """
    dest = classify_destination(out_path)
    if dest == "private" or LICENSE_STATUS == "confirmed":
        return dest
    if acknowledge_license:
        print(acknowledgment_warning(kind, out_path))
        return dest
    raise LicenseGuardError(refusal_message(out_path, kind))
