"""سیاست نگهداری و پاک‌سازی امن خروجی اجراها (Issue #10).

Safe, explicit, whitelist-based cleanup of run-generated artifacts that may
contain raw conversation text.

**Deletion model (deliberately conservative):**

- A cleanup *location* addresses one clearly-defined artifact set:
  ``default`` (the six standard single-run files directly in ``outputs/``),
  ``<run-dir>`` (a run directory under ``outputs/``),
  ``comparison/<run-id>`` (one per-run directory of a comparison), or
  ``translation-artifacts`` (the translation exports/sidecar/error/QA files
  in ``data/``).
- Only files whose **exact file name** is on the recognized-artifact
  whitelist are eligible. Everything else — original datasets, custom user
  files, unknown files inside a run directory, shared comparison summaries,
  logs, caches, model weights, gold data, source code — is **skipped**, not
  deleted. There is no recursive directory deletion anywhere; an emptied run
  directory is removed with a containment-checked ``rmdir`` only.
- Every candidate's **real path** must resolve inside the configured root
  (``OUTPUT_DIR`` / ``DATA_DIR``): absolute escapes, ``..`` traversal and
  symlink escapes are refused or skipped with a reason.
- ``execute_run_cleanup(confirm=False)`` is a pure preview; nothing is
  deleted without ``confirm=True`` (CLI ``--confirm`` / UI checkbox are the
  explicit confirmations).
- Deletion audit records contain **metadata only** (timestamp, relative
  location, counts, mode) — never conversation text and never absolute
  paths that could expose usernames.

The default retention period (30 days for temporary generated raw run
artifacts) is a **documented policy default**, not an enforced mechanism:
nothing in this module deletes anything automatically, on a schedule, or in
the background.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config

# Recognized run-generated artifacts (exact file names). Anything not on a
# whitelist is never a deletion candidate.
RUN_ARTIFACT_NAMES = frozenset(
    {
        "labeled.csv",
        "scores.npy",
        "gap_analysis.md",
        "gap_analysis.json",
        "gap_distribution.png",
        "run_meta.json",
    }
)
DEFAULT_RUN_EXTRA_NAMES = frozenset({"split_manifest.json"})

# data/ translation workflow artifacts (generated derivatives — never the
# original datasets, which are loaded from the Hugging Face Hub and never
# written into data/ by this project).
TRANSLATION_ARTIFACT_NAMES = (
    "mentalchat16k_en_for_translation.jsonl",
    "mentalchat16k_fa.jsonl",
    "translation_errors.jsonl",
    "translation_review.jsonl",
)

# Exclusive sidecar pairing used by export_for_translation.write_export:
# out_path.with_suffix(".export_meta.json") — one sidecar per export file.
SIDECAR_SUFFIX = ".export_meta.json"

DEFAULT_LOCATION = "default"
TRANSLATION_LOCATION = "translation-artifacts"
COMPARISON_DIRNAME = "comparison"

AUDIT_LOG_NAME = "deletion_audit.jsonl"


class RetentionError(ValueError):
    """Refused cleanup request (bad location, traversal, escape, ambiguity)."""


@dataclass(frozen=True)
class CleanupCandidate:
    """One file that would be deleted (display path is root-relative)."""

    display: str          # root-relative path — safe for logs/UI
    path: Path            # absolute path (internal use)
    kind: str             # "run_artifact" | "sidecar"
    size_bytes: int


@dataclass(frozen=True)
class CleanupPlan:
    """Exact preview of a cleanup operation."""

    location: str
    root: Path                                  # OUTPUT_DIR or DATA_DIR
    candidates: tuple[CleanupCandidate, ...]
    skipped: tuple[tuple[str, str], ...]        # (display, reason)
    empty_dirs_after: tuple[Path, ...]          # run dirs that become empty

    @property
    def total_bytes(self) -> int:
        return sum(c.size_bytes for c in self.candidates)


def _resolved_base(root: Path) -> Path:
    return Path(root).resolve()


def _contained(real_path: Path, base: Path) -> bool:
    try:
        real_path.relative_to(base)
    except ValueError:
        return False
    return True


def _candidate_for(path: Path, base: Path, kind: str) -> CleanupCandidate | None:
    """Build a candidate if (and only if) the real path stays in the base."""
    try:
        real = path.resolve(strict=False)
    except OSError:
        return None
    if not _contained(real, base):
        return None  # symlink escape / vanished — caller records a skip
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return CleanupCandidate(
        display=path.relative_to(base).as_posix(),
        path=path,
        kind=kind,
        size_bytes=size,
    )


def _resolve_location(location: str) -> tuple[Path, Path, str]:
    """Map a location identifier to (target, base, root-relative display).

    Raises RetentionError on traversal, escapes, or ambiguous targets.
    """
    if not location or location.strip() in ("", ".", "/"):
        raise RetentionError(
            "Empty cleanup location. Use --list to see valid locations."
        )
    from pathlib import PurePath

    if PurePath(location).is_absolute() or re.match(r"^[A-Za-z]:", location):
        raise RetentionError(
            "Refused: absolute or drive-qualified paths are not accepted — "
            "use a location identifier from --list."
        )
    location = location.strip().replace("\\", "/").strip("/")
    parts = location.split("/")
    if ".." in parts:
        raise RetentionError(
            f"Refused: location {location!r} contains path traversal "
            "('..') — never followed."
        )
    if location == DEFAULT_LOCATION:
        return config.OUTPUT_DIR, config.OUTPUT_DIR, DEFAULT_LOCATION
    if location == TRANSLATION_LOCATION:
        return config.DATA_DIR, config.DATA_DIR, TRANSLATION_LOCATION

    base = _resolved_base(config.OUTPUT_DIR)
    target = Path(config.OUTPUT_DIR, *parts)
    try:
        real_target = target.resolve(strict=False)
    except OSError as exc:
        raise RetentionError(f"Cannot resolve location {location!r}: {exc}")
    if not _contained(real_target, base):
        raise RetentionError(
            f"Refused: location {location!r} resolves outside the "
            f"configured output root — path traversal is never followed."
        )
    if real_target == base:
        raise RetentionError(
            "Refused: the storage root itself is not a deletable location. "
            "Use 'default', a run directory, 'comparison/<run-id>' or "
            f"'{TRANSLATION_LOCATION}'."
        )
    if real_target.name == COMPARISON_DIRNAME and real_target.parent == base:
        raise RetentionError(
            "Refused: the shared comparison root aggregates several runs — "
            "clean its per-run directories individually "
            f"('comparison/<run-id>'), never as one broad deletion."
        )
    return real_target, base, location


def plan_run_cleanup(location: str) -> CleanupPlan:
    """Build the exact preview for a cleanup location (deletes nothing)."""
    target, base, display_location = _resolve_location(location)
    candidates: list[CleanupCandidate] = []
    skipped: list[tuple[str, str]] = []
    empty_dirs: list[Path] = []

    if display_location == DEFAULT_LOCATION:
        for name in sorted(RUN_ARTIFACT_NAMES | DEFAULT_RUN_EXTRA_NAMES):
            path = target / name
            if not path.exists():
                continue
            candidate = _candidate_for(path, base, "run_artifact")
            if candidate is None:
                skipped.append((name, "resolves outside the storage root"))
            else:
                candidates.append(candidate)
        if not candidates:
            skipped.append(
                ("(outputs root)", "no default-run artifacts present")
            )
    elif display_location == TRANSLATION_LOCATION:
        for name in TRANSLATION_ARTIFACT_NAMES:
            path = target / name
            if not path.exists():
                # Audit fix (PR #38): the export may already be gone while
                # its exclusively-paired sidecar remains — include the
                # orphan sidecar instead of leaving a stale reference.
                # Pairing is exact (with_suffix), never a loose match; a
                # non-regular or escaping sidecar is skipped, not deleted.
                sidecar = path.with_suffix(SIDECAR_SUFFIX)
                if not sidecar.exists():
                    skipped.append((name, "already absent"))
                elif sidecar.is_symlink() or not sidecar.is_file():
                    skipped.append(
                        (
                            sidecar.name,
                            "orphan sidecar is not a regular file — "
                            "manual review",
                        )
                    )
                else:
                    sidecar_candidate = _candidate_for(
                        sidecar, base, "sidecar"
                    )
                    if sidecar_candidate is None:
                        skipped.append(
                            (
                                sidecar.name,
                                "orphan sidecar resolves outside the "
                                "storage root",
                            )
                        )
                    else:
                        candidates.append(sidecar_candidate)
                continue
            candidate = _candidate_for(path, base, "run_artifact")
            if candidate is None:
                skipped.append((name, "resolves outside the storage root"))
                continue
            candidates.append(candidate)
            sidecar = path.with_suffix(SIDECAR_SUFFIX)
            if sidecar.exists():
                if sidecar.is_symlink() or not sidecar.is_file():
                    skipped.append(
                        (
                            sidecar.name,
                            "sidecar is not a regular file — manual review",
                        )
                    )
                    continue
                sidecar_candidate = _candidate_for(
                    sidecar, base, "sidecar"
                )
                if sidecar_candidate is None:
                    skipped.append(
                        (sidecar.name, "resolves outside the storage root")
                    )
                else:
                    candidates.append(sidecar_candidate)
        if not candidates:
            skipped.append(("(data/)", "no translation artifacts present"))
    else:
        # a run directory under the output root (incl. comparison/<run-id>)
        if not target.is_dir():
            # already cleaned (or never existed) — a clean no-op: nothing
            # is deleted, the repeat cannot cause unrelated changes
            skipped.append(
                (display_location, "already absent or empty — nothing to do")
            )
            return CleanupPlan(
                location=display_location,
                root=base,
                candidates=(),
                skipped=tuple(skipped),
                empty_dirs_after=(),
            )
        for entry in sorted(target.iterdir()):
            if entry.is_file() and not entry.is_symlink() \
                    and entry.name in RUN_ARTIFACT_NAMES:
                candidate = _candidate_for(entry, base, "run_artifact")
                if candidate is None:
                    skipped.append(
                        (entry.name, "resolves outside the storage root")
                    )
                else:
                    candidates.append(candidate)
            else:
                skipped.append(
                    (
                        entry.relative_to(base).as_posix(),
                        "not a recognized run artifact — manual review",
                    )
                )
        if not candidates:
            raise RetentionError(
                f"Refused: {location!r} contains no recognized run "
                "artifacts — its ownership by a run cannot be established "
                "confidently, so nothing is deleted."
            )
        # the directory itself may become empty after a full deletion
        deletable_names = {
            c.display.rsplit("/", 1)[-1] for c in candidates
        }
        remaining = [
            p.name for p in target.iterdir() if p.name not in deletable_names
        ]
        if not remaining:
            empty_dirs.append(target)

    return CleanupPlan(
        location=display_location,
        root=base,
        candidates=tuple(candidates),
        skipped=tuple(skipped),
        empty_dirs_after=tuple(empty_dirs),
    )


def list_run_locations() -> list[str]:
    """Addressable cleanup locations (for --list and the UI picker)."""
    locations: list[str] = []
    if any(
        (config.OUTPUT_DIR / name).exists()
        for name in RUN_ARTIFACT_NAMES | DEFAULT_RUN_EXTRA_NAMES
    ):
        locations.append(DEFAULT_LOCATION)
    base = _resolved_base(config.OUTPUT_DIR)
    for entry in sorted(config.OUTPUT_DIR.iterdir()):
        if not entry.is_dir() or entry.name == COMPARISON_DIRNAME:
            continue
        real = entry.resolve(strict=False)
        if not _contained(real, base):
            continue
        if any(
            (entry / name).exists() for name in RUN_ARTIFACT_NAMES
        ):
            locations.append(entry.name)
    comparison = config.OUTPUT_DIR / COMPARISON_DIRNAME
    if comparison.is_dir():
        for entry in sorted(comparison.iterdir()):
            real = entry.resolve(strict=False)
            if not _contained(real, base):
                continue
            if entry.is_dir() and any(
                (entry / name).exists() for name in RUN_ARTIFACT_NAMES
            ):
                locations.append(f"{COMPARISON_DIRNAME}/{entry.name}")
    if any(
        (config.DATA_DIR / name).exists()
        for name in TRANSLATION_ARTIFACT_NAMES
    ):
        locations.append(TRANSLATION_LOCATION)
    return locations


def execute_run_cleanup(
    plan: CleanupPlan, *, confirm: bool = False, audit: bool = True
) -> dict:
    """Preview (``confirm=False``) or perform (``confirm=True``) a cleanup.

    Returns a metadata-only result dict. Missing files are reported as
    already-absent; per-file failures are recorded and do not abort the
    rest. Never deletes anything outside the plan's whitelist. With
    ``confirm=False`` nothing on disk is touched.
    """
    mode = "confirmed" if confirm else "preview"
    deleted: list[str] = []
    already_absent: list[str] = []
    failed: list[tuple[str, str]] = []
    if confirm:
        for candidate in plan.candidates:
            try:
                if not candidate.path.exists():
                    already_absent.append(candidate.display)
                    continue
                candidate.path.unlink()
                deleted.append(candidate.display)
            except OSError as exc:
                failed.append((candidate.display, type(exc).__name__))
        for run_dir in plan.empty_dirs_after:
            try:
                if run_dir.is_dir() and not any(run_dir.iterdir()):
                    run_dir.rmdir()  # containment-checked empty dir only
            except OSError as exc:
                failed.append((run_dir.name + "/", type(exc).__name__))

    result = {
        "location": plan.location,
        "mode": mode,
        "selected": len(plan.candidates),
        "deleted": len(deleted),
        "already_absent": len(already_absent),
        "skipped": len(plan.skipped),
        "failed": len(failed),
        "deleted_files": deleted,
        "failed_files": [name for name, _ in failed],
        "outcome": (
            "complete" if not failed else
            "partial" if deleted else
            "failed"
        ),
        "total_bytes": plan.total_bytes if not confirm else sum(
            c.size_bytes for c in plan.candidates
            if c.display in deleted
        ),
    }
    if confirm and audit:
        record_deletion_audit(
            location=plan.location,
            mode=mode,
            selected=result["selected"],
            deleted=result["deleted"],
            skipped=result["skipped"] + result["already_absent"],
            failed=result["failed"],
            outcome=result["outcome"],
        )
    return result


def record_deletion_audit(
    *,
    location: str,
    mode: str,
    selected: int,
    deleted: int,
    skipped: int,
    failed: int,
    outcome: str,
) -> Path:
    """Append a metadata-only deletion record to the audit log.

    Fields are fixed and content-free: timestamp, relative location,
    counts, mode, outcome. Never conversation text, never file contents,
    never absolute paths.
    """
    log_dir = config.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "run_cleanup",
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": location,
        "mode": mode,
        "files_selected": selected,
        "files_deleted": deleted,
        "files_skipped": skipped,
        "files_failed": failed,
        "outcome": outcome,
    }
    audit_path = log_dir / AUDIT_LOG_NAME
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return audit_path
