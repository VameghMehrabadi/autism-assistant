"""هم‌ترازی تاریخی runها (Issue #9) — الگوریتم کانونیکال + طبقه‌بندی legacy.

Canonical alignment helpers shared by ``compare_runs --verify-alignment``,
the comparison report and the legacy migration artifact. There is exactly
ONE fingerprint algorithm per purpose in this repository:

- ``labeled_text_fingerprint``   — sorted labeled texts, exactly the hash the
  Phase-4 verifier has always computed (``compare_runs.verify_alignment``).
- ``ordered_input_fingerprint``  — labeled input texts in row order, exactly
  the hash ``main.run_pipeline`` records as ``dataset_fingerprint``.

**Provenance rule (Issue #9):** a fingerprint computed from a persisted
``labeled.csv`` is *artifact-derived evidence*. It can compare what two runs
actually labeled, but it is NOT execution-time metadata and cannot prove the
original dataset-level provenance (pre-filtering records, ``dataset_total``,
preprocessing configuration). Retroactive metadata therefore lives in a
separate, clearly named artifact (``alignment_legacy.json``) that never
modifies ``run_meta.json`` or ``labeled.csv``, and classified runs are never
reported as execution-time verified.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TEXT_SEPARATOR = "\u241f"  # same separator the Phase-4 verifier uses
LEGACY_ARTIFACT_NAME = "alignment_legacy.json"

CLASSIFICATION_MODERN = "modern"
CLASSIFICATION_LEGACY_PARTIAL = "legacy-partial"
CLASSIFICATION_LEGACY = "legacy"

# Fields the Phase-4 execution-time contract requires in run_meta.json.
MODERN_REQUIRED_FIELDS = (
    "dataset_fingerprint",
    "text_window",
    "preprocess_version",
    "threshold",
    "dataset_total",
)


def _short_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def labeled_text_fingerprint(df: pd.DataFrame) -> str:
    """Canonical hash of the labeled texts (order-independent).

    Byte-for-byte the algorithm ``compare_runs.verify_alignment`` has used
    since Phase 4 — do not change one without the other.
    """
    return _short_hash(
        TEXT_SEPARATOR.join(sorted(df["text"].astype(str)))
    )


def ordered_input_fingerprint(texts: list[str]) -> str:
    """Canonical hash of the labeled input texts in row order.

    Byte-for-byte the algorithm ``main.run_pipeline`` uses for the
    execution-time ``dataset_fingerprint``.
    """
    return _short_hash(TEXT_SEPARATOR.join(texts))


def record_idx_fingerprint(df: pd.DataFrame) -> str:
    """Canonical hash of the labeled record-ID set."""
    idx = sorted(
        pd.to_numeric(df["idx"], errors="coerce").dropna().astype(int)
    )
    return _short_hash(TEXT_SEPARATOR.join(str(i) for i in idx))


def labeled_csv_sha256(path: Path) -> str:
    """Full sha256 of the labeled.csv artifact (artifact identity only)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Full sha256 of any file (source-artifact fingerprints, Issue #9)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Issue #9 review fixes: strict per-field contract validation -----------
# A run is `modern` only when EVERY required execution-time field is present
# with a schema-valid value. Legitimate schema values are preserved:
# `text_window: null` (the model's own default) and numeric zero are valid.


def _is_hex16(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


FIELD_VALIDATORS = {
    "dataset_fingerprint": _is_hex16,
    "text_window": lambda v: v is None or (_is_int(v) and v >= 1),
    "preprocess_version": _is_nonempty_str,
    "threshold": lambda v: _is_number(v) and v >= 0,
    "dataset_total": lambda v: _is_int(v) and v >= 0,
}


def _modern_field_issues(meta: dict) -> list[str]:
    """Missing or schema-invalid execution-time fields (sorted, stable)."""
    issues: list[str] = []
    for field in MODERN_REQUIRED_FIELDS:
        if field not in meta:
            issues.append(f"missing: {field}")
        elif not FIELD_VALIDATORS[field](meta[field]):
            issues.append(f"invalid: {field}")
    return issues


LEGACY_SCHEMA = "legacy-alignment-v1"

# Required evidence fields of a retroactive artifact, with validators.
LEGACY_EVIDENCE_FIELDS = {
    "labeled_csv_sha256": _is_hex64,
    "labeled_text_fingerprint": _is_hex16,
    "ordered_input_fingerprint_derived": _is_hex16,
    "record_idx_fingerprint": _is_hex16,
    "record_count": lambda v: _is_int(v) and v >= 0,
}


def _valid_legacy_artifact(artifact: object) -> bool:
    """Strict, defensive validation of a retroactive alignment artifact.

    Malformed, incomplete, or unexpected structures are rejected — they must
    never classify a run as ``legacy-partial`` and never raise.
    """
    if not isinstance(artifact, dict):
        return False
    if artifact.get("schema") != LEGACY_SCHEMA:
        return False
    if artifact.get("provenance") != "retroactive":
        return False
    if artifact.get("classification") != CLASSIFICATION_LEGACY_PARTIAL:
        return False
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        return False
    for field, check in LEGACY_EVIDENCE_FIELDS.items():
        if field not in evidence or not check(evidence[field]):
            return False
    missing = artifact.get("missing")
    if not isinstance(missing, dict):
        return False
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in missing.items()
    ):
        return False
    sources = artifact.get("sources", {})
    if not isinstance(sources, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in sources.items()
    ):
        return False
    return True


def read_json(path: Path) -> tuple[object | None, str | None]:
    """Tolerant JSON read: (value, None) or (None, error reason)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, None  # absent is normal, not an error
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def classify_alignment(run_dir: Path, meta: dict | None = None) -> dict:
    """Classify one run's alignment metadata status (never raises).

    Classification (Issue #9 review fix — strict contract validation):
    - ``modern``          — execution-time run_meta carries EVERY Phase-4
      contract field with a schema-valid value (``text_window: null`` is the
      legitimate model-default value; numeric zero is valid where allowed).
    - ``legacy-partial``  — no valid execution-time contract, but a strictly
      validated retroactive ``alignment_legacy.json`` artifact records the
      reconstructable evidence.
    - ``legacy``          — no execution-time contract and no valid artifact;
      the run cannot be treated as verified.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    meta_error: str | None = None
    if meta is None:
        meta, meta_error = read_json(meta_path)
        if meta is not None and not isinstance(meta, dict):
            meta, meta_error = None, "run_meta.json is not a JSON object"

    if isinstance(meta, dict):
        issues = _modern_field_issues(meta)
    else:
        issues = list(MODERN_REQUIRED_FIELDS)
    if isinstance(meta, dict) and not issues:
        return {
            "classification": CLASSIFICATION_MODERN,
            "reason": None,
            "artifact": None,
            "missing_fields": [],
        }

    artifact_path = run_dir / LEGACY_ARTIFACT_NAME
    artifact, artifact_error = read_json(artifact_path)
    if _valid_legacy_artifact(artifact):
        return {
            "classification": CLASSIFICATION_LEGACY_PARTIAL,
            "reason": (
                "execution-time alignment metadata absent or incomplete "
                f"({'; '.join(issues)}); retroactive artifact present "
                f"({LEGACY_ARTIFACT_NAME})"
            ),
            "artifact": LEGACY_ARTIFACT_NAME,
            "missing_fields": issues,
        }

    if meta_error:
        reason = f"run_meta.json malformed ({meta_error})"
    elif not isinstance(meta, dict):
        reason = "run_meta.json absent"
    else:
        reason = (
            "run_meta predates the Phase-4 alignment contract or has "
            "invalid values; " + "; ".join(issues)
        )
    if artifact_error:
        reason += f"; {LEGACY_ARTIFACT_NAME} malformed ({artifact_error})"
    elif artifact is not None:
        reason += (
            f"; {LEGACY_ARTIFACT_NAME} present but invalid or incomplete — "
            "treated as legacy"
        )
    return {
        "classification": CLASSIFICATION_LEGACY,
        "reason": reason,
        "artifact": None,
        "missing_fields": issues,
    }


def build_legacy_artifact(run_dir: Path) -> dict:
    """Build the retroactive alignment artifact for one historical run.

    Uses only evidence present in the run directory. Fields that cannot be
    proven from artifacts are listed under ``missing`` with reasons — never
    invented. The result is classified ``legacy-partial``: useful for
    cross-run comparison of labeled inputs, never a claim of execution-time
    verification.
    """
    run_dir = Path(run_dir)
    labeled_path = run_dir / "labeled.csv"
    df = pd.read_csv(labeled_path)

    gap, gap_error = read_json(run_dir / "gap_analysis.json")
    meta, meta_error = read_json(run_dir / "run_meta.json")
    if meta is not None and not isinstance(meta, dict):
        meta, meta_error = None, "run_meta.json is not a JSON object"

    texts_ordered = df["text"].astype(str).tolist()
    gap_threshold = (
        gap.get("threshold") if isinstance(gap, dict) else None
    )

    reconstructed: dict[str, object] = {}
    if gap_threshold is not None:
        reconstructed["threshold"] = gap_threshold
        reconstructed["threshold_source"] = "gap_analysis.json (execution-time report)"
    if isinstance(meta, dict):
        for field in ("model_key", "model_name", "prefix", "fact_lang",
                      "label_text_mode", "labeler_version", "sampling",
                      "seed", "id", "title"):
            if meta.get(field) is not None:
                reconstructed[field] = meta[field]
        reconstructed["execution_time_fields_source"] = (
            "historical run_meta.json (recorded at execution time)"
        )

    missing: dict[str, str] = {}
    missing["dataset_fingerprint_execution_time"] = (
        "not recorded by the historical run; the value derived from "
        "labeled.csv below is artifact-derived evidence, not execution-"
        "time metadata"
    )
    missing["text_window"] = (
        "effective preprocessing window was not recorded by historical runs"
    )
    missing["preprocess_version"] = (
        "preprocessing version was not recorded by historical runs"
    )
    missing["dataset_total"] = (
        "pre-validation dataset total was not recorded by historical runs"
    )
    if gap_error:
        missing["threshold"] = f"gap_analysis.json unreadable ({gap_error})"
    elif gap_threshold is None:
        missing["threshold"] = (
            "gap_analysis.json absent or contains no threshold"
        )

    # Issue #9 review fix — fingerprints of EVERY source artifact that
    # influences the reconstruction, so a change to any of them invalidates
    # the artifact. Missing sources are represented deterministically.
    sources = {
        name: (
            file_sha256(run_dir / name)
            if (run_dir / name).exists()
            else "absent"
        )
        for name in ("labeled.csv", "run_meta.json", "gap_analysis.json")
    }

    return {
        "schema": LEGACY_SCHEMA,
        "provenance": "retroactive",
        "classification": CLASSIFICATION_LEGACY_PARTIAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "sources": sources,
        "evidence": {
            "labeled_csv_sha256": labeled_csv_sha256(labeled_path),
            "labeled_text_fingerprint": labeled_text_fingerprint(df),
            "ordered_input_fingerprint_derived": ordered_input_fingerprint(
                texts_ordered
            ),
            "record_idx_fingerprint": record_idx_fingerprint(df),
            "record_count": int(len(df)),
        },
        "reconstructed": reconstructed,
        "missing": missing,
        "notes": [
            "Retroactively reconstructed metadata (Issue #9). Derived from "
            "the persisted labeled.csv and the run's own historical "
            "reports; NOT execution-time recorded and NOT proof of "
            "dataset-level provenance.",
        ],
    }


def migrate_legacy_alignment(
    run_dirs: list[Path], *, force: bool = False
) -> dict:
    """Write ``alignment_legacy.json`` for eligible historical runs.

    Idempotent (Issue #9 review fix): a run whose existing artifact is
    strictly valid AND whose recorded source fingerprints (labeled.csv,
    run_meta.json, gap_analysis.json) all match the current sources is left
    untouched unless ``force`` is set — a change to ANY source invalidates
    and regenerates the artifact. ``labeled.csv`` and ``run_meta.json``
    bytes are never modified and valid modern metadata is never overwritten.
    Returns a content-free summary.
    """
    migrated: list[str] = []
    already: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        name = run_dir.name
        if not (run_dir / "labeled.csv").exists():
            skipped.append((name, "no labeled.csv — nothing to migrate"))
            continue
        status = classify_alignment(run_dir)
        if status["classification"] == CLASSIFICATION_MODERN and not force:
            skipped.append(
                (name, "modern execution-time metadata present — skipped")
            )
            continue
        artifact_path = run_dir / LEGACY_ARTIFACT_NAME
        try:
            existing, _ = read_json(artifact_path)
            artifact = build_legacy_artifact(run_dir)
            current_sources = artifact.get("sources", {})
            if (
                _valid_legacy_artifact(existing)
                and existing.get("sources") == current_sources
                and not force
            ):
                already.append(name)
                continue
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            migrated.append(name)
        except (OSError, ValueError) as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))
    return {
        "migrated": migrated,
        "already_migrated": already,
        "skipped": skipped,
        "failed": failed,
    }
