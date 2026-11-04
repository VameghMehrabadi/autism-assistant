"""Dataset validation utilities for the Autism Assistant pipeline.

Covers the Phase-2 data-engineering requirements without replacing the
existing loader architecture:

- encoding detection and handling (no external dependencies)
- record normalization (missing values) and schema validation
- content fingerprinting, duplicate detection
- script-based language identification (Persian/English)
- dataset integrity reports
- reproducible dev/eval splitting with a persistent manifest and an
  explicit leakage guard
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import config

# Version of the preprocessing contract; recorded in reports and manifests.
PREPROCESS_VERSION: str = config.PREPROCESS_VERSION

# Fields considered when deciding whether a record has any usable content.
TEXT_FIELDS: tuple[str, ...] = ("instruction", "input", "output", "text")


class DataValidationError(ValueError):
    """Raised when strict validation encounters an unrecoverable record."""


class LeakageError(RuntimeError):
    """Raised when dev and eval splits share records (data leakage)."""


# --------------------------------------------------------------------------
# Issues
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding. Messages never contain record content."""

    code: str
    position: int
    message: str
    severity: str = "error"  # "error" | "warning" | "info"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Encoding detection
# --------------------------------------------------------------------------

# Single-byte fallbacks for legacy files; ordered most specific first.
_BYTE_ENCODINGS: tuple[str, ...] = ("cp1256", "cp1252", "latin-1")


def detect_encoding(path: str | Path, sample_size: int = 65536) -> str:
    """Detect a workable text encoding for ``path`` without new dependencies.

    Order: BOM sniffing, strict UTF-8, then single-byte fallbacks. Among the
    single-byte fallbacks, cp1256 is chosen when the decoded sample contains
    Arabic-script characters (typical for legacy Persian CSV exports),
    otherwise cp1252. latin-1 always succeeds, so a value is always returned.

    Boundary rule: a byte sample truncated mid-file may end inside a valid
    UTF-8 multibyte sequence (Persian characters are 2 bytes in UTF-8). The
    resulting strict-decode failure would misdetect large valid UTF-8 files
    as CP1256 and silently mojibake every Persian character. Therefore, when
    the *first* decode error starts within the last 3 bytes of the sample
    (the longest UTF-8 sequence is 4 bytes), the file is treated as UTF-8 —
    the incomplete tail is a sampling artifact, not file corruption. A
    genuinely non-UTF-8 file (e.g. CP1256) fails decoding at its first
    high byte, far earlier in the sample, so legitimate CP1256 detection is
    unaffected. Worst case for a pathological file: a loud read-time
    UnicodeDecodeError instead of silent corruption.
    """
    with Path(path).open("rb") as handle:
        raw = handle.read(sample_size)
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return "utf-32"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as exc:
        if exc.start >= len(raw) - 3 and len(raw) == sample_size:
            # Sample truncated mid-multibyte-sequence: the file itself is
            # (with overwhelming likelihood) valid UTF-8.
            return "utf-8"
    for enc in _BYTE_ENCODINGS:
        decoded = raw.decode(enc, errors="replace")
        if _has_arabic_script(decoded):
            return enc
    return _BYTE_ENCODINGS[-1]


def _has_arabic_script(text: str) -> bool:
    return any(0x0600 <= ord(ch) <= 0x06FF for ch in text)


# --------------------------------------------------------------------------
# Normalization / schema validation
# --------------------------------------------------------------------------


def normalize_value(value: Any) -> str:
    """Coerce one raw cell/JSON value to a clean string (missing → "")."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_record(row: Any) -> dict[str, str]:
    """Return a copy of ``row`` with clean keys and normalized string values."""
    if not isinstance(row, dict):
        raise TypeError(f"record must be a dict, got {type(row).__name__}")
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized[str(key).strip()] = normalize_value(value)
    return normalized


def has_content(row: dict[str, str]) -> bool:
    """True when at least one known text field is non-empty."""
    return any(row.get(k) for k in TEXT_FIELDS)


def validate_record(row: dict[str, str], position: int = 0) -> list[ValidationIssue]:
    """Validate one normalized record; returns a list of issues (may be empty).

    An ``error``-severity issue means the record cannot be labeled and should
    be skipped (or raise in strict mode). ``warning``/``info`` issues are
    reported but the record is still usable.
    """
    issues: list[ValidationIssue] = []
    if not has_content(row):
        issues.append(
            ValidationIssue(
                code="EMPTY_RECORD",
                position=position,
                message="no content in any known text field",
                severity="error",
            )
        )
    idx = row.get("idx")
    if idx and not str(idx).isdigit():
        issues.append(
            ValidationIssue(
                code="BAD_IDX",
                position=position,
                message="idx is not integer-like; positional index will be used",
                severity="warning",
            )
        )
    return issues


# --------------------------------------------------------------------------
# Language identification (script-based, dependency-free)
# --------------------------------------------------------------------------

_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)
# Persian-specific letters (گ چ پ ژ) that never occur in Arabic text.
_PERSIAN_MARKERS: frozenset[str] = frozenset("گچپژ")


def _arabic_letter(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_RANGES)


def detect_language(text: str) -> str:
    """Identify the dominant language/script of ``text``.

    Returns one of: ``"fa"`` (Arabic-script dominant; treated as Persian —
    Arabic is not distinguished beyond the Persian-marker hint), ``"en"``
    (Latin dominant), ``"mixed"`` (no ≥50% majority between the two) or
    ``"unknown"`` (no letters at all).
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "unknown"
    arabic = sum(1 for ch in letters if _arabic_letter(ch))
    latin = sum(1 for ch in letters if ch.isascii())
    total = len(letters)
    if arabic / total >= 0.5:
        return "fa"
    if latin / total >= 0.5:
        return "en"
    return "mixed"


def language_distribution(texts: Iterable[str]) -> dict[str, int]:
    """Count detected languages over a collection of texts."""
    return dict(Counter(detect_language(t) for t in texts))


# --------------------------------------------------------------------------
# Fingerprints / duplicate detection
# --------------------------------------------------------------------------


def record_fingerprint(row: dict[str, Any]) -> str:
    """Stable SHA-256 fingerprint of a record's text content.

    Whitespace is collapsed so purely cosmetic differences (extra spaces,
    line-ending changes) do not create false uniqueness. The fingerprint
    cannot be reversed to reconstruct the conversation.
    """
    parts = []
    for key in ("instruction", "input", "output", "text"):
        value = str(row.get(key) or "")
        parts.append(" ".join(value.split()))
    blob = "\u241f".join(parts)  # ␟ unit separator between fields
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def find_duplicate_groups(fingerprints: Sequence[str]) -> list[list[int]]:
    """Group positions that share the same fingerprint (groups of ≥2)."""
    by_hash: dict[str, list[int]] = {}
    for pos, fp in enumerate(fingerprints):
        by_hash.setdefault(fp, []).append(pos)
    return [positions for positions in by_hash.values() if len(positions) > 1]


# --------------------------------------------------------------------------
# Integrity report
# --------------------------------------------------------------------------


def integrity_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build a transparent integrity summary for a list of normalized rows."""
    field_coverage: Counter[str] = Counter()
    for row in rows:
        for key in TEXT_FIELDS:
            if row.get(key):
                field_coverage[key] += 1
    fingerprints = [record_fingerprint(row) for row in rows]
    duplicate_groups = find_duplicate_groups(fingerprints)
    merged_texts = [
        " ".join(filter(None, (str(row.get(k) or "") for k in TEXT_FIELDS)))
        for row in rows
    ]
    empty_records = sum(1 for text in merged_texts if not text.strip())
    return {
        "preprocess_version": PREPROCESS_VERSION,
        "total_rows": len(rows),
        "field_coverage": dict(field_coverage),
        "empty_records": empty_records,
        "n_unique": len(set(fingerprints)),
        "duplicate_rows": len(rows) - len(set(fingerprints)),
        "duplicate_groups": len(duplicate_groups),
        "language_distribution": language_distribution(merged_texts),
    }


# --------------------------------------------------------------------------
# dev/eval splits and leakage guard
# --------------------------------------------------------------------------


def assign_splits(
    hashes: Sequence[str],
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Deterministically assign each unique hash to ``"dev"`` or ``"eval"``.

    Same hash list + same seed → identical assignment (reproducible). The
    assignment is keyed by content hash so a record always lands in the same
    split regardless of row order in later loads.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    unique = list(dict.fromkeys(hashes))
    if not unique:
        return {}
    rng = random.Random(seed)
    order = list(range(len(unique)))
    rng.shuffle(order)
    n_eval = max(1, round(len(unique) * eval_fraction))
    eval_set = {unique[i] for i in order[:n_eval]}
    return {h: ("eval" if h in eval_set else "dev") for h in unique}


def _item_hash(item: Any) -> str | None:
    """Extract a record hash from a Sample-like object or a plain string."""
    if isinstance(item, str):
        return item
    meta = getattr(item, "meta", None)
    if isinstance(meta, dict):
        return meta.get("record_hash")
    if isinstance(item, dict):
        return item.get("record_hash")
    return None


def save_split_manifest(
    assignment: dict[str, str],
    path: str | Path | None = None,
    *,
    seed: int = 42,
    eval_fraction: float = 0.2,
    source: str | None = None,
) -> Path:
    """Persist the split assignment (hashes only — privacy-preserving)."""
    manifest_path = Path(path) if path else config.SPLIT_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    eval_hashes = sorted(h for h, s in assignment.items() if s == "eval")
    dev_hashes = sorted(h for h, s in assignment.items() if s == "dev")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preprocess_version": PREPROCESS_VERSION,
        "seed": seed,
        "eval_fraction": eval_fraction,
        "source": source,
        "n_dev": len(dev_hashes),
        "n_eval": len(eval_hashes),
        "dev_hashes": dev_hashes,
        "eval_hashes": eval_hashes,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def load_split_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the split manifest or raise with a helpful message."""
    manifest_path = Path(path) if path else config.SPLIT_MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No split manifest found at {manifest_path}. Create one first with "
            "data_validation.create_splits(...) before loading an eval split."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def check_no_leakage(
    dev_hashes: Iterable[str],
    eval_hashes: Iterable[str],
) -> None:
    """Raise :class:`LeakageError` if any record is in both splits."""
    overlap = set(dev_hashes) & set(eval_hashes)
    if overlap:
        raise LeakageError(
            f"Data leakage detected: {len(overlap)} record(s) present in both "
            "dev and eval splits. Regenerate the split manifest."
        )


def create_splits(
    items: Sequence[Any],
    eval_fraction: float = 0.2,
    seed: int | None = None,
    manifest_path: str | Path | None = None,
    source: str | None = None,
) -> tuple[list[Any], list[Any]]:
    """Split loaded items into (dev, eval), persist a manifest, verify disjoint.

    ``items`` must be Sample-like objects carrying ``meta["record_hash"]``
    (as produced by the Phase-2 loader) or plain hash strings.
    """
    seed = config.SAMPLING_SEED if seed is None else seed
    hashes: list[str] = []
    for i, item in enumerate(items):
        h = _item_hash(item)
        if not h:
            raise DataValidationError(
                f"item at position {i} has no record_hash; load data through "
                "data_loader.load_samples(...) so hashes are attached."
            )
        hashes.append(h)
    assignment = assign_splits(hashes, eval_fraction=eval_fraction, seed=seed)
    dev = [item for item, h in zip(items, hashes) if assignment[h] == "dev"]
    evaluation = [item for item, h in zip(items, hashes) if assignment[h] == "eval"]
    save_split_manifest(
        assignment,
        path=manifest_path,
        seed=seed,
        eval_fraction=eval_fraction,
        source=source,
    )
    check_no_leakage(
        (h for h in hashes if assignment[h] == "dev"),
        (h for h in hashes if assignment[h] == "eval"),
    )
    return dev, evaluation


def filter_by_split(
    items: Sequence[Any],
    split: str,
    manifest_path: str | Path | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Keep only items belonging to ``split`` ("dev"|"eval") per the manifest.

    Returns ``(kept_items, stats)``. Items without a hash are excluded and
    counted as ``missing_hash`` — they can never silently enter evaluation.
    """
    split = str(split).lower().strip()
    if split not in {"dev", "eval"}:
        raise ValueError(f"split must be 'dev' or 'eval', got {split!r}")
    manifest = load_split_manifest(manifest_path)
    eval_hashes = set(manifest.get("eval_hashes", []))
    dev_hashes = set(manifest.get("dev_hashes", []))
    if split == "dev" and not dev_hashes:
        # Older manifests: fall back to the complement of the eval set.
        dev_hashes = None  # type: ignore[assignment]
    kept: list[Any] = []
    excluded = 0
    missing_hash = 0
    for item in items:
        h = _item_hash(item)
        if not h:
            missing_hash += 1
            continue
        if split == "eval":
            keep = h in eval_hashes
        else:
            keep = (h not in eval_hashes) if dev_hashes is None else (h in dev_hashes)
        if keep:
            kept.append(item)
        else:
            excluded += 1
    stats = {
        "split": split,
        "kept": len(kept),
        "excluded": excluded,
        "missing_hash": missing_hash,
        "manifest": str(manifest_path or config.SPLIT_MANIFEST_PATH),
    }
    return kept, stats
