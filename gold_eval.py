"""Validator for the human-labeled gold evaluation set (data/gold/).

Enforces the governance rules from ``data/gold/README.md``:

- every record must declare ``origin: "human"`` — model-generated labels are
  invalid and fail validation;
- required fields, category vocabulary, and round/confidence vocabularies per
  ``data/gold/gold_eval.schema.json`` (validated natively — no new
  dependencies);
- unique ``gold_id`` / ``source_idx`` within one version;
- a file with zero records is valid but reported as ``empty`` (no labels
  exist yet — accuracy claims are forbidden until real human labels are
  committed).

The validator never fabricates, repairs, or scores labels.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORIES = ("A", "B", "C", "D", "E", "F", "G", "UNSURE")
CONFIDENCE = ("sure", "leaning", "guess")
_GOLD_ID_RE = re.compile(r"^gold-[0-9]{4,6}$")
_VERSION_RE = re.compile(r"^v[0-9]+$")

REQUIRED_FIELDS = ("gold_id", "source_idx", "labeler_id", "origin", "round", "final_label")


def validate_gold_record(record: Any, position: int) -> list[str]:
    """Validate one gold record; returns a list of error strings (may be empty)."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"line {position + 1}: record is not a JSON object"]
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"line {position + 1}: missing required field '{field}'")
    gold_id = record.get("gold_id")
    if gold_id is not None and not isinstance(gold_id, str):
        errors.append(f"line {position + 1}: gold_id must be a string")
    elif isinstance(gold_id, str) and not _GOLD_ID_RE.match(gold_id):
        errors.append(f"line {position + 1}: gold_id {gold_id!r} does not match gold-NNNN")
    if record.get("origin") != "human":
        errors.append(
            f"line {position + 1}: origin must be 'human' — model-generated "
            "labels are forbidden in the gold set"
        )
    if record.get("final_label") not in CATEGORIES:
        errors.append(f"line {position + 1}: final_label must be one of {CATEGORIES}")
    first = record.get("first_label")
    if first is not None and first not in CATEGORIES:
        errors.append(f"line {position + 1}: first_label must be one of {CATEGORIES}")
    if record.get("round") is not None:
        rnd = record["round"]
        if not isinstance(rnd, int) or not 1 <= rnd <= 2:
            errors.append(f"line {position + 1}: round must be 1 or 2")
    conf = record.get("confidence")
    if conf is not None and conf not in CONFIDENCE:
        errors.append(f"line {position + 1}: confidence must be one of {CONFIDENCE}")
    source_idx = record.get("source_idx")
    if source_idx is not None and (not isinstance(source_idx, int) or source_idx < 0):
        errors.append(f"line {position + 1}: source_idx must be a non-negative integer")
    unknown = set(record) - {
        "gold_id", "source_idx", "labeler_id", "origin", "round", "first_label",
        "final_label", "adjudicated", "confidence", "notes", "labeled_utc",
        "gold_set_version",
    }
    if unknown:
        errors.append(f"line {position + 1}: unknown fields {sorted(unknown)}")
    return errors


def validate_gold_file(path: str | Path) -> dict[str, Any]:
    """Validate a gold labels JSONL file; returns a content-free summary.

    ``valid`` is True only when the file parses completely, every record
    satisfies the schema rules, and no ids are duplicated. An empty file is
    valid with ``n_records == 0`` — no accuracy claims may be made until
    real human labels exist.
    """
    path = Path(path)
    report: dict[str, Any] = {
        "file": str(path),
        "exists": path.exists(),
        "n_records": 0,
        "n_valid": 0,
        "malformed_lines": 0,
        "duplicate_gold_ids": 0,
        "duplicate_source_idx": 0,
        "origin_non_human": 0,
        "label_distribution": {},
        "errors": [],
        "valid": False,
        "empty": True,
    }
    if not path.exists():
        report["errors"].append(f"file not found: {path}")
        return report

    seen_gold: set[str] = set()
    seen_source: set[int] = set()
    labels: Counter = Counter()
    position = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            position += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report["malformed_lines"] += 1
                report["errors"].append(f"line {position + 1}: invalid JSON ({exc})")
                continue
            report["n_records"] += 1
            errors = validate_gold_record(record, position)
            if not errors:
                report["n_valid"] += 1
                labels[record["final_label"]] += 1
            gold_id = record.get("gold_id") if isinstance(record, dict) else None
            if isinstance(gold_id, str):
                if gold_id in seen_gold:
                    report["duplicate_gold_ids"] += 1
                    errors.append(f"line {position + 1}: duplicate gold_id {gold_id}")
                seen_gold.add(gold_id)
            source_idx = record.get("source_idx") if isinstance(record, dict) else None
            if isinstance(source_idx, int):
                if source_idx in seen_source:
                    report["duplicate_source_idx"] += 1
                    errors.append(
                        f"line {position + 1}: duplicate source_idx {source_idx}"
                    )
                seen_source.add(source_idx)
            if isinstance(record, dict) and record.get("origin") != "human":
                report["origin_non_human"] += 1
            report["errors"].extend(errors)

    report["label_distribution"] = dict(labels)
    report["empty"] = report["n_records"] == 0
    report["valid"] = (
        report["malformed_lines"] == 0
        and report["n_records"] == report["n_valid"]
        and report["duplicate_gold_ids"] == 0
        and report["duplicate_source_idx"] == 0
        and report["origin_non_human"] == 0
    )
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate the gold evaluation set.")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parent / "data" / "gold" / "gold_labels_v1.jsonl"),
    )
    args = parser.parse_args(argv)
    report = validate_gold_file(args.file)
    print(
        f"[gold] records={report['n_records']} valid={report['n_valid']} "
        f"duplicates={report['duplicate_gold_ids'] + report['duplicate_source_idx']} "
        f"non_human={report['origin_non_human']} valid_file={report['valid']}"
    )
    if report["empty"]:
        print(
            "[gold] NOTE: no human labels exist yet — evaluation accuracy "
            "claims are forbidden until a labeled version is committed."
        )
    for error in report["errors"][:20]:
        print(f"[gold] ERROR: {error}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
