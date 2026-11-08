"""اعتبارسنجی تراز منبع↔فارسی (Issue #6) — لایه‌ای مستقل از --verify-alignment.

Source-to-Persian alignment validation for the Persian dataset
(``data/mentalchat16k_fa.jsonl``) against the English source export
(``data/mentalchat16k_en_for_translation.jsonl``).

This is **explicitly not** what ``compare_runs --verify-alignment`` does:
that mode compares the *outputs of executed runs* (labeled.csv record sets,
text fingerprints, thresholds). This module compares the **source export**
against the **translated dataset** by stable record ID, checking structure,
status, and coverage — the precondition for runs 4 and 5.

Checks (aggregate output only — never record text):

- ID correspondence: missing / unexpected / duplicate IDs (stable ``idx``,
  never row order);
- field-structure preservation per record: for each of ``instruction`` /
  ``input`` / ``output``, the Persian field is non-empty exactly when the
  English source field is non-empty (speaker-role and record-boundary
  preservation);
- translation-status correctness and distribution;
- empty translated fields among ``status == "translated"`` records (hard);
- suspicious unchanged English text (soft warning — proper nouns may
  legitimately survive translation);
- Persian-script coverage statistics (soft — script presence does not
  prove translation quality);
- eligible-record set: ``status == "translated"`` AND structurally intact —
  the exact population runs 4/5 must use;
- coverage percentage and ``complete`` / ``structurally_valid`` verdicts.

Read-only: never rewrites either dataset. Default mode is **strict**
(structural failures refuse runs 4/5); partial coverage is only ever
tolerated through an explicit caller decision, and even then the eligible
set is limited to structurally intact records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIELDS = ("instruction", "input", "output")
TRANSLATED = "translated"
SOURCE_EXPORT_NAME = "mentalchat16k_en_for_translation.jsonl"
TARGET_NAME = "mentalchat16k_fa.jsonl"


def _read_rows(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                rows.append(record)
    return rows, malformed


def _ids(rows: list[dict]) -> tuple[list[int], list[int]]:
    """(valid ids in order, malformed/missing-id positions)."""
    ids: list[int] = []
    bad_positions: list[int] = []
    for position, row in enumerate(rows):
        value = row.get("idx")
        if isinstance(value, int) and value >= 0:
            ids.append(value)
        else:
            bad_positions.append(position)
    return ids, bad_positions


def verify_translation_alignment(
    source_path: Path | str,
    target_path: Path | str,
) -> dict:
    """Compare the English source export with the Persian dataset.

    Read-only. Returns an aggregate, content-free report suitable for
    logging and for the runs 4/5 pre-flight gate.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    report: dict[str, Any] = {
        "source_file": str(source_path),
        "target_file": str(target_path),
    }
    if not source_path.exists():
        report.update({
            "status": "SOURCE_MISSING",
            "source_count": 0,
            "target_count": 0,
            "eligible_count": 0,
            "coverage_pct": 0.0,
            "complete": False,
            "structurally_valid": False,
        })
        return report
    if not target_path.exists():
        report.update({
            "status": "TARGET_MISSING",
            "source_count": None,
            "target_count": 0,
            "eligible_count": 0,
            "coverage_pct": 0.0,
            "complete": False,
            "structurally_valid": False,
            "reason": "the Persian dataset does not exist yet",
        })
        return report

    source_rows, source_malformed = _read_rows(source_path)
    target_rows, target_malformed = _read_rows(target_path)
    source_ids, source_bad = _ids(source_rows)
    target_ids, target_bad = _ids(target_rows)

    source_set = set(source_ids)
    target_set = set(target_ids)
    source_dupes = sorted(
        {i for i in source_ids if source_ids.count(i) > 1}
    )
    target_dupes = sorted(
        {i for i in target_ids if target_ids.count(i) > 1}
    )
    missing_ids = sorted(source_set - target_set)
    unexpected_ids = sorted(target_set - source_set)

    source_by_id = {row["idx"]: row for row in source_rows
                    if isinstance(row.get("idx"), int)}
    status_distribution: dict[str, int] = {}
    eligible_ids: list[int] = []
    empty_translated: list[int] = []
    structural_mismatches: list[int] = []
    suspicious_unchanged: list[int] = []
    persian_char_total = 0
    persian_char_total_eligible = 0
    eligible_chars = 0

    for row in target_rows:
        idx = row.get("idx")
        if not isinstance(idx, int):
            continue
        status = row.get("status")
        status_distribution[status if isinstance(status, str) else "<missing>"] = (
            status_distribution.get(
                status if isinstance(status, str) else "<missing>", 0
            )
            + 1
        )
        if idx not in source_set or idx in target_dupes:
            continue  # unexpected/duplicated IDs reported separately
        source_row = source_by_id.get(idx, {})
        record_chars = 0
        field_ok = True
        for field in FIELDS:
            source_value = source_row.get(field)
            source_nonempty = (
                isinstance(source_value, str) and bool(source_value.strip())
            )
            key_present = field in row
            target_value = row.get(field)
            target_value_str = (
                str(target_value) if isinstance(target_value, str) else ""
            )
            target_nonempty = bool(target_value_str.strip())
            if isinstance(target_value, str):
                record_chars += sum(
                    1 for ch in target_value
                    if "؀" <= ch <= "ۿ"
                )
            # field-presence preservation (speaker-role/record structure): the
            # Persian key must exist exactly when the source field is non-empty
            # - a missing key where the source had content (or an extra one
            # where it did not) is a structural mismatch.
            if source_nonempty and not key_present:
                field_ok = False
                structural_mismatches.append(idx)
            elif not source_nonempty and key_present:
                field_ok = False
                structural_mismatches.append(idx)
            elif source_nonempty and key_present and not target_nonempty:
                # key exists but the translated field is blank while the source
                # had content - an empty translation (hard failure)
                field_ok = False
                empty_translated.append(idx)
            elif (
                source_nonempty
                and target_nonempty
                and target_value_str.strip() == str(source_value).strip()
            ):
                # soft warning: identical English text - legitimate for proper
                # nouns/abbreviations, suspicious for full sentences
                suspicious_unchanged.append(idx)
        if record_chars:
            persian_char_total += record_chars
        if status == TRANSLATED and field_ok:
            eligible_ids.append(idx)
            persian_char_total_eligible += record_chars
            eligible_chars += sum(
                len(str(row.get(field) or "")) for field in FIELDS
            )

    source_count = len(source_set)
    eligible_ids = sorted(eligible_ids)  # order-independent (stable IDs)
    eligible_count = len(eligible_ids)
    coverage_pct = (
        round(eligible_count / source_count * 100, 2) if source_count else 0.0
    )
    complete = (
        not missing_ids
        and not unexpected_ids
        and not target_dupes
        and not source_dupes
        and not structural_mismatches
        and not empty_translated
        and not source_malformed
        and not target_malformed
        and eligible_count == source_count
        and source_count > 0
    )
    structurally_valid = (
        not unexpected_ids
        and not target_dupes
        and not structural_mismatches
        and not empty_translated
        and not target_malformed
    )
    report.update({
        "status": "OK" if (complete or structurally_valid) else "STRUCTURAL_FAILURES",
        "source_count": source_count,
        "target_count": len(target_set),
        "eligible_count": eligible_count,
        "coverage_pct": coverage_pct,
        "missing_ids_count": len(missing_ids),
        "missing_ids_sample": missing_ids[:20],
        "unexpected_ids_count": len(unexpected_ids),
        "duplicate_target_ids": target_dupes,
        "duplicate_source_ids": source_dupes,
        "source_malformed_records": source_malformed,
        "target_malformed_records": target_malformed,
        "structural_mismatch_records": structural_mismatches,
        "empty_translated_records": empty_translated,
        "suspicious_unchanged_count": len(sorted(set(suspicious_unchanged))),
        "status_distribution": status_distribution,
        "persian_chars_total": persian_char_total,
        "persian_chars_eligible": persian_char_total_eligible,
        "eligible_chars_total": eligible_chars,
        "eligible_source_idxs": eligible_ids,
        "complete": complete,
        "structurally_valid": structurally_valid,
    })
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: read-only source↔Persian alignment validation (Issue #6)."""
    import argparse
    import sys

    import config

    parser = argparse.ArgumentParser(
        description=(
            "Verify that the Persian dataset corresponds to the English "
            "source export (IDs, structure, status, coverage). Read-only."
        )
    )
    parser.add_argument(
        "--source", type=str,
        default=str(config.DATA_DIR / SOURCE_EXPORT_NAME),
    )
    parser.add_argument(
        "--target", type=str,
        default=str(config.PERSIAN_DATASET_PATH),
    )
    parser.add_argument(
        "--allow-partial-coverage", action="store_true",
        help="tolerate missing source IDs (coverage is reported); "
        "structural failures still refuse runs 4/5",
    )
    args = parser.parse_args(argv)
    report = verify_translation_alignment(args.source, args.target)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") == "TARGET_MISSING":
        print(
            "[alignment] the Persian dataset does not exist — runs 4/5 are "
            "blocked until the real translation is produced",
            file=sys.stderr,
        )
        return 2
    if not report["structurally_valid"]:
        print(
            "[alignment] structural failures — runs 4/5 are refused",
            file=sys.stderr,
        )
        return 2
    if not report["complete"] and not args.allow_partial_coverage:
        print(
            f"[alignment] coverage {report['coverage_pct']}% is incomplete "
            f"({report['missing_ids_count']} missing source IDs) and "
            "partial coverage was not authorized — refusing runs 4/5",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
