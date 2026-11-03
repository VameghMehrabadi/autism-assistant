"""Automated quality control for the translated Persian dataset (Phase 6).

Checks every record of ``data/mentalchat16k_fa.jsonl`` for:

- missing / empty translated fields
- invalid JSONL lines and non-object records
- duplicate record IDs
- missing original English fields
- corrupted Unicode (replacement chars / mojibake patterns)
- unexpected structural changes (translated field without an original)
- suspiciously short or long translations (length-ratio heuristics)
- non-Persian translated text (script-based language detection)
- translation completion status counts

Reports contain **no conversation content** — only record indices, issue
codes, and lengths. Automated QA validates structure and plausibility only;
it does NOT prove semantic accuracy. Records with issues are written to a
review file for manual inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import config
from data_validation import detect_language
from translate_dataset import SOURCE_FIELDS, STATUS_EMPTY_SOURCE, STATUS_FLAGGED, STATUS_TRANSLATED

VALID_STATUSES = {STATUS_TRANSLATED, STATUS_FLAGGED, STATUS_EMPTY_SOURCE}

# Length-ratio heuristics (Persian vs English character counts are usually
# comparable; boundaries are deliberately generous to avoid false alarms).
SHORT_RATIO = 0.3
LONG_RATIO = 2.5

# Common mojibake signatures for UTF-8 Persian misread as cp1252/latin-1,
# plus the Unicode replacement character.
_MOJIBAKE_PATTERNS: tuple[str, ...] = ("Ø", "Ù", "Û", "â€", "Ã˜", "\uFFFD")

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class QAIssue:
    idx: int | str
    code: str
    field: str
    detail: str  # counts/codes only — never record content
    severity: str = SEVERITY_ERROR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_corrupted_unicode(text: str) -> bool:
    return any(pattern in text for pattern in _MOJIBAKE_PATTERNS)


def validate_record(record: Any, position: int) -> tuple[list[QAIssue], int | str | None]:
    """Validate one translated record; returns (issues, idx-or-None)."""
    issues: list[QAIssue] = []
    if not isinstance(record, dict):
        return (
            [QAIssue(position, "INVALID_RECORD", "-", "record is not a JSON object")],
            None,
        )
    raw_idx = record.get("idx", position)
    idx = int(raw_idx) if str(raw_idx).isdigit() else position

    # Original English fields must be present (empty string is legitimate,
    # absent keys mean the original content was dropped).
    for field_name in SOURCE_FIELDS:
        if f"{field_name}_en" not in record:
            issues.append(
                QAIssue(idx, "MISSING_ORIGINAL_FIELD", field_name,
                        f"{field_name}_en key absent", SEVERITY_WARNING)
            )

    status = record.get("status")
    if status not in VALID_STATUSES:
        issues.append(
            QAIssue(idx, "BAD_STATUS", "-", f"unknown status {status!r}", SEVERITY_WARNING)
        )

    if status == STATUS_EMPTY_SOURCE:
        return issues, idx

    for field_name in SOURCE_FIELDS:
        fa = record.get(field_name)
        en = record.get(f"{field_name}_en", "")

        if fa is None:
            issues.append(
                QAIssue(idx, "MISSING_TRANSLATION", field_name, "translated field absent")
            )
            continue
        fa = str(fa)
        if not fa.strip():
            if status == STATUS_TRANSLATED and str(en).strip():
                issues.append(
                    QAIssue(idx, "EMPTY_TRANSLATED_FIELD", field_name,
                            f"translated field empty while EN length={len(str(en))}")
                )
            continue

        if _is_corrupted_unicode(fa):
            issues.append(
                QAIssue(idx, "CORRUPTED_UNICODE", field_name,
                        "replacement char or mojibake signature found")
            )
        lang = detect_language(fa)
        if lang in {"en", "unknown"}:
            issues.append(
                QAIssue(idx, "NOT_PERSIAN", field_name, f"detected language={lang}")
            )
        elif lang == "mixed":
            issues.append(
                QAIssue(idx, "NOT_PERSIAN", field_name, "detected language=mixed",
                        SEVERITY_WARNING)
            )
        if str(en).strip():
            ratio = len(fa) / len(str(en))
            if ratio < SHORT_RATIO:
                issues.append(
                    QAIssue(idx, "SUSPICIOUSLY_SHORT", field_name,
                            f"len_fa={len(fa)} len_en={len(str(en))} ratio={ratio:.2f}")
                )
            elif ratio > LONG_RATIO:
                issues.append(
                    QAIssue(idx, "SUSPICIOUSLY_LONG", field_name,
                            f"len_fa={len(fa)} len_en={len(str(en))} ratio={ratio:.2f}")
                )
    return issues, idx


def validate_translation_file(path: str | Path, max_listed_issues: int = 200) -> dict[str, Any]:
    """Validate a translated JSONL file and return a content-free report."""
    path = Path(path)
    report: dict[str, Any] = {
        "file": str(path),
        "exists": path.exists(),
        "n_records": 0,
        "n_valid": 0,
        "malformed_lines": 0,
        "non_object_records": 0,
        "duplicates": 0,
        "completion": Counter(),
        "issues_total": 0,
        "issues_by_code": Counter(),
        "issues": [],          # capped list of issue dicts (no content)
        "flagged_idx": [],
        "schema_expected_keys": sorted(
            ["idx", "status", "model", *SOURCE_FIELDS, *(f"{f}_en" for f in SOURCE_FIELDS)]
        ),
    }
    if not path.exists():
        report["issues_total"] = 1
        report["issues_by_code"]["FILE_MISSING"] = 1
        report["issues"].append(
            QAIssue("-", "FILE_MISSING", "-", f"file not found: {path}").to_dict()
        )
        return _finalize(report)

    seen_idx: set[int | str] = set()
    listed = 0
    with path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                report["malformed_lines"] += 1
                continue
            issues, idx = validate_record(record, position)
            report["n_records"] += 1
            if idx is None:
                report["non_object_records"] += 1
            if idx in seen_idx:
                report["duplicates"] += 1
                issues = issues + [
                    QAIssue(idx, "DUPLICATE_IDX", "-", "duplicate record id")
                ]
            seen_idx.add(idx)
            if isinstance(record, dict):
                report["completion"][str(record.get("status", "unknown"))] += 1
                if record.get("status") == STATUS_FLAGGED:
                    report["flagged_idx"].append(idx)
                if not issues:
                    report["n_valid"] += 1
            for issue in issues:
                report["issues_total"] += 1
                report["issues_by_code"][issue.code] += 1
                if listed < max_listed_issues:
                    report["issues"].append(issue.to_dict())
                    listed += 1
    return _finalize(report)


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    report["completion"] = dict(report["completion"])
    report["issues_by_code"] = dict(report["issues_by_code"])
    return report


def write_review_file(
    translated_path: str | Path,
    review_path: str | Path,
    report: dict[str, Any] | None = None,
) -> int:
    """Write records needing manual review to ``review_path``.

    The review file is a *data* artifact (gitignored under ``data/``): it
    necessarily contains the text a human must read. Issue codes travel in
    the ``qa_issues`` key; logs and reports stay content-free.
    """
    report = report or validate_translation_file(translated_path)
    flagged_idx = set(report.get("flagged_idx", []))
    issue_idx = {issue["idx"] for issue in report.get("issues", []) if issue["code"] != "FILE_MISSING"}
    review_idx = flagged_idx | issue_idx
    if not review_idx:
        return 0
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(translated_path).open("r", encoding="utf-8") as src, \
            review_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            idx = record.get("idx")
            record_issues = [
                i for i in report.get("issues", [])
                if i["idx"] == idx or i["code"] == "DUPLICATE_IDX"
            ]
            if idx in review_idx or record_issues:
                record["qa_issues"] = [i["code"] for i in record_issues]
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def check_completeness(
    src_path: str | Path, translated_path: str | Path
) -> dict[str, Any]:
    """Verify every source record has a corresponding translation.

    Returns counts only (no content): total source records, translated
    (unique idx present in the output), missing (pending/failed), duplicate
    output ids, and a capped list of missing idx values so a resumed run can
    target exactly the gap. Failed records stay absent from the output by
    design, so they appear here as missing until a later run retries them.
    """
    src_path, translated_path = Path(src_path), Path(translated_path)
    # _read_jsonl lives in translate_dataset; import locally to avoid cycles.
    from translate_dataset import _read_jsonl  # noqa: PLC0415

    src_idx: list[int] = []
    src_rows, _ = _read_jsonl(src_path)
    for position, row in enumerate(src_rows):
        if isinstance(row, dict):
            raw_idx = row.get("idx", position)
            src_idx.append(int(raw_idx) if str(raw_idx).isdigit() else position)

    translated_idx: list[int] = []
    if translated_path.exists():
        out_rows, _ = _read_jsonl(translated_path)
        for row in out_rows:
            if isinstance(row, dict) and "idx" in row:
                try:
                    translated_idx.append(int(row["idx"]))
                except (TypeError, ValueError):
                    continue

    translated_set = set(translated_idx)
    missing = [i for i in src_idx if i not in translated_set]
    duplicated = len(translated_idx) - len(translated_set)
    return {
        "src": str(src_path),
        "translated_file": str(translated_path),
        "total_source": len(src_idx),
        "translated": len(translated_set),
        "duplicated_output_ids": duplicated,
        "missing": len(missing),
        "missing_idx_sample": sorted(missing)[:50],
        "complete": len(src_idx) > 0 and not missing,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Human-readable, content-free summary."""
    print(
        f"[qa] records={report['n_records']} valid={report['n_valid']} "
        f"duplicates={report['duplicates']} malformed_lines={report['malformed_lines']} "
        f"issues={report['issues_total']}"
    )
    print(f"[qa] completion: {report['completion']}")
    if report["issues_by_code"]:
        print(f"[qa] issues by code: {report['issues_by_code']}")
    if report["flagged_idx"]:
        preview = ", ".join(str(i) for i in report["flagged_idx"][:10])
        more = " …" if len(report["flagged_idx"]) > 10 else ""
        print(f"[qa] flagged for manual review: {preview}{more}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the translated Persian dataset.")
    parser.add_argument("--file", type=str,
                        default=str(config.PERSIAN_DATASET_PATH))
    parser.add_argument("--review-out", type=str,
                        default=str(config.DATA_DIR / "translation_review.jsonl"))
    args = parser.parse_args()
    report = validate_translation_file(args.file)
    print_summary(report)
    if report["exists"] and report["issues_total"]:
        reviewed = write_review_file(args.file, args.review_out, report)
        print(f"[qa] wrote {reviewed} record(s) needing manual review -> {args.review_out}")
    # Missing file or structural issues are exit-worthy; heuristic warnings are not.
    has_structural = any(
        code in report["issues_by_code"]
        for code in ("FILE_MISSING", "INVALID_RECORD", "DUPLICATE_IDX", "MISSING_TRANSLATION")
    )
    return 2 if has_structural else 0


if __name__ == "__main__":
    sys.exit(main())
