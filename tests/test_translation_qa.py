"""Tests for translation_qa validation utilities (synthetic records only)."""
import json
from pathlib import Path

import pytest

import translation_qa as qa
from translation_qa import (
    validate_record,
    validate_translation_file,
    write_review_file,
)


def _record(idx: int = 0, **overrides) -> dict:
    base = {
        "idx": idx,
        "instruction": "دستیار مشاوره سلامت روان هستید.",
        "input": "برنامه‌ی روزانه‌ام که عوض می‌شود مضطرب می‌شوم.",
        "output": "ساختار و پیش‌بینی‌پذیری معمولاً کمک‌کننده است.",
        "instruction_en": "You are a mental health counselling assistant.",
        "input_en": "I get anxious when my routine changes.",
        "output_en": "Structure and predictability usually help.",
        "status": "translated",
        "model": "test",
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, records: list) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(rec, ensure_ascii=False) if isinstance(rec, dict) else rec
            )
            f.write("\n")
    return path


# --------------------------------------------------------------------------
# Per-record validation
# --------------------------------------------------------------------------


def test_valid_record_has_no_issues():
    issues, idx = validate_record(_record(), position=0)
    assert issues == []
    assert idx == 0


def test_missing_translation_field_detected():
    rec = _record()
    del rec["output"]  # translated Persian field absent
    issues, _ = validate_record(rec, position=0)
    codes = {i.code for i in issues}
    assert "MISSING_TRANSLATION" in codes


def test_empty_translated_field_flagged_when_source_present():
    issues, _ = validate_record(_record(output="   "), position=0)
    codes = {i.code for i in issues}
    assert "EMPTY_TRANSLATED_FIELD" in codes


def test_empty_input_field_ok_when_english_empty():
    issues, _ = validate_record(_record(input="", input_en=""), position=0)
    assert issues == []


def test_missing_original_english_field_flagged():
    rec = _record()
    del rec["output_en"]  # original content dropped
    issues, _ = validate_record(rec, position=0)
    codes = {i.code: i.severity for i in issues}
    assert codes.get("MISSING_ORIGINAL_FIELD") == "warning"


def test_corrupted_unicode_detected():
    for corrupted in ("متن با کاراکتر خراب \uFFFD", "Ø¯Ø§Ù†Ù„ÙˆØ¯ mojibake"):
        issues, _ = validate_record(_record(input=corrupted), position=0)
        assert any(i.code == "CORRUPTED_UNICODE" for i in issues)


def test_non_persian_translation_detected():
    issues, _ = validate_record(_record(input="totally english text here"), position=0)
    assert any(i.code == "NOT_PERSIAN" and i.severity == "error" for i in issues)


def test_suspiciously_short_translation_detected():
    issues, _ = validate_record(
        _record(output="کوتاه", output_en="x" * 200), position=0
    )
    assert any(i.code == "SUSPICIOUSLY_SHORT" for i in issues)


def test_suspiciously_long_translation_detected():
    issues, _ = validate_record(
        _record(output="سلام " * 60, output_en="hi"), position=0
    )
    assert any(i.code == "SUSPICIOUSLY_LONG" for i in issues)


def test_bad_status_flagged():
    issues, _ = validate_record(_record(status="weird"), position=0)
    assert any(i.code == "BAD_STATUS" for i in issues)


def test_non_object_record_flagged():
    issues, idx = validate_record(["not", "a", "dict"], position=4)
    assert issues[0].code == "INVALID_RECORD"
    assert idx is None


# --------------------------------------------------------------------------
# File-level validation
# --------------------------------------------------------------------------


def test_validate_file_counts_and_duplicates(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "fa.jsonl",
        [_record(0), _record(0), _record(1, status="flagged", flag_reason="ambiguous")],
    )
    report = validate_translation_file(path)
    assert report["n_records"] == 3
    assert report["duplicates"] == 1
    assert report["completion"] == {"translated": 2, "flagged": 1}
    assert any(i["code"] == "DUPLICATE_IDX" for i in report["issues"])
    assert 1 in report["flagged_idx"]  # only idx 1 has status="flagged"


def test_validate_file_skips_malformed_lines(tmp_path: Path):
    path = _write_jsonl(tmp_path / "fa.jsonl", [_record(0), "{not json", _record(1)])
    report = validate_translation_file(path)
    assert report["n_records"] == 2
    assert report["malformed_lines"] == 1


def test_validate_file_missing_reports_file_missing(tmp_path: Path):
    report = validate_translation_file(tmp_path / "nope.jsonl")
    assert report["exists"] is False
    assert report["issues_by_code"].get("FILE_MISSING") == 1


def test_report_never_contains_conversation_content(tmp_path: Path):
    secret_en = "I get anxious when my routine changes."
    secret_fa = "برنامه‌ی روزانه‌ام که عوض می‌شود مضطرب می‌شوم."
    path = _write_jsonl(
        tmp_path / "fa.jsonl",
        [
            _record(0),
            _record(1, status="flagged", flag_reason="ambiguous", input=secret_fa),
            "{not json",
        ],
    )
    report = validate_translation_file(path)
    dumped = json.dumps(report, ensure_ascii=False)
    assert secret_en not in dumped
    assert secret_fa not in dumped
    assert "anxious" not in dumped


def test_write_review_file_selects_flagged_and_issue_records(tmp_path: Path):
    flagged = _record(1, status="flagged", flag_reason="ambiguous")
    empty_issue = _record(2, output="   ")
    good = _record(0)
    path = _write_jsonl(tmp_path / "fa.jsonl", [good, flagged, empty_issue])
    report = validate_translation_file(path)
    review_path = tmp_path / "review.jsonl"
    count = write_review_file(path, review_path, report)
    assert count == 2
    reviewed = [json.loads(l) for l in review_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["idx"] for r in reviewed} == {1, 2}
    assert all("qa_issues" in r for r in reviewed)
    assert any("EMPTY_TRANSLATED_FIELD" in r["qa_issues"] for r in reviewed if r["idx"] == 2)


def test_write_review_file_empty_when_all_valid(tmp_path: Path):
    path = _write_jsonl(tmp_path / "fa.jsonl", [_record(0), _record(1)])
    report = validate_translation_file(path)
    review_path = tmp_path / "review.jsonl"
    assert write_review_file(path, review_path, report) == 0
    assert not review_path.exists()


def test_cli_exit_codes(tmp_path: Path, monkeypatch, capsys):
    good_file = _write_jsonl(tmp_path / "ok.jsonl", [_record(0)])
    monkeypatch.setattr("sys.argv", ["translation_qa.py", "--file", str(good_file)])
    assert qa.main() == 0

    monkeypatch.setattr(
        "sys.argv", ["translation_qa.py", "--file", str(tmp_path / "missing.jsonl")]
    )
    # main() returns the exit code; SystemExit is raised only by the
    # ``sys.exit(main())`` wrapper under ``__main__``.
    assert qa.main() == 2
    capsys.readouterr()
