"""Issue #6 — translation pipeline safety: test isolation, mock guards,
source-to-Persian alignment validation, and run 4/5 integration gates.

All fixtures are synthetic and live in isolated temporary directories. No
external provider is contacted; no real dataset or artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import config
import dataset_guard
import translation_alignment
from dataset_guard import MockDatasetError, assert_real_dataset

REPO = Path(__file__).resolve().parents[1]

# Files that must never be modified by tests or pipeline pre-flight gates.
PROTECTED_DATA_PATHS = (
    "data/mentalchat16k_en_for_translation.jsonl",
    "data/mentalchat16k_en_for_translation.export_meta.json",
    "data/mentalchat16k_fa.mock_sample.jsonl",
    "data/translation_errors.jsonl",
    "data/translation_mock_summary.json",
    "data/translation_summary.json",
    "data/translation_review.jsonl",
    "data/gold/gold_split_manifest.json",
)


def _file_states() -> dict[str, str | None]:
    states: dict[str, str | None] = {}
    for rel in PROTECTED_DATA_PATHS:
        path = REPO / rel
        if path.exists():
            states[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            states[rel] = None
    return states


def _protected_unchanged(before: dict[str, str | None]) -> None:
    assert _file_states() == before, "protected data files were modified"


def _write_source(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _source_records(n: int = 3) -> list[dict]:
    return [
        {
            "idx": i,
            "instruction": f"synthetic counselor prompt {i}",
            "input": f"synthetic patient message {i}",
            "output": f"synthetic counselor response {i}",
        }
        for i in range(n)
    ]


def _fa_target_rows(
    source: list[dict],
    *,
    persian: bool = True,
    status: str = "translated",
    model: str | None = None,
    drop_ids: set[int] | None = None,
) -> list[dict]:
    rows = []
    for record in source:
        if drop_ids and record["idx"] in drop_ids:
            continue
        suffix = "" if persian else ""
        marker = "فارسی" if persian else "english"
        row = {
            "idx": record["idx"],
            "instruction": f"پرامپت مشاور {record['idx']} {marker}"
            if persian
            else record["instruction"],
            "input": f"پیام مراجع {record['idx']} {marker}"
            if persian
            else record["input"],
            "output": f"پاسخ مشاور {record['idx']} {marker}"
            if persian
            else record["output"],
            "instruction_en": record["instruction"],
            "input_en": record["input"],
            "output_en": record["output"],
            "status": status,
        }
        if model:
            row["model"] = model
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# §2 — test isolation / protected data integrity
# --------------------------------------------------------------------------


def test_cli_provider_failure_writes_only_temporary_error_log(
    tmp_path, monkeypatch
):
    """A provider failure must write the error log ONLY to the explicitly
    supplied temporary path; protected data files stay byte-identical."""
    import translate_dataset as td

    src = _write_source(tmp_path / "src.jsonl", _source_records(3))
    real_before = _file_states()

    class FailingProvider(td.MockProvider):
        def translate(self, segments):
            raise td.TranslationError("api", "synthetic provider outage")

    monkeypatch.setattr(td, "MockProvider", lambda: FailingProvider())
    error_log = tmp_path / "errors.jsonl"
    summary_out = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(tmp_path / "out.jsonl"),
        "--summary-out", str(summary_out),
        "--error-log", str(error_log),
    ])
    assert td.main() == 1  # finished with failed records

    assert error_log.exists()  # explicit temporary destination
    # the temporary summary output is actually used and reflects the failure
    assert summary_out.exists()
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["engine"]["translated"] == 0
    assert summary["engine"]["total_source"] == 3
    _protected_unchanged(real_before)  # nothing else was touched


def test_cli_default_error_log_path_is_the_documented_production_path():
    """Production behavior is preserved: the CLI default remains the real
    data/ path (tests must opt out explicitly, not via a global patch)."""
    assert td_default_error_log() == str(
        config.DATA_DIR / "translation_errors.jsonl"
    )


def td_default_error_log() -> str:
    import translate_dataset as td

    # parse_args(argv) receives the argument list WITHOUT the program name
    args = td.parse_args([
        "--provider", "mock",
        "--src", "whatever.jsonl", "--out", "out.jsonl",
    ])
    return args.error_log


def test_retry_and_resume_tests_do_not_touch_protected_paths(
    tmp_path, monkeypatch
):
    """Running the interrupted-job CLI sequence twice must leave every
    protected data file byte-identical when explicit temporary error-log
    paths are used."""
    import translate_dataset as td

    src = _write_source(tmp_path / "src.jsonl", _source_records(4))
    real_before = _file_states()

    class FailOnceProvider(td.MockProvider):
        def translate(self, segments):
            raise td.TranslationError("api", "synthetic outage")

    monkeypatch.setattr(td, "MockProvider", lambda: FailOnceProvider())
    summary_out = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(tmp_path / "out.jsonl"),
        "--summary-out", str(summary_out),
        "--error-log", str(tmp_path / "errors.jsonl"),
    ])
    assert td.main() == 1
    assert summary_out.exists()  # temporary summary used, not the real one
    _protected_unchanged(real_before)


def test_protected_data_paths_exist_and_are_hashable():
    """The integrity guard itself is meaningful: the protected paths exist
    (so hashing them is not a silent no-op)."""
    states = _file_states()
    present = [rel for rel, sha in states.items() if sha is not None]
    assert len(present) >= 4


# --------------------------------------------------------------------------
# §3 — mock dataset protection
# --------------------------------------------------------------------------


def test_known_mock_filename_is_rejected(tmp_path):
    mock_file = tmp_path / "mentalchat16k_fa.mock_sample.jsonl"
    mock_file.write_text("{}", encoding="utf-8")
    with pytest.raises(MockDatasetError):
        assert_real_dataset(mock_file)


def test_mock_provider_metadata_is_rejected(tmp_path):
    clean_file = tmp_path / "renamed_but_clean.jsonl"
    clean_file.write_text("{}", encoding="utf-8")
    rows = _fa_target_rows(_source_records(2))
    for row in rows:
        row["provider"] = "mock"
    with pytest.raises(MockDatasetError):
        assert_real_dataset(clean_file, rows=rows)


def test_mock_echo_model_metadata_is_rejected(tmp_path):
    clean_file = tmp_path / "renamed_but_clean.jsonl"
    clean_file.write_text("{}", encoding="utf-8")
    rows = _fa_target_rows(_source_records(2), model="mock-echo")
    with pytest.raises(MockDatasetError):
        assert_real_dataset(clean_file, rows=rows)


def test_renamed_mock_file_with_mock_metadata_is_rejected(tmp_path):
    """Renaming the mock file does not bypass the guard when the metadata
    still identifies it as mock."""
    renamed = tmp_path / "totally_real_dataset.jsonl"
    renamed.write_text("{}", encoding="utf-8")
    rows = _fa_target_rows(_source_records(2), model="mock-echo")
    with pytest.raises(MockDatasetError):
        assert_real_dataset(renamed, rows=rows)


def test_legitimate_persian_dataset_with_provenance_is_accepted(tmp_path):
    clean_file = tmp_path / "mentalchat16k_fa.jsonl"
    rows = _fa_target_rows(_source_records(3))
    with clean_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert_real_dataset(clean_file)  # must not raise


def test_sample_filename_without_mock_provenance_is_accepted(tmp_path):
    """A 'sample' filename alone must NOT trigger rejection (the guard is
    provenance-based, not filename-based)."""
    file = tmp_path / "mysample.jsonl"
    rows = _fa_target_rows(_source_records(2))
    with file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert_real_dataset(file)  # must not raise


def test_mock_content_markers_in_text_are_rejected(tmp_path):
    file = tmp_path / "clean_named.jsonl"
    rows = _fa_target_rows(_source_records(2))
    for row in rows:
        row["input"] = "[MOCK-FA] synthetic placeholder"
    with pytest.raises(MockDatasetError):
        assert_real_dataset(file, rows=rows)


def test_runs_4_and_5_both_enforce_mock_guard(tmp_path, monkeypatch):
    """Both comparison runs refuse the mock dataset BEFORE embedding
    (run_pipeline would raise if it were ever invoked)."""
    import compare_runs as cr

    mock_file = tmp_path / "mentalchat16k_fa.mock_sample.jsonl"
    mock_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "comparison")
    monkeypatch.setattr(
        config, "PERSIAN_DATASET_PATH", mock_file
    )

    def _boom(*a, **kw):  # runs 1-3 must not even start
        raise AssertionError("run_pipeline must not run for a mock dataset")

    monkeypatch.setattr(cr, "run_pipeline", _boom)
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", "--only", "4,5",
        "--dataset-path", str(mock_file),
    ])
    assert cr.main() == 2  # refused before any embedding


# --------------------------------------------------------------------------
# §4 — source-to-Persian alignment validation
# --------------------------------------------------------------------------


def _aligned_pair(tmp_path: Path, n: int = 3):
    source = _write_source(tmp_path / "source.jsonl", _source_records(n))
    target = tmp_path / "target.jsonl"
    rows = _fa_target_rows(_source_records(n))
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return source, target, rows


def test_complete_aligned_pair_is_complete(tmp_path):
    source, target, _ = _aligned_pair(tmp_path, n=3)
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["complete"] is True
    assert report["structurally_valid"] is True
    assert report["eligible_count"] == 3
    assert report["coverage_pct"] == 100.0
    assert report["missing_ids_count"] == 0
    assert report["duplicate_target_ids"] == []
    # stable-ID based, not row-order based: shuffle the target rows and
    # the result must be identical
    import random as _random

    shuffled = list(_read_rows_shuffled(target))
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in shuffled) + "\n",
        encoding="utf-8",
    )
    report2 = translation_alignment.verify_translation_alignment(source, target)
    assert report2["eligible_source_idxs"] == report["eligible_source_idxs"]


def _read_rows_shuffled(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    import random as _r

    _r.Random(11).shuffle(rows)
    return rows


def test_missing_and_unexpected_ids_reported(tmp_path):
    source, target, _ = _aligned_pair(tmp_path, n=4)
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # drop idx 1 (missing), add idx 999 (unexpected)
    rows = [r for r in rows if r["idx"] != 1]
    rows.append({
        "idx": 999,
        "instruction": "غ", "input": "غ", "output": "غ",
        "instruction_en": "x", "input_en": "x", "output_en": "x",
        "status": "translated",
    })
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["missing_ids_count"] == 1
    assert report["unexpected_ids_count"] == 1
    # an unexpected Persian record has no source counterpart — a structural
    # failure even though the missing source ID would only be a coverage gap
    assert report["structurally_valid"] is False
    assert report["complete"] is False


def test_duplicate_target_ids_reported(tmp_path):
    source, target, rows = _aligned_pair(tmp_path, n=3)
    duplicated = rows + [rows[0]]
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in duplicated) + "\n",
        encoding="utf-8",
    )
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["duplicate_target_ids"] == [0]
    assert report["structurally_valid"] is False


def test_empty_translated_field_is_structural_failure(tmp_path):
    source, target, rows = _aligned_pair(tmp_path, n=2)
    rows[1]["output"] = ""
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["empty_translated_records"] == [1]
    assert report["structurally_valid"] is False


def test_structure_mismatch_field_presence(tmp_path):
    source, target, rows = _aligned_pair(tmp_path, n=2)
    # source has a non-empty instruction; the target drops it entirely
    del rows[0]["instruction"]
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["structural_mismatch_records"] == [rows[0]["idx"]]
    assert report["structurally_valid"] is False


def test_suspicious_unchanged_english_is_counted_as_warning(tmp_path):
    source, target, _ = _aligned_pair(tmp_path, n=2)
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # the target leaves the English output unchanged (a proper noun may
    # legitimately survive; the validator counts it as suspicious, not fatal)
    for row in rows:
        row["output"] = row["output_en"]  # unchanged English source text
    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["suspicious_unchanged_count"] == 2
    # warnings do not invalidate structure
    assert report["structurally_valid"] is True


def test_target_missing_reported_honestly(tmp_path):
    source = _write_source(tmp_path / "source.jsonl", _source_records(2))
    report = translation_alignment.verify_translation_alignment(
        source, tmp_path / "does_not_exist.jsonl"
    )
    assert report["status"] == "TARGET_MISSING"
    assert report["coverage_pct"] == 0.0
    assert report["structurally_valid"] is False


def test_eligible_set_identified_for_both_runs(tmp_path):
    """The eligible ID list is the explicit shared population for runs 4/5."""
    source, target, _ = _aligned_pair(tmp_path, n=3)
    report = translation_alignment.verify_translation_alignment(source, target)
    assert report["eligible_source_idxs"] == [0, 1, 2]


# --------------------------------------------------------------------------
# §5/§6 — QA MOCK_PROVIDER severity + run gate integration
# --------------------------------------------------------------------------


def test_qa_flags_mock_provider_as_hard_failure(tmp_path):
    from translation_qa import validate_translation_file

    fa_file = tmp_path / "fa.jsonl"
    rows = _fa_target_rows(_source_records(2), model="mock-echo")
    with fa_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = validate_translation_file(fa_file)
    assert report["issues_by_code"].get("MOCK_PROVIDER") == 2


def test_verify_translation_alignment_cli_matrix(tmp_path, monkeypatch):
    import translation_alignment as ta

    # target missing -> exit 2
    source = _write_source(tmp_path / "source.jsonl", _source_records(2))
    rc = ta.main([
        "--source", str(source),
        "--target", str(tmp_path / "missing.jsonl"),
    ])
    assert rc == 2

    # incomplete coverage -> exit 2 without --allow-partial-coverage
    source3, target3, _ = _aligned_pair(tmp_path, n=4)
    rows = [
        json.loads(line)
        for line in target3.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["idx"] != 3]
    target3.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    rc = ta.main(["--source", str(source3), "--target", str(target3)])
    assert rc == 2
    rc = ta.main([
        "--source", str(source3), "--target", str(target3),
        "--allow-partial-coverage",
    ])
    assert rc == 0

    # complete -> exit 0
    source_full, target_full, _ = _aligned_pair(tmp_path, n=3)
    rc = ta.main(["--source", str(source_full), "--target", str(target_full)])
    assert rc == 0


def test_compare_runs_strict_refusal_on_incomplete_coverage(
    tmp_path, monkeypatch
):
    """Strict mode: incomplete coverage refuses runs 4/5 before embedding;
    run_pipeline would raise if it were ever invoked."""
    import compare_runs as cr

    source = _write_source(
        tmp_path / "mentalchat16k_en_for_translation.jsonl",
        _source_records(4),
    )
    target = tmp_path / "mentalchat16k_fa.jsonl"
    rows = _fa_target_rows(_source_records(4))
    rows = [r for r in rows if r["idx"] != 3]  # incomplete coverage
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "comparison")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    def _boom(*a, **kw):
        raise AssertionError("runs must be refused before run_pipeline")

    monkeypatch.setattr(cr, "run_pipeline", _boom)
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", "--only", "4,5",
        "--dataset-path", str(target),
    ])
    assert cr.main() == 2


def test_compare_runs_partial_mode_uses_same_eligible_set(
    tmp_path, monkeypatch
):
    """--allow-partial-coverage: both fa runs proceed on the SAME eligible
    population; coverage is attached to both run summaries."""
    import compare_runs as cr

    source = _write_source(
        tmp_path / "mentalchat16k_en_for_translation.jsonl",
        _source_records(4),
    )
    target = tmp_path / "mentalchat16k_fa.jsonl"
    rows = _fa_target_rows(_source_records(4))
    rows = [r for r in rows if r["idx"] != 3]  # incomplete: idx 3 missing
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "comparison")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    captured: dict = {}

    def _fake_run_pipeline(**kwargs):
        captured.setdefault("runs", []).append(kwargs["output_dir"])
        # minimal report for summarize_run
        from facts import CATEGORY_KEYS

        per = {
            c: {"count": 1 if c == "B" else 0, "share": 0.0,
                "multi_count": 0, "multi_share": 0.0, "mean_score": 0.5,
                "median_score": 0.5, "std_score": 0.0,
                "confident_share": 0.0, "global_mean_score": 0.5}
            for c in CATEGORY_KEYS
        }
        return {
            "n_samples": 3, "threshold": 0.35, "n_confident": 3,
            "confident_rate": 1.0, "n_unlabeled_NONE": 0,
            "per_category": per,
            "global_mean_score_per_category": {c: 0.5 for c in CATEGORY_KEYS},
            "ranking_weak_to_strong": list(CATEGORY_KEYS),
            "weakest": list(CATEGORY_KEYS)[:2],
            "strongest": list(CATEGORY_KEYS)[-2:],
        }

    monkeypatch.setattr(cr, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", "--only", "4,5",
        "--dataset-path", str(target),
        "--allow-partial-coverage",
    ])
    assert cr.main() == 0
    assert sorted(Path(p).name for p in captured["runs"]) == [
        "4_fa_fa_bge-m3",
        "5_fa_fa_e5-large",
    ]
    summary = json.loads(
        (tmp_path / "comparison" / "comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    coverage = [e.get("translation_coverage") for e in summary]
    assert all(c is not None for c in coverage)
    assert all(c["coverage_pct"] == 75.0 for c in coverage)
