"""CLI-level tests for the Phase-6 translation engine gates and reporting.

All provider interaction is the offline mock; no API is contacted and no
credits are consumed. Source records are synthetic.
"""
import json
from pathlib import Path

import pytest

import translation_qa as qa
import translate_dataset as td


def _write_src(path: Path, n: int) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "idx": i,
                "instruction": f"counselor prompt {i}",
                "input": f"patient message {i}",
                "output": f"counselor response {i}",
            }, ensure_ascii=False) + "\n")
    return path


def test_main_openai_requires_confirm_external_sharing(tmp_path, monkeypatch):
    src = _write_src(tmp_path / "src.jsonl", 2)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "openai",
        "--src", str(src), "--out", str(tmp_path / "out.jsonl"),
    ])
    assert td.main() == 2  # blocked before any provider/network use


def test_main_mock_refuses_production_dataset_path(tmp_path, monkeypatch):
    src = _write_src(tmp_path / "src.jsonl", 2)
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(td.config.PERSIAN_DATASET_PATH),
    ])
    assert td.main() == 2  # mock output must never become the production dataset


def test_main_mock_smoke_end_to_end(tmp_path, monkeypatch):
    src = _write_src(tmp_path / "src.jsonl", 5)
    out = tmp_path / "out.jsonl"
    summary_out = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(out),
        "--summary-out", str(summary_out), "--batch-size", "2",
    ])
    assert td.main() == 0
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 5
    assert {r["idx"] for r in lines} == {0, 1, 2, 3, 4}
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["engine"]["translated"] == 5
    assert summary["completeness"]["complete"] is True
    assert summary["completeness"]["missing"] == 0


def test_check_completeness_counts_gaps(tmp_path: Path):
    src = _write_src(tmp_path / "src.jsonl", 5)
    out = tmp_path / "out.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i in (0, 1, 2, 2):  # 3 unique translated, one duplicate
            f.write(json.dumps({"idx": i, "status": "translated"}, ensure_ascii=False) + "\n")
    report = qa.check_completeness(src, out)
    assert report["total_source"] == 5
    assert report["translated"] == 3
    assert report["duplicated_output_ids"] == 1
    assert report["missing"] == 2
    assert report["missing_idx_sample"] == [3, 4]
    assert report["complete"] is False


def test_check_completeness_full_coverage(tmp_path: Path):
    src = _write_src(tmp_path / "src.jsonl", 3)
    out = tmp_path / "out.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"idx": i, "status": "translated"}, ensure_ascii=False) + "\n")
    report = qa.check_completeness(src, out)
    assert report["complete"] is True and report["missing"] == 0


def test_interrupted_job_recovery_via_cli(tmp_path, monkeypatch):
    """Simulate an interrupted run: partial output, then resume completes it."""
    src = _write_src(tmp_path / "src.jsonl", 6)
    out = tmp_path / "out.jsonl"

    class FailAfterProvider(td.MockProvider):
        """Translates only the first N segments, then fails permanently."""

        def __init__(self, succeed_segments: int):
            super().__init__()
            self.remaining = succeed_segments

        def translate(self, segments):
            if self.remaining <= 0:
                raise td.TranslationError("api", "interrupted")
            self.remaining -= len(segments)
            return super().translate(segments)

    real_provider = FailAfterProvider(succeed_segments=4)  # 1 record + 1 field
    original_mock = td.MockProvider  # capture before patching
    monkeypatch.setattr(
        td, "MockProvider", lambda: real_provider, raising=True
    )
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(out),
        "--summary-out", str(tmp_path / "s1.json"), "--batch-size", "2",
        "--max-retries", "0",
    ])
    assert td.main() == 1  # finished with failed records

    # Resume with a healthy provider: only the missing records are sent.
    monkeypatch.setattr(td, "MockProvider", original_mock, raising=True)
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(out),
        "--summary-out", str(tmp_path / "s2.json"), "--batch-size", "3",
    ])
    assert td.main() == 0
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    idxs = [r["idx"] for r in lines]
    assert sorted(idxs) == list(range(6))
    assert len(idxs) == len(set(idxs))  # no duplicates after resume
