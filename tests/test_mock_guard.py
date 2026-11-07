"""Issue #25 — mock datasets must never masquerade as real translations.

Synthetic fixtures only; the real mock sample file is never required.
Covers: provenance detection, default refusal, explicit --allow-mock
opt-in, blocked comparison runs, UI provenance labels, and regression of
real-data loading.
"""
import json
from pathlib import Path

import pytest

import config
import data_loader
from data_loader import load_samples_fa
from data_validation import (
    detect_dataset_kind,
    detect_dataset_kind_file,
)


def _mock_record(i: int) -> dict:
    return {
        "idx": i,
        "instruction": "[MOCK-FA]prompt[/MOCK-FA]",
        "input": f"[MOCK-FA]message {i}[/MOCK-FA]",
        "output": "[MOCK-FA]reply[/MOCK-FA]",
        "instruction_en": f"prompt {i}",
        "input_en": f"message {i}",
        "output_en": f"reply {i}",
        "status": "translated",
        "model": "mock-echo",
    }


def _real_record(i: int) -> dict:
    return {
        "idx": i,
        "instruction": f"مشکل شماره {i}",
        "input": f"فرزندم مضطرب می‌شود {i}",
        "output": f"پاسخ {i}",
        "status": "translated",
        "model": "glm-5.3-flash",
    }


def _write(path: Path, records) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------
# Provenance detection
# --------------------------------------------------------------------------


def test_detect_kind_mock_by_content_marker():
    assert detect_dataset_kind([_mock_record(0)]) == "mock"


def test_detect_kind_mock_by_model_id():
    rec = _real_record(0)
    rec["model"] = "mock-echo"
    assert detect_dataset_kind([rec]) == "mock"


def test_detect_kind_real_for_genuine_translations():
    assert detect_dataset_kind([_real_record(i) for i in range(5)]) == "real"


def test_detect_kind_fixture_by_path(tmp_path: Path):
    fixture_dir = tmp_path / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True)
    path = _write(fixture_dir / "sample.jsonl", [_real_record(0)])
    assert detect_dataset_kind([], path=path) == "fixture"
    # content markers win over path
    _write(path, [_mock_record(0)])
    assert detect_dataset_kind_file(path) == "mock"


def test_detect_kind_file_missing(tmp_path: Path):
    assert detect_dataset_kind_file(tmp_path / "nope.jsonl") == "missing"


# --------------------------------------------------------------------------
# Loader gate: mock refused by default, allowed explicitly
# --------------------------------------------------------------------------


def test_loader_refuses_mock_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_MOCK_DATASET", False)
    path = _write(tmp_path / "mock.jsonl", [_mock_record(i) for i in range(3)])
    with pytest.raises(data_loader.DataValidationError) as excinfo:
        load_samples_fa(path=path)
    assert "MOCK DATA refused" in str(excinfo.value)


def test_loader_accepts_mock_only_with_explicit_opt_in(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_MOCK_DATASET", False)
    path = _write(tmp_path / "mock.jsonl", [_mock_record(i) for i in range(3)])
    samples, report = load_samples_fa(path=path, allow_mock=True, return_report=True)
    assert report.dataset_kind == "mock"
    assert len(samples) == 3
    assert all(s.meta["translation_status"] == "translated" for s in samples)


def test_real_translated_data_loads_normally(tmp_path: Path):
    """Regression: genuine translations are untouched by the mock gate."""
    path = _write(tmp_path / "real.jsonl", [_real_record(i) for i in range(3)])
    samples, report = load_samples_fa(path=path, return_report=True)
    assert report.dataset_kind == "real"
    assert len(samples) == 3
    # patient input is the labeling target (not the counselor output)
    assert samples[0].text == "فرزندم مضطرب می‌شود 0"
    assert samples[0].response_text == "پاسخ 0"


def test_mock_guard_not_bypassed_via_en_loader(tmp_path: Path, monkeypatch):
    """Even the EN loader refuses mock content (defense in depth)."""
    monkeypatch.setattr(config, "ALLOW_MOCK_DATASET", False)

    class _FakeDS:
        def __init__(self, rows):
            self._rows = rows
            self.features = {k: SimpleNS("string") for k in rows[0]}

        def __getitem__(self, i):
            return dict(self._rows[i])

        def __len__(self):
            return len(self._rows)

    class SimpleNS:
        def __init__(self, dtype):
            self.dtype = dtype

    rows = [_mock_record(i) for i in range(3)]
    fake_ds = _FakeDS(rows)
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **kw: fake_ds
    )
    with pytest.raises(data_loader.DataValidationError):
        data_loader.load_samples_en(limit=3)


# --------------------------------------------------------------------------
# Comparison pipeline: default blocked, explicit opt-in labeled
# --------------------------------------------------------------------------


def _minimal_gap_report(n=3, dataset_kind="real"):
    per_cat = {
        c: {"count": 1, "share": 0.1, "multi_count": 0, "multi_share": 0.0,
            "mean_score": 0.5, "median_score": 0.5, "std_score": 0.0,
            "confident_share": 1.0, "global_mean_score": 0.5}
        for c in "ABCDEFG"
    }
    return {
        "n_samples": n,
        "threshold": 0.35,
        "confident_rate": 1.0,
        "top_score_stats": {"mean": 0.5, "median": 0.5, "p25": 0.5,
                            "p75": 0.5, "min": 0.5, "max": 0.5},
        "per_category": per_cat,
        "global_mean_score_per_category": {c: 0.5 for c in "ABCDEFG"},
        "ranking_weak_to_strong": list("ABCDEFG"),
        "run": {"dataset_kind": dataset_kind,
                "eligible_for_production_comparison": dataset_kind == "real"},
    }


def _write_fake_run(base: Path, run_id: str, dataset_kind="real"):
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gap_analysis.json").write_text(
        json.dumps(
            {**_minimal_gap_report(dataset_kind=dataset_kind),
             "run": {"dataset_kind": dataset_kind,
                     "eligible_for_production_comparison": dataset_kind == "real",
                     "model_key": "glm-5.3-flash"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_compare_blocked_when_fa_missing(tmp_path, monkeypatch):
    """No real Persian JSONL → runs 4/5 recorded as BLOCKED; no mock
    substitution; no fabricated record counts."""
    import compare_runs

    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "cmp")
    monkeypatch.setattr(
        config, "PERSIAN_DATASET_PATH", tmp_path / "missing_fa.jsonl"
    )
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", "--only", "4,5", "--skip-existing",
    ])
    assert compare_runs.main() == 3  # report generated, runs BLOCKED

    summary = json.loads(
        (tmp_path / "cmp" / "comparison_summary.json").read_text(encoding="utf-8")
    )
    assert [s["id"] for s in summary] == ["4_fa_fa_bge-m3", "5_fa_fa_e5-large"]
    assert all(s["status"] == "BLOCKED" for s in summary)
    assert all(s["n_samples"] is None for s in summary)
    md = (tmp_path / "cmp" / "comparison_report.md").read_text(encoding="utf-8")
    assert "BLOCKED runs (NOT executed" in md
    assert "16,005" not in md and "16005" not in md


def test_compare_mock_opt_in_labeled_ineligible(tmp_path, monkeypatch):
    """--allow-mock admits the mock dataset, but the run is labeled
    dataset_kind=mock and ineligible for production claims."""
    import compare_runs

    mock_fa = tmp_path / "fa.jsonl"
    _write(mock_fa, [_mock_record(i) for i in range(4)])

    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "cmp")
    monkeypatch.setattr(config, "PERSIAN_DATASET_PATH", mock_fa)

    fake_report = _minimal_gap_report(n=4, dataset_kind="mock")
    fake_report["run"] = {
        "dataset_kind": "mock", "eligible_for_production_comparison": False,
        "model_key": "glm-5.3-flash", "text_window": 512, "prefix": "none",
    }
    # compare_runs imports run_pipeline by name — patch its own reference
    monkeypatch.setattr(compare_runs, "run_pipeline", lambda **kw: fake_report)
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", "--only", "4", "--skip-existing", "--allow-mock",
        "--dataset-path", str(mock_fa),
    ])
    assert compare_runs.main() == 0

    summary = json.loads(
        (tmp_path / "cmp" / "comparison_summary.json").read_text(encoding="utf-8")
    )
    run4 = summary[0]
    assert run4["dataset_kind"] == "mock"
    assert run4["eligible_for_production_comparison"] is False


def test_compare_fa_present_runs_normally(tmp_path, monkeypatch):
    """Regression: when a real FA file exists, runs are not blocked."""
    import compare_runs

    real_fa = tmp_path / "fa.jsonl"
    _write(real_fa, [_real_record(i) for i in range(4)])
    fake_report = _minimal_gap_report(n=4, dataset_kind="real")
    fake_report["run"] = {
        "dataset_kind": "real", "eligible_for_production_comparison": True,
    }
    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "cmp")
    monkeypatch.setattr(config, "PERSIAN_DATASET_PATH", real_fa)
    monkeypatch.setattr(
        compare_runs, "run_pipeline", lambda **kw: {
            **fake_report, "run": {
                "dataset_kind": "real",
                "eligible_for_production_comparison": True,
            }
        },
    )
    monkeypatch.setattr("sys.argv", [
        "compare_runs.py", "--only", "4", "--skip-existing",
        "--dataset-path", str(real_fa),
    ])
    assert compare_runs.main() == 0
    summary = json.loads(
        (tmp_path / "cmp" / "comparison_summary.json").read_text(encoding="utf-8")
    )
    assert summary[0]["status"] != "BLOCKED"
    assert summary[0]["dataset_kind"] == "real"
    assert summary[0]["eligible_for_production_comparison"] is True


# --------------------------------------------------------------------------
# UI provenance labels
# --------------------------------------------------------------------------


def test_widget_provenance_labels(tmp_path: Path):
    from ui_lib.widgets import (
        FIXTURE_DATA_LABEL,
        MOCK_DATA_LABEL,
        NOT_TRANSLATED_FA,
        dataset_provenance,
    )

    kind, label = dataset_provenance(tmp_path / "missing.jsonl")
    assert kind == "missing" and label == NOT_TRANSLATED_FA

    kind, label = dataset_provenance(
        _write(tmp_path / "mock.jsonl", [_mock_record(0)])
    )
    assert kind == "mock" and label == MOCK_DATA_LABEL
    assert "NOT REAL TRANSLATION" in label

    fixture_dir = tmp_path / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True)
    kind, label = dataset_provenance(
        _write(fixture_dir / "sample.jsonl", [_real_record(0)])
    )
    assert kind == "fixture" and "TEST FIXTURE" in label

    kind, label = dataset_provenance(
        _write(tmp_path / "real.jsonl", [_real_record(0)])
    )
    assert kind == "real"
