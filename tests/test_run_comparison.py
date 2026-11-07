"""Tests for the longitudinal run comparison module (Issue #16).

Synthetic fixtures only — no real patient data, no API calls."""
import json
from pathlib import Path

import pandas as pd
import pytest

from run_comparison import (
    RunSnapshot,
    build_comparison_table,
    compare_runs,
    load_run_snapshot,
)


def _meta(run_id="run-001", timestamp="2026-01-15T10:00:00Z", **extra):
    meta = {
        "id": run_id,
        "model_key": "minilm",
        "fact_lang": "fa",
        "label_text_mode": "patient",
        "labeler_version": "3.0.0",
        "completed_utc": timestamp,
    }
    meta.update(extra)
    return meta


def _labeled_df(n=10, uncertain=2, margin_mean=0.15, top_shift=0):
    labels = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    decisions = ["uncertain"] * uncertain + ["labeled"] * (n - uncertain)
    margins = [round(margin_mean + (i % 5) * 0.01, 4) for i in range(n)]
    if top_shift:
        labels = labels[top_shift:] + labels[:top_shift]
    return pd.DataFrame({
        "idx": range(n),
        "text": [f"text {i}" for i in range(n)],
        "top_label": labels,
        "top_score": [0.5] * n,
        "labels": labels,
        "confident": [True] * n,
        "decision": decisions,
        "margin": margins,
        "matched_fact_ids": [f"8:fa"] * n,
    })


def _create_run(base: Path, run_id: str, timestamp: str, n=10, uncertain=2,
                margin_mean=0.15, top_shift=0) -> Path:
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(_meta(run_id, timestamp), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _labeled_df(n, uncertain, margin_mean, top_shift).to_csv(
        run_dir / "labeled.csv", index=False
    )
    return run_dir


@pytest.fixture()
def run_a(tmp_path):
    return _create_run(tmp_path, "run-a", "2026-01-10T08:00:00Z", n=10, uncertain=2, margin_mean=0.15)


@pytest.fixture()
def run_b(tmp_path):
    return _create_run(tmp_path, "run-b", "2026-01-20T12:00:00Z", n=10, uncertain=5, margin_mean=0.25, top_shift=2)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class TestLoadRunSnapshot:
    def test_loads_metadata_and_timestamp(self, run_a):
        snap = load_run_snapshot(run_a)
        assert snap.run_id == "run-a"
        assert snap.timestamp == "2026-01-10T08:00:00Z"
        assert snap.n_records == 10
        assert not snap.warnings

    def test_missing_timestamp_produces_warning(self, tmp_path):
        run_dir = tmp_path / "no-ts"
        run_dir.mkdir()
        meta = _meta("no-ts")
        del meta["completed_utc"]
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        pd.DataFrame({"top_label": ["A"]}).to_csv(run_dir / "labeled.csv", index=False)
        snap = load_run_snapshot(run_dir)
        assert snap.timestamp is None
        assert any("temporal metadata" in w for w in snap.warnings)

    def test_missing_run_meta(self, tmp_path):
        run_dir = tmp_path / "no-meta"
        run_dir.mkdir()
        pd.DataFrame({"top_label": ["A"]}).to_csv(run_dir / "labeled.csv", index=False)
        snap = load_run_snapshot(run_dir)
        assert snap.meta == {}
        assert snap.timestamp is None

    def test_missing_labeled_csv(self, tmp_path):
        run_dir = tmp_path / "no-csv"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(json.dumps(_meta("x")), encoding="utf-8")
        snap = load_run_snapshot(run_dir)
        assert snap.labeled.empty
        assert any("labeled.csv" in w for w in snap.warnings)

    def test_invalid_json_in_meta(self, tmp_path):
        run_dir = tmp_path / "bad-json"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text("{not valid", encoding="utf-8")
        snap = load_run_snapshot(run_dir)
        assert snap.meta == {}


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


class TestCompareRuns:
    def test_category_shares_and_delta_pp(self, run_a, run_b):
        result = compare_runs(run_a, run_b)
        assert len(result.categories) == 7  # A-G
        # run-a: 5 A, 5 B; run-b: shifted → different distribution
        for c in result.categories:
            assert c.share_a >= 0 and c.share_b >= 0
            assert c.delta_pp == pytest.approx(c.share_b - c.share_a)

    def test_uncertainty_comparison(self, run_a, run_b):
        result = compare_runs(run_a, run_b)
        unc = result.uncertainty
        assert unc is not None
        assert unc.count_a == 2  # uncertain=2 in run_a
        assert unc.count_b == 5  # uncertain=5 in run_b
        assert unc.rate_a == pytest.approx(2 / 10)
        assert unc.rate_b == pytest.approx(5 / 10)
        assert unc.delta_rate == pytest.approx(5 / 10 - 2 / 10)

    def test_margin_comparison(self, run_a, run_b):
        result = compare_runs(run_a, run_b)
        m = result.margin
        assert m is not None
        assert m.mean_a is not None and m.mean_b is not None
        assert m.delta == pytest.approx(m.mean_b - m.mean_a)

    def test_same_run_zero_differences(self, run_a):
        result = compare_runs(run_a, run_a)
        for c in result.categories:
            assert c.delta_pp == 0
        assert result.uncertainty.delta_rate == 0
        assert result.margin.delta == 0

    def test_direction_is_b_minus_a(self, run_a, run_b):
        """Verify delta = B − A for uncertainty rate."""
        result = compare_runs(run_a, run_b)
        # run_b has 5 uncertain, run_a has 2 → delta should be positive
        assert result.uncertainty.delta_rate > 0

    def test_empty_labeled_csv(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (empty_dir / "run_meta.json").write_text(json.dumps(_meta("empty")), encoding="utf-8")
        (empty_dir / "labeled.csv").write_text("", encoding="utf-8")
        normal_dir = _create_run(tmp_path, "normal", "2026-01-15T08:00:00Z")
        result = compare_runs(normal_dir, empty_dir)
        assert any("labeled.csv" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Missing Phase-3 columns (legacy CSV)
# --------------------------------------------------------------------------


class TestLegacyCsv:
    def test_missing_margin_column(self, tmp_path):
        run_dir = tmp_path / "legacy"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(json.dumps(_meta("legacy")), encoding="utf-8")
        pd.DataFrame({
            "idx": [0, 1], "top_label": ["A", "B"], "labels": ["A", "B"],
            "text": ["x", "y"], "top_score": [0.5, 0.5], "confident": [True, True],
            "decision": ["labeled", "labeled"],  # margin missing
        }).to_csv(run_dir / "labeled.csv", index=False)
        snap = load_run_snapshot(run_dir)
        assert snap.mean_margin is None  # margin column absent → N/A

    def test_missing_decision_column(self, tmp_path):
        run_dir = tmp_path / "legacy"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(json.dumps(_meta("legacy")), encoding="utf-8")
        pd.DataFrame({
            "idx": [0, 1], "top_label": ["A", "B"], "labels": ["A", "B"],
            "text": ["x", "y"], "top_score": [0.5, 0.5], "confident": [True, True],
            "margin": [0.1, 0.2],  # decision missing
        }).to_csv(run_dir / "labeled.csv", index=False)
        snap = load_run_snapshot(run_dir)
        assert snap.uncertain_count is None  # decision column absent → N/A

    def test_null_margin_values(self, tmp_path):
        run_dir = tmp_path / "null-margin"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(json.dumps(_meta("nm")), encoding="utf-8")
        pd.DataFrame({
            "idx": [0, 1], "top_label": ["A", "B"], "labels": ["A", "B"],
            "text": ["x", "y"], "top_score": [0.5, 0.5], "confident": [True, True],
            "decision": ["labeled", "labeled"],
            "margin": [None, None],  # all null
        }).to_csv(run_dir / "labeled.csv", index=False)
        snap = load_run_snapshot(run_dir)
        assert snap.mean_margin is None  # all margins null → N/A


# --------------------------------------------------------------------------
# Comparison table for UI
# --------------------------------------------------------------------------


def test_build_comparison_table_shape(run_a, run_b):
    result = compare_runs(run_a, run_b)
    table = build_comparison_table(result)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 7  # one row per category A–G
    assert "Δ pp (B−A)" in table.columns
