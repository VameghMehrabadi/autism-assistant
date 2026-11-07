"""Phase-4 evaluation-validity tests: score-scale honesty, alignment
verification, and run-metadata completeness (offline, synthetic only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compare_runs
import config
import main
from facts import CATEGORY_KEYS
from gap_analysis import build_report


# --------------------------------------------------------------------------
# 1. Score-scale statistics travel with every report
# --------------------------------------------------------------------------


def _labeled_df(n=10, score=0.5):
    return pd.DataFrame({
        "idx": range(n),
        "top_label": ["B"] * n,
        "top_score": [score] * n,
        "labels": ["B"] * n,
    }), np.full((n, len(CATEGORY_KEYS)), score)


def test_gap_report_includes_top_score_stats():
    df, scores = _labeled_df(4, 0.6)
    report = build_report(df, scores, threshold=0.35)
    stats = report["top_score_stats"]
    assert stats == {
        "mean": 0.6, "median": 0.6, "p25": 0.6, "p75": 0.6, "min": 0.6, "max": 0.6,
    }


def test_gap_report_score_stats_vary_with_scale():
    df_a, scores_a = _labeled_df(10, 0.42)
    df_b, scores_b = _labeled_df(10, 0.79)
    stats_a = build_report(df_a, scores_a)["top_score_stats"]
    stats_b = build_report(df_b, scores_b)["top_score_stats"]
    # Identical threshold 0.35, completely different score scales — the
    # numbers that prove thresholds are not comparable across models.
    assert stats_a["mean"] != stats_b["mean"]
    assert stats_b["mean"] > stats_a["mean"]


# --------------------------------------------------------------------------
# 2. Run metadata completeness (threshold, window, fingerprint, totals)
# --------------------------------------------------------------------------


class _FakeEncoder:
    def __init__(self, dim=16):
        import hashlib

        self.dim = dim
        self._hashlib = hashlib

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False, **kw):
        import numpy as np

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(self._hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vectors[i] = np.random.default_rng(seed).standard_normal(self.dim)
        return vectors


@pytest.fixture()
def offline_labeler(monkeypatch):
    real = SemanticLabeler = None  # noqa: F841 — imported below
    from labeler import SemanticLabeler as _SL

    real = _SL(model_key="minilm", fact_lang="both",
               _encoder=_FakeEncoder(), enable_cache=False)
    monkeypatch.setattr(main, "SemanticLabeler", lambda **kw: real)
    return real


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"


def test_run_meta_records_threshold_window_fingerprint(tmp_path, offline_labeler):
    main.run_pipeline(
        limit=3, model_key="minilm", fact_lang="fa", dataset="fa",
        dataset_path=str(FIXTURE), output_dir=tmp_path,
    )
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["threshold"] == config.LABEL_THRESHOLD
    assert meta["preprocess_version"] == config.PREPROCESS_VERSION
    assert meta["text_window"] is None  # minilm native default (128) → None = model default
    assert len(meta["dataset_fingerprint"]) == 16
    assert meta["dataset_total"] == 5  # fixture rows, before the limit
    # Issue #9 — the execution-time alignment contract is complete:
    assert set(
        __import__("alignment").MODERN_REQUIRED_FIELDS
    ) <= set(meta)


# --------------------------------------------------------------------------
# 3. Alignment verification over persisted runs
# --------------------------------------------------------------------------


def _make_run_dir(base: Path, run_id: str, *, idxs, threshold=0.35,
                  model="minilm", text_window=None, text_seed=1):
    import hashlib

    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    texts = [f"record {i} seed {text_seed}" for i in idxs]
    pd.DataFrame({"idx": idxs, "text": texts, "top_label": ["A"] * len(idxs)}).to_csv(
        run_dir / "labeled.csv", index=False
    )
    per_cat = {c: {"count": 0, "share": 0, "multi_count": 0, "multi_share": 0,
                   "mean_score": 0, "median_score": 0, "std_score": 0,
                   "confident_share": 0, "global_mean_score": 0}
               for c in CATEGORY_KEYS}
    per_cat["A"]["count"] = len(idxs)
    gap = {
        "n_samples": len(idxs), "threshold": threshold, "per_category": per_cat,
        "global_mean_score_per_category": {c: 0 for c in CATEGORY_KEYS},
        "ranking_weak_to_strong": list(CATEGORY_KEYS),
    }
    (run_dir / "gap_analysis.json").write_text(
        json.dumps(gap, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps({"model_key": model, "text_window": text_window,
                    "dataset_fingerprint": hashlib.sha256(
                        "\u241f".join(sorted(texts)).encode()).hexdigest()[:16]},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def test_verify_alignment_detects_equivalent_runs(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_a", idxs=[0, 1, 2], model="minilm", text_window=None)
    _make_run_dir(base, "2_b", idxs=[0, 1, 2], model="bge-m3", text_window=512)
    report = compare_runs.verify_alignment(["1_a", "2_b"])
    assert report["all_records_aligned"] is True
    by_id = {r["id"]: r for r in report["runs"]}
    assert by_id["2_b"]["text_window"] == 512
    assert by_id["1_a"]["records_match_reference"] is True


def test_verify_alignment_detects_record_mismatch(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_a", idxs=[0, 1, 2])
    _make_run_dir(base, "2_b", idxs=[0, 1, 3])  # different records
    report = compare_runs.verify_alignment(["1_a", "2_b"])
    assert report["all_records_aligned"] is False
    by_id = {r["id"]: r for r in report["runs"]}
    assert by_id["2_b"]["records_match_reference"] is False


def test_verify_alignment_reports_missing_artifacts(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    report = compare_runs.verify_alignment(["1_missing"])
    assert report["all_records_aligned"] is None
    assert report["runs"][0]["status"] == "MISSING_ARTIFACTS"


def test_verify_alignment_cli(tmp_path, monkeypatch, capsys):
    base = tmp_path / "comparison"
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_fa_en_minilm", idxs=[0, 1], model="minilm", text_window=None)
    _make_run_dir(base, "2_fa_en_bge-m3", idxs=[0, 1], model="bge-m3", text_window=512)
    # runs 3-5 have no artifacts → reported as missing, verdict from runs 1-2
    monkeypatch.setattr("sys.argv", ["compare_runs.py", "--verify-alignment"])
    assert compare_runs.main() == 0
    out = capsys.readouterr().out
    assert '"all_records_aligned": true' in out
    assert '"n_runs_missing_artifacts": 3' in out


def test_comparison_summary_carries_threshold_and_scale_stats():
    report = {
        "n_samples": 10, "confident_rate": 0.9, "threshold": 0.35,
        "top_score_stats": {"mean": 0.44, "median": 0.44, "p25": 0.44, "p75": 0.44,
                            "min": 0.44, "max": 0.44},
        "per_category": {
            c: {"count": 1, "share": 0.1, "multi_count": 0, "multi_share": 0.0,
                "mean_score": 0.44, "global_mean_score": 0.44}
            for c in CATEGORY_KEYS
        },
        "global_mean_score_per_category": {c: 0.44 for c in CATEGORY_KEYS},
        "ranking_weak_to_strong": list(CATEGORY_KEYS),
        "run": {"text_window": 512, "prefix": "none",
                "labeler_version": "3.0.0", "dataset_fingerprint": "abc123"},
    }
    summary = compare_runs.summarize_run(
        report, {"id": "x", "title": "t", "model_key": "bge-m3",
                 "fact_lang": "fa", "dataset": "en"}
    )
    assert summary["threshold"] == 0.35
    assert summary["top_score_stats"]["mean"] == 0.44
    assert summary["text_window"] == 512
    assert summary["dataset_fingerprint"] == "abc123"
    assert "model-dependent" in summary["score_scale_note"]
