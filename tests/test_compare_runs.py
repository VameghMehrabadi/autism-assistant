"""Unit tests for compare_runs logic (no models, no network).

Uses a synthetic gap-analysis report shaped exactly like the dict produced
by gap_analysis.build_report.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from facts import CATEGORY_KEYS
from compare_runs import (
    build_comparison_table,
    load_run_top_scores,
    plot_score_distributions,
    render_comparison_markdown,
    selected_runs,
    summarize_run,
)
from gap_analysis import (
    CONFIDENCE_NOT_ACCURACY_EN,
    CONFIDENCE_NOT_ACCURACY_FA,
)


def _synthetic_report(n_samples: int = 10, stats: dict | None = None) -> dict:
    """Minimal report dict matching gap_analysis.build_report's contract."""
    counts = {"A": 3, "B": 2, "C": 1, "D": 1, "E": 1, "F": 2, "G": 0}
    per_category = {}
    for c in CATEGORY_KEYS:
        cnt = counts[c]
        per_category[c] = {
            "title": f"Title {c}",
            "title_fa": f"عنوان {c}",
            "count": cnt,
            "share": round(cnt / n_samples, 4),
            "multi_count": max(0, cnt - 1),
            "multi_share": round(max(0, cnt - 1) / n_samples, 4),
            "mean_score": 0.5 if cnt else 0.0,
            "median_score": 0.5 if cnt else 0.0,
            "std_score": 0.05 if cnt else 0.0,
            "confident_share": 0.8 if cnt else 0.0,
            "global_mean_score": 0.3,
        }
    if stats is None:
        # below-threshold p25 → no saturation callout by default
        stats = {"mean": 0.42, "median": 0.43, "p05": 0.2, "p25": 0.3,
                 "p75": 0.55, "p95": 0.65, "min": 0.12, "max": 0.7}
    return {
        "n_samples": n_samples,
        "threshold": 0.35,
        "n_confident": 8,
        "confident_rate": 0.8,
        "n_unlabeled_NONE": 0,
        "top_score_stats": stats,
        "per_category": per_category,
        "global_mean_score_per_category": {c: 0.3 for c in CATEGORY_KEYS},
        "ranking_weak_to_strong": ["G", "E", "D", "C", "B", "F", "A"],
        "weakest": ["G", "E"],
        "strongest": ["F", "A"],
    }


def _run_cfg(run_id: str) -> dict:
    return {
        "id": run_id,
        "title": f"Title for {run_id}",
        "model_key": "minilm",
        "fact_lang": "fa",
        "dataset": "en",
    }


def test_summarize_run_extracts_expected_fields():
    summary = summarize_run(_synthetic_report(), _run_cfg("1_fa_en_minilm"))
    assert summary["id"] == "1_fa_en_minilm"
    assert summary["n_samples"] == 10
    assert summary["confident_rate"] == 0.8
    assert summary["counts"]["A"] == 3
    assert summary["shares"]["A"] == 0.3
    assert summary["multi_counts"]["A"] == 2
    assert summary["ranking_weak_to_strong"] == ["G", "E", "D", "C", "B", "F", "A"]
    assert summary["mean_top_score"] == pytest.approx(0.5)
    assert set(summary["counts"]) == set(CATEGORY_KEYS)


def test_build_comparison_table_shape_and_values():
    summaries = [
        summarize_run(_synthetic_report(), _run_cfg("1_fa_en_minilm")),
        summarize_run(_synthetic_report(n_samples=20), _run_cfg("2_fa_en_bge-m3")),
    ]
    table = build_comparison_table(summaries)
    assert len(table) == 2
    for column in ["run", "title", "model", "fact_lang", "n_samples",
                   "confident_rate", "mean_top_score", "weakest", "strongest"]:
        assert column in table.columns
    for c in CATEGORY_KEYS:
        assert f"count_{c}" in table.columns
        assert f"share_{c}" in table.columns
        assert f"multi_{c}" in table.columns
        assert f"mean_{c}" in table.columns
    assert table.iloc[0]["run"] == "1_fa_en_minilm"
    assert table.iloc[0]["count_A"] == 3
    assert table.iloc[1]["n_samples"] == 20


def test_render_comparison_markdown_contains_runs_and_sections():
    summaries = [summarize_run(_synthetic_report(), _run_cfg("1_fa_en_minilm"))]
    table = build_comparison_table(summaries)
    md = render_comparison_markdown(summaries, table)
    assert "# Five-way embedding comparison" in md
    assert "`1_fa_en_minilm`" in md
    assert "## Category share (top-1 %)" in md
    assert "## Notes for interpretation" in md


def test_selected_runs_filters_and_validates():
    runs = selected_runs("1,3")
    assert [r["id"] for r in runs] == ["1_fa_en_minilm", "3_fa_en_e5-large"]
    assert len(selected_runs(None)) == 5
    with pytest.raises(ValueError):
        selected_runs("9,99")


# --------------------------------------------------------------------------
# Issue #15 — confidence interpretation, raw score distributions,
# and the no-best-model rule
# --------------------------------------------------------------------------


def _comparison_markdown(summaries: list[dict]) -> str:
    return render_comparison_markdown(
        summaries, build_comparison_table(summaries)
    )


def test_comparison_markdown_carries_fixed_confidence_warning():
    summaries = [summarize_run(_synthetic_report(), _run_cfg("1_fa_en_minilm"))]
    md = _comparison_markdown(summaries)
    assert CONFIDENCE_NOT_ACCURACY_FA in md
    assert CONFIDENCE_NOT_ACCURACY_EN in md
    # the fixed sentence sits directly next to the Confident% table
    assert (
        md.index("| # | ID |")
        < md.index(CONFIDENCE_NOT_ACCURACY_FA)
        < md.index("## Raw score distribution per encoder")
    )


def test_comparison_markdown_no_best_model_presentation():
    low = summarize_run(_synthetic_report(n_samples=10),
                        _run_cfg("a_low_confident"))
    high = summarize_run(_synthetic_report(n_samples=10),
                         _run_cfg("b_high_confident"))
    # the higher confident rate comes second — never re-sorted to the top
    md = _comparison_markdown([low, high])
    assert "best model" not in md.lower()
    assert "بهترین مدل" not in md
    assert md.index("`a_low_confident`") < md.index("`b_high_confident`")


def test_comparison_markdown_score_distribution_section():
    summaries = [summarize_run(_synthetic_report(), _run_cfg("1_fa_en_minilm"))]
    md = _comparison_markdown(summaries)
    assert "## Raw score distribution per encoder (uncalibrated)" in md
    assert (
        "| `1_fa_en_minilm` | 10 | 0.42 | 0.43 | 0.2 | 0.65 | 80.0% |" in md
    )
    assert "threshold saturation" not in md


def test_comparison_markdown_flags_e5_style_saturation():
    e5_stats = {"mean": 0.805, "median": 0.81, "p05": 0.74, "p25": 0.78,
                "p75": 0.83, "p95": 0.86, "min": 0.55, "max": 0.92}
    summaries = [summarize_run(_synthetic_report(stats=e5_stats),
                               _run_cfg("3_fa_en_e5-large"))]
    md = _comparison_markdown(summaries)
    assert "threshold saturation" in md
    assert "not** a better model" in md


def test_comparison_markdown_legacy_stats_render_as_dashes():
    report = _synthetic_report()
    report["top_score_stats"] = {  # pre-#15 report: no p05/p95 keys
        "mean": 0.44, "median": 0.44, "p25": 0.3, "p75": 0.5,
        "min": 0.2, "max": 0.7,
    }
    summaries = [summarize_run(report, _run_cfg("1_fa_en_minilm"))]
    md = _comparison_markdown(summaries)
    assert "| `1_fa_en_minilm` | 10 | 0.44 | 0.44 | — | — |" in md


def test_plot_score_distributions_writes_shared_axes_figure(tmp_path):
    runs_dir = tmp_path / "comparison"
    specs = {
        "1_mini": np.linspace(0.05, 0.60, 60),
        "3_e5": np.full(60, 0.80),
    }
    for run_id, scores in specs.items():
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        pd.DataFrame({"top_score": scores}).to_csv(
            run_dir / "labeled.csv", index=False
        )
    summaries = [
        {"id": "1_mini", "model_key": "minilm"},
        {"id": "3_e5", "model_key": "e5-large"},
    ]
    out = tmp_path / "score_distributions.png"
    assert plot_score_distributions(summaries, out, runs_dir=runs_dir) is True
    assert out.exists() and out.stat().st_size > 0


def test_plot_score_distributions_without_raw_scores_is_honest(tmp_path):
    out = tmp_path / "score_distributions.png"
    summaries = [{"id": "missing", "model_key": "minilm"}]
    assert plot_score_distributions(summaries, out, runs_dir=tmp_path) is False
    assert not out.exists()


def test_load_run_top_scores_handles_missing_and_empty(tmp_path):
    assert load_run_top_scores("ghost", runs_dir=tmp_path) is None
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    pd.DataFrame({"top_score": pd.Series(dtype=float)}).to_csv(
        empty_run / "labeled.csv", index=False
    )
    assert load_run_top_scores("empty_run", runs_dir=tmp_path) is None


def test_augment_score_stats_derives_from_labeled_csv(tmp_path):
    """Legacy gap JSONs without stats get them derived from persisted raw
    scores — marked as derived, never invented for missing runs."""
    from compare_runs import augment_score_stats

    legacy_run = tmp_path / "1_legacy"
    legacy_run.mkdir()
    scores = np.full(20, 0.80)
    pd.DataFrame({"top_score": scores}).to_csv(
        legacy_run / "labeled.csv", index=False
    )
    summaries = [
        {"id": "1_legacy", "top_score_stats": None},
        {"id": "2_no_artifacts", "top_score_stats": None},
        {"id": "3_already_has", "top_score_stats": {"mean": 0.42}},
    ]
    augment_score_stats(summaries, runs_dir=tmp_path)
    derived = summaries[0]["top_score_stats"]
    assert derived["mean"] == 0.8
    assert derived["p25"] == 0.8
    assert "derived from labeled.csv" in derived["source"]
    assert summaries[1]["top_score_stats"] is None  # nothing fabricated
    assert summaries[2]["top_score_stats"] == {"mean": 0.42}  # untouched


def test_compare_models_page_wires_confidence_warning():
    """Textual snapshot lock: the Streamlit page must carry the warning."""
    page = Path(__file__).resolve().parents[1] / "pages" / "3_Compare_Models.py"
    src = page.read_text(encoding="utf-8")
    assert "CONFIDENCE_NOT_ACCURACY_FA" in src
    assert "st.warning" in src
