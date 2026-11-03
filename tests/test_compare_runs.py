"""Unit tests for compare_runs logic (no models, no network).

Uses a synthetic gap-analysis report shaped exactly like the dict produced
by gap_analysis.build_report.
"""
import pytest
from facts import CATEGORY_KEYS
from compare_runs import (
    build_comparison_table,
    render_comparison_markdown,
    selected_runs,
    summarize_run,
)


def _synthetic_report(n_samples: int = 10) -> dict:
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
    return {
        "n_samples": n_samples,
        "threshold": 0.35,
        "n_confident": 8,
        "confident_rate": 0.8,
        "n_unlabeled_NONE": 0,
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
