import numpy as np
import pandas as pd

from facts import CATEGORY_KEYS
from gap_analysis import build_report


def test_gap_report_counts_multi_label_even_when_top1_is_zero():
    n_cats = len(CATEGORY_KEYS)
    d_idx = CATEGORY_KEYS.index("D")
    e_idx = CATEGORY_KEYS.index("E")
    f_idx = CATEGORY_KEYS.index("F")

    scores = np.zeros((3, n_cats), dtype=float)
    scores[0, f_idx] = 0.60
    scores[0, d_idx] = 0.50
    scores[1, f_idx] = 0.55
    scores[1, e_idx] = 0.40
    scores[2, f_idx] = 0.70

    df = pd.DataFrame({
        "top_label": ["F", "F", "F"],
        "top_score": [0.60, 0.55, 0.70],
        "labels": ["F|D", "F|E", "F"],
    })

    report = build_report(df, scores, threshold=0.35)
    assert report["per_category"]["F"]["count"] == 3
    assert report["per_category"]["D"]["count"] == 0
    assert report["per_category"]["E"]["count"] == 0
    assert report["per_category"]["D"]["multi_count"] == 1
    assert report["per_category"]["E"]["multi_count"] == 1
    assert report["per_category"]["D"]["global_mean_score"] > 0
    assert report["per_category"]["D"]["mean_score"] == 0.0


def test_empty_multi_labels_are_not_counted_as_top_label():
    """Blank and CSV-NaN label cells must not count the diagnostic top-1."""
    n_cats = len(CATEGORY_KEYS)
    scores = np.full((2, n_cats), 0.2, dtype=float)
    df = pd.DataFrame({
        "top_label": ["F", "F"],
        "top_score": [0.2, 0.2],
        "labels": ["", np.nan],
    })

    report = build_report(df, scores, threshold=0.35)
    assert report["per_category"]["F"]["count"] == 2
    assert report["per_category"]["F"]["multi_count"] == 0


# --------------------------------------------------------------------------
# Phase-3 signals: uncertainty, margin, matched-prototype evidence
# --------------------------------------------------------------------------


def _df_with_phase3_signals() -> pd.DataFrame:
    n_cats = len(CATEGORY_KEYS)
    scores = np.full((3, n_cats), 0.5, dtype=float)
    return pd.DataFrame({
        "top_label": ["B", "B", "F"],
        "top_score": [0.61, 0.55, 0.70],
        "labels": ["B", "B", "F"],
        "decision": ["labeled", "labeled", "uncertain"],
        "margin": [0.11, 0.05, 0.02],
        "matched_fact_ids": ["8:en", "8:fa", "33:fa"],
    }), scores


def test_gap_report_uncertainty_margin_and_matched_evidence():
    df, scores = _df_with_phase3_signals()
    report = build_report(df, scores, threshold=0.35)
    assert report["n_uncertain"] == 1
    assert report["uncertain_rate"] == round(1 / 3, 4)
    assert report["margin_stats"]["mean"] == round((0.11 + 0.05 + 0.02) / 3, 4)
    assert report["margin_stats"]["min"] == 0.02
    # fact 8 → category B (twice), fact 33 → category G
    assert report["per_category"]["B"]["matched_count"] == 2
    assert report["per_category"]["G"]["matched_count"] == 1
    assert report["per_category"]["A"]["matched_count"] == 0


def test_gap_report_old_format_has_no_phase3_signals():
    n_cats = len(CATEGORY_KEYS)
    scores = np.full((2, n_cats), 0.5, dtype=float)
    df = pd.DataFrame({
        "top_label": ["F", "F"],
        "top_score": [0.6, 0.6],
        "labels": ["F", "F"],
    })
    report = build_report(df, scores, threshold=0.35)
    assert report["n_uncertain"] is None
    assert report["uncertain_rate"] is None
    assert report["margin_stats"] is None
    assert all(report["per_category"][c]["matched_count"] == 0 for c in CATEGORY_KEYS)


def test_gap_report_markdown_renders_uncertainty_section():
    from gap_analysis import render_markdown

    df, scores = _df_with_phase3_signals()
    report = build_report(df, scores, threshold=0.35)
    md = render_markdown(report)
    assert "## Uncertainty & prototype evidence" in md
    assert "Explicit uncertain decisions: **1**" in md
    assert "Matched-prototype evidence per category" in md
