"""Unit tests for the Streamlit display helpers (pure functions only —
no Streamlit runtime, no model, no network)."""
import pandas as pd
import pytest

from ui_lib.display import (
    decision_badge,
    detect_phase3_availability,
    matched_prototypes_table,
    phase3_missing_warning,
)


def test_decision_badge_prefers_explicit_decision():
    assert decision_badge({"decision": "uncertain"}) == "uncertain"
    assert decision_badge({"decision": "labeled"}) == "labeled"


def test_decision_badge_falls_back_to_confident_flag():
    assert decision_badge({"confident": True}) == "labeled"
    assert decision_badge({"confident": False}) == "uncertain"
    assert decision_badge({}) == "uncertain"


def test_matched_prototypes_table_shape():
    result = {
        "matched_prototypes": [
            {"category": "B", "fact_id": 8, "lang": "fa", "score": 0.51234},
            {"category": "D", "fact_id": 15, "lang": "en", "score": 0.401},
        ]
    }
    table = matched_prototypes_table(result)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["Category", "Title", "Fact ID", "Language", "Similarity"]
    assert table.iloc[0]["Title"] == "Sensory Processing"
    assert table.iloc[0]["Similarity"] == 0.5123


def test_matched_prototypes_table_none_when_absent():
    assert matched_prototypes_table({}) is None
    assert matched_prototypes_table({"matched_prototypes": []}) is None


# --------------------------------------------------------------------------
# Issue #19: honest UI labeling for pre-Phase-3 gap analysis CSV files
# --------------------------------------------------------------------------


def _legacy_df() -> pd.DataFrame:
    """A pre-Phase-3 labeled.csv — no decision, margin, or matched_fact_ids."""
    return pd.DataFrame({
        "idx": [0, 1],
        "text": ["patient text 0", "patient text 1"],
        "top_label": ["B", "C"],
        "top_score": [0.52, 0.48],
        "labels": ["B", "C"],
        "confident": [True, True],
        "score_A": [0.1, 0.1],
        "score_B": [0.52, 0.1],
        "score_C": [0.1, 0.48],
        "score_D": [0.1, 0.1],
        "score_E": [0.1, 0.1],
        "score_F": [0.1, 0.1],
        "score_G": [0.1, 0.1],
    })


def _current_df() -> pd.DataFrame:
    """A Phase-3 labeled.csv — all Phase-3 columns present."""
    df = _legacy_df()
    df["decision"] = ["labeled", "labeled"]
    df["margin"] = [0.11, 0.38]
    df["matched_fact_ids"] = ["8:fa", "15:en"]
    return df


def test_detect_phase3_availability_legacy_df():
    avail = detect_phase3_availability(_legacy_df())
    assert avail == {"decision": False, "margin": False, "matched_fact_ids": False}


def test_detect_phase3_availability_current_df():
    avail = detect_phase3_availability(_current_df())
    assert all(avail.values()), f"not all Phase-3 columns detected: {avail}"


def test_phase3_warning_returned_for_legacy_df():
    warning = phase3_missing_warning(_legacy_df())
    assert warning is not None
    assert "decision" in warning
    assert "margin" in warning
    assert "matched_fact_ids" in warning


def test_phase3_warning_none_for_current_df():
    assert phase3_missing_warning(_current_df()) is None


def test_phase3_warning_partial_schema_names_only_missing(tmp_path):
    """A CSV with decision but without margin identifies only margin as missing."""
    df = _legacy_df()
    df["decision"] = ["labeled", "uncertain"]
    warning = phase3_missing_warning(df)
    assert warning is not None
    assert "margin" in warning
    assert "matched_fact_ids" in warning
    assert "decision" not in warning.split("Missing columns:")[1].split("\n")[0]


def test_missing_margin_is_not_zero_in_warning():
    """The warning must not imply that a missing margin is zero."""
    warning = phase3_missing_warning(_legacy_df())
    assert "unavailable" in warning.lower()
    assert "= 0" not in warning
    assert "= 0.0" not in warning


def test_gap_summary_na_for_legacy_report():
    """render_gap_summary logic: Phase-3 metrics are None for legacy reports
    (verified via the gap report dict, not the UI function)."""
    # Legacy gap report (no Phase-3 fields)
    legacy_report = {
        "n_samples": 10,
        "confident_rate": 0.9,
        "threshold": 0.35,
        "per_category": {},
        "global_mean_score_per_category": {},
        "ranking_weak_to_strong": list("ABCDEFG"),
    }
    assert legacy_report.get("n_uncertain") is None
    assert legacy_report.get("margin_stats") is None


def test_gap_summary_has_phase3_for_current_report():
    """A current gap report includes Phase-3 metrics (not None)."""
    current_report = {
        "n_samples": 10,
        "confident_rate": 0.9,
        "threshold": 0.35,
        "n_uncertain": 2,
        "uncertain_rate": 0.2,
        "margin_stats": {"mean": 0.1, "median": 0.1, "min": 0.05},
        "per_category": {},
        "global_mean_score_per_category": {},
        "ranking_weak_to_strong": list("ABCDEFG"),
    }
    assert current_report["n_uncertain"] == 2
    assert current_report["margin_stats"] is not None
