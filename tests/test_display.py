"""Unit tests for the Streamlit display helpers (pure functions only —
no Streamlit runtime, no model, no network)."""
import pandas as pd

from ui_lib.display import decision_badge, matched_prototypes_table


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
