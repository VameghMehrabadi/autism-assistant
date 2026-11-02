import numpy as np
from facts import CATEGORY_KEYS
from labeler import (
    SemanticLabeler,
    apply_e5_prefix,
    assign_label_row,
    format_labels,
    parse_labels,
    resolve_model,
)


def _row(*scores: float) -> np.ndarray:
    return np.array(scores, dtype=float)


def test_assign_labels_multi_label():
    scores = np.array([
        [0.1, 0.5, 0.4],
        [0.35, 0.2, 0.1],
    ], dtype=float)

    result = [
        assign_label_row(row, CATEGORY_KEYS[:3], threshold=0.3, multi_label=True)
        for row in scores
    ]
    assert result[0]["top_label"] == CATEGORY_KEYS[1]
    assert result[0]["labels"] == [CATEGORY_KEYS[1], CATEGORY_KEYS[2]]
    assert result[1]["top_label"] == CATEGORY_KEYS[0]
    assert result[1]["labels"] == [CATEGORY_KEYS[0]]


def test_multi_label_below_threshold_stays_unlabeled():
    decision = assign_label_row(
        _row(0.1, 0.2, 0.34),
        CATEGORY_KEYS[:3],
        threshold=0.35,
        multi_label=True,
    )

    assert decision["labels"] == []
    assert decision["top_label"] == CATEGORY_KEYS[2]
    assert decision["top_score"] == 0.34
    assert decision["confident"] is False
    assert format_labels(decision["labels"]) == ""
    assert parse_labels(format_labels(decision["labels"])) == []


def test_multi_label_keeps_every_category_above_threshold():
    decision = assign_label_row(
        _row(0.2, 0.51, 0.36, 0.34),
        CATEGORY_KEYS[:4],
        threshold=0.35,
        multi_label=True,
    )

    assert decision["labels"] == [CATEGORY_KEYS[1], CATEGORY_KEYS[2]]
    assert decision["top_label"] == CATEGORY_KEYS[1]
    assert decision["confident"] is True
    assert parse_labels(format_labels(decision["labels"])) == decision["labels"]


def test_assign_labels_none_when_not_confident():
    decision = assign_label_row(
        _row(0.1, 0.2, 0.3),
        CATEGORY_KEYS[:3],
        threshold=0.5,
        multi_label=False,
    )
    assert decision["labels"] == ["NONE"]
    assert decision["confident"] is False


def test_assign_labels_uses_the_same_row_decision():
    scores = np.array([[0.1, 0.2, 0.3]], dtype=float)
    labeler = type("Labeler", (), {"cat_keys": CATEGORY_KEYS[:3]})()
    batched = SemanticLabeler.assign_labels(
        labeler, scores, threshold=0.5, multi_label=True
    )
    direct = assign_label_row(
        scores[0], CATEGORY_KEYS[:3], threshold=0.5, multi_label=True
    )
    assert batched[0] == direct


def test_resolve_model_keys():
    key, name, prefix = resolve_model(model_key="e5-large")
    assert key == "e5-large"
    assert "e5" in name.lower()
    assert prefix == "e5"

    key, name, prefix = resolve_model(model_key="bge-m3")
    assert key == "bge-m3"
    assert "bge-m3" in name.lower()
    assert prefix == "none"


def test_apply_e5_prefix():
    texts = apply_e5_prefix(["hello"], role="query")
    assert texts == ["query: hello"]
    texts = apply_e5_prefix(["passage: already"], role="passage")
    assert texts == ["passage: already"]
