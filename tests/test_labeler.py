import hashlib
from pathlib import Path

import numpy as np
import pytest
from facts import CATEGORY_KEYS
from labeler import (
    DECISION_LABELED,
    DECISION_UNCERTAIN,
    UNLABELED,
    SemanticLabeler,
    apply_e5_prefix,
    assign_label_row,
    format_labels,
    normalize_for_embedding,
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


# ==========================================================================
# Phase 3: labeling engine tests (offline via a deterministic fake encoder)
# ==========================================================================


class FakeEncoder:
    """Deterministic offline stand-in for SentenceTransformer.encode.

    Vectors are derived from the SHA-256 of the received (prepared) text, so
    results are reproducible across processes. It records every batch of
    prepared texts so tests can assert model-specific preprocessing (e.g.
    E5 query:/passage: prefixes).
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encode_calls = 0
        self.received: list[list[str]] = []

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False, **kwargs):
        self.encode_calls += 1
        self.received.append(list(texts))
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
            vectors[i] = vector
        return vectors


def _make_labeler(fact_lang: str = "both", **kwargs) -> SemanticLabeler:
    kwargs.setdefault("_encoder", FakeEncoder())
    kwargs.setdefault("enable_cache", False)
    return SemanticLabeler(model_key="minilm", fact_lang=fact_lang, **kwargs)


# --------------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------------


def test_normalize_for_embedding_multilingual():
    assert normalize_for_embedding("  سلام   دنیا\n") == "سلام دنیا"
    assert normalize_for_embedding("Hello   World \t") == "Hello World"
    assert normalize_for_embedding(None) == ""
    assert normalize_for_embedding("") == ""
    # ZWNJ inside Persian words must survive normalization
    assert "\u200c" in normalize_for_embedding("می‌شود")


# --------------------------------------------------------------------------
# Structured decisions
# --------------------------------------------------------------------------


def test_assign_label_row_margin_and_decision():
    result = assign_label_row(
        np.array([0.6, 0.2, 0.1]), ["A", "B", "C"], threshold=0.5, multi_label=True
    )
    assert result["decision"] == DECISION_LABELED
    assert result["margin"] == 0.4
    assert result["score_type"] == "cosine_similarity"
    assert result["labeler_version"]
    assert result["matched_prototypes"] == []


def test_assign_label_row_uncertain_is_explicit():
    result = assign_label_row(
        np.array([0.2, 0.19, 0.1]), ["A", "B", "C"], threshold=0.5, multi_label=True
    )
    assert result["decision"] == DECISION_UNCERTAIN
    assert result["labels"] == []
    assert result["confident"] is False
    assert result["top_label"] == "A"  # still reported, diagnostic only


def test_assign_label_row_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        assign_label_row(np.array([0.5, np.nan, 0.2]), ["A", "B", "C"])
    with pytest.raises(ValueError):
        assign_label_row(np.array([np.inf, 0.1, 0.2]), ["A", "B", "C"])


def test_assign_label_row_deterministic_tie_break():
    result = assign_label_row(
        np.array([0.5, 0.5, 0.1]), ["A", "B", "C"], threshold=0.4
    )
    assert result["top_label"] == "A"  # lowest index wins ties, deterministically


def test_assign_label_row_per_category_thresholds():
    row = np.array([0.40, 0.34, 0.10])
    result = assign_label_row(
        row, ["A", "B", "C"], threshold={"A": 0.5, "B": 0.3}, multi_label=True
    )
    assert result["top_label"] == "A"
    assert result["labels"] == ["B"]  # A below its own threshold, B above its
    assert result["confident"] is False  # top category did not clear 0.5
    assert result["decision"] == DECISION_LABELED  # B was accepted


def test_resolve_threshold_none_uses_config_overrides(monkeypatch):
    from labeler import resolve_threshold

    monkeypatch.setattr(
        "config.CATEGORY_THRESHOLDS", {"B": 0.42}, raising=False
    )
    assert resolve_threshold("B", None, 0.35) == 0.42
    assert resolve_threshold("A", None, 0.35) == 0.35
    assert resolve_threshold("A", 0.5, 0.35) == 0.5
    assert resolve_threshold("C", {"C": 0.6}, 0.35) == 0.6
    assert resolve_threshold("A", {"B": 0.6}, 0.35) == 0.35


# --------------------------------------------------------------------------
# Offline engine pipeline (fake encoder: no downloads, deterministic)
# --------------------------------------------------------------------------


def test_e5_prefix_roles_applied_query_and_passage():
    encoder = FakeEncoder()
    labeler = SemanticLabeler(
        model_key="e5-large", fact_lang="en", _encoder=encoder, enable_cache=False
    )
    labeler.embed_samples(["My child avoids eye contact."])
    # First batch = prototypes (passage role at init), last = sample (query).
    assert encoder.received[0][0].startswith("passage: ")
    assert encoder.received[-1][0].startswith("query: ")


def test_bge_and_minilm_get_no_prefix():
    for model_key in ("minilm", "bge-m3"):
        encoder = FakeEncoder()
        labeler = SemanticLabeler(
            model_key=model_key, fact_lang="fa", _encoder=encoder, enable_cache=False
        )
        labeler.embed_samples(["فرزندم تماس چشمی ندارد."])
        assert not encoder.received[-1][0].startswith(("query:", "passage:"))


def test_embed_samples_deterministic_multilingual():
    labeler = _make_labeler(fact_lang="both")
    texts = ["My child flaps his hands.", "فرزندم دست‌هایش را تکان می‌دهد."]
    first = labeler.embed_samples(texts)
    second = labeler.embed_samples(texts)
    assert np.array_equal(first, second)
    assert first.shape[0] == 2
    norms = np.linalg.norm(first, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # normalized embeddings


def test_score_samples_returns_valid_prototype_matches():
    labeler = _make_labeler(fact_lang="both")
    embeddings = labeler.embed_samples(["Eye contact is exhausting for my child."])
    scores, matches = labeler.score_samples(embeddings, return_matches=True)
    assert scores.shape == (1, 7)
    assert matches.shape == (1, 7)
    for ci, cat in enumerate(labeler.cat_keys):
        proto_index = int(matches[0, ci])
        meta = labeler.prototype_meta[proto_index]
        assert meta["category"] == cat
        assert 1 <= meta["fact_id"] <= 44
        assert meta["lang"] in {"en", "fa"}


def test_fact_lang_configurations_change_prototypes():
    en_labeler = _make_labeler(fact_lang="en")
    fa_labeler = _make_labeler(fact_lang="fa")
    both_labeler = _make_labeler(fact_lang="both")
    assert all(p["lang"] == "en" for p in en_labeler.prototype_meta)
    assert all(p["lang"] == "fa" for p in fa_labeler.prototype_meta)
    assert {p["lang"] for p in both_labeler.prototype_meta} == {"en", "fa"}


def test_multilingual_inputs_produce_structured_results():
    labeler = _make_labeler(fact_lang="both")
    texts = [
        "My child avoids eye contact.",            # English
        "فرزندم تماس چشمی برقرار نمی‌کند.",          # Persian
        "He lines up his toys for hours.",          # English
        "برنامه‌ی روزانه که عوض می‌شود اضطراب می‌گیرد.",  # Persian
    ]
    embeddings = labeler.embed_samples(texts)
    scores, matches = labeler.score_samples(embeddings, return_matches=True)
    results = labeler.assign_labels(
        scores, threshold=-1.0, prototype_indices=matches  # force accept-all
    )
    assert len(results) == 4
    for result in results:
        assert result["top_label"] in CATEGORY_KEYS
        assert len(result["score_vector"]) == 7
        assert all(
            -1.01 <= v <= 1.01 for v in result["score_vector"].values()
        )
        assert result["decision"] == DECISION_LABELED
        assert result["matched_prototypes"], "accept-all must match prototypes"
        for match in result["matched_prototypes"]:
            assert match["category"] in CATEGORY_KEYS
            assert 1 <= match["fact_id"] <= 44
            assert match["lang"] in {"en", "fa"}


def test_uncertain_inputs_never_forced_into_category():
    labeler = _make_labeler(fact_lang="both")
    texts = ["Random unrelated sentence about the weather."]
    embeddings = labeler.embed_samples(texts)
    scores, matches = labeler.score_samples(embeddings, return_matches=True)
    results = labeler.assign_labels(
        scores, threshold=10.0, prototype_indices=matches  # impossible threshold
    )
    result = results[0]
    assert result["decision"] == DECISION_UNCERTAIN
    assert result["labels"] == []  # multi-label: nothing accepted
    assert result["confident"] is False
    assert result["top_label"] in CATEGORY_KEYS  # diagnostic top-1 still present
    # the diagnostic top category keeps its matched prototype for inspection
    top_match = [m for m in result["matched_prototypes"]
                 if m["category"] == result["top_label"]]
    assert top_match


def test_results_reproducible_across_instances():
    texts = ["My child stims when the lights are bright.", "برنامه که عوض می‌شود ناراحت می‌شود."]
    scores_a = _make_labeler().score_samples(_make_labeler().embed_samples(texts))
    scores_b = _make_labeler().score_samples(_make_labeler().embed_samples(texts))
    assert np.array_equal(scores_a, scores_b)


def test_empty_and_whitespace_inputs_are_handled():
    labeler = _make_labeler(fact_lang="en")
    embeddings = labeler.embed_samples(["", "   ", "real text"])
    assert embeddings.shape[0] == 3
    assert np.all(np.isfinite(embeddings))
    scores = labeler.score_samples(embeddings)
    results = labeler.assign_labels(scores)
    assert len(results) == 3


def test_labeler_end_to_end_matches_and_decision(tmp_path):
    labeler = _make_labeler(fact_lang="both", cache_dir=str(tmp_path),
                            enable_cache=True)
    texts = ["We just received the autism diagnosis.", "تشخیص اوتیسم گرفتیم."]
    embeddings = labeler.embed_samples(texts)
    scores, matches = labeler.score_samples(embeddings, return_matches=True)
    results = labeler.assign_labels(scores, prototype_indices=matches)
    assert len(results) == len(texts)
    for result in results:
        assert set(result) >= {
            "top_label", "top_score", "labels", "score_vector", "confident",
            "decision", "margin", "score_type", "labeler_version",
            "matched_prototypes",
        }


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------


def test_embedding_cache_roundtrip(tmp_path: Path):
    cache_dir = str(tmp_path / "cache")
    encoder_a = FakeEncoder()
    labeler_a = SemanticLabeler(
        model_key="minilm", fact_lang="en", _encoder=encoder_a,
        enable_cache=True, cache_dir=cache_dir,
    )
    vectors_a = labeler_a.embed_samples(["text one", "متن دو"])
    prototype_calls = encoder_a.encode_calls  # 1 call: prototypes

    encoder_b = FakeEncoder()
    labeler_b = SemanticLabeler(
        model_key="minilm", fact_lang="en", _encoder=encoder_b,
        enable_cache=True, cache_dir=cache_dir,
    )
    vectors_b = labeler_b.embed_samples(["text one", "متن دو"])
    # Everything (prototypes + samples) served from disk: no encoder calls.
    assert encoder_b.encode_calls == 0
    assert np.allclose(vectors_a, vectors_b, atol=0)
    assert prototype_calls >= 1

    # New text must still be computed, then cached for a third instance.
    labeler_b.embed_samples(["brand new text"])
    assert encoder_b.encode_calls == 1
    encoder_c = FakeEncoder()
    labeler_c = SemanticLabeler(
        model_key="minilm", fact_lang="en", _encoder=encoder_c,
        enable_cache=True, cache_dir=cache_dir,
    )
    labeler_c.embed_samples(["brand new text"])
    assert encoder_c.encode_calls == 0


def test_embedding_cache_disabled_never_writes(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    encoder = FakeEncoder()
    labeler = SemanticLabeler(
        model_key="minilm", fact_lang="en", _encoder=encoder,
        enable_cache=False, cache_dir=str(cache_dir),
    )
    labeler.embed_samples(["some text"])
    labeler.embed_samples(["some text"])  # no cache → recomputed
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_embedding_cache_corrupt_files_rebuild(tmp_path: Path):
    from embedding_cache import EmbeddingCache

    cache_dir = tmp_path / "cache"
    encoder = FakeEncoder()
    labeler = SemanticLabeler(
        model_key="minilm", fact_lang="en", _encoder=encoder,
        enable_cache=True, cache_dir=str(cache_dir),
    )
    labeler.embed_samples(["cached text"])
    matrix_path = next(cache_dir.glob("*.npy"))
    matrix_path.write_bytes(b"definitely not a npy file")

    reloaded = EmbeddingCache(
        model_name=labeler.model_name, prefix_mode=labeler.prefix_mode,
        cache_dir=cache_dir, enabled=True,
    )
    assert reloaded.size == 0  # corrupt cache transparently discarded
    hits, matrix, missing = reloaded.lookup(["cached text"], role="query")
    assert hits == [] and missing == [0]
    reloaded.store(["cached text"], np.ones((1, 4), dtype=np.float32), role="query")
    hits, matrix, missing = reloaded.lookup(["cached text"], role="query")
    assert hits == [0] and missing == [] and matrix.shape == (1, 4)
