"""طبقه‌بندی embedding-based معنایی با multi-prototype (Phase 3 engine).

روش:
1. امبدینگ prototypeها (فکت‌های A–G به زبان انتخاب‌شده) با مدل چندزبانه.
2. امبدینگ همه‌ی نمونه‌های دیتاست (با کش دیسکی و batch inference).
3. ماتریس cosine similarity نمونه × prototype.
4. امتیاز هر دسته = max (یا میانگین top-k) شباهت روی prototypeهای همان دسته.
5. لیبل top-1 = دسته‌ای با بیشترین امتیاز؛ multi-label = دسته‌هایی بالای آستانه.

مهم درباره‌ی امتیازها / Score semantics
--------------------------------------
- Scores are **cosine similarities** in [-1, 1], NOT probabilities and NOT
  calibrated confidence. ``decision``/``confident`` only mean "the raw
  similarity crossed the configured threshold" — a geometric heuristic,
  not a statistically calibrated posterior. Thresholds must be validated
  against human-labeled data before any clinical use.
- ``top_label`` is always the highest-scoring category and is diagnostic
  only. Inputs below the threshold get an explicit ``uncertain`` decision
  and contribute no accepted label — nothing is ever forced into a
  category.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

import config
from embedding_cache import EmbeddingCache
from facts import CATEGORY_KEYS, get_prototypes
from model_loader import format_offline_state, load_sentence_transformer
from privacy_log import get_logger, log_event

logger = get_logger("autism.labeler")

# Version of the labeling engine; recorded in every structured result.
LABELER_VERSION = "3.0.0"

# Stored in labeled.csv. An empty field means "no accepted label", not "missing".
LABEL_SEPARATOR = "|"
UNLABELED = "NONE"

# Explicit decision values (uncertainty is a first-class outcome).
DECISION_LABELED = "labeled"
DECISION_UNCERTAIN = "uncertain"

# What the numbers are — and are not (cosine similarity, not a probability).
SCORE_TYPE = "cosine_similarity"


def resolve_model(model_key: str | None = None,
                  model_name: str | None = None) -> tuple[str, str, str]:
    """برگرداند (model_key, hf_name, prefix_mode)."""
    if model_name:
        # اگر نام کامل داده شده، prefix را از رجیستری حدس بزن
        for key, meta in config.EMBED_MODELS.items():
            if meta["name"] == model_name:
                return key, model_name, meta["prefix"]
        # مدل خارج از رجیستری
        prefix = "e5" if "e5" in model_name.lower() else "none"
        return "custom", model_name, prefix

    key = model_key or config.EMBED_MODEL_KEY
    if key not in config.EMBED_MODELS:
        raise ValueError(
            f"Unknown model key '{key}'. "
            f"Choose from: {sorted(config.EMBED_MODELS)}"
        )
    meta = config.EMBED_MODELS[key]
    return key, meta["name"], meta["prefix"]


def apply_e5_prefix(texts: list[str], role: str) -> list[str]:
    """prefix استاندارد multilingual-e5: query: / passage:"""
    tag = "query" if role == "query" else "passage"
    out: list[str] = []
    for t in texts:
        s = t.strip()
        if s.startswith("query:") or s.startswith("passage:"):
            out.append(s)
        else:
            out.append(f"{tag}: {s}")
    return out


def normalize_for_embedding(text: object) -> str:
    """Model-agnostic text normalization applied before embedding.

    Collapses all whitespace (including newlines and ZWNJ handling is left
    intact inside words) and strips the ends. Deterministic and versioned
    through ``LABELER_VERSION`` so results stay reproducible.
    """
    if text is None:
        return ""
    return " ".join(str(text).split())


def resolve_threshold(
    category: str | None,
    threshold: float | dict[str, float] | None,
    default: float,
) -> float:
    """Resolve the threshold for one category.

    Resolution order: an explicit ``float`` wins; a ``dict`` maps category
    → threshold with ``default`` for missing keys; ``None`` uses the
    optional per-category overrides in ``config.CATEGORY_THRESHOLDS`` and
    falls back to ``default`` (``config.LABEL_THRESHOLD``).
    """
    if threshold is None:
        overrides = getattr(config, "CATEGORY_THRESHOLDS", None) or {}
        if category is not None and category in overrides:
            return float(overrides[category])
        return float(default)
    if isinstance(threshold, dict):
        if category is not None and category in threshold:
            return float(threshold[category])
        return float(default)
    return float(threshold)


class SemanticLabeler:
    """Multi-prototype semantic labeler over the seven autism categories."""

    def __init__(
        self,
        model_name: str | None = None,
        model_key: str | None = None,
        fact_lang: str | None = None,
        device: str | None = None,
        prefix_mode: str | None = None,
        batch_size: int | None = None,
        max_seq_tokens: int | None = None,
        enable_cache: bool | None = None,
        cache_dir: str | None = None,
        _encoder: Any | None = None,
    ):
        """``_encoder`` is a test seam: an object with an ``encode`` API
        compatible with ``SentenceTransformer``. When given, no model is
        downloaded and no GPU is touched (offline, deterministic tests)."""
        self.model_key, resolved_name, resolved_prefix = resolve_model(
            model_key=model_key, model_name=model_name
        )
        self.model_name = resolved_name
        self.prefix_mode = prefix_mode or resolved_prefix
        self.fact_lang = (fact_lang or config.FACT_LANG).lower().strip()
        self.batch_size = int(batch_size or config.EMBED_BATCH_SIZE)
        configured_tokens = config.EMBED_MAX_SEQ_TOKENS.get(self.model_key)
        self.max_seq_tokens = (
            max_seq_tokens if max_seq_tokens is not None else configured_tokens
        )

        if _encoder is not None:
            self.device = device or "cpu"
            self.model = _encoder
        else:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            print(
                f"[labeler] Loading '{self.model_name}' "
                f"(key={self.model_key}, prefix={self.prefix_mode}, "
                f"fact_lang={self.fact_lang}) on {device} "
                f"[weights cache: {config.MODEL_CACHE_DIR}; "
                f"offline: {format_offline_state()}] ..."
            )
            # Single weights-cache configuration (Issue #14): missing or
            # offline-unavailable weights raise an actionable
            # ModelWeightsMissingError naming model, cache and prefetch cmd.
            self.model = load_sentence_transformer(
                self.model_name,
                model_key=self.model_key,
                device=device,
                cache_folder=str(config.MODEL_CACHE_DIR),
            )
            if self.max_seq_tokens:
                self.model.max_seq_length = int(self.max_seq_tokens)

        self.cat_keys = CATEGORY_KEYS
        self.prototypes = get_prototypes(lang=self.fact_lang)
        if not self.prototypes:
            raise RuntimeError(f"No prototypes for fact_lang={self.fact_lang!r}")

        self.prototype_meta = [dict(p) for p in self.prototypes]
        self.proto_texts = [p["text"] for p in self.prototypes]
        self.proto_cats = [p["category"] for p in self.prototypes]
        self.proto_cat_indices = {
            cat: np.fromiter(
                (i for i, c in enumerate(self.proto_cats) if c == cat),
                dtype=np.int64,
            )
            for cat in self.cat_keys
        }

        if enable_cache is None:
            enable_cache = config.EMBED_CACHE_ENABLED
        self._cache = EmbeddingCache(
            model_name=self.model_name,
            prefix_mode=self.prefix_mode,
            max_length=self.max_seq_tokens or 0,
            normalize=config.NORMALIZE_EMBEDDINGS,
            cache_dir=cache_dir,
            enabled=bool(enable_cache),
        )

        self._proto_emb = self._embed_prototypes()

    # ------------------------------------------------------------------
    # Embedding (normalization → disk cache → batched encode)
    # ------------------------------------------------------------------

    def _prepare_texts(self, texts: list[str], role: str) -> list[str]:
        if self.prefix_mode == "e5":
            return apply_e5_prefix(texts, role=role)
        return texts

    def _encode_raw(
        self, texts: list[str], role: str, batch_size: int
    ) -> np.ndarray:
        prepared = self._prepare_texts(texts, role=role)
        embs = self.model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
            show_progress_bar=len(prepared) >= 64,
        )
        return np.asarray(embs, dtype=np.float32)

    def _embed(
        self,
        texts: list[str],
        role: str,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed texts with cache lookup, batched computation, ordered output."""
        batch_size = int(batch_size or self.batch_size)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        normalized = [normalize_for_embedding(t) for t in texts]

        hit_positions: list[int] = []
        hit_matrix: np.ndarray | None = None
        missing: list[int] = list(range(len(normalized)))
        if self._cache.enabled:
            hit_positions, hit_matrix, missing = self._cache.lookup(
                normalized, role=role
            )

        new_vectors: np.ndarray | None = None
        if missing:
            missing_texts = [normalized[i] for i in missing]
            # Incremental persistence: encode and store in chunks so an
            # interruption loses at most one chunk of computed embeddings
            # instead of the whole pass.
            chunk_size = max(1, int(getattr(config, "EMBED_CACHE_WRITE_CHUNK", 256)))
            parts: list[np.ndarray] = []
            for start in range(0, len(missing_texts), chunk_size):
                chunk_texts = missing_texts[start : start + chunk_size]
                chunk_vectors = self._encode_raw(chunk_texts, role, batch_size)
                self._cache.store(chunk_texts, chunk_vectors, role=role)
                parts.append(chunk_vectors)
            if parts:
                new_vectors = (
                    parts[0] if len(parts) == 1 else np.vstack(parts)
                )

        dim = 0
        if new_vectors is not None:
            dim = new_vectors.shape[1]
        elif hit_matrix is not None and hit_matrix.size:
            dim = hit_matrix.shape[1]
        out = np.zeros((len(normalized), dim), dtype=np.float32)
        if hit_matrix is not None and hit_matrix.size:
            out[hit_positions] = hit_matrix
        if new_vectors is not None:
            out[missing] = new_vectors

        log_event(
            logger,
            "embed_complete",
            role=role,
            model=self.model_name,
            n_texts=len(normalized),
            n_cached=len(hit_positions),
            n_computed=0 if new_vectors is None else len(missing),
        )
        return out

    def _embed_prototypes(self) -> np.ndarray:
        # facts/prototypes as passages for e5 retrieval-style matching
        emb = self._embed(self.proto_texts, role="passage", batch_size=32)
        print(f"[labeler] Prototype embeddings: {emb.shape}")
        return emb

    def embed_samples(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """امبدینگ نمونه‌ها (query role؛ برای e5 با prefix «query:»)."""
        return self._embed(texts, role="query", batch_size=batch_size)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_samples(
        self,
        sample_embs: np.ndarray,
        top_k: int = config.TOP_K_PROTOTYPES_PER_CAT,
        return_matches: bool = False,
    ):
        """برگرداند ماتریس (n_samples, n_categories) از امتیاز cosine.

        With ``return_matches=True`` also returns an ``(n_samples,
        n_categories)`` int matrix holding, per category, the index of the
        best-matching prototype (``-1`` where a category has no prototypes).
        With ``top_k > 1`` the score is the mean of the top-k similarities
        while the matched prototype is still the single best one.
        """
        sim = sample_embs @ self._proto_emb.T  # (N, P)
        n_samples = sim.shape[0]
        scores = np.zeros((n_samples, len(self.cat_keys)), dtype=np.float32)
        matches = (
            np.full((n_samples, len(self.cat_keys)), -1, dtype=np.int64)
            if return_matches
            else None
        )

        for ci, cat in enumerate(self.cat_keys):
            cols = self.proto_cat_indices[cat]
            if cols.size == 0:
                continue
            sub = sim[:, cols]  # (N, P_cat)
            best = sub.argmax(axis=1)
            if top_k <= 1:
                scores[:, ci] = sub[np.arange(n_samples), best]
            else:
                k = min(top_k, sub.shape[1])
                part = np.partition(sub, -k, axis=1)[:, -k:]
                scores[:, ci] = part.mean(axis=1)
            if matches is not None:
                matches[:, ci] = cols[best]
        if return_matches:
            return scores, matches
        return scores

    # ------------------------------------------------------------------
    # Label decisions
    # ------------------------------------------------------------------

    def assign_labels(
        self,
        scores: np.ndarray,
        threshold: float | dict[str, float] | None = None,
        multi_label: bool = config.MULTI_LABEL,
        prototype_indices: np.ndarray | None = None,
    ) -> list[dict]:
        """Decide every row of ``scores`` (see :func:`assign_label_row`).

        ``prototype_indices`` (from ``score_samples(..., return_matches=True)``)
        enriches each result with ``matched_prototypes`` identifiers.
        """
        results = [
            assign_label_row(
                scores[i],
                self.cat_keys,
                threshold=threshold,
                multi_label=multi_label,
            )
            for i in range(scores.shape[0])
        ]
        if prototype_indices is not None:
            for i, result in enumerate(results):
                result["matched_prototypes"] = self._matches_for_row(
                    result, prototype_indices[i]
                )
        return results

    def _matches_for_row(
        self, result: dict, prototype_index_row: np.ndarray
    ) -> list[dict]:
        """Map a decision row to its matched prototype identifiers.

        Included: every accepted label plus the diagnostic top category.
        """
        meta = getattr(self, "prototype_meta", None) or []
        accepted = set(result["labels"]) - {UNLABELED}
        matched: list[dict] = []
        for ci, cat in enumerate(self.cat_keys):
            if cat not in accepted and cat != result["top_label"]:
                continue
            p_idx = int(prototype_index_row[ci])
            if 0 <= p_idx < len(meta):
                m = meta[p_idx]
                matched.append(
                    {
                        "category": cat,
                        "fact_id": m["fact_id"],
                        "lang": m["lang"],
                        "score": result["score_vector"][cat],
                    }
                )
        return matched


def assign_label_row(
    row: np.ndarray,
    categories: list[str],
    threshold: float | dict[str, float] | None = None,
    multi_label: bool = config.MULTI_LABEL,
) -> dict:
    """Decide one sample.

    top_label is always the highest-scoring category and is diagnostic only.
    labels is the set allowed to enter later stages:
    - multi-label: every category at or above its resolved threshold, best
      score first. This list is empty when nothing clears the threshold.
    - single-label: the top category, or "NONE" when it is below the threshold.
    confident is true only when the top score clears the top category's
    threshold; decision makes the uncertain outcome explicit. Ties are broken
    deterministically by the lowest category index. Non-finite scores are
    rejected instead of silently producing garbage labels.
    """
    values = np.asarray(row, dtype=float)
    if values.ndim != 1 or len(categories) != len(values):
        raise ValueError(
            f"Expected {len(categories)} scores, got shape {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Score vector contains non-finite values (NaN/inf); "
            "cannot assign labels."
        )

    # Deterministic: score descending, ties broken by lowest category index.
    order = np.lexsort((np.arange(len(values)), -values))
    top_idx = int(order[0])
    top_score = float(values[top_idx])
    second_score = float(values[order[1]]) if len(order) > 1 else None
    top_threshold = resolve_threshold(categories[top_idx], threshold, config.LABEL_THRESHOLD)
    confident = top_score >= top_threshold

    if multi_label:
        labels = [
            categories[int(j)]
            for j in order
            if float(values[int(j)])
            >= resolve_threshold(categories[int(j)], threshold, config.LABEL_THRESHOLD)
        ]
    else:
        labels = [categories[top_idx]] if confident else [UNLABELED]

    accepted_exists = any(label != UNLABELED for label in labels)
    return {
        "top_label": categories[top_idx],
        "top_score": round(top_score, 4),
        "labels": labels,
        "score_vector": {
            categories[j]: round(float(values[j]), 4) for j in range(len(categories))
        },
        "confident": confident,
        "decision": DECISION_LABELED if accepted_exists else DECISION_UNCERTAIN,
        "margin": (
            round(top_score - second_score, 4) if second_score is not None else None
        ),
        "score_type": SCORE_TYPE,
        "labeler_version": LABELER_VERSION,
        "matched_prototypes": [],
    }


def format_labels(labels: list[str]) -> str:
    """Serialize accepted labels. An empty list stays an empty cell."""
    return LABEL_SEPARATOR.join(labels)


def parse_labels(raw: object, fallback_top: str | None = None) -> list[str]:
    """Read accepted labels back from a CSV cell or an in-memory list.

    A blank or missing cell is an explicit empty decision and is not replaced
    by top_label. fallback_top is used only when the labels column itself is
    absent from an older file.
    """
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if item and str(item) != UNLABELED]
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [part for part in raw.split(LABEL_SEPARATOR) if part and part != UNLABELED]
    if fallback_top and fallback_top != UNLABELED:
        return [fallback_top]
    return []


def label_texts(
    texts: list[str],
    model_key: str | None = None,
    fact_lang: str | None = None,
    threshold: float | dict[str, float] | None = None,
    multi_label: bool = config.MULTI_LABEL,
) -> tuple[list[dict], np.ndarray]:
    """راحت‌ترین API: متن‌ها → (لیست نتایج ساختاریافته، ماتریس امتیازها)."""
    labeler = SemanticLabeler(model_key=model_key, fact_lang=fact_lang)
    embs = labeler.embed_samples(texts)
    scores, matches = labeler.score_samples(embs, return_matches=True)
    results = labeler.assign_labels(
        scores,
        threshold=threshold,
        multi_label=multi_label,
        prototype_indices=matches,
    )
    return results, scores


if __name__ == "__main__":
    demo = [
        "My child flaps his hands and gets very upset when the lights are too bright.",
        "He has a deep passion for trains and can talk about them for hours.",
        "We just got the autism diagnosis after the developmental assessment.",
        "I feel anxious when my daily routine changes unexpectedly.",
    ]
    res, _ = label_texts(demo)
    for txt, r in zip(demo, res):
        print("-" * 60)
        print(txt)
        print(
            f"  -> {r['decision']}: top {r['top_label']} ({r['top_score']}) | "
            f"labels: {r['labels']} | margin: {r['margin']}"
        )
        for m in r["matched_prototypes"]:
            print(f"     [{m['category']}] fact {m['fact_id']} ({m['lang']}): {m['score']}")
