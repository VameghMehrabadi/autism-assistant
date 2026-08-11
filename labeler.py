"""طبقه‌بندی embedding-based معنایی با multi-prototype.

روش:
1. امبدینگ prototypeها (فکت‌های A–G به زبان انتخاب‌شده) با مدل چندزبانه.
2. امبدینگ همه‌ی نمونه‌های دیتاست.
3. ماتریس cosine similarity نمونه × prototype.
4. امتیاز هر دسته = max (یا میانگین top-k) شباهت روی prototypeهای همان دسته.
5. لیبل top-1 = دسته‌ای با بیشترین امتیاز؛ multi-label = دسته‌هایی بالای آستانه.

این روش «معنا» را مقایسه می‌کند، نه کلمه؛ بنابراین متون دیتاست و
فکت‌ها را در یک فضای مشترک می‌نشاند.
"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config
from facts import CATEGORY_KEYS, get_prototypes


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


class SemanticLabeler:
    def __init__(
        self,
        model_name: str | None = None,
        model_key: str | None = None,
        fact_lang: str | None = None,
        device: str | None = None,
        prefix_mode: str | None = None,
    ):
        self.model_key, resolved_name, resolved_prefix = resolve_model(
            model_key=model_key, model_name=model_name
        )
        self.model_name = resolved_name
        self.prefix_mode = prefix_mode or resolved_prefix
        self.fact_lang = (fact_lang or config.FACT_LANG).lower().strip()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(
            f"[labeler] Loading '{self.model_name}' "
            f"(key={self.model_key}, prefix={self.prefix_mode}, "
            f"fact_lang={self.fact_lang}) on {device} ..."
        )
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=str(config.CACHE_DIR),
        )
        self.cat_keys = CATEGORY_KEYS
        self.prototypes = get_prototypes(lang=self.fact_lang)
        if not self.prototypes:
            raise RuntimeError(f"No prototypes for fact_lang={self.fact_lang!r}")

        self.proto_texts = [p["text"] for p in self.prototypes]
        self.proto_cats = [p["category"] for p in self.prototypes]
        self.proto_cat_indices = {
            cat: np.fromiter(
                (i for i, c in enumerate(self.proto_cats) if c == cat),
                dtype=np.int64,
            )
            for cat in self.cat_keys
        }
        self._proto_emb = self._embed_prototypes()

    def _prepare_texts(self, texts: list[str], role: str) -> list[str]:
        if self.prefix_mode == "e5":
            return apply_e5_prefix(texts, role=role)
        return texts

    def _embed(self, texts: list[str], batch_size: int,
               desc: str = "embedding") -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embs.astype(np.float32)

    def _embed_prototypes(self) -> np.ndarray:
        # facts/prototypes as passages for e5 retrieval-style matching
        texts = self._prepare_texts(self.proto_texts, role="passage")
        emb = self._embed(texts, batch_size=32, desc="prototypes")
        print(f"[labeler] Prototype embeddings: {emb.shape}")
        return emb

    def embed_samples(self, texts: list[str]) -> np.ndarray:
        prepared = self._prepare_texts(texts, role="query")
        return self._embed(
            prepared, batch_size=config.EMBED_BATCH_SIZE, desc="samples"
        )

    def score_samples(
        self,
        sample_embs: np.ndarray,
        top_k: int = config.TOP_K_PROTOTYPES_PER_CAT,
    ) -> np.ndarray:
        """برگرداند ماتریس (n_samples, n_categories) از امتیاز cosine."""
        sim = sample_embs @ self._proto_emb.T  # (N, P)
        n_samples = sim.shape[0]
        scores = np.zeros((n_samples, len(self.cat_keys)), dtype=np.float32)

        for ci, cat in enumerate(self.cat_keys):
            cols = self.proto_cat_indices[cat]
            if cols.size == 0:
                continue
            sub = sim[:, cols]  # (N, P_cat)
            if top_k <= 1:
                scores[:, ci] = sub.max(axis=1)
            else:
                k = min(top_k, sub.shape[1])
                part = np.partition(sub, -k, axis=1)[:, -k:]
                scores[:, ci] = part.mean(axis=1)
        return scores

    def assign_labels(
        self,
        scores: np.ndarray,
        threshold: float = config.LABEL_THRESHOLD,
        multi_label: bool = config.MULTI_LABEL,
    ) -> list[dict]:
        results: list[dict] = []
        order = np.argsort(scores, axis=1)[:, ::-1]
        n_categories = scores.shape[1]

        for i in range(scores.shape[0]):
            row_order = order[i]
            top_idx = int(row_order[0])
            top_cat = self.cat_keys[top_idx]
            top_score = float(scores[i, top_idx])

            if multi_label:
                above = [
                    self.cat_keys[j]
                    for j in row_order
                    if scores[i, j] >= threshold
                ]
                labels = above if above else [top_cat]
            else:
                labels = [top_cat] if top_score >= threshold else ["NONE"]

            results.append({
                "top_label": top_cat,
                "top_score": round(top_score, 4),
                "labels": labels,
                "score_vector": {
                    self.cat_keys[j]: round(float(scores[i, j]), 4)
                    for j in range(n_categories)
                },
                "confident": top_score >= threshold,
            })
        return results


def label_texts(
    texts: list[str],
    model_key: str | None = None,
    fact_lang: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    """راحت‌ترین API: متن‌ها → (لیست نتایج، ماتریس امتیازها)."""
    labeler = SemanticLabeler(model_key=model_key, fact_lang=fact_lang)
    embs = labeler.embed_samples(texts)
    scores = labeler.score_samples(embs)
    results = labeler.assign_labels(scores)
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
            f"  -> top: {r['top_label']} ({r['top_score']}) | "
            f"labels: {r['labels']} | scores: {r['score_vector']}"
        )
