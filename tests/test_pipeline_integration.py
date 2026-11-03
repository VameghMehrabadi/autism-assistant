"""Offline integration tests for main.run_pipeline using the FakeEncoder seam.

No model downloads: the real SemanticLabeler is constructed with a
deterministic fake encoder and injected into main. The dataset is the
synthetic Persian fixture. No real patient conversations are used anywhere.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import main
from facts import CATEGORY_KEYS
from labeler import LABELER_VERSION, SemanticLabeler

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"


class FakeEncoder:
    """Same deterministic stand-in as in test_labeler (kept self-contained)."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False, **kwargs):
        import hashlib

        self.encode_calls += 1
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


@pytest.fixture()
def offline_labeler(monkeypatch):
    """Patch main.SemanticLabeler to a deterministic offline instance."""
    real = SemanticLabeler(
        model_key="minilm", fact_lang="both",
        _encoder=FakeEncoder(), enable_cache=False,
    )
    monkeypatch.setattr(main, "SemanticLabeler", lambda **kwargs: real)
    return real


def test_run_pipeline_end_to_end_offline(tmp_path, offline_labeler):
    report = main.run_pipeline(
        limit=5,
        model_key="minilm",
        fact_lang="both",
        dataset="fa",
        dataset_path=str(FIXTURE),
        output_dir=tmp_path,
        run_meta={"source": "phase0-test"},
    )

    assert report["n_samples"] == 5

    labeled = pd.read_csv(tmp_path / "labeled.csv")
    assert len(labeled) == 5
    for column in (
        "idx", "text", "top_label", "top_score", "labels", "confident",
        "decision", "margin", "matched_fact_ids", *(f"score_{c}" for c in CATEGORY_KEYS),
    ):
        assert column in labeled.columns, f"missing column {column}"
    assert set(labeled["decision"]).issubset({"labeled", "uncertain"})
    assert set(labeled["top_label"]).issubset(set(CATEGORY_KEYS))

    scores = np.load(tmp_path / "scores.npy")
    assert scores.shape == (5, len(CATEGORY_KEYS))
    assert np.all(np.isfinite(scores))

    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["labeler_version"] == LABELER_VERSION
    assert meta["model_key"] == "minilm"
    assert meta["source"] == "phase0-test"

    gap = json.loads((tmp_path / "gap_analysis.json").read_text(encoding="utf-8"))
    assert gap["n_samples"] == 5
    assert set(gap["per_category"]) == set(CATEGORY_KEYS)
    assert (tmp_path / "gap_analysis.md").exists()
    assert (tmp_path / "gap_distribution.png").exists()


def test_run_pipeline_respects_limit(tmp_path, offline_labeler):
    report = main.run_pipeline(
        limit=3,
        model_key="minilm",
        fact_lang="fa",
        dataset="fa",
        dataset_path=str(FIXTURE),
        output_dir=tmp_path,
    )
    assert report["n_samples"] == 3
    assert len(pd.read_csv(tmp_path / "labeled.csv")) == 3


def test_run_pipeline_is_reproducible(tmp_path, offline_labeler):
    kwargs = dict(
        limit=5,
        model_key="minilm",
        fact_lang="both",
        dataset="fa",
        dataset_path=str(FIXTURE),
    )
    main.run_pipeline(**kwargs, output_dir=tmp_path / "run1")
    main.run_pipeline(**kwargs, output_dir=tmp_path / "run2")

    df1 = pd.read_csv(tmp_path / "run1" / "labeled.csv")
    df2 = pd.read_csv(tmp_path / "run2" / "labeled.csv")
    pd.testing.assert_frame_equal(df1, df2)

    s1 = np.load(tmp_path / "run1" / "scores.npy")
    s2 = np.load(tmp_path / "run2" / "scores.npy")
    assert np.array_equal(s1, s2)


def test_run_pipeline_no_samples_raises(tmp_path, offline_labeler):
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No samples loaded"):
        main.run_pipeline(
            limit=5,
            model_key="minilm",
            fact_lang="fa",
            dataset="fa",
            dataset_path=str(empty_jsonl),
            output_dir=tmp_path / "out",
        )


# --------------------------------------------------------------------------
# CLI completion (Phase-3b): explicit text-mode threading + new flags
# --------------------------------------------------------------------------


def test_run_pipeline_text_mode_no_global_mutation(tmp_path, offline_labeler, monkeypatch):
    """text_mode must be threaded explicitly, not by mutating global config."""
    monkeypatch.setattr(config, "LABEL_TEXT_MODE", "patient", raising=False)
    main.run_pipeline(
        limit=3,
        model_key="minilm",
        fact_lang="fa",
        dataset="fa",
        dataset_path=str(FIXTURE),
        output_dir=tmp_path,
        text_mode="all",
    )
    assert config.LABEL_TEXT_MODE == "patient"  # global config untouched
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["label_text_mode"] == "all"
    labeled = pd.read_csv(tmp_path / "labeled.csv")
    # 'all' mode merges instruction+input+output into the label text
    assert labeled["text"].str.contains("instruction:", regex=False).all()


def test_run_pipeline_default_mode_unchanged(tmp_path, offline_labeler):
    main.run_pipeline(
        limit=3,
        model_key="minilm",
        fact_lang="fa",
        dataset="fa",
        dataset_path=str(FIXTURE),
        output_dir=tmp_path,
    )
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["label_text_mode"] == config.LABEL_TEXT_MODE


def test_parse_args_new_flags():
    args = main.parse_args([
        "--sampling", "random", "--seed", "7", "--split", "dev",
    ])
    assert args.sampling == "random"
    assert args.seed == 7
    assert args.split == "dev"
    defaults = main.parse_args([])
    assert defaults.sampling is None and defaults.seed is None and defaults.split is None


def test_run_pipeline_meta_records_sampling_and_split(tmp_path, offline_labeler):
    main.run_pipeline(
        limit=4,
        model_key="minilm",
        fact_lang="fa",
        dataset="fa",
        dataset_path=str(FIXTURE),
        output_dir=tmp_path,
        sampling="first",
        seed=99,
    )
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["sampling"] == "first"
    assert meta["seed"] == 99
