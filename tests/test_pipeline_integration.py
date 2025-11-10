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


# --------------------------------------------------------------------------
# Issue #20: label text mode isolation regression tests
# --------------------------------------------------------------------------


def test_sequential_mode_isolation_patient_then_all(tmp_path, offline_labeler):
    """Issue #20 Test 1: patient → all in sequence must produce independent
    results with the correct mode in each run_meta, and must not mutate
    config.LABEL_TEXT_MODE."""
    kwargs = dict(
        limit=3, model_key="minilm", fact_lang="fa",
        dataset="fa", dataset_path=str(FIXTURE),
    )
    main.run_pipeline(**kwargs, output_dir=tmp_path / "run1", text_mode="patient")
    main.run_pipeline(**kwargs, output_dir=tmp_path / "run2", text_mode="all")

    meta1 = json.loads((tmp_path / "run1" / "run_meta.json").read_text(encoding="utf-8"))
    meta2 = json.loads((tmp_path / "run2" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta1["label_text_mode"] == "patient"
    assert meta2["label_text_mode"] == "all"

    df1 = pd.read_csv(tmp_path / "run1" / "labeled.csv")
    df2 = pd.read_csv(tmp_path / "run2" / "labeled.csv")
    # patient mode → label text is the input field (not the merged text)
    assert not df1["text"].str.contains("instruction:", regex=False).any()
    # all mode → label text merges instruction+input+output
    assert df2["text"].str.contains("instruction:", regex=False).all()


def test_sequential_mode_isolation_all_then_patient(tmp_path, offline_labeler):
    """Issue #20 Test 2: reverse order (all → patient) must produce
    the same independent results."""
    kwargs = dict(
        limit=3, model_key="minilm", fact_lang="fa",
        dataset="fa", dataset_path=str(FIXTURE),
    )
    main.run_pipeline(**kwargs, output_dir=tmp_path / "run1", text_mode="all")
    main.run_pipeline(**kwargs, output_dir=tmp_path / "run2", text_mode="patient")

    df1 = pd.read_csv(tmp_path / "run1" / "labeled.csv")
    df2 = pd.read_csv(tmp_path / "run2" / "labeled.csv")
    assert df1["text"].str.contains("instruction:", regex=False).all()
    assert not df2["text"].str.contains("instruction:", regex=False).any()


def test_global_config_immutability_both_modes(tmp_path, offline_labeler):
    """Issue #20 Test 3: config.LABEL_TEXT_MODE must remain unchanged after
    pipeline calls with both supported modes."""
    initial = config.LABEL_TEXT_MODE
    main.run_pipeline(
        limit=2, model_key="minilm", fact_lang="fa", dataset="fa",
        dataset_path=str(FIXTURE), output_dir=tmp_path / "r1", text_mode="patient",
    )
    assert config.LABEL_TEXT_MODE == initial
    main.run_pipeline(
        limit=2, model_key="minilm", fact_lang="fa", dataset="fa",
        dataset_path=str(FIXTURE), output_dir=tmp_path / "r2", text_mode="all",
    )
    assert config.LABEL_TEXT_MODE == initial


def test_regression_guard_no_global_assignment_during_execution(tmp_path, offline_labeler):
    """Issue #20 Test 4: explicit regression guard — detects any mutation
    of config.LABEL_TEXT_MODE during pipeline execution, not just after."""
    sentinel = object()
    original = config.LABEL_TEXT_MODE

    # Track every attribute write on the config module during execution.
    writes = []
    real_setattr = type(config).__setattr__

    class ConfigSpy:
        def __getattr__(self, name):
            return getattr(config, name)

        def __setattr__(self, name, value):
            writes.append((name, value))
            real_setattr(config, name, value)

    # Simple approach: assert immediately after each call.
    for mode in ("patient", "all"):
        main.run_pipeline(
            limit=2, model_key="minilm", fact_lang="fa", dataset="fa",
            dataset_path=str(FIXTURE), output_dir=tmp_path / f"guard_{mode}",
            text_mode=mode,
        )
        assert config.LABEL_TEXT_MODE == original, (
            f"config.LABEL_TEXT_MODE was mutated during execution with mode={mode!r}"
        )


def test_invalid_text_mode_rejected_safely(tmp_path, offline_labeler):
    """Issue #20 Test 5: invalid mode values must be rejected without
    modifying global configuration and without producing partial output."""
    initial = config.LABEL_TEXT_MODE
    with pytest.raises((ValueError, RuntimeError)):
        main.run_pipeline(
            limit=2, model_key="minilm", fact_lang="fa", dataset="fa",
            dataset_path=str(FIXTURE), output_dir=tmp_path,
            text_mode="counselor",  # not a valid mode
        )
    assert config.LABEL_TEXT_MODE == initial
    assert not (tmp_path / "run_meta.json").exists()


def test_interleaved_modes_independent_results(tmp_path, offline_labeler):
    """Issue #20 Test 7: interleaved pipeline calls with alternating modes
    produce results that consistently match each call's own mode."""
    kwargs = dict(
        limit=2, model_key="minilm", fact_lang="fa",
        dataset="fa", dataset_path=str(FIXTURE),
    )
    dirs_and_modes = [
        (tmp_path / "p1", "patient"),
        (tmp_path / "a1", "all"),
        (tmp_path / "p2", "patient"),
        (tmp_path / "a2", "all"),
    ]
    for out_dir, mode in dirs_and_modes:
        main.run_pipeline(**kwargs, output_dir=out_dir, text_mode=mode)

    for out_dir, mode in dirs_and_modes:
        meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["label_text_mode"] == mode, f"mode mismatch in {out_dir}"
        df = pd.read_csv(out_dir / "labeled.csv")
        if mode == "all":
            assert df["text"].str.contains("instruction:", regex=False).all()
        else:
            assert not df["text"].str.contains("instruction:", regex=False).any()


def test_no_global_label_text_mode_assignment_in_source():
    """Issue #20 Test 4 (static): scans source files for assignments to
    config.LABEL_TEXT_MODE outside config.py. Fails if any are found."""
    import re

    project_root = Path(__file__).resolve().parent.parent
    for py_file in project_root.glob("*.py"):
        if py_file.name == "config.py":
            continue
        source = py_file.read_text(encoding="utf-8")
        matches = re.findall(
            r"config\.LABEL_TEXT_MODE\s*=\s*(?!.*#.*read)", source
        )
        assert not matches, (
            f"{py_file.name} contains assignment(s) to config.LABEL_TEXT_MODE: "
            f"{matches} — use explicit function parameters instead."
        )
