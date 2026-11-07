"""Issue #9 — historical run_meta alignment metadata.

Scenarios: modern runs verify; historical runs classify as legacy with
stable reasons (never an exception, never falsely verified); retroactive
migration is idempotent, evidence-derived, and never touches labeled.csv or
run_meta.json; the comparison report distinguishes modern from legacy runs
deterministically. All fixtures are synthetic, in isolated tmp directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import alignment
import config
from compare_runs import (
    build_comparison_table,
    render_comparison_markdown,
    summarize_run,
    verify_alignment,
)
from facts import CATEGORY_KEYS

SENTINEL_TEXTS = [
    "synthetic conversation alpha one",
    "synthetic conversation beta two",
    "synthetic conversation gamma three",
]


def _modern_meta(texts: list[str]) -> dict:
    return {
        "model_key": "minilm",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": "none",
        "fact_lang": "fa",
        "label_text_mode": "patient",
        "labeler_version": "3.0.0",
        "dataset_fingerprint": alignment.ordered_input_fingerprint(texts),
        "text_window": 512,
        "preprocess_version": "2.0.0",
        "threshold": 0.35,
        "dataset_total": len(texts),
    }


def _legacy_meta() -> dict:
    return {
        "model_key": "minilm",
        "prefix": "none",
        "fact_lang": "fa",
        "labeler_version": "3.0.0",
        "label_text_mode": "patient",
        "id": "legacy_run",
    }


def _make_run_dir(
    base: Path,
    run_id: str,
    *,
    meta: dict | None = None,
    meta_raw: str | None = None,
    gap_raw: str | None = None,
    texts: list[str] | None = None,
) -> Path:
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    texts = texts or SENTINEL_TEXTS
    pd.DataFrame({
        "idx": range(len(texts)),
        "text": texts,
        "top_label": ["B"] * len(texts),
    }).to_csv(run_dir / "labeled.csv", index=False)
    per_cat = {
        c: {"count": len(texts) if c == "B" else 0}
        for c in CATEGORY_KEYS
    }
    gap = {
        "n_samples": len(texts),
        "threshold": 0.35,
        "per_category": per_cat,
    }
    (run_dir / "gap_analysis.json").write_text(
        gap_raw if gap_raw is not None else json.dumps(gap),
        encoding="utf-8",
    )
    if meta_raw is not None:
        (run_dir / "run_meta.json").write_text(meta_raw, encoding="utf-8")
    elif meta is not None:
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return run_dir


def _synthetic_summary_source(n: int = 3) -> tuple[dict, dict]:
    per = {
        c: {
            "count": n if c == "B" else 0,
            "share": 1.0 if c == "B" else 0.0,
            "multi_count": 0,
            "multi_share": 0.0,
            "mean_score": 0.5,
            "median_score": 0.5,
            "std_score": 0.0,
            "confident_share": 1.0,
            "global_mean_score": 0.4,
        }
        for c in CATEGORY_KEYS
    }
    report = {
        "n_samples": n,
        "threshold": 0.35,
        "n_confident": n,
        "confident_rate": 1.0,
        "n_unlabeled_NONE": 0,
        "per_category": per,
        "global_mean_score_per_category": {c: 0.4 for c in CATEGORY_KEYS},
        "ranking_weak_to_strong": list(CATEGORY_KEYS),
        "weakest": list(CATEGORY_KEYS)[:2],
        "strongest": list(CATEGORY_KEYS)[-2:],
    }
    cfg = {
        "id": "1_fa_en_minilm",
        "title": "synthetic",
        "model_key": "minilm",
        "fact_lang": "fa",
        "dataset": "en",
    }
    return report, cfg


# --------------------------------------------------------------------------
# 1. New runs carry the full execution-time alignment contract
# --------------------------------------------------------------------------


def test_modern_contract_fields_are_declared_and_satisfied(tmp_path):
    """The canonical contract list matches what run_pipeline writes (the
    offline end-to-end check lives in test_evaluation_validity.py)."""
    texts = SENTINEL_TEXTS
    run_dir = _make_run_dir(tmp_path, "modern_run", meta=_modern_meta(texts))
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert set(alignment.MODERN_REQUIRED_FIELDS) <= set(meta)
    assert meta["dataset_fingerprint"] == alignment.ordered_input_fingerprint(
        texts
    )
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] == alignment.CLASSIFICATION_MODERN


# --------------------------------------------------------------------------
# 2+5. --verify-alignment succeeds on modern runs; legacy never verified
# --------------------------------------------------------------------------


def test_verify_alignment_succeeds_on_modern_runs(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    base.mkdir()
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_fa_en_minilm", meta=_modern_meta(SENTINEL_TEXTS))
    _make_run_dir(base, "2_fa_en_bge-m3", meta=_modern_meta(SENTINEL_TEXTS))

    report = verify_alignment()
    assert report["all_records_aligned"] is True
    ok_runs = [r for r in report["runs"] if r.get("status") == "ok"]
    assert all(
        r["alignment"]["classification"] == alignment.CLASSIFICATION_MODERN
        for r in ok_runs
    )
    assert all(r["dataset_fingerprint"] for r in ok_runs)
    assert report["alignment_summary"] == {
        "modern": 2,
        "MISSING_ARTIFACTS": 3,  # runs 3-5 not present in the tmp fixture
    }
    assert report["all_runs_modern"] is True
    assert "NOT dataset-level provenance" in (
        report["all_records_aligned_semantics"]
    )


def test_mixed_modern_and_legacy_verification(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    base.mkdir()
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_fa_en_minilm", meta=_modern_meta(SENTINEL_TEXTS))
    _make_run_dir(base, "2_fa_en_bge-m3", meta=_legacy_meta())

    report = verify_alignment()
    ok_runs = [r for r in report["runs"] if r.get("status") == "ok"]
    classifications = [
        r["alignment"]["classification"] for r in ok_runs
    ]
    assert classifications == [
        alignment.CLASSIFICATION_MODERN,
        alignment.CLASSIFICATION_LEGACY,
    ]
    # record-set alignment is still computed across the modern/legacy mix
    assert report["all_records_aligned"] is True
    assert report["alignment_summary"] == {
        "modern": 1,
        "legacy": 1,
        "MISSING_ARTIFACTS": 3,
    }
    # Issue #9 review fix: unambiguous semantics — record-set correspondence
    # is NOT modern verification, and the legacy run never contributes to it
    assert "NOT dataset-level provenance" in (
        report["all_records_aligned_semantics"]
    )
    assert report["all_runs_modern"] is False


# --------------------------------------------------------------------------
# 3+4. Historical/malformed metadata: stable results, never an exception
# --------------------------------------------------------------------------


def test_legacy_run_without_metadata_is_stable(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    base.mkdir()
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_fa_en_minilm", meta=_legacy_meta())

    report = verify_alignment()  # must not raise
    entry = report["runs"][0]
    assert entry["status"] == "ok"
    assert entry["alignment"]["classification"] == (
        alignment.CLASSIFICATION_LEGACY
    )
    assert "predates" in entry["alignment"]["reason"]
    assert entry["text_fingerprint"]  # artifact-derived evidence still present
    assert entry["dataset_fingerprint"] is None  # execution-time: absent
    assert entry["text_window"] == "pre-alignment-metadata"


def test_malformed_metadata_never_raises(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    base.mkdir()
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(
        base, "1_fa_en_minilm",
        meta_raw="{ this is not json !!",
        gap_raw="also not json {{{",
    )

    report = verify_alignment()  # must not raise
    entry = report["runs"][0]
    assert entry["status"] == "ok"
    assert "malformed" in entry["alignment"]["reason"]
    # the error text never embeds absolute paths — type name only
    assert "JSONDecodeError" in entry.get("gap_report_error", "")
    assert entry["dataset_fingerprint"] is None
    assert entry["threshold"] is None  # not silently treated as matching
    assert entry.get("threshold_match") is None  # unset — honest absence


def test_run_meta_non_object_json_is_safe(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    base.mkdir()
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    _make_run_dir(base, "1_fa_en_minilm", meta_raw='["a", "list"]')

    report = verify_alignment()
    entry = report["runs"][0]
    assert entry["alignment"]["classification"] == (
        alignment.CLASSIFICATION_LEGACY
    )
    assert "not a JSON object" in entry["alignment"]["reason"]


# --------------------------------------------------------------------------
# 6+7+8. Retroactive migration: evidence-derived, idempotent, honest
# --------------------------------------------------------------------------


def test_migration_writes_retroactive_artifact(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())

    summary = alignment.migrate_legacy_alignment([run_dir])
    assert summary["migrated"] == ["1_fa_en_minilm"]
    artifact = json.loads(
        (run_dir / "alignment_legacy.json").read_text(encoding="utf-8")
    )
    assert artifact["provenance"] == "retroactive"
    assert artifact["classification"] == alignment.CLASSIFICATION_LEGACY_PARTIAL
    assert artifact["reconstructed"]["threshold"] == 0.35
    assert (
        artifact["reconstructed"]["threshold_source"] == "gap_analysis.json (execution-time report)"
    )
    assert artifact["reconstructed"]["labeler_version"] == "3.0.0"
    assert artifact["evidence"]["record_count"] == 3
    assert artifact["evidence"]["ordered_input_fingerprint_derived"] == (
        alignment.ordered_input_fingerprint(SENTINEL_TEXTS)
    )
    assert "dataset_fingerprint_execution_time" in artifact["missing"]

    # after migration the run is legacy-partial — never modern
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] == alignment.CLASSIFICATION_LEGACY_PARTIAL


def test_migration_idempotent_and_labeled_csv_bytes_preserved(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())
    labeled = run_dir / "labeled.csv"
    meta_file = run_dir / "run_meta.json"
    before = labeled.read_bytes()
    before_meta = meta_file.read_bytes()
    first_sha = hashlib.sha256(before).hexdigest()

    alignment.migrate_legacy_alignment([run_dir])
    second = alignment.migrate_legacy_alignment([run_dir])
    assert second["migrated"] == []
    assert second["already_migrated"] == ["1_fa_en_minilm"]

    assert labeled.read_bytes() == before          # byte-identical
    assert hashlib.sha256(labeled.read_bytes()).hexdigest() == first_sha
    assert meta_file.read_bytes() == before_meta   # run_meta untouched

    # evidence change (re-labeled artifact) -> artifact regenerated
    (run_dir / "labeled.csv").write_bytes(before + b"extra,row\n")
    third = alignment.migrate_legacy_alignment([run_dir])
    assert third["migrated"] == ["1_fa_en_minilm"]  # evidence changed
    assert labeled.read_bytes() == before + b"extra,row\n"


def test_migration_skips_modern_runs(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm",
                            meta=_modern_meta(SENTINEL_TEXTS))
    summary = alignment.migrate_legacy_alignment([run_dir])
    assert summary["migrated"] == []
    assert summary["skipped"] == [
        ("1_fa_en_minilm", "modern execution-time metadata present — skipped")
    ]
    assert not (run_dir / "alignment_legacy.json").exists()


def test_migration_skips_dirs_without_labeled_csv(tmp_path):
    empty = tmp_path / "4_fa_fa_bge-m3"
    empty.mkdir()
    summary = alignment.migrate_legacy_alignment([empty])
    assert summary["skipped"] == [("4_fa_fa_bge-m3", "no labeled.csv — nothing to migrate")]


# --------------------------------------------------------------------------
# 9. Comparison report distinguishes modern and legacy runs
# --------------------------------------------------------------------------


def _summaries_with_alignment(tmp_path):
    base = tmp_path / "comparison"
    base.mkdir()
    modern_dir = _make_run_dir(base, "1_fa_en_minilm",
                               meta=_modern_meta(SENTINEL_TEXTS))
    legacy_dir = _make_run_dir(base, "2_fa_en_bge-m3", meta=_legacy_meta())
    summaries = []
    for run_dir, cfg_id in (
        (modern_dir, "1_fa_en_minilm"),
        (legacy_dir, "2_fa_en_bge-m3"),
    ):
        report, cfg = _synthetic_summary_source()
        cfg = dict(cfg, id=cfg_id)
        summary = summarize_run(report, cfg)
        summary["alignment"] = alignment.classify_alignment(run_dir)
        summaries.append(summary)
    return summaries


def test_comparison_report_distinguishes_modern_and_legacy(tmp_path):
    summaries = _summaries_with_alignment(tmp_path)
    table = build_comparison_table(summaries)
    md = render_comparison_markdown(summaries, table)

    assert "## Run alignment" in md
    assert "**modern** — execution-time alignment metadata present" in md
    assert "**legacy**" in md
    assert "predates the Phase-4 alignment contract" in md
    assert "NOT execution-time verified" in md
    # Issue #9 review fix: the verified rerun uses the pipeline CLI into a
    # NEW directory (history preserved), not compare_runs --only (which
    # overwrites the historical run directory)
    assert (
        "python main.py --limit 0 --model minilm --fact-lang fa "
        "--dataset en --out outputs/verified_2_fa_en_bge-m3" in md
    )
    assert "classify_alignment('outputs/verified_2_fa_en_bge-m3')" in md
    assert "overwrites `outputs/comparison/<run-id>`" in md
    assert "NOT dataset-level provenance" in md
    # deterministic output
    assert md == render_comparison_markdown(summaries, table)


def test_comparison_report_back_compatible_without_alignment_key():
    """Summaries built by older callers (no alignment key) still render."""
    report, cfg = _synthetic_summary_source()
    summaries = [summarize_run(report, cfg)]
    table = build_comparison_table(summaries)
    md = render_comparison_markdown(summaries, table)
    assert "alignment status not recorded in this report" in md


# --------------------------------------------------------------------------
# Real historical runs (runs 1–3): regression when artifacts are present
# --------------------------------------------------------------------------


def test_real_historical_runs_are_legacy_partial():
    real = config.COMPARISON_DIR / "1_fa_en_minilm"
    if not (real / "labeled.csv").exists():
        pytest.skip("real comparison artifacts not present in this checkout")
    report = verify_alignment()
    classifications = {
        r["id"]: (r.get("alignment") or {}).get("classification")
        for r in report["runs"]
        if r.get("status") == "ok"
    }
    # the migrated real runs must be legacy-partial — never modern
    assert set(classifications.values()) == {
        alignment.CLASSIFICATION_LEGACY_PARTIAL
    }
    assert report["all_records_aligned"] is True


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# PR #39 review fixes — strict modern contract, source-fingerprint
# invalidation, strict legacy-artifact validation
# --------------------------------------------------------------------------


def _meta_variant(**overrides: object) -> dict:
    """A modern metadata object with fields popped/replaced (_POP removes)."""
    meta = _modern_meta(SENTINEL_TEXTS)
    for key, value in overrides.items():
        if value is _POP:
            meta.pop(key, None)
        else:
            meta[key] = value
    return meta


_POP = object()


@pytest.mark.parametrize(
    "overrides, expected_issue",
    [
        ({"dataset_fingerprint": _POP}, "missing: dataset_fingerprint"),
        ({"text_window": _POP}, "missing: text_window"),
        ({"preprocess_version": _POP}, "missing: preprocess_version"),
        ({"threshold": _POP}, "missing: threshold"),
        ({"dataset_total": _POP}, "missing: dataset_total"),
        ({"dataset_fingerprint": "xyz"}, "invalid: dataset_fingerprint"),
        ({"text_window": "abc"}, "invalid: text_window"),
        ({"preprocess_version": ""}, "invalid: preprocess_version"),
        ({"threshold": "high"}, "invalid: threshold"),
        ({"dataset_total": -1}, "invalid: dataset_total"),
        ({"dataset_total": 1.5}, "invalid: dataset_total"),
    ],
)
def test_incomplete_or_invalid_metadata_is_never_modern(
    tmp_path, overrides, expected_issue
):
    run_dir = _make_run_dir(
        tmp_path, "1_fa_en_minilm", meta=_meta_variant(**overrides)
    )
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] != alignment.CLASSIFICATION_MODERN
    assert expected_issue in status["missing_fields"]


def test_only_dataset_fingerprint_present_is_not_modern(tmp_path):
    meta = {
        "dataset_fingerprint": alignment.ordered_input_fingerprint(
            SENTINEL_TEXTS
        )
    }
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=meta)
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] == alignment.CLASSIFICATION_LEGACY
    assert len(status["missing_fields"]) == 4


def test_complete_valid_metadata_is_modern(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, "1_fa_en_minilm", meta=_modern_meta(SENTINEL_TEXTS)
    )
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] == alignment.CLASSIFICATION_MODERN
    assert status["missing_fields"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"text_window": None},      # legitimate: model's own default
        {"threshold": 0.0},         # legitimate numeric zero
        {"dataset_total": 0},       # legitimate numeric zero
        {"text_window": 1},         # legitimate minimum window
    ],
)
def test_legitimate_values_stay_modern(tmp_path, overrides):
    meta = _meta_variant(**overrides)
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=meta)
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] == alignment.CLASSIFICATION_MODERN


def test_migration_regenerates_when_run_meta_changes(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())
    meta_file = run_dir / "run_meta.json"
    alignment.migrate_legacy_alignment([run_dir])
    first_sources = json.loads(
        (run_dir / "alignment_legacy.json").read_text(encoding="utf-8")
    )["sources"]

    meta_file.write_text(
        json.dumps(dict(_legacy_meta(), sampling="random")),
        encoding="utf-8",
    )
    second = alignment.migrate_legacy_alignment([run_dir])
    assert second["migrated"] == ["1_fa_en_minilm"]
    artifact = json.loads(
        (run_dir / "alignment_legacy.json").read_text(encoding="utf-8")
    )
    assert (
        artifact["sources"]["run_meta.json"]
        != first_sources["run_meta.json"]
    )
    assert artifact["reconstructed"]["sampling"] == "random"


def test_migration_regenerates_when_gap_changes(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())
    gap_file = run_dir / "gap_analysis.json"
    alignment.migrate_legacy_alignment([run_dir])

    gap = json.loads(gap_file.read_text(encoding="utf-8"))
    gap["threshold"] = 0.4
    gap_file.write_text(json.dumps(gap), encoding="utf-8")
    second = alignment.migrate_legacy_alignment([run_dir])
    assert second["migrated"] == ["1_fa_en_minilm"]
    artifact = json.loads(
        (run_dir / "alignment_legacy.json").read_text(encoding="utf-8")
    )
    assert artifact["reconstructed"]["threshold"] == 0.4


def test_migration_missing_sources_deterministic(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())
    (run_dir / "run_meta.json").unlink()
    (run_dir / "gap_analysis.json").unlink()

    alignment.migrate_legacy_alignment([run_dir])
    artifact = json.loads(
        (run_dir / "alignment_legacy.json").read_text(encoding="utf-8")
    )
    assert artifact["sources"]["run_meta.json"] == "absent"
    assert artifact["sources"]["gap_analysis.json"] == "absent"
    assert artifact["sources"]["labeled.csv"] != "absent"
    assert "threshold" in artifact["missing"]
    repeat = alignment.migrate_legacy_alignment([run_dir])
    assert repeat["already_migrated"] == ["1_fa_en_minilm"]


def _write_artifact(run_dir: Path, artifact: object) -> None:
    if isinstance(artifact, str):
        (run_dir / "alignment_legacy.json").write_text(
            artifact, encoding="utf-8"
        )
    else:
        (run_dir / "alignment_legacy.json").write_text(
            json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
        )


def _valid_artifact_for(run_dir: Path) -> dict:
    alignment.migrate_legacy_alignment([run_dir])
    return json.loads(
        (run_dir / "alignment_legacy.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: {"missing": None},
        lambda a: {"evidence": None},
        lambda a: {"schema": "legacy-alignment-v0"},
        lambda a: {"provenance": "executed"},
        lambda a: {"classification": "modern"},
        lambda a: {"evidence": {"record_count": "3"}},
        lambda a: {"sources": 42},
        lambda a: "[]",
        lambda a: "{ broken json",
    ],
)
def test_invalid_legacy_artifacts_are_treated_as_legacy(tmp_path, mutate):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())
    artifact = _valid_artifact_for(run_dir)
    if isinstance(mutate(artifact), str):
        mutated = mutate(artifact)          # raw malformed/non-object JSON
    else:
        mutated = dict(artifact)
        mutated.update(mutate(artifact))
    _write_artifact(run_dir, mutated)

    status = alignment.classify_alignment(run_dir)  # must not raise
    assert status["classification"] == alignment.CLASSIFICATION_LEGACY
    assert status["artifact"] is None


def test_valid_legacy_artifact_is_legacy_partial(tmp_path):
    run_dir = _make_run_dir(tmp_path, "1_fa_en_minilm", meta=_legacy_meta())
    _valid_artifact_for(run_dir)
    status = alignment.classify_alignment(run_dir)
    assert status["classification"] == alignment.CLASSIFICATION_LEGACY_PARTIAL
    assert status["artifact"] == "alignment_legacy.json"


def test_verify_alignment_stable_with_invalid_artifact(tmp_path, monkeypatch):
    base = tmp_path / "comparison"
    base.mkdir()
    monkeypatch.setattr(config, "COMPARISON_DIR", base)
    run_dir = _make_run_dir(base, "1_fa_en_minilm", meta=_legacy_meta())
    artifact = _valid_artifact_for(run_dir)
    broken = dict(artifact)
    broken["evidence"] = None
    _write_artifact(run_dir, broken)

    report = verify_alignment()  # must not raise, must not count as verified
    entry = report["runs"][0]
    assert entry["alignment"]["classification"] == (
        alignment.CLASSIFICATION_LEGACY
    )
    assert "invalid or incomplete" in entry["alignment"]["reason"]
    assert report["all_runs_modern"] is False
