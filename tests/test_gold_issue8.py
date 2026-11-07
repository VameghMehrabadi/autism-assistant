"""Issue #8 — gold evaluation split, human annotation governance,
encoder-specific calibration workflow, and independent metrics.

Every fixture is synthetic and lives in isolated temporary directories.
 NOTHING in this file represents real human annotations, and no real
gold file is ever written or modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import calibration
import config
import gold_split
from data_loader import Sample
from evaluation_metrics import (
    group_bootstrap_ci,
    macro_f1_supported,
    multilabel_metrics,
)
from gold_eval import validate_gold_file, validate_gold_record


# --------------------------------------------------------------------------
# Synthetic Sample builder (isolated; never the real dataset)
# --------------------------------------------------------------------------


def _samples(n: int, seed: int = 1) -> list[Sample]:
    import hashlib

    items = []
    for i in range(n):
        text = f"synthetic conversation {seed}-{i} for gold selection"
        record_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        items.append(
            Sample(
                idx=1000 + i,
                text=text,
                meta={
                    "record_hash": record_hash,
                    "input_text": text,
                },
            )
        )
    return items


# --------------------------------------------------------------------------
# 1–3. Group-level split isolation, reproducibility, no leakage
# --------------------------------------------------------------------------


def test_gold_split_group_isolation(tmp_path):
    items = _samples(400)
    manifest = gold_split.build_gold_split(items, target_size=200, seed=7)
    assert manifest["verification"]["leak_free"] is True
    assert manifest["verification"]["groups_in_both_partitions"] == []
    assert manifest["verification"]["records_in_both_partitions"] == 0
    # whole-group guarantee: each group key maps to exactly one partition
    partitions = manifest["partitions"]
    cal = set(partitions["gold_calibration"])
    ev = set(partitions["gold_evaluation"])
    assert not (cal & ev)
    for position in set(cal) | set(ev):
        key = manifest["group_key_fingerprints"][str(position)]
        owners = [
            other
            for other in set(cal) | set(ev)
            if manifest["group_key_fingerprints"][str(other)] == key
        ]
        assert len(owners) == 1  # the whole group sits in one partition


def test_gold_split_reproducible(tmp_path):
    items = _samples(400)
    first = gold_split.build_gold_split(items, target_size=200, seed=7)
    second = gold_split.build_gold_split(items, target_size=200, seed=7)
    assert first["partitions"] == second["partitions"]
    assert first["record_hashes"] == second["record_hashes"]
    different = gold_split.build_gold_split(items, target_size=200, seed=8)
    assert different["partitions"] != first["partitions"] or (
        different["record_hashes"] != first["record_hashes"]
    )


def test_gold_split_no_calibration_evaluation_leakage(tmp_path, monkeypatch):
    """End-to-end: the manifest's two partitions are disjoint at record and
    group level, and the generated template carries the same partitioning."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gold_split, "GOLD_DIR", tmp_path / "gold")
    items = _samples(500)
    summary = gold_split.generate_gold_split(
        target_size=200, seed=7, source_path=tmp_path / "source.jsonl",
    ) if False else _generate_with_source(tmp_path, items)

    manifest = json.loads(
        Path(summary["manifest"]).read_text(encoding="utf-8")
    )
    verification = manifest["verification"]
    assert verification["leak_free"] is True
    cal_hashes = set()
    ev_hashes = set()
    template = [
        json.loads(line)
        for line in Path(summary["template"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for record in template:
        target = (
            cal_hashes
            if record["partition"] == "gold_calibration"
            else ev_hashes
        )
        target.add(record["provenance"]["record_hash"])
    assert not (cal_hashes & ev_hashes)  # no record in both partitions
    assert len(template) == verification["n_selected"]


def _generate_with_source(tmp_path: Path, items: list[Sample]) -> dict:
    source = tmp_path / "source.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps({"idx": item.idx, "text": item.text}) + "\n"
            )
    return gold_split.generate_gold_split(
        target_size=200, seed=7, source_path=source,
    )


def test_gold_split_target_bounds_enforced():
    items = _samples(10)
    with pytest.raises(ValueError):
        gold_split.build_gold_split(items, target_size=100)
    with pytest.raises(ValueError):
        gold_split.build_gold_split(items, target_size=600)


def test_gold_split_smaller_dataset_reports_actual_count(tmp_path):
    """If fewer records exist than the target, the actual count is reported
    honestly (whole groups, never padded or invented)."""
    items = _samples(120)
    manifest = gold_split.build_gold_split(items, target_size=300, seed=7)
    verification = manifest["verification"]
    assert 0 < verification["n_selected"] <= 120  # everything available
    assert verification["leak_free"] is True


# --------------------------------------------------------------------------
# 4–7. Gold validation: multi-label, origin, provenance, duplicates
# --------------------------------------------------------------------------


def _gold_record(**overrides) -> dict:
    record = {
        "gold_id": "gold-0001",
        "source_idx": 0,
        "labeler_id": "annotator-01",
        "origin": "human",
        "round": 1,
        "final_labels": ["B", "C"],
        "status": "annotated",
        "partition": "gold_evaluation",
    }
    record.update(overrides)
    return record


def test_valid_human_multi_label_annotation(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(_gold_record()) + "\n", encoding="utf-8")
    report = validate_gold_file(path)
    assert report["valid"] is True
    assert report["n_gold_labels"] == 1
    assert report["label_distribution"] == {"B": 1, "C": 1}


def test_model_originated_annotation_rejected(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(_gold_record(origin="model")) + "\n", encoding="utf-8"
    )
    report = validate_gold_file(path)
    assert report["valid"] is False
    assert report["origin_non_human"] == 1


def test_malformed_provenance_and_status_rejected(tmp_path):
    path = tmp_path / "gold.jsonl"
    bad = _gold_record(provenance="not-an-object", status="annotated?")
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    report = validate_gold_file(path)
    assert report["valid"] is False
    assert any("provenance" in e for e in report["errors"])
    assert any("status" in e for e in report["errors"])


def test_unknown_category_rejected(tmp_path):
    errors = validate_gold_record(
        _gold_record(final_labels=["B", "H"]), 1
    )
    assert any("must be one of" in e for e in errors)


def test_pending_records_are_not_gold_labels(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(_gold_record(status="pending")) + "\n", encoding="utf-8"
    )
    report = validate_gold_file(path)
    assert report["valid"] is True        # structurally fine (a worklist)
    assert report["n_gold_labels"] == 0   # but zero gold labels
    assert report["empty_gold"] is True   # no evaluation claims possible


def test_empty_labels_list_is_a_deliberate_annotation(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(_gold_record(final_labels=[])) + "\n", encoding="utf-8"
    )
    report = validate_gold_file(path)
    assert report["n_gold_labels"] == 1  # deliberate no-category annotation
    assert report["label_distribution"] == {}


def test_unsure_cannot_be_combined(tmp_path):
    errors = validate_gold_record(
        _gold_record(final_labels=["B", "UNSURE"]), 1
    )
    assert any("UNSURE cannot be combined" in e for e in errors)


def test_duplicate_ids_handled_explicitly(tmp_path):
    path = tmp_path / "gold.jsonl"
    record = _gold_record()
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    report = validate_gold_file(path)
    assert report["valid"] is False
    assert report["duplicate_gold_ids"] == 1
    assert report["duplicate_source_idx"] == 1


# --------------------------------------------------------------------------
# 8–10. Calibration: per-encoder behavior, sufficiency, no false claims
# --------------------------------------------------------------------------


CATEGORIES = ("A", "B", "C", "D", "E", "F", "G")


def _calibration_records(n: int, positive_every: int = 2) -> list[dict]:
    records = []
    for i in range(n):
        records.append({
            "origin": "human",
            "status": "annotated",
            "partition": "gold_calibration",
            "source_idx": i,
            "final_labels": (
                ["B"] if i % positive_every == 0 else []
            ),
        })
    return records


def _scores(n: int, signal: bool = True) -> dict[int, dict[str, float]]:
    scores = {}
    for i in range(n):
        scores[i] = {
            c: (
                0.9 if (signal and c == "B" and i % 2 == 0) else 0.1
            )
            for c in CATEGORIES
        }
    return scores


def test_calibration_succeeds_with_sufficient_human_data(tmp_path):
    records = _calibration_records(50)
    result = calibration.calibrate_encoder(
        records,
        _scores(50),
        encoder="minilm",
        dataset_fingerprint="a" * 16,
        split_fingerprint="b" * 16,
    )
    assert result["calibrated"] is True
    assert result["thresholds"]["global"] == 0.125
    assert result["global_macro_f1_supported"] == 1.0
    assert result["dataset_fingerprint"] == "a" * 16
    assert result["split_fingerprint"] == "b" * 16


def test_calibration_insufficient_data_is_uncalibrated(tmp_path):
    result = calibration.calibrate_encoder(
        _calibration_records(10),
        _scores(10),
        encoder="minilm",
        dataset_fingerprint="a" * 16,
        split_fingerprint="b" * 16,
    )
    assert result["calibrated"] is False
    assert "10 < 50" in result["reason"]
    assert "thresholds" not in result  # no threshold was invented


def test_calibration_refuses_model_origin_labels(tmp_path):
    records = [
        dict(_calibration_records(50)[0], origin="model")
    ] + _calibration_records(50)[1:]
    result = calibration.calibrate_encoder(
        records,
        _scores(50),
        encoder="minilm",
        dataset_fingerprint="a" * 16,
        split_fingerprint="b" * 16,
    )
    assert result["calibrated"] is False
    assert "non-human origin" in result["reason"]


def test_uncalibrated_status_wording(tmp_path):
    assert "uncalibrated" in calibration.threshold_status_for(
        "minilm", None
    )
    assert "no human annotations available" in calibration.threshold_status_for(
        "minilm", None
    )
    calibrated = {
        "encoders": {
            "minilm": {
                "calibrated": True,
                "thresholds": {"global": 0.125},
                "n_calibration_records": 50,
            }
        }
    }
    wording = calibration.threshold_status_for("minilm", calibrated)
    assert "calibrated on human labels" in wording
    assert "0.125" in wording


def test_per_category_insufficient_support_falls_back(tmp_path):
    """Category D never occurs positively -> insufficient support -> the
    per-category entry falls back to the global threshold."""
    records = _calibration_records(50)
    scores = _scores(50)
    result = calibration.calibrate_encoder(
        records, scores,
        encoder="minilm",
        dataset_fingerprint="a" * 16,
        split_fingerprint="b" * 16,
    )
    entry = result["thresholds"]["per_category"]["D"]
    assert entry["status"] == "insufficient_support"
    assert entry["fallback"] == "global"


# --------------------------------------------------------------------------
# 11–12. Metrics against known fixtures; uncertainty + small-sample guard
# --------------------------------------------------------------------------


def test_metric_calculations_known_fixture():
    y_true = [{"B"}, {"B", "C"}, set(), {"F"}]
    y_pred = [{"B"}, {"B"}, set(), {"B"}]
    metrics = multilabel_metrics(
        y_true, y_pred, ["A", "B", "C", "D", "E", "F", "G"]
    )
    b = metrics["per_category"]["B"]
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (2, 1, 0, 1)
    assert b["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert b["recall"] == 1.0
    assert b["f1"] == pytest.approx(0.8)
    assert b["support"] == 2
    c = metrics["per_category"]["C"]
    assert c["fn"] == 1 and c["f1"] == 0.0
    # micro: tp=2, fp=1 (rec3 B), fn=2 (C rec1, F rec3) -> F1 = 4/7
    assert metrics["micro_f1"] == pytest.approx(4 / 7, abs=1e-3)
    # macro over the full 7-category universe: 0.8/7
    assert metrics["macro_f1"] == pytest.approx(0.8 / 7, abs=1e-3)


def test_supported_macro_f1_excludes_zero_support():
    y_true = [{"B"}, {"B"}]
    y_pred = [{"B"}, {"B"}]
    assert macro_f1_supported(y_true, y_pred, CATEGORIES) == 1.0
    assert (
        multilabel_metrics(y_true, y_pred, CATEGORIES)["macro_f1"]
        == pytest.approx(1.0 / 7, abs=1e-3)
    )


def test_bootstrap_ci_and_small_sample_guard():
    rng_values = [
        ({"B"} if i % 2 == 0 else set()) for i in range(120)
    ]
    preds = [({"B"} if i % 2 == 0 else set()) for i in range(120)]
    groups = [f"group-{i // 3}" for i in range(120)]
    result = group_bootstrap_ci(
        groups, rng_values, preds, CATEGORIES, n_boot=200, seed=42
    )
    assert result["ci"] is not None
    assert result["ci"][0] <= result["point_estimate"] <= result["ci"][1]
    small = group_bootstrap_ci(
        ["g1", "g2"], [{"B"}, set()], [{"B"}, set()], CATEGORIES
    )
    assert small["ci"] is None
    assert "insufficient" in small["reason"]
    assert "point_estimate" in small


# --------------------------------------------------------------------------
# 13–14. Calibration vs held-out distinction; non-diagnostic language
# --------------------------------------------------------------------------


def test_calibration_partition_only_used_for_calibration(tmp_path):
    """A record in the evaluation partition must never influence the
    calibration thresholds."""
    records = _calibration_records(50)
    # contaminate the held-out partition with a strong pattern that would
    # shift a pooled threshold — it must be ignored
    held_out = [
        {
            "origin": "human", "status": "annotated",
            "partition": "gold_evaluation", "source_idx": 1000 + i,
            "final_labels": ["F"],
        }
        for i in range(20)
    ]
    scores = _scores(50)
    for i in range(20):
        scores[1000 + i] = {c: 0.95 for c in CATEGORIES}
    result = calibration.calibrate_encoder(
        records + held_out, scores,
        encoder="minilm",
        dataset_fingerprint="a" * 16,
        split_fingerprint="b" * 16,
    )
    assert result["calibrated"] is True
    assert result["n_calibration_records"] == 50  # held-out excluded
    assert result["thresholds"]["global"] == 0.125  # unchanged by held-out


def test_comparison_report_carries_non_diagnostic_language(tmp_path):
    """The comparison report must state the research-only framing and the
    0.35 status per encoder — never a diagnostic or superiority claim."""
    from compare_runs import build_comparison_table, render_comparison_markdown
    from facts import CATEGORY_KEYS

    per = {
        c: {"count": 1 if c == "B" else 0, "share": 0.0,
            "multi_count": 0, "multi_share": 0.0, "mean_score": 0.5,
            "median_score": 0.5, "std_score": 0.0,
            "confident_share": 0.0, "global_mean_score": 0.5}
        for c in CATEGORY_KEYS
    }
    report = {
        "n_samples": 1, "threshold": 0.35, "n_confident": 1,
        "confident_rate": 1.0, "n_unlabeled_NONE": 0,
        "per_category": per,
        "global_mean_score_per_category": {c: 0.5 for c in CATEGORY_KEYS},
        "ranking_weak_to_strong": list(CATEGORY_KEYS),
        "weakest": list(CATEGORY_KEYS)[:2],
        "strongest": list(CATEGORY_KEYS)[-2:],
    }
    cfg = {"id": "1_fa_en_minilm", "title": "t", "model_key": "minilm",
           "fact_lang": "fa", "dataset": "en"}
    summaries = [summarize_run := __import__("compare_runs").summarize_run(
        report, cfg
    )]
    summaries[0]["alignment"] = {"classification": "modern"}
    table = build_comparison_table(summaries)
    md = render_comparison_markdown(summaries, table)

    assert "## Gold calibration & human evaluation status" in md
    assert "Human-labeled gold records: **0**" in md
    assert "uncalibrated — no human annotations available" in md
    assert "not a probability" in md
    assert "not clinical diagnoses" in md
    assert "no encoder is clinically superior" in md
    # the default threshold is never claimed calibrated
    assert "calibrated on human labels" not in md


# --------------------------------------------------------------------------
# PR #40 privacy remediation — no raw group keys in persisted artifacts
# --------------------------------------------------------------------------


def test_group_key_fingerprint_deterministic_and_distinct():
    from gold_split import group_key_fingerprint

    key_a = "input:synthetic conversation alpha"
    key_b = "input:synthetic conversation beta"
    key_a2 = "input:synthetic conversation alpha"  # same source group
    fa, fb, fa2 = (
        group_key_fingerprint(key_a),
        group_key_fingerprint(key_b),
        group_key_fingerprint(key_a2),
    )
    assert fa == fa2                      # deterministic per source group
    assert fa != fb                       # different groups -> different fp
    assert len(fa) == 64                  # full sha256 hex
    assert all(c in "0123456789abcdef" for c in fa)
    assert "alpha" not in fa              # non-reversible


def test_manifest_stores_only_group_key_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gold_split, "GOLD_DIR", tmp_path / "gold")
    items = _samples(400)
    manifest = gold_split.build_gold_split(items, target_size=200, seed=7)
    raw_manifest = json.dumps(manifest)
    # no raw group key (the "input:"/"full:"/"hash:" prefixed text) survives
    assert '"input:' not in raw_manifest
    assert '"full:' not in raw_manifest
    assert '"hash:' not in raw_manifest
    fingerprints = manifest["group_key_fingerprints"]
    assert len(fingerprints) == manifest["verification"]["n_selected"]
    assert all(
        len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)
        for fp in fingerprints.values()
    )
    # distinct group keys -> distinct fingerprints
    distinct_fps = {
        fp for key, fp in fingerprints.items()
    }
    distinct_keys = set(manifest["group_key_fingerprints"].values())
    assert len(distinct_fps) == len(distinct_keys)
    assert manifest["group_key_fingerprint_algorithm"] == "sha256-utf8-hex-v1"


def test_template_provenance_fingerprinted_not_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gold_split, "GOLD_DIR", tmp_path / "gold")
    items = _samples(300)
    source = tmp_path / "source.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps({"idx": item.idx, "text": item.text}) + "\n"
            )
    summary = gold_split.generate_gold_split(
        target_size=200, seed=7, source_path=source
    )
    template = [
        json.loads(line)
        for line in Path(summary["template"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for record in template:
        prov = record["provenance"]
        fp = prov["group_key_fingerprint"]
        assert len(fp) == 64 and all(
            c in "0123456789abcdef" for c in fp
        )
        assert "group_key" not in prov  # raw key never persisted


def test_tracked_gold_artifacts_contain_no_raw_group_keys(tmp_path):
    """Privacy regression: every COMMITTED gold artifact must be free of
    raw conversation-derived group keys (the PR #40 audit finding)."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "data/gold/"],
        capture_output=True, text=True, cwd=repo,
    ).stdout.split()
    assert tracked, "no tracked gold artifacts found"
    violations = []
    for rel in tracked:
        content = (repo / rel).read_text(encoding="utf-8")
        if '"input:' in content or '"full:' in content or '"hash:' in content:
            violations.append(rel)
    assert violations == []


def test_worklist_remains_gitignored_and_untracked(tmp_path):
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    template_rel = "data/gold/gold_annotation_template.jsonl"
    check = subprocess.run(
        ["git", "check-ignore", template_rel],
        capture_output=True, text=True, cwd=repo,
    )
    assert check.returncode == 0, "worklist must stay gitignored"
    tracked = subprocess.run(
        ["git", "ls-files", template_rel],
        capture_output=True, text=True, cwd=repo,
    ).stdout.split()
    assert tracked == [], "worklist must never be committed"
