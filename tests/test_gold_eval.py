"""Tests for the gold evaluation-set validator (synthetic fixtures only —
no real human labels exist; nothing here may be cited as evaluation data)."""
import json
from pathlib import Path

import pytest

import gold_eval


def _gold(idx=0, **overrides):
    rec = {
        "gold_id": f"gold-{idx:04d}",
        "source_idx": idx,
        "labeler_id": "AB",
        "origin": "human",
        "round": 1,
        "first_label": "B",
        "final_label": "B",
        "confidence": "sure",
        "gold_set_version": "v1",
    }
    rec.update(overrides)
    return rec


def _write(path: Path, records) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def test_valid_gold_file_passes(tmp_path: Path):
    path = _write(tmp_path / "gold_labels_v1.jsonl", [_gold(0), _gold(1)])
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is True
    assert report["n_records"] == 2
    assert report["label_distribution"] == {"B": 2}


def test_missing_file_reported():
    report = gold_eval.validate_gold_file(Path("Z:/nope.jsonl"))
    assert report["exists"] is False and report["valid"] is False


def test_empty_file_is_valid_but_empty(tmp_path: Path):
    path = tmp_path / "gold_labels_v1.jsonl"
    path.write_text("", encoding="utf-8")
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is True and report["empty"] is True and report["n_records"] == 0


def test_model_generated_label_rejected(tmp_path: Path):
    path = _write(tmp_path / "g.jsonl", [_gold(0, origin="model", labeler_id="minilm")])
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is False
    assert report["origin_non_human"] == 1
    assert any("forbidden" in e for e in report["errors"])


def test_invalid_category_and_round_rejected(tmp_path: Path):
    path = _write(
        tmp_path / "g.jsonl",
        [_gold(0, final_label="H", round=3)],
    )
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is False
    assert any("final_label" in e for e in report["errors"])
    assert any("round" in e for e in report["errors"])


def test_duplicate_ids_rejected(tmp_path: Path):
    path = _write(tmp_path / "g.jsonl", [_gold(0), _gold(0)])
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is False
    assert report["duplicate_gold_ids"] == 1


def test_duplicate_source_idx_rejected(tmp_path: Path):
    path = _write(
        tmp_path / "g.jsonl",
        [_gold(0, gold_id="gold-0001"), _gold(0, gold_id="gold-0002")],
    )
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is False
    assert report["duplicate_source_idx"] == 1


def test_missing_required_field_rejected(tmp_path: Path):
    rec = _gold(0)
    del rec["labeler_id"]
    path = _write(tmp_path / "g.jsonl", [rec])
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is False
    assert any("labeler_id" in e for e in report["errors"])


def test_unsure_labels_retained_and_counted(tmp_path: Path):
    path = _write(
        tmp_path / "g.jsonl",
        [_gold(0, final_label="UNSURE", first_label="A"), _gold(1)],
    )
    report = gold_eval.validate_gold_file(path)
    assert report["valid"] is True
    assert report["label_distribution"] == {"UNSURE": 1, "B": 1}


def test_cli_exit_codes(tmp_path: Path, monkeypatch):
    good = _write(tmp_path / "g.jsonl", [_gold(0)])
    monkeypatch.setattr("sys.argv", ["gold_eval.py", "--file", str(good)])
    assert gold_eval.main() == 0
    bad = _write(tmp_path / "b.jsonl", [_gold(0, origin="model")])
    monkeypatch.setattr("sys.argv", ["gold_eval.py", "--file", str(bad)])
    assert gold_eval.main() == 2
