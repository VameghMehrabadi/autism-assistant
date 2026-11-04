"""Unit tests for data_validation utilities (encoding, schema, dedup,
language ID, integrity, splits)."""
import json
import math
from pathlib import Path

import pytest

import data_validation as dv

# --------------------------------------------------------------------------
# Encoding detection
# --------------------------------------------------------------------------


def test_detect_encoding_utf8(tmp_path: Path):
    path = tmp_path / "utf8.jsonl"
    path.write_text("سلام این یک متن فارسی است\n", encoding="utf-8")
    assert dv.detect_encoding(path) == "utf-8"


def test_detect_encoding_utf8_sig(tmp_path: Path):
    path = tmp_path / "bom.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8-sig")
    assert dv.detect_encoding(path) == "utf-8-sig"


def test_detect_encoding_cp1256_fallback(tmp_path: Path):
    path = tmp_path / "legacy.csv"
    # String restricted to cp1256-covered letters (no Farsi Yeh/Kaf).
    payload = "instruction,input,output\nاضطراب,برنامه,پاسخ\n"
    path.write_bytes(payload.encode("cp1256"))
    assert dv.detect_encoding(path) == "cp1256"


# --------------------------------------------------------------------------
# Phase-1 regression: UTF-8 samples truncated mid-multibyte-character
# --------------------------------------------------------------------------


def _persian_record_bytes(i: int) -> bytes:
    record = {
        "idx": i,
        "instruction": "برنامه‌ی روزانه‌ام که عوض می‌شود مضطرب می‌شوم.",
        "input": "",
        "output": "ساختار و پیش‌بینی‌پذیری معمولاً کمک‌کننده است.",
    }
    return json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"


def test_detect_encoding_utf8_sample_ends_mid_multibyte_char(tmp_path: Path):
    """A large valid UTF-8 Persian JSONL whose 65,536-byte sample ends after
    the lead byte of a 2-byte Persian character must be detected as UTF-8,
    not CP1256 (the original bug: silent mojibake of the whole file)."""
    path = tmp_path / "large_fa.jsonl"
    ascii_prefix = b'{"idx": 0, "instruction": "' + b"x" * (65535 - 27)
    assert len(ascii_prefix) == 65535
    payload = ascii_prefix + "ا".encode("utf-8") + _persian_record_bytes(1) * 400
    assert len(payload) > 65536
    path.write_bytes(payload)
    assert dv.detect_encoding(path) == "utf-8"


def test_detect_encoding_utf8_boundary_cut_positions(tmp_path: Path):
    """Regression for both cut offsets of a 2-byte char and a 3-byte char
    (ZWNJ, common in real Persian text such as 'می‌شود')."""
    prefix = b'{"a": "' + b"x" * (65535 - 8)
    for trailing_char in ("ا", "‌"):  # 2-byte U+0627, 3-byte U+200C
        path = tmp_path / f"cut_{len(trailing_char.encode('utf-8'))}.jsonl"
        path.write_bytes(prefix + trailing_char.encode("utf-8") + b"rest\"\n")
        assert dv.detect_encoding(path) == "utf-8", trailing_char


def test_detect_encoding_small_corrupt_file_not_rescued(tmp_path: Path):
    """A file smaller than the sample whose final bytes are a genuinely
    incomplete UTF-8 sequence is corrupt — it must not be labelled UTF-8."""
    path = tmp_path / "small_corrupt.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": "\xd8')  # lone lead byte at true EOF
    assert dv.detect_encoding(path) != "utf-8"


def test_detect_encoding_large_clean_boundary_still_utf8(tmp_path: Path):
    """Control: a large UTF-8 file whose sample ends exactly on a complete
    character continues to be detected as UTF-8."""
    path = tmp_path / "clean.jsonl"
    payload = b"".join(_persian_record_bytes(i) for i in range(800))
    assert len(payload) > 65536
    path.write_bytes(payload)
    assert dv.detect_encoding(path) == "utf-8"


def test_detect_encoding_utf16_bom(tmp_path: Path):
    path = tmp_path / "utf16.csv"
    path.write_bytes("a,b\n1,2\n".encode("utf-16"))
    assert dv.detect_encoding(path) == "utf-16"


# --------------------------------------------------------------------------
# Normalization / schema validation
# --------------------------------------------------------------------------


def test_normalize_record_handles_missing_values():
    row = {
        "instruction": "  hello world  ",
        "input": None,
        "output": float("nan"),
        "idx": 7,
        "extra": 3.5,
    }
    normalized = dv.normalize_record(row)
    assert normalized["instruction"] == "hello world"
    assert normalized["input"] == ""
    assert normalized["output"] == ""
    assert normalized["idx"] == "7"
    assert normalized["extra"] == "3.5"


def test_normalize_record_strips_keys():
    normalized = dv.normalize_record({" instruction ": "x"})
    assert "instruction" in normalized


def test_normalize_record_rejects_non_dict():
    with pytest.raises(TypeError):
        dv.normalize_record(["not", "a", "dict"])


def test_validate_record_empty_flags_error():
    issues = dv.validate_record(dv.normalize_record({"input": "   "}), position=3)
    assert len(issues) == 1
    assert issues[0].code == "EMPTY_RECORD"
    assert issues[0].severity == "error"
    assert issues[0].position == 3


def test_validate_record_bad_idx_flags_warning():
    issues = dv.validate_record(
        dv.normalize_record({"instruction": "text", "idx": "abc"}), position=0
    )
    codes = {i.code: i.severity for i in issues}
    assert codes.get("BAD_IDX") == "warning"


def test_validate_record_clean_record_has_no_issues():
    issues = dv.validate_record(
        dv.normalize_record({"instruction": "متن فارسی", "idx": "1"}), position=0
    )
    assert issues == []


# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------


def test_detect_language_persian():
    assert dv.detect_language("احساس اضطراب شدیدی دارم وقتی برنامه تغییر می‌کند") == "fa"


def test_detect_language_english():
    assert dv.detect_language("My child gets upset when plans change suddenly") == "en"


def test_detect_language_mixed_and_unknown():
    assert dv.detect_language("Привет سلام world") == "mixed"
    assert dv.detect_language("12345 !!!") == "unknown"
    assert dv.detect_language("") == "unknown"


def test_language_distribution_counts():
    dist = dv.language_distribution(["سلام متن", "hello text", ""])
    assert dist == {"fa": 1, "en": 1, "unknown": 1}


# --------------------------------------------------------------------------
# Fingerprints / duplicates
# --------------------------------------------------------------------------


def test_record_fingerprint_stable_across_whitespace():
    a = dv.record_fingerprint({"instruction": "one  two", "input": "", "output": "x"})
    b = dv.record_fingerprint({"instruction": "one two", "input": None, "output": " x "})
    assert a == b


def test_record_fingerprint_differs_for_different_content():
    a = dv.record_fingerprint({"instruction": "one", "input": "", "output": ""})
    b = dv.record_fingerprint({"instruction": "two", "input": "", "output": ""})
    assert a != b


def test_find_duplicate_groups():
    fps = [
        dv.record_fingerprint({"instruction": "a"}),
        dv.record_fingerprint({"instruction": "b"}),
        dv.record_fingerprint({"instruction": "a"}),
        dv.record_fingerprint({"instruction": "c"}),
        dv.record_fingerprint({"instruction": "b"}),
    ]
    groups = dv.find_duplicate_groups(fps)
    assert sorted(groups) == [[0, 2], [1, 4]]


# --------------------------------------------------------------------------
# Integrity report
# --------------------------------------------------------------------------


def test_integrity_report_counts():
    rows = [
        dv.normalize_record({"instruction": "سلام", "input": "", "output": "پاسخ"}),
        dv.normalize_record({"instruction": "سلام", "input": "", "output": "پاسخ"}),
        dv.normalize_record({"text": "hello there", "input": None}),
        dv.normalize_record({"input": "   "}),
    ]
    report = dv.integrity_report(rows)
    assert report["total_rows"] == 4
    assert report["empty_records"] == 1
    assert report["duplicate_rows"] == 1
    assert report["duplicate_groups"] == 1
    assert report["n_unique"] == 3
    assert report["field_coverage"]["instruction"] == 2
    assert report["preprocess_version"] == dv.PREPROCESS_VERSION
    assert report["language_distribution"]["fa"] == 2  # merged text language


# --------------------------------------------------------------------------
# Splits and leakage guard
# --------------------------------------------------------------------------


def test_assign_splits_deterministic_and_disjoint():
    hashes = [f"hash-{i:03d}" for i in range(100)]
    first = dv.assign_splits(hashes, eval_fraction=0.2, seed=42)
    again = dv.assign_splits(hashes, eval_fraction=0.2, seed=42)
    assert first == again

    eval_set = {h for h, s in first.items() if s == "eval"}
    dev_set = {h for h, s in first.items() if s == "dev"}
    assert eval_set.isdisjoint(dev_set)
    assert len(eval_set) == 20
    assert set(first) == set(hashes)


def test_assign_splits_invalid_fraction_raises():
    with pytest.raises(ValueError):
        dv.assign_splits(["a", "b"], eval_fraction=1.5)
    with pytest.raises(ValueError):
        dv.assign_splits(["a", "b"], eval_fraction=0.0)


def test_manifest_roundtrip(tmp_path: Path):
    assignment = dv.assign_splits([f"h{i}" for i in range(10)], 0.2, seed=5)
    path = dv.save_split_manifest(
        assignment, path=tmp_path / "m.json", seed=5, eval_fraction=0.2
    )
    loaded = dv.load_split_manifest(path)
    assert loaded["n_eval"] == 2
    assert loaded["n_dev"] == 8
    assert sorted(loaded["eval_hashes"] + loaded["dev_hashes"]) == sorted(assignment)


def test_load_split_manifest_missing_raises_helpful_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as excinfo:
        dv.load_split_manifest(tmp_path / "nope.json")
    assert "create_splits" in str(excinfo.value)


def test_check_no_leakage_raises_on_overlap():
    dv.check_no_leakage(["a", "b"], ["c", "d"])  # disjoint → no error
    with pytest.raises(dv.LeakageError) as excinfo:
        dv.check_no_leakage(["a", "b"], ["b", "c"])
    assert "1 record" in str(excinfo.value)


def test_create_splits_rejects_items_without_hash():
    class NoHash:
        pass

    with pytest.raises(dv.DataValidationError):
        dv.create_splits([NoHash(), NoHash()], manifest_path=None)


def test_filter_by_split_rejects_unknown_split(tmp_path: Path):
    with pytest.raises(ValueError):
        dv.filter_by_split([], "train", manifest_path=tmp_path / "m.json")


def test_filter_by_split_excludes_missing_hash(tmp_path: Path):
    assignment = {"h1": "dev", "h2": "eval"}
    dv.save_split_manifest(assignment, path=tmp_path / "m.json")

    class Item:
        def __init__(self, meta):
            self.meta = meta

    items = [Item({"record_hash": "h1"}), Item({}), Item({"record_hash": "h2"})]
    kept, stats = dv.filter_by_split(items, "eval", manifest_path=tmp_path / "m.json")
    assert [i.meta.get("record_hash") for i in kept] == ["h2"]
    assert stats["kept"] == 1
    assert stats["excluded"] == 1  # h1 belongs to dev
    assert stats["missing_hash"] == 1
