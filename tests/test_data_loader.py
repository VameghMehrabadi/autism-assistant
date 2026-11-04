import json
from pathlib import Path

import pytest

import config
import data_validation as dv
from data_loader import (
    LoadReport,
    Sample,
    _build_samples,
    _merge_fields,
    extract_label_text,
    load_samples,
    load_samples_fa,
)


# --------------------------------------------------------------------------
# Existing behavior (preserved)
# --------------------------------------------------------------------------


def test_merge_fields_combines_non_empty_values():
    row = {"instruction": "Speak clearly", "input": "", "output": "Response"}
    text = _merge_fields(row, ("instruction", "input", "output"))

    assert "instruction: Speak clearly" in text
    assert "output: Response" in text
    assert "input:" not in text


def test_merge_fields_returns_empty_for_no_text():
    row = {"instruction": "", "input": None, "output": "  "}
    assert _merge_fields(row, ("instruction", "input", "output")) == ""


def test_extract_label_text_uses_patient_input():
    row = {
        "instruction": (
            "You are a helpful mental health counselling assistant, "
            "please answer the mental health questions."
        ),
        "input": "My child gets upset when the daily plan changes suddenly.",
        "output": "Support and therapy can improve quality of life.",
    }
    assert extract_label_text(row, mode="patient") == row["input"]
    all_text = extract_label_text(row, mode="all")
    assert "instruction:" in all_text
    assert "output:" in all_text


def test_extract_label_text_falls_back_to_non_generic_instruction():
    row = {
        "text": (
            "instruction: احساس اضطراب شدیدی دارم وقتی برنامه‌ی روزانه‌ام ناگهان تغییر می‌کند.\n"
            "input: \n"
            "output: ساختار و پیش‌بینی‌پذیری معمولاً کمک‌کننده است."
        )
    }
    text = extract_label_text(row, mode="patient")
    assert "اضطراب" in text
    assert "پیش‌بینی‌پذیری" not in text


def test_load_samples_fa_from_sample_file():
    path = (
        Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    samples = load_samples_fa(path=path, limit=3)
    assert len(samples) == 3
    assert samples[0].text
    assert "اضطراب" in samples[0].text or "روتین" in samples[0].text
    assert "output:" not in samples[0].text.lower()


# --------------------------------------------------------------------------
# Phase 2: robust loading
# --------------------------------------------------------------------------

FIELDS = config.DATASET_TEXT_FIELDS
PATIENT_MODE = config.LABEL_TEXT_MODE


def _make_rows(n: int, *, with_idx: bool = True) -> list[dict]:
    rows = []
    for i in range(n):
        row = {
            "instruction": f"مشکل شماره {i}: کودکم در تغییر برنامه روزانه مضطرب می‌شود.",
            "input": "",
            "output": f"پاسخ درمانی شماره {i}.",
        }
        if with_idx:
            row["idx"] = i
        rows.append(row)
    return rows


def _build(rows: list[dict], **kwargs) -> tuple[list[Sample], LoadReport]:
    kwargs.setdefault("fields", FIELDS)
    kwargs.setdefault("mode", PATIENT_MODE)
    kwargs.setdefault("source", "test")
    return _build_samples(rows, **kwargs)


def test_build_samples_attaches_hashes_language_and_separation():
    samples, report = _build(_make_rows(3))
    assert report.kept == 3
    for s in samples:
        assert s.meta["record_hash"]
        assert s.meta["lang"] == "fa"
        assert s.meta["source"] == "test"
        assert s.input_text == ""  # input empty; instruction is labeled
        assert s.response_text.startswith("پاسخ درمانی")
        assert s.text.startswith("مشکل شماره")  # patient target preserved


def test_build_samples_separates_patient_input_from_response():
    rows = [
        {
            "instruction": "generic counsellor prompt",
            "input": "My child stims when lights are bright.",
            "output": "Sensory overload can cause distress.",
        }
    ]
    (s,), _ = _build(rows)
    assert s.input_text == "My child stims when lights are bright."
    assert s.response_text == "Sensory overload can cause distress."
    assert s.text == s.input_text  # patient input is the labeling target


def test_empty_records_are_skipped_transparently():
    rows = _make_rows(3) + [{"instruction": "", "input": "", "output": ""}]
    samples, report = _build(rows)
    assert report.total_rows == 4
    assert report.skipped_invalid == 1
    assert report.kept == 3
    assert any(i.code == "EMPTY_RECORD" for i in report.issues)


def test_strict_validation_raises_on_invalid_record():
    rows = _make_rows(2) + [{"input": "   "}]
    with pytest.raises(dv.DataValidationError):
        _build(rows, strict=True)


def test_malformed_jsonl_lines_are_skipped_and_counted(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    good = [
        json.dumps({"idx": 0, "text": "instruction: فرزندم به نور-sensitive است.\ninput: \noutput: پاسخ."}, ensure_ascii=False),
        json.dumps({"idx": 1, "text": "instruction: او روتین عوض می‌کند اضطراب دارد.\ninput: \noutput: پاسخ."}, ensure_ascii=False),
    ]
    path.write_text(
        good[0] + "\n{not valid json\n" + good[1] + "\n[1, 2, 3]\n\n",
        encoding="utf-8",
    )
    samples, report = load_samples_fa(path=path, return_report=True)
    assert report.malformed_lines == 1
    assert report.non_object_records == 1
    assert report.kept == 2
    assert [s.idx for s in samples] == [0, 1]


def test_load_csv_persian_with_utf8_sig(tmp_path: Path):
    path = tmp_path / "fa.csv"
    content = (
        "idx,instruction,input,output\n"
        "0,فرزندم تماس چشمی ندارد,,حمایت می‌تواند کمک کند.\n"
        "1,او به صدای بلند حساس است,,پرهیز از صدای بلید توصیه می‌شود.\n"
    )
    path.write_text(content, encoding="utf-8-sig")
    samples, report = load_samples_fa(path=path, return_report=True)
    assert report.encoding == "utf-8-sig"
    assert report.kept == 2
    assert samples[0].text == "فرزندم تماس چشمی ندارد"
    assert samples[0].response_text == "حمایت می‌تواند کمک کند."


def test_csv_missing_values_are_handled(tmp_path: Path):
    path = tmp_path / "missing.csv"
    content = (
        "instruction,input,output\n"
        "برنامه‌ی روزانه تغییر کرد,,پاسخ اول\n"
        ",,\n"  # entirely empty row → skipped
        "ساعت خواب بهم ریخت,,پاسخ دوم\n"
    )
    path.write_text(content, encoding="utf-8")
    samples, report = load_samples_fa(path=path, return_report=True)
    assert report.kept == 2
    assert report.skipped_invalid == 1


def test_duplicates_removed_keep_first(tmp_path: Path):
    dup = {"idx": 9, "instruction": "یکسان", "input": "", "output": "پاسخ"}
    rows = _make_rows(2) + [dict(dup), dict(dup)]  # extra whitespace variant
    rows[-1] = {**dup, "instruction": "  یکسان  "}
    samples, report = _build(rows)
    assert report.duplicates_removed == 1
    assert report.kept == 3
    assert sum(1 for s in samples if s.text.endswith("یکسان")) == 1


def test_duplicates_kept_when_keep_all():
    dup = {"instruction": "تکراری", "input": "", "output": "پاسخ"}
    samples, report = _build([dict(dup), dict(dup)], dedup="keep_all")
    assert report.duplicates_removed == 0
    assert report.kept == 2


def test_language_filter_keeps_only_fa():
    rows = _make_rows(1) + [
        {"instruction": "This is plain English text about routines.", "input": "", "output": "ok"}
    ]
    samples, report = _build(rows, language_filter="fa")
    assert report.kept == 1
    assert report.language_filtered == 1
    assert report.language_distribution.get("en") == 1


def test_language_counting_without_filter():
    rows = _make_rows(1) + [
        {"instruction": "This is plain English text about routines.", "input": "", "output": "ok"}
    ]
    samples, report = _build(rows)  # config.LANGUAGE_FILTER is None
    assert report.kept == 2
    assert report.language_filtered == 0


def test_random_sampling_is_reproducible():
    rows = _make_rows(20)
    a, _ = _build(rows, limit=7, sampling="random", seed=42)
    b, _ = _build(rows, limit=7, sampling="random", seed=42)
    c, rep_c = _build(rows, limit=7, sampling="random", seed=43)
    assert [s.meta["record_hash"] for s in a] == [s.meta["record_hash"] for s in b]
    assert [s.idx for s in a] == sorted(s.idx for s in a)  # stable order
    assert len(c) == 7 and rep_c.sampling == "random" and rep_c.seed == 43


def test_first_sampling_preserves_original_order():
    rows = _make_rows(10)
    samples, report = _build(rows, limit=4)
    assert [s.idx for s in samples] == [0, 1, 2, 3]
    assert report.sampling == "first" and report.seed is None


def test_limit_zero_and_none_mean_full_dataset():
    rows = _make_rows(5)
    assert len(_build(rows, limit=0)[0]) == 5
    assert len(_build(rows, limit=None)[0]) == 5


def test_original_identifiers_preserved():
    path = (
        Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    samples = load_samples_fa(path=path)
    assert [s.idx for s in samples] == [0, 1, 2, 3, 4]


def test_invalid_dedup_and_language_values_raise():
    with pytest.raises(ValueError):
        _build(_make_rows(1), dedup="nope")
    with pytest.raises(ValueError):
        _build(_make_rows(1), language_filter="fr")
    with pytest.raises(ValueError):
        _build(_make_rows(1), sampling="shuffle")


# --------------------------------------------------------------------------
# Phase 2: dev/eval split + leakage guard (integration)
# --------------------------------------------------------------------------


def test_dev_eval_split_roundtrip_without_leakage(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "split_manifest.json"
    rows = _make_rows(40)
    samples, _ = _build(rows)

    dev, evaluation = dv.create_splits(
        samples, eval_fraction=0.25, seed=7, manifest_path=manifest_path
    )
    assert len(dev) + len(evaluation) == 40
    dev_hashes = {s.meta["record_hash"] for s in dev}
    eval_hashes = {s.meta["record_hash"] for s in evaluation}
    assert dev_hashes.isdisjoint(eval_hashes)
    assert len(evaluation) == 10  # 25% of 40

    # Manifest roundtrip
    manifest = dv.load_split_manifest(manifest_path)
    assert manifest["n_eval"] == 10 and manifest["seed"] == 7

    # Reload from disk and filter by manifest — same membership
    reloaded, _ = _build(rows)
    dev_loaded, dev_stats = dv.filter_by_split(reloaded, "dev", manifest_path)
    eval_loaded, eval_stats = dv.filter_by_split(reloaded, "eval", manifest_path)
    assert dev_stats["kept"] == len(dev)
    assert eval_stats["kept"] == len(evaluation)
    assert {s.meta["record_hash"] for s in dev_loaded} == dev_hashes
    assert {s.meta["record_hash"] for s in eval_loaded} == eval_hashes

    # Leakage guard
    dv.check_no_leakage(dev_hashes, eval_hashes)  # must not raise
    with pytest.raises(dv.LeakageError):
        dv.check_no_leakage(dev_hashes, [next(iter(dev_hashes))])


def test_load_samples_eval_split_requires_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "SPLIT_MANIFEST_PATH", tmp_path / "missing.json")
    path = (
        Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    with pytest.raises(FileNotFoundError):
        load_samples(limit=5, dataset="fa", path=path, split="eval")


def test_load_samples_dev_split_excludes_eval(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "split_manifest.json"
    path = (
        Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    all_samples = load_samples_fa(path=path)
    monkeypatch.setattr(config, "SPLIT_MANIFEST_PATH", manifest_path)

    dev, evaluation = dv.create_splits(all_samples, eval_fraction=0.4, seed=3)
    assert len(dev) == 3 and len(evaluation) == 2

    dev_via_loader = load_samples(dataset="fa", path=path, split="dev")
    eval_via_loader = load_samples(dataset="fa", path=path, split="eval")
    assert {s.meta["record_hash"] for s in dev_via_loader} == {
        s.meta["record_hash"] for s in dev
    }
    assert {s.meta["record_hash"] for s in eval_via_loader} == {
        s.meta["record_hash"] for s in evaluation
    }


def test_split_rejects_unknown_split_name():
    path = (
        Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    with pytest.raises(ValueError):
        load_samples(dataset="fa", path=path, split="train")


# --------------------------------------------------------------------------
# Phase-2 (integrity): translation-status gating in the loader
# --------------------------------------------------------------------------


def _translated_rows() -> list[dict]:
    """Rows shaped like the Phase-6 translation output (status + *_en)."""
    return [
        {
            "idx": 0,
            "status": "translated",
            "instruction": "مشاور: سلامت روان مهم است.",
            "input": "فرزندم وقتی برنامه عوض می‌شود مضطرب می‌شود.",
            "output": "ساختار و پیش‌بینی‌پذیری کمک می‌کند.",
            "instruction_en": "Counsellor prompt.",
            "input_en": "My child gets anxious when routines change.",
            "output_en": "Structure and predictability help.",
        },
        {
            "idx": 1,
            "status": "flagged",
            "flag_reason": "ambiguous idiom",
            "instruction": "مشاور: پرامپت.",
            "input": "ورودی نامطمئن.",
            "output": "پاسخ نامطمئن.",
            "instruction_en": "prompt", "input_en": "unclear input", "output_en": "unclear output",
        },
        {
            "idx": 2,
            "status": "failed",
            "instruction": "", "input": " failed input", "output": "",
            "instruction_en": "", "input_en": "failed input", "output_en": "",
        },
        {
            "idx": 3,
            "status": "pending",
            "instruction": "", "input": "", "output": "",
            "instruction_en": "", "input_en": "pending input", "output_en": "",
        },
        {
            "idx": 4,
            "status": "rejected",
            "instruction": "", "input": "rejected", "output": "",
            "instruction_en": "", "input_en": "rejected", "output_en": "",
        },
        {
            "idx": 5,
            "status": "translated",
            "instruction": "مشاور: پرامپت دوم.",
            "input": "",  # EN input also empty → instruction is the patient target
            "output": "پاسخ دوم.",
            "instruction_en": "Second prompt.", "input_en": "", "output_en": "Second reply.",
        },
        {
            "idx": 6,
            "status": "translated",
            "instruction": "مشاور: ناقص.",
            "input": "",  # EN input PRESENT but FA input empty → incomplete
            "output": "پاسخ ناقص.",
            "instruction_en": "Incomplete prompt.", "input_en": "real patient input", "output_en": "reply",
        },
        {
            "idx": 7,
            "status": "empty_source",
            "instruction": "", "input": "", "output": "",
            "instruction_en": "", "input_en": "", "output_en": "",
        },
    ]


def test_translation_status_only_translated_enter_labeling(tmp_path: Path):
    path = tmp_path / "translated.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in _translated_rows():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    samples, report = load_samples_fa(path=path, return_report=True)

    # Only idx 0 and 5 (both status=translated with complete/legitimate input)
    assert [s.idx for s in samples] == [0, 5]
    assert report.translation_skipped == {
        "flagged": 1,
        "failed": 1,
        "pending": 1,
        "rejected": 1,
        "incomplete_input": 1,
        "empty_source": 1,
    }
    codes = {i.code for i in report.issues}
    assert {"TRANSLATION_STATUS_FLAGGED", "TRANSLATION_STATUS_FAILED",
            "TRANSLATION_STATUS_PENDING", "TRANSLATION_STATUS_REJECTED",
            "TRANSLATION_INCOMPLETE_INPUT"} <= codes
    assert report.skipped_invalid == 0  # skipped rows are counted separately


def test_translation_status_rejected_never_falls_back_to_counselor_output(tmp_path: Path):
    """The exact reported bug: a flagged/failed record with a blank patient
    input must never re-enter labeling through the counselor-output fallback."""
    path = tmp_path / "flagged.jsonl"
    row = {
        "idx": 0,
        "status": "flagged",
        "instruction": "generic counsellor prompt",
        "input": "",  # blank patient input — the dangerous fallback path
        "output": "counsellor output that must not become the label",
        "instruction_en": "generic counsellor prompt",
        "input_en": "", "output_en": "counsellor output",
    }
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    samples, report = load_samples_fa(path=path, return_report=True)
    assert samples == []  # excluded outright
    assert report.translation_skipped.get("flagged") == 1


def test_translation_status_preserves_english_originals_in_meta(tmp_path: Path):
    path = tmp_path / "one.jsonl"
    row = _translated_rows()[0]
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    samples, _ = load_samples_fa(path=path, return_report=True)
    s = samples[0]
    assert s.meta["translation_status"] == "translated"
    assert s.meta["input_en"] == "My child gets anxious when routines change."
    assert s.input_text == "فرزندم وقتی برنامه عوض می‌شود مضطرب می‌شود."
    assert s.response_text == "ساختار و پیش‌بینی‌پذیری کمک می‌کند."


def test_legacy_rows_without_status_unaffected(tmp_path: Path):
    """Status-less records (plain English dataset rows) keep the original
    behavior — no gating applies, and the pre-existing patient-mode fallback
    chain is unchanged (generic instructions are returned as-is by the final
    fallback when no better target exists; the output fallback fires only
    when input AND instruction are both empty)."""
    path = tmp_path / "legacy.jsonl"
    rows = [
        {"idx": 0, "instruction": "specific patient question", "input": "", "output": "counselor reply"},
        {"idx": 1,
         "instruction": "You are a helpful mental health counselling assistant, please answer the mental health questions.",
         "input": "", "output": "counselor reply"},
        {"idx": 2, "instruction": "", "input": "", "output": "output-only fallback reply"},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    samples, report = load_samples_fa(path=path, return_report=True)
    assert report.translation_skipped == {}
    assert [s.idx for s in samples] == [0, 1, 2]
    assert samples[0].text == "specific patient question"
    assert samples[1].text == rows[1]["instruction"]  # legacy chain: non-output fallback
    assert samples[2].text == "output-only fallback reply"  # legacy chain: output fallback


# --------------------------------------------------------------------------
# Phase-2 (integrity): export completeness metadata
# --------------------------------------------------------------------------


def test_write_export_meta_complete_and_incomplete(tmp_path: Path):
    from export_for_translation import build_export_record, write_export

    samples = []
    for i in range(5):
        s = Sample(
            idx=i,
            text=f"patient text {i}",
            meta={"instruction": f"inst {i}", "input": f"in {i}", "output": f"out {i}"},
            input_text=f"in {i}",
            response_text=f"out {i}",
        )
        samples.append(s)

    # Complete export
    meta = write_export(samples, tmp_path / "full.jsonl", dataset_total=5)
    assert meta["complete"] is True
    assert meta["total_source"] == 5 and meta["exported"] == 5
    meta_path = tmp_path / "full.export_meta.json"
    assert meta_path.exists()
    persisted = json.loads(meta_path.read_text(encoding="utf-8"))
    assert persisted["complete"] is True
    exported = [json.loads(l) for l in (tmp_path / "full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert exported[0] == build_export_record(samples[0])

    # Incomplete export (limited): clearly marked
    meta2 = write_export(samples[:2], tmp_path / "partial.jsonl", dataset_total=5)
    assert meta2["complete"] is False
    persisted2 = json.loads((tmp_path / "partial.export_meta.json").read_text(encoding="utf-8"))
    assert persisted2["complete"] is False
    assert persisted2["exported"] == 2 and persisted2["total_source"] == 5


def test_load_report_dataset_total_before_limit(tmp_path: Path):
    path = tmp_path / "small.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({"idx": i, "instruction": f"متن {i}"}, ensure_ascii=False) + "\n")
    samples, report = load_samples_fa(path=path, limit=2, return_report=True)
    assert len(samples) == 2
    assert report.dataset_total == 6  # total BEFORE the limit


# --------------------------------------------------------------------------
# Phase-1 regression: large UTF-8 Persian JSONL loads without mojibake
# --------------------------------------------------------------------------


def test_load_samples_fa_large_utf8_boundary_file(tmp_path: Path):
    """End-to-end: a large valid UTF-8 Persian JSONL whose 65,536-byte sample
    ends mid-multibyte-character must load as UTF-8 with text intact — the
    original bug detected such files as CP1256 and silently mojibaked every
    Persian character."""
    path = tmp_path / "large_fa.jsonl"
    records = [
        {
            "idx": i,
            "text": (
                f"instruction: برنامه‌ی شماره {i} عوض می‌شود مضطرب.\n"
                "input: \n"
                f"output: پاسخ درمانی {i}."
            ),
        }
        for i in range(800)
    ]
    payload = b"".join(
        json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in records
    )
    assert len(payload) > 65536
    path.write_bytes(payload)

    samples, report = load_samples_fa(path=path, return_report=True)
    assert report.encoding == "utf-8"
    assert report.kept == 800
    # Text integrity: the exact source Persian string survives the roundtrip.
    expected = "برنامه‌ی شماره 5 عوض می‌شود مضطرب."
    assert samples[5].text == expected
    # No mojibake signatures anywhere in the loaded labels.
    joined = " ".join(s.text for s in samples)
    assert "Ø" not in joined and "\ufffd" not in joined
    assert all(s.meta["lang"] == "fa" for s in samples)
