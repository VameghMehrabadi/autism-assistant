from pathlib import Path

from data_loader import _merge_fields, load_samples_fa


def test_merge_fields_combines_non_empty_values():
    row = {"instruction": "Speak clearly", "input": "", "output": "Response"}
    text = _merge_fields(row, ("instruction", "input", "output"))

    assert "instruction: Speak clearly" in text
    assert "output: Response" in text
    assert "input:" not in text


def test_merge_fields_returns_empty_for_no_text():
    row = {"instruction": "", "input": None, "output": "  "}
    assert _merge_fields(row, ("instruction", "input", "output")) == ""


def test_load_samples_fa_from_sample_file():
    path = (
        Path(__file__).resolve().parent / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    samples = load_samples_fa(path=path, limit=3)
    assert len(samples) == 3
    assert samples[0].text
    assert "اضطراب" in samples[0].text or "روتین" in samples[0].text
