from pathlib import Path

from data_loader import _merge_fields, extract_label_text, load_samples_fa


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
