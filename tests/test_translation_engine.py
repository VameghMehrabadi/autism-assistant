"""Tests for the Phase-6 translation engine (mock providers only).

No real API is ever called; no paid credits are consumed. Providers are
in-memory test doubles. Source records use synthetic text only.
"""
import json
from pathlib import Path

import pytest

import translate_dataset as td
from translate_dataset import (
    STATUS_EMPTY_SOURCE,
    STATUS_FLAGGED,
    STATUS_TRANSLATED,
    EngineError,
    MockProvider,
    OpenAIProvider,
    SourceRecord,
    TranslationEngine,
    TranslationError,
    build_output_record,
    normalize_source_record,
    parse_batch_response,
)


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class EchoProvider(MockProvider):
    """Same contract as MockProvider with call accounting."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def translate(self, segments):
        self.calls.append(list(segments))
        return super().translate(segments)


class FlakyProvider(EchoProvider):
    """Fails N times with the given kind, then behaves like EchoProvider."""

    def __init__(self, failures: int, kind: str = "api", retry_after: float | None = None):
        super().__init__()
        self.failures = failures
        self.kind = kind
        self.retry_after = retry_after

    def translate(self, segments):
        if self.failures > 0:
            self.failures -= 1
            raise TranslationError(self.kind, f"{self.kind} failure", retry_after=self.retry_after)
        return super().translate(segments)


class GarbageBatchProvider(EchoProvider):
    """Returns garbage for multi-segment batches, echoes single segments."""

    def translate(self, segments):
        self.calls.append(list(segments))
        if len(segments) > 1:
            return "this is not json at all"
        return super().translate(segments)


class WrongCountProvider(EchoProvider):
    """Always returns one element too many (invalid response shape)."""

    def translate(self, segments):
        self.calls.append(list(segments))
        return json.dumps(["x"] * (len(segments) + 1), ensure_ascii=False)


class UntranslatableProvider(EchoProvider):
    """Flags segments containing the word 'unintelligible'."""

    def translate(self, segments):
        self.calls.append(list(segments))
        out = [
            f"{td.UNTRANSLATABLE_MARKER} source is unintelligible"
            if "unintelligible" in text else f"[MOCK-FA]{text}[/MOCK-FA]"
            for text in segments
        ]
        return json.dumps(out, ensure_ascii=False)


class FixedPersianProvider(EchoProvider):
    """Returns a fixed Persian translation (with ZWNJ) for every segment."""

    def __init__(self, persian: str):
        super().__init__()
        self.persian = persian

    def translate(self, segments):
        self.calls.append(list(segments))
        return json.dumps([self.persian] * len(segments), ensure_ascii=False)


def _source_records(n: int = 3) -> list[SourceRecord]:
    records = []
    for i in range(n):
        records.append(
            normalize_source_record(
                {
                    "idx": i,
                    "instruction": f"counselor prompt {i}",
                    "input": f"patient message {i}",
                    "output": f"counselor response {i}",
                },
                position=i,
            )
        )
    return records


# --------------------------------------------------------------------------
# Source normalization
# --------------------------------------------------------------------------


def test_normalize_source_record_fields_and_idx():
    record = normalize_source_record(
        {"idx": "7", "instruction": "a", "input": "", "output": "c"}, position=3
    )
    assert record.idx == 7
    assert record.fields == {"instruction": "a", "output": "c"}
    assert record.originals["input"] == ""


def test_normalize_source_record_falls_back_to_positional_idx():
    record = normalize_source_record({"instruction": "a"}, position=5)
    assert record.idx == 5


def test_normalize_source_record_parses_old_tagged_text_format():
    record = normalize_source_record(
        {
            "idx": 1,
            "text": "instruction: old prompt\ninput: \noutput: old reply",
        },
        position=0,
    )
    assert record.fields == {"instruction": "old prompt", "output": "old reply"}


def test_normalize_source_record_maps_plain_text_to_input():
    record = normalize_source_record({"idx": 2, "text": "just a patient message"}, position=0)
    assert record.fields == {"input": "just a patient message"}


def test_normalize_source_record_rejects_non_object():
    assert normalize_source_record(["not", "a", "dict"], position=0) is None


# --------------------------------------------------------------------------
# Response protocol
# --------------------------------------------------------------------------


def test_parse_batch_response_valid_and_tolerates_fences():
    assert parse_batch_response('["a", "b"]', 2) == ["a", "b"]
    assert parse_batch_response("```json\n[\"a\"]\n```", 1) == ["a"]


def test_parse_batch_response_errors():
    with pytest.raises(TranslationError, match="not valid JSON"):
        parse_batch_response("nope", 1)
    with pytest.raises(TranslationError, match="not a JSON array"):
        parse_batch_response('{"a": 1}', 1)
    with pytest.raises(TranslationError, match="expected 2"):
        parse_batch_response('["a"]', 2)
    with pytest.raises(TranslationError, match="non-string"):
        parse_batch_response('["a", 3]', 2)


# --------------------------------------------------------------------------
# Engine behavior
# --------------------------------------------------------------------------


def test_engine_translates_records_end_to_end():
    provider = EchoProvider()
    engine = TranslationEngine(provider, batch_size=2)
    results, summary = engine.translate_records(_source_records(3))

    assert summary.translated == 3
    assert summary.failed == 0
    assert len(results) == 3
    first = results[0]
    assert first["idx"] == 0
    assert first["status"] == STATUS_TRANSLATED
    assert first["instruction"] == "[MOCK-FA]counselor prompt 0[/MOCK-FA]"
    assert first["model"] == "mock-echo"
    # Batching: batch_size=2 over 3 records × up to 3 fields → capped batches
    assert all(len(call) <= 2 for call in provider.calls)


def test_engine_preserves_original_english():
    engine = TranslationEngine(EchoProvider())
    results, _ = engine.translate_records(_source_records(2))
    for result, source in zip(results, _source_records(2)):
        for f in td.SOURCE_FIELDS:
            assert result[f + "_en"] == source.originals[f]


def test_engine_preserves_unicode_and_zwnj():
    persian = "برنامه‌ی روزانه که عوض می‌شود اضطراب می‌آورد — café ☕"
    engine = TranslationEngine(FixedPersianProvider(persian))
    results, _ = engine.translate_records(
        [normalize_source_record({"idx": 0, "input": "unicode test"}, position=0)]
    )
    assert results[0]["input"] == persian  # ZWNJ, dash, emoji intact
    assert results[0]["input_en"] == "unicode test"


def test_engine_empty_source_recorded_without_provider_call():
    provider = EchoProvider()
    engine = TranslationEngine(provider)
    results, summary = engine.translate_records(
        [normalize_source_record({"idx": 9, "instruction": "", "input": "", "output": ""}, position=0)]
    )
    assert summary.empty_source == 1
    assert results[0]["status"] == STATUS_EMPTY_SOURCE
    assert provider.calls == []  # never sent to the provider


def test_engine_skips_duplicate_idx():
    engine = TranslationEngine(EchoProvider())
    records = _source_records(2)
    duplicate = normalize_source_record(
        {"idx": 0, "instruction": "dup", "input": "dup", "output": "dup"}, position=9
    )
    _, summary = engine.translate_records(records + [duplicate])
    assert summary.duplicates_skipped == 1
    assert summary.total_source == 3
    assert summary.translated == 2


def test_engine_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(td.time, "sleep", lambda s: sleeps.append(s))
    provider = FlakyProvider(failures=2, kind="api")
    engine = TranslationEngine(provider, max_retries=3, base_backoff=2.0)
    results, summary = engine.translate_records(_source_records(1))
    assert summary.translated == 1
    # Failed attempts raise before reaching EchoProvider's call log, so only
    # the 3 successful per-field calls (batch_size=1) appear here.
    assert len(provider.calls) == 3
    assert len(sleeps) == 2  # backed off between the failed attempts


def test_engine_rate_limit_honors_retry_after(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(td.time, "sleep", lambda s: sleeps.append(s))
    provider = FlakyProvider(failures=1, kind="rate_limit", retry_after=7.0)
    engine = TranslationEngine(provider, max_retries=2)
    _, summary = engine.translate_records(_source_records(1))
    assert summary.translated == 1
    assert 7.0 <= sleeps[0] <= 120.0  # Retry-After respected, clamped


def test_engine_retries_exhausted_reported_not_silently_skipped():
    provider = FlakyProvider(failures=99, kind="api")
    engine = TranslationEngine(provider, max_retries=1, base_backoff=0.0)
    _, summary = engine.translate_records(_source_records(2))
    assert summary.translated == 0
    assert summary.failed == 2
    assert {e.kind for e in summary.errors} == {"api"}
    # Error messages carry no record content
    assert all("patient message" not in e.message for e in summary.errors)


def test_engine_garbage_batch_falls_back_to_single_calls():
    provider = GarbageBatchProvider()
    engine = TranslationEngine(provider, batch_size=4, max_retries=1)
    results, summary = engine.translate_records(_source_records(2))
    assert summary.translated == 2  # recovered via per-segment fallback
    garbage_calls = [c for c in provider.calls if len(c) > 1]
    single_calls = [c for c in provider.calls if len(c) == 1]
    assert garbage_calls, "batch protocol was attempted first"
    # 6 segments, each logged twice per single call (GarbageBatchProvider and
    # EchoProvider both append before delegating to the mock).
    assert len(single_calls) == 12


def test_engine_single_segment_invalid_response_fails_after_retries():
    provider = WrongCountProvider()
    engine = TranslationEngine(provider, max_retries=1, base_backoff=0.0)
    _, summary = engine.translate_records(_source_records(1))
    assert summary.failed == 1
    assert summary.errors[0].kind == "parse"


def test_engine_flags_untranslatable_segments():
    records = [
        normalize_source_record({"idx": 0, "input": "clear text"}, position=0),
        normalize_source_record({"idx": 1, "input": "unintelligible fragment"}, position=1),
    ]
    engine = TranslationEngine(UntranslatableProvider(), batch_size=4)
    results, summary = engine.translate_records(records)
    assert summary.translated == 1
    assert summary.flagged == 1
    flagged = results[1]
    assert flagged["status"] == STATUS_FLAGGED
    assert "unintelligible" in flagged["flag_reason"]
    assert flagged["input"] == ""  # no guessed translation


def test_engine_resume_after_interruption(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "idx": i,
                "instruction": f"p{i}",
                "input": f"m{i}",
                "output": f"r{i}",
            }, ensure_ascii=False) + "\n")
    out = tmp_path / "out.jsonl"

    # Run 1: provider dies after the first successful batch (2 records).
    class FailAfterProvider(FlakyProvider):
        def __init__(self, succeed_segments: int):
            super().__init__(failures=0)
            self.succeed_segments = succeed_segments

        def translate(self, segments):
            if self.succeed_segments <= 0:
                raise TranslationError("api", "interrupted")
            self.succeed_segments -= len(segments)
            return super().translate(segments)

    broken = FailAfterProvider(succeed_segments=3)  # 1st batch ok (3 segs), rest fail
    engine1 = TranslationEngine(broken, batch_size=3, max_retries=0)
    summary1 = engine1.run(src, out)
    assert summary1.translated == 1  # idx 0 fully translated (3 fields)
    assert summary1.failed == 3

    # Run 2: healthy provider resumes — only remaining records are sent.
    healthy = EchoProvider()
    engine2 = TranslationEngine(healthy, batch_size=8)
    summary2 = engine2.run(src, out)
    assert summary2.done_before == 1
    assert summary2.translated == 3
    assert summary2.failed == 0

    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    idxs = [rec["idx"] for rec in lines]
    assert sorted(idxs) == [0, 1, 2, 3]
    assert len(idxs) == len(set(idxs))  # no duplicates
    statuses = {rec["idx"]: rec["status"] for rec in lines}
    assert all(statuses[i] == STATUS_TRANSLATED for i in idxs)
    # Only the remaining records were sent to the provider in run 2
    translated_texts = {t for call in healthy.calls for t in call}
    assert "m0" not in translated_texts and "m1" in translated_texts


def test_engine_writes_structured_error_log(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps({"idx": 0, "input": "hello"}) + "\n", encoding="utf-8")
    err_log = tmp_path / "errors.jsonl"
    engine = TranslationEngine(FlakyProvider(failures=99), max_retries=0, base_backoff=0.0)
    engine.run(src, tmp_path / "out.jsonl", error_log_path=err_log)
    entries = [json.loads(l) for l in err_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(entries) == 1
    assert entries[0]["idx"] == 0
    assert entries[0]["kind"] == "api"
    assert "attempts" in entries[0]


def test_build_output_record_keeps_extra_source_metadata():
    record = SourceRecord(
        idx=3,
        fields={"input": "x"},
        originals={"instruction": "", "input": "x", "output": ""},
        extra={"custom_meta": "keep-me"},
    )
    out = build_output_record(record, {"input": "persian"}, STATUS_TRANSLATED, "gpt-4o")
    assert out["custom_meta_en"] == "keep-me"
    assert out["input"] == "persian" and out["input_en"] == "x"
    assert out["status"] == "translated" and out["model"] == "gpt-4o"
    assert "flag_reason" not in out


# --------------------------------------------------------------------------
# Provider configuration
# --------------------------------------------------------------------------


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(TranslationError) as excinfo:
        OpenAIProvider()
    assert excinfo.value.kind == "config"


def test_provider_exception_mapping():
    class FakeRateLimit(Exception):
        retry_after = 12

    err = td._map_provider_exception(FakeRateLimit("x"))
    assert err.kind == "rate_limit" and err.retry_after == 12.0

    class FakeTimeout(Exception):
        pass

    assert td._map_provider_exception(FakeTimeout("x")).kind == "timeout"
    assert td._map_provider_exception(ValueError("boom")).kind == "api"
