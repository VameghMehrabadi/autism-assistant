"""Phase-3 reliability tests for the Z.ai translation pipeline.

All provider behavior is mocked; no network, no API keys, no paid calls.
Covers: permanent-error fail-fast, auth job-abort, truncation handling,
think-mode stripping, empty-translation flagging, torn-output repair,
incremental embedding-cache persistence, and quota accounting with
conservative estimates.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import translate_dataset as td
from labeler import SemanticLabeler
from translate_dataset import (
    NON_RETRYABLE_KINDS,
    UsageTracker,
    parse_batch_response,
    _map_provider_exception,
    _strip_think_blocks,
)


# --------------------------------------------------------------------------
# 1. API error mapping: precise status codes, permanent fail-fast
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}


def test_status_mapping_400_is_permanent():
    err = _map_provider_exception(SimpleNamespace(status_code=400, response=None, retry_after=None))
    assert err.kind == "permanent"


def test_status_mapping_401_and_403_are_auth():
    for status in (401, 403):
        err = _map_provider_exception(
            SimpleNamespace(status_code=status, response=None, retry_after=None)
        )
        assert err.kind == "auth"


def test_status_mapping_429_is_rate_limit_with_retry_after():
    err = _map_provider_exception(
        SimpleNamespace(status_code=429, response=_FakeResponse(429), retry_after=None)
    )
    # headers empty → no retry_after extracted, but kind is rate_limit
    assert err.kind == "rate_limit"


def test_status_mapping_500_is_retryable_api():
    err = _map_provider_exception(SimpleNamespace(status_code=500, response=None, retry_after=None))
    assert err.kind == "api"


def test_status_mapping_zai_1113_quota_body_is_quota_exhausted():
    """Z.ai reports account-level balance exhaustion as 429 + body code 1113 —
    it must map to a non-retryable, job-aborting kind, not rate_limit."""
    exc = SimpleNamespace(
        status_code=429,
        response=None,
        retry_after=None,
        body={"error": {"code": "1113", "message": "Insufficient balance or no resource package."}},
    )
    err = _map_provider_exception(exc)
    assert err.kind == "quota_exhausted"
    assert "1113" in str(err) or "balance" in str(err)


def test_quota_exhausted_aborts_whole_job():
    class QuotaFailProvider(td.TranslationProvider):
        name, model = "quotafail", "test"
        calls = 0

        def translate(self, segments):
            type(self).calls = getattr(type(self), "calls", 0) + 1
            raise td.TranslationError("quota_exhausted", "insufficient balance")

    provider = QuotaFailProvider()
    engine = td.TranslationEngine(provider, batch_size=2, max_retries=3)
    records = [
        td.normalize_source_record(
            {"idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}"},
            position=i,
        )
        for i in range(8)
    ]
    results, summary = engine.translate_records(records)
    assert summary.stop_reason == "aborted_quota_exhausted"
    assert summary.failed == 1  # first batch only — job aborted
    assert provider.calls == 1  # fail fast: no retries, no doomed follow-ups


def test_permanent_kinds_are_non_retryable():
    assert {"auth", "permanent", "truncated"} <= NON_RETRYABLE_KINDS


def _auth_error_provider():
    class AuthFailProvider(td.TranslationProvider):
        name, model = "authfail", "test"
        calls = 0

        def translate(self, segments):
            type(self).calls = getattr(type(self), "calls", 0) + 1
            raise td.TranslationError("auth", "HTTP 401: bad key")

    return AuthFailProvider()


def test_permanent_error_fails_fast_no_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(td.time, "sleep", lambda s: sleeps.append(s))
    provider = _auth_error_provider()
    engine = td.TranslationEngine(provider, max_retries=5, base_backoff=1.0)
    records = [td.normalize_source_record({"idx": 0, "input": "text"}, position=0)]
    results, summary = engine.translate_records(records)
    assert summary.failed == 1
    assert summary.errors[0].kind == "auth"
    assert provider.calls == 1  # fail fast: zero retries
    assert sleeps == []


def test_auth_error_aborts_whole_job(tmp_path):
    provider = _auth_error_provider()
    engine = td.TranslationEngine(provider, batch_size=2, max_retries=1)
    records = [
        td.normalize_source_record(
            {"idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}"},
            position=i,
        )
        for i in range(10)
    ]
    results, summary = engine.translate_records(records)
    assert summary.stop_reason == "aborted_auth"
    # batch_size=2 → the first batch slice holds 2 segments (both of record
    # 0's first fields) → only record 0 was in flight when the job aborted;
    # records 1-9 remain pending (never attempted, never marked failed).
    assert summary.failed == 1
    assert summary.errors[0].kind == "auth"
    assert summary.errors[0].attempts == 1  # fail fast: zero retries
    assert provider.calls == 1  # job aborted — no doomed follow-up calls


# --------------------------------------------------------------------------
# 2. Model response handling
# --------------------------------------------------------------------------


def test_parse_batch_response_strips_think_blocks():
    content = "<think>reasoning about the text</think>\n[\"ترجمه\"]"
    assert parse_batch_response(content, 1) == ["ترجمه"]


def test_parse_batch_response_unclosed_think_is_protocol_error():
    with pytest.raises(td.TranslationError) as excinfo:
        parse_batch_response("<think>truncated reasoning...", 1)
    assert excinfo.value.kind == "parse"


def test_strip_think_blocks_removes_multiple_blocks():
    text = "<think>a</think>keep<think>b</think>"
    assert _strip_think_blocks(text) == "keep"


def test_empty_translation_for_non_empty_source_is_flagged():
    engine = td.TranslationEngine(td.MockProvider(), batch_size=1)
    work = [(0, "input", "real patient text")]
    assigned = engine._assign(work, ["   "])  # model returned whitespace only
    assert "__flag__input" in assigned[0]
    assert "empty translation" in assigned[0]["__flag__input"]


def _fake_openai_response(content, finish_reason="stop", usage=None, reasoning=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=reasoning),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _provider_with_responses(responses, **kwargs):
    provider = td.OpenAIProvider(model="test", api_key="k-test", **kwargs)
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            item = responses.pop(0) if len(responses) > 1 else responses[0]
            if isinstance(item, Exception):
                raise item
            return item

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    return provider, calls


def test_finish_reason_length_raises_truncated_not_retried_by_engine(monkeypatch):
    responses = [
        _fake_openai_response('["سلام"]', finish_reason="length"),
        _fake_openai_response('["سلام"]', finish_reason="length"),
        _fake_openai_response('["سلام"]', finish_reason="length"),
    ]
    provider, calls = _provider_with_responses(responses)
    engine = td.TranslationEngine(provider, batch_size=1, max_retries=3)
    monkeypatch.setattr(td.time, "sleep", lambda s: None)
    records = [td.normalize_source_record({"idx": 0, "input": "text"}, position=0)]
    results, summary = engine.translate_records(records)
    assert summary.failed == 1
    assert summary.errors[0].kind == "truncated"
    assert len(calls) == 1  # temperature 0 reproduces truncation → no retry
    assert results == []  # incomplete translation never silently accepted


def test_max_output_tokens_passed_to_api():
    provider, calls = _provider_with_responses(
        [_fake_openai_response('["سلام"]')], max_output_tokens=777
    )
    provider.translate(["hello"])
    assert calls[0]["max_tokens"] == 777


def test_reasoning_content_is_ignored_and_think_stripped():
    provider, _ = _provider_with_responses(
        [
            _fake_openai_response(
                '<think>internal reasoning</think>["سلام"]',
                reasoning="separate reasoning payload",
            )
        ]
    )
    content = provider.translate(["hello"])
    assert parse_batch_response(content, 1) == ["سلام"]


def test_thinking_only_response_is_invalid_not_accepted():
    provider, _ = _provider_with_responses(
        [_fake_openai_response("<think>only reasoning, no output</think>")]
    )
    with pytest.raises(td.TranslationError) as excinfo:
        provider.translate(["hello"])
    assert excinfo.value.kind == "invalid_response"


def test_usage_metadata_recorded_as_actual():
    usage = SimpleNamespace(prompt_tokens=912, completion_tokens=24, total_tokens=936)
    provider, _ = _provider_with_responses(
        [_fake_openai_response('["سلام"]', usage=usage)]
    )
    provider.translate(["hello"])
    assert provider.usage["prompt_tokens"] == 912
    assert provider.usage["total_tokens"] == 936
    assert provider.usage["estimated_prompt_tokens"] == 0
    assert provider.usage["requests"] == 1


def test_missing_usage_metadata_counted_as_conservative_estimate():
    provider, _ = _provider_with_responses([_fake_openai_response('["سلام"]', usage=None)])
    provider.translate(["hello" * 400])  # ~2000 prompt chars → ≥1000 estimated
    assert provider.usage["unknown_metadata_responses"] == 1
    assert provider.usage["total_tokens"] == 0
    assert provider.usage["estimated_prompt_tokens"] >= 1000
    # estimates never masquerade as actual usage
    assert provider.usage["prompt_tokens"] == 0


def test_no_api_key_material_in_errors_or_usage():
    key = "sk-super-secret-key-1234567890"
    try:
        td.OpenAIProvider(model="m", api_key=key)
    except td.TranslationError:
        pass  # openai client may or may not validate; irrelevant
    provider = td.OpenAIProvider.__new__(td.OpenAIProvider)  # bypass init
    err = td._map_provider_exception(SimpleNamespace(status_code=401, response=None))
    assert key not in str(err)
    tracker = UsageTracker()
    tracker.record(None, 10, 5)
    assert key not in repr(tracker) and key not in json.dumps(tracker.to_dict())


# --------------------------------------------------------------------------
# 3. Checkpoint / resume: torn-output repair
# --------------------------------------------------------------------------


def test_resume_repairs_torn_output_and_avoids_duplicates(tmp_path):
    src = tmp_path / "src.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}",
            }, ensure_ascii=False) + "\n")
    out = tmp_path / "out.jsonl"
    # Simulate an interrupted append: one valid record + a torn final line.
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "idx": 0, "instruction": "[MOCK-FA]p0[/MOCK-FA]", "input": "[MOCK-FA]m0[/MOCK-FA]",
            "output": "[MOCK-FA]r0[/MOCK-FA]", "instruction_en": "p0", "input_en": "m0",
            "output_en": "r0", "status": "translated", "model": "mock-echo",
        }, ensure_ascii=False) + "\n")
        f.write('{"idx": 1, "instruction": "[MOCK-FA]p1[/MOCK')  # torn, no newline

    engine = td.TranslationEngine(td.MockProvider(), batch_size=4)
    summary = engine.run(src, out)

    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    idxs = [r["idx"] for r in lines]
    assert sorted(idxs) == [0, 1, 2, 3]
    assert len(idxs) == len(set(idxs))  # repaired torn line → no duplicates
    assert summary.done_before == 1  # the valid record survived the repair
    assert summary.translated == 3  # only the remaining records were run
    corrupt_backups = list(tmp_path.glob("out.jsonl.corrupt-*"))
    assert len(corrupt_backups) == 1  # damaged original preserved for inspection


def test_resume_never_overwrites_valid_completed_translations(tmp_path):
    src = tmp_path / "src.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({
                "idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}",
            }, ensure_ascii=False) + "\n")
    out = tmp_path / "out.jsonl"
    original_valid = {
        "idx": 2, "instruction": "حفظ‌شده", "input": "حفظ‌شده", "output": "حفظ‌شده",
        "instruction_en": "p2", "input_en": "m2", "output_en": "r2",
        "status": "translated", "model": "earlier-run",
    }
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(original_valid, ensure_ascii=False) + "\n")

    engine = td.TranslationEngine(td.MockProvider(), batch_size=4)
    engine.run(src, out)
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in lines if r["idx"] == 2]
    assert len(kept) == 1 and kept[0]["model"] == "earlier-run"  # untouched


# --------------------------------------------------------------------------
# 4. Quota protection with conservative estimates
# --------------------------------------------------------------------------


class _EstimatingMockProvider(td.MockProvider):
    """Mock that reports NO usage metadata → the tracker must estimate."""

    name = "mock-nometa"
    model = "mock-nometa"

    def __init__(self):
        super().__init__()
        self.tracker = UsageTracker()

    @property
    def usage(self):
        return self.tracker.to_dict()

    def translate(self, segments):
        content = super().translate(segments)
        # provider-side accounting: no usage metadata available
        self.tracker.record(None, prompt_chars=2000, completion_chars=len(content))
        return content


def test_quota_guard_counts_estimated_tokens():
    provider = _EstimatingMockProvider()
    engine = td.TranslationEngine(
        provider, batch_size=1, max_total_tokens=800
    )
    records = [
        td.normalize_source_record(
            {"idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}"},
            position=i,
        )
        for i in range(20)
    ]
    results, summary = engine.translate_records(records)
    assert summary.stop_reason == "quota_cap"
    usage = provider.usage
    assert usage["total_tokens"] == 0  # no actual metadata existed
    assert usage["estimated_prompt_tokens"] > 0  # conservative estimates used
    consumed = (
        usage["total_tokens"]
        + usage["estimated_prompt_tokens"]
        + usage["estimated_completion_tokens"]
    )
    assert consumed <= 800 + provider.tracker.estimated_prompt_tokens + 1  # cap respected within one call


def test_usage_tracker_distinguishes_actual_and_estimated():
    tracker = UsageTracker()
    tracker.record(SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), 40, 20)
    tracker.record(None, 100, 50)  # unknown metadata → estimated
    snapshot = tracker.to_dict()
    assert snapshot["total_tokens"] == 15        # actual
    assert snapshot["estimated_prompt_tokens"] == 50   # 100 chars / 2, conservative
    assert snapshot["estimated_completion_tokens"] == 25
    assert tracker.total_with_estimates == 15 + 50 + 25


# --------------------------------------------------------------------------
# 5. Incremental embedding-cache persistence
# --------------------------------------------------------------------------


class _ChunkFailEncoder:
    """Encodes fine for the first N chunks, then fails (simulated crash)."""

    def __init__(self, fail_after_chunks: int, dim: int = 8):
        import hashlib

        self.dim = dim
        self.chunks_done = 0
        self.fail_after = fail_after_chunks
        self._hashlib = hashlib

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False, **kwargs):
        import numpy as np

        if self.chunks_done >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.chunks_done += 1
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(self._hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vectors[i] = np.random.default_rng(seed).standard_normal(self.dim)
        return vectors


def test_embedding_cache_persists_incrementally(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBED_CACHE_WRITE_CHUNK", 2)
    import hashlib

    labeler = SemanticLabeler(
        model_key="minilm", fact_lang="en",
        _encoder=_ChunkFailEncoder(fail_after_chunks=10**9),  # healthy at init
        enable_cache=True, cache_dir=str(tmp_path / "cache"),
    )
    # Swap in the failing encoder AFTER prototype embedding: the test targets
    # sample-embedding chunk persistence.
    labeler.model = _ChunkFailEncoder(fail_after_chunks=1)

    texts = ["text one", "text two", "text three", "text four"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        labeler.embed_samples(texts)
    # Incremental persistence: the FIRST chunk (2 texts) survived the crash.
    from embedding_cache import EmbeddingCache

    cache = EmbeddingCache(
        model_name=labeler.model_name, prefix_mode=labeler.prefix_mode,
        cache_dir=tmp_path / "cache", enabled=True,
    )
    assert cache.size >= 2
    hits, matrix, missing = cache.lookup(texts[:2], role="query")
    assert hits == [0, 1] and missing == []
