"""Tests for the Z.ai (GLM) provider integration — offline only.

No network is contacted: the OpenAI SDK client is constructed but never
called; token accounting and quota-guard logic use synthetic objects.
"""
import json
from pathlib import Path

import pytest

import config
import translate_dataset as td
from translate_dataset import (
    EngineSummary,
    MockProvider,
    OpenAIProvider,
    SourceRecord,
    TranslationEngine,
    TranslationError,
    _accumulate_usage,
    normalize_source_record,
    resolve_zai_config,
)


# --------------------------------------------------------------------------
# Z.ai configuration resolution (env-only credentials)
# --------------------------------------------------------------------------


def test_resolve_zai_config_missing_key_raises_with_var_name():
    with pytest.raises(TranslationError) as excinfo:
        resolve_zai_config(env={})
    assert "ZAI_API_KEY" in str(excinfo.value)
    assert excinfo.value.kind == "config"


def test_resolve_zai_config_reads_key_and_defaults():
    key, base_url, model = resolve_zai_config(env={"ZAI_API_KEY": "k-test"})
    assert key == "k-test"
    assert base_url == config.ZAI_API_BASE_URL
    assert model == config.ZAI_MODEL


def test_resolve_zai_config_honors_env_overrides():
    key, base_url, _model = resolve_zai_config(
        env={"ZAI_API_KEY": "k", "ZAI_BASE_URL": "https://example.invalid/v4"}
    )
    assert base_url == "https://example.invalid/v4"


def test_resolve_zai_config_falls_back_to_zhipu_key():
    key, _base, _model = resolve_zai_config(env={"ZHIPU_API_KEY": "z-key"})
    assert key == "z-key"


def test_zai_provider_constructed_openai_compatible(monkeypatch):
    """The zai path reuses the OpenAI-compatible client with a base URL —
    no model-specific logic is hardcoded (offline: the client is never called)."""
    monkeypatch.setenv("ZAI_API_KEY", "k-test")
    args = td.parse_args([
        "--provider", "zai", "--confirm-external-sharing",
        "--src", "whatever.jsonl", "--out", str(Path(config.DATA_DIR) / "x.jsonl"),
    ])
    provider = td.build_provider(args)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == config.ZAI_MODEL
    assert "api.z.ai" in str(provider._client.base_url)


# --------------------------------------------------------------------------
# Token accounting from usage metadata
# --------------------------------------------------------------------------


class _Usage:
    def __init__(self, prompt, completion, total):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


def test_accumulate_usage_adds_all_fields():
    totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    _accumulate_usage(_Usage(100, 50, 150), totals)
    _accumulate_usage(_Usage(30, 20, 50), totals)
    assert totals == {"requests": 2, "prompt_tokens": 130, "completion_tokens": 70, "total_tokens": 200}


def test_accumulate_usage_tolerates_missing_metadata():
    totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    _accumulate_usage(None, totals)
    _accumulate_usage(_Usage(None, None, None), totals)  # type: ignore[arg-type]
    assert totals["requests"] == 1  # response received but empty metadata
    assert totals["total_tokens"] == 0


# --------------------------------------------------------------------------
# Hard quota guard (--max-total-tokens)
# --------------------------------------------------------------------------


class BudgetedMockProvider(MockProvider):
    """Mock that reports growing usage metadata like a real provider."""

    name = "mock-budget"
    model = "mock-budget"

    def __init__(self, tokens_per_request: int = 100):
        super().__init__()
        self.tokens_per_request = tokens_per_request
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def translate(self, segments):
        result = super().translate(segments)
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += self.tokens_per_request
        self.usage["completion_tokens"] += self.tokens_per_request // 2
        self.usage["total_tokens"] += self.tokens_per_request + self.tokens_per_request // 2
        return result


def _records(n: int) -> list[SourceRecord]:
    return [
        normalize_source_record(
            {"idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}"},
            position=i,
        )
        for i in range(n)
    ]


def test_quota_guard_stops_gracefully_and_reports_pending():
    provider = BudgetedMockProvider(tokens_per_request=100)
    # batch_size=1 → one 150-token call per record FIELD (3 fields/record).
    # Cap 320 admits exactly 2 calls (300 tokens); the 3rd field-call is
    # blocked, so only record 0 completes before the graceful stop.
    engine = TranslationEngine(provider, batch_size=1, max_total_tokens=320)
    results, summary = engine.translate_records(_records(10))

    assert summary.stop_reason == "quota_cap"
    assert summary.translated == 1
    assert provider.usage["total_tokens"] <= 320 + 150  # cap respected per call
    # Unprocessed records are simply pending — never marked failed.
    assert summary.failed == 0


def test_no_cap_runs_to_completion():
    provider = BudgetedMockProvider(tokens_per_request=10)
    engine = TranslationEngine(provider, batch_size=2)
    results, summary = engine.translate_records(_records(4))
    assert summary.stop_reason == ""
    assert summary.translated == 4


def test_summary_persists_stop_reason_and_usage(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({
                "idx": i, "instruction": f"p{i}", "input": f"m{i}", "output": f"r{i}",
            }, ensure_ascii=False) + "\n")
    out = tmp_path / "out.jsonl"
    provider = BudgetedMockProvider(tokens_per_request=50)

    class TinyCapEngine(TranslationEngine):
        def __init__(self, *a, **kw):
            # 3 field-calls of 75 tokens fit under the 200-token cap → exactly
            # record 0 completes before the graceful stop.
            kw["max_total_tokens"] = 200
            super().__init__(*a, **kw)

    monkeypatch.setattr(td, "TranslationEngine", TinyCapEngine)
    monkeypatch.setattr(td, "MockProvider", lambda: provider, raising=True)
    monkeypatch.setattr("sys.argv", [
        "translate_dataset.py", "--provider", "mock",
        "--src", str(src), "--out", str(out),
        "--summary-out", str(tmp_path / "s.json"),
    ])
    assert td.main() == 0
    summary = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "quota_cap"
    assert summary["provider_usage"]["total_tokens"] > 0
    assert summary["engine"]["stop_reason"] == "quota_cap"
    # Progress checkpointed: exactly one complete record was persisted.
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1 and lines[0]["idx"] == 0
