"""Tests for privacy_log: sensitive-field redaction and structured events.

Privacy guarantee under test: conversational content must never appear in
log output — sensitive keys and long strings are replaced by non-reversible
fingerprints.
"""
import json
import logging
from pathlib import Path

import pytest

import config
import privacy_log
from privacy_log import (
    SENSITIVE_KEYS,
    fingerprint,
    get_logger,
    log_event,
    sanitize_fields,
    sanitize_value,
)


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch):
    """Point file logging at a temp dir and reset the logger cache per test."""
    monkeypatch.setattr(config, "LOG_JSON", True)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "LOG_MAX_INLINE_LEN", 120)
    saved_cache = dict(privacy_log._LOGGERS)
    privacy_log._LOGGERS.clear()
    yield
    for logger in list(privacy_log._LOGGERS.values()):
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    privacy_log._LOGGERS.clear()
    privacy_log._LOGGERS.update(saved_cache)


# --------------------------------------------------------------------------
# fingerprint / sanitize_value
# --------------------------------------------------------------------------


def test_fingerprint_is_non_reversible_and_deterministic():
    fp = fingerprint("patient conversation content")
    assert set(fp) == {"len", "sha256"}
    assert fp["len"] == len("patient conversation content")
    assert len(fp["sha256"]) == 12
    assert "patient" not in fp["sha256"]
    assert fingerprint("patient conversation content") == fp  # deterministic
    assert fingerprint("different content") != fp


def test_sanitize_value_redacts_sensitive_keys():
    for key in ("text", "input", "output", "instruction", "conversation"):
        assert key.lower() in SENSITIVE_KEYS
        out = sanitize_value(key, "hello sensitive world")
        assert out == {"len": 21, "sha256": fingerprint("hello sensitive world")["sha256"]}
        assert "hello" not in json.dumps(out)


def test_sanitize_value_redacts_case_insensitive_keys():
    out = sanitize_value("Text", "some conversation")
    assert isinstance(out, dict) and "len" in out


def test_sanitize_value_fingerprints_long_strings_even_on_safe_keys():
    long_public = "x" * 500  # exceeds config.LOG_MAX_INLINE_LEN
    out = sanitize_value("public_field", long_public)
    assert isinstance(out, dict) and out["len"] == 500


def test_sanitize_value_passes_through_short_safe_values():
    assert sanitize_value("model", "minilm") == "minilm"
    assert sanitize_value("count", 5) == 5
    assert sanitize_value("ratio", 0.25) == 0.25
    assert sanitize_value("enabled", True) is True
    assert sanitize_value("nothing", None) is None


def test_sanitize_fields_recurses_into_containers():
    fields = {
        "path": "outputs/labeled.csv",
        "text": "raw conversation must vanish",
        "counts": {"kept": 3, "note": "ok"},
        "items": ["short", "y" * 300],
    }
    out = sanitize_fields(fields)
    dumped = json.dumps(out, ensure_ascii=False)
    assert "raw conversation must vanish" not in dumped
    assert out["path"] == "outputs/labeled.csv"
    assert out["counts"] == {"kept": 3, "note": "ok"}
    assert out["items"][0] == "short"
    assert isinstance(out["items"][1], dict) and out["items"][1]["len"] == 300


# --------------------------------------------------------------------------
# log_event / get_logger / JsonFormatter
# --------------------------------------------------------------------------


def _read_log_lines(tmp_path: Path) -> list[dict]:
    log_file = tmp_path / "pipeline.log.jsonl"
    assert log_file.exists(), "log file was not created"
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_log_event_writes_sanitized_json_line(tmp_path):
    logger = get_logger("phase0.event_test")
    log_event(
        logger,
        "test_event",
        source="fa",
        text="secret conversation body",
        n=3,
    )
    events = _read_log_lines(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "test_event"
    assert event["level"] == "INFO"
    assert event["source"] == "fa"
    assert event["n"] == 3
    assert isinstance(event["text"], dict)  # fingerprinted
    dumped = json.dumps(event, ensure_ascii=False)
    assert "secret conversation body" not in dumped


def test_log_event_unbound_log_record_uses_message_field(tmp_path):
    logger = get_logger("phase0.plain")
    logger.warning("plain warning without event payload")
    events = _read_log_lines(tmp_path)
    assert events[0]["message"] == "plain warning without event payload"
    assert "event" not in events[0]


def test_get_logger_is_cached_per_name():
    first = get_logger("phase0.cached")
    second = get_logger("phase0.cached")
    assert first is second


def test_get_logger_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_JSON", False)
    logger = get_logger("phase0.disabled")
    log_event(logger, "should_not_persist", detail="x")
    assert not (tmp_path / "pipeline.log.jsonl").exists()
    # handler is a NullHandler → nothing crashes, nothing persists
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
