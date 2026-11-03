"""Structured, privacy-preserving JSON-line logging for the pipeline.

Design rules
------------
- Logs are JSON lines (one object per line) written to ``config.LOG_DIR``.
- Conversational content (patient text) must **never** reach the logs: any
  field whose name is in ``SENSITIVE_KEYS`` is replaced by a fingerprint
  (``{"len": <int>, "sha256": <12 hex chars>}``) that cannot be reversed.
- Long strings are fingerprinted even under non-sensitive keys, so an
  accidental key rename cannot leak conversations either.
- File logging can be disabled with ``config.LOG_JSON = False``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import config

# Keys that may carry patient conversation content. Matched case-insensitively.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "text",
        "input",
        "output",
        "instruction",
        "content",
        "message",
        "conversation",
        "response",
        "prompt",
        "sample_text",
        "label_text",
    }
)

# Length of the SHA-256 prefix used as a non-reversible fingerprint.
_FINGERPRINT_LEN = 12

_LOGGERS: dict[str, logging.Logger] = {}


def fingerprint(value: str) -> dict[str, Any]:
    """Return a privacy-safe summary of a sensitive string.

    The fingerprint is a salted-free SHA-256 prefix; it allows correlating
    log entries about the same record without revealing its content.
    """
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    return {"len": len(value), "sha256": digest[:_FINGERPRINT_LEN]}


def sanitize_value(key: str, value: Any) -> Any:
    """Recursively sanitize one (key, value) pair for logging."""
    if isinstance(value, str):
        if key.lower() in SENSITIVE_KEYS or len(value) > config.LOG_MAX_INLINE_LEN:
            return fingerprint(value)
        return value
    if isinstance(value, dict):
        return {k: sanitize_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(key, v) for v in value]
    return value


def sanitize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Sanitize every field of a structured log event."""
    return {k: sanitize_value(k, v) for k, v in fields.items()}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
            fields = getattr(record, "fields", {}) or {}
            payload.update(fields)
        else:
            payload["message"] = record.getMessage()
        if record.exc_info:
            payload["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).strip()
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a cached JSON-file logger; falls back to NullHandler on errors."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO))
    logger.propagate = False

    if config.LOG_JSON:
        try:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                config.LOG_DIR / "pipeline.log.jsonl", encoding="utf-8"
            )
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        except OSError as exc:  # unwritable disk, permission denied, ...
            # Transparent fallback: keep the pipeline running without file logs.
            print(
                f"[privacy_log] WARNING: cannot open log file ({exc}); "
                "file logging disabled.",
                file=sys.stderr,
            )
            logger.addHandler(logging.NullHandler())
    else:
        logger.addHandler(logging.NullHandler())

    _LOGGERS[name] = logger
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one sanitized structured event to the given logger."""
    logger.log(
        level,
        "pipeline_event",
        extra={"event": event, "fields": sanitize_fields(fields)},
    )
