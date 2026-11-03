"""Advanced Persian translation engine for MentalChat16K (Phase 6).

Improvements over the original single-provider script (backward-compatible CLI):

- configurable provider (``openai`` default; ``mock`` for tests/smoke runs)
- batch processing with configurable batch size and a strict JSON-array
  response protocol, with per-record fallback when a batch response cannot
  be parsed
- retry with exponential backoff, rate-limit handling (honors Retry-After),
  and per-request timeouts
- interrupted-job recovery via resume-from-output (existing records are
  never re-translated or duplicated)
- per-record translation status (``translated`` | ``flagged`` |
  ``empty_source``) and structured error reporting (JSONL error log)
- progress reporting and privacy-preserving logging: raw conversation
  content never reaches logs or error reports (sensitive strings are
  fingerprinted via :mod:`privacy_log`)

Schema of the translated output (``data/mentalchat16k_fa.jsonl``):

    {"idx": <original id>,
     "instruction": <FA>, "input": <FA>, "output": <FA>,
     "instruction_en": <original EN>, "input_en": ..., "output_en": ...,
     "text_en": <patient-view EN if present in the source record>,
     "status": "translated" | "flagged" | "empty_source",
     "model": <provider model id>,
     "flag_reason": <only when flagged>}

English originals are preserved under ``*_en`` keys; the Persian fields use
the plain keys so :func:`data_loader.load_samples_fa` consumes the file
natively (patient input stays the default labeling target).

Credentials come from the environment (``OPENAI_API_KEY``), never from code.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
import privacy_log
from data_loader import _parse_tagged_text
from privacy_log import sanitize_value

logger = privacy_log.get_logger("autism.translation")

SOURCE_FIELDS: tuple[str, ...] = ("instruction", "input", "output")
# Derived metadata keys recomputed by data_loader at load time — not original
# data, so they are dropped from translated records rather than passed through.
DERIVED_META_KEYS: frozenset[str] = frozenset({"record_hash", "lang", "source"})

STATUS_TRANSLATED = "translated"
STATUS_FLAGGED = "flagged"
STATUS_EMPTY_SOURCE = "empty_source"

UNTRANSLATABLE_MARKER = "[UNTRANSLATABLE]"

SYSTEM_PROMPT_PATH = config.PROMPTS_DIR / "semantic_persian_translation.txt"

_TEXT_VIEW_KEYS = ("text",)


class TranslationError(Exception):
    """Provider/protocol failure with a machine-readable kind."""

    def __init__(self, kind: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind  # config | rate_limit | timeout | api | parse | invalid_response
        self.retry_after = retry_after


# --------------------------------------------------------------------------
# Providers (raw-content contract: translate(segments) -> response string)
# --------------------------------------------------------------------------


class TranslationProvider:
    """Minimal provider interface: return the RAW model response for a
    list of segment strings. Parsing/retries live in the engine."""

    name = "base"
    model = "unknown"

    def translate(self, segments: list[str]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class MockProvider(TranslationProvider):
    """Deterministic test double — echoes input inside explicit markers.

    Output is intentionally NOT Persian: QA (translation_qa) must flag it,
    which the smoke run uses as positive evidence that validation works.
    Never used for production data.
    """

    name = "mock"
    model = "mock-echo"

    def translate(self, segments: list[str]) -> str:
        return json.dumps(
            [f"[MOCK-FA]{text}[/MOCK-FA]" for text in segments], ensure_ascii=False
        )


class OpenAIProvider(TranslationProvider):
    """OpenAI chat provider (lazy import; credentials from environment)."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise TranslationError(
                "config",
                "OPENAI_API_KEY is not set; export it or choose --provider mock.",
            )
        try:
            from openai import OpenAI  # lazy: only needed for real translation
        except ImportError as exc:
            raise TranslationError(
                "config", f"openai package is not installed ({exc}); pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self._api_key, timeout=timeout)

    def translate(self, segments: list[str]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _load_system_prompt()},
                    {"role": "user", "content": _build_user_message(segments)},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise TranslationError("invalid_response", "provider returned empty content")
            return content
        except TranslationError:
            raise
        except Exception as exc:  # map SDK exceptions to kinds
            raise _map_provider_exception(exc) from exc


def _map_provider_exception(exc: Exception) -> TranslationError:
    """Map an SDK exception to a TranslationError without leaking content.

    Error strings come from the provider (status text); long messages are
    fingerprinted by the logging layer before they can reach any report.
    """
    name = type(exc).__name__
    retry_after: float | None = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
    if getattr(exc, "retry_after", None) is not None:
        try:
            retry_after = float(exc.retry_after)  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            pass
    lowered = name.lower()
    if "ratelimit" in lowered or "rate_limit" in lowered:
        return TranslationError("rate_limit", f"{name}", retry_after=retry_after)
    if "timeout" in lowered or "apitimeout" in lowered:
        return TranslationError("timeout", f"{name}", retry_after=retry_after)
    if "connection" in lowered or "apiconnection" in lowered:
        return TranslationError("api", f"{name}", retry_after=retry_after)
    return TranslationError("api", f"{name}: {str(exc)[:200]}", retry_after=retry_after)


def _load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise TranslationError(
            "config", f"missing translation prompt: {SYSTEM_PROMPT_PATH}"
        )
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------
# Request/response protocol (JSON array of translated segments)
# --------------------------------------------------------------------------


def _build_user_message(segments: list[str]) -> str:
    numbered = "\n".join(
        f"{i + 1}. {json.dumps(text, ensure_ascii=False)}"
        for i, text in enumerate(segments)
    )
    return (
        "Translate every numbered JSON string below into natural Iranian "
        "Persian, following the system rules.\n"
        "Return ONLY a JSON array with exactly "
        f"{len(segments)} strings in the same order — no explanations.\n"
        "If a segment cannot be translated faithfully, return the string "
        f'"{UNTRANSLATABLE_MARKER} <short English reason>" for that position.\n'
        "Preserve paragraph breaks inside each translation.\n\n"
        f"{numbered}"
    )


def parse_batch_response(content: str, n_expected: int) -> list[str]:
    """Parse the provider response into exactly ``n_expected`` strings.

    Raises :class:`TranslationError` (kind ``parse`` / ``invalid_response``)
    on malformed JSON, wrong element count, or non-string elements.
    """
    text = content.strip()
    # Tolerate markdown fences around the JSON array.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranslationError("parse", f"response is not valid JSON ({exc})") from exc
    if not isinstance(parsed, list):
        raise TranslationError("parse", "response is not a JSON array")
    if len(parsed) != n_expected:
        raise TranslationError(
            "parse", f"expected {n_expected} segments, got {len(parsed)}"
        )
    if not all(isinstance(item, str) for item in parsed):
        raise TranslationError("invalid_response", "non-string element in response array")
    return parsed


def split_untranslatable(translation: str) -> tuple[str | None, str | None]:
    """Return ``(translation, None)`` or ``(None, reason)`` when flagged."""
    stripped = translation.strip()
    if stripped.startswith(UNTRANSLATABLE_MARKER):
        return None, stripped[len(UNTRANSLATABLE_MARKER):].strip(" :") or "unspecified"
    return translation, None


# --------------------------------------------------------------------------
# Source normalization
# --------------------------------------------------------------------------


@dataclass
class SourceRecord:
    idx: int
    fields: dict[str, str]                      # non-empty EN fields to translate
    originals: dict[str, str]                   # all original EN field values
    extra: dict[str, Any]                       # other preserved source keys
    status: str = STATUS_TRANSLATED


def normalize_source_record(raw: Any, position: int) -> SourceRecord | None:
    """Normalize one source record; returns None for non-object rows."""
    if not isinstance(raw, dict):
        return None
    raw_idx = raw.get("idx")
    idx = int(raw_idx) if isinstance(raw_idx, (int, str)) and str(raw_idx).isdigit() else position

    originals = {f: str(raw.get(f) or "").strip() for f in SOURCE_FIELDS}
    if not any(originals.values()) and raw.get("text"):
        # Old export format: label text (or tagged blocks) without fields.
        parsed = _parse_tagged_text(str(raw["text"]))
        for f in SOURCE_FIELDS:
            originals[f] = parsed.get(f, "")
        if not any(originals.values()):
            originals["input"] = str(raw["text"]).strip()  # patient-view text
    fields = {f: text for f, text in originals.items() if text}
    extra = {
        k: v
        for k, v in raw.items()
        if k not in SOURCE_FIELDS and k not in DERIVED_META_KEYS and k != "idx"
    }
    return SourceRecord(idx=idx, fields=fields, originals=originals, extra=extra)


def build_output_record(
    record: SourceRecord,
    translations: dict[str, str],
    status: str,
    model: str,
    flag_reason: str | None = None,
) -> dict[str, Any]:
    """Build one translated output record (English preserved under ``*_en``)."""
    out: dict[str, Any] = {"idx": record.idx}
    for f in SOURCE_FIELDS:
        out[f] = translations.get(f, "")
        out[f + "_en"] = record.originals.get(f, "")
    for key, value in record.extra.items():
        out[f"{key}_en"] = value
    out["status"] = status
    out["model"] = model
    if flag_reason:
        out["flag_reason"] = flag_reason
    return out


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class EngineError:
    idx: int
    kind: str
    message: str
    attempts: int


@dataclass
class EngineSummary:
    total_source: int = 0
    done_before: int = 0
    duplicates_skipped: int = 0
    translated: int = 0
    flagged: int = 0
    empty_source: int = 0
    failed: int = 0
    errors: list[EngineError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["errors"] = [e.__dict__ for e in self.errors]
        return data


class TranslationEngine:
    """Batched, retrying, resumable translation driver."""

    def __init__(
        self,
        provider: TranslationProvider,
        *,
        batch_size: int = 1,
        max_retries: int = 3,
        base_backoff: float = 2.0,
        inter_request_sleep: float = 0.0,
        progress_every: int = 25,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.provider = provider
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.base_backoff = float(base_backoff)
        self.inter_request_sleep = float(inter_request_sleep)
        self.progress_every = int(progress_every)
        self._rng = random.Random(42)  # jitter; deterministic seed

    # -- core: translate a mapping of work items --------------------------

    def _translate_segments(self, work: list[tuple[int, str, str]]) -> dict[int, dict[str, str]]:
        """Translate ``[(idx, field, text), ...]`` → ``{idx: {field: fa}}``.

        Raises the last :class:`TranslationError` after exhausting retries.
        """
        segments = [text for _, _, text in work]
        last_error: TranslationError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content = self.provider.translate(segments)
                translations = parse_batch_response(content, len(segments))
                return self._assign(work, translations)
            except TranslationError as exc:
                last_error = exc
                if exc.kind == "parse" and len(work) > 1:
                    # Batch protocol failed → fall back to one segment per call.
                    logger.warning(
                        "batch parse failed; falling back to per-segment calls",
                        extra={"event": "batch_fallback", "fields": {"n": len(work)}},
                    )
                    merged: dict[int, dict[str, str]] = {}
                    for item in work:
                        # Deep-merge per idx: dict.update on the outer map would
                        # replace a record's field map and lose every field but
                        # the last one.
                        for idx, field_map in self._translate_segments([item]).items():
                            merged.setdefault(idx, {}).update(field_map)
                    return merged
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt, exc.retry_after)
                    logger.warning(
                        "provider error; retrying",
                        extra={
                            "event": "provider_retry",
                            "fields": {
                                "kind": exc.kind,
                                "attempt": attempt + 1,
                                "delay_s": round(delay, 2),
                            },
                        },
                    )
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _assign(
        self, work: list[tuple[int, str, str]], translations: list[str]
    ) -> dict[int, dict[str, str]]:
        assigned: dict[int, dict[str, str]] = {}
        for (idx, field_name, _), translation in zip(work, translations):
            text, reason = split_untranslatable(translation)
            if text is None:
                # Marked untranslatable: record it as flagged, empty FA field.
                assigned.setdefault(idx, {})[f"__flag__{field_name}"] = reason or ""
            else:
                assigned.setdefault(idx, {})[field_name] = translation
        return assigned

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), 120.0)
        delay = self.base_backoff * (2**attempt)
        return delay + self._rng.uniform(0, delay * 0.1)

    # -- public API --------------------------------------------------------

    def _translate_accumulating(
        self,
        records: list[SourceRecord],
        summary: EngineSummary,
        on_result: Any,
        on_progress: Any = None,
    ) -> None:
        """Core accumulator: segment-batched translation emitting each record
        exactly once.

        ``batch_size`` caps the number of field segments per provider call; a
        record's fields may straddle a batch boundary, so per-idx assignments
        are merged across calls and the record is emitted only when all of its
        fields are accounted for (or flagged). Records whose batches failed are
        counted in ``summary.failed`` and stay absent from the output so a
        later run retries them — they are never silently skipped.
        """
        assignments: dict[int, dict[str, str]] = {}
        flag_map: dict[int, dict[str, str]] = {}
        failed_idx: set[int] = set()
        records_by_idx: dict[int, SourceRecord] = {}
        emitted: set[int] = set()

        def emit(idx: int) -> None:
            record = records_by_idx[idx]
            translations = assignments.get(idx, {})
            flags = flag_map.get(idx, {})
            if flags:
                status = STATUS_FLAGGED
                reason = "; ".join(
                    f"{field}: {reason}" for field, reason in sorted(flags.items())
                )
            else:
                status = STATUS_TRANSLATED
                reason = None
            on_result(
                build_output_record(record, translations, status, self.provider.model, reason)
            )
            emitted.add(idx)
            if status == STATUS_TRANSLATED:
                summary.translated += 1
            else:
                summary.flagged += 1

        def record_complete(idx: int) -> bool:
            record = records_by_idx[idx]
            return (
                len(assignments.get(idx, {})) + len(flag_map.get(idx, {}))
                >= len(record.fields)
            )

        def flush_slice(work_slice: list[tuple[int, str, str]]) -> None:
            try:
                assigned = self._translate_segments(work_slice)
            except TranslationError as exc:
                for idx in sorted({i for i, _, _ in work_slice}):
                    if idx not in failed_idx:
                        failed_idx.add(idx)
                        summary.failed += 1
                        summary.errors.append(
                            EngineError(
                                idx=idx,
                                kind=exc.kind,
                                message=str(sanitize_value("error", str(exc))),
                                attempts=self.max_retries + 1,
                            )
                        )
                return
            for idx, field_map in assigned.items():
                bucket = assignments.setdefault(idx, {})
                flags = flag_map.setdefault(idx, {})
                for key, value in field_map.items():
                    if key.startswith("__flag__"):
                        flags[key[len("__flag__"):]] = value
                    else:
                        bucket[key] = value
            for idx in sorted({i for i, _, _ in work_slice}):
                if idx not in emitted and idx not in failed_idx and record_complete(idx):
                    emit(idx)

        work: list[tuple[int, str, str]] = []
        for record in records:
            if record.idx in records_by_idx:
                summary.duplicates_skipped += 1
                continue
            records_by_idx[record.idx] = record
            if not record.fields:
                on_result(
                    build_output_record(
                        record, {}, STATUS_EMPTY_SOURCE, self.provider.model
                    )
                )
                emitted.add(record.idx)
                summary.empty_source += 1
                continue
            work.extend((record.idx, f, t) for f, t in record.fields.items())
            while len(work) >= self.batch_size:
                flush_slice(work[: self.batch_size])
                work = work[self.batch_size :]
                if self.inter_request_sleep:
                    time.sleep(self.inter_request_sleep)
                if on_progress and self.progress_every and len(emitted) and len(emitted) % self.progress_every == 0:
                    on_progress(len(emitted))
        while work:
            flush_slice(work[: self.batch_size])
            work = work[self.batch_size :]
        # Defensive drain: a record whose fields all succeeded but that was
        # never emitted (should not happen) is emitted here, never lost.
        for idx in list(records_by_idx):
            if idx not in emitted and idx not in failed_idx and records_by_idx[idx].fields:
                if record_complete(idx):
                    emit(idx)

    def translate_records(self, records: list[SourceRecord]) -> tuple[list[dict], EngineSummary]:
        """Translate normalized records in memory (no file I/O)."""
        summary = EngineSummary(total_source=len(records))
        results: list[dict] = []
        self._run_accumulator_core(records, summary, results)
        return results, summary

    def _run_accumulator_core(
        self,
        records: list[SourceRecord],
        summary: EngineSummary,
        results: list[dict],
    ) -> None:
        def on_result(row: dict) -> None:
            results.append(row)

        def on_progress(emitted_count: int) -> None:
            print(
                f"[translate] progress: {emitted_count}/{summary.total_source}"
                f" translated={summary.translated} flagged={summary.flagged}"
                f" failed={summary.failed}"
            )

        self._translate_accumulating(records, summary, on_result, on_progress)

    # -- resumable file runner ---------------------------------------------

    def run(
        self,
        src_path: str | Path,
        out_path: str | Path,
        error_log_path: str | Path | None = None,
        limit: int | None = None,
    ) -> EngineSummary:
        """Read source JSONL, resume from existing output, translate, write.

        Failed records are reported (never silently skipped) and simply
        remain absent from the output, so a later run retries them.
        """
        src_path, out_path = Path(src_path), Path(out_path)
        raw_rows, malformed = _read_jsonl(src_path)
        records: list[SourceRecord] = []
        skipped_non_object = 0
        for position, row in enumerate(raw_rows):
            record = normalize_source_record(row, position)
            if record is None:
                skipped_non_object += 1
                continue
            records.append(record)
        if limit is not None and limit > 0:
            records = records[:limit]

        done: dict[int, dict] = {}
        if out_path.exists():
            done_rows, _ = _read_jsonl(out_path)
            for row in done_rows:
                if isinstance(row, dict) and "idx" in row:
                    done[int(row["idx"])] = row

        pending = [r for r in records if r.idx not in done]
        summary = EngineSummary(
            total_source=len(records), done_before=len(done)
        )
        privacy_log.log_event(
            logger,
            "translation_start",
            src=str(src_path),
            out=str(out_path),
            total=len(records),
            done_before=len(done),
            provider=self.provider.name,
            model=self.provider.model,
            batch_size=self.batch_size,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fout:

            def on_result(row: dict) -> None:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

            self._translate_accumulating(pending, summary, on_result)

        # Recompute failed cleanly: unique idx values that ended in errors.
        summary.failed = len({e.idx for e in summary.errors})

        if error_log_path and summary.errors:
            _write_error_log(Path(error_log_path), summary.errors)
        privacy_log.log_event(
            logger,
            "translation_complete",
            translated=summary.translated,
            flagged=summary.flagged,
            empty_source=summary.empty_source,
            failed=summary.failed,
            duplicates_skipped=summary.duplicates_skipped,
            malformed_source_lines=malformed,
            non_object_source_rows=skipped_non_object,
        )
        print(
            f"[translate] done: translated={summary.translated} flagged={summary.flagged} "
            f"empty_source={summary.empty_source} failed={summary.failed} "
            f"duplicates_skipped={summary.duplicates_skipped} -> {out_path}"
        )
        if malformed or skipped_non_object:
            print(
                f"[translate] WARNING: source issues: malformed_lines={malformed} "
                f"non_object_rows={skipped_non_object}"
            )
        return summary


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> tuple[list[Any], int]:
    rows: list[Any] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def _write_error_log(path: Path, errors: list[EngineError]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error.__dict__, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# CLI (backward compatible)
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persian translation of MentalChat16K.")
    p.add_argument("--src", type=str,
                   default=str(config.DATA_DIR / "mentalchat16k_en_for_translation.jsonl"))
    p.add_argument("--out", type=str, default=str(config.PERSIAN_DATASET_PATH))
    p.add_argument("--prompt", type=str, default=str(SYSTEM_PROMPT_PATH),
                   help="kept for compatibility; the system prompt file is used by the engine")
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "mock"])
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--sleep", type=float, default=0.4,
                   help="inter-request politeness delay in seconds")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--error-log", type=str, default=str(config.DATA_DIR / "translation_errors.jsonl"))
    p.add_argument("--summary-out", type=str,
                   default=str(config.DATA_DIR / "translation_summary.json"))
    p.add_argument(
        "--confirm-external-sharing", action="store_true",
        help="acknowledge that conversation content will be sent to the configured "
             "external provider (required for --provider openai; review privacy "
             "and licensing constraints first)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        print(f"[translate] ERROR: missing source {src}", file=sys.stderr)
        print(
            "Export English records first: python export_for_translation.py --limit 500",
            file=sys.stderr,
        )
        return 2

    if args.provider == "mock" and out.resolve() == config.PERSIAN_DATASET_PATH.resolve():
        print(
            "[translate] ERROR: refusing to write mock output to the production "
            f"dataset path ({config.PERSIAN_DATASET_PATH}). Mock output is synthetic, "
            "not a translation — choose a different --out path.",
            file=sys.stderr,
        )
        return 2
    if args.provider == "openai" and not args.confirm_external_sharing:
        print(
            "[translate] ERROR: the openai provider uploads conversation content to "
            "an external service. Review your privacy and licensing constraints, "
            "then pass --confirm-external-sharing to acknowledge.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.provider == "mock":
            provider: TranslationProvider = MockProvider()
        else:
            provider = OpenAIProvider(model=args.model, timeout=args.timeout)
    except TranslationError as exc:
        print(f"[translate] ERROR: {exc}", file=sys.stderr)
        if exc.kind == "config":
            print(
                "[translate] BLOCKER: no API credentials available. "
                "Set OPENAI_API_KEY (never commit it) and rerun, "
                "or use --provider mock for a mechanics-only smoke test.",
                file=sys.stderr,
            )
        return 2

    privacy_log.log_event(
        logger,
        "provider_notice",
        provider=provider.name,
        model=provider.model,
        confirmed=args.provider != "openai" or args.confirm_external_sharing,
        note="source records will be sent to the configured external provider",
    )
    print(
        f"[translate] provider={provider.name} model={provider.model} "
        f"batch_size={args.batch_size} max_retries={args.max_retries}"
    )

    engine = TranslationEngine(
        provider,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        inter_request_sleep=args.sleep,
    )
    summary = engine.run(src, out, error_log_path=Path(args.error_log), limit=args.limit)

    # Always finish with automated quality control (never proves semantics)
    # and a completeness check against the source (every record accounted for).
    from translation_qa import check_completeness, validate_translation_file, print_summary

    report = validate_translation_file(out)
    print_summary(report)
    completeness = check_completeness(src, out)
    print(
        f"[translate] completeness: total_source={completeness['total_source']} "
        f"translated={completeness['translated']} missing={completeness['missing']} "
        f"duplicated_output_ids={completeness['duplicated_output_ids']}"
    )

    summary_out = {
        "engine": summary.to_dict(),
        "qa": {
            "n_records": report["n_records"],
            "n_valid": report["n_valid"],
            "duplicates": report["duplicates"],
            "malformed_lines": report["malformed_lines"],
            "issues_total": report["issues_total"],
            "issues_by_code": report["issues_by_code"],
            "completion": report["completion"],
        },
        "completeness": completeness,
        "provider": provider.name,
        "model": provider.model,
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[translate] summary -> {args.summary_out}")
    if report["issues_total"]:
        print(
            f"[translate] NOTE: {report['issues_total']} QA issue(s) recorded; "
            "automated QA does not prove semantic accuracy — review flagged records."
        )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
