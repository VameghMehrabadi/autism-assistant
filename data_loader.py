"""بارگذاری دیتاست MentalChat16K (انگلیسی) یا نسخه‌ی فارسی ترجمه‌شده.

- en: ShenLab/MentalChat16K از Hugging Face
- fa: فایل محلی JSONL/CSV/TSV/JSON (مسیر config.PERSIAN_DATASET_PATH یا سفارشی)

Phase-2 data engineering (backward compatible):
- encoding detection + robust file reading (malformed lines skipped and counted)
- schema validation, missing-value handling, duplicate detection
- script-based language identification (counting + optional filtering)
- input/response separation (``Sample.input_text`` / ``Sample.response_text``)
- reproducible limits & sampling (``first`` | seeded ``random``)
- dev/eval split loading with an explicit leakage guard (manifest-based)
- structured privacy-preserving logging (no raw conversations in logs)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

import config
import data_validation as dv
import privacy_log
from data_validation import DataValidationError, ValidationIssue

# `datasets` is imported lazily inside load_samples_en so Persian-only
# workflows do not require it at import time.

logger = privacy_log.get_logger("autism.data_loader")


@dataclass
class Sample:
    """One labeling unit: label text + separated input/response + metadata.

    ``input_text``/``response_text`` default to ``""`` so positional
    construction ``Sample(idx, text, meta)`` keeps working (backward
    compatibility). ``meta`` carries original identifiers (``idx``), the
    content ``record_hash``, detected ``lang`` and the ``source`` key.
    """

    idx: int
    text: str
    meta: dict
    input_text: str = ""
    response_text: str = ""


@dataclass
class LoadReport:
    """Transparent summary of one load call (never contains record text)."""

    source: str
    path: str | None
    total_rows: int
    kept: int
    encoding: str | None = None
    malformed_lines: int = 0
    non_object_records: int = 0
    skipped_invalid: int = 0
    duplicates_removed: int = 0
    language_filtered: int = 0
    empty_text: int = 0
    dedup: str = "keep_first"
    language_filter: str | None = None
    language_distribution: dict[str, int] = field(default_factory=dict)
    translation_skipped: dict[str, int] = field(default_factory=dict)
    dataset_total: int | None = None  # total source records available (before limit)
    dataset_kind: str | None = None  # "real" | "mock" | "fixture" (provenance)
    sampling: str = "first"
    seed: int | None = None
    split: str | None = None
    split_stats: dict[str, Any] | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    preprocess_version: str = dv.PREPROCESS_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data


_TAGGED_FIELDS: tuple[str, ...] = ("instruction", "input", "output")


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_generic_counselor_prompt(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in config.GENERIC_COUNSELOR_MARKERS)


def _parse_tagged_text(text: str) -> dict[str, str]:
    """Parse `instruction: ...\\ninput: ...\\noutput: ...` blocks."""
    buckets: dict[str, list[str]] = {k: [] for k in _TAGGED_FIELDS}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        matched = None
        for key in _TAGGED_FIELDS:
            prefix = f"{key}:"
            if stripped.lower().startswith(prefix):
                matched = key
                rest = stripped[len(prefix):].strip()
                current = key
                if rest:
                    buckets[key].append(rest)
                break
        if matched is None and current:
            buckets[current].append(line.rstrip())
    return {k: "\n".join(v).strip() for k, v in buckets.items() if "".join(v).strip()}


def _field_map(row: dict) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in _TAGGED_FIELDS:
        val = _clean(row.get(key))
        if val:
            fields[key] = val
    raw = _clean(row.get("text"))
    if raw:
        parsed = _parse_tagged_text(raw)
        for key, val in parsed.items():
            fields.setdefault(key, val)
        if not parsed:
            fields.setdefault("text", raw)
    return fields


def _merge_fields(row: dict, fields: tuple[str, ...]) -> str:
    """ادغام فیلدهای متنی دیتاست در یک رشته‌ی تمیز."""
    parts: list[str] = []
    for f in fields:
        val = _clean(row.get(f))
        if val:
            parts.append(f"{f}: {val}")
    return "\n".join(parts)


def extract_label_text(row: dict, mode: str | None = None) -> str:
    """متن مورد استفاده برای لیبل‌گذاری.

    patient: شرح مراجع (input). پرامپت تکراری مشاور و پاسخ درمانی حذف می‌شود
    تا دستهٔ F به‌خاطر زبان حمایت/درمان همه‌ی نمونه‌ها را نبرد.
    all: رفتار قبلی — instruction + input + output (یا کل فیلد text).
    """
    mode = (mode or config.LABEL_TEXT_MODE).lower().strip()
    if mode not in {"patient", "all"}:
        raise ValueError(f"Unsupported label text mode {mode!r}. Use patient|all.")

    fields = _field_map(row)
    if mode == "all":
        merged = _merge_fields(
            {k: fields.get(k, "") for k in _TAGGED_FIELDS},
            _TAGGED_FIELDS,
        )
        return merged or fields.get("text", "")

    inp = fields.get("input", "")
    if inp:
        return inp

    inst = fields.get("instruction", "")
    if inst and not _is_generic_counselor_prompt(inst):
        return inst

    raw = fields.get("text", "")
    if raw and not _is_generic_counselor_prompt(raw):
        return raw

    return inp or inst or fields.get("output", "") or raw


def _row_to_text(row: dict, fields: tuple[str, ...], mode: str | None = None) -> str:
    """متن لیبل از ردیف؛ fields فقط برای سازگاری با حالت all استفاده می‌شود."""
    mode = (mode or config.LABEL_TEXT_MODE).lower().strip()
    if mode == "all":
        mapped = _field_map(row)
        merged = _merge_fields(
            {k: mapped.get(k, "") for k in fields},
            fields,
        )
        if merged:
            return merged
        return mapped.get("text", "")
    return extract_label_text(row, mode=mode)


def _normalize_limit(limit: int | None) -> int | None:
    """None/<=0 → None (whole dataset); otherwise a positive int."""
    if limit is None or limit <= 0:
        return None
    return int(limit)


# --------------------------------------------------------------------------
# Robust file reading (Phase 2)
# --------------------------------------------------------------------------


def _load_rows(path: Path) -> tuple[list[dict], dict[str, Any]]:
    """Read a JSONL/JSON/CSV/TSV file into normalized dicts.

    Invalid pieces are handled transparently: malformed JSONL lines and
    non-object JSON values are skipped and counted — never logged verbatim
    (a malformed line may itself carry conversation content).
    """
    encoding = dv.detect_encoding(path)
    suffix = path.suffix.lower()
    stats: dict[str, Any] = {
        "encoding": encoding,
        "malformed_lines": 0,
        "non_object_records": 0,
    }
    rows: list[dict] = []

    if suffix == ".jsonl":
        with path.open("r", encoding=encoding) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_lines"] += 1
                    logger.warning(
                        "malformed jsonl line skipped",
                        extra={"event": "malformed_line", "fields": {"line_no": line_no}},
                    )
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    stats["non_object_records"] += 1
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding=encoding))
        if isinstance(payload, dict) and "data" in payload:
            rows = payload["data"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding)
        except pd.errors.ParserError:
            # Tolerate a few broken rows instead of failing the whole load.
            df = pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines="skip")
        rows = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use .jsonl/.csv/.json")

    return rows, stats


# --------------------------------------------------------------------------
# Shared build pipeline (validation → dedup → language → limit → Sample)
# --------------------------------------------------------------------------


def _build_samples(
    rows: list[dict],
    *,
    fields: tuple[str, ...],
    mode: str,
    source: str,
    path: str | None = None,
    load_stats: dict[str, Any] | None = None,
    limit: int | None = None,
    dedup: str | None = None,
    sampling: str | None = None,
    seed: int | None = None,
    language_filter: str | None = None,
    strict: bool | None = None,
) -> tuple[list[Sample], LoadReport]:
    """Turn raw rows into validated, deduplicated, sampled ``Sample`` objects.

    Every rejected record is counted in the returned :class:`LoadReport` and
    logged with its position and issue code only — never with its content.
    """
    load_stats = load_stats or {}
    limit = _normalize_limit(limit)
    strict = config.VALIDATION_STRICT if strict is None else strict
    dedup = (dedup or config.DEDUP_STRATEGY).lower().strip()
    if dedup not in {"keep_first", "keep_all"}:
        raise ValueError(f"dedup must be 'keep_first' or 'keep_all', got {dedup!r}")
    strategy = (sampling or config.SAMPLING_STRATEGY).lower().strip()
    if strategy not in {"first", "random"}:
        raise ValueError(f"sampling must be 'first' or 'random', got {strategy!r}")
    seed = config.SAMPLING_SEED if seed is None else seed
    language_filter = (
        language_filter.lower().strip()
        if language_filter is not None
        else config.LANGUAGE_FILTER
    )
    if language_filter is not None and language_filter not in {"fa", "en"}:
        raise ValueError(
            f"language_filter must be 'fa', 'en' or None, got {language_filter!r}"
        )

    report = LoadReport(
        source=source,
        path=path,
        total_rows=len(rows),
        kept=0,
        encoding=load_stats.get("encoding"),
        malformed_lines=load_stats.get("malformed_lines", 0),
        non_object_records=load_stats.get("non_object_records", 0),
        dedup=dedup,
        language_filter=language_filter,
        sampling=strategy,
        seed=seed if strategy == "random" else None,
    )

    privacy_log.log_event(
        logger,
        "load_start",
        source=source,
        path=path,
        total_rows=len(rows),
        sampling=strategy,
        dedup=dedup,
        language_filter=language_filter,
    )

    # 1) normalize + schema validation + translation-status gating
    candidates: list[tuple[int, dict[str, str]]] = []
    for position, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            report.non_object_records += 1
            continue
        row = dv.normalize_record(raw_row)

        # Translation-status gate runs BEFORE content validation: records
        # whose translation status is not `translated` are excluded outright
        # (even if their fields are empty) and counted by status — never
        # silently promoted, and never rescued by the patient-mode output
        # fallback.
        status = (row.get("status") or "").strip()
        if status:
            if status != "translated":
                report.translation_skipped[status] = (
                    report.translation_skipped.get(status, 0) + 1
                )
                report.issues.append(
                    ValidationIssue(
                        code=f"TRANSLATION_STATUS_{status.upper()}",
                        position=position,
                        message="record excluded: translation status is not 'translated'",
                        severity="warning",
                    )
                )
                continue
            en_input = (row.get("input_en") or "").strip()
            fa_input = (row.get("input") or "").strip()
            if en_input and not fa_input:
                report.translation_skipped["incomplete_input"] = (
                    report.translation_skipped.get("incomplete_input", 0) + 1
                )
                report.issues.append(
                    ValidationIssue(
                        code="TRANSLATION_INCOMPLETE_INPUT",
                        position=position,
                        message="status=translated but the patient input is untranslated "
                                "(FA input empty while EN input is present)",
                        severity="warning",
                    )
                )
                continue

        issues = dv.validate_record(row, position=position)
        report.issues.extend(issues)
        if any(i.severity == "error" for i in issues):
            if strict:
                raise DataValidationError(
                    f"record at position {position} failed validation: "
                    f"{[i.code for i in issues]}"
                )
            continue
        candidates.append((position, row))
    translation_skipped_total = sum(report.translation_skipped.values())
    report.skipped_invalid = (
        report.total_rows
        - report.non_object_records
        - translation_skipped_total
        - len(candidates)
    )

    # 2) duplicate detection / removal (content fingerprint)
    if dedup == "keep_first":
        seen: set[str] = set()
        deduped: list[tuple[int, dict[str, str]]] = []
        for position, row in candidates:
            fp = dv.record_fingerprint(row)
            if fp in seen:
                report.duplicates_removed += 1
                report.issues.append(
                    ValidationIssue(
                        code="DUPLICATE",
                        position=position,
                        message="duplicate content; kept first occurrence",
                        severity="info",
                    )
                )
                continue
            seen.add(fp)
            deduped.append((position, row))
        candidates = deduped

    # 3) language identification on the label text (+ optional filter)
    lang_texts = [_row_to_text(row, fields, mode=mode) for _, row in candidates]
    report.language_distribution = dv.language_distribution(lang_texts)
    if language_filter is not None:
        filtered: list[tuple[int, dict[str, str]]] = []
        for (position, row), text in zip(candidates, lang_texts):
            if dv.detect_language(text) != language_filter:
                report.language_filtered += 1
                continue
            filtered.append((position, row))
        candidates = filtered

    # 4) limit + reproducible sampling (over the validated pool)
    if limit is not None:
        if strategy == "random":
            pool_indices = range(len(candidates))
            picked = sorted(
                random.Random(seed).sample(
                    list(pool_indices), min(limit, len(candidates))
                )
            )
            candidates = [candidates[i] for i in picked]
        else:
            candidates = candidates[:limit]

    # 5) build samples (patient input is the default labeling target)
    samples: list[Sample] = []
    for position, row in candidates:
        text = _row_to_text(row, fields, mode=mode)
        if not text:
            report.empty_text += 1
            continue
        idx = (
            int(row["idx"]) if row.get("idx", "").isdigit() else position
        )
        mapped = _field_map(row)
        meta = {k: row[k] for k in fields if k in row}
        if "text" in row:
            meta["text"] = row["text"]
        meta["record_hash"] = dv.record_fingerprint(row)
        meta["lang"] = dv.detect_language(text)
        meta["source"] = source
        if row.get("status"):
            meta["translation_status"] = row["status"]
        # Preserve the original English content alongside the translation.
        for en_key in ("instruction_en", "input_en", "output_en"):
            if row.get(en_key):
                meta[en_key] = row[en_key]
        if row.get("flag_reason"):
            meta["flag_reason"] = row["flag_reason"]
        samples.append(
            Sample(
                idx=idx,
                text=text,
                meta=meta,
                input_text=mapped.get("input", ""),
                response_text=mapped.get("output", ""),
            )
        )

    report.kept = len(samples)
    privacy_log.log_event(
        logger,
        "load_complete",
        source=source,
        path=path,
        kept=report.kept,
        skipped_invalid=report.skipped_invalid,
        duplicates_removed=report.duplicates_removed,
        language_filtered=report.language_filtered,
        empty_text=report.empty_text,
        malformed_lines=report.malformed_lines,
        translation_skipped=report.translation_skipped,
        language_distribution=report.language_distribution,
    )
    return samples, report


# --------------------------------------------------------------------------
# Public loaders (signatures backward compatible)
# --------------------------------------------------------------------------


def load_samples_en(
    limit: int | None = None,
    *,
    dedup: str | None = None,
    sampling: str | None = None,
    seed: int | None = None,
    language_filter: str | None = None,
    text_mode: str | None = None,
    allow_mock: bool | None = None,
    return_report: bool = False,
) -> list[Sample] | tuple[list[Sample], LoadReport]:
    """بارگذاری MentalChat16K انگلیسی از Hugging Face."""
    from datasets import load_dataset  # lazy: FA-only workflows skip it

    limit = _normalize_limit(limit)
    strategy = (sampling or config.SAMPLING_STRATEGY).lower()

    print(
        f"[data_loader] Loading '{config.DATASET_NAME}' "
        f"(split={config.DATASET_SPLIT}) ..."
    )
    ds = load_dataset(
        config.DATASET_NAME,
        split=config.DATASET_SPLIT,
        cache_dir=str(config.CACHE_DIR),
    )

    cols = list(ds.features.keys())
    print(f"[data_loader] Columns: {cols} | total rows: {len(ds)}")

    fields = tuple(f for f in config.DATASET_TEXT_FIELDS if f in cols)
    if not fields:
        fields = tuple(c for c in cols if ds.features[c].dtype == "string")
    mode = (text_mode or config.LABEL_TEXT_MODE).lower().strip()
    print(f"[data_loader] Using text fields: {fields} | mode={mode}")

    # Fetch only what the strategy needs (first-N keeps the old behaviour).
    if strategy == "first" and limit is not None:
        n_fetch = min(limit, len(ds))
    else:
        n_fetch = len(ds)
    rows = [dict(ds[i]) for i in range(n_fetch)]
    dataset_kind = dv.detect_dataset_kind(rows)
    if dataset_kind == "mock" and not (config.ALLOW_MOCK_DATASET if allow_mock is None else allow_mock):
        raise DataValidationError(
            "MOCK DATA refused: the loaded dataset contains mock markers. "
            "It must never be treated as production data."
        )

    samples, report = _build_samples(
        rows,
        fields=fields,
        mode=mode,
        source="en",
        limit=limit,
        dedup=dedup,
        sampling=sampling,
        seed=seed,
        language_filter=language_filter,
    )
    report.dataset_total = len(ds)
    report.dataset_kind = dataset_kind
    print(
        f"[data_loader] Prepared {len(samples)} English samples "
        f"(duplicates removed: {report.duplicates_removed})."
    )
    return (samples, report) if return_report else samples


def load_samples_fa(
    path: str | Path | None = None,
    limit: int | None = None,
    *,
    dedup: str | None = None,
    sampling: str | None = None,
    seed: int | None = None,
    language_filter: str | None = None,
    text_mode: str | None = None,
    allow_mock: bool | None = None,
    return_report: bool = False,
) -> list[Sample] | tuple[list[Sample], LoadReport]:
    """بارگذاری دیتاست فارسی از JSONL/CSV/TSV/JSON با اعتبارسنجی کامل.

    قالب‌های پذیرفته‌شده:
    - JSONL: هر خط یک آبجکت با فیلد text یا instruction/input/output
    - CSV/TSV: ستون text یا همان فیلدهای MentalChat16K
    - JSON: لیست رکوردها یا {"data": [...]}
    """
    limit = _normalize_limit(limit)

    fa_path = Path(path) if path else config.PERSIAN_DATASET_PATH
    if not fa_path.exists():
        raise FileNotFoundError(
            f"Persian dataset not found: {fa_path}\n"
            "Export English records with `python export_for_translation.py`, "
            "translate them (GPT-4 + prompts/semantic_persian_translation.txt), "
            "then save as data/mentalchat16k_fa.jsonl."
        )

    print(f"[data_loader] Loading Persian dataset from {fa_path} ...")
    rows, load_stats = _load_rows(fa_path)
    dataset_kind = dv.detect_dataset_kind(rows, path=fa_path)
    if dataset_kind == "mock" and not (config.ALLOW_MOCK_DATASET if allow_mock is None else allow_mock):
        raise DataValidationError(
            f"MOCK DATA refused: {fa_path} is a mock/fixture dataset, not a "
            "real translated corpus. Pass allow_mock=True (--allow-mock) only "
            "for pipeline mechanics testing."
        )

    samples, report = _build_samples(
        rows,
        fields=config.DATASET_TEXT_FIELDS,
        mode=(text_mode or config.LABEL_TEXT_MODE).lower().strip(),
        source="fa",
        path=str(fa_path),
        load_stats=load_stats,
        limit=limit,
        dedup=dedup,
        sampling=sampling,
        seed=seed,
        language_filter=language_filter,
    )
    report.dataset_total = len(rows)
    report.dataset_kind = dataset_kind
    print(
        f"[data_loader] Prepared {len(samples)} Persian samples "
        f"(mode={config.LABEL_TEXT_MODE}, encoding={report.encoding}, "
        f"duplicates removed: {report.duplicates_removed})."
    )
    return (samples, report) if return_report else samples


def load_samples(
    limit: int | None = None,
    dataset: str = "en",
    path: str | Path | None = None,
    *,
    split: str | None = None,
    dedup: str | None = None,
    sampling: str | None = None,
    seed: int | None = None,
    language_filter: str | None = None,
    text_mode: str | None = None,
    allow_mock: bool | None = None,
    return_report: bool = False,
) -> list[Sample] | tuple[list[Sample], LoadReport]:
    """بارگذاری نمونه‌ها با پشتیبانی dev/eval split.

    Args:
        limit: حداکثر تعداد نمونه. None یا 0 یعنی کل.
        dataset: "en" | "fa" | مسیر فایل
        path: مسیر صریح برای دیتاست فارسی (یا هر فایل محلی)
        split: None | "dev" | "eval" — با manifest قابل‌تکرار؛ "eval" فقط
            رکوردهای manifest را برمی‌گرداند (جلوگیری از نشت داده).
        dedup: "keep_first" | "keep_all" (پیش‌فرض: config)
        sampling: "first" | "random" (پیش‌فرض: config)
        seed: seed نمونه‌گیری تصادفی (پیش‌فرض: config.SAMPLING_SEED)
        language_filter: "fa" | "en" | None (پیش‌فرض: config)
        return_report: اگر True، ``(samples, LoadReport)`` برمی‌گردد.
    """
    limit = _normalize_limit(limit)

    ds = (dataset or "en").strip()
    # مسیر فایل به‌عنوان dataset
    maybe_path = Path(ds)
    pass_through = {
        "dedup": dedup,
        "sampling": sampling,
        "seed": seed,
        "language_filter": language_filter,
        "text_mode": text_mode,
        "allow_mock": allow_mock,
    }
    if path is not None or maybe_path.suffix.lower() in {
        ".jsonl", ".csv", ".tsv", ".json"
    } or maybe_path.exists():
        result = load_samples_fa(
            path=path or ds,
            limit=limit,
            return_report=True,
            **pass_through,
        )
    else:
        key = ds.lower()
        if key in {"en", "english"}:
            result = load_samples_en(
                limit=limit,
                return_report=True,
                **pass_through,
            )
        elif key in {"fa", "persian", "farsi"}:
            result = load_samples_fa(
                path=path,
                limit=limit,
                return_report=True,
                **pass_through,
            )
        else:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Use en|fa or a path to jsonl/csv."
            )

    samples, report = result

    if split is not None:
        split_key = str(split).lower().strip()
        if split_key not in {"dev", "eval"}:
            raise ValueError(f"split must be None, 'dev' or 'eval', got {split!r}")
        samples, split_stats = dv.filter_by_split(
            samples, split_key, manifest_path=config.SPLIT_MANIFEST_PATH
        )
        report.split = split_key
        report.split_stats = split_stats
        privacy_log.log_event(logger, "split_filtered", **split_stats)

    return (samples, report) if return_report else samples


if __name__ == "__main__":
    # Privacy-safe demo: fingerprints instead of raw conversation text.
    demo, rep = load_samples(limit=5, dataset="en", return_report=True)
    for x in demo:
        print("-" * 60)
        print(
            f"idx={x.idx} len={len(x.text)} "
            f"hash={privacy_log.fingerprint(x.text)['sha256']} "
            f"lang={x.meta.get('lang')}"
        )
    print(rep.to_dict())
