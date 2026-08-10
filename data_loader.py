"""بارگذاری دیتاست MentalChat16K (انگلیسی) یا نسخه‌ی فارسی ترجمه‌شده.

- en: ShenLab/MentalChat16K از Hugging Face
- fa: فایل محلی JSONL/CSV در config.PERSIAN_DATASET_PATH (یا مسیر سفارشی)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset

import config


@dataclass
class Sample:
    idx: int
    text: str
    meta: dict


def _merge_fields(row: dict, fields: tuple[str, ...]) -> str:
    """ادغام فیلدهای متنی دیتاست در یک رشته‌ی تمیز."""
    parts: list[str] = []
    for f in fields:
        val = row.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(f"{f}: {val.strip()}")
    return "\n".join(parts) if parts else ""


def _row_to_text(row: dict, fields: tuple[str, ...]) -> str:
    """اگر ستون text موجود باشد همان را بگیر؛ وگرنه فیلدها را ادغام کن."""
    direct = row.get("text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return _merge_fields(row, fields)


def load_samples_en(limit: int | None = None) -> list[Sample]:
    """بارگذاری MentalChat16K انگلیسی از Hugging Face."""
    if limit is not None and limit <= 0:
        limit = None

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
    print(f"[data_loader] Using text fields: {fields}")

    n = len(ds) if limit is None else min(limit, len(ds))
    samples: list[Sample] = []
    for i in range(n):
        row = ds[i]
        text = _merge_fields(row, fields)
        if not text:
            continue
        samples.append(Sample(idx=i, text=text, meta={k: row[k] for k in fields}))

    print(f"[data_loader] Prepared {len(samples)} English samples.")
    return samples


def load_samples_fa(
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[Sample]:
    """بارگذاری دیتاست فارسی از JSONL یا CSV.

    قالب‌های پذیرفته‌شده:
    - JSONL: هر خط یک آبجکت با فیلد text یا instruction/input/output
    - CSV: ستون text یا همان فیلدهای MentalChat16K
    """
    if limit is not None and limit <= 0:
        limit = None

    fa_path = Path(path) if path else config.PERSIAN_DATASET_PATH
    if not fa_path.exists():
        raise FileNotFoundError(
            f"Persian dataset not found: {fa_path}\n"
            "Export English records with `python export_for_translation.py`, "
            "translate them (GPT-4 + prompts/semantic_persian_translation.txt), "
            "then save as data/mentalchat16k_fa.jsonl."
        )

    print(f"[data_loader] Loading Persian dataset from {fa_path} ...")
    suffix = fa_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict] = []
        with fa_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(fa_path, sep=sep, encoding="utf-8")
        rows = df.to_dict(orient="records")
    elif suffix == ".json":
        payload = json.loads(fa_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "data" in payload:
            rows = payload["data"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported JSON structure in {fa_path}")
    else:
        raise ValueError(
            f"Unsupported Persian dataset format '{suffix}'. Use .jsonl/.csv/.json"
        )

    fields = config.DATASET_TEXT_FIELDS
    samples: list[Sample] = []
    for i, row in enumerate(rows):
        if limit is not None and len(samples) >= limit:
            break
        text = _row_to_text(row, fields)
        if not text:
            continue
        idx = int(row["idx"]) if "idx" in row and str(row["idx"]).isdigit() else i
        meta = {k: row.get(k) for k in fields if k in row}
        if "text" in row:
            meta["text"] = row["text"]
        samples.append(Sample(idx=idx, text=text, meta=meta))

    print(f"[data_loader] Prepared {len(samples)} Persian samples.")
    return samples


def load_samples(
    limit: int | None = None,
    dataset: str = "en",
    path: str | Path | None = None,
) -> list[Sample]:
    """بارگذاری نمونهها.

    Args:
        limit: حداکثر تعداد نمونه. None یا 0 یعنی کل.
        dataset: "en" | "fa" | مسیر فایل
        path: مسیر صریح برای دیتاست فارسی (یا هر فایل محلی)
    """
    if limit is not None and limit <= 0:
        limit = None

    ds = (dataset or "en").strip()
    # مسیر فایل به‌عنوان dataset
    maybe_path = Path(ds)
    if path is not None or maybe_path.suffix.lower() in {
        ".jsonl", ".csv", ".tsv", ".json"
    } or maybe_path.exists():
        return load_samples_fa(path=path or ds, limit=limit)

    key = ds.lower()
    if key in {"en", "english"}:
        return load_samples_en(limit=limit)
    if key in {"fa", "persian", "farsi"}:
        return load_samples_fa(path=path, limit=limit)
    raise ValueError(
        f"Unknown dataset '{dataset}'. Use en|fa or a path to jsonl/csv."
    )


if __name__ == "__main__":
    s = load_samples(limit=5, dataset="en")
    for x in s:
        print("-" * 60)
        print(x.text[:300])
