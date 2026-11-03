"""Export English MentalChat16K records for Persian translation (Phase 6).

Usage:
    python export_for_translation.py --limit 500
    python export_for_translation.py --limit 0   # full dataset

Writes:
    data/mentalchat16k_en_for_translation.jsonl

Each record preserves the original ordering and identifiers:
    {"idx": <original id>,
     "text": <patient-view label text>,
     "instruction": <original EN>, "input": <original EN>, "output": <original EN>}

Translate each record's fields with the engine
(``python translate_dataset.py``) or manually with
``prompts/semantic_persian_translation.txt``. The translated output keeps
English originals under ``*_en`` keys and places Persian text under the
plain keys so ``data_loader.load_samples_fa`` consumes it natively.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from data_loader import load_samples_en

SOURCE_FIELDS: tuple[str, ...] = ("instruction", "input", "output")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EN records for FA translation.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--out",
        type=str,
        default=str(config.DATA_DIR / "mentalchat16k_en_for_translation.jsonl"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    limit = args.limit if args.limit is not None else config.SAMPLE_LIMIT
    if limit is not None and limit <= 0:
        limit = None

    samples, load_report = load_samples_en(limit=limit, return_report=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            rec: dict = {"idx": s.idx, "text": s.text}
            for field_name in SOURCE_FIELDS:
                value = s.meta.get(field_name, "")
                rec[field_name] = value if isinstance(value, str) else ""
            if not any(rec[f].strip() for f in SOURCE_FIELDS):
                n_empty += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[export] Wrote {len(samples)} records -> {out_path}")
    if n_empty:
        print(f"[export] WARNING: {n_empty} record(s) have no text fields (kept verbatim)")
    print(
        f"[export] load report: kept={load_report.kept} "
        f"duplicates_removed={load_report.duplicates_removed} "
        f"skipped_invalid={load_report.skipped_invalid}"
    )
    print(
        f"[export] completed_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    print(
        "[export] Next: python translate_dataset.py "
        "(or translate manually with prompts/semantic_persian_translation.txt)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
