"""Export English MentalChat16K records for Persian translation (GPT-4+).

Usage:
    python export_for_translation.py --limit 500
    python export_for_translation.py --limit 0   # full dataset

Writes:
    data/mentalchat16k_en_for_translation.jsonl

Translate each record's `text` (or instruction/input/output) with
`prompts/semantic_persian_translation.txt`, then save the Persian version as
`data/mentalchat16k_fa.jsonl` with the same schema (keep `idx`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
from data_loader import load_samples_en


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

    samples = load_samples_en(limit=limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            rec = {
                "idx": s.idx,
                "text": s.text,
                **s.meta,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[export] Wrote {len(samples)} records -> {out_path}")
    print(
        "[export] Translate with prompts/semantic_persian_translation.txt "
        "then save FA file as data/mentalchat16k_fa.jsonl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
