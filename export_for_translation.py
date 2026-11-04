"""Export English MentalChat16K records for Persian translation (Phase 6).

Usage:
    python export_for_translation.py --limit 500
    python export_for_translation.py --limit 0   # full dataset (default: full)

Writes:
    data/mentalchat16k_en_for_translation.jsonl
    data/mentalchat16k_en_for_translation.export_meta.json   ← completeness metadata

Each record preserves the original ordering and identifiers:
    {"idx": <original id>,
     "text": <patient-view label text>,
     "instruction": <original EN>, "input": <original EN>, "output": <original EN>}

Translate each record's fields with the engine
(``python translate_dataset.py``) or manually with
``prompts/semantic_persian_translation.txt``. The translated output keeps
English originals under ``*_en`` keys and places Persian text under the
plain keys so ``data_loader.load_samples_fa`` consumes it natively.

Export safety: a sidecar ``*.export_meta.json`` records ``total_source``,
``exported`` and ``complete`` so an incomplete (limited) export can never be
mistaken for a full-dataset export. Incomplete exports print a loud warning.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from data_loader import LoadReport, Sample, load_samples_en

SOURCE_FIELDS: tuple[str, ...] = ("instruction", "input", "output")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EN records for FA translation.")
    p.add_argument("--limit", type=int, default=None,
                   help="maximum records to export; 0 or omitted = full dataset")
    p.add_argument(
        "--out",
        type=str,
        default=str(config.DATA_DIR / "mentalchat16k_en_for_translation.jsonl"),
    )
    return p.parse_args(argv)


def build_export_record(sample: Sample) -> dict[str, Any]:
    """One export record: original identifier + text fields (English kept)."""
    rec: dict[str, Any] = {"idx": sample.idx, "text": sample.text}
    for field_name in SOURCE_FIELDS:
        value = sample.meta.get(field_name, "")
        rec[field_name] = value if isinstance(value, str) else ""
    return rec


def write_export(
    samples: list[Sample],
    out_path: str | Path,
    *,
    dataset_total: int | None = None,
    load_report: LoadReport | None = None,
) -> dict[str, Any]:
    """Write the JSONL export plus the completeness sidecar; returns the meta.

    ``dataset_total`` is the number of source records available **before**
    any limit (from the loader's ``LoadReport.dataset_total``). ``complete``
    is True only when every source record was exported.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            rec = build_export_record(s)
            if not any(rec[field].strip() for field in SOURCE_FIELDS):
                n_empty += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    complete = dataset_total is not None and len(samples) >= dataset_total
    meta: dict[str, Any] = {
        "export_file": str(out_path),
        "total_source": dataset_total,
        "exported": len(samples),
        "complete": complete,
        "empty_field_records": n_empty,
        "duplicates_removed": load_report.duplicates_removed if load_report else None,
        "skipped_invalid": load_report.skipped_invalid if load_report else None,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = out_path.with_suffix(".export_meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[export] Wrote {len(samples)} records -> {out_path}")
    if complete:
        print(f"[export] COMPLETE export: {len(samples)} of {dataset_total} source records.")
    else:
        print(
            f"[export] WARNING: INCOMPLETE export — {len(samples)} of "
            f"{dataset_total} source records. This file must NOT be treated as "
            f"the full dataset (see {meta_path.name})."
        )
    if n_empty:
        print(f"[export] WARNING: {n_empty} record(s) have no text fields (kept verbatim)")
    if load_report is not None:
        print(
            f"[export] load report: kept={load_report.kept} "
            f"duplicates_removed={load_report.duplicates_removed} "
            f"skipped_invalid={load_report.skipped_invalid}"
        )
    print(f"[export] meta -> {meta_path}")
    print(
        "[export] Next: python translate_dataset.py "
        "(or translate manually with prompts/semantic_persian_translation.txt)"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    limit = args.limit if args.limit is not None else 0  # default: full dataset
    if limit is not None and limit <= 0:
        limit = None

    samples, load_report = load_samples_en(limit=limit, return_report=True)
    meta = write_export(
        samples,
        args.out,
        dataset_total=load_report.dataset_total,
        load_report=load_report,
    )
    if not meta["complete"]:
        return 3  # distinguishable exit code for incomplete exports
    return 0


if __name__ == "__main__":
    sys.exit(main())
