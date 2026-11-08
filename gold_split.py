"""تقسیم‌بندی امن مجموعه طلایی (Issue #8) — انتخاب ۲۰۰–۵۰۰ رکورد برای برچسب انسانی.

Builds the human-annotation candidate split from the real dataset using the
EXISTING Phase-2 grouping machinery (``data_validation``): records are
selected and partitioned **whole-group**, so related records can never leak
between the calibration and evaluation partitions, and the selection is
reproducible from the recorded seed.

Outputs (operator-local, under ``data/gold/``):

- ``gold_split_manifest.json`` — content-free selection metadata: seed,
  algorithm, dataset fingerprint, per-record partition assignment
  (``gold_calibration`` / ``gold_evaluation``), group keys, and the
  leakage-verification results. Safe to commit.
- ``gold_annotation_template.jsonl`` — one pending record per selected
  source record: stable ``gold_id``, ``source_idx``, the patient text to
  annotate, empty multi-label ``labels``, ``status: "pending"``, null
  ``labeler_id``, empty ``notes``, and full provenance. **Contains raw
  conversation text — gitignored, never committed.** Humans fill it; until
  then zero human labels exist.

Nothing here fabricates labels, marks anything human-reviewed, or touches
``gold_labels_v1.jsonl``.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import config
from data_validation import _item_group_key

GOLD_DIR = config.DATA_DIR / "gold"
MANIFEST_NAME = "gold_split_manifest.json"
TEMPLATE_NAME = "gold_annotation_template.jsonl"

DEFAULT_TARGET = 300          # within the 200–500 range and the gold README cap
CALIBRATION_FRACTION = 0.35   # whole groups, by design — never per record
CATEGORIES = ("A", "B", "C", "D", "E", "F", "G")
PARTITION_CALIBRATION = "gold_calibration"
PARTITION_EVALUATION = "gold_evaluation"


def dataset_file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_gold_split(
    items: list,
    *,
    target_size: int = DEFAULT_TARGET,
    seed: int | None = None,
    source_path: Path | None = None,
) -> dict:
    """Select ≤ ``target_size`` records whole-group and partition them.

    Deterministic: same items + same seed → identical selection and
    partitions. Group integrity and partition disjointness are verified
    before returning; the verification results travel in the manifest.
    """
    from data_validation import _item_group_key

    seed = config.SAMPLING_SEED if seed is None else seed
    if target_size < 200 or target_size > 500:
        raise ValueError(
            f"target_size must be within 200-500 (issue #8), got {target_size}"
        )

    hashes: list[str] = []
    for i, item in enumerate(items):
        h = item.meta.get("record_hash") if isinstance(item.meta, dict) else None
        if not h:
            raise ValueError(
                f"item at position {i} has no record_hash — load data "
                "through data_loader.load_samples(...) so provenance and "
                "grouping are attached."
            )
        hashes.append(h)

    group_keys = [_item_group_key(item) for item in items]
    groups: dict[str, list[int]] = {}
    for position, key in enumerate(group_keys):
        groups.setdefault(key, []).append(position)

    # 1) whole-group selection: seeded shuffle of unique groups, accumulate
    #    groups until the target is reached (never split a group)
    order = list(groups)
    random.Random(seed).shuffle(order)
    selected_positions: list[int] = []
    selected_groups: list[str] = []
    for key in order:
        if len(selected_positions) >= target_size:
            break
        selected_positions.extend(groups[key])
        selected_groups.append(key)

    # 2) whole-group partition of the selection: seeded shuffle, accumulate
    #    calibration groups up to CALIBRATION_FRACTION of selected records
    cal_order = list(selected_groups)
    random.Random(f"{seed}:calibration").shuffle(cal_order)
    calibration_target = round(len(selected_positions) * CALIBRATION_FRACTION)
    partition: dict[int, str] = {}
    cal_count = 0
    for key in cal_order:
        size = len(groups[key])
        if cal_count < calibration_target:
            for position in groups[key]:
                partition[position] = PARTITION_CALIBRATION
            cal_count += size
        else:
            for position in groups[key]:
                partition[position] = PARTITION_EVALUATION

    # 3) verification: group integrity + partition disjointness
    group_partitions: dict[str, set[str]] = {}
    for position in selected_positions:
        group_partitions.setdefault(group_keys[position], set()).add(
            partition[position]
        )
    leaked_groups = [
        key for key, parts in group_partitions.items() if len(parts) > 1
    ]
    cal_positions = {
        p for p in selected_positions if partition[p] == PARTITION_CALIBRATION
    }
    eval_positions = {
        p for p in selected_positions if partition[p] == PARTITION_EVALUATION
    }
    overlap = cal_positions & eval_positions
    record_hashes = [hashes[p] for p in selected_positions]
    duplicate_hashes = [
        h for h, count in Counter(record_hashes).items() if count > 1
    ]

    verification = {
        "n_selected": len(selected_positions),
        "n_groups": len(selected_groups),
        "groups_in_both_partitions": leaked_groups,
        "records_in_both_partitions": len(overlap),
        "duplicate_record_hashes": duplicate_hashes,
        "calibration_records": len(cal_positions),
        "evaluation_records": len(eval_positions),
        "leak_free": not leaked_groups and not overlap and not duplicate_hashes,
    }

    source_sha = (
        dataset_file_sha256(source_path) if source_path else None
    )
    manifest = {
        "schema": "gold-split-v1",
        "purpose": "human annotation candidate selection (Issue #8)",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "seed": seed,
        "target_size": target_size,
        "algorithm": (
            "whole-group selection via data_validation._item_group_key; "
            "seeded group shuffle; accumulate groups to target; whole-group "
            f"partition at {CALIBRATION_FRACTION:.0%} calibration fraction"
        ),
        "source_file": str(source_path) if source_path else None,
        "source_sha256": source_sha,
        "n_source_records": len(items),
        "partitions": {
            PARTITION_CALIBRATION: sorted(cal_positions),
            PARTITION_EVALUATION: sorted(eval_positions),
        },
        "record_hashes": {
            str(pos): hashes[pos] for pos in selected_positions
        },
        "group_keys": {
            str(pos): group_keys[pos] for pos in selected_positions
        },
        "verification": verification,
    }
    return manifest


def write_template(
    manifest: dict,
    items: list,
    out_path: Path,
    source_sha: str | None = None,
) -> Path:
    """Write the unfilled human-annotation template (status: pending).

    One record per selected source record. Contains the patient text —
    gitignored by design, never committed. Nothing is labeled here: this is
    a worklist for humans, not a gold dataset.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cal = set(manifest["partitions"][PARTITION_CALIBRATION])
    positions = sorted(
        manifest["partitions"][PARTITION_CALIBRATION]
        + manifest["partitions"][PARTITION_EVALUATION]
    )
    group_keys = manifest["group_keys"]
    generated = manifest["generated_utc"]
    with out_path.open("w", encoding="utf-8") as handle:
        for gold_number, position in enumerate(positions, start=1):
            item = items[position]
            handle.write(
                json.dumps(
                    {
                        "gold_id": f"gold-{gold_number:04d}",
                        "source_idx": item.idx,
                        "partition": (
                            PARTITION_CALIBRATION
                            if position in cal
                            else PARTITION_EVALUATION
                        ),
                        "text": item.text,
                        "labels": [],
                        "status": "pending",
                        "labeler_id": None,
                        "notes": "",
                        "provenance": {
                            "record_hash": manifest["record_hashes"][
                                str(position)
                            ],
                            "group_key": group_keys[str(position)],
                            "dataset_sha256": source_sha
                            or manifest.get("source_sha256"),
                            "selection_seed": manifest["seed"],
                            "generated_utc": generated,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def generate_gold_split(
    *,
    target_size: int = DEFAULT_TARGET,
    seed: int | None = None,
    source_path: Path | None = None,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    """End-to-end: load the real dataset, carve the split, write outputs.

    Returns a content-free summary (counts + verification only — the
    manifest and template carry the details). Never touches
    ``gold_labels_v1.jsonl``.
    """
    from data_loader import load_samples

    source_path = Path(
        source_path
        or config.DATA_DIR / "mentalchat16k_en_for_translation.jsonl"
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"source dataset not found: {source_path} — generate it with "
            "python export_for_translation.py first."
        )
    items = load_samples(path=str(source_path), limit=limit)
    out_dir = Path(out_dir) if out_dir else GOLD_DIR
    manifest = build_gold_split(
        items,
        target_size=target_size,
        seed=seed,
        source_path=source_path,
    )
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template_path = write_template(
        manifest, items, out_dir / TEMPLATE_NAME,
        source_sha=manifest.get("source_sha256"),
    )
    verification = manifest["verification"]
    return {
        "manifest": str(manifest_path),
        "template": str(template_path),
        "n_source_records": manifest["n_source_records"],
        "n_selected": verification["n_selected"],
        "n_groups": verification["n_groups"],
        "calibration_records": verification["calibration_records"],
        "evaluation_records": verification["evaluation_records"],
        "leak_free": verification["leak_free"],
        "template_status": "all records pending human annotation",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Issue #8: carve the human-annotation gold split (200-500 "
            "records, group-safe, reproducible) and write the unfilled "
            "annotation template."
        )
    )
    parser.add_argument("--size", type=int, default=DEFAULT_TARGET,
                        help="target records within 200-500 (default 300)")
    parser.add_argument("--seed", type=int, default=None,
                        help="selection seed (default: config.SAMPLING_SEED)")
    parser.add_argument("--source", type=str, default=None,
                        help="source dataset JSONL (default: the EN export)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: data/gold/)")
    args = parser.parse_args(argv)
    try:
        summary = generate_gold_split(
            target_size=args.size,
            seed=args.seed,
            source_path=Path(args.source) if args.source else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[gold-split] ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"[gold-split] selected {summary['n_selected']} records in "
        f"{summary['n_groups']} groups "
        f"(calibration={summary['calibration_records']}, "
        f"evaluation={summary['evaluation_records']}) "
        f"leak_free={summary['leak_free']}"
    )
    print(f"[gold-split] manifest: {summary['manifest']}")
    print(f"[gold-split] template: {summary['template']}")
    print(
        "[gold-split] all records are status=pending — NO human labels "
        "exist yet; annotate the template, then validate with "
        "python gold_eval.py"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
