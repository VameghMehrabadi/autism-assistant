"""Orchestrator کل pipeline:

    دانلود/بارگذاری دیتاست → امبدینگ prototypeها و نمونه‌ها → لیبل‌گذاری معنایی
    → ذخیره‌ی خروجی CSV + امتیازها → gap analysis + گزارش/نمودار

اجرا:
    python main.py
    python main.py --limit 100 --model minilm --fact-lang both --dataset en
    python main.py --model bge-m3 --fact-lang en --dataset fa
    python main.py --model e5-large --fact-lang fa --dataset fa --out outputs/run5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader import load_samples
from gap_analysis import run as run_gap_analysis
from labeler import SemanticLabeler, resolve_model


def _reconfigure_stdio_utf8() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_stdio_utf8()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic labeling + gap analysis.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="تعداد نمونه‌ها. 0 یعنی کل دیتاست. default = config.SAMPLE_LIMIT",
    )
    p.add_argument(
        "--model",
        type=str,
        default=config.EMBED_MODEL_KEY,
        choices=sorted(config.EMBED_MODELS.keys()),
        help="کلید مدل امبدینگ: minilm | bge-m3 | e5-large",
    )
    p.add_argument(
        "--fact-lang",
        type=str,
        default=config.FACT_LANG,
        choices=["en", "fa", "both"],
        help="زبان prototypeهای فکت",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="en",
        help="en | fa | یا مسیر فایل JSONL/CSV فارسی",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="مسیر صریح دیتاست فارسی (اختیاری)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="پوشه‌ی خروجی (پیش‌فرض: outputs/)",
    )
    return p.parse_args()


def run_pipeline(
    limit: int | None = None,
    model_key: str = config.EMBED_MODEL_KEY,
    fact_lang: str = config.FACT_LANG,
    dataset: str = "en",
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_meta: dict | None = None,
) -> dict:
    """اجرای یک پیکربندی کامل؛ گزارش gap analysis را برمی‌گرداند."""
    if limit is None:
        limit = config.SAMPLE_LIMIT
    if limit is not None and limit <= 0:
        limit = None

    out_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "labeled.csv"
    scores_path = out_dir / "scores.npy"
    meta_path = out_dir / "run_meta.json"

    model_key, model_name, prefix = resolve_model(model_key=model_key)
    meta = {
        "model_key": model_key,
        "model_name": model_name,
        "prefix": prefix,
        "fact_lang": fact_lang,
        "dataset": dataset,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "limit": limit,
        **(run_meta or {}),
    }

    # 1) بارگذاری داده
    samples = load_samples(limit=limit, dataset=dataset, path=dataset_path)
    if not samples:
        raise RuntimeError("No samples loaded.")

    texts = [s.text for s in samples]

    # 2) لیبل‌گذاری معنایی
    labeler = SemanticLabeler(model_key=model_key, fact_lang=fact_lang)
    embs = labeler.embed_samples(texts)
    scores = labeler.score_samples(embs)
    results = labeler.assign_labels(scores)

    # 3) ذخیره‌ی خروجی
    rows = []
    for s, r in zip(samples, results):
        rows.append({
            "idx": s.idx,
            "text": s.text,
            "top_label": r["top_label"],
            "top_score": r["top_score"],
            "labels": "|".join(r["labels"]),
            "confident": r["confident"],
            **{f"score_{k}": r["score_vector"][k] for k in r["score_vector"]},
        })
    df = pd.DataFrame(rows)
    df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    np.save(scores_path, scores)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[main] Labeled data saved -> {labeled_path}")
    print(f"[main] Score matrix saved -> {scores_path} (shape={scores.shape})")

    # 4) Gap analysis
    report = run_gap_analysis(df, scores, output_dir=out_dir, run_meta=meta)

    # 5) خلاصه
    print("\n" + "=" * 60)
    print("SUMMARY / خلاصه")
    print("=" * 60)
    print(
        f"Model: {model_name} | fact_lang={fact_lang} | dataset={dataset}"
    )
    print(
        f"Samples: {report['n_samples']} | "
        f"Confident: {report['n_confident']} ({report['confident_rate']:.1%})"
    )
    from facts import CATEGORY_KEYS

    print("\nCategory distribution (top-1):")
    for c in CATEGORY_KEYS:
        d = report["per_category"][c]
        print(
            f"  {c}. {d['title']:32s} count={d['count']:5d} "
            f"share={d['share']:.1%} mean={d['mean_score']}"
        )
    print(f"\nWeak  → Strong: {' > '.join(report['ranking_weak_to_strong'])}")
    print(f"\nOutputs: {out_dir}")
    return report


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(
            limit=args.limit,
            model_key=args.model,
            fact_lang=args.fact_lang,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            output_dir=args.out,
        )
    except FileNotFoundError as e:
        print(f"[main] ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[main] ERROR: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
