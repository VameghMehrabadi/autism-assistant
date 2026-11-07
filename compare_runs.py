"""اجرای ماتریس ۵ پیکربندی و ساخت گزارش مقایسه‌ای.

پیکربندی‌ها (config.COMPARISON_RUNS) — همه از فکت فارسی استفاده می‌کنند:
1. FA facts × EN dataset × MiniLM   (rerun of task 1; facts may have changed)
2. FA facts × EN dataset × bge-m3
3. FA facts × EN dataset × multilingual-e5-large
4. FA facts × FA dataset × bge-m3   (after GPT-4+ translation)
5. FA facts × FA dataset × multilingual-e5-large

اجرا:
    python compare_runs.py --limit 100
    python compare_runs.py --limit 0          # کل دیتاست
    python compare_runs.py --only 1,2         # فقط بعضی runها
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import alignment
from facts import CATEGORY_KEYS
from main import run_pipeline


def _reconfigure_stdio_utf8() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_stdio_utf8()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-way embedding comparison.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample limit per run (0 = all). Default: config.SAMPLE_LIMIT",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override Persian dataset path",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated run numbers 1-5, e.g. 1,2,3",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its gap_analysis.json already exists",
    )
    p.add_argument(
        "--verify-alignment",
        action="store_true",
        help="Read-only verification that persisted runs used equivalent "
             "records, thresholds, and configurations; prints the alignment "
             "report and exits",
    )
    p.add_argument(
        "--migrate-legacy-alignment",
        action="store_true",
        help="Issue #9: write alignment_legacy.json (retroactive, "
             "artifact-derived evidence) for historical runs that lack "
             "execution-time alignment metadata. Idempotent; never touches "
             "labeled.csv or run_meta.json; migrated runs remain classified "
             "'legacy-partial', never execution-time verified",
    )
    return p.parse_args()


def selected_runs(only: str | None) -> list[dict]:
    """Parse `--only 1,2,3` into comparison run configs."""
    runs = list(config.COMPARISON_RUNS)
    if not only:
        return runs
    wanted = {int(x.strip()) for x in only.split(",") if x.strip()}
    selected = []
    for i, run in enumerate(runs, start=1):
        if i in wanted:
            selected.append(run)
    if not selected:
        raise ValueError(f"No runs matched --only={only!r}")
    return selected


# Back-compat alias
_selected_runs = selected_runs


def summarize_run(report: dict, run_cfg: dict) -> dict:
    pc = report["per_category"]
    run_meta = report.get("run") or {}
    top_score_stats = report.get("top_score_stats")
    return {
        "id": run_cfg["id"],
        "title": run_cfg["title"],
        "model_key": run_cfg["model_key"],
        "fact_lang": run_cfg["fact_lang"],
        "dataset": run_cfg["dataset"],
        "n_samples": report["n_samples"],
        "confident_rate": report["confident_rate"],
        "threshold": report.get("threshold"),
        "top_score_stats": top_score_stats,
        "score_scale_note": (
            # Cosine scales are model-dependent: identical thresholds are not
            # comparable across encoders, and higher scores do not indicate a
            # better model.
            "cosine scale is model-dependent; thresholds are not comparable "
            "across models"
        ),
        "text_window": run_meta.get("text_window"),
        "embed_prefix": run_meta.get("prefix"),
        "labeler_version": run_meta.get("labeler_version"),
        "dataset_fingerprint": run_meta.get("dataset_fingerprint"),
        "mean_top_score": round(
            float(np.mean([pc[c]["mean_score"] for c in CATEGORY_KEYS
                           if pc[c]["count"] > 0] or [0.0])),
            4,
        ),
        "ranking_weak_to_strong": report["ranking_weak_to_strong"],
        "counts": {c: pc[c]["count"] for c in CATEGORY_KEYS},
        "shares": {c: pc[c]["share"] for c in CATEGORY_KEYS},
        "multi_counts": {c: pc[c].get("multi_count", 0) for c in CATEGORY_KEYS},
        "multi_shares": {c: pc[c].get("multi_share", 0.0) for c in CATEGORY_KEYS},
        "mean_scores": {c: pc[c]["mean_score"] for c in CATEGORY_KEYS},
        "global_mean_score_per_category": report["global_mean_score_per_category"],
    }


# Back-compat alias
_summarize_run = summarize_run


def build_comparison_table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        row = {
            "run": s["id"],
            "title": s["title"],
            "model": s["model_key"],
            "fact_lang": s["fact_lang"],
            "n_samples": s["n_samples"],
            "confident_rate": s["confident_rate"],
            "mean_top_score": s["mean_top_score"],
            "weakest": " > ".join(s["ranking_weak_to_strong"][:2]),
            "strongest": " > ".join(s["ranking_weak_to_strong"][-2:]),
        }
        for c in CATEGORY_KEYS:
            row[f"count_{c}"] = s["counts"][c]
            row[f"share_{c}"] = s["shares"][c]
            row[f"multi_{c}"] = s.get("multi_counts", {}).get(c, 0)
            row[f"mean_{c}"] = s["mean_scores"][c]
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison_markdown(summaries: list[dict], table: pd.DataFrame) -> str:
    lines = [
        "# Five-way embedding comparison",
        "",
        "Persian MentalChat16K × autism facts (A–G).",
        "",
        "## Runs",
        "",
        "| # | ID | Facts | Dataset | Model | Confident% | Mean top-score | Weak→Strong |",
        "|---|----|-------|---------|-------|------------|----------------|-------------|",
    ]
    for i, s in enumerate(summaries, start=1):
        lines.append(
            f"| {i} | `{s['id']}` | {s['fact_lang']} | {s['dataset']} | "
            f"{s['model_key']} | {s['confident_rate']:.1%} | {s['mean_top_score']} | "
            f"{' → '.join(s['ranking_weak_to_strong'])} |"
        )

    # Issue #9 — per-run alignment classification. Legacy (historical) runs
    # are never presented as execution-time verified; each non-modern run
    # carries its reason and an actionable rerun instruction.
    lines += [
        "",
        "## Run alignment",
        "",
    ]
    for i, s in enumerate(summaries, start=1):
        info = s.get("alignment")
        if not info:
            lines.append(
                f"- `{s['id']}`: alignment status not recorded in this "
                "report."
            )
            continue
        classification = info.get("classification", "unknown")
        if classification == "modern":
            lines.append(
                f"- `{s['id']}`: **modern** — execution-time alignment "
                "metadata present (dataset fingerprint, text window, "
                "preprocessing version, threshold, dataset total)."
            )
        else:
            reason = info.get("reason") or "no alignment metadata available"
            lines.append(
                f"- `{s['id']}`: **{classification}** — {reason}. This run "
                "is NOT execution-time verified. Rerun for fully "
                "verifiable results: `python compare_runs.py --only "
                f"{i} --limit 0`."
            )

    lines += [
        "",
        "## Category share (top-1 %)",
        "",
        "| Run | " + " | ".join(CATEGORY_KEYS) + " |",
        "|-----|" + "|".join(["------"] * len(CATEGORY_KEYS)) + "|",
    ]
    for s in summaries:
        shares = " | ".join(f"{s['shares'][c]:.1%}" for c in CATEGORY_KEYS)
        lines.append(f"| `{s['id']}` | {shares} |")

    lines += [
        "",
        "## Category multi-label share (%)",
        "",
        "| Run | " + " | ".join(CATEGORY_KEYS) + " |",
        "|-----|" + "|".join(["------"] * len(CATEGORY_KEYS)) + "|",
    ]
    for s in summaries:
        multi = s.get("multi_shares") or {
            c: 0.0 for c in CATEGORY_KEYS
        }
        shares = " | ".join(f"{multi[c]:.1%}" for c in CATEGORY_KEYS)
        lines.append(f"| `{s['id']}` | {shares} |")

    lines += [
        "",
        "## Category global mean similarity",
        "",
        "| Run | " + " | ".join(CATEGORY_KEYS) + " |",
        "|-----|" + "|".join(["------"] * len(CATEGORY_KEYS)) + "|",
    ]
    for s in summaries:
        gmeans = s.get("global_mean_score_per_category") or s["mean_scores"]
        means = " | ".join(str(gmeans[c]) for c in CATEGORY_KEYS)
        lines.append(f"| `{s['id']}` | {means} |")

    lines += [
        "",
        "## Notes for interpretation",
        "",
        "- Runs 1–3 use **Persian facts × English dataset** to compare three encoders "
        "on the original MentalChat16K (task 1 rerun + two stronger multilingual models).",
        "- Runs 4–5 use **Persian facts × Persian dataset** after GPT-4+ translation "
        "(matched-language condition).",
        "- Labeling uses **patient-only** text by default (skips the repeated "
        "counselor prompt and therapy reply that used to collapse almost all "
        "samples onto category F).",
        "- **Thresholds are model-dependent**: all runs apply cosine threshold "
        "0.35, but cosine score scales differ per encoder (e.g. e5-large "
        "concentrates near 0.80). Confident-rate differences across models "
        "partly reflect score-scale differences, **not** better or worse "
        "model quality — higher cosine scores do not prove a better model.",
        "- **Effective text windows differ per model** (MiniLM 128 tokens by "
        "default; bge-m3 512; e5-large 512): long conversations truncate "
        "differently — part of each model's configured preprocessing, "
        "recorded per run in `run_meta.json` (`text_window`).",
        "- **Cosine similarity is not a probability and not clinical "
        "confidence.** Calibration against human-labeled data is required "
        "before any threshold carries operational meaning "
        "(see `data/gold/README.md`).",
        "- Prefer models whose category ranking is stable and whose confident_rate "
        "is not dominated by a single category unless that matches domain priors.",
        "",
        f"Full numeric table also saved as `comparison_summary.csv` "
        f"({len(table)} rows).",
        "",
    ]
    return "\n".join(lines)


def plot_comparison(summaries: list[dict], out_path: Path) -> None:
    if not summaries:
        return
    n_runs = len(summaries)
    n_cats = len(CATEGORY_KEYS)
    x = np.arange(n_cats)
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, s in enumerate(summaries):
        shares = [s["shares"][c] * 100 for c in CATEGORY_KEYS]
        ax.bar(x + i * width, shares, width=width, label=s["id"])
    ax.set_xticks(x + width * (n_runs - 1) / 2)
    ax.set_xticklabels(CATEGORY_KEYS)
    ax.set_ylabel("Top-1 share (%)")
    ax.set_title("Category distribution across comparison runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[compare] Chart saved -> {out_path}")


def verify_alignment(run_ids: list[str] | None = None) -> dict:
    """Verify that persisted comparison runs are built on equivalent inputs.

    Checks, per run and across runs (read-only):
    - identical labeled-record ID sets (same records, none missing)
    - identical label-text content fingerprint (same preprocessing output)
    - identical label definitions period (facts.py unchanged across runs)
    - identical decision threshold
    - effective text window + embedding prefix per run (alignment metadata)

    Returns a content-free verification report. Historical runs predating the
    alignment metadata report ``null`` for the new fields — absence is
    reported honestly, never backfilled.
    """
    import pandas as pd

    runs = run_ids or [r["id"] for r in config.COMPARISON_RUNS]
    base = config.COMPARISON_DIR
    report: dict[str, Any] = {
        "runs": [],
        "all_records_aligned": None,
    }
    ref_idx: frozenset | None = None
    ref_text_hash: str | None = None
    ref_threshold = None
    for run_id in runs:
        run_dir = base / run_id
        entry: dict[str, Any] = {"id": run_id, "dir": str(run_dir)}
        labeled_path = run_dir / "labeled.csv"
        meta_path = run_dir / "run_meta.json"
        gap_path = run_dir / "gap_analysis.json"
        if not (labeled_path.exists() and gap_path.exists()):
            entry["status"] = "MISSING_ARTIFACTS"
            report["runs"].append(entry)
            continue
        df = pd.read_csv(labeled_path)
        idx_set = frozenset(pd.to_numeric(df["idx"], errors="coerce").dropna().astype(int))
        # Issue #9: one canonical fingerprint implementation (alignment.py)
        text_hash = alignment.labeled_text_fingerprint(df)
        gap, gap_error = alignment.read_json(gap_path)
        if gap is not None and not isinstance(gap, dict):
            gap, gap_error = None, "gap_analysis.json is not a JSON object"
        meta, meta_error = alignment.read_json(meta_path)
        if meta is not None and not isinstance(meta, dict):
            meta, meta_error = None, "run_meta.json is not a JSON object"
        entry["alignment"] = alignment.classify_alignment(run_dir, meta)
        gap_threshold = gap.get("threshold") if isinstance(gap, dict) else None
        entry.update(
            {
                "status": "ok",
                "n_records": len(df),
                "records_match_reference": None,  # set below
                "text_fingerprint": text_hash,
                "threshold": gap_threshold,
                "model": (meta or {}).get("model_key") or run_id,
                "fact_lang": (meta or {}).get("fact_lang"),
                "labeler_version": (meta or {}).get("labeler_version"),
                "text_window": (meta or {}).get(
                    "text_window", "pre-alignment-metadata"
                ),
                "embed_prefix": (meta or {}).get("prefix"),
                "dataset_fingerprint": (meta or {}).get("dataset_fingerprint"),
            }
        )
        if gap_error:
            entry["gap_report_error"] = gap_error
        if isinstance(meta, dict) and meta_error:
            entry["run_meta_error"] = meta_error
        if isinstance(gap, dict):
            counts_sum = sum(
                gap["per_category"][c]["count"] for c in gap["per_category"]
            )
            entry["category_counts_sum"] = counts_sum
            entry["missing_records_vs_total"] = (
                ((meta or {}).get("dataset_total") or counts_sum) - counts_sum
            )
        else:
            entry["category_counts_sum"] = None
            entry["missing_records_vs_total"] = None
        # Issue #9: surface the retroactive artifact's derived fingerprint
        # under an explicitly derived name — never as execution-time metadata.
        artifact = alignment.read_json(
            run_dir / alignment.LEGACY_ARTIFACT_NAME
        )[0]
        if isinstance(artifact, dict):
            evidence = artifact.get("evidence", {})
            entry["dataset_fingerprint_derived"] = evidence.get(
                "ordered_input_fingerprint_derived"
            )
        if ref_idx is None:
            ref_idx, ref_text_hash, ref_threshold = (
                idx_set, text_hash, gap_threshold
            )
            entry["records_match_reference"] = True
        else:
            entry["records_match_reference"] = idx_set == ref_idx
            entry["text_fingerprint_match"] = text_hash == ref_text_hash
            entry["threshold_match"] = (
                gap_threshold == ref_threshold
                if gap_threshold is not None and ref_threshold is not None
                else None
            )
        report["runs"].append(entry)
    record_checks = [
        r.get("records_match_reference") for r in report["runs"] if r.get("status") == "ok"
    ]
    report["all_records_aligned"] = all(record_checks) if record_checks else None
    report["n_runs_missing_artifacts"] = sum(
        1 for r in report["runs"] if r.get("status") == "MISSING_ARTIFACTS"
    )
    # Issue #9: per-classification counts — legacy runs are never
    # counted as verified.
    summary: dict[str, int] = {}
    for r in report["runs"]:
        classification = (r.get("alignment") or {}).get(
            "classification", r.get("status", "unavailable")
        )
        summary[classification] = summary.get(classification, 0) + 1
    report["alignment_summary"] = summary
    return report


def main() -> int:
    args = parse_args()
    if args.verify_alignment:
        report = verify_alignment()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["all_records_aligned"] is False:
            return 4
        return 0

    if args.migrate_legacy_alignment:
        run_ids = (
            [r["id"] for r in selected_runs(args.only)]
            if args.only
            else [r["id"] for r in config.COMPARISON_RUNS]
        )
        run_dirs = [
            config.COMPARISON_DIR / run_id
            for run_id in run_ids
            if (config.COMPARISON_DIR / run_id / "labeled.csv").exists()
        ]
        summary = alignment.migrate_legacy_alignment(run_dirs)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["failed"]:
            return 4
        print(
            "[compare] Retroactive artifacts written. These runs remain "
            "classified 'legacy-partial' — they are NOT execution-time "
            "verified. For fully verifiable results rerun with current "
            "metadata: python compare_runs.py --limit 0"
        )
        return 0

    limit = args.limit if args.limit is not None else config.SAMPLE_LIMIT
    runs = selected_runs(args.only)

    fa_path = Path(args.dataset_path) if args.dataset_path else config.PERSIAN_DATASET_PATH
    needs_fa = any(r["dataset"] == "fa" for r in runs)
    if needs_fa and not fa_path.exists():
        print(
            f"[compare] ERROR: Persian dataset missing: {fa_path}\n"
            "1) python export_for_translation.py --limit 500\n"
            "2) Translate with prompts/semantic_persian_translation.txt (GPT-4+)\n"
            "3) Save as data/mentalchat16k_fa.jsonl\n"
            "Note: runs 1-3 use the English dataset and do not need this file.",
            file=sys.stderr,
        )
        return 2

    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for run_cfg in runs:
        run_dir = config.COMPARISON_DIR / run_cfg["id"]
        report_path = run_dir / "gap_analysis.json"
        print("\n" + "#" * 72)
        print(f"# {run_cfg['title']}")
        print("#" * 72)

        if args.skip_existing and report_path.exists():
            print(f"[compare] Skipping existing run -> {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            ds = run_cfg["dataset"]
            ds_path = fa_path if ds == "fa" else None
            report = run_pipeline(
                limit=limit,
                model_key=run_cfg["model_key"],
                fact_lang=run_cfg["fact_lang"],
                dataset=ds,
                dataset_path=ds_path,
                output_dir=run_dir,
                run_meta={
                    "id": run_cfg["id"],
                    "title": run_cfg["title"],
                },
            )
        summaries.append(summarize_run(report, run_cfg))
        # Issue #9 — attach the run's alignment classification so the
        # comparison report can distinguish modern runs from historical
        # (legacy) ones. Metadata-only; never raises on malformed meta.
        summaries[-1]["alignment"] = alignment.classify_alignment(run_dir)

    table = build_comparison_table(summaries)
    csv_path = config.COMPARISON_DIR / "comparison_summary.csv"
    json_path = config.COMPARISON_DIR / "comparison_summary.json"
    md_path = config.COMPARISON_DIR / "comparison_report.md"
    chart_path = config.COMPARISON_DIR / "comparison_shares.png"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(
        render_comparison_markdown(summaries, table), encoding="utf-8"
    )
    plot_comparison(summaries, chart_path)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Report: {md_path}")
    print(f"Table:  {csv_path}")
    print(f"Chart:  {chart_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
