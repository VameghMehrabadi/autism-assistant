"""اجرای ماتریس ۵ پیکربندی و ساخت گزارش مقایسه‌ای.

پیکربندی‌ها (config.COMPARISON_RUNS):
1. EN facts × FA dataset × MiniLM
2. EN facts × FA dataset × bge-m3
3. EN facts × FA dataset × e5-large
4. FA facts × FA dataset × bge-m3
5. FA facts × FA dataset × e5-large

اجرا:
    python compare_runs.py --limit 100
    python compare_runs.py --limit 0          # کل دیتاست فارسی
    python compare_runs.py --only 1,2         # فقط بعضی runها
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
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
    return p.parse_args()


def _selected_runs(only: str | None) -> list[dict]:
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


def _summarize_run(report: dict, run_cfg: dict) -> dict:
    pc = report["per_category"]
    return {
        "id": run_cfg["id"],
        "title": run_cfg["title"],
        "model_key": run_cfg["model_key"],
        "fact_lang": run_cfg["fact_lang"],
        "dataset": run_cfg["dataset"],
        "n_samples": report["n_samples"],
        "confident_rate": report["confident_rate"],
        "mean_top_score": round(
            float(np.mean([pc[c]["mean_score"] for c in CATEGORY_KEYS
                           if pc[c]["count"] > 0] or [0.0])),
            4,
        ),
        "ranking_weak_to_strong": report["ranking_weak_to_strong"],
        "counts": {c: pc[c]["count"] for c in CATEGORY_KEYS},
        "shares": {c: pc[c]["share"] for c in CATEGORY_KEYS},
        "mean_scores": {c: pc[c]["mean_score"] for c in CATEGORY_KEYS},
        "global_mean_score_per_category": report["global_mean_score_per_category"],
    }


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
        "## Category mean top-1 score",
        "",
        "| Run | " + " | ".join(CATEGORY_KEYS) + " |",
        "|-----|" + "|".join(["------"] * len(CATEGORY_KEYS)) + "|",
    ]
    for s in summaries:
        means = " | ".join(str(s["mean_scores"][c]) for c in CATEGORY_KEYS)
        lines.append(f"| `{s['id']}` | {means} |")

    lines += [
        "",
        "## Notes for interpretation",
        "",
        "- Run 1 repeats the original MiniLM setup on the **Persian** dataset "
        "(facts may have changed; always rerun).",
        "- Runs 2–3 keep English facts and swap in stronger multilingual encoders.",
        "- Runs 4–5 use Persian facts with the same Persian dataset "
        "(matched-language condition).",
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


def main() -> int:
    args = parse_args()
    limit = args.limit if args.limit is not None else config.SAMPLE_LIMIT
    runs = _selected_runs(args.only)

    fa_path = Path(args.dataset_path) if args.dataset_path else config.PERSIAN_DATASET_PATH
    if not fa_path.exists():
        print(
            f"[compare] ERROR: Persian dataset missing: {fa_path}\n"
            "1) python export_for_translation.py --limit 500\n"
            "2) Translate with prompts/semantic_persian_translation.txt (GPT-4+)\n"
            "3) Save as data/mentalchat16k_fa.jsonl",
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
            report = run_pipeline(
                limit=limit,
                model_key=run_cfg["model_key"],
                fact_lang=run_cfg["fact_lang"],
                dataset="fa",
                dataset_path=fa_path,
                output_dir=run_dir,
                run_meta={
                    "id": run_cfg["id"],
                    "title": run_cfg["title"],
                },
            )
        summaries.append(_summarize_run(report, run_cfg))

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
