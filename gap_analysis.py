"""تحلیل شکاف (Gap Analysis): کدام دسته‌های اوتیسم پوشش قوی/ضعیف دارند؟

خروجی:
- توزیع تعداد و درصد هر دسته (top-1).
- میانگین/میانه/انحراف معیار امتیاز top-1 برای هر دسته.
- نسبت نمونه‌های «بااعتماد» (top_score >= threshold).
- رتبه‌بندی قوی/ضعیف و توصیه برای فاز بعدی (داده‌ی مکمل).
- نمودار میله‌ای توزیع + نمودار میانگین امتیاز.
- گزارش Markdown و JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # بدون GUI / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from facts import CATEGORIES, CATEGORY_KEYS, FACTS
from labeler import parse_labels

# fact_id → category, for mapping matched prototype identifiers to categories
_FACT_CATEGORY: dict[int, str] = {f.id: f.category for f in FACTS}


def _category_label(cat: str) -> str:
    return f"{cat}. {CATEGORIES[cat]['title_en']}"


def _label_lists(df: pd.DataFrame) -> list[list[str]]:
    """Accepted labels per sample.

    Older files without a labels column fall back to top-1. A present but
    empty labels cell stays empty, so an uncertain top category is not counted
    as covered.
    """
    if "labels" not in df.columns:
        tops = (
            df["top_label"].astype(str).tolist()
            if "top_label" in df.columns
            else [""] * len(df)
        )
        return [[] if top in {"", "NONE"} else [top] for top in tops]
    return [parse_labels(raw) for raw in df["labels"]]


def build_report(df: pd.DataFrame, scores: np.ndarray,
                 threshold: float = config.LABEL_THRESHOLD) -> dict:
    """ساختن دیکشنری گزارش از DataFrame لیبل‌شده."""
    n = len(df)
    cat_keys = CATEGORY_KEYS

    # امتیاز top-1 هر نمونه و دسته‌ی top
    top_scores = df["top_score"].to_numpy(dtype=float)
    top_cats = df["top_label"].to_numpy()
    multi_lists = _label_lists(df)
    global_means = np.mean(scores, axis=0) if n else np.zeros(len(cat_keys))

    per_cat: dict[str, dict] = {}
    for ci, c in enumerate(cat_keys):
        mask = top_cats == c
        cnt = int(mask.sum())
        sc = top_scores[mask] if cnt else np.array([])
        multi_cnt = sum(1 for labs in multi_lists if c in labs)
        per_cat[c] = {
            "title": CATEGORIES[c]["title_en"],
            "title_fa": CATEGORIES[c]["title_fa"],
            "count": cnt,
            "share": round(cnt / n, 4) if n else 0.0,
            "multi_count": multi_cnt,
            "multi_share": round(multi_cnt / n, 4) if n else 0.0,
            "mean_score": round(float(sc.mean()), 4) if cnt else 0.0,
            "median_score": round(float(np.median(sc)), 4) if cnt else 0.0,
            "std_score": round(float(sc.std()), 4) if cnt else 0.0,
            "confident_share": (
                round(float((sc >= threshold).mean()), 4) if cnt else 0.0
            ),
            "global_mean_score": round(float(global_means[ci]), 4),
        }

    # میانگین امتیاز هر دسته روی کل نمونه‌ها (بدون شرط top-1) — نشان‌دهنده‌ی
    # میزان هم‌خوانی کلی داده‌ها با آن دسته.
    global_cat_score = {
        cat: per_cat[cat]["global_mean_score"] for cat in cat_keys
    }

    # --- Phase-3 signals: explicit uncertainty + prototype evidence -------
    # Old labeled.csv files may lack these columns; every addition is
    # optional and None when the data is absent.
    n_uncertain: int | None = None
    uncertain_rate: float | None = None
    margin_stats: dict[str, float] | None = None
    if "decision" in df.columns:
        decisions = df["decision"].astype(str)
        n_uncertain = int((decisions == "uncertain").sum())
        uncertain_rate = round(n_uncertain / n, 4) if n else 0.0
    if "margin" in df.columns:
        margins = pd.to_numeric(df["margin"], errors="coerce").dropna()
        if len(margins):
            margin_stats = {
                "mean": round(float(margins.mean()), 4),
                "median": round(float(margins.median()), 4),
                "min": round(float(margins.min()), 4),
            }
    matched_counts: dict[str, int] = {c: 0 for c in cat_keys}
    if "matched_fact_ids" in df.columns:
        for raw in df["matched_fact_ids"]:
            if not isinstance(raw, str) or not raw.strip():
                continue  # NaN / empty cells
            for part in raw.split(";"):
                fact_id = part.strip().split(":")[0]  # entries look like "12:fa"
                if fact_id.isdigit() and int(fact_id) in _FACT_CATEGORY:
                    matched_counts[_FACT_CATEGORY[int(fact_id)]] += 1
    for c in cat_keys:
        per_cat[c]["matched_count"] = matched_counts[c]

    # رتبه‌بندی ضعیف→قوی: top-1، سپس پوشش multi-label، سپس شباهت سراسری
    ranking = sorted(
        cat_keys,
        key=lambda c: (
            per_cat[c]["count"],
            per_cat[c]["multi_count"],
            per_cat[c]["global_mean_score"],
        ),
    )

    n_confident = int((top_scores >= threshold).sum())
    n_none = int((df["top_label"] == "NONE").sum()) if "NONE" in set(top_cats) else 0

    report = {
        "n_samples": n,
        "threshold": threshold,
        "n_confident": n_confident,
        "confident_rate": round(n_confident / n, 4) if n else 0.0,
        "n_unlabeled_NONE": n_none,
        "per_category": per_cat,
        "global_mean_score_per_category": global_cat_score,
        "ranking_weak_to_strong": ranking,
        "weakest": ranking[:2],
        "strongest": ranking[-2:],
        "n_uncertain": n_uncertain,
        "uncertain_rate": uncertain_rate,
        "margin_stats": margin_stats,
    }
    return report


def _recommendations(report: dict) -> list[str]:
    recs: list[str] = []
    for c in report["weakest"]:
        pc = report["per_category"][c]
        recs.append(
            f"- **{c}. {pc['title']}**: پوشش ضعیف "
            f"(top-1={pc['count']}, multi={pc.get('multi_count', 0)}, "
            f"share={pc['share']:.1%}, global_mean={pc.get('global_mean_score', 0)}). "
            f"برای فاز after-tuning/RAG داده‌ی مکمل از دیتاست‌های پزشکی/ASD بعدی "
            f"برای این دسته اضافه شود."
        )
    return recs


def render_markdown(report: dict) -> str:
    pc = report["per_category"]
    lines = [
        "# Gap Analysis — MentalChat16K × Autism Facts (A–F)",
        "",
        f"- تعداد نمونه‌ها / samples: **{report['n_samples']}**",
        f"- آستانه‌ی اعتماد / threshold: **{report['threshold']}**",
        f"- نمونه‌های بااعتماد / confident: "
        f"**{report['n_confident']} ({report['confident_rate']:.1%})**",
        "",
        "## Distribution per category (top-1 and multi-label)",
        "",
        "| Cat | Title (EN) | Top-1 | Share | Multi | Multi% | Mean top-score | Global mean |",
        "|-----|------------|-------|-------|-------|--------|----------------|-------------|",
    ]
    for c in CATEGORY_KEYS:
        d = pc[c]
        mean_top = f"{d['mean_score']}" if d["count"] else "—"
        lines.append(
            f"| {c} | {d['title']} | {d['count']} | {d['share']:.1%} | "
            f"{d.get('multi_count', 0)} | {d.get('multi_share', 0):.1%} | "
            f"{mean_top} | {d.get('global_mean_score', 0)} |"
        )
    lines += [
        "",
        "## Global mean similarity per category (over all samples)",
        "",
    ]
    for c in CATEGORY_KEYS:
        lines.append(
            f"- {c}. {pc[c]['title']}: {report['global_mean_score_per_category'][c]}"
        )
    lines += [
        "",
        "## Ranking (weak → strong)",
        "",
        " → ".join(report["ranking_weak_to_strong"]),
        "",
        "## Uncertainty & prototype evidence",
        "",
    ]
    if report.get("n_uncertain") is not None:
        lines.append(
            f"- Explicit uncertain decisions: **{report['n_uncertain']}** "
            f"({(report.get('uncertain_rate') or 0):.1%} of samples)"
        )
    if report.get("margin_stats"):
        ms = report["margin_stats"]
        lines.append(
            f"- Top-1 vs top-2 similarity margin — mean **{ms['mean']}**, "
            f"median **{ms['median']}**, min **{ms['min']}** "
            "(cosine units, an ambiguity heuristic — not a calibrated confidence)"
        )
    matched_total = sum(pc[c].get("matched_count", 0) for c in CATEGORY_KEYS)
    if matched_total:
        lines.append(
            "- Matched-prototype evidence per category "
            "(accepted labels + diagnostic top-1):"
        )
        for c in CATEGORY_KEYS:
            lines.append(f"  - {c}. {pc[c]['title']}: {pc[c].get('matched_count', 0)}")
    if report.get("n_uncertain") is None and not matched_total:
        lines.append(
            "- (labeled.csv predates the Phase-3 result schema — no signal columns)"
        )
    lines += _recommendations(report)
    lines += [
        "",
        "## Strongest categories",
        "",
        f"- **{report['strongest'][-1]}**: {pc[report['strongest'][-1]]['title']}",
        f"- **{report['strongest'][-2]}**: {pc[report['strongest'][-2]]['title']}",
        "",
    ]
    return "\n".join(lines)


def plot_distribution(report: dict, out_path: Path) -> None:
    cats = CATEGORY_KEYS
    counts = [report["per_category"][c]["count"] for c in cats]
    multi = [report["per_category"][c].get("multi_count", 0) for c in cats]
    means = [report["per_category"][c].get("global_mean_score", 0.0) for c in cats]
    labels = [_category_label(c) for c in cats]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(cats))
    width = 0.38
    bars = ax1.bar(
        x - width / 2, counts, width, color="#4C72B0", alpha=0.85, label="Top-1"
    )
    bars_m = ax1.bar(
        x + width / 2, multi, width, color="#55A868", alpha=0.75, label="Multi-label"
    )
    ax1.set_ylabel("Count of samples")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for b, cnt in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 str(cnt), ha="center", va="bottom", fontsize=8)
    for b, cnt in zip(bars_m, multi):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 str(cnt), ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, means, "-o", color="#C44E52", label="Global mean similarity")
    ax2.set_ylabel("Global mean cosine similarity", color="#C44E52")
    ax2.set_ylim(0, max(0.6, max(means) + 0.05) if means else 0.6)

    handles, legends = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + h2, legends + l2, loc="upper right", fontsize=8)
    plt.title("MentalChat16K — autism category distribution & similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[gap_analysis] Chart saved -> {out_path}")


def run(
    df: pd.DataFrame,
    scores: np.ndarray,
    output_dir: Path | None = None,
    run_meta: dict | None = None,
) -> dict:
    """اجرای gap analysis و ذخیره‌ی گزارش‌ها.

    Args:
        output_dir: اگر داده شود، خروجی‌ها آنجا نوشته می‌شوند
                    (برای مقایسه‌ی چندمدلی). در غیر این صورت config پیش‌فرض.
        run_meta: متادیتای اختیاری (model, fact_lang, ...) که به JSON اضافه می‌شود.
    """
    report = build_report(df, scores)
    if run_meta:
        report["run"] = run_meta

    if output_dir is None:
        md_path = config.GAP_REPORT_MD
        json_path = config.GAP_REPORT_JSON
        chart_path = config.GAP_CHART_PATH
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "gap_analysis.md"
        json_path = output_dir / "gap_analysis.json"
        chart_path = output_dir / "gap_distribution.png"

    title_extra = ""
    if run_meta:
        title_extra = (
            f"\n\n- run_id: **{run_meta.get('id', '')}**\n"
            f"- model: **{run_meta.get('model_name', run_meta.get('model_key', ''))}**\n"
            f"- fact_lang: **{run_meta.get('fact_lang', '')}**\n"
            f"- dataset: **{run_meta.get('dataset', '')}**\n"
        )
    md = render_markdown(report)
    if title_extra:
        # insert metadata after the first heading block
        parts = md.split("\n", 2)
        if len(parts) >= 3:
            md = parts[0] + "\n" + title_extra + parts[2]
        else:
            md = md + title_extra

    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_distribution(report, chart_path)
    print(f"[gap_analysis] Reports saved -> {md_path}, {json_path}")
    return report
