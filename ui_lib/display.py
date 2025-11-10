"""Result rendering helpers for Streamlit pages."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from facts import CATEGORIES

# Phase-3 columns that may be absent in legacy labeled.csv files.
PHASE3_COLUMNS: tuple[str, ...] = ("decision", "margin", "matched_fact_ids")

_MISSING_PHASE3_FA = (
    "این اجرا فیلدهای Phase 3 مانند margin و decision را ندارد؛ "
    "بنابراین این مقادیر قابل گزارش نیستند."
)
_MISSING_PHASE3_EN = (
    "This run does not contain Phase 3 fields such as margin and decision. "
    "These metrics are unavailable for this run."
)


def detect_phase3_availability(df: pd.DataFrame) -> dict[str, bool]:
    """Check which Phase-3 columns exist in a labeled DataFrame.

    Returns a dict mapping each Phase-3 column name to ``True`` (present)
    or ``False`` (absent). A column that exists but contains all-null
    values is reported as present — the caller must distinguish
    "column absent" from "column present but value is null/zero".
    """
    return {col: col in df.columns for col in PHASE3_COLUMNS}


def phase3_missing_warning(df: pd.DataFrame) -> str | None:
    """Return a human-readable warning when Phase-3 columns are absent.

    Returns ``None`` when all Phase-3 columns are present.
    """
    availability = detect_phase3_availability(df)
    missing = [col for col, present in availability.items() if not present]
    if not missing:
        return None
    return (
        f"⚠️ {_MISSING_PHASE3_EN}\n\n"
        f"Missing columns: {', '.join(missing)}\n\n"
        f"{_MISSING_PHASE3_FA}"
    )


def build_score_table(score_vector: dict[str, float]) -> pd.DataFrame:
    rows = []
    for cat, score in sorted(score_vector.items(), key=lambda x: x[1], reverse=True):
        title = CATEGORIES.get(cat, {}).get("title_en", cat)
        rows.append({"Category": cat, "Title": title, "Score": round(float(score), 4)})
    return pd.DataFrame(rows)


def decision_badge(result: dict) -> str:
    """Human-readable decision with legacy-result fallback."""
    decision = result.get("decision")
    if decision is None:
        decision = "labeled" if result.get("confident") else "uncertain"
    return str(decision)


def matched_prototypes_table(result: dict) -> pd.DataFrame | None:
    """Prototype evidence rows for the matched prototypes of one result."""
    matched = result.get("matched_prototypes") or []
    if not matched:
        return None
    rows = [
        {
            "Category": m.get("category", ""),
            "Title": CATEGORIES.get(m.get("category", ""), {}).get("title_en", ""),
            "Fact ID": m.get("fact_id", ""),
            "Language": m.get("lang", ""),
            "Similarity": round(float(m.get("score", 0.0)), 4),
        }
        for m in matched
    ]
    return pd.DataFrame(rows)


def render_label_result(result: dict) -> None:
    top_label = result["top_label"]
    title = CATEGORIES.get(top_label, {}).get("title_en", top_label)
    decision = decision_badge(result)
    left, right = st.columns([2, 3])
    with left:
        st.metric("Top category", f"{top_label} — {title}", delta=f"{result['top_score']:.4f}")
        if decision == "uncertain":
            st.warning(
                "⚠️ Uncertain — below threshold. No label is accepted; "
                "the top category is diagnostic only."
            )
        else:
            st.success("✅ Labeled")
        accepted = ", ".join(result["labels"]) or "none (below threshold)"
        st.markdown("**Labels:** " + accepted)
        if result.get("margin") is not None:
            st.caption(
                f"Margin vs runner-up: {result['margin']:.4f} — an ambiguity "
                "heuristic, not calibrated confidence."
            )
        st.caption(
            "Scores are cosine similarities in [-1, 1] — not probabilities. "
            "Thresholds are encoder-specific and require human validation; "
            "the default 0.35 is uncalibrated."
        )
    with right:
        score_df = build_score_table(result["score_vector"])
        st.bar_chart(score_df.set_index("Category")["Score"])
    matched_df = matched_prototypes_table(result)
    if matched_df is not None:
        st.markdown("**Matched prototypes (evidence):**")
        st.dataframe(matched_df, use_container_width=True, hide_index=True)
    st.dataframe(score_df, use_container_width=True, hide_index=True)


def render_gap_summary(report: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", report.get("n_samples", 0))
    c2.metric("Confident", f"{report.get('confident_rate', 0):.1%}")
    ranking = report.get("ranking_weak_to_strong", [])
    c3.metric("Weak → Strong", " → ".join(ranking) if ranking else "—")

    # Issue #19: display Phase-3 metrics honestly — N/A when unavailable.
    n_uncertain = report.get("n_uncertain")
    margin_stats = report.get("margin_stats")
    if n_uncertain is not None:
        c1.metric("Explicitly uncertain", str(n_uncertain))
    else:
        c1.metric("Explicitly uncertain", "N/A",
                  help="Phase-3 result schema not present in this run's labeled.csv")
    if margin_stats is not None:
        c2.metric("Margin (median)", str(margin_stats.get("median", "N/A")))
    else:
        c2.metric("Margin (median)", "N/A",
                  help="Phase-3 result schema not present in this run's labeled.csv")

    pc = report.get("per_category", {})
    rows = []
    for cat, data in pc.items():
        rows.append({
            "Cat": cat,
            "Title": data.get("title", ""),
            "Top-1": data.get("count", 0),
            "Share": data.get("share", 0),
            "Multi": data.get("multi_count", 0),
            "Multi share": data.get("multi_share", 0),
            "Global mean": data.get("global_mean_score", data.get("mean_score", 0)),
            "Matched": (
                data.get("matched_count")
                if data.get("matched_count") is not None
                else "N/A"
            ),
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Cat")[["Top-1", "Multi"]])

    # Issue #19: show a warning when ANY Phase-3 metric is unavailable.
    # Partial schemas (e.g. matched_fact_ids present but decision absent) are
    # named explicitly so the user knows exactly what is missing.
    missing_p3 = []
    if n_uncertain is None:
        missing_p3.append("decision")
    if margin_stats is None:
        missing_p3.append("margin")
    matched_unavailable = any(
        data.get("matched_count") is None for data in pc.values()
    )
    if matched_unavailable:
        missing_p3.append("matched_fact_ids")
    if missing_p3:
        st.warning(
            f"⚠️ Phase-3 result fields not available in this run: "
            f"{', '.join(missing_p3)}. These metrics are unavailable, not zero.",
            icon="⚠️",
        )


def list_output_runs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def show_file_if_exists(path: Path, kind: str = "text") -> None:
    if not path.exists():
        st.info(f"Not found: `{path}`")
        return
    if kind == "image":
        st.image(str(path))
    elif kind == "markdown":
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.code(path.read_text(encoding="utf-8")[:8000])
