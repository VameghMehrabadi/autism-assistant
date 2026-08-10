"""Result rendering helpers for Streamlit pages."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from facts import CATEGORIES


def build_score_table(score_vector: dict[str, float]) -> pd.DataFrame:
    rows = []
    for cat, score in sorted(score_vector.items(), key=lambda x: x[1], reverse=True):
        title = CATEGORIES.get(cat, {}).get("title_en", cat)
        rows.append({"Category": cat, "Title": title, "Score": round(float(score), 4)})
    return pd.DataFrame(rows)


def render_label_result(result: dict) -> None:
    top_label = result["top_label"]
    title = CATEGORIES.get(top_label, {}).get("title_en", top_label)
    left, right = st.columns([2, 3])
    with left:
        st.metric("Top category", f"{top_label} — {title}", delta=f"{result['top_score']:.4f}")
        st.markdown("**Labels:** " + ", ".join(result["labels"]))
        st.write("✅ Confident" if result["confident"] else "⚠️ Below confidence threshold")
    with right:
        score_df = build_score_table(result["score_vector"])
        st.bar_chart(score_df.set_index("Category")["Score"])
    st.dataframe(score_df, use_container_width=True, hide_index=True)


def render_gap_summary(report: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", report.get("n_samples", 0))
    c2.metric("Confident", f"{report.get('confident_rate', 0):.1%}")
    ranking = report.get("ranking_weak_to_strong", [])
    c3.metric("Weak → Strong", " → ".join(ranking) if ranking else "—")

    pc = report.get("per_category", {})
    rows = []
    for cat, data in pc.items():
        rows.append({
            "Cat": cat,
            "Title": data.get("title", ""),
            "Count": data.get("count", 0),
            "Share": data.get("share", 0),
            "Mean score": data.get("mean_score", 0),
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Cat")["Count"])


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
