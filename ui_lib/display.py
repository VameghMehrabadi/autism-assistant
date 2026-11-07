"""Result rendering helpers for Streamlit pages."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from facts import CATEGORIES


# Issue #10 — conversation-preview truncation (privacy affordance, NOT a
# security boundary against screenshots, browser inspection, or authorized
# users). Previews are truncated by default; revealing one record is an
# explicit, local action.
PREVIEW_MAX_CHARS = 120
PREVIEW_MARKER = " … [+{n} chars — reveal to view]"


def preview_text(text: object, max_chars: int = PREVIEW_MAX_CHARS,
                 reveal: bool = False) -> str:
    """Truncated single-value preview; full text only on explicit reveal."""
    value = "" if text is None else str(text)
    if reveal or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + PREVIEW_MARKER.format(
        n=len(value) - max_chars
    )


def preview_dataframe(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("text",),
    max_chars: int = PREVIEW_MAX_CHARS,
) -> pd.DataFrame:
    """Copy of ``df`` with the given text columns truncated for display.

    The underlying data (and any exported file) is never modified — only
    what the table shows.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: preview_text(v, max_chars=max_chars)
            )
    return out


def render_record_reveal(
    df: pd.DataFrame, text_col: str = "text", key_prefix: str = "reveal"
) -> None:
    """Explicit, local per-record reveal of the full conversation text.

    One record at a time, behind an explicit button. The revealed content
    is rendered straight from the dataframe on each rerun and is never
    written into session state, so it is not retained by unrelated
    components.
    """
    if df.empty or text_col not in df.columns:
        return
    st.caption(
        "Previews above are truncated by default — a privacy affordance, "
        "not a security boundary. Reveal one record at a time:"
    )
    labels = [
        f"row {pos} (idx={df.iloc[pos].get('idx', pos)})"
        for pos in range(len(df))
    ]
    choice = st.selectbox(
        "Record to inspect", options=range(len(labels)),
        format_func=lambda pos: labels[pos], key=f"{key_prefix}_record",
    )
    if st.button("Reveal full text for this record",
                 key=f"{key_prefix}_button"):
        st.text(preview_text(df.iloc[choice][text_col], reveal=True))


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
            "Scores are cosine similarities in [-1, 1] — not probabilities."
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
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Cat")[["Top-1", "Multi"]])


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
