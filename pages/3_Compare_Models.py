"""Launch the five-way embedding comparison from the UI."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from compare_runs import (
    build_comparison_table,
    plot_comparison,
    render_comparison_markdown,
    selected_runs,
    summarize_run,
)
from main import run_pipeline
from ui_lib.display import render_gap_summary
from ui_lib.widgets import persian_dataset_ready, render_limit_input

st.set_page_config(page_title="Compare Models", layout="wide")
st.title("Compare Models")
st.caption(
    "Five runs: EN/FA facts × Persian dataset × MiniLM / bge-m3 / e5-large."
)

st.dataframe(
    pd.DataFrame(config.COMPARISON_RUNS)[
        ["id", "title", "fact_lang", "dataset", "model_key"]
    ],
    use_container_width=True,
    hide_index=True,
)

with st.sidebar:
    st.header("Comparison options")
    limit = render_limit_input(key="cmp_limit", default=50)
    fa_default = str(config.PERSIAN_DATASET_PATH)
    sample = str(
        config.ROOT / "tests" / "fixtures" / "mentalchat16k_fa.sample.jsonl"
    )
    path_choice = st.selectbox(
        "Persian dataset",
        options=["production", "sample", "custom"],
        format_func={
            "production": "data/mentalchat16k_fa.jsonl",
            "sample": "Tiny fixture (smoke test)",
            "custom": "Custom path",
        }.__getitem__,
    )
    if path_choice == "production":
        fa_path = fa_default
    elif path_choice == "sample":
        fa_path = sample
    else:
        fa_path = st.text_input("Custom FA path", value=fa_default)

    only = st.multiselect(
        "Runs to execute",
        options=list(range(1, len(config.COMPARISON_RUNS) + 1)),
        default=list(range(1, len(config.COMPARISON_RUNS) + 1)),
        format_func=lambda i: f"{i}. {config.COMPARISON_RUNS[i - 1]['id']}",
    )
    skip_existing = st.checkbox("Skip runs with existing outputs", value=False)

if not persian_dataset_ready(fa_path):
    st.error(
        f"Persian dataset not found: `{fa_path}`. "
        "Export + translate first, or choose the tiny sample fixture."
    )
    st.stop()

if st.button("Run selected comparisons", type="primary"):
    if not only:
        st.warning("Select at least one run.")
        st.stop()

    runs = selected_runs(",".join(str(i) for i in only))
    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    progress = st.progress(0.0, text="Starting…")

    for i, run_cfg in enumerate(runs):
        progress.progress(
            i / len(runs),
            text=f"Running {run_cfg['id']} ({run_cfg['title']})",
        )
        run_dir = config.COMPARISON_DIR / run_cfg["id"]
        report_path = run_dir / "gap_analysis.json"

        if skip_existing and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            try:
                report = run_pipeline(
                    limit=int(limit),
                    model_key=run_cfg["model_key"],
                    fact_lang=run_cfg["fact_lang"],
                    dataset="fa",
                    dataset_path=fa_path,
                    output_dir=run_dir,
                    run_meta={"id": run_cfg["id"], "title": run_cfg["title"]},
                )
            except Exception as e:
                st.error(f"Failed on {run_cfg['id']}: {e}")
                st.exception(e)
                st.stop()
        summaries.append(summarize_run(report, run_cfg))
        with st.expander(f"Result · {run_cfg['id']}", expanded=False):
            render_gap_summary(report)

    progress.progress(1.0, text="Writing comparison report…")
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

    st.success(f"Comparison complete → `{config.COMPARISON_DIR}`")
    st.dataframe(table, use_container_width=True)
    if chart_path.exists():
        st.image(str(chart_path))
    st.markdown(md_path.read_text(encoding="utf-8"))
