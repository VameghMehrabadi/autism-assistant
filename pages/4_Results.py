"""Browse saved pipeline / comparison outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from ui_lib.display import list_output_runs, render_gap_summary, show_file_if_exists

st.set_page_config(page_title="Results", layout="wide")
st.title("Results")
st.caption("Inspect labeled CSVs, gap reports, and comparison summaries.")

tab_single, tab_compare = st.tabs(["Single runs", "Comparisons"])

with tab_single:
    runs = list_output_runs(config.OUTPUT_DIR)
    # exclude the comparison root container itself when listing children? 
    # OUTPUT_DIR contains both flat files and subdirs including comparison/
    run_dirs = [p for p in runs if p.name != "comparison"]
    if not run_dirs and (config.OUTPUT_DIR / "gap_analysis.json").exists():
        run_dirs = [config.OUTPUT_DIR]

    if not run_dirs:
        st.info("No run outputs yet. Use **Run Pipeline** first.")
    else:
        labels = [str(p.relative_to(config.ROOT)) for p in run_dirs]
        choice = st.selectbox("Output folder", labels)
        folder = config.ROOT / choice

        meta_path = folder / "run_meta.json"
        gap_json = folder / "gap_analysis.json"
        labeled = folder / "labeled.csv"
        chart = folder / "gap_distribution.png"

        if meta_path.exists():
            st.json(json.loads(meta_path.read_text(encoding="utf-8")))
        if gap_json.exists():
            report = json.loads(gap_json.read_text(encoding="utf-8"))
            render_gap_summary(report)
        if chart.exists():
            st.image(str(chart))
        if labeled.exists():
            st.subheader("Labeled samples")
            df = pd.read_csv(labeled)
            st.dataframe(df.head(100), use_container_width=True)
            st.download_button(
                "Download labeled.csv",
                data=labeled.read_bytes(),
                file_name=labeled.name,
                mime="text/csv",
            )
        with st.expander("Markdown report"):
            show_file_if_exists(folder / "gap_analysis.md", kind="markdown")

with tab_compare:
    cmp_dir = config.COMPARISON_DIR
    summary_csv = cmp_dir / "comparison_summary.csv"
    summary_md = cmp_dir / "comparison_report.md"
    summary_png = cmp_dir / "comparison_shares.png"

    if not cmp_dir.exists():
        st.info("No comparison outputs yet. Use **Compare Models**.")
    else:
        if summary_csv.exists():
            st.dataframe(
                pd.read_csv(summary_csv), use_container_width=True
            )
        if summary_png.exists():
            st.image(str(summary_png))
        if summary_md.exists():
            st.markdown(summary_md.read_text(encoding="utf-8"))

        sub = list_output_runs(cmp_dir)
        if sub:
            st.subheader("Per-run folders")
            for p in sub:
                st.write(f"- `{p.name}`")
