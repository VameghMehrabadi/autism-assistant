"""Run a full single-configuration labeling + gap analysis."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from main import run_pipeline
from ui_lib.display import render_gap_summary, show_file_if_exists
from ui_lib.widgets import (
    persian_dataset_ready,
    render_dataset_selector,
    render_fact_lang_selector,
    render_limit_input,
    render_model_selector,
)

st.set_page_config(page_title="Run Pipeline", layout="wide")
st.title("Run Pipeline")
st.caption("One model × one fact language × one dataset → labeled CSV + gap report.")

with st.sidebar:
    st.header("Run config")
    model_key = render_model_selector(key="rp_model")
    fact_lang = render_fact_lang_selector(key="rp_fact", default="en")
    dataset, dataset_path = render_dataset_selector(key_prefix="rp")
    limit = render_limit_input(key="rp_limit", default=100)
    out_name = st.text_input(
        "Output folder name",
        value=f"{model_key}_{fact_lang}_{Path(dataset_path or dataset).stem}",
        key="rp_out",
    )

out_dir = config.OUTPUT_DIR / out_name.strip()
st.write("**Will write to:**", f"`{out_dir}`")

if dataset != "en" and dataset_path and not persian_dataset_ready(dataset_path):
    st.warning(
        f"Persian dataset missing: `{dataset_path}`. "
        "Use **Translate / Export** or pick the tiny sample fixture."
    )

run = st.button("Start pipeline", type="primary")
if run:
    try:
        with st.spinner("Running labeling + gap analysis (may download models)…"):
            report = run_pipeline(
                limit=int(limit),
                model_key=model_key,
                fact_lang=fact_lang,
                dataset=dataset if dataset == "en" else "fa",
                dataset_path=dataset_path,
                output_dir=out_dir,
                run_meta={"source": "streamlit"},
            )
        st.success(f"Done → `{out_dir}`")
        render_gap_summary(report)
        chart = out_dir / "gap_distribution.png"
        if chart.exists():
            st.image(str(chart))
        with st.expander("Markdown report"):
            show_file_if_exists(out_dir / "gap_analysis.md", kind="markdown")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.exception(e)
