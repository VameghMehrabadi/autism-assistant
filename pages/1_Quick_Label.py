"""Interactive single-text semantic labeling."""
from __future__ import annotations

import streamlit as st

from ui_lib.caching import get_labeler, get_samples
from ui_lib.display import render_label_result
from ui_lib.widgets import (
    render_dataset_selector,
    render_fact_lang_selector,
    render_limit_input,
    render_model_selector,
)

st.set_page_config(page_title="Quick Label", layout="wide")
st.title("Quick Label")
st.caption("Score one utterance against autism fact categories.")

with st.sidebar:
    st.header("Settings")
    model_key = render_model_selector(key="ql_model")
    fact_lang = render_fact_lang_selector(key="ql_fact", default="both")
    dataset, dataset_path = render_dataset_selector(key_prefix="ql")
    limit = render_limit_input(key="ql_limit", default=50)

try:
    labeler = get_labeler(model_key, fact_lang)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

sample_text = ""
sample_meta = None
use_samples = st.checkbox("Pick from dataset samples", value=True)

if use_samples:
    try:
        samples = get_samples(dataset, int(limit) or 50, dataset_path)
    except FileNotFoundError as e:
        st.warning(str(e))
        samples = []
    except Exception as e:
        st.error(f"Dataset load failed: {e}")
        samples = []

    if samples:
        options = [
            f"#{s.idx}: {s.text[:100].replace(chr(10), ' ')}…" for s in samples
        ]
        choice = st.selectbox("Sample", options)
        idx = options.index(choice)
        sample_text = samples[idx].text
        sample_meta = samples[idx].meta

text = st.text_area("Text to label", value=sample_text, height=240)
if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Enter or select text first.")
    else:
        with st.spinner("Embedding & scoring…"):
            emb = labeler.embed_samples([text])
            scores = labeler.score_samples(emb)
            result = labeler.assign_labels(scores)[0]
        render_label_result(result)
        if sample_meta is not None:
            with st.expander("Sample metadata"):
                st.json(sample_meta)
