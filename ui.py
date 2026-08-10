"""Autism Assistant — Streamlit entrypoint.

Launch (from project root):
    streamlit run ui.py
    # or
    .\\launch_ui.ps1
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from ui_lib.widgets import render_category_legend

st.set_page_config(
    page_title="Autism Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Autism Assistant")
st.caption(
    "Semantic labeling & gap analysis for MentalChat16K × autism facts (A–G). "
    "Modular pipeline with a browser UI to run, compare, and inspect results."
)

st.markdown(
    """
### Launch checklist
1. Activate the venv and install deps (`pip install -r requirements.txt`).
2. Open this UI with `streamlit run ui.py` or `launch_ui.ps1`.
3. Use the **pages in the sidebar** to run each stage.
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("1 · Quick Label")
    st.write("Paste text or pick a sample and score categories interactively.")
with c2:
    st.subheader("2 · Run Pipeline")
    st.write("Full labeling + gap analysis for one model / fact-language setup.")
with c3:
    st.subheader("3 · Compare Models")
    st.write("Run the five-way MiniLM / bge-m3 / e5-large comparison matrix.")

st.markdown("---")
left, right = st.columns([3, 2])
with left:
    st.subheader("Project modules")
    st.markdown(
        """
| Module | Role |
|--------|------|
| `data_loader.py` | EN / FA dataset loading |
| `facts.py` | Autism prototypes A–G |
| `labeler.py` | Embeddings + cosine labeling |
| `gap_analysis.py` | Reports & charts |
| `main.py` | Single-run orchestrator |
| `compare_runs.py` | 5-way comparison |
| `export_for_translation.py` | EN → FA export helper |
| `ui.py` + `pages/` | Browser launcher UI |
| `ui_lib/` | Shared UI helpers |
"""
    )
with right:
    st.subheader("Current defaults")
    st.write("**Default model:**", config.EMBED_MODEL_KEY)
    st.write("**HF id:**", config.EMBED_MODEL_NAME)
    st.write("**Threshold:**", config.LABEL_THRESHOLD)
    st.write("**Outputs:**", str(config.OUTPUT_DIR))
    fa = config.PERSIAN_DATASET_PATH
    st.write(
        "**Persian dataset:**",
        "✅ ready" if fa.exists() else "⏳ missing — use Translate / Export page",
    )

st.markdown("---")
st.subheader("Categories A–G")
render_category_legend()

st.info(
    "Tip: keep heavy model downloads on GPU if available. "
    "For a smoke test, choose the tiny Persian sample fixture and limit=5."
)

# Ensure pages/ is discoverable next to this file
_pages = Path(__file__).resolve().parent / "pages"
if not _pages.exists():
    st.error("Missing `pages/` folder — multipage navigation will not work.")
