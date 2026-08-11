"""Export English records and prepare Persian translation workflow."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from data_loader import load_samples_en
from ui_lib.widgets import persian_dataset_ready, render_limit_input

st.set_page_config(page_title="Translate / Export", layout="wide")
st.title("Translate / Export")
st.caption(
    "Export English MentalChat16K utterances, then translate with GPT-4+ "
    "using the semantic Persian prompt."
)

prompt_path = config.PROMPTS_DIR / "semantic_persian_translation.txt"
out_default = config.DATA_DIR / "mentalchat16k_en_for_translation.jsonl"

st.subheader("1 · Translation prompt")
if prompt_path.exists():
    st.code(prompt_path.read_text(encoding="utf-8"), language="text")
else:
    st.error(f"Missing prompt file: {prompt_path}")

st.subheader("2 · Export English records")
limit = render_limit_input(key="tr_limit", default=500)
out_path = Path(
    st.text_input("Export path", value=str(out_default), key="tr_out")
)

if st.button("Export for translation", type="primary"):
    try:
        lim = int(limit)
        samples = load_samples_en(limit=None if lim <= 0 else lim)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                rec = {"idx": s.idx, "text": s.text, **s.meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st.success(f"Wrote {len(samples)} records → `{out_path}`")
    except Exception as e:
        st.exception(e)

st.subheader("3 · Save Persian dataset")
st.markdown(
    f"""
After translation (GPT-4+), save JSONL as:

`{config.PERSIAN_DATASET_PATH}`

Each line:

```json
{{"idx": 0, "text": "…Persian utterance…"}}
```
"""
)

status = (
    "✅ Persian dataset found"
    if persian_dataset_ready()
    else "⏳ Persian dataset not found yet"
)
st.info(status)

sample = config.ROOT / "tests" / "fixtures" / "mentalchat16k_fa.sample.jsonl"
if sample.exists():
    st.write("Smoke-test fixture available:", f"`{sample}`")
