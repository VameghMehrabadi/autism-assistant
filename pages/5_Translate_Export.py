"""Export English records and prepare Persian translation workflow."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import config
from data_loader import load_samples_en
from export_for_translation import write_export
from license_guard import (
    ASSESSMENT_REPORT,
    LICENSE_STATUS,
    LicenseGuardError,
    classify_destination,
)
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
limit = render_limit_input(key="tr_limit", default=0)
out_path = Path(
    st.text_input("Export path", value=str(out_default), key="tr_out")
)
st.caption(
    "Default: the **full dataset**. A limited export is marked INCOMPLETE in "
    "its sidecar metadata and must not be treated as the full dataset."
)

# Issue #13 — centralized licensing guard: paths outside the project
# workspace are public distribution destinations and need an explicit
# acknowledgment while the upstream license status is unclear.
public_destination = classify_destination(out_path) == "public"
acknowledge_license = False
if public_destination:
    st.warning(
        f"`{out_path}` is **outside the project workspace** — a public "
        "distribution destination. MentalChat16K mixes synthetic "
        "conversations with paraphrased clinical-trial interview "
        "transcripts; the upstream MIT tag does not by itself authorize "
        f"derivative redistribution (status: `{LICENSE_STATUS}`, see "
        f"`{ASSESSMENT_REPORT}`)."
    )
    acknowledge_license = st.checkbox(
        "I acknowledge the unresolved licensing risk "
        "(--acknowledge-license; this is NOT legal permission)"
    )

if st.button("Export for translation", type="primary"):
    try:
        lim = int(limit)
        samples, load_report = load_samples_en(
            limit=None if lim <= 0 else lim, return_report=True
        )
        meta = write_export(
            samples,
            out_path,
            dataset_total=load_report.dataset_total,
            load_report=load_report,
            acknowledge_license=acknowledge_license,
        )
        if meta["complete"]:
            st.success(
                f"COMPLETE export: {meta['exported']} of {meta['total_source']} "
                f"source records → `{out_path}`"
            )
        else:
            st.warning(
                f"INCOMPLETE export: {meta['exported']} of "
                f"{meta['total_source']} source records → `{out_path}`. "
                "This file must NOT be treated as the full dataset."
            )
    except LicenseGuardError as e:
        st.error(str(e))
    except Exception as e:
        st.exception(e)

st.subheader("3 · Persian dataset status")
meta_path = out_default.with_suffix(".export_meta.json")
if meta_path.exists():
    try:
        export_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.json(export_meta)
        if export_meta.get("complete"):
            st.info("✅ Export is COMPLETE (all source records included).")
        else:
            st.warning("⚠️ Export is INCOMPLETE — records are missing.")
    except json.JSONDecodeError:
        st.error(f"Unreadable export metadata: {meta_path}")

summary_path = config.DATA_DIR / "translation_summary.json"
if summary_path.exists():
    try:
        translation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with st.expander("Translation status (from the engine)", expanded=False):
            st.json(translation_summary)
        completeness = translation_summary.get("completeness") or {}
        if completeness.get("complete"):
            st.success("✅ Translation complete: every source record translated.")
        else:
            st.info(
                f"Translation progress: translated={completeness.get('translated')} "
                f"missing={completeness.get('missing')} "
                f"(total={completeness.get('total_source')})"
            )
    except json.JSONDecodeError:
        st.warning(f"Unreadable translation summary: {summary_path}")
else:
    st.caption("No translation has been run yet (no engine summary found).")

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
