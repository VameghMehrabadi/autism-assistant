"""Browse saved pipeline / comparison outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from ui_lib.display import (
    list_output_runs,
    preview_dataframe,
    render_gap_summary,
    render_record_reveal,
    show_file_if_exists,
)

st.set_page_config(page_title="Results", layout="wide")
st.title("Results")
st.caption("Inspect labeled CSVs, gap reports, and comparison summaries.")
st.caption(
    "⚠️ Research use only — labels are similarity heuristics, not diagnoses."
)

tab_single, tab_compare, tab_retention = st.tabs(
    ["Single runs", "Comparisons", "Retention & cleanup"]
)

with tab_retention:
    # Issue #10 — explicit, whitelist-based run-output cleanup.
    # Audit fix (PR #38): the confirmation checkbox and Delete button must
    # NOT live inside the "Preview deletions" click guard — Streamlit drops
    # clicks on widgets that vanish on the next rerun, which made the UI
    # deletion unreachable. Instead: Preview snapshots the selected location
    # into session state (workflow state only — never conversation text);
    # the pending plan is REBUILT from that snapshot on every rerun with the
    # same candidate-selection logic, so a changed file set is displayed as
    # an updated scope and must be re-confirmed before deletion.
    from retention import (  # noqa: E402
        RetentionError,
        execute_run_cleanup,
        list_run_locations,
        plan_run_cleanup,
    )

    st.caption(
        "Default policy: temporary generated raw run artifacts are kept "
        "for **30 days** — a documented default, NOT an enforced "
        "mechanism. Deletion here is always explicit and manual; original "
        "datasets, other runs, caches, model weights and gold data are "
        "never touched."
    )

    last_result = st.session_state.get("cleanup_last_result")
    if last_result:
        st.success(
            f"Deleted {last_result['deleted']} file(s) "
            f"(skipped {last_result['skipped']}, failed "
            f"{last_result['failed']}). Metadata-only audit record "
            "written to outputs/logs/deletion_audit.jsonl."
        )

    locations = list_run_locations()
    if not locations:
        st.info("No cleanup-eligible artifacts found.")
    else:
        location = st.selectbox("Run location", locations)
        if st.button("Preview deletions"):
            # snapshot the selected location — the pending review survives
            # reruns and cannot silently switch to a different location
            st.session_state["cleanup_pending_location"] = location
            st.session_state.pop("cleanup_last_result", None)

        pending = st.session_state.get("cleanup_pending_location")
        if pending:
            st.markdown(f"**Pending cleanup review for:** `{pending}`")
            if st.button("Discard pending preview"):
                st.session_state.pop("cleanup_pending_location", None)
                st.rerun()
            plan = None
            plan_error = None
            try:
                plan = plan_run_cleanup(pending)
            except RetentionError as exc:
                plan_error = str(exc)
            if plan_error:
                # refused — show the reason without killing the other tabs
                st.error(f"Refused: {plan_error}")
            elif not plan.candidates:
                st.info("Nothing to delete (already clean).")
            else:
                st.caption(
                    "The list below reflects the files currently on disk "
                    "— review it before confirming. Preview → confirm → "
                    "delete."
                )
                for cand in plan.candidates:
                    st.markdown(
                        f"- would delete `{cand.display}` "
                        f"({cand.kind}, {cand.size_bytes} bytes)"
                    )
                for display_name, reason in plan.skipped:
                    st.caption(f"skip: {display_name} — {reason}")
                confirm = st.checkbox(
                    "I confirm deletion of exactly the files listed above"
                )
                if confirm and st.button(
                    "Delete listed files", type="primary"
                ):
                    result = execute_run_cleanup(plan, confirm=True)
                    st.session_state["cleanup_last_result"] = result
                    st.session_state.pop(
                        "cleanup_pending_location", None
                    )
                    st.rerun()

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
            # Issue #10 — conversation previews truncated by default; the
            # full text is revealed one record at a time, explicitly.
            st.caption(
                "Conversation previews are truncated by default — a "
                "privacy affordance, not a security boundary."
            )
            st.dataframe(
                preview_dataframe(df.head(100)), use_container_width=True
            )
            render_record_reveal(df.head(100))
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
