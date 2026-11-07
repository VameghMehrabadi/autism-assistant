"""Issue #10 — retention, safe run-output cleanup, UI privacy previews.

All fixtures are synthetic and live in isolated temporary directories
(config.OUTPUT_DIR / config.DATA_DIR are re-anchored per test); no real
project data, no real conversations, and nothing under the actual
outputs/ or data/ is ever touched by these tests.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import config
import retention
from retention import (
    CleanupPlan,
    RetentionError,
    execute_run_cleanup,
    list_run_locations,
    plan_run_cleanup,
    record_deletion_audit,
)
from ui_lib.display import preview_dataframe, preview_text

SENTINEL = "SYNTHETIC-CONVERSATION-TEXT-abc123 — nobody should read this in logs"
SENTINEL2 = "SECOND-SYNTHETIC-CONVERSATION-xyz789"

RUN_FILES = {
    "labeled.csv": f"idx,text,top_label\n0,{SENTINEL},B\n",
    "scores.npy": "\x00\x01synthetic-binary",
    "gap_analysis.json": '{"n_samples": 1}',
    "gap_analysis.md": "# gap report",
    "run_meta.json": '{"model_key": "minilm"}',
}


def _make_roots(tmp_path: Path) -> tuple[Path, Path]:
    out_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    (out_dir / "logs").mkdir(parents=True)
    data_dir.mkdir()
    return out_dir, data_dir


def _anchor_roots(monkeypatch: pytest.MonkeyPatch, out_dir: Path, data_dir: Path):
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    # pages/4 resolves run folders against config.ROOT — anchor it too
    monkeypatch.setattr(config, "ROOT", out_dir.parent)


def _make_run(out_dir: Path, name: str, *, with_unknown: bool = False) -> Path:
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for file_name, content in RUN_FILES.items():
        (run_dir / file_name).write_text(content, encoding="utf-8")
    if with_unknown:
        (run_dir / "operator_notes.txt").write_text(
            "hand-written note — not a run artifact", encoding="utf-8"
        )
    return run_dir


# --------------------------------------------------------------------------
# 1+2. Preview identifies intended files; confirmed cleanup deletes them
# --------------------------------------------------------------------------


def test_preview_identifies_intended_files(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "my_run", with_unknown=True)

    plan = plan_run_cleanup("my_run")
    displays = {c.display for c in plan.candidates}
    assert displays == {f"my_run/{name}" for name in RUN_FILES}
    # the unknown file is never a candidate and is explained
    assert all("operator_notes.txt" not in c.display for c in plan.candidates)
    assert any("operator_notes.txt" in d for d, _ in plan.skipped)
    assert (
        "manual review"
        in dict(plan.skipped)["my_run/operator_notes.txt"]
    )


def test_confirmed_cleanup_deletes_eligible_artifacts(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    run_dir = _make_run(out_dir, "my_run")

    plan = plan_run_cleanup("my_run")
    result = execute_run_cleanup(plan, confirm=True)
    assert result["deleted"] == len(RUN_FILES)
    assert result["outcome"] == "complete"
    assert not any(run_dir.glob("*"))  # every whitelisted file is gone


# --------------------------------------------------------------------------
# 3. Sidecar consistency
# --------------------------------------------------------------------------


def test_sidecar_removed_with_its_export(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    export = data_dir / "mentalchat16k_en_for_translation.jsonl"
    sidecar = data_dir / "mentalchat16k_en_for_translation.export_meta.json"
    export.write_text("synthetic export record\n", encoding="utf-8")
    sidecar.write_text('{"complete": true}', encoding="utf-8")

    plan = plan_run_cleanup("translation-artifacts")
    kinds = {c.display: c.kind for c in plan.candidates}
    assert kinds["mentalchat16k_en_for_translation.export_meta.json"] == (
        "sidecar"
    )
    result = execute_run_cleanup(plan, confirm=True)
    assert result["deleted"] == len(plan.candidates)
    assert not export.exists()
    assert not sidecar.exists()  # exclusively paired sidecar removed


def test_missing_sidecar_reported_not_fatal(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    (data_dir / "mentalchat16k_fa.jsonl").write_text("synthetic\n",
                                                     encoding="utf-8")
    plan = plan_run_cleanup("translation-artifacts")
    assert not any(
        c.display.endswith(".export_meta.json") for c in plan.candidates
    )
    result = execute_run_cleanup(plan, confirm=True)
    assert result["outcome"] == "complete"


# --------------------------------------------------------------------------
# 4+5+6. Original datasets, unrelated runs, shared caches untouched
# --------------------------------------------------------------------------


def test_original_datasets_and_custom_files_untouched(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    original = data_dir / "original_source_dataset.csv"
    original.write_text("original,data\n1,2\n", encoding="utf-8")
    gold = data_dir / "gold"
    gold.mkdir()
    (gold / "gold_labels.jsonl").write_text('{"idx": 0}\n', encoding="utf-8")
    (data_dir / "mentalchat16k_fa.jsonl").write_text(SENTINEL2 + "\n",
                                                     encoding="utf-8")

    plan = plan_run_cleanup("translation-artifacts")
    execute_run_cleanup(plan, confirm=True)
    assert original.exists()          # custom/original files untouched
    assert (gold / "gold_labels.jsonl").exists()  # gold data untouched


def test_unrelated_runs_untouched(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    target = _make_run(out_dir, "run_to_delete")
    other = _make_run(out_dir, "other_run")

    plan = plan_run_cleanup("run_to_delete")
    execute_run_cleanup(plan, confirm=True)
    assert not target.exists() or not any(target.glob("*"))
    assert (other / "labeled.csv").exists()  # unrelated run untouched
    assert (other / "run_meta.json").exists()


def test_shared_caches_and_model_weights_untouched(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    cache_dir = tmp_path / "cache"
    (cache_dir / "embed_cache").mkdir(parents=True)
    (cache_dir / "embed_cache" / "chunk-0.parquet").write_text("vectors")
    (cache_dir / "models--minilm").mkdir()
    (cache_dir / "models--minilm" / "weights.bin").write_text("weights")
    _make_run(out_dir, "my_run")

    plan = plan_run_cleanup("my_run")
    execute_run_cleanup(plan, confirm=True)
    assert (cache_dir / "embed_cache" / "chunk-0.parquet").exists()
    assert (cache_dir / "models--minilm" / "weights.bin").exists()
    # the plan never even looked outside the output/data roots
    assert all(
        not str(c.path).startswith(str(cache_dir)) for c in plan.candidates
    )


def test_shared_comparison_summaries_not_deletable(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    comp = out_dir / "comparison"
    comp.mkdir()
    (comp / "comparison_summary.csv").write_text("aggregate,stats\n")
    _make_run(comp, "1_fa_en_minilm")

    with pytest.raises(RetentionError):  # shared root refused wholesale
        plan_run_cleanup("comparison")
    plan = plan_run_cleanup("comparison/1_fa_en_minilm")
    assert all(
        "comparison_summary" not in c.display for c in plan.candidates
    )


# --------------------------------------------------------------------------
# 7+8+9. Containment, traversal, symlinks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location",
    ["../data", "../../etc", "comparison/../../data", "my_run/../../..",
     "comparison/../my_run"],
)
def test_path_traversal_rejected(tmp_path, monkeypatch, location):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    with pytest.raises(RetentionError):
        plan_run_cleanup(location)


def test_nonexistent_location_is_a_clean_noop(tmp_path, monkeypatch):
    """A location that does not exist (typo or already cleaned) deletes
    nothing and does not raise — repeating a completed cleanup is safe."""
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    plan = plan_run_cleanup("never_existed")
    assert plan.candidates == ()
    result = execute_run_cleanup(plan, confirm=True)
    assert result["deleted"] == 0


def test_absolute_or_external_location_rejected(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(RetentionError):
        plan_run_cleanup(str(outside))  # absolute path outside the root


def test_storage_root_itself_is_never_a_target(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    with pytest.raises(RetentionError):
        plan_run_cleanup(".")
    with pytest.raises(RetentionError):
        plan_run_cleanup("comparison")  # shared root refused wholesale


def test_symlink_escape_is_skipped(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    run_dir = _make_run(out_dir, "my_run")
    outside = tmp_path / "outside.txt"
    outside.write_text("precious", encoding="utf-8")
    link = run_dir / "labeled.csv"  # replace with a symlink pointing outside
    link.unlink()
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")

    plan = plan_run_cleanup("my_run")
    assert all("labeled.csv" not in c.display for c in plan.candidates)
    assert any("labeled.csv" in d for d, _ in plan.skipped)
    execute_run_cleanup(plan, confirm=True)
    assert outside.exists()  # the symlink target was never touched


# --------------------------------------------------------------------------
# 10. Missing files and partial failures
# --------------------------------------------------------------------------


def test_missing_files_and_partial_failure_handled(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    run_dir = _make_run(out_dir, "my_run")
    plan = plan_run_cleanup("my_run")
    # a file disappears between planning and execution:
    (run_dir / "labeled.csv").unlink()

    stubborn = run_dir / "scores.npy"  # read-only -> deletion fails on win
    stubborn.chmod(stat.S_IREAD)
    try:
        result = execute_run_cleanup(plan, confirm=True)
        assert result["already_absent"] == 1
        assert result["failed"] == 1
        assert result["outcome"] == "partial"
        assert "my_run/scores.npy" in result["failed_files"]
        # everything else was still deleted despite the one failure:
        assert not (run_dir / "gap_analysis.json").exists()
        assert (run_dir / "scores.npy").exists()  # the failed one survives
    finally:
        stubborn.chmod(stat.S_IWRITE | stat.S_IREAD)


# --------------------------------------------------------------------------
# 11. Idempotency
# --------------------------------------------------------------------------


def test_repeated_cleanup_is_idempotent(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "my_run")

    first = execute_run_cleanup(plan_run_cleanup("my_run"), confirm=True)
    assert first["deleted"] == len(RUN_FILES)
    snapshot = sorted(p.name for p in out_dir.rglob("*") if p.is_file())

    second_plan = plan_run_cleanup("my_run")  # dir gone -> clean no-op
    assert second_plan.candidates == ()
    second = execute_run_cleanup(second_plan, confirm=True)
    assert second["deleted"] == 0
    assert sorted(p.name for p in out_dir.rglob("*") if p.is_file()) == (
        snapshot
    )


# --------------------------------------------------------------------------
# 12. Deletion audit logs contain no conversation text
# --------------------------------------------------------------------------


def test_audit_log_is_metadata_only(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "my_run")  # files contain SENTINEL text

    execute_run_cleanup(plan_run_cleanup("my_run"), confirm=True)
    audit = out_dir / "logs" / "deletion_audit.jsonl"
    assert audit.exists()
    records = [json.loads(line) for line in
               audit.read_text(encoding="utf-8").splitlines()]
    assert records and records[-1]["event"] == "run_cleanup"
    record = records[-1]
    assert set(record) == {
        "event", "ts_utc", "location", "mode", "files_selected",
        "files_deleted", "files_skipped", "files_failed", "outcome",
    }
    whole_log = audit.read_text(encoding="utf-8")
    assert SENTINEL not in whole_log
    assert SENTINEL2 not in whole_log
    assert "labeled.csv" not in whole_log  # not even file names — counts only
    assert str(tmp_path) not in whole_log  # no absolute paths


def test_preview_does_not_write_audit(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "my_run")
    execute_run_cleanup(plan_run_cleanup("my_run"), confirm=False)
    assert not (out_dir / "logs" / "deletion_audit.jsonl").exists()


def test_record_deletion_audit_shape(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    path = record_deletion_audit(
        location="x", mode="confirmed", selected=2, deleted=2,
        skipped=0, failed=0, outcome="complete",
    )
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["location"] == "x" and line["mode"] == "confirmed"


# --------------------------------------------------------------------------
# list_run_locations + CLI
# --------------------------------------------------------------------------


def test_list_run_locations(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "solo_run")
    comp = out_dir / "comparison"
    comp.mkdir()
    _make_run(comp, "1_fa_en_minilm")
    (data_dir / "mentalchat16k_fa.jsonl").write_text("synthetic\n",
                                                     encoding="utf-8")

    locations = list_run_locations()
    assert "solo_run" in locations
    assert "comparison/1_fa_en_minilm" in locations
    assert "translation-artifacts" in locations
    assert "comparison" not in locations  # shared root never addressable


def test_cli_list_preview_confirm_and_refusal(tmp_path, monkeypatch, capsys):
    import cleanup_runs

    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "my_run")

    assert cleanup_runs.main(["--list"]) == 0
    assert cleanup_runs.main(["--run", "my_run"]) == 0  # preview default
    out = capsys.readouterr().out
    assert "WOULD DELETE" in out and "preview only" in out
    assert (out_dir / "my_run" / "labeled.csv").exists()

    assert cleanup_runs.main(["--run", "my_run", "--confirm"]) == 0
    assert not (out_dir / "my_run" / "labeled.csv").exists()

    with pytest.raises(SystemExit):
        cleanup_runs.build_parser().parse_args(
            ["--run", "my_run", "--preview", "--confirm"]
        )
    rc = cleanup_runs.main(["--run", "../data", "--confirm"])
    assert rc == 2  # traversal refused with a clear exit code


# --------------------------------------------------------------------------
# 13+14. UI privacy previews
# --------------------------------------------------------------------------


def test_previews_truncated_by_default():
    long_text = SENTINEL + " " + "x" * 300
    shown = preview_text(long_text)
    assert len(shown) < len(long_text)
    assert SENTINEL[:40] in shown          # starts the preview
    assert "reveal to view" in shown       # explicit marker
    assert shown.endswith("]")             # never ends mid-sentence content
    short = preview_text("short")
    assert short == "short"                # short text untouched


def test_explicit_reveal_shows_full_text():
    long_text = SENTINEL + " " + "x" * 300
    assert preview_text(long_text, reveal=True) == long_text
    assert preview_text("short", reveal=True) == "short"
    assert preview_text(None) == ""


def test_preview_dataframe_truncates_without_touching_source():
    import pandas as pd

    df = pd.DataFrame({
        "idx": [0, 1],
        "text": [SENTINEL + " " + "y" * 300, SENTINEL2],
        "top_label": ["B", "B"],
    })
    shown = preview_dataframe(df.head(2))
    # source dataframe unchanged
    assert df.iloc[0]["text"] == SENTINEL + " " + "y" * 300
    assert df.iloc[1]["text"] == SENTINEL2
    # display copy truncated
    assert "reveal to view" in shown.iloc[0]["text"]
    assert SENTINEL2 == shown.iloc[1]["text"]  # short cell unchanged
    assert list(shown.columns) == list(df.columns)


def test_results_page_wires_truncation_and_reveal():
    src = (Path(__file__).resolve().parents[1] / "pages" / "4_Results.py"
           ).read_text(encoding="utf-8")
    assert "preview_dataframe(df.head(100))" in src  # table truncated
    assert "render_record_reveal(df.head(100))" in src  # local reveal
    assert "privacy affordance, not a security boundary" in src


def test_quick_label_captions_deliberate_load():
    src = (Path(__file__).resolve().parents[1] / "pages" / "1_Quick_Label.py"
           ).read_text(encoding="utf-8")
    assert "deliberately loads its full text" in src


def test_no_conversation_text_retained_in_session_state():
    """Revealed content is rendered from the dataframe, never copied into
    Streamlit session state by the privacy-mode widgets."""
    import ui_lib.display as display_mod

    src = Path(display_mod.__file__).read_text(encoding="utf-8")
    assert "session_state" not in src


# --------------------------------------------------------------------------
# PR #38 audit fixes — orphan sidecars + UI workflow regression
# --------------------------------------------------------------------------


def test_orphan_sidecar_detected_and_removed(tmp_path, monkeypatch):
    """Audit fix: the export is already gone while its exclusively-paired
    sidecar remains — the orphan is planned as a sidecar and removed
    instead of being left as a stale reference."""
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    sidecar = data_dir / "mentalchat16k_en_for_translation.export_meta.json"
    sidecar.write_text('{"complete": true}', encoding="utf-8")

    plan = plan_run_cleanup("translation-artifacts")
    displays = [c.display for c in plan.candidates]
    assert displays == [
        "mentalchat16k_en_for_translation.export_meta.json"
    ]
    assert all(c.kind == "sidecar" for c in plan.candidates)
    result = execute_run_cleanup(plan, confirm=True)
    assert result["deleted"] == 1
    assert not sidecar.exists()
    # idempotent: the repeat is a clean no-op
    repeat = execute_run_cleanup(
        plan_run_cleanup("translation-artifacts"), confirm=True
    )
    assert repeat["deleted"] == 0


def test_ambiguous_sidecar_is_never_deleted(tmp_path, monkeypatch):
    """A non-regular file occupying the sidecar path (here: a directory) is
    ambiguous — it is reported for manual review, never deleted."""
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    odd = data_dir / "mentalchat16k_en_for_translation.export_meta.json"
    odd.mkdir()
    other = data_dir / "mentalchat16k_fa.jsonl"
    other.write_text("synthetic\n", encoding="utf-8")

    plan = plan_run_cleanup("translation-artifacts")
    assert all(
        "export_meta" not in c.display for c in plan.candidates
    )
    assert any("not a regular file" in reason for _, reason in plan.skipped)
    execute_run_cleanup(plan, confirm=True)
    assert odd.is_dir()  # untouched
    assert not other.exists()  # the unambiguous artifact was cleaned


def test_paired_sidecar_symlink_escape_skipped(tmp_path, monkeypatch):
    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    (data_dir / "mentalchat16k_fa.jsonl").write_text("synthetic\n",
                                                     encoding="utf-8")
    outside = tmp_path / "outside_meta.json"
    outside.write_text("precious", encoding="utf-8")
    sidecar = data_dir / "mentalchat16k_fa.export_meta.json"
    try:
        sidecar.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")

    plan = plan_run_cleanup("translation-artifacts")
    assert all("export_meta" not in c.display for c in plan.candidates)
    execute_run_cleanup(plan, confirm=True)
    assert outside.exists()  # symlink target never touched


def test_results_page_cleanup_tab_uses_session_state_workflow():
    """Structural lock (supplements the AppTest): confirmation controls
    must not be nested inside the Preview-click guard, and the pending
    location must be snapshotted into session state."""
    src = (Path(__file__).resolve().parents[1] / "pages" / "4_Results.py"
           ).read_text(encoding="utf-8")
    tab_block = src.split('with tab_retention:')[1]
    preview_idx = tab_block.index('if st.button("Preview deletions")')
    confirm_idx = tab_block.index('confirm = st.checkbox(')
    delete_idx = tab_block.index('"Delete listed files"')
    # the confirm/delete controls live AFTER the preview guard block ends,
    # gated on the snapshotted pending location instead
    pending_idx = tab_block.index('pending = st.session_state.get(')
    assert preview_idx < pending_idx < confirm_idx < delete_idx
    assert 'st.session_state["cleanup_pending_location"] = location' in (
        tab_block
    )


def test_ui_cleanup_flow_app_test(tmp_path, monkeypatch):
    """AppTest regression for the audit finding: after the Preview rerun,
    the confirmation checkbox and Delete button must still exist, and the
    Delete click must reach the existing retention implementation."""
    from streamlit.testing.v1 import AppTest

    out_dir, data_dir = _make_roots(tmp_path)
    _anchor_roots(monkeypatch, out_dir, data_dir)
    _make_run(out_dir, "ui_run")

    page = Path(__file__).resolve().parents[1] / "pages" / "4_Results.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.run()

    # select the synthetic run and click Preview
    selectbox = next(
        sb for sb in at.selectbox if sb.label == "Run location"
    )
    selectbox.set_value("ui_run")
    at.run()
    next(
        b for b in at.button if b.label == "Preview deletions"
    ).click()
    at.run()

    # REGRESSION: controls persist after the Preview rerun
    assert any(
        "Pending cleanup review" in md.value for md in at.markdown
    )
    checkbox = next(
        cb for cb in at.checkbox if "I confirm deletion" in cb.label
    )
    checkbox.check()
    at.run()  # rerun with confirm=True -> the Delete button now renders
    next(
        b for b in at.button if b.label == "Delete listed files"
    ).click()
    at.run()

    # the existing retention implementation actually executed
    assert not (out_dir / "ui_run" / "labeled.csv").exists()
    assert "cleanup_pending_location" not in at.session_state
    result_state = at.session_state.get("cleanup_last_result")
    assert result_state is not None and result_state["deleted"] == 5
    assert result_state["outcome"] == "complete"
    assert any(
        "Deleted 5 file(s)" in s.value for s in at.success
    )
