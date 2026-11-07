"""Issue #13 — centralized dataset licensing guard.

All data in this file is synthetic (hand-written strings); no real
MentalChat16K records are used, required, or produced.

Boundary semantics in tests: the autouse conftest fixture anchors
``config.ROOT`` to each test's ``tmp_path``. These tests re-anchor it to a
nested fake workspace so that sibling directories are genuinely "outside"
— mirroring the real boundary (project workspace vs everything else).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import config
from data_loader import Sample
from export_for_translation import write_export
from license_guard import (
    ASSESSMENT_REPORT,
    DATASET_NAME,
    DATASET_URL,
    LICENSE_STATUS,
    OVERRIDE_FLAG,
    UPSTREAM_REVISION,
    LicenseGuardError,
    acknowledgment_warning,
    assert_export_allowed,
    classify_destination,
    refusal_message,
)


def _anchor_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Re-anchor the workspace so tmp_path/'public' is a real public dest."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(config, "ROOT", workspace)
    return workspace


def _synthetic_samples(n: int = 3) -> list[Sample]:
    return [
        Sample(idx=i, text=f"synthetic text {i}", meta={
            "instruction": f"synthetic instruction {i}",
            "input": f"synthetic input {i}",
            "output": f"synthetic output {i}",
        })
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Destination classification (project's actual path architecture)
# --------------------------------------------------------------------------


def test_classify_destination_private_roots(tmp_path, monkeypatch):
    workspace = _anchor_workspace(monkeypatch, tmp_path)
    assert classify_destination(workspace / "data" / "x.jsonl") == "private"
    assert classify_destination(workspace / "outputs" / "labeled.csv") == "private"
    # the real managed roots stay private even when ROOT is re-anchored
    assert classify_destination(config.DATA_DIR / "mentalchat16k_fa.jsonl") == "private"
    assert classify_destination(config.OUTPUT_DIR / "labeled.csv") == "private"


def test_classify_destination_outside_workspace_is_public(tmp_path, monkeypatch):
    _anchor_workspace(monkeypatch, tmp_path)
    assert classify_destination(tmp_path / "public" / "copy.jsonl") == "public"
    assert classify_destination(tmp_path / "other_repo" / "data.csv") == "public"


# --------------------------------------------------------------------------
# Refusal / acknowledgment behavior
# --------------------------------------------------------------------------


def test_public_export_refused_without_acknowledgment(tmp_path, monkeypatch):
    _anchor_workspace(monkeypatch, tmp_path)
    out = tmp_path / "public" / "mentalchat16k_copy.jsonl"
    with pytest.raises(LicenseGuardError) as ei:
        assert_export_allowed(out, kind="raw MentalChat16K export")
    msg = str(ei.value)
    assert DATASET_NAME in msg and DATASET_URL in msg   # dataset identified
    assert UPSTREAM_REVISION in msg                     # exact revision
    assert "unclear" in msg                             # unresolved status
    assert OVERRIDE_FLAG in msg                         # the explicit override
    assert ASSESSMENT_REPORT in msg                     # where to read more
    assert config.DATA_DIR.name in msg                  # private alternative
    assert not out.exists()                             # nothing written


def test_public_export_with_flag_prints_acknowledgment(tmp_path, monkeypatch, capsys):
    _anchor_workspace(monkeypatch, tmp_path)
    out = tmp_path / "public" / "copy.jsonl"
    dest = assert_export_allowed(out, kind="test records",
                                 acknowledge_license=True)
    assert dest == "public"
    captured = capsys.readouterr().out
    assert "LICENSE ACKNOWLEDGMENT" in captured
    assert "UNRESOLVED" in captured
    assert DATASET_NAME in captured


def test_acknowledgment_does_not_claim_legal_permission(capsys):
    print(acknowledgment_warning("test records", "somewhere/public.jsonl"))
    captured = capsys.readouterr().out
    assert "does NOT grant legal permission" in captured
    # the flag must never be described as permission or clearance
    assert "grants permission" not in captured
    assert "legally cleared" not in captured


def test_refusal_message_names_private_option(tmp_path, monkeypatch):
    _anchor_workspace(monkeypatch, tmp_path)
    msg = refusal_message(tmp_path / "public" / "x.jsonl", "records")
    assert "private" in msg.lower()


def test_private_processing_always_allowed(tmp_path, monkeypatch, capsys):
    workspace = _anchor_workspace(monkeypatch, tmp_path)
    dest = assert_export_allowed(
        workspace / "data" / "mentalchat16k_fa.jsonl",
        kind="translated dataset",
        acknowledge_license=False,
    )
    assert dest == "private"
    assert capsys.readouterr().out == ""  # no scary warning for local work


def test_confirmed_status_would_allow_public(tmp_path, monkeypatch):
    """Policy hook: a sourced status flip must actually unlock public export."""
    _anchor_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr("license_guard.LICENSE_STATUS", "confirmed")
    dest = assert_export_allowed(
        tmp_path / "public" / "copy.jsonl", acknowledge_license=False
    )
    assert dest == "public"


# --------------------------------------------------------------------------
# Every export entry point routes through the centralized guard
# --------------------------------------------------------------------------


def test_write_export_refuses_public_destination(tmp_path, monkeypatch):
    _anchor_workspace(monkeypatch, tmp_path)
    out = tmp_path / "public" / "en_export.jsonl"
    with pytest.raises(LicenseGuardError):
        write_export(
            _synthetic_samples(), out,
            dataset_total=3, load_report=None,
        )
    assert not out.exists()
    assert not out.with_suffix(".export_meta.json").exists()


def test_write_export_public_with_acknowledgment_writes_and_warns(
    tmp_path, monkeypatch, capsys
):
    _anchor_workspace(monkeypatch, tmp_path)
    out = tmp_path / "public" / "en_export.jsonl"
    meta = write_export(
        _synthetic_samples(), out,
        dataset_total=3, load_report=None,
        acknowledge_license=True,
    )
    assert meta["complete"] is True
    assert out.exists()
    assert "LICENSE ACKNOWLEDGMENT" in capsys.readouterr().out


def test_write_export_private_default_unchanged(tmp_path, monkeypatch):
    _anchor_workspace(monkeypatch, tmp_path)
    out = tmp_path / "workspace" / "data" / "en_export.jsonl"
    meta = write_export(
        _synthetic_samples(), out, dataset_total=3, load_report=None,
    )
    assert meta["exported"] == 3
    assert out.exists()


def test_translate_engine_run_guards_output_and_error_log(tmp_path, monkeypatch):
    """The translation engine checks the FA output AND the error log — the
    two public-copy surfaces of the translation workflow — before any I/O."""
    import json as _json

    from license_guard import LicenseGuardError as LGErr
    from translate_dataset import MockProvider, TranslationEngine

    _anchor_workspace(monkeypatch, tmp_path)
    src = tmp_path / "public" / "src.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        _json.dumps({
            "idx": 0, "instruction": "synthetic en instruction",
            "input": "synthetic en input", "output": "synthetic en output",
        }) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "public" / "fa.jsonl"
    err_log = tmp_path / "public" / "errors.jsonl"

    engine = TranslationEngine(MockProvider(), batch_size=1, max_retries=1)
    with pytest.raises(LGErr):
        engine.run(src, out, error_log_path=err_log)
    assert not out.exists()      # refused before any record I/O
    assert not err_log.exists()


def test_translate_engine_run_private_paths_functional(tmp_path, monkeypatch):
    import json as _json

    from translate_dataset import MockProvider, TranslationEngine

    _anchor_workspace(monkeypatch, tmp_path)
    src = tmp_path / "workspace" / "data" / "src.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        _json.dumps({
            "idx": 0, "instruction": "synthetic en instruction",
            "input": "synthetic en input", "output": "synthetic en output",
        }) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "workspace" / "data" / "fa.jsonl"
    err_log = tmp_path / "workspace" / "data" / "errors.jsonl"

    engine = TranslationEngine(MockProvider(), batch_size=1, max_retries=1)
    summary = engine.run(src, out, error_log_path=err_log, limit=1)
    assert summary.total_source == 1
    assert out.exists()  # ordinary local translation workflow intact


def test_all_entry_points_invoke_the_centralized_guard():
    """Structural lock: no export path may bypass license_guard."""
    root = Path(__file__).resolve().parents[1]
    expectations = {
        "export_for_translation.py": ["assert_export_allowed("],
        "translate_dataset.py": ["assert_export_allowed("],
        "translation_qa.py": ["assert_export_allowed("],
        "pages/5_Translate_Export.py": ["classify_destination("],
    }
    for rel, needles in expectations.items():
        src = (root / rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in src, f"{rel} bypasses the guard ({needle} missing)"


def test_cli_flag_registered_on_all_public_entry_points():
    from export_for_translation import parse_args as export_args
    import translate_dataset

    ns = export_args(["--out", "x.jsonl", "--acknowledge-license"])
    assert ns.acknowledge_license is True
    ns_fa = translate_dataset.parse_args(["--acknowledge-license"])
    assert ns_fa.acknowledge_license is True
    ns_qa = subprocess.run(
        [sys.executable, "-m", "translation_qa", "--help"],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert "--acknowledge-license" in ns_qa.stdout


# --------------------------------------------------------------------------
# .gitignore protections remain effective (verified via git itself)
# --------------------------------------------------------------------------


def _git_check_ignore(*rel_paths: str) -> list[str]:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "--", *rel_paths],
        capture_output=True, text=True, cwd=repo,
    )
    return result.stdout.split()


def test_gitignore_still_covers_dataset_and_exports():
    ignored = set(_git_check_ignore(
        "data/mentalchat16k_fa.jsonl",
        "data/mentalchat16k_en_for_translation.jsonl",
        "data/mentalchat16k_en_for_translation.export_meta.json",
        "data/translation_errors.jsonl",
        "data/translation_review.jsonl",
        "outputs/labeled.csv",
        "outputs/comparison/comparison_report.md",
        ".cache/some_blob.bin",
    ))
    assert len(ignored) == 8  # every dataset/derivative artifact ignored


def test_gitignore_placeholder_and_fixture_stay_tracked():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "data/", "tests/fixtures/"],
        capture_output=True, text=True, cwd=repo,
    )
    tracked = set(result.stdout.split())
    assert "data/.gitkeep" in tracked
    assert "tests/fixtures/mentalchat16k_fa.sample.jsonl" in tracked
    # and the fixture is the synthetic 5-record fixture, not real data
    fixture = repo / "tests/fixtures/mentalchat16k_fa.sample.jsonl"
    assert len(fixture.read_text(encoding="utf-8").splitlines()) == 5


def test_no_dataset_artifact_is_tracked():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=repo,
    )
    tracked = result.stdout.split()
    dataset_like = [
        p for p in tracked
        if p.endswith((".csv", ".tsv"))
        or (p.endswith(".jsonl") and not p.startswith("tests/fixtures/"))
        or p.startswith("data/mentalchat")
        or (p.startswith("outputs/") and p != "outputs/.gitkeep")
    ]
    assert dataset_like == []  # no real dataset record ever committed
