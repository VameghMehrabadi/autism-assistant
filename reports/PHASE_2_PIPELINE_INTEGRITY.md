# Phase 2: Data Pipeline Integrity & Leakage Prevention — Report

**Status: COMPLETE** — four confirmed/suspected issues fixed with regression coverage. **205/205 tests passing** (+15). No external API calls, no translations, no dataset replacement.

## 1. Translation Status Handling (loader)

### Investigation — confirmed
`data_loader.py` contained **zero** references to the Phase-6 translation `status` field: flagged, failed, pending, rejected and empty-source records all entered labeling as if valid, and `extract_label_text`'s patient-mode fallback chain (`inp or inst or output or raw`) could rescue a blank/flagged patient input with the **counselor output**.

### Fix (`data_loader.py::_build_samples`)
A **translation-status gate runs before content validation**:
- Records carrying a `status` other than `translated` (`flagged`, `failed`, `pending`, `rejected`, `empty_source`, …) are **excluded outright** — counted per status in `LoadReport.translation_skipped` with warning issues (`TRANSLATION_STATUS_<STATUS>`) — never silently promoted and never rescued by the output fallback.
- A `translated` record whose FA `input` is blank while `input_en` is present is excluded as **incomplete** (`TRANSLATION_INCOMPLETE_INPUT`) — a partial translation cannot pose as a valid one.
- Only complete `translated` records enter labeling; skipped rows are counted **separately** from `skipped_invalid` (arithmetic updated), so no double counting.
- **Original English preserved**: `instruction_en` / `input_en` / `output_en` (and `flag_reason`) now ride in `Sample.meta` alongside `translation_status`.
- **Legacy compatibility**: status-less rows (plain English dataset) are untouched — no gating, original fallback chain preserved (regression-tested, including the pre-existing generic-instruction and output-only fallbacks).

## 2. Data Leakage (splits)

### Investigation — confirmed mechanism
`assign_splits` hashed the **entire record content**. Two records sharing the same underlying patient input but differing in counselor response hash differently → can land in different splits → the shared input appears in both dev and eval.

### Grouping key (established, not invented)
**The normalized patient `input`** — the field named in the leak report, populated in 15,978/16,005 exported records. Records with an empty input are keyed by their full-content fingerprint (distinct singletons). Implemented as `data_validation.record_group_key()` / `_item_group_key()` (Sample-like items use separated `input_text`; plain hashes stay ungrouped).

### Fix (`data_validation.py`)
- `assign_splits(..., group_keys=None)`: with group keys, assignment is decided **per group** (seeded shuffle of unique group keys, whole groups assigned until `eval_fraction` is reached) — all related hashes share a split. Without group keys, behavior is byte-identical to before.
- `create_splits` computes group keys automatically; the manifest records `"grouped_by": "patient_input"`.
- Determinism preserved (seeded shuffle over a sorted-then-shuffled group list); disjointness re-verified via `check_no_leakage`.

**Documented limitation**: grouping is **exact-match** on normalized input — paraphrased or lightly-edited variants of the same input are *not* grouped (that would require embedding clustering or human judgment = inventing a fuzzy key). Recorded in the function docstring and here as the explicit limitation.

## 3. Review Report Completeness (translation_qa.py)

### Investigation — confirmed
`validate_translation_file` capped the displayed `issues` list at 200, and `write_review_file` derived its review selection from **that capped list** → records with issues beyond the first 200 were silently missing from manual review.

### Fix
- `validate_translation_file` now always builds the **uncapped machine-readable map** `idx_issue_codes` (idx → issue codes) plus explicit totals: `issues_total` (all) and `issues_exported` (capped display list length). The capped `issues` list remains purely for display.
- `write_review_file` selects from `idx_issue_codes` (with a legacy fallback to the old reports' capped list), so **every review-required record is exported regardless of any cap** — regression-tested with 300 issue records (review file contains all 300; `issues_total=300`, `issues_exported=200`).

## 4. Export Safety (export_for_translation.py)

### Investigation — confirmed
The default limit was `config.SAMPLE_LIMIT` (500) with **no completeness signal** — a limited export was byte-indistinguishable from a full one.

### Fix
- `LoadReport.dataset_total` added (total source records available **before** any limit: HF dataset size for EN, file row count for FA).
- New `write_export()` writes a sidecar **`*.export_meta.json`** with `total_source`, `exported`, `complete`, `empty_field_records`, dedup/skip counts, and UTC timestamp; `complete` is true only when every source record was exported.
- Incomplete exports print a **loud WARNING** ("INCOMPLETE export — N of M source records; must NOT be treated as the full dataset") and `main()` returns exit code **3** (distinguishable from success/config errors).
- **CLI change (documented)**: the default limit is now the **full dataset** (`--limit N` remains for deliberate partial exports; `--limit 0` unchanged). A silent 500-record default was the trap itself; partial exports are now opt-in and always flagged. `pages/5_Translate_Export.py` retains its own flow (noted limitation: it does not yet write the sidecar).

## 5. Files Modified
- `data_loader.py` — status gate, `translation_skipped` accounting, `*_en`/`flag_reason` in meta, `dataset_total`, log event field.
- `data_validation.py` — `record_group_key`, group-aware `assign_splits`, `_item_group_key`, `create_splits` grouping, manifest `grouped_by`.
- `translation_qa.py` — `idx_issue_codes` + `issues_exported`, uncapped review selection.
- `export_for_translation.py` — `build_export_record`/`write_export` refactor, meta sidecar, completeness warnings, exit code 3, full-dataset default.
- Tests: `test_data_loader.py` (+7), `test_data_validation.py` (+6), `test_translation_qa.py` (+3).

## 6. Regression Tests Added (16)
- Status gating: only `translated` enters labeling (flagged/failed/pending/rejected/incomplete/empty_source skipped with exact counts + issue codes); flagged record with blank input **never** reaches the counselor-output fallback; English originals + status in meta; legacy status-less rows unaffected (3-case fallback-chain pin).
- Split grouping: grouped assignment keeps shared-input records together; determinism; group-level eval fraction; length-mismatch error; end-to-end `create_splits` disjointness across 4 seeds (30 related pairs + 1 singleton); manifest `grouped_by`.
- Review completeness: 300 issues → all 300 reviewed despite the 200 display cap; legacy-report fallback; total/exported count reporting.
- Export safety: complete vs incomplete meta sidecars; `dataset_total` before limit.

## 7. Test Results (actual)
| Suite | Result |
|---|---|
| Focused (loader + validation + translation QA) | 89 passed |
| **Complete suite** | **205 passed** (190 + 15) |

## 8. Remaining Limitations
- Split grouping is exact-match on normalized input (paraphrase-level relatedness undetected — documented in code).
- Records without separated `input_text` (plain-hash items) cannot be grouped.
- `pages/5_Translate_Export.py` still writes its export without the meta sidecar.
- The legacy (status-less) fallback chain still permits counselor-output fallback for old English files — preserved deliberately for backward compatibility; the translation path is fully gated.
