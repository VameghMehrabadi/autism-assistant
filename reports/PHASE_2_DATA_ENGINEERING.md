# Phase 2: Data Engineering Report

**Status: COMPLETE** — all acceptance criteria met, 73/73 tests passing.

## Scope

Improved the existing dataset pipeline while preserving its architecture
(`Sample` dataclass, `load_samples_en` / `load_samples_fa` / `load_samples`
public API, patient-first label-text modes, Hugging Face path for English).

## Files Created

| File | Purpose |
|------|---------|
| `data_validation.py` | Validation utilities: encoding detection, record normalization, schema validation, fingerprints/dedup, language ID, integrity reports, dev/eval splits + leakage guard |
| `privacy_log.py` | Structured JSON-line logging that structurally redacts conversational content (sensitive keys and long strings → `{len, sha256[:12]}` fingerprint) |
| `tests/test_data_validation.py` | 26 unit tests for the utilities above |
| `reports/PHASE_2_DATA_ENGINEERING.md` | This report |

## Files Modified

| File | Change |
|------|--------|
| `data_loader.py` | Rewritten internally; public API backward compatible. New: robust file reader (`_load_rows`), shared build pipeline (`_build_samples`), `LoadReport`, split support in `load_samples`, input/response separation on `Sample`, privacy-safe `__main__` demo |
| `config.py` | Added Phase-2 settings (see below); added `LOG_DIR` to the created-dirs list |
| `tests/test_data_loader.py` | Kept all 5 original tests; added 20 unit/integration tests |

### New configuration keys (all optional, safe defaults)

| Key | Default | Meaning |
|-----|---------|---------|
| `PREPROCESS_VERSION` | `"2.0.0"` | Recorded in reports/manifests for reproducibility |
| `VALIDATION_STRICT` | `False` | `False`: invalid records skipped + counted; `True`: raise |
| `DEDUP_STRATEGY` | `"keep_first"` | `"keep_first"` removes later duplicates; `"keep_all"` only counts |
| `LANGUAGE_FILTER` | `None` | `"fa"`/`"en"` filters label text by detected language; `None` = count only |
| `SAMPLING_STRATEGY` | `"first"` | `"first"` = previous behavior; `"random"` = seeded, reproducible |
| `SAMPLING_SEED` | `42` | Seed for random sampling and split assignment |
| `SPLIT_MANIFEST_PATH` | `outputs/split_manifest.json` | Persistent dev/eval assignment (hashes only) |
| `LOG_DIR` / `LOG_JSON` / `LOG_LEVEL` / `LOG_MAX_INLINE_LEN` | `outputs/logs`, `True`, `"INFO"`, `120` | Privacy-preserving logging controls |

## Requirement Coverage

1. **Formats** — English MentalChat16K (HF, unchanged), Persian/any local
   JSONL, CSV, TSV, JSON (list or `{"data": [...]}`). CSV is read with the
   detected encoding; `utf-8-sig` BOMs are handled.
2. **Robustness**
   - *Schema validation*: `validate_record` — `EMPTY_RECORD` (error),
     `BAD_IDX` (warning); non-dict JSON values counted, never crash.
   - *Encoding*: BOM sniffing (UTF-8/16/32), strict UTF-8, then cp1256 /
     cp1252 / latin-1 fallback with Arabic-script heuristic (verified
     against a legacy-encoded Persian file).
   - *Missing values*: `normalize_record` maps `None`/`NaN`/blank → `""`
     before validation and labeling.
   - *Duplicates*: whitespace-insensitive SHA-256 content fingerprint;
     `keep_first` removal (default) or count-only; groups reported.
   - *Integrity checks*: `integrity_report` — field coverage, empty
     records, duplicate rows/groups, language distribution, version.
   - *Language ID*: dependency-free script-based detection returning
     `fa` / `en` / `mixed` / `unknown`, used for reporting and optional
     filtering.
   - *Input/response separation*: `Sample.input_text` / `.response_text`
     carry the parsed patient input and counselor response.
3. **Original records preserved** — source `idx` kept when integer-like
   (positional fallback), original text fields kept in `meta`, plus
   `record_hash`, `lang`, `source` metadata.
4. **Reproducibility** — deterministic pipeline; seeded sampling; split
   assignment keyed by content hash (same content ⇒ same split regardless
   of row order); `PREPROCESS_VERSION` in every report/manifest.
5. **Configurable limits & sampling** — `limit` (`None`/`0` = full) and
   `sampling="first"|"random"` with `seed`, applied over the *validated
   pool*; random subsets are returned in stable original order.
6. **Leakage prevention** — `create_splits()` builds a dev/eval split from
   content hashes and persists a manifest; `load_samples(..., split="eval")`
   returns *only* manifest-listed records and **fails fast** if no manifest
   exists; `split="dev"` excludes eval; `check_no_leakage()` raises
   `LeakageError` on any overlap; records without a hash can never enter a
   split (counted as `missing_hash`).
7. **Sensitive data handling** — logs never contain conversation text
   (sensitive keys and long strings are fingerprinted, not stored);
   malformed JSONL lines are logged by line number only (the line itself
   could contain patient text); split manifests store hashes only.
8. **Patient input default** — `LABEL_TEXT_MODE="patient"` preserved as the
   default everywhere; `Sample.text` is the patient description, verified by
   `test_build_samples_separates_patient_input_from_response`.
9. **Structured logging** — JSON lines in `outputs/logs/pipeline.log.jsonl`
   (`load_start`, `load_complete`, `split_filtered`, `malformed_line`),
   human console prints unchanged.
10. **Tests** — 46 new tests (see below).

## Test Results (actual)

```
tests\test_compare_config.py ....                                       [  5%]
tests\test_data_loader.py .........................                     [ 39%]
tests\test_data_validation.py ..........................                [ 75%]
tests\test_facts.py .........                                           [ 87%]
tests\test_gap_analysis.py ..                                           [ 90%]
tests\test_labeler.py .......                                           [100%]
73 passed in 13.57s
```

- 22 pre-existing tests: **all pass unmodified in behavior** (5 kept in
  `test_data_loader.py`, 17 in the other suites).
- 46 new tests: 25 in `test_data_loader.py`, 26 in
  `test_data_validation.py` — covering malformed JSONL, empty CSV cells,
  `utf-8-sig`/cp1256 files, dedup on/off, language filter/counting, seeded
  sampling reproducibility, identifier preservation, strict mode, full
  dev/eval split round-trip with disjointness + manifest reload, and
  fail-fast eval loading without a manifest.
- All modules compile; `python main.py --help` smoke-tested.
- Log file audited: counts/paths only, no conversational content.

## Design Notes & Decisions

- **Architecture preserved**: no consumer changes were required —
  `main.py`, `ui_lib/caching.py`, `pages/5_Translate_Export.py`,
  `export_for_translation.py` work unchanged (verified by import/compile
  and by signature compatibility: new parameters are keyword-only with
  `None` → config defaults).
- `datasets` (Hugging Face) is now imported lazily inside
  `load_samples_en`, so Persian-only workflows work without it.
- Language detection is script-based (Arabic-script ⇒ `fa`): it
  distinguishes Persian/English text reliably for filtering purposes but
  does not distinguish Arabic from Persian — acceptable for this
  bilingual dataset, documented as a limitation.
- Dedup defaults to `keep_first`, which can change sample counts when a
  dataset contains duplicates — this is intentional (requirement 2) and
  reported transparently in `LoadReport.duplicates_removed`.

## Known Limitations

- Encoding fallback for single-byte encodings is heuristic (cp1256 chosen
  when the decoded sample contains Arabic-script characters).
- `LoadReport.issues` grows with the number of invalid records; for very
  dirty datasets it may be large (bounded use: report saved per run).
- Split manifest hashes are content-based: if the source file itself
  changes a record's text, its hash changes and it may land in the other
  split — regenerate the manifest after dataset edits (documented).

## Remaining Tasks (for later phases)

- Surface `--split dev/eval` and `--sampling random` as CLI flags in
  `main.py` / Streamlit "Run Pipeline" page (API support already in place).
- Optional CSV column-mapping configuration for non-MentalChat schemas.
