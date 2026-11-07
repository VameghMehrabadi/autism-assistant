# Phase 4: Evaluation Validity & Model Comparison — Methodology Report

**Status: COMPLETE** — comparison verified equivalent-input, score-scale honesty enforced in artifacts, alignment metadata recorded going forward, gold-set governance created (no fabricated labels). **246/246 tests passing** (+41). No API calls, no model downloads, no translations.

## 1. Threshold Calibration — Investigation & Honest Terminology

### Investigation of the 0.35 cosine threshold
- All three comparison runs applied the **same** 0.35 threshold (recorded in each `gap_analysis.json`), yet produced **materially different confident rates**: MiniLM 84.4% (15.6% explicitly uncertain) vs bge-m3 100.0% vs e5-large 100.0%.
- Cause: cosine score **scales are model-dependent**. e5-large's top-1 scores concentrate near 0.80; MiniLM's spread lower. The identical threshold therefore cuts each model's score distribution at a different effective point. **The confident-rate differences across models partly measure score-scale differences, not model quality.**
- New `top_score_stats` (mean/median/p25/p75/min/max of top-1 similarity) are now computed in `gap_analysis.build_report` and carried into every gap report and the comparison summary, so the scale travels with the numbers.

### Honesty verification (application-wide)
| Surface | Status |
|---|---|
| Every labeling result | `score_type: "cosine_similarity"` field; `margin` labeled "ambiguity heuristic, not calibrated confidence" |
| Quick Label UI | "Scores are cosine similarities in [-1, 1] — not probabilities" + margin caption |
| Comparison report | new caveats: model-dependent thresholds, differing text windows, "higher cosine scores do not prove a better model", calibration requirement |
| Docstrings/reports | "uncalibrated geometric heuristics" language used consistently |
| Human-labeled calibration | **documented as required before any threshold carries operational meaning** (`data/gold/README.md`); gold set created in this phase (§4) |

No surface presents cosine similarity as a probability, calibrated confidence, or diagnostic certainty.

## 2. Three-Model Comparison — Verification Results (read-only, evidence)

Verified across `1_fa_en_minilm`, `2_fa_en_bge-m3`, `3_fa_en_e5-large` (all n=16,005):

| Check | Result |
|---|---|
| Same dataset records | ✅ identical labeled-`idx` sets across all three runs |
| Identical preprocessing output | ✅ **identical label-text content fingerprint** `cb4b81e2867729e0` (sorted labeled text, SHA-256) |
| Same intended facts | ✅ `fact_lang=fa` in all run metadata; `facts.py` last modified **19:40 Nov 3** (Phase-0 baseline) — predates every run → identical label definitions |
| Label definitions identical | ✅ same `facts.py` revision for all runs |
| Threshold recorded | ✅ `threshold: 0.35` in every `gap_analysis.json`; now also in `run_meta.json` for future runs |
| Missing records reported | ✅ per-run category counts sum to 16,005; the 79 upstream duplicates remain documented as absent (16,084 − 79) |
| Top-1 distributions consistent | ✅ computed by the same `build_report` code path for all runs |

Findings recap (from the executed runs): MiniLM C 33.0/F 26.1/B 23.8 with 15.6% explicit uncertainty; bge-m3 B 43.4/C 19.0 with 100% confident; e5-large F 31.3/G 30.6/C 25.9, scores clustered near 0.80.

## 3. Alignment

**Confirmed real**: the three runs used **different effective text windows** — MiniLM truncates at its native 128 tokens, bge-m3 ran with a 512-token window (pinned in this project after the original 8192-window run proved impractical), e5-large at its native 512. Different windows = different effective inputs for long conversations; this is part of each model's configured preprocessing and is now **recorded per run** (`run_meta.text_window`, `embed_prefix`).

Actions taken:
- `main.py` records `text_window`, `embed_prefix`, `threshold`, `preprocess_version`, `dataset_total`, and a `dataset_fingerprint` (SHA-256 of the exact labeled inputs) in `run_meta.json` for **future** runs.
- Historical runs are **not** retroactively modified — `verify_alignment` reports their metadata honestly as `"pre-alignment-metadata"`.
- E5 `query:`/`passage:` prefixes vs none for MiniLM/bge are intended model-specific preprocessing (documented in the comparison notes).
- English/Persian record alignment for future Runs 4–5 is by preserved `idx` (translation schema keeps `idx` + `*_en` originals); no unaligned records can arise silently because the loader keys everything on `idx`.
- New `python compare_runs.py --verify-alignment` performs the read-only equivalence check (records, text fingerprint, threshold, windows) — **live result on the real runs: `all_records_aligned: true`**, with runs 4–5 reported as `MISSING_ARTIFACTS` (n=2) rather than silently skipped.

## 4. Human-Labeled Gold Evaluation Set (schema + governance — NO labels fabricated)

Created under `data/gold/` (kept **separate** from development data and model outputs):
- **`README.md`** — labeling instructions: stratified 150–300-record sample (all uncertain records prioritized), category definitions bound to `facts.py::CATEGORIES`, two-round labeling with blind disagreement revision, third-reviewer adjudication, `UNSURE` handling (retained, excluded from per-category metrics), versioning (`gold_labels_v1.jsonl`, versions retained), and honesty rules (no model-generated labels, no accuracy claims before real labels exist).
- **`gold_eval.schema.json`** — normative record schema: `origin: "human"` (const), `labeler_id`, `round`, `first_label`/`final_label` (A–G/UNSURE), `adjudicated`, ordinal `confidence` (`sure|leaning|guess` — explicitly *not* a calibrated probability), `gold_set_version`; `additionalProperties: false`.
- **`gold_eval.py`** — dependency-free validator: rejects model-origin records, invalid categories/rounds/confidence, duplicate `gold_id`/`source_idx`, missing required fields; empty file is valid-but-empty ("no human labels exist yet"); CLI exit codes 0/2.

**No gold labels exist yet** — the validator reports an empty set as valid-but-empty, and no evaluation accuracy is claimed anywhere.

## 5. UI and Research Safeguards
- Home page: prominent warning — "Research prototype — not a medical device… does not diagnose autism, assess patients, or provide clinical guidance… scores are uncalibrated geometric heuristics."
- Run Pipeline + Results pages: "Research use only — not diagnosis or clinical assessment" captions.
- Quick Label already carries the cosine-honesty captions (Phase 3); unchanged.

## 6. Files Changed
- `gap_analysis.py` — `top_score_stats` (scale honesty).
- `compare_runs.py` — `summarize_run` carries threshold/scale/alignment metadata; `verify_alignment()` + `--verify-alignment`; methodology caveats in the Markdown renderer; `hashlib` import.
- `main.py` — `run_meta.json` now records `threshold`, `preprocess_version`, `text_window`, `dataset_fingerprint`, `dataset_total` (via `return_report=True`); `hashlib` import.
- `gold_eval.py` (new), `data/gold/README.md` (new), `data/gold/gold_eval.schema.json` (new).
- `ui.py`, `pages/2_Run_Pipeline.py`, `pages/4_Results.py` — research-use safeguards.
- Tests: `tests/test_evaluation_validity.py` (12), `tests/test_gold_eval.py` (10).

## 7. Test Results (actual)
| Suite | Result |
|---|---|
| Phase-4 tests (validity + gold) | 22 passed |
| **Complete suite** | **246 passed** |
| Live alignment check (`--verify-alignment`) | `all_records_aligned: true` on the three real runs; runs 4–5 reported missing |

## 8. Remaining Scientific Limitations
1. **No human-labeled calibration data exists** — all thresholds remain uncalibrated geometric heuristics; the gold set must be labeled (2 rounds + adjudication) before any operating point is meaningful.
2. **Historical runs lack execution-time alignment metadata** — addressed by Issue #9: retroactive, artifact-derived evidence is recorded in a separate `alignment_legacy.json` per run (provenance `retroactive`, classification `legacy-partial`); it is explicitly NOT execution-time metadata and never backfilled into `run_meta.json`. See section 9.
3. **Different text windows across models** (128 vs 512) are part of each model's configuration; a strict "identical truncation" study would require re-running all models at one window.
4. **Exact-match input grouping only** for leakage prevention (Phase-2 limitation, unchanged).
5. MentalChat16K contains almost no autism-specific terminology (Phase-1 finding) — comparison results measure category overlap with a general counseling corpus, not autism-specific performance.
6. "GPT-3.5 synthetic" provenance remains unverified locally (no network).

## 9. Historical Run Alignment Metadata (Issue #9)

### The alignment metadata contract for new runs

Every run created by `main.run_pipeline` records, at execution time, in
`run_meta.json`: `threshold`, `preprocess_version`, `text_window`
(effective per-model window), `dataset_fingerprint` (sha256 of the labeled
input texts **in row order**, first 16 hex chars — `main.py`), and
`dataset_total` (pre-validation total), alongside the model/prefix/
fact-lang/labeler-version fields historical runs already carried. The
canonical algorithms live in **one place** (`alignment.py`):
`ordered_input_fingerprint` (the `dataset_fingerprint` algorithm) and
`labeled_text_fingerprint` (the sorted-text algorithm
`--verify-alignment` has always used). There is no second implementation.

### Execution-time vs retroactive metadata

| Kind | Where | Classification |
|---|---|---|
| Recorded at execution time | `run_meta.json` (written by `run_pipeline`) | `modern` — verifiable |
| Reconstructed from persisted artifacts (Issue #9) | `alignment_legacy.json` (separate file, `provenance: "retroactive"`) | `legacy-partial` — **never** reported as execution-time verified |
| Nothing available | — | `legacy` |

The retroactive artifact contains: `labeled_csv_sha256` (artifact
identity), `labeled_text_fingerprint` (sorted-text canonical hash),
`ordered_input_fingerprint_derived` (row-order hash **derived from
labeled.csv** — clearly named as derived, because a hash of the labeled
artifact cannot by itself prove dataset-level provenance such as
pre-filtering records or `dataset_total`), `record_idx_fingerprint`,
`record_count`, and the execution-time fields the historical run did record
(`threshold` from its gap report; model/prefix/fact-lang/labeler-version
from its `run_meta.json` — with their source noted). Fields that cannot be
proven are listed under `missing` with reasons: `dataset_fingerprint_
execution_time`, `text_window`, `preprocess_version`, `dataset_total`.

### How historical (legacy) runs are identified

`python compare_runs.py --verify-alignment` classifies every run:
`modern` / `legacy-partial` / `legacy` / `MISSING_ARTIFACTS`, with
`alignment_summary` counts. Classification never raises on malformed or
non-object `run_meta.json` / `gap_analysis.json` (the error is reported in
the entry as `run_meta_error` / `gap_report_error`, type name only — no
absolute paths). `all_records_aligned` continues to be computed from the
labeled record-ID sets, so cross-run input equivalence remains auditable
for legacy runs; a missing threshold is reported as `null` with
`threshold_match` unset — never silently treated as matching.

### What the evidence cannot establish

A labeled.csv hash proves the identity of that artifact. It does not prove
the original dataset-level provenance of a historical run (pre-validation
records, ordering at load time, preprocessing version, effective text
window). Those fields remain `missing` in the retroactive artifact with
explicit reasons.

### Rerun procedure for fully verifiable results

```bash
# 1. retroactive artifacts for historical runs (idempotent; sources-fingerprinted)
python compare_runs.py --migrate-legacy-alignment
# 2. classification + cross-run record checks
python compare_runs.py --verify-alignment
# 3. fully verifiable re-run — into a NEW directory, preserving history:
python main.py --limit 0 --model minilm --fact-lang fa --dataset en     --out outputs/verified_1_fa_en_minilm
#    then confirm the metadata contract:
python -c "from alignment import classify_alignment; print(classify_alignment('outputs/verified_1_fa_en_minilm'))"
```

**Warning:** `python compare_runs.py --only <n>` re-executes a run AND
overwrites `outputs/comparison/<run-id>` — archive the historical
directory first if it must be kept (`--skip-existing` skips existing runs
instead of rerunning them).

**`all_records_aligned` semantics:** it means ONLY that the labeled
record-ID sets match the reference run. It is NOT dataset-level provenance
and NOT execution-time verification — the per-run classifications and
`all_runs_modern` carry that distinction (`legacy-partial` and `legacy`
runs never contribute to a modern-verified claim). Migration invalidation:
each retroactive artifact records sha256 fingerprints of ALL its sources
(`labeled.csv`, `run_meta.json`, `gap_analysis.json`; missing = `absent`)
— a change to any source regenerates the artifact on the next migration;
the artifact's own timestamp is excluded from the comparison.

Migrated runs remain classified `legacy-partial` forever — they are never
converted to `verified`. The comparison report
(`comparison_report.md`, section "Run alignment") marks every non-modern
run with its reason and the exact rerun command
(`python compare_runs.py --only <n> --limit 0`). Runs 1–3 have been
migrated (retroactive artifacts written; `labeled.csv` and `run_meta.json`
bytes untouched — sha256-verified) and verify as `legacy-partial` with
`all_records_aligned: true`.
