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
2. **Historical runs lack alignment metadata** — equivalence was verified empirically via content fingerprints, but `text_window`/`dataset_fingerprint` are recorded only for future runs (absence reported honestly, never backfilled).
3. **Different text windows across models** (128 vs 512) are part of each model's configuration; a strict "identical truncation" study would require re-running all models at one window.
4. **Exact-match input grouping only** for leakage prevention (Phase-2 limitation, unchanged).
5. MentalChat16K contains almost no autism-specific terminology (Phase-1 finding) — comparison results measure category overlap with a general counseling corpus, not autism-specific performance.
6. "GPT-3.5 synthetic" provenance remains unverified locally (no network).

## 10. Human-Labeled Gold Evaluation Set & Encoder-Specific Calibration (Issue #8)

### Gold dataset design (implemented)

- Selection: `python gold_split.py --size 300` carves **300 records**
  (within Issue #8's 200–500 range) from the 16,005-record EN export using
  the existing Phase-2 grouping machinery — records are selected and
  partitioned **whole-group** (297 groups), so related records can never
  leak across partitions. Seed (42), algorithm, dataset sha256
  (`a558909c0c62d05d…`), group keys and leakage verification are recorded
  in `data/gold/gold_split_manifest.json` (content-free, committed).
- Partitions: `gold_calibration` (105 records) for threshold selection,
  `gold_evaluation` (195 records) held out for final metrics. Verified
  disjoint at record and group level (`leak_free: true`).
- Worklist: `data/gold/gold_annotation_template.jsonl` — 300 pending
  records with stable `gold_id`, `source_idx`, the patient text, empty
  multi-label `labels`, `status: "pending"`, null `labeler_id`, and full
  provenance (record hash, group key, dataset sha256, seed). **Contains
  raw conversation text — gitignored by design, never committed.**

### Human annotation requirements

- Protocol: `data/gold/ANNOTATION_GUIDELINE.md` — operational A–G
  definitions (identical to `facts.py::CATEGORIES`), inclusion/exclusion
  criteria, multi-label rules (`UNSURE` stands alone), the empty-list
  "no applicable category" annotation, pseudonymous `labeler_id`, and the
  `origin: "human"` governance rule.
- Governance: `gold_eval.py` accepts **only** `origin == "human"`
  (model-generated labels are structurally rejected and counted); records
  with `status != "annotated"` are worklist entries, never gold labels;
  empty-list labels are deliberate "no category applies" annotations, not
  missing data; duplicates are rejected explicitly; `partition` and
  `provenance` are validated and preserved.
- **Zero human labels exist today.** The acceptance criterion "gold set
  with genuine human annotations" remains **blocked on human annotators**;
  nothing in this repository marks the template as annotated.

### Calibration methodology (implemented, pending human data)

- Per encoder: thresholds are selected by deterministic grid search
  (0.05–0.975, step 0.025) maximizing **macro F1 over categories with
  support > 0** on the `gold_calibration` partition only; ties resolve to
  the lower threshold. Per-category thresholds require ≥ 10 positives and
  ≥ 10 negatives — otherwise the category falls back to the global
  threshold with an explicit `insufficient_support` status.
- Sufficiency rule: fewer than 50 human-labeled calibration records →
  `calibrated: false` with the exact shortfall; model-origin records
  abort calibration outright.
- **Current status: `minilm`, `bge-m3`, `e5-large` are all
  UNCALIBRATED** — no human annotations exist, so the shared 0.35 default
  is the documented but **uncalibrated** default for every encoder. No
  threshold in this repository has been calibrated; none will be reported
  as calibrated until a validated human-labeled calibration partition
  exists (`data/gold/calibration.json`, written only by the calibration
  workflow).

### Evaluation metrics & uncertainty (implemented, pending human data)

- `evaluation_metrics.py`: per-category precision/recall/F1/support and
  confusion counts, micro P/R/F1, macro F1 (full universe — zero-support
  categories count as 0, reported with support), and a supported-categories
  macro used only as the calibration criterion. Uncertainty: **group-aware
  bootstrap** percentile intervals for macro F1 (groups resampled whole,
  1000 rounds, seeded) with an explicit refusal below 50 records / 30
  groups — a misleading interval is worse than none.
- Calibration vs held-out: calibration uses ONLY the `gold_calibration`
  partition; `gold_evaluation` metrics are computed separately (test-locked
  — held-out records cannot shift calibration thresholds). Until human
  labels exist, **all real evaluation metrics are pending**; the functions
  are tested against hand-computed fixtures only.

### Why cosine similarity is not probability

Cosine similarity is a geometric angle measure in [-1, 1] with no
probabilistic interpretation; a thresholded similarity is a decision rule,
not a confidence estimate, and threshold behavior differs per encoder
(e.g. e5-large concentrates scores near 0.80). Cross-encoder threshold
comparisons are meaningless without encoder-specific calibration on human
labels — which does not exist yet.

### Claims status

| Claim | Status |
|---|---|
| Gold selection (300 records, group-safe, reproducible) | Implemented + verified |
| Annotation protocol + unfilled template | Implemented |
| Gold validation (multi-label, origin enforcement, duplicates) | Implemented + tested |
| Calibration workflow (per encoder, sufficiency rules) | Implemented + tested on synthetic fixtures |
| Evaluation metrics + group-bootstrap uncertainty | Implemented + tested on fixtures |
| Genuine human-labeled calibration | **PENDING — blocked on human annotators** |
| Encoder-specific calibrated thresholds | **PENDING** |
| Held-out evaluation metrics on human labels | **PENDING** |
| Any accuracy/quality claim | **FORBIDDEN until human labels are committed** |
