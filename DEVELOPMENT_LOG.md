# Autism Assistant — Development Log

Chronological record of autonomous development phases. Detailed evidence lives in `reports/`.

## Phase 0 — Project Stabilization (commit 604b79c)
- Git initialized (`main`); baseline commit of Phases 1–3 work.
- Dependency graph pinned (`datasets==2.19.0`, `huggingface-hub==0.36.2`, `transformers==4.49.0`, `sentence-transformers==2.7.0`, `torch==2.15.0`, …) — fixes the reproducible fresh-install failure.
- CI hardened: explicit Python 3.12, pinned CPU torch, `pip check` gate, offline test guards.
- Fresh isolated venv install **tested**: 117/117 tests passed.
- Docs: README install sections, `reports/PHASE_0_STABILIZATION.md`.

## Phases 1–3 (pre-existing, commits 382502e era)
- Phase 1: repository audit reports.
- Phase 2: data engineering (validation, dedup, encodings, language ID, dev/eval leakage guard, privacy logging) — verified.
- Phase 3: labeling engine (E5 prefixes, cache, thresholds, uncertainty, matched prototypes, fact validation) — verified incl. a real MiniLM end-to-end run.

## Phase 6 — Persian translation pipeline (commit db04153)
- Adopted + repaired a parallel implementation (deep-merge data-loss bug, batch-straddle crash, missing import, 3 test bugs).
- Prompt reconciled to the engine protocol; `--confirm-external-sharing` privacy gate; mock→production guard; completeness check; `translation_summary.json`.
- Executed: full export **16,005 deduplicated records** (16,084 raw − 79 duplicates); mock smoke 100→250 with resume + zero duplicates; QA + loader compatibility verified.
- **Blocked**: real translation (no `ZAI_API_KEY`; `openai` package missing → installed later in 6b).

## Phase 6b — Z.ai (GLM-5.3-Flash) provider (commit 0067459)
- OpenAI-compatible integration (base_url, env creds, token accounting, `--max-total-tokens` quota guard, `--probe`).
- Verified offline + against a local OpenAI-compatible stub (auth/protocol/usage extraction OK).
- Pilot + full run **blocked** pending `ZAI_API_KEY`; explicit approval gate documented.

## Final completion phase (this log, commit see git)
- **Pipeline completion**: `run_pipeline` no longer mutates global `config.LABEL_TEXT_MODE` (explicit threading); CLI gained `--sampling`, `--seed`, `--split` (dev/eval leakage guard reachable from the command line); regression tests added (8/8 pipeline-integration tests).
- **Gap analysis v2**: reports now consume the Phase-3 signals — explicit `uncertain` decisions + rate, top-1/top-2 **margin** stats (labeled as heuristic), and **matched-prototype evidence per category** (fact→category mapping); Markdown section added; old CSVs degrade gracefully. 5/5 tests.
- **Model comparison executed (Phase 4)**:
  - Run 1 (Persian facts × **English MentalChat16K × MiniLM**) completed on the **full 16,005-record** dataset: 84.4% confident, 15.6% explicitly uncertain; top-1 distribution C 33.0%, F 26.1%, B 23.8%, A 10.2%, G 4.1%, E 1.5%, D 1.3%; ranking weak→strong D > E > G > A > B > F > C. Artifacts in `outputs/comparison/1_fa_en_minilm/` + `comparison_summary.{csv,json}` + `comparison_report.md` + chart.
  - Runs 2–3 (bge-m3 / e5-large × EN × 16,005) **completed** after their model downloads (bge-m3 pinned to a documented 512-token comparison window; the two 560M-model runs took multi-hour CPU compute). Unified 3-run summary regenerated (`--only 1,2,3 --skip-existing`):
    - Run 2 bge-m3: 100% confident, B 43.4% / C 19.0% / A 13.9%; ranking E > D > G > A > F > C > B.
    - Run 3 e5-large: 100% confident (cosine scores cluster near 0.80 — the 0.35 threshold is uncalibrated for e5), F 31.3% / G 30.6% / C 25.9%; ranking A > E > B > D > C > G > F.
    - Cross-model finding: materially different distributions and uncertainty rates per encoder at the same threshold — empirical confirmation that per-model threshold calibration on human-labeled data is required.
  - Runs 4–5 remain **blocked** on the Persian dataset (requires Z.ai credentials → translation execution).
- **GUI verified in a real browser** (Streamlit headless + in-app browser): Home, Quick Label (live MiniLM inference: top-1 C 0.4240, multi-label C/B/D/A, decision badge, margin, cosine-not-probability captions), Results (run metadata + gap metrics). Fixed during the pass: Quick Label now renders **matched prototype evidence** (page passes `prototype_indices`). Screenshots in `gui-test-screenshots/`.
- **Documentation**: README updated (roadmap status, Z.ai usage, limitations); this log; final report with API cost analysis (`reports/PROJECT_FINAL_REPORT.md`).

## Known external blockers (at the final-completion phase; unchanged since)
1. `ZAI_API_KEY` unavailable → Persian translation of the 16,005 records not executed (pipeline ready + quota-guarded).
2. Runs 4–5 of the comparison matrix depend on that translation.
3. Docker daemon not running in this environment → image build not exercised (Dockerfile/compose are pinned to the verified dependency set).

## Phase 1 (integrity) — Encoding fix & dataset provenance (commit 835c696)
- **Fixed** `detect_encoding`: a 65,536-byte sample ending mid-multibyte-character no longer misdetects large valid UTF-8 Persian JSONL as CP1256 (reproduced first: silent mojibake). Legitimate CP1256 files unaffected; small corrupt files still fail loudly.
- Dataset integrity verified read-only: raw sources **9,774 (Synthetic) + 6,310 (Interview) = 16,084** (matches HF metadata); export 16,005 records with 0 invalid JSON, 0 duplicate IDs, 0 Persian contamination; 32 gap ranges = exactly the 79 deduplicated records.
- Provenance: Interview_Data_6K = caregiver coaching from palliative/hospice content; Synthetic_Data_10K = general mental-health counseling; **autism terminology nearly absent dataset-wide** ("autism" ×5). Claimed composition (9,775+6,338=16,113) not reproducible from the distributed files — documented, nothing altered.

## Phase 2 (integrity) — Pipeline integrity & leakage prevention (commit 826b8ae)
- Loader: translation-status gate (only `translated` records enter labeling; flagged/failed/pending/rejected/empty_source/incomplete counted per status); `*_en` originals + `translation_status` + `flag_reason` preserved in `Sample.meta`; legacy status-less rows untouched.
- Splits: related records (same normalized patient input) now share a dev/eval split — group-aware `assign_splits`/`create_splits`, manifest `grouped_by: patient_input`; exact-match limitation documented.
- `translation_qa`: uncapped `idx_issue_codes` map — the 200-item display cap can no longer drop records from manual review; `issues_total` vs `issues_exported` reported.
- Export safety: `*.export_meta.json` sidecar (total/exported/complete), loud INCOMPLETE warning, exit code 3; `LoadReport.dataset_total`; default export is now the full dataset.

## Phase 3 (reliability) — Z.ai pipeline hardening (commit f82d17d)
- Precise HTTP status mapping: 400/404/405/422 permanent (fail fast), 401/403 auth (**whole-job abort**), 429 rate-limit (Retry-After), 5xx retryable.
- `finish_reason=length` rejected (`truncated`, no retry at temperature 0; batches fall back to singles); `--max-output-tokens` added.
- Thinking-mode tolerance (`<think>` stripping; unclosed think → invalid_response; `reasoning_content` ignored); empty translations for non-empty sources flagged.
- Checkpoint durability: per-record `fsync`; resume **repairs torn output lines** (corrupt original preserved as backup); embedding cache persisted **incrementally per 256-record chunk**.
- Centralized `UsageTracker`: actual vs conservative-estimated tokens tracked separately; `--max-total-tokens` guard counts both.

## Phase 4 (validity) — Evaluation methodology & comparison verification (commit 179d77a)
- Cross-run verification: all three comparison runs used identical record sets (identical labeled-text fingerprint), identical facts (unchanged since before the runs), identical 0.35 threshold; counts sum to 16,005 everywhere.
- `top_score_stats` (mean/median/p25/p75/min/max) added to every gap report — cosine scales are model-dependent; comparison Markdown carries explicit caveats (thresholds not comparable, higher scores ≠ better model, calibration requirement).
- `run_meta.json` now records threshold/preprocess_version/text_window/dataset_fingerprint/dataset_total; new `compare_runs --verify-alignment` (live result on the real runs: all_records_aligned=true; runs 4–5 reported missing).
- Gold evaluation set governance created (`data/gold/` + `gold_eval.py`): `origin: "human"` mandatory, two-round labeling + adjudication, versioning; **no labels fabricated** — empty set valid-but-empty.
- UI: research-use warnings on Home, Run Pipeline, Results; Quick Label/Compare captions.

## Phase 5 (release readiness) — GUI accuracy, Docker, documentation (this commit)

> **Correction (issue #1):** the Phase-5 report's release-readiness status was
> relabeled from `COMPLETE` to evidence-based statuses — the Docker image build
> and runtime smoke were `IMPLEMENTED — NOT EXECUTED`. The authoritative v1
> checklist now lives in `reports/PHASE_5_RELEASE_READINESS.md` §3; the
> Phase-5 execution record is preserved in its §6.
- Translate/Export page now uses the safe export path (`write_export`): COMPLETE/INCOMPLETE status shown in-UI, sidecar metadata displayed, translation-status panel added; default export is the full dataset.
- Docker compose: `ZAI_API_KEY` / `ZHIPU_API_KEY` / `ZAI_BASE_URL` forwarded to both services; **fixed pre-existing YAML typo** (`SENTENCE_TRANSFORMERS_HOME=` → `:` in the ui service); compose YAML-validated.
- Research-use disclaimers added to Quick Label and Compare Models pages.
- Consolidated `reports/PRIVACY_AND_SECURITY.md`; DEVELOPMENT_LOG extended; final report refreshed.

## Issue #14 (offline) — Model weight availability & prefetch workflow
- `config.MODEL_CACHE_DIR` (+ `MODEL_CACHE_SOURCE`): weights-cache resolution `AUTISM_MODEL_CACHE_DIR` > `SENTENCE_TRANSFORMERS_HOME` > `config.CACHE_DIR` (default keeps already-downloaded weights valid); distinct from `EMBED_CACHE_DIR` (computed vectors) — the two artifacts are never conflated.
- New `model_loader.py`: every SentenceTransformer load goes through `load_sentence_transformer`; missing/offline-unavailable/gated weights raise actionable `ModelWeightsMissingError`/`ModelLoadError` naming model key, HF id, expected cache path, offline state and the exact prefetch command (cause preserved). No other-encoder fallback, no hidden online retry; unrelated errors propagate unchanged. Classification verified against the REAL transformers stack error (plain `OSError` "couldn't connect ... couldn't find it in the cached files") — not only hub exception classes.
- New `prefetch_models.py` CLI: `--list` (registry + cache locations + approximate sizes), `--model <key|all>`; no dataset, no API key, no labeling/translation; failures reported per model with exit code 1 while still attempting the rest.
- `labeler.py` routes the single real model construction through the loader and prints the weights cache + offline state in its banner.
- docker-compose: additive `prefetch` one-off service (profile `prefetch`) writing into the existing `model_cache:/cache` volume (`SENTENCE_TRANSFORMERS_HOME=/cache/hf/sentence-transformers`); existing services/volumes untouched.
- REAL acceptance passed: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --limit 5 --model minilm --dataset en` — dataset served from local cache, MiniLM weights loaded from `D:\autism_assistant_cache`, 5 samples embedded and labeled end-to-end (zero hub requests possible under offline mode). Missing-model path verified with an empty cache dir: actionable error, exit 1. 274 tests green (26 new). Docker image build / volume-recreate verification BLOCKED — Docker daemon not running in this environment.

## Issue #15 (comparability) — Encoder confidence & raw score distribution
- Fixed Persian interpretation sentence (درصد اطمینان نشان‌دهنده سهم امتیازهای شباهت کسینوسی بالاتر از آستانه است و معادل دقت مدل نیست.) now sits next to every Confident% — per-run gap Markdown, five-way comparison report, and the Compare Models page (`CONFIDENCE_NOT_ACCURACY_FA`/`_EN` in `gap_analysis.py`).
- Raw top-score histogram per encoder (`outputs/comparison/score_distributions.png`): identical bins and shared x-axis for all models, dashed threshold line at 0.35, per-run threshold pass rate in each panel title; built from persisted `labeled.csv` raw scores (`plot_score_distributions`, `load_run_top_scores`).
- Comparison report gains a "Raw score distribution per encoder (uncalibrated)" section: n / mean / median / p05 / p95 / share ≥ threshold per run, automatic **threshold-saturation callout** when p05 ≥ threshold (bge-m3 and e5-large flagged on the real runs; MiniLM's threshold still discriminates), and an explicit no-winner rule — no run is ranked or declared best on these percentages (guard test).
- `top_score_stats` extended with p05/p95; legacy gap reports without stats get them derived from the persisted raw `top_score` column (marked as derived) — dashes when no raw data exists, never silently backfilled.
- UI snapshot test locks the warning into the Compare Models page; full suite 259 tests green; report regenerated from runs 1–3 (n=16,005 each) with the saturation visible.

## Issue #8 (gold) — Human annotation workflow, group-safe split, calibration & metrics
- New gold_split.py: carves the Issue #8 annotation split from the real 16,005-record EN export via the existing Phase-2 grouping machinery — whole-group selection and partitioning (seed 42; 300 records, 297 groups; calibration 105 / evaluation 195; manifest-verified leak-free at record and group level). Manifest (content-free, committed) records seed/algorithm/dataset sha256/partitions/verification; the per-record worklist (raw conversation text) is gitignored by design.
- Annotation protocol: data/gold/ANNOTATION_GUIDELINE.md (operational A–G definitions, inclusion/exclusion, multi-label rules, empty-list no-category annotation, UNSURE alone, pseudonymous labeler identity); gold README updated to the implemented selection.
- gold_eval.py strengthened: multi-label final_labels (XOR final_label; duplicates rejected; UNSURE alone; empty list = deliberate no-category annotation), status-aware accounting (pending worklist entries are never gold labels; a pending record with a broken schema still invalidates), partition/provenance validation. origin='human' enforcement unchanged.
- New calibration.py: per-encoder threshold grid search (maximize macro F1 over supported categories; ties to lower threshold; per-category requires >=10 positives/negatives; <50 human calibration records or any model-origin record -> calibrated: false with reasons). 0.35 reported UNCALIBRATED for every encoder — no human annotations exist.
- New evaluation_metrics.py: per-category P/R/F1/support + confusion, micro/macro F1, supported-categories macro (calibration criterion), group-aware bootstrap percentile CIs with an explicit insufficient-sample refusal. Verified against hand-computed fixtures.
- Comparison report: "Gold calibration & human evaluation status" section (human-labeled count 0, per-encoder 0.35 uncalibrated status, non-probability/non-diagnostic statements, no clinical-superiority claims). Quick Label caption: thresholds are encoder-specific, require human validation, 0.35 uncalibrated.
- 23 new tests (271 total green, synthetic only). PHASE_4 §10 documents design/requirements/methodology/status/claims table. ALL real evaluation, calibration, and accuracy claims remain PENDING on human annotators.
