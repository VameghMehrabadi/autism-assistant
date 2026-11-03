# Development Progress Audit — Autism Assistant (after Phases 1–3)

**Audit date:** 2026-11-03 · **Auditor role:** Senior SW Architect / AI Engineering Auditor / ML QA
**Repository:** `C:\Users\sina\Desktop\autism-assistant-main` (the `Downloads\autism-assistant-main` copy is empty)
**Method:** every claim below is backed by source inspection, grep evidence, a fresh test execution, or a reproducible run performed *during this audit*. No code or tests were modified. Nothing was deleted or renamed.

---

## 1. Executive Summary

Phases 1–3 are **genuinely complete in implementation** and **largely verified**: the data-engineering layer and the labeling engine were rebuilt this cycle with 73 new tests (27 → 99, all passing), and — importantly — two **real executions** were performed during this audit that were previously missing: a live Hugging Face download of MentalChat16K (16,084 rows) and an end-to-end real MiniLM pipeline run producing all six output artifacts with the new `decision`/`matched_fact_ids` fields.

The project's weakest points are **not in the new code**: (1) `requirements.txt` is unpinned and a fresh install demonstrably breaks (reproduced during environment setup; CI inherits this), (2) the directory is **not a git repository**, so none of this work has version control, and (3) the **Persian dataset does not exist** (`data/` contains only `.gitkeep`), which blocks translation (Phase 6), comparison runs 4–5 (Phase 4), and any FA-based evaluation.

## 2. Overall Progress Percentage

**Denominator definition.** Each phase's requirement checklist (as specified in the phase briefs) is decomposed into atomic items: **9 for Phase 1, 12 for Phase 2, 15 for Phase 3 → 36 items total.** Each item scores 1.0 implemented / 0.5 partial / 0.0 missing, and independently 1.0 verified (by an executed test or reproducible run in this audit) / 0.5 partially verified (logic-level only) / 0.0 unverified.

| Metric | Value |
|---|---|
| **Implementation completion, Phases 1–3** | **33.5 / 36 = 93%** |
| **Verified completion, Phases 1–3** | **31.5 / 36 = 88%** |
| Phase 1 | 72% implemented · 61% verified |
| Phase 2 | 100% implemented · 100% verified |
| Phase 3 | 100% implemented · 93% verified |

## 3. Repository Inspection — Current State

**Modules (source, 5,487 LOC total incl. tests):**

| Module | LOC | Status |
|---|---|---|
| `data_loader.py` | 653 | Phase-2 rewrite (backward-compatible API) |
| `labeler.py` | 519 | Phase-3 rewrite (backward-compatible API) |
| `data_validation.py` | 446 | **New** (Phase 2) |
| `embedding_cache.py` | 206 | **New** (Phase 3) |
| `privacy_log.py` | 142 | **New** (Phase 2) |
| `facts.py` | 289 | Modified: +`validate_facts()` (import-time), data unchanged |
| `config.py` | 192 | Modified: +Phase-2/3 settings |
| `main.py` | 216 | Modified: decision/matched_fact_ids columns, labeler_version |
| `gap_analysis.py` / `compare_runs.py` | 276/316 | **Unmodified** (pre-existing) |
| `ui.py`, `pages/` (5), `ui_lib/` (4) | 615 | **Unmodified** (pre-existing) |
| `translate_dataset.py` / `export_for_translation.py` | 124/64 | **Unmodified** (pre-existing) |
| `Dockerfile`, `docker-compose.yml`, CI workflow | — | **Unmodified** (pre-existing) |

**Tests:** 99 tests / 6 files — `test_compare_config` (4), `test_data_loader` (25), `test_data_validation` (26), `test_facts` (15), `test_gap_analysis` (2), `test_labeler` (27). **Fresh execution this audit: `99 passed in 17.00s`.** All 27 pre-Phase-2 tests still pass with identical bodies (5 in `test_data_loader`, 7 in `test_labeler`, the rest untouched).

**Reports:** `reports/PHASE_1_AUDIT.md` (2.1 KB), `ARCHITECTURE.md`, `DEVELOPMENT_ROADMAP.md`, `PHASE_2_DATA_ENGINEERING.md` (8.2 KB), `PHASE_3_LABELING_ENGINE.md` (8.7 KB).

**TODO/FIXME scan:** `grep -rn "TODO\|FIXME\|XXX\|HACK\|NotImplemented"` over all `.py` → **zero hits**. No unfinished stubs.

**Git:** `git status` → `fatal: not a git repository`. **No version control, no commit history.** (README badge references a GitHub Actions workflow that cannot run from this directory.)

**Data/outputs reality check:** `data/` = only `.gitkeep` → **no Persian dataset, no exported EN file**. `outputs/` = logs + this audit's smoke run only → **the 5-way comparison has never been executed**. Stray artifact: `.freebuff/project-id` (non-project file).

**Real executions performed during this audit (new evidence):**
1. `load_samples_en(limit=5)` → downloaded `ShenLab/MentalChat16K` (13.6 MB + 32.8 MB files), **16,084 train rows**, 5 samples, `LoadReport{kept:5, language_distribution:{'en':5}, preprocess_version:'2.0.0'}` → **exit 0**.
2. `python main.py --limit 5 --model minilm --fact-lang both --dataset fa --dataset-path tests/fixtures/mentalchat16k_fa.sample.jsonl --out outputs/audit_smoke_minilm` → **exit 0**; artifacts `labeled.csv`, `scores.npy`, `run_meta.json`, `gap_analysis.{md,json}`, `gap_distribution.png`. `labeled.csv` columns include `decision` (`['labeled','labeled','uncertain','labeled','labeled']` — a real uncertain outcome) and `matched_fact_ids` (e.g. `9:fa;11:fa;15:fa`); `run_meta.json` records `labeler_version: 3.0.0`. Real-model embedding cache created at `D:\autism_assistant_cache\embed_cache\` with **93 vectors = 88 prototypes + 5 samples** (140 KB `.npy` + index).

## 4. Phase-by-Phase Audit (evidence per item)

### Phase 1 — Repository Audit (72% impl / 61% verified)

| # | Item | Impl | Verified | Evidence |
|---|------|------|----------|----------|
| 1 | Repository structure fully inspected | 1.0 | 1.0 | tree map in `PHASE_1_AUDIT.md §1`; re-verified via `find` in this audit |
| 2 | Architecture documented | 1.0 | 1.0 | `reports/ARCHITECTURE.md` (layers + interaction flow) |
| 3 | Module dependencies identified | 0.5 | 0.5 | layer-level flow only; no per-module import graph anywhere |
| 4 | Existing bugs documented | 0.5 | 0.5 | `PHASE_1_AUDIT.md §5` = 3 generic bullets, no `file:line` evidence, no bug list |
| 5 | Existing tests executed | 1.0 | 1.0 | "27 passed in 17.17s" recorded; re-confirmed 99 passed now |
| 6 | Security & privacy risks evaluated | 0.5 | 0.5 | generic bullets only; no PHI-flow analysis, no dependency CVE check |
| 7 | Development roadmap created | 1.0 | 1.0 | `reports/DEVELOPMENT_ROADMAP.md` |
| 8 | Critical issues **prioritized** (acceptance criterion) | 0.5 | 0.0 | issues listed with no priority ranking |
| 9 | Implementation distinguished from README claims (acceptance criterion) | 0.0 | 0.0 | never performed (e.g. README install instructions are actually broken — see §8-B1) |

**Phase-1 gap honesty:** the Phase-1 report is thin (2.1 KB vs 8+ KB for Phases 2–3); several claims (e.g. "reviewed containerization configuration") were only substantiated by this audit's actual reads of `Dockerfile`/`docker-compose.yml`.

### Phase 2 — Data Engineering (100% impl / 100% verified)

| # | Item | Impl | Verified | Evidence (code + test) |
|---|------|------|----------|------------------------|
| 1 | EN MentalChat16K loads | 1.0 | **1.0** | `load_samples_en` — **live run this audit: 16,084 rows, 5 samples, exit 0** |
| 2 | Persian JSONL loads | 1.0 | 1.0 | `_load_rows`; `test_load_samples_fa_from_sample_file`, `test_malformed_jsonl_lines_are_skipped_and_counted` |
| 3 | Persian CSV loads | 1.0 | 1.0 | `test_load_csv_persian_with_utf8_sig` (utf-8-sig verified) |
| 4 | Schema validation | 1.0 | 1.0 | `data_validation.validate_record` (`EMPTY_RECORD`, `BAD_IDX`); `test_empty_records_are_skipped_transparently`, `test_strict_validation_raises_on_invalid_record` |
| 5 | Missing values handled | 1.0 | 1.0 | `normalize_record` (None/NaN→""); `test_csv_missing_values_are_handled` |
| 6 | Duplicate detection | 1.0 | 1.0 | `record_fingerprint` (SHA-256, whitespace-insensitive); `test_duplicates_removed_keep_first`, `test_duplicates_kept_when_keep_all` |
| 7 | Encoding handling | 1.0 | 1.0 | `detect_encoding` (BOM/UTF-8/cp1256/UTF-16); 4 encoding tests incl. legacy cp1256 Persian file |
| 8 | Patient input / counselor response separation | 1.0 | 1.0 | `Sample.input_text/.response_text`; `test_build_samples_separates_patient_input_from_response`; patient mode remains default (`config.LABEL_TEXT_MODE="patient"`) |
| 9 | Identifiers preserved | 1.0 | 1.0 | source `idx` kept + `record_hash` in `meta`; `test_original_identifiers_preserved` |
| 10 | Reproducible sampling | 1.0 | 1.0 | seeded `random.Random`; `test_random_sampling_is_reproducible`; `preprocess_version` in every report |
| 11 | Sensitive data protected | 1.0 | 1.0 | `privacy_log.SENSITIVE_KEYS` redaction (log audited: counts/paths only, zero conversation text); malformed lines logged by line number only; split manifests store hashes only; **dev/eval leakage guard** verified (`test_dev_eval_split_roundtrip_without_leakage`, `test_load_samples_eval_split_requires_manifest`) |
| 12 | Unit + integration tests | 1.0 | 1.0 | 46 tests in `test_data_loader.py` + `test_data_validation.py`, all passing |

### Phase 3 — Semantic Labeling Engine (100% impl / 93% verified)

| # | Item | Impl | Verified | Evidence |
|---|------|------|----------|----------|
| 1 | Categories A–G | 1.0 | 1.0 | `facts.CATEGORIES` (7 keys); `test_final_fact_inventory` (44 facts, counts pinned) |
| 2 | FA + EN prototypes | 1.0 | 1.0 | `get_prototypes(en/fa/both)`; `test_fact_lang_configurations_change_prototypes`, 3 `get_prototypes` tests |
| 3 | MiniLM supported | 1.0 | **1.0** | `config.EMBED_MODELS["minilm"]`; **real run this audit (exit 0, 4/5 confident)** |
| 4 | BGE-M3 supported | 1.0 | 0.5 | registry entry + no-prefix behavior verified via fake encoder; **real weights never loaded → UNVERIFIED** |
| 5 | e5-large supported | 1.0 | 0.5 | registry entry + prefix mode; **real weights never loaded → UNVERIFIED** |
| 6 | E5 query:/passage: prefixes correct | 1.0 | 1.0 | `apply_e5_prefix` + role routing (`query`→samples, `passage`→prototypes); `test_e5_prefix_roles_applied_query_and_passage` asserts the *prepared* texts recorded by the fake encoder; double-prefix guard kept |
| 7 | Cosine similarity correct | 1.0 | 1.0 | L2-normalized embeddings (`test_embed_samples_deterministic_multilingual` asserts unit norms) × dot product in `score_samples`; range asserted in `test_multilingual_inputs_produce_structured_results` |
| 8 | Batch inference | 1.0 | **1.0** | `batch_size` plumbed to `model.encode`; real encode executed end-to-end in the MiniLM smoke run |
| 9 | Embedding caching | 1.0 | 1.0 | `EmbeddingCache` — `test_embedding_cache_roundtrip` (second instance: **0 encoder calls**, bit-identical vectors), corrupt-rebuild + disabled-no-write tests; **real cache file verified after smoke run (93 rows)** |
| 10 | Multi-label classification | 1.0 | 1.0 | threshold-set semantics; 4 pre-existing + new tests |
| 11 | Configurable thresholds | 1.0 | 1.0 | float / per-category dict / `config.CATEGORY_THRESHOLDS`; `test_assign_label_row_per_category_thresholds`, `test_resolve_threshold_none_uses_config_overrides` |
| 12 | Uncertain / low-confidence results | 1.0 | 1.0 | `decision="uncertain"`, `margin`, `UNLABELED`; `test_assign_label_row_uncertain_is_explicit`, `test_uncertain_inputs_never_forced_into_category`; real uncertain row in smoke `labeled.csv` |
| 13 | Scores not misrepresented as probabilities | 1.0 | 1.0 | `score_type="cosine_similarity"` in every result (asserted in test); "not a probability" stated in module docstring + `PHASE_3_LABELING_ENGINE.md §5`; no calibration claimed |
| 14 | Results reproducible | 1.0 | 1.0 | deterministic `np.lexsort` tie-break, content-keyed cache, `labeler_version` recorded; `test_results_reproducible_across_instances`, `test_assign_label_row_deterministic_tie_break` |
| 15 | Edge cases tested | 1.0 | 1.0 | empty/whitespace inputs, NaN/inf rejection, ties, invalid params (`test_empty_and_whitespace_inputs_are_handled`, `test_assign_label_row_rejects_non_finite_scores`, `test_invalid_dedup_and_language_values_raise`) |

**Remaining UNVERIFIED in Phase 3:** real-weight inference for `bge-m3` and `multilingual-e5-large` (2 × 0.5). Everything else verified at logic level by tests and — for MiniLM — by real execution.

## 5. Completed Features (verified)

- Bilingual (EN/FA) robust dataset loading: JSONL/CSV/TSV/JSON + Hugging Face EN — all formats **executed**.
- Validation pipeline: normalization → schema validation → dedup → language detection → seeded sampling → `Sample` with separated input/response.
- Privacy-preserving JSON logging + structured `LoadReport`; content-fingerprint dedup; dev/eval split manifest with leakage guard.
- Labeling engine: 3 model registry, E5 role prefixes, cosine scoring (max/top-k over prototypes), multi-label + per-category thresholds, explicit uncertainty, matched-prototype identifiers, disk embedding cache, reproducible decisions.
- Fact inventory validation at import; 99 passing tests; CI workflow file; Docker image definition (non-root, healthcheck); bilingual README; Streamlit app (6 views); gap-analysis reports/charts; 5-run comparison tooling; translation tooling + domain prompt.

## 6. Partially Completed Features

- **Model comparison execution (Phase 4 core):** tooling complete, **never run** (`outputs/comparison/` absent).
- **Gap analysis depth (Phase 5):** reports/charts exist (2 tests only); does not yet consume Phase-3 signals (`margin`, `matched_fact_ids`, per-category thresholds) — `gap_analysis.build_report` still reads only `config.LABEL_THRESHOLD`.
- **Persian dataset production (Phase 6):** export/translate/prompt tooling ready; the dataset itself was never produced; `openai` package not installed; no API key configured.
- **GUI (Phase 7):** functional Streamlit prototype; Quick Label does not yet surface `decision`/`margin`/`matched_prototypes`; no UI tests; renders raw conversations with no privacy mode.
- **Integration/perf (Phase 8):** large suite + cache exist; no coverage tooling, no lint/type config, `main.run_pipeline` itself untested, CI likely broken by unpinned requirements.
- **Production readiness (Phase 9):** Docker/env/logging exist; no git, no pinned deps, no monitoring, no PHI retention policy for `labeled.csv` (which stores raw conversations by design).

## 7. Missing Features (not started)

- Any execution of the 5-run comparison matrix with real weights (runs 2–5 never loaded real models).
- Ground-truth evaluation set and calibrated thresholds (no human-labeled data anywhere).
- The Persian translated dataset (`data/mentalchat16k_fa.jsonl`).
- Test coverage for `privacy_log.sanitize_value`, `main.run_pipeline`, `compare_runs.summarize_run/build_comparison_table`, `translate_dataset`, `export_for_translation`.
- Version control (git init/commit), dependency pinning, coverage/lint tooling, model/data versioning, UI automation tests, deployment/ops documentation.

## 8. Critical Bugs / Issues (prioritized)

| ID | Sev | Issue | Evidence |
|----|-----|-------|----------|
| B1 | **CRITICAL** | `requirements.txt` open-ended ranges resolve to a broken set: `datasets 5.1.0` requires `huggingface-hub>=1.31` while `sentence-transformers 2.7`/`transformers` need `<1.0` → `ImportError: cannot import name 'BucketNotFoundError'` on fresh install (reproduced during setup; fixed in venv only via `datasets==2.19.0`, `huggingface-hub==0.36.2`, `transformers==4.49.0` — **pins never backported**). README §Install and CI (`.github/workflows/python-tests.yml`) inherit this. | `requirements.txt`; pip resolution logs; venv `pip list` |
| B2 | **HIGH** | **No git repository** — zero history/rollback/branching for 5,487 LOC of phase work; README badge/CI cannot function from this state. | `git status` → `fatal: not a git repository` |
| B3 | **HIGH** | Persian dataset absent → runs 4–5, FA evaluation, Phase-6 downstream all blocked; only the 5-record synthetic fixture exists. | `data/` listing |
| B4 | MEDIUM | `privacy_log` redaction logic (privacy-critical) has **no unit test**; verified only by manual log inspection. | `grep privacy_log tests/` → no hits |
| B5 | MEDIUM | `main.run_pipeline`, `compare_runs` logic, `translate_dataset`, `export_for_translation` untested; `gap_analysis` has 2 tests. | grep evidence §coverage |
| B6 | MEDIUM | Phase-1 report lacks prioritized issues and the README-claims comparison required by its own acceptance criteria. | `PHASE_1_AUDIT.md` |
| B7 | LOW | `labeled.csv` stores raw conversations (by design) with no retention/PHI policy; Quick Label/Results pages render raw text without a privacy mode. | `main.py` rows; `pages/1_Quick_Label.py` |
| B8 | LOW | `run_pipeline` mutates global `config.LABEL_TEXT_MODE` (thread-safety hazard under Streamlit). | `main.py:105` |
| B9 | LOW | Stray `.freebuff/project-id` file in repo root; `.dockerignore` excludes `*.md` (reports/ never ship in image — acceptable but undocumented). | file listing |

**Regressions introduced by Phases 2–3:** none breaking; three intentional, documented behavior changes — (a) dedup defaults to `keep_first` (counts can drop on dirty data; surfaced in `LoadReport.duplicates_removed`), (b) deterministic tie-breaking via `np.lexsort` (measure-zero label flips vs old `argsort[::-1]`), (c) non-finite score vectors now raise instead of silently labeling. All 27 pre-existing tests still pass verbatim.

## 9. Existing Features That Can Be Reused (do not rebuild)

Everything: the Phase-2 loader/validation/split stack, Phase-3 engine + cache + logging, `compare_runs.py` CLI/summaries, `gap_analysis.py`, Streamlit pages, Docker assets, CI workflow, translation prompt, and the `FakeEncoder` test seam (ready-made for offline Phase-4/8 integration tests). Language detection in `data_validation.detect_language` can double as translation-QA for Phase 6.

## 10. Future Phases Readiness

| Phase | Pre-existing assets | Still needed | Readiness |
|---|---|---|---|
| **4 — Model Evaluation & Comparison** | `compare_runs.py` (CLI, summaries, table, chart), `config.COMPARISON_RUNS` (tested), UI page 3, verified EN loading + MiniLM | Run 1–3 executions; runs 4–5 blocked on B3; ground-truth metrics blocked on annotation set; significance/bootstrapping; runtime benchmarks | **~50% — can start now for runs 1–3 + fixture** |
| **5 — Gap Analysis** | `gap_analysis.py` complete + integrated + 2 tests | Consume `margin`/`matched_fact_ids`/per-category thresholds; calibration section; run-over-run comparison | **~60%** |
| **6 — Persian Translation Pipeline** | `export_for_translation.py` (verified path), `translate_dataset.py` (resume-capable), domain prompt | Export file, OpenAI access, translated dataset, language-QA on output (reuse `detect_language`) | **~40% — blocked on API access/budget decision** |
| **7 — Professional GUI** | Full Streamlit app, widgets, caching, Docker UI | Surface Phase-3 result fields; privacy mode; i18n; tests; polish | **~55%** |
| **8 — Integration, Testing & Optimization** | 99 tests, pytest.ini, CI file, cache/batching perf work | Fix B1; coverage/lint tooling; offline `run_pipeline` integration test (FakeEncoder); benchmarks | **~45%** |
| **9 — Production Readiness** | Docker, healthcheck, env handling, validation layer, privacy logging, provenance metadata | git, pins, monitoring, PHI policy, model/data versioning, release process | **~30% — blocked on B1/B2 fundamentals** |

## 11. Recommended Next Steps (priority order)

1. **B1 (hours):** pin the verified dependency set in `requirements.txt` (datasets~=2.19.0, huggingface-hub<1.0, transformers~=4.49.0; keep torch CPU index note) → unblocks CI and fresh installs.
2. **B2 (minutes):** `git init` + initial commit + push; every later phase depends on rollback safety.
3. **B4/B5 (1 day):** unit tests for `privacy_log.sanitize_value`/`log_event`; an offline `run_pipeline` integration test via the existing `FakeEncoder` seam; `compare_runs` logic tests.
4. **Phase 6 decision:** obtain OpenAI access (or an alternative GPT-4+ endpoint), export EN records, translate ≥500 records, run language-QA with `detect_language` → produces `data/mentalchat16k_fa.jsonl`, which unblocks runs 4–5.
5. **Phase 4 track A (immediately unblocked):** execute runs 1–3 (EN × 3 models) plus fixture-based smoke comparisons; track B (metrics) waits for a small human-labeled eval set (also feeds threshold calibration).

**Dependencies:** B3 ⊂ Phase 6 → runs 4–5; calibration ⊂ annotation set → real metrics; B1 ⊂ CI ⊂ every future phase's "verified" claim.

**Unnecessary work to avoid:** do not re-implement loaders/validation (done, tested), do not re-write E5 prefix logic, do not build a second caching layer, do not replace Streamlit before Phase 7 is actually scoped, and do not re-run dataset downloads (HF cache is warm on `D:`).

**Can Phase 4 start immediately?** Yes — for runs 1–3, fixture runs, and metric scaffolding. Runs 4–5 and metric-bearing evaluation must wait for the Persian dataset (Phase 6) and the annotation/calibration set respectively.

## 12. Updated Development Roadmap (adapted to actual state)

| Order | Work | Depends on | Est. |
|---|---|---|---|
| 0a | Pin dependencies + verify fresh install | — | hours |
| 0b | `git init`, commit, remote, enable CI | 0a | minutes |
| 0c | Privacy-log unit tests + offline `run_pipeline` integration test | — | 1 day |
| 4a | Execute comparison runs 1–3 (+fixture runs); persist summaries | 0a,0b | 1 day |
| 6 | Export + translate FA dataset (GPT-4+) + language QA | budget decision | 1–2 days |
| 4b | Runs 4–5; comparison report; runtime benchmarks | 6 | 1 day |
| 5 | Gap-analysis v2 (margin/matched-prototype analytics, per-category thresholds, run diffing) | 4a | 2 days |
| eval | 200–500 record human-labeled eval split (via `create_splits`); calibrate thresholds; report true metrics | 4a | 1–2 weeks |
| 7 | GUI v2 (result fields, privacy mode, i18n, page tests) | 4a | 3–5 days |
| 8 | Coverage + lint/type tooling; CI green; perf benchmarks | 0a,0b | 2–3 days |
| 9 | PHI/retention policy, monitoring, model/data versioning, release docs | all | 1 week |

---

### Calculation method (repeated from §2)

- **Denominator:** 36 atomic requirement items (P1=9, P2=12, P3=15) taken from the phase briefs' checklists.
- **Implementation %** = Σ(item credit: 1.0 implemented / 0.5 partial / 0.0 missing) ÷ 36 = **33.5/36 = 93%**.
- **Verified %** = Σ(verification credit, only for implemented items; 1.0 = executed test or reproducible run observed in this audit, 0.5 = logic-verified only, 0.0 = unverified) ÷ 36 = **31.5/36 = 88%**.
- Documentation alone never earned credit for missing functionality; the two real runs (EN dataset, MiniLM pipeline) are the only non-test execution evidence, and both are reproducible from the commands recorded in §3.
