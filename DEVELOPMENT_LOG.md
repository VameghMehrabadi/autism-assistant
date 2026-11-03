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
  - Run 1 (Persian facts × **English MentalChat16K × MiniLM**) completed on the **full 16,005-record** dataset: 84.4% confident; top-1 distribution C 33.0%, F 26.1%, B 23.8%, A 10.2%, G 4.1%, E 1.5%, D 1.3%; ranking weak→strong D > E > G > A > B > F > C. Artifacts in `outputs/comparison/1_fa_en_minilm/` + `comparison_summary.{csv,json}` + `comparison_report.md` + chart.
  - Runs 2–3 (bge-m3 / e5-large) launched after model downloads completed (see final report for actual status).
  - Runs 4–5 remain **blocked** on the Persian dataset (requires Z.ai credentials → translation execution).
- **GUI verified in a real browser** (Streamlit headless + in-app browser): Home, Quick Label (live MiniLM inference: top-1 C 0.4240, multi-label C/B/D/A, decision badge, margin, cosine-not-probability captions), Results (run metadata + gap metrics). Fixed during the pass: Quick Label now renders **matched prototype evidence** (page passes `prototype_indices`). Screenshots in `gui-test-screenshots/`.
- **Documentation**: README updated (roadmap status, Z.ai usage, limitations); this log; final report with API cost analysis (`reports/PROJECT_FINAL_REPORT.md`).

## Known external blockers (final)
1. `ZAI_API_KEY` unavailable → Persian translation of the 16,005 records not executed (pipeline ready + quota-guarded).
2. Runs 4–5 of the comparison matrix depend on that translation.
3. Docker daemon not running in this environment → image build not exercised (Dockerfile/compose are pinned to the verified dependency set).
