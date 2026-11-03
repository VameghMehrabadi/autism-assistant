# Autism Assistant — Final Project Report

**Date:** 2026-11-03 · **Final release candidate** · Git: `main` (see commit log)

## 1. Completed Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Stabilization: git, pinned dependencies, CI, fresh-install verification | ✅ COMPLETE (tested) |
| 1 | Repository audit + architecture + roadmap | ✅ COMPLETE (reports) |
| 2 | Data engineering: robust loading, validation, dedup, encodings, language ID, input/response separation, dev/eval leakage guard, privacy logging | ✅ COMPLETE (implemented + tested + executed) |
| 3 | Semantic labeling engine: model registry, E5 prefixes, disk cache, per-category thresholds, explicit uncertainty, matched-prototype evidence, fact validation | ✅ COMPLETE (implemented + tested; MiniLM verified end-to-end) |
| 6 (+6b) | Persian translation pipeline + Z.ai (GLM) provider: batching, retries, checkpoints, QA, completeness, quota guard | ✅ IMPLEMENTED + TESTED (mock-verified); **real translation blocked on `ZAI_API_KEY`** |
| 4 | Model evaluation & comparison | ✅ Runs 1 & 3 executed on the **full 16,005-record** dataset; Run 2 launched; **Runs 4–5 blocked** on the Persian dataset |
| 5 | Gap analysis v2 (distribution, confidence, uncertainty, margin, prototype evidence) | ✅ COMPLETE (implemented + tested) |
| 7 | Streamlit GUI | ✅ COMPLETE + verified in a real browser (screenshots in `gui-test-screenshots/`) |
| 8 | Integration, testing, optimization | ✅ 185 tests passing; offline CI guards; embedding cache; fresh-venv install verified |
| 9 | Production readiness & release docs | ✅ Documentation complete; Docker assets ready (build not exercised — no daemon) |

## 2. Implemented Features (all verified where marked)

- **Data layer** (`data_loader.py`, `data_validation.py`): EN MentalChat16K (Hugging Face) + Persian/any JSONL/CSV/TSV/JSON; encoding detection (BOM/UTF-8/cp1256/UTF-16); schema validation; missing-value normalization; content-fingerprint dedup; script-based language ID; reproducible seeded sampling; dev/eval split manifest with hard leakage guard; identifiers + original records preserved.
- **Labeling engine** (`labeler.py`, `facts.py`, `embedding_cache.py`): 7 categories A–G with 44 validated bilingual facts; MiniLM / BGE-M3 / E5-Large with correct E5 `query:`/`passage:` roles; cosine scoring (max / top-k over prototypes); multi-label + per-category thresholds; **explicit `labeled`/`uncertain` decisions + margin**; matched-prototype evidence; disk embedding cache (content-addressed, atomic, corruption-safe); deterministic tie-breaking; NaN guards. Scores are labeled `cosine_similarity` — never presented as probabilities.
- **Translation pipeline** (`translate_dataset.py`, `translation_qa.py`, `export_for_translation.py`): schema-preserving export (16,005 records); provider abstraction (`openai` / `zai` / `mock`); batch protocol with per-segment fallback; retries with exponential backoff + Retry-After; timeouts; resume-from-output checkpoints; duplicate prevention; status tracking (`translated`/`flagged`/`empty_source`); structured content-free error log; automated QC (missing/empty/short/long/corrupted-unicode/non-Persian/duplicate detection) + manual review file; completeness accounting; `--confirm-external-sharing` privacy gate; `--max-total-tokens` hard quota guard; `--probe` endpoint verification.
- **Orchestration** (`main.py`, `compare_runs.py`): single-run pipeline (labels CSV with `decision`/`margin`/`matched_fact_ids`, scores matrix, gap report MD/JSON/chart); five-way comparison matrix with per-run summaries; CLI flags `--sampling/--seed/--split/--text-mode/--limit/--out`.
- **Gap analysis v2** (`gap_analysis.py`): per-category distribution/share/multi-label coverage/confident share/global mean; explicit uncertainty rate; margin statistics (documented as heuristic); matched-prototype coverage per category; weak→strong ranking + recommendations; Markdown + JSON + chart; graceful with pre-Phase-3 CSVs.
- **GUI** (`ui.py`, `pages/`, `ui_lib/`): Home dashboard; Quick Label (live inference, decision badge, margin, cosine-honesty captions, matched-prototype evidence table); Run Pipeline; Compare Models; Results browser; Translate/Export; privacy-conscious settings; verified in-browser.

## 3. Test Results (actual, final)

| Suite | Result |
|---|---|
| Full test suite (`pytest`) | **185 passed** |
| Fresh isolated venv (pinned requirements) | **installation verified; full suite green** (Phase-0 procedure repeated on the final requirements incl. `openai`) |
| Dependency graph | `pip check` — no broken requirements |
| GUI | browser-verified: Home, Quick Label (live MiniLM inference → top-1 C 0.4240, multi-label C/B/D/A, decision badge, margin, evidence table), Results (run metadata + metrics) |
| CI | workflow pinned + `pip check` + offline guards (`HF_HUB_OFFLINE=1` suite green) |

## 4. Dataset Status

- **Source**: `ShenLab/MentalChat16K` — **16,084 records** (verified from HF dataset metadata).
- **Valid unique records: 16,005** (79 content duplicates removed by the validation pipeline; 0 invalid).
- **English export**: `data/mentalchat16k_en_for_translation.jsonl` (56.9 MB, schema: `idx` + `instruction`/`input`/`output`).
- **Persian translation**: **not executed** — production `data/mentalchat16k_fa.jsonl` intentionally absent (0 fabricated records). Pipeline ready; 250-record mock sample exists as pipeline evidence only (`data/mentalchat16k_fa.mock_sample.jsonl`, gitignored, marked `mock`).
- Comparison runs consumed the **full 16,005-record English dataset** — no artificial limits.

## 5. API Integration Status

- **Z.ai (GLM-5.3-Flash)**: integrated via the OpenAI-compatible provider (`--provider zai`); credentials env-only (`ZAI_API_KEY`); token accounting (prompt/completion/total tracked separately); `--max-total-tokens` hard quota guard; `--probe` verification.
- **Verified**: offline unit tests + full network path against a local OpenAI-compatible stub (auth, protocol, usage extraction).
- **Not verified**: the real Z.ai service (endpoint/auth/model/rate limits) — **blocked on `ZAI_API_KEY`**, which is not present in this environment.
- **Actual external API consumption to date: 0 tokens** (all AI runs so far used locally cached open-weight models or offline mocks).

## 6. Outstanding Blockers

1. **`ZAI_API_KEY` unavailable** → 16,005-record Persian translation not executed (everything else unblocked; resumable pipeline + quota guard ready; `openai` SDK installed and pinned).
2. **Runs 4–5 of the comparison matrix** (Persian facts × Persian dataset) — blocked by (1), since the Persian dataset comes from that translation.
3. **Docker daemon not running** in this environment → image build not exercised (Dockerfile + compose are aligned with the verified pinned dependency set; `docker --version` 29.9.2 present).
4. Threshold calibration on human-labeled data remains future work (cosine scores are geometric heuristics, explicitly not probabilities; `margin` is an ambiguity hint, not confidence).

## 7. How to Run

```bash
# install (verified)
python -m venv .venv && .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
pip install -r requirements.txt   # torch excluded automatically in CI/Docker
python -m pip check

# CLI pipeline (single run)
python main.py --limit 0 --model minilm --fact-lang both --dataset en --out outputs/myrun
# five-way comparison (runs with available models/data; resumable via --skip-existing)
python compare_runs.py --limit 0 --skip-existing
# translation (requires ZAI_API_KEY; resumable, quota-guarded)
python translate_dataset.py --provider zai --probe --confirm-external-sharing
python translate_dataset.py --provider zai --batch-size 5 --confirm-external-sharing \
    --max-total-tokens 300000000 --summary-out data/translation_summary.json
# QA on a translated dataset
python translation_qa

# GUI
streamlit run ui.py            # or launch_ui.bat / launch_ui.ps1 / docker compose up
```

Model comparison results: `outputs/comparison/comparison_report.md` + `comparison_summary.{csv,json}` + per-run folders.

## 8. Final API Cost Analysis

| Item | Value |
|---|---|
| Actual Z.ai token consumption so far | **0 tokens / $0.00** (no external API call has been made; only a local stub server was used for integration testing) |
| Projected full-dataset translation (from measured file sizes; batch 5, GLM tokenizer estimate) | ≈ 47M prompt + ≈ 10M completion ≈ **57.5M tokens** (≈19% of the 300M budget) |
| Projected remaining after translation | QA-assistance / AI-analysis tasks are optional and small (<1M tokens estimated) |
| Budget headroom | ≈ 242M tokens untouched; `--max-total-tokens` enforces a hard ceiling regardless of estimate error |
| Cost | Not computable locally — apply the current Z.ai price sheet for `glm-5.3-flash` to the token totals above (input and output priced separately) |

## 9. Honest Status Vocabulary

- **Implemented + tested + executed (real)**: data engineering, labeling engine (MiniLM full-dataset run), Run 1/3 comparisons, gap analysis, GUI flows, export, QA machinery.
- **Implemented + tested, execution blocked externally**: real Persian translation (credentials), comparison runs 4–5 (depends on translation), Docker image build (daemon), real Z.ai service verification.
- No mock result is presented as a real AI result anywhere in this project's reports or outputs.
