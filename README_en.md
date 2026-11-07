# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

[![Python tests](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml/badge.svg)](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml)

A project to develop an LLM (using RAG or fine-tuning) for **autistic patients with language delay**.
This repository implements the data preparation stage: loading **MentalChat16K** (English or Persian translation),
semantically labeling samples against 7 autism fact categories (A–G), running **gap analysis**, and comparing
embedding models for reliability.

## Data licensing & redistribution (Issue #13)

- [ShenLab/MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)
  (reviewed 2026-11-07; the revision this project downloaded is
  `5f60cd380cfc58f0f12d44892bed41ee3670a70a` — still the current upstream
  `main` HEAD) **combines two components**: synthetic conversations
  (GPT-3.5 Turbo) and QA pairs paraphrased from interview transcripts of the
  PISCES clinical trial (palliative/hospice caregivers).
- The Hugging Face metadata carries a `license: mit` tag and the official
  repository ships an MIT LICENSE file — but **no source scopes that license**
  for the interview-derived component, nor explicitly authorizes translation,
  derivative creation, or derivative redistribution. The KDD paper's ACM
  copyright governs the *article*, not the dataset. Status: **UNCLEAR** —
  full assessment and permission matrix in
  `reports/PRIVACY_AND_SECURITY.md` (§8).
- Therefore `data/mentalchat16k_fa.jsonl` and every other derivative of this
  dataset **must not be committed or publicly redistributed**. The project's
  export paths **refuse by default** to write these records to a public
  destination (anywhere outside the project workspace); the
  `--acknowledge-license` override only records the user's acknowledgment of
  the unresolved risk — **it is not legal permission**.
- Citing the original dataset is not equivalent to obtaining redistribution
  permission. Private/local processing is never restricted.

## Project status (final release)

| Area | Status |
|---|---|
| Data engineering (validation, dedup, encodings, leakage guard, privacy logging) | ✅ implemented + tested (16,005 valid unique records exported) |
| Semantic labeling engine (E5 prefixes, disk cache, per-category thresholds, explicit uncertainty, matched-prototype evidence) | ✅ implemented + tested; MiniLM verified end-to-end on the full 16,005-record dataset |
| Model comparison | ✅ Runs 1–3 (MiniLM / bge-m3 / e5-large × EN × 16,005) **executed** — unified report in `outputs/comparison/`; runs 4–5 blocked on the Persian dataset |
| Gap analysis (distribution, uncertainty, margin, prototype evidence) | ✅ implemented + tested |
| Persian translation pipeline | ✅ implemented + tested (mock runs); **real translation blocked on `ZAI_API_KEY`** |
| Z.ai (GLM) provider integration | ✅ implemented + verified (offline + local live path); awaiting credentials |
| Streamlit GUI | ✅ implemented + verified in a real browser (labeling, results, evidence tables) |
| Tests | ✅ 174 passing (`pytest`); CI workflow in place |
| Docker image | ⚠️ Dockerfile/compose ready and pinned; build not exercised (no Docker daemon in the dev environment) |

Details: `reports/` (phase reports, final report, cost analysis), `DEVELOPMENT_LOG.md`.


## Pipeline

1. `data_loader.py` — Load English `ShenLab/MentalChat16K` or a local Persian JSONL/CSV.
2. `facts.py` — Categories A–G with bilingual prototypes; filter with `en` / `fa` / `both`.
3. `labeler.py` — Multilingual embeddings + cosine similarity labeling.
4. `gap_analysis.py` — Distribution, score stats, weak/strong categories, chart.
5. `main.py` — Single-run orchestration.
6. `compare_runs.py` — Five-way model/language comparison.
7. `export_for_translation.py` + `translate_dataset.py` + `prompts/semantic_persian_translation.txt` — Export and GPT-4+ Persian translation.

## Supported embedding models

| Key | Hugging Face ID | Notes |
|-----|-----------------|-------|
| `minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Original task-1 model |
| `bge-m3` | `BAAI/bge-m3` | Stronger multilingual dense model |
| `e5-large` | `intfloat/multilingual-e5-large` | Uses `query:` / `passage:` prefixes |

## Five-way comparison matrix — all runs use Persian facts

| # | Facts | Dataset | Model |
|---|-------|---------|--------|
| 1 | Persian | English | MiniLM *(rerun; facts may change)* |
| 2 | Persian | English | bge-m3 |
| 3 | Persian | English | multilingual-e5-large |
| 4 | Persian | Persian | bge-m3 |
| 5 | Persian | Persian | multilingual-e5-large |

> Runs 1–3 use the original English MentalChat16K; runs 4–5 use the GPT-4+ Persian translation.

## Install

Dependency versions are **pinned** in `requirements.txt` to a mutually
compatible, tested set (see `reports/PHASE_0_STABILIZATION.md`). Install the
CPU build of torch first, then the rest:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
grep -vE '^\s*torch([=<>]|$)' requirements.txt > /tmp/requirements.notorch.txt   # Windows: findstr /V /C:"torch" requirements.txt > requirements.notorch.txt
pip install -r /tmp/requirements.notorch.txt
python -m pip check             # verifies the dependency graph is consistent
```

A CUDA build of `torch==2.15.0` from the default PyPI index is drop-in
compatible for everything else in the stack. Unit tests run fully offline —
no model downloads and no API keys are required (`pytest`).

Engineering reports live in `reports/` (phase audits, architecture,
roadmap, dependency verification).

## Docker

```bash
cp .env.example .env    # optional HF_TOKEN / OPENAI_API_KEY
docker compose up --build
```

UI: [http://localhost:8501](http://localhost:8501). Models download on first use into the `model_cache` volume. Outputs land in `./outputs`, local datasets in `./data`.

```bash
# one-off CLI in the same image
docker compose run --rm pipeline main.py --limit 50 --model minilm --fact-lang fa --dataset en
```

## Translate dataset to Persian

```bash
python export_for_translation.py --limit 500
# optional if OPENAI_API_KEY is set:
# python translate_dataset.py --limit 500
# or translate manually with prompts/semantic_persian_translation.txt
# output: data/mentalchat16k_fa.jsonl
```

Schema for `data/mentalchat16k_fa.jsonl` (one JSON object per line):

```json
{"idx": 0, "text": "…Persian utterance…"}
```

A tiny smoke-test fixture lives at `tests/fixtures/mentalchat16k_fa.sample.jsonl`.

## UI (recommended launcher)

The project stays modular; use the browser UI to run stages without memorizing CLI flags:

```bash
# Windows
.\launch_ui.ps1
# or
launch_ui.bat
# or
streamlit run ui.py
# or
docker compose up --build
```

| Page | Purpose |
|------|---------|
| Home (`ui.py`) | Module overview + dataset status |
| Quick Label | Interactive single-text labeling |
| Run Pipeline | Full single-config labeling + gap analysis |
| Compare Models | Five-way model comparison matrix |
| Results | Browse outputs / reports / charts |
| Translate / Export | Export EN records + translation prompt |

UI code: `ui.py`, `pages/`, shared helpers in `ui_lib/`.

## Run (single configuration, CLI)

```bash
# Persian facts × English dataset × MiniLM  (run 1 of the matrix)
python main.py --dataset en --fact-lang fa --model minilm --out outputs/fa_en_minilm

# Persian facts × English dataset × bge-m3   (run 2)
python main.py --dataset en --fact-lang fa --model bge-m3 --out outputs/fa_en_bge-m3

# Persian facts × Persian dataset × e5-large (run 5)
python main.py --dataset fa --fact-lang fa --model e5-large --out outputs/fa_fa_e5-large

# Limit / full dataset
python main.py --limit 100
python main.py --limit 0
```

## Run (all 5 comparisons, CLI)

```bash
# Runs 1-3 use the English dataset (no FA file needed).
# Runs 4-5 require data/mentalchat16k_fa.jsonl.
python compare_runs.py --limit 500

# Smoke test with the sample fixture (only FA-dataset runs use it)
python compare_runs.py --limit 5 --dataset-path tests/fixtures/mentalchat16k_fa.sample.jsonl --only 4

# Skip runs that already have outputs
python compare_runs.py --skip-existing
```

Outputs:

- Single run → `outputs/` (or `--out` dir): `labeled.csv`, `scores.npy`, `gap_analysis.*`, `run_meta.json`
- Comparison → `outputs/comparison/`: per-run folders + `comparison_report.md`, `comparison_summary.csv`, `comparison_shares.png`

## Tests / CI

```bash
python -m pytest
```

GitHub Actions runs pytest on `push` / `pull_request` to `main`.

## Categories (A–G)

| Key | Title | #facts |
|-----|-------|--------|
| A | Social Communication | 6 |
| B | Sensory Processing | 3 |
| C | Emotional Regulation & Stress | 4 |
| D | Routine & Predictability | 3 |
| E | Special Interests & Strengths | 7 |
| F | Diagnosis & Support | 6 |
| G | Autism Knowledge & Awareness | 15 |

(Full facts are defined in `facts.py`. This is the expert-approved final set.)

## Notes

- Semantic matching is by embedding cosine similarity, not lexical overlap.
- Labeling uses patient text only by default (`LABEL_TEXT_MODE=patient`), not the counselor prompt/reply.
- Multi-prototype score per category = max similarity over that category’s selected-language prototypes.
- `multilingual-e5-large` prefixes samples as `query:` and facts as `passage:`.
- Gap analysis shows which autism categories are under-covered before RAG / fine-tuning.
