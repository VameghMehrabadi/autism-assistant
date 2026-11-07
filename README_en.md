# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

[![Python tests](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml/badge.svg)](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml)

A project to develop an LLM (using RAG or fine-tuning) for **autistic patients with language delay**.
This repository implements the data preparation stage: loading **MentalChat16K** (English or Persian translation),
semantically labeling samples against 7 autism fact categories (A–G), running **gap analysis**, and comparing
embedding models for reliability.

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

| Key | Hugging Face ID | Notes | Approx. download* |
|-----|-----------------|-------|-------------------|
| `minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Original task-1 model | ~0.5 GB |
| `bge-m3` | `BAAI/bge-m3` | Stronger multilingual dense model | ~2.3 GB |
| `e5-large` | `intfloat/multilingual-e5-large` | Uses `query:` / `passage:` prefixes | ~2.3 GB |

\* Approximate fp32 weight sizes from the Hugging Face model cards of the
pinned models — treat them as estimates, not exact numbers.

## Model cache, prefetch & offline runs (Issue #14)

Three distinct on-disk artifacts — do not confuse them:

| Artifact | Location | Purpose |
|----------|----------|---------|
| **Model weights** (`models--*`) | `config.MODEL_CACHE_DIR` | Downloaded encoder weights used by inference |
| **Embedding cache** | `config.EMBED_CACHE_DIR` (`CACHE_DIR/embed_cache`) | Computed sentence vectors (speeds up re-runs; safe to delete) |
| **Pipeline outputs** | `outputs/` | `labeled.csv`, reports, charts per run |

The weights cache resolves in this order: `AUTISM_MODEL_CACHE_DIR` (override)
> `SENTENCE_TRANSFORMERS_HOME` (standard variable) > `config.CACHE_DIR`
(`AUTISM_CACHE_DIR` → `D:\autism_assistant_cache` if a `D:` drive exists,
otherwise `<project>/.cache`). Set any of these **before** launching Python —
huggingface_hub reads its configuration at import time.

```bash
# Inspect the registry, cache locations and approximate sizes
python prefetch_models.py --list

# Prefetch one model (no dataset, no API key, no labeling)
python prefetch_models.py --model minilm

# Prefetch every registered model
python prefetch_models.py --model all
```

**Offline runs.** After a successful prefetch, set the offline switches and
run the pipeline normally — weights load from the local cache and no network
request is attempted (any hub request would fail loudly under
`HF_HUB_OFFLINE=1`):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --limit 5 --model minilm
```

**Missing-model errors.** If weights are absent (or offline without cache),
loading fails with an actionable message naming the model key, the Hugging
Face ID, the expected cache location, whether offline mode is enabled, and the
exact prefetch command — instead of a bare network stack trace. Diagnosis:
1. Run `python prefetch_models.py --list` and check the cache location line.
2. Verify that folder contains `models--<org>--<name>` for your model.
3. Run the printed `python prefetch_models.py --model <key>` command while
   online, then retry offline.
The loader never falls back to a different encoder and never retries online
in offline mode.

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

# one-off weight prefetch into the persistent model_cache volume (no dataset needed)
docker compose run --rm prefetch                 # all registered models
docker compose run --rm prefetch --model minilm  # one model
```

**Cache persistence.** All services set `AUTISM_CACHE_DIR=/cache` and mount
the named volume `model_cache` at `/cache`; inside the container the weights
cache resolves to `SENTENCE_TRANSFORMERS_HOME=/cache/hf/sentence-transformers`
on that volume, so downloaded weights survive container recreation. Host-side
prefetch (no env vars) puts weights in the host's `config.MODEL_CACHE_DIR`
instead — to reuse host-prefetched weights in the container, mount that host
folder at `/cache/hf/sentence-transformers`. Model weights (`models--*`),
the embedding cache (`/cache/embed_cache`) and pipeline outputs
(`./outputs`) are separate artifacts.

> Volume persistence is documented from the compose configuration; an actual
> build/recreate verification is pending Docker daemon availability
> (Issue #14 acceptance evidence tracked in the PR).

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
