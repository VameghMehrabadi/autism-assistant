# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

[![Python tests](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml/badge.svg)](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml)

A project to develop an LLM (using RAG or fine-tuning) for **autistic patients with language delay**.
This repository implements the data preparation stage: loading **MentalChat16K** (English or Persian translation),
semantically labeling samples against 7 autism fact categories (A–G), running **gap analysis**, and comparing
embedding models for reliability.

## Pipeline

1. `data_loader.py` — Load English `ShenLab/MentalChat16K` or a local Persian JSONL/CSV.
2. `facts.py` — Categories A–G with bilingual prototypes; filter with `en` / `fa` / `both`.
3. `labeler.py` — Multilingual embeddings + cosine similarity labeling.
4. `gap_analysis.py` — Distribution, score stats, weak/strong categories, chart.
5. `main.py` — Single-run orchestration.
6. `compare_runs.py` — Five-way model/language comparison.
7. `export_for_translation.py` + `prompts/semantic_persian_translation.txt` — Export EN records and translate to FA (GPT-4+).

## Supported embedding models

| Key | Hugging Face ID | Notes |
|-----|-----------------|-------|
| `minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Original task-1 model |
| `bge-m3` | `BAAI/bge-m3` | Stronger multilingual dense model |
| `e5-large` | `intfloat/multilingual-e5-large` | Uses `query:` / `passage:` prefixes |

## Five-way comparison matrix

| # | Facts | Dataset | Model |
|---|--------|---------|--------|
| 1 | English | Persian | MiniLM *(rerun; facts may change)* |
| 2 | English | Persian | bge-m3 |
| 3 | English | Persian | multilingual-e5-large |
| 4 | Persian | Persian | bge-m3 |
| 5 | Persian | Persian | multilingual-e5-large |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Translate dataset to Persian

```bash
# 1) Export English utterances for translation
python export_for_translation.py --limit 500

# 2) Translate each record with GPT-4+ using:
#    prompts/semantic_persian_translation.txt
#    Keep idx; write Persian text into data/mentalchat16k_fa.jsonl
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
# Original-style run (English dataset, both EN+FA facts, MiniLM)
python main.py --dataset en --fact-lang both --model minilm

# English facts × Persian dataset × bge-m3
python main.py --dataset fa --fact-lang en --model bge-m3 --out outputs/en_fa_bge-m3

# Persian facts × Persian dataset × e5-large
python main.py --dataset fa --fact-lang fa --model e5-large --out outputs/fa_fa_e5-large

# Limit / full dataset
python main.py --limit 100
python main.py --limit 0
```

## Run (all 5 comparisons, CLI)

```bash
# Requires data/mentalchat16k_fa.jsonl
python compare_runs.py --limit 500

# Smoke test with the sample fixture
python compare_runs.py --limit 5 --dataset-path tests/fixtures/mentalchat16k_fa.sample.jsonl --only 1

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
| G | Autism Knowledge & Awareness | 14 |

(Full facts are defined in `facts.py`. Expert-updated facts can replace this file when ready.)

## Notes

- Semantic matching is by embedding cosine similarity, not lexical overlap.
- Multi-prototype score per category = max similarity over that category’s selected-language prototypes.
- `multilingual-e5-large` prefixes samples as `query:` and facts as `passage:`.
- Gap analysis shows which autism categories are under-covered before RAG / fine-tuning.
