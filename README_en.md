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
| Docker image | ⚠️ **IMPLEMENTED — NOT EXECUTED**: Dockerfile/compose ready and pinned; YAML-validated, but the image build has never completed (see the release checklist) |

Details: `reports/` (phase reports, final report, cost analysis), `DEVELOPMENT_LOG.md`.

**Release status:** the authoritative v1 readiness checklist — including the
four outstanding blockers (Z.ai-funded translation, comparison runs 4–5,
Docker build + runtime smoke, human gold-data calibration) — lives in
[`reports/PHASE_5_RELEASE_READINESS.md`](reports/PHASE_5_RELEASE_READINESS.md)
§3. The project is **not** ready for a `v1` tag until every item there is
checked with linked evidence.


## Pipeline

1. `data_loader.py` — Load English `ShenLab/MentalChat16K` or a local Persian JSONL/CSV.
2. `facts.py` — Categories A–G with bilingual prototypes; filter with `en` / `fa` / `both`.
3. `labeler.py` — Multilingual embeddings + cosine similarity labeling.
4. `gap_analysis.py` — Distribution, score stats, weak/strong categories, chart.
5. `main.py` — Single-run orchestration.
6. `compare_runs.py` — Five-way model/language comparison.
7. `export_for_translation.py` + `translate_dataset.py` + `prompts/semantic_persian_translation.txt` — Export and LLM Persian translation (provider configurable via `--provider`).

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

> Runs 1–3 use the original English MentalChat16K; runs 4–5 use the LLM-translated Persian dataset (provider configurable; see translate_dataset.py).

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

### Quick start with the launch scripts (Windows)

The `launch_ui.bat` and `launch_ui.ps1` scripts automate the full setup:
venv creation, CPU-only PyTorch installation, dependency installation,
`pip check`, and Streamlit launch. They skip installation on subsequent
runs when dependencies are already present.

```bat
launch_ui.bat        (double-click or run from cmd)
.\launch_ui.ps1      (PowerShell)
```

**Model cache location:** embedding models and datasets are cached in
`config.CACHE_DIR` (default: `D:\autism_assistant_cache` if a `D:` drive
exists, otherwise `<project>\.cache`). Override with the `AUTISM_CACHE_DIR`
environment variable. The directory is created automatically on first use.

**Common failures:**
- `pip check` fails → dependency graph is inconsistent; delete `.venv` and re-run the launcher.
- PyTorch install fails → verify network access to `download.pytorch.org`.
- Streamlit not found → the venv exists but dependencies are missing; delete `.venv` and re-run.

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

## Raw conversation retention & deletion (Issue #10)

Generated artifacts that can contain raw conversation text (labeled CSVs,
translation exports, QA review files) live in `outputs/<run>/`,
`outputs/labeled.csv` (default run) and `data/mentalchat16k_*` /
`data/translation_*.jsonl`. **Default retention policy: 30 days for
temporary generated raw run artifacts — a documented default, not an
enforced mechanism; deletion is always an explicit operator action.**

```bash
python cleanup_runs.py --list                        # addressable locations
python cleanup_runs.py --run <location>              # exact preview
python cleanup_runs.py --run <location> --confirm    # explicit deletion
python cleanup_runs.py --run translation-artifacts --confirm
```

Or in the UI: **Results ▸ Retention & cleanup** (preview → confirm
checkbox → delete). Deletion is whitelist-based: only recognized run
artifacts of the selected location are removed; export sidecars are
removed with their export; original datasets, other runs, shared caches,
model weights, gold data and anything outside `outputs/`/`data/` are never
touched; every confirmed deletion writes a metadata-only audit record to
`outputs/logs/deletion_audit.jsonl`. Conversation previews in Results are
truncated by default with an explicit per-record reveal (a privacy
affordance, not a security boundary). Full inventory and limitations
(SSD wear-leveling, backups, snapshots): `reports/PRIVACY_AND_SECURITY.md` §8.
## Storage security / encryption at rest (Issue #11)

**Architecture: operator-managed encrypted host storage.** Every file the
pipeline writes (datasets, labels, translation outputs, QA review files,
reports, charts, logs, caches, temp/backup files) lives under `data/`,
`outputs/` or the cache root — so OS-level volume encryption (BitLocker /
LUKS / FileVault) covers all of them. The application deliberately contains
**no crypto code**; it cannot detect disk encryption and never pretends to.

```bash
# research mode (default — current workflow, explicit warnings):
python main.py --limit 50
# patient-data mode (fails closed without the declaration):
DATA_STORAGE_MODE=patient-data STORAGE_PROTECTION_DECLARATION=bitlocker python main.py --limit 50
```

| Mode | Behavior |
|------|----------|
| `research` (default) | Files written **unencrypted** unless the operator protects the volume; UI banners + CLI lines warn; unsuitable for PHI without separate, documented storage protection |
| `patient-data` | Requires `STORAGE_PROTECTION_DECLARATION` (`bitlocker` \| `luks` \| `filevault` \| `encrypted-volume`) — an **operator declaration, not an application-verified property**; missing/invalid configuration fails closed (exit code 2 / UI refuses content) |

- Protected paths (encrypt the volumes behind them): `data/`, `outputs/`
  (incl. `outputs/logs`), the cache root (`AUTISM_CACHE_DIR`), and any HF
  cache location (`HF_HOME`/`HF_HUB_CACHE` or `~/.cache/huggingface`).
- Docker: bind mounts inherit the host filesystem; **named volumes are NOT
  encrypted by Docker** (they live under the daemon's root, e.g.
  `/var/lib/docker/volumes`) — protect that host filesystem. Model-weight
  caching is not patient-data protection.
- OS verification is the operator's job: `manage-bde -status <drive>:`
  (Windows), `cryptsetup status` (Linux), `fdesetup status` (macOS).
- Existing unencrypted files are never auto-migrated; enable encryption on
  the volume or move copies manually. Git exclusion (`.gitignore`) is not
  encryption. No HIPAA/clinical-readiness claim is made or implied.
## Deployment & access — local vs public (Issue #12)

**Default = localhost only.** The UI binds to the loopback interface
(`.streamlit/config.toml`) and the Docker host port publishes on
`127.0.0.1` — nobody on the LAN/VPN/clinic network can open it. The
container itself listens on `0.0.0.0` internally (required for Docker port
forwarding); exposure is decided by the host-side published address, not by
the container.

```bash
# local mode (default — no credential needed, loopback only)
docker compose up --build            # -> http://127.0.0.1:8501 on this machine only
# or without Docker:
python -m streamlit run ui.py
```

**Public mode is explicit and requires a credential.** Setting
`BIND_PUBLIC=1` without a password fails closed (compose aborts; the app
refuses to serve content). Public exposure requires all three, deliberately:

```bash
BIND_PUBLIC=1 APP_AUTH_PASSWORD='choose-a-long-passphrase'   docker compose -f docker-compose.yml -f docker-compose.public.yml up --build
```

- `APP_AUTH_PASSWORD` (or `.streamlit/secrets.toml` → `[auth] password`,
  ≥ 8 chars in public mode) unlocks the single shared operator login.
  The value is never logged, hardcoded, or stored in session state.
- The login gate (`ui_lib/auth.py`, constant-time comparison) sits before
  any content on the home page and every sub-page; **Logout** clears it.
- **XSRF is not authentication** — it protects against cross-site request
  forgery, not against strangers opening the page.
- **This is not multi-user isolation**: one credential, one shared
  workspace — conversations, labels, `data/`, `outputs/` are common to all
  users of the instance.
- **Remote/clinical access should use TLS**: put an authenticated,
  TLS-terminating reverse proxy (e.g. Caddy `basic_auth`) in front and keep
  the Streamlit port loopback-only on the proxy host. The app serves plain
  HTTP — never expose it raw over an untrusted network.
- To stop public exposure: re-run plain `docker compose up -d` (loopback
  only) and remove the override file from the command.
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
- Comparison → `outputs/comparison/`: per-run folders + `comparison_report.md`, `comparison_summary.csv`, `comparison_shares.png`, `score_distributions.png`

> **Reading the Confident% column (Issue #15):** it is the share of raw cosine
> similarity scores above the threshold — it is **not** model accuracy, and
> cosine scales differ per encoder (e5-large concentrates near 0.80, so the
> shared 0.35 threshold passes everything). `comparison_report.md` carries the
> fixed Persian/English warning, per-encoder raw-score histograms on shared
> axes, and no "best model" ranking.

## Tests / CI

```bash
python -m pytest
```

GitHub Actions runs pytest on `push` / `pull_request` to `main`.

## Operational validation (manual — real Z.ai and Docker)

Offline automated tests (248 tests, `pytest`) verify pipeline logic without
real API calls. Operational validation exercises the **actual** Z.ai
endpoint and **actual** Docker build — these require manual triggering and
external services.

### How to trigger

Go to **Actions → Operational Validation → Run workflow** on GitHub.
Select which validations to run (Z.ai probe, Docker build, or both).

**Prerequisites:**
- `ZAI_API_KEY` must be configured as a GitHub Actions **repository secret**
  (Settings → Secrets and variables → Actions). Never commit it to the repo.
- Docker build requires no additional secrets.

### What each validation does

| Validation | What it does | Requires |
|---|---|---|
| Z.ai probe | Sends one benign test sentence to `api.z.ai` to verify authentication, model access, and response protocol | `ZAI_API_KEY` secret |
| Docker build | Builds the project image using `docker compose build ui` | Docker daemon on the runner |
| Docker healthcheck | Starts the container and waits for `/_stcore/health` → 200 | Docker build successful |

### Status meanings

| Status | Meaning |
|---|---|
| `SUCCESS` | Validation passed with execution evidence |
| `BLOCKED` | Authenticated/started but an external dependency prevented completion (e.g. code 1113 insufficient balance) |
| `SKIPPED` | Not requested or credentials not configured |
| `FAILED` | Attempted but failed (auth, network, build, etc.) |

### Where to find results

- Workflow logs: **Actions → Operational Validation → [run number]**
- Job summary: displayed at the top of each workflow run
- Issue #17: the workflow automatically posts a summary comment with the
  latest results

### What is NOT covered

- Real Persian translation of dataset records (requires manual `translate_dataset.py` execution)
- Model quality evaluation (requires gold-labeled calibration data)
- Clinical or diagnostic accuracy (this is a research prototype)

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
