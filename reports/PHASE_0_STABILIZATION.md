# Phase 0: Project Stabilization Report

**Status: COMPLETE** — all six tasks executed with evidence; 117/117 tests passing in both the working environment and a **freshly built isolated venv**.

## 1. Git Initialization (Task 1)

- Preflight: `git --version` → 2.55.0.windows.3; `git rev-parse --is-inside-work-tree` → `fatal: not a git repository` (confirmed no repo — nothing overwritten); global identity already configured.
- Sensitive-file scan: no `.env`, no keys, no secrets anywhere; only the empty `.env.example` template (tracked intentionally). `data/` contains no datasets (synthetic fixture in `tests/fixtures/` only); `outputs/` artifacts and logs are generated data.
- Added `.freebuff/` (local tool artifact) to `.gitignore`. Existing `.gitignore` already excluded: `.venv/`, `.env`/`.env.*` (except example), `data/*` + `outputs/*` (with `.gitkeep` exceptions), model/dataset caches, `__pycache__`, `.pytest_cache`, logs, IDE/OS files, `.streamlit/secrets.toml`.
- `git init -b main` → staged **52 files** (source, tests, reports, config); verified the staged list contains no virtualenv, cache, output, or tool-state files.
- **Baseline commit `382502e`** — "Phase 0 baseline: Phases 1-3 source, tests (99 passing), reports". **No push performed** (no remote configured).

## 2. Dependency Stabilization (Task 2)

### Resolution of the known incompatibility

The failure mode (reproduced during earlier setup): `pip install -r requirements.txt` with open ranges resolved to `datasets 5.1.0` + `huggingface-hub 2.4.0`, which is incompatible with `sentence-transformers 2.7.0`/`transformers` (require `huggingface-hub<1.0`) → `ImportError: cannot import name 'BucketNotFoundError'`.

`requirements.txt` is now pinned to the exact working set (candidate versions confirmed against the live environment, not assumed):

```
datasets==2.19.0  huggingface-hub==0.36.2  sentence-transformers==2.7.0
transformers==4.49.0  torch==2.15.0  numpy==2.5.4  tqdm==4.70.1
matplotlib==3.11.2  pandas==3.1.0  pytest==9.2.0  streamlit==1.66.1
```

### Graph verification (executed, not assumed)

- `pip check` → **"No broken requirements found"** (working venv *and* fresh venv).
- Constraint edges verified against installed package metadata:
  - `datasets → huggingface-hub >=0.21.2` → 0.36.2 OK
  - `transformers → huggingface-hub <1.0,>=0.26.0` → 0.36.2 OK
  - `sentence-transformers → huggingface-hub >=0.15.1` → 0.36.2 OK
  - `sentence-transformers → transformers <5.0.0,>=4.34.0` → 4.49.0 OK
  - `datasets → fsspec <=2024.3.1` → 2024.3.1 OK · `datasets → dill <0.3.9` → 0.3.8 OK
  - `pandas → numpy >=2.0.2` → 2.5.4 OK
- Transitive packages (scipy 1.18.1, scikit-learn 1.9.1, tokenizers 0.21.4, fsspec 2024.3.1, …) are left to the resolver and validated by `pip check`; no lock-tool dependency was added (Rule 6).

### CPU/GPU compatibility

- **CPU: verified.** `torch==2.15.0+cpu` ran the real MiniLM end-to-end pipeline (Phase-3 audit smoke, exit 0) and all 117 tests.
- **CUDA: untested** (no GPU exercised in this environment). A CUDA build of `torch==2.15.0` from the default PyPI index is drop-in for the rest of the stack; documented in `requirements.txt` comments.

### Fresh installation in an isolated environment — TESTED

Executed (not claimed): fresh venv at `%TEMP%\phase0_fresh_venv`, mirroring CI exactly:

1. `python -m venv` + pip upgrade
2. `pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"`
3. `pip install -r requirements.txt` (torch excluded via the same grep CI uses)
4. `python -m pip check` → **"No broken requirements found"**
5. `python -m pytest` from the repository → **`117 passed in 18.26s`**

Fresh-venv installed versions match the pins exactly (`datasets 2.19.0, huggingface_hub 0.36.2, transformers 4.49.0, torch 2.15.0+cpu, numpy 2.5.4, pandas 3.1.0, streamlit 1.66.1, pytest 9.2.0, matplotlib 3.11.2`). PyPI returned transient read-timeouts during resolution; pip retries succeeded — installs on flaky networks may be slow but complete.

## 3. Critical Test Coverage (Task 3)

18 new tests in 3 files (synthetic fixtures only; no real conversations):

| File | Tests | Covers |
|---|---|---|
| `tests/test_privacy_log.py` | 10 | `fingerprint` (non-reversible, deterministic, 12-hex, length-preserved); `sanitize_value` redacts `SENSITIVE_KEYS` case-insensitively, fingerprints >`LOG_MAX_INLINE_LEN` strings even on safe keys, passes short safe scalars through, recurses into dicts/lists; `sanitize_fields` end-to-end redaction; `log_event` writes one sanitized JSON line (`event`, ts, level) with **no raw content**; plain-log `message` path; `get_logger` caching; `LOG_JSON=False` → NullHandler, nothing persisted |
| `tests/test_pipeline_integration.py` | 4 | `main.run_pipeline` offline via the `FakeEncoder` seam injected through `main.SemanticLabeler`: full artifact set (`labeled.csv` with `decision`/`matched_fact_ids`/`score_A..G`, `scores.npy` shape (5,7) finite, `run_meta.json` with `labeler_version`, `gap_analysis.{json,md}`, `gap_distribution.png`); `limit=3` honored; **run-to-run reproducibility** (`assert_frame_equal` + `np.array_equal` across two output dirs); empty-dataset → `RuntimeError("No samples loaded")` |
| `tests/test_compare_runs.py` | 4 | `summarize_run` field extraction + `mean_top_score`; `build_comparison_table` shape/values (`count_*`/`share_*`/`multi_*`/`mean_*` per category); `render_comparison_markdown` sections/ids; `selected_runs("1,3")` filtering + invalid-input `ValueError` |

## 4. CI Stabilization (Task 4)

`.github/workflows/python-tests.yml` updated:

- Python **explicitly pinned: 3.12** (all pins support 3.11–3.13; 3.13 proven locally, Dockerfile uses 3.11).
- Install order made reproducible: pinned CPU `torch==2.15.0` from the PyTorch CPU index first, then pinned `requirements.txt` minus torch.
- Added **`python -m pip check`** gate — a broken graph now fails CI instead of surfacing as an import error later.
- Test step runs with **`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`** — proves and enforces that basic unit tests require **no model downloads** and **no secrets** (workflow has no `secrets:` usage).
- `workflow_dispatch` added for manual runs.

Offline guarantee proven locally: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest` → **117 passed in 14.20s**.

## 5. Regression Testing (Task 5)

| Environment | Result |
|---|---|
| Working `.venv` | **117 passed in 15.93s** |
| Working `.venv`, offline env guards | **117 passed in 14.20s** |
| **Fresh isolated venv** (from pinned requirements) | **117 passed in 18.26s** |

Breakdown: 99 pre-existing tests (all pass, bodies unchanged: 4 compare-config, 25 data-loader, 26 data-validation, 15 facts, 2 gap-analysis, 27 labeler) + 18 new Phase-0 tests.

Verified behaviors per the task checklist: data-loader backward compatibility (`test_merge_fields_*`, `test_extract_label_text_*`, `test_load_samples_fa_from_sample_file`), labeler behavior + multi-label + uncertain decisions (`test_assign_labels_multi_label`, `test_multi_label_below_threshold_stays_unlabeled`, `test_assign_label_row_uncertain_is_explicit`, real `uncertain` row in the audit smoke run), embedding cache (`test_embedding_cache_roundtrip` — zero encoder calls on second instance; real 93-vector cache file), privacy logging (10 new tests + log audit), pipeline integration (4 new offline `run_pipeline` tests).

## 6. Documentation (Task 6)

- `requirements.txt` — pinned with in-file rationale (torch index instructions, bump-as-a-graph rule).
- `.gitignore` — added `.freebuff/`; verified exclusions for env/secrets/caches/outputs/data.
- `README.md` + `README_en.md` — Install sections rewritten: pinned-set note, CPU-torch-first procedure (Windows + POSIX), `pip check` step, CUDA note, offline-tests statement, pointer to `reports/`.
- This report: `reports/PHASE_0_STABILIZATION.md`.

## Remaining Limitations

- **CI has not actually run on GitHub** — the repo has no remote (Rule 3: no push). The workflow mirrors the locally proven steps (identical install commands + pip check + offline pytest); first push will validate it.
- CUDA path untested (CPU-only machine); `torch==2.15.0` from PyPI (CUDA) is assumed compatible per NVIDIA wheel semantics, not executed here.
- Transitive dependencies are resolver-chosen and `pip check`-validated, not hash-locked; a full lock file (pip-tools/uv) would be stricter and was deliberately not added (Rule 6 — no unnecessary dependencies).
- bge-m3 / multilingual-e5-large still unverified with real weights (Phase 4 scope); Persian dataset still absent (Phase 6 scope).
- CI uses Python 3.12 while local verification used 3.13 (Docker: 3.11) — all pins span 3.11–3.13; a CI matrix is possible later if needed.

## Artifacts

- Commits: `382502e` (baseline) + Phase-0 stabilization commit (this report, pins, CI, tests, docs).
- Working tree after final commit: clean; branch `main`; no remote, nothing pushed.
