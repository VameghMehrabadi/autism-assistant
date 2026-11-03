# Phase 3: Semantic Labeling Engine — Technical Documentation

**Status: COMPLETE** — 99/99 tests passing (26 new engine tests, all previous suites untouched and green).

## 1. What was delivered

| File | Change |
|------|--------|
| `labeler.py` | Rewritten engine: text normalization, disk-backed embedding cache, batch inference, matched-prototype identifiers, per-category thresholds, explicit uncertainty, deterministic tie-breaking, NaN guards. All public names preserved (`resolve_model`, `apply_e5_prefix`, `SemanticLabeler`, `assign_label_row`, `format_labels`, `parse_labels`, `LABEL_SEPARATOR`, `UNLABELED`). |
| `facts.py` | Added `FactValidationError` + `validate_facts()` (runs at import: category validity, unique positive ids, non-empty bilingual texts, script sanity). Fact data and the seven categories A–G are unchanged. |
| `embedding_cache.py` | New: content-addressed disk cache for embeddings (atomic writes, lock-safe, transparent corruption recovery). |
| `main.py` | `labeled.csv` now also stores `decision` and `matched_fact_ids`; `run_meta.json` records `labeler_version`. |
| `config.py` | New keys: `LABELER_VERSION`, `EMBED_CACHE_ENABLED`, `EMBED_CACHE_DIR`, `EMBED_MAX_SEQ_TOKENS`, `CATEGORY_THRESHOLDS`. |
| `tests/test_labeler.py` | 7 existing tests kept + 20 new (multilingual, edge cases, cache, offline pipeline). |
| `tests/test_facts.py` | 6 new validation tests. |

## 2. The seven categories (unchanged)

`facts.py` still defines exactly A–G: Social Communication, Sensory
Processing, Emotional Regulation & Stress, Routine & Predictability, Special
Interests & Strengths, Diagnosis & Support, Autism Knowledge & Awareness —
44 facts × 2 languages = 88 prototypes. `test_final_fact_inventory` pins the
inventory; `validate_facts()` now also enforces integrity at import time
(fail fast on data corruption, e.g. an Arabic-script character inside an
English fact).

## 3. Model-specific preprocessing

| Model | Prefix | Notes |
|-------|--------|-------|
| `paraphrase-multilingual-MiniLM-L12-v2` | none | Symmetric similarity model; no instruction prefixes. |
| `BAAI/bge-m3` | none | bge-m3 needs no query/passage instruction; verified no prefix is added. |
| `intfloat/multilingual-e5-large` | `query: ` for dataset texts, `passage: ` for fact prototypes | E5 is trained asymmetrically; retrieval-style matching requires the correct role. Existing double-prefix guard retained. |

All texts pass through `normalize_for_embedding()` (strip + whitespace
collapse; Persian ZWNJ preserved) before embedding — deterministic and
covered by tests for Persian, English and empty inputs. Optional per-model
input truncation is available via `config.EMBED_MAX_SEQ_TOKENS`
(`None` = model default = previous behavior).

## 4. Embedding cache (`embedding_cache.py`)

- **Key**: SHA-256 of (schema version, model name, prefix mode, role,
  max-length, normalize flag, normalized text). Role is part of the key
  because E5 `query:`/`passage:` vectors differ for identical text.
- **Storage**: one `<model-slug>.npy` float32 matrix + `<model-slug>.index.json`
  (row order + metadata) under `config.EMBED_CACHE_DIR` (default
  `<CACHE_DIR>/embed_cache`).
- **Safety**: atomic writes (temp file + `os.replace`), `RLock` for
  Streamlit threads, and corruption handling that sets broken files aside
  as `*.corrupt` and rebuilds — the pipeline never fails because of the
  cache. Verified by test: a second labeler instance with the same cache
  performs **zero** encoder calls for already-seen texts and returns
  bit-identical vectors.
- Disable with `EMBED_CACHE_ENABLED = False` (verified: nothing is written).

## 5. Scoring & decision semantics

1. Cosine similarity matrix `samples × prototypes` (embeddings L2-normalized).
2. Category score = **max** similarity over the category's prototypes
   (`TOP_K_PROTOTYPES_PER_CAT > 1` → mean of top-k; the matched prototype
   remains the single argmax).
3. `score_samples(..., return_matches=True)` additionally returns, per
   sample and category, the index of the best-matching prototype; the
   decision stage maps those to `{category, fact_id, lang, score}` entries
   in `result["matched_prototypes"]` (accepted labels + the diagnostic top
   category). `main.py` persists them as `matched_fact_ids` (`fact:lang;...`)
   in `labeled.csv`.
4. `assign_label_row` produces a fully structured, self-describing result:
   - `top_label`, `top_score`, `score_vector` — cosine similarities in [-1, 1];
   - `labels` — accepted multi-label set (best first), possibly empty;
   - `confident` — top score ≥ top category's resolved threshold;
   - `decision` — `"labeled"` | `"uncertain"` (first-class outcome);
   - `margin` — top-1 minus top-2 score (ambiguity signal, heuristic);
   - `score_type: "cosine_similarity"`, `labeler_version`,
     `matched_prototypes`.
5. **Thresholds**: a single float, a per-category dict, or
   `config.CATEGORY_THRESHOLDS` overrides; missing keys fall back to
   `config.LABEL_THRESHOLD` (0.35). Resolution order is explicit param →
   per-category config → global default.

### Score honesty (cosine ≠ probability)

Scores are raw cosine similarities. `confident`/`decision` mean "crossed the
configured geometric threshold" — **not** a calibrated probability and **not**
a clinically validated confidence. No calibration set or human-labeled
ground truth exists in the repository yet, so no probabilistic
interpretation and no medical claim is attached to any label. Thresholds
must be calibrated against annotated data before production/clinical use;
the `margin` field is provided as a diagnostic for that future work, and the
docstrings, logs and this document state the limitation explicitly.

### Uncertainty & non-forcing

- Below-threshold inputs get `decision="uncertain"`; in multi-label mode
  `labels` is empty, in single-label mode it is `["NONE"]`.
- `top_label` is still reported as the diagnostic best guess with its
  matched prototype (useful for inspection) but never enters downstream
  label sets when uncertain. This behavior is pinned by
  `test_uncertain_inputs_never_forced_into_category` and the pre-existing
  `test_multi_label_below_threshold_stays_unlabeled`.
- NaN/inf score vectors raise `ValueError` instead of producing labels.
- Ties are broken deterministically (score descending, then lowest category
  index) via `np.lexsort`.

## 6. Reproducibility

- Deterministic tie-breaking; deterministic preprocessing; content-keyed
  cache (same text + settings ⇒ same vectors, same order).
- `labeler_version` recorded in every result row and in `run_meta.json`.
- Fake-encoder tests confirm cross-instance score equality
  (`test_results_reproducible_across_instances`).
- Remaining hardware caveat (documented): float nondeterminism across
  different GPU/CPU backends can perturb the last decimals; same
  machine + same backend is bit-stable.

## 7. Test results (actual)

```
tests\test_compare_config.py ....                                        [  4%]
tests\test_data_loader.py .........................                      [ 29%]
tests\test_data_validation.py ..........................                 [ 55%]
tests\test_facts.py ...............                                      [ 70%]
tests\test_gap_analysis.py ..                                            [ 72%]
tests\test_labeler.py ...........................                        [100%]
99 passed in 22.85s
```

New engine tests run **fully offline** through `FakeEncoder`, a deterministic
`SentenceTransformer`-compatible stand-in that records prepared texts — this
is how E5 prefix behavior, cache hit/miss accounting, and multilingual
pipelines are verified without downloading model weights.

## 8. Known limitations

- No calibration dataset exists, so thresholds remain geometric heuristics.
- `max_seq_length` overrides are plumbed but default to each model's own
  setting to preserve comparability with earlier runs.
- The cache is per-machine local storage; it does not validate that the
  SentenceTransformer library version changed (model name + settings are
  the key; bump `EMBED_CACHE_ENABLED=False` or the schema constant if the
  library's embedding behavior changes).
- Fake-encoder tests exercise the engine logic, not the real models'
  multilingual quality — that requires the 5-run comparison on real
  weights (out of scope for this phase; no performance numbers are claimed).

## 9. Remaining tasks

- Optional: expose `--threshold`, `--no-cache` CLI flags in `main.py`.
- Future phase: build a small human-labeled evaluation set to calibrate
  thresholds and report real classification metrics.
