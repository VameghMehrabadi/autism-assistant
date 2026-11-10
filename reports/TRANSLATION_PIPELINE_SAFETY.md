# Translation Pipeline Safety & Validation (Issue #6)

**Status: infrastructure ready — the real Persian translation has NOT
happened yet.** Runs 4 and 5 remain blocked until
`data/mentalchat16k_fa.jsonl` is produced by a real translation provider.
Nothing in this document claims that translation occurred.

## 1. Mock vs real Persian datasets

| | Mock pilot (`mentalchat16k_fa.mock_sample.jsonl`) | Real dataset (`mentalchat16k_fa.jsonl`) |
|---|---|---|
| Producer | `--provider mock` (`mock-echo`) | A real provider (e.g. Z.ai GLM) with `--confirm-external-sharing` |
| Content | English echo of the source (**0 Persian characters**) | Genuine Persian translation |
| QA | 750/750 fields flagged `NOT_PERSIAN` | Must pass `translation_qa.py` |
| Eligible for runs 4/5 | **Never** | Only after alignment validation passes |

The mock file is never implicitly loaded: `dataset_guard.py` refuses it by
filename **and** by provenance, and `translate_dataset.py` refuses to write
the production Persian path from the mock provider.

## 2. Required dataset provenance

A real Persian dataset must contain, per record:

- stable `idx` matching the English source export one-to-one;
- `instruction` / `input` / `output` in Persian (non-empty exactly where the
  English source is non-empty — speaker roles and record boundaries
  preserved), with the English originals preserved under `*_en` keys;
- `status`: `translated` / `flagged` / `empty_source`;
- `model` identifying the real provider model (mock model ids —
  `mock-echo`, `mock`, `mock-budget` — are structurally rejected).

## 3. Translation validation workflow

```bash
python compare_runs.py --verify-translation-alignment
# exit 0 = ready; exit 2 = structural failures or incomplete coverage
python compare_runs.py --only 4,5   # runs the same gates before embedding
```

The comparison runner enforces, **before any embedding generation**:

1. `dataset_guard.assert_real_dataset` — mock filename/metadata/content
   markers fail closed (`MockDatasetError`).
2. `translation_alignment.verify_translation_alignment` — source↔Persian
   validation (IDs, structure, status, coverage).
3. Strict mode (default): coverage must be **100%** — the eligible record
   set must equal the full source export.
4. `--allow-partial-coverage` (explicit opt-in): missing source IDs are
   tolerated (coverage is reported), but structural failures still refuse;
   both runs always use the **same eligible record set**.

## 4. Source↔target alignment guarantees

`translation_alignment.verify_translation_alignment` checks, by **stable
record ID** (never row order):

- unique source/target IDs, duplicates, missing and unexpected IDs;
- per-record field-structure preservation: `instruction` / `input` /
  `output` exist in the Persian record exactly when the English source has
  non-empty content (speaker roles and record boundaries preserved);
- empty translated fields among `status == "translated"` records (hard
  failure);
- translation-status distribution;
- suspicious unchanged English text (soft warning — proper nouns and
  abbreviations may legitimately survive);
- Persian-script coverage statistics (soft signal — script presence does
  NOT prove translation quality);
- eligible-record list + coverage percentage.

Output is aggregate-only: no conversation text is printed or persisted.

## 5. Completeness semantics

- `complete: true` — every source record has exactly one structurally
  intact, translated Persian counterpart (100% coverage).
- `structurally_valid: true` — no unexpected/duplicate/malformed targets
  and no structural mismatches; missing source IDs may still exist
  (incomplete coverage).
- **Strict mode requires `complete`;** partial mode requires
  `structurally_valid` and reports the coverage percentage.

## 6. Why `--verify-alignment` is not a substitute

`compare_runs.py --verify-alignment` compares the **outputs of executed
runs** (labeled.csv record-ID sets, text fingerprints, thresholds). It does
not compare the English source export with the Persian dataset and cannot
detect translation gaps, mock data, or structural mismatches. The new
`--verify-translation-alignment` mode is the dedicated source↔target
validation layer; both modes coexist.

## 7. Why Persian-script detection does not prove quality

`NOT_PERSIAN`/script-coverage checks catch echo-translation and placeholder
data, but Persian-script output can still be semantically wrong. Script
detection is a screening signal; semantic quality requires human review
(see `data/gold/ANNOTATION_GUIDELINE.md` for the human workflow that will
also cover Persian gold annotation).

## 8. Test isolation

The translation CLI's default error-log destination is the real
`data/translation_errors.jsonl` (production behavior, preserved). Tests
that exercise provider failures now pass explicit `--error-log`
temporary paths, and `tests/test_translation_safety.py` locks the guarantee:
protected data files are hash-verified unchanged across the translation CLI
failure/resume sequences.

## 9. Why runs 4/5 remain blocked

`data/mentalchat16k_fa.jsonl` does not exist; the only Persian-language
artifact is a 250-record mock pilot whose text is English echo. Until a
real provider translates the 16,005-record source export and the validation
gates pass, runs 4 and 5 are refused by design.
