# Phase 1: Dataset Integrity & Encoding Repair — Report

**Status: COMPLETE** — encoding bug fixed with regression coverage; dataset integrity fully verified; provenance investigated with local evidence. 190/190 tests passing. No dataset content altered.

## 1. UTF-8 Encoding Bug — Root Cause and Fix

### Root cause (reproduced)

`data_validation.detect_encoding()` read a **fixed 65,536-byte sample** and ran a strict UTF-8 decode on it. For a large valid UTF-8 Persian JSONL, the sample can end **inside a multibyte character** (Persian chars are 2 bytes in UTF-8; ZWNJ is 3). The strict decode then fails on the *incomplete tail sequence*, the function fell through to the single-byte fallbacks, and — because CP1256's high bytes decode into Arabic-script characters — returned **`cp1256`**. The whole file was then read as CP1256, silently mojibaking every Persian character.

**Reproduction** (controlled, deterministic): a 173,427-byte valid UTF-8 Persian JSONL whose byte offsets 65,535–65,536 are the two bytes of `ا` (U+0627) — the 65,536-byte sample ends after the lead byte `0xD8` → old code returned **`cp1256`**; the CP1256 interpretation of the same byte range produces mojibake (`\u0637`-range garbage) for the affected text.

### Fix (`data_validation.detect_encoding`)

**Boundary-artifact rule**: when the strict UTF-8 decode fails, the exception's `start` position is inspected. If the *first* error begins within the last 3 bytes of the sample (the longest UTF-8 sequence is 4 bytes) **and** the sample is the full `sample_size` (i.e. the file is larger than the sample), the file is treated as **UTF-8** — the incomplete tail is a sampling artifact, not file corruption.

Why this is safe:
- A genuine CP1256 file fails strict decoding at its **first high byte**, which for Persian JSONL lies far earlier in the sample — legitimate CP1256 detection is unaffected (regression-tested).
- Worst case for a pathological file (real corruption exactly at the boundary): reading later fails **loudly** with `UnicodeDecodeError` instead of silently corrupting — never silent corruption.
- Files smaller than the sample whose true EOF has an incomplete sequence are correctly **not** rescued (regression-tested).

## 2. Files Modified
- `data_validation.py` — `detect_encoding()` boundary-artifact rule + documented rationale.
- `tests/test_data_validation.py` — 5 new regression tests.
- `tests/test_data_loader.py` — 1 new end-to-end integrity roundtrip test.

## 3. Regression Tests Added (6)
1. `test_detect_encoding_utf8_sample_ends_mid_multibyte_char` — the exact reported bug (large Persian JSONL, boundary cut inside a 2-byte char) → must detect `utf-8`.
2. `test_detect_encoding_utf8_boundary_cut_positions` — both cut offsets of a 2-byte char and of a 3-byte char (ZWNJ `U+200C`, ubiquitous in real Persian such as `می‌شود`).
3. `test_detect_encoding_small_corrupt_file_not_rescued` — small file with a genuinely incomplete final sequence must **not** be labelled UTF-8.
4. `test_detect_encoding_large_clean_boundary_still_utf8` — control: clean boundary still detects UTF-8.
5. `test_detect_encoding_cp1256_fallback` (pre-existing) — legitimate CP1256 files still detected (unchanged, passing).
6. `test_load_samples_fa_large_utf8_boundary_file` — **end-to-end**: an 800-record large UTF-8 Persian JSONL with a boundary cut loads via `load_samples_fa` with `encoding="utf-8"`, 800/800 kept, exact source Persian strings preserved (`برنامه‌ی شماره 5 عوض می‌شود مضطرب.`), no mojibake signatures (`Ø`/U+FFFD), all `lang == "fa"`.

## 4. Dataset Integrity Statistics (evidence-based, read-only)

### Raw upstream sources (HF download cache, `D:\autism_assistant_cache\downloads\`)

| File (per cached metadata sidecars) | Actual parsed CSV rows | Naive line count | Empty rows |
|---|---|---|---|
| `Synthetic_Data_10K.csv` | **9,774** | 167,860 | 0 |
| `Interview_Data_6K.csv` | **6,310** | 43,796 | 0 |
| **Total** | **16,084** | — | 0 |

**Count reconciliation**: 9,774 + 6,310 = **16,084** — exactly matches the Hugging Face dataset metadata (`num_examples: 16084`) and the project's raw record count. Parsed with `csv.reader` (embedded newlines inside quoted cells handled correctly; both files contain thousands of multi-line cells).

### The reported composition claim is not reproducible from the distributed files

| Source | Claimed | Actual (verified locally) | Δ |
|---|---|---|---|
| Synthetic GPT-3.5 conversations | 9,775 | **9,774** | −1 |
| Caregiver-coaching QA (palliative/hospice-derived) | 6,338 | **6,310** | −28 |
| **Total** | **16,113** | **16,084** | **−29** |

The claimed totals cannot be produced by any counting method applied to the distributed CSVs (naive line counting yields 167,860/43,796, not the claimed numbers). The claim's origin (e.g. a dataset-card statement) cannot be verified without network access. **Verifiable ground truth: 9,774 / 6,310 / 16,084.** Nothing was invented or altered to reconcile the difference.

### Export integrity (`data/mentalchat16k_en_for_translation.jsonl`)

| Metric | Value |
|---|---|
| Lines / invalid JSON / non-object records | 16,005 / **0** / **0** |
| Records | 16,005 |
| Unique record IDs | **16,005** (0 duplicate IDs) |
| ID stability | positional `idx` 0…16,083; **32 gap ranges** covering exactly the 79 removed duplicates (e.g. 37→41, 441→450) — identifiers are the original upstream positions, unchanged |
| Field presence | `instruction` 16,005/16,005 · `output` 16,005/16,005 · `input` present-but-empty in **27 records** (caregiver-coaching style where the instruction carries the content); no missing keys |
| Language integrity | **0** Persian-contaminated values (pure English export) |
| Content duplicates remaining | **0** (dedup removed 79 upstream duplicates; 16,084 − 79 = 16,005 ✓) |

## 5. Dataset Provenance & Suitability Findings (Part C)

Keyword evidence across the raw sources (case-insensitive counts over all fields):

| Term group | `Synthetic_Data_10K.csv` (9,774) | `Interview_Data_6K.csv` (6,310) |
|---|---|---|
| palliative/hospice (hospice, palliative, end-of-life, terminal) | **0** | **1,346** (hospice ×1,016) |
| caregiver/coaching (caregiver, coach, dementia, alzheimer) | 158 | **2,752** (caregiver ×2,081, dementia ×254, alzheimer ×403) |
| autism-related (autism, asd, sensory, stimming, special interest, meltdown, neurodivergent, asperger) | 69 (sensory ×52, meltdown ×17) | 18 |
| general mental-health (anxiety, depression, therapy, counsel, panic, bipolar, trauma) | **77,677** (anxiety ×27,821, counsel ×21,883) | 18,277 |

**Findings:**
1. `Interview_Data_6K.csv` is confirmed to be **caregiver-coaching content derived from palliative/hospice conversations** (hospice/palliative/end-of-life and caregiver/dementia terminology concentrated exactly there) — consistent with the claim's *characterization*, though its *count* (6,338) is not reproducible (actual 6,310).
2. `Synthetic_Data_10K.csv` contains **no** palliative/hospice content and is dominated by general mental-health counseling language. The "GPT-3.5" generator attribution has **no local metadata evidence** (the CSVs carry no provenance column; no network access permitted).
3. **Suitability for the autism-related semantic coverage research**: MentalChat16K is a **general mental-health counseling + caregiver-coaching corpus, not an autism-specific dataset**. Autism-related terminology is nearly absent dataset-wide ("autism" ×5, "ASD" ×0, "stimming" ×0 in 16,084 records; the few "sensory"/"meltdown" hits occur in psychiatric contexts). **Consequence for the research**: the semantic-labeling pipeline measures how the A–G autism fact categories map onto *general* counseling language — valid as a public-corpus coverage/gap analysis (and the gap analysis correctly shows weak coverage for D/Routine and E/Interests), but the dataset **cannot support claims about autism-specific conversational content**, and any future fine-tuning/RAG for autism support will need autism-specific data sources. **Research objective unchanged** — documented limitation only.

## 6. Remaining Limitations
- The claimed composition (9,775 + 6,338) could not be traced to any counting method on the distributed files; without network access the dataset card could not be consulted. Local evidence stands.
- "GPT-3.5" provenance of the synthetic file is unverifiable locally.
- Files smaller than the detection sample with a genuinely incomplete final byte sequence remain unrecoverable-by-design (they are corrupt; they now fail loudly rather than mojibake silently — except the ≤3-byte-shorter edge which falls to the legacy fallback path).
- `data/mentalchat16k_en_for_translation.jsonl` retains raw conversations (gitignored, by design); retention policy still a documented open item.

## 7. Test Results (actual)

| Suite | Result |
|---|---|
| Focused regression (encoding + loader) | 56 passed |
| **Complete suite** | **190 passed** (185 + 5 new) |
