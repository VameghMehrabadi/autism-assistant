# Phase 6: Advanced Persian Dataset Translation — Report

**Status: IMPLEMENTATION COMPLETE · FULL-DATASET EXECUTION BLOCKED (no API credentials)**

## 1. Executive Summary

The Phase-6 translation stack is implemented, tested (47 new tests; **164/164 passing**), and proven end-to-end with the offline mock provider: complete-dataset export (16,005 deduplicated records), batched translation, interrupted-job recovery, duplicate prevention, automated QC, completeness accounting, and `data_loader` compatibility were all **executed**, not merely written.

Full-dataset translation **cannot be executed in this environment**: `OPENAI_API_KEY` is **not set** and the `openai` package is **not installed**. Per the phase rules, no translations were fabricated, and the production `data/mentalchat16k_fa.jsonl` was **intentionally not created**. The exact operator command to finish the job is in §8.

**Dataset-size correction (important):** the task brief states "approximately 140,000 records". The **actual** source (`ShenLab/MentalChat16K`, verified from the Hugging Face `dataset_info.json` cached during this project) contains **16,084 records** (`Interview_Data_6K.csv` + `Synthetic_Data_10K.csv`). The pipeline was run against the **complete actual dataset with no artificial limit**: all 16,084 records were loaded; the Phase-2 pipeline removed 79 content duplicates, leaving **16,005 valid unique records** exported for translation. Nothing was limited to a subset.

## 2. What Was Found — Parallel Implementation Preserved

At the start of the execution step, a **complete parallel Phase-6 implementation** was discovered in the working tree (engine, QA module, export update, prompt, 41 tests — uncommitted, timestamps 20:11–20:12, created while this phase was underway). Per the preserve-and-reuse rules it was **adopted, not replaced**. It had **never been run**: 11 of its 41 tests failed. This phase's work became: repair, gap-fill, reconcile, execute.

### Bugs found and fixed (with evidence)

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| B1 | `translation_qa.py` crashed at import/use (`NameError: name 'config' is not defined`, 2 test failures) | missing `import config` | added import (`translation_qa.py:27`) |
| B2 | **Batch-straddle crash**: a record's fields spanning a batch boundary hit `KeyError` in `by_idx` (`pending` reset while `work` still held that record's segments) — 8 engine tests failed | `translate_records`/`run` reset the pending-record list per slice | replaced both paths with a per-idx **accumulator core** (`_translate_accumulating`): assignments merge across batch slices; every record is emitted **exactly once**, when all its fields are translated/flagged; failed records stay absent (retried next run) and are counted once |
| B3 | **Deep-merge data loss**: batch-parse fallback ran per-segment calls but `merged.update({idx: {...}})` *replaced* each record's field map — records kept only their **last** field, so nothing was ever emitted after a fallback (proven by instrumented trace: `{0: {'output': …}}` after 3 successful single calls) | `dict.update` on the outer map | per-idx deep merge (`merged.setdefault(idx, {}).update(field_map)`) |
| B4 | Test bug: `assert 0 in report["flagged_idx"]` — only idx 1 was flagged | wrong expected value | corrected to `1 in …` |
| B5 | Test bug: expected `SystemExit` from `qa.main()`, which only *returns* the exit code | wrapper confusion | assert `qa.main() == 2` |
| B6 | Test bug: garbage-batch call-count expectations contradicted the test-double's logging (each single call logged twice) and the engine's retry semantics | unrun tests | corrected assertions with explanatory comments (`12` double-logged singles; `3` successful calls because failed attempts raise before the call log) |

### Gaps filled (requirement → implementation)

- **Prompt ↔ engine protocol mismatch (critical):** the prompt on disk described a JSON-object protocol the engine does not implement. `prompts/semantic_persian_translation.txt` was rewritten to the engine's actual contract (numbered JSON-string segments → JSON array of Persian strings, `[UNTRANSLATABLE] <reason>` flagging), while keeping the strong original content: translate-don't-summarize, speaker roles, terminology glossary, uncertainty/hedging preservation, no added/removed information, brand-name rule.
- **Privacy gate (Step 7):** `--provider openai` now **requires `--confirm-external-sharing`** — external upload of conversation content is an explicit operator action (verified: exit 2 with the message before any network use).
- **Mock→production guard:** `--provider mock` refuses to write to `data/mentalchat16k_fa.jsonl` (synthetic output must never become the production dataset).
- **Completeness accounting (Critical Req. #9):** `translation_qa.check_completeness(src, dst)` reports `total_source / translated / missing / duplicated_output_ids / missing_idx_sample / complete`; wired into the CLI and the persisted summary.
- **Final report (Critical Req. #10):** every run writes `data/translation_summary.json` = engine counts (translated/flagged/empty_source/failed/errors) + QA report + completeness — content-free, privacy-safe.

## 3. Requirements Coverage (Steps 2–8)

| Requirement | Where | Evidence |
|---|---|---|
| Preserve record IDs, ordering, English content, metadata | `normalize_source_record`/`build_output_record` (`*_en` keys + `extra` passthrough) | `test_engine_preserves_original_english`, `test_build_output_record_keeps_extra_source_metadata`; export keeps `idx` ordering |
| Source schema fidelity (instruction/input/output only — no invented fields) | `export_for_translation.py` | export log: 16,005 records, load-report counts printed |
| Translate patient + counselor content separately | field-level segments; roles preserved structurally | `test_normalize_source_record_fields_and_idx` |
| Configurable provider / model | `--provider openai\|mock`, `--model` | `test_openai_provider_requires_api_key` |
| Batch processing, configurable size | `--batch-size` (field segments per call) | `test_engine_translates_records_end_to_end` (call sizes ≤ batch) |
| Retry / rate-limit / timeout | exponential backoff + jitter, Retry-After honored (clamped ≤120 s), SDK timeout param, batch→single fallback | `test_engine_retries_then_succeeds`, `test_engine_rate_limit_honors_retry_after`, `test_engine_garbage_batch_falls_back_to_single_calls` |
| Interrupted-job recovery + duplicate prevention | resume-from-output (output file = checkpoint truth), `duplicates_skipped` | `test_engine_resume_after_interruption`, `test_interrupted_job_recovery_via_cli`, CLI resume run (§5) |
| Status tracking / progress / structured errors | statuses `translated\|flagged\|empty_source`; progress prints; `data/translation_errors.jsonl` (idx+code only) | `test_engine_flags_untranslatable_segments`, `test_engine_writes_structured_error_log` |
| No hardcoded keys; env credentials | `OPENAI_API_KEY` only | `test_openai_provider_requires_api_key` |
| Prompt improvements (Step 5) | rewritten system prompt incl. ambiguity flagging + JSON-array protocol | §2 gap table |
| QC checks (Step 6) | `translation_qa`: MISSING_TRANSLATION, EMPTY_TRANSLATED_FIELD, DUPLICATE_IDX, MISSING_ORIGINAL_FIELD, CORRUPTED_UNICODE (mojibake/U+FFFD), NOT_PERSIAN, SUSPICIOUSLY_SHORT/LONG, BAD_STATUS, INVALID_RECORD + **manual review file** (`write_review_file` → `data/translation_review.jsonl`) | 21 QA tests incl. `test_report_never_contains_conversation_content` |
| Tests with mocks only, no paid credits (Step 8) | all provider interaction is in-memory doubles | suite is fully offline (Phase-0 CI guards prove it) |

## 4. Execution Results (actual)

| Step | Command | Result |
|---|---|---|
| Full export | `python export_for_translation.py --limit 0` | **16,005 records** (16,084 raw − 79 duplicates − 0 invalid), 56.9 MB → `data/mentalchat16k_en_for_translation.jsonl` |
| Test suite | `python -m pytest` | **164 passed** (117 pre-existing + 47 translation) |
| Mock smoke | `translate_dataset.py --provider mock --limit 100 --batch-size 4` | 100/100 translated, 0 failed; QA flags 300×NOT_PERSIAN (mock output is deliberately non-Persian — **validation proven**); summary persisted |
| Resume run | same command, `--limit 250` | +150 new, **250 unique lines, 0 duplicates** — recovery + dedup proven |
| Loader compatibility | `load_samples_fa(path=mock_sample)` | 250/250 samples load; patient `input`/counselor `output` separation works on the translated schema; `idx` preserved |
| Real provider | `--provider openai` | **BLOCKED**: gate fires without `--confirm-external-sharing`; with it: `OPENAI_API_KEY is not set` + `openai` package not installed → exit 2, zero records sent |

**Actual translated record counts (production dataset): 0.** Mock-sample artifacts (`data/mentalchat16k_fa.mock_sample.jsonl`, 250 records) are synthetic, gitignored, and marked `model: mock-echo` — they are pipeline evidence, not translations. Pending: **16,005 − 250 = 15,755** beyond the mock sample; effectively all 16,005 await real translation.

## 5. Privacy (Step 7) — verified

- Logs and error reports carry **indices, codes, and counts only**; provider exception strings are sanitized via `privacy_log.sanitize_value` (`test_report_never_contains_conversation_content` asserts conversation text absent from reports).
- Credentials: environment-only; `--confirm-external-sharing` gates external upload; `provider_notice` log event records the acknowledgment.
- `data/*` is gitignored → translated dataset and review files are not committed; **licensing review is still required before any external release** (MentalChat16K's HF license field is empty in the cached metadata — treat as unresolved).
- Mock output never touches the production path (guard tested).

## 6. Known Limitations

- **No semantic validation is possible without real translations** — automated QA checks structure, plausibility, and language script only; the phase explicitly does *not* claim clinical or semantic accuracy. Flagged/ambiguous records go to the review file for human translators.
- Segment-level batching sends each field independently; the per-segment protocol does not pass field-role context (patient vs counselor) in the request — the system prompt instructs register inference from content. A role-aware payload is a recommended future improvement (requires engine change; out of scope to avoid destabilizing the repaired engine).
- The engine's per-field retry means a failing record consumes up to `max_retries × fields` calls before being recorded as failed; acceptable at these scales.
- `--prompt` CLI flag is retained for compatibility but the engine reads the fixed system-prompt path (documented in `--help`).

## 7. Remaining Work

1. Obtain OpenAI access (`pip install openai` + `export OPENAI_API_KEY=…`), review privacy/licensing, then run §8's command — checkpoint/resume makes it interruptible at will.
2. Human review of `data/translation_review.jsonl` after each batch (flagged/ambiguous records).
3. Wire the review-file workflow into the Streamlit Translate/Export page (optional).
4. After real translation: re-run `translation_qa` CLI, then Phase-4 runs 4–5 (FA dataset × bge-m3 / e5-large).

## 8. Exact Command to Complete the Translation

```bash
pip install openai
export OPENAI_API_KEY="<key>"            # never commit it
python translate_dataset.py --provider openai --model gpt-4o \
    --batch-size 5 --max-retries 5 --checkpoint-every 25 \
    --confirm-external-sharing           # processes ALL 16,005 pending records
# resume after any interruption by re-running the same command
python translation_qa                    # QC + review file when done
```

Cost/duration note: ≈16,005 records ≈ 48,000 field segments; at batch-size 5 and GPT-4o pricing this is a paid, multi-hour job — it was **not** executed here for lack of credentials, and no translation is claimed beyond what actually ran.
