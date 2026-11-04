# Phase 3: Z.ai Translation Pipeline Reliability — Report

**Status: COMPLETE** — provider and engine hardened for reliability, resumability, and quota safety. **228/228 tests passing** (+23). **No external API calls were made** — every test uses mocked providers/responses.

## 1. API Error Handling

### Status-code → kind mapping (precise, in `_map_provider_exception`)

| HTTP | Kind | Retry behavior |
|---|---|---|
| 400 / 404 / 405 / 422 | `permanent` | **fail fast** — the request can never succeed as-is |
| 401 / 403 | `auth` | **fail fast + whole-job abort** (`stop_reason="aborted_auth"`) — every future request would fail identically |
| 429 | `rate_limit` | retried; honors `Retry-After` header / attribute, then exponential backoff + jitter |
| 5xx | `api` | retried |
| timeouts / connection errors | `timeout` / `api` | retried |
| malformed responses (`parse`) | retried; batch→single fallback |
| empty responses (`invalid_response`) | retried |
| truncation (`finish_reason=length`) | `truncated` — **not retried** (deterministic at temperature 0); batch falls back to singles |

`NON_RETRYABLE_KINDS = {config, auth, permanent, truncated}` — the retry loop raises immediately for these (verified: a permanent/auth failure makes exactly **1** provider call, `attempts=1` in the error log, zero sleeps).

**Job abort on auth**: an `auth` failure marks the in-flight batch's records failed and aborts the entire job (`_JobAbort` control flow → `summary.stop_reason="aborted_auth"`), preventing thousands of doomed calls on a 16K-record run. Records never attempted remain pending for a healthy rerun (tested: 10-record job aborts after 1 call, 1 failed, 9 pending).

## 2. Model Response Handling

- **`finish_reason`** now inspected: `finish_reason == "length"` → `TranslationError("truncated")` — the response is rejected, never silently accepted. At temperature 0 the truncation reproduces, so a truncated **single** fails without retry; a truncated **batch** falls back to per-segment singles (shorter outputs fit the limit). The error message points the operator at `--max-output-tokens`.
- **`max_tokens`**: new `--max-output-tokens` CLI flag (default: provider default — no artificial cap) wired through `OpenAIProvider.max_output_tokens` → `create(max_tokens=…)`; passthrough unit-tested.
- **Thinking-mode compatibility**: `<think>…</think>` blocks are stripped (`_strip_think_blocks`, applied in the provider and defensively in `parse_batch_response`); an **unclosed** `<think>` removes everything from the tag on → surfaces as `invalid_response` instead of parsing reasoning text as a translation; GLM's separate `reasoning_content` payload is explicitly read-and-ignored; a thinking-only response raises `invalid_response` (retried, then failed) — never accepted.
- **JSON validation**: unchanged strict protocol (valid JSON array, exact element count, string elements) **plus** a new engine guard in `_assign`: an **empty translation for a non-empty source segment** is flagged (`empty translation for non-empty source`) — incomplete translations can never silently enter downstream analysis.
- **Interrupted requests**: connection/timeout kinds retried with backoff; usage for such attempts is recorded conservatively (see §4).

## 3. Checkpoint and Resume

- **Translation**: progress is persisted **per record** — `write` + `flush` + **`os.fsync`** (durable checkpoint). Resume reads the output file as the source of truth; duplicates are impossible (done-map); valid completed translations are never overwritten (tested: a record translated by an earlier run survives a rerun untouched).
- **NEW — torn-output repair**: an interrupted append leaves a torn final line. On resume the engine now **rewrites the output file keeping only parseable records**, moves the damaged original to `out.jsonl.corrupt-<timestamp>` for inspection, and logs the repair loudly. Tested: a file with 1 valid record + a torn line resumes to exactly 4 unique records, 0 duplicates, corrupt backup present.
- **Embedding progress**: previously the cache was written only **after the entire encode pass** — an interruption lost the whole computation. `_embed` now encodes and stores in **chunks** (`config.EMBED_CACHE_WRITE_CHUNK = 256`): progress persists after every chunk. Tested with a crash-injecting encoder: the first chunk's vectors survive the crash and are served from cache afterwards.

## 4. Token Quota Protection

- **Centralized accounting**: new `UsageTracker` — one canonical recorder per provider (`OpenAIProvider.tracker`, exposed as `.usage`).
- **Actual vs estimated, distinguished**: `prompt_tokens`/`completion_tokens`/`total_tokens` are **actual** (response metadata); `estimated_prompt_tokens`/`estimated_completion_tokens`/`unknown_metadata_responses` are **estimates** used only when the API reports no usage. Estimates are conservative by construction (`ceil(chars/2)` ≥ true tokens for EN/FA text) and never overwrite actual values.
- **Guard**: `--max-total-tokens` now compares against `actual + estimated`, so responses lacking metadata can never bypass the cap (tested: an estimate-only provider trips `quota_cap` with `total_tokens == 0`).
- Error-path requests (timeouts, SDK errors) are recorded as conservative estimates — the server may still have served them.

## 5. Security and Privacy
- API keys appear **nowhere**: not in exceptions (HTTP error messages carry status codes only), not in `UsageTracker` repr/dict, not in logs (error strings sanitized via `sanitize_value` before the error log) — regression-tested.
- External transmission remains gated behind `--confirm-external-sharing`; the probe sends one benign sentence. **No external requests were made during this phase** — all provider behavior was mocked.

## 6. Files Modified
- `translate_dataset.py` — `UsageTracker`; precise status mapping; `NON_RETRYABLE_KINDS` + fail-fast; `JOB_ABORT_KINDS` + `_JobAbort` abort flow; `finish_reason`/truncation handling; `_strip_think_blocks`; think-strip in `parse_batch_response`; empty-translation flagging in `_assign`; `max_output_tokens` (provider + CLI); torn-output repair; `fsync`; estimate-aware quota guard.
- `labeler.py` — incremental chunked embedding-cache persistence.
- `config.py` — `EMBED_CACHE_WRITE_CHUNK = 256`.
- Tests: `tests/test_translation_reliability.py` (23 new tests).

## 7. Test Results (actual)
| Suite | Result |
|---|---|
| Phase-3 reliability tests | 23 passed |
| **Complete suite** | **228 passed** |

## 8. Known Limitations
- Truncated outputs are failed, not auto-resolved: the operator raises `--max-output-tokens` and reruns (resume retries the failed records). An automatic per-request backoff-and-retry-with-larger-cap was deliberately avoided (unbounded cost).
- Conservative estimates overcount (~2× for English) by design; with real Z.ai metadata the estimate bucket stays at zero.
- Exact-message integrity of SDK exceptions is trusted for status extraction (`status_code` / `response.status_code`); exotic providers deviating from the OpenAI-compatible convention fall back to name-based mapping.
- Job abort on auth means a mid-run key expiry stops the run (by design — rerun after fixing credentials; resume is lossless).

## 9. Pilot Readiness
**Ready for an explicitly authorized pilot.** The pipeline now has: precise permanent-error fail-fast, auth abort, truncation rejection, think-mode tolerance, empty-translation flagging, durable per-record checkpoints with torn-write repair, chunked embedding persistence, centralized actual-vs-estimated token accounting, and a hard quota guard. Awaiting `ZAI_API_KEY` + explicit pilot authorization (no pilot was executed in this phase).
