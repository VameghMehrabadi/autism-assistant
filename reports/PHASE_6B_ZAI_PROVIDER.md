# Phase 6b: Z.ai (GLM-5.3-Flash) Provider Integration — Report

**Status: INTEGRATION COMPLETE AND VERIFIED (offline + local live path) · PILOT BLOCKED — `ZAI_API_KEY` not configured**

## 1. What Was Built (no architecture rebuild)

The Z.ai GLM API is OpenAI-compatible, so the integration **reuses the existing `OpenAIProvider`** — one provider class serves `openai` and `zai`; nothing model-specific is hardcoded (requirement 14).

| Change | File | Purpose |
|---|---|---|
| `base_url` support + `max_retries=0` (engine owns retries) | `translate_dataset.py::OpenAIProvider` | point the same client at Z.ai's endpoint |
| `resolve_zai_config()` | `translate_dataset.py` | credentials **from environment only**: `ZAI_API_KEY` (fallback `ZHIPU_API_KEY`), endpoint from `ZAI_BASE_URL` (default `config.ZAI_API_BASE_URL = https://api.z.ai/api/paas/v4`), default model `config.ZAI_MODEL = glm-5.3-flash` |
| Token accounting | `OpenAIProvider.usage` + `_accumulate_usage()` | per-request **usage metadata**: `requests / prompt_tokens / completion_tokens / total_tokens` — input and output tracked **separately** (requirement 9) |
| Hard quota guard | `TranslationEngine(max_total_tokens=…)` | graceful stop with `stop_reason="quota_cap"` before any call that would push reported usage past the cap; checkpointed progress, unprocessed records stay pending (requirement 10) |
| `--provider zai` + `--base-url` + `--max-total-tokens` + `--probe` | CLI | configurable provider selection; `--probe` verifies endpoint/auth/model/protocol with **one benign request** (no dataset content) |
| `openai==3.42.1` pinned | `requirements.txt` | the designed optional dependency is now required for real runs (`pip check` clean) |

## 2. Verification Evidence (executed, not claimed)

| Check | Result |
|---|---|
| Test suite | **174 passed** (164 + 10 new Z.ai tests in `tests/test_translation_zai.py`); `pip check` clean |
| Config resolution | key required → clear `ZAI_API_KEY` blocker; env overrides honored; `ZHIPU_API_KEY` fallback — unit-tested |
| Token accounting | accumulates `usage` metadata across requests; tolerates missing metadata — unit-tested with synthetic usage objects |
| Quota guard | graceful `quota_cap` stop, records persisted, remainder pending, `failed == 0`; `failed == 0` + summary fields (`stop_reason`, `provider_usage`) persisted — unit-tested + CLI-tested |
| **Live network path** | `--probe` against a local OpenAI-compatible stub: `OK: endpoint reachable, authentication accepted, model responded` — Bearer auth, model-id passthrough (`glm-5.3-flash`), 1-element JSON-array protocol parse, usage extraction (`prompt_tokens=912, completion_tokens=24, total_tokens=936`), exit 0 |
| Blocker path | `--probe` without key → `CONFIG ERROR: ZAI_API_KEY is not set` (exit 2) — exact message users see |

**What is verified vs. not:** request shape, auth handling, protocol, usage accounting, and quota enforcement are verified end-to-end against an OpenAI-compatible server. The **real Z.ai service** (actual endpoint reachability, account authentication, model availability of `glm-5.3-flash`, real rate limits, real per-token pricing) **cannot be verified without the account key** — `ZAI_API_KEY` is not set in this environment (verified; the harness's own `ZAI_*` OAuth variables are runtime plumbing, not project credentials, and were not used). Rate limits are server-side and not queryable via API; the engine handles 429 + Retry-After dynamically.

## 3. Pilot Plan (ready to execute the moment the key is provided)

```bash
export ZAI_API_KEY=<paste-your-key-here>          # never commit it
python translate_dataset.py --provider zai --probe --confirm-external-sharing
# → verifies real endpoint + auth + glm-5.3-flash access (1 benign request)
python translate_dataset.py --provider zai \
    --batch-size 5 --max-retries 5 --checkpoint-every 25 \
    --max-total-tokens 30000000 \
    --limit 100 --confirm-external-sharing \
    --summary-out data/translation_pilot_summary.json
# → 100-record pilot, hard-capped at 30M tokens, checkpointed + resumable
python translation_qa --file data/mentalchat16k_fa.jsonl
```

Pilot evaluation checklist (requirement 6): automated QA (Persian script, length ratios, corruption), autism-terminology consistency counts against the glossary, patient/counselor register spot-check on a few short records, plus token accounting per record from API metadata (requirement 7).

## 4. Projected Token Consumption (formula-based — to be recalibrated from pilot measurements)

Measured inputs: system prompt **5,103 chars ≈ ~1,300 tokens** (sent with every request); average record **2,884 EN chars ≈ ~720 input tokens** (counselor responses dominate: 2,071 of 2,884 chars); estimated Persian output ≈ **~650 tokens/record**. Formula: `requests = 16,005 × 3 fields ÷ batch_size`; `total = requests × (system + input-per-call) + 16,005 × output`.

| batch-size | requests | prompt tokens | completion tokens | **total** | % of 300M |
|---|---|---|---|---|---|
| 1 | 48,015 | ≈ 73.9M | ≈ 10.4M | **≈ 84.3M** | 28% |
| **5 (recommended)** | 9,603 | ≈ 47.1M | ≈ 10.4M | **≈ 57.5M** | 19% |
| 10 | 4,802 | ≈ 42.8M | ≈ 10.4M | **≈ 53.2M** | 18% |

- Batch 5–10 keeps the full dataset around **~53–58M tokens — roughly 18–19% of the 300M budget**, leaving ample margin for retries, flagged re-translation, and QA-assistance calls.
- These are **estimates** (tokenizer- and model-dependent). The pilot replaces them with measured `usage` metadata before any full run; the full run additionally carries `--max-total-tokens` as a hard stop so the account quota can never be exceeded regardless of estimate error.
- Z.ai pricing is not applied here (no public price for `glm-5.3-flash` available to this environment) — token totals are the verifiable quantity; apply the current price sheet to these numbers.

## 5. Approval Gate (requirement 11)

Per the phase rules the full 16,005-record run **starts only after explicit approval**. The pilot (100 records) also requires the key. To proceed:

1. Provide `ZAI_API_KEY` in the environment (or a local uncommitted `.env` sourced by the shell).
2. Approve the pilot (I run the probe + 100-record pilot + evaluation + measured cost table).
3. After reviewing pilot quality + measured consumption, approve the full run (resumable; ~53–58M projected tokens; `--max-total-tokens` enforces the ceiling).

Nothing was sent to any external service during this phase: the only network traffic was to a local stub server on 127.0.0.1 containing one benign test sentence.

## 6. Test Results (actual)

```
tests/test_translation_zai.py  10 passed
full suite                     174 passed in 28.99s
pip check                      No broken requirements found
```
