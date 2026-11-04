# Privacy & Security Documentation

**Scope:** what the Autism Assistant protects, how, and what remains open.
Everything below describes **implemented and tested** behavior unless marked
otherwise.

## 1. Conversational data (patient content)

| Surface | Protection |
|---|---|
| Structured logs (`outputs/logs/pipeline.log.jsonl`) | Sensitive keys (`text`, `input`, `output`, `instruction`, …) and any string longer than `LOG_MAX_INLINE_LEN` are replaced by non-reversible SHA-256 fingerprints (`{len, sha256[:12]}`) — enforced by `privacy_log.sanitize_value`; unit-tested |
| Translation error reports | idx, code, attempt count only; provider exception strings pass through `sanitize_value`; raw record content never logged (tested: `test_report_never_contains_conversation_content`) |
| Malformed JSONL lines | logged by **line number only** — the line itself (potentially patient text) is never written anywhere |
| Split manifests | content hashes only (cannot reconstruct conversations) |
| Exported datasets / labeled.csv | contain raw conversations **by design** (they are the research inputs/outputs); stored under `data/` and `outputs/`, both **gitignored** — never committed |
| GUI | renders conversation text to the signed-in local user (required for labeling); no network transmission |

**Open item:** no retention/deletion policy for `labeled.csv` and export
files yet — operators must apply their own data-governance rules.

## 2. API credentials

- Credentials come **only** from environment variables: `ZAI_API_KEY`
  (fallback `ZHIPU_API_KEY`) for Z.ai, `OPENAI_API_KEY` for OpenAI,
  `HF_TOKEN` for Hugging Face. Nothing is hardcoded (verified by grep + tests).
- `.env` and `.env.*` are gitignored (`!.env.example` — the committed template
  contains empty values only).
- Keys never appear in exceptions, usage snapshots, or logs (regression-tested:
  `test_no_api_key_material_in_errors_or_usage`).
- Docker: credentials are injected via `docker-compose.yml` environment
  mappings (`${ZAI_API_KEY:-}` etc.) sourced from the host environment — never
  baked into the image.

## 3. External transmission

- **Explicit authorization gate**: `translate_dataset.py --provider openai|zai`
  refuses to run without `--confirm-external-sharing` — sending conversation
  content to an external provider is always a deliberate operator action
  (exit code 2 otherwise; gate tested).
- The engine's own `--probe` sends **one benign sentence** (no dataset
  content) and is likewise gated.
- No telemetry: Streamlit usage stats disabled (`browser.gatherUsageStats
  = false` in `.streamlit/config.toml` and compose).

## 4. Cost / quota safety

- Token accounting is centralized in `UsageTracker`: **actual** (API usage
  metadata) and **conservative estimated** (chars/2 when metadata is absent)
  tokens are tracked separately and both count toward the cap.
- `--max-total-tokens` is a hard guard: the engine stops gracefully at the
  cap regardless of estimate error (tested, including the estimate-only path).
- Permanent client errors (HTTP 400/401/403/404/422) fail fast — no wasted
  retries; auth failures abort the whole job.

## 5. Data integrity & separation

- **dev/eval leakage guard**: related records (same normalized patient input)
  always share a split; disjointness verified at creation (`check_no_leakage`)
  and recorded in the split manifest (hashes only).
- **Translation status gating**: only `status: "translated"` records enter
  labeling; flagged/failed/pending/rejected/incomplete records are excluded
  and counted — invalid translations cannot silently enter analysis.
- **Export safety**: `*.export_meta.json` sidecars distinguish complete from
  incomplete exports; incomplete exports print loud warnings and exit 3.
- **Gold evaluation set**: `origin: "human"` is mandatory (`gold_eval.py`
  rejects model-generated labels) — model outputs can never pose as human
  ground truth.
- **Research-use safeguards**: non-diagnostic disclaimers on the Home,
  Quick Label, Run Pipeline, Compare Models, and Results pages; scores are
  consistently labeled cosine similarities, never probabilities or clinical
  confidence.

## 6. Supply chain
- Dependency graph **pinned** and `pip check`-verified; CI installs CPU torch
  from the PyTorch index and runs tests with network disabled
  (`HF_HUB_OFFLINE=1`).
- Fresh isolated-venv installation re-verified on the final requirements
  (185 → 246 tests across phases; latest run: 246 passing).

## 7. Open items (honest)
- No encryption-at-rest for `data/` / `outputs/` (local research workspace
  assumption).
- No multi-user authentication on the Streamlit app (local tool assumption;
  XSRF protection enabled).
- MentalChat16K licensing unresolved (empty license field upstream) —
  review before any external release of derived datasets.
- Docker image build not exercised in this environment (daemon unavailable);
  compose configuration is YAML-validated.
