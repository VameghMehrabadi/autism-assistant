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

**Retention (Issue #10):** a documented default retention period and a
tested, explicit cleanup path now exist — see section 8. Enforcement
remains manual: nothing deletes anything automatically.

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

## 8. Raw conversation retention, deletion & UI privacy (Issue #10 — 2026-11-08)

### Where raw conversation text is stored (data inventory)

| Artifact | Path / pattern | Raw text? | Run-specific? | Cleanup |
|---|---|---|---|---|
| Labeled CSV | `outputs/<run>/labeled.csv`, `outputs/labeled.csv` (default run), `outputs/comparison/<run-id>/labeled.csv` | **Yes** | Yes | Eligible (`retention` whitelist) |
| Score matrices | `outputs/<run>/scores.npy` | No (vectors only) | Yes | Eligible |
| Gap/comparison reports & charts | `outputs/<run>/gap_analysis.*`, `outputs/comparison/*.{csv,json,md,png}` (shared summaries) | No (aggregates) | Yes / shared | Run dirs eligible; shared summaries **not** |
| Run metadata / split manifest | `run_meta.json`, `outputs/split_manifest.json` | No (fingerprints/counts) | Yes | Eligible |
| EN export + sidecar | `data/mentalchat16k_en_for_translation.jsonl{,.export_meta.json}` | **Yes** | Shared workflow artifact | Eligible (`translation-artifacts`) |
| FA translation | `data/mentalchat16k_fa.jsonl` | **Yes** | Shared workflow artifact | Eligible |
| Translation error log | `data/translation_errors.jsonl` | idx + codes only (sanitized) | Shared | Eligible |
| QA review file | `data/translation_review.jsonl` | **Yes** (text under review) | Shared | Eligible |
| Original datasets | Never written to `data/` by this project (loaded from the HF Hub); any user-placed source files | **Yes** | — | **Never deleted** |
| Embedding cache | `CACHE_DIR/embed_cache/` | No (vectors keyed by hash — but hash keys are linkable to repeated texts; treat as derived) | Shared | **Not** part of run cleanup |
| Model weights | `CACHE_DIR/models--*` | No (public artifacts) | Shared | Never deleted |
| Gold evaluation data | `data/gold/` | Labels only (`origin: human`) | Shared | **Never deleted** |
| App logs | `outputs/logs/pipeline.log.jsonl` | No (sanitized/fingerprinted) | Shared | Not part of run cleanup |
| Deletion audit | `outputs/logs/deletion_audit.jsonl` | No (counts/mode only) | Shared | Never deleted |
| Temp/backup files | next to their outputs (e.g. `*.corrupt-*`) | **Yes** | Yes | Delete with the same run/artifact manually (same whitelist rules apply when named otherwise) |

A hash or fingerprint is **not** treated as anonymous: embedding-cache keys
can be linked back to repeated source texts, so the cache is documented as
derived data outside run cleanup, deletable only manually.

### Retention policy (default — manually enforced)

- **Default retention period for temporary, generated raw run artifacts
  (labeled CSVs, translation exports, QA review files): 30 days.**
- This is a **policy default, not an enforced mechanism**: this issue
  introduces **no background, scheduled, or automatic deletion**. Operators
  are responsible for reviewing and running the cleanup.
- Operators own retention decisions; the application provides the tooling
  (`retention.py`, `cleanup_runs.py`, Results ▸ "Retention & cleanup").

### How to inspect and delete a run's raw outputs

```bash
python cleanup_runs.py --list                       # addressable locations
python cleanup_runs.py --run <location>             # exact preview (default)
python cleanup_runs.py --run <location> --confirm   # explicit deletion
python cleanup_runs.py --run translation-artifacts --confirm
```

Or in the UI: **Results ▸ Retention & cleanup** — pick a location, preview
the exact file list, tick the confirmation checkbox, delete. Deletion
boundaries (enforced by whitelist + canonical-path containment):

- **Never deleted:** original/source datasets, unrelated runs, shared
  comparison summaries, model weights, embedding/model caches, gold
  evaluation data (`data/gold/`), source code, anything outside
  `outputs/`/`data/`, and any file whose run ownership cannot be
  established confidently (unknown names are skipped with a reason).
- **Sidecars:** the export sidecar (`*.export_meta.json`) is exclusively
  paired with its export file and is removed in the same operation;
  missing sidecars are reported, never fatal.
- **Audit:** every confirmed deletion appends a metadata-only record
  (timestamp, relative location, counts, mode, outcome) to
  `outputs/logs/deletion_audit.jsonl` — never conversation text, file
  contents, or absolute paths.

### UI privacy mode

Conversation previews in **Results** are truncated by default
(`… [+N chars — reveal to view]`); the full text of one record is shown
only behind an explicit per-record reveal button, rendered from the
underlying data on each rerun and never copied into session state. In
**Quick Label**, the sample list is truncated; picking a sample
deliberately loads its full text into the editor (that choice is the
reveal). This is a privacy affordance — **not** a security boundary
against screenshots, browser inspection, or authorized users.

### Limitations of deletion (honest)

- Deletion is a filesystem `unlink` — **not secure erasure**: SSD
  wear-leveling, filesystem journals, backups and volume snapshots may
  retain recoverable copies outside the application's control.
- Docker named volumes holding `data/`/`outputs/` on other hosts are not
  covered by a cleanup run against this checkout; run cleanup where the
  data lives.
- Nothing here claims GDPR/HIPAA/clinical compliance, and retention
  requirements for real studies must be set by the study's own governance.
