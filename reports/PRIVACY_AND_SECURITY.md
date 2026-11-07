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

## 8. Encryption at rest & storage security (Issue #11 — reviewed 2026-11-07)

### Architectural decision

**Protection mechanism: operator-managed encrypted host storage** (BitLocker
/ LUKS / FileVault / an encrypted volume) — not application-level crypto.
Rationale: every file the pipeline writes lands under `config.DATA_DIR`,
`config.OUTPUT_DIR` or `config.CACHE_DIR` (verified write-path inventory:
labeled CSVs, score matrices, gap/comparison reports, charts, translation
outputs + checkpoints + error logs, QA review files, export sidecars,
sanitized JSON logs, embedding-cache chunks), so OS-level encryption covers
**all** of them — including temporary and backup files, which are written
next to their outputs on the same volume — without touching any read/write
path. Application-level authenticated encryption was evaluated and
**deliberately not implemented**: covering the incremental JSONL
checkpoint/resume flow, torn-output repair, numpy caches and chart output
safely would require a storage-architecture rewrite, and a partial crypto
layer would falsely imply PHI-grade protection. No fake crypto layer was
added to satisfy the issue's application-level test list.

**Four-way distinction:**

| Kind | Status |
|---|---|
| Encryption implemented by this application | **None** — by design |
| Encryption provided by the host OS | Available, **not detectable** by this app |
| Operator-declared external protection | The `patient-data` mode's basis |
| Encryption actually verified by the app | **Nothing — the app cannot verify disk state, and does not attempt to** |

### Storage-security modes (`DATA_STORAGE_MODE`)

- **`research` (default):** existing workflow, unchanged. The UI (Home
  banner + per-page captions) and every CLI entry point (`main.py`,
  `compare_runs.py`, `export_for_translation.py`, `translate_dataset.py`,
  `translation_qa.py`) print the effective mode and warn that files are
  stored **unencrypted** unless the operator protects the volume — and that
  this is unsuitable for PHI unless the operator separately establishes and
  documents equivalent protection.
- **`patient-data`:** requires `STORAGE_PROTECTION_DECLARATION` naming the
  mechanism (`bitlocker` | `luks` | `filevault` | `encrypted-volume`). The
  declaration is an **operator-declared prerequisite — not an
  application-verified property**. Missing/invalid declaration or an
  invalid mode value **fails closed** (CLI exit code 2; UI refuses to
  render content). No silent downgrade exists.
- Invalid values in either variable are rejected in every mode.

### Path coverage requirements (operator checklist)

The encryption must cover the physical volumes behind:

1. `<project>/data/` — datasets, exports, translation outputs, QA review files
2. `<project>/outputs/` — labeled.csv, scores, reports, charts, logs
3. the cache root (`AUTISM_CACHE_DIR`, default `D:\autism_assistant_cache`
   or `<project>/.cache`) — embedding cache + model/dataset caches
4. **any HF cache outside the project** — on the host, `load_dataset` uses
   `HF_HOME`/`HF_HUB_CACHE` if set, else the user profile default
   (`~/.cache/huggingface`); set `HF_HOME` onto the encrypted volume or
   include the default location in the protected volume.

### OS setup & verification guidance (operator actions — the app cannot do these)

- **Windows (BitLocker):** enable per volume (`manage-bde -on <drive>:`),
  verify with `manage-bde -status <drive>:`; see Microsoft BitLocker docs.
- **Linux (LUKS/dm-crypt):** encrypt the volume/block device
  (`cryptsetup luksFormat` / `luksOpen`), verify with `cryptsetup status`
  and `lsblk -f`; see dm-crypt wiki.
- **macOS (FileVault):** System Settings → Privacy & Security → FileVault,
  verify with `fdesetup status`; see Apple Platform Security.
- **Docker:** bind mounts (`./data`, `./outputs`) inherit the **host**
  filesystem — encrypt the host volume that hosts them. Named volumes
  (e.g. `model_cache`) physically live under the Docker daemon's root
  (typically `/var/lib/docker/volumes/...` on Linux) — they are **not**
  encrypted by Docker; they are protected only when that host filesystem
  is. Model-weight caching is not patient-data protection. Temporary
  files/backups (e.g. translation `.corrupt-*` files) are written next to
  their outputs — same volume, same coverage. Backups made outside these
  volumes must be encrypted separately by the operator.

### Existing data & migration

No automatic migration: existing unencrypted files are left untouched.
To protect existing data, either enable encryption on the volume holding
it (OS-level, transparent to existing files) or move copies onto an
encrypted volume manually. Document any such procedure locally; this
repository performs none.

### Limitations & remaining risks

- Disk encryption protects **at rest** only; once the volume is
  unlocked/mounted, any process running with the user's privileges can read
  the files. It does not protect against a compromised OS session.
- **No HIPAA compliance, clinical readiness, or complete PHI protection is
  claimed or implied** — a login screen (Issue #12) plus disk encryption do
  not make this multi-tenant or clinical-grade.
- Git exclusion rules (`.gitignore`) prevent commits; they are **not**
  encryption.
- The application's verification of any of the above is **zero by
  design**; everything in this section beyond the mode/declaration
  enforcement is operator responsibility.
