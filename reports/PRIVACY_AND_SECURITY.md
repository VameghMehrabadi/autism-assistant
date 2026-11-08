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
- MentalChat16K licensing status **unclear for derivatives** — MIT tag is
  present upstream but its scope for the interview-derived component and for
  translated/redistributed derivatives is not established; see the dated
  assessment in section 8 and the `license_guard` export refusal.
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

## 8. Network isolation & UI authentication (Issue #12 — reviewed 2026-11-07)

### Deployment boundary — three distinct layers

| Layer | Default | Controlled by |
|---|---|---|
| **Container listening address** | `0.0.0.0` (inside the container — required for Docker port forwarding; does NOT expose the host) | Dockerfile CMD (`--server.address=0.0.0.0`) |
| **Host published address** | `127.0.0.1` only — other machines cannot connect | `docker-compose.yml` → `${UI_BIND_HOST:-127.0.0.1}:${UI_HOST_PORT:-8501}:8501` |
| **Authentication boundary** | none required (service is loopback-only) | `BIND_PUBLIC` / `APP_AUTH_PASSWORD` via `security_config.py` + `ui_lib/auth.py` |

Local (non-Docker) runs bind to `localhost` via `.streamlit/config.toml`
(`server.address = "localhost"`) — the UI is not reachable from another
machine by default. XSRF protection stays enabled and CORS is enabled
(`server.enableCORS = true`); neither was weakened.

### Public mode — explicit, authenticated, fail-closed

`BIND_PUBLIC=1` is the only switch that requests public exposure, and it
requires **both** of these to take effect:

```bash
# Default local mode (localhost only, no credential needed):
docker compose up --build
# or locally: .\launch_ui.ps1   /   python -m streamlit run ui.py

# Explicit PUBLIC mode (fails closed without the credential):
BIND_PUBLIC=1 APP_AUTH_PASSWORD='choose-a-long-passphrase'   docker compose -f docker-compose.yml -f docker-compose.public.yml up --build
```

- `docker-compose.public.yml` publishes `0.0.0.0:${UI_HOST_PORT:-8501}:8501`
  (override file, `!override` tag) and sets `BIND_PUBLIC=1`.
- `APP_AUTH_PASSWORD` is required by compose interpolation
  (`${APP_AUTH_PASSWORD:?…}`) — `up`/`config` abort without it (verified via
  `docker compose config`).
- The application validates independently (`security_config.py`, called by
  `ui_lib.auth.require_auth()` on every page): public mode without a
  credential, an invalid `BIND_PUBLIC` value, or a credential shorter than 8
  characters refuses to serve content (defense in depth for non-Docker
  launches too).
- **Stop public exposure:** stop using the override file and re-run plain
  `docker compose up -d` (or unset `BIND_PUBLIC`); the port is loopback-only
  again.

### Authentication mechanism (selected)

- **Implemented:** application-level password gate (`ui_lib/auth.py`) —
  single shared operator credential from `APP_AUTH_PASSWORD` or
  `.streamlit/secrets.toml` (`[auth] password`), constant-time comparison
  (`hmac.compare_digest`), session flag only (no password in session
  state), gate before any content on `ui.py` and all five pages, logout
  clears state. Streamlit 1.66's native `st.login` was evaluated and
  rejected: it is OIDC-provider-based and requires an external identity
  provider plus registered redirect URLs — unsuitable for an offline
  research prototype.
- **Not implemented (recommended for production/clinical use):** an
  authenticated, TLS-terminating reverse proxy (e.g. Caddy with basic_auth
  or forward_auth) in front of the app, with the Streamlit port kept
  loopback-only on the proxy host. TLS is REQUIRED for any remote/clinical
  network access — passwords cross the network unencrypted otherwise, and
  the app itself serves plain HTTP. A proxy claim is never treated as
  proof of authentication; client-supplied forwarded-auth headers are not
  trusted.

### What this is NOT (limitations)

- **XSRF protection is not authentication** — it guards cross-site request
  forgery, not who opens the page.
- **No multi-user data isolation**: one shared credential, one shared
  workspace — conversations, labels, exports, `data/`, `outputs/` are common
  to all users of the instance. This is not a multi-tenant system; a login
  screen does not make it suitable for clinical deployment.
- Browser-level authentication behavior was not exercised here (no browser
  test in this environment); the gate logic, page coverage and fail-closed
  paths are unit-tested, and compose validation is configuration-level.
- Docker runtime verification (image build, real port exposure, remote
  connection attempt) was not executed — Docker daemon unavailable in this
  environment; reported as BLOCKED in the Issue #12 PR.
## 8. Dataset licensing assessment — MentalChat16K (reviewed 2026-11-07)

**Dataset:** [ShenLab/MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)
**Revision used by this project:** `5f60cd380cfc58f0f12d44892bed41ee3670a70a`
(recovered from the project's `datasets` download metadata, which pins
`hf://datasets/ShenLab/MentalChat16K@5f60cd38…/{Interview_Data_6K,Synthetic_Data_10K}.csv`).
The Hugging Face API confirms this exact commit is **still the current `main`
HEAD** (checked 2026-11-07), so current metadata equals the historical
revision — no drift between the project's copy and today's upstream metadata.

### Sources examined

| Source | Finding |
|---|---|
| HF API + dataset card ([page](https://huggingface.co/datasets/ShenLab/MentalChat16K), raw `README.md` @ `5f60cd38`) | License metadata: `license: mit` tag and `cardData.license: mit`. Card body does **not** scope the license; it describes both components but its Limitations section claims the dataset is "entirely synthetic" while its own Interview Data section describes 6,338 QA pairs paraphrased from 378 real clinical-trial interview transcripts — an internal inconsistency that reinforces the scope ambiguity |
| Official repo ([PennShenLab/MentalChat16K](https://github.com/PennShenLab/MentalChat16K)) | Full MIT `LICENSE` file ("Copyright (c) 2024 PennShenLab") covering the repository; GitHub license API: MIT. The repo hosts pipelines/code, not the dataset CSVs; the README does not distinguish rights between synthetic and interview-derived data ("handle the data with care" exhortation only) |
| Paper (KDD 2025, [DOI 10.1145/3711896.3737393](https://doi.org/10.1145/3711896.3737393), [PMC12520247](https://pmc.ncbi.nlm.nih.gov/articles/PMC12520247/)) | ACM copyright notice governs the *article* (republishing/redistribution requires ACM permission) — **article copyright does not define the dataset license**. No explicit dataset license stated. Interview-derived subset: paraphrased from 378 PISCES clinical-trial transcripts; sensitive/identifiable information removed before release; release consent obtained "upon removal of all sensitive and identifiable information" — a conditional release permission, not an unrestricted license |

### Component distinction

- **Synthetic component** (GPT-3.5 Turbo + Airoboros): machine-generated text;
  MIT (if it applies to the data) most plausibly covers this.
- **Interview-derived component** (paraphrased PISCES caregiver/coach
  transcripts): originates from **human speech of research participants**.
  Upstream consent language authorizes *release of the de-identified
  paraphrased data*; no source states that recipients receive redistribution,
  translation, or commercial rights for this component.

### Permission matrix (as of 2026-11-07)

| Permission | Status | Basis |
|---|---|---|
| Access / download | **CONFIRMED** | Public, non-gated HF dataset |
| Use for research | **CONFIRMED** | HF card + repo README state research intent; MIT tag |
| Modify / translate (private local processing) | **CONFIRMED** | MIT tag is the strongest available statement; private processing creates no distribution |
| Redistribute derivatives (e.g. `data/mentalchat16k_fa.jsonl`) | **UNCLEAR** | No source explicitly authorizes derivative redistribution; interview-derived provenance + conditional consent language leave scope unresolved |
| Commercial use | **UNCLEAR** | MIT text would permit it for code, but no source establishes it for the dataset, particularly the interview-derived component |
| Publish data in a public repository | **UNCLEAR** — refused by default | No explicit upstream permission for re-publishing the corpus or derivatives |

### Project policy while uncertainty remains

1. `license_guard.py` is the **single centralized guard** for every export
   path (`export_for_translation.write_export`, `translate_dataset` FA output
   + error log, `translation_qa.write_review_file`, Translate/Export UI):
   destinations **inside** the project workspace (`config.ROOT`) are private
   and always allowed; destinations **outside** it are treated as public
   distribution and refused while `LICENSE_STATUS == "unclear"`.
2. CLI override `--acknowledge-license` (UI: acknowledgment checkbox)
   proceeds after printing a prominent warning that records the user's
   acknowledgment of the unresolved risk — **it does not grant legal
   permission**.
3. The Persian dataset `data/mentalchat16k_fa.jsonl` and every raw export
   stay **gitignored**; they must not be committed or publicly redistributed
   until the derivative-redistribution permission is established.
4. Citing the original dataset is **not** equivalent to holding
   redistribution permission.
5. Re-assessment triggers: upstream publishes an explicit data license
   covering derivatives, or written permission is obtained. Any status flip
   must update `license_guard.LICENSE_STATUS` and this section with a new
   review date.

**Unresolved legal questions** (not legal advice): exact scope of the MIT
tag over data files; whether interview-derived paraphrases carry any
license at all; whether the PISCES consent permits downstream
redistribution/translation; interaction with third-party generation-tool
terms (OpenAI, Mistral). These require the upstream authors or counsel —
this report deliberately draws no definitive legal conclusion.
