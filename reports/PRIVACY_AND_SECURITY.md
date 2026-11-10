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
- MentalChat16K licensing status **unclear for derivatives** — MIT tag is
  present upstream but its scope for the interview-derived component and for
  translated/redistributed derivatives is not established; see the dated
  assessment in section 8 and the `license_guard` export refusal.
- Docker image build not exercised in this environment (daemon unavailable);
  compose configuration is YAML-validated.


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
