# Release Readiness — Autism Assistant (Authoritative Checklist)

> **TL;DR (5-minute read): the project is NOT ready for a `v1` tag.**
> All engineering phases are implemented and their local verifications pass
> (248/248 tests, browser-verified GUI, three executed model-comparison runs).
> However, **four release blockers remain**, each requiring execution that has
> not happened yet: (1) a funded Z.ai account + real translation run,
> (2) comparison runs 4–5, (3) a completed Docker image build + runtime smoke,
> (4) human labeling of the gold evaluation set. Do **not** merge
> `release/main` into `main` or tag `v1` until every checkbox in
> [§3 Release Checklist](#3-release-checklist-authoritative) is checked with
> linked evidence.

## 1. Status vocabulary and evidence rules

A requirement may be labeled **`COMPLETE`** only when this document (or a
report it links) contains **all three** of:

1. the actual command or procedure that was executed;
2. the observed result (output, artifact path, or test reference);
3. the date/location of the verification.

The following labels are used for everything else:

| Label | Meaning |
|---|---|
| `COMPLETE` | Executed and verified; evidence linked in this document |
| `IMPLEMENTED — NOT EXECUTED` | Code/docs exist and are tested at unit level, but the real-world execution step has not been run |
| `BLOCKED` | Execution attempted or required, but an external dependency is missing (credentials, billing, daemon, network) — exact missing prerequisite stated |
| `PENDING HUMAN REVIEW` | Requires human judgment (e.g. gold labeling) that has not occurred |

**Implementation alone is never proof of execution.** Evidence is never
manufactured: every claim in §3 links to a committed report, artifact, or test.

## 2. Current status at a glance

| Area | Label | Evidence / reference |
|---|---|---|
| Data pipeline (loading, validation, dedup, encodings, leakage guard) | `COMPLETE` | `reports/PHASE_2_DATA_ENGINEERING.md`, `reports/PHASE_1_ENCODING_INTEGRITY.md`; 248-test suite |
| Semantic labeling engine (E5 prefixes, cache, thresholds, uncertainty, prototype evidence) | `COMPLETE` | `reports/PHASE_3_LABELING_ENGINE.md`; real MiniLM end-to-end run (`outputs/comparison/1_fa_en_minilm/`) |
| Comparison runs 1–3 (MiniLM / bge-m3 / e5-large × EN × 16,005) | `COMPLETE` | `outputs/comparison/comparison_summary.json` (3 runs, n=16,005 each); `compare_runs.py --verify-alignment` → `all_records_aligned: true` |
| Evaluation methodology (score-scale honesty, alignment check, gold governance) | `COMPLETE` | `reports/PHASE_4_EVALUATION_VALIDITY.md`; `data/gold/README.md`, `gold_eval.py` |
| Translation pipeline implementation (providers, checkpoint/resume, QA, quota guard) | `COMPLETE` (implementation + mocked tests) | `reports/PHASE_3_ZAI_RELIABILITY.md`; `reports/PHASE_6_FINALIZATION.md` |
| GUI (Streamlit, disclaimers, status panels) | `COMPLETE` | browser-verified (screenshots in `gui-test-screenshots/`) |
| Docker configuration (compose + Dockerfile) | `IMPLEMENTED — NOT EXECUTED` | YAML-validated only; **image build never completed** |
| Real Z.ai translation | `BLOCKED` | Z.ai error **code 1113 — insufficient balance / no resource package** (probe `2026-11-05`, `reports/PHASE_6_FINALIZATION.md`) |
| Comparison runs 4–5 (FA dataset) | `BLOCKED` | depends on real translation output (`data/mentalchat16k_fa.jsonl` absent) |
| Docker build + runtime smoke | `BLOCKED` | Docker daemon became available but the base-image pull did not complete in the execution window (stopped cleanly; no image produced) |
| Human calibration of gold data | `PENDING HUMAN REVIEW` | no gold labels exist (`data/gold/` contains schema + instructions only) |

Test evidence for the implementation work: **248 passed** (`pytest`, working
venv) and 246 in a fresh isolated venv — recorded in
`reports/PHASE_6_FINALIZATION.md` and `DEVELOPMENT_LOG.md`.

## 3. Release Checklist (authoritative)

**Every box must be checked with linked evidence before the `v1` tag is
created.** Close items in future issues by appending the execution evidence
(command + result + artifact reference) next to the box.

### A. Credentials & external services
- [ ] **Z.ai credentials/configuration verified** — status: `BLOCKED`.
      Evidence required: `--probe` output showing a successful (`exit 0`)
      request against `https://api.z.ai/api/paas/v4` with model
      `glm-5.3-flash`. Last attempt (2026-11-05): `HTTP 429`, provider code
      **1113 "Insufficient balance or no resource package"** — the account
      needs a recharge/active resource package, and the plan must be attached
      to **this** key's account (`ZAI_API_KEY`).

### B. Translation
- [ ] **Real Z.ai translation successfully executed and its output validated** —
      status: `BLOCKED` (by A) + `IMPLEMENTED — NOT EXECUTED`.
      Evidence required: `translate_dataset.py --provider zai …` run log with
      the engine summary (translated/flagged/failed counts),
      `data/translation_summary.json` with `completeness.complete: true` for
      all 16,005 source records, and a `translation_qa` report
      (`issues_total`, `NOT_PERSIAN = 0` for translated records, flagged
      records routed to `data/translation_review.jsonl`).
- [ ] **20-record pilot quality reviewed and accepted by a human** — status:
      `PENDING HUMAN REVIEW`. Evidence required: pilot output + reviewer
      sign-off on autism terminology, emotional tone, and patient/counselor
      separation.

### C. Comparison matrix
- [ ] **Pipeline run 4 completed and evidence recorded** — status: `BLOCKED`
      (by B). Evidence required: `outputs/comparison/4_fa_fa_bge-m3/` with the
      full artifact set (`gap_analysis.{json,md}`, `labeled.csv`, `scores.npy`,
      `run_meta.json`, chart) and its row present in the regenerated
      `comparison_summary.json`.
- [ ] **Pipeline run 5 completed and evidence recorded** — status: `BLOCKED`
      (by B). Evidence required: `outputs/comparison/5_fa_fa_e5-large/` with
      the same artifact set and summary row.
- [ ] **Unified comparison report regenerated including runs 1–5** — status:
      `PENDING`. Evidence required: `comparison_summary.json` containing five
      runs and `compare_runs.py --verify-alignment` reporting
      `all_records_aligned: true` with runs 4–5 included.

### D. Docker
- [ ] **Docker image successfully built using the documented build command** —
      status: `IMPLEMENTED — NOT EXECUTED` (daemon available but the build did
      not finish: base-image pull stalled; stopped cleanly, no image produced).
      Evidence required: full output of
      `docker build --progress plain -t autism-assistant:latest .` ending in a
      successful image, plus `docker images` showing
      `autism-assistant:latest`.
- [ ] **Docker container/runtime smoke test completed** — status:
      `IMPLEMENTED — NOT EXECUTED`. Evidence required:
      `docker compose up` (or `docker run`) with
      `GET /_stcore/health → 200` from the containerized UI, and one CLI
      pipeline command executed inside the container
      (`docker compose run --rm pipeline main.py --limit 5 …`).

### E. Evaluation quality
- [ ] **Human calibration of gold evaluation data completed** — status:
      `PENDING HUMAN REVIEW`. Evidence required: committed
      `data/gold/gold_labels_v*.jsonl` passing `python gold_eval.py`
      (`valid: true`), with two labeler rounds + adjudication recorded per
      `data/gold/README.md`; no model-generated labels.
- [ ] **Per-model threshold calibration computed from gold labels and
      reviewed** — status: `PENDING HUMAN REVIEW` (depends on previous item).
      Evidence required: per-model operating points (macro-F1 / confusion
      matrix vs gold) recorded in a calibration report replacing the current
      uncalibrated-heuristic caveat.

### F. Release finalization
- [ ] **Relevant evaluation results reviewed and accepted** — status:
      `PENDING HUMAN REVIEW`. Evidence required: sign-off note in this
      checklist referencing the accepted reports.
- [ ] **All required release documentation updated** — status: `PENDING`
      (README status tables, this checklist, and `PROJECT_FINAL_REPORT.md`
      must reflect the post-translation state; current versions already
      distinguish implemented vs executed work).
- [ ] **Final release verification completed before creating the `v1` tag** —
      status: `PENDING`. Evidence required: full `pytest` suite green on the
      release commit, `compare_runs.py --verify-alignment` →
      `all_records_aligned: true`, translation QA report attached, and a
      fresh-venv installation run of the release commit.

## 4. What is already verified (kept as `COMPLETE`)

The following retain `COMPLETE` because execution evidence exists and is
linked; these are **not** downgraded:

- Dataset integrity: 16,084 raw → 16,005 valid unique records; 0 invalid JSON,
  0 duplicate IDs, 0 Persian contamination in the export
  (`reports/PHASE_1_ENCODING_INTEGRITY.md`).
- UTF-8 boundary-detection fix with end-to-end roundtrip test (same report).
- Comparison runs 1–3 on the full dataset with equivalent-input verification
  (`--verify-alignment`: identical records, identical label-text fingerprint
  `cb4b81e2867729e0`, threshold 0.35 recorded).
- Z.ai pipeline reliability hardening (fail-fast kinds, auth abort, truncation
  rejection, fsync checkpoints, torn-output repair, incremental embedding
  cache, centralized actual-vs-estimated token accounting) — mocked tests only;
  real execution deliberately not attempted after code 1113
  (`reports/PHASE_3_ZAI_RELIABILITY.md`, `reports/PHASE_6_FINALIZATION.md`).
- 248-test suite green (working venv) and 246-test fresh-venv installation.

## 5. Historical note

This document was `reports/PHASE_5_RELEASE_READINESS.md` from the Phase-5
execution, whose status table labeled the release phase `COMPLETE` although the
Docker build had never been exercised — the discrepancy this checklist corrects
(issue #1). The original Phase-5 execution evidence (GUI browser verification,
246-test suite, fresh-venv install, secret scan) remains valid and is preserved
in the sections above and in `DEVELOPMENT_LOG.md`.

---

## 6. Phase-5 execution record (preserved, issue #1)

> Original document from the Phase-5 execution. Its top-line status read
> `COMPLETE`, which was misleading for *release readiness* (the Docker build
> had never been exercised). Corrected by issue #1: the phase's **implementation
> and local verification** were complete (evidence below); **release readiness
> itself is governed by §3 above.**

**Original status line (corrected):** ~~"Status: COMPLETE — GUI verified in a
real browser with the new safeguards, Docker credential forwarding fixed and
YAML-validated, documentation consolidated and refreshed, full verification
executed."~~ → *Implementation and local verification complete; Docker image
build and runtime smoke remained `IMPLEMENTED — NOT EXECUTED` and are now
release-checklist items D-1/D-2.*

### 1. GUI (verified in-browser, port 8602)

| Change | Verified |
|---|---|
| Translate/Export page now uses the safe export path (`export_for_translation.write_export`) instead of its own ad-hoc loop: **COMPLETE/INCOMPLETE export status shown in-UI**, sidecar metadata (`*.export_meta.json`) rendered, translation-status panel (engine summary + completeness) added, honest "No translation has been run yet" caption | ✅ DOM-verified + screenshot |
| Default export limit on the page changed 500 → **full dataset**, with a caption explaining that limited exports are marked INCOMPLETE | ✅ |
| Research-use / non-diagnostic captions added to **Quick Label** and **Compare Models** (Home/Run Pipeline/Results already had them from Phase 4) | ✅ Quick Label DOM-verified; Compare caption follows the identical pattern |
| Quick Label retains the Phase-3 honesty captions (cosine ≠ probability; margin heuristic) | ✅ (previously browser-verified) |

No redesign — additive captions and panels only.

### 2. Docker

- **Confirmed**: `docker-compose.yml` did not forward `ZAI_API_KEY` (nor `ZHIPU_API_KEY`/`ZAI_BASE_URL`) — the Z.ai provider could never authenticate from containers. **Fixed**: both `ui` and `pipeline` services now forward `ZAI_API_KEY`, `ZHIPU_API_KEY`, `ZAI_BASE_URL` (all `${VAR:-}` host-sourced; nothing hardcoded).
- **Bonus fix**: pre-existing YAML typo in the `ui` service — `SENTENCE_TRANSFORMERS_HOME=/cache/...` (`=` instead of `:`) — which mis-declared that environment variable. Both services now validate identically.
- Compose file **YAML-parsed and asserted**: 10 env vars per service, `ZAI_API_KEY`/`ZAI_BASE_URL` present, `SENTENCE_TRANSFORMERS_HOME` correct in both.
- Secure configuration documented (`reports/PRIVACY_AND_SECURITY.md` §2): host env → compose injection, never baked into the image.
- Image **build** still not exercised (no Docker daemon in this environment) — now tracked as release-checklist item D.

### 3. Documentation
- **`reports/PRIVACY_AND_SECURITY.md`** (new): consolidated privacy/security posture — log sanitization, credential handling, external-sharing gate, quota safety, leakage guard, status gating, export safety, gold-set origin rule, research-use safeguards, supply chain, and honest open items.
- **`DEVELOPMENT_LOG.md`**: Phase 1–5 entries appended (encoding fix, pipeline integrity, reliability hardening, evaluation validity, release readiness) with commit hashes; duplicate-blockers section removed.
- **`reports/PROJECT_FINAL_REPORT.md`**: phase table refreshed to the current 14-phase reality; test table updated to 246 with the fresh-venv evidence.
- **READMEs**: status tables updated as phases landed (Runs 1–3 executed; Phase-1–4 fixes referenced).

### 4. Full Verification (actual)

| Check | Result |
|---|---|
| Working-venv test suite | **246 passed** |
| Fresh isolated venv (final pinned requirements) | **246 passed in 77.99s** + `pip check` clean |
| GUI smoke (Streamlit 8602 + in-app browser) | Home warning, Translate/Export status panels, Quick Label disclaimer — all DOM-verified; server stopped afterward |
| Docker compose | YAML-parsed; both services' env asserted |
| Secret scan | `git ls-files` scanned for `sk-…` keys and quoted `ZAI_API_KEY=`/`OPENAI_API_KEY=` assignments: **0 real secrets**; one documentation placeholder reworded to avoid false positives |

### 5. Git
- Change set reviewed file-by-file (docs + pages + compose + one report placeholder fix); no unrelated changes touched; nothing discarded.
- **No secrets committed** (scan clean post-fix); generated artifacts (`data/`, `outputs/`) remain gitignored; only intentional tracked files committed.
- **Nothing pushed** — no remote configured at the time of the Phase-5 execution.

### 6. Known Limitations
- Docker image build remains unexercised (daemon unavailable); compose is validated to the YAML level only.
- `run_meta.json` for historical comparison runs predates the alignment metadata (documented; `--verify-alignment` reports it honestly).
- Gold evaluation set has schema + governance but no human labels yet.
- Retention policy for raw-conversation exports/labeled.csv still an operator responsibility.
