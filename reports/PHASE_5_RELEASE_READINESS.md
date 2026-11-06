# Phase 5: GUI, Documentation & Release Readiness — Report

**Status: IMPLEMENTED — NOT EXECUTED (release readiness).** The Phase-5
implementation (GUI, docs, Docker config, test suite) is complete and locally
verified (246/246 tests, browser-verified GUI). However, release readiness
requires execution evidence that is still missing: the Docker image has never
been built, the real Z.ai translation has never been run, and the gold
evaluation data has never been labeled.

## Release checklist (authoritative — added by Issue #21)

**Every box must be checked with linked execution evidence before `v1`.**

- [ ] Z.ai credentials verified (`--probe` exit 0) — BLOCKED: account code 1113
- [ ] Real Z.ai translation executed (20-record pilot → 16,005 full run) and output validated — BLOCKED (by above)
- [ ] Pipeline run 4 completed and evidence recorded (`outputs/comparison/4_fa_fa_bge-m3/`) — BLOCKED
- [ ] Pipeline run 5 completed and evidence recorded (`outputs/comparison/5_fa_fa_e5-large/`) — BLOCKED
- [ ] Docker image built (`docker build --progress plain -t autism-assistant:latest .`) — IMPLEMENTED — NOT EXECUTED
- [ ] Docker container/runtime smoke (`docker compose up` + health check) — IMPLEMENTED — NOT EXECUTED
- [ ] Human calibration of gold evaluation data (`gold_eval.py valid: true`) — PENDING HUMAN REVIEW
- [ ] Per-model threshold calibration computed from gold labels — PENDING HUMAN REVIEW
- [ ] Relevant evaluation results reviewed and accepted — PENDING HUMAN REVIEW
- [ ] All release documentation updated — PENDING
- [ ] Final release verification before `v1` tag — PENDING

## 1. GUI (verified in-browser, port 8602)

| Change | Verified |
|---|---|
| Translate/Export page now uses the safe export path (`export_for_translation.write_export`) instead of its own ad-hoc loop: **COMPLETE/INCOMPLETE export status shown in-UI**, sidecar metadata (`*.export_meta.json`) rendered, translation-status panel (engine summary + completeness) added, honest "No translation has been run yet" caption | ✅ DOM-verified + screenshot |
| Default export limit on the page changed 500 → **full dataset**, with a caption explaining that limited exports are marked INCOMPLETE | ✅ |
| Research-use / non-diagnostic captions added to **Quick Label** and **Compare Models** (Home/Run Pipeline/Results already had them from Phase 4) | ✅ Quick Label DOM-verified; Compare caption follows the identical pattern |
| Quick Label retains the Phase-3 honesty captions (cosine ≠ probability; margin heuristic) | ✅ (previously browser-verified) |

No redesign — additive captions and panels only.

## 2. Docker

- **Confirmed**: `docker-compose.yml` did not forward `ZAI_API_KEY` (nor `ZHIPU_API_KEY`/`ZAI_BASE_URL`) — the Z.ai provider could never authenticate from containers. **Fixed**: both `ui` and `pipeline` services now forward `ZAI_API_KEY`, `ZHIPU_API_KEY`, `ZAI_BASE_URL` (all `${VAR:-}` host-sourced; nothing hardcoded).
- **Bonus fix**: pre-existing YAML typo in the `ui` service — `SENTENCE_TRANSFORMERS_HOME=/cache/...` (`=` instead of `:`) — which mis-declared that environment variable. Both services now validate identically.
- Compose file **YAML-parsed and asserted**: 10 env vars per service, `ZAI_API_KEY`/`ZAI_BASE_URL` present, `SENTENCE_TRANSFORMERS_HOME` correct in both.
- Secure configuration documented (`reports/PRIVACY_AND_SECURITY.md` §2): host env → compose injection, never baked into the image, never committed. Image **build** still not exercised (no Docker daemon in this environment).

## 3. Documentation
- **`reports/PRIVACY_AND_SECURITY.md`** (new): consolidated privacy/security posture — log sanitization, credential handling, external-sharing gate, quota safety, leakage guard, status gating, export safety, gold-set origin rule, research-use safeguards, supply chain, and honest open items.
- **`DEVELOPMENT_LOG.md`**: Phase 1–5 entries appended (encoding fix, pipeline integrity, reliability hardening, evaluation validity, release readiness) with commit hashes; duplicate-blockers section removed.
- **`reports/PROJECT_FINAL_REPORT.md`**: phase table refreshed to the current 14-phase reality; test table updated to 246 with the fresh-venv evidence.
- **READMEs**: status tables updated as phases landed (Runs 1–3 executed; Phase-1–4 fixes referenced).

## 4. Full Verification (actual)

| Check | Result |
|---|---|
| Working-venv test suite | **246 passed** |
| Fresh isolated venv (final pinned requirements) | **246 passed in 77.99s** + `pip check` clean |
| GUI smoke (Streamlit 8602 + in-app browser) | Home warning, Translate/Export status panels, Quick Label disclaimer — all DOM-verified; server stopped afterward |
| Docker compose | YAML-parsed; both services' env asserted |
| Secret scan | `git ls-files` scanned for `sk-…` keys and quoted `ZAI_API_KEY=`/`OPENAI_API_KEY=` assignments: **0 real secrets**; one documentation placeholder reworded to avoid false positives |

## 5. Git
- Change set reviewed file-by-file (docs + pages + compose + one report placeholder fix); no unrelated changes touched; nothing discarded.
- **No secrets committed** (scan clean post-fix); generated artifacts (`data/`, `outputs/`) remain gitignored; only intentional tracked files committed.
- **Nothing pushed** — no remote configured.

## 6. Known Limitations
- Docker image build remains unexercised (daemon unavailable); compose is validated to the YAML level only.
- `run_meta.json` for historical comparison runs predates the alignment metadata (documented; `--verify-alignment` reports it honestly).
- Gold evaluation set has schema + governance but no human labels yet.
- Retention policy for raw-conversation exports/labeled.csv still an operator responsibility.
