# Development Roadmap — Autism Assistant

**Test count:** 248 passing on `release/main` (commit `64293cd`).
Historical counts (27, 185, 246, 261) reflect earlier snapshots or unmerged
feature branches — see `DEVELOPMENT_LOG.md` for the chronology.

## Release blockers (must resolve before v1 tag)

The authoritative checklist with per-item evidence requirements is in
`reports/PHASE_5_RELEASE_READINESS.md` §3. Summary:

| # | Blocker | Status | What's needed |
|---|---------|--------|---------------|
| 1 | Z.ai account funded (code 1113) | BLOCKED | Recharge or activate a resource package on the account owning `ZAI_API_KEY` |
| 2 | Real Persian translation executed + validated | BLOCKED (by 1) | Run the 20-record pilot, review quality, then the full 16,005-record resumable run |
| 3 | Comparison runs 4–5 (Persian facts × Persian dataset) | BLOCKED (by 2) | Execute after the Persian dataset exists; verify with `--verify-alignment` |
| 4 | Docker image build + container runtime smoke | IMPLEMENTED — NOT EXECUTED | Run `docker build` and `docker compose up` on a machine with Docker daemon |
| 5 | Human calibration of gold evaluation data | PENDING HUMAN REVIEW | 2 labelers × 150–300 records + adjudication per `data/gold/README.md` |

## What currently works (executed and verified)

- **Data pipeline**: EN MentalChat16K loading (16,084 raw → 16,005 valid unique
  after dedup), Persian JSONL/CSV loading, schema validation, duplicate
  detection, encoding repair, language identification, dev/eval leakage guard.
- **Semantic labeling**: 7 categories A–G, 44 bilingual prototypes, three
  embedding models (MiniLM, bge-m3, e5-large), E5 query/passage prefixes,
  disk cache, per-category thresholds, explicit uncertainty decisions,
  matched-prototype evidence, deterministic results.
- **Comparison runs 1–3**: Persian facts × English dataset ×
  {MiniLM, bge-m3, e5-large} on the full 16,005 records — executed and
  verified (`outputs/comparison/`).
- **GUI**: Streamlit app with research-use disclaimers on all pages,
  browser-verified labeling and results.
- **Test suite**: 248 tests passing on the release baseline; CI workflow
  with offline guards and `pip check`.

## What is implemented but not executed

- **Translation pipeline**: providers (openai, zai, mock), checkpoint/resume,
  QA validation, review workflow, completeness accounting — all implemented
  and tested with mocked providers. Real execution blocked by Z.ai billing.
- **Docker build**: Dockerfile and compose validated; image build attempted
  but not completed.

## What requires human review

- Gold evaluation data labeling (2 labelers + adjudicator).
- Threshold calibration against human labels (current thresholds are
  uncalibrated geometric heuristics, not probabilities).
- Acceptance of evaluation results before any model-quality claim.

## Near-term actions (in order)

1. Recharge/activate the Z.ai resource package → re-run `--probe`.
2. Approve and execute the 20-record pilot → review quality.
3. Approve and execute the full 16,005-record translation.
4. Execute comparison runs 4–5 → regenerate the unified 5-run report.
5. Complete gold-data labeling → calibrate thresholds per model.
6. Docker build + runtime smoke on a machine with Docker daemon.

## Longer-term (post-v1)

- Fine-tuning embedding models on labeled data.
- RAG pipeline for autism-support responses.
- Longitudinal monitoring dashboards.
- Clinical validation (requires ethics approval and clinical partners).
