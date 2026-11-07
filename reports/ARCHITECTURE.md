# Architecture — Autism Assistant

Verified against source code on `release/main` (commit `64293cd`).

## Import Graph

The graph below shows **verified direct imports** extracted from AST parsing
of the source files. Standard-library imports are omitted. Optional/lazy
imports are annotated.

```
ui.py (Streamlit entrypoint)
  ├── config
  ├── ui_lib/                      ← shared UI helpers
  │     ├── caching.py → data_loader, labeler
  │     ├── display.py → facts
  │     └── widgets.py → config, facts, data_validation
  └── pages/
        ├── 1_Quick_Label.py       ← data_validation (mock guard), ui_lib
        ├── 2_Run_Pipeline.py      ← config, main (run_pipeline), ui_lib
        ├── 3_Compare_Models.py    ← config, compare_runs, main, ui_lib
        ├── 4_Results.py           ← config, ui_lib
        └── 5_Translate_Export.py  ← config, data_loader,
                                     export_for_translation, ui_lib

main.py (single-run orchestrator)
  ├── config
  ├── data_loader      ← config, data_validation, privacy_log
  │     └── data_validation  ← config
  ├── facts            ← (standalone: 7 categories A–G, bilingual prototypes)
  ├── gap_analysis     ← config, facts, labeler
  └── labeler          ← config, embedding_cache, facts, privacy_log
        ├── embedding_cache  ← config, numpy
        └── privacy_log      ← config

compare_runs.py (5-way comparison orchestrator)
  ├── config
  ├── facts
  ├── main (run_pipeline)
  └── matplotlib / numpy / pandas

translate_dataset.py (translation engine)
  ├── config
  ├── data_loader      ← (SourceRecord._parse_tagged_text)
  ├── privacy_log
  └── translation_qa   ← config, data_validation, translate_dataset
        └── data_validation ← config

export_for_translation.py
  ├── config
  └── data_loader (load_samples_en)

gold_eval.py (standalone validator — no project imports)
```

### Key observations

- `data_validation.py` is imported by `data_loader.py`, `translation_qa.py`,
  and `ui_lib/widgets.py` — it is the shared validation/provenance layer.
- `privacy_log.py` is imported by `data_loader.py`, `labeler.py`, and
  `translate_dataset.py` — all three pipeline stages share the same
  sanitization layer.
- `config.py` is imported by every module — it is the single source of
  truth for all tunable settings.
- `translate_dataset.py` imports `openai` **lazily** (inside the provider
  constructor) so the package is only required when actually calling the
  API. The Z.ai provider reuses the same OpenAI-compatible client with a
  different `base_url`.
- `gold_eval.py` has **zero project imports** — it is a standalone validator
  for the human-labeled gold evaluation set.

## Provider Architecture

`translate_dataset.py` implements a provider abstraction:

```
TranslationProvider (interface: translate(segments) → str)
  ├── OpenAIProvider   ← OpenAI-compatible client; serves openai AND zai
  │                      (base_url switch; Z.ai = api.z.ai/api/paas/v4)
  └── MockProvider     ← deterministic test double (content-marked output)
```

Provider selection: `--provider openai|zai|mock` CLI flag.
Default: `openai`. The `zai` provider reads `ZAI_API_KEY` from the
environment. The `--confirm-external-sharing` gate is required for both
real providers. The mock provider is refused at the production dataset path.

**Execution status:** no real translation has been executed. The Z.ai
endpoint returned HTTP 429 / code 1113 (insufficient balance) on every
attempt. The provider implementation and all reliability mechanisms are
tested with mocked responses only.

## Evaluation Flow

```
compare_runs.py
  ├── runs 1–5 via main.run_pipeline (labeling + gap analysis)
  ├── summarize_run / summarize_blocked_run
  ├── build_comparison_table → comparison_summary.{csv,json}
  ├── render_comparison_markdown → comparison_report.md
  ├── plot_comparison → comparison_shares.png
  └── verify_alignment (read-only cross-run equivalence check)
```

Runs 1–3 (Persian facts × English MentalChat16K × MiniLM/bge-m3/e5-large)
have been executed on the full 16,005-record dataset. Runs 4–5 (Persian
facts × Persian dataset) are BLOCKED — the Persian translated dataset does
not exist yet.
