# Annotation Guideline — Gold Evaluation Set (Issue #8)

**Read this fully before annotating.** This protocol defines the human
labeling task for the gold evaluation set. It is part of the governance in
`data/gold/README.md` and the schema in `gold_eval.py` /
`gold_eval.schema.json`.

**Status: annotation workflow ready — zero human labels exist yet.** The
template (`gold_annotation_template.jsonl`, 300 records) is unfilled. No
accuracy, calibration, or evaluation claim may be made until genuinely
human-annotated records are committed and validated by `python gold_eval.py`.

## 1. What you are labeling

Each worklist record contains the **patient-side text** (the exact text the
pipeline labels) drawn from `data/mentalchat16k_en_for_translation.jsonl`
(16,005 valid unique records; selection seed and provenance are recorded in
`gold_split_manifest.json`). Records carry a `partition`:
`gold_calibration` (used to select thresholds) or `gold_evaluation`
(held out — used only for final metrics). Never mix the two during
annotation or analysis.

## 2. Categories and operational meaning

Definitions come from `facts.py::CATEGORIES` — the same definitions the
prototypes use. They must never be re-interpreted per labeler.

| Label | Category | Operational meaning — assign when the record is substantially about… |
|---|---|---|
| `A` | Social Communication | difficulties with social interaction, reciprocity, reading social cues, conversation pragmatics |
| `B` | Sensory Processing | sensory sensitivities, overload, stimming described as sensory regulation, sensory-seeking/avoiding |
| `C` | Emotional Regulation & Stress | anxiety, meltdowns/shutdowns, emotional dysregulation, coping with distress |
| `D` | Routine & Predictability | distress at plan changes, need for sameness, routines, transitions |
| `E` | Special Interests & Strengths | intense focused interests, strengths, deep expertise topics |
| `F` | Diagnosis & Support | seeking/describing diagnosis, assessment, services, support needs |
| `G` | Autism Knowledge & Awareness | general discussion of what autism is, awareness, myths, education |

## 3. Inclusion and exclusion criteria

- **Include** a category when the text explicitly describes experiences,
  needs, questions, or content matching the operational meaning — in the
  author's own framing, not one you infer from a single keyword.
- **Exclude** a category when the mention is incidental, metaphorical, or
  solely a DSM-style term used without personal/contextual meaning.
- Do **not** infer autism, diagnosis status, or identity from tone, writing
  style, or the fact that the record appears in this dataset. The corpus is
  general mental-health counseling text; most records plausibly match **no**
  category.

## 4. Multi-label annotation

- Assign **every** category that substantially applies (`final_labels` is a
  list — e.g. `["B", "C"]`). Records legitimately have zero, one, or several
  categories.
- Do not add a weaker category "just in case" — multi-label means all that
  apply, not all that might.

## 5. No applicable category

- If **no** category applies, keep `final_labels: []` (an empty list). That
  is a deliberate, valid annotation — the most common outcome for this
  corpus. Never force a category onto an unrelated record.

## 6. Ambiguity and insufficient context

- Genuinely torn between two categories: include both.
- Text too short, fragmented, or context-free to decide: set
  `final_labels: ["UNSURE"]`. `UNSURE` must appear **alone** (never
  combined with A–G).
- Use `notes` for a one-line rationale when the case was hard; rationales
  are optional and should never contain identifying information.

## 7. Confidence and rounds

- Optional `confidence`: `sure` / `leaning` / `guess`.
- Two-round protocol (round 1 independent, round 2 reconciliation) is
  defined in `data/gold/README.md`; `round` is 1 or 2, `adjudicated` marks
  reconciled records.

## 8. Identity and provenance (privacy rules)

- Use a **pseudonymous** `labeler_id` (e.g. `annotator-01`) — never a real
  name, email, or handle that identifies you.
- Do not copy identifying details from the text into `notes`; the text
  itself stays in the worklist record and is not duplicated anywhere.
- `provenance` (record hash, dataset sha256, selection seed) is pre-filled —
  do not alter it. It lets auditors trace every gold record back to its
  source without exposing anyone.

## 9. Deliverable

Fill the worklist fields (`labels`, `status: "annotated"`, `labeler_id`,
optional `confidence`/`notes`), keep `origin: "human"` and the provenance
untouched, save as `gold_labels_v1.jsonl`, and run:

```bash
python gold_eval.py     # must report valid_file=true and gold_labels>0
```

Records with `status != "annotated"` are counted as pending worklist
entries — they are never gold labels. Model-generated labels are forbidden:
`origin` must be `"human"`, and copying model predictions into the gold set
is a governance violation, not a shortcut.
