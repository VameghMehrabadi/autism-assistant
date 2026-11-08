# Gold Evaluation Set — Labeling Instructions & Governance

**Purpose.** A small, version-controlled, **human-labeled** evaluation set used to
calibrate the semantic-labeling thresholds and to measure classification quality
against ground truth. Model-generated labels are **never** part of this set.

**Status: annotation workflow ready (Issue #8) — zero human labels exist
yet.** The selection and the unfilled annotation template are generated
(`gold_split.py`, seed 42, 300 records — see
`gold_split_manifest.json`); **no accuracy, calibration, or evaluation
claim may be made until genuinely human-annotated records are committed
and validated by `python gold_eval.py`.** See
`ANNOTATION_GUIDELINE.md` for the full protocol.

## 1. What gets labeled (implemented)

- `python gold_split.py --size 300` selects **300 records** (within the
  200–500 range of Issue #8) from
  `data/mentalchat16k_en_for_translation.jsonl` (16,005 valid unique
  records) using the existing Phase-2 grouping machinery: records are
  selected and partitioned **whole-group** (297 groups for 300 records) so
  related records never leak between the calibration (105) and evaluation
  (195) partitions. Seed, algorithm, dataset fingerprint, group keys and
  leakage verification are recorded in `gold_split_manifest.json`
  (content-free — committed); the per-record worklist
  `gold_annotation_template.jsonl` contains raw conversation text and is
  **gitignored by design**.
- Reproducible: same dataset + same seed → identical selection.

## 2. Labeling task

For each record, the labeler reads the **patient-side text** (exactly what
the pipeline labels) and assigns a **multi-label** set — every category that
substantially applies (Issue #8); the empty list is a deliberate "no
applicable category" annotation, and `UNSURE` stands alone when the record
is undecidable (full protocol: `ANNOTATION_GUIDELINE.md`):

| Label | Meaning |
|---|---|
| `A` | Social communication / interaction difficulty |
| `B` | Sensory processing / overload |
| `C` | Emotional regulation & stress |
| `D` | Routine & predictability needs |
| `E` | Special interests & strengths |
| `F` | Diagnosis & support seeking |
| `G` | General autism knowledge/awareness discussion |
| `UNSURE` | The record does not clearly fit any category |

Definitions are the category titles/titles_fa in `facts.py::CATEGORIES` — the
**same** definitions the prototypes use. Category definitions must never be
re-interpreted per labeler.

## 3. Process & disagreement handling

1. **Two independent labelers** label every record (round 1), blind to each
   other and to all model outputs.
2. Agreement is computed per record; **disagreements go to round 2**: both
   labelers see only the *other* labeler's label (not the model's) and may
   revise once.
3. Records still disagreeing are **adjudicated** by a third reviewer whose
   decision is final (`adjudicated: true`). If no consensus is possible, the
   record's `final_label` is `UNSURE` and it is excluded from accuracy
   metrics but retained in the file (audit trail).
4. `UNSURE` records are **reported** (counted) and excluded from per-category
   metrics.

## 4. Versioning & separation

- Labels live in `data/gold/gold_labels_v1.jsonl` (one record per line,
  conforming to `gold_eval.schema.json`). Version increments on any
  relabeling; every version is retained.
- **Human labels are distinguishable from model labels**: each gold record
  requires `labeler_id` (human initials), `origin: "human"`, and `round`.
  Files under `outputs/` are model labels and must never be merged into
  `data/gold/`.
- **Evaluation data is separate from development data**: gold record `idx`
  values must be excluded from any future threshold-fitting development set;
  `data_validation.create_splits` must be regenerated after gold selection if
  a gold idx lands in an eval/dev boundary that would bias calibration.
- The committed schema (`gold_eval.schema.json`) is normative; the validator
  (`gold_eval.py::validate_gold_file`) must pass on every committed version.

## 5. What this set is used for

- **Threshold calibration** per embedding model: choose the operating point
  on the model's own score scale that maximizes agreement with gold labels
  (report the metric, e.g. macro-F1, and the confusion matrix).
- **Model comparison**: agreement with gold is the *only* basis for model
  quality claims. Raw cosine scores never are.

## 6. Honesty rules

- Do not fabricate, model-generate, or "pre-fill" labels. A labels file with
  any `origin != "human"` record is invalid and must fail validation.
- Until a gold file exists, all reports must state: *"No human-labeled
  calibration data exists; thresholds are uncalibrated geometric heuristics."*
