"""کالیبراسیون آستانه‌ی اختصاصی هر encoder (Issue #8) — فقط با برچسب انسانی.

Encoder-specific cosine threshold calibration from **genuine human-labeled
gold records**. Everything here refuses to run on model-generated labels:
the caller must pass records validated by ``gold_eval.validate_gold_file``
(``origin == "human"`` is re-checked defensively), and a run with fewer
human-labeled calibration records than ``MIN_CALIBRATION_RECORDS`` returns
``calibrated: False`` with the exact missing evidence — the shared 0.35
default stays **uncalibrated**.

Method: deterministic grid search over candidate thresholds maximizing
**macro F1 over categories with support > 0** (``evaluation_metrics.
macro_f1_supported``); ties resolve to the LOWER threshold (more conservative
recall). Per-category thresholds are selected only where both positives and
negatives reach ``MIN_POSITIVES_PER_CATEGORY`` — otherwise the category
falls back to the global threshold with an explicit status.

Cosine similarity is never interpreted as a probability; the selected
thresholds are decision operating points, not confidence estimates.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation_metrics import macro_f1_supported

MIN_CALIBRATION_RECORDS = 50
MIN_POSITIVES_PER_CATEGORY = 10
CATEGORIES = ("A", "B", "C", "D", "E", "F", "G")
GRID = [round(0.05 + 0.025 * i, 3) for i in range(38)]  # 0.05 … 0.975
CALIBRATION_STATUS_FILE = (
    Path(__file__).resolve().parent / "data" / "gold" / "calibration.json"
)


def select_threshold(
    y_true: list[set],
    score_rows: list[dict[str, float]],
    *,
    categories: tuple[str, ...] = CATEGORIES,
    grid: list[float] | None = None,
) -> dict:
    """Grid search for the threshold maximizing supported macro F1.

    Deterministic: ties resolve to the lower threshold. Returns the selected
    threshold, the achieved F1, and the grid size.
    """
    grid = GRID if grid is None else sorted(grid)
    best: dict | None = None
    for threshold in grid:
        y_pred = [
            {c for c, score in scores.items() if score >= threshold}
            for scores in score_rows
        ]
        f1 = macro_f1_supported(y_true, y_pred, categories)
        if best is None or f1 > best["macro_f1_supported"]:
            best = {"threshold": threshold, "macro_f1_supported": f1}
    if best is None:
        best = {"threshold": 0.35, "macro_f1_supported": 0.0}
    best["grid_size"] = len(grid)
    return best


def calibrate_encoder(
    gold_records: list[dict],
    scores_by_source_idx: dict[int, dict[str, float]],
    *,
    encoder: str,
    dataset_fingerprint: str | None,
    split_fingerprint: str | None,
    min_records: int = MIN_CALIBRATION_RECORDS,
    min_positives: int = MIN_POSITIVES_PER_CATEGORY,
    categories: tuple[str, ...] = CATEGORIES,
) -> dict:
    """Calibrate one encoder's thresholds from human-labeled gold records.

    Uses ONLY the ``gold_calibration`` partition of annotated
    (``status == "annotated"``, ``origin == "human"``) records. Insufficient
    human data → ``calibrated: False`` with the exact missing evidence; no
    threshold is invented.
    """
    for record in gold_records:
        if record.get("origin") != "human":
            return {
                "encoder": encoder,
                "calibrated": False,
                "reason": (
                    "non-human origin detected in gold records — calibration "
                    "refused (model-generated labels can never calibrate)"
                ),
            }
    calibration_rows = [
        record
        for record in gold_records
        if record.get("partition") == "gold_calibration"
        and record.get("status") == "annotated"
    ]
    n = len(calibration_rows)
    base = {
        "encoder": encoder,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "n_calibration_records": n,
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if n < min_records:
        base.update({
            "calibrated": False,
            "reason": (
                f"insufficient human-labeled calibration records: {n} < "
                f"{min_records} — the encoder's threshold remains "
                "uncalibrated"
            ),
        })
        return base

    y_true: list[set] = []
    score_rows: list[dict[str, float]] = []
    missing_scores = 0
    for record in calibration_rows:
        labels = set(
            record.get("final_labels")
            or ([record["final_label"]] if record.get("final_label") else [])
        )
        scores = scores_by_source_idx.get(record.get("source_idx"))
        if scores is None:
            missing_scores += 1
            continue
        y_true.append(labels)
        score_rows.append(scores)
    if len(score_rows) < min_records:
        base.update({
            "calibrated": False,
            "reason": (
                f"insufficient encoder score coverage: {len(score_rows)} < "
                f"{min_records} (missing scores for {missing_scores} "
                "records) — the encoder's threshold remains uncalibrated"
            ),
        })
        return base

    best_global = select_threshold(y_true, score_rows, categories=categories)
    per_category: dict[str, dict] = {}
    for category in categories:
        positives = sum(1 for labels in y_true if category in labels)
        negatives = len(y_true) - positives
        if positives >= min_positives and negatives >= min_positives:
            # single-category F1 = macro over the 1-category universe
            best_cat = select_threshold(
                y_true, score_rows, categories=(category,)
            )
            per_category[category] = {
                "threshold": best_cat["threshold"],
                "f1": best_cat["macro_f1_supported"],
                "positives": positives,
                "negatives": negatives,
            }
        else:
            per_category[category] = {
                "status": "insufficient_support",
                "positives": positives,
                "negatives": negatives,
                "fallback": "global",
            }
    base.update({
        "calibrated": True,
        "method": (
            "grid search maximizing macro F1 over categories with support "
            "> 0; ties resolve to the lower threshold; per-category "
            f"thresholds only with >= {min_positives} positives and "
            "negatives"
        ),
        "thresholds": {
            "global": best_global["threshold"],
            "per_category": per_category,
        },
        "global_macro_f1_supported": best_global["macro_f1_supported"],
    })
    return base


def threshold_status_for(encoder: str, calibration: dict | None) -> str:
    """Human-readable 0.35 status for one encoder (report/UI wording)."""
    entry = (calibration or {}).get("encoders", {}).get(encoder)
    if not entry:
        return (
            "uncalibrated — no human annotations available; 0.35 is the "
            "uncalibrated documented default"
        )
    if entry.get("calibrated"):
        thresholds = entry.get("thresholds", {})
        return (
            f"calibrated on human labels (global threshold "
            f"{thresholds.get('global')}; calibration records: "
            f"{entry.get('n_calibration_records')})"
        )
    return f"uncalibrated — {entry.get('reason', 'insufficient human labels')}"


def load_calibration(path: Path | None = None) -> dict | None:
    path = Path(path) if path else CALIBRATION_STATUS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def save_calibration(
    statuses: dict, path: Path | None = None
) -> Path:
    path = Path(path) if path else CALIBRATION_STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
