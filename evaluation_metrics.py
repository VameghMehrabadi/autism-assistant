"""ارزیابی مستقل (Issue #8) — متریک‌های چندبرچسبی + باز نمونه‌گیری گروهی.

Independent evaluation metrics for the human-labeled gold evaluation
partition. Pure functions over explicit label sets — no model, no network,
no threshold assumptions (the caller supplies thresholded predictions).

Multi-label convention: each record carries a **set** of true categories and
a **set** of predicted categories. Per category: tp/fp/fn/tn, precision,
recall, F1, support. Macro F1 = unweighted mean of per-category F1 (over the
category universe, counting zero-support categories as F1 = 0 — reported
with their support so readers can judge). Micro F1 = pooled tp/fp/fn across
categories.

Uncertainty: group-aware bootstrap resampling over the macro-F1 statistic —
resampling **groups** (never individual records) preserves the leakage
structure. Deliberately returns **no interval** when the sample is too small
(``min_groups``/``min_records``): a misleading interval is worse than none.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from collections.abc import Callable, Sequence

MIN_GROUPS_FOR_CI = 30
MIN_RECORDS_FOR_CI = 50
DEFAULT_BOOTSTRAP_ROUNDS = 1000


def _counts_for_category(
    y_true: Sequence[set], y_pred: Sequence[set], category: str
) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for true_set, pred_set in zip(y_true, y_pred):
        true_has = category in true_set
        pred_has = category in pred_set
        if true_has and pred_has:
            tp += 1
        elif pred_has:
            fp += 1
        elif true_has:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def multilabel_metrics(
    y_true: Sequence[set],
    y_pred: Sequence[set],
    categories: Sequence[str],
) -> dict:
    """Per-category precision/recall/F1/support + macro/micro F1.

    ``support`` is the number of true positives-plus-negatives... i.e. the
    number of records where the category is truly present (standard
    classification support = true instances of that category).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    per_category: dict[str, dict] = {}
    pooled_tp = pooled_fp = pooled_fn = 0
    for category in categories:
        counts = _counts_for_category(y_true, y_pred, category)
        precision, recall, f1 = _prf(counts["tp"], counts["fp"], counts["fn"])
        per_category[category] = {
            **counts,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": counts["tp"] + counts["fn"],
        }
        pooled_tp += counts["tp"]
        pooled_fp += counts["fp"]
        pooled_fn += counts["fn"]
    micro_p, micro_r, micro_f1 = _prf(pooled_tp, pooled_fp, pooled_fn)
    f1_values = [per_category[c]["f1"] for c in categories]
    macro_f1 = (
        round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0
    )
    return {
        "per_category": per_category,
        "macro_f1": macro_f1,
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "n_records": len(y_true),
    }


def macro_f1_for(
    y_true: Sequence[set], y_pred: Sequence[set], categories: Sequence[str]
) -> float:
    return multilabel_metrics(y_true, y_pred, categories)["macro_f1"]


def group_bootstrap_ci(
    groups: Sequence[str],
    y_true: Sequence[set],
    y_pred: Sequence[set],
    categories: Sequence[str],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_ROUNDS,
    seed: int = 42,
    min_groups: int = MIN_GROUPS_FOR_CI,
    min_records: int = MIN_RECORDS_FOR_CI,
) -> dict:
    """Group-aware bootstrap interval for macro F1 — or an honest refusal.

    Resamples whole groups with replacement (leakage-preserving structure).
    Returns ``{"ci": None, "reason": …}`` when the sample is too small for a
    meaningful interval: fewer than ``min_groups`` groups or fewer than
    ``min_records`` records.
    """
    if not (len(groups) == len(y_true) == len(y_pred)):
        raise ValueError("groups, y_true, y_pred must have the same length")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < min_groups or len(y_true) < min_records:
        return {
            "ci": None,
            "reason": (
                f"insufficient sample for an interval: {len(y_true)} "
                f"records in {len(unique_groups)} groups — at least "
                f"{min_records} records in {min_groups} groups required"
            ),
            "point_estimate": macro_f1_for(y_true, y_pred, categories),
        }
    by_group: dict[str, list[int]] = {}
    for position, group in enumerate(groups):
        by_group.setdefault(group, []).append(position)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sampled_groups = rng.choices(unique_groups, k=len(unique_groups))
        positions = [
            pos for group in sampled_groups for pos in by_group[group]
        ]
        true_subset = [y_true[pos] for pos in positions]
        pred_subset = [y_pred[pos] for pos in positions]
        values.append(macro_f1_for(true_subset, pred_subset, categories))
    values.sort()

    def _percentile(p: float) -> float:
        index = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
        return round(values[index], 4)

    return {
        "ci": [_percentile(0.025), _percentile(0.975)],
        "method": f"group-bootstrap percentile ({n_boot} rounds, seed {seed})",
        "point_estimate": macro_f1_for(y_true, y_pred, categories),
        "n_groups": len(unique_groups),
        "n_records": len(y_true),
    }




def macro_f1_supported(
    y_true: Sequence[set], y_pred: Sequence[set], categories: Sequence[str]
) -> float:
    """Macro F1 over categories with support > 0 (calibration criterion).

    Zero-support categories carry no signal for threshold selection and are
    excluded; the convention is reported wherever the value is used.
    """
    metrics = multilabel_metrics(y_true, y_pred, categories)
    values = [
        metrics["per_category"][category]["f1"]
        for category in categories
        if metrics["per_category"][category]["support"] > 0
    ]
    return round(sum(values) / len(values), 4) if values else 0.0
