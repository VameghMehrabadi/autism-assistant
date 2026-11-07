"""Longitudinal comparison of two independent pipeline runs (Issue #16).

Compares two run directories using their ``run_meta.json`` and
``labeled.csv`` outputs. Reports category distribution, uncertainty,
and margin metrics with their differences (Run B minus Run A).

Descriptive only — does not establish causality or clinical significance.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from facts import CATEGORY_KEYS

# Phase-3 columns that may be absent in legacy labeled.csv files.
PHASE3_RUN_COLUMNS: tuple[str, ...] = ("decision", "margin")

_RUN_META_FILE = "run_meta.json"
_LABELED_FILE = "labeled.csv"


@dataclass
class RunSnapshot:
    """Normalized representation of one pipeline run."""

    run_id: str
    run_dir: str
    meta: dict[str, Any]
    labeled: pd.DataFrame
    timestamp: str | None  # ISO string or None if unavailable
    warnings: list[str] = field(default_factory=list)

    @property
    def category_counts(self) -> dict[str, int]:
        return self.labeled["top_label"].value_counts().to_dict()

    @property
    def n_records(self) -> int:
        return len(self.labeled)

    def category_share(self, cat: str) -> float:
        if self.n_records == 0:
            return 0.0
        return self.category_counts.get(cat, 0) / self.n_records

    @property
    def uncertain_count(self) -> int | None:
        if "decision" not in self.labeled.columns:
            return None
        return int((self.labeled["decision"] == "uncertain").sum())

    @property
    def uncertain_rate(self) -> float | None:
        if self.uncertain_count is None or self.n_records == 0:
            return None
        return self.uncertain_count / self.n_records

    @property
    def mean_margin(self) -> float | None:
        if "margin" not in self.labeled.columns:
            return None
        margins = pd.to_numeric(self.labeled["margin"], errors="coerce").dropna()
        if margins.empty:
            return None
        return round(float(margins.mean()), 4)


@dataclass
class CategoryComparison:
    cat: str
    count_a: int
    count_b: int
    share_a: float
    share_b: float
    delta_pp: float  # share_b - share_a (percentage points)


@dataclass
class UncertaintyComparison:
    count_a: int | None
    count_b: int | None
    rate_a: float | None
    rate_b: float | None
    delta_rate: float | None  # rate_b - rate_a


@dataclass
class MarginComparison:
    mean_a: float | None
    mean_b: float | None
    delta: float | None  # mean_b - mean_a


@dataclass
class RunComparisonResult:
    run_a: RunSnapshot
    run_b: RunSnapshot
    categories: list[CategoryComparison]
    uncertainty: UncertaintyComparison | None
    margin: MarginComparison | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": {
                "run_id": self.run_a.run_id,
                "timestamp": self.run_a.timestamp,
                "n_records": self.run_a.n_records,
            },
            "run_b": {
                "run_id": self.run_b.run_id,
                "timestamp": self.run_b.timestamp,
                "n_records": self.run_b.n_records,
            },
            "categories": [
                {
                    "cat": c.cat,
                    "count_a": c.count_a,
                    "count_b": c.count_b,
                    "share_a": round(c.share_a, 4),
                    "share_b": round(c.share_b, 4),
                    "delta_pp": round(c.delta_pp, 4),
                }
                for c in self.categories
            ],
            "uncertainty": {
                "count_a": self.uncertainty.count_a,
                "count_b": self.uncertainty.count_b,
                "rate_a": round(self.uncertainty.rate_a, 4) if self.uncertainty.rate_a is not None else None,
                "rate_b": round(self.uncertainty.rate_b, 4) if self.uncertainty.rate_b is not None else None,
                "delta_rate": round(self.uncertainty.delta_rate, 4) if self.uncertainty.delta_rate is not None else None,
            } if self.uncertainty else None,
            "margin": {
                "mean_a": self.margin.mean_a,
                "mean_b": self.margin.mean_b,
                "delta": round(self.margin.delta, 4) if self.margin.delta is not None else None,
            } if self.margin else None,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_labeled_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


def _extract_timestamp(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    for key in ("timestamp", "completed_utc", "labeled_utc", "created_utc"):
        val = meta.get(key)
        if val and isinstance(val, str) and len(val) >= 10:
            return val
    return None


def load_run_snapshot(run_dir: str | Path) -> RunSnapshot:
    """Load one pipeline run from a directory.

    Returns a :class:`RunSnapshot` with metadata, labeled data, timestamp,
    and any warnings about missing or incompatible files.
    """
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / _RUN_META_FILE)
    labeled = _read_labeled_csv(run_dir / _LABELED_FILE)
    warnings: list[str] = []

    run_id = (
        (meta or {}).get("id")
        or (meta or {}).get("run_id")
        or run_dir.name
    )
    timestamp = _extract_timestamp(meta)
    if timestamp is None:
        warnings.append(
            f"Run '{run_id}': temporal metadata is missing — longitudinal "
            "comparison cannot establish chronological ordering."
        )
    if labeled is None or labeled.empty:
        warnings.append(f"Run '{run_id}': labeled.csv is missing, empty, or unreadable.")

    return RunSnapshot(
        run_id=run_id,
        run_dir=str(run_dir),
        meta=meta or {},
        labeled=labeled if labeled is not None else pd.DataFrame(),
        timestamp=timestamp,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _pct_diff(a: float, b: float) -> float:
    return b - a


def _safe_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def compare_runs(
    run_a_dir: str | Path,
    run_b_dir: str | Path,
) -> RunComparisonResult:
    """Compare two independent pipeline runs.

    Direction: Run B minus Run A for all differences.

    Returns a :class:`RunComparisonResult` with category distribution,
    uncertainty, and margin comparisons. Handles missing files, empty CSVs,
    missing columns, and incompatible schemas without raising.
    """
    snap_a = load_run_snapshot(run_a_dir)
    snap_b = load_run_snapshot(run_b_dir)
    all_warnings = snap_a.warnings + snap_b.warnings

    categories: list[CategoryComparison] = []
    if snap_a.labeled.empty or snap_b.labeled.empty:
        all_warnings.append(
            "One or both runs have empty labeled data — category comparison unavailable."
        )
    else:
        for cat in CATEGORY_KEYS:
            ca = snap_a.category_counts.get(cat, 0)
            cb = snap_b.category_counts.get(cat, 0)
            sa = snap_a.category_share(cat)
            sb = snap_b.category_share(cat)
            categories.append(
                CategoryComparison(
                    cat=cat, count_a=ca, count_b=cb,
                    share_a=sa, share_b=sb, delta_pp=_pct_diff(sa, sb),
                )
            )

    # Uncertainty comparison
    ua = snap_a.uncertain_count
    ub = snap_b.uncertain_count
    ra = snap_a.uncertain_rate
    rb = snap_b.uncertain_rate
    uncertainty = UncertaintyComparison(
        count_a=ua, count_b=ub, rate_a=ra, rate_b=rb,
        delta_rate=_safe_delta(ra, rb),
    ) if ua is not None or ub is not None else None

    # Margin comparison
    ma = snap_a.mean_margin
    mb = snap_b.mean_margin
    margin = MarginComparison(
        mean_a=ma, mean_b=mb, delta=_safe_delta(ma, mb),
    ) if ma is not None or mb is not None else None

    return RunComparisonResult(
        run_a=snap_a, run_b=snap_b,
        categories=categories,
        uncertainty=uncertainty,
        margin=margin,
        warnings=all_warnings,
    )


# --------------------------------------------------------------------------
# Comparison table for UI display
# --------------------------------------------------------------------------


def build_comparison_table(result: RunComparisonResult) -> pd.DataFrame:
    """Build a DataFrame suitable for ``st.dataframe`` display."""
    rows = []
    for c in result.categories:
        rows.append({
            "Category": c.cat,
            f"Count ({result.run_a.run_id})": c.count_a,
            f"Count ({result.run_b.run_id})": c.count_b,
            f"Share % (A)": f"{c.share_a:.1%}",
            f"Share % (B)": f"{c.share_b:.1%}",
            f"Δ pp (B−A)": f"{c.delta_pp:+.1%}",
        })
    return pd.DataFrame(rows)
