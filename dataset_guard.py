"""دیتاست گارد (Issue #6) — رد قطعی دیتاست‌های موک برای اجراهای ۴ و ۵.

Fail-closed guard that prevents mock/synthetic datasets from entering the
Persian comparison runs (4 and 5). Detection uses **provenance, not
filenames alone**:

- known mock filename markers (``*.mock_sample.jsonl`` and friends);
- mock model identifiers in record metadata (``mock-echo``, ``mock``,
  ``mock-budget``);
- mock provider declarations (``provider == "mock"``);
- mock content markers embedded in record text (``[MOCK-FA]``/``[MOCK-EN]``,
  the vocabulary shared with the loader-level guard in PR #26).

The marker vocabulary is intentionally identical to the loader-level guard
implemented for Issue #25 (PR #26, unmerged at the time of writing): this
module adds the *comparison-run pre-flight* layer so runs 4/5 fail **before
embedding generation**, independently of which loader path is used. When
both PRs merge, the two layers compose (loader gate + run gate).

Legitimate temporary synthetic datasets used by unit tests contain none of
these markers and pass unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

MOCK_FILENAME_MARKERS: tuple[str, ...] = (".mock_sample.",)
MOCK_MODEL_IDS: frozenset[str] = frozenset({"mock-echo", "mock", "mock-budget"})
MOCK_PROVIDER_VALUES: frozenset[str] = frozenset({"mock"})
MOCK_CONTENT_MARKERS: tuple[str, ...] = ("[MOCK-FA]", "[MOCK-EN]")


class MockDatasetError(RuntimeError):
    """Refused: the dataset is mock/synthetic, not a real translated corpus."""


def _row_mock_reasons(row: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(row, dict):
        return reasons
    model = row.get("model")
    if isinstance(model, str) and model.strip().lower() in MOCK_MODEL_IDS:
        reasons.append(f"record declares mock model {model!r}")
    provider = row.get("provider")
    if (
        isinstance(provider, str)
        and provider.strip().lower() in MOCK_PROVIDER_VALUES
    ):
        reasons.append("record declares a mock provider")
    for field in ("instruction", "input", "output", "text"):
        value = row.get(field)
        if isinstance(value, str):
            for marker in MOCK_CONTENT_MARKERS:
                if marker in value:
                    reasons.append(
                        f"{field} contains mock content marker {marker}"
                    )
    return reasons


def scan_records_for_mock_markers(
    rows: Sequence[Any],
) -> list[str]:
    """Scan loaded records for mock provenance; returns deduped reasons."""
    seen: list[str] = []
    for row in rows:
        for reason in _row_mock_reasons(row):
            if reason not in seen:
                seen.append(reason)
    return seen


def _read_jsonl_rows(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append(None)  # counted as unparseable, not mock
    return rows


def check_dataset_guard(path: str | Path, rows: Sequence[Any] | None = None) -> list[str]:
    """Evaluate the mock guard for a candidate dataset (read-only).

    Returns the list of violation reasons (empty = passes). Raises nothing.
    """
    path = Path(path)
    reasons: list[str] = []
    name = path.name.lower()
    for marker in MOCK_FILENAME_MARKERS:
        if marker in name:
            reasons.append(
                f"filename contains the mock marker {marker!r}"
            )
    if rows is None and path.exists() and path.suffix.lower() == ".jsonl":
        rows = _read_jsonl_rows(path)
    if rows is not None:
        reasons.extend(scan_records_for_mock_markers(rows))
    return reasons


def assert_real_dataset(path: str | Path, rows: Sequence[Any] | None = None) -> None:
    """Fail closed when the candidate dataset is mock/synthetic.

    Raises :class:`MockDatasetError` with an actionable message listing the
    violation reasons. Read-only.
    """
    path = Path(path)
    reasons = check_dataset_guard(path, rows=rows)
    if reasons:
        raise MockDatasetError(
            f"MOCK DATA refused: {path} is not a real translated corpus. "
            "Reasons: " + "; ".join(reasons) + ". Runs 4 and 5 require the "
            "genuine Persian dataset (data/mentalchat16k_fa.jsonl) produced "
            "by translate_dataset.py with a real provider."
        )
