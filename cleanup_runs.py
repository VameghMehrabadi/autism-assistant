"""حذف امن خروجی‌های اجرا — CLI نگهداری داده (Issue #10).

Explicit, whitelist-based cleanup of run-generated artifacts that may
contain raw conversation text. Nothing is deleted without ``--confirm``;
a preview always shows the exact files first.

Usage:
    python cleanup_runs.py --list
    python cleanup_runs.py --run <location> --preview
    python cleanup_runs.py --run <location> --confirm

Locations (see --list):
    default                 the standard single-run files in outputs/
    <run-dir>               a run directory under outputs/
    comparison/<run-id>     one per-run directory of a comparison
    translation-artifacts   translation exports, sidecar, error/QA files

Never touched: original datasets, unrelated runs, model weights, shared
embedding/model caches, gold evaluation data (data/gold/), source code,
shared comparison summaries, logs. The default retention period (30 days
for temporary generated raw run artifacts) is a documented policy default —
deletion is always an explicit operator action.
"""
from __future__ import annotations

import argparse
import sys

from retention import (
    RetentionError,
    execute_run_cleanup,
    list_run_locations,
    plan_run_cleanup,
)


def _reconfigure_stdio_utf8() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_stdio_utf8()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Preview or delete run-generated artifacts (raw conversation "
            "retention policy, Issue #10). Deletes ONLY whitelisted run "
            "artifacts of the selected location; original datasets, other "
            "runs, caches, model weights and gold data are never touched."
        )
    )
    p.add_argument("--list", action="store_true", dest="list_locations",
                   help="list addressable cleanup locations and exit")
    p.add_argument("--run", type=str, default=None,
                   help="cleanup location (see --list)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true",
                      help="show exactly what would be deleted (default)")
    mode.add_argument("--confirm", action="store_true",
                      help="explicitly confirm deletion of the listed files")
    return p


def _print_plan(location: str) -> int:
    try:
        plan = plan_run_cleanup(location)
    except RetentionError as exc:
        print(f"[cleanup] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"[cleanup] location: {plan.location}")
    if not plan.candidates:
        print("[cleanup] nothing to delete (already clean or nothing found)")
    for candidate in plan.candidates:
        print(
            f"  WOULD DELETE [{candidate.kind}] {candidate.display} "
            f"({candidate.size_bytes} bytes)"
        )
    for display, reason in plan.skipped:
        print(f"  skip: {display} — {reason}")
    print(
        f"[cleanup] {len(plan.candidates)} file(s), "
        f"{plan.total_bytes} bytes; run again with --confirm to delete."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_locations:
        locations = list_run_locations()
        if not locations:
            print("[cleanup] no cleanup-eligible artifacts found.")
            return 0
        print("[cleanup] addressable locations:")
        for location in locations:
            print(f"  {location}")
        return 0

    if not args.run:
        build_parser().print_help()
        return 2

    if args.confirm:
        try:
            plan = plan_run_cleanup(args.run)
        except RetentionError as exc:
            print(f"[cleanup] REFUSED: {exc}", file=sys.stderr)
            return 2
        if not plan.candidates:
            print("[cleanup] nothing to delete (already clean).")
            return 0
        print("[cleanup] confirmed deletion of:")
        for candidate in plan.candidates:
            print(f"  - {candidate.display} ({candidate.kind})")
        result = execute_run_cleanup(plan, confirm=True)
    else:
        # preview is the default — explicit confirmation is required
        result = None
        rc = _print_plan(args.run)
        if rc:
            return rc
        print(
            "[cleanup] preview only — nothing was deleted. Re-run with "
            "--confirm to delete these files."
        )
        return 0

    print(
        f"[cleanup] done ({result['mode']}): "
        f"selected={result['selected']} deleted={result['deleted']} "
        f"skipped={result['skipped']} failed={result['failed']} "
        f"outcome={result['outcome']}"
    )
    if result["failed_files"]:
        print(
            "[cleanup] partial failure — these files could not be "
            f"deleted: {', '.join(result['failed_files'])}",
            file=sys.stderr,
        )
        return 3
    if result["outcome"] == "failed":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
