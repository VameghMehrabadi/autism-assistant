"""Translate exported MentalChat16K records to Persian with GPT-4+.

Requires OPENAI_API_KEY. Uses prompts/semantic_persian_translation.txt.

    python export_for_translation.py --limit 500
    python translate_dataset.py --limit 500
    python translate_dataset.py --model gpt-4o --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-4+ Persian translation of MentalChat16K.")
    p.add_argument(
        "--src",
        type=str,
        default=str(config.DATA_DIR / "mentalchat16k_en_for_translation.jsonl"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(config.PERSIAN_DATASET_PATH),
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=str(config.PROMPTS_DIR / "semantic_persian_translation.txt"),
    )
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "[translate] ERROR: OPENAI_API_KEY is not set.\n"
            "Set the key, then rerun. Output must be GPT-4+ for runs 4–5.",
            file=sys.stderr,
        )
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("[translate] ERROR: pip install openai", file=sys.stderr)
        return 2

    src = Path(args.src)
    out = Path(args.out)
    prompt_path = Path(args.prompt)
    if not src.exists():
        print(f"[translate] ERROR: missing source {src}", file=sys.stderr)
        return 2
    if not prompt_path.exists():
        print(f"[translate] ERROR: missing prompt {prompt_path}", file=sys.stderr)
        return 2

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    rows: list[dict] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    done: dict[int, dict] = {}
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "idx" in rec:
                    done[int(rec["idx"])] = rec
        print(f"[translate] Resuming with {len(done)} existing rows in {out}")

    client = OpenAI(api_key=api_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = [r for r in rows if int(r.get("idx", -1)) not in done]
    print(f"[translate] {len(pending)} / {len(rows)} remaining  model={args.model}")

    with out.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(pending, start=1):
            text = rec.get("text") or ""
            if not text.strip():
                continue
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            fa = (resp.choices[0].message.content or "").strip()
            out_rec = {"idx": rec.get("idx"), "text": fa}
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[translate] {i}/{len(pending)} idx={rec.get('idx')}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"[translate] Wrote Persian records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
