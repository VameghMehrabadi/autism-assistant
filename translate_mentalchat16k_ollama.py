#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate MentalChat16K JSONL to Persian with a local Ollama model.
Run this on the same computer where Ollama is running.
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests

SYSTEM_PROMPT = """You are an expert English-to-Persian (Farsi) translator.
Translate the supplied text faithfully into natural, clear Persian.
Rules:
- Preserve the complete meaning; do not summarize, omit, add, or interpret.
- Preserve paragraph breaks, lists, quotations, punctuation, and formatting where possible.
- Preserve names, numbers, dates, URLs, and placeholders.
- Use accurate, consistent mental-health and counseling terminology.
- Use respectful, non-stigmatizing language.
- Do not add a diagnosis, advice, warning, or explanation that is not in the source.
- Output ONLY the Persian translation. No preface or commentary.
"""

TEXT_FIELDS = ("text", "instruction", "input", "output")


def translate(session, base_url, model, source, timeout, retries):
    if not source:
        return source
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Translate this text into Persian:\n\n" + source},
        ],
        "options": {"temperature": 0.1},
    }
    last_error = None
    for attempt in range(retries):
        try:
            r = session.post(
                base_url.rstrip("/") + "/api/chat",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            result = data.get("message", {}).get("content", "").strip()
            if not result:
                raise ValueError("Ollama returned an empty translation")
            return result
        except Exception as exc:
            last_error = exc
            wait = min(60, 2 ** attempt)
            print(f"  Request failed ({attempt + 1}/{retries}): {exc}; retry in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Translation failed after {retries} attempts: {last_error}")


def main():
    ap = argparse.ArgumentParser(description="Translate MentalChat16K JSONL to Persian via Ollama")
    ap.add_argument("--input", default="mentalchat16k_en_for_translation.jsonl")
    ap.add_argument("--output", default="mentalchat16k_fa.jsonl")
    ap.add_argument("--model", default="translategemma:4b",
                    help="Use the exact model tag shown by `ollama list`")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="Translate only the first N records (0 = all); useful for a pilot")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    if not src.is_file():
        raise SystemExit(f"Input file not found: {src.resolve()}")

    # SQLite cache avoids translating repeated strings again and supports resume.
    db_path = Path(str(out) + ".cache.sqlite3")
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS translations (source TEXT PRIMARY KEY, target TEXT NOT NULL)")
    db.commit()

    session = requests.Session()
    done = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    # Ignore a malformed final line; the output is still append-only.
                    continue

    total = sum(1 for _ in src.open("r", encoding="utf-8"))
    target_total = min(total, args.limit) if args.limit else total
    print(f"Records: {total}; this run targets: {target_total}")
    print(f"Model: {args.model} | Ollama: {args.url}")
    print("Fields translated: text, instruction, input, output")

    written = 0
    with src.open("r", encoding="utf-8") as fin, out.open("a", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if args.limit and line_no > args.limit:
                break
            row = json.loads(line)
            idx = row.get("idx", line_no - 1)
            if idx in done:
                continue

            translated = dict(row)
            for field in TEXT_FIELDS:
                value = row.get(field)
                if not isinstance(value, str):
                    continue
                cached = db.execute(
                    "SELECT target FROM translations WHERE source = ?", (value,)
                ).fetchone()
                if cached:
                    translated[field] = cached[0]
                else:
                    print(f"  idx={idx}: translating {field} ({len(value)} chars)")
                    target = translate(session, args.url, args.model, value,
                                       args.timeout, args.retries)
                    db.execute(
                        "INSERT OR REPLACE INTO translations(source, target) VALUES (?, ?)",
                        (value, target),
                    )
                    db.commit()
                    translated[field] = target

            fout.write(json.dumps(translated, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 10 == 0:
                print(f"Progress: {len(done) + written}/{target_total} records written")

    db.close()
    print(f"Finished this run. New records: {written}")
    print(f"Output: {out.resolve()}")
    print("Check the output before using it for research or model training.")


if __name__ == "__main__":
    main()
