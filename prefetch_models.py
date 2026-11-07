"""دانلود یک‌باره‌ی وزن مدل‌های امبدینگ (Issue #14).

Separate from pipeline execution: no dataset, no API key, no labeling, no
translation — only the model weights are fetched into the configured cache
(``config.MODEL_CACHE_DIR``) so later runs work offline.

اجرا:
    python prefetch_models.py --list            # registry + cache locations
    python prefetch_models.py --model minilm    # one registered model
    python prefetch_models.py --model all       # every registered model

The cache location follows the same resolution as inference:
AUTISM_MODEL_CACHE_DIR > SENTENCE_TRANSFORMERS_HOME > config.CACHE_DIR.
Approximate sizes are from the Hugging Face model cards (fp32 weights) —
treat them as estimates, not exact numbers.
"""
from __future__ import annotations

import argparse
import sys

import config
from model_loader import load_sentence_transformer, offline_state

# Approximate weight download sizes (Hugging Face model cards, fp32).
# "approx" — do not treat as exact; local disk usage varies with format.
APPROX_SIZES_MB: dict[str, str] = {
    "minilm": "~0.5 GB",
    "bge-m3": "~2.3 GB",
    "e5-large": "~2.3 GB",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prefetch embedding-model weights into the configured cache "
            "(offline usage afterwards). No dataset or API key needed."
        )
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[*sorted(config.EMBED_MODELS), "all"],
        help="model key from the registry, or 'all' "
        f"(default: {config.EMBED_MODEL_KEY})",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="print the model registry, cache locations and offline state",
    )
    return p


def print_registry() -> None:
    print(f"Model weights cache : {config.MODEL_CACHE_DIR}  [{config.MODEL_CACHE_SOURCE}]")
    print(f"Embedding cache     : {config.EMBED_CACHE_DIR}  (computed vectors — separate artifact)")
    print(f"Pipeline outputs    : {config.OUTPUT_DIR}")
    for key in sorted(config.EMBED_MODELS):
        meta = config.EMBED_MODELS[key]
        approx = APPROX_SIZES_MB.get(key, "?")
        print(
            f"  {key:9s} {meta['name']}  "
            f"prefix={meta['prefix']}  approx={approx}"
        )
    print(f"Default model: {config.EMBED_MODEL_KEY}")
    print(f"Offline switches: {offline_state()}")


def prefetch_one(model_key: str) -> bool:
    """Prefetch a single registered model; True on success."""
    meta = config.EMBED_MODELS[model_key]
    print(f"[prefetch] {model_key}: {meta['name']}")
    print(f"[prefetch] cache -> {config.MODEL_CACHE_DIR}  [{config.MODEL_CACHE_SOURCE}]")
    try:
        model = load_sentence_transformer(
            meta["name"],
            model_key=model_key,
            device="cpu",
            cache_folder=str(config.MODEL_CACHE_DIR),
        )
    except Exception as exc:
        print(f"[prefetch] FAILED {model_key}: {exc}", file=sys.stderr)
        return False
    try:
        n_params = sum(p.numel() for p in model.parameters())
        detail = f" ({n_params / 1e6:.0f}M params)"
    except Exception:  # never let introspection fail the prefetch
        detail = ""
    print(f"[prefetch] OK {model_key}{detail} -> {config.MODEL_CACHE_DIR}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        print_registry()
        return 0

    if args.model == "all":
        keys = sorted(config.EMBED_MODELS)
    else:
        keys = [args.model or config.EMBED_MODEL_KEY]

    failed = [key for key in keys if not prefetch_one(key)]
    if failed:
        print(
            f"[prefetch] {len(failed)}/{len(keys)} model(s) failed: "
            f"{', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[prefetch] Done: {len(keys)} model(s) available offline at "
        f"{config.MODEL_CACHE_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
