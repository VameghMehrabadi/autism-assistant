"""بارگذاری وزن مدل با خطای قابل‌فهم (Issue #14).

همه‌ی بارگیری‌های SentenceTransformer از این ماژول می‌گذرند تا اگر وزن مدل در
کش نبود یا در حالت آفلاین در دسترس نبود، پیام خطا شامل نام مدل، شناسه‌ی
Hugging Face، مسیر کش، وضعیت آفلاین و دستور prefetch باشد — نه فقط یک
stacktrace شبکه‌ی Hugging Face.

قواعد:
- هرگز به encoder دیگری fallback نمی‌شود.
- در حالت آفلاین هرگز تلاش آنلاین مخفی انجام نمی‌شود (HF_HUB_OFFLINE /
  TRANSFORMERS_OFFLINE توسط خود huggingface_hub اعمال می‌شود؛ این ماژول فقط
  وضعیت را گزارش و خطاها را طبقه‌بندی می‌کند).
- خطاهای نامرتبط (شبکه‌ی عمومی، خطای داخلی، ...) دست‌نخورده عبور می‌کنند.
"""
from __future__ import annotations

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

import config

# Offline switches honored by huggingface_hub / transformers. They are read
# by those libraries when they are first imported, so they must be set
# BEFORE launching Python — this module reports them, it cannot set them.
OFFLINE_ENV_VARS: tuple[str, ...] = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

try:
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    _MISSING_WEIGHTS_ERRORS: tuple[type[Exception], ...] = (
        OfflineModeIsEnabled,
        LocalEntryNotFoundError,
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    _AUTH_ERRORS: tuple[type[Exception], ...] = (GatedRepoError,)
except ImportError:  # pragma: no cover — unexpected hub version
    _MISSING_WEIGHTS_ERRORS = ()
    _AUTH_ERRORS = ()


def offline_state() -> dict[str, bool]:
    """Which offline switches are currently set (truthy ≠ "" and ≠ "0")."""
    return {
        var: os.environ.get(var, "").strip() not in ("", "0")
        for var in OFFLINE_ENV_VARS
    }


def format_offline_state(offline: dict[str, bool] | None = None) -> str:
    offline = offline_state() if offline is None else offline
    if not any(offline.values()):
        return "disabled (network access allowed)"
    enabled = " ".join(f"{var}=1" for var, on in offline.items() if on)
    return f"ENABLED ({enabled})"


def prefetch_command(model_key: str | None = None) -> str:
    """Exact command that downloads the given model's weights."""
    key = model_key if model_key in config.EMBED_MODELS else config.EMBED_MODEL_KEY
    return f"python prefetch_models.py --model {key}"


class ModelLoadError(RuntimeError):
    """Actionable base error for model-weight loading problems."""

    def __init__(self, message: str, model_key: str, model_name: str):
        super().__init__(message)
        self.model_key = model_key
        self.model_name = model_name


class ModelWeightsMissingError(ModelLoadError):
    """The requested model's weights are not usable from the local cache."""

    def __init__(
        self,
        model_key: str,
        model_name: str,
        cache_dir: Path | str,
        cause: Exception,
        offline: dict[str, bool] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.offline = dict(offline) if offline else offline_state()
        lines = [
            f"Model weights not available for {model_key!r} ({model_name}).",
            f"  Expected cache location: {self.cache_dir}",
            f"  Cache setting: {config.MODEL_CACHE_SOURCE}",
            f"  Offline mode: {format_offline_state(self.offline)}",
            "  Fetch the weights once (online), then run offline:",
            f"      {prefetch_command(model_key)}",
            f"  Original error: {type(cause).__name__}: {cause}",
        ]
        super().__init__("\n".join(lines), model_key, model_name)


def _looks_like_missing_weights(exc: Exception) -> bool:
    """Classify 'weights are not usable here' without over-reach.

    hub's missing/offline errors are exact matches; a FileNotFoundError or an
    OSError mentioning the model files counts too. Unrelated OSErrors (disk
    full, permissions, generic network failures) must NOT be swallowed.
    """
    if isinstance(exc, _AUTH_ERRORS):
        return False  # gated/private repos are reported as auth problems
    if isinstance(exc, _MISSING_WEIGHTS_ERRORS):
        return True
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, OSError):
        text = str(exc).lower()
        # Phrases observed from the real huggingface_hub / transformers /
        # sentence-transformers stacks when weights cannot be resolved:
        # empty cache, offline mode, hub unreachable, corrupt snapshot.
        markers = (
            "no such file",
            "does not appear to have",
            "pytorch_model",
            "unrecognized configuration",
            "couldn't find it in the cached files",
            "couldn't connect to 'https://huggingface.co'",
            "offline mode is enabled",
            "max retries exceeded",
        )
        return any(marker in text for marker in markers)
    return False


def load_sentence_transformer(
    model_name: str,
    model_key: str = "custom",
    device: str = "cpu",
    cache_folder: Path | str | None = None,
) -> SentenceTransformer:
    """Load model weights through the app's single cache configuration.

    ``cache_folder`` defaults to ``config.MODEL_CACHE_DIR`` (weights cache —
    distinct from ``config.EMBED_CACHE_DIR`` which stores computed vectors).

    Missing / offline-unavailable weights raise ``ModelWeightsMissingError``
    with the full actionable context (original exception preserved as
    ``__cause__``). Everything else propagates unchanged: no other-encoder
    fallback, no hidden online retry when offline mode is set.
    """
    cache_dir = config.MODEL_CACHE_DIR if cache_folder is None else Path(cache_folder)
    try:
        return SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_dir),
        )
    except Exception as exc:
        if isinstance(exc, _AUTH_ERRORS):
            raise ModelLoadError(
                f"Access to {model_name!r} is gated or unauthorized. Accept the "
                "model's conditions on huggingface.co and provide a token via "
                "the HF_TOKEN environment variable (the token value is never "
                f"logged). Original error: {type(exc).__name__}: {exc}",
                model_key,
                model_name,
            ) from exc
        if _looks_like_missing_weights(exc):
            raise ModelWeightsMissingError(
                model_key, model_name, cache_dir, exc
            ) from exc
        raise
