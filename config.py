"""تنظیمات مرکزی پروژه / Central configuration for the pipeline.

این فایل همه‌ی پارامترهای قابل تنظیم را در یک جا نگه می‌دارد تا با اجرای
`python main.py` رفتار pipeline را بدون دستکاری بقیه‌ی ماژول‌ها کنترل کنید.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- مسیرها / Paths ---
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"
OUTPUT_DIR: Path = ROOT / "outputs"
PROMPTS_DIR: Path = ROOT / "prompts"
# JSON-line logs (privacy-preserving; no raw conversational content).
LOG_DIR: Path = OUTPUT_DIR / "logs"


def _resolve_cache_dir() -> Path:
    """Prefer AUTISM_CACHE_DIR, then D: when C: is too small for large models."""
    override = os.environ.get("AUTISM_CACHE_DIR")
    if override:
        return Path(override)
    d_drive = Path("D:/autism_assistant_cache")
    if Path("D:/").exists():
        return d_drive
    return ROOT / ".cache"


CACHE_DIR: Path = _resolve_cache_dir()

# --- کش وزن مدل‌ها / Model-weights cache (Issue #14) ---
# مسیر کش وزن‌های دانلودشده (models--* از Hugging Face) — جداست از
# EMBED_CACHE_DIR که فقط امبدینگ‌های محاسبه‌شده را نگه می‌دارد؛ این دو نباید
# با هم اشتباه گرفته شوند.
# Resolution order: AUTISM_MODEL_CACHE_DIR (app-specific override)
# > SENTENCE_TRANSFORMERS_HOME (the standard Sentence-Transformers variable)
# > CACHE_DIR (existing layout — keeps already-downloaded weights valid).
# Environment variables must be set BEFORE launching Python: huggingface_hub
# reads HF_HUB_OFFLINE / cache locations when its modules are first imported.
def _resolve_model_cache_dir() -> tuple[Path, str]:
    """Return (model_weights_cache_dir, which_setting_won)."""
    override = os.environ.get("AUTISM_MODEL_CACHE_DIR")
    if override:
        return Path(override), "AUTISM_MODEL_CACHE_DIR"
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if st_home:
        return Path(st_home), "SENTENCE_TRANSFORMERS_HOME"
    return CACHE_DIR, "config.CACHE_DIR (default)"


MODEL_CACHE_DIR: Path
MODEL_CACHE_SOURCE: str
MODEL_CACHE_DIR, MODEL_CACHE_SOURCE = _resolve_model_cache_dir()

for _p in (DATA_DIR, OUTPUT_DIR, CACHE_DIR, PROMPTS_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- دیتاست / Dataset ---
DATASET_NAME: str = "ShenLab/MentalChat16K"
DATASET_SPLIT: str = "train"
# برای تست سریع روی CPU می‌توانید مقدار زیر را کوچک نگه دارید (مثلاً 200).
# None یعنی کل دیتاست. / None means the whole dataset.
SAMPLE_LIMIT: int | None = 500
# فیلدهای متنی دیتاست. در حالت patient فقط شرح مراجع استفاده می‌شود.
# Dataset text fields. In "patient" mode only the client's description is labeled.
DATASET_TEXT_FIELDS: tuple[str, ...] = ("instruction", "input", "output")
# "patient" = فقط input (یا instruction غیرقالبی)؛ "all" = instruction+input+output
# "patient" avoids the repeated counselor prompt + therapy reply that collapse labels to F.
LABEL_TEXT_MODE: str = "patient"
GENERIC_COUNSELOR_MARKERS: tuple[str, ...] = (
    "mental health counselling assistant",
    "mental health counseling assistant",
    "helpful mental health",
    "دستیار مشاوره",
    "مشاوره سلامت روان",
)
# نسخه‌ی فارسی ترجمه‌شده‌ی MentalChat16K (JSONL یا CSV).
# Persian translation of MentalChat16K (JSONL or CSV).
PERSIAN_DATASET_PATH: Path = DATA_DIR / "mentalchat16k_fa.jsonl"

# --- مدل‌های امبدینگ / Embedding models ---
# کلید کوتاه → شناسه‌ی Hugging Face + شیوه‌ی prefix (برای e5 ضروری است).
EMBED_MODELS: dict[str, dict[str, str]] = {
    "minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": "none",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "prefix": "none",
    },
    "e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "prefix": "e5",
    },
}
# پیش‌فرض سازگار با task اول / default matches the first task
EMBED_MODEL_KEY: str = "minilm"
EMBED_MODEL_NAME: str = EMBED_MODELS[EMBED_MODEL_KEY]["name"]
EMBED_BATCH_SIZE: int = 64
# نرمال‌سازی امبدینگ‌ها برای محاسبه‌ی cosine similarity.
NORMALIZE_EMBEDDINGS: bool = True
# زبان prototypeهای فکت: "en" | "fa" | "both"
FACT_LANG: str = "both"

# --- استراتژی لیبل‌گذاری / Labeling strategy ---
# هر دسته چند «prototype» (فکت) دارد؛ امتیاز دسته = max cosine sim روی prototypeها.
# Each category has several prototype facts; category score = max cosine sim over prototypes.
TOP_K_PROTOTYPES_PER_CAT: int = 1  # 1 یعنی max; مقادیر بالاتر میانگین top-k می‌گیرد.
# آستانه‌ی اعتماد برای پذیرش لیبل (cosine similarity). / Confidence threshold.
LABEL_THRESHOLD: float = 0.35
# اگر بیش از یک دسته بالای آستانه باشند، خروجی multi-label می‌شود.
MULTI_LABEL: bool = True

# --- مقایسه‌ی ۵ پیکربندی / Five-way comparison matrix ---
# همه‌ی اجراها از فکت‌های فارسی استفاده می‌کنند (FA facts).
# 1) FA facts × EN dataset × MiniLM   (rerun of task 1; facts may have changed)
# 2) FA facts × EN dataset × bge-m3
# 3) FA facts × EN dataset × e5-large
# 4) FA facts × FA dataset × bge-m3   (after GPT-4+ translation)
# 5) FA facts × FA dataset × e5-large
COMPARISON_RUNS: list[dict[str, str]] = [
    {
        "id": "1_fa_en_minilm",
        "fact_lang": "fa",
        "dataset": "en",
        "model_key": "minilm",
        "title": "Persian facts × English dataset × MiniLM",
    },
    {
        "id": "2_fa_en_bge-m3",
        "fact_lang": "fa",
        "dataset": "en",
        "model_key": "bge-m3",
        "title": "Persian facts × English dataset × bge-m3",
    },
    {
        "id": "3_fa_en_e5-large",
        "fact_lang": "fa",
        "dataset": "en",
        "model_key": "e5-large",
        "title": "Persian facts × English dataset × multilingual-e5-large",
    },
    {
        "id": "4_fa_fa_bge-m3",
        "fact_lang": "fa",
        "dataset": "fa",
        "model_key": "bge-m3",
        "title": "Persian facts × Persian dataset × bge-m3",
    },
    {
        "id": "5_fa_fa_e5-large",
        "fact_lang": "fa",
        "dataset": "fa",
        "model_key": "e5-large",
        "title": "Persian facts × Persian dataset × multilingual-e5-large",
    },
]
COMPARISON_DIR: Path = OUTPUT_DIR / "comparison"

# --- خروجی‌های تک‌اجرا / Single-run outputs ---
LABELED_PATH = OUTPUT_DIR / "labeled.csv"
SCORES_PATH = OUTPUT_DIR / "scores.npy"
GAP_REPORT_MD = OUTPUT_DIR / "gap_analysis.md"
GAP_REPORT_JSON = OUTPUT_DIR / "gap_analysis.json"
GAP_CHART_PATH = OUTPUT_DIR / "gap_distribution.png"

# --- Phase 2: مهندسی داده / Data engineering ---
# نسخه‌ی پیش‌پردازش؛ در گزارش‌ها و manifest ذخیره می‌شود تا اجراها قابل‌بازتولید باشند.
# Preprocessing version stored in reports/manifests for reproducibility.
PREPROCESS_VERSION: str = "2.0.0"
# True: رکورد نامعتبر خطا می‌دهد؛ False: رکورد نامعتبر شفاف رد می‌شود (شمرده‌شده در گزارش).
# True: invalid records raise; False (default): invalid records are skipped
# transparently and counted in the load report.
VALIDATION_STRICT: bool = False
# حذف تکراری‌ها: "keep_first" (اولین نگه داشته می‌شود) یا "keep_all" (فقط شمارش).
# Duplicate handling: "keep_first" removes later duplicates, "keep_all" only counts them.
DEDUP_STRATEGY: str = "keep_first"
# فیلتر زبان لیبل: "fa" | "en" | None (فقط شمارش، بدون حذف).
# Language filter for label text: "fa" | "en" | None (None = count only).
LANGUAGE_FILTER: str | None = None
# استراتژی نمونه‌گیری: "first" (رفتار قبلی) یا "random" (با seed تکرارپذیر).
# Sampling strategy: "first" (previous behavior) or "random" (seeded, reproducible).
SAMPLING_STRATEGY: str = "first"
SAMPLING_SEED: int = 42
# Manifest جداسازی dev/eval برای جلوگیری از نشت داده بین توسعه و ارزیابی.
# dev/eval split manifest guarding against train/evaluation data leakage.
SPLIT_MANIFEST_PATH: Path = OUTPUT_DIR / "split_manifest.json"
# حداکثر طول رشته‌ای که بدون fingerprint در لاگ JSON می‌نشیند.
# Max inline string length in JSON logs; longer strings are fingerprinted.
LOG_MAX_INLINE_LEN: int = 120
# نوشتن لاگ ساختاریافته در LOG_DIR (بدون محتوای محاوره‌ای).
# Structured JSON-line logging into LOG_DIR (never conversational content).
LOG_JSON: bool = True
LOG_LEVEL: str = "INFO"

# --- Phase 3: موتور لیبل‌گذاری / Labeling engine ---
# نسخه‌ی موتور لیبل‌گذاری؛ در نتایج و گزارش‌ها ذخیره می‌شود.
# Labeling engine version recorded in results and reports.
LABELER_VERSION: str = "3.0.0"

# --- Phase 6b: ارائه‌دهنده‌ی Z.ai (GLM) / Z.ai (GLM) provider ---
# OpenAI-compatible endpoint of the Z.ai platform; override with ZAI_BASE_URL.
# Credentials come from the ZAI_API_KEY environment variable (never hardcoded).
ZAI_API_BASE_URL: str = "https://api.z.ai/api/paas/v4"
ZAI_MODEL: str = "glm-5.3-flash"
# کش دیسکی امبدینگ‌ها (کلید = مدل + prefix + نقش + متن نرمال‌شده).
# Disk cache for embeddings (key = model + prefix + role + normalized text).
EMBED_CACHE_ENABLED: bool = True
EMBED_CACHE_DIR: Path = CACHE_DIR / "embed_cache"
# نوشتن امبدینگ‌های جدید به کش در تکه‌های این اندازه — پیشرفت پس از هر تکه
# ماندگار می‌شود، پس قطع‌شدن حداکثر یک تکه را از دست می‌دهد.
# Embedding cache writes are chunked: progress persists after every chunk,
# so an interruption loses at most one chunk of computed embeddings.
EMBED_CACHE_WRITE_CHUNK: int = 256
# حداکثر توکن ورودی برای هر مدل؛ None = پیش‌فرض خود مدل (رفتار قبلی).
# Per-model max input tokens; None = the model's own default (previous behavior).
# bge-m3 pinned to 512 for the comparison matrix: comparable with e5-large's
# window and ~16× faster on CPU than its native 8192-token window.
EMBED_MAX_SEQ_TOKENS: dict[str, int | None] = {
    "minilm": None,      # model default (128)
    "bge-m3": 512,
    "e5-large": None,    # model default (512)
}
# آستانه‌ی اختصاصی هر دسته (اختیاری). دسته‌ی غایب = LABEL_THRESHOLD.
# Optional per-category threshold overrides; missing category = LABEL_THRESHOLD.
CATEGORY_THRESHOLDS: dict[str, float] = {}
