"""تنظیمات مرکزی پروژه / Central configuration for the pipeline.

این فایل همه‌ی پارامترهای قابل تنظیم را در یک جا نگه می‌دارد تا با اجرای
`python main.py` رفتار pipeline را بدون دستکاری بقیه‌ی ماژول‌ها کنترل کنید.
"""
from __future__ import annotations

from pathlib import Path

# --- مسیرها / Paths ---
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"
OUTPUT_DIR: Path = ROOT / "outputs"
CACHE_DIR: Path = ROOT / ".cache"
PROMPTS_DIR: Path = ROOT / "prompts"

for _p in (DATA_DIR, OUTPUT_DIR, CACHE_DIR, PROMPTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- دیتاست / Dataset ---
DATASET_NAME: str = "ShenLab/MentalChat16K"
DATASET_SPLIT: str = "train"
# برای تست سریع روی CPU می‌توانید مقدار زیر را کوچک نگه دارید (مثلاً 200).
# None یعنی کل دیتاست. / None means the whole dataset.
SAMPLE_LIMIT: int | None = 500
# فیلدهای متنی دیتاست که با هم به‌عنوان «utterance» برای لیبل‌گذاری استفاده می‌شوند.
# Fields merged to form one utterance for semantic labeling.
DATASET_TEXT_FIELDS: tuple[str, ...] = ("instruction", "input", "output")
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
# 1) EN facts × FA dataset × MiniLM  (rerun of task 1 on Persian data)
# 2) EN facts × FA dataset × bge-m3
# 3) EN facts × FA dataset × e5-large
# 4) FA facts × FA dataset × bge-m3
# 5) FA facts × FA dataset × e5-large
COMPARISON_RUNS: list[dict[str, str]] = [
    {
        "id": "1_en_fa_minilm",
        "fact_lang": "en",
        "dataset": "fa",
        "model_key": "minilm",
        "title": "English facts × Persian dataset × MiniLM",
    },
    {
        "id": "2_en_fa_bge-m3",
        "fact_lang": "en",
        "dataset": "fa",
        "model_key": "bge-m3",
        "title": "English facts × Persian dataset × bge-m3",
    },
    {
        "id": "3_en_fa_e5-large",
        "fact_lang": "en",
        "dataset": "fa",
        "model_key": "e5-large",
        "title": "English facts × Persian dataset × multilingual-e5-large",
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
