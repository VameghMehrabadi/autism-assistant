# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

[![Python tests](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml/badge.svg)](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml)

پروژه‌ی توسعه‌ی یک LLM (با RAG یا fine-tuning) برای **بیماران اوتیستیک با تأخیر زبانی (language delay)**.
این مخزن مرحله‌ی آماده‌سازی داده را پیاده‌سازی می‌کند: بارگذاری **MentalChat16K** (انگلیسی یا ترجمه‌ی فارسی)،
لیبل‌گذاری معنایی بر اساس ۷ دسته‌ی فکت اوتیسم (A تا G)، تحلیل شکاف (gap analysis)، و **مقایسه‌ی چندمدلی** برای اطمینان از پایداری نتایج.

برای راهنمای کامل انگلیسی → [`README_en.md`](README_en.md)

## Pipeline / خط لوله

1. `data_loader.py` — بارگذاری انگلیسی از Hugging Face یا فارسی از JSONL/CSV محلی.
2. `facts.py` — ۷ دسته (A–G) با prototype دوزبانه؛ فیلتر `en` / `fa` / `both`.
3. `labeler.py` — امبدینگ چندزبانه + cosine similarity.
4. `gap_analysis.py` — توزیع دسته‌ها، آمار امتیاز، دسته‌های قوی/ضعیف و نمودار.
5. `main.py` — اجرای تک‌پیکربندی.
6. `compare_runs.py` — مقایسه‌ی ۵ پیکربندی.
7. `export_for_translation.py` + `prompts/semantic_persian_translation.txt` — خروجی برای ترجمه‌ی GPT-4+.

## مدل‌های امبدینگ

| کلید | مدل | توضیح |
|------|------|--------|
| `minilm` | `paraphrase-multilingual-MiniLM-L12-v2` | مدل task اول |
| `bge-m3` | `BAAI/bge-m3` | مدل قوی‌تر چندزبانه |
| `e5-large` | `intfloat/multilingual-e5-large` | با prefixهای `query:` / `passage:` |

## ماتریس مقایسه (۵ اجرا)

| # | فکت | دیتاست | مدل |
|---|-----|--------|-----|
| 1 | انگلیسی | فارسی | MiniLM *(اجرای مجدد)* |
| 2 | انگلیسی | فارسی | bge-m3 |
| 3 | انگلیسی | فارسی | multilingual-e5-large |
| 4 | فارسی | فارسی | bge-m3 |
| 5 | فارسی | فارسی | multilingual-e5-large |

## Install / نصب

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## ترجمه‌ی دیتاست به فارسی

```bash
python export_for_translation.py --limit 500
# سپس با پرامپت prompts/semantic_persian_translation.txt (GPT-4+) ترجمه کنید
# و خروجی را در data/mentalchat16k_fa.jsonl ذخیره کنید
```

## Run / اجرا

```bash
# تک‌اجرا (مثال)
python main.py --dataset fa --fact-lang en --model bge-m3 --out outputs/en_fa_bge-m3

# هر ۵ مقایسه (نیاز به data/mentalchat16k_fa.jsonl)
python compare_runs.py --limit 500

# smoke test با fixture کوچک
python compare_runs.py --limit 5 --dataset-path tests/fixtures/mentalchat16k_fa.sample.jsonl --only 1
```

خروجی مقایسه در `outputs/comparison/` نوشته می‌شود.

## Tests / تست‌ها

```bash
python -m pytest
```

## Categories / دسته‌ها (A–G)

| Key | Title                         | #facts |
| --- | ----------------------------- | ------ |
| A   | Social Communication          | 6      |
| B   | Sensory Processing            | 3      |
| C   | Emotional Regulation & Stress | 4      |
| D   | Routine & Predictability      | 3      |
| E   | Special Interests & Strengths | 7      |
| F   | Diagnosis & Support           | 6      |
| G   | Autism Knowledge & Awareness  | 14     |

(فکت‌های کامل در `facts.py`؛ نسخه‌ی تأییدشده توسط متخصص را می‌توان جایگزین کرد.)

## Notes / یادداشت‌ها

* تطبیق معنایی است، نه کلمه‌ای.
* امتیاز هر دسته = حداکثر شباهت روی prototypeهای همان دسته (با زبان انتخاب‌شده).
* برای e5، نمونه‌ها `query:` و فکت‌ها `passage:` prefix می‌گیرند.
* gap analysis مشخص می‌کند کدام دسته‌ها پوشش ضعیف دارند و برای RAG/fine-tuning به داده‌ی مکمل نیاز دارند.
