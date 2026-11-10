# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

[![Python tests](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml/badge.svg)](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml)

پروژه‌ی توسعه‌ی یک LLM (با RAG یا fine-tuning) برای **بیماران اوتیستیک با تأخیر زبانی (language delay)**.
این مخزن مرحله‌ی آماده‌سازی داده را پیاده‌سازی می‌کند: بارگذاری **MentalChat16K** (انگلیسی یا ترجمه‌ی فارسی)،
لیبل‌گذاری معنایی بر اساس ۷ دسته‌ی فکت اوتیسم (A تا G)، تحلیل شکاف (gap analysis)، و **مقایسه‌ی چندمدلی** برای اطمینان از پایداری نتایج.

## وضعیت پروژه (نسخه‌ی نهایی)

| بخش | وضعیت |
|---|---|
| مهندسی داده (اعتبارسنجی، حذف تکراری، انکودینگ، جداسازی dev/eval، لاگ محرمانه) | ✅ پیاده‌سازی + تست (۱۶٬۰۰۵ رکورد معتبر صادر شده) |
| موتور لیبل‌گذاری معنایی (prefix های E5، کش دیسکی، آستانه‌های قابل‌تنظیم، وضعیت نامطمئن، شواهد prototype) | ✅ پیاده‌سازی + تست؛ MiniLM روی کل ۱۶٬۰۰۵ رکورد اجرای واقعی شد |
| مقایسه‌ی مدل‌ها | ✅ اجراهای ۱–۳ (MiniLM / bge-m3 / e5-large × انگلیسی × ۱۶٬۰۰۵) **انجام شد** — گزارش یکپارچه در `outputs/comparison/`؛ اجراهای ۴–۵ منتظر دیتاست فارسی |
| تحلیل شکاف (توزیع، عدم‌قطعیت، margin، شواهد prototype) | ✅ پیاده‌سازی + تست |
| خط لوله‌ی ترجمه‌ی فارسی | ✅ پیاده‌سازی + تست (اجرای موک)؛ **اجرای واقعی منتظر `ZAI_API_KEY`** |
| یکپارچه‌سازی Z.ai (GLM) | ✅ پیاده‌سازی + تأیید؛ منتظر کلید |
| رابط کاربری Streamlit | ✅ پیاده‌سازی + تأیید در مرورگر واقعی |
| تست‌ها | ✅ ۲۷۴ تست سبز (`pytest`)؛ CI برقرار |
| ایمیج Docker | ⚠️ **پیاده‌سازی شده — اجرا نشده**: Dockerfile/compose آماده و پین‌شده؛ YAML اعتبارسنجی شده اما build هرگز کامل نشده (چک‌لیست انتشار را ببینید) |

جزئیات: `reports/` (گزارش فازها، گزارش نهایی، تحلیل هزینه) و `DEVELOPMENT_LOG.md`.

**وضعیت انتشار:** چک‌لیست مرجع برای آمادگی `v1` — از جمله چهار مسدودکننده
باقی‌مانده (ترجمه‌ی واقعی با اعتبار Z.ai، اجراهای ۴–۵ مقایسه، build و اجرای
Docker، برچسب‌گذاری انسانی داده‌ی gold) — در
[`reports/PHASE_5_RELEASE_READINESS.md`](reports/PHASE_5_RELEASE_READINESS.md)
§3 است. پروژه تا علامت‌گذاری همه‌ی موارد با شواهد اجرا، **برای تگ `v1` آماده نیست**.

برای راهنمای کامل انگلیسی → [`README_en.md`](README_en.md)

## Pipeline / خط لوله

1. `data_loader.py` — بارگذاری انگلیسی از Hugging Face یا فارسی از JSONL/CSV محلی.
2. `facts.py` — ۷ دسته (A–G) با prototype دوزبانه؛ فیلتر `en` / `fa` / `both`.
3. `labeler.py` — امبدینگ چندزبانه + cosine similarity.
4. `gap_analysis.py` — توزیع دسته‌ها، آمار امتیاز، دسته‌های قوی/ضعیف و نمودار.
5. `main.py` — اجرای تک‌پیکربندی.
6. `compare_runs.py` — مقایسه‌ی ۵ پیکربندی.
7. `export_for_translation.py` + `translate_dataset.py` + `prompts/semantic_persian_translation.txt` — خروجی و ترجمه‌ی LLM (ارائه‌دهنده از طریق `--provider` قابل‌تنظیم است).

## مدل‌های امبدینگ

| کلید | مدل | توضیح | حجم تقریبی دانلود* |
|------|------|--------|---------------------|
| `minilm` | `paraphrase-multilingual-MiniLM-L12-v2` | مدل task اول | ~۰٫۵ GB |
| `bge-m3` | `BAAI/bge-m3` | مدل قوی‌تر چندزبانه | ~۲٫۳ GB |
| `e5-large` | `intfloat/multilingual-e5-large` | با prefixهای `query:` / `passage:` | ~۲٫۳ GB |

\* حجم تقریبی وزن‌های fp32 بر اساس کارت مدل‌های Hugging Face — عدد تقریبی است، نه دقیق.

## کش مدل‌ها، Prefetch و اجرای آفلاین (Issue #14)

سه مسیر دیسکی متمایز — این‌ها را با هم اشتباه نگیرید:

| مسیر | مکان | محتوا |
|------|------|-------|
| **وزن مدل** (`models--*`) | `config.MODEL_CACHE_DIR` | وزن‌های دانلودشده‌ی encoder برای استنتاج |
| **کش امبدینگ** | `config.EMBED_CACHE_DIR` (`CACHE_DIR/embed_cache`) | بردارهای محاسبه‌شده (سرعت‌بخش به اجرای مجدد؛ حذفش بی‌خطر است) |
| **خروجی‌های pipeline** | `outputs/` | `labeled.csv`، گزارش‌ها و نمودار هر اجرا |

ترتیب تعیین مسیر کش وزن‌ها: `AUTISM_MODEL_CACHE_DIR` (override اختصاصی) >
`SENTENCE_TRANSFORMERS_HOME` (متغیر استاندارد) > `config.CACHE_DIR`
(`AUTISM_CACHE_DIR` → در صورت وجود درایو `D:` مسیر `D:\autism_assistant_cache`،
وگرنه `<project>/.cache`). این متغیرها را **قبل از اجرای Python** تنظیم کنید —
huggingface_hub پیکربندی خود را هنگام import می‌خواند.

```bash
# دیدن رجیستری مدل‌ها، مسیر کش و حجم تقریبی
python prefetch_models.py --list

# دانلود یک مدل (بدون دیتاست، بدون کلید API، بدون لیبل‌گذاری)
python prefetch_models.py --model minilm

# دانلود همه‌ی مدل‌های رجیستری
python prefetch_models.py --model all
```

**اجرای آفلاین.** پس از یک prefetch موفق، کلیدهای آفلاین را تنظیم کنید و
pipeline را معمول اجرا کنید — وزن‌ها از کش محلی می‌آیند و هیچ درخواست شبکه‌ای
زده نمی‌شود (در حالت `HF_HUB_OFFLINE=1` هر درخواست hub با خطا قطع می‌شود، پس
اجرای موفق یعنی واقعاً آفلاین):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --limit 5 --model minilm
```

**خطای مدل غایب.** اگر وزن در کش نباشد (یا آفلاین باشید و کش خالی باشد)،
بارگیری با پیامی قابل‌فهم شکست می‌خورد که کلید مدل، شناسه‌ی Hugging Face، مسیر
انتظار کش، وضعیت آفلاین و دقیقاً دستور prefetch لازم را می‌گوید — نه فقط یک
stacktrace شبکه. تشخیص:
1. `python prefetch_models.py --list` را اجرا کنید و خط مسیر کش را ببینید.
2. بررسی کنید همان پوشه `models--<org>--<name>` مدل شما را دارد.
3. همان دستور `python prefetch_models.py --model <key>` چاپ‌شده را آنلاین
   اجرا کنید و سپس آفلاین تلاش کنید.
بارگیر هرگز به encoder دیگری fallback نمی‌کند و در حالت آفلاین تلاش آنلاین
مخفی انجام نمی‌دهد.

## ماتریس مقایسه (۵ اجرا) — همه با فکت فارسی

| # | فکت | دیتاست | مدل |
|---|-----|--------|-----|
| 1 | فارسی | انگلیسی | MiniLM *(اجرای مجدد)* |
| 2 | فارسی | انگلیسی | bge-m3 |
| 3 | فارسی | انگلیسی | multilingual-e5-large |
| 4 | فارسی | فارسی | bge-m3 |
| 5 | فارسی | فارسی | multilingual-e5-large |

> اجرا‌های ۱-۳ روی دیتاست انگلیسی MentalChat16K و اجرا‌های ۴-۵ روی نسخه‌ی ترجمه‌شده‌ی فارسی (با LLM) انجام می‌شوند.

## Install / نصب

نسخه‌ی پکیج‌ها در `requirements.txt` **پین** شده است (مجموعه‌ی سازگار و تست‌شده؛
جزئیات در `reports/PHASE_0_STABILIZATION.md`). ابتدا torch را از ایندکس CPU نصب کنید:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
grep -vE '^\s*torch([=<>]|$)' requirements.txt > /tmp/requirements.notorch.txt   # ویندوز: findstr /V /C:"torch" requirements.txt > requirements.notorch.txt
pip install -r /tmp/requirements.notorch.txt
python -m pip check             # بررسی سازگاری گراف وابستگی‌ها
```

buildی CUDA از همان نسخه هم سازگار است. تست‌های واحد کاملاً آفلاین اجرا می‌شوند —
بدون دانلود مدل و بدون نیاز به کلید API (`pytest`).

### اجرای سریع با اسکریپت‌های راه‌اندازی (ویندوز)

اسکریپت‌های `launch_ui.bat` و `launch_ui.ps1` تمام مراحل را خودکار می‌کنند:
ساخت venv، نصب CPU-only PyTorch، نصب وابستگی‌ها، `pip check`، و اجرای
Streamlit. در اجراهای بعدی اگر وابستگی‌ها موجود باشند، نصب رد می‌شود.

```bat
launch_ui.bat        (دابل‌کلیک یا اجرا از cmd)
.\launch_ui.ps1      (PowerShell)
```

**محل کش مدل‌ها:** مدل‌های امبدینگ و دیتاست‌ها در `config.CACHE_DIR` ذخیره
می‌شوند (پیش‌فرض: `D:\autism_assistant_cache` در صورت وجود درایو `D:`،
در غیر این صورت `<project>\.cache`). با متغیر محیطی `AUTISM_CACHE_DIR`
قابل‌تغییر است. دایرکتوری در اولین استفاده ساخته می‌شود.

**خطاهای رایج:**
- `pip check` ناموفق → گراف وابستگی ناسازگار است؛ `.venv` را حذف و مجدد اجرا کنید.
- نصب PyTorch ناموفق → دسترسی شبکه به `download.pytorch.org` را بررسی کنید.
- Streamlit پیدا نشد → venv موجود است اما وابستگی‌ها نصب نیستند؛ `.venv` را حذف و مجدد اجرا کنید.

گزارش‌های مهندسی در پوشه‌ی `reports/` است (آدیت فازها، معماری، نقشه‌ی راه،
تأیید وابستگی‌ها).

## Docker

```bash
cp .env.example .env    # اختیاری؛ HF_TOKEN یا OPENAI_API_KEY
docker compose up --build
```

UI روی [http://localhost:8501](http://localhost:8501) باز می‌شود. مدل‌ها بار اول دانلود می‌شوند و در volume `model_cache` می‌مانند. خروجی‌ها در `./outputs` و داده در `./data` ذخیره می‌شوند.

```bash
# یک اجرای CLI داخل همان ایمیج
docker compose run --rm pipeline main.py --limit 50 --model minilm --fact-lang fa --dataset en

# prefetch یک‌باره‌ی وزن مدل‌ها در volume ماندگار model_cache (بدون دیتاست)
docker compose run --rm prefetch                 # همه‌ی مدل‌ها
docker compose run --rm prefetch --model minilm  # یک مدل
```

**ماندگاری کش.** همه‌ی سرویس‌ها `AUTISM_CACHE_DIR=/cache` دارند و volume نام‌دار
`model_cache` روی `/cache` سوار است؛ داخل کانتینر مسیر کش وزن‌ها از طریق
`SENTENCE_TRANSFORMERS_HOME=/cache/hf/sentence-transformers` روی همان volume
حل می‌شود، پس وزن‌های دانلودشده پس از حذف/ساخت مجدد کانتینر می‌مانند. prefetch
سمت هاست (بدون env) وزن‌ها را در `config.MODEL_CACHE_DIR` هاست می‌گذارد —
برای استفاده از همان وزن‌ها در کانتینر، پوشه‌ی هاست را روی
`/cache/hf/sentence-transformers` سوار کنید. وزن مدل‌ها (`models--*`)، کش
امبدینگ (`/cache/embed_cache`) و خروجی‌ها (`./outputs`) مسیرهای جدایی هستند.

> ماندگاری volume بر اساس پیکربندی compose مستند شده است؛ تأیید build واقعی و
> ساخت مجدد کانتینر به در دسترس بودن Docker daemon وابسته است (در PR شماره
> Issue #14 به‌صورت BLOCKED گزارش شده).

## ترجمه‌ی دیتاست به فارسی

```bash
python export_for_translation.py --limit 500
# اختیاری، اگر OPENAI_API_KEY دارید:
# python translate_dataset.py --limit 500
# یا دستی با prompts/semantic_persian_translation.txt ترجمه کنید
# خروجی: data/mentalchat16k_fa.jsonl
```

## UI / رابط کاربری (پیشنهادی)

پروژه **modular** است؛ برای اجرای آسان از UI مرورگر استفاده کنید:

```bash
# Windows
.\launch_ui.ps1
# یا
launch_ui.bat
# یا
streamlit run ui.py
# یا
docker compose up --build
```

صفحات UI:

| صفحه | کار |
|------|-----|
| Home (`ui.py`) | نمای کلی ماژول‌ها و وضعیت دیتاست |
| Quick Label | لیبل‌گذاری تعاملی یک متن |
| Run Pipeline | اجرای کامل تک‌پیکربندی |
| Compare Models | ماتریس ۵تایی مدل‌ها |
| Results | مشاهده خروجی‌ها و گزارش‌ها |
| Translate / Export | خروجی برای ترجمه‌ی فارسی + پرامپت |

کد UI در `ui.py` + `pages/` و کمک‌کننده‌های مشترک در `ui_lib/` است.

## Run / اجرا (CLI)

```bash
# فکت فارسی × دیتاست انگلیسی × MiniLM  (اجرای ۱ ماتریس)
python main.py --dataset en --fact-lang fa --model minilm --out outputs/fa_en_minilm

# هر ۵ مقایسه:
# اجرا‌های ۱-۳: دیتاست انگلیسی (نیازی به فایل FA نیست)
# اجرا‌های ۴-۵: نیاز به data/mentalchat16k_fa.jsonl
python compare_runs.py --limit 500

# smoke test با fixture کوچک (فقط برای run‌های دیتاست فارسی)
python compare_runs.py --limit 5 --dataset-path tests/fixtures/mentalchat16k_fa.sample.jsonl --only 4
```

خروجی تک‌اجرا در `outputs/` (یا `--out`) و خروجی مقایسه در `outputs/comparison/` نوشته می‌شود
(شامل `comparison_report.md`، `comparison_shares.png` و هیستوگرام توزیع نمره‌ی خام هر مدل `score_distributions.png`).

> **خواندن ستون «مطمئن%» (Issue #15):** این درصد سهم امتیازهای شباهت کسینوسی بالاتر از آستانه است،
> **نه دقت مدل** — مقیاس نمره‌ی کسینوسی بین encoderها متفاوت است (مثلاً e5-large نزدیک ۰٫۸۰ متمرکز می‌شود
> و آستانه‌ی مشترک ۰٫۳۵ همه را رد می‌کند). گزارش مقایسه جمله‌ی هشدار ثابت، هیستوگرام توزیع نمره‌ی خام
> با محور مشترک و بدون رتبه‌بندی «بهترین مدل» را دارد.

## Tests / تست‌ها

```bash
python -m pytest
```

```bash
python -m pytest
```

## اعتبارسنجی عملیاتی (دستی — Z.ai واقعی و Docker)

تست‌های آفلاین (۲۴۸ تست، `pytest`) منطق خط لوله را بدون تماس با API واقعی
بررسی می‌کنند. اعتبارسنجی عملیاتی **endpoint واقعی** Z.ai و **build واقعی**
Docker را آزمایش می‌کند — این‌ها نیازمند فعال‌سازی دستی و سرویس‌های بیرونی هستند.

### نحوه‌ی فعال‌سازی

در GitHub به **Actions → Operational Validation → Run workflow** بروید.
اعتبارسنجی مورد نظر (Z.ai probe، Docker build، یا هر دو) را انتخاب کنید.

**پیش‌نیازها:**
- `ZAI_API_KEY` باید به‌عنوان **repository secret** در GitHub Actions تنظیم
  شده باشد (Settings → Secrets and variables → Actions). هرگز در ریپو کامیت نکنید.
- Docker build نیازی به secret اضافه ندارد.

### چه چیزی بررسی می‌شود

| اعتبارسنجی | عملکرد | پیش‌نیاز |
|---|---|---|
| Z.ai probe | ارسال یک جمله‌ی ساده به `api.z.ai` برای بررسی auth و دسترسی به مدل | `ZAI_API_KEY` secret |
| Docker build | ساخت ایمیج پروژه با `docker compose build ui` | Docker daemon |
| Docker healthcheck | اجرای کانتینر و انتظار برای `/_stcore/health` → 200 | Docker build موفق |

### معنی وضعیت‌ها

| وضعیت | معنی |
|---|---|
| `SUCCESS` | اعتبارسنجی با شواهد اجرا موفق بود |
| `BLOCKED` | احراز هویت/اجرا شروع شد اما وابستگی بیرونی مانع شد (مثلاً code 1113) |
| `SKIPPED` | درخواست نشد یا credentials تنظیم نشده بود |
| `FAILED` | اجرا شد اما ناموفق بود (auth، network، build و غیره) |

### یافتن نتایج

- لاگ workflow: **Actions → Operational Validation → [شماره اجرا]**
- خلاصه‌ی job: بالای هر اجرای workflow نمایش داده می‌شود
- Issue #17: workflow به‌صورت خودکار خلاصه‌ای از آخرین نتایج را در Issue #17 ثبت می‌کند

### چه چیزی پوشش داده نمی‌شود

- ترجمه‌ی واقعی رکوردهای دیتاست (نیازمند اجرای دستی `translate_dataset.py`)
- ارزیابی کیفیت مدل (نیازمند داده‌ی gold برچسب‌خورده)
- دقت بالینی (این یک پروتوتایپ پژوهشی است)

## Categories / دسته‌ها (A–G)

| Key | Title                         | #facts |
| --- | ----------------------------- | ------ |
| A   | Social Communication          | 6      |
| B   | Sensory Processing            | 3      |
| C   | Emotional Regulation & Stress | 4      |
| D   | Routine & Predictability      | 3      |
| E   | Special Interests & Strengths | 7      |
| F   | Diagnosis & Support           | 6      |
| G   | Autism Knowledge & Awareness  | 15     |

(فکت‌های کامل در `facts.py`؛ نسخه‌ی نهایی تأییدشده.)

## Notes / یادداشت‌ها

* تطبیق معنایی است، نه کلمه‌ای.
* به‌طور پیش‌فرض فقط متن بیمار (`input`) لیبل می‌شود، نه پرامپت مشاور و پاسخ درمانی (`LABEL_TEXT_MODE=patient`).
* امتیاز هر دسته = حداکثر شباهت روی prototypeهای همان دسته (با زبان انتخاب‌شده).
* برای e5، نمونه‌ها `query:` و فکت‌ها `passage:` prefix می‌گیرند.
* gap analysis مشخص می‌کند کدام دسته‌ها پوشش ضعیف دارند و برای RAG/fine-tuning به داده‌ی مکمل نیاز دارند.
