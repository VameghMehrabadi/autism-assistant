"""فکت‌های اوتیسم در ۷ دسته (A–G) به‌عنوان prototypeهای معنایی.

هر فکت دو نسخه‌ی انگلیسی و فارسی دارد. هنگام امبدینگ، دو نسخه به‌صورت دو prototype
مستقل در نظر گرفته می‌شوند تا فضای معنایی چندزبانه بهتر پوشش داده شود.
دسته‌ی G (دانش و آگاهی درباره‌ی اوتیسم) نیز در فاز لیبل‌گذاری داده استفاده می‌شود.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: int
    category: str  # "A" .. "G"
    en: str
    fa: str


# --- تعریف دسته‌ها / Category metadata ---
CATEGORIES: dict[str, dict[str, str]] = {
    "A": {"key": "A", "title_en": "Social Communication",
          "title_fa": "ارتباطات و تعامل اجتماعی"},
    "B": {"key": "B", "title_en": "Sensory Processing",
          "title_fa": "پردازش حسی"},
    "C": {"key": "C", "title_en": "Emotional Regulation & Stress",
          "title_fa": "تنظیم هیجان و استرس"},
    "D": {"key": "D", "title_en": "Routine & Predictability",
          "title_fa": "روتین و پیش‌بینی‌پذیری"},
    "E": {"key": "E", "title_en": "Special Interests & Strengths",
          "title_fa": "علایق ویژه و نقاط قوت"},
    "F": {"key": "F", "title_en": "Diagnosis & Support",
          "title_fa": "تشخیص و حمایت"},
    "G": {"key": "G", "title_en": "Autism Knowledge & Awareness",
          "title_fa": "دانش و آگاهی درباره اوتیسم"},
}

CATEGORY_KEYS: list[str] = list(CATEGORIES.keys())


FACTS: list[Fact] = [
    # ---------- A. Social Communication ----------
    Fact(1, "A",
         "Autistic people may struggle with social communication and reading non-verbal cues.",
         "افراد اوتیستیک ممکن است در ارتباطات اجتماعی و درک نشانه‌های غیرکلامی چالش داشته باشند."),
    Fact(2, "A",
         "Eye contact can be difficult or exhausting for some autistic people.",
         "تماس چشمی برای برخی افراد اوتیستیک دشوار یا خسته‌کننده است."),
    Fact(3, "A",
         "Difficulty understanding body language, tone of voice, sarcasm and social cues is common in autism.",
         "دشواری در درک زبان بدن، لحن صدا، کنایه و نشانه‌های اجتماعی در اوتیسم شایع است."),
    Fact(4, "A",
         "Autistic people usually want to connect, but their way of communicating may be different.",
         "افراد اوتیستیک معمولاً به ارتباط علاقه دارند، اما ممکن است شیوه برقراری ارتباط آن‌ها متفاوت باشد."),
    Fact(5, "A",
         "Verbal ability in autistic people varies widely, from non-speaking to highly articulate.",
         "توانایی کلامی در افراد اوتیستیک بسیار متنوع است؛ از افراد غیرکلامی تا افراد بسیار فصیح."),
    Fact(6, "A",
         "Autistic people have feelings and empathy, but may express them differently from others.",
         "افراد اوتیستیک احساسات و همدلی دارند، اما ممکن است آن را متفاوت از دیگران ابراز کنند."),

    # ---------- B. Sensory Processing ----------
    Fact(7, "B",
         "Many autistic people have differences in sensory processing.",
         "بسیاری از افراد اوتیستیک تفاوت‌هایی در پردازش حسی دارند."),
    Fact(8, "B",
         "Hyper- or hypo-reactivity to sound, light, touch, smell or other stimuli is common.",
         "حساسیت بیش از حد یا کمتر از حد معمول به صدا، نور، لمس، بو یا سایر محرک‌ها شایع است."),
    Fact(9, "B",
         "Sensory overload can cause anxiety, exhaustion or severe distress.",
         "اضافه‌بار حسی می‌تواند موجب اضطراب، خستگی یا ناراحتی شدید شود."),

    # ---------- C. Emotional Regulation & Stress ----------
    Fact(10, "C",
         "Repetitive behaviors (stimming) often help regulate emotion or reduce stress.",
         "رفتارهای تکراری اغلب به تنظیم هیجان یا کاهش استرس کمک می‌کنند."),
    Fact(11, "C",
         "Sudden changes in routine or environment can create anxiety and stress.",
         "تغییرات ناگهانی در برنامه یا محیط می‌تواند اضطراب و استرس ایجاد کند."),
    Fact(12, "C",
         "Structure, consistency and predictability usually help reduce anxiety.",
         "ساختار، ثبات و پیش‌بینی‌پذیری معمولاً به کاهش اضطراب کمک می‌کند."),
    Fact(13, "C",
         "Early interventions and appropriate support can improve quality of life and adaptation.",
         "مداخلات زودهنگام و حمایت مناسب می‌توانند کیفیت زندگی و سازگاری فرد را بهبود دهند."),

    # ---------- D. Routine & Predictability ----------
    Fact(14, "D",
         "Many autistic people need consistency and predictability to feel safe.",
         "بسیاری از افراد اوتیستیک برای احساس امنیت به ثبات و پیش‌بینی‌پذیری نیاز دارند."),
    Fact(15, "D",
         "Sudden changes in plans or environment can be very hard for some autistic people.",
         "تغییر ناگهانی برنامه‌ها یا محیط می‌تواند برای برخی افراد اوتیستیک بسیار دشوار باشد."),
    Fact(16, "D",
         "A preference for clearly defined routines is a common trait in the autism spectrum.",
         "ترجیح روتین‌های مشخص یکی از ویژگی‌های رایج در طیف اوتیسم است."),

    # ---------- E. Special Interests & Strengths ----------
    Fact(17, "E",
         "Many autistic people have intense, deep special interests.",
         "بسیاری از افراد اوتیستیک علایق ویژه و عمیقی دارند."),
    Fact(18, "E",
         "Some autistic people show remarkable skill or expertise in specific areas.",
         "برخی افراد اوتیستیک در زمینه‌های خاص مهارت یا تخصص چشمگیری دارند."),
    Fact(19, "E",
         "Long focus and high attention to detail are seen in some autistic people.",
         "تمرکز طولانی‌مدت و توجه زیاد به جزئیات در برخی افراد اوتیستیک دیده می‌شود."),
    Fact(20, "E",
         "Autism does not mean low intelligence.",
         "اوتیسم به معنای کم‌هوشی نیست."),
    Fact(21, "E",
         "Autistic people can have average, high or low intelligence.",
         "افراد اوتیستیک می‌توانند هوش متوسط، بالا یا پایین داشته باشند."),
    Fact(22, "E",
         "Some autistic people can live independently, study, work and have successful relationships.",
         "برخی افراد اوتیستیک می‌توانند زندگی مستقل، تحصیل، اشتغال و روابط موفق داشته باشند."),
    Fact(23, "E",
         "Not all autistic people have savant syndrome or extraordinary talents.",
         "همه افراد اوتیستیک دارای استعداد خارق‌العاده یا سندرم ساوان نیستند."),

    # ---------- F. Diagnosis & Support ----------
    Fact(24, "F",
         "Signs of autism usually appear in early childhood.",
         "علائم اوتیسم معمولاً در اوایل کودکی ظاهر می‌شوند."),
    Fact(25, "F",
         "Autism diagnosis is usually based on behavioral assessment and developmental history.",
         "تشخیص اوتیسم معمولاً بر اساس ارزیابی رفتاری و تاریخچه رشد انجام می‌شود."),
    Fact(26, "F",
         "There is no blood test or definitive medical test for diagnosing autism.",
         "هیچ آزمایش خون یا تست پزشکی قطعی برای تشخیص اوتیسم وجود ندارد."),
    Fact(27, "F",
         "Early diagnosis can help improve life outcomes.",
         "تشخیص زودهنگام می‌تواند به بهبود نتایج زندگی کمک کند."),
    Fact(28, "F",
         "Supportive, educational, speech and occupational therapies can be helpful.",
         "مداخلات حمایتی، آموزشی، گفتاردرمانی و کاردرمانی می‌توانند مفید باشند."),
    Fact(29, "F",
         "The goal of support is to improve quality of life and independence, not to change identity.",
         "هدف حمایت‌ها افزایش کیفیت زندگی و استقلال فرد است، نه تغییر هویت او."),

    # ---------- G. Autism Knowledge & Awareness ----------
    Fact(30, "G",
         "Autism is a neurodevelopmental condition, not a mental illness.",
         "اوتیسم یک وضعیت عصبی-رشدی است، نه یک بیماری روانی."),
    Fact(31, "G",
         "Autism is part of human neurodiversity.",
         "اوتیسم بخشی از تنوع عصبی انسان محسوب می‌شود."),
    Fact(32, "G",
         "Autism is a spectrum and autistic people can have very different traits and needs.",
         "اوتیسم یک طیف است و افراد اوتیستیک می‌توانند ویژگی‌ها و نیازهای بسیار متفاوتی داشته باشند."),
    Fact(33, "G",
         "Autism is a lifelong condition.",
         "اوتیسم یک وضعیت مادام‌العمر است."),
    Fact(34, "G",
         "Autism has no specific physical appearance.",
         "اوتیسم ظاهر فیزیکی مشخصی ندارد."),
    Fact(35, "G",
         "The exact cause of autism is not yet fully known.",
         "علت دقیق اوتیسم هنوز کاملاً مشخص نیست."),
    Fact(36, "G",
         "Genetic factors play an important role in autism.",
         "عوامل ژنتیکی نقش مهمی در بروز اوتیسم دارند."),
    Fact(37, "G",
         "Some environmental factors may contribute alongside genetics.",
         "برخی عوامل محیطی ممکن است در کنار ژنتیک نقش داشته باشند."),
    Fact(38, "G",
         "Autism is not caused by parenting style or parents' behavior.",
         "اوتیسم نتیجه سبک فرزندپروری یا رفتار والدین نیست."),
    Fact(39, "G",
         "Autism is diagnosed more often in boys than in girls.",
         "اوتیسم در پسران بیشتر از دختران تشخیص داده می‌شود."),
    Fact(40, "G",
         "Some autistic girls may be diagnosed less often due to masking.",
         "برخی دختران اوتیستیک ممکن است به دلیل ماسکینگ کمتر تشخیص داده شوند."),
    Fact(41, "G",
         "Autism occurs across all social and cultural groups.",
         "اوتیسم در همه گروه‌های اجتماعی و فرهنگی دیده می‌شود."),
    Fact(42, "G",
         "Vaccines do not cause autism.",
         "واکسن‌ها باعث اوتیسم نمی‌شوند."),
    Fact(43, "G",
         "Respecting neurodiversity and reducing social stigma matters.",
         "احترام به تفاوت‌های عصبی و کاهش انگ اجتماعی اهمیت دارد."),
]


def get_prototypes(lang: str = "both") -> list[dict]:
    """برگرداندن prototypeها به‌صورت لیست دیکشنری برای امبدینگ.

    Args:
        lang: "en" | "fa" | "both"
            - both: هر فکت دو prototype (EN و FA) می‌سازد (رفتار task اول).
            - en / fa: فقط همان زبان برای مقایسه‌ی کنترل‌شده‌ی چندمدلی.
    """
    lang = (lang or "both").lower().strip()
    if lang not in {"en", "fa", "both"}:
        raise ValueError(f"Unsupported fact lang '{lang}'. Use en|fa|both.")

    protos: list[dict] = []
    for f in FACTS:
        if lang in ("en", "both"):
            protos.append(
                {"fact_id": f.id, "category": f.category, "lang": "en", "text": f.en}
            )
        if lang in ("fa", "both"):
            protos.append(
                {"fact_id": f.id, "category": f.category, "lang": "fa", "text": f.fa}
            )
    return protos


def category_titles_bilingual() -> dict[str, str]:
    return {k: f"{v['key']}. {v['title_en']} — {v['title_fa']}"
            for k, v in CATEGORIES.items()}


if __name__ == "__main__":
    protos = get_prototypes()
    print(f"Total facts: {len(FACTS)} | Total prototypes (en+fa): {len(protos)}")
    for k in CATEGORY_KEYS:
        n = sum(1 for f in FACTS if f.category == k)
        print(f"  {k}: {n} facts -> {n*2} prototypes | {CATEGORIES[k]['title_en']}")
