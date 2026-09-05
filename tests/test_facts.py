from collections import Counter

from facts import CATEGORIES, CATEGORY_KEYS, FACTS, get_prototypes

EXPECTED_PER_CATEGORY = {"A": 6, "B": 3, "C": 4, "D": 3, "E": 7, "F": 6, "G": 15}


def test_final_fact_inventory():
    assert len(FACTS) == 44
    assert [f.id for f in FACTS] == list(range(1, 45))
    counts = Counter(f.category for f in FACTS)
    assert counts == EXPECTED_PER_CATEGORY
    assert CATEGORY_KEYS == list(CATEGORIES.keys()) == list("ABCDEFG")


def test_person_first_persian_wording():
    """Expert-approved facts use person-first ASD wording, not identity-first."""
    person_facts = [f for f in FACTS if "افراد" in f.fa or "دختران" in f.fa]
    assert person_facts
    for f in person_facts:
        assert "افراد اوتیستیک" not in f.fa
        assert "دختران اوتیستیک" not in f.fa


def test_emotion_facts_do_not_overlap_routine_or_support():
    """C must stay about emotion so D/F can win their own samples."""
    c_facts = [f for f in FACTS if f.category == "C"]
    banned = (
        "routine",
        "predictability",
        "روتین",
        "پیش‌بینی",
        "early intervention",
        "مداخلات زودهنگام",
    )
    for f in c_facts:
        blob = f"{f.en} {f.fa}".lower()
        for word in banned:
            assert word.lower() not in blob, f"C fact {f.id} still overlaps: {word}"


def test_comorbidity_fact_in_knowledge_category():
    comorbidity = next(f for f in FACTS if f.id == 33)
    assert comorbidity.category == "G"
    assert "فلج مغزی" in comorbidity.fa
    assert "cerebral palsy" in comorbidity.en.lower()


def test_get_prototypes_count_both():
    prototypes = get_prototypes()
    assert len(prototypes) == len(FACTS) * 2
    assert all(p["category"] in CATEGORY_KEYS for p in prototypes)
    assert set(p["lang"] for p in prototypes) == {"en", "fa"}


def test_get_prototypes_en_only():
    prototypes = get_prototypes(lang="en")
    assert len(prototypes) == len(FACTS)
    assert all(p["lang"] == "en" for p in prototypes)


def test_get_prototypes_fa_only():
    prototypes = get_prototypes(lang="fa")
    assert len(prototypes) == len(FACTS)
    assert all(p["lang"] == "fa" for p in prototypes)


def test_get_prototypes_texts():
    prototypes = get_prototypes()
    first = prototypes[0]
    assert "text" in first
    assert first["fact_id"] == FACTS[0].id


def test_get_prototypes_invalid_lang():
    try:
        get_prototypes(lang="de")
        assert False, "expected ValueError"
    except ValueError:
        pass
