from facts import CATEGORY_KEYS, FACTS, get_prototypes


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
