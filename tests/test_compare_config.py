import config


def test_comparison_runs_are_five():
    assert len(config.COMPARISON_RUNS) == 5


def test_comparison_matrix_matches_supervisor_plan():
    # All five runs use Persian facts (FA). Runs 1-3 use the English dataset;
    # runs 4-5 use the Persian dataset (after GPT-4+ translation).
    expected = [
        ("fa", "en", "minilm"),
        ("fa", "en", "bge-m3"),
        ("fa", "en", "e5-large"),
        ("fa", "fa", "bge-m3"),
        ("fa", "fa", "e5-large"),
    ]
    actual = [
        (r["fact_lang"], r["dataset"], r["model_key"])
        for r in config.COMPARISON_RUNS
    ]
    assert actual == expected


def test_comparison_run_ids_match_matrix():
    expected_ids = [
        "1_fa_en_minilm",
        "2_fa_en_bge-m3",
        "3_fa_en_e5-large",
        "4_fa_fa_bge-m3",
        "5_fa_fa_e5-large",
    ]
    actual_ids = [r["id"] for r in config.COMPARISON_RUNS]
    assert actual_ids == expected_ids


def test_embed_models_registered():
    for key in ("minilm", "bge-m3", "e5-large"):
        assert key in config.EMBED_MODELS
        assert "name" in config.EMBED_MODELS[key]
