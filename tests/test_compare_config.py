import config


def test_comparison_runs_are_five():
    assert len(config.COMPARISON_RUNS) == 5


def test_comparison_matrix_matches_supervisor_plan():
    expected = [
        ("en", "fa", "minilm"),
        ("en", "fa", "bge-m3"),
        ("en", "fa", "e5-large"),
        ("fa", "fa", "bge-m3"),
        ("fa", "fa", "e5-large"),
    ]
    actual = [
        (r["fact_lang"], r["dataset"], r["model_key"])
        for r in config.COMPARISON_RUNS
    ]
    assert actual == expected


def test_embed_models_registered():
    for key in ("minilm", "bge-m3", "e5-large"):
        assert key in config.EMBED_MODELS
        assert "name" in config.EMBED_MODELS[key]
