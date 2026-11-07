"""Issue #14 — model-weights cache, prefetch CLI, missing-model error, offline.

All unit tests are fully offline: the SentenceTransformer constructor is
mocked, so no download happens and no real weights are touched. The real
offline acceptance run (prefetch → HF_HUB_OFFLINE=1 → --limit 5) is executed
separately and reported in the PR; it is not asserted here.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from huggingface_hub.utils import (
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)

import config
import model_loader
import prefetch_models
from model_loader import (
    ModelLoadError,
    ModelWeightsMissingError,
    format_offline_state,
    load_sentence_transformer,
    offline_state,
    prefetch_command,
)


# --------------------------------------------------------------------------
# Fix 1 — model-weights cache configuration
# --------------------------------------------------------------------------


def _resolve(monkeypatch: pytest.MonkeyPatch, **env: str):
    for var in ("AUTISM_MODEL_CACHE_DIR", "SENTENCE_TRANSFORMERS_HOME"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    return config._resolve_model_cache_dir()


def test_model_cache_resolution_priority(monkeypatch, tmp_path):
    # no env -> existing default layout (keeps already-downloaded weights)
    path, source = _resolve(monkeypatch)
    assert path == config.CACHE_DIR
    assert "default" in source
    # the standard Sentence-Transformers variable is honored
    path, source = _resolve(
        monkeypatch, SENTENCE_TRANSFORMERS_HOME=str(tmp_path / "st")
    )
    assert path == tmp_path / "st"
    assert source == "SENTENCE_TRANSFORMERS_HOME"
    # app-specific override wins over both
    path, source = _resolve(
        monkeypatch,
        SENTENCE_TRANSFORMERS_HOME=str(tmp_path / "st"),
        AUTISM_MODEL_CACHE_DIR=str(tmp_path / "app"),
    )
    assert path == tmp_path / "app"
    assert source == "AUTISM_MODEL_CACHE_DIR"


def test_model_weights_cache_distinct_from_embedding_cache(monkeypatch, tmp_path):
    """Weights and computed-embedding caches are separate artifacts; a
    model-cache override must never move the embedding cache."""
    monkeypatch.setenv("AUTISM_MODEL_CACHE_DIR", str(tmp_path / "weights"))
    path, _ = config._resolve_model_cache_dir()
    assert path != config.EMBED_CACHE_DIR
    # embedding cache stays anchored to CACHE_DIR (backward compatible)
    assert config.EMBED_CACHE_DIR == config.CACHE_DIR / "embed_cache"


def test_model_registry_consistency():
    run_keys = {r["model_key"] for r in config.COMPARISON_RUNS}
    assert run_keys <= set(config.EMBED_MODELS)
    assert config.EMBED_MODEL_KEY in config.EMBED_MODELS
    for key, meta in config.EMBED_MODELS.items():
        assert meta["name"], f"{key} has no HF id"
        assert meta["prefix"] in {"none", "e5"}


# --------------------------------------------------------------------------
# Fix 2 — prefetch CLI
# --------------------------------------------------------------------------


def test_prefetch_list_prints_registry_and_cache(capsys):
    assert prefetch_models.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert str(config.MODEL_CACHE_DIR) in out
    assert str(config.EMBED_CACHE_DIR) in out
    for key, meta in config.EMBED_MODELS.items():
        assert key in out
        assert meta["name"] in out
    assert "approx" in out  # sizes marked approximate


def test_prefetch_single_model_selection(monkeypatch):
    called = []
    monkeypatch.setattr(
        prefetch_models, "prefetch_one", lambda k: called.append(k) or True
    )
    assert prefetch_models.main(["--model", "minilm"]) == 0
    assert called == ["minilm"]


def test_prefetch_all_models(monkeypatch):
    called = []
    monkeypatch.setattr(
        prefetch_models, "prefetch_one", lambda k: called.append(k) or True
    )
    assert prefetch_models.main(["--model", "all"]) == 0
    assert sorted(called) == sorted(config.EMBED_MODELS)


def test_prefetch_defaults_to_registry_default(monkeypatch):
    called = []
    monkeypatch.setattr(
        prefetch_models, "prefetch_one", lambda k: called.append(k) or True
    )
    assert prefetch_models.main([]) == 0
    assert called == [config.EMBED_MODEL_KEY]


def test_prefetch_rejects_unknown_model():
    with pytest.raises(SystemExit):
        prefetch_models.build_parser().parse_args(["--model", "nope"])


def test_prefetch_uses_configured_cache_destination(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_load(name, model_key="custom", device="cpu", cache_folder=None):
        seen.update(name=name, model_key=model_key, cache_folder=cache_folder)
        return SimpleNamespace(parameters=lambda: iter(()))

    monkeypatch.setattr(prefetch_models, "load_sentence_transformer", fake_load)
    monkeypatch.setattr(config, "MODEL_CACHE_DIR", tmp_path / "weights")
    assert prefetch_models.prefetch_one("minilm") is True
    assert seen["name"] == config.EMBED_MODELS["minilm"]["name"]
    assert seen["model_key"] == "minilm"
    assert Path(seen["cache_folder"]) == tmp_path / "weights"


def test_prefetch_reports_failure_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(prefetch_models, "prefetch_one", lambda k: False)
    assert prefetch_models.main(["--model", "minilm"]) == 1
    err = capsys.readouterr().err
    assert "failed" in err and "minilm" in err


def test_prefetch_all_continues_after_a_failure(monkeypatch):
    calls = []

    def fake(key: str) -> bool:
        calls.append(key)
        return key != "bge-m3"

    monkeypatch.setattr(prefetch_models, "prefetch_one", fake)
    assert prefetch_models.main(["--model", "all"]) == 1
    # every registered model is still attempted — one failure is reported,
    # the rest are not silently skipped
    assert sorted(calls) == sorted(config.EMBED_MODELS)


def test_prefetch_command_matches_cli():
    assert prefetch_command("minilm") == (
        "python prefetch_models.py --model minilm"
    )
    # unknown keys fall back to the default model, never a broken command
    assert prefetch_command("ghost") == (
        f"python prefetch_models.py --model {config.EMBED_MODEL_KEY}"
    )


# --------------------------------------------------------------------------
# Fix 3 — actionable missing-model error, no fallback
# --------------------------------------------------------------------------


def test_missing_weights_error_is_actionable(monkeypatch, tmp_path):
    def raise_offline(*args, **kwargs):
        raise OfflineModeIsEnabled("offline mode is enabled")

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_offline)
    monkeypatch.setattr(config, "MODEL_CACHE_DIR", tmp_path / "weights")
    monkeypatch.setattr(config, "MODEL_CACHE_SOURCE", "TEST")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelWeightsMissingError) as ei:
        load_sentence_transformer(
            config.EMBED_MODELS["bge-m3"]["name"], model_key="bge-m3"
        )
    msg = str(ei.value)
    assert "bge-m3" in msg                                       # model key
    assert config.EMBED_MODELS["bge-m3"]["name"] in msg          # HF model id
    assert str(tmp_path / "weights") in msg                      # cache location
    assert "ENABLED" in msg and "HF_HUB_OFFLINE=1" in msg        # offline state
    assert prefetch_command("bge-m3") in msg                     # exact command
    assert isinstance(ei.value.__cause__, OfflineModeIsEnabled)  # cause kept


def test_offline_load_makes_a_single_attempt_no_online_retry(monkeypatch):
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise OfflineModeIsEnabled("offline mode is enabled")

    monkeypatch.setattr(model_loader, "SentenceTransformer", failing)
    with pytest.raises(ModelWeightsMissingError):
        load_sentence_transformer("org/model", model_key="minilm")
    assert len(calls) == 1  # no hidden second (online) attempt


def test_no_silent_fallback_to_another_encoder(monkeypatch):
    requested = []

    def failing(name, *args, **kwargs):
        requested.append(name)
        raise LocalEntryNotFoundError("not in local cache")

    monkeypatch.setattr(model_loader, "SentenceTransformer", failing)
    with pytest.raises(ModelWeightsMissingError):
        load_sentence_transformer("BAAI/bge-m3", model_key="bge-m3")
    assert requested == ["BAAI/bge-m3"]  # only the requested model was touched


def test_unrelated_errors_propagate_unchanged(monkeypatch):
    class DiskOnFire(RuntimeError):
        pass

    def raise_weird(*args, **kwargs):
        raise DiskOnFire("controller failure")

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_weird)
    with pytest.raises(DiskOnFire) as ei:  # NOT wrapped in ModelLoadError
        load_sentence_transformer("org/model", model_key="minilm")
    assert not isinstance(ei.value, ModelLoadError)


def test_gated_repo_reports_auth_not_missing(monkeypatch):
    def raise_gated(*args, **kwargs):
        raise GatedRepoError("401 — gated repository")

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_gated)
    with pytest.raises(ModelLoadError) as ei:
        load_sentence_transformer("org/gated", model_key="minilm")
    assert not isinstance(ei.value, ModelWeightsMissingError)
    assert "HF_TOKEN" in str(ei.value)


def test_corrupt_cache_oserror_classified_as_missing(monkeypatch):
    def raise_os(*args, **kwargs):
        raise OSError("Error no file named pytorch_model.bin in the cache")

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_os)
    with pytest.raises(ModelWeightsMissingError):
        load_sentence_transformer("org/model", model_key="minilm")


def test_transformers_style_offline_oserror_classified_as_missing(monkeypatch):
    """Regression lock on the exact OSError text the real transformers stack
    raises when offline with an empty cache (builtins.OSError, not a hub
    exception class)."""
    def raise_os(*args, **kwargs):
        raise OSError(
            "We couldn't connect to 'https://huggingface.co' to load this "
            "file, couldn't find it in the cached files and it looks like "
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 is "
            "not the path to a directory containing a file named config.json."
        )

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_os)
    with pytest.raises(ModelWeightsMissingError):
        load_sentence_transformer("org/model", model_key="minilm")


def test_generic_oserror_not_misclassified(monkeypatch):
    def raise_os(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_os)
    with pytest.raises(OSError) as ei:  # propagates, not a "missing weights"
        load_sentence_transformer("org/model", model_key="minilm")
    assert not isinstance(ei.value, ModelLoadError)


def test_repository_not_found_is_missing_weights(monkeypatch):
    def raise_notfound(*args, **kwargs):
        raise RepositoryNotFoundError("404 — repository not found")

    monkeypatch.setattr(model_loader, "SentenceTransformer", raise_notfound)
    with pytest.raises(ModelWeightsMissingError):
        load_sentence_transformer("org/absent", model_key="e5-large")


# --------------------------------------------------------------------------
# Fix 4 — offline mode plumbing
# --------------------------------------------------------------------------


def test_offline_state_helpers(monkeypatch):
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    assert offline_state() == {
        "HF_HUB_OFFLINE": False,
        "TRANSFORMERS_OFFLINE": False,
    }
    assert "disabled" in format_offline_state()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    state = offline_state()
    assert state["HF_HUB_OFFLINE"] is True
    assert "HF_HUB_OFFLINE=1" in format_offline_state(state)
    # explicit "0" counts as disabled
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert offline_state()["HF_HUB_OFFLINE"] is False


def test_labeler_routes_through_model_loader():
    """The pipeline's only SentenceTransformer construction goes through the
    loader (single cache configuration + actionable errors)."""
    src = (Path(__file__).resolve().parents[1] / "labeler.py").read_text(
        encoding="utf-8"
    )
    assert "load_sentence_transformer(" in src
    assert "SentenceTransformer(" not in src  # no direct construction left
    assert "config.MODEL_CACHE_DIR" in src


def test_loader_defaults_to_configured_model_cache(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_st(name, device="cpu", cache_folder=None):
        seen.update(name=name, device=device, cache_folder=cache_folder)
        return SimpleNamespace()

    monkeypatch.setattr(model_loader, "SentenceTransformer", fake_st)
    monkeypatch.setattr(config, "MODEL_CACHE_DIR", tmp_path / "weights")
    load_sentence_transformer("org/model", model_key="minilm")
    assert seen["cache_folder"] == str(tmp_path / "weights")
    assert seen["device"] == "cpu"


# --------------------------------------------------------------------------
# Fix 5 — Docker volume configuration
# --------------------------------------------------------------------------


def _compose() -> dict:
    path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_docker_compose_model_cache_volume():
    compose = _compose()
    assert "model_cache" in compose.get("volumes", {})
    for svc in ("ui", "pipeline", "prefetch"):
        service = compose["services"][svc]
        mounts = service["volumes"]
        # weights cache volume present exactly once, nothing mounted over it
        targets = [m.split(":")[1] for m in mounts]
        assert targets.count("/cache") == 1
        assert "model_cache:/cache" in mounts
        env = service["environment"]
        assert env["AUTISM_CACHE_DIR"] == "/cache"
        assert env["SENTENCE_TRANSFORMERS_HOME"].startswith("/cache")
        # existing application volumes preserved
        assert "./data:/app/data" in mounts
        assert "./outputs:/app/outputs" in mounts


def test_docker_prefetch_service_downloads_all_models():
    compose = _compose()
    prefetch = compose["services"]["prefetch"]
    assert prefetch["command"] == ["prefetch_models.py", "--model", "all"]
    assert prefetch["entrypoint"] == ["python"]
    assert "prefetch" in prefetch.get("profiles", [])
