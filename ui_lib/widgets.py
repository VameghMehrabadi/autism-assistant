"""Reusable Streamlit controls for model / dataset / fact language."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from facts import category_titles_bilingual


MODEL_LABELS = {
    "minilm": "MiniLM (task-1 baseline)",
    "bge-m3": "BGE-M3",
    "e5-large": "multilingual-e5-large",
}


def render_model_selector(key: str = "model_key", default: str | None = None) -> str:
    keys = list(config.EMBED_MODELS.keys())
    default = default or config.EMBED_MODEL_KEY
    index = keys.index(default) if default in keys else 0
    return st.selectbox(
        "Embedding model",
        options=keys,
        index=index,
        format_func=lambda k: f"{k} — {MODEL_LABELS.get(k, k)}",
        key=key,
        help=config.EMBED_MODELS[keys[index]]["name"],
    )


def render_fact_lang_selector(
    key: str = "fact_lang",
    default: str = "en",
    allow_both: bool = True,
) -> str:
    options = ["en", "fa", "both"] if allow_both else ["en", "fa"]
    default = default if default in options else options[0]
    return st.selectbox(
        "Fact language (prototypes)",
        options=options,
        index=options.index(default),
        key=key,
        help="en / fa for controlled comparisons; both = original bilingual setup",
    )


def render_dataset_selector(key_prefix: str = "ds") -> tuple[str, str | None]:
    source = st.radio(
        "Dataset",
        options=["en", "fa", "sample", "custom"],
        format_func={
            "en": "English MentalChat16K (Hugging Face)",
            "fa": "Persian JSONL (data/mentalchat16k_fa.jsonl)",
            "sample": "Tiny Persian sample (tests/fixtures)",
            "custom": "Custom file path",
        }.__getitem__,
        key=f"{key_prefix}_source",
        horizontal=False,
    )
    if source == "en":
        return "en", None
    if source == "fa":
        return "fa", str(config.PERSIAN_DATASET_PATH)
    if source == "sample":
        path = config.ROOT / "tests" / "fixtures" / "mentalchat16k_fa.sample.jsonl"
        return "fa", str(path)
    custom = st.text_input(
        "Custom dataset path",
        value=str(config.PERSIAN_DATASET_PATH),
        key=f"{key_prefix}_custom",
    )
    return "fa", custom.strip() or None


def render_limit_input(key: str = "limit", default: int = 50) -> int:
    return st.number_input(
        "Sample limit (0 = all)",
        min_value=0,
        max_value=50_000,
        value=default,
        step=50,
        key=key,
    )


def render_category_legend() -> None:
    titles = category_titles_bilingual()
    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            [(k, v) for k, v in titles.items()],
            columns=["Category", "Title (EN / FA)"],
        ),
        use_container_width=True,
        hide_index=True,
    )


def persian_dataset_ready(path: str | Path | None = None) -> bool:
    p = Path(path) if path else config.PERSIAN_DATASET_PATH
    return p.exists()
