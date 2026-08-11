"""Cached loaders for Streamlit — keyed by model / language / dataset."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from data_loader import Sample, load_samples
from labeler import SemanticLabeler


@st.cache_resource(show_spinner="Loading embedding model…")
def get_labeler(model_key: str, fact_lang: str) -> SemanticLabeler:
    return SemanticLabeler(model_key=model_key, fact_lang=fact_lang)


@st.cache_data(show_spinner="Loading dataset samples…")
def get_samples(
    dataset: str,
    limit: int,
    dataset_path: str | None = None,
) -> list[Sample]:
    path = Path(dataset_path) if dataset_path else None
    return load_samples(limit=limit, dataset=dataset, path=path)


def clear_caches() -> None:
    get_labeler.clear()
    get_samples.clear()
