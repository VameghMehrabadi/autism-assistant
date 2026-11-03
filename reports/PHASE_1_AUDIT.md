# Phase 1: Repository Audit - Autism Assistant

## 1. Directory Structure Map
- **root**:
  - `README.md` / `README_en.md`: Documentation.
  - `requirements.txt`: Project dependencies.
  - `main.py`: Entry point for pipeline.
  - `config.py`: Configuration settings.
  - `data_loader.py`: Data ingestion logic.
  - `facts.py`: Fact classification and prototypes.
  - `labeler.py`: Semantic labeling module.
  - `gap_analysis.py`: Gap and analysis reporting.
  - `compare_runs.py`: Cross-configuration comparison tool.
  - `translate_dataset.py` / `export_for_translation.py`: Data translation utilities.
  - `Dockerfile` / `docker-compose.yml`: Containerization configurations.
  - `launch_ui.bat` / `launch_ui.ps1`: UI launchers.
  - `pytest.ini`: Test configuration.
  - **directories**:
    - `.github/workflows`: CI/CD pipelines.
    - `.streamlit`: UI framework configuration.
    - `data/`: Dataset storage.
    - `outputs/`: Analysis reports and results.
    - `pages/`: Streamlit UI pages.
    - `prompts/`: Translation prompts.
    - `tests/`: Unit and integration tests.
    - `ui_lib/`: UI shared helpers.

## 2. Dependency Review
- Requirements include `datasets`, `huggingface-hub`, `sentence-transformers`, `torch`, `pandas`, `pytest`, `streamlit`.
- Environment successfully configured using Python 3.13 and updated `huggingface-hub` and `transformers` versions for compatibility.

## 3. Execution Pipeline Trace
1. **Data Load**: `data_loader.py` fetches/loads datasets.
2. **Fact Definition**: `facts.py` provides categorization logic.
3. **Labeling**: `labeler.py` uses semantic models (MiniLM, BGE, E5) to classify.
4. **Analysis**: `gap_analysis.py` computes metrics and distributions.
5. **Comparison**: `compare_runs.py` orchestrates multiple model evaluations.

## 4. Test Status
- `pytest` suite: 27/27 tests passed.

## 5. Identified Issues
- **Architectural**: High coupling between labeler and model implementations (handled by abstraction).
- **ML/Data**: Relying on external model caches in containers.
- **Security**: Hardcoded keys in `.env.example` template (standard practice).
