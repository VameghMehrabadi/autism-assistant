# Architecture Overview - Autism Assistant

## Design Principles
- Modular Python components for pipeline stages.
- Streamlit-based UI for accessibility and rapid prototyping.
- Containerized environment for reproducible execution.

## Core Modules
- **Data Layer**: `data_loader.py` handles input sanitization and loading.
- **Inference Layer**: `labeler.py` abstracts semantic embedding and label assignment.
- **Reporting Layer**: `gap_analysis.py` performs diagnostic assessment.
- **Orchestration Layer**: `main.py` and `compare_runs.py` control workflow.
- **Presentation Layer**: `ui.py` and `pages/` directory provide user interface.

## Interaction Flow
- User input/selection via Streamlit UI.
- UI triggers pipeline script.
- Pipeline reads from `data/`, writes to `outputs/`.
- UI reads outputs for visual dashboarding.
