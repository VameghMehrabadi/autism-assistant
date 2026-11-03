# Development Roadmap - Autism Assistant

## Current Status
- Pipeline audit complete.
- Test suite passing (27 tests).
- Modular architecture confirmed.

## Short-term Objectives
1. **Model Optimization**: Evaluate performance benchmarks for `bge-m3` vs `e5-large`.
2. **Data Pipeline**: Refine Persian translation workflows to improve classification accuracy for local languages.
3. **UI Enhancements**: Add custom dashboard features for longitudinal monitoring of mental chat sessions.

## Long-term Objectives
1. **Model Fine-tuning**: Investigate fine-tuning embedding models on the annotated MentalChat16K dataset.
2. **Advanced Analytics**: Add causal analysis between interventions and linguistic delays.
3. **Deployment**: Simplify Docker deployment for clinical settings.

## Known Limitations
- Heavy dependence on external Hugging Face model downloads during initial setup.
- Translation accuracy relies on GPT-4+ API availability.
