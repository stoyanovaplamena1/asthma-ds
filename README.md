# asthma-ds

This project explores which factors are associated with a positive asthma diagnosis.  
The analysis uses the Kaggle dataset: [Asthma Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/asthma-disease-dataset)

## Goal
Explore which factors are associated with a positive asthma diagnosis using the Kaggle dataset.

## Structure
- `data/raw/`         — raw input files (not committed)
- `data/processed/`   — cleaned/feature tables (not committed)
- `notebooks/`        — EDA, cleaning experiments, feature work
- `src/asthma_ds/`    — reusable data prep & feature code
- `tests/`            — unit tests for src/ functions
- `reports/`          — exported figures/tables (ignored)
- `configs/`          — optional configs (paths, toggles)

## Reproducibility
- Python 3.10+ recommended  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
