# Retention-Time-Prediction-With-KA-GNN-and-PGM


[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Predict chromatographic retention times (RT) using state-of-the-art KA-GNN + PGM models using the METLIN SMRT dataset.

## Novel Contributions
1. Developed novel residual hybrid models combining KAGNN→PGM and PGM→KAGNN.
2. Comprehensive evaluation pipeline with statistical tests and custom RT metrics.
3. Achieved state-of-the-art performance and reproducibility on METLIN SMRT dataset.

## Installation
### Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate rt-prediction
```

### Pip
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# Run baseline KA-GNN experiment
python experiments/01_baseline_kagnn.py

# Generate plots and metrics
python src/evaluation/visualization.py
```

## Results
| Model                | MedAE (s) | MAE (s) | RMSE (s) | R²   | % ≤ 30s |
|----------------------|-----------|---------|----------|-------|---------|
| KAGNN Baseline       |           |         |          |       |         |
| KAGNN → PGM (Forward)|           |         |          |       |         |
| PGM → KAGNN (Reverse)|           |         |          |       |         |

## Usage Example
Refer to the [notebooks/](notebooks/) directory for example workflows.

## Citation
```bibtex
@article{RetentionTimePrediction,
  author  = {Faruk ZnB},
  title   = {Retention Time Prediction With KA-GNN},
  journal = {GitHub},
  year    = {2026},
  url     = {https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN}
}
```

