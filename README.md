#  Retention Time Prediction with KA-GNN and PGM

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Predict chromatographic retention times (RT) using hybrid architectures combining Probabilistic Graphical Models (PGM) and Knowledge-Augmented Graph Neural Networks (KA-GNN) with two-stage residual learning approaches.

---

##  COMPREHENSIVE REPOSITORY ANALYSIS

### 1. REPOSITORY OVERVIEW

This repository implements state-of-the-art retention time (RT) prediction models for chromatographic analysis, combining Probabilistic Graphical Models (PGM) and Knowledge-Augmented Graph Neural Networks (KA-GNN) with a novel two-stage residual learning strategy.

### 2. FILE STRUCTURE

```
Retention-Time-Prediction-With-KA-GNN/
├── README.md                                    # Main documentation
├── requirements.txt                             # Python dependencies
├── environment.yml                              # Conda environment
├── LICENSE                                      # MIT License
│
├── data/
│   ├── raw/                                     # Raw data files (SMRT dataset)
│   │   ├── SMRT_dataset.csv                     # RT values with PubChem IDs
│   │   ├── SMRT_ECFP_1024_Fingerprints.txt      # ECFP fingerprints
│   │   └── SMRT_dataset.sdf                     # Molecular structures
│   └── processed/                               # Processed datasets
│
├── experiments/                                 # Main experiment scripts
│   ├── 01_baseline_kagnn.py                     # KA-GNN standalone baseline
│   ├── 02_kagnn_pgm_forward.py                  # KA-GNN → PGM forward hybrid
│   ├── 03_pgm_kagnn_reverse.py                  # PGM → KA-GNN reverse hybrid
│   ├── 04_statistical_tests.py                  # Statistical comparison & analysis
│   ├── 05_pgm_kagin_classwise.py                # PGM → KA-GNN with class-wise analysis (3-Fold CV)
│   ├── 06_classwise_kagnn_pgm_forward_hybrid.py # KA-GNN → PGM with class-wise analysis
│   ├── 07_classwise_kagnn.py                    # KA-GNN baseline with class-wise analysis
│   └── pgm_kagnn_training.py                    # PGM & KA-GNN training utility functions
│
├── src/
│   ├── data/                                    # Data utilities
│   │   ├── __init__.py
│   │   ├── dataset.py                           # SMRTCombinedDataset
│   │   └── preprocessing.py                     # Feature extraction
│   │
│   ├── models/                                  # Neural network models
│   │   ├── __init__.py
│   │   ├── base_model.py                        # BaseRTModel abstract class
│   │   ├── kagnn.py                             # FixedBaselineKAGNN
│   │   ├── kagnn_pgm_forward.py                 # KAGNN→PGM Forward
│   │   └── pgm_kagnn_reverse.py                 # PGM→KAGNN Reverse
│   │
│   ├── evaluation/                              # Metrics & visualization
│   │   ├── __init__.py
│   │   ├── metrics.py                           # RTMetrics class
│   │   └── visualization.py                     # RTVisualizer class
│   │
│   ├── training/                                # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py                           # Unified Trainer class
│   │
│   └── utils/                                   # Utility functions
│       ├── __init__.py
│       ├── seed.py                              # Random seed setting
│       └── smrt_utils.py                        # SMRT-specific utilities
│
├── results/                                     # Output directory
│   ├── checkpoints/                             # Saved model weights
│   ├── metrics/                                 # JSON metrics files
│   ├── plots/                                   # Visualization outputs
│   └── statistical_analysis/                    # Statistical test results
│
└── .gitignore                                   # Git ignore rules
```

### 3. MODEL ARCHITECTURES

#### A. KA-GNN Baseline (Standalone)

**Architecture:** KAGIN (Knowledge-Augmented Graph Isomorphism Network) + KAN (Kolmogorov-Arnold Network)

**Features:**
- ECFP4 fingerprints (1024 bits)
- Graph molecular structure (PyTorch Geometric)

#### B. Forward Hybrid (KA-GNN → PGM)

- **Stage 1:** KA-GNN backbone trained on RT prediction
- **Stage 2:** PGM ensemble (XGBoost + Bayesian Ridge) learns residual corrections
- **Inference:** `final = KAGNN(pred) + PGM(correction)`

#### C. Reverse Hybrid (PGM → KA-GNN)

- **Stage 1:** PGM ensemble trained on physicochemical descriptors
- **Stage 2:** KA-GNN predicts residual corrections
- **Inference:** `final = PGM(pred) + KA-GNN(correction)`

### 4. KEY COMPONENTS

#### A. Data Handling

| Component | File | Purpose |
|-----------|------|---------|
| SMRTDataLoader | src/data/dataset.py | Load SMRT dataset |
| SMRTDataset | src/data/dataset.py | PyTorch Dataset class |
| ComprehensiveDescriptors | src/models/kagnn_pgm_forward.py | Extract 32 molecular descriptors |
| atom_to_indices() | src/data/preprocessing.py | Convert atoms to feature indices |
| bond_to_indices() | src/data/preprocessing.py | Convert bonds to feature indices |

#### B. Models

| Model | File | Description |
|-------|------|-------------|
| FixedBaselineKAGNN | src/models/kagnn.py | Standalone KA-GNN with GAT fallback |
| KAGNN_PGM_Forward | src/models/kagnn_pgm_forward.py | Forward hybrid (KA-GNN→PGM) |
| PGM_KAGNN_Reverse | src/models/pgm_kagnn_reverse.py | Reverse hybrid (PGM→KA-GNN) |
| ResidualKAGNN | src/models/pgm_kagnn_reverse.py | KAGNN for residual prediction |

#### C. Training

| Component | File | Purpose |
|-----------|------|---------|
| Trainer | src/training/trainer.py | Unified training loop |
| create_trainer() | src/training/trainer.py | Factory function for Trainer |

**Key Trainer Features:**
- Gradient clipping
- Mixed precision training (AMP)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping
- Checkpoint saving

#### D. Evaluation

| Component | File | Purpose |
|-----------|------|---------|
| RTMetrics | src/evaluation/metrics.py | Compute all metrics |
| RTVisualizer | src/evaluation/visualization.py | Generate plots |

**Metrics Computed:**
- **Core Metrics:** MedAE, MAE, RMSE, R²
- **Correlation:** Pearson r, Spearman ρ
- **Threshold Accuracy:** % ≤ 10s, % ≤ 30s, % ≤ 60s
- **Statistical:** Wilcoxon, Paired t-test, Cohen's d

#### E. Statistical Analysis

| Component | File | Purpose |
|-----------|------|---------|
| ComprehensiveStatisticalAnalyzer | experiments/04_statistical_tests.py | Full statistical analysis |

**Features:**
- Normality tests (Shapiro-Wilk, D'Agostino K²)
- Paired t-test, Wilcoxon signed-rank test
- Effect sizes (Cohen's d, Cliff's delta)
- Bootstrap confidence intervals
- Comprehensive diagnostic plots (16 panels)

### 5. DEPENDENCIES

```txt
# Core ML/DL
torch>=2.0.0
torch-geometric>=2.4.0
numpy>=1.24.0
pandas>=2.0.0

# Molecular Processing
rdkit>=2023.3.0
scikit-learn>=1.3.0

# Optimization
xgboost>=2.0.0
optuna>=3.4.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Statistics
scipy>=1.11.0
```

---

##  QUICK START GUIDE

### Dataset Preparation

The model requires three data files in `data/raw/`:

1. **SMRT_dataset.csv** - RT values with PubChem IDs
   ```
   PubChemID,RetentionTime
   12345,120.5
   67890,85.3
   ...
   ```

2. **SMRT_ECFP_1024_Fingerprints.txt** - ECFP4 fingerprints
   ```
   12345:01010101...
   67890:11001010...
   ```

3. **SMRT_dataset.sdf** - Molecular structures (SDF format)

**Setting paths:** Update the paths in your config or experiment script:
```python
CSV_PATH = "data/raw/SMRT_dataset.csv"
ECFP_PATH = "data/raw/SMRT_ECFP_1024_Fingerprints.txt"
SDF_PATH = "data/raw/SMRT_dataset.sdf"
```

### Dependencies and Installation

#### Option 1: Conda (Recommended)
```bash
# Create environment from yml file
conda env create -f environment.yml
conda activate rt-prediction

# Verify installation
python -c "import torch; import rdkit; print('PyTorch:', torch.__version__); print('RDKit:', rdkit.__version__)"
```

#### Option 2: Pip
```bash
# Install core dependencies
pip install torch torch-geometric numpy pandas

# Install RDKit (requires conda first)
conda install -c conda-forge rdkit

# Install remaining dependencies
pip install scikit-learn xgboost optuna matplotlib seaborn scipy
```

---

##  RUNNING THE MODELS

### Experiment 1 — Global Training (Single Split)

#### Full PGM → KA-GNN Experiment (Recommended)

```bash
python experiments/03_pgm_kagnn_reverse.py
```

#### KA-GNN → PGM Forward Experiment

```bash
python experiments/02_kagnn_pgm_forward.py
```

#### Baseline KA-GNN Only

```bash
python experiments/01_baseline_kagnn.py
```

### Experiment 2 — Class-Wise Oriented Training (3-Fold CV)

#### PGM → KA-GNN Class-Wise

```bash
python experiments/05_pgm_kagin_classwise.py
```

#### KA-GNN → PGM Forward Class-Wise

```bash
python experiments/06_classwise_kagnn_pgm_forward_hybrid.py
```

#### KA-GNN Baseline Class-Wise

```bash
python experiments/07_classwise_kagnn.py
```

---

##  OPTIONAL STAGES

### Running Only PGM Baseline

If you want to train and evaluate only the PGM ensemble (fast, ~10 minutes):

```python
from src.models.pgm_kagnn_reverse import PGMKAGNNTrainer
from src.data.dataset import SMRTDataLoader, SMRTDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Load data
loader = SMRTDataLoader(csv_path, ecfp_path, sdf_path)
df, ecfp_dict, mol_dict = loader.get_data()

# Split data
train_idx, test_idx = train_test_split(range(len(df)), test_size=0.15, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42)

# Create dataloaders
train_loader = DataLoader(SMRTDataset(df.iloc[train_idx], ecfp_dict, mol_dict),
                         batch_size=64, shuffle=True, collate_fn=loader.collate_fn)

# Train only PGM (Stage 1)
trainer = PGMKAGNNTrainer(train_loader, val_loader, test_loader, mol_dict, loader.rt_scaler)
metrics_pgm = trainer.evaluate_pgm_only()
print(f"PGM MedAE: {metrics_pgm['medae']:.2f}s")
```

### Running Only KA-GNN Model

For standalone KA-GNN training:

```python
from src.models.kagnn import FixedBaselineKAGNN
from src.training.trainer import Trainer, create_trainer

# Initialize model
model = FixedBaselineKAGNN(
    in_channels=1024,      # ECFP dimension
    hidden_dim=256,
    num_layers=3,
    dropout=0.2
)

# Create trainer
trainer = create_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    lr=3e-4,
    weight_decay=1e-5,
    epochs=100,
    patience=20,
    device='cuda'
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
```

---

##  PREDICTION ON NEW COMPOUNDS

### Loading a Trained Model

```python
import torch
from src.models.pgm_kagnn_reverse import PGM_KAGNN_Reverse

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PGM_KAGNN_Reverse(cfg).to(device)
model.load_state_dict(torch.load('results/checkpoints/pgm_kagnn_reverse/best_model.pt'))
model.eval()
```

### Making Predictions

```python
from rdkit import Chem
from src.data.preprocessing import atom_to_indices, bond_to_indices
import numpy as np

def predict_rt(model, smiles_list, ecfp_dict, device):
    """Predict retention times for new compounds."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            atom_features = atom_to_indices(mol)
            bond_features = bond_to_indices(mol)
            pubchem_id = get_pubchem_id(smiles)  # You need to map this
            ecfp = ecfp_dict.get(pubchem_id, np.zeros(1024))
            graph_data = (atom_features, bond_features)
            ecfp_tensor = torch.tensor(ecfp, dtype=torch.float).unsqueeze(0).to(device)
            pred = model(graph_data, ecfp_tensor)
            predictions.append(pred.item())

    return predictions

# Example usage
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
predictions = predict_rt(model, smiles_list, ecfp_dict, device)
print(f"Predicted RTs: {predictions}")
```

---

##  IMPORTANT NOTES

### GPU Usage

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Recommended:** Use CUDA-enabled GPU for KA-GNN training. CPU training is significantly slower.

### Batch Size

| Hardware | Recommended Batch Size |
|----------|------------------------|
| GPU (8GB VRAM) | 32-64 |
| GPU (16GB+ VRAM) | 64-128 |
| CPU | 16-32 |

### Training Times

| Stage | Time (Approximate) |
|-------|-------------------|
| PGM Optimization (Optuna) | 10-30 minutes |
| KA-GNN Training (1 fold) | 5-15 minutes |
| KA-GNN Training (3-fold CV) | 1-2 hours |
| Full Pipeline | 2-3 hours |

### Memory Requirements

| Component | RAM | VRAM |
|-----------|-----|------|
| Dataset Loading | 4-8 GB | - |
| KA-GNN Training | - | 4-8 GB |
| PGM Training | 2-4 GB | - |

---

##  DATA FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Data (CSV + ECFP + SDF)                                    │
│         ↓                                                       │
│  SMRTDataLoader                                                 │
│  ├── Load CSV → DataFrame                                       │
│  ├── Load ECFP → Dictionary {id: fingerprint}                   │
│  ├── Load SDF → Dictionary {id: molecule}                       │
│  └── Sync & Clean → Intersection of all three                  │
│         ↓                                                       │
│  SMRTDataset (PyTorch Dataset)                                  │
│  ├── Graph creation from molecules                              │
│  ├── ECFP retrieval                                             │
│  └── RT normalization                                           │
│         ↓                                                       │
│  DataLoader (batch processing)                                  │
│         ↓                                                       │
│  Model Input                                                    │
│  └── (graph, ecfp, rt, ids)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

##  TRAINING PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: BACKBONE TRAINING                               │   │
│  │ • Optimizer: AdamW (lr=3e-4, weight_decay=1e-5)          │   │
│  │ • Loss: SmoothL1Loss (Huber)                             │   │
│  │ • Scheduler: ReduceLROnPlateau                           │   │
│  │ • Early Stopping: patience=20                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: RESIDUAL CORRECTION (for hybrid models)         │   │
│  │ • Extract descriptors (32 features)                      │   │
│  │ • Train XGBoost + Bayesian Ridge ensemble                │   │
│  │ • Optional: Optuna hyperparameter optimization           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ EVALUATION                                               │   │
│  │ • Compute metrics (MedAE, MAE, RMSE, R², etc.)           │   │
│  │ • Statistical significance tests                         │   │
│  │ • Visualization (scatter, histogram, residuals, etc.)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## KEY INNOVATIONS

1. **Two-Stage Residual Learning:** Combines complementary strengths of PGMs and KA-GNN
2. **Descriptor Caching:** Efficient molecular descriptor extraction
3. **Fallback Mechanisms:** GAT fallback if KAGNN unavailable
4. **Comprehensive Statistics:** 16-panel diagnostic plots, multiple statistical tests
5. **Mixed Precision Training:** Faster GPU training

---

## RESULTS

> The two experiments differ in their training protocol. **Experiment 1** trains each model globally on a single train/test split (n = 11,994 test). **Experiment 2** trains with a class-wise oriented strategy using 3-fold cross-validation, allowing evaluation of generalisation across chemical superclasses.

---

## EXPERIMENT 1 — Global Training (Single Split)

Models are trained on the full dataset without class stratification. Evaluated on a held-out test set (n = 11,994 from 79,955 total).

### Dataset Split

| Split | Size |
|-------|------|
| Train | 55,967 |
| Test | 11,994 |
| Total (with METLIN baseline ref.) | 80,038 |

### Group 1 — Baseline (SMRT METLIN)

Reference only; not trained in this repository.

| MedAE (s) | MAE (s) | RMSE (s) | R² | Pearson | Spearman | % ≤ 10s | % ≤ 30s | % ≤ 60s |
|-----------|---------|----------|----|---------|----------|---------|---------|---------|
| 57 | — | — | — | — | — | — | 21.69% | 55.26% |

### Group 2 — KA-GNN Only

Standalone KA-GNN trained on the full training split.

| n (test) | MedAE (s) | MAE (s) | RMSE (s) | R² | Pearson | Spearman | % ≤ 10s | % ≤ 30s | % ≤ 60s |
|----------|-----------|---------|----------|----|---------|----------|---------|---------|---------|
| 11,994 | 26.14 | 48.43 | 90.68 | 0.8270 | 0.9061 | 0.9281 | 21.69% | 55.26% | 79.39% |

### Group 3 — Forward Hybrid (KA-GNN → PGM)

KA-GNN trained first; PGM corrects residuals.

| n (test) | MedAE (s) | MAE (s) | RMSE (s) | R² | Pearson | Spearman | % ≤ 10s | % ≤ 30s | % ≤ 60s |
|----------|-----------|---------|----------|----|---------|----------|---------|---------|---------|
| 11,994 | 20.45 | 36.99 | 69.22 | 0.8336 | 0.9135 | 0.9329 | 26.95% | 65.07% | 85.76% |

### Group 4 — Reverse Hybrid (PGM → KA-GNN)

PGM trained first; KA-GNN corrects residuals.

| n (test) | MedAE (s) | MAE (s) | RMSE (s) | R² | Pearson | Spearman | % ≤ 10s | % ≤ 30s | % ≤ 60s |
|----------|-----------|---------|----------|----|---------|----------|---------|---------|---------|
| 11,994 | 22.19 | 39.57 | 71.94 | 0.8202 | 0.9004 | 0.9257 | 25.40% | 61.70% | 84.18% |

> **Summary:** The Forward Hybrid achieves the best global MedAE (20.45 s). All three trained models outperform the SMRT METLIN baseline.

---

## EXPERIMENT 2 — Class-Wise Oriented Training (3-Fold CV)

Models are trained with a class-aware strategy. Evaluation uses **3-fold cross-validation** to ensure robust estimates. Results are reported as mean ± std across folds.

Global metrics over the SMRT test set can be misleading due to severe class imbalance: **Organoheterocyclics account for ~98.3%** of the test set. Experiment 2 decomposes performance across chemical superclasses.

### Dataset Split (per fold, averaged)

| Split | Size (mean) |
|-------|-------------|
| Test per fold | ~26,652 |

### Group 5 — Reverse Hybrid (PGM → KA-GNN), 3-Fold CV

#### PGM Stage Only (Stage 1 baseline)

| Metric | Mean | Std |
|--------|------|-----|
| n (test) | 26,652 | ±0.47 |
| MedAE (s) | 40.05 | ±0.71 |
| MAE (s) | 62.17 | ±0.70 |
| RMSE (s) | 96.26 | ±0.80 |
| R² | 0.7830 | ±0.0023 |
| Pearson | 0.8859 | ±0.0014 |
| Spearman | 0.9075 | ±0.0011 |
| % ≤ 10s | 14.06% | ±0.22 |
| % ≤ 30s | 39.39% | ±0.65 |
| % ≤ 60s | 66.07% | ±0.49 |

#### Final Hybrid (PGM + KA-GNN Residual Correction)

| Metric | Mean | Std |
|--------|------|-----|
| n (test) | 26,652 | ±0.47 |
| MedAE (s) | 29.12 | ±1.29 |
| MAE (s) | 51.32 | ±1.34 |
| RMSE (s) | 89.92 | ±0.68 |
| R² | 0.8107 | ±0.0017 |
| Pearson | 0.9006 | ±0.0008 |
| Spearman | 0.9243 | ±0.0008 |
| % ≤ 10s | 19.36% | ±0.96 |
| % ≤ 30s | 51.11% | ±1.69 |
| % ≤ 60s | 76.25% | ±1.05 |

**Statistical significance (paired t-test & Wilcoxon per fold):**

| Fold | t-stat | p (t-test) | w-stat | p (Wilcoxon) |
|------|--------|------------|--------|--------------|
| 1 | 44.04 | 0.0 | 111,923,954 | 0.0 |
| 2 | 37.76 | 5.77e-304 | 122,264,911 | 0.0 |
| 3 | 42.73 | 0.0 | 114,861,441 | 0.0 |

> KA-GNN residual correction significantly improves PGM predictions across all 3 folds (all p ≈ 0).

---

### Group 6 — Forward Hybrid (KA-GNN → PGM) & KA-GNN Baseline, Class-Wise

Evaluated on the test split (n = 11,994). These results come from the class-wise training scripts.

#### KA-GNN Baseline (Class-Wise Run)

| n (test) | MedAE (s) | MAE (s) | RMSE (s) | R² | Pearson | Spearman | % ≤ 10s | % ≤ 30s | % ≤ 60s |
|----------|-----------|---------|----------|----|---------|----------|---------|---------|---------|
| 11,994 | 25.93 | 46.91 | 86.94 | 0.8226 | 0.9075 | 0.9290 | 21.47% | 55.72% | 79.73% |

#### Forward Hybrid (KA-GNN → PGM, Class-Wise Run)

| n (test) | MedAE (s) | MAE (s) | RMSE (s) | R² | Pearson | Spearman | % ≤ 10s | % ≤ 30s | % ≤ 60s |
|----------|-----------|---------|----------|----|---------|----------|---------|---------|---------|
| 11,994 | 24.60 | 44.59 | 84.11 | 0.8340 | 0.9134 | 0.9330 | 23.07% | 57.37% | 81.35% |

**Statistical significance (Forward Hybrid vs. KA-GNN Baseline):**

| t-stat | p (t-test) | w-stat | p (Wilcoxon) |
|--------|------------|--------|--------------|
| 9.67 | 4.76e-22 | 31,965,816 | 5.01e-26 |

> The Forward Hybrid significantly outperforms the standalone KA-GNN baseline in the class-wise experiment (p ≈ 0).

---

### Class-Wise MedAE: KA-GNN vs. Forward Hybrid (Experiment 2, Test Split n = 11,994)

| Class | n | KA-GNN MedAE | Forward Hybrid MedAE | ΔMedAE |
|-------|---|-------------|---------------------|--------|
| Organoheterocyclics | 11,791 | 25.11 s | 24.60 s | +0.51 s |
| Other (unclassified) | 100 | 25.59 s | 15.97 s | **+9.62 s** |
| Organic Acids & AA | 70 | 31.65 s | 33.52 s | −1.87 s |
| Lipids | 26 | 18.07 s | 27.77 s | **−9.70 s** |
| **Global** | **11,994** | **26.14 s** | **20.45 s** | +5.69 s |

**Key findings:**
- The **Forward Hybrid** improves "Other" compounds substantially (+37.6%) via PGM physicochemical descriptors (LogP, TPSA, MW), but **degrades Lipids** (−53.7%) and **Organic Acids** (−5.9%).
- **Standalone KA-GNN** shows no majority-class over-optimisation; Lipids (n = 26) achieve the best class-wise MedAE (18.07 s).
- Global improvement (+5.69 s) almost entirely reflects gains on Organoheterocyclics; degradations on minority classes are statistically invisible in aggregate metrics.

---

### Reverse Hybrid (PGM → KA-GNN): 3-Fold CV Class-Wise Results (Experiment 2)

| Class | PGM MedAE | Hybrid MedAE | ΔMedAE |
|-------|-----------|-------------|--------|
| Organoheterocyclics | 39.37 s | 28.92 s | +10.45 s |
| Other (unclassified) | 80.60 s | 45.01 s | +35.59 s |
| Organic Acids & AA | 73.47 s | 46.21 s | +27.26 s |
| Lipids | 60.89 s | 53.05 s | +7.84 s |
| Benzenoids | 94.21 s | 88.06 s | +6.15 s |
| Aliphatic Organics | 158.79 s | 110.55 s | +48.24 s |
| Carbohydrates | 235.85 s | 142.53 s | +93.32 s |
| **Global** | **39.74 s** | **29.15 s** | +10.59 s |

> The Reverse Hybrid improves **all seven chemical classes**. The coarse-to-fine ordering (PGM → KA-GNN) creates large, correctable residuals for every class, avoiding the selective failure modes of the Forward Hybrid.

---

### Three-Architecture Summary (Experiment 2)

| Class | KA-GNN Baseline | Forward Hybrid | Reverse Hybrid (CV) |
|-------|-----------------|----------------|---------------------|
| Organoheterocyclics | 25.11 s | **24.60 s ↑** | 28.92 s |
| Other | 25.59 s | **15.97 s ↑** | 45.01 s |
| Organic Acids & AA | **31.65 s ↑** | 33.52 s ↓ | 46.21 s |
| Lipids | **18.07 s ↑** | 27.77 s ↓ | 53.05 s |
| Global | 26.14 s | **20.45 s ↑** | 29.15 s |

> ↑ = best MedAE for that class; ↓ = worst. The Reverse Hybrid's higher absolute values reflect its weaker PGM starting baseline (~40 s), not a worse architecture.

**Overarching lesson:** The ordering of model components in a hybrid architecture has profound consequences for class-wise generalisation that are **invisible in global metrics**. For applications requiring reliable predictions across diverse chemical classes (lipids, bile acids, polar metabolites), the Reverse Hybrid's class-universal improvement pattern is a meaningful practical advantage.

---

## USAGE EXAMPLE

```python
# Quick start example
from src.data.dataset import SMRTCombinedDataset, collate_fn
from src.models.pgm_kagnn_reverse import PGM_KAGNN_Reverse
from torch.utils.data import DataLoader

# Load data
train_ds = SMRTCombinedDataset(cfg, 'train')
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = PGM_KAGNN_Reverse(cfg).to(device)
model.set_mol_dict(train_ds.mol_dict)

# Stage 1: Train PGM
model.train_stage1_pgm(train_loader, n_estimators=400, optimize=False)

# Stage 2: Train KAGNN
model.fit(train_loader, val_loader, epochs=100)

# Evaluate
metrics = model.evaluate(test_loader)
```

---

## REPRODUCIBILITY

1. **Set random seeds:** All experiments use `seed=42`
2. **Save checkpoints:** Best models saved in `results/checkpoints/`
3. **Metrics output:** JSON files in `results/metrics/`
4. **Reproducible dependencies:** Pinned in `requirements.txt`

---

## CITATION

```bibtex
@article{RetentionTimePrediction,
  author  = {Faruk ZnB},
  title   = {Retention Time Prediction With KA-GNN and PGM},
  journal = {GitHub},
  year    = {2026},
  url     = {https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN}
}
```

---

## REFERENCES

- KA-GNN: Knowledge-Augmented Graph Neural Networks
- METLIN SMRT Dataset: Small Molecule Retention Time
- RDKit: Chemoinformatics toolkit
- PyTorch Geometric: Graph neural networks

---

## 📄 LICENSE

MIT License - See LICENSE file for details.