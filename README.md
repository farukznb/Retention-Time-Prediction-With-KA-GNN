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
│   ├── 01_baseline_kagnn.py                     # KA-GNN standalone model
│   ├── 02_kagnn_pgm_forward.py                  # KA-GNN → PGM hybrid
│   ├── 03_pgm_kagnn_reverse.py                  # PGM → KA-GNN hybrid
│   └── 04_statistical_tests.py                  # Statistical comparison
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
- Performance: MedAE ~27s, R² ~0.817

#### B. Forward Hybrid (KA-GNN → PGM)

- **Stage 1:** KA-GNN backbone trained on RT prediction
- **Stage 2:** PGM ensemble (XGBoost + Bayesian Ridge) learns residual corrections
- **Inference:** `final = KAGNN(pred) + PGM(correction)`
- **Performance:** MedAE ~20.56s, MAE ~39.57s

#### C. Reverse Hybrid (PGM → KA-GNN)

- **Stage 1:** PGM ensemble trained on physicochemical descriptors
- **Stage 2:** KA-GNN predicts residual corrections
- **Inference:** `final = PGM(pred) + KA-GNN(correction)`
- **Performance:** MedAE ~22.19s, MAE ~39.57s

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
| RTMetrics | src/evaluation/metrics.py | Compute all  metrics |
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

### Full PGM → KA-GNN Experiment (Recommended)

This runs the complete reverse hybrid pipeline with both PGM baseline and KA-GNN residual learning:

```bash
python experiments/03_pgm_kagnn_reverse.py
```

**What happens:**
1. Loads and preprocesses the SMRT dataset
2. Trains PGM ensemble (XGBoost + Bayesian Ridge) as Stage 1
3. Trains KA-GNN for residual correction as Stage 2
4. Evaluates on test set with comprehensive metrics
5. Generates visualizations and saves results

### KA-GNN → PGM Forward Experiment

```bash
python experiments/02_kagnn_pgm_forward.py
```

**What happens:**
1. Trains KA-GNN backbone first
2. PGM corrects the residuals from KA-GNN
3. Best for capturing global trends first

### Baseline KA-GNN Only

```bash
python experiments/01_baseline_kagnn.py
```

Runs standalone KA-GNN without any hybrid correction.

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
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            
            # Create graph representation
            atom_features = atom_to_indices(mol)
            bond_features = bond_to_indices(mol)
            
            # Get ECFP fingerprint
            pubchem_id = get_pubchem_id(smiles)  # You need to map this
            ecfp = ecfp_dict.get(pubchem_id, np.zeros(1024))
            
            # Prepare input
            graph_data = (atom_features, bond_features)
            ecfp_tensor = torch.tensor(ecfp, dtype=torch.float).unsqueeze(0).to(device)
            
            # Predict
            pred = model(graph_data, ecfp_tensor)
            predictions.append(pred.item())
    
    return predictions

# Example usage
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
predictions = predict_rt(model, smiles_list, ecfp_dict, device)
print(f"Predicted RTs: {predictions}")
```

### Using Pre-computed Descriptors

```python
from src.models.pgm_kagnn_reverse import PGMEnsemble

# Initialize PGM ensemble
pgm = PGMEnsemble()
pgm.load('results/checkpoints/pgm_ensemble.pkl')

# Predict with molecular descriptors
descriptors = np.array([[32.0, 1.5, 45.0, ...]])  # 32 features
rt_prediction = pgm.predict(descriptors)
```

---

##  IMPORTANT NOTES

### GPU Usage

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Recommended:** Use CUDA-enabled GPU for KA-GNN training. CPU training is significantly slower.

### Batch Size

| Hardware | Recommended Batch Size |
|----------|------------------------|
| GPU (8GB VRAM) | 32-64 |
| GPU (16GB+ VRAM) | 64-128 |
| CPU | 16-32 |

```python
# In your config
BATCH_SIZE = 64  # Adjust based on your GPU memory
```

### Training Times

| Stage | Time (Approximate) |
|-------|-------------------|
| PGM Optimization (Optuna) | 10-30 minutes |
| KA-GNN Training (1 fold) | 5-15 minutes |
| KA-GNN Training (3-fold CV) | 1-2 hours |
| Full Pipeline | 2-3 hours |

### Early Stopping

The models use early stopping to prevent overfitting:

```python
# Configuration
EARLY_STOPPING_PATIENCE = 20  # Number of epochs without improvement
MIN_DELTA = 0.001  # Minimum improvement threshold

# Monitor validation loss
if val_loss < best_loss - MIN_DELTA:
    best_loss = val_loss
    patience_counter = 0
    save_checkpoint()
else:
    patience_counter += 1
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered!")
        break
```

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
│  │ • Compute  metrics (MedAE, MAE, RMSE, R², etc.)        │   │
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

## PERFORMANCE SUMMARY

| Model | MedAE (s) | MAE (s) | RMSE (s) | R² | % ≤ 30s |
|-------|-----------|---------|----------|-----|---------|
| KA-GNN Only | 27.23 | 48.89 | 88.43 | 0.817 | 53.78% |
| Forward Hybrid | 20.56 | 39.57 | 71.94 | 0.824 | 65.07% |
| Reverse Hybrid | 22.19 | 39.57 | 71.94 | 0.820 | 61.70% |

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

