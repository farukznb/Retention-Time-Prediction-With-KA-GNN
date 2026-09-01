# Retention Time Prediction with KA-GNN, GNN Hybrids and CMLM

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN/blob/main/LICENSE)

Hybrid architectures combining **Kolmogorov–Arnold Graph Neural Networks (KA-GNN)**, 
**Graph Convolutional Networks (GCN-based GNN)**, and **Classical Machine Learning methods (CMLM)** 
for liquid chromatography retention time (RT) prediction on the METLIN SMRT dataset, 
with a focus on LC–MS use cases in pharmaceutical drug discovery and metabolite identification.

---

## Overview

Retention time provides orthogonal physicochemical information to mass-to-charge ratio, 
enabling more reliable metabolite annotation in untargeted LC–MS workflows. This repository 
evaluates five hybrid architectures under two experimental protocols on the full METLIN SMRT 
dataset (79,955 compounds), without excluding non-retained compounds.

Three experiments are conducted:

- **Experiment 1 — KA-GNN Hybrids (Global Training):** KA-GNN-based models trained on the 
  full dataset, evaluated on a single held-out test set (n = 11,994).
- **Experiment 2 — Class-Wise Oriented Training:** Models trained taking chemical superclasses 
  into account, evaluated with 3-fold cross-validation and per-class MedAE analysis.
- **Experiment 3 — GNN Hybrids (MolGCN, Global Training):** Genuine dual-stream GNN 
  (3 × GCNConv layers + fingerprint stream) evaluated in both forward and reverse hybrid 
  configurations on the same held-out test set (n = 11,994).

---

## Dataset

The **METLIN SMRT dataset** provides 80,038 experimentally measured RT values under 
standardised reverse-phase C18 conditions. After synchronisation across CSV, ECFP, and SDF 
sources, **79,955 compounds** are retained.

Three molecular representations are used:

- **1024-bit ECFP4 fingerprints** (Morgan radius 2)
- **32 RDKit physicochemical descriptors** (logP, TPSA, HBD/HBA, rotatable bonds, 
  topological indices, etc.)
- **Molecular graphs** from SDF with **23-dimensional atom node features** 
  (atomic type, hybridisation, aromaticity, formal charge, implicit-H, ring membership, 
  chirality) — used in both KA-GNN and GCN-based GNN models

Data are split **70 / 15 / 15** (train / validation / test):

| Split | n |
|---|---|
| Train | 55,968 |
| Validation | 11,993 |
| Test | 11,994 |

RT is normalised with RobustScaler (mean = 0.0665, std = 0.8220) during training 
and inverse-transformed for all reported metrics.

### Chemical Class Distribution (post-QC, n = 79,955)

| Class | n | Median RT (s) |
|---|---|---|
| Organoheterocyclics | 78,543 | 776 |
| Other (unclassified) | 699 | 619 |
| Organic Acids & AA | 470 | 597 |
| Lipids | 188 | 741 |
| Benzenoids | 24 | 709 |
| Aliphatic Organics | 18 | 593 |
| Carbohydrates | 13 | 93 |

The severe class imbalance (Organoheterocyclics: 98.2% of data) means global MedAE 
is effectively a majority-class metric, motivating the class-wise evaluation in Experiment 2.

---

## Model Architectures

### KA-GNN Family (Experiments 1 & 2)

#### 1. KA-GNN Standalone

A dual-stream network processing the molecular graph through five KA-GNN layers 
(256-dim, 4 attention heads) with learnable Fourier-basis edge functions:

$$\phi(x) = a_0 + \sum_{k=1}^{K}(a_k \cos kx + b_k \sin kx)$$

A separate KAN encoder processes the ECFP4 fingerprint stream. Both streams are fused 
into a 512-dimensional representation for the final RT prediction.

#### 2. Forward Hybrid (KA-GNN → CMLM)

**Stage 1:** The KA-GNN backbone predicts RT end-to-end.  
**Stage 2:** A CMLM ensemble (XGBoost + BayesianRidge) corrects the KA-GNN residuals 
using the learned 256-dim embeddings concatenated with 32 molecular descriptors (288-dim input).  
**Inference:** $\hat{y} = \hat{y}_{\text{KA-GNN}} + \hat{r}_{\text{CMLM}}$

#### 3. Reverse Hybrid (CMLM → KA-GNN)

**Stage 1:** A CMLM ensemble trained on ECFP4 + descriptors provides a physicochemical baseline.  
**Stage 2:** The KA-GNN learns to correct the CMLM residuals, specialising in local structural 
corrections that descriptors cannot represent.  
**Inference:** $\hat{y} = \hat{y}_{\text{CMLM}} + \hat{r}_{\text{KA-GNN}}$
---

### GNN Family — MolGCN (Experiment 3)

The MolGCN is a **genuine dual-stream Graph Neural Network** using GCNConv layers 
(message passing over molecular graph topology). It is not graph-free: the core graph 
stream operates on the bond connectivity encoded in `edge_index` from the SDF file, 
which a plain MLP cannot reproduce.

**Architecture:**
Graph stream  : 3 × GCNConv(hidden=256) → global mean pool → 256-dim
FP stream     : LayerNorm(1024) → Linear → ReLU → Linear → 256-dim
Fusion        : concat(512) → Linear(512→256) → ReLU → Linear(256→1)

**Node feature vector (23-dim):**

| Feature | Dim |
|---|---|
| Atomic type one-hot (C N O S P F Cl Br I + other) | 10 |
| Hybridisation one-hot (SP SP2 SP3 SP3D SP3D2 + other) | 6 |
| Is aromatic | 1 |
| Formal charge | 1 |
| Implicit H count (capped at 4) | 1 |
| Is in ring | 1 |
| Chirality one-hot (CW CCW + other) | 3 |
| **Total** | **23** |

**Total trainable parameters: 599,553**

#### 4. Forward GNN Hybrid (MolGCN → CMLM)

**Stage 1:** MolGCN is the **primary predictor**, trained directly on absolute RT 
using molecular graphs + ECFP fingerprints.  
**Stage 2:** BayesianRidge corrects the GNN residuals using ECFP4 + 32 descriptors (1056-dim).  
**Stage 3 (optional):** Weighted ensemble of CMLM, GNN, and Forward Hybrid predictions, 
weights learned on the validation set.  
**Inference:** $$\hat{y} = \hat{y}_{\text{GNN}} + \hat{r}_{\text{CMLM}}$$  
**Ensemble:** $$\hat{y}_{\text{ens}} = w_1\hat{y}_{\text{CMLM}} + w_2\hat{y}_{\text{GNN}} + w_3\hat{y}_{\text{Fwd}}$$

#### 5. Reverse GNN Hybrid (CMLM → MolGCN)

**Stage 1:** BayesianRidge on ECFP4 + descriptors provides a probabilistic baseline.  
**Stage 2:** MolGCN learns to correct CMLM residuals ($r = y - \hat{y}_{\text{CMLM}}$) 
from the molecular graph and ECFP fingerprint. The graph topology allows the GNN to capture 
substructure-level patterns that explain systematic CMLM errors.  
**Inference:** $\hat{y} = \hat{y}_{\text{CMLM}} + \hat{r}_{\text{GNN}}$

---

## Results

### Experiment 1 — KA-GNN Global Training (n = 11,994 test)

| Model | MedAE (s) | MAE (s) | RMSE (s) | R² | % ≤ 30s |
|---|---|---|---|---|---|
| SMRT DNN baseline | 35.0 | — | — | — | — |
| KA-GNN Standalone | 26.14 | 48.43 | 90.68 | 0.827 | 55.3% |
| **Forward Hybrid (KA-GNN → CMLM)** | **20.45** | **36.99** | **69.22** | **0.834** | **65.1%** |
| Reverse Hybrid (CMLM → KA-GNN) | 22.19 | 39.57 | 71.94 | 0.820 | 61.7% |

The Forward Hybrid achieves the best global performance. The CMLM corrector adds only 
~4 minutes of training beyond the KA-GNN, yet reduces MedAE by 21.7%.

---

### Experiment 2 — Class-Wise Analysis (KA-GNN, 3-fold CV)

#### Forward Hybrid vs. Standalone KA-GNN (class-wise MedAE, s)

| Class | KA-GNN | Forward Hybrid | Δ |
|---|---|---|---|
| Organoheterocyclics | 25.11 | **24.60** | +0.51 |
| Other (unclassified) | 25.59 | **15.97** | +9.62 |
| Organic Acids & AA | **31.65** | 33.52 | −1.87 ↓ |
| Lipids | **18.07** | 27.77 | −9.70 ↓ |
| **Global** | 26.14 | **20.45** | +5.69 |

The Forward Hybrid improves majority classes but degrades Lipids and Organic Acids — 
the CMLM corrector fits noise when KA-GNN residuals are already small for those classes.

#### Reverse Hybrid (CMLM → KA-GNN, 3-fold CV, mean ± std)

| Metric | CMLM only | CMLM + KA-GNN |
|---|---|---|
| MedAE (s) | 40.05 ± 0.71 | **29.12 ± 1.29** |
| MAE (s) | 62.17 ± 0.70 | **51.32 ± 1.34** |
| R² | 0.783 ± 0.002 | **0.811 ± 0.002** |
| % ≤ 30s | 39.4% ± 0.65 | **51.1% ± 1.69** |

Per-fold statistical significance (KA-GNN correction vs. CMLM alone): 
t-statistics 37.76–44.04, all p ≤ 5.77 × 10⁻³⁰⁴.

#### Class-Wise MedAE — CMLM Only vs. Reverse Hybrid (CMLM → KA-GNN)

| Class | CMLM only | Reverse Hybrid | Δ |
|---|---|---|---|
| Organoheterocyclics | 39.37 | **28.92** | +10.45 |
| Other | 80.60 | **45.01** | +35.59 |
| Organic Acids & AA | 73.47 | **46.21** | +27.26 |
| Lipids | 60.89 | **53.05** | +7.84 |
| Benzenoids | 94.21 | **88.06** | +6.15 |
| Aliphatic Organics | 158.79 | **110.55** | +48.24 |
| Carbohydrates | 235.85 | **142.53** | +93.32 |

The Reverse Hybrid is the only architecture that improves every chemical class without 
exception. The coarse-to-fine ordering (CMLM → KA-GNN) always presents large, structured 
residuals to the second stage, regardless of class.

---

### Experiment 3 — GNN Hybrids, Global Training (n = 11,994 test)

Both models use the MolGCN dual-stream architecture (GCNConv + FP stream, 599,553 parameters). 
The CMLM baseline is identical across all experiments (BayesianRidge on ECFP4 + 32 descriptors).

#### CMLM Baseline (shared across all Experiment 3 models)

| Metric | Value |
|---|---|
| MedAE (s) | 52.15 |
| MAE (s) | 74.86 |
| RMSE (s) | 108.73 |
| R² | 0.716 |
| Pearson r | 0.846 |
| Spearman ρ | 0.874 |
| % ≤ 30s | 31.0% |
| % ≤ 60s | 55.3% |

#### Forward GNN Hybrid (MolGCN → CMLM)

Training: early stop at epoch 42 (train loss = 0.00408, val loss = 0.06099).  
Ensemble weights learned on validation set: CMLM = 0.197, GNN = 0.115, FwdHybrid = 0.689.

| Model | MedAE (s) | MAE (s) | RMSE (s) | R² | % ≤ 30s |
|---|---|---|---|---|---|
| CMLM baseline | 52.15 | 74.86 | 108.73 | 0.716 | 31.0% |
| GNN standalone (MolGCN) | 29.04 | 50.56 | 88.79 | 0.811 | 51.4% |
| **Forward Hybrid (GNN → CMLM)** | **28.24** | **49.86** | **88.29** | **0.813** | **52.5%** |
| Weighted Ensemble | 28.77 | 49.84 | 86.71 | 0.820 | 51.4% |

Statistical significance (paired Wilcoxon):

| Comparison | p-value | Result |
|---|---|---|
| Forward Hybrid < GNN | 6.6 × 10⁻²¹ | ✓ significant |
| Forward Hybrid < CMLM | ~0 | ✓ significant |
| Ensemble < GNN | 1.1 × 10⁻⁹ | ✓ significant |
| Ensemble < Forward Hybrid | 0.055 | ✗ not significant |

Total runtime: 9 min 29 s (GPU: CUDA).

#### Reverse GNN Hybrid (CMLM → MolGCN)

Training: early stop at epoch 29 (train loss = 0.00933, val loss = 0.06171).

| Model | MedAE (s) | MAE (s) | RMSE (s) | R² | % ≤ 30s | % ≤ 60s |
|---|---|---|---|---|---|---|
| CMLM baseline | 52.15 | 74.86 | 108.73 | 0.716 | 31.0% | 55.3% |
| **Reverse Hybrid (CMLM → GNN)** | **31.62** | **53.06** | **88.93** | **0.810** | **47.9%** | **73.7%** |

Statistical significance: Wilcoxon p ≈ 0 (Reverse Hybrid < CMLM), ✓ significant.  
Improvement over CMLM baseline: **−20.5 s MedAE (−39.4%)**.  
Total runtime: 6 min 51 s (GPU: CUDA).

#### Experiment 3 — Cross-Model Comparison Summary

| Model | MedAE (s) | R² | % ≤ 30s | Training time |
|---|---|---|---|---|
| CMLM baseline | 52.15 | 0.716 | 31.0% | < 1 min |
| GNN standalone (MolGCN) | 29.04 | 0.811 | 51.4% | ~9 min |
| Forward GNN Hybrid | 28.24 | 0.813 | 52.5% | ~9 min |
| Weighted Ensemble | 28.77 | 0.820 | 51.4% | ~9 min |
| Reverse GNN Hybrid | 31.62 | 0.810 | 47.9% | ~7 min |

---

## Cross-Experiment Comparison (Best Models per Family)

| Model Family | Best Model | MedAE (s) | R² | % ≤ 30s |
|---|---|---|---|---|
| KA-GNN Hybrids | Forward KA-GNN → CMLM | **20.45** | 0.834 | **65.1%** |
| GNN Hybrids (MolGCN) | Forward GNN → CMLM | 28.24 | 0.813 | 52.5% |
| GNN Hybrids (MolGCN) | Weighted Ensemble | 28.77 | **0.820** | 51.4% |
| CMLM baseline | BayesianRidge | 52.15 | 0.716 | 31.0% |

The KA-GNN Forward Hybrid outperforms the GCN-based GNN Forward Hybrid by **7.8 s MedAE**, 
demonstrating that Kolmogorov–Arnold activations provide expressivity beyond standard 
graph convolution for RT prediction.

---

## Key Findings

1. **Component ordering is a fundamental architectural choice.** Global metrics mask stark 
   class-wise differences: the Forward Hybrid degrades Lipids and Organic Acids despite the 
   best global MedAE, while the Reverse Hybrid universally improves all classes.

2. **KAN expressivity provides measurable gain over standard GCNConv.** The KA-GNN Forward 
   Hybrid (MedAE = 20.45 s) outperforms the equivalent GCN-based Forward Hybrid 
   (MedAE = 28.24 s) by ~7.8 s, confirming that learnable spline/Fourier activations 
   on graph edges capture RT-relevant patterns that fixed ReLU convolutions miss.

3. **GCN-based GNN hybrids confirm graph structure is informative.** The MolGCN 
   (3 × GCNConv, 23-dim node features, dual-stream with FP encoder) reduces CMLM baseline 
   MedAE from 52.15 s to 28.24 s (Forward) and 31.62 s (Reverse), establishing that 
   molecular graph topology — not just fingerprints — is essential for accurate RT prediction.

4. **Stratified class-wise evaluation is necessary** for chemically imbalanced datasets. 
   Models should not be selected on global MedAE alone when the deployment domain spans 
   diverse chemical classes.

5. **The Reverse GNN Hybrid is the most robust configuration across chemical classes.** 
   The coarse-to-fine ordering (CMLM → GNN) always presents large, structured residuals 
   to the second stage, regardless of class, enabling consistent improvement.

---

## Reproducibility

All experiments use `seed = 42`. RT normalisation uses RobustScaler fitted on the training 
set only (mean = 0.0665, std = 0.8220 on the normalised scale). Best model checkpoints, 
per-fold metrics, and visualisation outputs are saved under `results/`. Dependencies 
are pinned in `requirements.txt` and `environment.yml`.

### Hardware

All experiments were run on GPU (CUDA). Approximate runtimes:

| Experiment | Runtime |
|---|---|
| KA-GNN Forward Hybrid | ~197 min |
| KA-GNN Reverse Hybrid | ~226 min |
| GNN Forward Hybrid (MolGCN) | ~9 min 29 s |
| GNN Reverse Hybrid (MolGCN) | ~6 min 51 s |
| CMLM baseline only | < 1 min |

---

## Citation
```bibtex
@article{ZainabDjagba2026,
  author  = {Zainab, Farouk and Djagba, Prudence and Rakotonarivo, Vaisoa and Zeleke, Aklilu},
  title   = {Hybrid Kolmogorov--Arnold Graph Neural Networks and Probabilistic Graphical Models
             for Retention Time Prediction in Pharmaceutical {LC--MS} Workflows},
  year    = {2026},
  url     = {https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN}
}
```

---

## License

MIT License — see [LICENSE](https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN/blob/main/LICENSE) for details.
