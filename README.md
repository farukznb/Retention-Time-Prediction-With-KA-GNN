# Retention Time Prediction with KA-GNN and PGM

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Hybrid architectures combining **Kolmogorov–Arnold Graph Neural Networks (KA-GNN)** and **Probabilistic Graphical Models (PGM)** for liquid chromatography retention time (RT) prediction on the METLIN SMRT dataset, with a focus on LC–MS use cases in pharmaceutical drug discovery and metabolite identification.

---

## Overview

Retention time provides orthogonal physicochemical information to mass-to-charge ratio, enabling more reliable metabolite annotation in untargeted LC–MS workflows. This repository evaluates three hybrid architectures under two experimental protocols on the full METLIN SMRT dataset (79,955 compounds), without excluding non-retained compounds.

Three experiments are conducted:

- **Experiment 1 — Global Training:** Models trained on the full dataset without class differentiation, evaluated on a single held-out test set (n = 11,994).
- **Experiment 2 — Class-Wise Oriented Training:** Models trained taking chemical superclasses into account, evaluated with 3-fold cross-validation and per-class MedAE analysis.
- **Experiment 3 — Graph-Free MLP–PGM Ablation:** The KA-GNN backbone is replaced with a standard MLP operating on tabular features only, isolating the contribution of molecular graph structure and KAN expressivity to the performance gains observed in Experiments 1 and 2.

---

## Dataset

The **METLIN SMRT dataset** provides 80,038 experimentally measured RT values under standardised reverse-phase C18 conditions. Three molecular representations are used:

- **1024-bit ECFP4 fingerprints** (Morgan radius 2)
- **32 RDKit physicochemical descriptors** (logP, TPSA, HBD/HBA, rotatable bonds, topological indices, etc.)
- **Molecular graphs** with 9-dimensional atom features and 4-dimensional bond features

Data are split 70 / 15 / 15 (train / validation / test) stratified by chemical class.

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

The severe class imbalance (Organoheterocyclics: 98.2 % of data) means global MedAE is effectively a majority-class metric, motivating the class-wise evaluation in Experiment 2.

---

## Model Architectures

### 1. KA-GNN Standalone

A dual-stream network processing the molecular graph through five KA-GNN layers (256-dim, 4 attention heads) with learnable Fourier-basis edge functions:

$$\phi(x) = a_0 + \sum_{k=1}^{K}(a_k \cos kx + b_k \sin kx)$$

A separate KAN encoder processes the ECFP4 fingerprint stream. Both streams are fused into a 512-dimensional representation for the final RT prediction.

### 2. Forward Hybrid (KA-GNN → PGM)

**Stage 1:** The KA-GNN backbone predicts RT end-to-end.  
**Stage 2:** A PGM ensemble (XGBoost + Bayesian Ridge) corrects the KA-GNN residuals using the learned 256-dim embeddings concatenated with 32 molecular descriptors (288-dim input).  
**Inference:** $\hat{y} = \hat{y}_{\text{KA-GNN}} + \hat{r}_{\text{PGM}}$

### 3. Reverse Hybrid (PGM → KA-GNN)

**Stage 1:** A PGM ensemble trained on ECFP4 + descriptors provides a physicochemical baseline.  
**Stage 2:** The KA-GNN learns to correct the PGM residuals, specialising in local structural corrections that descriptors cannot represent.  
**Inference:** $\hat{y} = \hat{y}_{\text{PGM}} + \hat{r}_{\text{KA-GNN}}$

### 4. Graph-Free MLP Ablation (Experiment 3)

The KA-GNN is replaced with a standard 4-layer MLP (512→256→128→1, BatchNorm + ReLU + Dropout) operating on tabular features only (ECFP4 + descriptors), with **no molecular graph construction and no KAN activations**. This configuration — referred to as **GNN(MLP)** to signal its role as the neural component in a GNN–PGM pipeline — isolates the joint contribution of molecular graph structure and KAN expressivity to the performance gap observed in Experiments 1 and 2. Both forward (MLP → PGM) and reverse (PGM → MLP) integration orders are evaluated, along with a learned weighted ensemble of all three graph-free configurations.

---

## Results

### Experiment 1 — Global Training (n = 11,994 test)

| Model | MedAE (s) | MAE (s) | RMSE (s) | R² | % ≤ 30s |
|---|---|---|---|---|---|
| SMRT DNN baseline | 35.0 | — | — | — | — |
| KA-GNN Standalone | 26.14 | 48.43 | 90.68 | 0.827 | 55.3% |
| **Forward Hybrid (KA-GNN → PGM)** | **20.45** | **36.99** | **69.22** | **0.834** | **65.1%** |
| Reverse Hybrid (PGM → KA-GNN) | 22.19 | 39.57 | 71.94 | 0.820 | 61.7% |

The Forward Hybrid achieves the best global performance. The PGM corrector adds only 4 minutes of training beyond the KA-GNN, yet reduces MedAE by 21.7%.

### Experiment 2 — Class-Wise Analysis

#### Forward Hybrid vs. Standalone KA-GNN (class-wise MedAE, s)

| Class | KA-GNN | Forward Hybrid | Δ |
|---|---|---|---|
| Organoheterocyclics | 25.11 | **24.60** | +0.51 |
| Other (unclassified) | 25.59 | **15.97** | +9.62 |
| Organic Acids & AA | **31.65** | 33.52 | −1.87 ↓ |
| Lipids | **18.07** | 27.77 | −9.70 ↓ |
| **Global** | 26.14 | **20.45** | +5.69 |

The Forward Hybrid improves majority classes but **degrades Lipids and Organic Acids** — the PGM corrector fits noise when the KA-GNN residuals are already small for those classes.

#### Reverse Hybrid (PGM → KA-GNN, 3-fold CV, mean ± std)

| Metric | PGM only | PGM + KA-GNN |
|---|---|---|
| MedAE (s) | 40.05 ± 0.71 | **29.12 ± 1.29** |
| MAE (s) | 62.17 ± 0.70 | **51.32 ± 1.34** |
| R² | 0.783 ± 0.002 | **0.811 ± 0.002** |
| % ≤ 30s | 39.4% ± 0.65 | **51.1% ± 1.69** |

Per-fold statistical significance (KA-GNN correction vs. PGM alone): t-statistics 37.76–44.04, all p ≤ 5.77 × 10⁻³⁰⁴.

#### Class-Wise MedAE: PGM Only vs. Reverse Hybrid

| Class | PGM only | Reverse Hybrid | Δ |
|---|---|---|---|
| Organoheterocyclics | 39.37 | **28.92** | +10.45 |
| Other | 80.60 | **45.01** | +35.59 |
| Organic Acids & AA | 73.47 | **46.21** | +27.26 |
| Lipids | 60.89 | **53.05** | +7.84 |
| Benzenoids | 94.21 | **88.06** | +6.15 |
| Aliphatic Organics | 158.79 | **110.55** | +48.24 |
| Carbohydrates | 235.85 | **142.53** | +93.32 |

The Reverse Hybrid is the **only architecture that improves every chemical class without exception**. The coarse-to-fine ordering (PGM → KA-GNN) always presents large, structured residuals to the second stage, regardless of class.

### Experiment 3 — Graph-Free MLP–PGM Ablation: Isolating KAN Expressivity and Graph Structure

| Model | MedAE (s) | R² | % ≤ 30s |
|---|---|---|---|
| GNN(MLP) standalone | 26.57 | 0.829 | 54.6% |
| Forward GNN(MLP)–PGM | 27.14 | 0.827 | 54.1% |
| GNN(MLP)–PGM ensemble | 25.33 | 0.834 | 56.2% |
| **KA-GNN + PGM (Forward)** | **20.56** | 0.824 | **63.8%** |

The 4.8 s MedAE gap between the best graph-free configuration and the Forward KA-GNN hybrid is statistically robust (p < 10⁻⁴⁸, Cohen's d = 0.41), attributable specifically to molecular graph structure and KAN expressivity. The GNN(MLP) ensemble (< 2 min training) nonetheless provides a practical low-infrastructure option achieving 96% of best KA-GNN performance.

---

## Key Findings

1. **Component ordering is a fundamental architectural choice.** Global metrics mask stark class-wise differences: the Forward Hybrid degrades Lipids and Organic Acids despite the best global MedAE, while the Reverse Hybrid universally improves all classes.

2. **Molecular graph structure and KAN expressivity jointly contribute ~4.8 s of MedAE gain** over tabular descriptor-only approaches, confirmed by the graph-free MLP ablation (Experiment 3, p < 10⁻⁴⁸, Cohen's d = 0.41).

3. **Stratified class-wise evaluation is necessary** for chemically imbalanced datasets. Models should not be selected on global MedAE alone when the deployment domain spans diverse chemical classes.

4. **The graph-free MLP–PGM ensemble** (< 2 min training, no graph construction needed) achieves 96% of best KA-GNN hybrid performance and is recommended for resource-constrained deployment.

---

## Reproducibility

All experiments use `seed = 42`. Best model checkpoints, per-fold metrics, and visualisation outputs are saved under `results/`. Dependencies are pinned in `requirements.txt` and `environment.yml`.

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

MIT License — see [LICENSE](LICENSE) for details.
