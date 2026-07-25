# Model Architectures — Retention-Time-Prediction-With-KA-GNN

Repository: https://github.com/farukznb/Retention-Time-Prediction-With-KA-GNN

Five architectures are implemented, evaluated under two experimental
protocols on the full METLIN SMRT dataset (79,955 compounds after QC).
Throughout this document, the classical-ML residual corrector (XGBoost +
Bayesian Ridge) is referred to as **CMLM** (Classical Machine Learning
Model) rather than PGM — it defines no factor graph, no conditional
independence structure, and no message-passing inference over random
variables, so it does not meet the formal definition of a Probabilistic
Graphical Model (Koller & Friedman, 2009). Only its Bayesian Ridge
sub-component carries genuine probabilistic semantics.

---

## Shared inputs

Every architecture consumes some combination of three precomputed
molecular representations:

| Representation | Dimension | Source |
|---|---|---|
| Molecular graph | 23-dim atom node features | parsed from `.sdf` |
| ECFP4 fingerprint | 1024-bit | precomputed, `.txt` |
| Physicochemical descriptors | 32-dim | computed with RDKit (logP, TPSA, HBD/HBA, rotatable bonds, topological indices, etc.) |

**Atom node feature vector (23-dim), used by both the KA-GNN and MolGCN
families:**

| Feature | Dim |
|---|---|
| Atomic type one-hot (C, N, O, S, P, F, Cl, Br, I + other) | 10 |
| Hybridisation one-hot (SP, SP2, SP3, SP3D, SP3D2 + other) | 6 |
| Is aromatic | 1 |
| Formal charge | 1 |
| Implicit-H count (capped at 4) | 1 |
| Is in ring | 1 |
| Chirality one-hot (CW, CCW + other) | 3 |
| **Total** | **23** |

All edges are undirected (bonds represented in both directions in
`edge_index`). Retention time is normalised with `RobustScaler` fitted
on the training set only, and predictions are inverse-transformed before
any reported metric.

---

## 1. Standalone KA-GNN

Dual-stream network, no residual correction — the reference baseline for
both KA-GNN hybrids.

**Graph stream:** 5 KA-GNN layers (256-dim hidden, 4 attention heads),
where each layer replaces the fixed scalar weights of standard
message-passing with learnable edge functions:

```
phi(x) = a_0 + sum_{k=1}^{K} (a_k * cos(kx) + b_k * sin(kx))
```

**Fingerprint stream:** a separate KAN encoder projects the 1024-bit
ECFP4 fingerprint into the same latent space.

**Fusion:** the two streams are concatenated into a 512-dim
representation and decoded to a single scalar RT prediction.

**Result:** MedAE = 26.14 s, R² = 0.827, 55.3% within ±30 s. A 25.3%
improvement over the original SMRT DNN baseline (35 s).

---

## 2. Forward Hybrid: KA-GNN → CMLM

Two-stage residual learning. KA-GNN predicts first; CMLM corrects it.

**Stage 1 — KA-GNN backbone.**
Identical to the standalone KA-GNN above: trained end-to-end on the
molecular graph and ECFP4 fingerprint to predict RT directly,
$\hat{y}_{\text{KAGNN}}$.

**Stage 2 — CMLM residual corrector.**
Input feature vector: the KA-GNN's learned 256-dim embedding,
concatenated with the 32 RDKit descriptors (288-dim total). An ensemble
of **XGBoost** (captures non-linear residual structure via sequential
tree-based error correction) and **Bayesian Ridge** (provides a Gaussian
posterior over the residual via Automatic Relevance Determination, so
XGBoost's point corrections come with a calibrated uncertainty estimate)
is trained to predict the residual $r = y - \hat{y}_{\text{KAGNN}}$.

**Inference:**
```
y_hat_final = y_hat_KAGNN + r_hat_CMLM
```

**Result:** MedAE = 20.45 s, R² = 0.834, 65.1% within ±30 s — the best
absolute accuracy of all five architectures (21.7% improvement over
standalone KA-GNN). Adds only ~4 minutes of training time beyond the
KA-GNN stage.

**Known limitation (class-wise):** improves majority classes
(Organoheterocyclics +0.51 s, Other +9.62 s) but *degrades* Lipids
(−9.70 s) and Organic Acids & AA (−1.87 s) relative to standalone
KA-GNN — the CMLM stage fits noise when Stage-1 residuals for a class
are already small and unstructured.

---

## 3. Reverse Hybrid: CMLM → KA-GNN

Coarse-to-fine residual learning. CMLM predicts first; KA-GNN refines it.

**Stage 1 — CMLM baseline.**
The same XGBoost + Bayesian Ridge ensemble is trained directly on ECFP4
fingerprints + 32 descriptors, producing a physicochemical baseline
$\hat{y}_{\text{CMLM}}$ that captures global trends (hydrophobicity,
polarity) but has no access to graph topology.

**Stage 2 — KA-GNN residual learner.**
A KA-GNN (same dual-stream design) is trained not on absolute RT, but on
the residual $r = y - \hat{y}_{\text{CMLM}}$, using the molecular graph.
This lets it specialise in local structural corrections — substituent
effects, stereochemistry — that global descriptors cannot represent.

**Inference:**
```
y_hat_final = y_hat_CMLM + r_hat_KAGNN
```

**Result (single split):** MedAE = 22.19 s, R² = 0.820, 61.7% within
±30 s.

**Result (3-fold, class-stratified CV):** MedAE = 29.12 ± 1.29 s vs.
40.05 ± 0.71 s for the CMLM stage alone (per-fold t = 37.76–44.04,
p ≤ 5.77×10⁻³⁰⁴). This is the **only architecture that improves every
one of the seven chemical superclasses without exception**, including
the hardest ones: Carbohydrates (−93.3 s vs. CMLM baseline), Aliphatic
Organics (−48.2 s), Other (−35.6 s). The coarse-to-fine ordering always
hands the second stage large, structured residuals, regardless of class
— unlike the forward ordering, where residuals for easy classes (e.g.
Lipids) are already near zero and get treated as noise.

---

## 4. Forward GNN Hybrid: MolGCN → CMLM

Same forward residual strategy as Architecture 2, with the KA-GNN
backbone replaced by a standard Graph Convolutional Network (MolGCN),
to isolate the contribution of KAN expressivity from the residual
learning strategy itself.

**MolGCN architecture** (599,553 trainable parameters):

```
Graph stream : 3 x GCNConv(hidden=256) -> global mean pool -> 256-dim
FP stream    : LayerNorm(1024) -> Linear -> ReLU -> Linear -> 256-dim
Fusion       : concat(512) -> Linear(512->256) -> ReLU -> Linear(256->1)
```

Unlike KA-GNN's learnable edge functions, `GCNConv` uses fixed ReLU
activations — this is the controlled comparison point.

**Stage 1:** MolGCN trained directly on absolute RT from the graph +
fingerprint, $\hat{y}_{\text{GNN}}$.

**Stage 2:** Bayesian Ridge (CMLM) corrects the GNN's residuals using
ECFP4 + 32 descriptors (1056-dim input).

**Stage 3 (optional):** a weighted ensemble of the CMLM baseline, the
standalone GNN, and this forward hybrid, with weights learned on the
validation set: $w_{\text{CMLM}} = 0.197$, $w_{\text{GNN}} = 0.115$,
$w_{\text{Fwd}} = 0.689$.

**Inference:**
```
y_hat_final = y_hat_GNN + r_hat_CMLM
y_hat_ens   = w1*y_hat_CMLM + w2*y_hat_GNN + w3*y_hat_Fwd     (optional)
```

**Result:** MedAE = 28.24 s, R² = 0.813, 52.5% within ±30 s, trained in
~9 minutes (vs. ~197 min for the KA-GNN forward hybrid). The weighted
ensemble reaches MedAE = 28.77 s but is not significantly better than
the forward hybrid alone (Wilcoxon p = 0.055).

**Gap to KA-GNN forward hybrid:** 7.68 s higher MedAE (28.24 s vs.
20.45 s), attributed to KAN's learnable edge functions capturing
structure–retention patterns that fixed ReLU convolution misses — though
part of this gap may also reflect the KA-GNN family's richer node
featurization in some configurations, so it should be read as an upper
bound on KAN's specific contribution.

---

## 5. Reverse GNN Hybrid: CMLM → MolGCN

Same reverse residual strategy as Architecture 3, with MolGCN in place
of KA-GNN.

**Stage 1:** Bayesian Ridge (CMLM) on ECFP4 + 32 descriptors provides
the physicochemical baseline $\hat{y}_{\text{CMLM}}$ (MedAE ≈ 52 s
alone).

**Stage 2:** MolGCN (same dual-stream architecture as Architecture 4)
learns the residual $r = y - \hat{y}_{\text{CMLM}}$ from the molecular
graph and fingerprint — the graph topology lets it capture
substructure-level patterns behind the CMLM's systematic errors.

**Inference:**
```
y_hat_final = y_hat_CMLM + r_hat_GNN
```

**Result:** MedAE = 31.62 s, R² = 0.810, 47.9% within ±30 s, 73.7%
within ±60 s — a 39.4% improvement over the CMLM baseline alone
(Wilcoxon p ≈ 0), trained in ~7 minutes.

---

## Cross-architecture summary

| # | Architecture | MedAE (s) | R² | % ≤ 30 s | Train time |
|---|---|---|---|---|---|
| — | CMLM baseline (Bayesian Ridge only) | 52.15 | 0.716 | 31.0% | < 1 min |
| 1 | Standalone KA-GNN | 26.14 | 0.827 | 55.3% | ~193 min |
| 2 | **Forward Hybrid (KA-GNN → CMLM)** | **20.45** | **0.834** | **65.1%** | ~197 min |
| 3 | Reverse Hybrid (CMLM → KA-GNN) | 22.19 | 0.820 | 61.7% | ~226 min |
| — | Standalone MolGCN | 29.04 | 0.811 | 51.4% | ~9 min |
| 4 | Forward GNN Hybrid (MolGCN → CMLM) | 28.24 | 0.813 | 52.5% | ~9 min |
| 5 | Reverse GNN Hybrid (CMLM → MolGCN) | 31.62 | 0.810 | 47.9% | ~7 min |

**Two independent design axes for practitioners:**
- **Integration order** — forward for maximum global accuracy; reverse
  for class-universal robustness on chemically diverse data.
- **Backbone** — KA-GNN for maximum accuracy and better-calibrated
  uncertainty; MolGCN/GCNConv for rapid, resource-constrained deployment
  at roughly 20× lower training cost.
