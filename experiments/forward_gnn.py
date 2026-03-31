
"""
Forward Hybrid Pipeline: GNN (MolGCN) → PGM (BayesianRidge) residual corrector.

This script trains a graph neural network (MolGCN) on absolute retention times,
then uses a BayesianRidge model on tabular features (ECFP + RDKit descriptors)
to correct the GNN residuals. A weighted ensemble of PGM, GNN, and the hybrid
is also evaluated.

Usage:
    python forward_hybrid_gnn.py --csv <csv_path> --ecfp <ecfp_path> --sdf <sdf_path>
    [--epochs 150] [--batch-size 64] [--hidden-dim 256] [--device cuda]
"""

import argparse
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, wilcoxon

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


# =============================================================================
# 0)  CONFIGURATION (defaults, can be overridden by command line)
# =============================================================================
DEFAULT_CSV_PATH  = "data/SMRT_dataset.csv"
DEFAULT_ECFP_PATH = "data/SMRT_ECFP_1024_Fingerprints.txt"
DEFAULT_SDF_PATH  = "data/SMRT_dataset.sdf"

SEED         = 42
GNN_EPOCHS   = 150
PATIENCE     = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 64
HIDDEN_DIM   = 256
DROPOUT      = 0.1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 1)  MOLECULAR GRAPH CONSTRUCTION
# =============================================================================
ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]   # 9 → +1 other = 10
HYBRID_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]                                                                 # 5 → +1 other = 6
CHIRAL_TYPES = [
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]                                                                 # 2 → +1 other = 3


def one_hot(value, vocab):
    vec = [0] * (len(vocab) + 1)
    if value in vocab:
        vec[vocab.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom):
    """
    23-dimensional node feature vector:
      10 atomic-type one-hot
       6 hybridisation one-hot
       1 is aromatic
       1 formal charge
       1 implicit H count (capped at 4)
       1 is in ring
       3 chirality one-hot
    """
    feat  = one_hot(atom.GetSymbol(),         ATOM_TYPES)    # 10
    feat += one_hot(atom.GetHybridization(),  HYBRID_TYPES)  # 6
    feat += [int(atom.GetIsAromatic())]                      # 1
    feat += [float(atom.GetFormalCharge())]                  # 1
    feat += [float(min(atom.GetTotalNumHs(), 4))]            # 1
    feat += [int(atom.IsInRing())]                           # 1
    feat += one_hot(atom.GetChiralTag(), CHIRAL_TYPES)       # 3
    assert len(feat) == 23
    return feat


def mol_to_graph(mol, y_target):
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j];  dst += [j, i]
    if not src:  # single-atom molecule
        src = [0];  dst = [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([y_target], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


# =============================================================================
# 2)  DATA LOADER (CSV + ECFP + SDF)
# =============================================================================
class DataLoader_SMRT:
    def __init__(self, csv_path, ecfp_path, sdf_path):
        self.df        = self._load_csv(csv_path)
        self.ecfp_dict = self._load_ecfp(ecfp_path)
        self.mol_dict  = self._load_sdf(sdf_path)
        self._sync()
        self.rt_scaler = RobustScaler()
        self.df["rt_original"] = self.df["rt"].copy()
        self.df["rt"] = self.rt_scaler.fit_transform(
            self.df[["rt"]]).flatten()
        print(f"RT normalised — mean={self.df['rt'].mean():.4f}, "
              f"std={self.df['rt'].std():.4f}")

    def _load_csv(self, path):
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        df.columns = df.columns.str.strip().str.lower()
        id_col = next((c for c in df.columns
                       if any(k in c for k in
                              ["pubchem", "cid", "molecule", "id"])), None)
        rt_col = next((c for c in df.columns
                       if any(k in c for k in
                              ["rt", "retention", "time"])), None)
        if not id_col or not rt_col:
            raise KeyError(f"ID/RT column not found. Got: {list(df.columns)}")
        df = df.rename(columns={id_col: "pubchem", rt_col: "rt"})
        df["pubchem"] = pd.to_numeric(df["pubchem"], errors="coerce")
        df["rt"]      = pd.to_numeric(df["rt"],      errors="coerce")
        df = df.dropna(subset=["pubchem", "rt"]).reset_index(drop=True)
        df["pubchem"] = df["pubchem"].astype(int)
        print(f"CSV loaded: {len(df)} rows")
        return df

    def _load_ecfp(self, path):
        d = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    mid = int(parts[0].replace("ID=", "").strip())
                    bits = np.array([int(c) for c in parts[1].strip()], dtype=np.float32)
                    if bits.shape[0] == 1024:
                        d[mid] = bits
                except Exception:
                    continue
        print(f"ECFP loaded: {len(d)} molecules")
        return d

    def _load_sdf(self, path):
        d = {}
        for mol in Chem.SDMolSupplier(path):
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                Chem.GetSymmSSSR(mol)
                props = mol.GetPropsAsDict()
                mid = (props.get("PUBCHEM_COMPOUND_CID")
                       or props.get("ID")
                       or props.get("pubchem"))
                if mid is not None:
                    d[int(mid)] = mol
            except Exception:
                continue
        print(f"SDF loaded: {len(d)} molecules")
        return d

    def _sync(self):
        valid = (set(self.df["pubchem"])
                 & set(self.ecfp_dict)
                 & set(self.mol_dict))
        before = len(self.df)
        self.df = self.df[self.df["pubchem"].isin(valid)].reset_index(drop=True)
        print(f"After sync: {before} → {len(self.df)} rows")
        if len(self.df) == 0:
            raise ValueError("Empty dataset after sync. Check ID formats.")

    def denorm(self, y_norm):
        return self.rt_scaler.inverse_transform(
            np.asarray(y_norm).reshape(-1, 1)).flatten()


# =============================================================================
# 3)  TABULAR FEATURE ENGINEERING (for PGM)
# =============================================================================
def rdkit_descriptors(mol):
    try:
        if mol is None:
            return np.zeros(32, dtype=np.float32)
        if not mol.GetRingInfo().IsInitialized():
            Chem.GetSymmSSSR(mol)
        return np.array([
            Descriptors.MolWt(mol),               Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),                Descriptors.MolMR(mol),
            Descriptors.NumHDonors(mol),          Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),   Descriptors.FractionCSP3(mol),
            Descriptors.NumAromaticRings(mol),    Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),           Descriptors.NumHeteroatoms(mol),
            Descriptors.HeavyAtomCount(mol),      Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.BertzCT(mol),             Descriptors.LabuteASA(mol),
            Descriptors.HallKierAlpha(mol),       Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),               Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),              Descriptors.Kappa3(mol),
            Descriptors.BalabanJ(mol),            Descriptors.Ipc(mol),
            mol.GetNumAtoms(),                    mol.GetNumBonds(),
            0, 0, 0, 0, 0, 0,
        ], dtype=np.float32)[:32]
    except Exception:
        return np.zeros(32, dtype=np.float32)


def build_tabular_features(df, ecfp_dict, mol_dict):
    ids = df["pubchem"].astype(int).tolist()
    ecfp = np.stack([ecfp_dict[i] for i in ids]).astype(np.float32)
    desc = np.array([rdkit_descriptors(mol_dict.get(i)) for i in ids], dtype=np.float32)
    X = np.concatenate([ecfp, desc], axis=1)   # (N, 1056)
    y_norm = df["rt"].astype(np.float32).values
    y_orig = df["rt_original"].astype(np.float32).values
    return X, y_norm, y_orig


# =============================================================================
# 4)  GNN MODEL (MolGCN)
# =============================================================================
class MolGCN(nn.Module):
    def __init__(self, node_dim=23, hidden_dim=HIDDEN_DIM, fp_dim=1024, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(node_dim,   hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)

        self.fp_encoder = nn.Sequential(
            nn.LayerNorm(fp_dim),
            nn.Linear(fp_dim,   hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data, fp):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index));  x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index));  x = self.drop(x)
        x = F.relu(self.conv3(x, edge_index))
        graph_emb = global_mean_pool(x, batch)

        fp_emb = self.fp_encoder(fp)

        fused = torch.cat([graph_emb, fp_emb], dim=1)
        return self.head(fused).squeeze(-1)


# =============================================================================
# 5)  GRAPH DATASET BUILDER
# =============================================================================
def build_graph_dataset(ids, mol_dict, ecfp_dict, targets):
    data_list = []
    for i, mid in enumerate(ids):
        mol = mol_dict.get(mid)
        if mol is None:
            continue
        g = mol_to_graph(mol, float(targets[i]))
        if g is None:
            continue
        g.fp = torch.tensor(ecfp_dict[mid], dtype=torch.float).unsqueeze(0)
        data_list.append(g)
    return data_list


# =============================================================================
# 6)  TRAINING LOOP
# =============================================================================
def train_gnn(model, train_data, val_data,
              epochs=GNN_EPOCHS, patience=PATIENCE,
              lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE):

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    loss_fn = nn.HuberLoss(delta=1.0)

    train_loader = PyGLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_data,   batch_size=batch_size, shuffle=False)

    best_val, bad, best_state = float("inf"), 0, None
    hist_tr, hist_va = [], []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            fp    = batch.fp.squeeze(1).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(batch, fp), batch.y.squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
        tr_loss = total_loss / n

        model.eval()
        total_vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                fp    = batch.fp.squeeze(1).to(DEVICE)
                total_vl += loss_fn(model(batch, fp), batch.y.squeeze()).item() * batch.num_graphs
        va_loss = total_vl / len(val_data)

        sched.step(va_loss)
        hist_tr.append(tr_loss)
        hist_va.append(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stop at epoch {ep}")
                break

        if ep % 20 == 0:
            print(f"  Epoch {ep:4d} | train={tr_loss:.5f} | val={va_loss:.5f}")

    model.load_state_dict(best_state)
    return model, hist_tr, hist_va


# =============================================================================
# 7)  INFERENCE
# =============================================================================
@torch.no_grad()
def predict_gnn(model, data_list, batch_size=BATCH_SIZE):
    model.eval()
    preds = []
    for batch in PyGLoader(data_list, batch_size=batch_size, shuffle=False):
        batch = batch.to(DEVICE)
        fp    = batch.fp.squeeze(1).to(DEVICE)
        preds.append(model(batch, fp).cpu().numpy())
    return np.concatenate(preds)


# =============================================================================
# 8)  ALIGNMENT HELPER
# =============================================================================
def align_predictions(ids, mol_dict, gnn_preds, fallback_norm):
    out = fallback_norm.copy()
    ptr = 0
    for i, mid in enumerate(ids):
        if mol_dict.get(mid) is not None:
            out[i] = gnn_preds[ptr]
            ptr += 1
    return out


# =============================================================================
# 9)  METRICS
# =============================================================================
def compute_metrics(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    rel = err / np.maximum(np.abs(y_true), 1e-6) * 100
    return {
        "MedAE (s)":  float(median_absolute_error(y_true, y_pred)),
        "MedRE (%)":  float(np.median(rel)),
        "MAE (s)":    float(mean_absolute_error(y_true, y_pred)),
        "RMSE (s)":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²":         float(r2_score(y_true, y_pred)),
        "Pearson r":  float(pearsonr(y_true, y_pred)[0]),
        "Spearman ρ": float(spearmanr(y_true, y_pred)[0]),
        "% ≤ 60s":    float((err <= 60).mean() * 100),
        "% ≤ 30s":    float((err <= 30).mean() * 100),
        "% ≤ 10s":    float((err <= 10).mean() * 100),
    }


def print_metrics(title, m):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")
    for k, v in m.items():
        print(f"  {k:<14}: {v:10.4f}")


# =============================================================================
# 10)  PLOTTING
# =============================================================================
def plot_training(hist_tr, hist_va, title):
    plt.figure(figsize=(7, 3))
    plt.plot(hist_tr, label="train loss", linewidth=1)
    plt.plot(hist_va, label="val loss",   linewidth=1)
    plt.title(title);  plt.xlabel("epoch");  plt.ylabel("Huber loss")
    plt.legend();  plt.tight_layout();  plt.show()


def scatter_plot(y_true, y_pred, title):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10)
    mn, mx = y_true.min(), y_true.max()
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.title(title);  plt.xlabel("True RT (s)");  plt.ylabel("Pred RT (s)")
    plt.tight_layout();  plt.show()


def residual_plot(y_true, y_pred, title):
    resid = y_pred - y_true
    plt.figure(figsize=(7, 3))
    plt.scatter(y_pred, resid, alpha=0.4, s=10)
    plt.axhline(0, color="r", linestyle="--", linewidth=1)
    plt.title(title);  plt.xlabel("Predicted");  plt.ylabel("Residual")
    plt.tight_layout();  plt.show()


# =============================================================================
# 11)  WEIGHTED ENSEMBLE (learned on val set)
# =============================================================================
def learn_ensemble_weights(preds_val_list, y_val):
    n = len(preds_val_list)
    def obj(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blended = sum(w[i] * preds_val_list[i] for i in range(n))
        return mean_squared_error(y_val, blended)
    res = minimize(obj, np.ones(n) / n, method="Nelder-Mead",
                   options={"maxiter": 5000})
    w = np.abs(res.x) / np.abs(res.x).sum()
    return w


# =============================================================================
# 12)  MAIN PIPELINE
# =============================================================================
def run_forward_hybrid(csv_path, ecfp_path, sdf_path):
    t0 = time.time()

    # Load data
    loader = DataLoader_SMRT(csv_path, ecfp_path, sdf_path)
    X, y_norm, y_orig = build_tabular_features(loader.df, loader.ecfp_dict, loader.mol_dict)
    ids = loader.df["pubchem"].astype(int).tolist()

    # Split
    idx = np.arange(len(X))
    tr_idx, tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
    va_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=SEED)
    print(f"Split — train:{len(tr_idx)}  val:{len(va_idx)}  test:{len(te_idx)}")

    # Scale tabular features
    xs = RobustScaler()
    Xtr = xs.fit_transform(X[tr_idx])
    Xva = xs.transform(X[va_idx])
    Xte = xs.transform(X[te_idx])

    ytr, yva, yte = y_norm[tr_idx], y_norm[va_idx], y_norm[te_idx]
    yte_s = y_orig[te_idx]

    ids_tr = [ids[i] for i in tr_idx]
    ids_va = [ids[i] for i in va_idx]
    ids_te = [ids[i] for i in te_idx]

    # ======================================================================
    # STEP A — PGM baseline
    # ======================================================================
    print("\n" + "="*60)
    print("STEP A: PGM baseline  (BayesianRidge)")
    pgm = BayesianRidge()
    pgm.fit(Xtr, ytr)

    pgm_tr_norm = pgm.predict(Xtr)
    pgm_va_norm = pgm.predict(Xva)
    pgm_te_norm = pgm.predict(Xte)
    pgm_te_s    = loader.denorm(pgm_te_norm)

    metrics_pgm = compute_metrics(yte_s, pgm_te_s)
    print_metrics("PGM — BayesianRidge", metrics_pgm)
    scatter_plot(yte_s, pgm_te_s, "PGM")

    # ======================================================================
    # STEP B — GNN baseline (MolGCN)
    # ======================================================================
    print("\n" + "="*60)
    print("STEP B: GNN baseline  (MolGCN — trained on absolute RT)")

    print("  Building molecular graphs …")
    train_graphs = build_graph_dataset(ids_tr, loader.mol_dict, loader.ecfp_dict, ytr)
    val_graphs   = build_graph_dataset(ids_va, loader.mol_dict, loader.ecfp_dict, yva)
    test_graphs  = build_graph_dataset(ids_te, loader.mol_dict, loader.ecfp_dict,
                                       np.zeros(len(ids_te), np.float32))

    print(f"  Graphs — train:{len(train_graphs)}  val:{len(val_graphs)}  test:{len(test_graphs)}")

    gnn = MolGCN(node_dim=23, hidden_dim=HIDDEN_DIM, fp_dim=1024, dropout=DROPOUT)
    n_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    print(f"  MolGCN parameters: {n_params:,}")

    gnn, hist_tr_gnn, hist_va_gnn = train_gnn(
        gnn, train_graphs, val_graphs,
        epochs=GNN_EPOCHS, patience=PATIENCE,
        lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE)

    plot_training(hist_tr_gnn, hist_va_gnn, "GNN training curve")

    gnn_raw_tr = predict_gnn(gnn, train_graphs)
    gnn_raw_va = predict_gnn(gnn, val_graphs)
    gnn_raw_te = predict_gnn(gnn, test_graphs)

    gnn_tr_norm = align_predictions(ids_tr, loader.mol_dict, gnn_raw_tr, pgm_tr_norm)
    gnn_va_norm = align_predictions(ids_va, loader.mol_dict, gnn_raw_va, pgm_va_norm)
    gnn_te_norm = align_predictions(ids_te, loader.mol_dict, gnn_raw_te, pgm_te_norm)

    gnn_te_s = loader.denorm(gnn_te_norm)
    metrics_gnn = compute_metrics(yte_s, gnn_te_s)
    print_metrics("GNN baseline  (MolGCN)", metrics_gnn)
    scatter_plot(yte_s,  gnn_te_s, "GNN")
    residual_plot(yte_s, gnn_te_s, "GNN residuals")

    # ======================================================================
    # STEP C — Forward Hybrid: GNN → PGM residual corrector
    # ======================================================================
    print("\n" + "="*60)
    print("STEP C: Forward Hybrid  (GNN → PGM residual corrector)")

    resid_tr = ytr - gnn_tr_norm
    pgm_corrector = BayesianRidge()
    pgm_corrector.fit(Xtr, resid_tr)

    corr_te = pgm_corrector.predict(Xte)

    fwd_te_norm = gnn_te_norm + corr_te
    fwd_te_s    = loader.denorm(fwd_te_norm)

    metrics_fwd = compute_metrics(yte_s, fwd_te_s)
    print_metrics("Forward Hybrid  (GNN + PGM correction)", metrics_fwd)
    scatter_plot(yte_s,  fwd_te_s, "Forward Hybrid")
    residual_plot(yte_s, fwd_te_s, "Forward Hybrid residuals")

    # ======================================================================
    # STEP D — Weighted ensemble (PGM, GNN, ForwardHybrid)
    # ======================================================================
    print("\n" + "="*60)
    print("STEP D: Weighted ensemble")

    preds_va = [pgm_va_norm, gnn_va_norm, fwd_te_norm]  # careful: fwd_te_norm is test, but we need val predictions for learning.
    # Actually, we need val predictions of the forward hybrid. Let's compute them correctly.
    corr_va = pgm_corrector.predict(Xva)
    fwd_va_norm = gnn_va_norm + corr_va
    preds_va = [pgm_va_norm, gnn_va_norm, fwd_va_norm]

    weights = learn_ensemble_weights(preds_va, yva)
    print(f"  Learned weights — PGM:{weights[0]:.3f}  GNN:{weights[1]:.3f}  FwdHybrid:{weights[2]:.3f}")

    ens_te_norm = (weights[0] * pgm_te_norm
                 + weights[1] * gnn_te_norm
                 + weights[2] * fwd_te_norm)
    ens_te_s = loader.denorm(ens_te_norm)

    metrics_ens = compute_metrics(yte_s, ens_te_s)
    print_metrics("Weighted Ensemble", metrics_ens)
    scatter_plot(yte_s, ens_te_s, "Ensemble")

    # ======================================================================
    # STEP E — Statistical tests
    # ======================================================================
    print("\n" + "="*60)
    print("STEP E: Statistical tests")

    abs_pgm = np.abs(yte_s - pgm_te_s)
    abs_gnn = np.abs(yte_s - gnn_te_s)
    abs_fwd = np.abs(yte_s - fwd_te_s)
    abs_ens = np.abs(yte_s - ens_te_s)

    tests = {
        "Fwd < GNN  (p)": wilcoxon(abs_fwd, abs_gnn, alternative="less").pvalue,
        "Fwd < PGM  (p)": wilcoxon(abs_fwd, abs_pgm, alternative="less").pvalue,
        "Ens < GNN  (p)": wilcoxon(abs_ens, abs_gnn, alternative="less").pvalue,
        "Ens < Fwd  (p)": wilcoxon(abs_ens, abs_fwd, alternative="less").pvalue,
    }
    for k, v in tests.items():
        sig = "✓ significant" if v < 0.05 else "✗ not significant"
        print(f"  {k:<20}: {v:.4g}  {sig}")

    # ======================================================================
    # STEP F — Summary table
    # ======================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    summary = pd.DataFrame({
        "Model":     ["PGM", "GNN", "Forward Hybrid", "Ensemble"],
        "R²":        [metrics_pgm["R²"],        metrics_gnn["R²"],
                      metrics_fwd["R²"],         metrics_ens["R²"]],
        "MAE (s)":   [metrics_pgm["MAE (s)"],   metrics_gnn["MAE (s)"],
                      metrics_fwd["MAE (s)"],    metrics_ens["MAE (s)"]],
        "RMSE (s)":  [metrics_pgm["RMSE (s)"],  metrics_gnn["RMSE (s)"],
                      metrics_fwd["RMSE (s)"],   metrics_ens["RMSE (s)"]],
        "MedAE (s)": [metrics_pgm["MedAE (s)"], metrics_gnn["MedAE (s)"],
                      metrics_fwd["MedAE (s)"],  metrics_ens["MedAE (s)"]],
        "MedRE (%)": [metrics_pgm["MedRE (%)"], metrics_gnn["MedRE (%)"],
                      metrics_fwd["MedRE (%)"],  metrics_ens["MedRE (%)"]],
    }).set_index("Model")
    print(summary.to_string())

    print(f"\nTotal runtime: {timedelta(seconds=int(time.time() - t0))}")

    return {
        "metrics": {"PGM": metrics_pgm, "GNN": metrics_gnn,
                    "ForwardHybrid": metrics_fwd, "Ensemble": metrics_ens},
        "weights": weights,
        "stats":   tests,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Hybrid GNN→PGM for retention time prediction")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH,
                        help="Path to the SMRT CSV file")
    parser.add_argument("--ecfp", type=str, default=DEFAULT_ECFP_PATH,
                        help="Path to the ECFP fingerprint file")
    parser.add_argument("--sdf", type=str, default=DEFAULT_SDF_PATH,
                        help="Path to the SDF file")
    parser.add_argument("--epochs", type=int, default=GNN_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for GNN training")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM,
                        help="Hidden dimension of GNN")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # Override globals with command line arguments
    global GNN_EPOCHS, BATCH_SIZE, HIDDEN_DIM, DEVICE
    GNN_EPOCHS   = args.epochs
    BATCH_SIZE   = args.batch_size
    HIDDEN_DIM   = args.hidden_dim
    DEVICE       = args.device

    print(f"Device: {DEVICE}")
    results = run_forward_hybrid(args.csv, args.ecfp, args.sdf)
