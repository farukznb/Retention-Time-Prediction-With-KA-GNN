"""
=============================================================================
REVERSE HYBRID PIPELINE  —  PGM (primary predictor) → GNN residual corrector
=============================================================================
Flow:
  1. Load CSV + ECFP + SDF
  2. Build tabular features: ECFP1024 + 32 RDKit descriptors  (for PGM)
  3. Build molecular graphs from SDF                          (for GNN)
  4. Train/Val/Test split (70 / 15 / 15)
  5. Scale tabular features with RobustScaler
  6. Normalise RT with RobustScaler
  7. [PGM baseline]    BayesianRidge on scaled tabular X
  8. [Reverse Hybrid]  Compute PGM residuals on train set →
                       train GCN corrector on molecular graphs
                       to predict those residuals.
                       final pred = pgm_pred + gcn_correction
  9. Metrics + Wilcoxon tests + plots

Architecture of MolGCN:
  Graph stream  : 3 × GCNConv(hidden=256) → global mean pool → 256-dim
  FP stream     : LayerNorm(1024) → Linear → ReLU → Linear → 256-dim
  Fusion        : concat(512) → Linear(512→256) → ReLU → Linear(256→1)

Node feature vector (23-dim):
  atomic-type one-hot (10) | hybridisation one-hot (6) | aromatic (1)
  | formal charge (1) | implicit-H count (1) | in-ring (1) | chirality (3)

Why this is a genuine GNN and not an MLP:
  GCNConv performs iterative neighbourhood aggregation over the molecular
  graph topology.  Each atom's embedding is updated by a normalised mean
  of its bonded neighbours' embeddings followed by a learnable linear
  transform + ReLU.  The output depends on the graph adjacency structure
  (edge_index) and cannot be reproduced by a plain feedforward network
  operating on a flat feature vector.
=============================================================================
"""

# ── standard lib ─────────────────────────────────────────────────────────────
import time, warnings
from datetime import timedelta

# ── scientific stack ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ── rdkit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

# ── torch ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── torch_geometric ───────────────────────────────────────────────────────────
from torch_geometric.data import Data, DataLoader as PyGLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ── viz ───────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


# =============================================================================
# 0)  CONFIG
# =============================================================================
DATASET = {
    "name":      "SMRT",
    "csv_path":  "/content/drive/MyDrive/SMRT/KAGNN-main/8038913/SMRT_dataset.csv",
    "ecfp_path": "/content/drive/MyDrive/SMRT/KAGNN-main/8038913/SMRT_ECFP_1024_Fingerprints.txt",
    "sdf_path":  "/content/drive/MyDrive/SMRT/KAGNN-main/8038913/SMRT_dataset.sdf",
}

SEED         = 42
GNN_EPOCHS   = 150
PATIENCE     = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 64
HIDDEN_DIM   = 256
DROPOUT      = 0.1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


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
    """One-hot with a trailing 'other' bucket — length = len(vocab) + 1."""
    vec = [0] * (len(vocab) + 1)
    if value in vocab:
        vec[vocab.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom):
    """
    Returns a 23-dimensional node feature vector.
      10  atomic-type one-hot  (9 elements + other)
       6  hybridisation one-hot (5 types  + other)
       1  is aromatic
       1  formal charge
       1  implicit H count (capped at 4)
       1  is in ring
       3  chirality one-hot (2 types + other)
    ─────────────────────────────────────────────
      23  total
    """
    feat  = one_hot(atom.GetSymbol(),        ATOM_TYPES)    # 10
    feat += one_hot(atom.GetHybridization(), HYBRID_TYPES)  # 6
    feat += [int(atom.GetIsAromatic())]                     # 1
    feat += [float(atom.GetFormalCharge())]                 # 1
    feat += [float(min(atom.GetTotalNumHs(), 4))]           # 1
    feat += [int(atom.IsInRing())]                          # 1
    feat += one_hot(atom.GetChiralTag(), CHIRAL_TYPES)      # 3
    assert len(feat) == 23, f"Expected 23 features, got {len(feat)}"
    return feat


def mol_to_graph(mol, y_target):
    """Convert an RDKit mol to a torch_geometric Data object."""
    if mol is None:
        return None

    # node features
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float)                      # (N_atoms, 23)

    # edge index — both directions (undirected graph)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j];  dst += [j, i]
    if not src:                                              # single-atom mol
        src = [0];  dst = [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.tensor([y_target], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


# =============================================================================
# 2)  DATA LOADER  (CSV + ECFP + SDF)
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
                    mid  = int(parts[0].replace("ID=", "").strip())
                    bits = np.array([int(c) for c in parts[1].strip()],
                                    dtype=np.float32)
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
        valid  = (set(self.df["pubchem"])
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
# 3)  TABULAR FEATURE ENGINEERING  (for PGM stage)
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
            Descriptors.NumHDonors(mol),           Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),    Descriptors.FractionCSP3(mol),
            Descriptors.NumAromaticRings(mol),     Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),            Descriptors.NumHeteroatoms(mol),
            Descriptors.HeavyAtomCount(mol),       Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.BertzCT(mol),              Descriptors.LabuteASA(mol),
            Descriptors.HallKierAlpha(mol),        Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),                Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),               Descriptors.Kappa3(mol),
            Descriptors.BalabanJ(mol),             Descriptors.Ipc(mol),
            mol.GetNumAtoms(),                     mol.GetNumBonds(),
            0, 0, 0, 0, 0, 0,
        ], dtype=np.float32)[:32]
    except Exception:
        return np.zeros(32, dtype=np.float32)


def build_tabular_features(df, ecfp_dict, mol_dict):
    ids  = df["pubchem"].astype(int).tolist()
    ecfp = np.stack([ecfp_dict[i] for i in ids]).astype(np.float32)
    desc = np.array([rdkit_descriptors(mol_dict.get(i)) for i in ids],
                    dtype=np.float32)
    X      = np.concatenate([ecfp, desc], axis=1)   # (N, 1056)
    y_norm = df["rt"].astype(np.float32).values
    y_orig = df["rt_original"].astype(np.float32).values
    return X, y_norm, y_orig


# =============================================================================
# 4)  GNN MODEL
# =============================================================================
class MolGCN(nn.Module):
    """
    Dual-stream GNN.

    Graph stream  : 3 × GCNConv → global mean pool → 256-dim embedding
    FP stream     : LayerNorm → Linear → ReLU → Linear → 256-dim embedding
    Fusion        : concat(512) → Linear(512→256) → ReLU → Linear(256→1)

    In the reverse hybrid this model is trained to predict PGM residuals,
    not absolute RT.  The message-passing architecture is identical to the
    forward hybrid; only the training targets differ.
    """
    def __init__(self, node_dim=23, hidden_dim=HIDDEN_DIM,
                 fp_dim=1024, dropout=DROPOUT):
        super().__init__()

        # graph stream
        self.conv1 = GCNConv(node_dim,   hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)

        # fingerprint stream
        self.fp_encoder = nn.Sequential(
            nn.LayerNorm(fp_dim),
            nn.Linear(fp_dim,   hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # predictor head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data, fp):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # graph stream — neighbourhood aggregation
        x = F.relu(self.conv1(x, edge_index));  x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index));  x = self.drop(x)
        x = F.relu(self.conv3(x, edge_index))
        graph_emb = global_mean_pool(x, batch)   # (B, hidden_dim)

        # fingerprint stream
        fp_emb = self.fp_encoder(fp)             # (B, hidden_dim)

        # fusion + prediction
        fused = torch.cat([graph_emb, fp_emb], dim=1)
        return self.head(fused).squeeze(-1)      # (B,)


# =============================================================================
# 5)  GRAPH DATASET BUILDER
# =============================================================================
def build_graph_dataset(ids, mol_dict, ecfp_dict, targets):
    """
    Build a list of torch_geometric Data objects.
    targets[i] is the regression target (residual or placeholder) for ids[i].
    Molecules missing from mol_dict are silently skipped.
    """
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
              lr=LR, weight_decay=WEIGHT_DECAY,
              batch_size=BATCH_SIZE):

    model    = model.to(DEVICE)
    opt      = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                   opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    loss_fn  = nn.HuberLoss(delta=1.0)

    train_loader = PyGLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_data,   batch_size=batch_size, shuffle=False)

    best_val, bad, best_state = float("inf"), 0, None
    hist_tr, hist_va = [], []

    for ep in range(1, epochs + 1):
        # train
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            fp    = batch.fp.squeeze(1).to(DEVICE)
            opt.zero_grad()
            loss  = loss_fn(model(batch, fp), batch.y.squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            n          += batch.num_graphs
        tr_loss = total_loss / n

        # validate
        model.eval()
        total_vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                fp    = batch.fp.squeeze(1).to(DEVICE)
                total_vl += (loss_fn(model(batch, fp), batch.y.squeeze()).item()
                             * batch.num_graphs)
        va_loss = total_vl / len(val_data)

        sched.step(va_loss)
        hist_tr.append(tr_loss)
        hist_va.append(va_loss)

        if va_loss < best_val - 1e-8:
            best_val   = va_loss
            bad        = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
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
#     build_graph_dataset silently drops molecules missing from SDF.
#     This maps GNN predictions back to the full index array, using the
#     provided fallback for any molecule without a valid graph.
# =============================================================================
def align_predictions(ids, mol_dict, gnn_preds, fallback_norm):
    """
    Returns a full-length array (one entry per id).
    Positions with a valid mol → GNN prediction.
    Positions without a valid mol → fallback (e.g. PGM prediction).
    """
    out = fallback_norm.copy()
    ptr = 0
    for i, mid in enumerate(ids):
        if mol_dict.get(mid) is not None:
            out[i] = gnn_preds[ptr]
            ptr   += 1
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
# 11)  MAIN PIPELINE  —  REVERSE HYBRID  (PGM → GNN)
# =============================================================================
def run_reverse_hybrid(ds):
    t0 = time.time()

    # ── load data ─────────────────────────────────────────────────────────────
    loader = DataLoader_SMRT(ds["csv_path"], ds["ecfp_path"], ds["sdf_path"])
    X, y_norm, y_orig = build_tabular_features(
        loader.df, loader.ecfp_dict, loader.mol_dict)
    ids = loader.df["pubchem"].astype(int).tolist()

    # ── 70 / 15 / 15 split ────────────────────────────────────────────────────
    idx = np.arange(len(X))
    tr_idx, tmp    = train_test_split(idx, test_size=0.30, random_state=SEED)
    va_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=SEED)
    print(f"Split — train:{len(tr_idx)}  val:{len(va_idx)}  test:{len(te_idx)}")

    # ── scale tabular features ────────────────────────────────────────────────
    xs  = RobustScaler()
    Xtr = xs.fit_transform(X[tr_idx])
    Xva = xs.transform(X[va_idx])
    Xte = xs.transform(X[te_idx])

    ytr, yva, yte = y_norm[tr_idx], y_norm[va_idx], y_norm[te_idx]
    yte_s         = y_orig[te_idx]   # original seconds — for final metrics

    ids_tr = [ids[i] for i in tr_idx]
    ids_va = [ids[i] for i in va_idx]
    ids_te = [ids[i] for i in te_idx]

    # =========================================================================
    # STEP A — PGM baseline (BayesianRidge on tabular features)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP A: PGM baseline  (BayesianRidge)")
    pgm = BayesianRidge()
    pgm.fit(Xtr, ytr)

    pgm_tr_norm = pgm.predict(Xtr)   # kept for residual computation
    pgm_va_norm = pgm.predict(Xva)
    pgm_te_norm = pgm.predict(Xte)
    pgm_te_s    = loader.denorm(pgm_te_norm)

    metrics_pgm = compute_metrics(yte_s, pgm_te_s)
    print_metrics("PGM — BayesianRidge", metrics_pgm)
    scatter_plot(yte_s, pgm_te_s, f"{ds['name']} | PGM")

    # =========================================================================
    # STEP B — REVERSE HYBRID:  PGM → GCN residual corrector
    #
    #   1. Compute train residuals of the PGM:  r_i = y_i − pgm_pred_i
    #   2. Train a MolGCN to predict those residuals from the molecular graph
    #      and ECFP fingerprint.  The GCN captures substructure-level patterns
    #      that explain the systematic error left by the PGM.
    #   3. Final prediction = pgm_pred + gcn_correction
    # =========================================================================
    print("\n" + "="*60)
    print("STEP B: Reverse Hybrid  (PGM → GCN residual corrector)")

    resid_tr = ytr - pgm_tr_norm   # what the GCN must learn
    resid_va = yva - pgm_va_norm   # used only for monitoring (not training)

    print("  Building molecular graphs …")
    train_graphs = build_graph_dataset(ids_tr, loader.mol_dict,
                                       loader.ecfp_dict, resid_tr)
    val_graphs   = build_graph_dataset(ids_va, loader.mol_dict,
                                       loader.ecfp_dict, resid_va)
    # test targets are unused at inference; pass zeros as placeholder
    test_graphs  = build_graph_dataset(ids_te, loader.mol_dict,
                                       loader.ecfp_dict,
                                       np.zeros(len(ids_te), np.float32))

    print(f"  Graphs — train:{len(train_graphs)}  "
          f"val:{len(val_graphs)}  test:{len(test_graphs)}")

    model    = MolGCN(node_dim=23, hidden_dim=HIDDEN_DIM,
                      fp_dim=1024,  dropout=DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  MolGCN parameters: {n_params:,}")

    model, hist_tr, hist_va = train_gnn(
        model, train_graphs, val_graphs,
        epochs=GNN_EPOCHS, patience=PATIENCE,
        lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE)

    plot_training(hist_tr, hist_va,
                  f"{ds['name']} | GCN corrector training curve")

    # raw GNN corrections (only for molecules that had a valid graph)
    gcn_raw_corr_te = predict_gnn(model, test_graphs)

    # align corrections back to full index array;
    # molecules missing a graph receive a zero correction (PGM prediction kept)
    gcn_corr_te = align_predictions(ids_te, loader.mol_dict,
                                    gcn_raw_corr_te,
                                    np.zeros(len(ids_te), np.float32))

    rev_te_norm = pgm_te_norm + gcn_corr_te
    rev_te_s    = loader.denorm(rev_te_norm)

    metrics_rev = compute_metrics(yte_s, rev_te_s)
    print_metrics("Reverse Hybrid  (PGM + GCN correction)", metrics_rev)
    scatter_plot(yte_s,  rev_te_s, f"{ds['name']} | Reverse Hybrid")
    residual_plot(yte_s, rev_te_s, f"{ds['name']} | Reverse Hybrid residuals")

    # =========================================================================
    # STEP C — STATISTICAL TESTS  (paired Wilcoxon on absolute errors)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP C: Statistical tests")

    abs_pgm = np.abs(yte_s - pgm_te_s)
    abs_rev = np.abs(yte_s - rev_te_s)

    p_rev_lt_pgm = wilcoxon(abs_rev, abs_pgm, alternative="less").pvalue
    sig = "✓ significant" if p_rev_lt_pgm < 0.05 else "✗ not significant"
    print(f"  Rev < PGM (Wilcoxon p): {p_rev_lt_pgm:.4g}  {sig}")

    # =========================================================================
    # STEP D — SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    summary = pd.DataFrame({
        "Model":     ["PGM", "Reverse Hybrid"],
        "R²":        [metrics_pgm["R²"],        metrics_rev["R²"]],
        "MAE (s)":   [metrics_pgm["MAE (s)"],   metrics_rev["MAE (s)"]],
        "RMSE (s)":  [metrics_pgm["RMSE (s)"],  metrics_rev["RMSE (s)"]],
        "MedAE (s)": [metrics_pgm["MedAE (s)"], metrics_rev["MedAE (s)"]],
        "MedRE (%)": [metrics_pgm["MedRE (%)"], metrics_rev["MedRE (%)"]],
        "% ≤ 60s":   [metrics_pgm["% ≤ 60s"],   metrics_rev["% ≤ 60s"]],
    }).set_index("Model")
    print(summary.to_string())

    print(f"\nTotal runtime: {timedelta(seconds=int(time.time() - t0))}")

    return {
        "metrics": {"PGM": metrics_pgm, "ReverseHybrid": metrics_rev},
        "stats":   {"p_rev_lt_pgm": p_rev_lt_pgm},
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    results = run_reverse_hybrid(DATASET)
