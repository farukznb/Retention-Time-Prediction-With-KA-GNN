"""
05_pgm_kagin_classwise.py
=========================
PGM → KA-GNN Reverse Hybrid with Class-wise Analysis (3-Fold CV)

Architecture: Two-stage residual learning
  Stage 1 — PGM ensemble (XGBoost + BayesianRidge) on ECFP + RDKit descriptors
  Stage 2 — RefinementKAGIN (5×GINEConv(KAN) + KAN head) predicts residuals

Key additions vs 03_pgm_kagnn_reverse.py:
  - Chemical class heuristic (assign_chemical_class)
  - Class-wise MedAE / MedRE analysis per fold and in aggregate
  - Majority-bias verdict
  - ECDF of absolute errors
  - RBF-KAN (FastKAN) preferred; B-spline efficient-kan as fallback

Usage:
    python experiments/05_pgm_kagin_classwise.py

Results saved to:
    results/pgm_kagin_classwise/
        checkpoints/best_fold{N}.pth
        plots/
        fold{N}_metrics.json
        aggregate_metrics.json
        all_fold_predictions.csv
"""

# ── Standard library ─────────────────────────────────────────────────────────
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_add_pool

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Repo paths ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results" / "pgm_kagin_classwise"

# KAGNN source (for ekan.py / featurization.py)
KAGNN_PATH = REPO_ROOT / "src"

# ── Helper: silent pip install ────────────────────────────────────────────────
def _pip(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


# ─────────────────────────────────────────────────────────────────────────────
# KAN backend resolution (RBF preferred, B-spline fallback)
# ─────────────────────────────────────────────────────────────────────────────
KAN = None

_ekan_file = Path(KAGNN_PATH) / "ekan.py"
if _ekan_file.exists():
    try:
        _spec = importlib.util.spec_from_file_location("ekan", str(_ekan_file))
        _ekan = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ekan)
        if hasattr(_ekan, "FastKAN"):
            KAN = _ekan.FastKAN
            print("✓ FastKAN (RBF ekan.py) loaded from KAGNN repo")
        elif hasattr(_ekan, "KAN"):
            KAN = _ekan.KAN
            print("✓ KAN (ekan.py) loaded from KAGNN repo")
    except Exception as e:
        print(f"  ekan.py load failed: {e}")

if KAN is None:
    try:
        from fastkan import FastKAN as _FastKAN
        KAN = _FastKAN
        print("✓ FastKAN (RBF, pip fast-kan) loaded")
    except ImportError:
        pass

USE_AMP = True
if KAN is None:
    try:
        _pip("git+https://github.com/Blealtan/efficient-kan.git")
        from efficient_kan import KANLinear

        class KAN(nn.Module):
            """Minimal B-spline KAN wrapper — AMP will be disabled."""
            def __init__(self, layers_hidden, grid_size=5, spline_order=3, **kw):
                super().__init__()
                self.layers = nn.ModuleList([
                    KANLinear(i, o, grid_size=grid_size, spline_order=spline_order)
                    for i, o in zip(layers_hidden, layers_hidden[1:])
                ])
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        USE_AMP = False
        print("✓ KAN (fallback B-spline efficient-kan) — AMP DISABLED")
    except Exception as e:
        raise RuntimeError(f"No KAN implementation available: {e}")
else:
    USE_AMP = torch.cuda.is_available()

print(f"AMP (fp16): {'ENABLED' if USE_AMP else 'DISABLED'}")


# ─────────────────────────────────────────────────────────────────────────────
# KAGNN featurization (official if available, fallback otherwise)
# ─────────────────────────────────────────────────────────────────────────────
KAGNN_AVAILABLE = False
try:
    from featurization import get_atom_fdim, get_bond_fdim, mol2graph
    KAGNN_AVAILABLE = True
    ATOM_FDIM = get_atom_fdim()
    BOND_FDIM  = get_bond_fdim()
    print(f"Official KAGNN featurization | Atom: {ATOM_FDIM} | Bond: {BOND_FDIM}")
except ImportError as e:
    print(f"  KAGNN featurization not available ({e}) — using fallback")
    ATOM_FDIM = 133
    BOND_FDIM  = 14

if not KAGNN_AVAILABLE:
    def get_atom_features_kagnn(atom):
        f = []
        f.extend([int(atom.GetAtomicNum() == x) for x in range(1, 101)])
        f.extend([int(atom.GetDegree() == x) for x in range(11)])
        f.extend([int(atom.GetFormalCharge() == x) for x in range(-2, 3)])
        hybs = [
            Chem.rdchem.HybridizationType.S,    Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,  Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
        ]
        f.extend([int(atom.GetHybridization() == h) for h in hybs])
        f.append(int(atom.GetIsAromatic()))
        f.extend([int(atom.GetTotalNumHs() == x) for x in range(5)])
        f.append(int(atom.IsInRing()))
        f.append(int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
        f.append(int(atom.GetNumRadicalElectrons()))
        f.append(atom.GetImplicitValence())
        f.append(atom.GetExplicitValence())
        return f

    def get_bond_features_kagnn(bond):
        bt = bond.GetBondType()
        f  = [
            int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE),
            int(bt == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        stereo = bond.GetStereo()
        f.extend([int(stereo == s) for s in [
            Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,    Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,  Chem.rdchem.BondStereo.STEREOTRANS,
        ]])
        bd = bond.GetBondDir()
        f.extend([
            int(bd == Chem.rdchem.BondDir.ENDUPRIGHT),
            int(bd == Chem.rdchem.BondDir.ENDDOWNRIGHT),
        ])
        return f


# ─────────────────────────────────────────────────────────────────────────────
# Chemical class heuristic
# ─────────────────────────────────────────────────────────────────────────────
def assign_chemical_class(mol):
    try:
        mw     = Descriptors.MolWt(mol);        logp   = Descriptors.MolLogP(mol)
        tpsa   = Descriptors.TPSA(mol);         n_ar   = Descriptors.NumAromaticRings(mol)
        fsp3   = Descriptors.FractionCSP3(mol); hbd    = Descriptors.NumHDonors(mol)
        n_ring = Descriptors.RingCount(mol)
        if mw > 350 and logp > 4 and tpsa < 80 and fsp3 > 0.6:        return "Lipids"
        if tpsa > 120 and logp < -0.5 and n_ar == 0 and mw < 500:     return "Carbohydrates"
        if tpsa > 100 and logp < 1 and hbd >= 2:                      return "Organic Acids & AA"
        if n_ar >= 1 and Descriptors.NumHeteroatoms(mol) >= 2:        return "Organoheterocyclics"
        if n_ar >= 1:                                                  return "Benzenoids"
        if n_ring == 0 and fsp3 > 0.5:                                return "Aliphatic Organics"
        return "Other"
    except Exception:
        return "Unknown"


CLASS_PALETTE = {
    "Organoheterocyclics": "#2196F3", "Benzenoids":         "#4CAF50",
    "Organic Acids & AA":  "#FF5722", "Lipids":             "#9C27B0",
    "Carbohydrates":       "#FF9800", "Aliphatic Organics": "#00BCD4",
    "Other":               "#795548", "Unknown":            "#9E9E9E",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    csv_path  = DATA_DIR / "SMRT_dataset.csv"
    ecfp_path = DATA_DIR / "SMRT_ECFP_1024_Fingerprints.txt"
    sdf_path  = DATA_DIR / "SMRT_dataset.sdf"

    # Stage 1 — PGM
    optuna_trials   = 15
    optuna_cv_folds = 3

    # Stage 2 — KAGIN architecture (faithful to KAGNN paper, RBF variant)
    hidden_dim       = 256
    dropout          = 0.1
    # B-spline KAN allocates O(batch × atoms × grid × spline_order) intermediates.
    # 512 OOMs on T4/A100 with h=256. 128 is safe; increase only if using RBF FastKAN.
    batch_size       = 128
    epochs           = 120
    # B-spline grid: each extra grid point adds a full (batch × atoms) tensor.
    # 5 is the paper default; reduce to 3 if still OOMing with batch_size=128.
    kan_grid_size    = 5
    kan_hidden_layers= 2

    # Optimizer
    learning_rate = 3e-4
    weight_decay  = 1e-5
    grad_clip     = 1.0

    # Scheduler
    lr_patience = 7
    lr_factor   = 0.5
    min_lr      = 1e-6

    early_stop_patience = 20
    n_folds             = 3
    seed                = 42
    min_class_samples   = 10

    results_dir = RESULTS_DIR

    def __init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        self.use_amp = USE_AMP
        print(f"Results → {self.results_dir}")
        print(f"AMP: {self.use_amp} | batch: {self.batch_size} | h: {self.hidden_dim}")


# ─────────────────────────────────────────────────────────────────────────────
# Safe plot-size helper
# ─────────────────────────────────────────────────────────────────────────────
def _safe_figsize(n_items, base_w=13.0, per_item_h=0.55, min_h=4.0, max_h=20.0):
    h = float(np.clip(n_items * per_item_h, min_h, max_h))
    return float(base_w), h


# ─────────────────────────────────────────────────────────────────────────────
# RDKit descriptors
# ─────────────────────────────────────────────────────────────────────────────
class ComprehensiveDescriptors:
    _cache = {}

    @staticmethod
    def extract(mol):
        try:
            if not mol.GetRingInfo().IsInitialized():
                Chem.GetSymmSSSR(mol)
            return np.array([
                Descriptors.MolWt(mol),             Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),              Descriptors.MolMR(mol),
                Descriptors.NumHDonors(mol),        Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol),
                Descriptors.NumAromaticRings(mol),  Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),         Descriptors.NumHeteroatoms(mol),
                Descriptors.HeavyAtomCount(mol),    Descriptors.MaxAbsPartialCharge(mol),
                Descriptors.BertzCT(mol),           Descriptors.LabuteASA(mol),
                Descriptors.HallKierAlpha(mol),     Descriptors.Chi0v(mol),
                Descriptors.Chi1v(mol),             Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),            Descriptors.Kappa3(mol),
                Descriptors.BalabanJ(mol),          Descriptors.Ipc(mol),
                mol.GetNumAtoms(),                  mol.GetNumBonds(),
                Descriptors.ExactMolWt(mol),        Descriptors.NumValenceElectrons(mol),
                Descriptors.NumRadicalElectrons(mol), Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol),  Descriptors.MinAbsPartialCharge(mol),
            ], dtype=np.float32)
        except Exception:
            return np.zeros(32, dtype=np.float32)

    @staticmethod
    def extract_batch(mol_dict, mol_ids):
        out = []
        for mid in mol_ids:
            mid = int(mid)
            if mid not in ComprehensiveDescriptors._cache:
                ComprehensiveDescriptors._cache[mid] = \
                    ComprehensiveDescriptors.extract(mol_dict[mid])
            out.append(ComprehensiveDescriptors._cache[mid])
        return np.array(out, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────
class SMRTDataLoader:
    def __init__(self, csv_path, ecfp_path, sdf_path):
        print("=" * 60 + "\nLOADING SMRT DATASET\n" + "=" * 60)
        df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="skip")
        df.columns = df.columns.str.strip().str.lower()
        id_col = next(
            (c for c in df.columns if "pubchem" in c or "cid" in c or "#" in c),
            df.columns[0],
        )
        rt_col = next((c for c in df.columns if "rt" in c or "retention" in c), None)
        df = df.rename(columns={id_col: "pubchem_id", rt_col: "rt"})
        df["pubchem_id"] = pd.to_numeric(df["pubchem_id"], errors="coerce")
        df["rt"]         = pd.to_numeric(df["rt"],         errors="coerce")
        self.df = df.dropna(subset=["pubchem_id", "rt"]).reset_index(drop=True)

        self.ecfp_dict = {}
        with open(ecfp_path, "r") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        cid = int(parts[0].replace("ID=", ""))
                        self.ecfp_dict[cid] = np.array(
                            [int(b) for b in parts[1][:1024]], dtype=np.float32
                        )
                    except Exception:
                        continue
        print(f"  {len(self.ecfp_dict)} fingerprints")

        self.mol_dict = {}
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        valid = errors = 0
        for i, mol in enumerate(suppl):
            if mol is None:
                errors += 1
                continue
            try:
                cid = int(mol.GetProp("PUBCHEM_COMPOUND_CID"))
                Chem.SanitizeMol(mol)
                Chem.GetSymmSSSR(mol)
                self.mol_dict[cid] = mol
                valid += 1
            except Exception:
                errors += 1
            if (i + 1) % 10000 == 0:
                print(f"  {i+1} | valid={valid} errors={errors}")
        print(f"  SDF: {valid} valid, {errors} skipped")

        common = (
            set(self.df.pubchem_id)
            & set(self.ecfp_dict)
            & set(self.mol_dict)
        )
        self.df        = self.df[self.df.pubchem_id.isin(common)].reset_index(drop=True)
        self.ecfp_dict = {k: v for k, v in self.ecfp_dict.items() if k in common}
        self.mol_dict  = {k: v for k, v in self.mol_dict.items()  if k in common}
        self.rt_scaler = RobustScaler()
        self.df["rt_norm"] = (
            self.rt_scaler.fit_transform(self.df[["rt"]]).flatten().astype(np.float32)
        )
        print(f"Dataset ready: {len(self.df):,} molecules")

    def denormalize(self, rt_norm):
        return self.rt_scaler.inverse_transform(
            np.array(rt_norm, dtype=np.float32).reshape(-1, 1)
        ).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# PrecomputedSMRTDataset
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph(mol):
    if mol.GetNumAtoms() == 0:
        return Data(
            x=torch.zeros((1, ATOM_FDIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, BOND_FDIM), dtype=torch.float32),
        )
    if KAGNN_AVAILABLE:
        gd     = mol2graph(mol)
        x      = torch.tensor(gd["node_features"], dtype=torch.float32)
        raw_ei = gd["edge_index"]
        raw_ea = gd["edge_features"]
        if (
            raw_ei is None
            or len(raw_ei) == 0
            or (
                hasattr(raw_ei, "__len__")
                and (
                    len(raw_ei) == 0
                    or (hasattr(raw_ei[0], "__len__") and len(raw_ei[0]) == 0)
                )
            )
        ):
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, BOND_FDIM), dtype=torch.float32)
        else:
            edge_index = torch.tensor(raw_ei, dtype=torch.long)
            edge_attr  = torch.tensor(raw_ea, dtype=torch.float32)
            if edge_index.dim() == 2 and edge_index.shape[0] != 2:
                edge_index = edge_index.t().contiguous()
    else:
        x  = torch.tensor(
            [get_atom_features_kagnn(a) for a in mol.GetAtoms()], dtype=torch.float32
        )
        ei, ea = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf   = get_bond_features_kagnn(bond)
            ei.extend([[i, j], [j, i]])
            ea.extend([bf, bf])
        if len(ei) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, BOND_FDIM), dtype=torch.float32)
        else:
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(ea, dtype=torch.float32)
    return Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr.float())


class PrecomputedSMRTDataset(Dataset):
    """All graphs built once at construction. __getitem__ = pure dict lookup."""

    def __init__(self, df, ecfp_dict, mol_dict, desc=""):
        self.df        = df.reset_index(drop=True)
        self.ecfp_dict = ecfp_dict
        self.graphs    = {}
        unique_cids    = list(dict.fromkeys(self.df["pubchem_id"].astype(int).tolist()))
        print(f"  Pre-computing {len(unique_cids):,} graphs [{desc}]...", end="", flush=True)
        t0 = time.time()
        for cid in unique_cids:
            self.graphs[cid] = _build_graph(mol_dict[cid])
        print(f" done in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid = int(row.pubchem_id)
        return (
            self.graphs[cid],
            torch.tensor(self.ecfp_dict[cid], dtype=torch.float32),
            torch.tensor(row.rt_norm,          dtype=torch.float32),
            cid,
        )


def collate_fn(batch):
    graphs, ecfps, rts, ids = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(ecfps),
        torch.stack(rts),
        tuple(int(x) for x in ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RefinementKAGIN
# ─────────────────────────────────────────────────────────────────────────────
def _make_kan(in_dim, h, out_dim, hl, gs):
    sizes = [in_dim] + [h] * (hl - 1) + [out_dim]
    try:
        return KAN(layers_hidden=sizes, num_grids=gs)
    except TypeError:
        try:
            return KAN(layers_hidden=sizes, grid_size=gs, spline_order=3)
        except TypeError:
            return KAN(sizes)


class RefinementKAGIN(nn.Module):
    """
    KAGNN residual corrector implemented as RBF-KAGIN (preferred) or BS-KAGIN.
    Faithfully mirrors graph_regression/models.py → class KAGIN (eq.9).
    """

    def __init__(self, config):
        super().__init__()
        h  = config.hidden_dim
        gs = config.kan_grid_size
        hl = config.kan_hidden_layers

        self.atom_embed = nn.Linear(ATOM_FDIM, h)
        self.bond_embed = nn.Linear(BOND_FDIM, h)

        self.conv    = nn.ModuleList([GINEConv(_make_kan(h, h, h, hl, gs)) for _ in range(5)])
        self.bn      = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(5)])
        self.dropout = nn.Dropout(config.dropout)

        self.ecfp_enc = _make_kan(1024, h * 2, h, hl, gs)

        try:
            self.residual_head = KAN(layers_hidden=[h * 2, h, h // 2, 1], num_grids=gs)
        except TypeError:
            try:
                self.residual_head = KAN(
                    layers_hidden=[h * 2, h, h // 2, 1], grid_size=gs, spline_order=3
                )
            except TypeError:
                self.residual_head = KAN([h * 2, h, h // 2, 1])

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"RefinementKAGIN | {total:,} params | 5×GINEConv(KAN) | h={h} | grid={gs} | "
            f"ATOM={ATOM_FDIM} | BOND={BOND_FDIM} | "
            f"KAN={'RBF-KAGIN' if USE_AMP else 'BS-KAGIN'}"
        )

    def forward(self, graph, ecfp):
        x = self.atom_embed(graph.x)
        e = self.bond_embed(graph.edge_attr)
        for i in range(5):
            x = self.conv[i](x, graph.edge_index, e)
            x = self.bn[i](x)
            x = self.dropout(x)
        g_emb = global_add_pool(x, graph.batch)
        e_emb = self.ecfp_enc(ecfp)
        fused = torch.cat([g_emb, e_emb], dim=-1)
        return self.residual_head(fused).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    y_true = np.clip(y_true, 1e-3, None) if (y_true <= 0).any() else y_true
    ae = np.abs(y_true - y_pred)
    re = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6)) * 100
    return {
        "n_samples":  int(len(y_true)),
        "MedAE":      float(np.median(ae)),
        "MedRE":      float(np.median(re)),
        "MAE":        float(np.mean(ae)),
        "RMSE":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":         float(r2_score(y_true, y_pred)),
        "Pearson":    float(pearsonr(y_true, y_pred)[0]),
        "Spearman":   float(spearmanr(y_true, y_pred)[0]),
        "Pct_le_60s": float((ae <= 60).mean() * 100),
        "Pct_le_30s": float((ae <= 30).mean() * 100),
        "Pct_le_10s": float((ae <= 10).mean() * 100),
        "MeanError":  float(np.mean(y_pred - y_true)),
        "Skewness":   float(pd.Series(y_pred - y_true).skew()),
        "Kurtosis":   float(pd.Series(y_pred - y_true).kurt()),
    }


def print_metrics(metrics, label):
    print(f"  {label}:")
    for k in [
        "MedAE", "MedRE", "MAE", "RMSE", "R2", "Pearson", "Spearman",
        "Pct_le_10s", "Pct_le_30s", "Pct_le_60s", "MeanError",
    ]:
        u = (
            "s" if k in ("MedAE", "MAE", "RMSE", "MeanError")
            else "%" if ("Pct" in k or k == "MedRE")
            else ""
        )
        print(f"    {k:<15}: {metrics[k]:>9.4f}{u}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — PGM
# ─────────────────────────────────────────────────────────────────────────────
def extract_pgm_features(loader, mol_dict):
    X_list, y_list, cid_list = [], [], []
    for _, ecfp, rt_norm, cids in loader:
        desc = ComprehensiveDescriptors.extract_batch(mol_dict, cids)
        X_list.append(np.concatenate([ecfp.numpy(), desc], axis=1))
        y_list.append(rt_norm.numpy())
        cid_list.extend(list(cids))
    return (
        np.nan_to_num(np.concatenate(X_list), 0),
        np.concatenate(y_list),
        cid_list,
    )


def train_pgm_stage(train_loader, val_loader, mol_dict, config, xgb_device):
    print("\n" + "=" * 60 + "\nSTAGE 1: PGM (XGBoost + BayesianRidge)\n" + "=" * 60)
    t0     = time.time()
    scaler = StandardScaler()
    X_tr, y_tr, _ = extract_pgm_features(train_loader, mol_dict)
    X_va, y_va, _ = extract_pgm_features(val_loader,   mol_dict)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    print(f"  Features: {X_tr_s.shape}  (1024 ECFP + 32 RDKit)")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 150, 400),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method": "hist", "device": xgb_device, "random_state": 42,
        }
        return cross_val_score(
            xgb.XGBRegressor(**params), X_tr_s, y_tr,
            cv=config.optuna_cv_folds, scoring="r2", n_jobs=-1,
        ).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=True, n_jobs=1)
    print(f"  Best CV R2: {study.best_value:.4f}")

    pgm_xgb = xgb.XGBRegressor(**study.best_params, random_state=42)
    pgm_xgb.fit(X_tr_s, y_tr, verbose=False)
    pgm_br  = BayesianRidge(max_iter=500)
    pgm_br.fit(X_tr_s, y_tr)
    ens_va = (pgm_xgb.predict(X_va_s) + pgm_br.predict(X_va_s)) / 2
    dur    = time.time() - t0
    print(
        f"  Val R2={r2_score(y_va, ens_va):.4f} | "
        f"MedAE(norm)={np.median(np.abs(y_va - ens_va)):.4f} | "
        f"done in {str(timedelta(seconds=int(dur)))}"
    )
    return pgm_xgb, pgm_br, scaler, dur


def get_pgm_pred_dict(loader, mol_dict, pgm_xgb, pgm_br, scaler):
    X, _, cids = extract_pgm_features(loader, mol_dict)
    X_s   = scaler.transform(np.nan_to_num(X, 0))
    preds = (pgm_xgb.predict(X_s) + pgm_br.predict(X_s)) / 2
    return {int(c): float(p) for c, p in zip(cids, preds)}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — KAGIN training
# ─────────────────────────────────────────────────────────────────────────────
def train_kagin(train_loader, val_loader, pgm_train_dict, pgm_val_dict,
                config, device, fold):
    print("\n" + "=" * 60)
    print(f"STAGE 2: RefinementKAGIN — 5×GINEConv(KAN) + KAN head")
    print(f"  KAN variant : {'RBF-KAGIN' if USE_AMP else 'BS-KAGIN'}")
    print(f"  AMP (fp16)  : {'ON' if config.use_amp else 'OFF'}")
    print(f"  Scheduler   : ReduceLROnPlateau(patience={config.lr_patience})")
    print("=" * 60)
    t0 = time.time()

    model = RefinementKAGIN(config).to(device)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler  = ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor,
        patience=config.lr_patience, min_lr=config.min_lr,
    )
    _dev_type  = "cuda" if torch.cuda.is_available() else "cpu"
    scaler_amp = GradScaler(device=_dev_type) if config.use_amp else None

    ckpt_path    = config.results_dir / "checkpoints" / f"best_fold{fold}.pth"
    best_val     = float("inf")
    patience_ctr = 0
    start_epoch  = 1
    history      = {"train_loss": [], "val_loss": [], "lr": []}

    if ckpt_path.exists():
        print(f"  Checkpoint found: {ckpt_path.name}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            if not (isinstance(ckpt, dict) and "model_state" in ckpt):
                raise ValueError("Legacy format.")
            cur = set(model.state_dict().keys())
            sav = set(ckpt["model_state"].keys())
            mis = cur - sav; unex = sav - cur
            if mis or unex:
                raise RuntimeError(
                    f"Architecture mismatch.\n"
                    f"  Missing  : {sorted(mis)[:4]}\n"
                    f"  Unexpected: {sorted(unex)[:4]}"
                )
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val     = ckpt.get("best_val_loss", float("inf"))
            start_epoch  = ckpt.get("epoch", 0) + 1
            patience_ctr = ckpt.get("patience_ctr", 0)
            history      = ckpt.get("history", history)
            print(
                f"  Resumed | epoch={start_epoch} | "
                f"best_val={best_val:.5f} | patience={patience_ctr}"
            )
        except (RuntimeError, ValueError) as err:
            print(f"  WARNING: {err}\n  Deleting stale checkpoint.")
            ckpt_path.unlink()

    for epoch in range(start_epoch, config.epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for graph, ecfp, rt, cids in train_loader:
            graph = graph.to(device, non_blocking=True)
            ecfp  = ecfp.to(device,  non_blocking=True)
            rt    = rt.to(device,    non_blocking=True)
            pgm_p = torch.tensor(
                [pgm_train_dict[int(c)] for c in cids], dtype=torch.float32, device=device
            )
            target = rt - pgm_p
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast(device_type=_dev_type, enabled=config.use_amp):
                    pred = model(graph, ecfp)
                    loss = F.smooth_l1_loss(pred, target)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    if scaler_amp is not None:
                        scaler_amp.scale(loss).backward()
                        scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        optimizer.step()
                    tr_loss += loss.item() * len(rt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"  [OOM] skipped a batch at epoch {epoch} — "
                    f"consider reducing batch_size or kan_grid_size"
                )
        tr_loss /= len(train_loader.dataset)
        history["train_loss"].append(tr_loss)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph, ecfp, rt, cids in val_loader:
                graph = graph.to(device, non_blocking=True)
                ecfp  = ecfp.to(device,  non_blocking=True)
                rt    = rt.to(device,    non_blocking=True)
                pgm_p = torch.tensor(
                    [pgm_val_dict[int(c)] for c in cids], dtype=torch.float32, device=device
                )
                target = rt - pgm_p
                with torch.amp.autocast(device_type=_dev_type, enabled=config.use_amp):
                    val_loss += F.smooth_l1_loss(model(graph, ecfp), target).item() * len(rt)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr_now)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  E{epoch:3d}/{config.epochs} | "
                f"tr={tr_loss:.5f} val={val_loss:.5f} lr={lr_now:.2e}"
            )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_ctr = 0
            torch.save(
                {
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss":   best_val,
                    "epoch":           epoch,
                    "patience_ctr":    0,
                    "history":         history,
                },
                ckpt_path,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= config.early_stop_patience:
                print(f"  Early stop at epoch {epoch}")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) else ckpt)
    dur = time.time() - t0
    print(f"  Stage 2 done: {str(timedelta(seconds=int(dur)))} | best_val={best_val:.5f}")
    return model, history, dur


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(test_loader, model, pgm_dict, data_loader_obj, device, config):
    model.eval()
    _dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    y_true_l, y_pgm_l, y_final_l, cid_l = [], [], [], []
    for graph, ecfp, rt_norm, cids in test_loader:
        graph = graph.to(device, non_blocking=True)
        ecfp  = ecfp.to(device,  non_blocking=True)
        pgm_p = np.array([pgm_dict[int(c)] for c in cids])
        with torch.amp.autocast(device_type=_dev_type, enabled=config.use_amp):
            residual = model(graph, ecfp).cpu().float().numpy()
        y_true_l.append(rt_norm.numpy())
        y_pgm_l.append(pgm_p)
        y_final_l.append(pgm_p + residual)
        cid_l.extend([int(c) for c in cids])
    y_true  = data_loader_obj.denormalize(np.concatenate(y_true_l))
    y_pgm   = data_loader_obj.denormalize(np.concatenate(y_pgm_l))
    y_final = data_loader_obj.denormalize(np.concatenate(y_final_l))
    return y_true, y_pgm, y_final, cid_l


# ─────────────────────────────────────────────────────────────────────────────
# Class-wise analysis
# ─────────────────────────────────────────────────────────────────────────────
def classwise_analysis_cv(results_df, config, fold=None, aggregate=False):
    label = "AGGREGATE" if aggregate else f"FOLD {fold}"
    print(f"\n{'=' * 60}")
    print(f"CLASS-WISE ANALYSIS — {label}")
    print(f"{'=' * 60}")

    df = results_df.copy()
    df["ae_pgm"]   = np.abs(df["y_true"] - df["y_pgm"])
    df["ae_final"] = np.abs(df["y_true"] - df["y_final"])
    df["re_pgm"]   = np.abs((df["y_true"] - df["y_pgm"])   / np.maximum(np.abs(df["y_true"]), 1e-6)) * 100
    df["re_final"] = np.abs((df["y_true"] - df["y_final"]) / np.maximum(np.abs(df["y_true"]), 1e-6)) * 100

    agg = (
        df.groupby("chemical_class")
          .agg(
              n           =("ae_final", "count"),
              MedAE_pgm   =("ae_pgm",   "median"),
              MedAE_final =("ae_final", "median"),
              MedRE_pgm   =("re_pgm",   "median"),
              MedRE_final =("re_final", "median"),
          )
          .reset_index()
          .rename(columns={"chemical_class": "Class"})
    )
    agg = agg[agg["n"] >= config.min_class_samples].copy()
    agg["delta_MedAE"] = agg["MedAE_pgm"] - agg["MedAE_final"]

    gme_p = np.median(df["ae_pgm"]);  gme_f = np.median(df["ae_final"])
    gmr_p = np.median(df["re_pgm"]);  gmr_f = np.median(df["re_final"])

    total = len(df)
    print(f"\n  Class distribution (n={total:,}):")
    print(f"  {'Class':<35} {'n':>6}  {'%':>6}")
    print("  " + "-" * 52)
    for cls, cnt in df["chemical_class"].value_counts().items():
        print(f"  {cls:<35} {cnt:>6}  {cnt / total * 100:>5.1f}%")

    print(f"\n  Global MedAE — PGM: {gme_p:.2f}s  |  KAGNN: {gme_f:.2f}s")
    print(f"  Global MedRE — PGM: {gmr_p:.2f}%  |  KAGNN: {gmr_f:.2f}%")
    print(f"\n  Classes shown: {len(agg)} (n >= {config.min_class_samples})")
    print(
        f"  {'Class':<35} {'n':>6}  {'PGM MedAE':>10}  "
        f"{'KAGNN MedAE':>12}  {'Delta':>8}  {'KAGNN MedRE':>12}"
    )
    print("  " + "-" * 90)
    for _, row in agg.sort_values("MedAE_final").iterrows():
        arrow     = "↓" if row["delta_MedAE"] > 0 else "↑"
        color_str = f"{arrow}{abs(row['delta_MedAE']):>6.2f}s"
        print(
            f"  {row['Class']:<35} {int(row['n']):>6}  "
            f"{row['MedAE_pgm']:>9.2f}s  {row['MedAE_final']:>11.2f}s  "
            f"{color_str:>8}  {row['MedRE_final']:>11.2f}%"
        )

    suffix = "aggregate" if aggregate else f"fold{fold}"
    n_cls  = max(1, len(agg))

    print(f"\n  Statistical tests (paired, per class):")
    for _, row in agg.iterrows():
        cls_df = df[df["chemical_class"] == row["Class"]]
        if len(cls_df) >= 10:
            try:
                t_stat, p_t = ttest_rel(cls_df["ae_pgm"], cls_df["ae_final"])
                _, p_w      = wilcoxon(cls_df["ae_pgm"], cls_df["ae_final"])
                sig = (
                    "***" if p_t < 0.001 else
                    "**"  if p_t < 0.01  else
                    "*"   if p_t < 0.05  else "ns"
                )
                print(
                    f"    {row['Class']:<35} t={t_stat:>7.3f} "
                    f"p_t={p_t:.3e} p_w={p_w:.3e} {sig}"
                )
            except Exception as e:
                print(f"    {row['Class']:<35} test failed: {e}")

    # ── Plot 1: MedAE bar ─────────────────────────────────────────────────────
    fw, fh = _safe_figsize(n_cls)
    a_s     = agg.sort_values("MedAE_final").reset_index(drop=True)
    bcolors = ["#2ecc71" if v <= gme_f else "#e74c3c" for v in a_s["MedAE_final"]]
    patches = [
        mpatches.Patch(color="#2ecc71", label="Below global KAGNN MedAE"),
        mpatches.Patch(color="#e74c3c", label="Above global KAGNN MedAE"),
    ]
    fig, ax = plt.subplots(figsize=(fw, fh))
    bars = ax.barh(a_s["Class"], a_s["MedAE_final"], color=bcolors, edgecolor="white", height=0.7)
    for bar, (_, row) in zip(bars, a_s.iterrows()):
        w = bar.get_width()
        ax.text(
            w + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}s  (n={int(row['n'])})",
            va="center", fontsize=8.5, color="#2c3e50",
        )
    ax.axvline(gme_f, color="#2c3e50", linestyle="--", lw=1.8,
               label=f"Global KAGNN MedAE={gme_f:.1f}s")
    ax.legend(
        handles=patches + [ax.get_legend_handles_labels()[0][-1]],
        fontsize=10, loc="lower right",
    )
    ax.set_xlabel("MedAE (s)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Class-wise MedAE — PGM+KAGNN (KAGIN) [{label}]",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, float(a_s["MedAE_final"].max()) * 1.28)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        config.results_dir / "plots" / f"cw_MedAE_{suffix}.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # ── Plot 2: Grouped PGM vs KAGNN ─────────────────────────────────────────
    fw2, fh2 = _safe_figsize(n_cls, base_w=20.0)
    a_g  = agg.sort_values("MedAE_pgm").reset_index(drop=True)
    y_pos = np.arange(len(a_g))
    bw    = 0.38
    fig, axes2 = plt.subplots(1, 2, figsize=(fw2, fh2))
    fig.suptitle(
        f"PGM Only vs PGM+KAGNN (KAGIN) [{label}]", fontsize=14, fontweight="bold"
    )
    for ax2, mp, mf, gp, gf, unit, title in [
        (axes2[0], "MedAE_pgm", "MedAE_final", gme_p, gme_f, "s", "MedAE"),
        (axes2[1], "MedRE_pgm", "MedRE_final", gmr_p, gmr_f, "%", "MedRE"),
    ]:
        ax2.barh(y_pos - bw / 2, a_g[mp], bw, label="PGM Only",
                 color="royalblue", edgecolor="white", alpha=0.8)
        ax2.barh(y_pos + bw / 2, a_g[mf], bw, label="PGM+KAGNN",
                 color="mediumseagreen", edgecolor="white", alpha=0.8)
        for i2, (_, row) in enumerate(a_g.iterrows()):
            delta = row[mp] - row[mf]
            sign  = "↓" if delta > 0 else "↑"
            clr   = "#27ae60" if delta > 0 else "#c0392b"
            ax2.text(
                max(row[mp], row[mf]) + 0.3, i2,
                f"{sign}{abs(delta):.1f}{unit}",
                va="center", fontsize=7.5, color=clr, fontweight="bold",
            )
        ax2.axvline(gp, color="royalblue",      linestyle=":",  lw=1.5, alpha=0.7)
        ax2.axvline(gf, color="mediumseagreen", linestyle="--", lw=1.5, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(a_g["Class"], fontsize=8.5)
        ax2.set_xlabel(f"{title} ({unit})", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8.5)
        ax2.invert_yaxis()
        ax2.set_title(f"{title}: PGM vs KAGNN", fontsize=13, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(
        config.results_dir / "plots" / f"cw_grouped_{suffix}.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # ── Plot 3: Size vs error scatter ─────────────────────────────────────────
    fig, axes3 = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Class Size vs Error — Majority Bias [{label}]", fontsize=14, fontweight="bold"
    )
    for ax3, col, gval, unit in [
        (axes3[0], "MedAE_final", gme_f, "s"),
        (axes3[1], "MedRE_final", gmr_f, "%"),
    ]:
        sc = ax3.scatter(
            agg["n"], agg[col], c=agg[col], cmap="RdYlGn_r",
            s=80, edgecolors="grey", linewidths=0.5, alpha=0.85,
        )
        for _, row in agg.iterrows():
            ax3.annotate(
                row["Class"][:18], (row["n"], row[col]),
                textcoords="offset points", xytext=(5, 3),
                fontsize=6.5, color="#2c3e50",
            )
        ax3.axhline(gval, color="#2c3e50", linestyle="--", lw=1.5)
        ax3.set_xlabel("Class size (n)",       fontsize=12, fontweight="bold")
        ax3.set_ylabel(f"{col[:5]} ({unit})",  fontsize=12, fontweight="bold")
        ax3.set_title(f"n vs {col[:5]}",       fontsize=13, fontweight="bold")
        ax3.grid(True, alpha=0.3, linestyle="--")
        plt.colorbar(sc, ax=ax3, label=f"{col[:5]} ({unit})")
    plt.tight_layout()
    plt.savefig(
        config.results_dir / "plots" / f"cw_size_{suffix}.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    # ── Plot 4: ECDF of absolute errors ───────────────────────────────────────
    fig, ax4 = plt.subplots(figsize=(10, 6))
    for ae, label_str, color in [
        (df["ae_pgm"].values,   "PGM Only",   "royalblue"),
        (df["ae_final"].values, "PGM+KAGNN", "mediumseagreen"),
    ]:
        sorted_ae = np.sort(ae)
        cdf = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae)
        ax4.plot(sorted_ae, cdf * 100, label=label_str, color=color, lw=2)
    for thresh, ls in [(10, "--"), (30, "-."), (60, ":")]:
        ax4.axvline(thresh, color="grey", linestyle=ls, lw=1, alpha=0.6, label=f"±{thresh}s")
    ax4.set_xlabel("Absolute Error (s)",            fontsize=12, fontweight="bold")
    ax4.set_ylabel("Cumulative % of predictions",   fontsize=12, fontweight="bold")
    ax4.set_title(f"ECDF of Absolute Errors [{label}]", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 200)
    plt.tight_layout()
    plt.savefig(
        config.results_dir / "plots" / f"ecdf_{suffix}.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    return agg, gme_f, gmr_f


def majority_bias_verdict(agg, metrics_final, mol_dict, cid_list):
    cls_counts   = Counter(
        assign_chemical_class(mol_dict[c]) for c in cid_list if c in mol_dict
    )
    majority_cls = cls_counts.most_common(1)[0][0]
    majority_pct = cls_counts[majority_cls] / len(cid_list) * 100
    gme          = metrics_final["MedAE"]
    above = agg[agg["MedAE_final"] >  gme]
    below = agg[agg["MedAE_final"] <= gme]
    if majority_cls in agg["Class"].values:
        maj = agg[agg["Class"] == majority_cls].iloc[0]
        print(
            f"\n  Majority: {majority_cls} ({majority_pct:.1f}%) "
            f"MedAE: PGM {maj['MedAE_pgm']:.1f}s → KAGNN {maj['MedAE_final']:.1f}s"
        )
    print("  Well predicted (≤ global): " + ", ".join(
        f"{r['Class'][:14]}({r['MedAE_final']:.0f}s)" for _, r in below.iterrows()
    ))
    if len(above) > 0:
        print("  Harder (> global): " + ", ".join(
            f"{r['Class'][:14]}({r['MedAE_final']:.0f}s)" for _, r in above.iterrows()
        ))
    if majority_cls in agg["Class"].values:
        maj = agg[agg["Class"] == majority_cls].iloc[0]
        print(
            f"  Majority-bias verdict: "
            f"{'No bias detected.' if maj['MedAE_final'] <= gme else 'Majority-class pull detected.'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fold summary plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_fold(y_true, y_pgm, y_final, history, fold, config):
    e_p = np.abs(y_true - y_pgm)
    e_f = np.abs(y_true - y_final)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Fold {fold} — PGM vs PGM+KAGNN (KAGIN)", fontsize=16, fontweight="bold"
    )
    lims = [
        min(y_true.min(), y_pgm.min(), y_final.min()),
        max(y_true.max(), y_pgm.max(), y_final.max()),
    ]
    for ax, yp, color, title in [
        (axes[0, 0], y_pgm,   "royalblue",
         f"PGM Only  R²={r2_score(y_true, y_pgm):.4f}"),
        (axes[0, 1], y_final, "mediumseagreen",
         f"KAGNN     R²={r2_score(y_true, y_final):.4f}"),
    ]:
        ax.scatter(y_true, yp, alpha=0.3, s=15, color=color, edgecolors="none")
        ax.plot(lims, lims, "r--", lw=2)
        ax.set_xlabel("True RT (s)", fontsize=11)
        ax.set_ylabel("Predicted RT (s)", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)

    axes[0, 2].scatter(
        y_true, y_final - y_true, alpha=0.2, s=10, color="mediumseagreen", edgecolors="none"
    )
    axes[0, 2].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[0, 2].set_xlabel("True RT (s)", fontsize=11)
    axes[0, 2].set_ylabel("Residual (s)", fontsize=11)
    axes[0, 2].set_title("KAGNN Residuals vs True RT", fontweight="bold")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].hist(e_p, bins=60, alpha=0.6, color="royalblue",
                    label=f"PGM MedAE={np.median(e_p):.1f}s")
    axes[1, 0].hist(e_f, bins=60, alpha=0.6, color="mediumseagreen",
                    label=f"KAGNN MedAE={np.median(e_f):.1f}s")
    axes[1, 0].set_xlabel("Absolute Error (s)", fontsize=11)
    axes[1, 0].set_ylabel("Count", fontsize=11)
    axes[1, 0].set_title("Error Distribution", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    if history["train_loss"]:
        axes[1, 1].plot(history["train_loss"], label="Train", color="steelblue")
        axes[1, 1].plot(history["val_loss"],   label="Val",   color="darkorange")
        axes[1, 1].set_xlabel("Epoch", fontsize=11)
        axes[1, 1].set_ylabel("SmoothL1 Loss", fontsize=11)
        axes[1, 1].set_title("Training Curves", fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    if history["lr"]:
        axes[1, 2].plot(history["lr"], color="purple")
        axes[1, 2].set_xlabel("Epoch", fontsize=11)
        axes[1, 2].set_ylabel("Learning Rate", fontsize=11)
        axes[1, 2].set_title("LR Schedule", fontweight="bold")
        axes[1, 2].set_yscale("log")
        axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        config.results_dir / "plots" / f"fold{fold}_summary.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"  Fold {fold} summary plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main cross-validation loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    config     = Config()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | XGB: {xgb_device}")

    data_loader_obj = SMRTDataLoader(config.csv_path, config.ecfp_path, config.sdf_path)
    df        = data_loader_obj.df
    ecfp_dict = data_loader_obj.ecfp_dict
    mol_dict  = data_loader_obj.mol_dict

    print("Assigning chemical classes...", end="", flush=True)
    df["chemical_class"] = [
        assign_chemical_class(mol_dict[int(cid)]) for cid in df["pubchem_id"]
    ]
    print(f" done. Distribution:\n{df['chemical_class'].value_counts().to_string()}")

    kf          = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    all_results = []
    fold_metrics = {"pgm": [], "kagnn": []}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(df), 1):
        print(f"\n{'=' * 60}\nFOLD {fold}/{config.n_folds}\n{'=' * 60}")
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]

        ds_tr = PrecomputedSMRTDataset(df_tr, ecfp_dict, mol_dict, f"fold{fold}-train")
        ds_va = PrecomputedSMRTDataset(df_va, ecfp_dict, mol_dict, f"fold{fold}-val")

        # num_workers=2: each worker holds the full precomputed graph dict in RAM.
        loader_kwargs = dict(
            batch_size=config.batch_size, collate_fn=collate_fn,
            num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        )
        tr_loader = DataLoader(ds_tr, shuffle=True,  **loader_kwargs)
        va_loader = DataLoader(ds_va, shuffle=False, **loader_kwargs)

        pgm_xgb, pgm_br, pgm_scaler, pgm_dur = train_pgm_stage(
            tr_loader, va_loader, mol_dict, config, xgb_device
        )

        # Cache PGM predictions to avoid redundant loader passes
        pgm_tr_dict = get_pgm_pred_dict(tr_loader, mol_dict, pgm_xgb, pgm_br, pgm_scaler)
        pgm_va_dict = get_pgm_pred_dict(va_loader, mol_dict, pgm_xgb, pgm_br, pgm_scaler)

        model, history, kagin_dur = train_kagin(
            tr_loader, va_loader, pgm_tr_dict, pgm_va_dict, config, device, fold
        )

        y_true, y_pgm, y_final, cid_list = run_inference(
            va_loader, model, pgm_va_dict, data_loader_obj, device, config
        )

        m_pgm   = compute_metrics(y_true, y_pgm)
        m_final = compute_metrics(y_true, y_final)
        print_metrics(m_pgm,   f"Fold {fold} — PGM Only")
        print_metrics(m_final, f"Fold {fold} — PGM+KAGNN")

        ae_pgm   = np.abs(y_true - y_pgm)
        ae_final = np.abs(y_true - y_final)
        t_stat, p_t = ttest_rel(ae_pgm, ae_final)
        _, p_w      = wilcoxon(ae_pgm, ae_final)
        print(f"  Paired t-test: t={t_stat:.4f} p={p_t:.4e}")
        print(f"  Wilcoxon:      W-stat p={p_w:.4e}")

        fold_metrics["pgm"].append(m_pgm)
        fold_metrics["kagnn"].append(m_final)

        # Build O(1) lookup dict once — avoids O(n²) df scan per molecule
        cid_to_class = dict(zip(df["pubchem_id"].astype(int), df["chemical_class"]))
        fold_df = pd.DataFrame({
            "y_true":         y_true,
            "y_pgm":          y_pgm,
            "y_final":        y_final,
            "pubchem_id":     cid_list,
            "chemical_class": [cid_to_class.get(c, "Unknown") for c in cid_list],
        })
        agg, gme, gmr = classwise_analysis_cv(fold_df, config, fold=fold)
        majority_bias_verdict(agg, m_final, mol_dict, cid_list)
        plot_fold(y_true, y_pgm, y_final, history, fold, config)

        fold_df["fold"] = fold
        all_results.append(fold_df)

        # Free GPU memory between folds — prevents cross-fold OOM accumulation
        del model, ds_tr, ds_va, tr_loader, va_loader
        del pgm_xgb, pgm_br, pgm_tr_dict, pgm_va_dict
        torch.cuda.empty_cache()

        fold_out = {
            "fold":         fold,
            "pgm_dur_s":    pgm_dur,
            "kagin_dur_s":  kagin_dur,
            "metrics_pgm":  m_pgm,
            "metrics_kagnn": m_final,
            "p_ttest":      float(p_t),
            "p_wilcoxon":   float(p_w),
        }
        with open(config.results_dir / f"fold{fold}_metrics.json", "w") as f:
            json.dump(fold_out, f, indent=2)
        print(f"  Fold {fold} metrics saved.")

    print(f"\n{'=' * 60}\nAGGREGATE RESULTS ({config.n_folds}-FOLD CV)\n{'=' * 60}")
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(config.results_dir / "all_fold_predictions.csv", index=False)

    agg_all, _, _ = classwise_analysis_cv(all_df, config, aggregate=True)
    majority_bias_verdict(
        agg_all,
        compute_metrics(all_df.y_true.values, all_df.y_final.values),
        mol_dict,
        all_df.pubchem_id.tolist(),
    )

    print(f"\n  {'Metric':<15} {'PGM mean±std':>18} {'KAGNN mean±std':>18}")
    print("  " + "-" * 55)
    for key in ["MedAE", "MAE", "RMSE", "R2", "Pct_le_10s", "Pct_le_30s", "Pct_le_60s"]:
        pgm_vals   = [m[key] for m in fold_metrics["pgm"]]
        kagnn_vals = [m[key] for m in fold_metrics["kagnn"]]
        print(
            f"  {key:<15} "
            f"{np.mean(pgm_vals):>8.3f}±{np.std(pgm_vals):.3f}   "
            f"{np.mean(kagnn_vals):>8.3f}±{np.std(kagnn_vals):.3f}"
        )

    def _agg(ms, fn):
        return {k: float(fn([m[k] for m in ms])) for k in ms[0]}

    aggregate_out = {
        "n_folds":    config.n_folds,
        "pgm_mean":   _agg(fold_metrics["pgm"],   np.mean),
        "pgm_std":    _agg(fold_metrics["pgm"],   np.std),
        "kagnn_mean": _agg(fold_metrics["kagnn"], np.mean),
        "kagnn_std":  _agg(fold_metrics["kagnn"], np.std),
    }
    with open(config.results_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_out, f, indent=2)

    print(f"\nAll outputs saved to: {config.results_dir}")


if __name__ == "__main__":
    main()
