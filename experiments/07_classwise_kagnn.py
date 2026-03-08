"""
07_classwise_kagnn.py — KA-GNN Baseline with Class-wise Analysis
=================================================================
5×GATv2 backbone + KAN ECFP encoder + KAN predictor head.

Extends the baseline KA-GNN (01_baseline_kagnn.py) with:
  • Chemical-class assignment heuristic (Lipids, Benzenoids, etc.)
  • Deep EDA: class imbalance, per-class RT violin plots, descriptor KDEs
  • Class-wise MedAE / MedRE performance table + horizontal bar charts
  • Majority-bias diagnostic (size-vs-error scatter + verdict)
  • All outputs written to results/classwise_kagnn/

GPU optimisations (architecture/logic unchanged vs baseline):
  A. AMP (fp16) via torch.amp.autocast + GradScaler     — ~1.5× speedup
  B. PrecomputedSMRTDataset — graphs built once, not per __getitem__
  C. num_workers=0 (avoids worker-fork RAM duplication of graph dict)
  D. torch.backends.cudnn.benchmark = True              — cuDNN auto-tune
  E. torch.cuda.empty_cache() at phase boundaries       — less fragmentation
  F. OOM guard in train loop                            — graceful batch skip

Usage:
    python experiments/07_classwise_kagnn.py

Paths default to data/raw/. Override via CLI:
    python experiments/07_classwise_kagnn.py \
        --csv  data/raw/SMRT_dataset.csv \
        --ecfp data/raw/SMRT_ECFP_1024_Fingerprints.txt \
        --sdf  data/raw/SMRT_dataset.sdf \
        --out  results/classwise_kagnn
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ── Optional imports ──────────────────────────────────────────────────────────
# FastKAN (RBF) — AMP-compatible, preferred over B-spline KAN
try:
    from fastkan import FastKAN as _KAN_CLS
    _KAN_BACKEND = "FastKAN (RBF)"
except ImportError:
    _KAN_CLS = None
    _KAN_BACKEND = "unavailable"

# Official KAGNN featurization (clone https://github.com/RomanBresson/KAGNN.git)
_KAGNN_AVAILABLE = False
_ATOM_FDIM, _BOND_FDIM = 133, 14
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "KAGNN" / "graph_regression"))
    from featurization import get_atom_fdim, get_bond_fdim, mol2graph  # noqa: F401
    _KAGNN_AVAILABLE = True
    _ATOM_FDIM = get_atom_fdim()
    _BOND_FDIM  = get_bond_fdim()
    print(f"✓ Official KAGNN featurization | Atom: {_ATOM_FDIM} | Bond: {_BOND_FDIM}")
except ImportError:
    print("  Official KAGNN not found — using local featurization fallback")

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True  # GPU OPT D


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Paths (overridden by CLI args)
    csv_path:  Path = Path("data/raw/SMRT_dataset.csv")
    ecfp_path: Path = Path("data/raw/SMRT_ECFP_1024_Fingerprints.txt")
    sdf_path:  Path = Path("data/raw/SMRT_dataset.sdf")
    results_dir: Path = Path("results/classwise_kagnn")

    # Architecture
    hidden_dim: int   = 256
    gnn_layers: int   = 5
    dropout: float    = 0.1
    batch_size: int   = 128
    epochs: int       = 150

    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float  = 1e-5
    grad_clip: float     = 1.0

    # Scheduler
    lr_factor: float   = 0.5
    lr_patience: int   = 8
    min_lr: float      = 5e-6

    # Early stopping
    early_stop_patience: int = 20

    # Splits
    val_size: float  = 0.15
    test_size: float = 0.15
    seed: int        = 42

    # Analysis
    min_class_samples: int = 10
    num_workers: int       = 0   # see module docstring

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        (self.results_dir / "plots" / "eda").mkdir(parents=True, exist_ok=True)
        self.use_amp     = torch.cuda.is_available()
        self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Results dir : {self.results_dir}")
        print(f"  AMP (fp16)  : {'ON' if self.use_amp else 'OFF'} | "
              f"batch: {self.batch_size} | workers: {self.num_workers}")
        print(f"  KAN backend : {_KAN_BACKEND}")

    # Allow dataclass-style construction without @dataclass decorator
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init__()


def _kan(layers):
    """Instantiate KAN (FastKAN if available, else raise)."""
    if _KAN_CLS is None:
        raise ImportError(
            "FastKAN not found. Install with:\n"
            "  pip install git+https://github.com/ZiyaoLi/fast-kan.git"
        )
    return _KAN_CLS(layers_hidden=layers, num_grids=5)


# =============================================================================
# Chemical-class heuristic
# =============================================================================

CLASS_PALETTE = {
    "Organoheterocyclics": "#2196F3",
    "Benzenoids":          "#4CAF50",
    "Organic Acids & AA":  "#FF5722",
    "Lipids":              "#9C27B0",
    "Carbohydrates":       "#FF9800",
    "Aliphatic Organics":  "#00BCD4",
    "Other":               "#795548",
    "Unknown":             "#9E9E9E",
}


def assign_chemical_class(mol) -> str:
    """Heuristic chemical-class assignment based on physicochemical descriptors."""
    try:
        mw     = Descriptors.MolWt(mol)
        logp   = Descriptors.MolLogP(mol)
        tpsa   = Descriptors.TPSA(mol)
        n_ar   = Descriptors.NumAromaticRings(mol)
        fsp3   = Descriptors.FractionCSP3(mol)
        hbd    = Descriptors.NumHDonors(mol)
        n_ring = Descriptors.RingCount(mol)
        if mw > 350 and logp > 4 and tpsa < 80 and fsp3 > 0.6:          return "Lipids"
        if tpsa > 120 and logp < -0.5 and n_ar == 0 and mw < 500:       return "Carbohydrates"
        if tpsa > 100 and logp < 1 and hbd >= 2:                        return "Organic Acids & AA"
        if n_ar >= 1 and Descriptors.NumHeteroatoms(mol) >= 2:          return "Organoheterocyclics"
        if n_ar >= 1:                                                    return "Benzenoids"
        if n_ring == 0 and fsp3 > 0.5:                                  return "Aliphatic Organics"
        return "Other"
    except Exception:
        return "Unknown"


# =============================================================================
# Local featurization fallback
# =============================================================================

def _atom_features_local(atom):
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


def _bond_features_local(bond):
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
    f.extend([
        int(stereo == Chem.rdchem.BondStereo.STEREONONE),
        int(stereo == Chem.rdchem.BondStereo.STEREOANY),
        int(stereo == Chem.rdchem.BondStereo.STEREOZ),
        int(stereo == Chem.rdchem.BondStereo.STEREOE),
        int(stereo == Chem.rdchem.BondStereo.STEREOCIS),
        int(stereo == Chem.rdchem.BondStereo.STEREOTRANS),
    ])
    bd = bond.GetBondDir()
    f.extend([
        int(bd == Chem.rdchem.BondDir.ENDUPRIGHT),
        int(bd == Chem.rdchem.BondDir.ENDDOWNRIGHT),
    ])
    return f


# =============================================================================
# Data loading
# =============================================================================

class SMRTDataLoader:
    """Loads CSV + ECFP fingerprints + SDF molecules and merges to common CIDs."""

    def __init__(self, csv_path: Path, ecfp_path: Path, sdf_path: Path):
        print("=" * 60)
        print("LOADING SMRT DATASET")
        print("=" * 60)

        # ── CSV ───────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower()
        id_col = next((c for c in df.columns
                       if "pubchem" in c or "cid" in c or "#" in c), df.columns[0])
        rt_col = next((c for c in df.columns if "rt" in c or "retention" in c), None)
        df = df.rename(columns={id_col: "pubchem_id", rt_col: "rt"})
        df["pubchem_id"] = pd.to_numeric(df["pubchem_id"], errors="coerce")
        df["rt"]         = pd.to_numeric(df["rt"],         errors="coerce")
        self.df = df.dropna(subset=["pubchem_id", "rt"]).reset_index(drop=True)
        print(f"  CSV rows    : {len(self.df):,}")

        # ── ECFP ─────────────────────────────────────────────────────────
        print("Loading ECFP fingerprints...")
        self.ecfp_dict: dict[int, np.ndarray] = {}
        with open(ecfp_path, "r") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        cid = int(parts[0].replace("ID=", ""))
                        fp  = np.array([int(b) for b in parts[1][:1024]], dtype=np.float32)
                        self.ecfp_dict[cid] = fp
                    except Exception:
                        continue
        print(f"  Fingerprints: {len(self.ecfp_dict):,}")

        # ── SDF ──────────────────────────────────────────────────────────
        print("Loading SDF structures...")
        self.mol_dict: dict[int, Chem.Mol] = {}
        suppl  = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        valid, errors = 0, 0
        for i, mol in enumerate(suppl):
            if mol is None:
                errors += 1
                continue
            try:
                cid = int(mol.GetProp("PUBCHEM_COMPOUND_CID"))
                Chem.SanitizeMol(mol)
                self.mol_dict[cid] = mol
                valid += 1
            except Exception:
                errors += 1
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} processed — valid: {valid:,} / errors: {errors:,}")
        print(f"  SDF complete : {valid:,} valid, {errors:,} skipped")

        # ── Intersect ─────────────────────────────────────────────────────
        common = (set(self.df.pubchem_id) &
                  set(self.ecfp_dict)     &
                  set(self.mol_dict))
        self.df        = self.df[self.df.pubchem_id.isin(common)].reset_index(drop=True)
        self.ecfp_dict = {k: v for k, v in self.ecfp_dict.items() if k in common}
        self.mol_dict  = {k: v for k, v in self.mol_dict.items()  if k in common}
        print(f"  Intersection : {len(self.df):,} molecules")

        # ── Chemical classes ──────────────────────────────────────────────
        print("Assigning chemical classes...")
        self.df["chemical_class"] = [
            assign_chemical_class(self.mol_dict[int(cid)])
            for cid in self.df.pubchem_id
        ]
        print("  Class distribution:")
        for cls, cnt in self.df["chemical_class"].value_counts().items():
            print(f"    {cls:<30} {cnt:>6}  ({cnt/len(self.df)*100:.1f}%)")

        # ── RT normalisation ──────────────────────────────────────────────
        self.rt_scaler     = RobustScaler()
        self.df["rt_norm"] = self.rt_scaler.fit_transform(
            self.df[["rt"]]).flatten().astype(np.float32)

        print(f"✓ Dataset ready: {len(self.df):,} molecules")
        print("=" * 60)

    def denormalize(self, rt_norm: np.ndarray) -> np.ndarray:
        return self.rt_scaler.inverse_transform(
            np.array(rt_norm, dtype=np.float32).reshape(-1, 1)
        ).flatten()


# =============================================================================
# EDA
# =============================================================================

def deep_eda(df: pd.DataFrame, mol_dict: dict, config: Config) -> pd.DataFrame:
    """Produce three EDA figures and an elution-tendency table."""
    print("\n" + "=" * 60)
    print("DEEP EDA — PRE-TRAINING DATA CHARACTERISATION")
    print("=" * 60)

    eda_dir = config.results_dir / "plots" / "eda"

    # Build EDA dataframe via vectorised descriptor extraction
    cids = df["pubchem_id"].astype(int).tolist()
    rows = []
    for cid in cids:
        mol = mol_dict.get(cid)
        if mol is None:
            continue
        try:
            rows.append({
                "MW":   Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "_cid": cid,
            })
        except Exception:
            continue

    _rt_map  = dict(zip(df["pubchem_id"].astype(int), df["rt"]))
    _cls_map = dict(zip(df["pubchem_id"].astype(int), df["chemical_class"]))
    for r in rows:
        r["rt"]             = _rt_map.get(r["_cid"], np.nan)
        r["chemical_class"] = _cls_map.get(r["_cid"], "Unknown")
    eda_df    = pd.DataFrame(rows).drop(columns=["_cid"]).dropna(subset=["rt"])
    cls_order = eda_df["chemical_class"].value_counts().index.tolist()
    palette   = CLASS_PALETTE

    # ── Figure 1: class imbalance ─────────────────────────────────────────
    counts = eda_df["chemical_class"].value_counts()
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[palette.get(c, "#607D8B") for c in counts.index],
                  edgecolor="white", linewidth=0.7)
    for bar, (cls, cnt) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 60,
                f"{cnt:,}\n({cnt/len(eda_df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Number of Molecules", fontsize=13, fontweight="bold")
    ax.set_xlabel("Chemical Class",      fontsize=13, fontweight="bold")
    ax.set_title("Class Imbalance — SMRT Dataset", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(eda_dir / "eda_01_class_imbalance.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: eda_01_class_imbalance.png")

    # ── Figure 2: RT violin + box per class ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(6, len(cls_order) * 0.75)))
    sns.violinplot(data=eda_df, x="chemical_class", y="rt", order=cls_order,
                   palette=palette, inner=None, alpha=0.5, ax=ax, cut=0)
    sns.boxplot(data=eda_df, x="chemical_class", y="rt", order=cls_order,
                palette=palette, width=0.22, fliersize=1.5,
                boxprops=dict(alpha=0.9), ax=ax)
    gm = eda_df["rt"].median()
    ax.axhline(gm, color="black", linestyle="--", linewidth=1.8,
               label=f"Global median RT = {gm:.0f} s")
    ax.set_xlabel("Chemical Class",     fontsize=13, fontweight="bold")
    ax.set_ylabel("Retention Time (s)", fontsize=13, fontweight="bold")
    ax.set_title("RT Distribution per Chemical Class", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(eda_dir / "eda_02_rt_per_class.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: eda_02_rt_per_class.png")

    # ── Figure 3: MW / LogP / TPSA KDEs per class ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, feat, unit, xlim in zip(
        axes,
        ["MW",    "LogP", "TPSA"],
        ["Da",    "",     "Å²"],
        [(0,900), (-5,12),(0,300)],
    ):
        for cls in cls_order:
            sub = eda_df[eda_df["chemical_class"] == cls][feat].dropna()
            if len(sub) > 10:
                sub.plot.kde(ax=ax, label=cls,
                             color=palette.get(cls, "#607D8B"),
                             linewidth=2.2, alpha=0.85)
        ax.set_xlabel(f"{feat} ({unit})" if unit else feat,
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax.set_title(f"{feat} Distribution", fontsize=13, fontweight="bold")
        ax.set_xlim(xlim)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(eda_dir / "eda_03_descriptor_kde.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: eda_03_descriptor_kde.png")

    # ── Elution tendency table ────────────────────────────────────────────
    global_median = eda_df["rt"].median()
    print(f"\n  {'Class':<30} {'n':>6}  {'Median':>10}  {'IQR':>10}  "
          f"{'Delta':>9}  Tendency")
    print("  " + "-" * 82)
    for cls in cls_order:
        sub   = eda_df[eda_df["chemical_class"] == cls]["rt"]
        q1    = sub.quantile(0.25)
        q3    = sub.quantile(0.75)
        delta = sub.median() - global_median
        tend  = ("earlier elution" if delta < -30 else
                 "later elution"   if delta >  30 else
                 "near-global median")
        print(f"  {cls:<30} {len(sub):>6}  {sub.median():>9.0f}s  "
              f"{(q3-q1):>9.0f}s  {delta:>+8.0f}s  {tend}")

    return eda_df


# =============================================================================
# Graph construction
# =============================================================================

def _build_graph(mol: Chem.Mol) -> Data:
    """Build a PyG Data object from an RDKit mol (called once per molecule)."""
    if mol.GetNumAtoms() == 0:
        return Data(
            x=torch.zeros((1, _ATOM_FDIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, _BOND_FDIM), dtype=torch.float32),
        )

    if _KAGNN_AVAILABLE:
        gd         = mol2graph(mol)
        x          = torch.tensor(gd["node_features"], dtype=torch.float32)
        raw_ei     = gd["edge_index"]
        raw_ea     = gd["edge_features"]
        if (raw_ei is None or len(raw_ei) == 0 or
                (hasattr(raw_ei, "__len__") and len(raw_ei) > 0 and
                 hasattr(raw_ei[0], "__len__") and len(raw_ei[0]) == 0)):
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, _BOND_FDIM), dtype=torch.float32)
        else:
            edge_index = torch.tensor(raw_ei, dtype=torch.long)
            edge_attr  = torch.tensor(raw_ea, dtype=torch.float32)
            if edge_index.dim() == 2 and edge_index.shape[0] != 2:
                edge_index = edge_index.t().contiguous()
    else:
        x  = torch.tensor(
            [_atom_features_local(a) for a in mol.GetAtoms()],
            dtype=torch.float32,
        )
        ei, ea = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf   = _bond_features_local(bond)
            ei.extend([[i, j], [j, i]])
            ea.extend([bf, bf])
        if len(ei) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, _BOND_FDIM), dtype=torch.float32)
        else:
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(ea, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# Dataset  (GPU OPT B — pre-compute graphs once)
# =============================================================================

class PrecomputedSMRTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ecfp_dict: dict, mol_dict: dict,
                 desc: str = ""):
        self.df        = df.reset_index(drop=True)
        self.ecfp_dict = ecfp_dict
        unique_cids    = list(dict.fromkeys(self.df["pubchem_id"].astype(int).tolist()))
        print(f"  Pre-computing {len(unique_cids):,} graphs [{desc}]...",
              end="", flush=True)
        t0           = time.time()
        self.graphs  = {cid: _build_graph(mol_dict[cid]) for cid in unique_cids}
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
    return Batch.from_data_list(graphs), torch.stack(ecfps), torch.stack(rts), ids


# =============================================================================
# Model
# =============================================================================

class KAGNNModel(nn.Module):
    """
    5×GATv2 backbone with KAN ECFP encoder and KAN predictor.

    KAN blocks (FastKAN, RBF):
      ecfp_kan  : [1024 → 512 → 256]
      predictor : [512  → 256 → 128 → 1]

    Atom/bond projections and GATv2 layers are identical to baseline v1.
    """

    def __init__(self, config: Config):
        super().__init__()
        h     = config.hidden_dim
        heads = 4

        self.atom_embed = nn.Linear(_ATOM_FDIM, h)
        self.bond_embed = nn.Linear(_BOND_FDIM, h // heads)

        self.gat1 = GATv2Conv(h, h // heads, heads=heads,
                              dropout=config.dropout, edge_dim=h // heads)
        self.gat2 = GATv2Conv(h, h // heads, heads=heads,
                              dropout=config.dropout, edge_dim=h // heads)
        self.gat3 = GATv2Conv(h, h // heads, heads=heads,
                              dropout=config.dropout, edge_dim=h // heads)
        self.gat4 = GATv2Conv(h, h // heads, heads=heads,
                              dropout=config.dropout, edge_dim=h // heads)
        self.gat5 = GATv2Conv(h, h, heads=1,
                              dropout=config.dropout, edge_dim=h // heads)

        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.norm3 = nn.LayerNorm(h)
        self.norm4 = nn.LayerNorm(h)
        self.norm5 = nn.LayerNorm(h)

        self.ecfp_ln  = nn.LayerNorm(1024)
        self.ecfp_kan = _kan([1024, h * 2, h])
        self.predictor = _kan([h * 2, h, h // 2, 1])

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ KAGNNModel | params: {n_params:,} | h={h} | "
              f"ATOM={_ATOM_FDIM} | BOND={_BOND_FDIM}")

    def forward(self, graph: Batch, ecfp: torch.Tensor) -> torch.Tensor:
        x = self.atom_embed(graph.x)
        e = self.bond_embed(graph.edge_attr)
        x = x + self.norm1(F.elu(self.gat1(x, graph.edge_index, e)))
        x = x + self.norm2(F.elu(self.gat2(x, graph.edge_index, e)))
        x = x + self.norm3(F.elu(self.gat3(x, graph.edge_index, e)))
        x = x + self.norm4(F.elu(self.gat4(x, graph.edge_index, e)))
        x =     self.norm5(F.elu(self.gat5(x, graph.edge_index, e)))
        g_emb = (global_mean_pool(x, graph.batch) +
                 global_max_pool(x, graph.batch)) / 2.0
        e_emb = self.ecfp_kan(self.ecfp_ln(ecfp))
        return self.predictor(torch.cat([g_emb, e_emb], dim=-1)).squeeze(-1)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_zero = int((y_true <= 0).sum())
    if n_zero > 0:
        print(f"   {n_zero} samples with RT ≤ 0 — clipping to 1e-3 for MedRE")
        y_true = np.clip(y_true, 1e-3, None)
    ae = np.abs(y_true - y_pred)
    return {
        "n_samples":  int(len(y_true)),
        "MedAE":      float(np.median(ae)),
        "MAE":        float(np.mean(ae)),
        "RMSE":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":         float(r2_score(y_true, y_pred)),
        "Pearson":    float(pearsonr(y_true, y_pred)[0]),
        "Spearman":   float(spearmanr(y_true, y_pred)[0]),
        "Pct_le_60s": float((ae <= 60).mean() * 100),
        "Pct_le_30s": float((ae <= 30).mean() * 100),
        "Pct_le_10s": float((ae <= 10).mean() * 100),
        "MedRE":      float(
            np.median(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100)
        ),
    }


# =============================================================================
# Training
# =============================================================================

def train_model(model: KAGNNModel, train_loader: DataLoader,
                val_loader: DataLoader, config: Config,
                device: torch.device) -> dict:

    print("\n" + "=" * 60)
    print("TRAINING KA-GNN (classwise experiment)")
    print(f"  AMP (fp16): {'ON' if config.use_amp else 'OFF'}")
    print("=" * 60)

    ckpt_path = config.results_dir / "checkpoints" / "best_kagnn_classwise.pth"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )
    scaler = GradScaler(device=config._device_str) if config.use_amp else None

    best_val_loss    = float("inf")
    patience_counter = 0
    start_epoch      = 1
    history          = {"train_loss": [], "val_loss": [], "lr": []}

    if ckpt_path.exists():
        print(f"  Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt.get("scheduler_state", {}))
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            start_epoch   = ckpt.get("epoch", 0) + 1
            history       = ckpt.get("history", history)
        else:
            model.load_state_dict(ckpt)

    torch.cuda.empty_cache()  # GPU OPT E
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for graph, ecfp, rt, _ in train_loader:
            graph = graph.to(device, non_blocking=True)
            ecfp  = ecfp.to(device,  non_blocking=True)
            rt    = rt.to(device,    non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast(device_type=config._device_str,
                                        enabled=config.use_amp):
                    loss = criterion(model(graph, ecfp), rt)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        optimizer.step()
                    train_loss += loss.item() * len(rt)
            except torch.cuda.OutOfMemoryError:  # GPU OPT F
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                print(f"  [OOM] batch skipped at epoch {epoch}")

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        torch.cuda.empty_cache()  # GPU OPT E

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph, ecfp, rt, _ in val_loader:
                graph = graph.to(device, non_blocking=True)
                ecfp  = ecfp.to(device,  non_blocking=True)
                rt    = rt.to(device,    non_blocking=True)
                with torch.amp.autocast(device_type=config._device_str,
                                        enabled=config.use_amp):
                    val_loss += criterion(model(graph, ecfp), rt).item() * len(rt)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        if lr_now < prev_lr - 1e-10:
            print(f"  LR reduced: {prev_lr:.2e} → {lr_now:.2e}  (epoch {epoch})")
            prev_lr = lr_now
        history["lr"].append(lr_now)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr_now:.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss":   best_val_loss,
                "epoch":           epoch,
                "history":         history,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val={best_val_loss:.5f})")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(
        ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )
    print(f"✓ Best checkpoint loaded (val={best_val_loss:.5f})")
    return history


# =============================================================================
# Plots
# =============================================================================

def plot_training_history(history: dict, config: Config) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   linewidth=2)
    axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("SmoothL1 Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].plot(history["val_loss"], color="darkorange", linewidth=2)
    axes[1].set_title("Validation Loss", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    axes[2].plot(history["lr"], color="purple", linewidth=2)
    axes[2].set_title("Learning Rate", fontsize=13, fontweight="bold")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("KA-GNN (Classwise) — Training History",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.results_dir / "plots" / "training_history.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: training_history.png")


def plot_test_results(y_true: np.ndarray, y_pred: np.ndarray,
                      config: Config) -> None:
    ae    = np.abs(y_true - y_pred)
    medae = float(np.median(ae))
    mae   = float(np.mean(ae))
    r2    = r2_score(y_true, y_pred)
    pct10 = (ae <= 10).mean() * 100
    pct30 = (ae <= 30).mean() * 100
    pct60 = (ae <= 60).mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("KA-GNN Classwise — Test Set Performance",
                 fontsize=18, fontweight="bold", y=1.01)

    # Scatter
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c="royalblue", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2.5, label="Perfect", zorder=10)
    ax.set_xlabel("True RT (s)",      fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted RT (s)", fontsize=13, fontweight="bold")
    ax.set_title(f"Predicted vs True\nMedAE={medae:.2f}s  R²={r2:.3f}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--"); ax.legend(fontsize=11)
    ax.text(0.05, 0.95,
            f"MedAE: {medae:.2f}s\nMAE:   {mae:.2f}s\nR²:    {r2:.3f}",
            transform=ax.transAxes, fontsize=12, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    # CDF
    ax  = axes[1]
    sa  = np.sort(ae)
    cdf = np.arange(1, len(sa) + 1) / len(sa)
    ax.plot(sa, cdf, linewidth=2.5, color="royalblue", alpha=0.8)
    ax.fill_between(sa, 0, cdf, alpha=0.2, color="royalblue")
    ax.axhline(0.5, color="purple", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.axvline(medae, color="royalblue", linestyle=":", linewidth=2)
    ax.text(medae + 3, 0.45, f"{medae:.1f}s",
            color="royalblue", fontweight="bold", fontsize=11)
    ax.set_xlabel("Absolute Error (s)",     fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Probability", fontsize=13, fontweight="bold")
    ax.set_title("CDF of Absolute Errors",  fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, min(150, sa.max())); ax.set_ylim(0, 1.05)

    # Threshold bars
    ax   = axes[2]
    bars = ax.bar(["≤10s", "≤30s", "≤60s"], [pct10, pct30, pct60],
                  color="royalblue", edgecolor="black", linewidth=1.5, alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("% Correct",       fontsize=13, fontweight="bold")
    ax.set_xlabel("Error Threshold", fontsize=13, fontweight="bold")
    ax.set_title("Threshold Performance", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(config.results_dir / "plots" / "test_results.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: test_results.png")


# =============================================================================
# Class-wise analysis
# =============================================================================

def classwise_analysis(results_df: pd.DataFrame,
                        config: Config) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("CLASS-WISE PERFORMANCE ANALYSIS")
    print("=" * 60)

    df = results_df.copy()
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["rel_error"] = (np.abs(df["y_true"] - df["y_pred"]) /
                       (np.abs(df["y_true"]) + 1e-8)) * 100

    agg = (
        df.groupby("chemical_class")
          .agg(
              n      = ("abs_error", "count"),
              MedAE  = ("abs_error", "median"),
              MAE    = ("abs_error", "mean"),
              MedRE  = ("rel_error", "median"),
              MeanRE = ("rel_error", "mean"),
          )
          .reset_index()
          .rename(columns={"chemical_class": "Class"})
    )
    agg = agg[agg["n"] >= config.min_class_samples].copy()
    agg["pct_of_test"] = agg["n"] / len(df) * 100

    global_medae = float(np.median(df["abs_error"]))
    global_medre = float(np.median(df["rel_error"]))

    print(f"\n  Global MedAE : {global_medae:.2f}s  |  "
          f"Global MedRE : {global_medre:.2f}%")
    print(f"  Classes shown: {len(agg)} (n ≥ {config.min_class_samples})\n")
    print(f"  {'Class':<35} {'n':>6} {'%Test':>6} {'MedAE':>8} {'MedRE':>8}")
    print("  " + "-" * 68)
    for _, row in agg.sort_values("MedAE").iterrows():
        print(f"  {row['Class']:<35} {int(row['n']):>6} "
              f"{row['pct_of_test']:>5.1f}% "
              f"{row['MedAE']:>7.2f}s {row['MedRE']:>7.2f}%")

    def _bar_colors(vals, threshold):
        return ["#2ecc71" if v <= threshold else "#e74c3c" for v in vals]

    fig_h   = max(6, len(agg) * 0.55)
    patches = [mpatches.Patch(color="#2ecc71", label="Below global MedAE"),
               mpatches.Patch(color="#e74c3c", label="Above global MedAE")]

    for metric, unit, gval in [
        ("MedAE", "s", global_medae),
        ("MedRE",  "%", global_medre),
    ]:
        agg_s  = agg.sort_values(metric, ascending=True).reset_index(drop=True)
        colors = _bar_colors(agg_s[metric], gval)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        bars    = ax.barh(agg_s["Class"], agg_s[metric],
                          color=colors, edgecolor="white",
                          linewidth=0.6, height=0.7)
        for bar, (_, row) in zip(bars, agg_s.iterrows()):
            w = bar.get_width()
            ax.text(w + (0.5 if unit == "s" else 0.2),
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}{unit}  (n={int(row['n'])})",
                    va="center", ha="left", fontsize=8.5, color="#2c3e50")
        ax.axvline(gval, color="#2c3e50", linestyle="--", linewidth=1.8,
                   label=f"Global {metric} = {gval:.1f}{unit}")
        ax.legend(handles=patches + [ax.get_legend_handles_labels()[0][-1]],
                  fontsize=10, loc="lower right")
        lbl = ("Median Absolute Error (s)" if unit == "s"
               else "Median Relative Error (%)")
        ax.set_xlabel(lbl, fontsize=13, fontweight="bold")
        ax.set_title(f"Class-wise {metric} — KA-GNN Classwise",
                     fontsize=14, fontweight="bold")
        ax.set_xlim(0, agg_s[metric].max() * 1.28)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()
        plt.tight_layout()
        fname = f"classwise_{metric}.png"
        plt.savefig(config.results_dir / "plots" / fname,
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {fname}")

    # Size-vs-error scatter
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Class Size vs Error — Majority-Bias Diagnostic",
                 fontsize=14, fontweight="bold")
    for ax, metric, unit, gval in [
        (axes[0], "MedAE", "s", global_medae),
        (axes[1], "MedRE",  "%", global_medre),
    ]:
        sc = ax.scatter(agg["n"], agg[metric],
                        c=agg[metric], cmap="RdYlGn_r",
                        s=80, edgecolors="grey", linewidths=0.5, alpha=0.85)
        for _, row in agg.iterrows():
            ax.annotate(row["Class"][:18], (row["n"], row[metric]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color="#2c3e50")
        ax.axhline(gval, color="#2c3e50", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Class size (n)",     fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{metric} ({unit})", fontsize=12, fontweight="bold")
        ax.set_title(f"n vs {metric}",      fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.colorbar(sc, ax=ax, label=f"{metric} ({unit})")
    plt.tight_layout()
    plt.savefig(config.results_dir / "plots" / "classwise_size_vs_error.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: classwise_size_vs_error.png")

    return agg


def majority_bias_verdict(agg: pd.DataFrame, metrics: dict,
                           mol_dict: dict, test_cids: list) -> None:
    print("\n" + "=" * 60)
    print("MAJORITY-BIAS DIAGNOSTIC SUMMARY")
    print("=" * 60)

    cls_counts   = Counter(assign_chemical_class(mol_dict[c]) for c in test_cids)
    majority_cls = cls_counts.most_common(1)[0][0]
    majority_pct = cls_counts[majority_cls] / len(test_cids) * 100
    global_medae = metrics["MedAE"]
    global_medre = metrics["MedRE"]

    above = agg[agg["MedAE"] >  global_medae]
    below = agg[agg["MedAE"] <= global_medae]

    if majority_cls in agg["Class"].values:
        maj = agg[agg["Class"] == majority_cls].iloc[0]
        print(f"\n  Majority class : {majority_cls} ({majority_pct:.1f}% of test)")
        print(f"  MedAE : {maj['MedAE']:.2f}s  (global: {global_medae:.2f}s)")
        print(f"  MedRE : {maj['MedRE']:.2f}%  (global: {global_medre:.2f}%)")

    print(f"\n  Classes BELOW global MedAE ({len(below)}) — well predicted:")
    for _, row in below.sort_values("MedAE").iterrows():
        print(f"     {row['Class']:<35} "
              f"MedAE={row['MedAE']:.2f}s  MedRE={row['MedRE']:.2f}%  "
              f"n={int(row['n'])}")

    print(f"\n  Classes ABOVE global MedAE ({len(above)}) — harder to predict:")
    for _, row in above.sort_values("MedAE", ascending=False).iterrows():
        print(f"     {row['Class']:<35} "
              f"MedAE={row['MedAE']:.2f}s  MedRE={row['MedRE']:.2f}%  "
              f"n={int(row['n'])}")

    if majority_cls in agg["Class"].values:
        maj     = agg[agg["Class"] == majority_cls].iloc[0]
        verdict = (
            "Model does NOT appear over-optimised for the majority class. "
            "Uniform SmoothL1 trained without bias."
            if maj["MedAE"] <= global_medae else
            "Potential majority-class pull detected. "
            "Consider mild class-weighted loss (max_ratio ≤ 3) in next experiment."
        )
        print(f"\n  Verdict: {verdict}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KA-GNN Classwise Experiment")
    p.add_argument("--csv",  default="data/raw/SMRT_dataset.csv")
    p.add_argument("--ecfp", default="data/raw/SMRT_ECFP_1024_Fingerprints.txt")
    p.add_argument("--sdf",  default="data/raw/SMRT_dataset.sdf")
    p.add_argument("--out",  default="results/classwise_kagnn")
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--hidden-dim",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        csv_path    = Path(args.csv),
        ecfp_path   = Path(args.ecfp),
        sdf_path    = Path(args.sdf),
        results_dir = Path(args.out),
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        hidden_dim  = args.hidden_dim,
        learning_rate = args.lr,
        seed        = args.seed,
    )

    # ── Load data ─────────────────────────────────────────────────────────
    data_loader = SMRTDataLoader(config.csv_path, config.ecfp_path, config.sdf_path)
    df          = data_loader.df

    # ── EDA ───────────────────────────────────────────────────────────────
    deep_eda(df, data_loader.mol_dict, config)

    # ── Splits ────────────────────────────────────────────────────────────
    val_fraction = config.val_size / (1 - config.test_size)
    indices      = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=config.test_size,
                                           random_state=config.seed)
    train_idx, val_idx  = train_test_split(train_idx, test_size=val_fraction,
                                           random_state=config.seed)

    # ── Device ────────────────────────────────────────────────────────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"\n Device     : {device}"
          f" | pin_memory: {pin_memory}"
          f" | num_workers: {config.num_workers}"
          f" | AMP: {config.use_amp}")

    # ── Datasets / loaders ────────────────────────────────────────────────
    train_ds = PrecomputedSMRTDataset(
        df.iloc[train_idx], data_loader.ecfp_dict, data_loader.mol_dict, "train")
    val_ds   = PrecomputedSMRTDataset(
        df.iloc[val_idx],   data_loader.ecfp_dict, data_loader.mol_dict, "val")
    test_ds  = PrecomputedSMRTDataset(
        df.iloc[test_idx],  data_loader.ecfp_dict, data_loader.mol_dict, "test")

    loader_kw = dict(
        batch_size  = config.batch_size,
        collate_fn  = collate_fn,
        num_workers = config.num_workers,
        pin_memory  = pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Build + train ─────────────────────────────────────────────────────
    model   = KAGNNModel(config).to(device)
    t0      = time.time()
    history = train_model(model, train_loader, val_loader, config, device)
    elapsed = time.time() - t0

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENCE ON TEST SET")
    print("=" * 60)
    torch.cuda.empty_cache()

    model.eval()
    y_true_l, y_pred_l, cid_list = [], [], []
    with torch.no_grad():
        for graph, ecfp, rt_norm, cids in test_loader:
            graph = graph.to(device, non_blocking=True)
            ecfp  = ecfp.to(device,  non_blocking=True)
            with torch.amp.autocast(device_type=config._device_str,
                                    enabled=config.use_amp):
                pred = model(graph, ecfp).cpu().float().numpy()
            y_true_l.append(rt_norm.numpy())
            y_pred_l.append(pred)
            cid_list.extend([int(c) for c in cids])

    y_true = data_loader.denormalize(np.concatenate(y_true_l))
    y_pred = data_loader.denormalize(np.concatenate(y_pred_l))

    # ── Overall metrics ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL TEST METRICS — KA-GNN Classwise")
    print("=" * 60)
    metrics = compute_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:10.4f}")
    with open(config.results_dir / "metrics_classwise_kagnn.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: metrics_classwise_kagnn.json")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_training_history(history, config)
    plot_test_results(y_true, y_pred, config)

    # ── Class-wise analysis ───────────────────────────────────────────────
    test_cids      = [int(df.iloc[i].pubchem_id) for i in test_idx]
    chem_class_map = {cid: assign_chemical_class(data_loader.mol_dict[cid])
                      for cid in test_cids}

    print(f"\n  Class distribution in test split (n={len(test_cids):,}):")
    cls_counts_test = Counter(chem_class_map.values())
    for cls, cnt in sorted(cls_counts_test.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / len(test_cids) * 30)
        print(f"  {cls:<35} {cnt:>5}  {cnt/len(test_cids)*100:>5.1f}%  {bar}")

    results_df = pd.DataFrame({
        "pubchem_id":     cid_list,
        "y_true":         y_true,
        "y_pred":         y_pred,
        "chemical_class": [chem_class_map.get(c, "Unknown") for c in cid_list],
    })

    class_summary = classwise_analysis(results_df, config)
    class_summary.to_csv(config.results_dir / "classwise_metrics.csv", index=False)
    print("  Saved: classwise_metrics.csv")

    majority_bias_verdict(class_summary, metrics, data_loader.mol_dict, test_cids)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"""
============================================================
  KA-GNN CLASSWISE EXPERIMENT — COMPLETE
============================================================
  Architecture  : 5×GATv2 + KAN ECFP encoder + KAN predictor
  KAN variant   : FastKAN (RBF, num_grids=5)
  ECFP encoder  : KAN([1024, 512, 256])
  Predictor     : KAN([512, 256, 128, 1])
  Loss          : SmoothL1 (uniform)
  Scheduler     : ReduceLROnPlateau
  AMP (fp16)    : {"ON" if config.use_amp else "OFF"}
  Training time : {elapsed/60:.1f} min
------------------------------------------------------------
  Test MedAE    : {metrics["MedAE"]:>7.2f} s
  Test MedRE    : {metrics["MedRE"]:>7.2f} %
  R² Score      : {metrics["R2"]:>7.4f}
  % within 30s  : {metrics["Pct_le_30s"]:>7.2f} %
------------------------------------------------------------
  Output directory: {config.results_dir}
    checkpoints/best_kagnn_classwise.pth
    plots/training_history.png
    plots/test_results.png
    plots/eda/  (3 EDA figures)
    plots/classwise_MedAE.png
    plots/classwise_MedRE.png
    plots/classwise_size_vs_error.png
    metrics_classwise_kagnn.json
    classwise_metrics.csv
============================================================
""")


if __name__ == "__main__":
    main()
