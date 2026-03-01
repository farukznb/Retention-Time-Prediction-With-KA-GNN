"""
kagnn_pgm_forward_hybrid.py
===========================
Forward Hybrid: KA-GNN → PGM Residual Correction
--------------------------------------------------
Experiment 2 — Class-wise Performance Analysis
METLIN SMRT Dataset (n = 79,955)

Architecture
------------
Stage 1  TRUE KA-GNN (primary predictor):
    • Dual-stream: 5-layer GATv2 (graph branch) + KAN ECFP encoder
    • KAN prediction head (Kolmogorov–Arnold, Gaussian RBF basis)
    • Class-weighted SmoothL1 loss  (sqrt-capped inverse-frequency, 10×)
    • ReduceLROnPlateau learning-rate schedule

Stage 2  PGM residual correction:
    • Features: 256-dim GNN embedding + ECFP-1024 + 32 RDKit descriptors
    • XGBoost + Bayesian Ridge ensemble
    • Final:  ŷ_final = ŷ_KA-GNN + r̂_PGM

Key Results (Experiment 1 — authoritative test split, n = 11,994)
------------------------------------------------------------------
    KA-GNN only    MedAE = 26.14 s   R² = 0.807
    Forward Hybrid MedAE = 20.45 s   R² = 0.834   (+21.8 % improvement)

Outputs (saved to Google Drive)
--------------------------------
    /content/drive/MyDrive/SMRT/results/kagnn_pgm_forward/
        plots/best_kagnn.pth
        plots/training_history.png
        plots/TRUE_KAGNN_Baseline_results.png
        plots/KAGNN_+_PGM_results.png
        plots/comprehensive_comparison.png
        plots/classwise_MedAE_hybrid.png
        plots/classwise_MedRE_hybrid.png
        plots/classwise_size_vs_error.png
        plots/classwise_kagnn_vs_hybrid_grouped.png
        metrics_kagnn.json
        metrics_final.json
        classwise_metrics.csv

Requirements
------------
    torch, torch-geometric, scikit-learn, scipy, matplotlib,
    seaborn, pandas, numpy, tqdm, xgboost, joblib, rdkit

Usage
-----
    Open in Google Colab (GPU runtime — A100 recommended).
    Run cells sequentially. All outputs are written to Drive automatically.

Author
------
    Farouk — AIMS Senegal, 2025
"""

# ─── Cell 1: Install Dependencies ────────────────────────────────────────────
!pip install torch torch-geometric scikit-learn scipy matplotlib seaborn pandas numpy tqdm xgboost joblib -q
!pip install rdkit 2>/dev/null || pip install rdkit-pypi 2>/dev/null

# ─── Cell 2: Mount Drive ──────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─── Cell 3: Imports ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import BayesianRidge
from scipy.stats import pearsonr, spearmanr, ttest_rel
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import joblib
from pathlib import Path
from collections import Counter
import sys
import time
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ─── Cell 4: Try Official KAGNN ───────────────────────────────────────────────
KAGNN_PATH = '/content/drive/MyDrive/SMRT/KAGNN-main/graph_regression'
sys.path.insert(0, KAGNN_PATH)

KAGNN_AVAILABLE = False
try:
    from models import KAGIN
    from featurization import get_atom_fdim, get_bond_fdim, mol2graph
    KAGNN_AVAILABLE = True
    ATOM_FDIM = get_atom_fdim()
    BOND_FDIM  = get_bond_fdim()
    print(f"Official KAGNN loaded (OK) | Atom: {ATOM_FDIM} | Bond: {BOND_FDIM}")
except ImportError as e:
    print(f"WARNING: Official KAGNN not available: {e}")
    print("Using optimized GAT fallback")
    ATOM_FDIM = 133
    BOND_FDIM  = 14

# ─── Cell 5: Fallback Featurization ──────────────────────────────────────────
if not KAGNN_AVAILABLE:
    def get_atom_features_kagnn(atom):
        features = []
        features.extend([int(atom.GetAtomicNum() == x) for x in range(1, 101)])
        features.extend([int(atom.GetDegree() == x) for x in range(11)])
        features.extend([int(atom.GetFormalCharge() == x) for x in range(-2, 3)])
        hyb_types = [
            Chem.rdchem.HybridizationType.S,    Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,  Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2
        ]
        features.extend([int(atom.GetHybridization() == x) for x in hyb_types])
        features.append(int(atom.GetIsAromatic()))
        features.extend([int(atom.GetTotalNumHs() == x) for x in range(5)])
        features.append(int(atom.IsInRing()))
        features.append(int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
        features.append(int(atom.GetNumRadicalElectrons()))
        features.append(atom.GetImplicitValence())
        features.append(atom.GetExplicitValence())
        return features

    def get_bond_features_kagnn(bond):
        features = []
        bt = bond.GetBondType()
        features.extend([
            int(bt == Chem.rdchem.BondType.SINGLE), int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE), int(bt == Chem.rdchem.BondType.AROMATIC)
        ])
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))
        stereo = bond.GetStereo()
        features.extend([
            int(stereo == Chem.rdchem.BondStereo.STEREONONE),
            int(stereo == Chem.rdchem.BondStereo.STEREOANY),
            int(stereo == Chem.rdchem.BondStereo.STEREOZ),
            int(stereo == Chem.rdchem.BondStereo.STEREOE),
            int(stereo == Chem.rdchem.BondStereo.STEREOCIS),
            int(stereo == Chem.rdchem.BondStereo.STEREOTRANS)
        ])
        bond_dir = bond.GetBondDir()
        features.extend([
            int(bond_dir == Chem.rdchem.BondDir.ENDUPRIGHT),
            int(bond_dir == Chem.rdchem.BondDir.ENDDOWNRIGHT)
        ])
        return features

# ─── Cell 6: Chemical class heuristic (same as reverse model) ─────────────────
def assign_chemical_class(mol):
    """Rule-based ClassyFire-style superclass assignment from RDKit descriptors."""
    try:
        mw     = Descriptors.MolWt(mol)
        logp   = Descriptors.MolLogP(mol)
        tpsa   = Descriptors.TPSA(mol)
        n_ar   = Descriptors.NumAromaticRings(mol)
        n_het  = Descriptors.NumHeteroatoms(mol)
        fsp3   = Descriptors.FractionCSP3(mol)
        hbd    = Descriptors.NumHDonors(mol)
        n_ring = Descriptors.RingCount(mol)
        if mw > 350 and logp > 4 and tpsa < 80 and fsp3 > 0.6:   return 'Lipids'
        if tpsa > 120 and logp < -0.5 and n_ar == 0:              return 'Carbohydrates'
        if tpsa > 100 and logp < 1 and hbd >= 2:                  return 'Organic Acids & AA'
        if n_ar >= 1 and n_het >= 2:                              return 'Organoheterocyclics'
        if n_ar >= 1 and n_het < 2:                               return 'Benzenoids'
        if n_ring == 0 and fsp3 > 0.5:                            return 'Aliphatic Organics'
        return 'Other'
    except Exception:
        return 'Unknown'

CLASS_PALETTE = {
    'Organoheterocyclics': '#2196F3', 'Benzenoids':         '#4CAF50',
    'Organic Acids & AA':  '#FF5722', 'Lipids':             '#9C27B0',
    'Carbohydrates':       '#FF9800', 'Aliphatic Organics': '#00BCD4',
    'Other':               '#795548', 'Unknown':            '#9E9E9E',
}

# ─── Cell 7: Config ───────────────────────────────────────────────────────────
DATA_DIR = Path('/content/drive/MyDrive/SMRT/KAGNN-main/8038913')

class Config:
    csv_path  = DATA_DIR / 'SMRT_dataset.csv'
    ecfp_path = DATA_DIR / 'SMRT_ECFP_1024_Fingerprints.txt'
    sdf_path  = DATA_DIR / 'SMRT_dataset.sdf'

    # Stage 1 — TRUE KAGNN
    hidden_dim          = 256
    gnn_layers          = 5
    dropout             = 0.1
    batch_size          = 128         # larger batch for better GPU saturation
    epochs              = 150
    learning_rate       = 3e-4
    weight_decay        = 1e-5
    early_stop_patience = 20
    grad_clip           = 1.0
    use_amp             = True        # automatic mixed precision (Tensor Cores)

    # KAN (Kolmogorov–Arnold Networks, Gaussian RBF)
    kan_num_basis  = 5
    kan_num_layers = 2

    # Stage 2 — PGM residual correction
    pgm_n_estimators  = 50
    pgm_max_depth     = 6
    pgm_learning_rate = 0.05

    val_size  = 0.15
    test_size = 0.15
    seed      = 42

    min_class_samples = 10

    # ── All results go to Drive ───────────────────────────────────────────────
    results_dir = Path('/content/drive/MyDrive/SMRT/results/kagnn_pgm_forward')

    def __init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")

config = Config()

# ─── Cell 8: Molecular Descriptors ───────────────────────────────────────────
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
                Descriptors.MinPartialCharge(mol),  Descriptors.MinAbsPartialCharge(mol)
            ], dtype=np.float32)
        except:
            return np.zeros(32, dtype=np.float32)

    @staticmethod
    def extract_batch(mol_dict, mol_ids, use_cache=True):
        results = []
        for mid in mol_ids:
            if use_cache and mid in ComprehensiveDescriptors._cache:
                results.append(ComprehensiveDescriptors._cache[mid])
            else:
                desc = ComprehensiveDescriptors.extract(mol_dict[mid])
                if use_cache:
                    ComprehensiveDescriptors._cache[mid] = desc
                results.append(desc)
        return np.array(results, dtype=np.float32)

# ─── Cell 9: Data Loader ──────────────────────────────────────────────────────
# NOTE: chemical class assignment is NOT done here — moved to after training.
class SMRTDataLoader:
    def __init__(self, csv_path, ecfp_path, sdf_path):
        print("=" * 60)
        print("LOADING SMRT DATASET")
        print("=" * 60)

        df = pd.read_csv(csv_path, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        id_col = next((c for c in df.columns if 'pubchem' in c or 'cid' in c or '#' in c),
                      df.columns[0])
        rt_col = next((c for c in df.columns if 'rt' in c or 'retention' in c), None)
        df = df.rename(columns={id_col: 'pubchem_id', rt_col: 'rt'})
        df['pubchem_id'] = pd.to_numeric(df['pubchem_id'], errors='coerce')
        df['rt']         = pd.to_numeric(df['rt'],         errors='coerce')
        self.df = df.dropna(subset=['pubchem_id', 'rt']).reset_index(drop=True)

        print("Loading ECFP...")
        self.ecfp_dict = {}
        with open(ecfp_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        cid = int(parts[0].replace('ID=', ''))
                        fp  = np.array([int(b) for b in parts[1][:1024]], dtype=np.float32)
                        self.ecfp_dict[cid] = fp
                    except:
                        continue
        print(f"  Loaded {len(self.ecfp_dict)} fingerprints")

        print("Loading SDF...")
        self.mol_dict = {}
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        valid, errors = 0, 0
        for i, mol in enumerate(suppl):
            if mol is None:
                errors += 1; continue
            try:
                cid = int(mol.GetProp('PUBCHEM_COMPOUND_CID'))
                try:
                    Chem.SanitizeMol(mol)
                    self.mol_dict[cid] = mol; valid += 1
                except:
                    errors += 1
            except:
                errors += 1
            if (i + 1) % 10000 == 0:
                print(f"  {i+1} processed | Valid: {valid} | Errors: {errors}")
        print(f"  SDF complete: {valid} valid, {errors} skipped")

        common_ids     = set(self.df.pubchem_id) & set(self.ecfp_dict) & set(self.mol_dict)
        self.df        = self.df[self.df.pubchem_id.isin(common_ids)].reset_index(drop=True)
        self.ecfp_dict = {k: v for k, v in self.ecfp_dict.items() if k in common_ids}
        self.mol_dict  = {k: v for k, v in self.mol_dict.items()  if k in common_ids}

        self.rt_scaler     = RobustScaler()
        self.df['rt_norm'] = self.rt_scaler.fit_transform(self.df[['rt']]).flatten()

        print(f"Dataset: {len(self.df)} samples")
        print("=" * 60)

    def denormalize(self, rt_norm):
        return self.rt_scaler.inverse_transform(
            np.array(rt_norm).reshape(-1, 1)).flatten()

# ─── Cell 10: Dataset ─────────────────────────────────────────────────────────
class SMRTDataset(Dataset):
    def __init__(self, df, ecfp_dict, mol_dict):
        self.df        = df.reset_index(drop=True)
        self.ecfp_dict = ecfp_dict
        self.mol_dict  = mol_dict

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid = int(row.pubchem_id)
        mol = self.mol_dict[cid]

        if KAGNN_AVAILABLE:
            gd         = mol2graph(mol)
            x          = torch.tensor(gd['node_features'], dtype=torch.float)
            edge_index = torch.tensor(gd['edge_index'],    dtype=torch.long)
            edge_attr  = torch.tensor(gd['edge_features'], dtype=torch.float)
        else:
            x = torch.tensor(
                [get_atom_features_kagnn(a) for a in mol.GetAtoms()], dtype=torch.float)
            edge_index, edge_attr = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = get_bond_features_kagnn(bond)
                edge_index.extend([[i, j], [j, i]]); edge_attr.extend([bf, bf])
            if len(edge_index) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr  = torch.zeros((0, BOND_FDIM), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

        return (Data(x=x, edge_index=edge_index, edge_attr=edge_attr),
                torch.tensor(self.ecfp_dict[cid], dtype=torch.float),
                torch.tensor(row.rt_norm,          dtype=torch.float),
                cid)

def collate_fn(batch):
    graphs, ecfps, rts, ids = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(ecfps), torch.stack(rts), ids

# ─── Cell 11: KAN Layer (Kolmogorov–Arnold Networks, Gaussian RBF basis) ──────
class SimpleKANLayer(nn.Module):
    """
    KAN layer: each input neuron activates n_basis learnable Gaussian RBFs.
    More expressive than MLP — activation is per-connection, not per-neuron.
    """
    def __init__(self, in_features, out_features, num_basis=5):
        super().__init__()
        self.centers = nn.Parameter(
            torch.linspace(-2, 2, num_basis).unsqueeze(0).repeat(in_features, 1))
        self.widths  = nn.Parameter(torch.ones(in_features, num_basis) * 0.5)
        self.coef    = nn.Parameter(
            torch.randn(in_features, out_features, num_basis) * 0.1)
        self.base    = nn.Linear(in_features, out_features)
        self.scale   = nn.Parameter(torch.ones(1))

    def forward(self, x):
        base_out  = self.base(x)
        x_norm    = torch.tanh(x)
        x_exp     = x_norm.unsqueeze(-1)
        c_exp     = self.centers.unsqueeze(0)
        w_exp     = torch.abs(self.widths).unsqueeze(0) + 0.1
        basis     = torch.exp(-((x_exp - c_exp) ** 2) / (2 * w_exp ** 2))
        basis     = basis / (basis.sum(dim=-1, keepdim=True) + 1e-8)
        basis_out = torch.einsum('bin,ion->bo', basis, self.coef)
        return base_out + self.scale * basis_out


class KANNetwork(nn.Module):
    """Multi-layer KAN network with LayerNorm + GELU between layers."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_basis=5):
        super().__init__()
        layers = []
        dims   = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(SimpleKANLayer(dims[i], dims[i+1], num_basis=num_basis))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.1))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)

# ─── Cell 12: TRUE KAGNN Backbone — ARCHITECTURE ──────────────────────────────
class TrueKAGNNBackbone(nn.Module):
    """
    Dual-stream architecture:
      Graph stream:  5-layer GATv2 with residual connections → mean+max pooling
      ECFP stream:   KAN encoder (1024 → h)
      Combined: cat(graph_emb, ecfp_emb) → KAN predictor → RT
    """
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim; heads = 4

        self.atom_embed = nn.Linear(ATOM_FDIM, h)
        self.bond_embed = nn.Linear(BOND_FDIM, h // heads)

        self.gat1 = GATv2Conv(h, h//heads, heads=heads, dropout=config.dropout, edge_dim=h//heads)
        self.gat2 = GATv2Conv(h, h//heads, heads=heads, dropout=config.dropout, edge_dim=h//heads)
        self.gat3 = GATv2Conv(h, h//heads, heads=heads, dropout=config.dropout, edge_dim=h//heads)
        self.gat4 = GATv2Conv(h, h//heads, heads=heads, dropout=config.dropout, edge_dim=h//heads)
        self.gat5 = GATv2Conv(h, h,        heads=1,     dropout=config.dropout, edge_dim=h//heads)

        self.norm1 = nn.LayerNorm(h); self.norm2 = nn.LayerNorm(h)
        self.norm3 = nn.LayerNorm(h); self.norm4 = nn.LayerNorm(h)
        self.norm5 = nn.LayerNorm(h)

        # ✅ ECFP encoding with KAN instead of MLP
        self.ecfp_encoder = KANNetwork(
            input_dim=1024, hidden_dim=h*2, output_dim=h,
            num_layers=config.kan_num_layers, num_basis=config.kan_num_basis)

        # ✅ Prediction head with KAN instead of MLP
        self.predictor = KANNetwork(
            input_dim=2*h, hidden_dim=h, output_dim=1,
            num_layers=config.kan_num_layers, num_basis=config.kan_num_basis)

        print(f"TrueKAGNNBackbone initialised "
              f"(5×GATv2 + KAN basis={config.kan_num_basis})")

    def forward(self, graph, ecfp, return_emb=False):
        x = self.atom_embed(graph.x)
        e = self.bond_embed(graph.edge_attr)

        x = x + self.norm1(F.elu(self.gat1(x, graph.edge_index, e)))
        x = x + self.norm2(F.elu(self.gat2(x, graph.edge_index, e)))
        x = x + self.norm3(F.elu(self.gat3(x, graph.edge_index, e)))
        x = x + self.norm4(F.elu(self.gat4(x, graph.edge_index, e)))
        x = self.norm5(F.elu(self.gat5(x, graph.edge_index, e)))

        g_emb    = (global_mean_pool(x, graph.batch) +
                    global_max_pool(x, graph.batch)) / 2
        e_emb    = self.ecfp_encoder(ecfp)
        combined = torch.cat([g_emb, e_emb], dim=-1)
        pred     = self.predictor(combined).squeeze(-1)

        return (pred, combined) if return_emb else pred

# ─── Cell 13: Class-weighted loss ────────────────────────────────────────────
class ClassWeightedSmoothL1(nn.Module):
    """
    SmoothL1 re-weighted by sqrt-capped inverse-frequency class weights.
    Minority classes receive proportional gradient signal.
    """
    def __init__(self, class_weights, cid_to_class, beta=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.cid_to_class  = cid_to_class
        self.beta          = beta

    def forward(self, pred, target, cids):
        w     = torch.tensor(
            [self.class_weights.get(self.cid_to_class.get(int(c), 'Unknown'), 1.0)
             for c in cids],
            dtype=torch.float, device=pred.device)
        diff    = pred - target
        loss_el = torch.where(diff.abs() < self.beta,
                              0.5 * diff**2 / self.beta,
                              diff.abs() - 0.5 * self.beta)
        return (w * loss_el).mean()

# ─── Cell 14: KAGNN→PGM Model — ARCHITECTURE ──────────────────────────────────
class KAGNN_PGM_Model(nn.Module):
    def __init__(self, config, mol_dict, class_weights, cid_to_class, device):
        super().__init__()
        self.config       = config
        self.mol_dict     = mol_dict
        self.device       = device

        # Stage 1: TRUE KAGNN (GAT + KAN)
        self.kagnn        = TrueKAGNNBackbone(config)
        self.criterion    = ClassWeightedSmoothL1(class_weights, cid_to_class)

        # Stage 2: PGM residual correction
        self.pgm_xgb      = None
        self.pgm_br       = None
        self.pgm_scaler   = StandardScaler()

    def forward(self, graph, ecfp, return_emb=False):
        return self.kagnn(graph, ecfp, return_emb)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    def train_stage1_kagnn(self, train_loader, val_loader, optimizer, scheduler, epochs, patience):
        print("\n" + "=" * 60)
        print("STAGE 1: TRAINING TRUE KAGNN BACKBONE (GAT + KAN)")
        amp_on = self.config.use_amp and self.device.type == 'cuda'
        print(f"  AMP (mixed precision): {'ON' if amp_on else 'OFF'}")
        print("=" * 60)

        scaler           = GradScaler(enabled=amp_on)
        best_val_loss    = float('inf')
        patience_counter = 0
        history          = {'train_loss': [], 'val_loss': [], 'lr': []}
        ckpt_path        = self.config.results_dir / 'plots' / 'best_kagnn.pth'
        prev_lr          = optimizer.param_groups[0]['lr']

        for epoch in range(1, epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            self.train()
            train_loss = 0
            for graph, ecfp, rt, cids in train_loader:
                graph = graph.to(self.device, non_blocking=True)
                ecfp  = ecfp.to(self.device,  non_blocking=True)
                rt    = rt.to(self.device,     non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_on):
                    pred = self(graph, ecfp)
                    loss = self.criterion(pred, rt, cids)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item() * len(rt)
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)

            # ── Val ───────────────────────────────────────────────────────────
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for graph, ecfp, rt, cids in val_loader:
                    graph = graph.to(self.device, non_blocking=True)
                    ecfp  = ecfp.to(self.device,  non_blocking=True)
                    rt    = rt.to(self.device,     non_blocking=True)
                    with autocast(enabled=amp_on):
                        val_loss += self.criterion(self(graph, ecfp), rt, cids).item() * len(rt)
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)

            # ── LR schedule ───────────────────────────────────────────────────
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']
            if lr_now < prev_lr - 1e-10:
                print(f"  LR reduced: {prev_lr:.2e} -> {lr_now:.2e} (epoch {epoch})")
                prev_lr = lr_now
            history['lr'].append(lr_now)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.5f} | "
                      f"Val: {val_loss:.5f} | LR: {lr_now:.2e}")

            # ── Checkpoint ────────────────────────────────────────────────────
            if val_loss < best_val_loss - 1e-5:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        print(f"Stage 1 complete — best val loss: {best_val_loss:.5f}")
        print(f"Checkpoint saved to Drive: {ckpt_path}")
        return history

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    def train_stage2_pgm(self, train_loader, verbose=True):
        amp_on = self.config.use_amp and self.device.type == 'cuda'
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: TRAINING PGM RESIDUAL CORRECTION")
            print("(Features: 256-dim GNN emb + ECFP-1024 + 32 RDKit desc)")
            print("=" * 60)

        self.eval()
        X_list, residual_list = [], []

        with torch.no_grad():
            for graph, ecfp, rt_norm, cids in train_loader:
                graph   = graph.to(self.device,   non_blocking=True)
                ecfp    = ecfp.to(self.device,    non_blocking=True)
                rt_norm = rt_norm.to(self.device, non_blocking=True)
                with autocast(enabled=amp_on):
                    kagnn_pred, emb = self(graph, ecfp, return_emb=True)
                residual = (rt_norm.cpu().float() - kagnn_pred.cpu().float()).numpy()
                desc     = ComprehensiveDescriptors.extract_batch(
                    self.mol_dict, cids, use_cache=True)
                features = np.concatenate(
                    [emb.cpu().float().numpy(), ecfp.cpu().float().numpy(), desc], axis=1)
                X_list.append(features)
                residual_list.append(residual)

        X        = np.nan_to_num(np.concatenate(X_list), 0)
        y        = np.concatenate(residual_list)
        X_scaled = self.pgm_scaler.fit_transform(X)

        xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if verbose:
            print(f"  Feature matrix: {X_scaled.shape}  "
                  f"(256 GNN emb + 1024 ECFP + 32 desc)")
            print(f"  Training XGBoost (device={xgb_device})...")

        self.pgm_xgb = xgb.XGBRegressor(
            n_estimators=self.config.pgm_n_estimators,
            max_depth=self.config.pgm_max_depth,
            learning_rate=self.config.pgm_learning_rate,
            tree_method='hist', device=xgb_device,
            random_state=42)
        self.pgm_xgb.fit(X_scaled, y, verbose=False)

        if verbose:
            print("  Training Bayesian Ridge...")

        self.pgm_br = BayesianRidge(max_iter=500)
        self.pgm_br.fit(X_scaled, y)

        ens = (self.pgm_xgb.predict(X_scaled) + self.pgm_br.predict(X_scaled)) / 2
        mae = np.mean(np.abs(y - ens))
        if verbose:
            print(f"Stage 2 complete | Residual train MAE: {mae:.5f}")

    # ── PGM correction helper ─────────────────────────────────────────────────
    def _pgm_correct(self, emb_np, ecfp_np, cids):
        desc = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids, use_cache=True)
        X    = np.nan_to_num(np.concatenate([emb_np, ecfp_np, desc], axis=1), 0)
        X_sc = self.pgm_scaler.transform(X)
        return (self.pgm_xgb.predict(X_sc) + self.pgm_br.predict(X_sc)) / 2

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, data_loader, data_loader_obj):
        """Returns y_true, y_pred_kagnn, y_pred_final (all denormalized), cid_list."""
        amp_on = self.config.use_amp and self.device.type == 'cuda'
        self.eval()
        y_true_list, y_kagnn_list, y_final_list, cid_list = [], [], [], []

        for graph, ecfp, rt_norm, cids in data_loader:
            graph = graph.to(self.device, non_blocking=True)
            ecfp  = ecfp.to(self.device,  non_blocking=True)
            with autocast(enabled=amp_on):
                kagnn_norm, emb = self(graph, ecfp, return_emb=True)
            kagnn_np = kagnn_norm.cpu().float().numpy()
            emb_np   = emb.cpu().float().numpy()
            ecfp_np  = ecfp.cpu().float().numpy()

            if self.pgm_xgb is not None:
                corr     = self._pgm_correct(emb_np, ecfp_np, cids)
                final_np = kagnn_np + corr
            else:
                final_np = kagnn_np

            y_true_list.append(rt_norm.numpy())
            y_kagnn_list.append(kagnn_np)
            y_final_list.append(final_np)
            cid_list.extend([int(c) for c in cids])

        y_true  = data_loader_obj.denormalize(np.concatenate(y_true_list))
        y_kagnn = data_loader_obj.denormalize(np.concatenate(y_kagnn_list))
        y_final = data_loader_obj.denormalize(np.concatenate(y_final_list))
        return y_true, y_kagnn, y_final, cid_list

# ─── Cell 15: Metrics ─────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    ae = np.abs(y_true - y_pred)
    return {
        'n_samples':  len(y_true),
        'MedAE':      float(np.median(ae)),
        'MAE':        float(np.mean(ae)),
        'RMSE':       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2':         float(r2_score(y_true, y_pred)),
        'Pearson':    float(pearsonr(y_true, y_pred)[0]),
        'Spearman':   float(spearmanr(y_true, y_pred)[0]),
        'Pct_le_60s': float((ae <= 60).mean() * 100),
        'Pct_le_30s': float((ae <= 30).mean() * 100),
        'Pct_le_10s': float((ae <= 10).mean() * 100),
        'MedRE':      float(np.median(
                          np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100)),
    }

# ─── Cell 16: Standard Result Plots ──────────────────────────────────────────
def plot_results(y_true, y_pred, label, color, history, config):
    ae     = np.abs(y_true - y_pred)
    medae  = np.median(ae); mae = np.mean(ae)
    r2     = r2_score(y_true, y_pred)
    pct_10 = (ae <= 10).mean() * 100
    pct_30 = (ae <= 30).mean() * 100
    pct_60 = (ae <= 60).mean() * 100

    if history is not None:
        has_lr = 'lr' in history and len(history['lr']) > 0
        ncols  = 3 if has_lr else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
        axes[0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(history['val_loss'],   label='Val',   linewidth=2)
        axes[0].set_title('Stage 1: KAGNN Training (Class-weighted SmoothL1)',
                          fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].legend(); axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[1].plot(history['val_loss'], color='darkorange', linewidth=2)
        axes[1].set_title('Validation Loss', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        if has_lr:
            axes[2].plot(history['lr'], color='purple', linewidth=2)
            axes[2].set_title('LR (ReduceLROnPlateau)', fontsize=13, fontweight='bold')
            axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3, linestyle='--')
        plt.suptitle('Stage 1: TRUE KAGNN Training History',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = config.results_dir / 'plots' / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved to Drive: {save_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{label} — Test Set Performance', fontsize=18, fontweight='bold', y=1.01)

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, c=color, edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', lw=2.5, label='Perfect', zorder=10)
    ax1.set_xlabel('True RT (s)',      fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted RT (s)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Predicted vs True\nMedAE={medae:.2f}s, R²={r2:.3f}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--'); ax1.legend(fontsize=11)
    ax1.text(0.05, 0.95, f'MedAE: {medae:.2f}s\nMAE:   {mae:.2f}s\nR²:    {r2:.3f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax2 = axes[1]
    sa  = np.sort(ae); cdf = np.arange(1, len(sa) + 1) / len(sa)
    ax2.plot(sa, cdf, linewidth=2.5, color=color, alpha=0.8)
    ax2.fill_between(sa, 0, cdf, alpha=0.2, color=color)
    ax2.axhline(0.5, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(medae, color=color, linestyle=':', linewidth=2)
    ax2.text(medae + 3, 0.45, f'{medae:.1f}s', color=color, fontweight='bold', fontsize=11)
    ax2.set_xlabel('Absolute Error (s)',     fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax2.set_title('CDF of Absolute Errors',  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, min(150, sa.max())); ax2.set_ylim(0, 1.05)

    ax3 = axes[2]
    bars = ax3.bar(['≤10s', '≤30s', '≤60s'], [pct_10, pct_30, pct_60],
                   color=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., h + 1,
                 f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('% Correct', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Error Threshold', fontsize=13, fontweight='bold')
    ax3.set_title('Threshold Performance', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105); ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    save_path = config.results_dir / 'plots' / f'{label.replace(" ", "_")}_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to Drive: {save_path}")

# ─── Cell 17: Comprehensive Comparison Plot ───────────────────────────────────
def plot_comparison(y_true, y_kagnn, y_final, config):
    ae_kagnn = np.abs(y_true - y_kagnn)
    ae_final = np.abs(y_true - y_final)

    medae_kagnn = np.median(ae_kagnn); medae_final = np.median(ae_final)
    mae_kagnn   = np.mean(ae_kagnn);   mae_final   = np.mean(ae_final)
    r2_kagnn    = r2_score(y_true, y_kagnn); r2_final = r2_score(y_true, y_final)
    pct60_k     = (ae_kagnn <= 60).mean() * 100; pct60_f = (ae_final <= 60).mean() * 100
    pct30_k     = (ae_kagnn <= 30).mean() * 100; pct30_f = (ae_final <= 30).mean() * 100
    pct10_k     = (ae_kagnn <= 10).mean() * 100; pct10_f = (ae_final <= 10).mean() * 100
    improv      = (medae_kagnn - medae_final) / medae_kagnn * 100

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('TRUE KAGNN Baseline vs KAGNN + PGM — Comprehensive Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    for ax, y_pred, color, title_str, box_color in [
        (axes[0, 0], y_kagnn, 'royalblue',
         f'TRUE KAGNN Baseline\nMedAE={medae_kagnn:.2f}s, R²={r2_kagnn:.3f}', 'lightblue'),
        (axes[0, 1], y_final, 'mediumseagreen',
         f'KAGNN + PGM\nMedAE={medae_final:.2f}s, R²={r2_final:.3f} (↑{improv:.1f}%)',
         'lightgreen'),
    ]:
        ae = np.abs(y_true - y_pred)
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=color, edgecolors='none')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=2.5, label='Perfect', zorder=10)
        ax.set_xlabel('True RT (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted RT (s)', fontsize=13, fontweight='bold')
        ax.set_title(title_str, fontsize=14, fontweight='bold',
                     color='darkgreen' if color == 'mediumseagreen' else 'black')
        ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(fontsize=11)
        ax.text(0.05, 0.95,
                f'MedAE: {np.median(ae):.2f}s\nMAE:   {np.mean(ae):.2f}s\n'
                f'R²:    {r2_score(y_true, y_pred):.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    ax3 = axes[1, 0]
    for ae, color, lbl, medae in [
        (ae_kagnn, 'royalblue',      'TRUE KAGNN',  medae_kagnn),
        (ae_final, 'mediumseagreen', 'KAGNN + PGM', medae_final),
    ]:
        sa = np.sort(ae); cdf = np.arange(1, len(sa) + 1) / len(sa)
        ax3.plot(sa, cdf, linewidth=2.5, color=color, label=lbl, alpha=0.8)
        ax3.fill_between(sa, 0, cdf, alpha=0.2, color=color)
        ax3.axvline(medae, color=color, linestyle=':', linewidth=2)
        ax3.text(medae + 3, 0.45 if color == 'royalblue' else 0.55,
                 f'{medae:.1f}s', color=color, fontweight='bold', fontsize=11)
    ax3.axhline(0.5, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Absolute Error (s)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax3.set_title('CDF Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=12, loc='lower right')
    ax3.set_xlim(0, min(150, max(ae_kagnn.max(), ae_final.max()))); ax3.set_ylim(0, 1.05)

    ax4 = axes[1, 1]
    x = np.arange(3); w = 0.35
    b1 = ax4.bar(x - w/2, [pct10_k, pct30_k, pct60_k], w,
                 label='TRUE KAGNN',  color='royalblue',     edgecolor='black', alpha=0.8)
    b2 = ax4.bar(x + w/2, [pct10_f, pct30_f, pct60_f], w,
                 label='KAGNN + PGM', color='mediumseagreen', edgecolor='black', alpha=0.8)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., h + 1,
                     f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_ylabel('% Correct', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Error Threshold', fontsize=13, fontweight='bold')
    ax4.set_title('Threshold Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x); ax4.set_xticklabels(['≤10s', '≤30s', '≤60s'])
    ax4.set_ylim(0, 105); ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_path = config.results_dir / 'plots' / 'comprehensive_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to Drive: {save_path}")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: TRUE KAGNN Baseline vs KAGNN + PGM")
    print("=" * 70)
    print(f"{'Metric':<20} {'KAGNN Only':>15} {'KAGNN+PGM':>15} {'Improvement':>15}")
    print("-" * 70)
    for lbl, v1, v2 in [
        ('MedAE (s)',   medae_kagnn, medae_final),
        ('MAE (s)',     mae_kagnn,   mae_final),
        ('R² Score',    r2_kagnn,    r2_final),
        ('% ≤10s',      pct10_k,     pct10_f),
        ('% ≤30s',      pct30_k,     pct30_f),
        ('% ≤60s',      pct60_k,     pct60_f),
    ]:
        impv = (v2 - v1) / abs(v1) * 100 if abs(v1) > 1e-9 else 0
        print(f"  {lbl:<18} {v1:>15.3f} {v2:>15.3f} {impv:>14.1f}%")
    print("=" * 70)

# ─── Cell 18: Class-wise Analysis ─────────────────────────────────────────────
def classwise_analysis(results_df, config):
    print("\n" + "=" * 60)
    print("CLASS-WISE PERFORMANCE ANALYSIS")
    print("=" * 60)

    df = results_df.copy()
    df['ae_kagnn'] = np.abs(df['y_true'] - df['y_pred_kagnn'])
    df['ae_final'] = np.abs(df['y_true'] - df['y_pred_final'])
    df['re_kagnn'] = np.abs((df['y_true'] - df['y_pred_kagnn']) /
                             (np.abs(df['y_true']) + 1e-8)) * 100
    df['re_final'] = np.abs((df['y_true'] - df['y_pred_final']) /
                             (np.abs(df['y_true']) + 1e-8)) * 100

    agg = (df.groupby('chemical_class').agg(
        n            = ('ae_final', 'count'),
        MedAE_kagnn  = ('ae_kagnn', 'median'),
        MedAE_final  = ('ae_final', 'median'),
        MAE_kagnn    = ('ae_kagnn', 'mean'),
        MAE_final    = ('ae_final', 'mean'),
        MedRE_kagnn  = ('re_kagnn', 'median'),
        MedRE_final  = ('re_final', 'median'),
    ).reset_index().rename(columns={'chemical_class': 'Class'}))

    agg = agg[agg['n'] >= config.min_class_samples].copy()
    agg['pct_of_test'] = agg['n'] / len(df) * 100
    agg['delta_MedAE'] = agg['MedAE_kagnn'] - agg['MedAE_final']
    agg['delta_MedRE'] = agg['MedRE_kagnn'] - agg['MedRE_final']

    global_medae_kagnn = np.median(df['ae_kagnn'])
    global_medae_final = np.median(df['ae_final'])
    global_medre_kagnn = np.median(df['re_kagnn'])
    global_medre_final = np.median(df['re_final'])

    print(f"\n  Global MedAE — KAGNN: {global_medae_kagnn:.2f}s  |  Hybrid: {global_medae_final:.2f}s")
    print(f"  Global MedRE — KAGNN: {global_medre_kagnn:.2f}%  |  Hybrid: {global_medre_final:.2f}%")
    print(f"  Classes shown: {len(agg)} (n ≥ {config.min_class_samples})\n")
    print(f"{'Class':<35} {'n':>6} {'MedAE_K':>9} {'MedAE_H':>10} "
          f"{'ΔMAE':>8} {'MedRE_H':>9}")
    print("-" * 82)
    for _, row in agg.sort_values('MedAE_final').iterrows():
        print(f"  {row['Class']:<33} {int(row['n']):>6} "
              f"{row['MedAE_kagnn']:>8.2f}s {row['MedAE_final']:>9.2f}s "
              f"{row['delta_MedAE']:>+7.2f}s {row['MedRE_final']:>8.2f}%")

    patches = [mpatches.Patch(color='#2ecc71', label='Below global MedAE'),
               mpatches.Patch(color='#e74c3c', label='Above global MedAE')]
    fig_h = max(6, len(agg) * 0.48)

    def bar_colors(values, gval):
        return ['#2ecc71' if v <= gval else '#e74c3c' for v in values]

    # ── Plot 1: MedAE Hybrid ──────────────────────────────────────────────────
    agg_me = agg.sort_values('MedAE_final', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    colors  = bar_colors(agg_me['MedAE_final'], global_medae_final)
    bars    = ax.barh(agg_me['Class'], agg_me['MedAE_final'],
                      color=colors, edgecolor='white', linewidth=0.6, height=0.7)
    for bar, (_, row) in zip(bars, agg_me.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{w:.1f}s  (n={int(row["n"])})',
                va='center', ha='left', fontsize=8.5, color='#2c3e50')
    ax.axvline(global_medae_final, color='#2c3e50', linestyle='--', linewidth=1.8,
               label=f'Global MedAE = {global_medae_final:.1f}s (Hybrid)')
    ax.legend(handles=patches + [ax.get_legend_handles_labels()[0][-1]],
              fontsize=10, loc='lower right')
    ax.set_xlabel('Median Absolute Error (s)', fontsize=13, fontweight='bold')
    ax.set_title('Class-wise MedAE — KAGNN + PGM Hybrid\n'
                 '(Reviewer: class imbalance bias check)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, agg_me['MedAE_final'].max() * 1.28)
    ax.tick_params(axis='y', labelsize=9); ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis(); plt.tight_layout()
    sp = config.results_dir / 'plots' / 'classwise_MedAE_hybrid.png'
    plt.savefig(sp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show(); print(f"Saved to Drive: {sp}")

    # ── Plot 2: MedRE Hybrid ──────────────────────────────────────────────────
    agg_re = agg.sort_values('MedRE_final', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    colors  = bar_colors(agg_re['MedRE_final'], global_medre_final)
    bars    = ax.barh(agg_re['Class'], agg_re['MedRE_final'],
                      color=colors, edgecolor='white', linewidth=0.6, height=0.7)
    for bar, (_, row) in zip(bars, agg_re.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{w:.1f}%  (n={int(row["n"])})',
                va='center', ha='left', fontsize=8.5, color='#2c3e50')
    ax.axvline(global_medre_final, color='#2c3e50', linestyle='--', linewidth=1.8,
               label=f'Global MedRE = {global_medre_final:.1f}% (Hybrid)')
    ax.legend(handles=patches + [ax.get_legend_handles_labels()[0][-1]],
              fontsize=10, loc='lower right')
    ax.set_xlabel('Median Relative Error (%)', fontsize=13, fontweight='bold')
    ax.set_title('Class-wise MedRE — KAGNN + PGM Hybrid\n'
                 '(Reviewer: class imbalance bias check)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, agg_re['MedRE_final'].max() * 1.28)
    ax.tick_params(axis='y', labelsize=9); ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis(); plt.tight_layout()
    sp = config.results_dir / 'plots' / 'classwise_MedRE_hybrid.png'
    plt.savefig(sp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show(); print(f"Saved to Drive: {sp}")

    # ── Plot 3: Class size vs error scatter ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Class Size vs Error — Majority Bias Diagnostic (Hybrid)',
                 fontsize=14, fontweight='bold')
    for ax, metric_col, gval, unit in [
        (axes[0], 'MedAE_final', global_medae_final, 's'),
        (axes[1], 'MedRE_final', global_medre_final, '%'),
    ]:
        sc = ax.scatter(agg['n'], agg[metric_col], c=agg[metric_col],
                        cmap='RdYlGn_r', s=80, edgecolors='grey',
                        linewidths=0.5, alpha=0.85)
        for _, row in agg.iterrows():
            ax.annotate(row['Class'][:18], (row['n'], row[metric_col]),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=6.5, color='#2c3e50')
        ax.axhline(gval, color='#2c3e50', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Class size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_col[:5]} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'n vs {metric_col[:5]}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(sc, ax=ax, label=f'{metric_col[:5]} ({unit})')
    plt.tight_layout()
    sp = config.results_dir / 'plots' / 'classwise_size_vs_error.png'
    plt.savefig(sp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show(); print(f"Saved to Drive: {sp}")

    # ── Plot 4: Grouped bar KAGNN vs Hybrid ───────────────────────────────────
    agg_g = agg.sort_values('MedAE_kagnn', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(agg_g)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(20, fig_h))
    fig.suptitle('TRUE KAGNN Baseline vs KAGNN + PGM Hybrid — Class-wise Comparison\n'
                 '(PGM residual rescue across classes)',
                 fontsize=14, fontweight='bold')

    for ax, m_k, m_f, gk, gf, unit, title in [
        (axes[0], 'MedAE_kagnn', 'MedAE_final',
         global_medae_kagnn, global_medae_final, 's', 'MedAE'),
        (axes[1], 'MedRE_kagnn', 'MedRE_final',
         global_medre_kagnn, global_medre_final, '%', 'MedRE'),
    ]:
        ax.barh(y_pos - w/2, agg_g[m_k], w, label='TRUE KAGNN',
                color='royalblue',      edgecolor='white', alpha=0.8, height=w)
        ax.barh(y_pos + w/2, agg_g[m_f], w, label='KAGNN + PGM',
                color='mediumseagreen', edgecolor='white', alpha=0.8, height=w)
        for i, (_, row) in enumerate(agg_g.iterrows()):
            delta = row[m_k] - row[m_f]
            sign  = 'v' if delta > 0 else '^'
            clr   = '#27ae60' if delta > 0 else '#c0392b'
            ax.text(max(row[m_k], row[m_f]) + 0.5, i,
                    f'{sign}{abs(delta):.1f}{unit}',
                    va='center', ha='left', fontsize=7.5, color=clr, fontweight='bold')
        ax.axvline(gk, color='royalblue',      linestyle=':', linewidth=1.5,
                   label=f'Global KAGNN {gk:.1f}{unit}', alpha=0.7)
        ax.axvline(gf, color='mediumseagreen', linestyle='--', linewidth=1.5,
                   label=f'Global Hybrid {gf:.1f}{unit}', alpha=0.7)
        ax.set_yticks(y_pos); ax.set_yticklabels(agg_g['Class'], fontsize=8.5)
        ax.set_xlabel(f'{title} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'Class-wise {title}: KAGNN vs KAGNN+PGM',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8.5, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        ax.set_xlim(0, agg_g[[m_k, m_f]].values.max() * 1.28)

    plt.tight_layout()
    sp = config.results_dir / 'plots' / 'classwise_kagnn_vs_hybrid_grouped.png'
    plt.savefig(sp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show(); print(f"Saved to Drive: {sp}")

    return agg

# ─── Cell 19: Majority-bias Verdict ──────────────────────────────────────────
def majority_bias_verdict(agg, metrics_kagnn, metrics_final, test_cids, mol_dict):
    print("\n" + "=" * 60)
    print("MAJORITY-BIAS DIAGNOSTIC SUMMARY")
    print("=" * 60)

    cls_counts   = Counter(
        assign_chemical_class(mol_dict[cid]) for cid in test_cids)
    majority_cls = cls_counts.most_common(1)[0][0]
    majority_pct = cls_counts[majority_cls] / len(test_cids) * 100
    global_medae = metrics_final['MedAE']

    above = agg[agg['MedAE_final'] >  global_medae]
    below = agg[agg['MedAE_final'] <= global_medae]

    if majority_cls in agg['Class'].values:
        maj = agg[agg['Class'] == majority_cls].iloc[0]
        print(f"\n  Majority class : {majority_cls} ({majority_pct:.1f}% of test split)")
        print(f"  MedAE KAGNN -> Hybrid : {maj['MedAE_kagnn']:.2f}s → {maj['MedAE_final']:.2f}s")
        print(f"  MedRE KAGNN -> Hybrid : {maj['MedRE_kagnn']:.2f}% → {maj['MedRE_final']:.2f}%")

    print(f"\n  Classes BELOW global MedAE = {global_medae:.2f}s  ({len(below)} — well predicted):")
    for _, row in below.sort_values('MedAE_final').iterrows():
        rescued = '[rescued]' if row['delta_MedAE'] > 5 else ''
        print(f"    OK {row['Class']:<35} MedAE={row['MedAE_final']:.1f}s "
              f"MedRE={row['MedRE_final']:.1f}%  n={int(row['n'])}  {rescued}")

    print(f"\n  Classes ABOVE global MedAE ({len(above)} — harder):")
    for _, row in above.sort_values('MedAE_final', ascending=False).iterrows():
        worsened = '[worsened]' if row['delta_MedAE'] < -2 else ''
        print(f"    -- {row['Class']:<35} MedAE={row['MedAE_final']:.1f}s "
              f"MedRE={row['MedRE_final']:.1f}%  n={int(row['n'])}  {worsened}")

    if majority_cls in agg['Class'].values:
        maj = agg[agg['Class'] == majority_cls].iloc[0]
        verdict = (
            "The hybrid model does NOT appear over-optimised for the majority class. "
            "Class-weighted SmoothL1 loss in Stage 1 successfully equalised gradient "
            "signal across minority classes without sacrificing majority-class performance."
            if maj['MedAE_final'] <= global_medae else
            "Potential majority-class pull detected. Consider increasing max_weight_ratio "
            "or adding a class-conditioned residual head in Stage 2."
        )
        print(f"\n  Verdict: {verdict}")

# ─── Cell 20: Class Weights Helper ───────────────────────────────────────────
def compute_class_weights(df, max_ratio=10.0):
    """
    Sqrt-capped inverse-frequency weights, normalised to sample-weighted mean = 1.
    Prevents the 98%+ majority class from dominating training.
    """
    counts = df['rt'].groupby(df['pubchem_id']).count()   # dummy; use df directly
    # Use chemical class proxy: assign placeholder classes to get RT distribution
    # Actual class weights are computed in Main after data is loaded
    pass   # See Main block below

# ─── Cell 21: Main ────────────────────────────────────────────────────────────
data_loader = SMRTDataLoader(config.csv_path, config.ecfp_path, config.sdf_path)
df = data_loader.df

# ── Train / Val / Test splits ─────────────────────────────────────────────────
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=config.test_size, random_state=config.seed)
train_idx, val_idx  = train_test_split(
    train_idx, test_size=config.val_size / (1 - config.test_size),
    random_state=config.seed)

train_ds = SMRTDataset(df.iloc[train_idx], data_loader.ecfp_dict, data_loader.mol_dict)
val_ds   = SMRTDataset(df.iloc[val_idx],   data_loader.ecfp_dict, data_loader.mol_dict)
test_ds  = SMRTDataset(df.iloc[test_idx],  data_loader.ecfp_dict, data_loader.mol_dict)

# ── Device setup — done here so DataLoader params can use it ─────────────────
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pin_memory  = device.type == 'cuda'
num_workers = 4 if device.type == 'cuda' else 0
print(f"Device      : {device}")
print(f"AMP         : {'ON' if config.use_amp and device.type == 'cuda' else 'OFF'}")
print(f"pin_memory  : {pin_memory}  |  num_workers: {num_workers}")

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers > 0))
val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers > 0))
test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers > 0))

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# ── Pre-compute class weights on TRAINING CIDs only ───────────────────────────
# Chemical class assignment runs on ALL training mols once before Stage 1.
# This is the ONLY place classes are computed for weighting; class-wise
# performance analysis runs AFTER training on test split only.
print("\nPre-computing chemical classes for Stage-1 class weights…")
print("(on training split only — does not affect training speed)")
train_cids    = [int(df.iloc[i].pubchem_id) for i in train_idx]
train_classes = {cid: assign_chemical_class(data_loader.mol_dict[cid])
                 for cid in train_cids}
cid_to_class  = train_classes   # used by ClassWeightedSmoothL1

class_counts  = Counter(train_classes.values())
n_max         = float(max(class_counts.values()))
MAX_W         = 10.0

sqrt_w        = {cls: min(float(np.sqrt(n_max / n)), MAX_W)
                 for cls, n in class_counts.items()}
total_s       = sum(class_counts.values())
norm_f        = sum(sqrt_w[c] * class_counts[c] for c in class_counts) / total_s
class_weights = {cls: w / norm_f for cls, w in sqrt_w.items()}

print("\n  Stage-1 class weights (sqrt-capped 10×, normalised):")
for cls, w in sorted(class_weights.items(), key=lambda x: -x[1]):
    n = int(class_counts.get(cls, 0))
    print(f"    {cls:<35}  n={n:>7,}  weight={w:.4f}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = KAGNN_PGM_Model(
    config, data_loader.mol_dict, class_weights, cid_to_class, device
).to(device)

# ── Stage 1: TRUE KAGNN ───────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

t1      = time.time()
history = model.train_stage1_kagnn(
    train_loader, val_loader, optimizer, scheduler,
    config.epochs, config.early_stop_patience)
stage1_time = time.time() - t1

# ── KAGNN-only evaluation ─────────────────────────────────────────────────────
print("\n" + "=" * 60); print("EVALUATING KAGNN ONLY"); print("=" * 60)
y_true_k, y_pred_k, _, cid_list_k = model.predict(test_loader, data_loader)
metrics_kagnn = compute_metrics(y_true_k, y_pred_k)
print("TRUE KAGNN Metrics:")
for k, v in metrics_kagnn.items():
    print(f"  {k:15s}: {v:10.4f}")

# ── Stage 2: PGM residual correction ─────────────────────────────────────────
t2 = time.time()
model.train_stage2_pgm(train_loader, verbose=True)
stage2_time = time.time() - t2

# ── Final inference ───────────────────────────────────────────────────────────
print("\n" + "=" * 60); print("FINAL EVALUATION (KAGNN + PGM)"); print("=" * 60)
y_true, y_kagnn, y_final, cid_list = model.predict(test_loader, data_loader)

metrics_final = compute_metrics(y_true, y_final)
print("KAGNN + PGM Metrics:")
for k, v in metrics_final.items():
    print(f"  {k:15s}: {v:10.4f}")

improv_medae = (metrics_kagnn['MedAE'] - metrics_final['MedAE']) / metrics_kagnn['MedAE'] * 100
improv_medre = (metrics_kagnn['MedRE'] - metrics_final['MedRE']) / metrics_kagnn['MedRE'] * 100
print(f"\n  MedAE improvement : {improv_medae:.2f}%")
print(f"  MedRE improvement : {improv_medre:.2f}%")

# ── Statistical significance tests ───────────────────────────────────────────
from scipy.stats import wilcoxon

errors_k = np.abs(y_true - y_kagnn)
errors_f = np.abs(y_true - y_final)

t_stat, p_ttest = ttest_rel(errors_k, errors_f)
try:
    w_stat, p_wilcoxon = wilcoxon(errors_k, errors_f, alternative='two-sided')
except ValueError:
    w_stat, p_wilcoxon = float('nan'), float('nan')

print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE TESTS (KAGNN errors vs Hybrid errors)")
print("=" * 60)
print(f"  Paired t-test       :  t = {t_stat:+.4f}   p = {p_ttest:.6f}"
      f"  [{('SIGNIFICANT' if p_ttest < 0.05 else 'not significant')} at alpha=0.05]")
print(f"  Wilcoxon signed-rank:  W = {w_stat:.1f}     p = {p_wilcoxon:.6f}"
      f"  [{('SIGNIFICANT' if p_wilcoxon < 0.05 else 'not significant')} at alpha=0.05]")
print(f"  Mean error reduction: {np.mean(errors_k):.3f}s -> {np.mean(errors_f):.3f}s"
      f"  (delta = {np.mean(errors_k) - np.mean(errors_f):+.3f}s)")
print(f"  Med  error reduction: {np.median(errors_k):.3f}s -> {np.median(errors_f):.3f}s"
      f"  (delta = {np.median(errors_k) - np.median(errors_f):+.3f}s)")
print("=" * 60)

# ── Save metrics to Drive ─────────────────────────────────────────────────────
metrics_final['t_stat']     = float(t_stat)
metrics_final['p_ttest']    = float(p_ttest)
metrics_final['w_stat']     = float(w_stat)
metrics_final['p_wilcoxon'] = float(p_wilcoxon)

mk = config.results_dir / 'metrics_kagnn.json'
mf = config.results_dir / 'metrics_final.json'
with open(mk, 'w') as f: json.dump(metrics_kagnn, f, indent=2)
with open(mf, 'w') as f: json.dump(metrics_final, f, indent=2)
print(f"Saved to Drive: {mk}")
print(f"Saved to Drive: {mf}")

# ── Standard plots ─────────────────────────────────────────────────────────────
plot_results(y_true_k, y_pred_k, 'TRUE_KAGNN_Baseline', 'royalblue',      history, config)
plot_results(y_true,   y_final,  'KAGNN_+_PGM',         'mediumseagreen', None,    config)
plot_comparison(y_true, y_kagnn, y_final, config)

# ── Class-wise analysis (runs AFTER training — test split only, ~30s) ─────────
print("\n" + "=" * 60)
print("CLASS-WISE PERFORMANCE ANALYSIS")
print("Assigning classes to test split only (~12k mols) — training unaffected")
print("=" * 60)

test_cids      = [int(df.iloc[i].pubchem_id) for i in test_idx]
chem_class_map = {
    cid: assign_chemical_class(data_loader.mol_dict[cid])
    for cid in test_cids
}

cls_counts_test = Counter(chem_class_map.values())
print(f"\n  Class distribution in test split (n={len(test_cids):,}):")
print("  " + "-" * 56)
for cls, cnt in sorted(cls_counts_test.items(), key=lambda x: -x[1]):
    bar = '#' * int(cnt / len(test_cids) * 32)
    print(f"  {cls:<35} {cnt:>5}  {cnt/len(test_cids)*100:>5.1f}%  {bar}")
print("  " + "-" * 56)

results_df = pd.DataFrame({
    'pubchem_id':     cid_list,
    'y_true':         y_true,
    'y_pred_kagnn':   y_kagnn,
    'y_pred_final':   y_final,
    'chemical_class': [chem_class_map.get(int(c), 'Unknown') for c in cid_list],
})

class_summary = classwise_analysis(results_df, config)

csv_out = config.results_dir / 'classwise_metrics.csv'
class_summary.to_csv(csv_out, index=False)
print(f"Saved to Drive: {csv_out}")

majority_bias_verdict(class_summary, metrics_kagnn, metrics_final,
                      test_cids, data_loader.mol_dict)

# ── Final summary ─────────────────────────────────────────────────────────────
t_total = stage1_time + stage2_time
print(f"""
============================================================
  KAGNN -> PGM FORWARD HYBRID -- EXPERIMENT COMPLETE
============================================================
  Stage 1  ({stage1_time/60:>5.1f} min)  KAGNN (5xGATv2 + KAN)
  Stage 2  ({stage2_time/60:>5.1f} min)  PGM (XGBoost + BayesRidge)
  Total    ({t_total/60:>5.1f} min)
------------------------------------------------------------
  KAGNN  MedAE  : {metrics_kagnn['MedAE']:>7.2f} s
  Hybrid MedAE  : {metrics_final['MedAE']:>7.2f} s     y_final = y_KAGNN + r_PGM
  R2 Score      : {metrics_final['R2']:>7.4f}
  % within 30s  : {metrics_final['Pct_le_30s']:>7.2f} %
  Paired t-test : p = {p_ttest:.6f}  [{('SIGNIFICANT' if p_ttest < 0.05 else 'not significant')}]
  Wilcoxon      : p = {p_wilcoxon:.6f}  [{('SIGNIFICANT' if p_wilcoxon < 0.05 else 'not significant')}]
------------------------------------------------------------
  All results saved to: {config.results_dir}/
    plots/best_kagnn.pth
    plots/training_history.png
    plots/TRUE_KAGNN_Baseline_results.png
    plots/KAGNN_+_PGM_results.png
    plots/comprehensive_comparison.png
    plots/classwise_MedAE_hybrid.png       <- REVIEWER CHART
    plots/classwise_MedRE_hybrid.png
    plots/classwise_size_vs_error.png
    plots/classwise_kagnn_vs_hybrid_grouped.png
    metrics_kagnn.json
    metrics_final.json
    classwise_metrics.csv
============================================================
""")
