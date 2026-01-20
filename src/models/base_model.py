"""
KAGNN BASELINE NOTEBOOK
Complete implementation with metrics, plots, and statistical tests
"""

# ============================================================================
# CELL 1: IMPORTS AND SETUP
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from models import KAGIN, make_kan, allowable_features, get_atom_feature_dims, get_bond_feature_dims
import warnings
import time
import json
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print("✓ All imports successful")

# ============================================================================
# CELL 2: CONFIGURATION
# ============================================================================
class Config:
    # Data paths - UPDATE THESE FOR YOUR ENVIRONMENT
    ecfp_file = "/kaggle/input/smrtdatas/SMRT_ECFP_1024_Fingerprints.txt"
    rt_file = "/kaggle/input/smrtdt/SMRT_dataset.csv"
    sdf_file = "/kaggle/input/smrtdatas/SMRT_dataset.sdf"

    # Model hyperparameters
    batch_size = 64
    epochs = 150
    lr = 3e-4
    weight_decay = 1e-5
    gnn_layers = 5
    hidden_dim = 256
    hidden_layers = 2
    grid_size = 4
    spline_order = 3
    dropout = 0.1

    # Data split
    val_size = 0.15
    test_size = 0.15
    seed = 42
    
    # Output paths
    output_dir = Path("kagnn_baseline_outputs")
    checkpoint_path = output_dir / "best_kagnn_baseline.pt"
    metrics_path = output_dir / "metrics.json"

cfg = Config()
cfg.output_dir.mkdir(exist_ok=True)

# Set seeds
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ Configuration set")
print(f"  Device: {device}")
print(f"  Output directory: {cfg.output_dir}")

# ============================================================================
# CELL 3: FEATURE EXTRACTION
# ============================================================================
def atom_to_indices(atom):
    zs = allowable_features['possible_atomic_num_list']
    ch_list = allowable_features['possible_chirality_list']
    deg_list = allowable_features['possible_degree_list']
    fc_list = allowable_features['possible_formal_charge_list']
    h_list = allowable_features['possible_numH_list']
    rad_list = allowable_features['possible_number_radical_e_list']
    hyb_list = allowable_features['possible_hybridization_list']
    arom_list = allowable_features['possible_is_aromatic_list']
    ring_list = allowable_features['possible_is_in_ring_list']

    feats = []
    feats.append(zs.index(atom.GetAtomicNum()) if atom.GetAtomicNum() in zs else zs.index('misc'))

    ch_val = str(atom.GetChiralTag())
    if 'CHI_TETRAHEDRAL_CW' in ch_val: ch_val = 'CHI_TETRAHEDRAL_CW'
    elif 'CHI_TETRAHEDRAL_CCW' in ch_val: ch_val = 'CHI_TETRAHEDRAL_CCW'
    elif 'CHI_UNSPECIFIED' in ch_val: ch_val = 'CHI_UNSPECIFIED'
    else: ch_val = 'CHI_OTHER'
    feats.append(ch_list.index(ch_val) if ch_val in ch_list else ch_list.index('misc'))

    deg = atom.GetDegree()
    feats.append(deg_list.index(deg) if deg in deg_list else deg_list.index('misc'))

    fc = atom.GetFormalCharge()
    feats.append(fc_list.index(fc) if fc in fc_list else fc_list.index('misc'))

    nh = atom.GetTotalNumHs()
    feats.append(h_list.index(nh) if nh in h_list else h_list.index('misc'))

    nr = atom.GetNumRadicalElectrons()
    feats.append(rad_list.index(nr) if nr in rad_list else rad_list.index('misc'))

    hyb = str(atom.GetHybridization())
    hval = 'misc'
    if 'SP' in hyb: hval = 'SP'
    elif 'SP2' in hyb: hval = 'SP2'
    elif 'SP3' in hyb: hval = 'SP3'
    elif 'SP3D' in hyb: hval = 'SP3D'
    elif 'SP3D2' in hyb: hval = 'SP3D2'
    feats.append(hyb_list.index(hval) if hval in hyb_list else hyb_list.index('misc'))

    feats.append(arom_list.index(atom.GetIsAromatic()))
    feats.append(ring_list.index(atom.IsInRing()))
    return feats

def bond_to_indices(bond):
    bt_list = allowable_features['possible_bond_type_list']
    st_list = allowable_features['possible_bond_stereo_list']
    conj_list = allowable_features['possible_is_conjugated_list']

    btype = str(bond.GetBondType())
    bval = 'misc'
    if 'SINGLE' in btype: bval = 'SINGLE'
    elif 'DOUBLE' in btype: bval = 'DOUBLE'
    elif 'TRIPLE' in btype: bval = 'TRIPLE'
    elif 'AROMATIC' in btype: bval = 'AROMATIC'

    stype = str(bond.GetStereo())
    mapped = 'STEREOANY'
    for s in st_list:
        if s in stype: mapped = s
    return [
        bt_list.index(bval) if bval in bt_list else bt_list.index('misc'),
        st_list.index(mapped),
        conj_list.index(bond.GetIsConjugated())
    ]

print("✓ Feature extraction functions defined")

# ============================================================================
# CELL 4: DATASET
# ============================================================================
class SMRTCombinedDataset(Dataset):
    def __init__(self, cfg, split='train'):
        # Load RT data
        rt_df = pd.read_csv(cfg.rt_file, sep=';')
        cid_col = next((c for c in rt_df.columns if 'cid' in c.lower()), rt_df.columns[0])
        rt_col = next((c for c in rt_df.columns if 'rt' in c.lower() or 'retention' in c.lower() or 'time' in c.lower()), None)
        if rt_col is None:
            for c in rt_df.columns:
                if c != cid_col and pd.api.types.is_numeric_dtype(rt_df[c]):
                    rt_col = c
                    break
        rt_df = rt_df[[cid_col, rt_col]].copy()
        rt_df.columns = ['cid', 'rt']
        rt_df['cid'] = pd.to_numeric(rt_df['cid'], errors='coerce').astype(int)
        rt_df = rt_df.dropna()

        # Load ECFP
        with open(cfg.ecfp_file, 'r') as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('Extended') and line[0].isdigit()]
        cid_list, bit_strings = [], []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cid_list.append(int(parts[0]))
                    bit_strings.append(parts[1][:1024])
                except:
                    continue
        ecfp_array = np.array([[int(b) for b in bs] for bs in bit_strings], dtype=np.float32)
        ecfp_df = pd.DataFrame(ecfp_array, index=cid_list)
        merged_df = pd.merge(rt_df, ecfp_df, left_on='cid', right_index=True, how='inner')

        # Load SDF and build graphs
        suppl = Chem.SDMolSupplier(cfg.sdf_file)
        self.graph_dict = {}
        cids_with_graphs = set()
        for mol in tqdm(suppl, desc=f"Building graphs ({split})"):
            if mol is None: continue
            try:
                cid = int(mol.GetProp("PUBCHEM_COMPOUND_CID"))
            except KeyError:
                continue
            if cid not in merged_df['cid'].values:
                continue
            x = torch.tensor([atom_to_indices(a) for a in mol.GetAtoms()], dtype=torch.long)
            edge_index, edge_attr = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                feats = bond_to_indices(bond)
                edge_index.extend([[i,j],[j,i]])
                edge_attr.extend([feats, feats])
            if len(edge_index) == 0:
                edge_index = [[0,0]]
                edge_attr = [[0,0,0]]
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            self.graph_dict[cid] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            cids_with_graphs.add(cid)

        self.data_df = merged_df[merged_df['cid'].isin(cids_with_graphs)].reset_index(drop=True)
        self.ecfp = torch.tensor(self.data_df.iloc[:, 2:].values, dtype=torch.float32)
        self.cids = self.data_df['cid'].values
        self.rt_raw = self.data_df['rt'].values.astype(np.float32)
        self.rt_mean = np.mean(self.rt_raw)
        self.rt_std = np.std(self.rt_raw)
        self.rt_norm = (self.rt_raw - self.rt_mean) / self.rt_std

        n = len(self.data_df)
        indices = np.arange(n)
        np.random.seed(cfg.seed)
        np.random.shuffle(indices)
        n_val = int(cfg.val_size * n)
        n_test = int(cfg.test_size * n)

        if split == 'train': self.indices = indices[: n - n_val - n_test]
        elif split == 'val': self.indices = indices[n - n_val - n_test: n - n_test]
        elif split == 'test': self.indices = indices[n - n_test:]
        else: raise ValueError("Invalid split")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        cid = self.cids[i]
        return self.graph_dict[cid], self.ecfp[i], torch.tensor(self.rt_norm[i], dtype=torch.float32)

def collate_fn(batch):
    graphs, ecfps, rts = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    return batch_graph, torch.stack(ecfps), torch.stack(rts)

print("✓ Dataset class defined")

# ============================================================================
# CELL 5: LOAD DATA
# ============================================================================
print("Loading datasets...")
train_ds = SMRTCombinedDataset(cfg, 'train')
val_ds = SMRTCombinedDataset(cfg, 'val')
test_ds = SMRTCombinedDataset(cfg, 'test')

train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, cfg.batch_size, shuffle=False, collate_fn=collate_fn)

print(f"\n✓ Data loaded successfully")
print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")
print(f"  RT normalization: mean={train_ds.rt_mean:.2f}, std={train_ds.rt_std:.2f}")

# ============================================================================
# CELL 6: MODEL DEFINITION
# ============================================================================
class KAGIN_ECFP_Combined(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kagnn = KAGIN(len(get_atom_feature_dims()), len(get_bond_feature_dims()),
                           cfg.gnn_layers, cfg.hidden_dim, cfg.hidden_layers,
                           cfg.grid_size, cfg.spline_order, cfg.hidden_dim, cfg.dropout,
                           ogb_encoders=True)
        self.ecfp_kan = make_kan(1024, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.grid_size, cfg.spline_order)
        self.final_kan = make_kan(2*cfg.hidden_dim, cfg.hidden_dim//2, 1, cfg.hidden_layers, cfg.grid_size, cfg.spline_order)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, graph_data, ecfp):
        graph_emb = self.kagnn(graph_data)
        ecfp_emb = self.ecfp_kan(ecfp)
        x = torch.cat([graph_emb, ecfp_emb], dim=-1)
        x = self.dropout(x)
        return self.final_kan(x).squeeze(-1)

model = KAGIN_ECFP_Combined(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
criterion = nn.SmoothL1Loss()

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model initialized")
print(f"  Total parameters: {total_params:,}")

# ============================================================================
# CELL 7: TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING KAGNN BASELINE")
print("="*80)

best_val_loss = float("inf")
patience = 20
patience_counter = 0
train_losses = []
val_losses = []

start_time = time.time()

for epoch in range(cfg.epochs):
    # Training
    model.train()
    train_loss = 0
    for g, ecfp, rt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
        g, ecfp, rt = g.to(device), ecfp.to(device), rt.to(device)
        optimizer.zero_grad()
        pred = model(g, ecfp)
        loss = criterion(pred, rt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(rt)
    
    train_loss /= len(train_ds)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for g, ecfp, rt in val_loader:
            g, ecfp, rt = g.to(device), ecfp.to(device), rt.to(device)
            pred = model(g, ecfp)
            val_loss += criterion(pred, rt).item() * len(rt)
    val_loss /= len(val_ds)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss - 1e-3:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), cfg.checkpoint_path)
        print(f"  ✓ Saved checkpoint (Val Loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

# Load best model
model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

# ============================================================================
# CELL 8: METRICS COMPUTATION
# ============================================================================
def compute_all_metrics(y_true, y_pred):
    """Compute all metrics"""
    abs_errors = np.abs(y_true - y_pred)
    
    metrics = {
        'MedAE': np.median(abs_errors),
        'MAE': np.mean(abs_errors),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0],
        'Spearman': spearmanr(y_true, y_pred)[0],
        'Pct_le_60s': (abs_errors <= 60).mean() * 100,
        'Pct_le_30s': (abs_errors <= 30).mean() * 100,
        'Pct_le_10s': (abs_errors <= 10).mean() * 100,
        'Median_Rel_Error': np.median(np.abs((y_true - y_pred) / y_true) * 100),
        'Mean_Rel_Error': np.mean(np.abs((y_true - y_pred) / y_true) * 100)
    }
    return metrics

def perform_statistical_tests(y_true, y_pred_current, y_pred_baseline=None):
    """Statistical significance tests"""
    errors_current = np.abs(y_true - y_pred_current)
    
    results = {
        'Mean Error': np.mean(errors_current),
        'Std Error': np.std(errors_current),
        'Error 95% CI': (np.percentile(errors_current, 2.5), np.percentile(errors_current, 97.5))
    }
    
    if y_pred_baseline is not None:
        errors_baseline = np.abs(y_true - y_pred_baseline)
        
        wilcoxon_stat, wilcoxon_p = wilcoxon(errors_current, errors_baseline)
        results['Wilcoxon_statistic'] = wilcoxon_stat
        results['Wilcoxon_p_value'] = wilcoxon_p
        
        ttest_stat, ttest_p = ttest_rel(errors_current, errors_baseline)
        results['T_test_statistic'] = ttest_stat
        results['T_test_p_value'] = ttest_p
        
        mean_diff = np.mean(errors_current - errors_baseline)
        pooled_std = np.sqrt((np.std(errors_current)**2 + np.std(errors_baseline)**2) / 2)
        results['Cohens_d'] = mean_diff / pooled_std
        
    return results

print("✓ Metrics functions defined")

# ============================================================================
# CELL 9: EVALUATION
# ============================================================================
print("\n" + "="*80)
print("EVALUATION ON TEST SET")
print("="*80)

model.eval()
y_true_all, y_pred_all = [], []

with torch.no_grad():
    for g, ecfp, rt in tqdm(test_loader, desc="Evaluating"):
        g, ecfp, rt = g.to(device), ecfp.to(device), rt.to(device)
        pred = model(g, ecfp)
        y_true_all.append(rt.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

y_true_norm = np.concatenate(y_true_all)
y_pred_norm = np.concatenate(y_pred_all)

# Denormalize
y_true = y_true_norm * train_ds.rt_std + train_ds.rt_mean
y_pred = y_pred_norm * train_ds.rt_std + train_ds.rt_mean

# Compute metrics
metrics = compute_all_metrics(y_true, y_pred)

# Statistical tests (vs mean predictor baseline)
y_pred_baseline = np.full_like(y_true, train_ds.rt_mean)
stat_results = perform_statistical_tests(y_true, y_pred, y_pred_baseline)

# Save metrics
output_data = {
    'model_name': 'KAGNN+ECFP Baseline',
    'training_time_seconds': training_time,
    'n_samples': len(y_true),
    'metrics': {k: float(v) for k, v in metrics.items()},
    'statistical_tests': {k: float(v) if not isinstance(v, tuple) else [float(x) for x in v] 
                         for k, v in stat_results.items()}
}

with open(cfg.metrics_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Metrics saved to {cfg.metrics_path}")

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nCORE METRICS:")
print(f"  MedAE       : {metrics['MedAE']:8.2f} s")
print(f"  MAE         : {metrics['MAE']:8.2f} s")
print(f"  RMSE        : {metrics['RMSE']:8.2f} s")
print(f"  R²          : {metrics['R2']:8.3f}")
print(f"  Pearson     : {metrics['Pearson']:8.3f}")
print(f"  Spearman    : {metrics['Spearman']:8.3f}")

print(f"\nTHRESHOLD ACCURACY:")
print(f"  % ≤ 60s     : {metrics['Pct_le_60s']:8.2f} %")
print(f"  % ≤ 30s     : {metrics['Pct_le_30s']:8.2f} %")
print(f"  % ≤ 10s     : {metrics['Pct_le_10s']:8.2f} %")

print(f"\nSTATISTICAL TESTS:")
print(f"  Wilcoxon p  : {stat_results['Wilcoxon_p_value']:.2e}")
print(f"  T-test p    : {stat_results['T_test_p_value']:.2e}")
print(f"  Cohen's d   : {stat_results['Cohens_d']:.3f}")
print("="*80)

# ============================================================================
# CELL 10: COMPREHENSIVE VISUALIZATION
# ============================================================================
from scipy import stats

print("\nGenerating comprehensive visualization...")

abs_errors = np.abs(y_true - y_pred)
errors = y_pred - y_true

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
fig.suptitle('KAGNN+ECFP Baseline - Comprehensive RT Prediction Analysis', 
             fontsize=24, fontweight='bold')

# 1. Scatter plot
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(y_true, y_pred, c=abs_errors, cmap='RdYlGn_r', 
                     alpha=0.6, s=15, vmin=0, vmax=np.percentile(abs_errors, 95))
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax1.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Experimental RT (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted RT (s)', fontsize=12, fontweight='bold')
ax1.set_title(f'Prediction Scatter\nPearson={metrics["Pearson"]:.3f}, R²={metrics["R2"]:.3f}',
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Abs Error (s)')

# 2. Error distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(errors, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, 
           label=f'Median={np.median(errors):.1f}s')
ax2.set_xlabel('Prediction Error (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title(f'Error Distribution\nMean={np.mean(errors):.1f}s, Std={np.std(errors):.1f}s',
             fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Absolute error distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(abs_errors, bins=60, color='coral', edgecolor='black', alpha=0.7)
ax3.axvline(metrics['MedAE'], color='red', linestyle='--', linewidth=2, 
           label=f'Median={metrics["MedAE"]:.1f}s')
ax3.axvline(metrics['MAE'], color='blue', linestyle='--', linewidth=2, 
           label=f'Mean={metrics["MAE"]:.1f}s')
ax3.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title(f'Absolute Error Distribution', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Relative error distribution
ax4 = fig.add_subplot(gs[0, 3])
rel_errors = np.abs((y_true - y_pred) / y_true) * 100
rel_errors_clipped = np.clip(rel_errors, 0, 100)
ax4.hist(rel_errors_clipped, bins=60, color='forestgreen', edgecolor='black', alpha=0.7)
ax4.axvline(metrics['Median_Rel_Error'], color='red', linestyle='--', linewidth=2, 
           label=f'Median={metrics["Median_Rel_Error"]:.1f}%')
ax4.set_xlabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Residuals plot
ax5 = fig.add_subplot(gs[1, 0])
ax5.scatter(y_pred, errors, alpha=0.4, s=10, color='purple')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.axhline(np.mean(errors) + 1.96*np.std(errors), color='orange', linestyle=':', linewidth=1.5)
ax5.axhline(np.mean(errors) - 1.96*np.std(errors), color='orange', linestyle=':', linewidth=1.5)
ax5.set_xlabel('Predicted RT (s)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Residual (s)', fontsize=12, fontweight='bold')
ax5.set_title('Residuals Plot (±1.96σ bands)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. CDF of absolute errors
ax6 = fig.add_subplot(gs[1, 1])
sorted_errs = np.sort(abs_errors)
cdf = np.arange(1, len(sorted_errs)+1) / len(sorted_errs)
ax6.plot(sorted_errs, cdf, 'b-', linewidth=2, label='KAGNN Model')
ax6.axvline(10, color='green', linestyle='--', alpha=0.7, label='10s threshold')
ax6.axvline(30, color='orange', linestyle='--', alpha=0.7, label='30s threshold')
ax6.axvline(60, color='red', linestyle='--', alpha=0.7, label='60s threshold')
ax6.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
ax6.set_title('CDF of Absolute Errors', fontsize=14
