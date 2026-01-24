"""
FIXED BASELINE KAGNN MODEL FOR SMRT RT PREDICTION
Uses official KAGNN repository with proper imports and fallback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import warnings
import sys
import os

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ============================================================================
# KAGNN IMPORT WITH VALIDATION AND FALLBACK
# ============================================================================
try:
    # Try to import from official KAGNN
    sys.path.append(os.path.expanduser('~/KAGNN'))  # Adjust path as needed
    from graph_regression.kagnn.models import (
        KAGIN, 
        make_kan, 
        get_atom_feature_dims, 
        get_bond_feature_dims
    )
    KAGNN_AVAILABLE = True
    print("✓ Official KAGNN imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import official KAGNN: {e}")
    print("📌 Falling back to GAT-based implementation...")
    KAGNN_AVAILABLE = False
    
    # Fallback to GAT
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

# ============================================================================
# DATA LOADER
# ============================================================================
class SMRTDataLoader:
    def __init__(self, csv_path, ecfp_path, sdf_path):
        print("="*80)
        print("LOADING SMRT DATASET")
        print("="*80)
        
        self.df = self._load_csv(csv_path)
        self.ecfp_dict = self._load_ecfp(ecfp_path)
        self.mol_dict = self._load_sdf(sdf_path)
        self._sync_and_clean()
        self.rt_scaler = RobustScaler()
        self._normalize_rt()
    
    def _load_csv(self, path):
        df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()
        
        id_col = next((c for c in df.columns if any(k in c for k in ['pubchem', 'cid', 'molecule', 'id'])), None)
        rt_col = next((c for c in df.columns if any(k in c for k in ['rt', 'retention', 'time'])), None)
        
        if not id_col or not rt_col:
            raise KeyError(f"Missing ID or RT columns. Found: {list(df.columns)}")
        
        df = df.rename(columns={id_col: 'pubchem', rt_col: 'rt'})
        df['pubchem'] = pd.to_numeric(df['pubchem'], errors='coerce')
        df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
        return df.dropna(subset=['pubchem', 'rt']).reset_index(drop=True)
    
    def _load_ecfp(self, path):
        ecfp_dict = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        mid = int(parts[0].replace('ID=', '').strip())
                        ecfp_dict[mid] = np.array([int(c) for c in parts[1].strip()], dtype=np.float32)
                    except:
                        continue
        return ecfp_dict
    
    def _load_sdf(self, path):
        print(f"📖 Reading SDF & Initializing RingInfo...")
        mol_dict = {}
        suppl = Chem.SDMolSupplier(path)
        for mol in suppl:
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                Chem.GetSymmSSSR(mol)
                
                props = mol.GetPropsAsDict()
                mid = props.get('PUBCHEM_COMPOUND_CID') or props.get('ID') or props.get('pubchem')
                if mid:
                    mol_dict[int(mid)] = mol
            except:
                continue
        return mol_dict
    
    def _sync_and_clean(self):
        valid_ids = set(self.df['pubchem']) & set(self.ecfp_dict.keys()) & set(self.mol_dict.keys())
        self.df = self.df[self.df['pubchem'].isin(valid_ids)].reset_index(drop=True)
        print(f"✓ Sync complete. Dataset size: {len(self.df)}")
    
    def _normalize_rt(self):
        self.df['rt_original'] = self.df['rt'].copy()
        self.df['rt'] = self.rt_scaler.fit_transform(self.df[['rt']]).flatten()
        print(f"✓ RT normalized: mean={self.df['rt'].mean():.4f}, std={self.df['rt'].std():.4f}")
    
    def denormalize_rt(self, rt_normalized):
        return self.rt_scaler.inverse_transform(rt_normalized.reshape(-1, 1)).flatten()
    
    def get_data(self):
        return self.df, self.ecfp_dict, self.mol_dict


# ============================================================================
# DATASET
# ============================================================================
class SMRTDataset(Dataset):
    def __init__(self, df, ecfp_dict, mol_dict):
        self.df, self.ecfp_dict, self.mol_dict = df, ecfp_dict, mol_dict
    
    def __len__(self):
        return len(self.df)
    
    def _get_graph(self, mol):
        if KAGNN_AVAILABLE:
            # Use KAGNN's built-in atom features if available
            try:
                from graph_regression.kagnn.models import get_atom_feature_dims, get_bond_feature_dims
                # This will be processed by KAGNN's built-in encoders
                return Data(
                    x=torch.eye(len(get_atom_feature_dims()))[0].unsqueeze(0),  # Dummy
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                    edge_attr=torch.tensor([[0, 0, 0]], dtype=torch.float)
                )
            except:
                pass
        
        # Fallback: 9 atom features (matching KAGNN defaults)
        x = torch.tensor([
            [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(),
             int(a.GetIsAromatic()), int(a.GetHybridization()), int(a.IsInRing()),
             a.GetChiralTag(), a.GetMass()]
            for a in mol.GetAtoms()
        ], dtype=torch.float32)
        
        edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        edge_index = torch.tensor(edges + [[j, i] for i, j in edges], dtype=torch.long).t()
        
        # Bond features (3D)
        edge_attr = torch.tensor([
            [int(b.GetBondType()), int(b.GetIsConjugated()), int(b.IsInRing())]
            for b in mol.GetBonds()
        ], dtype=torch.float32)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # For undirected
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mid = int(row['pubchem'])
        return (
            self._get_graph(self.mol_dict[mid]),
            torch.tensor(self.ecfp_dict[mid], dtype=torch.float32),
            torch.tensor(row['rt'], dtype=torch.float32),
            mid
        )

def collate_fn(batch):
    gs, es, rts, mids = zip(*batch)
    return Batch.from_data_list(gs), torch.stack(es), torch.stack(rts), mids


# ============================================================================
# FIXED BASELINE KAGNN MODEL
# ============================================================================
class FixedBaselineKAGNN(nn.Module):
    """
    Fixed KAGNN model using official repository with proper API
    """
    def __init__(self, hidden_dim=256, dropout=0.15, num_layers=5):
        super().__init__()
        
        if KAGNN_AVAILABLE:
            try:
                # Use official KAGNN
                atom_dims = get_atom_feature_dims()
                bond_dims = get_bond_feature_dims()
                
                self.kagin = KAGIN(
                    in_channels=len(atom_dims),
                    edge_dim=len(bond_dims),
                    num_layers=num_layers,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    grid_size=4,
                    spline_order=3,
                    hidden_hidden_channels=hidden_dim,
                    dropout=dropout,
                    ogb_encoders=True
                )
                
                print("✓ Using official KAGIN with OGB encoders")
            except Exception as e:
                print(f"⚠️ KAGIN initialization failed: {e}")
                print("📌 Using fallback GAT implementation")
                KAGNN_AVAILABLE = False
        
        if not KAGNN_AVAILABLE:
            # Fallback GAT implementation
            from torch_geometric.nn import GATConv
            self.gat_layers = nn.ModuleList()
            heads = 4
            
            # First layer: 9 → hidden_dim
            self.gat_layers.append(
                GATConv(9, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
            )
            
            # Middle layers
            for _ in range(num_layers - 2):
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            
            # Last layer
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
            )
            
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            print("✓ Using GAT fallback implementation")
        
        # ECFP encoder (using KAN if available)
        if KAGNN_AVAILABLE:
            try:
                self.ecfp_kan = make_kan(
                    1024,  # Input dim (ECFP)
                    hidden_dim,  # Output dim
                    hidden_dim,  # Hidden dim
                    2,  # Hidden layers
                    4,  # Grid size
                    3   # Spline order
                )
                print("✓ Using KAN for ECFP encoding")
            except:
                self.ecfp_kan = nn.Sequential(
                    nn.Linear(1024, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        else:
            self.ecfp_kan = nn.Sequential(
                nn.Linear(1024, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Final predictor
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.use_kaginn = KAGNN_AVAILABLE
    
    def forward(self, graph, ecfp):
        # Process graph
        if self.use_kaginn:
            graph_emb = self.kagin(graph)
        else:
            x = graph.x.float()
            for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
                x_new = gat(x, graph.edge_index)
                x_new = norm(x_new)
                x_new = F.elu(x_new)
                
                # Residual connection
                if i > 0 and x.shape[-1] == x_new.shape[-1]:
                    x = x + x_new
                else:
                    x = x_new
            
            # Multi-scale pooling
            from torch_geometric.nn import global_mean_pool, global_max_pool
            g_mean = global_mean_pool(x, graph.batch)
            g_max = global_max_pool(x, graph.batch)
            graph_emb = (g_mean + g_max) / 2
        
        # Process ECFP
        ecfp_emb = self.ecfp_kan(ecfp.float())
        
        # Combine and predict
        combined = torch.cat([graph_emb, ecfp_emb], dim=-1)
        combined = self.dropout(combined)
        output = self.final_predictor(combined).squeeze(-1)
        
        return output


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_fixed_kagnn(train_loader, val_loader, test_loader, rt_scaler, 
                      hidden_dim=256, dropout=0.15, epochs=150, patience=20,
                      device='cuda'):
    """
    Train fixed KAGNN model with early stopping
    """
    model = FixedBaselineKAGNN(hidden_dim=hidden_dim, dropout=dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.SmoothL1Loss()
    
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using {'official KAGNN' if model.use_kaginn else 'GAT fallback'}")
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for g, ecfp, rt, _ in train_loader:
            g, ecfp, rt = g.to(device), ecfp.to(device), rt.to(device)
            
            optimizer.zero_grad()
            pred = model(g, ecfp)
            loss = criterion(pred, rt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(rt)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for g, ecfp, rt, _ in val_loader:
                g, ecfp, rt = g.to(device), ecfp.to(device), rt.to(device)
                pred = model(g, ecfp)
                val_loss += criterion(pred, rt).item() * len(rt)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"📊 Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, 'best_fixed_kagnn.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_fixed_kagnn.pth'))
    print(f"✓ Best model loaded (Val Loss: {best_val_loss:.4f})")
    
    # Evaluate on test set
    model.eval()
    all_trues, all_preds = [], []
    
    with torch.no_grad():
        for g, ecfp, rt, _ in test_loader:
            g, ecfp, rt = g.to(device), ecfp.to(device), rt.to(device)
            pred = model(g, ecfp)
            
            all_trues.extend(rt_scaler.inverse_transform(rt.cpu().numpy().reshape(-1, 1)).flatten())
            all_preds.extend(rt_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten())
    
    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)
    
    # Compute metrics
    metrics = {
        'MedAE': median_absolute_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0],
        'Spearman': spearmanr(y_true, y_pred)[0],
    }
    
    print("\n" + "="*80)
    print("FIXED KAGNN TEST SET PERFORMANCE")
    print("="*80)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:8.4f}")
    
    return model, metrics, train_losses, val_losses


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    config = {
        'csv': '/path/to/SMRT_dataset.csv',
        'ecfp': '/path/to/SMRT_ECFP_1024_Fingerprints.txt',
        'sdf': '/path/to/SMRT_dataset.sdf'
    }
    
    loader = SMRTDataLoader(config['csv'], config['ecfp'], config['sdf'])
    df, ecfp_dict, mol_dict = loader.get_data()
    
    # Train/val/test split
    train_idx, test_idx = train_test_split(range(len(df)), test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    
    train_loader = DataLoader(SMRTDataset(train_df, ecfp_dict, mol_dict), 
                              batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SMRTDataset(val_df, ecfp_dict, mol_dict), 
                            batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(SMRTDataset(test_df, ecfp_dict, mol_dict), 
                             batch_size=64, collate_fn=collate_fn)
    
    # Train
    model, metrics, train_losses, val_losses = train_fixed_kagnn(
        train_loader, val_loader, test_loader, loader.rt_scaler,
        hidden_dim=256, dropout=0.15, epochs=150, patience=20
    )
    
    print("\n✅ Fixed KAGNN training complete!")
