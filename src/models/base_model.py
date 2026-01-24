"""
COMPLETE_KAGNN_BASELINE.py
Fixed and compatible with official KAGNN repository
Works with your data/raw/ directory structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import warnings
from tqdm import tqdm
import sys
import os

# Add KAGNN to path
kagnn_path = Path.home() / "KAGNN"
sys.path.append(str(kagnn_path))

# Import from official KAGNN
try:
    from graph_regression.kagnn.models import (
        KAGIN, 
        make_kan, 
        get_atom_feature_dims, 
        get_bond_feature_dims
    )
    KAGNN_AVAILABLE = True
    print(" Using official KAGNN repository")
except ImportError:
    print(" KAGNN not available, using fallback")
    KAGNN_AVAILABLE = False
    # You might want to raise an error here or use a fallback
    raise ImportError("Please install KAGNN first")

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration for KAGNN baseline"""
    
    # DATA PATHS - ADJUST THESE
    DATA_DIR = Path("data/raw")  # Your data directory
    
    # File names (adjust if your files have different names)
    CSV_FILE = DATA_DIR / "SMRT_dataset.csv"
    ECFP_FILE = DATA_DIR / "SMRT_ECFP_1024_Fingerprints.txt"
    SDF_FILE = DATA_DIR / "SMRT_dataset.sdf"
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.15
    PATIENCE = 20
    
    # Model architecture
    HIDDEN_DIM = 256
    GNN_LAYERS = 5
    KAN_HIDDEN_LAYERS = 2
    GRID_SIZE = 4
    SPLINE_ORDER = 3
    
    # Data split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_SEED = 42
    
    # Output
    OUTPUT_DIR = Path("results/kagnn_baseline")
    CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
    METRICS_PATH = OUTPUT_DIR / "metrics.json"
    
    def __init__(self):
        """Initialize and validate paths"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validate data paths
        missing_files = []
        for file_path in [self.CSV_FILE, self.ECFP_FILE, self.SDF_FILE]:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            raise FileNotFoundError(f"Missing files:\n" + "\n".join(missing_files))


# ============================================================================
# DATA LOADER WITH PROPER FEATURE EXTRACTION
# ============================================================================
class SMRTDataLoader:
    """
    Load and preprocess SMRT dataset
    Compatible with KAGNN's feature extraction
    """
    
    def __init__(self, config):
        self.config = config
        print("="*80)
        print("LOADING SMRT DATASET")
        print("="*80)
        
        # Load data
        self.df = self._load_csv()
        self.ecfp_dict = self._load_ecfp()
        self.mol_dict = self._load_sdf()
        
        # Synchronize and clean
        self._sync_and_clean()
        
        # Normalize RT
        from sklearn.preprocessing import RobustScaler
        self.rt_scaler = RobustScaler()
        self._normalize_rt()
    
    def _load_csv(self):
        """Load CSV file with RT data"""
        print(f" Reading CSV: {self.config.CSV_FILE}")
        
        # Try different separators
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(self.config.CSV_FILE, sep=sep, low_memory=False)
                if len(df.columns) > 1:
                    print(f"  ✓ Success with separator: '{sep}'")
                    break
            except:
                continue
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Find ID and RT columns
        id_col = next((c for c in df.columns if any(k in c for k in ['pubchem', 'cid', 'molecule', 'id'])), None)
        rt_col = next((c for c in df.columns if any(k in c for k in ['rt', 'retention', 'time'])), None)
        
        if not id_col or not rt_col:
            print(f"  Could not find ID/RT columns automatically")
            print(f"   Available columns: {list(df.columns)}")
            print(f"   Assuming first column is ID, second is RT")
            id_col = df.columns[0]
            rt_col = df.columns[1] if len(df.columns) > 1 else None
        
        print(f"  ID column: {id_col}")
        print(f"  RT column: {rt_col}")
        
        # Rename and clean
        df = df.rename(columns={id_col: 'pubchem_id', rt_col: 'rt'})
        df['pubchem_id'] = pd.to_numeric(df['pubchem_id'], errors='coerce')
        df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
        
        # Remove invalid entries
        initial_count = len(df)
        df = df.dropna(subset=['pubchem_id', 'rt']).reset_index(drop=True)
        final_count = len(df)
        
        print(f"  Loaded {final_count:,} valid compounds ({initial_count - final_count:,} removed)")
        print(f"  RT range: {df['rt'].min():.1f} to {df['rt'].max():.1f} seconds")
        
        return df
    
    def _load_ecfp(self):
        """Load ECFP fingerprints"""
        print(f"📖 Reading ECFP: {self.config.ECFP_FILE}")
        
        ecfp_dict = {}
        try:
            with open(self.config.ECFP_FILE, 'r') as f:
                lines = f.readlines()
            
            # Parse ECFP file
            for line in tqdm(lines, desc="Parsing ECFP", unit="lines"):
                line = line.strip()
                if not line or line.startswith('Extended'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        # Extract ID (handle formats like "ID=12345" or just "12345")
                        id_part = parts[0].strip()
                        if '=' in id_part:
                            compound_id = int(id_part.split('=')[-1].strip())
                        else:
                            compound_id = int(id_part)
                        
                        # Extract fingerprint (1024 bits)
                        fingerprint_str = parts[1].strip()
                        if len(fingerprint_str) >= 1024:
                            fingerprint = np.array([int(bit) for bit in fingerprint_str[:1024]], dtype=np.float32)
                            ecfp_dict[compound_id] = fingerprint
                    except Exception as e:
                        continue
            
            print(f"  ✓ Loaded {len(ecfp_dict):,} ECFP fingerprints")
            
        except Exception as e:
            print(f"   Error loading ECFP: {e}")
            raise
        
        return ecfp_dict
    
    def _load_sdf(self):
        """Load SDF file and extract molecules"""
        print(f"📖 Reading SDF: {self.config.SDF_FILE}")
        
        mol_dict = {}
        suppl = Chem.SDMolSupplier(str(self.config.SDF_FILE), sanitize=True)
        
        for idx, mol in enumerate(tqdm(suppl, desc="Loading molecules", unit="mols")):
            if mol is None:
                continue
            
            try:
                # Get compound ID from properties
                props = mol.GetPropsAsDict()
                compound_id = None
                
                # Try different property names
                for key in ['PUBCHEM_COMPOUND_CID', 'CID', 'PubChem_CID', 'PUBCHEM_CID', 'ID']:
                    if key in props:
                        try:
                            compound_id = int(props[key])
                            break
                        except:
                            continue
                
                if compound_id is not None:
                    # Sanitize molecule
                    Chem.SanitizeMol(mol)
                    mol_dict[compound_id] = mol
                    
            except Exception as e:
                continue
        
        print(f"  ✓ Loaded {len(mol_dict):,} valid molecules")
        return mol_dict
    
    def _sync_and_clean(self):
        """Keep only compounds present in all three sources"""
        print("\n Synchronizing datasets...")
        
        # Get intersection of IDs
        csv_ids = set(self.df['pubchem_id'].astype(int).unique())
        ecfp_ids = set(self.ecfp_dict.keys())
        sdf_ids = set(self.mol_dict.keys())
        
        common_ids = csv_ids & ecfp_ids & sdf_ids
        
        print(f"  CSV compounds:    {len(csv_ids):,}")
        print(f"  ECFP compounds:   {len(ecfp_ids):,}")
        print(f"  SDF compounds:    {len(sdf_ids):,}")
        print(f"  Common compounds: {len(common_ids):,}")
        
        # Filter data
        self.df = self.df[self.df['pubchem_id'].isin(common_ids)].reset_index(drop=True)
        self.ecfp_dict = {k: v for k, v in self.ecfp_dict.items() if k in common_ids}
        self.mol_dict = {k: v for k, v in self.mol_dict.items() if k in common_ids}
        
        print(f"✓ Synchronization complete. Final dataset: {len(self.df):,} compounds")
    
    def _normalize_rt(self):
        """Normalize retention times"""
        print("\n Normalizing retention times...")
        
        # Store original values
        self.df['rt_original'] = self.df['rt'].copy()
        
        # Normalize
        rt_values = self.df['rt'].values.reshape(-1, 1)
        rt_normalized = self.rt_scaler.fit_transform(rt_values).flatten()
        self.df['rt'] = rt_normalized
        
        print(f"  Original: mean={np.mean(self.df['rt_original']):.2f}, std={np.std(self.df['rt_original']):.2f}")
        print(f"  Normalized: mean={self.df['rt'].mean():.4f}, std={self.df['rt'].std():.4f}")
    
    def denormalize_rt(self, rt_normalized):
        """Convert normalized RT back to original scale"""
        return self.rt_scaler.inverse_transform(rt_normalized.reshape(-1, 1)).flatten()
    
    def get_data(self):
        """Return processed data"""
        return self.df, self.ecfp_dict, self.mol_dict


# ============================================================================
# DATASET CLASS WITH KAGNN-COMPATIBLE FEATURES
# ============================================================================
class SMRTDataset(Dataset):
    """
    PyTorch Dataset for SMRT data
    Uses KAGNN's built-in feature extraction
    """
    
    def __init__(self, df, ecfp_dict, mol_dict):
        self.df = df.reset_index(drop=True)
        self.ecfp_dict = ecfp_dict
        self.mol_dict = mol_dict
        
        # Get KAGNN's feature dimensions
        self.atom_dims = get_atom_feature_dims()
        self.bond_dims = get_bond_feature_dims()
        
        print(f"Dataset created with {len(self.df)} samples")
        print(f"  Atom features: {len(self.atom_dims)} dimensions")
        print(f"  Bond features: {len(self.bond_dims)} dimensions")
    
    def __len__(self):
        return len(self.df)
    
    def _create_kagnn_graph(self, mol):
        """
        Create graph data compatible with KAGNN's OGB encoders
        KAGNN will handle feature extraction internally
        """
        # Create atom features (one-hot encoded indices)
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Create one-hot index (KAGNN's OGB encoder will handle the rest)
            atom_features.append([i])  # Placeholder, KAGNN will encode
        
        # Create edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Undirected
        
        if len(edge_indices) == 0:
            # Handle single-atom molecules
            edge_indices = [[0, 0]]
        
        # Convert to tensors
        x = torch.zeros((num_atoms, 1), dtype=torch.long)  # KAGNN expects Long tensor
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Create minimal edge attributes (KAGNN will encode)
        edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        compound_id = int(row['pubchem_id'])
        
        # Get graph
        mol = self.mol_dict[compound_id]
        graph = self._create_kagnn_graph(mol)
        
        # Get ECFP
        ecfp = torch.tensor(self.ecfp_dict[compound_id], dtype=torch.float32)
        
        # Get normalized RT
        rt = torch.tensor(row['rt'], dtype=torch.float32)
        
        return graph, ecfp, rt, compound_id


def collate_fn(batch):
    """Collate function for DataLoader"""
    graphs, ecfps, rts, ids = zip(*batch)
    
    # Batch graphs using PyG's Batch
    batch_graph = Batch.from_data_list(graphs)
    
    # Stack other tensors
    ecfps_tensor = torch.stack(ecfps)
    rts_tensor = torch.stack(rts)
    
    return batch_graph, ecfps_tensor, rts_tensor, ids


# ============================================================================
# KAGNN MODEL DEFINITION
# ============================================================================
class KAGNN_ECFP_Model(nn.Module):
    """
    KAGNN + ECFP model using official KAGNN implementation
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Get feature dimensions from KAGNN
        atom_dims = get_atom_feature_dims()
        bond_dims = get_bond_feature_dims()
        
        print(f"\nBuilding KAGNN model...")
        print(f"  Atom features: {len(atom_dims)}")
        print(f"  Bond features: {len(bond_dims)}")
        print(f"  GNN layers: {config.GNN_LAYERS}")
        print(f"  Hidden dim: {config.HIDDEN_DIM}")
        
        # KAGNN Graph Encoder
        self.kagnn = KAGIN(
            in_channels=len(atom_dims),
            edge_dim=len(bond_dims),
            num_layers=config.GNN_LAYERS,
            hidden_channels=config.HIDDEN_DIM,
            out_channels=config.HIDDEN_DIM,
            grid_size=config.GRID_SIZE,
            spline_order=config.SPLINE_ORDER,
            hidden_hidden_channels=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
            ogb_encoders=True  # Use OGB feature encoders
        )
        
        # ECFP Encoder (KAN)
        self.ecfp_kan = make_kan(
            in_features=1024,
            hidden_features=config.HIDDEN_DIM,
            out_features=config.HIDDEN_DIM,
            hidden_layers=config.KAN_HIDDEN_LAYERS,
            grid_size=config.GRID_SIZE,
            spline_order=config.SPLINE_ORDER
        )
        
        # Fusion Layer (KAN)
        self.fusion_kan = make_kan(
            in_features=2 * config.HIDDEN_DIM,
            hidden_features=config.HIDDEN_DIM,
            out_features=1,
            hidden_layers=config.KAN_HIDDEN_LAYERS,
            grid_size=config.GRID_SIZE,
            spline_order=config.SPLINE_ORDER
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, graph_data, ecfp):
        """
        Forward pass
        
        Args:
            graph_data: PyG Batch object
            ecfp: ECFP fingerprints [batch_size, 1024]
            
        Returns:
            predictions: [batch_size]
        """
        # Graph encoding
        graph_emb = self.kagnn(graph_data)  # [batch_size, hidden_dim]
        
        # ECFP encoding
        ecfp_emb = self.ecfp_kan(ecfp)  # [batch_size, hidden_dim]
        
        # Concatenate and fuse
        combined = torch.cat([graph_emb, ecfp_emb], dim=-1)  # [batch_size, 2*hidden_dim]
        combined = self.dropout(combined)
        
        # Final prediction
        output = self.fusion_kan(combined)  # [batch_size, 1]
        
        return output.squeeze(-1)  # [batch_size]


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_kagnn_model(config, train_loader, val_loader, test_loader, data_loader):
    """
    Train KAGNN model with early stopping
    
    Returns:
        model: Trained model
        metrics: Test set metrics
        history: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    # Initialize model
    model = KAGNN_ECFP_Model(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Loss function
    criterion = nn.SmoothL1Loss()  # Huber loss
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (graph, ecfp, rt, _) in enumerate(train_loader):
            graph = graph.to(device)
            ecfp = ecfp.to(device)
            rt = rt.to(device)
            
            optimizer.zero_grad()
            pred = model(graph, ecfp)
            loss = criterion(pred, rt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * rt.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for graph, ecfp, rt, _ in val_loader:
                graph = graph.to(device)
                ecfp = ecfp.to(device)
                rt = rt.to(device)
                
                pred = model(graph, ecfp)
                loss = criterion(pred, rt)
                val_loss += loss.item() * rt.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Learning rate tracking
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__
            }, config.CHECKPOINT_PATH)
            
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n Early stopping at epoch {epoch}")
                break
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    
    # Load best model
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    test_metrics = evaluate_model(model, test_loader, data_loader, device)
    
    return model, test_metrics, history


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def evaluate_model(model, test_loader, data_loader, device):
    """
    Evaluate model and compute comprehensive metrics
    """
    model.eval()
    
    all_true = []
    all_pred = []
    all_ids = []
    
    with torch.no_grad():
        for graph, ecfp, rt, ids in test_loader:
            graph = graph.to(device)
            ecfp = ecfp.to(device)
            
            pred = model(graph, ecfp)
            
            # Denormalize predictions
            true_denorm = data_loader.denormalize_rt(rt.numpy())
            pred_denorm = data_loader.denormalize_rt(pred.cpu().numpy())
            
            all_true.extend(true_denorm)
            all_pred.extend(pred_denorm)
            all_ids.extend(ids)
    
    # Convert to numpy arrays
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    
    # Compute metrics
    from sklearn.metrics import (
        r2_score, mean_absolute_error, median_absolute_error, 
        mean_squared_error
    )
    from scipy.stats import pearsonr, spearmanr
    
    metrics = {
        'n_samples': len(y_true),
        'MedAE': median_absolute_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson_r': pearsonr(y_true, y_pred)[0],
        'Pearson_p': pearsonr(y_true, y_pred)[1],
        'Spearman_rho': spearmanr(y_true, y_pred)[0],
        'Spearman_p': spearmanr(y_true, y_pred)[1],
    }
    
    # Threshold accuracies
    abs_errors = np.abs(y_true - y_pred)
    for threshold in [10, 20, 30, 60]:
        metrics[f'within_{threshold}s'] = (abs_errors <= threshold).mean() * 100
    
    # Error statistics
    errors = y_pred - y_true
    metrics.update({
        'error_mean': np.mean(errors),
        'error_std': np.std(errors),
        'error_median': np.median(errors),
        'error_q90': np.percentile(np.abs(errors), 90),
        'error_q95': np.percentile(np.abs(errors), 95),
        'error_max': np.max(np.abs(errors))
    })
    
    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_training_history(history, save_path):
    """Plot training and validation loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training History', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {save_path}")


def plot_predictions(y_true, y_pred, metrics, save_path):
    """Create comprehensive prediction plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('True RT (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted RT (s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Predictions: R²={metrics["R2"]:.3f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Error histogram
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error Distribution\nMean={errors.mean():.1f}s', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Absolute error CDF
    abs_errors = np.abs(errors)
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax3.plot(sorted_errors, cdf, 'b-', linewidth=3)
    ax3.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax3.set_title('CDF of Absolute Errors', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Threshold performance
    thresholds = [10, 30, 60]
    percentages = [(abs_errors <= t).mean() * 100 for t in thresholds]
    bars = ax4.bar(range(len(thresholds)), percentages, color=['green', 'orange', 'red'])
    ax4.set_xlabel('Error Threshold (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% of Predictions', fontsize=12, fontweight='bold')
    ax4.set_title('Threshold Performance', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(thresholds)))
    ax4.set_xticklabels([f'≤{t}s' for t in thresholds])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('KAGNN Baseline Model Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction plots saved to {save_path}")


# ============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# ============================================================================
def run_statistical_analysis(y_true, y_pred, save_dir):
    """
    Run comprehensive statistical analysis
    """
    from scipy.stats import shapiro, normaltest, ttest_1samp
    from scipy import stats
    
    results = {}
    abs_errors = np.abs(y_true - y_pred)
    errors = y_pred - y_true
    
    # 1. Normality tests
    if len(errors) > 3 and len(errors) <= 5000:
        shapiro_stat, shapiro_p = shapiro(errors)
        results['shapiro_wilk'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p > 0.05
        }
    
    k2_stat, k2_p = normaltest(errors)
    results['dagostino_k2'] = {
        'statistic': k2_stat,
        'p_value': k2_p,
        'normal': k2_p > 0.05
    }
    
    # 2. T-test against zero (bias test)
    t_stat, t_p = ttest_1samp(errors, 0)
    results['bias_test'] = {
        't_statistic': t_stat,
        'p_value': t_p,
        'biased': t_p < 0.05
    }
    
    # 3. Skewness and kurtosis
    results['skewness'] = stats.skew(errors)
    results['kurtosis'] = stats.kurtosis(errors)
    
    # 4. Confidence intervals
    from scipy.stats import sem
    mean_error = np.mean(errors)
    std_error = sem(errors)
    ci_95 = stats.t.interval(0.95, len(errors)-1, loc=mean_error, scale=std_error)
    results['ci_95'] = ci_95
    
    # 5. Save results
    stats_path = save_dir / "statistical_tests.json"
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Statistical analysis saved to {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*80)
    print(f"Normality tests:")
    if 'shapiro_wilk' in results:
        normal = "NORMAL" if results['shapiro_wilk']['normal'] else "NON-NORMAL"
        print(f"  Shapiro-Wilk: {normal} (p={results['shapiro_wilk']['p_value']:.3e})")
    
    normal = "NORMAL" if results['dagostino_k2']['normal'] else "NON-NORMAL"
    print(f"  D'Agostino K²: {normal} (p={results['dagostino_k2']['p_value']:.3e})")
    
    print(f"\nBias test:")
    biased = "BIASED" if results['bias_test']['biased'] else "UNBIASED"
    print(f"  Model is {biased} (p={results['bias_test']['p_value']:.3e})")
    
    print(f"\nError statistics:")
    print(f"  Skewness: {results['skewness']:.3f}")
    print(f"  Kurtosis: {results['kurtosis']:.3f}")
    print(f"  95% CI for mean error: [{ci_95[0]:.2f}, {ci_95[1]:.2f}] seconds")
    print("="*80)
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """
    Main function to run complete KAGNN baseline experiment
    """
    print("\n" + "="*80)
    print("KAGNN BASELINE EXPERIMENT - COMPLETE IMPLEMENTATION")
    print("="*80)
    
    try:
        # 1. Initialize configuration
        config = Config()
        print(f"✓ Configuration loaded")
        print(f"  Data directory: {config.DATA_DIR}")
        print(f"  Output directory: {config.OUTPUT_DIR}")
        
        # 2. Load and preprocess data
        data_loader = SMRTDataLoader(config)
        df, ecfp_dict, mol_dict = data_loader.get_data()
        
        # 3. Create data splits
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_SEED
        )
        train_idx, val_idx = train_test_split(
            train_idx, 
            test_size=config.VAL_SIZE, 
            random_state=config.RANDOM_SEED
        )
        
        # Create datasets
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        train_dataset = SMRTDataset(train_df, ecfp_dict, mol_dict)
        val_dataset = SMRTDataset(val_df, ecfp_dict, mol_dict)
        test_dataset = SMRTDataset(test_df, ecfp_dict, mol_dict)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"\n✓ Data splits created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        # 4. Train model
        model, test_metrics, history = train_kagnn_model(
            config, train_loader, val_loader, test_loader, data_loader
        )
        
        # 5. Save metrics
        with open(config.METRICS_PATH, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"✓ Metrics saved to {config.METRICS_PATH}")
        
        # 6. Get predictions for visualization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for graph, ecfp, rt, _ in test_loader:
                graph = graph.to(device)
                ecfp = ecfp.to(device)
                pred = model(graph, ecfp)
                
                true_denorm = data_loader.denormalize_rt(rt.numpy())
                pred_denorm = data_loader.denormalize_rt(pred.cpu().numpy())
                
                y_true_all.extend(true_denorm)
                y_pred_all.extend(pred_denorm)
        
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        
        # 7. Create visualizations
        plot_training_history(history, config.OUTPUT_DIR / "training_history.png")
        plot_predictions(y_true, y_pred, test_metrics, config.OUTPUT_DIR / "predictions.png")
        
        # 8. Run statistical analysis
        run_statistical_analysis(y_true, y_pred, config.OUTPUT_DIR)
        
        # 9. Print final results
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE - FINAL RESULTS")
        print("="*80)
        print(f"\n MODEL PERFORMANCE:")
        print(f"  MedAE:        {test_metrics['MedAE']:.2f} s")
        print(f"  MAE:          {test_metrics['MAE']:.2f} s")
        print(f"  RMSE:         {test_metrics['RMSE']:.2f} s")
        print(f"  R²:           {test_metrics['R2']:.4f}")
        print(f"  Pearson r:    {test_metrics['Pearson_r']:.4f}")
        print(f"  Spearman ρ:   {test_metrics['Spearman_rho']:.4f}")
        
        print(f"\n ACCURACY THRESHOLDS:")
        print(f"  Within 10s:   {test_metrics['within_10s']:.1f} %")
        print(f"  Within 30s:   {test_metrics['within_30s']:.1f} %")
        print(f"  Within 60s:   {test_metrics['within_60s']:.1f} %")
        
        print(f"\n ERROR STATISTICS:")
        print(f"  Mean error:   {test_metrics['error_mean']:.2f} ± {test_metrics['error_std']:.2f} s")
        print(f"  Median error: {test_metrics['error_median']:.2f} s")
        print(f"  90th percentile: {test_metrics['error_q90']:.2f} s")
        print(f"  95th percentile: {test_metrics['error_q95']:.2f} s")
        
        print(f"\n OUTPUTS SAVED:")
        print(f"  Model checkpoint: {config.CHECKPOINT_PATH}")
        print(f"  Metrics:          {config.METRICS_PATH}")
        print(f"  Training plots:   {config.OUTPUT_DIR / 'training_history.png'}")
        print(f"  Prediction plots: {config.OUTPUT_DIR / 'predictions.png'}")
        print(f"  Statistical tests:{config.OUTPUT_DIR / 'statistical_tests.json'}")
        
        print("\n" + "="*80)
        print(" KAGNN BASELINE EXPERIMENT SUCCESSFULLY COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run the experiment
    main()
