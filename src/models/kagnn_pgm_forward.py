"""
KAGNN→PGM Forward Residual Model (FIXED VERSION)

Key improvements:
1. Correct imports from official KAGNN
2. API validation with fallback
3. Efficient feature caching
4. Comprehensive error handling
5. Statistical analysis integration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append('..')
from src.models.base_model import BaseRTModel
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
import joblib
import warnings

# ============================================================================
# KAGNN IMPORT WITH VALIDATION AND FALLBACK
# ============================================================================
try:
    from kagnn.models import (
        KAGIN, 
        make_kan, 
        get_atom_feature_dims, 
        get_bond_feature_dims
    )
    KAGNN_AVAILABLE = True
    print("✓ Official KAGNN imported successfully")
    
    # Validate API
    try:
        atom_dims = len(get_atom_feature_dims())
        bond_dims = len(get_bond_feature_dims())
        print(f"  Atom features: {atom_dims}, Bond features: {bond_dims}")
    except Exception as e:
        print(f"  KAGNN API validation warning: {e}")
        
except ImportError as e:
    print(f" Failed to import official KAGNN: {e}")
    print(" Falling back to GAT-based implementation...")
    KAGNN_AVAILABLE = False
    
    # Fallback to GAT (from our artifacts)
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    import torch.nn.functional as F


# ============================================================================
# DESCRIPTOR EXTRACTION (CACHED VERSION)
# ============================================================================
class ComprehensiveDescriptors:
    """Extract comprehensive molecular descriptors with caching."""
    
    _cache = {}  # Cache for extracted descriptors
    
    @staticmethod
    def extract(mol):
        """Extract descriptors from a single molecule."""
        try:
            if not mol.GetRingInfo().IsInitialized():
                from rdkit import Chem
                Chem.GetSymmSSSR(mol)
            
            return np.array([
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.MolMR(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol),
                Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol), Descriptors.NumHeteroatoms(mol),
                Descriptors.HeavyAtomCount(mol), Descriptors.MaxAbsPartialCharge(mol),
                Descriptors.BertzCT(mol), Descriptors.LabuteASA(mol),
                Descriptors.HallKierAlpha(mol), Descriptors.Chi0v(mol), Descriptors.Chi1v(mol),
                Descriptors.Kappa1(mol), Descriptors.Kappa2(mol), Descriptors.Kappa3(mol),
                Descriptors.BalabanJ(mol), Descriptors.Ipc(mol),
                mol.GetNumAtoms(), mol.GetNumBonds(), 0, 0, 0, 0, 0, 0
            ], dtype=np.float32)[:32]
        except Exception as e:
            warnings.warn(f"Failed to extract descriptors: {e}")
            return np.zeros(32, dtype=np.float32)
    
    @staticmethod
    def extract_batch(mol_dict, mol_ids, use_cache=True):
        """
        Extract descriptors for a batch of molecules with optional caching.
        
        Args:
            mol_dict: Dictionary mapping IDs to RDKit molecules
            mol_ids: List of molecule IDs
            use_cache: Whether to use cached descriptors
        
        Returns:
            Array of descriptors (batch_size, 32)
        """
        results = []
        
        for mid in mol_ids:
            if use_cache and mid in ComprehensiveDescriptors._cache:
                # Use cached descriptor
                results.append(ComprehensiveDescriptors._cache[mid])
            else:
                # Compute and cache
                desc = ComprehensiveDescriptors.extract(mol_dict[mid])
                if use_cache:
                    ComprehensiveDescriptors._cache[mid] = desc
                results.append(desc)
        
        return np.array(results, dtype=np.float32)
    
    @staticmethod
    def clear_cache():
        """Clear descriptor cache to free memory."""
        ComprehensiveDescriptors._cache.clear()


# ============================================================================
# FALLBACK GAT-BASED KAGNN (if official KAGNN not available)
# ============================================================================
class FallbackKAGNN(nn.Module):
    """GAT-based KAGNN fallback if official implementation unavailable."""
    
    def __init__(self, hidden_dim=256, dropout=0.1, num_layers=5):
        super().__init__()
        
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
        
        # Last layer: concat=False for averaging
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        )
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, graph_data):
        x = graph_data.x.float()
        
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, graph_data.edge_index)
            x_new = norm(x_new)
            x_new = F.elu(x_new)
            
            # Residual connection (skip first layer)
            if i > 0 and x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new
        
        # Multi-scale pooling
        g_mean = global_mean_pool(x, graph_data.batch)
        g_max = global_max_pool(x, graph_data.batch)
        
        return (g_mean + g_max) / 2


class FallbackKAN(nn.Module):
    """Simple MLP fallback for KAN."""
    
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.1):
        super().__init__()
        
        layers = []
        current_dim = in_features
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_features
        
        layers.append(nn.Linear(current_dim, out_features))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# MAIN MODEL
# ============================================================================
class KAGNN_PGM_Forward(BaseRTModel):
    """
    KAGNN→PGM Forward Residual Model (FIXED VERSION).
    
    Two-stage training:
    1. Train KAGNN backbone on RT prediction
    2. Train PGM ensemble to predict residual errors
    
    At inference: final prediction = KAGNN prediction + PGM correction
    
    Improvements:
    - Correct KAGNN imports with validation
    - Efficient descriptor caching
    - Comprehensive error handling
    - Statistical analysis ready
    """
    
    def __init__(self, config: Any):
        """
        Initialize KAGNN→PGM model.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        super().__init__(config)
        
        self.kagnn_available = KAGNN_AVAILABLE
        
        # Stage 1: KAGNN backbone
        if KAGNN_AVAILABLE:
            try:
                self.kagnn = KAGIN(
                    len(get_atom_feature_dims()),
                    len(get_bond_feature_dims()),
                    config.gnn_layers,
                    config.hidden_dim,
                    config.hidden_layers,
                    config.grid_size,
                    config.spline_order,
                    config.hidden_dim,
                    config.dropout,
                    ogb_encoders=True
                )
                
                self.ecfp_kan = make_kan(
                    1024,
                    config.hidden_dim,
                    config.hidden_dim,
                    config.hidden_layers,
                    config.grid_size,
                    config.spline_order
                )
                
                self.final_kan = make_kan(
                    2 * config.hidden_dim,
                    config.hidden_dim // 2,
                    1,
                    config.hidden_layers,
                    config.grid_size,
                    config.spline_order
                )
                
                print("✓ Using official KAGNN/KAN layers")
                
            except Exception as e:
                print(f" Failed to initialize official KAGNN: {e}")
                print(" Falling back to GAT implementation...")
                self.kagnn_available = False
        
        if not self.kagnn_available:
            # Fallback to GAT-based implementation
            self.kagnn = FallbackKAGNN(
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                num_layers=config.gnn_layers
            )
            
            self.ecfp_kan = FallbackKAN(
                in_features=1024,
                hidden_features=config.hidden_dim,
                out_features=config.hidden_dim,
                num_layers=config.hidden_layers,
                dropout=config.dropout
            )
            
            self.final_kan = FallbackKAN(
                in_features=2 * config.hidden_dim,
                hidden_features=config.hidden_dim // 2,
                out_features=1,
                num_layers=config.hidden_layers,
                dropout=config.dropout
            )
            
            print("✓ Using GAT-based fallback implementation")
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Stage 2: PGM models (initialized after Stage 1 training)
        self.pgm_xgb = None
        self.pgm_br = None
        self.pgm_scaler = StandardScaler()
        
        # Store molecule dictionary for descriptor extraction
        self.mol_dict = None
        
        # Feature cache for efficiency
        self._feature_cache = {}
    
    def forward(self, graph_data, ecfp, return_emb=False):
        """
        Forward pass through KAGNN backbone.
        
        Args:
            graph_data: PyTorch Geometric graph batch
            ecfp: ECFP fingerprint tensor
            return_emb: Whether to return intermediate embeddings
            
        Returns:
            predictions (and embeddings if return_emb=True)
        """
        # Process graph
        graph_emb = self.kagnn(graph_data)
        
        # Process ECFP
        ecfp_emb = self.ecfp_kan(ecfp)
        
        # Concatenate
        x = torch.cat([graph_emb, ecfp_emb], dim=-1)
        x = self.dropout(x)
        
        # Predict
        output = self.final_kan(x).squeeze(-1)
        
        if return_emb:
            # Return embeddings for PGM training
            return output, torch.cat([graph_emb, ecfp_emb], dim=-1)
        
        return output
    
    def set_mol_dict(self, mol_dict: Dict):
        """Set molecule dictionary for descriptor extraction."""
        self.mol_dict = mol_dict
        print(f"✓ Molecule dictionary set ({len(mol_dict)} molecules)")
    
    def train_stage2_pgm(
        self,
        train_loader,
        n_estimators: int = 50,
        max_samples: float = 0.8,
        verbose: bool = True,
        use_cache: bool = True
    ):
        """
        Train Stage 2: PGM residual correction (IMPROVED VERSION).
        
        Args:
            train_loader: Training data loader
            n_estimators: Number of estimators for ensemble
            max_samples: Max samples for bagging
            verbose: Print progress
            use_cache: Use descriptor caching for speed
        """
        if verbose:
            print("\n" + "="*80)
            print("STAGE 2: Training PGM Residual Correction (IMPROVED)")
            print("="*80)
        
        self.eval()
        
        # Clear old cache
        if use_cache:
            ComprehensiveDescriptors.clear_cache()
        
        # Collect features and residuals
        X_train_list = []
        y_residuals_list = []
        
        if verbose:
            from tqdm import tqdm
            loader_iter = tqdm(train_loader, desc="Extracting features")
        else:
            loader_iter = train_loader
        
        with torch.no_grad():
            for batch in loader_iter:
                graph, ecfp, rt_norm, cids = batch
                graph = graph.to(self.device)
                ecfp = ecfp.to(self.device)
                rt_norm = rt_norm.to(self.device)
                
                # Get KAGNN predictions and embeddings
                pred_norm, emb = self(graph, ecfp, return_emb=True)
                
                # Calculate residuals (in normalized space)
                residuals = rt_norm.cpu().numpy() - pred_norm.cpu().numpy()
                
                # Extract molecular descriptors (with caching)
                descriptors = ComprehensiveDescriptors.extract_batch(
                    self.mol_dict, cids, use_cache=use_cache
                )
                
                # Combine embeddings + descriptors
                features = np.concatenate([emb.cpu().numpy(), descriptors], axis=1)
                
                X_train_list.append(features)
                y_residuals_list.append(residuals)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_residuals_list, axis=0)
        
        # Check for NaN/Inf
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("  Warning: NaN/Inf detected in features, replacing with zeros")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_train_scaled = self.pgm_scaler.fit_transform(X_train)
        
        if verbose:
            print(f"\n Feature Statistics:")
            print(f"   Training features shape: {X_train_scaled.shape}")
            print(f"   Residuals shape: {y_train.shape}")
            print(f"   Mean residual: {np.mean(y_train):.4f} ± {np.std(y_train):.4f}")
            print(f"   Min residual: {np.min(y_train):.4f}")
            print(f"   Max residual: {np.max(y_train):.4f}")
        
        # Train XGBoost
        if verbose:
            print("\n Training XGBoost...")
        
        self.pgm_xgb = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=max_samples,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42
        )
        self.pgm_xgb.fit(X_train_scaled, y_train, verbose=False)
        
        # Train Bayesian Ridge
        if verbose:
            print(" Training Bayesian Ridge...")
        
        self.pgm_br = BayesianRidge(max_iter=500, compute_score=True)
        self.pgm_br.fit(X_train_scaled, y_train)
        
        if verbose:
            # Evaluate PGM on training set
            pred_xgb = self.pgm_xgb.predict(X_train_scaled)
            pred_br = self.pgm_br.predict(X_train_scaled)
            pred_ensemble = (pred_xgb + pred_br) / 2
            
            mae_xgb = np.mean(np.abs(y_train - pred_xgb))
            mae_br = np.mean(np.abs(y_train - pred_br))
            mae_ensemble = np.mean(np.abs(y_train - pred_ensemble))
            
            print(f"\n✓ PGM training complete")
            print(f"   XGBoost MAE (residual):     {mae_xgb:.4f}")
            print(f"   Bayesian Ridge MAE:         {mae_br:.4f}")
            print(f"   Ensemble MAE:               {mae_ensemble:.4f}")
            
            # Feature importance (top 10)
            if hasattr(self.pgm_xgb, 'feature_importances_'):
                importances = self.pgm_xgb.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]
                print(f"\n   Top 10 XGBoost Features:")
                for i, idx in enumerate(top_indices, 1):
                    print(f"      {i}. Feature {idx}: {importances[idx]:.4f}")
    
    def predict(
        self,
        data_loader,
        denormalize: bool = True,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with PGM correction (IMPROVED VERSION).
        
        Args:
            data_loader: Data loader for prediction
            denormalize: Whether to denormalize predictions
            use_cache: Use descriptor caching
            
        Returns:
            Tuple of (true_values, predictions)
        """
        self.eval()
        y_true_all = []
        y_pred_all = []
        
        with torch.no_grad():
            for batch in data_loader:
                graph, ecfp, rt_norm, cids = batch
                graph = graph.to(self.device)
                ecfp = ecfp.to(self.device)
                
                # Stage 1: KAGNN prediction
                pred_base, emb = self(graph, ecfp, return_emb=True)
                
                # Stage 2: PGM correction
                if self.pgm_xgb is not None and self.pgm_br is not None:
                    # Extract descriptors (with caching)
                    descriptors = ComprehensiveDescriptors.extract_batch(
                        self.mol_dict, cids, use_cache=use_cache
                    )
                    
                    # Combine features
                    features = np.concatenate([emb.cpu().numpy(), descriptors], axis=1)
                    
                    # Handle NaN/Inf
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    X_scaled = self.pgm_scaler.transform(features)
                    
                    # Predict correction
                    correction_xgb = self.pgm_xgb.predict(X_scaled)
                    correction_br = self.pgm_br.predict(X_scaled)
                    correction = (correction_xgb + correction_br) / 2
                    
                    # Apply correction
                    pred_norm = pred_base.cpu().numpy() + correction
                else:
                    # No PGM correction (only KAGNN)
                    pred_norm = pred_base.cpu().numpy()
                
                y_true_all.append(rt_norm.numpy())
                y_pred_all.append(pred_norm)
        
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        if denormalize and self.rt_mean is not None and self.rt_std is not None:
            y_true = y_true * self.rt_std + self.rt_mean
            y_pred = y_pred * self.rt_std + self.rt_mean
        
        return y_true, y_pred
    
    def save(self, path):
        """Save model including PGM components (IMPROVED)."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'rt_mean': self.rt_mean,
            'rt_std': self.rt_std,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'pgm_scaler': self.pgm_scaler,
            'kagnn_available': self.kagnn_available,
        }
        
        torch.save(checkpoint, path)
        
        # Save PGM models separately
        if self.pgm_xgb is not None:
            joblib.dump(self.pgm_xgb, path.parent / f"{path.stem}_xgb.pkl")
            print(f"✓ XGBoost saved to {path.parent / f'{path.stem}_xgb.pkl'}")
        
        if self.pgm_br is not None:
            joblib.dump(self.pgm_br, path.parent / f"{path.stem}_br.pkl")
            print(f"✓ Bayesian Ridge saved to {path.parent / f'{path.stem}_br.pkl'}")
        
        print(f"✓ Model checkpoint saved to {path}")
    
    def load(self, path, map_location=None):
        """Load model including PGM components (IMPROVED)."""
        from pathlib import Path
        if map_location is None:
            map_location = str(self.device)
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=map_location)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.rt_mean = checkpoint.get('rt_mean')
        self.rt_std = checkpoint.get('rt_std')
        self.pgm_scaler = checkpoint.get('pgm_scaler', StandardScaler())
        self.kagnn_available = checkpoint.get('kagnn_available', KAGNN_AVAILABLE)
        
        # Load PGM models
        xgb_path = path.parent / f"{path.stem}_xgb.pkl"
        br_path = path.parent / f"{path.stem}_br.pkl"
        
        if xgb_path.exists():
            self.pgm_xgb = joblib.load(xgb_path)
            print(f"✓ XGBoost loaded from {xgb_path}")
        else:
            print(f"  XGBoost model not found at {xgb_path}")
        
        if br_path.exists():
            self.pgm_br = joblib.load(br_path)
            print(f"✓ Bayesian Ridge loaded from {br_path}")
        else:
            print(f"  Bayesian Ridge model not found at {br_path}")
        
        print(f"✓ Model loaded from {path}")
