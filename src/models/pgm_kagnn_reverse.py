"""
PGM→KAGNN Reverse Residual Model.

This model first trains PGM, then uses KAGNN to predict and correct
the residual errors made by PGM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import numpy as np
from typing import Tuple, Dict, Any
import sys
sys.path.append('..')
from src.models.base_model import BaseRTModel
from rdkit.Chem import Descriptors
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
import joblib


class ComprehensiveDescriptors:
    """Extract comprehensive molecular descriptors for PGM stage."""
    
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
        except:
            return np.zeros(32, dtype=np.float32)
    
    @staticmethod
    def extract_batch(mol_dict, mol_ids):
        """Extract descriptors for a batch of molecules."""
        return np.array([ComprehensiveDescriptors.extract(mol_dict[mid]) for mid in mol_ids], 
                       dtype=np.float32)


class RefinementKAGNN(nn.Module):
    """Lightweight KAGNN for residual refinement."""
    
    def __init__(self, hidden_dim=256, dropout=0.15, heads=4):
        super().__init__()
        
        # Graph attention layers
        self.conv1 = GATConv(9, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        
        # ECFP encoder
        self.ecfp_enc = nn.Sequential(
            nn.Linear(1024, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual predictor
        self.residual_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)
        )
    
    def forward(self, graph, ecfp):
        # Graph encoding
        gx = F.elu(self.conv1(graph.x.float(), graph.edge_index))
        gx = F.elu(self.conv2(gx, graph.edge_index))
        gx = F.elu(self.conv3(gx, graph.edge_index))
        
        # Dual pooling
        g_mean = global_mean_pool(gx, graph.batch)
        g_max = global_max_pool(gx, graph.batch)
        g = (g_mean + g_max) / 2
        
        # ECFP encoding
        e = self.ecfp_enc(ecfp.float())
        
        # Fusion
        emb = self.fusion(torch.cat([g, e], dim=1))
        
        # Residual correction
        residual = self.residual_predictor(emb).squeeze(-1)
        
        return residual


class PGM_KAGNN_Reverse(BaseRTModel):
    """
    PGM→KAGNN Reverse Residual Model.
    
    Two-stage training:
    1. Train PGM (XGBoost + Bayesian Ridge) on descriptors
    2. Train KAGNN to predict residual errors
    
    At inference, final prediction = PGM prediction + KAGNN correction
    """
    
    def __init__(self, config: Any):
        """
        Initialize PGM→KAGNN model.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        super().__init__(config)
        
        # Stage 1: PGM models
        self.pgm_xgb = None
        self.pgm_br = None
        self.pgm_scaler = RobustScaler()
        
        # Stage 2: KAGNN refinement network
        self.kagnn = RefinementKAGNN(
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            heads=4
        )
        
        # Store molecule dictionary
        self.mol_dict = None
        
        # Track PGM predictions for training
        self.pgm_train_preds = None
        self.pgm_val_preds = None
    
    def set_mol_dict(self, mol_dict: Dict):
        """Set molecule dictionary for descriptor extraction."""
        self.mol_dict = mol_dict
    
    def train_stage1_pgm(
        self,
        train_loader,
        val_loader,
        n_estimators: int = 400,
        optimize: bool = False,
        verbose: bool = True
    ):
        """
        Train Stage 1: PGM ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_estimators: Number of estimators
            optimize: Whether to optimize hyperparameters
            verbose: Print progress
        """
        if verbose:
            print("\n" + "="*80)
            print("STAGE 1: Training PGM Ensemble (XGBoost + Bayesian Ridge)")
            print("="*80)
        
        # Extract features from training set
        if verbose:
            print("📊 Extracting features from training set...")
        
        X_train_list = []
        y_train_list = []
        
        for batch in train_loader:
            graph, ecfp, rt_norm, cids = batch
            
            # Extract ECFP + descriptors
            descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
            features = np.concatenate([ecfp.numpy(), descriptors], axis=1)
            
            X_train_list.append(features)
            y_train_list.append(rt_norm.numpy())
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Extract validation features
        X_val_list = []
        y_val_list = []
        
        for batch in val_loader:
            graph, ecfp, rt_norm, cids = batch
            descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
            features = np.concatenate([ecfp.numpy(), descriptors], axis=1)
            
            X_val_list.append(features)
            y_val_list.append(rt_norm.numpy())
        
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        
        # Scale features
        X_train_scaled = self.pgm_scaler.fit_transform(X_train)
        X_val_scaled = self.pgm_scaler.transform(X_val)
        
        if verbose:
            print(f"Training features shape: {X_train_scaled.shape}")
            print(f"Validation features shape: {X_val_scaled.shape}")
        
        # Train XGBoost
        if verbose:
            print("\n🌲 Training XGBoost...")
        
        self.pgm_xgb = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42
        )
        self.pgm_xgb.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # Train Bayesian Ridge
        if verbose:
            print("📈 Training Bayesian Ridge...")
        
        self.pgm_br = BayesianRidge(max_iter=500, compute_score=True)
        self.pgm_br.fit(X_train_scaled, y_train)
        
        # Store PGM predictions
        train_pred_xgb = self.pgm_xgb.predict(X_train_scaled)
        train_pred_br = self.pgm_br.predict(X_train_scaled)
        self.pgm_train_preds = (train_pred_xgb + train_pred_br) / 2
        
        val_pred_xgb = self.pgm_xgb.predict(X_val_scaled)
        val_pred_br = self.pgm_br.predict(X_val_scaled)
        self.pgm_val_preds = (val_pred_xgb + val_pred_br) / 2
        
        # Evaluate PGM
        from sklearn.metrics import r2_score, mean_absolute_error
        
        train_r2 = r2_score(y_train, self.pgm_train_preds)
        val_r2 = r2_score(y_val, self.pgm_val_preds)
        train_mae = mean_absolute_error(y_train, self.pgm_train_preds)
        val_mae = mean_absolute_error(y_val, self.pgm_val_preds)
        
        if verbose:
            print(f"\n📊 PGM Performance:")
            print(f"  Train - R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
            print(f"  Val   - R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
            print("✓ PGM training complete")
    
    def forward(self, graph_data, ecfp):
        """
        Forward pass through KAGNN refinement network.
        
        Args:
            graph_data: PyTorch Geometric graph batch
            ecfp: ECFP fingerprint tensor
            
        Returns:
            Residual corrections
        """
        return self.kagnn(graph_data, ecfp)
    
    def _train_step(self, batch, optimizer, criterion):
        """Override train step to use PGM predictions."""
        graph, ecfp, rt_norm, cids = batch
        graph = graph.to(self.device)
        ecfp = ecfp.to(self.device)
        rt_norm = rt_norm.to(self.device)
        
        # Get PGM predictions
        descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
        features = np.concatenate([ecfp.cpu().numpy(), descriptors], axis=1)
        X_scaled = self.pgm_scaler.transform(features)
        
        pgm_pred = (self.pgm_xgb.predict(X_scaled) + self.pgm_br.predict(X_scaled)) / 2
        pgm_pred = torch.tensor(pgm_pred, dtype=torch.float32, device=self.device)
        
        # Train KAGNN to predict residuals
        optimizer.zero_grad()
        residual = self(graph, ecfp)
        final_pred = pgm_pred + residual
        
        loss = criterion(final_pred, rt_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item() * len(rt_norm), len(rt_norm)
    
    def _val_step(self, batch, criterion):
        """Override validation step to use PGM predictions."""
        graph, ecfp, rt_norm, cids = batch
        graph = graph.to(self.device)
        ecfp = ecfp.to(self.device)
        rt_norm = rt_norm.to(self.device)
        
        # Get PGM predictions
        descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
        features = np.concatenate([ecfp.cpu().numpy(), descriptors], axis=1)
        X_scaled = self.pgm_scaler.transform(features)
        
        pgm_pred = (self.pgm_xgb.predict(X_scaled) + self.pgm_br.predict(X_scaled)) / 2
        pgm_pred = torch.tensor(pgm_pred, dtype=torch.float32, device=self.device)
        
        # Predict residuals
        residual = self(graph, ecfp)
        final_pred = pgm_pred + residual
        
        loss = criterion(final_pred, rt_norm)
        
        return loss.item() * len(rt_norm), len(rt_norm)
    
    def predict(self, data_loader, denormalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with KAGNN correction on top of PGM.
        
        Args:
            data_loader: Data loader for prediction
            denormalize: Whether to denormalize predictions
            
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
                
                # Stage 1: PGM prediction
                descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
                features = np.concatenate([ecfp.cpu().numpy(), descriptors], axis=1)
                X_scaled = self.pgm_scaler.transform(features)
                
                pgm_pred = (self.pgm_xgb.predict(X_scaled) + self.pgm_br.predict(X_scaled)) / 2
                
                # Stage 2: KAGNN correction
                residual = self(graph, ecfp)
                final_pred = pgm_pred + residual.cpu().numpy()
                
                y_true_all.append(rt_norm.numpy())
                y_pred_all.append(final_pred)
        
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        if denormalize and self.rt_mean is not None and self.rt_std is not None:
            y_true = y_true * self.rt_std + self.rt_mean
            y_pred = y_pred * self.rt_std + self.rt_mean
        
        return y_true, y_pred
    
    def save(self, path):
        """Save model including PGM components."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'rt_mean': self.rt_mean,
            'rt_std': self.rt_std,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'pgm_scaler': self.pgm_scaler,
        }
        
        torch.save(checkpoint, path)
        
        # Save PGM models separately
        if self.pgm_xgb is not None:
            joblib.dump(self.pgm_xgb, path.parent / f"{path.stem}_xgb.pkl")
        if self.pgm_br is not None:
            joblib.dump(self.pgm_br, path.parent / f"{path.stem}_br.pkl")
        
        print(f"Model saved to {path}")
    
    def load(self, path, map_location=None):
        """Load model including PGM components."""
        from pathlib import Path
        if map_location is None:
            map_location = str(self.device)
        
        path = Path(path)
        checkpoint = torch.load(path, map_location=map_location)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.rt_mean = checkpoint.get('rt_mean')
        self.rt_std = checkpoint.get('rt_std')
        self.pgm_scaler = checkpoint.get('pgm_scaler', RobustScaler())
        
        # Load PGM models
        xgb_path = path.parent / f"{path.stem}_xgb.pkl"
        br_path = path.parent / f"{path.stem}_br.pkl"
        
        if xgb_path.exists():
            self.pgm_xgb = joblib.load(xgb_path)
        if br_path.exists():
            self.pgm_br = joblib.load(br_path)
        
        print(f"Model loaded from {path}")
