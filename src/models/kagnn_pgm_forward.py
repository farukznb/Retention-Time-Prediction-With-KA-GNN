"""
KAGNN→PGM Forward Residual Model.

This model first trains KAGNN, then uses PGM (XGBoost + Bayesian Ridge)
to predict and correct the residual errors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import sys
sys.path.append('..')
from src.models.base_model import BaseRTModel
from models import KAGIN, make_kan, get_atom_feature_dims, get_bond_feature_dims
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
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


class KAGNN_PGM_Forward(BaseRTModel):
    """
    KAGNN→PGM Forward Residual Model.
    
    Two-stage training:
    1. Train KAGNN backbone on RT prediction
    2. Train PGM ensemble to predict residual errors
    
    At inference, final prediction = KAGNN prediction + PGM correction
    """
    
    def __init__(self, config: Any):
        """
        Initialize KAGNN→PGM model.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        super().__init__(config)
        
        # Stage 1: KAGNN backbone
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
        
        # ECFP processing
        self.ecfp_kan = make_kan(
            1024,
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_layers,
            config.grid_size,
            config.spline_order
        )
        
        # Final prediction layer
        self.final_kan = make_kan(
            2 * config.hidden_dim,
            config.hidden_dim // 2,
            1,
            config.hidden_layers,
            config.grid_size,
            config.spline_order
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Stage 2: PGM models (initialized after Stage 1 training)
        self.pgm_xgb = None
        self.pgm_br = None
        self.pgm_scaler = StandardScaler()
        
        # Store molecule dictionary for descriptor extraction
        self.mol_dict = None
    
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
    
    def train_stage2_pgm(
        self,
        train_loader,
        n_estimators: int = 50,
        max_samples: float = 0.8,
        verbose: bool = True
    ):
        """
        Train Stage 2: PGM residual correction.
        
        Args:
            train_loader: Training data loader
            n_estimators: Number of estimators for ensemble
            max_samples: Max samples for bagging
            verbose: Print progress
        """
        if verbose:
            print("\n" + "="*80)
            print("STAGE 2: Training PGM Residual Correction")
            print("="*80)
        
        self.eval()
        
        # Collect features and residuals
        X_train_list = []
        y_residuals_list = []
        
        with torch.no_grad():
            for batch in train_loader:
                graph, ecfp, rt_norm, cids = batch
                graph = graph.to(self.device)
                ecfp = ecfp.to(self.device)
                rt_norm = rt_norm.to(self.device)
                
                # Get KAGNN predictions and embeddings
                pred_norm, emb = self(graph, ecfp, return_emb=True)
                
                # Calculate residuals (in normalized space)
                residuals = rt_norm.cpu().numpy() - pred_norm.cpu().numpy()
                
                # Extract molecular descriptors
                descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
                
                # Combine embeddings + descriptors
                features = np.concatenate([emb.cpu().numpy(), descriptors], axis=1)
                
                X_train_list.append(features)
                y_residuals_list.append(residuals)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_residuals_list, axis=0)
        
        # Scale features
        X_train_scaled = self.pgm_scaler.fit_transform(X_train)
        
        if verbose:
            print(f"Training features shape: {X_train_scaled.shape}")
            print(f"Residuals shape: {y_train.shape}")
            print(f"Mean residual: {np.mean(y_train):.4f}")
            print(f"Std residual: {np.std(y_train):.4f}")
        
        # Train XGBoost
        if verbose:
            print("\n🌲 Training XGBoost...")
        
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
            print("📈 Training Bayesian Ridge...")
        
        self.pgm_br = BayesianRidge(max_iter=500, compute_score=True)
        self.pgm_br.fit(X_train_scaled, y_train)
        
        if verbose:
            # Evaluate PGM on training set
            pred_xgb = self.pgm_xgb.predict(X_train_scaled)
            pred_br = self.pgm_br.predict(X_train_scaled)
            pred_ensemble = (pred_xgb + pred_br) / 2
            
            mae = np.mean(np.abs(y_train - pred_ensemble))
            print(f"\n✓ PGM training complete")
            print(f"  Training MAE (residual): {mae:.4f}")
    
    def predict(
        self,
        data_loader,
        denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with PGM correction.
        
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
                
                # Stage 1: KAGNN prediction
                pred_base, emb = self(graph, ecfp, return_emb=True)
                
                # Stage 2: PGM correction
                if self.pgm_xgb is not None and self.pgm_br is not None:
                    # Extract descriptors
                    descriptors = ComprehensiveDescriptors.extract_batch(self.mol_dict, cids)
                    
                    # Combine features
                    features = np.concatenate([emb.cpu().numpy(), descriptors], axis=1)
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
        self.pgm_scaler = checkpoint.get('pgm_scaler', StandardScaler())
        
        # Load PGM models
        xgb_path = path.parent / f"{path.stem}_xgb.pkl"
        br_path = path.parent / f"{path.stem}_br.pkl"
        
        if xgb_path.exists():
            self.pgm_xgb = joblib.load(xgb_path)
        if br_path.exists():
            self.pgm_br = joblib.load(br_path)
        
        print(f"Model loaded from {path}")
