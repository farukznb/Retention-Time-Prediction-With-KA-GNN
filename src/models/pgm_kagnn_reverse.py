"""
FIXED PGM→KAGNN MODEL (Two-Stage)
Stage 1: Train PGM ensemble (XGBoost + Bayesian Ridge)
Stage 2: Train KAGNN to predict residuals
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
import optuna
import warnings
import sys
import os

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ============================================================================
# KAGNN IMPORT WITH VALIDATION
# ============================================================================
try:
    sys.path.append(os.path.expanduser('~/KAGNN'))
    from graph_regression.kagnn.models import (
        KAGIN, 
        make_kan, 
        get_atom_feature_dims, 
        get_bond_feature_dims
    )
    KAGNN_AVAILABLE = True
    print("✓ Official KAGNN imported successfully")
except ImportError as e:
    print(f" Failed to import official KAGNN: {e}")
    print(" Falling back to GAT-based implementation...")
    KAGNN_AVAILABLE = False
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

# ============================================================================
# DESCRIPTOR EXTRACTION (CACHED)
# ============================================================================
class CachedDescriptors:
    """Extract molecular descriptors with caching for efficiency"""
    
    _cache = {}
    
    @staticmethod
    def extract(mol):
        """Extract 32 molecular descriptors"""
        try:
            if not mol.GetRingInfo().IsInitialized():
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
    def extract_batch(mol_dict, mol_ids, use_cache=True):
        """Extract descriptors for batch with caching"""
        results = []
        for mid in mol_ids:
            if use_cache and mid in CachedDescriptors._cache:
                results.append(CachedDescriptors._cache[mid])
            else:
                desc = CachedDescriptors.extract(mol_dict[mid])
                if use_cache:
                    CachedDescriptors._cache[mid] = desc
                results.append(desc)
        return np.array(results, dtype=np.float32)
    
    @staticmethod
    def clear_cache():
        CachedDescriptors._cache.clear()


# ============================================================================
# DATA LOADER (SAME AS BEFORE)
# ============================================================================
class SMRTDataLoader:
    # ... (Same as previous implementation) ...
    pass


# ============================================================================
# DATASET (SAME AS BEFORE)
# ============================================================================
class SMRTDataset(Dataset):
    # ... (Same as previous implementation) ...
    pass

def collate_fn(batch):
    gs, es, rts, mids = zip(*batch)
    return Batch.from_data_list(gs), torch.stack(es), torch.stack(rts), mids


# ============================================================================
# RESIDUAL KAGNN MODEL
# ============================================================================
class ResidualKAGNN(nn.Module):
    """
    KAGNN model to predict residual corrections
    """
    def __init__(self, hidden_dim=256, dropout=0.15, num_layers=5):
        super().__init__()
        
        if KAGNN_AVAILABLE:
            try:
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
                
                print("✓ Using official KAGIN for residual prediction")
            except Exception as e:
                print(f" KAGIN initialization failed: {e}")
                KAGNN_AVAILABLE = False
        
        if not KAGNN_AVAILABLE:
            # Fallback GAT
            from torch_geometric.nn import GATConv
            self.gat_layers = nn.ModuleList()
            heads = 4
            
            self.gat_layers.append(
                GATConv(9, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
            )
            
            for _ in range(num_layers - 2):
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
            )
            
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # ECFP encoder
        self.ecfp_encoder = nn.Sequential(
            nn.Linear(1024, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer with attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Residual predictor (predicts correction)
        self.residual_predictor = nn.Sequential(
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
                
                if i > 0 and x.shape[-1] == x_new.shape[-1]:
                    x = x + x_new
                else:
                    x = x_new
            
            from torch_geometric.nn import global_mean_pool, global_max_pool
            g_mean = global_mean_pool(x, graph.batch)
            g_max = global_max_pool(x, graph.batch)
            graph_emb = (g_mean + g_max) / 2
        
        # Process ECFP
        ecfp_emb = self.ecfp_encoder(ecfp.float())
        
        # Cross-attention fusion
        fusion_input = torch.stack([graph_emb, ecfp_emb], dim=1)
        attn_out, _ = self.cross_attention(fusion_input, fusion_input, fusion_input)
        attn_out = attn_out.mean(dim=1)
        
        # Combine
        combined = torch.cat([graph_emb, ecfp_emb], dim=-1)
        fused = combined + attn_out
        
        # Predict residual
        fused = self.dropout(fused)
        residual = self.residual_predictor(fused).squeeze(-1)
        
        return residual


# ============================================================================
# PGM→KAGNN TRAINER
# ============================================================================
class PGMKAGNNTrainer:
    def __init__(self, train_loader, val_loader, test_loader, mol_dict, rt_scaler, device='cuda'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mol_dict = mol_dict
        self.rt_scaler = rt_scaler
        self.device = device
        
        # PGM models
        self.pgm_xgb = None
        self.pgm_br = None
        self.pgm_scaler = RobustScaler()
        
        # KAGNN model
        self.kagnn = None
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.pgm_train_preds = None
        self.pgm_val_preds = None
    
    def extract_pgm_features(self, loader):
        """Extract ECFP + descriptors for PGM training"""
        all_ecfp, all_desc, all_rt, all_mids = [], [], [], []
        
        for g, e, rt, mids in loader:
            all_ecfp.append(e.numpy())
            all_desc.append(CachedDescriptors.extract_batch(self.mol_dict, mids, use_cache=True))
            all_rt.append(rt.numpy())
            all_mids.extend(mids)
        
        X = np.concatenate([np.vstack(all_ecfp), np.vstack(all_desc)], axis=1)
        y = np.concatenate(all_rt)
        
        return X, y, all_mids
    
    def optimize_pgm(self, X_train, y_train, n_trials=30):
        """Optimize XGBoost hyperparameters"""
        print(f"\n🔍 Optimizing PGM hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tree_method': 'hist'
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"✓ Best 3-fold CV R²: {study.best_value:.4f}")
        return study.best_params
    
    def train_stage1_pgm(self, optimize=True, n_trials=30):
        """Train PGM ensemble"""
        print("\n" + "="*80)
        print("STAGE 1: Training PGM Ensemble")
        print("="*80)
        
        # Extract features
        X_train, y_train, _ = self.extract_pgm_features(self.train_loader)
        X_val, y_val, _ = self.extract_pgm_features(self.val_loader)
        
        # Scale
        X_train_scaled = self.pgm_scaler.fit_transform(X_train)
        X_val_scaled = self.pgm_scaler.transform(X_val)
        
        # Train XGBoost
        if optimize:
            best_params = self.optimize_pgm(X_train_scaled, y_train, n_trials)
            self.pgm_xgb = xgb.XGBRegressor(**best_params, random_state=42)
        else:
            self.pgm_xgb = xgb.XGBRegressor(
                n_estimators=400, max_depth=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method='hist'
            )
        
        self.pgm_xgb.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        
        # Train Bayesian Ridge
        self.pgm_br = BayesianRidge(max_iter=500, compute_score=True)
        self.pgm_br.fit(X_train_scaled, y_train)
        
        # Ensemble predictions
        train_pred_xgb = self.pgm_xgb.predict(X_train_scaled)
        train_pred_br = self.pgm_br.predict(X_train_scaled)
        self.pgm_train_preds = (train_pred_xgb + train_pred_br) / 2
        
        val_pred_xgb = self.pgm_xgb.predict(X_val_scaled)
        val_pred_br = self.pgm_br.predict(X_val_scaled)
        self.pgm_val_preds = (val_pred_xgb + val_pred_br) / 2
        
        # Evaluate
        train_r2 = r2_score(y_train, self.pgm_train_preds)
        val_r2 = r2_score(y_val, self.pgm_val_preds)
        
        print(f"\n PGM Performance:")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Val R²: {val_r2:.4f}")
        
        # Save
        joblib.dump(self.pgm_xgb, 'pgm_xgboost.pkl')
        joblib.dump(self.pgm_br, 'pgm_bayesian_ridge.pkl')
        joblib.dump(self.pgm_scaler, 'pgm_scaler.pkl')
        print("✓ PGM models saved")
    
    def train_stage2_kagnn(self, hidden_dim=256, dropout=0.2, epochs=100, patience=15):
        """Train KAGNN to predict residuals"""
        print("\n" + "="*80)
        print("STAGE 2: Training KAGNN Residual Correction")
        print("="*80)
        
        self.kagnn = ResidualKAGNN(hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.kagnn.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
        
        print(f" KAGNN parameters: {sum(p.numel() for p in self.kagnn.parameters()):,}")
        
        # Create PGM prediction dictionaries
        pgm_train_dict = self._get_pgm_dict(self.train_loader)
        pgm_val_dict = self._get_pgm_dict(self.val_loader)
        
        best_val_loss = float('inf')
        counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            self.kagnn.train()
            train_loss = 0.0
            
            for g, e, rt, mids in self.train_loader:
                g, e, rt = g.to(self.device), e.to(self.device), rt.to(self.device)
                pgm_pred = torch.tensor([pgm_train_dict[mid] for mid in mids], 
                                       dtype=torch.float32, device=self.device)
                
                optimizer.zero_grad()
                residual = self.kagnn(g, e)
                final_pred = pgm_pred + residual
                
                loss = F.l1_loss(final_pred, rt)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.kagnn.parameters(), max_norm=0.5)
                    optimizer.step()
                    train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            self.kagnn.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for g, e, rt, mids in self.val_loader:
                    g, e, rt = g.to(self.device), e.to(self.device), rt.to(self.device)
                    pgm_pred = torch.tensor([pgm_val_dict[mid] for mid in mids],
                                           dtype=torch.float32, device=self.device)
                    
                    residual = self.kagnn(g, e)
                    final_pred = pgm_pred + residual
                    
                    loss = F.l1_loss(final_pred, rt)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            if epoch % 5 == 0:
                print(f"📊 Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.kagnn.state_dict().items()}
                torch.save(best_state, 'best_kagnn_residual.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f" Early stopping at epoch {epoch}")
                    break
        
        # Load best
        self.kagnn.load_state_dict(torch.load('best_kagnn_residual.pth'))
        print(f"✓ Best KAGNN loaded (Val Loss: {best_val_loss:.4f})")
    
    def _get_pgm_dict(self, loader):
        """Get PGM predictions as dictionary"""
        X, _, mids = self.extract_pgm_features(loader)
        X_scaled = self.pgm_scaler.transform(X)
        
        pred_xgb = self.pgm_xgb.predict(X_scaled)
        pred_br = self.pgm_br.predict(X_scaled)
        ensemble_pred = (pred_xgb + pred_br) / 2
        
        return {mid: pred for mid, pred in zip(mids, ensemble_pred)}
    
    def evaluate(self):
        """Evaluate complete model"""
        self.kagnn.eval()
        pgm_test_dict = self._get_pgm_dict(self.test_loader)
        
        all_trues, all_preds_pgm, all_preds_final = [], [], []
        
        with torch.no_grad():
            for g, e, rt, mids in self.test_loader:
                g, e, rt = g.to(self.device), e.to(self.device), rt.to(self.device)
                pgm_pred = torch.tensor([pgm_test_dict[mid] for mid in mids],
                                       dtype=torch.float32, device=self.device)
                
                residual = self.kagnn(g, e)
                final_pred = pgm_pred + residual
                
                # Denormalize
                pgm_pred_orig = self.rt_scaler.inverse_transform(pgm_pred.cpu().numpy().reshape(-1, 1)).flatten()
                final_pred_orig = self.rt_scaler.inverse_transform(final_pred.cpu().numpy().reshape(-1, 1)).flatten()
                true_rt = self.rt_scaler.inverse_transform(rt.cpu().numpy().reshape(-1, 1)).flatten()
                
                all_trues.extend(true_rt)
                all_preds_pgm.extend(pgm_pred_orig)
                all_preds_final.extend(final_pred_orig)
        
        y_true = np.array(all_trues)
        y_pred_pgm = np.array(all_preds_pgm)
        y_pred_final = np.array(all_preds_final)
        
        # Metrics
        def compute_metrics(y_true, y_pred, name):
            return {
                'name': name,
                'MedAE': median_absolute_error(y_true, y_pred),
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'Pearson': pearsonr(y_true, y_pred)[0],
                'Spearman': spearmanr(y_true, y_pred)[0],
            }
        
        metrics_pgm = compute_metrics(y_true, y_pred_pgm, "PGM Only")
        metrics_final = compute_metrics(y_true, y_pred_final, "PGM + KAGNN")
        
        # Statistical test
        errors_pgm = np.abs(y_true - y_pred_pgm)
        errors_final = np.abs(y_true - y_pred_final)
        t_stat, t_pval = ttest_rel(errors_pgm, errors_final)
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TEST")
        print("="*80)
        print(f"Paired t-test: t={t_stat:.4f}, p={t_pval:.6f}")
        
        improvement = (np.mean(errors_pgm) - np.mean(errors_final)) / np.mean(errors_pgm) * 100
        if t_pval < 0.05:
            print(f"✓ SIGNIFICANT improvement: {improvement:.2f}% error reduction (p < 0.05)")
        else:
            print(f"✗ NOT statistically significant: {improvement:.2f}% (p ≥ 0.05)")
        
        return metrics_pgm, metrics_final


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_pgm_kagnn_experiment():
    """Complete PGM→KAGNN experiment"""
    config = {
        'csv': '/path/to/SMRT_dataset.csv',
        'ecfp': '/path/to/SMRT_ECFP_1024_Fingerprints.txt',
        'sdf': '/path/to/SMRT_dataset.sdf'
    }
    
    # Load data
    loader = SMRTDataLoader(config['csv'], config['ecfp'], config['sdf'])
    df, ecfp_dict, mol_dict = loader.get_data()
    
    # 3-fold cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_results_pgm = []
    fold_results_final = []
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold+1}/3")
        print(f"{'='*80}")
        
        # Split data
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]
        
        train_idx, val_idx = train_test_split(range(len(train_val_df)), test_size=0.15, random_state=42)
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
        
        # Create dataloaders
        train_loader = DataLoader(SMRTDataset(train_df, ecfp_dict, mol_dict), 
                                  batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(SMRTDataset(val_df, ecfp_dict, mol_dict), 
                                batch_size=64, collate_fn=collate_fn)
        test_loader = DataLoader(SMRTDataset(test_df, ecfp_dict, mol_dict), 
                                 batch_size=64, collate_fn=collate_fn)
        
        # Train
        trainer = PGMKAGNNTrainer(train_loader, val_loader, test_loader, mol_dict, loader.rt_scaler)
        
        trainer.train_stage1_pgm(optimize=True, n_trials=30)
        trainer.train_stage2_kagnn(hidden_dim=256, dropout=0.2, epochs=100, patience=15)
        
        metrics_pgm, metrics_final = trainer.evaluate()
        
        fold_results_pgm.append(metrics_pgm)
        fold_results_final.append(metrics_final)
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATED 3-FOLD CV RESULTS")
    print("="*80)
    
    for model_name, results in [("PGM Only", fold_results_pgm), ("PGM + KAGNN", fold_results_final)]:
        print(f"\n{model_name}:")
        for metric in ['MedAE', 'MAE', 'RMSE', 'R2', 'Pearson']:
            values = [r[metric] for r in results]
            print(f"  {metric:12s}: {np.mean(values):8.4f} ± {np.std(values):6.4f}")
    
    print("\n PGM→KAGNN experiment complete!")


if __name__ == "__main__":
    run_pgm_kagnn_experiment()
