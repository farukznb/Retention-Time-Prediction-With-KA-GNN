"""
KAGNN+ECFP model for retention time prediction.

This module implements the baseline KAGNN model combined with ECFP fingerprints.
"""

import torch
import torch.nn as nn
from typing import Any
from models import KAGIN, make_kan, get_atom_feature_dims, get_bond_feature_dims
import sys
sys.path.append('..')
from base_model import BaseRTModel


class KAGIN_ECFP_Combined(BaseRTModel):
    """
    KAGNN model combined with ECFP fingerprints for RT prediction.
    
    This model uses:
    - KAGNN (KAN-based GNN) for molecular graph representation
    - KAN network for ECFP fingerprint processing
    - Combined representation for final RT prediction
    """
    
    def __init__(self, config: Any):
        """
        Initialize KAGNN+ECFP model.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        super().__init__(config)
        
        # KAGNN for graph processing
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
        
        # KAN for ECFP processing
        self.ecfp_kan = make_kan(
            1024,  # ECFP dimension
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_layers,
            config.grid_size,
            config.spline_order
        )
        
        # Final KAN for prediction
        self.final_kan = make_kan(
            2 * config.hidden_dim,  # Concatenated graph + ECFP embeddings
            config.hidden_dim // 2,
            1,  # Output: single RT value
            config.hidden_layers,
            config.grid_size,
            config.spline_order
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, graph_data, ecfp):
        """
        Forward pass of the model.
        
        Args:
            graph_data: PyTorch Geometric graph batch
            ecfp: ECFP fingerprint tensor [batch_size, 1024]
            
        Returns:
            torch.Tensor: Predicted retention times (normalized) [batch_size]
        """
        # Process graph through KAGNN
        graph_emb = self.kagnn(graph_data)
        
        # Process ECFP through KAN
        ecfp_emb = self.ecfp_kan(ecfp)
        
        # Concatenate representations
        x = torch.cat([graph_emb, ecfp_emb], dim=-1)
        x = self.dropout(x)
        
        # Final prediction
        output = self.final_kan(x).squeeze(-1)
        
        return output
