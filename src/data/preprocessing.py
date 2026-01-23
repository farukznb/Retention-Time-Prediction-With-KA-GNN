"""
Data preprocessing utilities for molecular featurization.

This module provides utilities for feature extraction from molecules,
including atom/bond featurization for graph neural networks.
"""

import torch
from typing import List
from models import allowable_features


def atom_to_indices(atom) -> List[int]:
    """
    Convert RDKit atom to feature indices.
    
    Args:
        atom: RDKit atom object
        
    Returns:
        List of 9 feature indices representing atom properties
    """
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
    
    # Atomic number
    feats.append(zs.index(atom.GetAtomicNum()) if atom.GetAtomicNum() in zs else zs.index('misc'))

    # Chirality
    ch_val = str(atom.GetChiralTag())
    if 'CHI_TETRAHEDRAL_CW' in ch_val: 
        ch_val = 'CHI_TETRAHEDRAL_CW'
    elif 'CHI_TETRAHEDRAL_CCW' in ch_val: 
        ch_val = 'CHI_TETRAHEDRAL_CCW'
    elif 'CHI_UNSPECIFIED' in ch_val: 
        ch_val = 'CHI_UNSPECIFIED'
    else: 
        ch_val = 'CHI_OTHER'
    feats.append(ch_list.index(ch_val) if ch_val in ch_list else ch_list.index('misc'))

    # Degree
    deg = atom.GetDegree()
    feats.append(deg_list.index(deg) if deg in deg_list else deg_list.index('misc'))

    # Formal charge
    fc = atom.GetFormalCharge()
    feats.append(fc_list.index(fc) if fc in fc_list else fc_list.index('misc'))

    # Number of hydrogens
    nh = atom.GetTotalNumHs()
    feats.append(h_list.index(nh) if nh in h_list else h_list.index('misc'))

    # Radical electrons
    nr = atom.GetNumRadicalElectrons()
    feats.append(rad_list.index(nr) if nr in rad_list else rad_list.index('misc'))

    # Hybridization
    hyb = str(atom.GetHybridization())
    hval = 'misc'
    if 'SP' in hyb: hval = 'SP'
    elif 'SP2' in hyb: hval = 'SP2'
    elif 'SP3' in hyb: hval = 'SP3'
    elif 'SP3D' in hyb: hval = 'SP3D'
    elif 'SP3D2' in hyb: hval = 'SP3D2'
    feats.append(hyb_list.index(hval) if hval in hyb_list else hyb_list.index('misc'))

    # Aromaticity
    feats.append(arom_list.index(atom.GetIsAromatic()))
    
    # Ring membership
    feats.append(ring_list.index(atom.IsInRing()))
    
    return feats


def bond_to_indices(bond) -> List[int]:
    """
    Convert RDKit bond to feature indices.
    
    Args:
        bond: RDKit bond object
        
    Returns:
        List of 3 feature indices representing bond properties
    """
    bt_list = allowable_features['possible_bond_type_list']
    st_list = allowable_features['possible_bond_stereo_list']
    conj_list = allowable_features['possible_is_conjugated_list']

    # Bond type
    btype = str(bond.GetBondType())
    bval = 'misc'
    if 'SINGLE' in btype: 
        bval = 'SINGLE'
    elif 'DOUBLE' in btype: 
        bval = 'DOUBLE'
    elif 'TRIPLE' in btype: 
        bval = 'TRIPLE'
    elif 'AROMATIC' in btype: 
        bval = 'AROMATIC'

    # Bond stereo
    stype = str(bond.GetStereo())
    mapped = 'STEREOANY'
    for s in st_list:
        if s in stype: 
            mapped = s
            break
    
    return [
        bt_list.index(bval) if bval in bt_list else bt_list.index('misc'),
        st_list.index(mapped),
        conj_list.index(bond.GetIsConjugated())
    ]


def create_train_val_test_split(n_samples: int, val_size: float = 0.15, 
                                test_size: float = 0.15, seed: int = 42):
    """
    Create train/validation/test split indices.
    
    Args:
        n_samples: Total number of samples
        val_size: Fraction for validation set
        test_size: Fraction for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    import numpy as np
    
    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_val = int(val_size * n_samples)
    n_test = int(test_size * n_samples)
    
    train_indices = indices[: n_samples - n_val - n_test]
    val_indices = indices[n_samples - n_val - n_test: n_samples - n_test]
    test_indices = indices[n_samples - n_test:]
    
    return train_indices, val_indices, test_indices
