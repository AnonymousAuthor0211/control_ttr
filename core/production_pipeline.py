"""
Production pipeline wrapper for Control-TTR.

This module provides a clean interface to the full production pipeline
(spectral_refinement_baseline) with all features, fusion, Stage 1b, and FDR.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Import from standalone module (NO dependencies on original codebase)
from core.spectral_refinement_standalone import spectral_refinement_baseline

# Import from control_ttr core modules
from core.data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from core.metrics import compute_base_r1


# Feature weight configurations
EQUAL_WEIGHTS = {
    'w_rts': 0.2, 'w_ccs': 0.2, 'w_rtps': 0.2, 'w_dhc': 0.2, 'w_curv': 0.2,
    'w_contra': 0.0, 'w_rtrc': 0.0,
}

PRODUCTION_WEIGHTS = {
    'w_rts': 0.60, 'w_ccs': 0.05, 'w_rtps': 0.05, 'w_dhc': 0.15, 'w_curv': 0.05,
    'w_contra': 0.0, 'w_rtrc': 0.0,
}

# Base parameters
BASE_PARAMS = {
    'topK': 50, 'M_rts': 24, 'tau_ccs': 0.7, 'n_drops_rtps': 5,
    'lambda_smooth': 0.3, 'boost_scale': 0.35, 'fusion_weight': 0.5,
    'k_curv': 12, 'beta_curv': 0.4,
}

# Feature toggles
FEATURES_ON = {
    'use_rts': True, 'use_ccs': True, 'use_rtps': True,
    'use_dhc': True, 'use_curv': True, 'use_contra': False, 'use_rtrc': False,
}


def run_production_pipeline(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    gt_mapping: Dict,
    image_bank: Optional[torch.Tensor],
    text_bank: Optional[torch.Tensor],
    config: Dict,
    direction: str = 'i2t',
) -> Dict:
    """
    Run the full production pipeline with all features.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        query_ids: Query IDs
        gallery_ids: Gallery IDs
        gt_mapping: Ground truth mapping
        image_bank: Image bank for RTS (optional)
        text_bank: Text bank for RTS (optional)
        config: Configuration dict with keys:
            - budget_percentages: List of budget fractions for Stage 1a
            - use_polarity_aware: Enable Stage 1b fine veto
            - use_significance_gate: Enable Layer 3 FDR validation
            - fusion_weight: Weight for spectral vs features (0.0 = features only, 1.0 = spectral only)
            - feature_weights: 'equal' or 'production'
            - polarity_threshold: Threshold for Stage 1b
        direction: Retrieval direction ('i2t', 't2i', etc.)
    
    Returns:
        Dict with results:
            - r1: Recall@1
            - delta_r1: Change from baseline
            - flip_pct: Percentage of queries that flipped
            - A: Number of good flips
            - B: Number of bad flips
            - net_flip_per_1k: Net flips per 1000 queries
            - sel_pct: Selection percentage (Stage 1a)
            - spec_pct: Spectral rate (after Stage 1b)
            - purity: A/(A+B) if applicable
    """
    device = queries.device
    N = queries.shape[0]
    
    # Get configuration
    budget_percentages = config.get('budget_percentages', [1.0])
    use_polarity_aware = config.get('use_polarity_aware', False)
    use_significance_gate = config.get('use_significance_gate', False)
    fusion_weight = config.get('fusion_weight', 0.5)
    feature_weights_name = config.get('feature_weights', 'equal')
    polarity_threshold = config.get('polarity_threshold', 0.1)
    
    # Select feature weights
    if feature_weights_name == 'production':
        feature_weights = PRODUCTION_WEIGHTS
    else:
        feature_weights = EQUAL_WEIGHTS
    
    # Build baseline_params for spectral_refinement_baseline
    baseline_params = {
        **BASE_PARAMS,
        **FEATURES_ON,
        **feature_weights,
        'use_polarity_aware': use_polarity_aware,
        'use_significance_gate': use_significance_gate,
        'polarity_threshold': polarity_threshold,
        'fusion_weight': fusion_weight,
        'budget_percentages': budget_percentages,
        'frozen_bcg_mode': config.get('frozen_bcg_mode', 'rank_sum_oriented'),
    }
    
    # Override with variant-specific parameters if provided
    if 'topK' in config:
        baseline_params['topK'] = config['topK']
    if 'lambda_smooth' in config:
        baseline_params['lambda_smooth'] = config['lambda_smooth']
    
    # Stage 1b feature flags (for Table 3)
    if 'stage1b_use_rts' in config:
        baseline_params['stage1b_use_rts'] = config['stage1b_use_rts']
    if 'stage1b_use_ccs' in config:
        baseline_params['stage1b_use_ccs'] = config['stage1b_use_ccs']
    if 'stage1b_use_curv' in config:
        baseline_params['stage1b_use_curv'] = config['stage1b_use_curv']
    
    # Convert IDs to lists (spectral_refinement_baseline expects lists)
    query_ids_list = [str(qid) for qid in query_ids]
    gallery_ids_list = [str(gid) for gid in gallery_ids]
    
    # Run the production pipeline
    results_dict = spectral_refinement_baseline(
        queries=queries,
        gallery=gallery,
        query_ids=query_ids_list,
        gallery_ids=gallery_ids_list,
        gt_mapping=gt_mapping,
        baseline_params=baseline_params,
        direction=direction,
        image_bank=image_bank,
        text_bank=text_bank,
    )
    
    # Extract metrics - match the structure from scripts/compute_complementarity_evidence.py
    base_r1 = compute_base_r1(queries, gallery, gt_mapping, query_ids)[0]
    refined_r1 = results_dict.get('recall@1', base_r1)  # Function returns 'recall@1'
    delta_r1 = refined_r1 - base_r1
    
    # Get flip statistics - function returns 'flips_A' and 'flips_B'
    A = results_dict.get('flips_A', 0)
    B = results_dict.get('flips_B', 0)
    
    # Use flip_rate from results if available (matches original), otherwise compute
    flip_rate = results_dict.get('flip_rate', 0.0)  # Fraction from function
    flip_pct = flip_rate * 100.0 if flip_rate > 0 else (100.0 * (A + B) / N if N > 0 else 0.0)
    
    # Net flip per 1k - use from results if available
    netflip_1k = results_dict.get('netflip_per_1k', 0.0)
    if netflip_1k == 0.0 and (A + B) > 0:
        # Fallback calculation if not provided
        netflip_1k = 1000.0 * (A - B) / N if N > 0 else 0.0
    
    purity = A / (A + B) if (A + B) > 0 else None
    
    # Get selection statistics - match original naming
    # stage1a_sel is already a percentage (from the function)
    stage1a_sel = results_dict.get('stage1a_sel', 100.0)
    
    # Spectral rate: gate_rate is the actual refinement rate (after Stage 1b vetoes)
    # Convert from fraction to percentage
    gate_rate_frac = results_dict.get('gate_rate', 1.0)
    gate_pct = gate_rate_frac * 100.0
    spectral_rate = gate_pct  # Alias for consistency with original
    
    return {
        # Match original naming convention
        'R1': refined_r1,
        'delta_R1': delta_r1,
        'flip_pct': flip_pct,
        'A': A,
        'B': B,
        'A/B': f"{A}/{B}",
        'netflip_1k': netflip_1k,
        'netflip_per_1k': netflip_1k,  # Also provide with underscore for compatibility
        'gate_pct': gate_pct,
        'spectral_rate': spectral_rate,
        'stage1a_sel': stage1a_sel,
        'purity': purity,
        'base_r1': base_r1,
        # Also provide lowercase versions for backward compatibility
        'r1': refined_r1,
        'delta_r1': delta_r1,
        'sel_pct': stage1a_sel,
        'spec_pct': spectral_rate,
        'net_flip_per_1k': netflip_1k,
    }


def run_variant_production(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    gt_mapping: Dict,
    image_bank: Optional[torch.Tensor],
    text_bank: Optional[torch.Tensor],
    variant: Dict,
    direction: str = 'i2t',
) -> Dict:
    """
    Run a single variant using the production pipeline.
    
    Args:
        variant: Dict with keys:
            - name: Variant name
            - l1a: Use BCG selection (bool)
            - l1b: Use fine veto (bool)
            - l2: Use feature scoring + spectral (bool)
            - l3: Use FDR validation (bool)
            - budget: Budget fraction for L1a (float, default 0.10)
            - fusion_weight: Weight for spectral (0.0=features only, 1.0=spectral only, 0.5=both)
            - feature_weights: 'equal' or 'production'
    
    Returns:
        Dict with results
    """
    # Build config from variant
    if variant.get('l1a', False):
        budget_percentages = [variant.get('budget', 0.10)]
    else:
        budget_percentages = [1.0]  # All queries
    
    # Determine fusion_weight
    if not variant.get('l2', False):
        # No Layer 2: no refinement
        fusion_weight = 0.0
    elif variant.get('fusion_weight') is not None:
        fusion_weight = variant['fusion_weight']
    else:
        # Default: both spectral and features
        fusion_weight = 0.5
    
    config = {
        'budget_percentages': budget_percentages,
        'use_polarity_aware': variant.get('l1b', False),
        'use_significance_gate': variant.get('l3', False),
        'fusion_weight': fusion_weight,
        'feature_weights': variant.get('feature_weights', 'equal'),
        'polarity_threshold': 0.1,
    }
    
    # Pass through variant-specific parameters
    if 'topK' in variant:
        config['topK'] = variant['topK']
    if 'lambda_smooth' in variant:
        config['lambda_smooth'] = variant['lambda_smooth']
    if 'stage1b_use_rts' in variant:
        config['stage1b_use_rts'] = variant['stage1b_use_rts']
    if 'stage1b_use_ccs' in variant:
        config['stage1b_use_ccs'] = variant['stage1b_use_ccs']
    if 'stage1b_use_curv' in variant:
        config['stage1b_use_curv'] = variant['stage1b_use_curv']
    if 'frozen_bcg_mode' in variant:
        config['frozen_bcg_mode'] = variant['frozen_bcg_mode']
    
    # Run pipeline
    results = run_production_pipeline(
        queries, gallery, query_ids, gallery_ids, gt_mapping,
        image_bank, text_bank, config, direction
    )
    
    # Add variant name
    results['variant'] = variant['name']
    
    return results
