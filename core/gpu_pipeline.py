"""
GPU-Accelerated Production Pipeline for Control-TTR.

This module provides fast batched implementations for all table experiments.
Uses GPU operations from gpu_ops.py for 10-50x speedup over CPU pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .gpu_ops import (
    batch_spectral_smooth,
    compute_rts_batched,
    compute_ccs_batched,
    compute_curv_batched,
    compute_dhc_batched,
    compute_rtps_batched,
    batch_lfa_refinement,
    compute_bcg_scores,
    compute_flips_from_arrays,
    PROD_PARAMS,
)
from .data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from .metrics import compute_base_r1


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================
EQUAL_WEIGHTS = {
    'rts_weight': 0.20,
    'ccs_weight': 0.20,
    'curv_weight': 0.20,
    'dhc_weight': 0.20,
    'rtps_weight': 0.20,
}

PRODUCTION_WEIGHTS = {
    'rts_weight': 0.60,
    'ccs_weight': 0.05,
    'curv_weight': 0.05,
    'dhc_weight': 0.15,
    'rtps_weight': 0.05,
}


# =============================================================================
# DATA LOADING HELPER
# =============================================================================
def load_data_gpu(
    dataset: str,
    direction: str,
    backbone: str,
    embeddings_root: Optional[str] = None,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, List, np.ndarray]:
    """
    Load and prepare all data for GPU-accelerated evaluation.
    
    Returns:
        (query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, gallery_ids)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    query_emb, gallery_emb, query_ids, gallery_ids = load_embeddings(
        dataset, direction, backbone, embeddings_root=embeddings_root
    )
    image_bank, text_bank = load_rts_distractor_bank(
        dataset, direction, backbone, embeddings_root=embeddings_root
    )
    distractor_bank = text_bank if direction == 't2i' else image_bank
    
    # Dedupe for i2t
    if direction in ['i2t', 'a2t']:
        unique_ids, unique_idx = np.unique(query_ids, return_index=True)
        query_emb = query_emb[unique_idx]
        query_ids = query_ids[unique_idx]
    
    gt_mapping = build_gt_mapping(query_ids, gallery_ids, direction)
    
    # Move to GPU and normalize
    query_emb = F.normalize(query_emb.to(device), dim=-1)
    gallery_emb = F.normalize(gallery_emb.to(device), dim=-1)
    if distractor_bank is not None:
        distractor_bank = F.normalize(distractor_bank.to(device), dim=-1)
    
    query_ids_list = query_ids.tolist() if hasattr(query_ids, 'tolist') else list(query_ids)
    
    return query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids_list, gallery_ids


# =============================================================================
# LAYER ABLATION (Table 1) - GPU Version
# =============================================================================
def run_layer_ablation_gpu(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    distractor_bank: Optional[torch.Tensor],
    gt_mapping: Dict,
    query_ids: List,
    direction: str = 't2i',
    batch_size: int = 256,
) -> List[Dict]:
    """
    GPU-accelerated Layer-wise ablation (Table 1).
    
    Returns list of results for each variant.
    """
    device = query_emb.device
    n_queries = len(query_ids)
    
    # Get base R@1
    base_r1, _ = compute_base_r1(query_emb, gallery_emb, gt_mapping, query_ids)
    
    # Compute base top-1 for all queries
    all_sims = query_emb @ gallery_emb.T
    base_top1 = all_sims.argmax(dim=1).cpu().numpy()
    
    # Get top-K for each query
    topK = PROD_PARAMS['topK']
    topk_scores, topk_idx = torch.topk(all_sims, k=topK, dim=1)
    cand_embs = gallery_emb[topk_idx]  # [N, K, D]
    
    # Compute all features
    all_smoothed, all_rts, all_ccs, all_curv, all_dhc, all_rtps = [], [], [], [], [], []
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_query = query_emb[start:end]
        batch_scores = topk_scores[start:end]
        batch_cands = cand_embs[start:end]
        
        # Spectral smoothing
        smoothed = batch_spectral_smooth(batch_scores, batch_cands, PROD_PARAMS['lambda_smooth'])
        all_smoothed.append(smoothed.cpu())
        
        # Features
        if distractor_bank is not None:
            rts = compute_rts_batched(batch_query, batch_cands, distractor_bank, M=24)
        else:
            rts = torch.zeros(end - start, topK, device=device)
        all_rts.append(rts.cpu())
        all_ccs.append(compute_ccs_batched(batch_cands).cpu())
        all_curv.append(compute_curv_batched(batch_cands).cpu())
        all_dhc.append(compute_dhc_batched(batch_cands).cpu())
        all_rtps.append(compute_rtps_batched(batch_query, batch_cands).cpu())
    
    all_smoothed = torch.cat(all_smoothed, dim=0)
    all_rts = torch.cat(all_rts, dim=0)
    all_ccs = torch.cat(all_ccs, dim=0)
    all_curv = torch.cat(all_curv, dim=0)
    all_dhc = torch.cat(all_dhc, dim=0)
    all_rtps = torch.cat(all_rtps, dim=0)
    topk_scores_cpu = topk_scores.cpu()
    topk_idx_cpu = topk_idx.cpu()
    
    def zscore(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std
    
    # Z-score features
    rts_z = zscore(all_rts)
    ccs_z = zscore(all_ccs)
    curv_z = zscore(all_curv)
    dhc_z = zscore(all_dhc)
    rtps_z = zscore(all_rtps)
    base_z = zscore(topk_scores_cpu)
    spectral_z = zscore(all_smoothed)
    
    # Feature boost (production weights)
    f_boost = (
        PROD_PARAMS['rts_weight'] * rts_z +
        PROD_PARAMS['ccs_weight'] * ccs_z -
        PROD_PARAMS['curv_weight'] * curv_z -
        PROD_PARAMS['dhc_weight'] * dhc_z +
        PROD_PARAMS['rtps_weight'] * rtps_z
    )
    
    results = []
    
    # 1. Base (no refinement)
    results.append({
        'variant': 'Base (no refinement)',
        'sel_pct': 0.0, 'spec_pct': 0.0,
        'R1': base_r1, 'delta_R1': 0.0, 'flip_pct': 0.0,
        'A': 0, 'B': 0, 'A/B': '0/0', 'netflip_1k': 0.0, 'purity': None,
    })
    
    # 2. PLD (Spectral only)
    pld_scores = spectral_z
    pld_top1_local = pld_scores.argmax(dim=1)
    pld_top1 = topk_idx_cpu[torch.arange(n_queries), pld_top1_local].numpy()
    A_pld, B_pld = compute_flips_from_arrays(base_top1, pld_top1, gt_mapping, query_ids)
    delta_pld = (A_pld - B_pld) / n_queries
    purity_pld = A_pld / (A_pld + B_pld) if (A_pld + B_pld) > 0 else None
    results.append({
        'variant': 'PLD (Spectral only)',
        'sel_pct': 100.0, 'spec_pct': 100.0,
        'R1': base_r1 + delta_pld, 'delta_R1': delta_pld,
        'flip_pct': 100.0 * (A_pld + B_pld) / n_queries,
        'A': A_pld, 'B': B_pld, 'A/B': f'{A_pld}/{B_pld}',
        'netflip_1k': 1000 * delta_pld, 'purity': purity_pld,
    })
    
    # 3. Features-only (no spectral)
    feat_scores = base_z + PROD_PARAMS['boost_scale'] * f_boost
    feat_top1_local = feat_scores.argmax(dim=1)
    feat_top1 = topk_idx_cpu[torch.arange(n_queries), feat_top1_local].numpy()
    A_feat, B_feat = compute_flips_from_arrays(base_top1, feat_top1, gt_mapping, query_ids)
    delta_feat = (A_feat - B_feat) / n_queries
    purity_feat = A_feat / (A_feat + B_feat) if (A_feat + B_feat) > 0 else None
    results.append({
        'variant': 'Features-only (Always-on)',
        'sel_pct': 100.0, 'spec_pct': 0.0,
        'R1': base_r1 + delta_feat, 'delta_R1': delta_feat,
        'flip_pct': 100.0 * (A_feat + B_feat) / n_queries,
        'A': A_feat, 'B': B_feat, 'A/B': f'{A_feat}/{B_feat}',
        'netflip_1k': 1000 * delta_feat, 'purity': purity_feat,
    })
    
    # 4. LFA (Spectral + Features)
    fw = PROD_PARAMS['fusion_weight']
    lfa_scores = fw * spectral_z + (1 - fw) * (base_z + PROD_PARAMS['boost_scale'] * f_boost)
    lfa_top1_local = lfa_scores.argmax(dim=1)
    lfa_top1 = topk_idx_cpu[torch.arange(n_queries), lfa_top1_local].numpy()
    A_lfa, B_lfa = compute_flips_from_arrays(base_top1, lfa_top1, gt_mapping, query_ids)
    delta_lfa = (A_lfa - B_lfa) / n_queries
    purity_lfa = A_lfa / (A_lfa + B_lfa) if (A_lfa + B_lfa) > 0 else None
    results.append({
        'variant': 'LFA (Spectral+Features)',
        'sel_pct': 100.0, 'spec_pct': 100.0,
        'R1': base_r1 + delta_lfa, 'delta_R1': delta_lfa,
        'flip_pct': 100.0 * (A_lfa + B_lfa) / n_queries,
        'A': A_lfa, 'B': B_lfa, 'A/B': f'{A_lfa}/{B_lfa}',
        'netflip_1k': 1000 * delta_lfa, 'purity': purity_lfa,
    })
    
    # 5. BCG @ 10% and 20% (without strict gates for cleaner comparison)
    margin_scores = (topk_scores_cpu[:, 0] - topk_scores_cpu[:, 1]).numpy()
    bcg_scores = compute_bcg_scores(query_emb, gallery_emb, distractor_bank, margin_scores, batch_size=500)
    
    for budget in [0.10, 0.20]:
        n_select = int(budget * n_queries)
        bcg_order = np.argsort(bcg_scores)[::-1][:n_select]
        
        # Apply LFA to selected queries (no gates for fair comparison)
        bcg_top1 = base_top1.copy()
        bcg_mask = np.isin(np.arange(n_queries), bcg_order)
        bcg_top1[bcg_mask] = lfa_top1[bcg_mask]
        
        A_bcg, B_bcg = compute_flips_from_arrays(base_top1, bcg_top1, gt_mapping, query_ids, bcg_mask)
        delta_bcg = (A_bcg - B_bcg) / n_queries
        purity_bcg = A_bcg / (A_bcg + B_bcg) if (A_bcg + B_bcg) > 0 else None
        
        results.append({
            'variant': f'BCG @ {int(budget*100)}%',
            'sel_pct': budget * 100.0,
            'spec_pct': 100.0,
            'R1': base_r1 + delta_bcg, 'delta_R1': delta_bcg,
            'flip_pct': 100.0 * (A_bcg + B_bcg) / n_queries,
            'A': A_bcg, 'B': B_bcg, 'A/B': f'{A_bcg}/{B_bcg}',
            'netflip_1k': 1000 * delta_bcg, 'purity': purity_bcg,
        })
    
    return results


# =============================================================================
# BUDGET SWEEP (Table 5) - GPU Version
# =============================================================================
def run_budget_sweep_gpu(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    distractor_bank: Optional[torch.Tensor],
    gt_mapping: Dict,
    query_ids: List,
    direction: str = 't2i',
    budgets: List[float] = None,
    batch_size: int = 256,
    n_seeds: int = 5,
    use_gates: bool = False,  # Set to False for simpler budget comparison
) -> List[Dict]:
    """
    GPU-accelerated Budget Sweep (Table 5).
    
    Compares BCG vs Margin vs Random at different budgets.
    Note: For clean comparison, we apply LFA to ALL selected queries without gates.
    """
    if budgets is None:
        budgets = [0.05, 0.10, 0.20]
    
    device = query_emb.device
    n_queries = len(query_ids)
    
    # Get base R@1
    base_r1, _ = compute_base_r1(query_emb, gallery_emb, gt_mapping, query_ids)
    
    # Pre-compute margin scores
    all_sims = query_emb @ gallery_emb.T
    top2_scores, _ = torch.topk(all_sims, k=2, dim=1)
    margin_scores = (top2_scores[:, 0] - top2_scores[:, 1]).cpu().numpy()
    base_top1 = all_sims.argmax(dim=1).cpu().numpy()
    
    # Compute BCG scores
    bcg_scores = compute_bcg_scores(query_emb, gallery_emb, distractor_bank, margin_scores, batch_size=500)
    
    # Pre-compute LFA refined top-1 for all queries (no gates, just LFA)
    topK = PROD_PARAMS['topK']
    topk_scores, topk_idx = torch.topk(all_sims, k=topK, dim=1)
    cand_embs = gallery_emb[topk_idx]
    
    # Compute features
    all_smoothed, all_rts, all_ccs, all_curv, all_dhc, all_rtps = [], [], [], [], [], []
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_query = query_emb[start:end]
        batch_scores = topk_scores[start:end]
        batch_cands = cand_embs[start:end]
        
        smoothed = batch_spectral_smooth(batch_scores, batch_cands, PROD_PARAMS['lambda_smooth'])
        all_smoothed.append(smoothed.cpu())
        
        if distractor_bank is not None:
            rts = compute_rts_batched(batch_query, batch_cands, distractor_bank, M=24)
        else:
            rts = torch.zeros(end - start, topK, device=device)
        all_rts.append(rts.cpu())
        all_ccs.append(compute_ccs_batched(batch_cands).cpu())
        all_curv.append(compute_curv_batched(batch_cands).cpu())
        all_dhc.append(compute_dhc_batched(batch_cands).cpu())
        all_rtps.append(compute_rtps_batched(batch_query, batch_cands).cpu())
    
    all_smoothed = torch.cat(all_smoothed, dim=0)
    all_rts = torch.cat(all_rts, dim=0)
    all_ccs = torch.cat(all_ccs, dim=0)
    all_curv = torch.cat(all_curv, dim=0)
    all_dhc = torch.cat(all_dhc, dim=0)
    all_rtps = torch.cat(all_rtps, dim=0)
    topk_scores_cpu = topk_scores.cpu()
    topk_idx_cpu = topk_idx.cpu()
    
    def zscore(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std
    
    rts_z = zscore(all_rts)
    ccs_z = zscore(all_ccs)
    curv_z = zscore(all_curv)
    dhc_z = zscore(all_dhc)
    rtps_z = zscore(all_rtps)
    base_z = zscore(topk_scores_cpu)
    spectral_z = zscore(all_smoothed)
    
    f_boost = (
        PROD_PARAMS['rts_weight'] * rts_z +
        PROD_PARAMS['ccs_weight'] * ccs_z -
        PROD_PARAMS['curv_weight'] * curv_z -
        PROD_PARAMS['dhc_weight'] * dhc_z +
        PROD_PARAMS['rtps_weight'] * rtps_z
    )
    
    fw = PROD_PARAMS['fusion_weight']
    lfa_scores = fw * spectral_z + (1 - fw) * (base_z + PROD_PARAMS['boost_scale'] * f_boost)
    lfa_top1_local = lfa_scores.argmax(dim=1)
    lfa_top1 = topk_idx_cpu[torch.arange(n_queries), lfa_top1_local].numpy()
    
    results = []
    
    for budget in budgets:
        n_select = max(1, int(budget * n_queries))
        
        # BCG selection (select top n_select by BCG score)
        bcg_order = np.argsort(bcg_scores)[::-1][:n_select]
        bcg_mask = np.isin(np.arange(n_queries), bcg_order)
        
        bcg_final = base_top1.copy()
        bcg_final[bcg_mask] = lfa_top1[bcg_mask]
        
        A_bcg, B_bcg = compute_flips_from_arrays(base_top1, bcg_final, gt_mapping, query_ids, bcg_mask)
        
        # Margin selection (select n_select lowest margin queries)
        margin_order = np.argsort(margin_scores)[:n_select]
        margin_mask = np.isin(np.arange(n_queries), margin_order)
        
        margin_final = base_top1.copy()
        margin_final[margin_mask] = lfa_top1[margin_mask]
        
        A_margin, B_margin = compute_flips_from_arrays(base_top1, margin_final, gt_mapping, query_ids, margin_mask)
        
        # Random selection (avg over seeds)
        random_As, random_Bs = [], []
        for seed in range(n_seeds):
            np.random.seed(seed + 42)
            random_order = np.random.choice(n_queries, size=n_select, replace=False)
            random_mask = np.isin(np.arange(n_queries), random_order)
            
            random_final = base_top1.copy()
            random_final[random_mask] = lfa_top1[random_mask]
            
            A_r, B_r = compute_flips_from_arrays(base_top1, random_final, gt_mapping, query_ids, random_mask)
            random_As.append(A_r)
            random_Bs.append(B_r)
        
        A_rand = int(np.mean(random_As))
        B_rand = int(np.mean(random_Bs))
        
        exec_pct = 100.0 * n_select / n_queries
        
        for sel, A, B in [('BCG', A_bcg, B_bcg), ('Margin', A_margin, B_margin), ('Random', A_rand, B_rand)]:
            delta = (A - B) / n_queries
            r1 = base_r1 + delta
            purity = A / (A + B) if (A + B) > 0 else None
            results.append({
                'variant': f'{sel} @ {int(budget*100)}%',
                'selection': sel, 'budget': budget, 'exec_pct': exec_pct,
                'R1': r1, 'delta_R1': delta, 'A': A, 'B': B,
                'A/B': f'{A}/{B}', 'netflip_1k': 1000 * delta, 'purity': purity,
            })
    
    return results


# =============================================================================
# FEATURE ABLATION (Table 3/6) - GPU Version
# =============================================================================
def run_feature_ablation_gpu(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    distractor_bank: Optional[torch.Tensor],
    gt_mapping: Dict,
    query_ids: List,
    direction: str = 't2i',
    batch_size: int = 256,
    use_equal_weights: bool = False,
) -> List[Dict]:
    """
    GPU-accelerated Feature Ablation (drop one feature at a time).
    """
    device = query_emb.device
    n_queries = len(query_ids)
    
    # Get base R@1
    base_r1, _ = compute_base_r1(query_emb, gallery_emb, gt_mapping, query_ids)
    
    # Compute base top-1
    all_sims = query_emb @ gallery_emb.T
    base_top1 = all_sims.argmax(dim=1).cpu().numpy()
    
    topK = PROD_PARAMS['topK']
    topk_scores, topk_idx = torch.topk(all_sims, k=topK, dim=1)
    cand_embs = gallery_emb[topk_idx]
    
    # Compute all features
    all_rts, all_ccs, all_curv, all_dhc, all_rtps = [], [], [], [], []
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_query = query_emb[start:end]
        batch_cands = cand_embs[start:end]
        
        if distractor_bank is not None:
            rts = compute_rts_batched(batch_query, batch_cands, distractor_bank, M=24)
        else:
            rts = torch.zeros(end - start, topK, device=device)
        all_rts.append(rts.cpu())
        all_ccs.append(compute_ccs_batched(batch_cands).cpu())
        all_curv.append(compute_curv_batched(batch_cands).cpu())
        all_dhc.append(compute_dhc_batched(batch_cands).cpu())
        all_rtps.append(compute_rtps_batched(batch_query, batch_cands).cpu())
    
    all_rts = torch.cat(all_rts, dim=0)
    all_ccs = torch.cat(all_ccs, dim=0)
    all_curv = torch.cat(all_curv, dim=0)
    all_dhc = torch.cat(all_dhc, dim=0)
    all_rtps = torch.cat(all_rtps, dim=0)
    topk_scores_cpu = topk_scores.cpu()
    topk_idx_cpu = topk_idx.cpu()
    
    def zscore(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std
    
    rts_z = zscore(all_rts)
    ccs_z = zscore(all_ccs)
    curv_z = zscore(all_curv)
    dhc_z = zscore(all_dhc)
    rtps_z = zscore(all_rtps)
    base_z = zscore(topk_scores_cpu)
    
    # Select weights
    weights = EQUAL_WEIGHTS if use_equal_weights else PRODUCTION_WEIGHTS
    
    # Feature configurations
    configs = {
        'Full (all features)': {'rts': True, 'ccs': True, 'curv': True, 'dhc': True, 'rtps': True},
        'w/o RTS': {'rts': False, 'ccs': True, 'curv': True, 'dhc': True, 'rtps': True},
        'w/o CCS': {'rts': True, 'ccs': False, 'curv': True, 'dhc': True, 'rtps': True},
        'w/o CURV': {'rts': True, 'ccs': True, 'curv': False, 'dhc': True, 'rtps': True},
        'w/o DHC': {'rts': True, 'ccs': True, 'curv': True, 'dhc': False, 'rtps': True},
        'w/o RTPS': {'rts': True, 'ccs': True, 'curv': True, 'dhc': True, 'rtps': False},
    }
    
    results = []
    
    for name, cfg in configs.items():
        # Compute feature boost with this config
        f_boost = torch.zeros_like(rts_z)
        if cfg['rts']:
            f_boost += weights['rts_weight'] * rts_z
        if cfg['ccs']:
            f_boost += weights['ccs_weight'] * ccs_z
        if cfg['curv']:
            f_boost -= weights['curv_weight'] * curv_z
        if cfg['dhc']:
            f_boost -= weights['dhc_weight'] * dhc_z
        if cfg['rtps']:
            f_boost += weights['rtps_weight'] * rtps_z
        
        # Features-only scoring
        feat_scores = base_z + PROD_PARAMS['boost_scale'] * f_boost
        feat_top1_local = feat_scores.argmax(dim=1)
        feat_top1 = topk_idx_cpu[torch.arange(n_queries), feat_top1_local].numpy()
        
        A, B = compute_flips_from_arrays(base_top1, feat_top1, gt_mapping, query_ids)
        
        delta = (A - B) / n_queries
        r1 = base_r1 + delta
        purity = A / (A + B) if (A + B) > 0 else None
        
        results.append({
            'variant': name,
            'R1': r1, 'delta_R1': delta,
            'flip_pct': 100.0 * (A + B) / n_queries,
            'A': A, 'B': B, 'A/B': f'{A}/{B}',
            'netflip_1k': 1000 * delta, 'purity': purity,
        })
    
    return results


# =============================================================================
# K SWEEP (Table 7) - GPU Version
# =============================================================================
def run_k_sweep_gpu(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    distractor_bank: Optional[torch.Tensor],
    gt_mapping: Dict,
    query_ids: List,
    direction: str = 't2i',
    k_values: List[int] = None,
    batch_size: int = 256,
) -> List[Dict]:
    """
    GPU-accelerated K sweep (candidate pool size).
    """
    if k_values is None:
        k_values = [20, 50, 100]
    
    device = query_emb.device
    n_queries = len(query_ids)
    
    # Get base R@1
    base_r1, _ = compute_base_r1(query_emb, gallery_emb, gt_mapping, query_ids)
    
    all_sims = query_emb @ gallery_emb.T
    base_top1 = all_sims.argmax(dim=1).cpu().numpy()
    
    results = []
    
    for K in k_values:
        topk_scores, topk_idx = torch.topk(all_sims, k=K, dim=1)
        cand_embs = gallery_emb[topk_idx]
        
        # Compute features
        all_smoothed, all_rts, all_ccs, all_curv, all_dhc, all_rtps = [], [], [], [], [], []
        
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_query = query_emb[start:end]
            batch_scores = topk_scores[start:end]
            batch_cands = cand_embs[start:end]
            
            smoothed = batch_spectral_smooth(batch_scores, batch_cands, PROD_PARAMS['lambda_smooth'])
            all_smoothed.append(smoothed.cpu())
            
            if distractor_bank is not None:
                rts = compute_rts_batched(batch_query, batch_cands, distractor_bank, M=24)
            else:
                rts = torch.zeros(end - start, K, device=device)
            all_rts.append(rts.cpu())
            all_ccs.append(compute_ccs_batched(batch_cands).cpu())
            all_curv.append(compute_curv_batched(batch_cands).cpu())
            all_dhc.append(compute_dhc_batched(batch_cands).cpu())
            all_rtps.append(compute_rtps_batched(batch_query, batch_cands).cpu())
        
        all_smoothed = torch.cat(all_smoothed, dim=0)
        all_rts = torch.cat(all_rts, dim=0)
        all_ccs = torch.cat(all_ccs, dim=0)
        all_curv = torch.cat(all_curv, dim=0)
        all_dhc = torch.cat(all_dhc, dim=0)
        all_rtps = torch.cat(all_rtps, dim=0)
        topk_scores_cpu = topk_scores.cpu()
        topk_idx_cpu = topk_idx.cpu()
        
        def zscore(x):
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
            return (x - mean) / std
        
        rts_z = zscore(all_rts)
        ccs_z = zscore(all_ccs)
        curv_z = zscore(all_curv)
        dhc_z = zscore(all_dhc)
        rtps_z = zscore(all_rtps)
        base_z = zscore(topk_scores_cpu)
        spectral_z = zscore(all_smoothed)
        
        f_boost = (
            PROD_PARAMS['rts_weight'] * rts_z +
            PROD_PARAMS['ccs_weight'] * ccs_z -
            PROD_PARAMS['curv_weight'] * curv_z -
            PROD_PARAMS['dhc_weight'] * dhc_z +
            PROD_PARAMS['rtps_weight'] * rtps_z
        )
        
        fw = PROD_PARAMS['fusion_weight']
        lfa_scores = fw * spectral_z + (1 - fw) * (base_z + PROD_PARAMS['boost_scale'] * f_boost)
        lfa_top1_local = lfa_scores.argmax(dim=1)
        lfa_top1 = topk_idx_cpu[torch.arange(n_queries), lfa_top1_local].numpy()
        
        A, B = compute_flips_from_arrays(base_top1, lfa_top1, gt_mapping, query_ids)
        
        delta = (A - B) / n_queries
        r1 = base_r1 + delta
        purity = A / (A + B) if (A + B) > 0 else None
        
        results.append({
            'variant': f'K={K}',
            'K': K, 'R1': r1, 'delta_R1': delta,
            'flip_pct': 100.0 * (A + B) / n_queries,
            'A': A, 'B': B, 'A/B': f'{A}/{B}',
            'netflip_1k': 1000 * delta, 'purity': purity,
            'cost': K / 50.0,
        })
    
    return results


# =============================================================================
# LAMBDA SWEEP (Table 9) - GPU Version
# =============================================================================
def run_lambda_sweep_gpu(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    distractor_bank: Optional[torch.Tensor],
    gt_mapping: Dict,
    query_ids: List,
    direction: str = 't2i',
    lambda_values: List[float] = None,
    batch_size: int = 256,
) -> List[Dict]:
    """
    GPU-accelerated Lambda sweep (spectral smoothing strength).
    """
    if lambda_values is None:
        lambda_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    device = query_emb.device
    n_queries = len(query_ids)
    
    # Get base R@1
    base_r1, _ = compute_base_r1(query_emb, gallery_emb, gt_mapping, query_ids)
    
    all_sims = query_emb @ gallery_emb.T
    base_top1 = all_sims.argmax(dim=1).cpu().numpy()
    
    topK = PROD_PARAMS['topK']
    topk_scores, topk_idx = torch.topk(all_sims, k=topK, dim=1)
    cand_embs = gallery_emb[topk_idx]
    
    # Pre-compute features (same for all λ)
    all_rts, all_ccs, all_curv, all_dhc, all_rtps = [], [], [], [], []
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_query = query_emb[start:end]
        batch_cands = cand_embs[start:end]
        
        if distractor_bank is not None:
            rts = compute_rts_batched(batch_query, batch_cands, distractor_bank, M=24)
        else:
            rts = torch.zeros(end - start, topK, device=device)
        all_rts.append(rts.cpu())
        all_ccs.append(compute_ccs_batched(batch_cands).cpu())
        all_curv.append(compute_curv_batched(batch_cands).cpu())
        all_dhc.append(compute_dhc_batched(batch_cands).cpu())
        all_rtps.append(compute_rtps_batched(batch_query, batch_cands).cpu())
    
    all_rts = torch.cat(all_rts, dim=0)
    all_ccs = torch.cat(all_ccs, dim=0)
    all_curv = torch.cat(all_curv, dim=0)
    all_dhc = torch.cat(all_dhc, dim=0)
    all_rtps = torch.cat(all_rtps, dim=0)
    topk_scores_cpu = topk_scores.cpu()
    topk_idx_cpu = topk_idx.cpu()
    
    def zscore(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std
    
    rts_z = zscore(all_rts)
    ccs_z = zscore(all_ccs)
    curv_z = zscore(all_curv)
    dhc_z = zscore(all_dhc)
    rtps_z = zscore(all_rtps)
    base_z = zscore(topk_scores_cpu)
    
    f_boost = (
        PROD_PARAMS['rts_weight'] * rts_z +
        PROD_PARAMS['ccs_weight'] * ccs_z -
        PROD_PARAMS['curv_weight'] * curv_z -
        PROD_PARAMS['dhc_weight'] * dhc_z +
        PROD_PARAMS['rtps_weight'] * rtps_z
    )
    
    results = []
    
    for lam in lambda_values:
        # Compute spectral smoothing with this λ
        all_smoothed = []
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            smoothed = batch_spectral_smooth(topk_scores[start:end], cand_embs[start:end], lam)
            all_smoothed.append(smoothed.cpu())
        all_smoothed = torch.cat(all_smoothed, dim=0)
        spectral_z = zscore(all_smoothed)
        
        fw = PROD_PARAMS['fusion_weight']
        lfa_scores = fw * spectral_z + (1 - fw) * (base_z + PROD_PARAMS['boost_scale'] * f_boost)
        lfa_top1_local = lfa_scores.argmax(dim=1)
        lfa_top1 = topk_idx_cpu[torch.arange(n_queries), lfa_top1_local].numpy()
        
        A, B = compute_flips_from_arrays(base_top1, lfa_top1, gt_mapping, query_ids)
        
        delta = (A - B) / n_queries
        r1 = base_r1 + delta
        purity = A / (A + B) if (A + B) > 0 else None
        
        results.append({
            'variant': f'λ={lam}',
            'lambda': lam, 'R1': r1, 'delta_R1': delta,
            'flip_pct': 100.0 * (A + B) / n_queries,
            'A': A, 'B': B, 'A/B': f'{A}/{B}',
            'netflip_1k': 1000 * delta, 'purity': purity,
        })
    
    return results


# =============================================================================
# QUICK TEST RUNNER
# =============================================================================
def run_quick_test(
    dataset: str = 'flickr30k',
    direction: str = 't2i',
    backbone: str = 'clip',
    embeddings_root: str = None,
    table: str = 'layer',
) -> List[Dict]:
    """
    Quick test runner for GPU-accelerated tables.
    
    Args:
        table: 'layer', 'budget', 'feature', 'k', 'lambda'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data: {dataset}/{direction}/{backbone}")
    query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, gallery_ids = load_data_gpu(
        dataset, direction, backbone, embeddings_root, device
    )
    print(f"  Queries: {len(query_ids)}, Gallery: {len(gallery_ids)}")
    
    if table == 'layer':
        results = run_layer_ablation_gpu(query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, direction)
    elif table == 'budget':
        results = run_budget_sweep_gpu(query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, direction)
    elif table == 'feature':
        results = run_feature_ablation_gpu(query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, direction)
    elif table == 'k':
        results = run_k_sweep_gpu(query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, direction)
    elif table == 'lambda':
        results = run_lambda_sweep_gpu(query_emb, gallery_emb, distractor_bank, gt_mapping, query_ids, direction)
    else:
        raise ValueError(f"Unknown table: {table}")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {table.upper()}")
    print(f"{'='*80}")
    for r in results:
        purity_str = f"{r['purity']:.2f}" if r.get('purity') is not None else "-"
        print(f"  {r['variant']:<30} R@1={r['R1']:.4f} Δ={r['delta_R1']:+.4f} A/B={r['A/B']:>8} NF/1k={r['netflip_1k']:+.1f} Purity={purity_str}")
    
    return results
