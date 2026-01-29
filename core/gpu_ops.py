"""
GPU-Accelerated Operations for Control-TTR.

This module provides batched GPU implementations of:
- Spectral smoothing
- Feature computation (RTS, CCS, CURV, DHC, RTPS)
- LFA refinement pipeline
- BCG score computation

Extracted from scripts/random_vs_bcg_selection_gpu.py for standalone use.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from scipy.stats import rankdata


# =============================================================================
# PRODUCTION PARAMETERS
# =============================================================================
PROD_PARAMS = {
    'topK': 50,
    'lambda_smooth': 0.3,
    'boost_scale': 0.35,
    'fusion_weight': 0.5,
    'polarity_threshold': 0.1,
    # Feature weights (production)
    'rts_weight': 0.60,
    'dhc_weight': 0.15,
    'ccs_weight': 0.05,
    'rtps_weight': 0.05,
    'curv_weight': 0.05,
    # Quality gate parameters
    'use_sharpness_gate': True,
    'entropy_min': 0.3,
    'use_consensus_gate': True,
    'min_community_votes': 1,
    'use_ls_margin_gate': True,
    # LS weights
    'ls_alpha': 0.6,
    'ls_beta': 0.2,
    'ls_gamma': 0.15,
    'ls_delta': 0.05,
}


# =============================================================================
# BATCHED SPECTRAL SMOOTHING
# =============================================================================
def batch_spectral_smooth(
    batch_scores: torch.Tensor,      # [B, K] - base scores for top-K candidates
    batch_cand_embs: torch.Tensor,   # [B, K, D] - candidate embeddings
    lambda_smooth: float = 0.3,
) -> torch.Tensor:
    """
    Batched spectral smoothing using (I + λL)^-1 @ s
    
    Args:
        batch_scores: [B, K] base similarity scores
        batch_cand_embs: [B, K, D] candidate embeddings
        lambda_smooth: Laplacian regularization weight
    
    Returns:
        [B, K] smoothed scores
    """
    B, K, D = batch_cand_embs.shape
    device = batch_cand_embs.device
    
    # Build similarity matrix: [B, K, K]
    A = torch.bmm(batch_cand_embs, batch_cand_embs.transpose(1, 2))
    A = torch.clamp(A, min=0)  # ReLU for non-negative adjacency
    
    # Degree matrix diagonal: [B, K]
    D_diag = A.sum(dim=-1).clamp(min=1e-8)
    
    # D^{-1/2}: [B, K]
    D_inv_sqrt = 1.0 / torch.sqrt(D_diag)
    
    # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt_mat = D_inv_sqrt.unsqueeze(-1)  # [B, K, 1]
    normalized_A = D_inv_sqrt_mat * A * D_inv_sqrt.unsqueeze(1)  # [B, K, K]
    
    I = torch.eye(K, device=device).unsqueeze(0).expand(B, -1, -1)  # [B, K, K]
    L_sym = I - normalized_A
    
    # Low-pass filter: (I + λL)^{-1}
    filter_matrix = I + lambda_smooth * L_sym
    
    # Batched matrix inverse
    smooth_filter = torch.linalg.inv(filter_matrix)  # [B, K, K]
    
    # Apply filter: [B, K, K] @ [B, K, 1] -> [B, K, 1] -> [B, K]
    smoothed = torch.bmm(smooth_filter, batch_scores.unsqueeze(-1)).squeeze(-1)
    
    return smoothed


# =============================================================================
# FEATURE COMPUTATIONS (Batched)
# =============================================================================
def compute_rts_batched(
    query_emb: torch.Tensor,         # [B, D]
    cand_embs: torch.Tensor,         # [B, K, D]
    distractor_bank: torch.Tensor,   # [N_dist, D]
    M: int = 24,
    sub_batch: int = 500,
) -> torch.Tensor:
    """
    Batched RTS computation.
    RTS = candidate's score with query - best distractor score
    
    Returns: [B, K] RTS scores (higher = better candidate specificity)
    """
    B, K, D = cand_embs.shape
    device = cand_embs.device
    N_dist = distractor_bank.shape[0]
    
    rts_all = torch.zeros(B, K, device=device)
    
    # Process in sub-batches to avoid OOM
    for start in range(0, B, sub_batch):
        end = min(start + sub_batch, B)
        
        batch_query = query_emb[start:end]  # [b, D]
        batch_cands = cand_embs[start:end]  # [b, K, D]
        
        # Find top-M distractors per query: [b, N_dist]
        query_dist_sims = batch_query @ distractor_bank.T
        _, topM_idx = torch.topk(query_dist_sims, k=min(M, N_dist), dim=1)  # [b, M]
        batch_distractors = distractor_bank[topM_idx]  # [b, M, D]
        
        # Candidate scores with query: [b, K]
        cand_query_scores = torch.bmm(batch_cands, batch_query.unsqueeze(-1)).squeeze(-1)
        
        # Candidate scores with distractors: [b, K, M]
        cand_dist_scores = torch.bmm(batch_cands, batch_distractors.transpose(1, 2))
        cand_dist_best = cand_dist_scores.max(dim=-1)[0]  # [b, K]
        
        # RTS = query_score - best_distractor_score
        rts_all[start:end] = cand_query_scores - cand_dist_best
        
        del query_dist_sims, batch_distractors, cand_dist_scores
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rts_all


def compute_ccs_batched(
    cand_embs: torch.Tensor,  # [B, K, D]
) -> torch.Tensor:
    """
    Caption Cluster Support: mutual similarity among candidates.
    High CCS = candidates form a coherent cluster = more reliable.
    
    Returns: [B, K] CCS scores
    """
    B, K, D = cand_embs.shape
    
    # Pairwise similarity: [B, K, K]
    sim_matrix = torch.bmm(cand_embs, cand_embs.transpose(1, 2))
    
    # CCS for candidate i = mean similarity to other candidates (excluding self)
    mask = ~torch.eye(K, dtype=torch.bool, device=cand_embs.device).unsqueeze(0)
    sim_matrix_masked = sim_matrix * mask.float()
    
    # Mean over other candidates: [B, K]
    ccs = sim_matrix_masked.sum(dim=-1) / (K - 1)
    
    return ccs


def compute_curv_batched(
    cand_embs: torch.Tensor,  # [B, K, D]
) -> torch.Tensor:
    """
    Graph Curvature: local density around each candidate.
    High curvature = isolated candidate = less reliable.
    
    Returns: [B, K] CURV scores (higher = more isolated)
    """
    B, K, D = cand_embs.shape
    
    # Pairwise similarity: [B, K, K]
    sim_matrix = torch.bmm(cand_embs, cand_embs.transpose(1, 2))
    
    # Curvature proxy: mean similarity of each candidate to others
    mask = ~torch.eye(K, dtype=torch.bool, device=cand_embs.device).unsqueeze(0)
    sim_matrix_masked = sim_matrix * mask.float()
    mean_sim = sim_matrix_masked.sum(dim=-1) / (K - 1)
    
    # Invert: curvature = 1 - mean_similarity
    curv = 1.0 - mean_sim
    
    return curv


def compute_dhc_batched(
    cand_embs: torch.Tensor,  # [B, K, D]
    threshold: float = 0.95,
) -> torch.Tensor:
    """
    Duplication/Hallucination Control: penalize near-duplicates.
    
    Returns: [B, K] DHC scores (penalty, higher = more duplicates)
    """
    B, K, D = cand_embs.shape
    
    # Pairwise similarity: [B, K, K]
    sim_matrix = torch.bmm(cand_embs, cand_embs.transpose(1, 2))
    
    # Count near-duplicates (similarity > threshold, excluding self)
    mask = ~torch.eye(K, dtype=torch.bool, device=cand_embs.device).unsqueeze(0)
    near_dup = ((sim_matrix > threshold) & mask).float().sum(dim=-1)  # [B, K]
    
    return near_dup


def compute_rtps_batched(
    query_emb: torch.Tensor,   # [B, D]
    cand_embs: torch.Tensor,   # [B, K, D]
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Round-Trip Paraphrase Stability: sensitivity to query perturbation.
    Low sensitivity = more stable = better.
    
    Returns: [B, K] RTPS scores (stability, higher = more stable)
    """
    B, K, D = cand_embs.shape
    
    # Original scores: [B, K]
    orig_scores = torch.bmm(cand_embs, query_emb.unsqueeze(-1)).squeeze(-1)
    
    # Perturbed query (add noise)
    noise = torch.randn_like(query_emb) * noise_scale
    perturbed_query = F.normalize(query_emb + noise, dim=-1)
    
    # Perturbed scores: [B, K]
    pert_scores = torch.bmm(cand_embs, perturbed_query.unsqueeze(-1)).squeeze(-1)
    
    # Stability = 1 - |original - perturbed| / max_diff
    diff = torch.abs(orig_scores - pert_scores)
    max_diff = diff.max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    rtps = 1.0 - diff / max_diff
    
    return rtps


# =============================================================================
# FULL LFA REFINEMENT (Batched) with Quality Gates
# =============================================================================
def batch_lfa_refinement(
    query_emb: torch.Tensor,         # [B, D]
    gallery_emb: torch.Tensor,       # [N_gal, D]
    distractor_bank: Optional[torch.Tensor],   # [N_dist, D] or None
    params: dict = None,
    direction: str = 't2i',
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Full LFA refinement matching spectral_refinement_baseline with quality gates.
    
    Returns:
        base_top1_idx: [B] base top-1 indices
        refined_top1_idx: [B] refined top-1 indices
        gate_info: dict with polarity, sharpness, consensus, ls_margin pass info
    """
    if params is None:
        params = PROD_PARAMS
    
    B = query_emb.shape[0]
    device = query_emb.device
    topK = params['topK']
    
    # Compute all similarities: [B, N_gal]
    all_sims = query_emb @ gallery_emb.T
    
    # Get top-K: [B, K]
    topk_scores, topk_idx = torch.topk(all_sims, k=topK, dim=1)
    base_top1_idx = topk_idx[:, 0]
    
    # Base margin: [B] (top1 - top2)
    base_margin = topk_scores[:, 0] - topk_scores[:, 1]
    
    # Base entropy: [B] - entropy of softmax(top-K scores)
    probs = F.softmax(topk_scores, dim=1)
    base_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    
    # Gather candidate embeddings: [B, K, D]
    cand_embs = gallery_emb[topk_idx]
    
    # 1. Spectral smoothing: [B, K]
    smoothed_scores = batch_spectral_smooth(
        topk_scores, cand_embs, 
        lambda_smooth=params['lambda_smooth']
    )
    
    # 2. Feature computation
    # RTS: [B, K]
    if distractor_bank is not None:
        rts = compute_rts_batched(query_emb, cand_embs, distractor_bank, M=24)
    else:
        rts = torch.zeros(B, topK, device=device)
    
    # CCS: [B, K]
    ccs = compute_ccs_batched(cand_embs)
    
    # CURV: [B, K]
    curv = compute_curv_batched(cand_embs)
    
    # DHC: [B, K]
    dhc = compute_dhc_batched(cand_embs)
    
    # RTPS: [B, K]
    rtps = compute_rtps_batched(query_emb, cand_embs)
    
    # 3. Z-score normalization (per-query)
    def zscore(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std
    
    rts_z = zscore(rts)
    ccs_z = zscore(ccs)
    curv_z = zscore(curv)
    dhc_z = zscore(dhc)
    rtps_z = zscore(rtps)
    base_z = zscore(topk_scores)
    spectral_z = zscore(smoothed_scores)
    
    # 4. Weighted feature combination
    f_boost = (
        params['rts_weight'] * rts_z +
        params['ccs_weight'] * ccs_z -
        params['curv_weight'] * curv_z -
        params['dhc_weight'] * dhc_z +
        params['rtps_weight'] * rtps_z
    )
    
    # 5. Score fusion
    fw = params['fusion_weight']
    boost_scale = params['boost_scale']
    lfa_scores = fw * spectral_z + (1 - fw) * (base_z + boost_scale * f_boost)
    
    # Get refined top-1
    refined_top1_local = lfa_scores.argmax(dim=1)  # [B] indices in top-K
    refined_top1_idx = topk_idx[torch.arange(B, device=device), refined_top1_local]
    
    # 6. Quality gates (Stage 1b)
    # Polarity veto: (CCS + CURV - RTS) / 3 > threshold
    polarity = (ccs_z[:, 0] + curv_z[:, 0] - rts_z[:, 0]) / 3.0
    pass_polarity = polarity > params['polarity_threshold']
    
    # Sharpness gate: low entropy = confident
    entropy_threshold = params.get('entropy_min', 0.3)
    pass_sharpness = base_entropy < entropy_threshold * np.log(topK)
    
    # LS margin gate: refined margin > threshold
    lfa_margin = lfa_scores[:, 0] - lfa_scores[:, 1]
    pass_ls_margin = lfa_margin > 0.01
    
    # Consensus gate: community vote
    # Simplified: check if top-1 is supported by multiple candidates
    pass_consensus = ccs[:, 0] > 0.5
    
    # All gates pass
    all_gates_pass = pass_polarity & pass_sharpness & pass_ls_margin & pass_consensus
    
    gate_info = {
        'pass_polarity': pass_polarity.cpu().numpy(),
        'pass_sharpness': pass_sharpness.cpu().numpy(),
        'pass_ls_margin': pass_ls_margin.cpu().numpy(),
        'pass_consensus': pass_consensus.cpu().numpy(),
        'all_gates_pass': all_gates_pass.cpu().numpy(),
        'polarity': polarity.cpu().numpy(),
        'entropy': base_entropy.cpu().numpy(),
    }
    
    return base_top1_idx, refined_top1_idx, gate_info


# =============================================================================
# BCG SCORE COMPUTATION
# =============================================================================
def compute_bcg_scores(
    query_emb: torch.Tensor,         # [N, D]
    gallery_emb: torch.Tensor,       # [M, D]
    distractor_bank: Optional[torch.Tensor],   # [N_dist, D] or None
    margin_scores: np.ndarray,       # [N] pre-computed margins
    batch_size: int = 500,
) -> np.ndarray:
    """
    Compute BCG selection scores for all queries.
    
    BCG = Polarity + RSC (rank-sum fusion)
    
    Memory-efficient: computes RSC in batches to avoid OOM on large datasets.
    
    Returns:
        bcg_scores: [N] BCG scores (higher = more uncertain = should refine)
    """
    N = query_emb.shape[0]
    device = query_emb.device
    topK = 50
    
    # Compute RSC in batches (memory-efficient)
    rsc_scores = np.zeros(N)
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_query = query_emb[start:end]  # [B, D]
        
        # Forward retrieval: query -> gallery
        batch_sims = batch_query @ gallery_emb.T  # [B, M]
        forward_top1_idx = batch_sims.argmax(dim=1)  # [B]
        forward_top1_emb = gallery_emb[forward_top1_idx]  # [B, D]
        
        # Backward retrieval: gallery[forward_top1] -> all queries
        backward_sims = forward_top1_emb @ query_emb.T  # [B, N]
        
        # RSC = rank of original query in backward retrieval
        # For each query i in batch, find rank of (start+i) in backward_sims[i]
        batch_size_actual = end - start
        for i in range(batch_size_actual):
            query_idx = start + i
            row_sims = backward_sims[i]  # [N]
            # Count how many queries have higher similarity than the original query
            original_sim = row_sims[query_idx].item()
            rank = (row_sims > original_sim).sum().item()  # 0-based rank (lower = better)
            rsc_scores[query_idx] = rank
        
        del batch_sims, backward_sims
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Polarity = margin-based uncertainty
    # Lower margin = more uncertain = should refine
    polarity_scores = -margin_scores
    
    # CURV: compute per-query curvature (batched)
    curv_scores = np.zeros(N)
    all_sims = None  # Compute on demand
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_query = query_emb[start:end]
        batch_sims = batch_query @ gallery_emb.T  # [B, M]
        topk_scores, topk_idx = torch.topk(batch_sims, k=topK, dim=1)
        cand_embs = gallery_emb[topk_idx]  # [B, K, D]
        curv = compute_curv_batched(cand_embs)  # [B, K]
        curv_scores[start:end] = curv[:, 0].cpu().numpy()  # Top-1 curvature
        del batch_sims, cand_embs
    
    # RTS: compute if distractor bank available
    rts_scores = np.zeros(N)
    if distractor_bank is not None:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_query = query_emb[start:end]
            batch_sims = batch_query @ gallery_emb.T
            topk_scores, topk_idx = torch.topk(batch_sims, k=topK, dim=1)
            cand_embs = gallery_emb[topk_idx]
            rts = compute_rts_batched(batch_query, cand_embs, distractor_bank, M=24)
            rts_scores[start:end] = rts[:, 0].cpu().numpy()
            del batch_sims, cand_embs
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Rank-sum fusion (Borda count)
    # Higher polarity (lower margin) = should refine
    # Higher CURV = should refine
    # Lower RTS = should refine (invert)
    # Higher RSC rank = less consistent = should refine
    
    polarity_rank = rankdata(polarity_scores)  # Higher = more uncertain
    curv_rank = rankdata(curv_scores)          # Higher = more uncertain
    rts_rank = rankdata(-rts_scores)           # Lower RTS = higher rank = more uncertain
    rsc_rank = rankdata(rsc_scores)            # Higher RSC = less consistent = should refine
    
    # BCG = sum of ranks
    bcg_scores = polarity_rank + curv_rank + rts_rank + rsc_rank
    
    return bcg_scores


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def compute_flips_from_arrays(
    base_top1: np.ndarray, 
    refined_top1: np.ndarray,
    gt_mapping: Dict, 
    query_ids: list, 
    mask: np.ndarray = None
) -> Tuple[int, int]:
    """Compute A/B flips from numpy arrays."""
    A, B = 0, 0
    n = len(query_ids)
    
    for i in range(n):
        if mask is not None and not mask[i]:
            continue
        
        qid = str(query_ids[i])
        gt_indices = set(gt_mapping.get(qid, []))
        if not gt_indices:
            continue
        
        base_correct = int(base_top1[i]) in gt_indices
        refined_correct = int(refined_top1[i]) in gt_indices
        
        if base_top1[i] != refined_top1[i]:
            if not base_correct and refined_correct:
                A += 1
            elif base_correct and not refined_correct:
                B += 1
    
    return A, B
