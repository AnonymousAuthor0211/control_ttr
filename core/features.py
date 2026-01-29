"""
Feature computation for Control-TTR.

This module implements the geometric features used in the paper:
- RTS (Round-Trip Specificity)
- CCS (Caption Cluster Support)
- CURV (Graph Curvature)
- RTPS (Round-Trip Paraphrase Stability)
- DHC (Duplication/Hallucination Control)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.stats import rankdata


def robust_zscore(x: np.ndarray, eps: float = 1e-6, max_z: float = 10.0) -> np.ndarray:
    """
    Robust z-score using MAD (Median Absolute Deviation).
    
    Returns zeros when there's no meaningful spread (no signal).
    
    Args:
        x: Input array
        eps: Minimum spread threshold
        max_z: Maximum absolute z-score value (clip extreme values)
        
    Returns:
        Z-scored array
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return np.zeros_like(x)
    
    # Check for NaN/inf
    if not np.isfinite(x).any():
        return np.zeros_like(x)
    
    # Check if constant - if range is tiny, return zeros (no signal)
    x_range = np.nanmax(x) - np.nanmin(x)
    if x_range < 1e-6 or np.allclose(x, x[0], atol=1e-6):
        return np.zeros_like(x)
    
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    
    # If MAD is too small, fall back to std
    if mad < eps:
        std = np.nanstd(x)
        if std > eps and x_range > 1e-6:
            z = (x - np.nanmean(x)) / (std + eps)
        else:
            # No meaningful spread - return zeros (no signal)
            return np.zeros_like(x)
    else:
        # Use 1.4826 factor to make MAD consistent with std for Gaussian
        z = (x - med) / (1.4826 * mad + eps)
    
    # Clip extreme z-scores to prevent huge values from dominating
    z = np.clip(z, -max_z, max_z)
    return z


def compute_rts(
    query: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    modal_bank: torch.Tensor,
    M: int = 24
) -> float:
    """
    Compute Round-Trip Specificity (RTS) for a query-candidate pair.
    
    RTS measures how specific a candidate is to the query vs. a background
    distribution (modal_bank). Higher RTS = more confident match.
    
    Args:
        query: Query embedding [D]
        candidate_embeddings: Candidate embeddings [K, D] (top-K candidates)
        modal_bank: Background embeddings for distractor bank [N_bank, D]
        M: Number of neighbors to consider
        
    Returns:
        RTS score (margin of top-1 candidate)
    """
    if len(candidate_embeddings) == 0:
        return 0.0
    
    # Normalize
    query_norm = F.normalize(query, dim=0)
    candidates_norm = F.normalize(candidate_embeddings, dim=1)
    modal_bank_norm = F.normalize(modal_bank, dim=1)
    
    # Find M nearest neighbors in modal bank (distractors)
    modal_sims = modal_bank_norm @ query_norm  # [N_bank]
    topM = min(M, len(modal_bank_norm))
    _, topM_idx = torch.topk(modal_sims, k=topM, dim=0)
    visual_neighbors = modal_bank_norm[topM_idx]  # [M, D]
    
    # Specificity margin: query_score - best_distractor_score
    query_scores = candidates_norm @ query_norm  # [K]
    distractor_scores = candidates_norm @ visual_neighbors.T  # [K, M]
    distractor_bests = distractor_scores.max(dim=1)[0]  # [K]
    rts_margins = query_scores - distractor_bests  # [K]
    
    # RTS for this query = margin of top-1 candidate
    return rts_margins[0].item()


def compute_ccs(candidate_embeddings: torch.Tensor) -> float:
    """
    Compute Caption Cluster Support (CCS).
    
    CCS measures mutual support/cohesion among top candidates.
    Higher CCS = more cohesive cluster (potentially ambiguous).
    
    Args:
        candidate_embeddings: Candidate embeddings [K, D]
        
    Returns:
        CCS score (average pairwise similarity, excluding self)
    """
    K = len(candidate_embeddings)
    if K < 3:
        return 0.5  # Default neutral value
    
    # Normalize
    candidates_norm = F.normalize(candidate_embeddings, dim=1)
    
    # Compute pairwise similarities
    cand_sim = candidates_norm @ candidates_norm.T  # [K, K]
    
    # Average similarity excluding diagonal (self-similarity)
    mask = torch.ones(K, K, device=cand_sim.device) - torch.eye(K, device=cand_sim.device)
    ccs = (cand_sim * mask).sum().item() / max(K * (K - 1), 1)
    
    return ccs


def compute_curv(
    candidate_embeddings: torch.Tensor,
    scores: torch.Tensor
) -> float:
    """
    Compute Graph Curvature (CURV) using Laplacian-based measure.
    
    CURV measures how much the top candidate's score disagrees with its
    neighbors in the graph. Higher CURV = more uncertain/ambiguous.
    
    Args:
        candidate_embeddings: Candidate embeddings [K, D]
        scores: Base scores for candidates [K]
        
    Returns:
        CURV score (curvature of top-1 candidate)
    """
    K = len(candidate_embeddings)
    if K < 3:
        return 0.0
    
    # Normalize
    candidates_norm = F.normalize(candidate_embeddings, dim=1)
    
    # Build adjacency matrix (pairwise similarities)
    cand_sim = candidates_norm @ candidates_norm.T  # [K, K]
    A = torch.clamp(cand_sim, min=0)
    
    # Compute degree
    D = A.sum(dim=1)
    D_inv = 1.0 / torch.clamp(D, min=1e-8)
    
    # Normalize scores
    scores_normalized = scores / scores.max()
    
    # Curvature: score - smoothed score (disagreement with neighbors)
    curvature = scores_normalized - D_inv * (A @ scores_normalized)
    
    # Return curvature of top-1 candidate
    return curvature[0].item()


def compute_rtps(
    query: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    n_drops: int = 5
) -> float:
    """
    Compute Round-Trip Paraphrase Stability (RTPS).
    
    RTPS measures sensitivity to word-drop perturbations.
    Higher RTPS = more stable/confident match.
    
    Note: This is a simplified version. Full implementation would
    require text tokenization and word-drop simulation.
    
    Args:
        query: Query embedding [D]
        candidate_embeddings: Candidate embeddings [K, D]
        n_drops: Number of token drops to simulate (not used in simplified version)
        
    Returns:
        RTPS score (approximated as margin for now)
    """
    if len(candidate_embeddings) < 2:
        return 0.0
    
    # Normalize
    query_norm = F.normalize(query, dim=0)
    candidates_norm = F.normalize(candidate_embeddings, dim=1)
    
    # Compute similarities
    sims = candidates_norm @ query_norm  # [K]
    
    # RTPS approximated as margin (top1 - top2)
    # Higher margin = more stable
    top2_vals, _ = torch.topk(sims, k=min(2, len(sims)), dim=0)
    if len(top2_vals) >= 2:
        margin = (top2_vals[0] - top2_vals[1]).item()
        return margin
    else:
        return 0.0


def compute_dhc(candidate_embeddings: torch.Tensor) -> float:
    """
    Compute Duplication/Hallucination Control (DHC).
    
    DHC penalizes near-duplicate candidates (high similarity clusters).
    Higher DHC = more diverse candidates (less duplication risk).
    
    Args:
        candidate_embeddings: Candidate embeddings [K, D]
        
    Returns:
        DHC score (inverse of average similarity)
    """
    K = len(candidate_embeddings)
    if K < 2:
        return 1.0
    
    # Normalize
    candidates_norm = F.normalize(candidate_embeddings, dim=1)
    
    # Compute pairwise similarities
    cand_sim = candidates_norm @ candidates_norm.T  # [K, K]
    
    # Average similarity (excluding diagonal)
    mask = torch.ones(K, K, device=cand_sim.device) - torch.eye(K, device=cand_sim.device)
    avg_sim = (cand_sim * mask).sum().item() / max(K * (K - 1), 1)
    
    # DHC = inverse of average similarity (higher diversity = higher DHC)
    dhc = 1.0 / max(avg_sim, 1e-6)
    
    return dhc


def compute_polarity_scores(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    direction: str,
    topK: int = 50,
    M_rts: int = 24,
    image_bank: Optional[torch.Tensor] = None,
    text_bank: Optional[torch.Tensor] = None,
    return_components: bool = False,
) -> np.ndarray:
    """
    Compute CORE Polarity scores (RTS + CURV + Margin) via rank aggregation.
    
    CORE Polarity = rank(-RTS) + rank(CURV) + rank(-Margin)
    
    Semantics:
        - High RTS = confident → DON'T refine (negative contribution)
        - High CURV = uncertain → DO refine (positive contribution)
        - High Margin = confident → DON'T refine (negative contribution)
    
    Higher polarity = more likely to need refinement.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        direction: 'i2t', 't2i', 'a2t', 't2a'
        topK: Number of top candidates for feature computation
        M_rts: Number of neighbors for RTS computation
        image_bank: Image embeddings for RTS distractor bank (for i2t/a2t)
        text_bank: Text embeddings for RTS distractor bank (for t2i/t2a)
        return_components: If True, also return individual RTS, CURV, Margin scores
        
    Returns:
        If return_components=False:
            polarity: CORE polarity scores [N] normalized to [0, 1]
        If return_components=True:
            (polarity, rts_scores, curv_scores, margin_scores): Tuple of arrays
    """
    device = queries.device
    N = queries.shape[0]
    
    queries_norm = F.normalize(queries, dim=1)
    gallery_norm = F.normalize(gallery, dim=1)
    
    rts_scores = np.zeros(N)
    curv_scores = np.zeros(N)
    margin_scores = np.zeros(N)
    
    # Determine which bank to use for RTS
    use_rts = False
    modal_bank = None
    if direction in ['i2t', 'a2t'] and image_bank is not None:
        modal_bank = F.normalize(image_bank.to(device), dim=1)
        use_rts = True
    elif direction in ['t2i', 't2a'] and text_bank is not None:
        modal_bank = F.normalize(text_bank.to(device), dim=1)
        use_rts = True
    
    for i in range(N):
        q = queries_norm[i:i+1]
        sims = (q @ gallery_norm.T).squeeze(0)
        
        topk_vals, topk_idx = torch.topk(sims, k=min(topK, len(gallery_norm)), dim=0)
        K_actual = len(topk_vals)
        candidate_emb = gallery_norm[topk_idx]
        
        # RTS (Round-Trip Specificity)
        if use_rts and K_actual >= 1:
            query_emb = q.squeeze(0)  # [D]
            rts_scores[i] = compute_rts(query_emb, candidate_emb, modal_bank, M_rts)
        
        # CURV (Laplacian Curvature)
        if K_actual >= 3:
            curv_scores[i] = compute_curv(candidate_emb, topk_vals)
        
        # Margin (top1 - top2)
        if K_actual >= 2:
            margin_scores[i] = (topk_vals[0] - topk_vals[1]).item()
    
    # Rank aggregation
    # polarity = rank(-RTS) + rank(CURV) + rank(-Margin)
    rts_rank = rankdata(-rts_scores)       # Descending: high RTS → low rank
    curv_rank = rankdata(curv_scores)      # Ascending: high CURV → high rank
    margin_rank = rankdata(-margin_scores) # Descending: high Margin → low rank
    
    # Borda-style sum of ranks
    polarity = rts_rank + curv_rank + margin_rank
    
    # Normalize to [0, 1] for interpretability
    polarity_min = polarity.min()
    polarity_max = polarity.max()
    if polarity_max - polarity_min > 1e-8:
        polarity = (polarity - polarity_min) / (polarity_max - polarity_min)
    
    if return_components:
        return polarity, rts_scores, curv_scores, margin_scores
    return polarity


def compute_margin_scores(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    topK: int = 50,
) -> np.ndarray:
    """
    Compute per-query margin scores (top1 - top2) for the base retriever.
    
    Used for margin-based gating.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        topK: Number of top candidates to consider
        
    Returns:
        Margin scores [N]
    """
    N = queries.shape[0]
    queries_norm = F.normalize(queries, dim=1)
    gallery_norm = F.normalize(gallery, dim=1)
    
    margin_scores = np.zeros(N)
    
    for i in range(N):
        q = queries_norm[i:i+1]
        sims = (q @ gallery_norm.T).squeeze(0)
        topk_vals, _ = torch.topk(sims, k=min(topK, len(gallery_norm)), dim=0)
        
        if len(topk_vals) >= 2:
            margin_scores[i] = (topk_vals[0] - topk_vals[1]).item()
    
    return margin_scores


def compute_rsc_scores(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    direction: str,
    topK: int = 50,
) -> np.ndarray:
    """
    Compute RSC (Relative Self-Consistency) scores.
    
    RSC(q) = <q, b(q)> / max_{q' ≠ q} <q, q'>
    
    where b(q) is the backward-retrieved embedding from round-trip:
        1. Forward: q → top-1 gallery item g
        2. Backward: g → top-1 query from query bank
        3. b(q) = embedding of that backward-retrieved query
    
    Higher RSC = more self-consistent round-trip = reliable match.
    Lower RSC = hub interference = may benefit from refinement.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        direction: 'i2t', 't2i', 'a2t', 't2a'
        topK: Number of top candidates (not used, kept for API compat)
        
    Returns:
        rsc_scores: RSC scores [N] in range [0, ~1], higher = more stable
    """
    device = queries.device
    N = queries.shape[0]
    
    queries_norm = F.normalize(queries, dim=1)
    gallery_norm = F.normalize(gallery, dim=1)
    
    # For RSC, the query bank for backward retrieval is the queries themselves
    query_bank_norm = queries_norm  # [N, D]
    
    # Compute full similarity matrix once for efficiency
    similarity = queries_norm @ gallery_norm.T  # [N, M]
    
    # Get forward top-1 for all queries
    forward_top1_idx = torch.argmax(similarity, dim=1)  # [N]
    
    # Get embeddings of forward top-1 gallery items
    forward_top1_emb = gallery_norm[forward_top1_idx]  # [N, D]
    
    # Backward retrieval: forward_top1 → query_bank
    backward_sims = forward_top1_emb @ query_bank_norm.T  # [N, N_bank]
    backward_top1_idx = torch.argmax(backward_sims, dim=1)  # [N]
    
    # Get embeddings of backward-retrieved queries
    backward_top1_emb = query_bank_norm[backward_top1_idx]  # [N, D]
    
    # Compute RSC: <q, b(q)> / max_{q' ≠ q} <q, q'>
    # Self-similarity: original query vs backward-retrieved query
    self_sim = (queries_norm * backward_top1_emb).sum(dim=1)  # [N]
    
    # Max similarity: original query vs all queries, EXCLUDING SELF
    all_query_sims = queries_norm @ query_bank_norm.T  # [N, N_bank]
    
    # Mask out self-similarity: set sim to -inf at self position
    mask = torch.eye(N, device=device, dtype=torch.bool)
    all_query_sims.masked_fill_(mask, -float('inf'))
    
    max_sim = all_query_sims.max(dim=1).values  # [N] max over non-self entries
    
    # RSC = self_sim / max_sim
    max_sim = torch.clamp(max_sim, min=1e-8)
    rsc_scores_tensor = self_sim / max_sim
    
    # Clamp to [0, 1] for stability (can exceed 1 if backward is self)
    rsc_scores = torch.clamp(rsc_scores_tensor, 0.0, 1.0).cpu().numpy()
    
    return rsc_scores
