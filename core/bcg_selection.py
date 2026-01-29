"""
BCG (Bidirectional Consistency Gating) query selection.

This module implements Stage 1a: Coarse query selection using
Polarity + RSC rank-sum fusion.
"""

import numpy as np
import torch
from typing import Optional
from scipy.stats import rankdata, spearmanr

from .features import compute_polarity_scores, compute_rsc_scores


def select_queries_bcg(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    direction: str,
    budget_frac: float,
    image_bank: Optional[torch.Tensor] = None,
    text_bank: Optional[torch.Tensor] = None,
    mode: str = 'rank_sum_oriented',
    topK: int = 50,
    M_rts: int = 24,
) -> np.ndarray:
    """
    Select queries using BCG (Bidirectional Consistency Gating).
    
    Stage 1a: Coarse selection using Polarity + RSC rank-sum fusion.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        direction: Retrieval direction ('i2t', 't2i', 'a2t', 't2a')
        budget_frac: Budget fraction (e.g., 0.10 for 10%)
        image_bank: Image embeddings for RTS distractor bank (for i2t/a2t)
        text_bank: Text embeddings for RTS distractor bank (for t2i/t2a)
        mode: Selection mode:
            - 'rank_sum_oriented': Rank-sum with label-free sign alignment
            - 'rank_sum': Simple rank-sum
            - 'polarity_only': Only use polarity
            - 'rsc_only': Only use RSC
        topK: Number of top candidates for feature computation
        M_rts: Number of neighbors for RTS computation
        
    Returns:
        selected_mask: Boolean array [N], True for selected queries
    """
    N = queries.shape[0]
    K = int(budget_frac * N)
    K = max(1, min(K, N))  # Ensure at least 1, at most all
    
    # Compute Polarity scores for ALL queries
    polarity_scores = compute_polarity_scores(
        queries=queries,
        gallery=gallery,
        direction=direction,
        topK=topK,
        M_rts=M_rts,
        image_bank=image_bank,
        text_bank=text_bank,
    )
    
    # Compute RSC scores for ALL queries
    rsc_scores = compute_rsc_scores(
        queries=queries,
        gallery=gallery,
        direction=direction,
        topK=topK,
    )
    
    # Apply rank-sum fusion based on mode
    if mode == 'rank_sum_oriented':
        # Rank-sum with label-free sign alignment
        polarity_rank = rankdata(-polarity_scores, method='average')  # Higher polarity = lower rank (better)
        rsc_rank = rankdata(-rsc_scores, method='average')  # Higher RSC = lower rank (better)
        
        # Check correlation to determine if RSC should be inverted
        corr, _ = spearmanr(polarity_rank, rsc_rank)
        
        if corr < 0:
            # Invert RSC if negatively correlated with polarity
            rsc_rank = rankdata(rsc_scores, method='average')  # Flip direction
        
        # Rank-sum score: lower = better (more uncertain, should refine)
        selection_scores = polarity_rank + rsc_rank
        
    elif mode == 'rank_sum':
        # Simple rank-sum without orientation
        polarity_rank = rankdata(-polarity_scores, method='average')
        rsc_rank = rankdata(-rsc_scores, method='average')
        selection_scores = polarity_rank + rsc_rank
        
    elif mode == 'polarity_only':
        # Only use polarity
        selection_scores = rankdata(-polarity_scores, method='average')
        
    elif mode == 'rsc_only':
        # Only use RSC
        selection_scores = rankdata(-rsc_scores, method='average')
        
    else:
        # Default: rank_sum_oriented
        polarity_rank = rankdata(-polarity_scores, method='average')
        rsc_rank = rankdata(-rsc_scores, method='average')
        selection_scores = polarity_rank + rsc_rank
    
    # Select top K queries (lowest selection scores = most uncertain = should refine)
    selected_indices = np.argsort(selection_scores)[:K]
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_indices] = True
    
    return selected_mask
