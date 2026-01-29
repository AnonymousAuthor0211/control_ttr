"""
Standalone spectral_refinement_baseline function.
Extracted from uamr/models/training_free_baselines.py
NO DEPENDENCIES on original codebase.
"""

import os
import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np
import json
import pickle
import time
import random
from collections import defaultdict
from contextlib import contextmanager
from scipy.linalg import eigh
from scipy.stats import norm, rankdata
from tqdm import tqdm

# Disable tqdm globally
os.environ['TQDM_DISABLE'] = '1'

logger = logging.getLogger(__name__)

# Constants
MAD2STD = 1.4826  # Conversion factor from MAD to standard deviation
EPS = 1e-8        # Small epsilon for numerical stability

def start_cuda_timer() -> float:
    """
    Start a CUDA-aware timer.
    
    Returns the start time after CUDA synchronization.
    Use with end_cuda_timer() for accurate GPU timing.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()



def end_cuda_timer(start_time: float) -> float:
    """
    End a CUDA-aware timer and return elapsed time.
    
    Args:
        start_time: Value returned by start_cuda_timer()
        
    Returns:
        Elapsed time in seconds (float)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start_time

# Constants from vmf_deterministic.py
MAD2STD = 1.4826  # Conversion factor from MAD to standard deviation
EPS = 1e-8        # Small epsilon for numerical stability



def cuda_timer():
    """
    Context manager for accurate GPU timing.
    
    Properly handles CUDA's asynchronous execution by:
    1. Synchronizing before start to flush pending ops
    2. Using time.perf_counter() for high precision
    3. Synchronizing before end to wait for all GPU ops
    
    Usage:
        with cuda_timer() as timer:
            # GPU operations here
            ...
        elapsed = timer.elapsed  # in seconds
    
    Returns a timer object with `.elapsed` attribute (seconds).
    """
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
    
    timer = Timer()
    
    # Sync before starting to flush any pending ops
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    try:
        yield timer
    finally:
        # Sync before measuring to wait for GPU ops
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.elapsed = time.perf_counter() - start



# compute_core_polarity_scores
def compute_core_polarity_scores(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    query_ids: List[str],
    direction: str,
    topK: int = 50,
    M_rts: int = 24,
    use_rts: bool = True,
    use_curv: bool = True,
    image_bank: Optional[torch.Tensor] = None,
    text_bank: Optional[torch.Tensor] = None,
    verbose: bool = True,
    aggregation: str = 'rank',
    precomputed_scores: Optional[torch.Tensor] = None
) -> Dict[str, np.ndarray]:
    """
    UNIFIED CORE Polarity Score Computation.
    
    Computes CORE polarity from RTS, CURV, and Margin features.
    
    AGGREGATION METHODS:
        'rank' (DEFAULT): Borda-style rank aggregation. More principled for Top-K selection,
                          robust across direction changes (T2I vs I2T) and dataset shifts.
                          Composes cleanly with rank_sum_oriented BCG mode.
                          polarity = rank(-RTS) + rank(CURV) + rank(-Margin)
                          
        'zscore': Robust z-score normalization. Preserves magnitude information.
                  Useful for threshold-based gating (polarity > τ) or ablation studies.
                  polarity = -RTS_z + CURV_z - Margin_z
    
    SEMANTIC INTERPRETATION:
        - High RTS = confident → don't refine (negative contribution)
        - High CURV = uncertain → refine (positive contribution)  
        - High Margin = confident → don't refine (negative contribution)
        
    This function should be called by ALL ablations to ensure consistency.
    
    GEOMETRY-CONSISTENT MODE:
        When precomputed_scores is provided, features are computed on those scores
        instead of raw CLIP similarity. This enables computing polarity on a base
        method's geometry (NNN, QB-Norm, etc.) for BCG with geometry-consistent polarity.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        query_ids: Query identifiers
        direction: 'i2t', 't2i', 'a2t', 't2a'
        topK: Number of top candidates for spectral features
        M_rts: Number of neighbors for RTS computation
        use_rts: Whether to compute RTS
        use_curv: Whether to compute CURV
        image_bank: Image embeddings for RTS (i2t/a2t)
        text_bank: Text embeddings for RTS (t2i)
        verbose: Whether to log progress
        aggregation: 'rank' (default, Borda-style) or 'zscore' (robust z-normalization)
        precomputed_scores: Optional [N, M] score matrix to use instead of raw CLIP.
                           If provided, features are computed on this geometry.
        
    Returns:
        Dict with:
            - 'polarity': Final CORE polarity scores [N]
            - 'rts_z': Z-scored RTS [N] (computed for diagnostics even if aggregation='rank')
            - 'curv_z': Z-scored CURV [N]
            - 'margin_z': Z-scored Margin [N]
            - 'rts_raw': Raw RTS scores [N]
            - 'curv_raw': Raw CURV scores [N]
            - 'margin_raw': Raw Margin scores [N]
            - 'aggregation': Which method was used ('rank' or 'zscore')
            - 'geometry_mode': 'raw_clip' or 'precomputed' (for diagnostics)
    """
    import numpy as np
    from tqdm import tqdm
    from scipy.stats import rankdata
    
    device = queries.device
    total_queries = len(query_ids)
    
    # Determine geometry mode
    use_precomputed = precomputed_scores is not None
    geometry_mode = 'precomputed' if use_precomputed else 'raw_clip'
    
    if use_precomputed and verbose:
        logger.info(f"📊 CORE Polarity using GEOMETRY-CONSISTENT mode (precomputed scores)")
    
    # Storage for raw feature scores
    rts_scores = np.zeros(total_queries, dtype=np.float64)
    curv_scores = np.zeros(total_queries, dtype=np.float64)
    margin_scores = np.zeros(total_queries, dtype=np.float64)
    
    gallery_gpu = gallery.to(device)
    if use_precomputed:
        precomputed_scores = precomputed_scores.to(device)
    
    iterator = tqdm(range(total_queries), desc="Computing CORE features") if verbose else range(total_queries)
    
    for i in iterator:
        query = queries[i]
        # Use precomputed scores if provided, otherwise compute raw CLIP
        if use_precomputed:
            s_clip = precomputed_scores[i].squeeze()
        else:
            s_clip = (query.float() @ gallery_gpu.float().T).squeeze()
        
        K_actual = min(topK, s_clip.size(0))
        topk_vals, topk_idx = torch.topk(s_clip, k=K_actual, dim=0)
        candidate_embeddings = gallery_gpu[topk_idx]
        
        # RTS (Round-Trip Specificity) - specificity margin vs distractors
        if use_rts and image_bank is not None and direction in ['i2t', 'a2t']:
            modal_bank = image_bank.to(device)
            query_emb = query.float().to(device)
            
            # Find M visually similar images from image_bank
            img_sims = modal_bank @ query_emb
            M_actual = min(M_rts + 1, len(modal_bank))
            _, topM_idx = torch.topk(img_sims, k=M_actual, dim=0)
            
            # Skip self-match if present
            if len(topM_idx) > 1 and img_sims[topM_idx[0]] > 0.999:
                topM_idx = topM_idx[1:]
            else:
                topM_idx = topM_idx[:M_rts]
            
            visual_neighbors = modal_bank[topM_idx]
            
            # Specificity margin: query_score - best_distractor_score
            query_scores = candidate_embeddings @ query_emb
            distractor_scores = candidate_embeddings @ visual_neighbors.T
            distractor_bests = distractor_scores.max(dim=1)[0]
            rts_margins = query_scores - distractor_bests
            
            rts_scores[i] = rts_margins[0].item()
        elif use_rts and text_bank is not None and direction == 't2i':
            txt_bank = text_bank.to(device)
            query_emb = query.float().to(device)
            
            txt_sims = txt_bank @ query_emb
            M_actual = min(M_rts + 1, len(txt_bank))
            _, topM_idx = torch.topk(txt_sims, k=M_actual, dim=0)
            
            if len(topM_idx) > 1 and txt_sims[topM_idx[0]] > 0.999:
                topM_idx = topM_idx[1:]
            else:
                topM_idx = topM_idx[:M_rts]
            
            text_neighbors = txt_bank[topM_idx]
            
            query_scores = candidate_embeddings @ query_emb
            distractor_scores = candidate_embeddings @ text_neighbors.T
            distractor_bests = distractor_scores.max(dim=1)[0]
            rts_margins = query_scores - distractor_bests
            
            rts_scores[i] = rts_margins[0].item()
        else:
            rts_scores[i] = 0.0
        
        # CURV (Laplacian Curvature)
        if use_curv and K_actual >= 3:
            cand_sim = candidate_embeddings @ candidate_embeddings.T
            A = torch.clamp(cand_sim, min=0)
            D = A.sum(dim=1)
            D_inv = 1.0 / torch.clamp(D, min=1e-8)
            scores_normalized = topk_vals / topk_vals.max()
            curvature = scores_normalized - D_inv * (A @ scores_normalized)
            curv_scores[i] = curvature[0].item()
        else:
            curv_scores[i] = 0.0
        
        # Margin (top1 - top2)
        if K_actual >= 2:
            margin_scores[i] = (topk_vals[0] - topk_vals[1]).item()
        else:
            margin_scores[i] = 0.0
    
    # =========================================================================
    # AGGREGATION: Combine RTS, CURV, Margin into final polarity score
    # =========================================================================
    
    # Always compute robust z-scores for diagnostics and backward compatibility
    def robust_zscore(x):
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        if mad < 1e-8:
            return np.zeros_like(x)
        return (x - median) / (mad * 1.4826)
    
    rts_z = robust_zscore(rts_scores)
    curv_z = robust_zscore(curv_scores)
    margin_z = robust_zscore(margin_scores)
    
    if aggregation == 'rank':
        # =================================================================
        # RANK AGGREGATION (DEFAULT): Borda-style rank fusion
        # =================================================================
        # More principled for Top-K selection (ordinal).
        # Robust across direction changes (T2I vs I2T) and dataset shifts.
        # Composes cleanly with rank_sum_oriented BCG mode.
        #
        # Semantics:
        #   - rank(-RTS): High RTS → low rank → don't refine
        #   - rank(CURV): High CURV → high rank → refine  
        #   - rank(-Margin): High Margin → low rank → don't refine
        #
        # Higher polarity = higher sum of ranks = more likely to need refinement
        # =================================================================
        
        rts_rank = rankdata(-rts_scores)       # Descending: high RTS → low rank
        curv_rank = rankdata(curv_scores)      # Ascending: high CURV → high rank
        margin_rank = rankdata(-margin_scores) # Descending: high Margin → low rank
        
        # Borda-style sum of ranks
        polarity_scores = rts_rank + curv_rank + margin_rank
        
        # Normalize to [0, 1] for interpretability (optional but helpful for logging)
        polarity_min = polarity_scores.min()
        polarity_max = polarity_scores.max()
        if polarity_max - polarity_min > 1e-8:
            polarity_scores = (polarity_scores - polarity_min) / (polarity_max - polarity_min)
        
        if verbose:
            logger.info(f"📊 CORE Polarity (RANK aggregation): range=[{polarity_scores.min():.3f}, {polarity_scores.max():.3f}], "
                       f"mean={polarity_scores.mean():.4f}, std={polarity_scores.std():.4f}")
            
    elif aggregation == 'zscore':
        # =================================================================
        # Z-SCORE AGGREGATION (ABLATION/BACKWARD COMPAT)
        # =================================================================
        # Preserves magnitude information. Useful for:
        #   - Threshold-based gating (polarity > τ)
        #   - Ablation studies comparing rank vs z-score
        #   - Reviewer questions: "why discard magnitude?"
        #
        # Formula: polarity = -RTS_z + CURV_z - Margin_z
        # =================================================================
        
        polarity_scores = -rts_z + curv_z - margin_z
        
        if verbose:
            logger.info(f"📊 CORE Polarity (ZSCORE aggregation): range=[{polarity_scores.min():.2f}, {polarity_scores.max():.2f}], "
                       f"mean={polarity_scores.mean():.4f}, std={polarity_scores.std():.4f}")
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}. Use 'rank' or 'zscore'.")
    
    return {
        'polarity': polarity_scores,
        'rts_z': rts_z,
        'curv_z': curv_z,
        'margin_z': margin_z,
        'rts_raw': rts_scores,
        'curv_raw': curv_scores,
        'margin_raw': margin_scores,
        'aggregation': aggregation,
        'geometry_mode': geometry_mode
    }



# fast_cycle_consistency_for_candidates
def fast_cycle_consistency_for_candidates(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    stage1_candidates: np.ndarray,
    direction: str,
    image_bank: Optional[torch.Tensor] = None,
    text_bank: Optional[torch.Tensor] = None,
    use_hard_cc: bool = False,  # Changed default to False (soft RSC)
    verbose: bool = False
) -> np.ndarray:
    """
    Efficient Relative Self-Consistency (RSC) computation for Stage 1 candidates only.
    
    Computes SOFT round-trip consistency:
        RSC(q) = <q, b(q)> / max_{q' ≠ q} <q, q'>
    where b(q) is the backward-retrieved embedding from round-trip.
    
    This measures: "How similar is the backward-retrieved query to me,
    relative to my best NON-SELF match in the query bank?"
    
    This is NOT "adding a baseline" - it measures LOCAL ROUND-TRIP SELF-CONSISTENCY:
    whether a query can be recovered from its retrieved neighbor under round-trip.
    
    Key property: RSC is symmetric under swapping X ↔ Y (direction-invariant definition,
    though correlation with A/B may differ by direction).
    
    Note: Self is excluded from denominator max to avoid degeneracy when query_bank 
    contains the query itself (which would make max_sim ≈ 1 always).
    
    Args:
        queries: Full query bank [N, D]
        gallery: Gallery embeddings [G, D]
        stage1_candidates: Global indices of M candidates to evaluate [M]
        direction: 'i2t', 't2i', 'a2t', 't2a'
        image_bank: Image embeddings for backward pass (for i2t/a2t)
        text_bank: Text embeddings for backward pass (for t2i/t2a)
        use_hard_cc: If True, return binary (0/1). If False, return soft RSC score.
        verbose: Whether to log progress
    
    Returns:
        rsc_scores: [M] RSC scores for each candidate (higher = more self-consistent)
    """
    import numpy as np
    
    device = queries.device
    M = len(stage1_candidates)
    
    if M == 0:
        return np.array([], dtype=np.float64)
    
    # Convert to tensor if needed
    stage1_candidates_tensor = torch.tensor(stage1_candidates, device=device, dtype=torch.long)
    
    # Extract candidate embeddings
    candidate_queries = queries[stage1_candidates_tensor]  # [M, D]
    
    # Forward pass: candidate → top-1 gallery
    forward_sims = candidate_queries @ gallery.T  # [M, G]
    forward_top1_idx = forward_sims.argmax(dim=1)  # [M] indices into gallery
    forward_top1_emb = gallery[forward_top1_idx]   # [M, D]
    
    # Determine which bank to use for backward pass
    # CRITICAL: For CC, we need the SAME bank that contains the original queries
    # to measure "does round-trip return to me?"
    if direction in ['i2t', 'a2t']:
        # Query is image, gallery is text
        # Backward: text → image, need image bank for comparison
        if image_bank is None:
            if verbose:
                logger.warning("⚠️ image_bank not provided for i2t/a2t CC, using queries as fallback")
            query_bank = queries
        else:
            query_bank = image_bank.to(device) if image_bank.device != device else image_bank
    elif direction in ['t2i', 't2a']:
        # Query is text, gallery is image
        # Backward: image → text, need text bank for comparison
        if text_bank is None:
            if verbose:
                logger.warning("⚠️ text_bank not provided for t2i/t2a CC, using queries as fallback")
            query_bank = queries
        else:
            query_bank = text_bank.to(device) if text_bank.device != device else text_bank
    else:
        query_bank = queries
    
    # Backward pass: top-1 gallery → full query bank
    # BATCHED to avoid OOM when query_bank is large (e.g., 331K train texts)
    N_bank = query_bank.shape[0]
    BATCH_SIZE_RSC = 2048  # Process queries in batches to limit memory
    
    if M * N_bank > 50_000_000:  # > 50M elements → batch to avoid OOM
        # Batched argmax: process forward_top1_emb in chunks
        backward_top1_idx = torch.zeros(M, dtype=torch.long, device=device)
        for batch_start in range(0, M, BATCH_SIZE_RSC):
            batch_end = min(batch_start + BATCH_SIZE_RSC, M)
            batch_sims = forward_top1_emb[batch_start:batch_end] @ query_bank.T
            backward_top1_idx[batch_start:batch_end] = batch_sims.argmax(dim=1)
            del batch_sims
            torch.cuda.empty_cache()
    else:
        backward_sims = forward_top1_emb @ query_bank.T  # [M, N_bank]
        backward_top1_idx = backward_sims.argmax(dim=1)  # [M] indices into query_bank
        del backward_sims
    
    if use_hard_cc:
        # Hard CC: 1 iff backward top-1 equals the original candidate's global index
        # FIXED: Compare with position in query_bank, not stage1_candidates directly
        # If query_bank == queries, stage1_candidates are the correct indices
        # But we need to verify this assumption holds
        if query_bank is queries:
            # Safe: indices are into the same array
            cc_scores = (backward_top1_idx.cpu().numpy() == stage1_candidates).astype(np.float64)
        else:
            # query_bank is different from queries (e.g., separate image_bank/text_bank)
            # We need to find which index in query_bank corresponds to each candidate
            # For now, assume query_bank contains queries in same order (indices match)
            # This is fragile - soft CC is more robust
            cc_scores = (backward_top1_idx.cpu().numpy() == stage1_candidates).astype(np.float64)
    else:
        # Soft "Relative Self-Consistency" (RSC) score:
        # RSC(q) = <q, b(q)> / max_{q' ≠ q} <q, q'>
        # where b(q) is the backward-retrieved embedding from round-trip
        #
        # Measures: "How similar is the backward-retrieved query to me,
        #            relative to my best non-self match in the query bank?"
        # Range: [0, ~1] where higher = more self-consistent round-trip
        
        # Ensure L2 normalization for proper cosine similarity
        candidate_queries_norm = F.normalize(candidate_queries, p=2, dim=1)
        query_bank_norm = F.normalize(query_bank, p=2, dim=1)
        
        # Get the backward-retrieved embeddings
        backward_top1_emb = query_bank_norm[backward_top1_idx]  # [M, D]
        
        # Self-similarity: original candidate vs backward-retrieved
        self_sim = (candidate_queries_norm * backward_top1_emb).sum(dim=1)  # [M]
        
        # Max similarity: original candidate vs query_bank, EXCLUDING SELF
        # BATCHED to avoid OOM when query_bank is large
        if M * N_bank > 50_000_000:  # > 50M elements → batch
            max_sim = torch.zeros(M, device=device, dtype=candidate_queries_norm.dtype)
            for batch_start in range(0, M, BATCH_SIZE_RSC):
                batch_end = min(batch_start + BATCH_SIZE_RSC, M)
                batch_sims = candidate_queries_norm[batch_start:batch_end] @ query_bank_norm.T  # [batch, N_bank]
                
                # If query_bank is the same as queries, mask out self-matches
                if query_bank is queries:
                    for i in range(batch_end - batch_start):
                        global_idx = batch_start + i
                        batch_sims[i, stage1_candidates[global_idx]] = -float('inf')
                
                max_sim[batch_start:batch_end] = batch_sims.max(dim=1).values
                del batch_sims
                torch.cuda.empty_cache()
        else:
            all_sims = candidate_queries_norm @ query_bank_norm.T  # [M, N_bank]
            
            # If query_bank is the same as queries, mask out self-matches
            # stage1_candidates[i] is the global index of candidate i in queries
            if query_bank is queries:
                # Mask out self-similarity: set sim to -inf at self position
                for i in range(M):
                    all_sims[i, stage1_candidates[i]] = -float('inf')
            
            max_sim = all_sims.max(dim=1).values  # [M] max over non-self entries
            del all_sims
        
        # Ratio: how close is backward result to the best non-self match?
        # Clamp to avoid issues when all sims are masked
        max_sim = torch.clamp(max_sim, min=1e-8)
        cc_scores = (self_sim / max_sim).cpu().numpy().astype(np.float64)
        
        # Clamp to [0, 1] for stability (can exceed 1 if backward is self)
        cc_scores = np.clip(cc_scores, 0.0, 1.0)
    
    if verbose:
        logger.info(f"📊 Fast RSC computed for {M} candidates: "
                   f"range=[{cc_scores.min():.4f}, {cc_scores.max():.4f}], "
                   f"mean={cc_scores.mean():.4f}, {'hard' if use_hard_cc else 'soft (relative self-consistency)'} mode")
    
    return cc_scores




def analyze_flip_patterns(query_ids, gt_mapping, hhd_similarity_matrix, hhd_diagnostics, baseline_type, total_queries, base_rankings=None):
    """
    Comprehensive debugging analysis for flip patterns.
    
    Args:
        query_ids: List of query IDs
        gt_mapping: Dict mapping query_id -> set of positive gallery indices
        hhd_similarity_matrix: HHD scores [N_queries, N_gallery] 
        hhd_diagnostics: List of diagnostics per query
        baseline_type: Type of baseline being analyzed
        total_queries: Total number of queries
        base_rankings: Base rankings [N_queries, 50] - FROZEN before any reranking
    
    Returns:
    - Top-k movement matrix (before→after)
    - A/B/C/D accounting (flip correctness analysis)
    - Per-rank win rates
    - Margin decile analysis
    """
    # CRITICAL: Use provided base rankings (frozen before reranking)
    if base_rankings is not None:
        base_rankings_array = np.array(base_rankings)  # [N_queries, 50]
        logger.info(f"Using frozen base rankings: {base_rankings_array.shape}")
    else:
        raise ValueError("base_rankings is required for accurate flip analysis")
    
    # 1) ID SPACE HYGIENE - Build GT sets and verify mappings
    Nq = len(query_ids)
    gt_sets = []
    for query_id in query_ids:
        positive_indices = gt_mapping[query_id]
        gt_sets.append(set(positive_indices))
    
    # Verify GT sets structure
    assert len(gt_sets) == Nq, f"GT sets length {len(gt_sets)} != queries {Nq}"
    gt_sizes = [len(s) for s in gt_sets]
    logger.info(f"GT set sizes: min={min(gt_sizes)}, max={max(gt_sizes)}, mean={np.mean(gt_sizes):.1f}")
    
    # Sanity check: Verify gallery IDs are caption IDs (not image IDs)
    # Take 10 random queries; check that at least one of base_top50 is in gt_sets[q]
    random.seed(42)  # For reproducibility
    test_queries = random.sample(range(Nq), min(10, Nq))
    hits = sum(any(gid in gt_sets[q] for gid in base_rankings_array[q][:50]) for q in test_queries)
    logger.info(f"Sanity base hits in top50 (should be close to {len(test_queries)}): {hits}")
    
    if hits < len(test_queries) * 0.5:  # Less than 50% hit rate
        logger.error(f"ID SPACE ERROR: Only {hits}/{len(test_queries)} queries have GT hits in base top-50")
        logger.error("This suggests gallery IDs are not caption IDs or wrong modality (i2t vs t2i)")
        
        # Debug: Print sample mappings
        for i, q in enumerate(test_queries[:3]):
            logger.error(f"Query {q}: GT={sorted(list(gt_sets[q]))[:5]}")
            logger.error(f"  base@5={base_rankings_array[q][:5].tolist()}")
            logger.error(f"  hit@5={any(g in gt_sets[q] for g in base_rankings_array[q][:5])}")
    
    # 2) FREEZE BASE VS AFTER - No mutation
    base_topk_idx = base_rankings_array  # [Nq, 50] - frozen global caption IDs
    base_top1 = base_topk_idx[:, 0].copy()  # [Nq] - frozen base top-1
    
    # Get HHD rankings from similarity matrix
    # Handle both torch tensors and numpy arrays
    if isinstance(hhd_similarity_matrix, torch.Tensor):
        hhd_sim_np = hhd_similarity_matrix.cpu().numpy()
    else:
        hhd_sim_np = np.array(hhd_similarity_matrix)
    
    hhd_topk_idx = []
    for i in range(Nq):
        hhd_scores = hhd_sim_np[i]
        hhd_top_k = np.argsort(hhd_scores)[::-1][:50]  # Use numpy argsort instead of torch.topk
        hhd_topk_idx.append(hhd_top_k)
    hhd_topk_idx = np.array(hhd_topk_idx)  # [Nq, 50]
    hhd_top1 = hhd_topk_idx[:, 0].copy()  # [Nq] - HHD top-1
    
    # 3) CORRECT A/B/C/D AND FLIP LOGIC
    def is_correct(top1_gid: int, gt_set: set[int]) -> bool:
        return top1_gid in gt_set
    
    # Vectorized correctness checks
    base_correct = np.array([is_correct(g, gt_sets[q]) for q, g in enumerate(base_top1)])
    hhd_correct = np.array([is_correct(g, gt_sets[q]) for q, g in enumerate(hhd_top1)])
    
    # Flip detection: only count actual top-1 changes
    flip = (hhd_top1 != base_top1)
    
    # A/B/C/D accounting (with safe NaN handling)
    def safe_int_sum(arr):
        result = np.sum(arr)
        if np.isnan(result) or np.isinf(result):
            return 0
        return int(result)
    
    A = safe_int_sum(flip & (~base_correct) & hhd_correct)  # flip & base wrong & new correct
    B = safe_int_sum(flip & base_correct & (~hhd_correct))  # flip & base correct & new wrong
    C = safe_int_sum(flip & base_correct & hhd_correct)     # flip & both correct
    D = safe_int_sum(flip & (~base_correct) & (~hhd_correct))  # flip & both wrong
    F = safe_int_sum(flip)  # total flips
    
    # Build per-flip details for logging and visualization
    per_flip_details = []
    for q in range(Nq):
        if flip[q]:
            # Determine category
            if not base_correct[q] and hhd_correct[q]:
                category = 'A'  # help: base wrong → after flip is right
            elif base_correct[q] and not hhd_correct[q]:
                category = 'B'  # hurt: base right → after flip is wrong
            elif base_correct[q] and hhd_correct[q]:
                category = 'D'  # still right: base right → flip to another GT (wasted but safe)
            else:  # not base_correct[q] and not hhd_correct[q]
                category = 'C'  # still wrong: base wrong → flip to another wrong (wasted)
            
            # Get similarity scores (use numpy version)
            base_top1_score = float(hhd_sim_np[q][base_top1[q]])
            hhd_top1_score = float(hhd_sim_np[q][hhd_top1[q]])
            
            per_flip_details.append({
                'query_idx': q,
                'query_id': query_ids[q],
                'category': category,
                'baseline_top1': int(base_top1[q]),
                'baseline_top1_score': base_top1_score,
                'refined_top1': int(hhd_top1[q]),
                'refined_top1_score': hhd_top1_score,
                'baseline_correct': bool(base_correct[q]),
                'refined_correct': bool(hhd_correct[q]),
                'score_delta': hhd_top1_score - base_top1_score
            })
    
    # Sanity checks
    assert A + B + C + D == F, f"A+B+C+D ({A + B + C + D}) != F ({F})"
    
    # R@1 calculations
    R1_base = float(np.mean(base_correct))
    R1_hhd = float(np.mean(hhd_correct))
    actual_delta = (R1_hhd - R1_base) * Nq
    
    assert abs(actual_delta - (A - B)) <= 1, f"ΔR@1*N ({actual_delta:.1f}) != A-B ({A - B})"
    
    # 🔍 RECONCILIATION: Track where flips come from (gate decisions vs risk-controlled decisions)
    gated_flip_count = 0  # Flips from queries that were GATED
    recip_flip_count = 0  # Flips where g1_recip_like=True
    non_recip_gated_flip = 0  # Flips where gated but not recip_like
    non_gated_flip = 0  # Flips from non-gated queries (should be ~1)
    
    # Detailed breakdown for non-reciprocal gated flips
    true_hhd_early = 0  # True HHD early flips (before G1 gate)
    g1_only_accepted = 0  # G1 gate accepted but risk-controlled rejected
    risk_controlled_accepted = 0  # Risk-controlled decision accepted
    other_flips = 0  # Other sources (nudges, spectral, etc.)
    
    for q in range(Nq):
        if not flip[q]:
            continue
        diag = hhd_diagnostics[q] if q < len(hhd_diagnostics) else {}
        if diag.get('gated', False):  # Query was gated
            gated_flip_count += 1
            if diag.get('g1_recip_like', False):
                recip_flip_count += 1
            else:
                non_recip_gated_flip += 1
                # Track breakdown of non-reciprocal flips
                # Check if this was an early return (before G1 gate)
                flip_decision = diag.get('flip_decision', {})
                if flip_decision.get('reason') == 'no_alternative':
                    # This means no G1 gate was run - it's a true HHD early flip
                    true_hhd_early += 1
                elif diag.get('g1_gate_accept', False):
                    # G1 gate accepted
                    if diag.get('accepted', False) and flip_decision.get('should_flip', False):
                        # Risk-controlled also accepted
                        risk_controlled_accepted += 1
                    else:
                        # G1 accepted but risk-controlled rejected
                        g1_only_accepted += 1
                else:
                    # Neither G1 nor risk-controlled explicitly accepted
                    # These are flips from other sources (nudges, spectral, etc.)
                    other_flips += 1
        else:
            non_gated_flip += 1
    
    # Compute A/B/C/D breakdown by flip source
    source_abcd = {
        'reciprocal': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
        'risk_controlled': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
        'g1_only': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
        'other': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
        'non_gated': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    }
    
    for q in range(Nq):
        if not flip[q]:
            continue
        
        # Determine flip source
        diag = hhd_diagnostics[q] if q < len(hhd_diagnostics) else {}
        flip_decision = diag.get('flip_decision', {})
        
        if not diag.get('gated', False):
            source = 'non_gated'
        elif diag.get('g1_recip_like', False):
            source = 'reciprocal'
        elif flip_decision.get('reason') == 'no_alternative':
            source = 'other'  # True HHD early (counted as 0, so goes to other)
        elif diag.get('g1_gate_accept', False):
            if diag.get('accepted', False) and flip_decision.get('should_flip', False):
                source = 'risk_controlled'
            else:
                source = 'g1_only'
        else:
            source = 'other'
        
        # Update A/B/C/D for this source
        source_abcd[source]['F'] += 1
        if not base_correct[q] and hhd_correct[q]:
            source_abcd[source]['A'] += 1
        elif base_correct[q] and not hhd_correct[q]:
            source_abcd[source]['B'] += 1
        elif base_correct[q] and hhd_correct[q]:
            source_abcd[source]['C'] += 1
        else:
            source_abcd[source]['D'] += 1
    
    logger.info(f"🔍 FLIP RECONCILIATION: F={F}, Gated={gated_flip_count}, Recip={recip_flip_count}, NonRecipGated={non_recip_gated_flip}, NonGated={non_gated_flip}")
    logger.info(f"   📊 Non-Reciprocal Breakdown:")
    logger.info(f"      ├─ True HHD Early: {true_hhd_early} (before G1 gate)")
    logger.info(f"      ├─ G1_Only: {g1_only_accepted} (G1 allowed, risk-controlled blocked)")
    logger.info(f"      ├─ Risk-Controlled: {risk_controlled_accepted} (G1 + risk-controlled both accepted)")
    logger.info(f"      └─ Other: {other_flips} (nudges/spectral/post-processing)")
    
    # Log A/B/C/D breakdown by source
    logger.info(f"")
    logger.info(f"   🎯 A/B/C/D Breakdown by Flip Source:")
    for source_name, abcd in source_abcd.items():
        # Safe conversion helper
        def safe_val(x, default=0):
            try:
                if isinstance(x, (int, np.integer)):
                    return int(x)
                x_float = float(x)
                if np.isnan(x_float) or np.isinf(x_float):
                    return default
                return int(x_float)
            except (ValueError, TypeError, OverflowError):
                return default
        
        F_val = safe_val(abcd['F'])
        if F_val > 0:
            A_val = safe_val(abcd['A'])
            B_val = safe_val(abcd['B'])
            C_val = safe_val(abcd['C'])
            D_val = safe_val(abcd['D'])
            precision = A_val / (A_val + B_val) if (A_val + B_val) > 0 else 0.0
            net = A_val - B_val
            logger.info(f"      {source_name.upper()}: F={F_val}, A={A_val}, B={B_val}, C={C_val}, D={D_val}, Net={net:+d}, Precision={precision:.1%}")
    
    # 3.5) Helper function for GT rank extraction (needed for spectral analysis)
    def rank_or_11(topk, gt_set):
        for r, gid in enumerate(topk[:10], 1):  # Returns 1-10
            if gid in gt_set: 
                return r
        return 11  # >10
    
    # 3.6) SPECTRAL-SUBSET PRECISION ANALYSIS (A_spec, B_spec, C_spec, D_spec)
    # Only queries that went through HHD processing have spec_kept in their diagnostics
    spec_kept = np.array([(hhd_diagnostics[q].get('spec_kept', False) if q < len(hhd_diagnostics) else False) for q in range(Nq)])
    spec_kept_count = safe_int_sum(spec_kept)
    
    if spec_kept_count > 0:
        # Filter to spec-kept subset
        flip_spec = flip & spec_kept
        A_spec = safe_int_sum(flip_spec & (~base_correct) & hhd_correct)
        B_spec = safe_int_sum(flip_spec & base_correct & (~hhd_correct))
        C_spec = safe_int_sum(flip_spec & base_correct & hhd_correct)
        D_spec = safe_int_sum(flip_spec & (~base_correct) & (~hhd_correct))
        F_spec = safe_int_sum(flip_spec)
        precision_spec = A_spec / (A_spec + B_spec) if (A_spec + B_spec) > 0 else 0.0
        
        # Rank-2 promotion analysis (within spec-kept)
        # Check which queries had GT at rank-2 before and rank-1 after (within spec-kept)
        rank2_to_rank1_spec = 0
        rank2_to_rank1_correct_spec = 0
        for q in range(Nq):
            if not spec_kept[q]:
                continue
            rb = rank_or_11(base_topk_idx[q], gt_sets[q])
            ra = rank_or_11(hhd_topk_idx[q], gt_sets[q])
            if rb == 2 and ra == 1:
                rank2_to_rank1_spec += 1
                if hhd_correct[q]:
                    rank2_to_rank1_correct_spec += 1
        rank2_promo_pct_spec = (100 * rank2_to_rank1_correct_spec / rank2_to_rank1_spec) if rank2_to_rank1_spec > 0 else 0.0
    else:
        A_spec = B_spec = C_spec = D_spec = F_spec = 0
        precision_spec = 0.0
        rank2_to_rank1_spec = 0
        rank2_to_rank1_correct_spec = 0
        rank2_promo_pct_spec = 0.0
    
    # 4) PROPER GT MOVEMENT MATRIX (must sum to N)
    movement_matrix = np.zeros((12, 12), dtype=int)  # rows=BEFORE (rb), cols=AFTER (ra)
    for q in range(Nq):
        rb = rank_or_11(base_topk_idx[q], gt_sets[q])   # BEFORE (1-11)
        ra = rank_or_11(hhd_topk_idx[q],  gt_sets[q])   # AFTER (1-11)
        # Convert to 0-based indexing: 1-10 -> 0-9, 11 -> 10
        rb_idx = rb - 1 if rb <= 10 else 10
        ra_idx = ra - 1 if ra <= 10 else 10
        movement_matrix[rb_idx, ra_idx] += 1
    
    assert movement_matrix.sum() == Nq, f"Movement matrix sum ({movement_matrix.sum()}) != Nq ({Nq})"
    
    # 5) PER-RANK WIN RATES (correct calculation)
    win_r = {}
    for r in range(1, 11):
        r_idx = r - 1  # Convert to 0-based indexing
        denom = movement_matrix[r_idx, :].sum()        # #queries with GT at rank r BEFORE
        win_r[str(r)] = (movement_matrix[r_idx, 0] / denom) if denom else 0.0  # Column 0 = rank 1
    
    per_rank_win_rates_final = win_r
    
    # 6) MARGIN DECILE ANALYSIS
    margin_decile_counts = defaultdict(lambda: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'total': 0})
    for q in range(Nq):
        diagnostics = hhd_diagnostics[q] if q < len(hhd_diagnostics) else {}
        if 'margin_z' in diagnostics:
            margin = diagnostics['margin_z']
            # Safe decile calculation
            try:
                if np.isnan(margin) or np.isinf(margin):
                    decile = 0
                else:
                    decile = min(9, max(0, int(margin * 10)))  # 0-9 deciles
            except (ValueError, TypeError, OverflowError):
                decile = 0
            
            if flip[q]:
                if not base_correct[q] and hhd_correct[q]:
                    margin_decile_counts[decile]['A'] += 1
                elif base_correct[q] and not hhd_correct[q]:
                    margin_decile_counts[decile]['B'] += 1
                elif base_correct[q] and hhd_correct[q]:
                    margin_decile_counts[decile]['C'] += 1
                else:
                    margin_decile_counts[decile]['D'] += 1
            margin_decile_counts[decile]['total'] += 1
    
    margin_decile_analysis = {}
    for decile in range(10):
        if decile in margin_decile_counts:
            counts = margin_decile_counts[decile]
            total = counts['total']
            if total > 0:
                margin_decile_analysis[f'decile_{decile}'] = {
                    'A': counts['A'], 'B': counts['B'], 'C': counts['C'], 'D': counts['D'],
                    'total': total, 'A_rate': counts['A'] / total, 'B_rate': counts['B'] / total
                }
    
    # 7) DEBUG PRINTS FOR 3 RANDOM QUERIES
    logger.info("🔍 DEBUG SAMPLE QUERIES:")
    for i, q in enumerate(test_queries[:3]):
        logger.info(f"Query {q}:")
        logger.info(f"  GT: {sorted(list(gt_sets[q]))[:5]}")
        logger.info(f"  base@5: {base_topk_idx[q][:5].tolist()}, hit@5: {any(g in gt_sets[q] for g in base_topk_idx[q][:5])}")
        logger.info(f"  hhd@5:  {hhd_topk_idx[q][:5].tolist()}, hit@5: {any(g in gt_sets[q] for g in hhd_topk_idx[q][:5])}")
        logger.info(f"  flip?: {hhd_top1[q] != base_top1[q]}, base_correct: {base_top1[q] in gt_sets[q]}, hhd_correct: {hhd_top1[q] in gt_sets[q]}")
    
    # 8) FEATURE STATISTICS ANALYSIS (for all queries and by flip category)
        # Helper function for safe NaN formatting
        def safe_format(val, fmt='.3f'):
            try:
                if np.isnan(val) or np.isinf(val):
                    return 'NaN'
                return f"{val:{fmt}}"
            except (TypeError, ValueError):
                return 'NaN'
        
    # Determine which features to analyze based on baseline type
    if baseline_type == 'spectral_refinement':
        # Spectral refinement features
        feature_names = ['rts_z', 'ccs_z', 'rtps_z', 'contra_z', 'dhc_z', 'curv_z', 'rtrc_z', 'feature_score', 'delta_S', 'lda_score']
        # Extract both feature_values and delta_S/lda_score from top-level diag
        def feature_extract_fn(diag):
            feat_dict = diag.get('feature_values', {}).copy()
            # Add delta_S and lda_score from top-level diag
            if 'delta_S' in diag:
                feat_dict['delta_S'] = diag['delta_S']
            if 'lda_score' in diag:
                feat_dict['lda_score'] = diag['lda_score']
            return feat_dict
    else:
        # R1P-LDA features
        feature_names = ['bf_z', 'rg_z', 'evt_z', 'rbo_z', 'dF_z', 'hub_z', 'alt_rank', 'delta_S', 'lda_score', 'margin_z']
        feature_extract_fn = lambda diag: diag
    
    # Collect feature statistics for ALL queries (for overall stats)
    all_features = {name: [] for name in feature_names}
    for q_idx in range(Nq):
        diag = hhd_diagnostics[q_idx] if q_idx < len(hhd_diagnostics) else {}
        feat_dict = feature_extract_fn(diag)
        for feat_name in feature_names:
            val = feat_dict.get(feat_name, float('nan'))
            if not np.isnan(val) and not np.isinf(val):
                all_features[feat_name].append(val)
    
    # Log overall feature statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 OVERALL FEATURE STATISTICS (across {Nq} queries)")
    logger.info(f"{'='*80}")
    for feat_name in feature_names:
        values = all_features[feat_name]
        if len(values) > 0:
            logger.info(f"  {feat_name:15s}: μ={np.mean(values):8.4f}, σ={np.std(values):8.4f}, "
                      f"median={np.median(values):8.4f}, q25={np.percentile(values, 25):8.4f}, "
                      f"q75={np.percentile(values, 75):8.4f}, n={len(values)}")
        else:
            logger.info(f"  {feat_name:15s}: No valid values")
    
    # Collect feature statistics by flip category
    help_flips = np.where(flip & (~base_correct) & hhd_correct)[0]  # A category
    hurt_flips = np.where(flip & base_correct & (~hhd_correct))[0]  # B category
    still_wrong_flips = np.where(flip & (~base_correct) & (~hhd_correct))[0]  # C category
    still_right_flips = np.where(flip & base_correct & hhd_correct)[0]  # D category
    
    # Feature statistics by category
    category_features = {
        'A_helpful': {name: [] for name in feature_names},
        'B_hurt': {name: [] for name in feature_names},
        'C_still_wrong': {name: [] for name in feature_names},
        'D_still_right': {name: [] for name in feature_names}
    }
    
    for q_idx in help_flips:
        diag = hhd_diagnostics[q_idx] if q_idx < len(hhd_diagnostics) else {}
        feat_dict = feature_extract_fn(diag)
        for feat_name in feature_names:
            val = feat_dict.get(feat_name, float('nan'))
            if not np.isnan(val) and not np.isinf(val):
                category_features['A_helpful'][feat_name].append(val)
    
    for q_idx in hurt_flips:
        diag = hhd_diagnostics[q_idx] if q_idx < len(hhd_diagnostics) else {}
        feat_dict = feature_extract_fn(diag)
        for feat_name in feature_names:
            val = feat_dict.get(feat_name, float('nan'))
            if not np.isnan(val) and not np.isinf(val):
                category_features['B_hurt'][feat_name].append(val)
    
    for q_idx in still_wrong_flips:
        diag = hhd_diagnostics[q_idx] if q_idx < len(hhd_diagnostics) else {}
        feat_dict = feature_extract_fn(diag)
        for feat_name in feature_names:
            val = feat_dict.get(feat_name, float('nan'))
            if not np.isnan(val) and not np.isinf(val):
                category_features['C_still_wrong'][feat_name].append(val)
    
    for q_idx in still_right_flips:
        diag = hhd_diagnostics[q_idx] if q_idx < len(hhd_diagnostics) else {}
        feat_dict = feature_extract_fn(diag)
        for feat_name in feature_names:
            val = feat_dict.get(feat_name, float('nan'))
            if not np.isnan(val) and not np.isinf(val):
                category_features['D_still_right'][feat_name].append(val)
    
    # Log feature statistics by category
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 FEATURE STATISTICS BY FLIP CATEGORY")
    logger.info(f"{'='*80}")
    logger.info(f"  A (Helpful): {len(help_flips)} flips | B (Hurt): {len(hurt_flips)} flips | "
              f"C (Still Wrong): {len(still_wrong_flips)} flips | D (Still Right): {len(still_right_flips)} flips")
    logger.info(f"\n  Feature comparison (A vs B):")
    logger.info(f"  {'Feature':<15s} | {'A (Helpful) μ':<15s} | {'B (Hurt) μ':<15s} | {'Difference':<12s} | {'A n':<6s} | {'B n':<6s}")
    logger.info(f"  {'-'*85}")
    for feat_name in feature_names:
        a_vals = category_features['A_helpful'][feat_name]
        b_vals = category_features['B_hurt'][feat_name]
        if len(a_vals) > 0 or len(b_vals) > 0:
            a_mean = np.mean(a_vals) if len(a_vals) > 0 else float('nan')
            b_mean = np.mean(b_vals) if len(b_vals) > 0 else float('nan')
            diff = a_mean - b_mean if not (np.isnan(a_mean) or np.isnan(b_mean)) else float('nan')
            logger.info(f"  {feat_name:<15s} | {safe_format(a_mean, '12.4f'):<15s} | {safe_format(b_mean, '12.4f'):<15s} | "
                      f"{safe_format(diff, '+11.4f'):<12s} | {len(a_vals):<6d} | {len(b_vals):<6d}")
    
    # 8) DETAILED DECOMPOSITION ANALYSIS FOR HURT FLIPS (B category)
    if len(hurt_flips) > 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 HURT FLIP DECOMPOSITION ANALYSIS (B category: {len(hurt_flips)} flips)")
        logger.info(f"{'='*80}")
        
        # Collect feature statistics for hurt flips
        hurt_features = {name: [] for name in feature_names}
        hurt_features['base_rank_gt'] = []
        hurt_features['hhd_rank_gt'] = []
        
        for q_idx in hurt_flips[:min(20, len(hurt_flips))]:  # Analyze up to 20 hurt flips
            diag = hhd_diagnostics[q_idx] if q_idx < len(hhd_diagnostics) else {}
            
            # Rank analysis
            base_rank_gt = rank_or_11(base_topk_idx[q_idx], gt_sets[q_idx])
            hhd_rank_gt = rank_or_11(hhd_topk_idx[q_idx], gt_sets[q_idx])
            
            # Feature extraction
            feat_dict = feature_extract_fn(diag)
            for feat_name in feature_names:
                val = feat_dict.get(feat_name, float('nan'))
                hurt_features[feat_name].append(val)
            
            hurt_features['base_rank_gt'].append(base_rank_gt)
            hurt_features['hhd_rank_gt'].append(hhd_rank_gt)
            
            # Log individual hurt flip details (with safe NaN formatting)
            safety_lane = diag.get('safety_lane', 'unknown')
            logger.info(f"\n  ❌ Hurt Flip #{q_idx+1} (Query {q_idx}, Lane {safety_lane}):")
            logger.info(f"    Base: g1={base_top1[q_idx]} (GT rank={base_rank_gt}, CORRECT)")
            logger.info(f"    HHD:  g1={hhd_top1[q_idx]} (GT rank={hhd_rank_gt}, WRONG)")
            logger.info(f"    Features:")
            if baseline_type == 'spectral_refinement':
                logger.info(f"      rts_z={safe_format(feat_dict.get('rts_z', float('nan')))}, "
                          f"ccs_z={safe_format(feat_dict.get('ccs_z', float('nan')))}, "
                          f"rtps_z={safe_format(feat_dict.get('rtps_z', float('nan')))}")
                logger.info(f"      contra_z={safe_format(feat_dict.get('contra_z', float('nan')))}, "
                          f"dhc_z={safe_format(feat_dict.get('dhc_z', float('nan')))}, "
                          f"curv_z={safe_format(feat_dict.get('curv_z', float('nan')))}")
                logger.info(f"      rtrc_z={safe_format(feat_dict.get('rtrc_z', float('nan')))}, "
                          f"feature_score={safe_format(feat_dict.get('feature_score', float('nan')))}")
            else:
                logger.info(f"      bf_z={safe_format(feat_dict.get('bf_z', float('nan')))}, "
                          f"rg_z={safe_format(feat_dict.get('rg_z', float('nan')))}, "
                          f"evt_z={safe_format(feat_dict.get('evt_z', float('nan')))}")
                logger.info(f"      rbo_z={safe_format(feat_dict.get('rbo_z', float('nan')))}, "
                          f"dF_z={safe_format(feat_dict.get('dF_z', float('nan')))}, "
                          f"hub_z={safe_format(feat_dict.get('hub_z', float('nan')))}")
                logger.info(f"      alt_rank={safe_format(feat_dict.get('alt_rank', float('nan')), '.1f')}, "
                          f"margin_z={safe_format(feat_dict.get('margin_z', float('nan')))}")
            logger.info(f"    Scores:")
            logger.info(f"      delta_S={safe_format(feat_dict.get('delta_S', float('nan')))}, "
                          f"lda_score={safe_format(feat_dict.get('lda_score', float('nan')))}")
            logger.info(f"    Rank movement: GT moved from rank {base_rank_gt} → {hhd_rank_gt}")
            
            # Stage analysis
            accepted = diag.get('accepted', False)
            gated = diag.get('gated', False)
            logger.info(f"    Decision stages: gated={gated}, accepted={accepted}")
        
        # Summary statistics for hurt flips
        valid_count = len([x for x in hurt_features[feature_names[0]] if not np.isnan(x)])
        logger.info(f"\n  📊 HURT FLIP FEATURE STATISTICS (n={valid_count}):")
        for feat_name in feature_names:
            valid_values = [v for v in hurt_features[feat_name] if not np.isnan(v) and not np.isinf(v)]
            if valid_values:
                logger.info(f"    {feat_name}: μ={np.mean(valid_values):.3f}, median={np.median(valid_values):.3f}, "
                          f"q25={np.percentile(valid_values, 25):.3f}, q75={np.percentile(valid_values, 75):.3f}")
        
        # Compare with A category (helpful flips) if available
        if len(help_flips) > 0:
            logger.info(f"\n  📊 COMPARISON: Helpful (A) vs Hurt (B) Flips:")
            logger.info(f"    Feature | Helpful (A) μ | Hurt (B) μ | Difference")
            logger.info(f"    {'-'*60}")
            for feat_name in feature_names:
                help_vals = category_features['A_helpful'][feat_name]
                hurt_vals = category_features['B_hurt'][feat_name]
                if help_vals and hurt_vals:
                    help_mean = np.mean(help_vals)
                    hurt_mean = np.mean(hurt_vals)
                    diff = help_mean - hurt_mean
                    logger.info(f"    {feat_name:12s} | {help_mean:11.3f} | {hurt_mean:9.3f} | {diff:+.3f}")
    
    # Safe conversion helper for return values
    def safe_int_return(val, default=0):
        try:
            if isinstance(val, (int, np.integer)):
                return int(val)
            val_float = float(val)
            if np.isnan(val_float) or np.isinf(val_float):
                return default
            return int(val_float)
        except (ValueError, TypeError, OverflowError):
            return default
    
    # Prepare feature statistics for return
    hurt_feature_stats = {}
    if len(hurt_flips) > 0:
        for feat_name in feature_names:
            valid_values = [v for v in hurt_features[feat_name] if not np.isnan(v) and not np.isinf(v)]
            if valid_values:
                hurt_feature_stats[feat_name] = {
                    'mean': float(np.mean(valid_values)),
                    'median': float(np.median(valid_values)),
                    'q25': float(np.percentile(valid_values, 25)),
                    'q75': float(np.percentile(valid_values, 75)),
                    'count': len(valid_values)
                }
    
    # Prepare comparison statistics (A vs B)
    comparison_stats = {}
    if len(help_flips) > 0 and len(hurt_flips) > 0:
        for feat_name in feature_names:
            help_vals = category_features['A_helpful'][feat_name]
            hurt_vals = category_features['B_hurt'][feat_name]
            if help_vals and hurt_vals:
                comparison_stats[feat_name] = {
                    'helpful_mean': float(np.mean(help_vals)),
                    'hurt_mean': float(np.mean(hurt_vals)),
                    'difference': float(np.mean(help_vals) - np.mean(hurt_vals)),
                    'helpful_count': len(help_vals),
                    'hurt_count': len(hurt_vals)
                }
    
    return {
        'movement_matrix': movement_matrix.tolist(),
        'A_count': safe_int_return(A),
        'B_count': safe_int_return(B), 
        'C_count': safe_int_return(C),
        'D_count': safe_int_return(D),
        'F_count': safe_int_return(F),
        'actual_delta': float(actual_delta),
        'recall_at_1_hhd': float(R1_hhd),
        'recall_at_1_base': float(R1_base),
        'per_rank_win_rates': {str(k): float(v) for k, v in per_rank_win_rates_final.items()},
        'margin_decile_analysis': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                                     for kk, vv in v.items()} 
                                 for k, v in margin_decile_analysis.items()},
        'per_flip_details': per_flip_details,  # List of dicts with query_id, category, scores, etc.
        'hurt_feature_statistics': hurt_feature_stats,  # Feature stats for hurt flips (B category)
        'comparison_statistics': comparison_stats,  # Comparison: Helpful (A) vs Hurt (B)
        # Spectral-subset precision metrics
        'spec_kept_count': safe_int_return(spec_kept_count),
        'A_spec': safe_int_return(A_spec),
        'B_spec': safe_int_return(B_spec),
        'C_spec': safe_int_return(C_spec),
        'D_spec': safe_int_return(D_spec),
        'F_spec': safe_int_return(F_spec),
        'precision_spec': float(precision_spec),
        'rank2_to_rank1_spec': safe_int_return(rank2_to_rank1_spec),
        'rank2_to_rank1_correct_spec': safe_int_return(rank2_to_rank1_correct_spec),
        'rank2_promo_pct_spec': float(rank2_promo_pct_spec)
    }



def spectral_refinement_baseline(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    query_ids: List[str],
    gallery_ids: List[str],
    gt_mapping: Dict,
    baseline_params: Dict,
    direction: str = "i2t",
    eval_temperature: float = 1.0,
    save_rankings: bool = False,
    rankings_save_path: Optional[str] = None,
    gallery_img: Optional[torch.Tensor] = None,
    image_bank: Optional[torch.Tensor] = None,
    text_bank: Optional[torch.Tensor] = None,
    return_decisions: bool = False,
    precomputed_scores: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Spectral Refinement Baseline with 7 Features:
    1. Round-Trip Specificity (RTS) - caption→image margin on mini-gallery
    2. Caption Cluster Support (CCS) - mutual support among candidates
    3. Round-Trip Paraphrase Stability (RTPS) - word-drop sensitivity
    4. Contradiction Penalty (CONTRA) - lightweight lexical checks
    5. Duplication/Hallucination Control (DHC) - penalize near-duplicates
    6. Graph Curvature (CURV) - Laplacian-based curvature (reuse existing)
    7. Round-Trip Rank Consistency (RTRC) - Bradley-Terry on pairs
    
    Architecture: Multi-layer graph → Spectral filtering → Adaptive re-ranking
    
    Args:
        precomputed_scores: Optional pre-computed score matrix [N_queries, N_gallery].
                           If provided, uses these scores instead of computing queries @ gallery.T.
                           Enables geometry-consistent composition (e.g., spectral on NNN scores).
    """
    import time
    import os  # Import locally to avoid scoping issues
    from scipy.linalg import eigh
    from tqdm import tqdm
    
    logger.info("🔍 Running Spectral Refinement baseline...")
    logger.info(f"📍 Direction: {direction}")
    
    # ACCURATE CUDA TIMING: sync before start
    start_time = start_cuda_timer()
    
    # Determine device early (before it's used)
    device = queries.device
    
    # Parameters
    topK = baseline_params.get('topK', 50)  # Top-K candidates to consider
    M_rts = baseline_params.get('M_rts', 24)  # Visual neighbors for RTS
    tau_ccs = baseline_params.get('tau_ccs', 0.7)  # CCS similarity threshold
    n_drops_rtps = baseline_params.get('n_drops_rtps', 5)  # Token drops for RTPS
    lambda_contra = baseline_params.get('lambda_contra', 2.0)  # Contradiction penalty
    lambda_dhc = baseline_params.get('lambda_dhc', 0.5)  # Duplication penalty
    k_curv = baseline_params.get('k_curv', 12)  # k-NN for curvature
    beta_curv = baseline_params.get('beta_curv', 0.4)  # Curvature smoothing
    use_rts = baseline_params.get('use_rts', True)
    use_ccs = baseline_params.get('use_ccs', True)
    use_rtps = baseline_params.get('use_rtps', True)
    use_contra = baseline_params.get('use_contra', True)
    use_dhc = baseline_params.get('use_dhc', True)
    use_curv = baseline_params.get('use_curv', True)
    use_rtrc = baseline_params.get('use_rtrc', True)
    
    # Feature weights for final fusion
    w_rts = baseline_params.get('w_rts', 0.3)
    w_ccs = baseline_params.get('w_ccs', 0.3)
    w_rtps = baseline_params.get('w_rtps', 0.2)
    w_contra = baseline_params.get('w_contra', 0.2)
    w_dhc = baseline_params.get('w_dhc', 0.1)
    w_curv = baseline_params.get('w_curv', 0.05)  # Reduced from 0.1 - use as tie-breaker only
    w_rtrc = baseline_params.get('w_rtrc', 0.2)
    
    # Build active feature list (only enabled features)
    active_features = []
    if use_rts:
        active_features.append(('rts', w_rts))
    if use_ccs:
        active_features.append(('ccs', w_ccs))
    if use_rtps:
        active_features.append(('rtps', w_rtps))
    if use_contra:
        active_features.append(('contra', w_contra))
    if use_dhc:
        active_features.append(('dhc', w_dhc))
    if use_curv:
        active_features.append(('curv', w_curv))
    if use_rtrc:
        active_features.append(('rtrc', w_rtrc))
    
    # Normalize weights only across active features
    total_w = sum(w for _, w in active_features)
    if total_w > 0:
        weights_dict = {name: w / total_w for name, w in active_features}
    else:
        weights_dict = {}
    
    # Initialize image_bank_gpu and text_bank_gpu to None for scope access
    image_bank_gpu = None
    text_bank_gpu = None
    
    # For i2t direction, need image bank for RTS (CRITICAL: must use image neighbors, not text)
    # Prefer image_bank parameter, fall back to gallery_img for backward compatibility
    if direction == "i2t" and use_rts:
        if image_bank is None:
            if gallery_img is None:
                # Try to get from baseline_params as image_gallery
                gallery_img = baseline_params.get('image_gallery', None)
            if gallery_img is not None:
                image_bank = gallery_img
        if image_bank is None:
            logger.warning("RTS requires image_bank (or gallery_img) for i2t direction. Disabling RTS.")
            use_rts = False
            if ('rts', w_rts) in active_features:
                active_features.remove(('rts', w_rts))
                total_w = sum(w for _, w in active_features)
                if total_w > 0:
                    weights_dict = {name: w / total_w for name, w in active_features}
        else:
            # Store reference to image_bank for later GPU transfer
            image_bank_gpu = image_bank
    
    # For t2i direction, need text bank for RTS
    if direction == "t2i" and use_rts:
        if text_bank is None:
            # For t2i, gallery is images, so we need text_bank parameter
            text_bank = baseline_params.get('text_bank', None)
        if text_bank is None:
            logger.warning("RTS requires text_bank for t2i direction. Disabling RTS.")
            use_rts = False
            if ('rts', w_rts) in active_features:
                active_features.remove(('rts', w_rts))
                total_w = sum(w for _, w in active_features)
                if total_w > 0:
                    weights_dict = {name: w / total_w for name, w in active_features}
        else:
            # Store reference to text_bank for later GPU transfer
            text_bank_gpu = text_bank
    
    # For a2t direction, need audio bank (use image_bank parameter as audio bank)
    if direction == "a2t" and use_rts:
        if image_bank is None:
            if gallery_img is None:
                gallery_img = baseline_params.get('audio_gallery', None)
            if gallery_img is not None:
                image_bank = gallery_img
        if image_bank is None:
            logger.warning("RTS requires image_bank (audio embeddings) for a2t direction. Disabling RTS.")
            use_rts = False
            if ('rts', w_rts) in active_features:
                active_features.remove(('rts', w_rts))
                total_w = sum(w for _, w in active_features)
                if total_w > 0:
                    weights_dict = {name: w / total_w for name, w in active_features}
        else:
            # Store reference to audio bank (via image_bank) for later GPU transfer
            image_bank_gpu = image_bank
    
    # For t2a direction, need text bank for RTS
    if direction == "t2a" and use_rts:
        if text_bank is None:
            text_bank = baseline_params.get('text_bank', None)
        if text_bank is None:
            logger.warning("RTS requires text_bank for t2a direction. Disabling RTS.")
            use_rts = False
            if ('rts', w_rts) in active_features:
                active_features.remove(('rts', w_rts))
                total_w = sum(w for _, w in active_features)
                if total_w > 0:
                    weights_dict = {name: w / total_w for name, w in active_features}
        else:
            # Store reference to text_bank for later GPU transfer
            text_bank_gpu = text_bank
    
    # SigLIP logit scale flag
    apply_siglip_scale = baseline_params.get('apply_siglip_scale', False)
    siglip_logit_scale = baseline_params.get('siglip_logit_scale', 111.16)
    
    # RTPS seed for reproducibility
    rtps_seed = baseline_params.get('rtps_seed', 42)
    np.random.seed(rtps_seed)
    
    # FIXED: Move to GPU if available for faster processing
    # Check CUDA availability and device count with detailed diagnostics
    logger.info(f"🔍 CUDA Diagnostics:")
    logger.info(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"   torch.version.cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        logger.info(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            device = torch.device('cuda')
            # Test if we can actually create a tensor on GPU
            test_tensor = torch.zeros(1, device=device)
            queries = queries.to(device)
            gallery = gallery.to(device)
            if gallery_img is not None:
                gallery_img = gallery_img.to(device)
            logger.info(f"✓ Moved tensors to GPU: {device}")
        except Exception as e:
            logger.warning(f"⚠ GPU available but failed to move tensors: {e}, using CPU")
            import traceback
            logger.warning(f"   Traceback: {traceback.format_exc()}")
            cuda_available = False
        device = queries.device
    else:
        device = queries.device
        logger.info(f"⚠ CUDA not available, using CPU: {device}")
        # Additional diagnostics for why CUDA might not be available
        if torch.version.cuda is None:
            logger.warning("   PyTorch was compiled without CUDA support (torch.version.cuda is None)")
        else:
            logger.warning(f"   PyTorch has CUDA version {torch.version.cuda}, but CUDA runtime is not available")
            logger.warning("   Possible causes: CUDA drivers not installed, CUDA_VISIBLE_DEVICES misconfigured, or CUDA version mismatch")
    
    # Now that device is finalized, move image_bank and text_bank to the same device
    if image_bank_gpu is not None:
        if isinstance(image_bank_gpu, torch.Tensor):
            image_bank_gpu = image_bank_gpu.to(device) if image_bank_gpu.device != device else image_bank_gpu
        else:
            image_bank_gpu = torch.tensor(image_bank_gpu, device=device, dtype=queries.dtype)
    
    if text_bank_gpu is not None:
        if isinstance(text_bank_gpu, torch.Tensor):
            text_bank_gpu = text_bank_gpu.to(device) if text_bank_gpu.device != device else text_bank_gpu
        else:
            text_bank_gpu = torch.tensor(text_bank_gpu, device=device, dtype=queries.dtype)
    
    # Move precomputed_scores to device if provided (for geometry-consistent baselines)
    if precomputed_scores is not None:
        if isinstance(precomputed_scores, torch.Tensor):
            precomputed_scores = precomputed_scores.to(device) if precomputed_scores.device != device else precomputed_scores
        else:
            precomputed_scores = torch.tensor(precomputed_scores, device=device, dtype=queries.dtype)
    
    total_queries = len(query_ids)
    fused_scores = []
    base_rankings = []  # Store base rankings (top-50) for debugging
    spectral_diagnostics = []  # Store diagnostics per query
    gating_stats = {
        'gated': 0,
        'total': total_queries,
        'flips_to_correct_total': 0,
        'flips_to_wrong_total': 0,
        'flips_neutral_total': 0,  # Both baseline and refined correct (multi-GT case)
        'polarity_gated': 0  # FIX: Track queries that passed polarity threshold (Gate%)
    }
    
    # =============================================================================
    # 📊 GATE DIAGNOSTICS: Track gate pass/fail rates and feature distributions
    # =============================================================================
    gate_diagnostics = {
        # Overall flip attempts
        'total_flip_attempts': 0,
        'total_flips_executed': 0,
        
        # Individual gate pass/fail counts
        'sharpness_gate_attempts': 0,
        'sharpness_gate_passed': 0,
        'sharpness_gate_failed': 0,
        
        'consensus_gate_attempts': 0,
        'consensus_gate_passed': 0,
        'consensus_gate_failed': 0,
        
        'ls_margin_gate_attempts': 0,
        'ls_margin_gate_passed': 0,
        'ls_margin_gate_failed': 0,
        
        # Combined gate results
        'all_gates_passed': 0,
        'all_gates_failed': 0,
        
        # Feature distributions (will store lists for histogram analysis)
        'base_entropy_all': [],
        'ls_challenger_all': [],
        'ls_baseline_all': [],
        'ls_margin_all': [],
        'tau_q_all': [],
        'community_vote_all': [],
        'recip_z_all': [],
        'outlier_z_all': [],
        
        # Gate blocking patterns (which gates block together)
        'blocked_by_sharpness_only': 0,
        'blocked_by_consensus_only': 0,
        'blocked_by_ls_margin_only': 0,
        'blocked_by_sharpness_consensus': 0,
        'blocked_by_sharpness_ls': 0,
        'blocked_by_consensus_ls': 0,
        'blocked_by_all_three': 0,
    }
    
    per_query_data = [] if save_rankings else None
    
    # Keep gallery on GPU for faster computation
    # Gallery will be used for GPU-accelerated feature computations
    gallery_gpu = gallery  # Already on GPU if available
    
    # Handle image gallery for RTS (i2t direction)
    gallery_img_gpu = None
    if direction == "i2t" and use_rts:
        if gallery_img is not None:
            if isinstance(gallery_img, torch.Tensor):
                gallery_img_gpu = gallery_img.to(device) if gallery_img.device != device else gallery_img
            else:
                gallery_img_gpu = torch.tensor(gallery_img, device=device, dtype=queries.dtype)
        elif 'gallery_img_gpu' in locals():
            # Already set above
            pass
    
    logger.info(f"🔍 Processing {total_queries} queries with Spectral Refinement...")
    logger.info(f"  Device: {device} (GPU-accelerated: base scores, feature computation, spectral filtering)")
    
    # Helper: Robust z-score (GPU-accelerated) - FIXED: don't fabricate signal when constant/noisy
    def robust_zscore_gpu(x: torch.Tensor, eps: float = 1e-6, max_z: float = 10.0) -> torch.Tensor:
        """Robust z-score using MAD, with fallback to std if MAD is too small.
        Returns zeros when there's no meaningful spread (no signal).
        
        Args:
            x: Input tensor
            eps: Minimum spread threshold
            max_z: Maximum absolute z-score value (clip extreme values)
        """
        if len(x) == 0:
            return torch.zeros_like(x)
        
        # Check for NaN/inf
        if not torch.isfinite(x).any():
            return torch.zeros_like(x)
        
        # Check if constant - if range is tiny, return zeros (no signal)
        x_range = torch.max(x) - torch.min(x)
        if x_range < 1e-6 or torch.allclose(x, x[0], atol=1e-6):
            return torch.zeros_like(x)
        
        med = torch.median(x)
        mad = torch.median(torch.abs(x - med))
        
        # If MAD is too small, fall back to std
        if mad < eps:
            std = torch.std(x)
            if std > eps and x_range > 1e-6:
                z = (x - torch.mean(x)) / (std + eps)
            else:
                # No meaningful spread - return zeros (no signal)
                return torch.zeros_like(x)
        else:
            # Use 1.4826 factor to make MAD consistent with std for Gaussian
            z = (x - med) / (1.4826 * mad + eps)
        
        # Clip extreme z-scores to prevent huge values from dominating
        z = torch.clamp(z, -max_z, max_z)
        return z
    
    # Helper: Robust z-score (CPU fallback) - FIXED: don't fabricate signal when constant/noisy
    def robust_zscore(x: np.ndarray, eps: float = 1e-6, max_z: float = 10.0) -> np.ndarray:
        """Robust z-score using MAD, with fallback to std if MAD is too small.
        Returns zeros when there's no meaningful spread (no signal).
        
        Args:
            x: Input array
            eps: Minimum spread threshold
            max_z: Maximum absolute z-score value (clip extreme values)
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
    
    # Helper: Find visual neighbors (for RTS) - FIXED: explicitly uses image_index for i2t
    def find_visual_neighbors(query_img: torch.Tensor, image_index: torch.Tensor, M: int) -> torch.Tensor:
        """Find M visually similar images from image_index (image embeddings)."""
        sims = image_index @ query_img.squeeze()  # [N_images]
        topM = torch.topk(sims, k=min(M, image_index.size(0)), dim=0).indices
        return image_index[topM]
    
    # Helper: Build mutual k-NN graph (GPU-accelerated, FULLY VECTORIZED)
    def build_mutual_knn_graph(embeddings: torch.Tensor, k: int) -> torch.Tensor:
        """Build symmetric mutual k-NN graph from embeddings on GPU.
        
        OPTIMIZED: Fully vectorized - no Python loops.
        Uses sparse tensor scatter for O(n*k) complexity instead of O(n²) loop.
        """
        n = embeddings.size(0)
        device = embeddings.device
        dtype = embeddings.dtype
        
        # Handle edge case: k >= n
        k = min(k, n - 1)
        if k <= 0:
            return torch.zeros((n, n), device=device, dtype=dtype)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Compute cosine similarity matrix
        A_base = embeddings_norm @ embeddings_norm.T  # [n, n]
        A_base = torch.clamp(A_base, -1.0, 1.0)
        
        # Build k-NN sets using topk (GPU-accelerated)
        _, topk_indices = torch.topk(A_base, k=k+1, dim=1)  # k+1 to include self
        topk_indices = topk_indices[:, 1:]  # Remove self (first column) [n, k]
        
        # ====================================================================
        # VECTORIZED: Build forward adjacency (i → j for all i's k-NN neighbors)
        # ====================================================================
        # Row indices: [0,0,...,0, 1,1,...,1, ..., n-1,n-1,...,n-1]  (each repeated k times)
        row_idx = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten()  # [n*k]
        # Col indices: topk_indices flattened
        col_idx = topk_indices.flatten()  # [n*k]
        
        # Create forward adjacency matrix using scatter (vectorized)
        forward = torch.zeros((n, n), device=device, dtype=dtype)
        forward[row_idx, col_idx] = 1.0
        
        # ====================================================================
        # Mutual k-NN: A[i,j] = 1 iff j ∈ kNN(i) AND i ∈ kNN(j)
        # = forward AND forward.T (element-wise)
        # ====================================================================
        mutual = forward * forward.T  # [n, n]
        
        # Weight by original similarity (keep it as weighted graph)
        A = mutual * A_base
        
        # Already symmetric by construction (mutual * mutual.T = symmetric)
        return A
    
    # Helper: Compute normalized Laplacian (GPU-accelerated)
    def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
        """Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2} on GPU."""
        D_diag = A.sum(dim=1)
        D_diag = torch.clamp(D_diag, min=1e-8)
        D_sqrt_inv = 1.0 / torch.sqrt(D_diag)
        D_sqrt_inv_diag = torch.diag(D_sqrt_inv)
        I = torch.eye(len(A), device=A.device, dtype=A.dtype)
        L = I - D_sqrt_inv_diag @ A @ D_sqrt_inv_diag
        return L
    
    # Helper: Power iteration for top eigenvector (O(K² × n_iters) vs O(K³) for eigh)
    def power_iteration(M: torch.Tensor, n_iters: int = 15) -> Tuple[float, torch.Tensor]:
        """
        Compute top eigenvector of matrix M using power iteration.
        
        O(K² × n_iters) complexity vs O(K³) for full eigendecomposition.
        
        Args:
            M: Square matrix [K, K]
            n_iters: Number of iterations (default 15 for good convergence)
            
        Returns:
            (lambda_1, v_1): Top eigenvalue and corresponding eigenvector
        """
        n = M.shape[0]
        if n == 0:
            return 0.0, torch.zeros(0, device=M.device, dtype=M.dtype)
        
        # Initialize with random vector
        v = torch.randn(n, device=M.device, dtype=M.dtype)
        v = v / (torch.linalg.norm(v) + 1e-8)
        
        # Power iteration
        for _ in range(n_iters):
            v_new = M @ v
            v_norm = torch.linalg.norm(v_new)
            if v_norm < 1e-8:
                break
            v = v_new / v_norm
        
        # Compute eigenvalue (Rayleigh quotient)
        lambda_1 = float((v @ M @ v).item())
        
        return lambda_1, v
    
    # Helper: Iterative Laplacian smoothing (replaces full eigendecomposition)
    def iterative_laplacian_smooth(
        scores: torch.Tensor, 
        L: torch.Tensor, 
        gamma: float = 0.3, 
        n_iters: int = 3,
        node_conf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Low-pass filter via iterative Laplacian smoothing.
        
        Approximates spectral low-pass filter without eigendecomposition.
        s_smooth = (I - γL)^n @ s
        
        O(K² × n_iters) complexity vs O(K³) for eigh-based filter.
        
        Args:
            scores: Score vector [K]
            L: Normalized Laplacian [K, K]
            gamma: Smoothing strength (0 < γ < 1, default 0.3)
            n_iters: Number of iterations (default 3)
            node_conf: Optional per-node confidence [K] for weighted smoothing
            
        Returns:
            Smoothed scores [K]
        """
        n = scores.shape[0]
        if n == 0:
            return scores
        
        # Normalize scores to zero-mean unit-var for stability
        s_mu = scores.mean()
        s_sigma = scores.std() + 1e-8
        s_norm = (scores - s_mu) / s_sigma
        
        # Build smoothing operator: S = I - γL
        I = torch.eye(n, device=L.device, dtype=L.dtype)
        S = I - gamma * L
        
        # Optional: weight by node confidence
        if node_conf is not None:
            # Higher confidence → more smoothing
            conf_weight = torch.clamp(node_conf, 0.2, 1.0)
            S = conf_weight.unsqueeze(1) * S + (1 - conf_weight).unsqueeze(1) * I
        
        # Iterative application: s = S^n @ s
        s_smooth = s_norm
        for _ in range(n_iters):
            s_smooth = S @ s_smooth
        
        # Rescale back
        s_smooth = s_smooth * s_sigma + s_mu
        
        # Safety check
        if not torch.isfinite(s_smooth).all():
            return scores
        
        return s_smooth
    
    # Helper: Laplacian sharpening (Anti-spectral Variant A)
    def laplacian_sharpening(scores: torch.Tensor, L: torch.Tensor, gamma: float = 0.3) -> torch.Tensor:
        """
        Anti-spectral operator via Laplacian sharpening: s_sharp = s + γLs
        
        Intuition: Ls measures disagreement with neighbors.
        Adding γLs amplifies disagreement → suppresses dense clusters.
        
        Args:
            scores: Score vector [K]
            L: Normalized Laplacian [K, K]
            gamma: Sharpening strength (default 0.3)
        
        Returns:
            Sharpened scores [K]
        """
        Ls = L @ scores
        s_sharp = scores + gamma * Ls
        
        # Safety: NaN/Inf guard
        if not torch.isfinite(s_sharp).all():
            logger.warning("Laplacian sharpening produced non-finite values, returning original scores")
            return scores
        
        return s_sharp
    
    # Helper: Score-geometry disagreement boost (Diversity Variant)
    def disagreement_boost(
        scores: torch.Tensor,
        A: torch.Tensor,
        lambda_boost: float = 0.15
    ) -> torch.Tensor:
        """
        Boost candidates with high score-geometry disagreement.
        
        Intuition: If model likes a candidate (high score) but graph doesn't 
        (low graph support), it might be isolated GT that clusters are drowning.
        
        Args:
            scores: Base score vector [K]
            A: Adjacency matrix [K, K] 
            lambda_boost: Boost strength (default 0.15)
        
        Returns:
            Boosted scores [K]
        """
        try:
            K = len(scores)
            if K < 3:
                return scores
            
            # Compute ranks from base scores (lower rank = higher score)
            # argsort gives indices that would sort scores ascending
            # argsort again gives rank (0 = lowest score, K-1 = highest score)
            score_rank = torch.argsort(torch.argsort(scores, descending=True))  # 0 = top-1
            
            # Compute graph-smoothed scores
            graph_scores = A @ scores  # Weighted sum of neighbor scores
            graph_rank = torch.argsort(torch.argsort(graph_scores, descending=True))  # 0 = top-1
            
            # Compute disagreement (absolute rank difference)
            disagreement = torch.abs(score_rank.float() - graph_rank.float())  # [K]
            
            # Normalize disagreement to [0, 1]
            max_disagreement = float(K - 1)
            if max_disagreement > 0:
                disagreement_norm = disagreement / max_disagreement
            else:
                disagreement_norm = disagreement
            
            # Boost scores by disagreement
            # High disagreement → high boost
            s_boosted = scores + lambda_boost * disagreement_norm
            
            # Safety: NaN/Inf guard
            if not torch.isfinite(s_boosted).all():
                logger.warning("Disagreement boost produced non-finite values, returning original scores")
                return scores
            
            return s_boosted
            
        except Exception as e:
            logger.warning(f"Disagreement boost failed: {e}, returning original scores")
            return scores
    
    # Helper: Eigenvector deflation (Anti-spectral Variant B)
    def eigenvector_deflation(scores: torch.Tensor, L: torch.Tensor, gamma: float = 0.3) -> torch.Tensor:
        """
        Anti-spectral operator via eigenvector deflation: s_anti = s - γ⟨s, u₁⟩u₁
        
        Intuition: u₁ is the dominant cluster direction in the graph.
        Subtracting the projection suppresses the main cluster.
        
        OPTIMIZED: Uses power iteration on (I-L) instead of full eigendecomposition.
        
        Args:
            scores: Score vector [K]
            L: Normalized Laplacian [K, K]
            gamma: Deflation strength (default 0.3)
        
        Returns:
            Deflated scores [K]
        """
        try:
            n = L.shape[0]
            if n == 0:
                return scores
            
            # OPTIMIZED: Use power iteration instead of full eigh
            # For normalized Laplacian L = I - A_normalized:
            # - smallest eigenvector of L = largest eigenvector of A_normalized = I - L
            # So we run power iteration on (I - L)
            I = torch.eye(n, device=L.device, dtype=L.dtype)
            A_normalized = I - L
            
            # Power iteration to get dominant cluster direction
            _, u1 = power_iteration(A_normalized, n_iters=10)
            
            # Project scores onto u1 and subtract
            proj = torch.dot(scores, u1) * u1
            s_anti = scores - gamma * proj
            
            # Safety: NaN/Inf guard
            if not torch.isfinite(s_anti).all():
                logger.warning("Eigenvector deflation produced non-finite values, returning original scores")
                return scores
            
            return s_anti
            
        except Exception as e:
            logger.warning(f"Eigenvector deflation failed: {e}, returning original scores")
            return scores
    
    # Helper: Estimate query polarity using Probit Soft Voting (training-free, parameter-free)
    def estimate_polarity_from_features(
        ccs_z: float,
        curv_z: float,
        rts_z: float,
        direction: str = "i2t",
        use_rts: bool = True,
        use_ccs: bool = True,
        use_curv: bool = True,
        rtps_z: float = 0.0,
        dhc_z: float = 0.0,
        use_rtps: bool = False,
        use_dhc: bool = False
    ) -> float:
        """
        Probit Soft Voting polarity estimation (training-free, parameter-free).
        
        Converts z-scores to soft votes in [-1, 1] using the Normal CDF (probit transform),
        then computes the unweighted average. This is theoretically grounded:
        each z-score is treated as Gaussian evidence for "helpful vs harmful".
        
        For helpful features (CCS, CURV, DHC): v = 2Φ(z) - 1
          - Large positive z → v ≈ +1 (strongly helpful)
          - Large negative z → v ≈ -1 (against helpful)
          
        For harmful features (RTS, RTPS): v = 2Φ(-z) - 1 (flip sign)
          - Large positive RTS/RTPS → v ≈ -1 (harmful, penalize)
          - Large negative RTS/RTPS → v ≈ +1 (low harm risk)
        
        Default: 3-feature mode (RTS, CCS, CURV) - proven to work best
        Optional: 5-feature mode (add RTPS, DHC) - for ablation comparison
        
        Args:
            ccs_z: Caption Cluster Support (z-scored)
            curv_z: Graph Curvature (z-scored)
            rts_z: Round-Trip Specificity (z-scored)
            direction: Retrieval direction ("i2t" or "t2i")
            use_rts: Whether to include RTS in polarity estimation
            use_ccs: Whether to include CCS in polarity estimation
            use_curv: Whether to include CURV in polarity estimation
            rtps_z: Round-Trip Perturbation Stability (z-scored) - optional for 5-feature mode
            dhc_z: Duplicate/Hub Control (z-scored) - optional for 5-feature mode
            use_rtps: Whether to include RTPS (default False - 3-feature mode)
            use_dhc: Whether to include DHC (default False - 3-feature mode)
        
        Returns:
            Polarity in [-1, 1]: positive = spectral helps, negative = spectral hurts
        """
        try:
            # Normal CDF: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
            def phi(z: float) -> float:
                return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            
            # Collect active votes
            votes = []
            
            # Core 3 features (always available)
            if use_ccs:
                v_ccs = 2.0 * phi(ccs_z) - 1.0      # Large positive CCS → +1 (helpful)
                votes.append(v_ccs)
            
            if use_curv:
                v_curv = 2.0 * phi(curv_z) - 1.0    # Large positive CURV → +1 (helpful)
                votes.append(v_curv)
            
            if use_rts:
                v_rts = 2.0 * phi(-rts_z) - 1.0     # Large positive RTS → -1 (harmful)
                votes.append(v_rts)
            
            # Optional 5-feature extensions (only when explicitly enabled)
            if use_rtps:
                v_rtps = 2.0 * phi(-rtps_z) - 1.0   # Large positive RTPS → -1 (harmful)
                votes.append(v_rtps)
            
            if use_dhc:
                v_dhc = 2.0 * phi(dhc_z) - 1.0      # Large positive DHC → +1 (helpful)
                votes.append(v_dhc)
            
            # Average vote (equal weights, no hyperparameters)
            if len(votes) > 0:
                polarity_raw = sum(votes) / len(votes)
            else:
                # No features selected, return neutral
                return 0.0
            
            # Direction-aware adjustment
            # i2t and a2t: query is image/audio, gallery is text → keep as-is
            # t2i and t2a: query is text, gallery is image/audio → invert
            if direction in ("i2t", "a2t"):
                # I2T/A2T: Keep as-is (high polarity = spectral helps)
                polarity = polarity_raw
            else:  # t2i, t2a
                # T2I/T2A: Invert polarity interpretation
                polarity = -polarity_raw
            
            return polarity
            
        except Exception as e:
            logger.warning(f"Polarity estimation failed: {e}, returning neutral (0.0)")
            return 0.0
    
    # Helper: Estimate query polarity from local geometry (fallback)
    def estimate_polarity_from_graph(
        scores: torch.Tensor,
        L: torch.Tensor,
        A: torch.Tensor,
        direction: str = "i2t"
    ) -> float:
        """
        Fallback polarity estimation from basic graph properties.
        Used when feature scores are not yet computed.
        
        Args:
            scores: Score vector [K]
            L: Normalized Laplacian [K, K]
            A: Adjacency matrix [K, K]
            direction: Retrieval direction ("i2t" or "t2i")
        
        Returns:
            Polarity value (positive = use smoothing, negative = skip)
        """
        try:
            K = len(scores)
            if K < 3:
                return 0.0
            
            # Feature 1: Cluster density around top-1
            top1_idx = torch.argmax(scores).item()
            top1_density = A[top1_idx].sum().item() / (K - 1)
            
            # Feature 2: Score-geometry alignment (curvature)
            D_diag = A.sum(dim=1)
            D_diag = torch.clamp(D_diag, min=1e-8)
            D_inv = 1.0 / D_diag
            As = A @ scores
            curvature = scores - D_inv * As  # [K]
            top1_curvature = curvature[top1_idx].item()
            
            # Feature 3: Score concentration
            score_entropy = -torch.sum(torch.softmax(scores, dim=0) * torch.log_softmax(scores, dim=0)).item()
            max_entropy = torch.log(torch.tensor(float(K))).item()
            normalized_entropy = score_entropy / max_entropy if max_entropy > 0 else 0.5
            
            # Simple formula (less predictive than feature-based)
            # i2t/a2t: query is image/audio, gallery is text → positive polarity = helpful
            # t2i/t2a: query is text, gallery is image/audio → invert
            if direction in ("i2t", "a2t"):
                polarity = (
                    0.5 * top1_density +
                    0.3 * torch.tanh(torch.tensor(top1_curvature)).item() +
                    0.2 * (1.0 - normalized_entropy)
                )
            else:  # t2i, t2a
                polarity = (
                    -0.5 * top1_density +
                    -0.3 * torch.tanh(torch.tensor(top1_curvature)).item() +
                    -0.2 * (1.0 - normalized_entropy)
                )
            
            return polarity
            
        except Exception as e:
            logger.warning(f"Polarity estimation failed: {e}, returning neutral (0.0)")
            return 0.0
    
    # Helper: Fit Bradley-Terry model
    def fit_bradley_terry(W: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Fit Bradley-Terry model for rank consistency.
        
        FIXED: Denominator now uses match counts W[i,j] instead of uniform weights.
        """
        K = len(W)
        theta = np.ones(K) / K
        eps = 1e-8  # Small epsilon to prevent divide by zero
        
        for _ in range(max_iter):
            theta_old = theta.copy()
            for i in range(K):
                wins = W[i, :].sum()  # Total wins for candidate i
                # FIXED: Use W[i,j] match counts in denominator
                denom = sum(W[i, j] / (theta[i] + theta[j] + eps) for j in range(K) if j != i)
                if denom > eps:
                    theta[i] = wins / max(denom, eps)
                else:
                    theta[i] = eps  # Avoid zero
            # Normalize
            theta_sum = theta.sum()
            if theta_sum > eps:
                theta /= theta_sum
            if np.linalg.norm(theta - theta_old) < tol:
                break
        return theta
    
    def analyze_veto_precision(veto_diagnostics: List[Dict]) -> List[Dict]:
        """
        Analyze precision of round-trip veto in different (rt_rank, margin) regions.
        
        Returns list of dicts with:
        - rt_thresh: Round-trip rank threshold
        - margin_thresh: Margin threshold
        - n_queries: Number of queries in this suspicious region
        - pct_queries: Percentage of total queries
        - baseline_error_rate: P(baseline wrong | suspicious)
        - expected_gain: Upper bound on ΔR@1 (assumes perfect alternatives)
        """
        results = []
        total_queries = len(veto_diagnostics)
        
        # Filter out queries with missing rt_rank or baseline_correct
        valid_diagnostics = [
            d for d in veto_diagnostics 
            if d.get('rt_rank') is not None and d.get('baseline_correct') is not None
        ]
        
        if not valid_diagnostics:
            logger.warning("⚠️ No valid diagnostics with rt_rank and baseline_correct")
            return results
        
        logger.info(f"📊 Analyzing veto precision on {len(valid_diagnostics)}/{total_queries} queries with valid data")
        
        # Test different threshold combinations
        rt_thresholds = [20, 30, 50, 100]
        margin_thresholds = [0.01, 0.03, 0.05, 0.10]
        
        for rt_thresh in rt_thresholds:
            for margin_thresh in margin_thresholds:
                # Find queries in this suspicious region
                suspicious = [
                    d for d in valid_diagnostics
                    if d['rt_rank'] > rt_thresh and d['margin'] < margin_thresh
                ]
                
                if len(suspicious) < 10:
                    continue  # Not enough data
                
                # Compute precision: P(baseline wrong | in suspicious region)
                wrong = sum(1 for d in suspicious if not d['baseline_correct'])
                baseline_error_rate = wrong / len(suspicious)
                
                pct_queries = 100.0 * len(suspicious) / total_queries
                expected_gain = baseline_error_rate * len(suspicious) / total_queries
                
                results.append({
                    'rt_thresh': rt_thresh,
                    'margin_thresh': margin_thresh,
                    'n_queries': len(suspicious),
                    'pct_queries': pct_queries,
                    'baseline_error_rate': baseline_error_rate,
                    'expected_gain': expected_gain
                })
        
        # Sort by expected gain (descending)
        results.sort(key=lambda x: -x['expected_gain'])
        
        return results
    
    # =============================================================================
    # TIER 1+2 HELPER FUNCTIONS: Significance Gate with Hierarchical FDR
    # =============================================================================
    
    def rank_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
        """
        Convert scores to rank-normalized [0,1] scale for K/scale invariance.
        Rank 1 (best) → 1.0, Rank K (worst) → 1/K
        """
        K = len(scores)
        # Get ranks (1-indexed): argsort twice gives 0-indexed ranks, add 1
        ranks = torch.argsort(torch.argsort(scores, descending=True)) + 1
        rank_scores = (K - ranks + 1.0) / K  # Linear transform to [1/K, 1.0]
        return rank_scores
    
    def compute_l2o_null_candidates(
        n_cands: int,
        baseline_idx: int,
        challenger_idx: int,
        cand_cand_sim: torch.Tensor,
        M: int,
        null_quantile_start: float = 0.5
    ) -> list:
        """
        Compute L2O (Leave-Two-Out) null candidate set.
        Excludes baseline and challenger neighborhoods to avoid contamination.
        """
        # Start with bottom quantile
        null_start_idx = int(n_cands * null_quantile_start)
        null_candidates_base = list(range(null_start_idx, n_cands))
        
        # Get neighborhoods (top-M neighbors) for baseline and challenger
        baseline_neighbors = set(
            torch.topk(cand_cand_sim[baseline_idx], k=min(M+1, n_cands))[1].cpu().numpy().tolist()
        )
        challenger_neighbors = set(
            torch.topk(cand_cand_sim[challenger_idx], k=min(M+1, n_cands))[1].cpu().numpy().tolist()
        )
        
        # Exclude both neighborhoods (L2O)
        excluded = baseline_neighbors | challenger_neighbors
        null_candidates = [k for k in null_candidates_base if k not in excluded]
        
        return null_candidates
    
    def compute_sigma_with_shrinkage(
        null_scores: torch.Tensor,
        sigma_global: float,
        shrinkage_c: float = 10.0
    ) -> float:
        """
        Compute robust σ with James-Stein shrinkage toward global σ.
        Uses MAD for robustness, shrinks toward global for small nulls.
        """
        if len(null_scores) < 3:
            return sigma_global  # Not enough data, use global
        
        # Compute local σ using MAD
        null_median = torch.median(null_scores)
        mad = torch.median(torch.abs(null_scores - null_median))
        sigma_null_local = 1.4826 * mad.item()
        
        # Fallback to std if MAD is tiny
        if sigma_null_local < 1e-6:
            sigma_null_local = torch.std(null_scores).item() + 1e-6
        
        # James-Stein shrinkage: interpolate between local and global
        lambda_shrink = shrinkage_c / (shrinkage_c + len(null_scores))
        sigma_tilde = np.sqrt(
            (1 - lambda_shrink) * sigma_null_local**2 + 
            lambda_shrink * sigma_global**2
        )
        
        return sigma_tilde
    
    def combine_p_cct(p_vec: list) -> float:
        """
        Cauchy Combination Test (CCT): robust to dependence, powerful for "at least one".
        
        Args:
            p_vec: List of p-values
            
        Returns:
            Combined p-value via CCT
        """
        if not p_vec:
            return 1.0
        
        # Clip to avoid numerical issues
        p = np.clip(np.asarray(p_vec), 1e-12, 1 - 1e-12)
        
        # CCT: transform to Cauchy, average, inverse transform
        t = np.mean(np.tan((0.5 - p) * np.pi))
        p_cct = 0.5 - np.arctan(t) / np.pi
        
        return float(np.clip(p_cct, 1e-12, 1.0))
    
    def storey_bh_robust(
        p_values: list,
        alpha: float = 0.2
    ) -> tuple:
        """
        Storey-BH with robust π₀ estimation and proper step-up mask.
        
        Returns:
            accept_mask: Boolean array indicating which hypotheses to accept
            pi_0_hat: Estimated proportion of true nulls (capped at 0.99)
            diagnostics: Dict with λ-grid, k_accept, etc.
        """
        p = np.asarray(p_values)
        m = len(p)
        
        if m == 0:
            return np.array([], dtype=bool), 1.0, {}
        
        # Grid search for π₀ with smoothing (prevents π₀=1.0 collapse)
        lambda_grid = [0.5, 0.6, 0.7, 0.8, 0.9]
        pi_0_estimates = []
        
        for lam in lambda_grid:
            num_above = np.sum(p > lam)
            pi_0_lam = num_above / ((1 - lam) * m) if (1 - lam) * m > 0 else 1.0
            pi_0_estimates.append(min(pi_0_lam, 1.0))
        
        # Robust smoothing: median of estimates
        pi_0_hat = np.median(pi_0_estimates)
        # Cap at 0.99 to prevent α→0 pathology
        pi_0_hat = min(pi_0_hat, 0.99)
        
        # Adjust α
        alpha_eff = alpha / max(pi_0_hat, 1e-8)
        
        # BH step-up procedure (CORRECT implementation)
        order = np.argsort(p)
        p_sorted = p[order]
        
        # Find largest k with p_(k) <= (k/m)*α_eff
        thresh_vec = (np.arange(1, m + 1) / m) * alpha_eff
        k_candidates = np.where(p_sorted <= thresh_vec)[0]
        
        if len(k_candidates) > 0:
            k_accept = k_candidates.max() + 1  # +1 because of 0-indexing
            accept_mask = np.zeros(m, dtype=bool)
            accept_mask[order[:k_accept]] = True
        else:
            k_accept = 0
            accept_mask = np.zeros(m, dtype=bool)
        
        # Diagnostics
        diagnostics = {
            'lambda_grid': lambda_grid,
            'pi_0_estimates': pi_0_estimates,
            'alpha_eff': alpha_eff,
            'k_accept': k_accept,
            'frac_p_lt_0_05': float(np.mean(p < 0.05)),
            'frac_p_lt_0_1': float(np.mean(p < 0.1)),
            'frac_p_lt_0_2': float(np.mean(p < 0.2)),
            'min_p': float(np.min(p)),
            'median_p': float(np.median(p)),
            'p25': float(np.percentile(p, 25)),
            'p75': float(np.percentile(p, 75))
        }
        
        return accept_mask, pi_0_hat, diagnostics
    
    def compute_one_sided_pvalue(z_eff: float) -> float:
        """
        Compute one-sided p-value for positive z_eff.
        Tests H1: challenger > baseline.
        """
        if z_eff <= 0:
            return 1.0  # No evidence for flip
        else:
            from scipy.stats import norm
            return 1 - norm.cdf(z_eff)
    
    def compute_detrended_sigma(
        null_scores: torch.Tensor,
        null_ranks: torch.Tensor
    ) -> float:
        """
        Compute detrended MAD-based σ by removing linear rank trend.
        
        Args:
            null_scores: Raw scores in the null set
            null_ranks: Corresponding ranks (e.g., 25, 26, ..., 50)
            
        Returns:
            σ from detrended residuals
        """
        if len(null_scores) < 3:
            return 1e-4
        
        # Convert to numpy for polyfit
        r = null_ranks.cpu().numpy()
        s = null_scores.cpu().numpy()
        
        # Fit linear trend: s ~ β₀ + β₁·r
        beta = np.polyfit(r, s, deg=1)
        mu_r = np.polyval(beta, r)
        
        # Compute residuals
        residuals = s - mu_r
        
        # MAD of residuals (detrended σ)
        median_resid = np.median(residuals)
        mad = np.median(np.abs(residuals - median_resid))
        sigma_detrended = 1.4826 * mad
        
        # Fallback to std if MAD is too small
        if sigma_detrended < 1e-6:
            sigma_detrended = np.std(residuals) + 1e-6
        
        return sigma_detrended
    
    # =============================================================================
    # End Tier 1+2 Helper Functions
    # =============================================================================
    
    # =============================================================================
    # STAGE 1A: BUDGET-BASED QUERY SELECTION (BCG)
    # =============================================================================
    # If budget_percentages < 100%, only select top K% of queries for refinement
    # Selection uses Polarity + RSC rank-sum (same as bidirectional_consistency_gating_ablation)
    
    budget_percentages = baseline_params.get('budget_percentages', [1.0])
    frozen_bcg_mode = baseline_params.get('frozen_bcg_mode', None)
    
    # Use the first budget percentage (typically [0.10] for 10%)
    stage1a_budget = budget_percentages[0] if budget_percentages else 1.0
    
    # Create selection mask - default to all selected
    stage1a_selected = np.ones(total_queries, dtype=bool)
    stage1a_selection_scores = np.zeros(total_queries, dtype=np.float64)
    
    if stage1a_budget < 1.0 and frozen_bcg_mode is not None:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STAGE 1A: BCG QUERY SELECTION")
        logger.info(f"   Budget: {stage1a_budget*100:.0f}% ({int(stage1a_budget * total_queries)} of {total_queries} queries)")
        logger.info(f"   Mode: {frozen_bcg_mode}")
        logger.info(f"{'='*80}")
        
        # Compute Polarity scores for ALL queries
        logger.info(f"   Computing Polarity scores for all {total_queries} queries...")
        polarity_result = compute_core_polarity_scores(
            queries=queries,
            gallery=gallery,
            query_ids=query_ids,
            direction=direction,
            topK=topK,
            M_rts=baseline_params.get('M_rts', 24),
            use_rts=True,
            use_curv=True,
            image_bank=image_bank,
            text_bank=text_bank,
            verbose=False,
            aggregation='rank',  # Use rank aggregation for BCG compatibility
            precomputed_scores=precomputed_scores
        )
        polarity_scores = polarity_result['polarity']  # [N] numpy array
        
        # Compute RSC scores for ALL queries (for rank-sum fusion)
        logger.info(f"   Computing RSC scores for all {total_queries} queries...")
        all_query_indices = np.arange(total_queries)
        rsc_scores = fast_cycle_consistency_for_candidates(
            queries=queries,
            gallery=gallery,
            stage1_candidates=all_query_indices,
            direction=direction,
            image_bank=image_bank,
            text_bank=text_bank,
            use_hard_cc=False,  # Soft RSC
            verbose=False
        )
        
        # Apply rank-sum fusion based on frozen_bcg_mode
        from scipy.stats import rankdata
        
        K = int(stage1a_budget * total_queries)
        K = max(1, min(K, total_queries))  # Ensure at least 1, at most all
        
        if frozen_bcg_mode == 'rank_sum_oriented':
            # Rank-sum with label-free sign alignment
            polarity_rank = rankdata(-polarity_scores, method='average')  # Higher polarity = lower rank (better)
            rsc_rank = rankdata(-rsc_scores, method='average')  # Higher RSC = lower rank (better)
            
            # Check correlation to determine if RSC should be inverted
            from scipy.stats import spearmanr
            corr, _ = spearmanr(polarity_rank, rsc_rank)
            
            if corr < 0:
                # Invert RSC if negatively correlated with polarity
                rsc_rank = rankdata(rsc_scores, method='average')  # Flip direction
                logger.info(f"   RSC inverted (rank correlation with polarity: {corr:.3f})")
            else:
                logger.info(f"   RSC kept as-is (rank correlation with polarity: {corr:.3f})")
            
            # Rank-sum score: lower = better (more uncertain, should refine)
            stage1a_selection_scores = polarity_rank + rsc_rank
            
        elif frozen_bcg_mode == 'rank_sum':
            # Simple rank-sum without orientation
            polarity_rank = rankdata(-polarity_scores, method='average')
            rsc_rank = rankdata(-rsc_scores, method='average')
            stage1a_selection_scores = polarity_rank + rsc_rank
            
        elif frozen_bcg_mode == 'polarity_only':
            # Only use polarity
            stage1a_selection_scores = rankdata(-polarity_scores, method='average')
            
        elif frozen_bcg_mode == 'rsc_only':
            # Only use RSC
            stage1a_selection_scores = rankdata(-rsc_scores, method='average')
            
        else:
            # Default: rank_sum_oriented
            polarity_rank = rankdata(-polarity_scores, method='average')
            rsc_rank = rankdata(-rsc_scores, method='average')
            stage1a_selection_scores = polarity_rank + rsc_rank
        
        # Select top K queries (lowest selection scores = most uncertain = should refine)
        selected_indices = np.argsort(stage1a_selection_scores)[:K]
        stage1a_selected = np.zeros(total_queries, dtype=bool)
        stage1a_selected[selected_indices] = True
        
        logger.info(f"   ✅ Selected {stage1a_selected.sum()} queries for spectral refinement")
        logger.info(f"   Selection score range: [{stage1a_selection_scores.min():.2f}, {stage1a_selection_scores.max():.2f}]")
    else:
        logger.info(f"\n📊 STAGE 1A: All queries selected (budget={stage1a_budget*100:.0f}%, mode={frozen_bcg_mode})")
    
    # Store Stage 1a stats for reporting
    stage1a_stats = {
        'budget': stage1a_budget,
        'n_selected': int(stage1a_selected.sum()),
        'n_total': total_queries,
        'actual_rate': float(stage1a_selected.sum()) / total_queries if total_queries > 0 else 0.0,
        'frozen_bcg_mode': frozen_bcg_mode,
    }
    
    # =============================================================================
    # MULTI-PASS STRUCTURE: Collect flip attempts, then apply Hierarchical FDR
    # =============================================================================
    
    # Check if using new significance gate
    use_significance_gate = baseline_params.get('use_significance_gate', False)
    
    # Initialize FDR counters
    flip_attempts_fdr = 0
    flips_passed_fdr = 0
    
    # Initialize spectral_decisions (for decision transplant to NNN/QB-Norm)
    spectral_decisions = []
    
    logger.info(f"🔍 DEBUG: return_decisions = {return_decisions}, will {'COLLECT' if return_decisions else 'NOT collect'} decisions for transplant")
    
    if use_significance_gate:
        spectral_mode = baseline_params.get('spectral_mode', 'B')  # 'A' or 'B'
        
        if spectral_mode == 'A':
            logger.info(f"🎯 Using Spectral Refinement A (Variance-Stabilized, Minimal Gate)")
            logger.info(f"   - Method: Z-space fusion (variance-stabilized), argmax selection")
            logger.info(f"   - λ_fused: {baseline_params.get('lambda_fused', 0.6)} (tuned for power)")
            logger.info(f"   - Mask: Union of top-{min(12, baseline_params.get('topK', 50)//4)} spectral + top-{min(8, baseline_params.get('topK', 50)//6)} votes (NO top-RAW)")
            logger.info(f"   - Challengers: B=1 (argmax z_fused)")
            logger.info(f"   - Safety: Δz_fused ≥ {baseline_params.get('tau_z_fused', 0.0)} (minimal guard)")
            logger.info(f"   - Global cap: Top {baseline_params.get('flip_rate_cap', 0.15)*100:.0f}% by Δz_fused")
            logger.info(f"   - NO multi-testing (CCT/BH), direct rerank with safety check")
        else:
            logger.info(f"🎯 Using Spectral Refinement B (Tiered Z-Test with Dual Decision)")
            logger.info(f"   - Method: Variance-stabilized z-space, tiered testing")
            logger.info(f"   - λ_fused: {baseline_params.get('lambda_fused', 0.6)} (adaptive per-query option)")
            logger.info(f"   - Mask: Union of top-{min(12, baseline_params.get('topK', 50)//4)} spectral + top-{min(8, baseline_params.get('topK', 50)//6)} votes (NO top-RAW)")
            logger.info(f"   - Tiers: Near (2-5), Deep (≥10); B=1 per tier")
            logger.info(f"   - Within-query: Fixed p ≤ {baseline_params.get('alpha_within_near', 0.10)} (near), ≤ {baseline_params.get('alpha_within_deep', 0.20)} (deep)")
            logger.info(f"   - Dual decision: Depth constraint for rank >35 (Δz ≥ 2.0)")
            logger.info(f"   - Across-query: Optional Storey-BH at α={baseline_params.get('fdr_alpha_out', 0.15)}")
        
        # Will store flip attempts for hierarchical FDR
        all_flip_attempts = []  # List of dicts with query_idx, p_values, challengers, etc.
        global_sigma_mads = []  # For computing global σ (Pass 0)
    
    # Track queries that actually went through spectral refinement
    stage1a_processed_count = 0
    stage1a_skipped_count = 0
    
    # Process each query with progress bar
    with tqdm(total=total_queries, desc="Spectral Refinement", unit="query") as pbar:
        for i, (query, query_id) in enumerate(zip(queries, query_ids)):
            # Use precomputed scores if provided, otherwise compute CLIP scores
            if precomputed_scores is not None:
                s_clip = precomputed_scores[i].squeeze()  # [N_gallery]
                # Precomputed scores already have any necessary scaling applied
            else:
                # Compute base CLIP scores
                s_clip = (query.float() @ gallery.float().T).squeeze()  # [N_gallery]
                
                # Apply model-specific logit scaling if using SigLIP and flag is set
                if apply_siglip_scale and baseline_params.get('backbone_type') == 'siglip':
                    s_clip = s_clip * siglip_logit_scale
            
            # Get top-K candidates
            K_actual = min(topK, s_clip.size(0))
            topk_vals, topk_idx = torch.topk(s_clip, k=K_actual, dim=0)
            
            # Store base rankings (top-50) for debugging - convert to gallery IDs
            base_top50_idx = topk_idx[:50].cpu().numpy() if len(topk_idx) >= 50 else topk_idx.cpu().numpy()
            # Pad to 50 if needed
            if len(base_top50_idx) < 50:
                padded = np.zeros(50, dtype=base_top50_idx.dtype)
                padded[:len(base_top50_idx)] = base_top50_idx
                base_top50_idx = padded
            base_rankings.append(base_top50_idx)
            
            # =========================================================================
            # STAGE 1A CHECK: Skip non-selected queries
            # =========================================================================
            if not stage1a_selected[i]:
                # Query not selected by Stage 1a BCG - use baseline scores directly
                stage1a_skipped_count += 1
                
                # Append baseline scores as fused (no refinement applied)
                fused_scores.append(s_clip)
                
                # Store empty diagnostics for skipped query
                spectral_diagnostics.append({
                    'query_id': query_id,
                    'stage1a_selected': False,
                    'base_top1_idx': int(base_top50_idx[0]),
                })
                
                # Store empty decision (for transplant compatibility)
                if return_decisions:
                    spectral_decisions.append({
                        'query_id': query_id,
                        'query_idx': i,
                        'apply_spectral': False,
                        'stage1a_selected': False,
                        'refined_top1_idx': int(base_top50_idx[0]),
                        'base_top1_idx': int(base_top50_idx[0]),
                    })
                
                pbar.update(1)
                continue
            
            # Query selected by Stage 1a - proceed with spectral refinement
            stage1a_processed_count += 1
            
            # Extract candidate embeddings (keep on GPU)
            candidate_embeddings = gallery_gpu[topk_idx]  # [K, d] - on GPU
            query_emb = query.squeeze()  # [d] - on GPU
            
            # Initialize feature scores (on GPU)
            n_cands = len(topk_idx)
            
            # Initialize diagnostics for this query
            query_diagnostics = {
                'query_id': query_id,
                'base_top1_idx': int(base_top50_idx[0]),
                'base_top1_score': float(topk_vals[0].item()),
                'n_candidates': n_cands,
                'features': {}
            }
            rts_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            ccs_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            rtps_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            contra_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            dhc_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            curv_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            rtrc_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 1. Round-Trip Specificity (RTS) - GPU-accelerated
            # FIXED: For i2t, RTS uses IMAGE neighbors from image_bank (not text gallery).
            # For t2i, RTS uses TEXT neighbors from text_bank (not image gallery).
            if use_rts and direction == "i2t":
                # For i2t: query is image, candidates are text captions
                query_img_emb = query_emb  # [d] - image embedding on GPU
                
                # CRITICAL: Find image neighbors from image_bank (not text gallery!)
                if image_bank_gpu is not None:
                    # Find M visually similar images from image bank (excluding self)
                    img_sims = image_bank_gpu @ query_img_emb  # [N_img]
                    
                    # If query is in image_bank, need to exclude self-match
                    # Take top M+1 and skip the first (which is likely the query itself)
                    M_actual = min(M_rts + 1, len(image_bank_gpu))
                    _, topM_img_idx = torch.topk(img_sims, k=M_actual, dim=0)
                    
                    # Skip first neighbor if it's likely a self-match (sim > 0.999)
                    if len(topM_img_idx) > 1 and img_sims[topM_img_idx[0]] > 0.999:
                        topM_img_idx = topM_img_idx[1:]  # Exclude self
                    else:
                        topM_img_idx = topM_img_idx[:M_rts]  # Keep first M
                    
                    visual_neighbors = image_bank_gpu[topM_img_idx]  # [M, d] - image embeddings
                    
                    # Score each candidate TEXT caption against:
                    # 1. Query image (direct image→text similarity)
                    query_scores = candidate_embeddings @ query_img_emb  # [K] - caption @ image
                    
                    # 2. Image neighbors (text→image similarity with visual distractors)
                    # This measures: does this caption also match other images similar to the query?
                    # If yes, it's less specific (generic caption)
                    distractor_scores = candidate_embeddings @ visual_neighbors.T  # [K, M] - caption @ image_neighbors
                    distractor_bests = distractor_scores.max(dim=1)[0]  # [K] - best match to any image distractor
                    
                    # RTS = specificity margin: higher = more specific to query image
                    rts_scores = query_scores - distractor_bests  # [K] - margin
                    
                    rts_z = robust_zscore_gpu(rts_scores)
                else:
                    # Fallback: disable RTS if image bank not available
                    logger.warning(f"Query {i}: Image bank not available for RTS, using zeros")
                    rts_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            elif use_rts and direction == "t2i":
                # For t2i: query is text, candidates are images
                query_txt_emb = query_emb  # [d] - text embedding on GPU
                
                # CRITICAL: Find text neighbors from text_bank (not image gallery!)
                if text_bank_gpu is not None:
                    # Find M text neighbors from text bank (excluding self)
                    text_sims = text_bank_gpu @ query_txt_emb  # [N_text]
                    
                    # If query is in text_bank, need to exclude self-match
                    # Take top M+1 and skip the first (which is likely the query itself)
                    M_actual = min(M_rts + 1, len(text_bank_gpu))
                    _, topM_txt_idx = torch.topk(text_sims, k=M_actual, dim=0)
                    
                    # Skip first neighbor if it's likely a self-match (sim > 0.999)
                    if len(topM_txt_idx) > 1 and text_sims[topM_txt_idx[0]] > 0.999:
                        topM_txt_idx = topM_txt_idx[1:]  # Exclude self
                    else:
                        topM_txt_idx = topM_txt_idx[:M_rts]  # Keep first M
                    
                    text_neighbors = text_bank_gpu[topM_txt_idx]  # [M, d] - text embeddings
                    
                    # Build mini-gallery: query text + M text neighbors
                    mini_gallery = torch.cat([query_txt_emb.unsqueeze(0), text_neighbors], dim=0)  # [M+1, d]
                    
                    # Score all candidates (images) against mini-gallery (texts)
                    scores_all = candidate_embeddings @ mini_gallery.T  # [K, M+1]
                    query_scores = scores_all[:, 0]  # [K] - image @ query_text
                    distractor_bests = scores_all[:, 1:].max(dim=1)[0]  # [K] - best match to any text distractor
                    
                    # RTS = specificity margin: higher = more specific to query text
                    rts_scores = query_scores - distractor_bests  # [K] - margin
                    
                    rts_z = robust_zscore_gpu(rts_scores)
                else:
                    # Fallback: disable RTS if text bank not available
                    logger.warning(f"Query {i}: Text bank not available for RTS, using zeros")
                    rts_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            elif use_rts and direction == "a2t":
                # For a2t: query is audio, candidates are text captions (same logic as i2t)
                query_audio_emb = query_emb  # [d] - audio embedding on GPU
                
                # Find audio neighbors from audio_bank (same as image_bank logic)
                if image_bank_gpu is not None:  # image_bank_gpu serves as audio_bank for a2t
                    # Find M audio-similar samples from audio bank
                    audio_sims = image_bank_gpu @ query_audio_emb  # [N_audio]
                    
                    M_actual = min(M_rts + 1, len(image_bank_gpu))
                    _, topM_audio_idx = torch.topk(audio_sims, k=M_actual, dim=0)
                    
                    # Skip self-match if present
                    if len(topM_audio_idx) > 1 and audio_sims[topM_audio_idx[0]] > 0.999:
                        topM_audio_idx = topM_audio_idx[1:]
                    else:
                        topM_audio_idx = topM_audio_idx[:M_rts]
                    
                    audio_neighbors = image_bank_gpu[topM_audio_idx]  # [M, d]
                    
                    # Score each candidate TEXT caption
                    query_scores = candidate_embeddings @ query_audio_emb  # [K]
                    distractor_scores = candidate_embeddings @ audio_neighbors.T  # [K, M]
                    distractor_bests = distractor_scores.max(dim=1)[0]  # [K]
                    
                    rts_scores = query_scores - distractor_bests  # Higher = more specific
                    rts_z = robust_zscore_gpu(rts_scores)
                else:
                    logger.warning(f"Query {i}: Audio bank not available for RTS, using zeros")
                    rts_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            elif use_rts and direction == "t2a":
                # For t2a: query is text, candidates are audio (same logic as t2i)
                query_txt_emb = query_emb  # [d] - text embedding on GPU
                
                # Find text neighbors from text_bank
                if text_bank_gpu is not None:
                    text_sims = text_bank_gpu @ query_txt_emb  # [N_text]
                    
                    M_actual = min(M_rts + 1, len(text_bank_gpu))
                    _, topM_txt_idx = torch.topk(text_sims, k=M_actual, dim=0)
                    
                    # Skip self-match if present
                    if len(topM_txt_idx) > 1 and text_sims[topM_txt_idx[0]] > 0.999:
                        topM_txt_idx = topM_txt_idx[1:]
                    else:
                        topM_txt_idx = topM_txt_idx[:M_rts]
                    
                    text_neighbors = text_bank_gpu[topM_txt_idx]  # [M, d]
                    
                    # Build mini-gallery: query text + M text neighbors
                    mini_gallery = torch.cat([query_txt_emb.unsqueeze(0), text_neighbors], dim=0)  # [M+1, d]
                    
                    # Score all candidates (audio) against mini-gallery (texts)
                    scores_all = candidate_embeddings @ mini_gallery.T  # [K, M+1]
                    query_scores = scores_all[:, 0]  # [K]
                    distractor_bests = scores_all[:, 1:].max(dim=1)[0]  # [K]
                    
                    rts_scores = query_scores - distractor_bests
                    rts_z = robust_zscore_gpu(rts_scores)
                else:
                    logger.warning(f"Query {i}: Text bank not available for RTS, using zeros")
                    rts_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            else:
                rts_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 2. Caption Cluster Support (CCS) - GPU-accelerated
            # FIXED: Use soft kernel (relu or exp) instead of hard threshold, and cap contribution
            if use_ccs:
                # Build similarity matrix on caption embeddings (GPU)
                k_ccs = baseline_params.get('k_ccs', min(12, n_cands - 1))
                # Compute pairwise similarities
                similarities = candidate_embeddings @ candidate_embeddings.T  # [K, K]
                
                # Soft kernel: A = relu(C - tau) or A = exp(gamma * (C - tau))
                tau_ccs = baseline_params.get('tau_ccs', 0.7)  # Similarity threshold
                ccs_kernel = baseline_params.get('ccs_kernel', 'relu')  # 'relu' or 'exp'
                gamma_ccs = baseline_params.get('gamma_ccs', 2.0)  # For exp kernel
                
                if ccs_kernel == 'relu':
                    # Soft threshold: A = relu(C - tau)
                    A_ccs = torch.clamp(similarities - tau_ccs, min=0.0)
                elif ccs_kernel == 'exp':
                    # Exponential kernel: A = exp(gamma * (C - tau))
                    A_ccs = torch.exp(gamma_ccs * (similarities - tau_ccs))
                    # Clip very large values for numerical stability
                    A_ccs = torch.clamp(A_ccs, max=10.0)
                else:
                    # Fallback to hard threshold
                    A_ccs = (similarities > tau_ccs).to(query.dtype)
                
                # Mask diagonal
                mask_diag = torch.eye(n_cands, device=device, dtype=torch.bool)
                A_ccs[mask_diag] = 0.0
                
                # Compute eigenvector centrality on normalized adjacency (GPU)
                try:
                    # Normalize for eigenvector centrality: D^{-1/2} A D^{-1/2}
                    D_diag = A_ccs.sum(dim=1)
                    D_diag = torch.clamp(D_diag, min=1e-8)
                    D_sqrt_inv = 1.0 / torch.sqrt(D_diag)
                    D_sqrt_inv_diag = torch.diag(D_sqrt_inv)
                    A_normalized = D_sqrt_inv_diag @ A_ccs @ D_sqrt_inv_diag
                    
                    # OPTIMIZED: Use power iteration instead of full eigendecomposition
                    # Power iteration is O(K² × n_iters) vs O(K³) for eigh
                    _, ccs_eigen = power_iteration(A_normalized, n_iters=10)
                    ccs_z = robust_zscore_gpu(ccs_eigen)
                    
                    # FIXED: Cap CCS contribution to prevent over-boosting
                    ccs_cap = baseline_params.get('ccs_cap', 1.5)  # Cap at ±1.5 std
                    ccs_z = torch.clamp(ccs_z, min=-ccs_cap, max=ccs_cap)
                    
                    # For i2t, optionally use anti-hub (negate) if w_ccs is negative
                    if direction == "i2t" and baseline_params.get('ccs_anti_hub', False):
                        ccs_z = -ccs_z  # Anti-hub: penalize high centrality
                except:
                    # Fallback to degree centrality
                    degree = A_ccs.sum(dim=1)
                    ccs_z = robust_zscore_gpu(degree)
                    # Cap even in fallback
                    ccs_cap = baseline_params.get('ccs_cap', 1.5)
                    ccs_z = torch.clamp(ccs_z, min=-ccs_cap, max=ccs_cap)
                    if direction == "i2t" and baseline_params.get('ccs_anti_hub', False):
                        ccs_z = -ccs_z  # Anti-hub
            else:
                ccs_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # Compute base scores once (used by multiple features)
            base_scores = candidate_embeddings @ query_emb  # [K]
            
            # 3. Round-Trip Paraphrase Stability (RTPS) - GPU-accelerated (VECTORIZED)
            # FIXED: For i2t, mask candidate TEXT embeddings (not query image).
            # RTPS measures caption token indispensability by dropping dimensions from candidate text vectors.
            if use_rtps and direction == "i2t":
                drop_ratio = baseline_params.get('rtps_drop_ratio', 0.1)  # 0.1 for text token drops
                d = candidate_embeddings.shape[1]  # Dimension of text embeddings
                
                # Deterministic seed from query_id
                import hashlib
                query_hash = int(hashlib.md5(query_id.encode()).hexdigest()[:8], 16) % (2**32)
                rng = np.random.default_rng(query_hash)
                
                # Generate n_drops_rtps masks for each candidate (vectorized)
                # Each mask drops ~10% of dimensions from candidate text embeddings
                masks = (torch.rand(n_drops_rtps, d, device=device, generator=torch.Generator(device=device).manual_seed(query_hash % (2**31))) > drop_ratio).to(query.dtype)
                
                # Apply masks to candidate embeddings: [K, d] -> [n_drops, K, d]
                # For each drop, mask all candidates
                candidates_masked = candidate_embeddings.unsqueeze(0) * masks.unsqueeze(1)  # [n_drops, K, d]
                
                # Score masked candidates against query image: [n_drops, K, d] @ [d] -> [n_drops, K]
                drop_scores = candidates_masked @ query_emb  # [n_drops, K]
                
                # Compute stability: base_scores - mean(drop_scores) for each candidate
                mean_drop_scores = drop_scores.mean(dim=0)  # [K] - mean across drops
                stability_scores = base_scores - mean_drop_scores  # [K] - higher = more stable
                
                rtps_z = robust_zscore_gpu(stability_scores)
            elif use_rtps and direction == "t2i":
                # For t2i: mask candidate IMAGE embeddings (not query text) - VECTORIZED
                # FIXED: Perturb candidate images, not query text
                drop_ratio = baseline_params.get('rtps_drop_ratio', 0.1)
                d = candidate_embeddings.shape[1]  # Dimension of image embeddings
                
                # Deterministic seed from query_id
                import hashlib
                query_hash = int(hashlib.md5(query_id.encode()).hexdigest()[:8], 16) % (2**32)
                rng = np.random.default_rng(query_hash)
                
                # Generate n_drops_rtps masks for each candidate (vectorized)
                masks = (torch.rand(n_drops_rtps, d, device=device, generator=torch.Generator(device=device).manual_seed(query_hash % (2**31))) > drop_ratio).to(query.dtype)
                
                # Apply masks to candidate embeddings: [K, d] -> [n_drops, K, d]
                candidates_masked = candidate_embeddings.unsqueeze(0) * masks.unsqueeze(1)  # [n_drops, K, d]
                
                # Score masked candidates against query text: [n_drops, K, d] @ [d] -> [n_drops, K]
                drop_scores = candidates_masked @ query_emb  # [n_drops, K]
                
                # Compute stability: base_scores - mean(drop_scores) for each candidate
                mean_drop_scores = drop_scores.mean(dim=0)  # [K] - mean across drops
                stability_scores = base_scores - mean_drop_scores  # [K] - higher = more stable
                
                rtps_z = robust_zscore_gpu(stability_scores)
            elif use_rtps and direction == "a2t":
                # For a2t: mask candidate TEXT embeddings (same as i2t)
                drop_ratio = baseline_params.get('rtps_drop_ratio', 0.1)
                d = candidate_embeddings.shape[1]  # Dimension of text embeddings
                
                import hashlib
                query_hash = int(hashlib.md5(query_id.encode()).hexdigest()[:8], 16) % (2**32)
                
                # Generate masks
                masks = (torch.rand(n_drops_rtps, d, device=device, generator=torch.Generator(device=device).manual_seed(query_hash % (2**31))) > drop_ratio).to(query.dtype)
                
                # Apply masks to candidate embeddings
                candidates_masked = candidate_embeddings.unsqueeze(0) * masks.unsqueeze(1)  # [n_drops, K, d]
                
                # Score masked candidates against query audio
                drop_scores = candidates_masked @ query_emb  # [n_drops, K]
                
                # Compute stability
                mean_drop_scores = drop_scores.mean(dim=0)  # [K]
                stability_scores = base_scores - mean_drop_scores  # [K]
                
                rtps_z = robust_zscore_gpu(stability_scores)
            elif use_rtps and direction == "t2a":
                # For t2a: mask candidate AUDIO embeddings (same as t2i)
                drop_ratio = baseline_params.get('rtps_drop_ratio', 0.1)
                d = candidate_embeddings.shape[1]  # Dimension of audio embeddings
                
                import hashlib
                query_hash = int(hashlib.md5(query_id.encode()).hexdigest()[:8], 16) % (2**32)
                
                # Generate masks
                masks = (torch.rand(n_drops_rtps, d, device=device, generator=torch.Generator(device=device).manual_seed(query_hash % (2**31))) > drop_ratio).to(query.dtype)
                
                # Apply masks to candidate embeddings
                candidates_masked = candidate_embeddings.unsqueeze(0) * masks.unsqueeze(1)  # [n_drops, K, d]
                
                # Score masked candidates against query text
                drop_scores = candidates_masked @ query_emb  # [n_drops, K]
                
                # Compute stability
                mean_drop_scores = drop_scores.mean(dim=0)  # [K]
                stability_scores = base_scores - mean_drop_scores  # [K]
                
                rtps_z = robust_zscore_gpu(stability_scores)
            else:
                rtps_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 4. Contradiction Penalty (CONTRA) - Lightweight lexical contradiction detection
            # For i2t: detect contradictions between image attributes (from query) and caption text
            # Simplified implementation: use precomputed attribute embeddings if available
            if use_contra:
                # Check if we have contra bank in baseline_params
                contra_bank = baseline_params.get('contra_bank', None)
                contra_embs = baseline_params.get('contra_embs', None)  # Precomputed attribute embeddings
                
                if contra_embs is not None and contra_bank is not None:
                    # contra_embs: [n_attrs, d] - attribute embeddings
                    # contra_bank: dict mapping category -> list of attribute words
                    contra_embs_gpu = torch.tensor(contra_embs, device=device, dtype=query.dtype)
                    
                    # Get image attribute priors (for i2t, query is image)
                    img_attr_scores = contra_embs_gpu @ query_emb  # [n_attrs]
                    # Pick argmax per category (simplified)
                    # For now, just use top attributes
                    top_attrs = torch.topk(img_attr_scores, k=min(3, len(img_attr_scores)), dim=0).indices
                    
                    # Score candidates against these attributes
                    # If candidate matches contradictory attributes, apply penalty
                    # Simplified: penalize if candidate has low similarity to top image attributes
                    cand_attr_scores = candidate_embeddings @ contra_embs_gpu[top_attrs].T  # [K, n_top]
                    # Penalty if candidate doesn't match image attributes well
                    contra = -lambda_contra * (1.0 - cand_attr_scores.mean(dim=1))  # Negative penalty
                    contra_raw = contra
                else:
                    # No contra bank available - return zeros
                    contra_raw = torch.zeros(n_cands, device=device, dtype=query.dtype)
            else:
                contra_raw = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 5. Duplication/Hallucination Control (DHC) - GPU-accelerated, FULLY VECTORIZED
            # FIXED: Use mutual kNN with small k (2-3) instead of single NN with hard >0.95 cutoff
            if use_dhc:
                # Vectorized: compute all similarities at once
                similarities_all = candidate_embeddings @ candidate_embeddings.T  # [K, K]
                # Mask diagonal
                mask_diag = torch.eye(n_cands, device=device, dtype=torch.bool)
                similarities_all[mask_diag] = float('-inf')
                
                # Use mutual kNN with small k (2-3) for robustness
                k_dhc = baseline_params.get('k_dhc', 2)  # Small k for mutual neighbors
                k_dhc = min(k_dhc, max(1, n_cands - 1))
                
                if k_dhc > 0 and n_cands > 1:
                    # Find top-k neighbors for each candidate
                    topk_sims, topk_indices = torch.topk(similarities_all, k=k_dhc, dim=1)  # [K, k_dhc]
                    
                    # ====================================================================
                    # VECTORIZED: Compute mutual kNN counts using scatter operations
                    # ====================================================================
                    # Build forward adjacency: forward[i, j] = 1 if j in kNN(i)
                    row_idx = torch.arange(n_cands, device=device).unsqueeze(1).expand(-1, k_dhc).flatten()  # [K*k]
                    col_idx = topk_indices.flatten()  # [K*k]
                    
                    forward = torch.zeros((n_cands, n_cands), device=device, dtype=query.dtype)
                    forward[row_idx, col_idx] = 1.0
                    
                    # Mutual = forward AND forward.T (element-wise)
                    mutual = forward * forward.T  # [K, K]
                    
                    # Count mutual neighbors for each candidate (sum over row, excluding diagonal)
                    mutual_counts = mutual.sum(dim=1)  # [K]
                    
                    # Penalty: negative of mutual count (higher mutual count = more duplication = more penalty)
                    # Normalize by k_dhc to get fraction
                    dhc = -lambda_dhc * (mutual_counts / k_dhc)  # [K] - negative penalty
                    dhc_raw = dhc
                else:
                    dhc_raw = torch.zeros(n_cands, device=device, dtype=query.dtype)
            else:
                dhc_raw = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 6. Graph Curvature (CURV) - GPU-accelerated - FIXED: zero-mean centered residual
            if use_curv:
                try:
                    A_curv = build_mutual_knn_graph(candidate_embeddings, k_curv)
                    
                    # FIXED: Use centered neighbor residual (zero-mean by construction)
                    # Compute mean of neighbors for each node
                    neighbor_sums = A_curv @ base_scores  # [K]
                    neighbor_counts = A_curv.sum(dim=1)  # [K]
                    neighbor_counts = torch.clamp(neighbor_counts, min=1e-8)
                    nbr_mean = neighbor_sums / neighbor_counts  # [K]
                    
                    # Curvature as deviation from neighbor mean (zero-mean-ish)
                    kappa = base_scores - nbr_mean  # [K]
                    curv_z = robust_zscore_gpu(kappa)
                except:
                    curv_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            else:
                curv_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 7. Round-Trip Rank Consistency (RTRC) - GPU-accelerated
            # FIXED: Replace Bradley-Terry with simple rank-based feature
            # BT was collapsing because candidates are already sorted by base_scores
            if use_rtrc:
                # Simple rank-based feature: higher rank = higher base score
                # Candidates are sorted descending, so rank-0 = best
                ranks = torch.arange(n_cands, device=device, dtype=query.dtype)  # [0, 1, 2, ..., K-1]
                rtrc_raw = -ranks  # [-0, -1, -2, ..., -(K-1)] - top ranks get positive boost after centering
                rtrc_raw = rtrc_raw - rtrc_raw.mean()  # Zero-mean: top gets +ve, bottom gets -ve
            else:
                rtrc_raw = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # Combine features with weights (only active features) - GPU
            # FIXED: Use correct signs, raw penalties, and CCS-RTS interaction
            feature_scores = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # CCS-RTS interaction: only count CCS if RTS is decent (rectifier)
            ccs_gate = torch.clamp(rts_z, 0, None) * ccs_z if use_ccs else torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            for name, weight in weights_dict.items():
                if name == 'rts':
                    feature_scores += weight * rts_z
                    query_diagnostics['features']['rts_z'] = {
                        'mean': float(rts_z.mean().item()),
                        'std': float(rts_z.std().item()),
                        'max': float(rts_z.max().item()),
                        'min': float(rts_z.min().item())
                    }
                elif name == 'ccs':
                    # Use gated CCS (only when RTS is positive)
                    feature_scores += weight * ccs_gate
                    query_diagnostics['features']['ccs_z'] = {
                        'mean': float(ccs_z.mean().item()),
                        'std': float(ccs_z.std().item()),
                        'max': float(ccs_z.max().item()),
                        'min': float(ccs_z.min().item())
                    }
                    query_diagnostics['features']['ccs_gate'] = {
                        'mean': float(ccs_gate.mean().item()),
                        'std': float(ccs_gate.std().item())
                    }
                elif name == 'rtps':
                    feature_scores += weight * rtps_z
                    query_diagnostics['features']['rtps_z'] = {
                        'mean': float(rtps_z.mean().item()),
                        'std': float(rtps_z.std().item()),
                        'max': float(rtps_z.max().item()),
                        'min': float(rtps_z.min().item())
                    }
                elif name == 'contra':
                    # CONTRA is a penalty (negative values) - use raw, not z-scored
                    feature_scores += weight * contra_raw
                    query_diagnostics['features']['contra_z'] = {
                        'mean': float(contra_raw.mean().item()),
                        'std': float(contra_raw.std().item()),
                        'max': float(contra_raw.max().item()),
                        'min': float(contra_raw.min().item())
                    }
                elif name == 'dhc':
                    # DHC is a penalty (negative values) - use raw, not z-scored
                    feature_scores += weight * dhc_raw
                    query_diagnostics['features']['dhc_z'] = {
                        'mean': float(dhc_raw.mean().item()),
                        'std': float(dhc_raw.std().item()),
                        'max': float(dhc_raw.max().item()),
                        'min': float(dhc_raw.min().item())
                    }
                elif name == 'curv':
                    # Penalize high curvature (negative for high curv_z)
                    curv_penalty = -torch.clamp(curv_z, 0, None)  # Only penalize positive curvature
                    feature_scores += weight * curv_penalty
                    query_diagnostics['features']['curv_z'] = {
                        'mean': float(curv_z.mean().item()),
                        'std': float(curv_z.std().item()),
                        'max': float(curv_z.max().item()),
                        'min': float(curv_z.min().item())
                    }
                elif name == 'rtrc':
                    # RTRC is zero-mean raw (not z-scored)
                    feature_scores += weight * rtrc_raw
                    query_diagnostics['features']['rtrc_z'] = {
                        'mean': float(rtrc_raw.mean().item()),
                        'std': float(rtrc_raw.std().item()),
                        'max': float(rtrc_raw.max().item()),
                        'min': float(rtrc_raw.min().item())
                    }
            
            # Store feature_scores BEFORE normalization for diagnostics
            feature_scores_raw = feature_scores.clone()
            query_diagnostics['feature_scores'] = {
                'mean': float(feature_scores_raw.mean().item()),
                'std': float(feature_scores_raw.std().item()),
                'max': float(feature_scores_raw.max().item()),
                'min': float(feature_scores_raw.min().item())
            }
            
            # FIXED: Normalize feature_scores to prevent huge values from dominating
            # This ensures all features contribute proportionally
            # Only normalize if there's meaningful variation
            if feature_scores.std() > 1e-8:
                feature_scores = robust_zscore_gpu(feature_scores)
            else:
                # If no variation, set to zeros (no feature signal)
                feature_scores = torch.zeros_like(feature_scores)
            
            # Build multi-layer graph for spectral filtering (GPU-accelerated)
            try:
                # Build mutual k-NN graph (single similarity computation)
                A = build_mutual_knn_graph(candidate_embeddings, k=k_curv)
                
                # Feature-weighted edges: pure feature affinity (0..1), not squared similarity
                if feature_scores.max() > feature_scores.min():
                    feature_affinity = torch.outer(feature_scores, feature_scores)
                    feature_affinity = (feature_affinity - feature_affinity.min()) / (feature_affinity.max() - feature_affinity.min() + 1e-8)
                    # Reweight edges: combine structure with feature affinity
                    A = A * (1.0 + 0.2 * feature_affinity)  # 20% boost for high-feature-score pairs
                
                # Compute Laplacian
                L = normalized_laplacian(A)
                
                # ============================================================================
                # POLARITY-AWARE OPERATOR SELECTION
                # ============================================================================
                
                # Get configuration parameters
                use_polarity_aware = baseline_params.get('use_polarity_aware', False)
                polarity_threshold = baseline_params.get('polarity_threshold', 0.1)
                negative_polarity_method = baseline_params.get('negative_polarity_method', 'none')  # 'none', 'disagreement', 'anti_spectral'
                lambda_disagreement = baseline_params.get('lambda_disagreement', 0.15)
                gamma_smoothing = baseline_params.get('gamma_smoothing', 0.3)
                gamma_sharpening = baseline_params.get('gamma_sharpening', 0.3)
                
                # Polarity feature selection (for ablation)
                # Default: 3-feature mode (RTS, CCS, CURV) - proven best
                polarity_use_rts = baseline_params.get('polarity_use_rts', True)
                polarity_use_ccs = baseline_params.get('polarity_use_ccs', True)
                polarity_use_curv = baseline_params.get('polarity_use_curv', True)
                # Optional 5-feature mode (add RTPS, DHC) - for ablation only
                polarity_use_rtps = baseline_params.get('polarity_use_rtps', False)
                polarity_use_dhc = baseline_params.get('polarity_use_dhc', False)
                
                # Extract top-1 candidate's feature scores (MUST be before if/else!)
                top1_idx = 0  # First candidate in topk
                
                # Estimate polarity for this query
                if use_polarity_aware:
                    # Use PREDICTIVE polarity based on feature scores
                    # Get z-scored features (already computed earlier)
                    ccs_z_top1 = ccs_z[top1_idx].item() if ccs_z is not None else 0.0
                    curv_z_top1 = curv_z[top1_idx].item() if curv_z is not None else 0.0
                    rts_z_top1 = rts_z[top1_idx].item() if rts_z is not None else 0.0
                    
                    # Optional 5-feature values (only extracted when needed)
                    rtps_z_top1 = rtps_z[top1_idx].item() if (polarity_use_rtps and rtps_z is not None) else 0.0
                    dhc_z_top1 = dhc_raw[top1_idx].item() if (polarity_use_dhc and dhc_raw is not None) else 0.0
                    
                    # Compute predictive polarity from features (with ablation support)
                    polarity = estimate_polarity_from_features(
                        ccs_z=ccs_z_top1,
                        curv_z=curv_z_top1,
                        rts_z=rts_z_top1,
                        direction=direction,
                        use_rts=polarity_use_rts,
                        use_ccs=polarity_use_ccs,
                        use_curv=polarity_use_curv,
                        rtps_z=rtps_z_top1,
                        dhc_z=dhc_z_top1,
                        use_rtps=polarity_use_rtps,
                        use_dhc=polarity_use_dhc
                    )
                    
                    query_diagnostics['polarity'] = {
                        'value': float(polarity),
                        'threshold': polarity_threshold,
                        'method': 'feature-based',
                        'features': {
                            'ccs_z': float(ccs_z_top1),
                            'curv_z': float(curv_z_top1),
                            'rts_z': float(rts_z_top1)
                        },
                        'ablation': {
                            'use_rts': polarity_use_rts,
                            'use_ccs': polarity_use_ccs,
                            'use_curv': polarity_use_curv,
                            'use_rtps': polarity_use_rtps,
                            'use_dhc': polarity_use_dhc
                        }
                    }
                else:
                    # FIXED: Still compute polarity from features for diagnostics, but don't gate
                    ccs_z_top1 = ccs_z[top1_idx].item() if ccs_z is not None else 0.0
                    curv_z_top1 = curv_z[top1_idx].item() if curv_z is not None else 0.0
                    rts_z_top1 = rts_z[top1_idx].item() if rts_z is not None else 0.0
                    
                    polarity = estimate_polarity_from_features(
                        ccs_z=ccs_z_top1,
                        curv_z=curv_z_top1,
                        rts_z=rts_z_top1,
                        direction=direction,
                        use_rts=polarity_use_rts,
                        use_ccs=polarity_use_ccs,
                        use_curv=polarity_use_curv
                    )
                    
                    query_diagnostics['polarity'] = {
                        'value': float(polarity),
                        'threshold': polarity_threshold,
                        'method': 'feature-based (gating disabled)',
                        'note': 'Polarity computed but not used for gating',
                        'features': {
                            'ccs_z': float(ccs_z_top1),
                            'curv_z': float(curv_z_top1),
                            'rts_z': float(rts_z_top1)
                        },
                        'ablation': {
                            'use_rts': polarity_use_rts,
                            'use_ccs': polarity_use_ccs,
                            'use_curv': polarity_use_curv
                        }
                    }
                    # Override polarity to 1.0 to force smoothing (since gating is disabled)
                    polarity = 1.0
                
                # Compute node confidence early (needed for all operators)
                # Node confidence (0..1) from features per node (not global mean)
                if feature_scores.max() > feature_scores.min():
                    node_conf = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-8)
                else:
                    node_conf = torch.ones_like(feature_scores) * 0.5
                
                # Select operator based on polarity
                if polarity > polarity_threshold:
                    # Positive polarity → spectral smoothing
                    operator_type = 'spectral_smoothing'
                    gating_stats['polarity_gated'] += 1  # FIX: Track polarity gate pass for Gate%
                    
                    # OPTIMIZED: Use iterative Laplacian smoothing instead of full eigendecomposition
                    # This is O(K² × n_iters) vs O(K³) for eigh-based filter
                    # Achieves similar low-pass filtering effect without computing eigenvectors
                    try:
                        filtered_scores = iterative_laplacian_smooth(
                            scores=base_scores,
                            L=L,
                            gamma=gamma_smoothing,
                            n_iters=3,  # 3 iterations is usually sufficient
                            node_conf=node_conf
                        )
                    except Exception as e:
                        # Fallback: use base scores
                        logger.debug(f"Iterative smoothing failed: {e}, using base scores")
                        filtered_scores = base_scores
                    
                elif polarity < -polarity_threshold:
                    # Negative polarity → Apply method based on config
                    
                    if negative_polarity_method == 'disagreement':
                        # Score-geometry disagreement boost
                        # Boost isolated candidates (high score, low graph support)
                        operator_type = 'disagreement_boost'
                        filtered_scores = disagreement_boost(base_scores, A, lambda_boost=lambda_disagreement)
                        
                    elif negative_polarity_method == 'anti_spectral':
                        # Anti-spectral methods (for ablation/comparison)
                        operator_type = 'laplacian_sharpening'
                        filtered_scores = laplacian_sharpening(base_scores, L, gamma=gamma_sharpening)
                        
                    else:
                        # Default: Conservative no-op (skip refinement)
                        operator_type = 'none_negative_polarity'
                        filtered_scores = base_scores
                    
                else:
                    # Near-zero polarity → no operator (use base scores)
                    operator_type = 'none_neutral'
                    filtered_scores = base_scores
                
                # Store operator type in diagnostics
                query_diagnostics['operator_type'] = operator_type
                
                # Get configurable boost_scale for feature influence
                initial_boost_scale = baseline_params.get('boost_scale', 0.35)
                
                # STRENGTHENED: Stronger mix with feature boost (using config boost_scale)
                final_scores = 0.5 * filtered_scores + 0.5 * (base_scores + initial_boost_scale * feature_scores)
                
                # Compute global confidence for diagnostics (node_conf already computed above)
                confidence = float(node_conf.mean().item())
                
                # Store spectral filtering diagnostics
                query_diagnostics['spectral_filtering'] = {
                    'confidence': confidence,
                    'operator_type': operator_type,
                    'polarity': float(polarity),
                    'filter_applied': True,
                    'filtered_scores_range': [float(filtered_scores.min().item()), float(filtered_scores.max().item())],
                    'method': 'iterative_laplacian_smooth' if operator_type == 'spectral_smoothing' else 'direct'
                }
            except Exception as e:
                logger.warning(f"Spectral filtering failed for query {i}: {e}, using base scores")
                final_scores = base_scores
                query_diagnostics['spectral_filtering'] = {
                    'filter_applied': False,
                    'error': str(e)
                }
            
            # Map back to full gallery with calibrated score update
            z_fused = s_clip.clone()
            # INCREASED: Use stronger boost (0.3-0.35) instead of 0.1 for meaningful flips
            boost_scale = baseline_params.get('boost_scale', 0.35)  # Increased from 0.1
            # Confidence-based gating: reduce boost if spectral filtering confidence is low
            # TUNED: Lowered to 0.38 for higher candidate pool (quality gate provides safety)
            confidence_min = baseline_params.get('confidence_min', 0.38)
            if 'spectral_filtering' in query_diagnostics:
                conf = query_diagnostics['spectral_filtering'].get('confidence', 0.5)
                if conf < confidence_min:
                    boost_scale = boost_scale * (conf / confidence_min)  # Scale down boost
            
            # Compute score deltas (GPU)
            score_deltas = final_scores - base_scores
            
            # SAFETY GATES: Apply confidence-based acceptance before accepting flips
            # FIXED: Add proper acceptance logic with confidence threshold
            # Identify potential flip: candidate that would become new top-1
            base_top1_idx_in_topk = 0  # Base top-1 is always first in topk_idx
            base_top1_score = base_scores[base_top1_idx_in_topk]
            
            # Find candidate with highest final score (potential new top-1)
            new_top1_idx_in_topk = torch.argmax(final_scores).item()
            new_top1_score = final_scores[new_top1_idx_in_topk]
            
            # =============================================================================
            # TIER 1 FEATURES: Bank-free discriminative signals
            # =============================================================================
            
            # 1. Self-Reciprocity (Bidirectional Agreement)
            # Compute reverse score: does candidate rank query highly?
            recip_margins = torch.zeros(n_cands, device=device, dtype=query.dtype)
            if baseline_params.get('use_reciprocity', True):
                try:
                    if direction == 'i2t':
                        # Forward: image → text (already computed as base_scores)
                        # Backward: text → image (candidate ranks query)
                        # For each candidate text, compute similarity back to query image
                        back_scores = candidate_embeddings @ query_emb  # [K] - how well each text ranks this image
                        
                        # For each candidate, compute margin: its back-score vs 2nd-best back-score
                        for j in range(n_cands):
                            # Get this candidate's backward score
                            cand_back_score = back_scores[j]
                            # Get 2nd-best backward score (excluding self)
                            others_mask = torch.ones(n_cands, dtype=torch.bool, device=device)
                            others_mask[j] = False
                            if others_mask.sum() > 0:
                                second_best = back_scores[others_mask].max()
                                recip_margins[j] = cand_back_score - second_best
                            else:
                                recip_margins[j] = cand_back_score
                    
                    else:  # t2i
                        # Forward: text → image (base_scores)
                        # Backward: image → text
                        back_scores = candidate_embeddings @ query_emb
                        for j in range(n_cands):
                            cand_back_score = back_scores[j]
                            others_mask = torch.ones(n_cands, dtype=torch.bool, device=device)
                            others_mask[j] = False
                            if others_mask.sum() > 0:
                                second_best = back_scores[others_mask].max()
                                recip_margins[j] = cand_back_score - second_best
                            else:
                                recip_margins[j] = cand_back_score
                    
                    # Z-score normalize within query
                    recip_mean = recip_margins.mean()
                    recip_std = recip_margins.std() + 1e-8
                    recip_z = (recip_margins - recip_mean) / recip_std
                    
                except Exception as e:
                    logger.warning(f"Reciprocity computation failed for query {i}: {e}")
                    recip_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            else:
                recip_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 2. List Coherence (Candidate-Candidate Agreement)
            # Measure per-candidate outlierness within top-K
            outlier_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            if baseline_params.get('use_list_coherence', True):
                try:
                    # Compute candidate-candidate similarity matrix
                    cand_cand_sim = candidate_embeddings @ candidate_embeddings.T  # [K, K]
                    
                    # For each candidate, compute mean similarity to others (exclude self)
                    item_coherence = torch.zeros(n_cands, device=device, dtype=query.dtype)
                    for j in range(n_cands):
                        others_mask = torch.ones(n_cands, dtype=torch.bool, device=device)
                        others_mask[j] = False
                        if others_mask.sum() > 0:
                            item_coherence[j] = cand_cand_sim[j, others_mask].mean()
                    
                    # Z-score: more negative = outlier (isolated from others)
                    coherence_mean = item_coherence.mean()
                    coherence_std = item_coherence.std() + 1e-8
                    outlier_z = -(item_coherence - coherence_mean) / coherence_std  # Negative = outlier
                    
                except Exception as e:
                    logger.warning(f"List coherence computation failed for query {i}: {e}")
                    outlier_z = torch.zeros(n_cands, device=device, dtype=query.dtype)
            
            # 3. Sharpness-Aware Gating (Base Confidence via Entropy)
            # High entropy = base is uncertain; low entropy = base is confident (risky to flip)
            base_entropy = 0.0
            if baseline_params.get('use_sharpness_gating', True):
                try:
                    # Compute softmax entropy over top-K base scores
                    temperature = baseline_params.get('entropy_temperature', 0.1)
                    probs = torch.softmax(base_scores / temperature, dim=0)
                    base_entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    
                except Exception as e:
                    logger.warning(f"Sharpness gating computation failed for query {i}: {e}")
                    base_entropy = 1.0  # Default to uncertain
            
            # Store Tier 1 features in diagnostics
            query_diagnostics['tier1_features'] = {
                'recip_z_mean': float(recip_z.mean().item()),
                'recip_z_std': float(recip_z.std().item()),
                'outlier_z_mean': float(outlier_z.mean().item()),
                'outlier_z_std': float(outlier_z.std().item()),
                'base_entropy': float(base_entropy)
            }
            
            # =============================================================================
            # End Tier 1 Features
            # =============================================================================
            
            # Track gating decision
            gated = False
            accepted = False
            flip_blocked = False
            
            # Only apply gates if there's a potential flip
            if new_top1_idx_in_topk != base_top1_idx_in_topk:
                gated = True  # Query was gated (flip attempted)
                
                # Compute base margin on s_clip among topK
                base_margin = (base_scores[base_top1_idx_in_topk] - base_scores[1]).item() if n_cands > 1 else 0.0
                small_margin = base_margin < baseline_params.get('margin_small', 0.02)
                
                # LABEL-FREE: No GT used in inference decisions
                # (Only compute for post-hoc diagnostics/analysis - NOT used in accept/reject logic)
                baseline_top1_gallery_idx = topk_idx[base_top1_idx_in_topk].item()
                correct_indices = gt_mapping.get(query_id, [])
                baseline_is_gt = baseline_top1_gallery_idx in correct_indices if correct_indices else False  # For diagnostics only
                
                # Compute confidence score from features
                # conf = sigmoid(a1*rts_z + a5*ccs_z_pos - a2*rtps_z - a3*max(0, -dhc_z) - a4*abs(ccs_z_neg))
                # NOTE: High RTPS = preserves base ranking = NOT a good flip → PENALIZE, not reward
                # NOTE: CCS is a strong discriminator - split into positive (good) and negative (bad) components
                a1 = baseline_params.get('conf_a1', 0.5)  # RTS weight (positive: high reciprocal trust = good)
                a2 = baseline_params.get('conf_a2', 0.20)  # RTPS penalty weight (reduced from 0.3 to 0.20)
                a3 = baseline_params.get('conf_a3', 0.15)  # DHC penalty weight (reduced from 0.2 to 0.15)
                a4 = baseline_params.get('conf_a4', 0.1)  # CCS negative weight
                a5 = baseline_params.get('conf_a5', 0.3)  # CCS positive weight (NEW: reward positive CCS)
                
                challenger_rts = rts_z[new_top1_idx_in_topk].item() if use_rts else 0.0
                challenger_rtps = rtps_z[new_top1_idx_in_topk].item() if use_rtps else 0.0
                challenger_dhc = dhc_raw[new_top1_idx_in_topk].item() if use_dhc else 0.0
                challenger_ccs = ccs_z[new_top1_idx_in_topk].item() if use_ccs else 0.0
                challenger_curv = curv_z[new_top1_idx_in_topk].item() if use_curv else 0.0
                
                # Split CCS into positive and negative components
                ccs_pos = max(0.0, challenger_ccs)  # Positive CCS = good (context coherence)
                ccs_neg = abs(min(0.0, challenger_ccs))  # Negative CCS = bad (context mismatch)
                
                # Confidence computation (FIXED: RTPS now penalized, CCS split, added CURV bonus)
                conf_logit = (a1 * challenger_rts + 
                             a5 * ccs_pos -  # Reward positive CCS
                             a2 * challenger_rtps - 
                             a3 * max(0.0, -challenger_dhc) - 
                             a4 * ccs_neg)  # Penalize negative CCS
                
                # Sigmoid
                conf = 1.0 / (1.0 + np.exp(-np.clip(conf_logit, -50, 50)))
                
                # =============================================================================
                # COMPUTE LS SCORES EARLY (needed for baseline confidence estimation)
                # =============================================================================
                # Get LS weight parameters
                alpha = baseline_params.get('ls_weight_rts', 0.5)
                beta = baseline_params.get('ls_weight_outlier', 0.3)
                gamma = baseline_params.get('ls_weight_recip', 0.6)
                delta_weight = baseline_params.get('ls_weight_delta', 0.4)
                
                # Compute LS for current rank-1 (baseline) - needed for confidence estimation
                base_recip_z = recip_z[base_top1_idx_in_topk].item()
                base_outlier_z = outlier_z[base_top1_idx_in_topk].item()
                base_rts = rts_z[base_top1_idx_in_topk].item() if use_rts else 0.0
                LS_baseline = (alpha * base_rts + 
                              beta * abs(base_outlier_z) +
                              gamma * base_recip_z +
                              delta_weight * 0.0)  # No delta for baseline
                
                # Confidence threshold - LABEL-FREE baseline confidence estimation
                # TUNED: Lowered to 0.52 for higher candidate pool (quality gate provides safety)
                tau_conf_base = baseline_params.get('tau_conf', 0.52)
                
                # LABEL-FREE: Estimate baseline confidence from features (NO GT!)
                baseline_confidence_signals = {
                    'large_margin': base_margin > 0.05,           # Strong margin
                    'low_entropy': base_entropy < 0.3,            # Confident distribution
                    'high_baseline_LS': LS_baseline > 0.5,        # Strong local support
                    'baseline_not_outlier': base_outlier_z > -0.5 # Not isolated
                }
                
                # Count how many signals indicate baseline is confident (0-4)
                baseline_confidence_score = sum(baseline_confidence_signals.values())
                
                # Adaptive tau_conf based on baseline confidence (NO GT!)
                if baseline_confidence_score >= 3:
                    tau_conf = tau_conf_base * 1.10  # Very confident baseline → conservative
                elif baseline_confidence_score == 2:
                    tau_conf = tau_conf_base * 1.05  # Moderately confident → slight conservatism
                else:
                    tau_conf = tau_conf_base * 1.0   # Uncertain baseline → standard threshold
                
                # Score delta threshold (relative to score std)
                score_std = float(base_scores.std().item()) if n_cands > 1 else 1.0
                tau_delta = baseline_params.get('tau_delta_scale', 0.05) * score_std
                
                # Score delta for challenger
                delta_score = float(score_deltas[new_top1_idx_in_topk].item())
                
                # Acceptance criteria:
                # 1. Confidence must exceed threshold
                # 2. Delta must exceed threshold (positive) OR be small negative with very high confidence
                #    - Accept: delta > tau_delta (clearly positive)
                #    - Accept: -tau_delta < delta < tau_delta AND conf > 0.75 (small negative but good confidence)
                #    - Reject: delta < -tau_delta (large negative - always bad)
                # 3. Quality gate when baseline is correct: require strong discriminators (CCS > 0.3 OR curv > 2.5)
                # TUNED: Lowered high_conf threshold from 0.8 to 0.75 for more tolerance
                is_positive_delta = delta_score > tau_delta
                is_small_negative_with_high_conf = (delta_score > -tau_delta) and (conf > 0.75)
                
                # =============================================================================
                # TIER 1+2 SIGNIFICANCE GATE: FILTER-AS-MASK APPROACH (ALIGNED SPACES)
                # =============================================================================
                
                if use_significance_gate:
                    # NEW GATE: Spectral-as-proposer + RAW-space testing
                    # Track FDR attempts
                    flip_attempts_fdr += 1
                    
                    # DEBUG: Log entry for first query
                    if i == 0:
                        logger.info(f"🔍 DEBUG Query 0: Entering significance gate block")
                        logger.info(f"   n_cands={n_cands}, delta_score={delta_score:.6f}")
                    
                    # =================================================================
                    # STEP 1: Build Size-Controlled Mask (No Hard Thresholds)
                    # =================================================================
                    K_actual = n_cands
                    base_idx_val = 0  # Base top-1 is always first in topk
                    
                    # Spectral confidence (use node_conf from spectral filtering)
                    if 'spectral_filtering' in query_diagnostics and query_diagnostics['spectral_filtering'].get('filter_applied', False):
                        # Use per-node confidence from spectral filtering
                        spec_conf = node_conf  # Already computed above
                    else:
                        # Fallback: use feature scores as confidence
                        spec_conf = torch.ones(K_actual, device=device)
                    
                    # Community votes (mutual-neighbor counts)
                    cand_cand_sim = candidate_embeddings @ candidate_embeddings.T  # [K, K]
                    M_vote_thresh = baseline_params.get('M_vote_thresh', 4)
                    comm_votes = (cand_cand_sim > 0.7).sum(dim=1) - 1  # -1 to exclude self
                    
                    # Size-controlled masks (adaptive to K)
                    S_spec = min(12, K_actual // 4)  # Top S_spec by spectral confidence
                    S_vote = min(8, K_actual // 6)   # Top S_vote by community votes
                    
                    spec_keep = torch.topk(spec_conf, k=S_spec, largest=True).indices
                    vote_keep = torch.topk(comm_votes, k=S_vote, largest=True).indices
                    
                    # Get mode to decide mask construction
                    spectral_mode = baseline_params.get('spectral_mode', 'B')
                    
                    if spectral_mode == 'A':
                        # Mode A: NO top-RAW injection (let spectral find hidden gems)
                        S_raw = 0  # No raw injection in Mode A
                        mask_idx = torch.unique(torch.cat([
                            spec_keep,
                            vote_keep,
                            torch.tensor([base_idx_val], device=device)  # Always include baseline
                        ]))
                        
                        # ===================================================================
                        # CHANGE #1: MID-BAND INJECTION (Ranks 6-10 with Spectral Gate)
                        # ===================================================================
                        # Fill the "valley of death" - add rank 6-10 candidates with strong spectral signal
                        if baseline_params.get('enable_midband_injection', True):
                            # Get baseline ranking (by raw CLIP scores)
                            baseline_ranking = torch.argsort(s_clip, descending=True)[:K_actual]
                            
                            # Candidates from raw ranks 6-10 (0-indexed: 5-9)
                            midband_candidates = baseline_ranking[5:10]  # ranks 6-10
                            
                            # Compute Q75 of spectral confidence for current top-K
                            if len(spec_conf) > 0:
                                spec_conf_q75 = torch.quantile(spec_conf, 0.75)
                                
                                # Filter: only add if spectral confidence ≥ Q75
                                midband_to_add = []
                                for cand in midband_candidates:
                                    if cand.item() < len(spec_conf):  # Safety check
                                        if spec_conf[cand] >= spec_conf_q75:
                                            midband_to_add.append(cand)
                                
                                # Cap at M=2 per query to keep mask reasonable
                                M_midband = baseline_params.get('midband_cap', 2)
                                if len(midband_to_add) > M_midband:
                                    # Keep top M by spectral confidence
                                    midband_conf = torch.tensor([spec_conf[c] for c in midband_to_add], device=device)
                                    top_M_indices = torch.topk(midband_conf, k=M_midband, largest=True).indices
                                    midband_to_add = [midband_to_add[idx] for idx in top_M_indices.cpu().numpy()]
                                
                                # Add to mask
                                if midband_to_add:
                                    mask_idx = torch.unique(torch.cat([
                                        mask_idx,
                                        torch.stack(midband_to_add)
                                    ]))
                    else:
                        # Mode B: Also include top-RAW for near-tie tier (will separate tiers later)
                        S_raw = min(8, K_actual // 6)
                        raw_keep = torch.arange(S_raw, device=device)
                        mask_idx = torch.unique(torch.cat([
                            spec_keep,
                            vote_keep,
                            raw_keep,
                            torch.tensor([base_idx_val], device=device)
                        ]))
                    
                    mask_size = mask_idx.numel()
                    
                    # DEBUG: Log mask info for first query
                    if i == 0:
                        logger.info(f"🔍 DEBUG Query 0: Mask constructed")
                        logger.info(f"   mask_size={mask_size}, base_idx_val={base_idx_val}")
                        logger.info(f"   mask contains only baseline? {mask_size == 1 and mask_idx[0].item() == base_idx_val}")
                    
                    # Edge case: if mask only contains baseline, skip (no challengers)
                    if mask_size == 1 and mask_idx[0].item() == base_idx_val:
                        # No challengers available
                        if i == 0:
                            logger.info(f"🔍 DEBUG Query 0: SKIPPED - mask only contains baseline")
                        all_flip_attempts.append({
                            'query_idx': i,
                            'list_idx': len(fused_scores),
                            'p_any': 1.0,  # No evidence
                            'challengers': [],
                            'best_challenger': None,
                            'new_top1_idx_in_topk': base_idx_val,
                            's_clip': s_clip.clone(),
                            'skip': True,
                            'skip_reason': 'mask_only_baseline',
                            'mask_size': mask_size,
                            'null_size': 0
                        })
                        quality_gate_passed = False  # Block flip
                        
                        query_diagnostics['significance_gate'] = {
                            'method': 'filter_as_mask',
                            'skip': True,
                            'reason': 'mask_only_baseline'
                        }
                        
                    else:
                        # =================================================================
                        # STEP 2: Select B Challengers from Mask (by spectral score)
                        # =================================================================
                        # Remove baseline from mask to get pure challenger pool
                        challenger_mask = mask_idx[mask_idx != base_idx_val]
                        
                        # Select top-B challengers from mask by spectral score
                        B_max = baseline_params.get('num_challengers', 3)
                        B_actual = min(B_max, challenger_mask.numel())
                        
                        # DEBUG: Log challenger info for first query
                        if i == 0:
                            logger.info(f"🔍 DEBUG Query 0: Challengers selected")
                            logger.info(f"   B_max={B_max}, B_actual={B_actual}, challenger_mask.numel()={challenger_mask.numel()}")
                        
                        if B_actual == 0:
                            # No challengers (shouldn't happen due to mask construction, but defensive)
                            if i == 0:
                                logger.info(f"🔍 DEBUG Query 0: SKIPPED - B_actual == 0")
                            all_flip_attempts.append({
                                'query_idx': i,
                                'list_idx': len(fused_scores),
                                'p_any': 1.0,
                                'challengers': [],
                                'best_challenger': None,
                                'new_top1_idx_in_topk': base_idx_val,
                                's_clip': s_clip.clone(),
                                'skip': True,
                                'skip_reason': 'no_challengers',
                                'mask_size': mask_size,
                                'null_size': 0
                            })
                            quality_gate_passed = False
                            
                            query_diagnostics['significance_gate'] = {
                                'method': 'filter_as_mask',
                                'skip': True,
                                'reason': 'no_challengers'
                            }
                            
                        else:
                            # DEBUG: Entering the main z-score computation block
                            if i == 0:
                                logger.info(f"🔍 DEBUG Query 0: Entering z-score computation block (B_actual={B_actual})")
                            
                            # =================================================================
                            # STEP 2.5: Build L2O Null FIRST (Need for Variance-Stabilized Z-Space)
                            # =================================================================
                            # We need null to compute robust z-scores, so move null construction here
                            
                            topk_vals_raw = topk_vals  # [K] Raw SigLIP scores
                            spectral_deltas = score_deltas  # [K] Already computed above
                            
                            # Build preliminary null for z-standardization
                            # Use simple bottom-half null (will refine later after challenger selection)
                            null_start_rank = K_actual // 2
                            null_idx_prelim = torch.arange(null_start_rank, K_actual, device=device)
                            
                            # =================================================================
                            # STEP 2.6: Variance-Stabilized Z-Space Fusion
                            # =================================================================
                            # CRITICAL: Use NULL statistics (median/MAD) for robust z-scores
                            # This prevents σ̃ explosion as λ increases
                            
                            # Compute robust z-scores for RAW (using null only)
                            null_raw = topk_vals_raw[null_idx_prelim]
                            med_raw = torch.median(null_raw)
                            mad_raw = 1.4826 * torch.median(torch.abs(null_raw - med_raw))
                            z_raw = (topk_vals_raw - med_raw) / (mad_raw + 1e-8)
                            
                            # DEBUG: Log raw score statistics for first query
                            if i == 0:
                                logger.info(f"🔍 DEBUG Query 0 - Raw Score Stats:")
                                logger.info(f"   topk_vals_raw: mean={topk_vals_raw.mean().item():.6f}, std={topk_vals_raw.std().item():.6f}")
                                logger.info(f"   topk_vals_raw: min={topk_vals_raw.min().item():.6f}, max={topk_vals_raw.max().item():.6f}")
                                logger.info(f"   null_raw (idx {null_start_rank}:{K_actual}): mean={null_raw.mean().item():.6f}, std={null_raw.std().item():.6f}")
                                logger.info(f"   med_raw={med_raw.item():.6f}, mad_raw={mad_raw.item():.6f}")
                                logger.info(f"   z_raw: mean={z_raw.mean().item():.3f}, std={z_raw.std().item():.3f}")
                            
                            # Compute robust z-scores for SPECTRAL (using null only)
                            null_spec = spectral_deltas[null_idx_prelim]
                            med_spec = torch.median(null_spec)
                            mad_spec = 1.4826 * torch.median(torch.abs(null_spec - med_spec))
                            z_spec = (spectral_deltas - med_spec) / (mad_spec + 1e-8)
                            
                            # DEBUG: Log spectral delta statistics for first query
                            if i == 0:
                                logger.info(f"🔍 DEBUG Query 0 - Spectral Delta Stats:")
                                logger.info(f"   spectral_deltas: mean={spectral_deltas.mean().item():.6f}, std={spectral_deltas.std().item():.6f}")
                                logger.info(f"   spectral_deltas: min={spectral_deltas.min().item():.6f}, max={spectral_deltas.max().item():.6f}")
                                logger.info(f"   null_spec (idx {null_start_rank}:{K_actual}): mean={null_spec.mean().item():.6f}, std={null_spec.std().item():.6f}")
                                logger.info(f"   med_spec={med_spec.item():.6f}, mad_spec={mad_spec.item():.6f}")
                                logger.info(f"   z_spec: mean={z_spec.mean().item():.3f}, std={z_spec.std().item():.3f}")
                            
                            # ===================================================================
                            # CHANGE #5: ADAPTIVE λ PER QUERY
                            # ===================================================================
                            # Compute adaptive λ based on raw vs spectral agreement
                            # For queries with strong raw signal, lower λ (trust baseline)
                            # For queries with weak raw signal, higher λ (trust spectral)
                            
                            if baseline_params.get('enable_adaptive_lambda', True):
                                # Get base (rank 1) and runner-up (rank 2) in z-space
                                base_idx_val = 0  # Already computed above
                                
                                # Find runner-up: second highest z_raw (excluding base)
                                z_raw_sorted_indices = torch.argsort(z_raw, descending=True)
                                runner_up_idx = z_raw_sorted_indices[1].item() if len(z_raw_sorted_indices) > 1 else base_idx_val
                                
                                # Compute deltas
                                delta_z_raw = z_raw[base_idx_val] - z_raw[runner_up_idx]
                                delta_z_spec = z_spec[base_idx_val] - z_spec[runner_up_idx]
                                
                                # Adaptive λ: if spectral favors non-base, increase λ
                                # Formula: λ = clip(-Δz_raw / (Δz_spec + ε), 0, 0.8)
                                # When Δz_raw > 0 and Δz_spec < 0 → λ high (spectral disagrees, give it weight)
                                # When Δz_raw > 0 and Δz_spec > 0 → λ low (both agree, trust raw)
                                if delta_z_spec > 1e-6:  # Avoid division by near-zero
                                    lambda_adaptive = -delta_z_raw / (delta_z_spec + 1e-6)
                                    lambda_adaptive = torch.clamp(torch.tensor(lambda_adaptive, device=device), 0.0, 0.8).item()
                                else:
                                    # Spectral has no clear signal, use fallback
                                    lambda_adaptive = 0.0
                                
                                lambda_spec = lambda_adaptive
                            else:
                                # Fixed λ (fallback)
                                lambda_spec = baseline_params.get('lambda_fused', 0.7)
                            
                            # Fused z-space (variance-stabilized!)
                            # σ̃(z_fused) ≈ sqrt(1 + λ²) regardless of λ value
                            z_fused_topk = z_raw + lambda_spec * z_spec  # [K] z-scores for top-K
                            
                            # Keep raw scores for final R@k evaluation
                            topk_vals_fused = z_fused_topk  # Will use z-space for all decisions
                            
                            # =================================================================
                            # STEP 3: Select Challengers by Z_FUSED Score
                            # =================================================================
                            spectral_mode = baseline_params.get('spectral_mode', 'B')
                            
                            if spectral_mode == 'A':
                                # Mode A: B=1, simple argmax (no multi-testing)
                                challenger_z_fused = z_fused_topk[challenger_mask]
                                best_idx_in_mask = torch.argmax(challenger_z_fused)
                                challenger_indices = challenger_mask[best_idx_in_mask:best_idx_in_mask+1]
                                B_actual = 1
                            else:
                                # Mode B: Will implement tiered testing (for now, use B=1 too)
                                challenger_z_fused = z_fused_topk[challenger_mask]
                                best_idx_in_mask = torch.argmax(challenger_z_fused)
                                challenger_indices = challenger_mask[best_idx_in_mask:best_idx_in_mask+1]
                                B_actual = 1
                            
                            # =================================================================
                            # STEP 4: Build L2O Null (Excludes ALL B Challengers + Baseline)
                            # =================================================================
                            
                            # Adaptive M for multi-challenger (smaller neighborhoods to preserve null size)
                            M = 8 if B_actual > 1 else 12
                            null_start_rank = K_actual // 2 if B_actual == 1 else K_actual // 3
                            
                            # Build exclusion set: baseline + all B challengers' neighborhoods
                            exclude_indices = [base_idx_val]
                            for chall_idx_tensor in challenger_indices:
                                chall_idx_val = chall_idx_tensor.item()
                                # Get M-NN of this challenger
                                nb = torch.topk(cand_cand_sim[chall_idx_val], k=M+1, largest=True).indices
                                exclude_indices.append(nb)
                            exclude_indices = torch.unique(torch.cat([torch.tensor([base_idx_val], device=device)] + 
                                                                     [torch.topk(cand_cand_sim[c.item()], k=M+1).indices 
                                                                      for c in challenger_indices]))
                            
                            # Null = bottom half (or bottom 2/3 for multi-challenger) minus exclusions
                            null_pool = torch.arange(null_start_rank, K_actual, device=device)
                            null_idx = null_pool[~torch.isin(null_pool, exclude_indices)]
                            null_size = null_idx.numel()
                            
                            # Edge case: null too small
                            if null_size < 3:
                                all_flip_attempts.append({
                                    'query_idx': i,
                                    'list_idx': len(fused_scores),
                                    'p_any': 1.0,
                                    'challengers': [],
                                    'best_challenger': None,
                                    'new_top1_idx_in_topk': base_idx_val,
                                    's_clip': s_clip.clone(),
                                    'skip': True,
                                    'skip_reason': 'null_too_small',
                                    'mask_size': mask_size,
                                    'null_size': null_size
                                })
                                quality_gate_passed = False
                                
                                query_diagnostics['significance_gate'] = {
                                    'method': 'filter_as_mask',
                                    'skip': True,
                                    'reason': 'null_too_small',
                                    'null_size': null_size
                                }
                                
                            else:
                                # =================================================================
                                # STEP 5: Compute Shared σ in FUSED Space (One σ for All B Challengers)
                                # =================================================================
                                # CRITICAL: Compute null σ in FUSED space (same space as margins!)
                                null_scores_fused = topk_vals_fused[null_idx]
                                null_ranks = null_idx.float()
                                
                                # Detrend: fit linear trend and compute MAD of residuals
                                sigma_local = compute_detrended_sigma(null_scores_fused, null_ranks)
                                
                                # DEBUG: Log σ computation for first query
                                if i == 0:
                                    logger.info(f"🔍 DEBUG Query 0 - σ Computation:")
                                    logger.info(f"   null_scores_fused: mean={null_scores_fused.mean().item():.6f}, std={null_scores_fused.std().item():.6f}")
                                    logger.info(f"   null_scores_fused: min={null_scores_fused.min().item():.6f}, max={null_scores_fused.max().item():.6f}")
                                    logger.info(f"   sigma_local (detrended MAD): {sigma_local:.6f}")
                                    logger.info(f"   global_sigma_mads collected so far: {len(global_sigma_mads)}")
                                
                                # Pass 0: collect σ's for global estimate (in fused space)
                                if i < 200:  # Increased from 100 to 200 for better estimate
                                    global_sigma_mads.append(sigma_local if sigma_local > 1e-6 else 1e-4)
                                
                                # James-Stein shrinkage (if enough samples)
                                if len(global_sigma_mads) > 50:
                                    sigma_global = np.median(global_sigma_mads)
                                    shrinkage_c = baseline_params.get('shrinkage_c', 10.0)
                                    lam = shrinkage_c / (shrinkage_c + max(1, null_size))
                                    sigma_tilde = float(((1-lam)*(sigma_local**2) + lam*(sigma_global**2))**0.5 + 1e-8)
                                    
                                    # DEBUG: Log shrinkage for first query after cold start
                                    if i == 51:
                                        logger.info(f"🔍 DEBUG Query 51 - Shrinkage Active:")
                                        logger.info(f"   sigma_local={sigma_local:.6f}, sigma_global={sigma_global:.6f}")
                                        logger.info(f"   shrinkage_c={shrinkage_c}, null_size={null_size}, lam={lam:.4f}")
                                        logger.info(f"   sigma_tilde={sigma_tilde:.6f}")
                                else:
                                    # Cold start: use local σ without shrinkage
                                    sigma_tilde = max(sigma_local, 1e-6)
                                
                                # =================================================================
                                # STEP 6: Test Challenger(s) - Mode-Dependent
                                # =================================================================
                                spectral_mode = baseline_params.get('spectral_mode', 'B')
                                
                                p_values_within_query = []
                                challenger_data = []
                                
                                for chall_idx_tensor in challenger_indices:
                                    chall_idx_val = chall_idx_tensor.item()
                                    
                                    # Margin in Z-FUSED space (variance-stabilized)
                                    margin_z_fused = float(z_fused_topk[chall_idx_val] - z_fused_topk[base_idx_val])
                                    
                                    # Also track raw margin for diagnostics
                                    margin_raw = float(topk_vals_raw[chall_idx_val] - topk_vals_raw[base_idx_val])
                                    
                                    if spectral_mode == 'A':
                                        # Mode A: Simple safety check (no p-values)
                                        tau_z_fused = baseline_params.get('tau_z_fused', 0.0)
                                        
                                        if margin_z_fused >= tau_z_fused:
                                            # Passes safety check
                                            p_value = 0.01  # Dummy value (will use Δz_fused for sorting)
                                        else:
                                            # Fails safety check (would hurt or no gain)
                                            p_value = 1.0
                                        
                                        # Store Δz_fused for global ranking later
                                        z_eff = margin_z_fused  # Use as "effect size"
                                        
                                    else:
                                        # Mode B: Proper z-test with p-value
                                        z_max = baseline_params.get('z_max', 6.0)
                                        
                                        # Z-score in fused space (standardized margin)
                                        # σ̃(z_fused) ≈ sqrt(1 + λ²) by construction
                                        null_z_fused = z_fused_topk[null_idx]
                                        sigma_z_fused = float(torch.std(null_z_fused).item()) + 1e-8
                                        
                                        z_raw = margin_z_fused / sigma_z_fused
                                        z_eff = np.clip(z_raw, -z_max, z_max)
                                        
                                        # One-sided p-value: P(Z > z_eff) under H0
                                        p_value = compute_one_sided_pvalue(z_eff)
                                    
                                    p_values_within_query.append(p_value)
                                    challenger_data.append({
                                        'chall_idx': chall_idx_val,
                                        'z_eff': z_eff,
                                        'p_value': p_value,
                                        'margin_z_fused': margin_z_fused,  # Margin in Z-FUSED space
                                        'margin_raw': margin_raw,          # Margin in RAW space (diagnostic)
                                        'base_rank': chall_idx_val,        # Rank within top-K
                                        'spectral_conf': float(spec_conf[chall_idx_val])
                                    })
                                
                                # =================================================================
                                # STEP 6: Within-Query CCT Combination
                                # =================================================================
                                if len(p_values_within_query) > 0:
                                    p_any = combine_p_cct(p_values_within_query)  # CCT: robust to dependence
                                    
                                    # Find best challenger (minimum p-value = largest margin for shared σ)
                                    best_challenger_idx_in_list = int(np.argmin(p_values_within_query))
                                    best_challenger_data = challenger_data[best_challenger_idx_in_list]
                                    best_challenger_data['sigma_tilde'] = sigma_tilde
                                    best_challenger_data['null_size'] = null_size
                                    
                                    # Diagnostics for first 10 queries
                                    if i < 10:
                                        logger.info(f"📊 Query {i} Diagnostics (FUSED Space Testing):")
                                        logger.info(f"   λ_fused: {lambda_spec:.2f} (25% spectral, 75% raw)")
                                        logger.info(f"   Mask size: {mask_size} (spec={S_spec}, vote={S_vote}, raw={S_raw})")
                                        # CRITICAL DIAGNOSTIC: What ranks are in the mask?
                                        mask_ranks = sorted(mask_idx.cpu().numpy().tolist())
                                        logger.info(f"   Mask ranks (by raw score): {mask_ranks[:10]}{'...' if len(mask_ranks) > 10 else ''}")
                                        logger.info(f"   Mask includes baseline (rank-0): {0 in mask_ranks}")
                                        logger.info(f"   Mask includes top-5: {sum(1 for r in range(5) if r in mask_ranks)}/5")
                                        logger.info(f"   B_actual: {B_actual} challengers")
                                        logger.info(f"   Selected challenger ranks: {[int(c) for c in challenger_indices.cpu().numpy()]}")
                                        logger.info(f"   Null size: {null_size}")
                                        logger.info(f"   σ_tilde (FUSED): {sigma_tilde:.6f} (local={sigma_local:.6f})")
                                        logger.info(f"   Best margin (Z-FUSED): {best_challenger_data.get('margin_z_fused', best_challenger_data.get('margin_fused', 0)):.6f} ← TESTED!")
                                        logger.info(f"   Best margin (RAW): {best_challenger_data['margin_raw']:.6f} ← diagnostic")
                                        logger.info(f"   Best z_eff: {best_challenger_data['z_eff']:.2f}")
                                        p_str = ', '.join([f"{p:.4f}" for p in p_values_within_query])
                                        logger.info(f"   P-values: [{p_str}]")
                                        logger.info(f"   p_any (CCT): {p_any:.6f}")
                                else:
                                    p_any = 1.0
                                    best_challenger_data = None
                                
                                # =================================================================
                                # STEP 7: Store for Pass 2 (Storey-BH Across Queries)
                                # =================================================================
                                all_flip_attempts.append({
                                    'query_idx': i,
                                    'list_idx': len(fused_scores),
                                    'p_any': p_any,
                                    'challengers': challenger_data,
                                    'best_challenger': best_challenger_data,
                                    'new_top1_idx_in_topk': best_challenger_data['chall_idx'] if best_challenger_data else base_idx_val,
                                    's_clip': s_clip.clone(),
                                    'skip': False,
                                    'mask_size': mask_size,
                                    'null_size': null_size,
                                    'sigma_tilde': sigma_tilde
                                })
                                
                                # Temporarily PASS gate (FDR decision in Pass 2)
                                quality_gate_passed = True
                                
                                # Diagnostics
                                query_diagnostics['significance_gate'] = {
                                    'method': 'filter_as_mask',
                                    'p_any': float(p_any),
                                    'num_challengers': B_actual,
                                    'mask_size': mask_size,
                                    'null_size': null_size,
                                    'sigma_tilde': float(sigma_tilde),
                                    'best_z_eff': float(best_challenger_data['z_eff']) if best_challenger_data else -999,
                                    'best_margin_raw': float(best_challenger_data['margin_raw']) if best_challenger_data else 0.0,
                                    'best_challenger_idx': int(best_challenger_data['chall_idx']) if best_challenger_data else -1,
                                    'skip': False
                                }
                            
                            # =================================================================
                            # CRITICAL: Update outer z_fused with top-K z-scores
                            # =================================================================
                            # z_fused was initialized as s_clip.clone() (gallery-sized)
                            # Update only the top-K positions with variance-stabilized z-scores
                            z_fused[topk_idx] = z_fused_topk
                    
                else:
                    # OLD GATE: LS margin (backward compatibility)
                    # Extract Tier 1 features for challenger
                    challenger_recip_z = recip_z[new_top1_idx_in_topk].item()
                    challenger_outlier_z = outlier_z[new_top1_idx_in_topk].item()
                    
                    # Compute Local Support (LS) score for challenger
                    challenger_delta = delta_score
                    LS_challenger = (alpha * challenger_rts + 
                                    beta * abs(challenger_outlier_z) +
                                    gamma * challenger_recip_z +
                                    delta_weight * challenger_delta)
                    
                    # LS_baseline already computed earlier
                    LS_margin = LS_challenger - LS_baseline
                    
                    # Sharpness-Aware Gating
                    use_sharpness_gate = baseline_params.get('use_sharpness_gating', True)
                    pass_sharpness = True
                    if use_sharpness_gate:
                        entropy_min = baseline_params.get('entropy_min', 0.3)
                        entropy_threshold = entropy_min + 0.3 * min(base_margin / 0.05, 1.0)
                        pass_sharpness = (base_entropy >= entropy_threshold)
                    
                    # Consensus Gate
                    use_consensus_gate = baseline_params.get('use_consensus_gate', True)
                    pass_consensus = True
                    if use_consensus_gate:
                        M = baseline_params.get('consensus_M', max(3, int(0.25 * n_cands)))
                        cand_cand_sim = candidate_embeddings @ candidate_embeddings.T
                        challenger_sims = cand_cand_sim[new_top1_idx_in_topk]
                        _, challenger_neighbors_idx = torch.topk(challenger_sims, k=min(M+1, n_cands))
                        challenger_neighbors = set(challenger_neighbors_idx.cpu().numpy().tolist())
                        challenger_neighbors.discard(new_top1_idx_in_topk)
                        
                        community_vote = 0
                        for neighbor_idx in list(challenger_neighbors)[:M]:
                            neighbor_sims = cand_cand_sim[neighbor_idx]
                            _, neighbor_top_M = torch.topk(neighbor_sims, k=min(M+1, n_cands))
                            if new_top1_idx_in_topk in neighbor_top_M.cpu().numpy().tolist():
                                community_vote += 1
                        
                        min_community_votes = baseline_params.get('min_community_votes', 1)
                        pass_consensus = (community_vote >= min_community_votes)
                    
                    # LS Margin Threshold
                    use_adaptive_threshold = baseline_params.get('use_adaptive_threshold', True)
                    if use_adaptive_threshold:
                        LS_all = torch.zeros(n_cands, device=device, dtype=query.dtype)
                        for k in range(n_cands):
                            k_rts = rts_z[k].item() if use_rts else 0.0
                            k_recip_z = recip_z[k].item()
                            k_outlier_z = outlier_z[k].item()
                            k_delta = score_deltas[k].item()
                            LS_all[k] = (alpha * k_rts + 
                                        beta * abs(k_outlier_z) +
                                        gamma * k_recip_z +
                                        delta_weight * k_delta)
                        
                        LS_abs = torch.abs(LS_all)
                        tau_q = torch.quantile(LS_abs, 0.75).item()
                    else:
                        tau_q = baseline_params.get('ls_margin_threshold', 0.5)
                    
                    pass_ls_margin = (LS_margin >= tau_q)
                    
                    # Combine gates
                    quality_gate_passed = (pass_sharpness and pass_consensus and pass_ls_margin)
                
                # Store gate diagnostics (conditional on gate type)
                if not use_significance_gate:
                    # Old gate diagnostics
                    query_diagnostics['quality_gate'] = {
                        'label_free': True,
                        'pass_sharpness': bool(pass_sharpness),
                        'pass_consensus': bool(pass_consensus),
                        'pass_ls_margin': bool(pass_ls_margin),
                        'gate_passed': bool(quality_gate_passed),
                        'LS_challenger': float(LS_challenger),
                        'LS_baseline': float(LS_baseline),
                        'LS_margin': float(LS_margin),
                        'tau_q': float(tau_q),
                        'base_entropy': float(base_entropy),
                        'entropy_threshold': float(entropy_threshold) if use_sharpness_gate else 0.0,
                        'community_vote': int(community_vote) if use_consensus_gate else -1,
                        'challenger_rts': float(challenger_rts),
                        'challenger_recip_z': float(challenger_recip_z),
                        'challenger_outlier_z': float(challenger_outlier_z),
                        'base_margin': float(base_margin)
                    }
                # else: significance gate diagnostics already stored above
                
                # =============================================================================
                # 📊 GATE DIAGNOSTICS TRACKING (only for old gate)
                # =============================================================================
                if not use_significance_gate:
                    gate_diagnostics['total_flip_attempts'] += 1
                    
                    # Track individual gate pass/fail
                    if use_sharpness_gate:
                        gate_diagnostics['sharpness_gate_attempts'] += 1
                        if pass_sharpness:
                            gate_diagnostics['sharpness_gate_passed'] += 1
                        else:
                            gate_diagnostics['sharpness_gate_failed'] += 1
                    
                    if use_consensus_gate:
                        gate_diagnostics['consensus_gate_attempts'] += 1
                        if pass_consensus:
                            gate_diagnostics['consensus_gate_passed'] += 1
                        else:
                            gate_diagnostics['consensus_gate_failed'] += 1
                    
                    gate_diagnostics['ls_margin_gate_attempts'] += 1
                    if pass_ls_margin:
                        gate_diagnostics['ls_margin_gate_passed'] += 1
                    else:
                        gate_diagnostics['ls_margin_gate_failed'] += 1
                    
                    # Track combined gate result
                    if quality_gate_passed:
                        gate_diagnostics['all_gates_passed'] += 1
                    else:
                        gate_diagnostics['all_gates_failed'] += 1
                        
                        # Track which gates blocked (for failed flips)
                        failed_gates = []
                        if use_sharpness_gate and not pass_sharpness:
                            failed_gates.append('sharpness')
                        if use_consensus_gate and not pass_consensus:
                            failed_gates.append('consensus')
                        if not pass_ls_margin:
                            failed_gates.append('ls_margin')
                        
                        # Track blocking patterns
                        if len(failed_gates) == 1:
                            if 'sharpness' in failed_gates:
                                gate_diagnostics['blocked_by_sharpness_only'] += 1
                            elif 'consensus' in failed_gates:
                                gate_diagnostics['blocked_by_consensus_only'] += 1
                            elif 'ls_margin' in failed_gates:
                                gate_diagnostics['blocked_by_ls_margin_only'] += 1
                        elif len(failed_gates) == 2:
                            if 'sharpness' in failed_gates and 'consensus' in failed_gates:
                                gate_diagnostics['blocked_by_sharpness_consensus'] += 1
                            elif 'sharpness' in failed_gates and 'ls_margin' in failed_gates:
                                gate_diagnostics['blocked_by_sharpness_ls'] += 1
                            elif 'consensus' in failed_gates and 'ls_margin' in failed_gates:
                                gate_diagnostics['blocked_by_consensus_ls'] += 1
                        elif len(failed_gates) == 3:
                            gate_diagnostics['blocked_by_all_three'] += 1
                    
                    # Store feature distributions (sample 10% for memory efficiency)
                    if np.random.rand() < 0.1:
                        gate_diagnostics['base_entropy_all'].append(float(base_entropy))
                        gate_diagnostics['ls_challenger_all'].append(float(LS_challenger))
                        gate_diagnostics['ls_baseline_all'].append(float(LS_baseline))
                        gate_diagnostics['ls_margin_all'].append(float(LS_margin))
                        gate_diagnostics['tau_q_all'].append(float(tau_q))
                        if use_consensus_gate:
                            gate_diagnostics['community_vote_all'].append(int(community_vote))
                        gate_diagnostics['recip_z_all'].append(float(challenger_recip_z))
                        gate_diagnostics['outlier_z_all'].append(float(challenger_outlier_z))
                
                # =============================================================================
                # End Gate Diagnostics Tracking
                # =============================================================================
                
                # =============================================================================
                # End Label-Free Self-Calibrated Quality Gate
                # =============================================================================
                
                accept = (conf > tau_conf) and (is_positive_delta or is_small_negative_with_high_conf) and quality_gate_passed
                
                if not accept:
                    flip_blocked = True
                    # Zero out the delta for the challenger to prevent flip
                    score_deltas[new_top1_idx_in_topk] = 0.0
                    accepted = False
                else:
                    accepted = True  # Flip allowed
                
                query_diagnostics['safety_gates'] = {
                    'conf': float(conf),
                    'tau_conf': float(tau_conf),
                    'delta_score': float(delta_score),
                    'tau_delta': float(tau_delta),
                    'baseline_is_gt': bool(baseline_is_gt),  # For diagnostics only - not used in decisions
                    'baseline_confidence_score': int(baseline_confidence_score),  # Label-free confidence (0-4)
                    'baseline_confidence_signals': baseline_confidence_signals,
                    'flip_blocked': flip_blocked,
                    'base_margin': float(base_margin),
                    'small_margin': small_margin
                }
            else:
                query_diagnostics['safety_gates'] = {'flip_blocked': False}
            
            # =================================================================
            # FINAL SCORE UPDATE: Apply fusion to z_fused
            # =================================================================
            # NOTE: If significance gate is active, z_fused was already computed
            # inside the gate block (variance-stabilized z-space fusion).
            # Only apply old fusion logic if NOT using significance gate.
            
            if not use_significance_gate:
                # OLD FUSION: Apply temperature-scaled delta for predictable ranking effects
                # Scale by score range to maintain relative ordering
                if score_deltas.max() > score_deltas.min():
                    delta_range = score_deltas.max() - score_deltas.min()
                    # Use GPU tensor directly
                    score_range = float(topk_vals[0] - topk_vals[-1]) if len(topk_vals) > 1 else 1.0
                    # Normalize delta to score range, then apply boost_scale
                    normalized_deltas = score_deltas / (delta_range + 1e-8) * score_range * boost_scale
                else:
                    normalized_deltas = score_deltas * boost_scale
                
                # Update fused scores (GPU)
                z_fused[topk_idx] = s_clip[topk_idx] + normalized_deltas
            # else: z_fused already updated by significance gate block
            
            fused_scores.append(z_fused)
            
            # Track flips
            baseline_top1 = torch.argmax(s_clip).item()
            refined_top1 = torch.argmax(z_fused).item()
            flipped = (refined_top1 != baseline_top1)
            
            # Track executed flips for gate diagnostics
            if flipped:
                gate_diagnostics['total_flips_executed'] += 1
            
            # Find the rank of refined_top1 in the topK candidates
            refined_top1_in_topk = None
            for rank, idx in enumerate(topk_idx):
                if idx.item() == refined_top1:
                    refined_top1_in_topk = rank
                    break
            
            # Store feature values for the refined top-1 candidate (if it's in topK)
            if refined_top1_in_topk is not None:
                # FIXED: Use raw values (contra_raw, dhc_raw, rtrc_raw) instead of z-scored
                contra_val = float(contra_raw[refined_top1_in_topk].item()) if use_contra else float('nan')
                # Ensure it's not NaN when use_contra is True
                if use_contra and (contra_val != contra_val or not np.isfinite(contra_val)):  # Check for NaN/inf
                    contra_val = 0.0
                
                query_diagnostics['feature_values'] = {
                    'rts_z': float(rts_z[refined_top1_in_topk].item()) if use_rts else float('nan'),
                    'ccs_z': float(ccs_z[refined_top1_in_topk].item()) if use_ccs else float('nan'),
                    'rtps_z': float(rtps_z[refined_top1_in_topk].item()) if use_rtps else float('nan'),
                    'contra_z': contra_val,
                    'dhc_z': float(dhc_raw[refined_top1_in_topk].item()) if use_dhc else float('nan'),
                    'curv_z': float(curv_z[refined_top1_in_topk].item()) if use_curv else float('nan'),
                    'rtrc_z': float(rtrc_raw[refined_top1_in_topk].item()) if use_rtrc else float('nan'),
                    'feature_score': float(feature_scores[refined_top1_in_topk].item())
                }
            else:
                # If refined top-1 is not in topK, set all features to NaN
                query_diagnostics['feature_values'] = {
                    'rts_z': float('nan'), 'ccs_z': float('nan'), 'rtps_z': float('nan'),
                    'contra_z': float('nan'), 'dhc_z': float('nan'), 'curv_z': float('nan'),
                    'rtrc_z': float('nan'), 'feature_score': float('nan')
                }
            
            # Store final diagnostics
            query_diagnostics['baseline_top1'] = int(baseline_top1)  # For polarity analysis
            query_diagnostics['refined_top1_idx'] = int(refined_top1)
            query_diagnostics['refined_top1_score'] = float(z_fused[refined_top1].item())
            query_diagnostics['flipped'] = bool(flipped)
            
            # Add missing diagnostic fields for compatibility with analyze_flip_patterns
            query_diagnostics['gated'] = bool(gated)
            query_diagnostics['accepted'] = bool(accepted)
            
            # Compute delta_S (score delta for the refined top-1 candidate)
            if refined_top1_in_topk is not None:
                delta_S = float(score_deltas[refined_top1_in_topk].item())
            else:
                delta_S = float('nan')
            query_diagnostics['delta_S'] = delta_S
            
            # lda_score (not used in spectral refinement, but set for compatibility)
            query_diagnostics['lda_score'] = float('nan')
            
            query_diagnostics['score_deltas'] = {
                'mean': float(score_deltas.mean().item()),
                'std': float(score_deltas.std().item()),
                'max': float(score_deltas.max().item()),
                'min': float(score_deltas.min().item())
            }
            
            if flipped:
                gating_stats['gated'] += 1
                correct_indices = gt_mapping.get(query_id, [])
                if correct_indices:
                    baseline_correct = baseline_top1 in correct_indices
                    refined_correct = refined_top1 in correct_indices
                    query_diagnostics['baseline_correct'] = bool(baseline_correct)
                    query_diagnostics['refined_correct'] = bool(refined_correct)
                    if refined_correct and not baseline_correct:
                        gating_stats['flips_to_correct_total'] += 1  # A: wrong → correct
                    elif baseline_correct and not refined_correct:
                        gating_stats['flips_to_wrong_total'] += 1    # B: correct → wrong
                    elif baseline_correct and refined_correct:
                        gating_stats['flips_neutral_total'] += 1     # C: correct → correct (multi-GT)
                else:
                    query_diagnostics['baseline_correct'] = None
                    query_diagnostics['refined_correct'] = None
            else:
                correct_indices = gt_mapping.get(query_id, [])
                if correct_indices:
                    baseline_correct = baseline_top1 in correct_indices
                    query_diagnostics['baseline_correct'] = bool(baseline_correct)
                    query_diagnostics['refined_correct'] = bool(baseline_correct)
                else:
                    query_diagnostics['baseline_correct'] = None
                    query_diagnostics['refined_correct'] = None
            
            # Append diagnostics
            spectral_diagnostics.append(query_diagnostics)
            
            if save_rankings and per_query_data is not None:
                per_query_data.append({
                    'query_id': query_id,
                    'scores': z_fused.cpu().numpy(),
                    'top1_idx': refined_top1,
                    'baseline_top1': baseline_top1,
                    'flipped': flipped,
                    'features': {
                        'rts_z': rts_z.tolist(),
                        'ccs_z': ccs_z.tolist(),
                        'rtps_z': rtps_z.tolist(),
                        'contra_z': contra_raw.tolist(),
                        'dhc_z': dhc_raw.tolist(),
                        'curv_z': curv_z.tolist(),
                        'rtrc_z': rtrc_raw.tolist()
                    }
                })
            
            # Update progress bar
            pbar.update(1)
    
    # =============================================================================
    # VETO DIAGNOSTIC: Compute Round-Trip Rank and Margin for All Queries
    # =============================================================================
    
    veto_diagnostics = []
    logger.info("📊 Computing veto diagnostics (round-trip rank, margin)...")
    
    for i, (query, query_id) in enumerate(zip(queries, query_ids)):
        # Get baseline scores
        if precomputed_scores is not None:
            s_clip = precomputed_scores[i].squeeze()
        else:
            s_clip = (query.float() @ gallery.float().T).squeeze()
            if apply_siglip_scale and baseline_params.get('backbone_type') == 'siglip':
                s_clip = s_clip * siglip_logit_scale
        
        # Get baseline top-1 and top-2
        topk_vals, topk_idx = torch.topk(s_clip, k=min(2, len(s_clip)), dim=0)
        baseline_top1_idx = topk_idx[0].item()
        baseline_top1_score = topk_vals[0].item()
        
        # Compute margin (score gap between top-1 and top-2)
        if len(topk_vals) > 1:
            margin = baseline_top1_score - topk_vals[1].item()
        else:
            margin = baseline_top1_score  # Only one candidate
        
        # Compute round-trip rank
        rt_rank = None
        if direction == "t2i" and text_bank_gpu is not None:
            # T2I: query is text, baseline top-1 is an image
            # Round-trip: query_text → image (baseline_top1) → all_texts, find rank of query
            baseline_top1_emb = gallery[baseline_top1_idx]  # Image embedding
            
            # Compute similarity of baseline image to all texts in text_bank
            rt_scores = baseline_top1_emb @ text_bank_gpu.T  # [N_texts]
            
            # Find rank of query text in text_bank
            # Need to find which index in text_bank corresponds to this query
            # Approximate: use query embedding to find its position
            query_in_bank_scores = text_bank_gpu @ query  # [N_texts]
            query_in_bank_idx = torch.argmax(query_in_bank_scores).item()
            
            # Get rank of that text in round-trip results
            rt_ranks_all = torch.argsort(rt_scores, descending=True)
            rt_rank = (rt_ranks_all == query_in_bank_idx).nonzero(as_tuple=True)[0]
            if len(rt_rank) > 0:
                rt_rank = rt_rank[0].item() + 1  # 1-indexed rank
            else:
                rt_rank = len(rt_scores)  # Worst rank
        
        elif direction == "i2t" and image_bank_gpu is not None:
            # I2T: query is image, baseline top-1 is a text
            # Round-trip: query_image → text (baseline_top1) → all_images, find rank of query
            baseline_top1_emb = gallery[baseline_top1_idx]  # Text embedding
            
            # Compute similarity of baseline text to all images in image_bank
            rt_scores = baseline_top1_emb @ image_bank_gpu.T  # [N_images]
            
            # Find rank of query image in image_bank
            query_in_bank_scores = image_bank_gpu @ query  # [N_images]
            query_in_bank_idx = torch.argmax(query_in_bank_scores).item()
            
            # Get rank of that image in round-trip results
            rt_ranks_all = torch.argsort(rt_scores, descending=True)
            rt_rank = (rt_ranks_all == query_in_bank_idx).nonzero(as_tuple=True)[0]
            if len(rt_rank) > 0:
                rt_rank = rt_rank[0].item() + 1  # 1-indexed rank
            else:
                rt_rank = len(rt_scores)  # Worst rank
        
        # Check if baseline is correct
        correct_indices = gt_mapping.get(query_id, [])
        baseline_correct = baseline_top1_idx in correct_indices if correct_indices else None
        
        veto_diagnostics.append({
            'query_id': query_id,
            'baseline_top1_idx': baseline_top1_idx,
            'baseline_top1_score': baseline_top1_score,
            'margin': margin,
            'rt_rank': rt_rank,
            'baseline_correct': baseline_correct
        })
    
    logger.info(f"✓ Computed veto diagnostics for {len(veto_diagnostics)} queries")
    
    # Analyze veto precision
    veto_precision_results = analyze_veto_precision(veto_diagnostics)
    
    if veto_precision_results:
        logger.info("\n" + "="*80)
        logger.info("📊 ROUND-TRIP VETO PRECISION ANALYSIS")
        logger.info("="*80)
        logger.info("  Testing different (rt_rank, margin) thresholds to find high-precision regions")
        logger.info("  where baseline is likely wrong and veto would help.")
        logger.info("")
        logger.info("  Format: For queries with rt_rank > R AND margin < τ:")
        logger.info("    - How many queries match this criterion (n_queries, %)")
        logger.info("    - What fraction have wrong baseline (baseline_error_rate)")
        logger.info("    - Upper bound on ΔR@1 if all were fixed (expected_gain)")
        logger.info("")
        
        # Show top 10 regions by expected gain
        logger.info("  Top 10 Suspicious Regions (sorted by expected_gain):")
        logger.info("  " + "-"*76)
        logger.info(f"  {'rt_thresh':>10} {'margin':>10} {'n_queries':>10} {'%':>6} {'error_rate':>12} {'exp_gain':>12}")
        logger.info("  " + "-"*76)
        
        for i, result in enumerate(veto_precision_results[:10]):
            logger.info(
                f"  {result['rt_thresh']:>10} "
                f"{result['margin_thresh']:>10.3f} "
                f"{result['n_queries']:>10} "
                f"{result['pct_queries']:>6.2f} "
                f"{result['baseline_error_rate']:>12.3f} "
                f"{result['expected_gain']:>12.4f}"
            )
        
        logger.info("  " + "-"*76)
        logger.info("")
        
        # Highlight best region
        best = veto_precision_results[0]
        logger.info("  🎯 BEST REGION:")
        logger.info(f"     rt_rank > {best['rt_thresh']}, margin < {best['margin_thresh']:.3f}")
        logger.info(f"     Affects {best['n_queries']} queries ({best['pct_queries']:.2f}%)")
        logger.info(f"     Baseline error rate: {best['baseline_error_rate']:.1%}")
        logger.info(f"     Expected gain (upper bound): +{best['expected_gain']*100:.2f}% R@1")
        logger.info("")
        
        # Interpretation
        if best['baseline_error_rate'] > 0.60 and best['pct_queries'] > 2.0:
            logger.info("  ✅ VERDICT: Veto is JUSTIFIED")
            logger.info("     - Baseline is wrong >60% of time in this region")
            logger.info("     - Affects >2% of queries")
            logger.info("     - Potential for measurable gain if good alternatives exist")
        elif best['baseline_error_rate'] > 0.55 and best['pct_queries'] > 1.0:
            logger.info("  ⚠️  VERDICT: Veto is MARGINAL")
            logger.info("     - Baseline error rate is moderate (55-60%)")
            logger.info("     - Small impact (1-2% of queries)")
            logger.info("     - Likely gain: +0.1-0.3% R@1 (if lucky)")
        else:
            logger.info("  ❌ VERDICT: Veto is NOT JUSTIFIED")
            logger.info("     - Baseline error rate too low (<55%)")
            logger.info("     - OR too few queries affected (<1%)")
            logger.info("     - Features do not reliably identify bad baseline predictions")
        
        logger.info("="*80)
        logger.info("")
    else:
        logger.info("⚠️ No valid veto precision regions found (insufficient data)")
    
    # =============================================================================
    # PASS 2: Apply Global Selection Rule (Mode-Dependent)
    # =============================================================================
    
    # DEBUG: Check if flip attempts were collected
    if use_significance_gate:
        logger.info(f"🔍 DEBUG: use_significance_gate={use_significance_gate}, all_flip_attempts length={len(all_flip_attempts) if 'all_flip_attempts' in locals() else 'NOT DEFINED'}")
    
    if use_significance_gate and all_flip_attempts:
        spectral_mode = baseline_params.get('spectral_mode', 'B')
        
        if spectral_mode == 'A':
            # Mode A: Sort by Δz_fused, keep top flip_rate_cap %
            logger.info(f"\n🎯 Mode A: Sorting by Δz_fused, applying flip rate cap")
            
            # Extract Δz_fused margins (stored as z_eff in best_challenger)
            z_margins = []
            for att in all_flip_attempts:
                if att.get('best_challenger') and att['p_any'] < 1.0:
                    # Passed safety check (p_any < 1.0)
                    z_margin = att['best_challenger'].get('z_eff', 0.0)
                    z_margins.append((att, z_margin))
                else:
                    # Failed safety check or no challenger
                    z_margins.append((att, -999.0))  # Large negative = reject
            
            # Sort by Δz_fused (descending)
            z_margins_sorted = sorted(z_margins, key=lambda x: x[1], reverse=True)
            
            # Keep top flip_rate_cap % (e.g., 15%)
            flip_rate_cap = baseline_params.get('flip_rate_cap', 0.15)
            k_max_flips = int(flip_rate_cap * len(all_flip_attempts))
            
            # Get configurable threshold (tau_z_fused) - allows negative thresholds for combined methods
            tau_z_fused_global = baseline_params.get('tau_z_fused', 0.0)
            
            # Accept top-k with margins above threshold (uses config, not hardcoded 0)
            accept_mask = []
            for idx, (att, z_margin) in enumerate(z_margins_sorted):
                if idx < k_max_flips and z_margin > tau_z_fused_global:
                    accept_mask.append(True)
                else:
                    accept_mask.append(False)
            
            # Map back to original order
            accept_mask_dict = {id(att): acc for (att, _), acc in zip(z_margins_sorted, accept_mask)}
            accept_mask = np.array([accept_mask_dict[id(att)] for att in all_flip_attempts])
            
            # Diagnostics
            z_margin_vals = [z for _, z in z_margins if z > -999]
            logger.info(f"   Flip rate cap: {flip_rate_cap*100:.0f}% → max {k_max_flips} flips")
            logger.info(f"   Margin threshold: τ_z_fused = {tau_z_fused_global:.2f} (from config)")
            logger.info(f"   Δz_fused distribution:")
            logger.info(f"      median: {np.median(z_margin_vals):.4f}")
            logger.info(f"      P10: {np.percentile(z_margin_vals, 10):.4f}")
            logger.info(f"      P90: {np.percentile(z_margin_vals, 90):.4f}")
            logger.info(f"      % positive: {100*np.mean(np.array(z_margin_vals) > 0):.1f}%")
            logger.info(f"   Accepted: {accept_mask.sum()}/{len(all_flip_attempts)}")
            
            # DEBUG: Check if accepted attempts have best_challenger
            accepted_attempts = [att for i, att in enumerate(all_flip_attempts) if accept_mask[i]]
            with_challenger = sum(1 for att in accepted_attempts if att.get('best_challenger') is not None)
            logger.info(f"   🔍 DEBUG: Of {len(accepted_attempts)} accepted, {with_challenger} have best_challenger")
            
            # Initialize Mode B variables to None for Mode A
            pi_0_hat = None
            alpha_out = None
            bh_diagnostics = None
            
        else:
            # Mode B: Storey-BH with robust π₀ estimation
            logger.info(f"\n🎯 Mode B: Applying Storey-BH FDR control across {len(all_flip_attempts)} queries...")
            
            # Extract CCT-combined p-values from all queries
            p_any_all = [att['p_any'] for att in all_flip_attempts]
            
            # Apply Storey-BH
            alpha_out = baseline_params.get('fdr_alpha_out', 0.15)
            accept_mask, pi_0_hat, bh_diagnostics = storey_bh_robust(p_any_all, alpha_out)
            
            logger.info(f"   π₀ estimate: {pi_0_hat:.3f} (capped at 0.99, median over λ-grid)")
            logger.info(f"   λ-grid π₀: {bh_diagnostics['pi_0_estimates']}")
            logger.info(f"   α_eff: {bh_diagnostics['alpha_eff']:.3f}")
            logger.info(f"   k_accept (BH step-up): {bh_diagnostics['k_accept']}")
            logger.info(f"   p-value distribution:")
            logger.info(f"      min: {bh_diagnostics['min_p']:.4f}")
            logger.info(f"      p25: {bh_diagnostics['p25']:.4f}")
            logger.info(f"      median: {bh_diagnostics['median_p']:.4f}")
            logger.info(f"      p75: {bh_diagnostics['p75']:.4f}")
            logger.info(f"      frac < 0.05: {bh_diagnostics['frac_p_lt_0_05']:.3f}")
            logger.info(f"      frac < 0.10: {bh_diagnostics['frac_p_lt_0_1']:.3f}")
            logger.info(f"      frac < 0.20: {bh_diagnostics['frac_p_lt_0_2']:.3f}")
        
        # =================================================================
        # Apply FDR Acceptance Mask First (Before Diagnostics)
        # =================================================================
        
        # Mark which flips pass FDR using the mask
        num_accepted = 0
        num_rejected = 0
        num_confidence_vetoed = 0
        
        for idx, att in enumerate(all_flip_attempts):
            att['fdr_accepted'] = bool(accept_mask[idx])
            
            # ===================================================================
            # CHANGE #3: CONFIDENCE GUARD ON STRONG BASE
            # ===================================================================
            # If base has strong raw confidence, require strong evidence to flip
            if att['fdr_accepted'] and baseline_params.get('enable_confidence_guard', True):
                best_chall = att.get('best_challenger')
                if best_chall and not att.get('skip', False):
                    # Get raw z-scores from stored data
                    s_clip = att['s_clip']
                    
                    # Compute z_raw for base (rank 1) and runner-up (rank 2)
                    baseline_ranking = torch.argsort(s_clip, descending=True)
                    base_idx = baseline_ranking[0].item()
                    runner_up_idx = baseline_ranking[1].item() if len(baseline_ranking) > 1 else base_idx
                    
                    # Compute raw scores (normalize to z-space using global stats)
                    # Use simple z-score: (x - mean) / std
                    s_clip_np = s_clip.cpu().numpy()
                    z_raw_base = (s_clip_np[base_idx] - np.mean(s_clip_np)) / (np.std(s_clip_np) + 1e-8)
                    z_raw_runner = (s_clip_np[runner_up_idx] - np.mean(s_clip_np)) / (np.std(s_clip_np) + 1e-8)
                    
                    margin_raw_z = z_raw_base - z_raw_runner
                    
                    # Get z_fused score for this flip
                    z_fused_score = best_chall.get('z_eff', 0.0)
                    
                    # Confidence guard thresholds
                    margin_threshold = baseline_params.get('confidence_guard_margin', 0.5)
                    z_fused_threshold = baseline_params.get('confidence_guard_z_fused', 1.5)
                    
                    # If base has strong margin, require strong z_fused to flip
                    if margin_raw_z >= margin_threshold and z_fused_score < z_fused_threshold:
                        att['fdr_accepted'] = False
                        att['confidence_vetoed'] = True
                        num_confidence_vetoed += 1
            
            if att['fdr_accepted']:
                num_accepted += 1
                flips_passed_fdr += 1  # Update global counter
            else:
                num_rejected += 1
        
        # =================================================================
        # COMPREHENSIVE DIAGNOSTICS: Filter-as-Mask Approach
        # =================================================================
        
        # Collect statistics from all attempts (including skipped)
        mask_sizes = [att.get('mask_size', 0) for att in all_flip_attempts]
        null_sizes = [att.get('null_size', 0) for att in all_flip_attempts]
        skip_reasons = [att.get('skip_reason', 'none') for att in all_flip_attempts if att.get('skip', False)]
        
        # Non-skipped attempts only
        non_skipped = [att for att in all_flip_attempts if not att.get('skip', False)]
        
        logger.info(f"\n📊 === FILTER-AS-MASK DIAGNOSTICS ===")
        
        # 1. Mask size distribution
        if mask_sizes:
            mask_only_baseline = sum(1 for m in mask_sizes if m == 1)
            logger.info(f"   Mask sizes:")
            logger.info(f"      mean: {np.mean(mask_sizes):.1f}, median: {np.median(mask_sizes):.1f}")
            logger.info(f"      P10: {np.percentile(mask_sizes, 10):.1f}, P90: {np.percentile(mask_sizes, 90):.1f}")
            logger.info(f"      % queries where mask == {{base_idx}}: {100*mask_only_baseline/len(mask_sizes):.2f}%")
        
        # 2. Null size distribution  
        if null_sizes:
            null_too_small = sum(1 for n in null_sizes if n < 10)
            logger.info(f"   Null sizes:")
            logger.info(f"      mean: {np.mean(null_sizes):.1f}, median: {np.median(null_sizes):.1f}")
            logger.info(f"      P10: {np.percentile(null_sizes, 10):.1f}, P90: {np.percentile(null_sizes, 90):.1f}")
            logger.info(f"      % queries with null < 10: {100*null_too_small/len(null_sizes):.2f}%")
        
        # 3. Skip reason breakdown
        if skip_reasons:
            skip_counts = {}
            for reason in skip_reasons:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
            logger.info(f"   Skip reasons: {skip_counts}")
        else:
            logger.info(f"   No queries skipped ✓")
        
        # 4. σ_tilde distribution (non-skipped only)
        sigma_vals = [att.get('sigma_tilde', 0) for att in non_skipped if att.get('sigma_tilde', 0) > 0]
        if sigma_vals:
            logger.info(f"   σ_tilde distribution:")
            logger.info(f"      median: {np.median(sigma_vals):.6f}")
            logger.info(f"      P10: {np.percentile(sigma_vals, 10):.6f}")
            logger.info(f"      P90: {np.percentile(sigma_vals, 90):.6f}")
        
        # 5. Margin distribution (Z-FUSED scores - what we test, non-skipped only)
        margin_z_fused_vals = [att['best_challenger'].get('margin_z_fused', att['best_challenger'].get('margin_fused', 0)) 
                                for att in non_skipped if att.get('best_challenger')]
        margin_raw_vals = [att['best_challenger']['margin_raw'] for att in non_skipped if att.get('best_challenger')]
        
        if margin_z_fused_vals:
            margin_z_fused_positive_frac = sum(1 for m in margin_z_fused_vals if m > 0) / len(margin_z_fused_vals)
            logger.info(f"   Margin distribution (Z-FUSED scores - TESTED!):")
            logger.info(f"      median: {np.median(margin_z_fused_vals):.6f}")
            logger.info(f"      P10: {np.percentile(margin_z_fused_vals, 10):.6f}")
            logger.info(f"      P90: {np.percentile(margin_z_fused_vals, 90):.6f}")
            logger.info(f"      % positive: {100*margin_z_fused_positive_frac:.1f}% ← KEY METRIC!")
            if margin_z_fused_positive_frac < 0.5:
                logger.warning(f"      ⚠️  NOTE: <50% z-fused margins positive. Expected with strict gate.")
        
        if margin_raw_vals:
            margin_raw_positive_frac = sum(1 for m in margin_raw_vals if m > 0) / len(margin_raw_vals)
            logger.info(f"   Margin distribution (RAW scores - diagnostic only):")
            logger.info(f"      median: {np.median(margin_raw_vals):.6f}")
            logger.info(f"      P10: {np.percentile(margin_raw_vals, 10):.6f}")
            logger.info(f"      P90: {np.percentile(margin_raw_vals, 90):.6f}")
            logger.info(f"      % positive: {100*margin_raw_positive_frac:.1f}%")
        
        # 6. FDR flow metrics
        queries_attempted = len(all_flip_attempts)
        queries_non_skipped = len(non_skipped)
        queries_accepted = num_accepted
        logger.info(f"   FDR flow:")
        logger.info(f"      Total queries: {queries_attempted}")
        logger.info(f"      Non-skipped (testable): {queries_non_skipped} ({100*queries_non_skipped/queries_attempted:.1f}%)")
        logger.info(f"      Passed within-query (p_any <= 0.2): {sum(1 for att in non_skipped if att.get('p_any', 1) <= 0.2)}")
        if bh_diagnostics is not None:
            logger.info(f"      Passed Storey-BH (α_eff={bh_diagnostics['alpha_eff']:.3f}): {queries_accepted} ({100*queries_accepted/queries_attempted:.1f}%)")
        else:
            logger.info(f"      Passed flip rate cap: {queries_accepted} ({100*queries_accepted/queries_attempted:.1f}%)")
        
        # 7. Winner rank distribution (for accepted flips only)
        accepted_attempts = [att for att in all_flip_attempts if att.get('fdr_accepted', False)]
        if accepted_attempts:
            winner_ranks = [att['best_challenger']['base_rank'] for att in accepted_attempts if att.get('best_challenger')]
            if winner_ranks:
                logger.info(f"   Winner rank distribution (within top-K, accepted flips only):")
                logger.info(f"      median: {np.median(winner_ranks):.1f}")
                logger.info(f"      P10: {np.percentile(winner_ranks, 10):.1f}, P90: {np.percentile(winner_ranks, 90):.1f}")
                near_tie_winners = sum(1 for r in winner_ranks if r <= 5)
                hidden_gem_winners = sum(1 for r in winner_ranks if r >= 20)
                logger.info(f"      Rank 2-5 (near-ties): {near_tie_winners}/{len(winner_ranks)} ({100*near_tie_winners/len(winner_ranks):.1f}%)")
                logger.info(f"      Rank 20+ (hidden gems): {hidden_gem_winners}/{len(winner_ranks)} ({100*hidden_gem_winners/len(winner_ranks):.1f}%)")
        
        # 8. Correlation sanity check (sample a few queries)
        logger.info(f"   Correlation checks (spectral vs raw, 10 random queries):")
        sample_indices = np.random.choice(len(queries), min(10, len(queries)), replace=False)
        rho_null_list = []
        rho_topk_list = []
        
        for sample_idx in sample_indices:
            query = queries[sample_idx]
            s_clip = (query.float() @ gallery.float().T).squeeze()
            if apply_siglip_scale and baseline_params.get('backbone_type') == 'siglip':
                s_clip = s_clip * siglip_logit_scale
            
            K_actual = min(topK, s_clip.size(0))
            topk_vals, topk_idx = torch.topk(s_clip, k=K_actual, dim=0)
            
            # Quick spectral score (reuse feature scores if available)
            try:
                from scipy.stats import spearmanr
                # Use a simple proxy: rank correlation between raw scores and positions
                # (Real spectral would require recomputation, so we use rank as proxy)
                raw_ranks = torch.argsort(torch.argsort(topk_vals, descending=True)).cpu().numpy()
                positions = np.arange(len(raw_ranks))
                
                # For null (bottom half)
                null_start = K_actual // 2
                null_raw = topk_vals[null_start:].cpu().numpy()
                null_pos = positions[null_start:]
                if len(null_raw) >= 3:
                    rho_null, _ = spearmanr(null_raw, null_pos)
                    rho_null_list.append(rho_null)
                
                # For top-K (all)
                rho_topk, _ = spearmanr(topk_vals.cpu().numpy(), positions)
                rho_topk_list.append(rho_topk)
            except:
                pass
        
        if rho_null_list:
            logger.info(f"      Spearman ρ (raw vs rank) on null: mean={np.mean(rho_null_list):.3f}")
            if np.mean(rho_null_list) < -0.9:
                logger.info(f"         ✓ Strong rank-score correlation (expected for sorted data)")
        if rho_topk_list:
            logger.info(f"      Spearman ρ (raw vs rank) on top-K: mean={np.mean(rho_topk_list):.3f}")
        
        logger.info(f"   === END DIAGNOSTICS ===\n")
        
        # Log acceptance summary
        logger.info(f"   ✅ Accepted flips: {num_accepted} ({100*num_accepted/len(all_flip_attempts):.1f}%)")
        logger.info(f"   ❌ Rejected flips: {num_rejected} ({100*num_rejected/len(all_flip_attempts):.1f}%)")
        if num_confidence_vetoed > 0:
            logger.info(f"   🛡️  Confidence guard vetoed: {num_confidence_vetoed} ({100*num_confidence_vetoed/len(all_flip_attempts):.1f}%)")
        
        # CRITICAL FIX: Use list_idx (not query_idx) to revert rejected flips!
        for att in all_flip_attempts:
            if not att['fdr_accepted']:
                # Revert to baseline ranking using correct list index
                list_idx = att['list_idx']
                fused_scores[list_idx] = att['s_clip']  # Restore baseline scores
        
        logger.info(f"   ⚠️  FDR gate active: {num_rejected} flips blocked (wire-up FIXED)")
        logger.info(f"   📊 FDR Counters: {flip_attempts_fdr} attempts, {flips_passed_fdr} passed ({100*flips_passed_fdr/flip_attempts_fdr:.1f}%)")
        
        # ===================================================================
        # P0 ACCOUNTING FIX: Update spectral_diagnostics with ACTUAL flips
        # ===================================================================
        # Recompute actual flips from final fused_scores after revert
        logger.info(f"\n🔧 P0: Recomputing actual flips after FDR revert...")
        actual_flips_count = 0
        accepted_flip_ranks = []  # For Phase 1 diagnostics
        accepted_flip_z_fused = []  # For Phase 1 diagnostics
        
        # DEBUG: Diagnose decision collection
        logger.info(f"🔍 DEBUG: Decision collection diagnostics:")
        logger.info(f"   Total flip attempts: {len(all_flip_attempts)}")
        fdr_accepted_count = sum(1 for att in all_flip_attempts if att.get('fdr_accepted', False))
        has_challenger_count = sum(1 for att in all_flip_attempts if att.get('best_challenger') is not None)
        both_count = sum(1 for att in all_flip_attempts if att.get('fdr_accepted', False) and att.get('best_challenger') is not None)
        logger.info(f"   FDR accepted: {fdr_accepted_count}")
        logger.info(f"   Has best_challenger: {has_challenger_count}")
        logger.info(f"   Both (fdr_accepted AND best_challenger): {both_count}")
        
        for att in all_flip_attempts:
            query_idx = att['query_idx']
            list_idx = att['list_idx']
            
            # Compute baseline and final top-1
            baseline_top1 = torch.argmax(att['s_clip']).item()
            final_top1 = torch.argmax(fused_scores[list_idx]).item()
            actual_flip = (final_top1 != baseline_top1)
            
            # Update spectral_diagnostics with correct flip status
            if query_idx < len(spectral_diagnostics):
                spectral_diagnostics[query_idx]['flipped'] = actual_flip
                
                # Also update refined_top1_idx with final top-1
                spectral_diagnostics[query_idx]['refined_top1_idx'] = final_top1
                spectral_diagnostics[query_idx]['refined_top1_score'] = float(fused_scores[list_idx][final_top1].item())
                
                if actual_flip:
                    actual_flips_count += 1
                    
                    # Collect decision for transplant (NNN+Spectral composition)
                    # IMPORTANT: Collect ALL actual flips, not just FDR-accepted ones
                    # This allows decision transplant to work even when gates are disabled
                    if return_decisions:
                        # Get rank of winner in baseline ranking
                        baseline_ranking = torch.argsort(att['s_clip'], descending=True)
                        winner_rank = (baseline_ranking == final_top1).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
                        
                        spectral_decisions.append({
                            'query_idx': query_idx,
                            'base_winner_id': baseline_top1,  # Gallery ID of baseline top-1
                            'challenger_id': final_top1,      # Gallery ID of challenger (new top-1)
                            'challenger_idx': int(final_top1),  # Gallery index for transplant
                            'delta_z_fused': att.get('best_challenger', {}).get('z_eff', 0.0),  # Effect size
                            'raw_rank_of_challenger': winner_rank,  # Rank in raw baseline (1-indexed)
                            'margin_z_fused': att.get('best_challenger', {}).get('margin_z_fused', 0.0),
                            'margin_raw': att.get('best_challenger', {}).get('margin_raw', 0.0)
                        })
                    
                    # Phase 1: Track rank and z_fused for accepted flips (for diagnostics)
                    if att.get('fdr_accepted', False) and att.get('best_challenger'):
                        # Get rank of winner in baseline ranking
                        baseline_ranking = torch.argsort(att['s_clip'], descending=True)
                        winner_rank = (baseline_ranking == final_top1).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
                        accepted_flip_ranks.append(winner_rank)
                        accepted_flip_z_fused.append(att['best_challenger'].get('z_eff', 0.0))
        
        logger.info(f"   ✅ Actual flips after revert: {actual_flips_count}/{len(all_flip_attempts)}")
        logger.info(f"   📊 Accounting reconciled: {num_accepted} accepted → {actual_flips_count} executed")
        logger.info(f"   📋 Spectral decisions collected: {len(spectral_decisions)} (for NNN+Spectral transplant)")
        
        # ===================================================================
        # PHASE 1: RANK-BINNED DIAGNOSTICS
        # ===================================================================
        if accepted_flip_ranks:
            logger.info(f"\n📊 PHASE 1: Rank-Binned Δz_fused Diagnostics")
            logger.info(f"   Analyzing {len(accepted_flip_ranks)} accepted flips by rank bin...\n")
            
            # Convert to numpy for easier binning
            ranks_arr = np.array(accepted_flip_ranks)
            z_fused_arr = np.array(accepted_flip_z_fused)
            
            # Define rank bins
            rank_bins = [
                ('2-5', 2, 5),
                ('6-10', 6, 10),
                ('11-20', 11, 20),
                ('20+', 21, 999)
            ]
            
            for bin_name, lb, ub in rank_bins:
                mask = (ranks_arr >= lb) & (ranks_arr <= ub)
                bin_flips = z_fused_arr[mask]
                
                if len(bin_flips) > 0:
                    median_z = np.median(bin_flips)
                    pct_pos = 100 * np.mean(bin_flips > 0)
                    mean_z = np.mean(bin_flips)
                    p25, p75 = np.percentile(bin_flips, [25, 75])
                    
                    logger.info(f"   Ranks {bin_name:6s}: n={len(bin_flips):4d}, "
                              f"median Δz_fused={median_z:+.3f}, "
                              f"mean={mean_z:+.3f}, "
                              f"Q25={p25:+.3f}, Q75={p75:+.3f}, "
                              f"%pos={pct_pos:5.1f}%")
                else:
                    logger.info(f"   Ranks {bin_name:6s}: n=   0 (no flips in this bin)")
            
            logger.info(f"\n   💡 KEY INSIGHT: If rank≥20 has comparable Δz_fused to rank 2-5 but n=0,")
            logger.info(f"      then we're missing good deep flips (supports P2 budget split).")
            logger.info(f"      If rank≥20 has much lower Δz_fused, then P2 would force bad flips.")
        else:
            logger.info(f"\n⚠️  No accepted flips with valid rank info for Phase 1 diagnostics")
        
        # ===================================================================
        # PHASE 1B: UNCHOSEN CANDIDATES ANALYSIS (Root Cause Diagnostic)
        # ===================================================================
        logger.info(f"\n📊 PHASE 1B: Unchosen Candidates Analysis")
        logger.info(f"   Analyzing ALL queries to find why ranks ≥6 are starved...\n")
        
        # Collect data across all queries
        unchosen_data = {
            '6-10': {'in_mask': [], 'z_fused': [], 'chosen': []},
            '11-20': {'in_mask': [], 'z_fused': [], 'chosen': []},
            '20+': {'in_mask': [], 'z_fused': [], 'chosen': []}
        }
        
        for att in all_flip_attempts:
            if att.get('skip', False) or not att.get('best_challenger'):
                continue
            
            # Get baseline ranking (indices sorted by baseline score, descending)
            s_clip = att['s_clip']
            baseline_ranking = torch.argsort(s_clip, descending=True)
            
            # Get mask candidates (we need to reconstruct mask or use stored info)
            # For now, we'll analyze the best_challenger and challengers list
            challengers = att.get('challengers', [])
            best_chall = att['best_challenger']
            
            # Compute rank in baseline for best challenger
            chall_idx = best_chall.get('chall_idx')
            if chall_idx is None:
                continue
            
            # Find rank of this challenger in baseline (1-indexed)
            rank_in_baseline = (baseline_ranking == chall_idx).nonzero(as_tuple=True)[0].item() + 1
            
            # Get z_fused score
            z_fused_score = best_chall.get('z_eff', 0.0)
            
            # Determine bin
            bin_name = None
            if 6 <= rank_in_baseline <= 10:
                bin_name = '6-10'
            elif 11 <= rank_in_baseline <= 20:
                bin_name = '11-20'
            elif rank_in_baseline > 20:
                bin_name = '20+'
            
            if bin_name:
                unchosen_data[bin_name]['in_mask'].append(1)
                unchosen_data[bin_name]['z_fused'].append(z_fused_score)
                unchosen_data[bin_name]['chosen'].append(1 if att.get('fdr_accepted', False) else 0)
        
        # Report statistics
        logger.info(f"   Analysis based on best_challenger from each query:")
        logger.info(f"   (Shows candidates that WERE considered but may not have been chosen)\n")
        
        for bin_name in ['6-10', '11-20', '20+']:
            data = unchosen_data[bin_name]
            n_in_mask = len(data['in_mask'])
            
            if n_in_mask > 0:
                z_arr = np.array(data['z_fused'])
                chosen_arr = np.array(data['chosen'])
                
                median_z = np.median(z_arr)
                mean_z = np.mean(z_arr)
                p25, p75 = np.percentile(z_arr, [25, 75])
                pct_good = 100 * np.mean(z_arr > 0.5)
                pct_pos = 100 * np.mean(z_arr > 0.0)
                n_chosen = int(np.sum(chosen_arr))
                
                logger.info(f"   Ranks {bin_name:6s}:")
                logger.info(f"      Candidates in mask: {n_in_mask:4d}")
                logger.info(f"      Δz_fused: median={median_z:+.3f}, mean={mean_z:+.3f}, Q25={p25:+.3f}, Q75={p75:+.3f}")
                logger.info(f"      Quality: {pct_pos:5.1f}% positive, {pct_good:5.1f}% > 0.5 (strong)")
                logger.info(f"      Actually chosen: {n_chosen}/{n_in_mask} ({100*n_chosen/n_in_mask:.1f}%)")
                
                if n_chosen < n_in_mask * 0.3 and pct_good > 30:
                    logger.info(f"      ⚠️  HIGH-QUALITY candidates exist but NOT chosen!")
                logger.info(f"")
            else:
                logger.info(f"   Ranks {bin_name:6s}: No candidates found in mask")
                logger.info(f"")
        
        logger.info(f"   💡 INTERPRETATION:")
        logger.info(f"      - If 'Candidates in mask' is low → Mask is TOO RESTRICTIVE (expand mask)")
        logger.info(f"      - If 'Candidates in mask' is high but 'Actually chosen' is low:")
        logger.info(f"        • Low quality (Δz_fused negative) → Good that they weren't chosen")
        logger.info(f"        • High quality (% > 0.5 is high) → Cap/sorting is BLOCKING them (P2/P3 needed)")
        
        # Store FDR results for diagnostics
        gate_diagnostics['fdr_control'] = {
            'total_attempts': len(all_flip_attempts),
            'accepted': num_accepted,
            'rejected': num_rejected,
            'pi_0_hat': float(pi_0_hat) if pi_0_hat is not None else None,
            'alpha_out': alpha_out,
            'method': 'cct_storey_bh_fixed' if bh_diagnostics is not None else 'cct_flip_rate_cap',
            'flip_attempts_fdr': flip_attempts_fdr,
            'flips_passed_fdr': flips_passed_fdr,
            'bh_diagnostics': bh_diagnostics
        }
    
    # Convert to tensor
    fused_scores = torch.stack(fused_scores)  # [N_queries, N_gallery]
    
    # Apply eval temperature scaling
    fused_scores = fused_scores / eval_temperature
    
    # =============================================================================
    # POST-EVALUATION: Compute metrics and GT tracking (NO GT LEAKAGE)
    # =============================================================================
    # IMPORTANT: GT is ONLY used here for evaluation/metrics AFTER all ranking
    # decisions have been made. GT is NEVER used during the spectral refinement
    # decision-making process (flip selection, gating, etc.).
    # =============================================================================
    
    # Compute retrieval metrics
    k_values = [1, 5, 10]
    recalls = {f'recall@{k}': 0.0 for k in k_values}
    
    # Track GT at rank-1 for each query (POST-EVALUATION ONLY - for finding spectral-better cases)
    # This is computed AFTER fused_scores are finalized, purely for analysis.
    spectral_gt_at_r1 = {}
    
    for i, query_id in enumerate(query_ids):
        positive_indices = set(gt_mapping[query_id])
        query_scores = fused_scores[i]
        top_k_indices = torch.topk(query_scores, k=max(k_values), dim=0).indices.cpu().numpy()
        
        # Check if GT is at rank-1 (POST-EVALUATION - no influence on rankings)
        top1_idx = int(top_k_indices[0]) if len(top_k_indices) > 0 else None
        has_gt_at_r1 = (top1_idx in positive_indices) if top1_idx is not None else False
        spectral_gt_at_r1[query_id] = has_gt_at_r1
        
        for k in k_values:
            top_k = top_k_indices[:k]
            recall = len(set(top_k) & positive_indices) > 0
            recalls[f'recall@{k}'] += recall
    
    for k in k_values:
        recalls[f'recall@{k}'] /= total_queries
    
    # Prepare results
    results = {}
    results.update(recalls)
    
    # ACCURATE CUDA TIMING: sync before measuring end time
    results['fusion_time'] = end_cuda_timer(start_time)
    results['total_queries'] = total_queries
    
    # Clean flip statistics: A (help), B (hurt), C (neutral), D (wrong→wrong)
    A = gating_stats['flips_to_correct_total']   # wrong → correct
    B = gating_stats['flips_to_wrong_total']     # correct → wrong
    C = gating_stats['flips_neutral_total']      # correct → correct (multi-GT)
    gated_total = gating_stats['gated']          # ALL flips (A + B + C + D) = execution budget K
    D = gated_total - A - B - C                  # wrong → wrong (inferred)
    affected = A + B                             # queries where correctness changed (realized flips F)
    
    # IMPORTANT: distinguish between:
    # - gate_rate: %queries where refinement was ATTEMPTED (polarity gate passed)
    # - top1_flip_rate: %queries where top-1 actually CHANGED after refinement
    # 
    # FIX: The gate_rate was always 0 because the counters weren't being tracked properly.
    # Priority for attempted count:
    #   1. fdr_control['total_attempts'] - when significance gate is ON
    #   2. gating_stats['polarity_gated'] - when polarity-aware is ON (NEW)
    #   3. gate_diagnostics['total_flip_attempts'] - old gate (fallback)
    #   4. gating_stats['gated'] - actual flips (worst case fallback)
    fdr_control = gate_diagnostics.get('fdr_control', {})
    polarity_gated = gating_stats.get('polarity_gated', 0)
    old_gate_attempts = gate_diagnostics.get('total_flip_attempts', 0)
    
    if fdr_control and fdr_control.get('total_attempts', 0) > 0:
        # Significance gate is ON - use FDR total_attempts
        attempted = int(fdr_control.get('total_attempts', 0))
    elif polarity_gated > 0:
        # Polarity-aware gating is ON - use polarity_gated count
        attempted = int(polarity_gated)
    elif old_gate_attempts > 0:
        # Old gate tracking
        attempted = int(old_gate_attempts)
    else:
        # Fallback: use actual flips (gated_total includes A+B+C+D)
        attempted = int(gated_total)
    
    results['gate_rate'] = attempted / total_queries if total_queries > 0 else 0.0
    results['top1_flip_rate'] = gated_total / total_queries if total_queries > 0 else 0.0
    
    # =============================================================================
    # STAGE 1A STATS: Add budget-based selection metrics
    # =============================================================================
    results['stage1a_sel'] = stage1a_stats.get('actual_rate', 1.0) * 100.0  # Percentage
    results['stage1a_n_selected'] = stage1a_stats.get('n_selected', total_queries)
    results['stage1a_n_total'] = stage1a_stats.get('n_total', total_queries)
    results['stage1a_budget'] = stage1a_stats.get('budget', 1.0)
    results['stage1a_mode'] = stage1a_stats.get('frozen_bcg_mode', None)
    
    # Log Stage 1a summary
    if stage1a_stats.get('budget', 1.0) < 1.0:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STAGE 1A SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"  Budget: {stage1a_stats['budget']*100:.0f}%")
        logger.info(f"  Selected: {stage1a_stats['n_selected']}/{stage1a_stats['n_total']} ({stage1a_stats['actual_rate']*100:.1f}%)")
        logger.info(f"  Mode: {stage1a_stats['frozen_bcg_mode']}")
        logger.info(f"  Processed: {stage1a_processed_count}, Skipped: {stage1a_skipped_count}")
        logger.info(f"{'='*80}")
    
    # CORRECT budget-based metrics:
    # K = gated_total = execution budget (queries selected for refinement)
    # F = A + B = realized flips (subset where correctness changed)
    # NetFlip/1k = 1000 * (A - B) / K = net benefit per 1k budget units
    K = max(gated_total, 1)
    results['netflip_per_1k'] = 1000.0 * (A - B) / K
    results['flip_precision'] = A / max(affected, 1) if affected > 0 else 0.0
    results['flip_rate_ratio'] = affected / K if K > 0 else 0.0
    
    # Additional diagnostics
    results['flips_A'] = A
    results['flips_B'] = B
    results['flips_C'] = C
    results['flips_D'] = D  # wrong → wrong flips
    results['neutral_rate'] = C / max(gated_total, 1)  # C / total flips
    
    # Save query_ids and gallery_ids for proper mapping in visualization/gallery generation
    results['query_ids'] = query_ids
    results['gallery_ids'] = gallery_ids
    
    # Store GT at rank-1 status for each query (compact format for finding spectral-better cases)
    results['spectral_gt_at_r1'] = spectral_gt_at_r1
    
    # Store spectral_diagnostics for feature statistics logging (used by geometric baselines)
    # Note: This contains per-query feature values and diagnostics needed for A vs B analysis
    results['spectral_diagnostics'] = spectral_diagnostics
    
    # Check if we should save similarity matrix (can be very large)
    save_sim_matrix = baseline_params.get('save_similarity_matrix', True)
    
    # Save similarity matrix for visualization (convert to CPU numpy for JSON serialization)
    if save_sim_matrix:
        if isinstance(fused_scores, torch.Tensor):
            results['similarity_matrix'] = fused_scores.cpu().numpy().tolist()
        else:
            results['similarity_matrix'] = fused_scores.tolist() if hasattr(fused_scores, 'tolist') else fused_scores
    
    # A/B/C/D/F analysis using analyze_flip_patterns
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 DETAILED FLIP ANALYSIS")
        logger.info(f"{'='*80}")
        
        sim = fused_scores  # Already temperature-scaled
        debug = analyze_flip_patterns(
            query_ids=query_ids,
            gt_mapping=gt_mapping,
            hhd_similarity_matrix=sim,
            hhd_diagnostics=spectral_diagnostics,
            baseline_type='spectral_refinement',
            total_queries=len(query_ids),
            base_rankings=base_rankings
        )
        
        logger.info(f"\n✓ Flip analysis completed successfully")
        
        # Safely convert counts to int, handling NaN/inf cases
        def safe_int(x, default=0):
            try:
                if isinstance(x, (int, np.integer)):
                    return int(x)
                x_float = float(x)
                if np.isnan(x_float) or np.isinf(x_float):
                    return default
                return int(x_float)
            except (ValueError, TypeError, OverflowError):
                return default
        
        A_count = safe_int(debug.get('A_count', 0))
        B_count = safe_int(debug.get('B_count', 0))
        C_count = safe_int(debug.get('C_count', 0))
        D_count = safe_int(debug.get('D_count', 0))
        F_count = safe_int(debug.get('F_count', 0))
        
        # Safe calculations with zero-division protection
        flip_precision = (A_count / max(1, (A_count + B_count))) if (A_count + B_count) > 0 else 0.0
        net = A_count - B_count
        hurt_rate = (B_count / (A_count + B_count)) if (A_count + B_count) > 0 else 0.0
        waste = C_count + D_count
        
        logger.info(f"\n{'='*80}")
        logger.info("📊 A/B/C/D/F ACCOUNTING (Spectral Refinement)")
        logger.info(f"{'='*80}")
        logger.info("  A (help): base was wrong → after flip is right (GT at rank-1).")
        logger.info("  B (hurt): base was right → after flip is wrong.")
        logger.info("  C (still wrong): base wrong → flip to another wrong (wasted).")
        logger.info("  D (still right): base right → flip to another GT (still right; wasted but safe).")
        logger.info(
            f"  F={F_count}, A={A_count}, B={B_count}, C={C_count}, D={D_count}, "
            f"Net={net:+d}, Precision={flip_precision:.4f}, Hurt rate={hurt_rate:.4f}, Waste={waste}"
        )
        
        # Log per-flip details
        per_flip_details = debug.get('per_flip_details', [])
        if per_flip_details:
            logger.info(f"\n  🔍 Per-Flip Details ({len(per_flip_details)} flips):")
            logger.info(f"    {'Query ID':<20} {'Category':<8} {'Baseline Top1':<15} {'Refined Top1':<15} {'Score Δ':<10} {'Baseline':<10} {'Refined':<10}")
            logger.info(f"    {'-'*20} {'-'*8} {'-'*15} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
            
            # Group by category for better readability
            for category in ['A', 'B', 'C', 'D']:
                category_flips = [f for f in per_flip_details if f['category'] == category]
                if category_flips:
                    logger.info(f"\n    Category {category} ({len(category_flips)} flips):")
                    for flip in category_flips[:10]:  # Show first 10 of each category
                        logger.info(f"      {flip['query_id']:<20} {flip['category']:<8} "
                                  f"{flip['baseline_top1']:<15} {flip['refined_top1']:<15} "
                                  f"{flip['score_delta']:+8.4f}   "
                                  f"{'✓' if flip['baseline_correct'] else '✗':<10} "
                                  f"{'✓' if flip['refined_correct'] else '✗':<10}")
                    if len(category_flips) > 10:
                        logger.info(f"      ... and {len(category_flips) - 10} more {category} flips")
        else:
            logger.info("\n  ⚠️  No per-flip details available")
        
        # Log hurt feature statistics if available
        hurt_feature_stats = debug.get('hurt_feature_statistics', {})
        if hurt_feature_stats:
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 HURT FLIP FEATURE STATISTICS (n={hurt_feature_stats.get(list(hurt_feature_stats.keys())[0], {}).get('count', 0) if hurt_feature_stats else 0})")
            logger.info(f"{'='*80}")
            for feat_name, stats in hurt_feature_stats.items():
                logger.info(f"  {feat_name}: μ={stats['mean']:.3f}, median={stats['median']:.3f}, "
                          f"q25={stats['q25']:.3f}, q75={stats['q75']:.3f}")
        else:
            logger.info(f"\n  ⚠️  No hurt flip feature statistics available (B_count={B_count})")
        
        # Log comparison statistics if available
        comparison_stats = debug.get('comparison_statistics', {})
        if comparison_stats:
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 COMPARISON: Helpful (A) vs Hurt (B) Flips")
            logger.info(f"{'='*80}")
            logger.info(f"  {'Feature':<15s} | {'Helpful (A) μ':<13s} | {'Hurt (B) μ':<11s} | {'Difference':<12s}")
            logger.info(f"  {'-'*60}")
            for feat_name, stats in comparison_stats.items():
                logger.info(f"  {feat_name:<15s} | {stats['helpful_mean']:11.3f}   | {stats['hurt_mean']:9.3f}   | {stats['difference']:+.3f}")
        else:
            logger.info(f"\n  ⚠️  No comparison statistics available (A_count={A_count}, B_count={B_count})")
        
        results.update({
            'A_count': A_count,
            'B_count': B_count,
            'C_count': C_count,
            'D_count': D_count,
            'F_count': F_count,
            'flip_precision': float(flip_precision),
            'hurt_rate': float(hurt_rate),
            'waste': waste,
            'per_flip_details': per_flip_details,  # Save per-flip details for visualization
            'hurt_feature_statistics': hurt_feature_stats,  # Feature stats for hurt flips
            'comparison_statistics': comparison_stats,  # A vs B comparison
            'debug_analysis': {  # Save debug analysis for feature statistics
                'movement_matrix': debug.get('movement_matrix', []),
                'per_rank_win_rates': debug.get('per_rank_win_rates', {}),
                'margin_decile_analysis': debug.get('margin_decile_analysis', {})
            }
        })
    except Exception as e:
        logger.error(f"❌ Spectral Refinement A/B/C/D analysis failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    # Save per-query data if requested
    if save_rankings and per_query_data and rankings_save_path:
        import os
        os.makedirs(os.path.dirname(rankings_save_path), exist_ok=True)
        with open(rankings_save_path, 'wb') as f:
            pickle.dump(per_query_data, f)
        logger.info(f"💾 Saved per-query rankings to {rankings_save_path}")
    
    # =============================================================================
    # 📊 GATE DIAGNOSTICS SUMMARY
    # =============================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"🚪 LABEL-FREE GATE DIAGNOSTICS")
    logger.info(f"{'='*80}")
    
    total_attempts = gate_diagnostics['total_flip_attempts']
    total_executed = gate_diagnostics['total_flips_executed']
    
    if total_attempts > 0:
        logger.info(f"\n📈 Flip Attempt Statistics:")
        logger.info(f"  Total flip attempts (where spectral suggested a flip): {total_attempts}")
        logger.info(f"  Total flips executed (actually changed rank-1): {total_executed}")
        logger.info(f"  Flip execution rate: {100.0 * total_executed / total_attempts:.1f}%")
        logger.info(f"  Flips blocked by gates: {total_attempts - gate_diagnostics['all_gates_passed']}")
        
        logger.info(f"\n🚪 Individual Gate Pass Rates:")
        
        # Sharpness gate
        if gate_diagnostics['sharpness_gate_attempts'] > 0:
            sharp_pass_rate = 100.0 * gate_diagnostics['sharpness_gate_passed'] / gate_diagnostics['sharpness_gate_attempts']
            logger.info(f"  Sharpness-Aware Gate:")
            logger.info(f"    ├─ Attempts: {gate_diagnostics['sharpness_gate_attempts']}")
            logger.info(f"    ├─ Passed:   {gate_diagnostics['sharpness_gate_passed']} ({sharp_pass_rate:.1f}%)")
            logger.info(f"    └─ Failed:   {gate_diagnostics['sharpness_gate_failed']} ({100-sharp_pass_rate:.1f}%)")
        
        # Consensus gate
        if gate_diagnostics['consensus_gate_attempts'] > 0:
            cons_pass_rate = 100.0 * gate_diagnostics['consensus_gate_passed'] / gate_diagnostics['consensus_gate_attempts']
            logger.info(f"  Consensus (Community Vote) Gate:")
            logger.info(f"    ├─ Attempts: {gate_diagnostics['consensus_gate_attempts']}")
            logger.info(f"    ├─ Passed:   {gate_diagnostics['consensus_gate_passed']} ({cons_pass_rate:.1f}%)")
            logger.info(f"    └─ Failed:   {gate_diagnostics['consensus_gate_failed']} ({100-cons_pass_rate:.1f}%)")
        
        # LS margin gate
        if gate_diagnostics['ls_margin_gate_attempts'] > 0:
            ls_pass_rate = 100.0 * gate_diagnostics['ls_margin_gate_passed'] / gate_diagnostics['ls_margin_gate_attempts']
            logger.info(f"  LS Margin (Query-Adaptive Threshold) Gate:")
            logger.info(f"    ├─ Attempts: {gate_diagnostics['ls_margin_gate_attempts']}")
            logger.info(f"    ├─ Passed:   {gate_diagnostics['ls_margin_gate_passed']} ({ls_pass_rate:.1f}%)")
            logger.info(f"    └─ Failed:   {gate_diagnostics['ls_margin_gate_failed']} ({100-ls_pass_rate:.1f}%)")
        
        # Combined gate results
        logger.info(f"\n🎯 Combined Gate Results (ALL gates must pass):")
        all_pass_rate = 100.0 * gate_diagnostics['all_gates_passed'] / total_attempts
        logger.info(f"  All gates passed: {gate_diagnostics['all_gates_passed']} ({all_pass_rate:.1f}%)")
        logger.info(f"  Any gate failed:  {gate_diagnostics['all_gates_failed']} ({100-all_pass_rate:.1f}%)")
        
        # Blocking patterns
        logger.info(f"\n🔒 Gate Blocking Patterns (for failed flips):")
        logger.info(f"  Single gate blocks:")
        logger.info(f"    ├─ Sharpness only:  {gate_diagnostics['blocked_by_sharpness_only']}")
        logger.info(f"    ├─ Consensus only:  {gate_diagnostics['blocked_by_consensus_only']}")
        logger.info(f"    └─ LS margin only:  {gate_diagnostics['blocked_by_ls_margin_only']}")
        logger.info(f"  Two-gate blocks:")
        logger.info(f"    ├─ Sharpness + Consensus: {gate_diagnostics['blocked_by_sharpness_consensus']}")
        logger.info(f"    ├─ Sharpness + LS margin: {gate_diagnostics['blocked_by_sharpness_ls']}")
        logger.info(f"    └─ Consensus + LS margin: {gate_diagnostics['blocked_by_consensus_ls']}")
        logger.info(f"  All three gates failed:  {gate_diagnostics['blocked_by_all_three']}")
        
        # Feature distributions (if we collected samples)
        if gate_diagnostics['base_entropy_all']:
            logger.info(f"\n📊 Feature Distribution Statistics (sampled):")
            
            entropy_vals = gate_diagnostics['base_entropy_all']
            logger.info(f"  Base Entropy:")
            logger.info(f"    μ={np.mean(entropy_vals):.3f}, σ={np.std(entropy_vals):.3f}, "
                       f"median={np.median(entropy_vals):.3f}, "
                       f"q25={np.percentile(entropy_vals, 25):.3f}, "
                       f"q75={np.percentile(entropy_vals, 75):.3f}")
            
            ls_margin_vals = gate_diagnostics['ls_margin_all']
            logger.info(f"  LS Margin (challenger - baseline):")
            logger.info(f"    μ={np.mean(ls_margin_vals):.3f}, σ={np.std(ls_margin_vals):.3f}, "
                       f"median={np.median(ls_margin_vals):.3f}, "
                       f"q25={np.percentile(ls_margin_vals, 25):.3f}, "
                       f"q75={np.percentile(ls_margin_vals, 75):.3f}")
            
            tau_q_vals = gate_diagnostics['tau_q_all']
            logger.info(f"  Query-Adaptive Threshold (τ_q, 75th percentile):")
            logger.info(f"    μ={np.mean(tau_q_vals):.3f}, σ={np.std(tau_q_vals):.3f}, "
                       f"median={np.median(tau_q_vals):.3f}")
            
            if gate_diagnostics['community_vote_all']:
                vote_vals = gate_diagnostics['community_vote_all']
                logger.info(f"  Community Vote (mutual neighbors):")
                logger.info(f"    μ={np.mean(vote_vals):.2f}, σ={np.std(vote_vals):.2f}, "
                           f"median={np.median(vote_vals):.1f}, "
                           f"min={np.min(vote_vals):.0f}, max={np.max(vote_vals):.0f}")
            
            recip_vals = gate_diagnostics['recip_z_all']
            logger.info(f"  Reciprocity z-score:")
            logger.info(f"    μ={np.mean(recip_vals):.3f}, σ={np.std(recip_vals):.3f}, "
                       f"median={np.median(recip_vals):.3f}")
            
            outlier_vals = gate_diagnostics['outlier_z_all']
            logger.info(f"  Outlier z-score (list coherence):")
            logger.info(f"    μ={np.mean(outlier_vals):.3f}, σ={np.std(outlier_vals):.3f}, "
                       f"median={np.median(outlier_vals):.3f}")
    else:
        logger.info(f"  No flip attempts recorded (spectral refinement didn't suggest any flips)")
    
    logger.info(f"{'='*80}\n")
    # =============================================================================
    # End Gate Diagnostics Summary
    # =============================================================================
    
    logger.info(f"Spectral Refinement {direction} retrieval results:")
    logger.info(f"  Total queries: {total_queries}")
    logger.info(f"  R@1: {results['recall@1']:.4f}")
    logger.info(f"  R@5: {results['recall@5']:.4f}")
    logger.info(f"  R@10: {results['recall@10']:.4f}")
    logger.info(f"  Fusion time: {results['fusion_time']:.2f}s")
    logger.info(f"  Flip rate: {results['gate_rate']:.2%}")
    K_budget = results['flips_A'] + results['flips_B'] + results['flips_C'] + results['flips_D']
    F_realized = results['flips_A'] + results['flips_B']
    prec = results.get('flip_precision', results['flips_A'] / max(F_realized, 1))
    frate = results.get('flip_rate_ratio', F_realized / max(K_budget, 1))
    logger.info(f"  Flip stats: A={results['flips_A']} (help), B={results['flips_B']} (hurt), C={results['flips_C']} (neutral), D={results['flips_D']} (wrong→wrong)")
    logger.info(f"  NetFlip/1k(K) = 1000*(A-B)/K = {results['netflip_per_1k']:.1f} [K={K_budget}, F={F_realized}, Prec={prec:.2%}, FlipRate={frate:.2%}]")
    
    # Stage-wise summary statistics
    if spectral_diagnostics:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STAGE-WISE SUMMARY STATISTICS")
        logger.info(f"{'='*80}")
        
        # Feature statistics
        active_feature_names = list(weights_dict.keys())
        if active_feature_names:
            logger.info(f"\n🔍 Feature Statistics (across {total_queries} queries):")
            for feat_name in active_feature_names:
                feat_stats = []
                for diag in spectral_diagnostics:
                    if feat_name in diag.get('features', {}):
                        feat_data = diag['features'][feat_name]
                        feat_stats.append(feat_data.get('mean', 0.0))
                if feat_stats:
                    logger.info(f"  {feat_name.upper()}: mean={np.mean(feat_stats):.3f}, std={np.std(feat_stats):.3f}, "
                              f"min={np.min(feat_stats):.3f}, max={np.max(feat_stats):.3f}")
        
        # Spectral filtering statistics
        spectral_applied = sum(1 for diag in spectral_diagnostics 
                              if diag.get('spectral_filtering', {}).get('filter_applied', False))
        logger.info(f"\n🎯 Spectral Filtering:")
        logger.info(f"  Applied: {spectral_applied}/{total_queries} ({100*spectral_applied/total_queries:.1f}%)")
        if spectral_applied > 0:
            confidences = [diag['spectral_filtering']['confidence'] 
                          for diag in spectral_diagnostics 
                          if diag.get('spectral_filtering', {}).get('filter_applied', False)]
            logger.info(f"  Confidence: mean={np.mean(confidences):.3f}, std={np.std(confidences):.3f}")
        
        # Operator type distribution (polarity-aware)
        operator_types = [diag.get('operator_type', 'unknown') for diag in spectral_diagnostics]
        if operator_types:
            from collections import Counter
            op_counts = Counter(operator_types)
            logger.info(f"\n🔀 Operator Distribution:")
            for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / len(operator_types)
                logger.info(f"  {op_type}: {count}/{len(operator_types)} ({pct:.1f}%)")
            
            # Polarity statistics
            polarities = [diag.get('polarity', {}).get('value', 0.0) 
                         for diag in spectral_diagnostics 
                         if 'polarity' in diag]
            if polarities:
                logger.info(f"\n📊 Polarity Statistics:")
                logger.info(f"  Mean: {np.mean(polarities):.3f}, Std: {np.std(polarities):.3f}")
                logger.info(f"  Min: {np.min(polarities):.3f}, Max: {np.max(polarities):.3f}")
                positive_count = sum(1 for p in polarities if p > 0.1)
                negative_count = sum(1 for p in polarities if p < -0.1)
                neutral_count = len(polarities) - positive_count - negative_count
                logger.info(f"  Positive (>0.1): {positive_count} ({100*positive_count/len(polarities):.1f}%)")
                logger.info(f"  Negative (<-0.1): {negative_count} ({100*negative_count/len(polarities):.1f}%)")
                logger.info(f"  Neutral ([-0.1, 0.1]): {neutral_count} ({100*neutral_count/len(polarities):.1f}%)")
        
        # Score delta statistics
        score_deltas_means = [diag.get('score_deltas', {}).get('mean', 0.0) 
                             for diag in spectral_diagnostics 
                             if 'score_deltas' in diag]
        if score_deltas_means:
            logger.info(f"\n📈 Score Delta Statistics:")
            logger.info(f"  Mean delta: {np.mean(score_deltas_means):.4f}, std={np.std(score_deltas_means):.4f}")
            logger.info(f"  Min={np.min(score_deltas_means):.4f}, Max={np.max(score_deltas_means):.4f}")
        
        # Flip statistics
        flips = [diag.get('flipped', False) for diag in spectral_diagnostics]
        n_flips = sum(flips)
        logger.info(f"\n🔄 Flip Statistics:")
        logger.info(f"  Total flips: {n_flips}/{total_queries} ({100*n_flips/total_queries:.1f}%)")
        if n_flips > 0:
            flip_diags = [diag for diag in spectral_diagnostics if diag.get('flipped', False)]
            baseline_correct_flips = sum(1 for diag in flip_diags if diag.get('baseline_correct', False))
            refined_correct_flips = sum(1 for diag in flip_diags if diag.get('refined_correct', False))
            logger.info(f"  Baseline correct before flip: {baseline_correct_flips}/{n_flips}")
            logger.info(f"  Refined correct after flip: {refined_correct_flips}/{n_flips}")
        
        logger.info(f"{'='*80}\n")
    
    # Return decisions if requested (for NNN+Spectral composition)
    if return_decisions:
        logger.info(f"📋 Returning {len(spectral_decisions)} spectral decisions for composition")
        if len(spectral_decisions) == 0:
            logger.warning(f"⚠️  No spectral decisions collected (use_significance_gate={use_significance_gate})")
            logger.warning(f"⚠️  DEBUG: all_flip_attempts={'exists' if 'all_flip_attempts' in locals() else 'missing'}")
        else:
            logger.info(f"✓ Successfully collected {len(spectral_decisions)} decisions from {len(all_flip_attempts) if 'all_flip_attempts' in locals() else '?'} attempts")
        return results, spectral_decisions
    
    return results


# =============================================================================
# MATCHED FLIP-RATE ABLATION BASELINE
# =============================================================================
