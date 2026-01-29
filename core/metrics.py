"""
Evaluation metrics for Control-TTR.

This module provides functions for computing retrieval metrics,
flip statistics, and other evaluation measures.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def compute_base_r1(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    gt_mapping: Dict,
    query_ids: np.ndarray,
) -> Tuple[float, int]:
    """
    Compute base R@1 (recall at 1) for the base retriever.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        gt_mapping: Ground truth mapping {query_id: [gallery_indices]}
        query_ids: Query identifiers [N]
        
    Returns:
        (r1, n_queries): R@1 score and number of queries with ground truth
    """
    queries_norm = F.normalize(queries, dim=1)
    gallery_norm = F.normalize(gallery, dim=1)
    
    # Compute similarities
    sims = queries_norm @ gallery_norm.T  # [N, M]
    
    # Get top-1 predictions
    _, top1_idx = torch.topk(sims, k=1, dim=1)  # [N, 1]
    top1_idx = top1_idx.squeeze(1).cpu().numpy()  # [N]
    
    # Check correctness
    correct = 0
    total = 0
    
    query_ids_list = [str(qid) for qid in query_ids]
    
    for i, qid in enumerate(query_ids_list):
        if qid in gt_mapping:
            total += 1
            if top1_idx[i] in gt_mapping[qid]:
                correct += 1
    
    r1 = correct / total if total > 0 else 0.0
    return r1, total


def compute_flip_outcomes(
    base_scores: np.ndarray,
    refined_scores: np.ndarray,
    gt_mapping: Dict,
    query_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute flip outcomes: A flips (wrong→correct) and B flips (correct→wrong).
    
    Args:
        base_scores: Base retriever scores [N, M]
        refined_scores: Refined scores [N, M]
        gt_mapping: Ground truth mapping {query_id: [gallery_indices]}
        query_ids: Query identifiers [N]
        
    Returns:
        (is_A_flip, is_B_flip, is_flip):
            - is_A_flip: Boolean array [N], True if wrong→correct flip
            - is_B_flip: Boolean array [N], True if correct→wrong flip
            - is_flip: Boolean array [N], True if any flip occurred
    """
    N = base_scores.shape[0]
    is_A_flip = np.zeros(N, dtype=bool)
    is_B_flip = np.zeros(N, dtype=bool)
    is_flip = np.zeros(N, dtype=bool)
    
    query_ids_list = [str(qid) for qid in query_ids]
    
    for i, qid in enumerate(query_ids_list):
        if qid not in gt_mapping:
            continue
        
        # Get top-1 for base and refined
        base_top1 = np.argmax(base_scores[i])
        refined_top1 = np.argmax(refined_scores[i])
        
        # Check if flip occurred
        if base_top1 != refined_top1:
            is_flip[i] = True
            
            # Check base correctness
            base_correct = base_top1 in gt_mapping[qid]
            refined_correct = refined_top1 in gt_mapping[qid]
            
            # A flip: wrong → correct
            if not base_correct and refined_correct:
                is_A_flip[i] = True
            
            # B flip: correct → wrong
            elif base_correct and not refined_correct:
                is_B_flip[i] = True
    
    return is_A_flip, is_B_flip, is_flip


def compute_complementarity_table(
    polarity: np.ndarray,
    rsc: np.ndarray,
    is_A_flip: np.ndarray,
    is_B_flip: np.ndarray,
    is_flip: np.ndarray
) -> Dict:
    """
    Compute 2×2 complementarity table and metrics.
    
    For flipped queries only:
    - Polarity predicts "should flip" (high polarity = susceptible)
    - RSC predicts "flip is reliable" (high RSC = consistent)
    
    Ground truth: A-flip = good flip, B-flip = bad flip
    
    This matches the implementation in scripts/compute_complementarity_evidence.py
    
    Args:
        polarity: Polarity scores [N]
        rsc: RSC scores [N]
        is_A_flip: Boolean array [N], True for A flips (wrong→correct)
        is_B_flip: Boolean array [N], True for B flips (correct→wrong)
        is_flip: Boolean array [N], True for any flip
        
    Returns:
        Dictionary with complementarity statistics including AUROCs
    """
    # Only consider queries that actually flipped
    flip_mask = is_flip
    n_flips = flip_mask.sum()
    
    if n_flips < 10:
        return {
            'n_flips': n_flips,
            'error': 'Too few flips for analysis'
        }
    
    # Ground truth for flips: A=good, B=bad
    is_good_flip = is_A_flip[flip_mask]
    
    # Predictions based on median threshold
    polarity_flip = polarity[flip_mask]
    rsc_flip = rsc[flip_mask]
    
    # For A vs B classification:
    # High polarity + high RSC should predict A-flip
    # We'll use median as threshold for "high"
    pi_thresh = np.median(polarity_flip)
    eta_thresh = np.median(rsc_flip)
    
    pi_high = polarity_flip > pi_thresh
    eta_high = rsc_flip > eta_thresh
    
    # Predictions: high polarity AND high RSC → predict A-flip
    pi_pred_A = pi_high  # Simplified: high polarity = predict A
    eta_pred_A = eta_high  # Simplified: high RSC = predict A
    
    # Error sets
    E_pi = (pi_pred_A != is_good_flip)  # Polarity wrong
    E_eta = (eta_pred_A != is_good_flip)  # RSC wrong
    
    # Complementarity metrics
    eta_rescues_pi = E_pi & ~E_eta  # pi wrong, eta right
    pi_rescues_eta = E_eta & ~E_pi  # eta wrong, pi right
    both_correct = ~E_pi & ~E_eta
    both_wrong = E_pi & E_eta
    
    # Conditional probabilities
    if E_pi.sum() > 0:
        p_eta_correct_given_E_pi = (~E_eta[E_pi]).mean()
    else:
        p_eta_correct_given_E_pi = np.nan
    
    if E_eta.sum() > 0:
        p_pi_correct_given_E_eta = (~E_pi[E_eta]).mean()
    else:
        p_pi_correct_given_E_eta = np.nan
    
    # AUROCs
    if is_good_flip.sum() > 0 and (~is_good_flip).sum() > 0:
        auroc_pi = roc_auc_score(is_good_flip.astype(int), polarity_flip)
        auroc_eta = roc_auc_score(is_good_flip.astype(int), rsc_flip)
        # Combined score
        combined = (rankdata(polarity_flip) + rankdata(rsc_flip)) / 2
        auroc_combined = roc_auc_score(is_good_flip.astype(int), combined)
    else:
        auroc_pi = auroc_eta = auroc_combined = np.nan
    
    return {
        'n_flips': int(n_flips),
        'n_A_flips': int(is_good_flip.sum()),
        'n_B_flips': int((~is_good_flip).sum()),
        'purity': float(is_good_flip.mean()),
        
        # 2×2 Table
        'both_correct': int(both_correct.sum()),
        'eta_rescues_pi': int(eta_rescues_pi.sum()),
        'pi_rescues_eta': int(pi_rescues_eta.sum()),
        'both_wrong': int(both_wrong.sum()),
        
        # Percentages
        'pct_both_correct': float(100 * both_correct.mean()),
        'pct_eta_rescues_pi': float(100 * eta_rescues_pi.mean()),
        'pct_pi_rescues_eta': float(100 * pi_rescues_eta.mean()),
        'pct_both_wrong': float(100 * both_wrong.mean()),
        
        # Complementarity metrics
        'P_eta_correct_given_E_pi': float(p_eta_correct_given_E_pi) if not np.isnan(p_eta_correct_given_E_pi) else None,
        'P_pi_correct_given_E_eta': float(p_pi_correct_given_E_eta) if not np.isnan(p_pi_correct_given_E_eta) else None,
        
        # AUROCs
        'AUROC_polarity': float(auroc_pi) if not np.isnan(auroc_pi) else None,
        'AUROC_rsc': float(auroc_eta) if not np.isnan(auroc_eta) else None,
        'AUROC_combined': float(auroc_combined) if not np.isnan(auroc_combined) else None,
        
        # Error counts
        'n_E_pi': int(E_pi.sum()),
        'n_E_eta': int(E_eta.sum()),
    }
