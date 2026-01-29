"""
FDR (False Discovery Rate) validation for flip decisions.

This module implements Layer 3: FDR flip validation to ensure
flips are statistically significant.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm


def compute_one_sided_pvalue(z_eff: float) -> float:
    """
    Compute one-sided p-value for positive z_eff.
    
    Tests H1: challenger > baseline.
    
    Args:
        z_eff: Effective z-score (challenger - baseline) / sigma
        
    Returns:
        p-value (one-sided)
    """
    if z_eff <= 0:
        return 1.0  # No evidence for flip
    else:
        return 1 - norm.cdf(z_eff)


def storey_bh_robust(
    p_values: List[float],
    alpha: float = 0.15
) -> Tuple[np.ndarray, float, Dict]:
    """
    Storey-BH with robust π₀ estimation.
    
    Args:
        p_values: List of p-values
        alpha: FDR threshold (default 0.15)
        
    Returns:
        (accept_mask, pi_0_hat, diagnostics):
            - accept_mask: Boolean array indicating which hypotheses to accept
            - pi_0_hat: Estimated proportion of true nulls
            - diagnostics: Dict with diagnostic information
    """
    p = np.asarray(p_values)
    m = len(p)
    
    if m == 0:
        return np.array([], dtype=bool), 1.0, {}
    
    # Grid search for π₀
    lambda_grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    pi_0_estimates = []
    
    for lam in lambda_grid:
        num_above = np.sum(p > lam)
        pi_0_lam = num_above / ((1 - lam) * m) if (1 - lam) * m > 0 else 1.0
        pi_0_estimates.append(min(pi_0_lam, 1.0))
    
    # Robust smoothing: median of estimates
    pi_0_hat = np.median(pi_0_estimates)
    pi_0_hat = min(pi_0_hat, 0.99)  # Cap at 0.99
    
    # Adjust α
    alpha_eff = alpha / max(pi_0_hat, 1e-8)
    
    # BH step-up procedure
    order = np.argsort(p)
    p_sorted = p[order]
    
    # Find largest k with p_(k) <= (k/m)*α_eff
    thresh_vec = (np.arange(1, m + 1) / m) * alpha_eff
    k_candidates = np.where(p_sorted <= thresh_vec)[0]
    
    if len(k_candidates) > 0:
        k_accept = k_candidates.max() + 1
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
        'min_p': float(np.min(p)),
        'median_p': float(np.median(p)),
    }
    
    return accept_mask, pi_0_hat, diagnostics


def validate_flip_fdr(
    base_top1_idx: int,
    refined_top1_idx: int,
    base_scores: np.ndarray,
    refined_scores: np.ndarray,
    null_candidates: List[int],
    alpha: float = 0.15
) -> Tuple[bool, float, Dict]:
    """
    Validate a flip using FDR testing.
    
    Tests H0: flip is random (challenger not significantly better than baseline).
    
    Args:
        base_top1_idx: Index of base top-1 candidate
        refined_top1_idx: Index of refined top-1 candidate (challenger)
        base_scores: Base scores [M]
        refined_scores: Refined scores [M]
        null_candidates: List of candidate indices for null distribution
        alpha: FDR threshold (default 0.15)
        
    Returns:
        (is_valid, p_value, diagnostics):
            - is_valid: True if flip is statistically significant
            - p_value: p-value for the flip
            - diagnostics: Dict with diagnostic information
    """
    if base_top1_idx == refined_top1_idx:
        # No flip, accept by default
        return True, 0.0, {'no_flip': True}
    
    # Get scores
    baseline_score = base_scores[base_top1_idx]
    challenger_score = refined_scores[refined_top1_idx]
    
    # Build null distribution from null candidates
    if len(null_candidates) < 3:
        # Not enough null samples, accept flip
        return True, 0.0, {'insufficient_null': True}
    
    null_scores = base_scores[null_candidates]
    
    # Compute robust sigma from null
    null_median = np.median(null_scores)
    mad = np.median(np.abs(null_scores - null_median))
    sigma_null = 1.4826 * mad if mad > 1e-6 else np.std(null_scores) + 1e-6
    
    # Compute z-score
    delta = challenger_score - baseline_score
    z_eff = delta / sigma_null if sigma_null > 1e-6 else 0.0
    
    # Compute p-value
    p_value = compute_one_sided_pvalue(z_eff)
    
    # Accept if p < alpha
    is_valid = p_value < alpha
    
    diagnostics = {
        'z_eff': float(z_eff),
        'p_value': float(p_value),
        'sigma_null': float(sigma_null),
        'delta': float(delta),
        'baseline_score': float(baseline_score),
        'challenger_score': float(challenger_score),
    }
    
    return is_valid, p_value, diagnostics
