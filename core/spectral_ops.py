"""
Spectral operations for Control-TTR.

This module implements spectral smoothing using Laplacian-based filtering.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def build_mutual_knn_graph(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build symmetric mutual k-NN graph from embeddings.
    
    Args:
        embeddings: Candidate embeddings [K, D]
        k: Number of neighbors
        
    Returns:
        Adjacency matrix [K, K]
    """
    n = embeddings.size(0)
    device = embeddings.device
    dtype = embeddings.dtype
    
    # Handle edge case
    k = min(k, n - 1)
    if k <= 0:
        return torch.zeros((n, n), device=device, dtype=dtype)
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    A_base = embeddings_norm @ embeddings_norm.T  # [n, n]
    A_base = torch.clamp(A_base, -1.0, 1.0)
    
    # Build k-NN sets using topk
    _, topk_indices = torch.topk(A_base, k=k+1, dim=1)  # k+1 to include self
    topk_indices = topk_indices[:, 1:]  # Remove self [n, k]
    
    # Build forward adjacency
    row_idx = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten()
    col_idx = topk_indices.flatten()
    
    forward = torch.zeros((n, n), device=device, dtype=dtype)
    forward[row_idx, col_idx] = 1.0
    
    # Mutual k-NN: A[i,j] = 1 iff j ∈ kNN(i) AND i ∈ kNN(j)
    mutual = forward * forward.T
    
    # Weight by original similarity
    A = mutual * A_base
    
    return A


def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    
    Args:
        A: Adjacency matrix [K, K]
        
    Returns:
        Normalized Laplacian [K, K]
    """
    D_diag = A.sum(dim=1)
    D_diag = torch.clamp(D_diag, min=1e-8)
    D_sqrt_inv = 1.0 / torch.sqrt(D_diag)
    D_sqrt_inv_diag = torch.diag(D_sqrt_inv)
    I = torch.eye(len(A), device=A.device, dtype=A.dtype)
    L = I - D_sqrt_inv_diag @ A @ D_sqrt_inv_diag
    return L


def laplacian_smooth(
    scores: torch.Tensor,
    L: torch.Tensor,
    lambda_smooth: float = 0.3,
    n_iters: int = 3
) -> torch.Tensor:
    """
    Low-pass filter via iterative Laplacian smoothing.
    
    Approximates: s_smooth = (I - γL)^n @ s
    
    Args:
        scores: Score vector [K]
        L: Normalized Laplacian [K, K]
        lambda_smooth: Smoothing strength (0 < γ < 1, default 0.3)
        n_iters: Number of iterations (default 3)
        
    Returns:
        Smoothed scores [K]
    """
    n = scores.shape[0]
    if n == 0:
        return scores
    
    # Normalize scores for stability
    s_mu = scores.mean()
    s_sigma = scores.std() + 1e-8
    s_norm = (scores - s_mu) / s_sigma
    
    # Build smoothing operator: S = I - γL
    I = torch.eye(n, device=L.device, dtype=L.dtype)
    S = I - lambda_smooth * L
    
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


def spectral_refinement(
    query: torch.Tensor,
    gallery: torch.Tensor,
    base_scores: torch.Tensor,
    topK: int = 50,
    lambda_smooth: float = 0.3,
    k_nn: int = 12
) -> torch.Tensor:
    """
    Apply spectral refinement to base scores.
    
    Args:
        query: Query embedding [D]
        gallery: Gallery embeddings [M, D]
        base_scores: Base retriever scores [M]
        topK: Number of top candidates to consider
        lambda_smooth: Smoothing strength
        k_nn: k for k-NN graph construction
        
    Returns:
        Refined scores [M] (same shape as base_scores)
    """
    # Get top-K candidates
    K_actual = min(topK, len(gallery))
    topk_vals, topk_idx = torch.topk(base_scores, k=K_actual, dim=0)
    
    if K_actual < 3:
        # Not enough candidates for graph, return original
        return base_scores
    
    # Get candidate embeddings
    candidate_emb = gallery[topk_idx]  # [K, D]
    
    # Build mutual k-NN graph
    A = build_mutual_knn_graph(candidate_emb, k=k_nn)
    
    # Compute normalized Laplacian
    L = normalized_laplacian(A)
    
    # Apply Laplacian smoothing to top-K scores
    topk_scores_smooth = laplacian_smooth(topk_vals, L, lambda_smooth)
    
    # Create refined scores (copy base, update top-K)
    refined_scores = base_scores.clone()
    refined_scores[topk_idx] = topk_scores_smooth
    
    return refined_scores
