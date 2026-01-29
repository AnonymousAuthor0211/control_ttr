"""
Table 14: Embedding Geometry Analysis.

Analyzes geometry metrics (isotropy, hubness, RTS variance) for different encoders
and correlates with BCG performance in T2I regimes.

Usage:
    python experiments/paper_ablations/table14_geometry_analysis.py \
        --dataset coco_captions \
        --direction t2i \
        --backbone clip \
        --output results/table14_geometry_analysis.json \
        --embeddings_root /path/to/UAMR
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from core.production_pipeline import run_variant_production
from core.spectral_refinement_standalone import compute_core_polarity_scores


def compute_isotropy(embeddings: torch.Tensor) -> float:
    """
    Compute isotropy measure: variance of embedding norms.
    
    Higher isotropy = more uniform embedding norms = less hubness.
    
    Args:
        embeddings: Embeddings [N, D]
        
    Returns:
        Isotropy score (higher = more isotropic)
    """
    norms = torch.norm(embeddings, dim=1)
    isotropy = 1.0 / (1.0 + norms.std().item())  # Inverse of std (higher = more uniform)
    return isotropy


def compute_hubness(embeddings: torch.Tensor, k: int = 10) -> float:
    """
    Compute hubness measure: concentration of kNN neighbors.
    
    Higher hubness = more concentrated neighbors = hub problem.
    
    Args:
        embeddings: Embeddings [N, D]
        k: Number of neighbors to consider
        
    Returns:
        Hubness score S_N (higher = more hubness)
    """
    N = embeddings.shape[0]
    embeddings_norm = F.normalize(embeddings, dim=1)
    
    # Compute similarity matrix
    sims = embeddings_norm @ embeddings_norm.T  # [N, N]
    
    # Get top-k neighbors for each embedding
    _, topk_idx = torch.topk(sims, k=min(k+1, N), dim=1)  # +1 to exclude self
    
    # Count how many times each embedding appears in top-k of others
    hub_counts = torch.zeros(N, device=embeddings.device)
    for i in range(N):
        # Exclude self from top-k
        neighbors = topk_idx[i, 1:]  # Skip first (self)
        hub_counts[neighbors] += 1
    
    # Hubness = variance of hub counts (higher = more concentrated)
    hubness = hub_counts.std().item()
    return hubness


def compute_rts_variance(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    direction: str,
    image_bank: Optional[torch.Tensor] = None,
    text_bank: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute variance of RTS scores across queries.
    
    Higher variance = more discriminative RTS signal.
    
    Args:
        queries: Query embeddings [N, D]
        gallery: Gallery embeddings [M, D]
        direction: 'i2t', 't2i', etc.
        image_bank: Image bank for RTS
        text_bank: Text bank for RTS
        
    Returns:
        Variance of RTS scores
    """
    polarity_result = compute_core_polarity_scores(
        queries=queries,
        gallery=gallery,
        query_ids=[str(i) for i in range(len(queries))],
        direction=direction,
        topK=50,
        M_rts=24,
        use_rts=True,
        use_curv=False,  # Only RTS
        image_bank=image_bank,
        text_bank=text_bank,
        verbose=False,
        aggregation='rank',
    )
    
    # Extract RTS component (if available) or use polarity as proxy
    # Note: This is a simplified version. Full implementation would extract RTS directly.
    polarity_scores = polarity_result['polarity']
    rts_variance = np.var(polarity_scores)
    
    return rts_variance


def run_table14_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None
):
    """
    Run Table 14: Embedding geometry analysis.
    
    Computes:
    1. Isotropy (variance of embedding norms)
    2. Hubness (concentration of kNN neighbors)
    3. RTS Variance (discriminative power of RTS signal)
    4. BCG T2I performance (ΔR@1)
    
    Note: This table compares different encoders (CLIP, SigLIP, BLIP).
    For a single backbone, we compute the geometry metrics and BCG performance.
    """
    print(f"\n{'='*80}")
    print("TABLE 14: Embedding Geometry Analysis")
    print(f"Dataset: {dataset} | Direction: {direction} | Backbone: {backbone}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading embeddings...")
    queries, gallery, query_ids, gallery_ids = load_embeddings(
        dataset, direction, backbone, embeddings_root=embeddings_root
    )
    
    # For i2t, deduplicate queries
    if direction in ['i2t', 'a2t']:
        unique_ids, unique_idx = np.unique(query_ids, return_index=True)
        queries = queries[unique_idx]
        query_ids = query_ids[unique_idx]
    
    gt_mapping = build_gt_mapping(query_ids, gallery_ids, direction)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    queries = queries.to(device)
    gallery = gallery.to(device)
    
    # Load RTS distractor banks
    image_bank, text_bank = load_rts_distractor_bank(
        dataset, direction, backbone, embeddings_root=embeddings_root
    )
    if image_bank is not None:
        image_bank = image_bank.to(device)
    if text_bank is not None:
        text_bank = text_bank.to(device)
    
    print(f"Loaded {len(queries)} queries, {len(gallery)} gallery items")
    print(f"Device: {device}\n")
    
    # Compute geometry metrics
    print("Computing geometry metrics...")
    
    # Isotropy (on queries)
    isotropy = compute_isotropy(queries)
    print(f"  Isotropy: {isotropy:.4f}")
    
    # Hubness (on queries)
    hubness = compute_hubness(queries, k=10)
    print(f"  Hubness (S_N): {hubness:.4f}")
    
    # RTS Variance
    print("  Computing RTS variance...")
    rts_variance = compute_rts_variance(queries, gallery, direction, image_bank, text_bank)
    print(f"  RTS Variance: {rts_variance:.4f}\n")
    
    # Compute BCG T2I performance (full system)
    print("Computing BCG performance...")
    variant = {
        'name': 'BCG (Full)',
        'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
        'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production'
    }
    
    try:
        result = run_variant_production(
            queries, gallery, query_ids, gallery_ids, gt_mapping,
            image_bank, text_bank, variant, direction
        )
        bcg_delta_r1 = result['delta_r1']
        bcg_r1 = result['r1']
        print(f"  BCG R@1: {bcg_r1:.4f} (Δ: {bcg_delta_r1:+.4f})\n")
    except Exception as e:
        print(f"  ❌ Error computing BCG: {e}\n")
        bcg_delta_r1 = 0.0
        bcg_r1 = 0.0
    
    # Compile results
    results = {
        'encoder': f"{backbone.upper()}-B/32" if backbone == 'clip' else backbone.upper(),
        'isotropy': float(isotropy),
        'hubness': float(hubness),
        'rts_variance': float(rts_variance),
        'bcg_t2i_delta_r1': float(bcg_delta_r1),
        'bcg_t2i_r1': float(bcg_r1),
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Encoder':<15} {'Isotropy ↑':<12} {'Hubness (S_N)':<15} {'RTS Var.':<12} {'BCG T2I ΔR@1':<15}")
    print("-" * 70)
    bcg_str = f"{bcg_delta_r1:+.4f}" if bcg_delta_r1 >= 0 else f"{bcg_delta_r1:.4f}"
    print(f"{results['encoder']:<15} {isotropy:>10.4f} {hubness:>13.4f} {rts_variance:>10.4f} {bcg_str:>13}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 14: Embedding geometry analysis")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="t2i", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table14_geometry_analysis.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    
    args = parser.parse_args()
    
    run_table14_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root
    )
