"""
Table 11: Computational cost breakdown with two-stage control.

This is primarily an analysis table showing FLOPs and wall-clock time.
The timing information is already captured in the production pipeline.

Usage:
    python experiments/paper_ablations/table11_computational_cost.py \
        --dataset coco_captions \
        --direction i2t \
        --backbone clip \
        --output results/table11_computational_cost.json \
        --embeddings_root /path/to/UAMR
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from core.production_pipeline import run_variant_production
from core.metrics import compute_base_r1


def run_table11_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None
):
    """
    Run Table 11: Computational cost breakdown.
    
    This table measures:
    - Base retrieval time
    - Stage 1a cost (Polarity + RSC computation)
    - Stage 1b cost (per-query TOP-1 polarity)
    - Layer 2+3 cost (feature scoring + spectral smoothing + FDR)
    - Total system cost
    
    Note: FLOPs are estimated based on operations. Actual FLOPs would require
    detailed profiling. This script focuses on wall-clock time measurements.
    """
    print(f"\n{'='*80}")
    print("TABLE 11: Computational Cost Breakdown")
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
    
    N = len(queries)
    print(f"Loaded {N} queries, {len(gallery)} gallery items")
    print(f"Device: {device}\n")
    
    # Measure base retrieval time
    print("Measuring base retrieval time...")
    queries_norm = torch.nn.functional.normalize(queries, dim=1)
    gallery_norm = torch.nn.functional.normalize(gallery, dim=1)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    base_scores = queries_norm @ gallery_norm.T
    torch.cuda.synchronize() if device.type == 'cuda' else None
    base_time = time.time() - start_time
    
    base_r1 = compute_base_r1(queries, gallery, gt_mapping, query_ids)[0]
    print(f"  Base retrieval time: {base_time:.4f}s, R@1: {base_r1:.4f}\n")
    
    # Define variants for cost breakdown
    variants = [
        {
            'name': 'Full system (Stage 1a+1b+L2+L3)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production'
        },
        {
            'name': 'Without Stage 1b (L1a+L2+L3)',
            'l1a': True, 'l1b': False, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production'
        },
        {
            'name': 'Always-on spectral (no control)',
            'l1a': False, 'l1b': False, 'l2': True, 'l3': False,
            'budget': 1.0, 'fusion_weight': 1.0, 'feature_weights': 'equal'  # Spectral only
        },
    ]
    
    # Run experiments with timing
    results = {
        'base_retrieval': {
            'operation': 'Base retrieval (q ⊗ G)',
            'time_seconds': base_time,
            'overhead': 1.0,
        }
    }
    
    for variant in variants:
        print(f"Running: {variant['name']}...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        try:
            result = run_variant_production(
                queries, gallery, query_ids, gallery_ids, gt_mapping,
                image_bank, text_bank, variant, direction
            )
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed_time = time.time() - start_time
            
            results[variant['name']] = {
                'operation': variant['name'],
                'time_seconds': elapsed_time,
                'overhead': elapsed_time / base_time,
                'sel_pct': result['sel_pct'],
                'spec_pct': result['spec_pct'],
                'r1': result['r1'],
                'delta_r1': result['delta_r1'],
            }
            
            print(f"  ✅ Time: {elapsed_time:.4f}s ({elapsed_time/base_time:.1f}× overhead)")
            print(f"     Sel%: {result['sel_pct']:.1f}%, Spec%: {result['spec_pct']:.1f}%")
            print(f"     R@1: {result['r1']:.4f} (Δ: {result['delta_r1']:+.4f})\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
    
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
    print(f"{'Operation':<40} {'Time (s)':<12} {'Overhead':<12} {'Sel%':<8} {'Spec%':<8} {'R@1':<8} {'ΔR@1':<8}")
    print("-" * 100)
    for key, result in results.items():
        sel_str = f"{result.get('sel_pct', 0):.1f}%" if 'sel_pct' in result else "--"
        spec_str = f"{result.get('spec_pct', 0):.1f}%" if 'spec_pct' in result else "--"
        r1_str = f"{result.get('r1', 0):.4f}" if 'r1' in result else "--"
        delta_str = f"{result.get('delta_r1', 0):+.4f}" if 'delta_r1' in result else "--"
        print(f"{result['operation']:<40} {result['time_seconds']:>10.4f} {result['overhead']:>10.1f}× {sel_str:>6} {spec_str:>6} {r1_str:>7} {delta_str:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 11: Computational cost")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="i2t", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table11_computational_cost.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    
    args = parser.parse_args()
    
    run_table11_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root
    )
