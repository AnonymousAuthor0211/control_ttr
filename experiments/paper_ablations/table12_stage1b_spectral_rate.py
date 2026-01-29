"""
Table 12: Stage 1b fine veto reduces spectral rate.

Compares L1a only vs L1a+L1b to show Stage 1b reduces computational cost.

Usage:
    python experiments/paper_ablations/table12_stage1b_spectral_rate.py \
        --dataset coco_captions \
        --direction t2i \
        --backbone clip \
        --output results/table12_stage1b_spectral_rate.json \
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


def run_table12_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None
):
    """
    Run Table 12: Stage 1b spectral rate reduction.
    
    Variants:
    1. L1a only (no Stage 1b) - all selected queries get spectral
    2. L1a+L1b (with veto) - Stage 1b filters out low-confidence queries
    
    Measures the reduction in spectral rate and computational savings.
    """
    print(f"\n{'='*80}")
    print("TABLE 12: Stage 1b Fine Veto Reduces Spectral Rate")
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
    
    # Define variants
    variants = [
        # L1a only (no Stage 1b)
        {
            'name': 'L1a only (no Stage 1b)',
            'l1a': True, 'l1b': False, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production'
        },
        
        # L1a+L1b (with veto)
        {
            'name': 'L1a+L1b (with veto)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production'
        },
    ]
    
    # Run experiments with timing
    results = []
    for i, variant in enumerate(variants):
        print(f"Running {i+1}/{len(variants)}: {variant['name']}...")
        start_time = time.time()
        
        try:
            result = run_variant_production(
                queries, gallery, query_ids, gallery_ids, gt_mapping,
                image_bank, text_bank, variant, direction
            )
            
            elapsed_time = time.time() - start_time
            result['time_seconds'] = elapsed_time
            
            results.append(result)
            
            # Print progress
            print(f"  ✅ R@1: {result['r1']:.4f} (Δ: {result['delta_r1']:+.4f})")
            print(f"     Stage 1a Sel%: {result['sel_pct']:.1f}%, Spectral Rate%: {result['spec_pct']:.1f}%")
            print(f"     NetFlip/1k: {result['net_flip_per_1k']:+.1f}, Time: {elapsed_time:.2f}s\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({
                'variant': variant['name'],
                'error': str(e),
                'r1': 0.0,
                'delta_r1': 0.0,
                'flip_pct': 0.0,
                'A': 0,
                'B': 0,
                'A/B': '0/0',
                'net_flip_per_1k': 0.0,
                'sel_pct': 0.0,
                'spec_pct': 0.0,
                'time_seconds': 0.0,
            })
    
    # Compute reduction
    if len(results) == 2 and 'error' not in results[0] and 'error' not in results[1]:
        l1a_only = results[0]
        l1a_l1b = results[1]
        
        sel_reduction = 0.0  # Stage 1a selection is the same
        spec_reduction = l1a_only['spec_pct'] - l1a_l1b['spec_pct']
        spec_reduction_pct = (spec_reduction / l1a_only['spec_pct'] * 100) if l1a_only['spec_pct'] > 0 else 0.0
        time_reduction = l1a_only['time_seconds'] - l1a_l1b['time_seconds']
        time_reduction_pct = (time_reduction / l1a_only['time_seconds'] * 100) if l1a_only['time_seconds'] > 0 else 0.0
        
        reduction_result = {
            'variant': 'Reduction (Stage 1b savings)',
            'sel_pct': sel_reduction,
            'spec_pct': spec_reduction,
            'spec_reduction_pct': spec_reduction_pct,
            'time_seconds': time_reduction,
            'time_reduction_pct': time_reduction_pct,
        }
        results.append(reduction_result)
    
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
    print(f"{'Configuration':<40} {'Stage 1a Sel%':<15} {'Spectral Rate%':<15} {'R@1':<8} {'ΔR@1':<8} {'NF/1k':<12} {'Time (s)':<10}")
    print("-" * 110)
    for r in results:
        if 'reduction' in r.get('variant', '').lower():
            print(f"{r['variant']:<40} {r['sel_pct']:>13.1f}% {r['spec_pct']:>13.1f}% ({r.get('spec_reduction_pct', 0):.1f}% reduction) {'-':>8} {'-':>8} {'-':>12} {r['time_seconds']:>9.2f}s ({r.get('time_reduction_pct', 0):.1f}% reduction)")
        else:
            print(f"{r['variant']:<40} {r['sel_pct']:>13.1f}% {r['spec_pct']:>13.1f}% {r['r1']:>7.4f} {r['delta_r1']:>+7.4f} {r['net_flip_per_1k']:>+11.1f} {r.get('time_seconds', 0):>9.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 12: Stage 1b spectral rate")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="t2i", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table12_stage1b_spectral_rate.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    
    args = parser.parse_args()
    
    run_table12_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root
    )
