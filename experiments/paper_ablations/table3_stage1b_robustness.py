"""
Table 3: Stage 1b robustness (TOP-1 feature set).

Tests robustness of Stage 1b fine veto by removing features from probit_vote.

Usage:
    python experiments/paper_ablations/table3_stage1b_robustness.py \
        --dataset coco_captions \
        --direction i2t \
        --backbone clip \
        --output results/table3_stage1b_robustness.json \
        --embeddings_root /path/to/UAMR
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from core.production_pipeline import run_variant_production


def run_table3_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None
):
    """
    Run Table 3: Stage 1b robustness ablation.
    
    Variants:
    1. Full system (Stage 1b uses RTS+CCS+CURV at TOP-1)
    2. Stage 1b w/o CURV (TOP-1: RTS+CCS only)
    3. Stage 1b w/o CCS (TOP-1: RTS+CURV only)
    
    Note: Stage 1b feature removal is controlled via baseline_params in the
    production pipeline. We need to pass this through the variant config.
    """
    print(f"\n{'='*80}")
    print("TABLE 3: Stage 1b Robustness (TOP-1 Feature Set)")
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
    # Note: Stage 1b feature removal is handled internally by spectral_refinement_baseline
    # via estimate_polarity_from_features with use_rts, use_ccs, use_curv flags
    variants = [
        # 1. Full system (RTS+CCS+CURV at TOP-1)
        {
            'name': 'Full system (Stage 1b uses RTS+CCS+CURV at TOP-1)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'equal',
            'stage1b_use_rts': True, 'stage1b_use_ccs': True, 'stage1b_use_curv': True,
        },
        
        # 2. Stage 1b w/o CURV (TOP-1: RTS+CCS only)
        {
            'name': 'Stage 1b w/o CURV (TOP-1: RTS+CCS only)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'equal',
            'stage1b_use_rts': True, 'stage1b_use_ccs': True, 'stage1b_use_curv': False,
        },
        
        # 3. Stage 1b w/o CCS (TOP-1: RTS+CURV only)
        {
            'name': 'Stage 1b w/o CCS (TOP-1: RTS+CURV only)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'equal',
            'stage1b_use_rts': True, 'stage1b_use_ccs': False, 'stage1b_use_curv': True,
        },
    ]
    
    # Run experiments
    results = []
    for i, variant in enumerate(variants):
        print(f"Running {i+1}/{len(variants)}: {variant['name']}...")
        try:
            # Note: stage1b feature flags need to be passed through to baseline_params
            # This requires updating run_variant_production to handle these
            result = run_variant_production(
                queries, gallery, query_ids, gallery_ids, gt_mapping,
                image_bank, text_bank, variant, direction
            )
            results.append(result)
            
            # Print progress
            purity_str = f"{result['purity']:.2f}" if result['purity'] is not None else "-"
            print(f"  ✅ R@1: {result['r1']:.4f} (Δ: {result['delta_r1']:+.4f})")
            print(f"     Sel%: {result['sel_pct']:.1f}%, Spec%: {result['spec_pct']:.1f}%")
            print(f"     Flips: {result['flip_pct']:.1f}%, A/B: {result['A/B']}, NF/1k: {result['net_flip_per_1k']:+.1f}, Purity: {purity_str}\n")
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
                'purity': None,
            })
    
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
    print(f"{'Variant':<50} {'Sel%':<8} {'Spec%':<8} {'R@1':<8} {'ΔR@1':<8} {'Flip%':<8} {'A/B':<10} {'NF/1k':<12} {'Purity':<8}")
    print("-" * 110)
    for r in results:
        purity_str = f"{r['purity']:.2f}" if r.get('purity') is not None else "-"
        print(f"{r['variant']:<50} {r['sel_pct']:>6.1f}% {r['spec_pct']:>6.1f}% {r['r1']:>7.4f} {r['delta_r1']:>+7.4f} {r['flip_pct']:>6.1f}% {r['A/B']:>10} {r['net_flip_per_1k']:>+11.1f} {purity_str:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 3: Stage 1b robustness")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="i2t", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table3_stage1b_robustness.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    
    args = parser.parse_args()
    
    run_table3_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root
    )
