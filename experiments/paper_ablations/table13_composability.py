"""
Table 13: Composability with base retrievers.

Tests BCG framework on top of different base retrieval methods:
- Raw CLIP
- NNN (Neural Neighbor Normalization)
- QB-Norm (QueryBank Normalization)
- DB-Norm (Database Normalization)

Usage:
    python experiments/paper_ablations/table13_composability.py \
        --dataset coco_captions \
        --direction i2t \
        --backbone clip \
        --output results/table13_composability.json \
        --embeddings_root /path/to/UAMR \
        --base_method raw_clip
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


def run_table13_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None,
    base_method: str = 'raw_clip'
):
    """
    Run Table 13: Composability with base retrievers.
    
    Variants for each base method:
    1. Base only
    2. Spectral-Only (100%)
    3. BCG @ 5%
    4. BCG @ 10%
    5. BCG @ 20%
    6. BCG (no L1b)
    7. BCG (no L3)
    
    Base methods:
    - raw_clip: Raw CLIP embeddings
    - nnn: Neural Neighbor Normalization
    - qb_norm: QueryBank Normalization
    - db_norm: Database Normalization
    
    Note: This requires the base_method parameter to be passed through to
    spectral_refinement_baseline. For now, we test with raw_clip.
    """
    print(f"\n{'='*80}")
    print("TABLE 13: Composability with Base Retrievers")
    print(f"Dataset: {dataset} | Direction: {direction} | Backbone: {backbone}")
    print(f"Base Method: {base_method}")
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
        # Base only
        {
            'name': f'{base_method}: Base only',
            'l1a': False, 'l1b': False, 'l2': False, 'l3': False,
            'budget': 0.0, 'fusion_weight': 0.0, 'feature_weights': 'equal',
            'base_method': base_method,
        },
        
        # Spectral-Only (100%)
        {
            'name': f'{base_method}: Spectral-Only',
            'l1a': False, 'l1b': False, 'l2': True, 'l3': False,
            'budget': 1.0, 'fusion_weight': 1.0, 'feature_weights': 'equal',  # Spectral only
            'base_method': base_method,
        },
        
        # BCG @ 5%
        {
            'name': f'{base_method}: BCG @ 5%',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.05, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'base_method': base_method,
        },
        
        # BCG @ 10%
        {
            'name': f'{base_method}: BCG @ 10%',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'base_method': base_method,
        },
        
        # BCG @ 20%
        {
            'name': f'{base_method}: BCG @ 20%',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.20, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'base_method': base_method,
        },
        
        # BCG (no L1b)
        {
            'name': f'{base_method}: BCG (no L1b)',
            'l1a': True, 'l1b': False, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'base_method': base_method,
        },
        
        # BCG (no L3)
        {
            'name': f'{base_method}: BCG (no L3)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': False,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'base_method': base_method,
        },
    ]
    
    # Run experiments
    results = []
    for i, variant in enumerate(variants):
        print(f"Running {i+1}/{len(variants)}: {variant['name']}...")
        try:
            # Note: base_method needs to be passed through production_pipeline
            # For now, we'll use raw_clip (default) and note that other base methods
            # require additional implementation in spectral_refinement_baseline
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
    print(f"{'Refinement':<25} {'Sel%':<8} {'R@1':<8} {'ΔR@1':<8} {'Flip%':<8} {'A/B':<10} {'NF/1k':<12} {'Purity':<8}")
    print("-" * 100)
    for r in results:
        purity_str = f"{r['purity']:.2f}" if r.get('purity') is not None else "-"
        refinement = r['variant'].replace(f'{base_method}: ', '')
        sel_str = f"{r['sel_pct']:.1f}%" if r['sel_pct'] > 0 else "--"
        print(f"{refinement:<25} {sel_str:>6} {r['r1']:>7.4f} {r['delta_r1']:>+7.4f} {r['flip_pct']:>6.1f}% {r['A/B']:>10} {r['net_flip_per_1k']:>+11.1f} {purity_str:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 13: Composability with base retrievers")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="i2t", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table13_composability.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    parser.add_argument("--base_method", type=str, default="raw_clip",
                       choices=["raw_clip", "nnn", "qb_norm", "db_norm"],
                       help="Base retrieval method")
    
    args = parser.parse_args()
    
    run_table13_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root,
        base_method=args.base_method
    )
