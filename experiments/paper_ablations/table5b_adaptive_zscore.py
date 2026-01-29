"""
Table 5b: Adaptive selection via z-score threshold.

Instead of fixing Stage 1a budget at 10%, uses signal-based threshold:
refine queries where z_Polarity + z_RSC > τ.

Usage:
    python experiments/paper_ablations/table5b_adaptive_zscore.py \
        --dataset coco_captions \
        --direction i2t \
        --backbone clip \
        --output results/table5b_adaptive_zscore.json \
        --embeddings_root /path/to/UAMR
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from scipy.stats import rankdata

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from core.production_pipeline import run_production_pipeline
from core.spectral_refinement_standalone import compute_core_polarity_scores, fast_cycle_consistency_for_candidates


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score normalization using median and MAD."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    mad = np.clip(mad, 1e-8, None)  # Avoid division by zero
    return (x - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal dist


def run_table5b_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None
):
    """
    Run Table 5b: Adaptive z-score threshold selection.
    
    Variants:
    1. z > 0.0 (median)
    2. z > 0.5
    3. z > 1.0 (1 std)
    4. z > 1.5
    5. z > 2.0 (2 std)
    6. Fixed 10% (baseline)
    
    For z-score variants, we compute z_polarity + z_rsc and select queries above threshold.
    """
    print(f"\n{'='*80}")
    print("TABLE 5b: Adaptive Selection (Z-Score Threshold)")
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
    
    # Compute Polarity and RSC scores for ALL queries
    print("Computing Polarity and RSC scores for all queries...")
    
    # Compute Polarity scores
    polarity_result = compute_core_polarity_scores(
        queries=queries,
        gallery=gallery,
        query_ids=[str(qid) for qid in query_ids],
        direction=direction,
        topK=50,
        M_rts=24,
        use_rts=True,
        use_curv=True,
        image_bank=image_bank,
        text_bank=text_bank,
        verbose=False,
        aggregation='rank',
    )
    polarity_scores = polarity_result['polarity']  # [N] numpy array
    
    # Compute RSC scores for all queries
    all_query_indices = np.arange(len(queries))
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
    
    # Compute z-scores
    z_polarity = robust_zscore(polarity_scores)
    z_rsc = robust_zscore(rsc_scores)
    z_combined = z_polarity + z_rsc
    
    print(f"Z-score ranges: Polarity=[{z_polarity.min():.2f}, {z_polarity.max():.2f}], "
          f"RSC=[{z_rsc.min():.2f}, {z_rsc.max():.2f}], "
          f"Combined=[{z_combined.min():.2f}, {z_combined.max():.2f}]\n")
    
    # Define z-score thresholds
    z_thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Run experiments
    results = []
    
    # Z-score threshold variants
    for z_thresh in z_thresholds:
        # Select queries where z_combined > threshold
        selected_mask = z_combined > z_thresh
        selected_indices = np.where(selected_mask)[0]
        adaptive_sel_pct = len(selected_indices) / len(queries) * 100
        
        print(f"Running z > {z_thresh}: {len(selected_indices)} queries selected ({adaptive_sel_pct:.1f}%)...")
        
        if len(selected_indices) == 0:
            print(f"  ⚠️  No queries selected, skipping...\n")
            results.append({
                'variant': f'z > {z_thresh}',
                'adaptive_sel_pct': 0.0,
                'error': 'No queries selected',
                'r1': 0.0,
                'delta_r1': 0.0,
                'flip_pct': 0.0,
                'A': 0,
                'B': 0,
                'A/B': '0/0',
                'net_flip_per_1k': 0.0,
                'sel_pct': 0.0,
                'spec_pct': 0.0,
            })
            continue
        
        # For z-score selection, we need to use a custom budget that matches the selected percentage
        # We'll use the production pipeline with the adaptive budget
        try:
            config = {
                'budget_percentages': [adaptive_sel_pct / 100.0],
                'use_polarity_aware': True,  # Stage 1b
                'use_significance_gate': True,  # FDR
                'fusion_weight': 0.5,
                'feature_weights': 'production',
                'zscore_threshold': z_thresh,  # Pass threshold for logging
            }
            
            result = run_production_pipeline(
                queries, gallery, query_ids, gallery_ids, gt_mapping,
                config, direction, image_bank, text_bank
            )
            
            # Add adaptive selection percentage
            result['variant'] = f'z > {z_thresh}'
            result['adaptive_sel_pct'] = adaptive_sel_pct
            
            results.append(result)
            
            # Print progress
            print(f"  ✅ R@1: {result['r1']:.4f} (Δ: {result['delta_r1']:+.4f})")
            print(f"     Adaptive Sel%: {adaptive_sel_pct:.1f}%, Spec%: {result['spec_pct']:.1f}%")
            print(f"     Flips: {result['flip_pct']:.1f}%, A/B: {result['A/B']}, NF/1k: {result['net_flip_per_1k']:+.1f}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({
                'variant': f'z > {z_thresh}',
                'adaptive_sel_pct': adaptive_sel_pct,
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
            })
    
    # Fixed 10% baseline
    print(f"Running Fixed 10% (baseline)...")
    try:
        config = {
            'budget_percentages': [0.10],
            'use_polarity_aware': True,
            'use_significance_gate': True,
            'fusion_weight': 0.5,
            'feature_weights': 'production',
        }
        
        result = run_production_pipeline(
            queries, gallery, query_ids, gallery_ids, gt_mapping,
            config, direction, image_bank, text_bank
        )
        
        result['variant'] = 'Fixed 10% (baseline)'
        result['adaptive_sel_pct'] = 10.0
        
        results.append(result)
        
        print(f"  ✅ R@1: {result['r1']:.4f} (Δ: {result['delta_r1']:+.4f})")
        print(f"     Sel%: {result['sel_pct']:.1f}%, Spec%: {result['spec_pct']:.1f}%")
        print(f"     Flips: {result['flip_pct']:.1f}%, A/B: {result['A/B']}, NF/1k: {result['net_flip_per_1k']:+.1f}\n")
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        results.append({
            'variant': 'Fixed 10% (baseline)',
            'adaptive_sel_pct': 10.0,
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
    print(f"{'Z-Threshold':<15} {'Adaptive Sel%':<15} {'Spec%':<8} {'R@1':<8} {'ΔR@1':<8} {'Flip%':<8} {'A/B':<10} {'NF/1k':<12}")
    print("-" * 100)
    for r in results:
        z_str = r['variant'].replace('z > ', '').replace('Fixed 10% (baseline)', 'Fixed 10%')
        print(f"{z_str:<15} {r.get('adaptive_sel_pct', r.get('sel_pct', 0)):>13.1f}% {r['spec_pct']:>6.1f}% {r['r1']:>7.4f} {r['delta_r1']:>+7.4f} {r['flip_pct']:>6.1f}% {r['A/B']:>10} {r['net_flip_per_1k']:>+11.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 5b: Adaptive z-score selection")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="i2t", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table5b_adaptive_zscore.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    
    args = parser.parse_args()
    
    run_table5b_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root
    )
