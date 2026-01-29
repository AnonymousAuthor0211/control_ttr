"""
Table 10: Statistical significance of query selection strategies.

Results averaged over 5 random query orderings with paired t-test vs base.

Usage:
    python experiments/paper_ablations/table10_statistical_significance.py \
        --dataset coco_captions \
        --direction t2i \
        --backbone clip \
        --output results/table10_statistical_significance.json \
        --embeddings_root /path/to/UAMR \
        --n_runs 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
from scipy.stats import ttest_rel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from core.production_pipeline import run_variant_production
from core.metrics import compute_base_r1


def run_table10_experiment(
    dataset: str,
    direction: str,
    backbone: str,
    output_path: str,
    embeddings_root: Optional[str] = None,
    n_runs: int = 5
):
    """
    Run Table 10: Statistical significance ablation.
    
    Variants:
    1. Base (no refinement)
    2. Always-on (no selection)
    3. V0: Random selection
    4. V1: Polarity only
    5. V2: RSC only
    6. V3: BCG (Polarity+RSC, Ours)
    
    Each variant is run n_runs times with different query orderings.
    Results are averaged and p-values computed via paired t-test vs base.
    """
    print(f"\n{'='*80}")
    print("TABLE 10: Statistical Significance")
    print(f"Dataset: {dataset} | Direction: {direction} | Backbone: {backbone}")
    print(f"Number of runs: {n_runs}")
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
    
    # Compute baseline R@1 (for all runs, it should be the same)
    base_r1 = compute_base_r1(queries, gallery, gt_mapping, query_ids)[0]
    print(f"Baseline R@1: {base_r1:.4f}\n")
    
    # Define variants
    variants = [
        {
            'name': 'Base (no refinement)',
            'l1a': False, 'l1b': False, 'l2': False, 'l3': False,
            'budget': 0.0, 'fusion_weight': 0.0, 'feature_weights': 'equal'
        },
        {
            'name': 'Always-on (no selection)',
            'l1a': False, 'l1b': False, 'l2': True, 'l3': False,
            'budget': 1.0, 'fusion_weight': 0.5, 'feature_weights': 'equal'
        },
        {
            'name': 'V0: Random selection',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'frozen_bcg_mode': 'random',
        },
        {
            'name': 'V1: Polarity only',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'frozen_bcg_mode': 'polarity_only',
        },
        {
            'name': 'V2: RSC only',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'frozen_bcg_mode': 'rsc_only',
        },
        {
            'name': 'V3: BCG (Polarity+RSC, Ours)',
            'l1a': True, 'l1b': True, 'l2': True, 'l3': True,
            'budget': 0.10, 'fusion_weight': 0.5, 'feature_weights': 'production',
            'frozen_bcg_mode': 'rank_sum_oriented',
        },
    ]
    
    # Run experiments with multiple random orderings
    all_results = {}
    
    for variant in variants:
        variant_name = variant['name']
        print(f"\n{'='*60}")
        print(f"Running: {variant_name}")
        print(f"{'='*60}")
        
        run_results = []
        
        for run_idx in range(n_runs):
            print(f"  Run {run_idx+1}/{n_runs}...")
            
            # Shuffle query order for this run (affects selection when using random)
            if run_idx > 0:
                # Create shuffled indices
                shuffled_idx = np.random.permutation(len(queries))
                queries_shuffled = queries[shuffled_idx]
                query_ids_shuffled = query_ids[shuffled_idx]
            else:
                queries_shuffled = queries
                query_ids_shuffled = query_ids
            
            try:
                result = run_variant_production(
                    queries_shuffled, gallery, query_ids_shuffled, gallery_ids, gt_mapping,
                    image_bank, text_bank, variant, direction
                )
                
                # Store delta_r1 and net_flip_per_1k for statistical analysis
                run_results.append({
                    'r1': result['r1'],
                    'delta_r1': result['delta_r1'],
                    'net_flip_per_1k': result['net_flip_per_1k'],
                    'sel_pct': result['sel_pct'],
                })
                
                print(f"    R@1: {result['r1']:.4f} (Δ: {result['delta_r1']:+.4f}), "
                      f"NF/1k: {result['net_flip_per_1k']:+.1f}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                run_results.append({
                    'r1': base_r1,
                    'delta_r1': 0.0,
                    'net_flip_per_1k': 0.0,
                    'sel_pct': 0.0,
                })
        
        # Compute statistics
        delta_r1_values = [r['delta_r1'] for r in run_results]
        net_flip_values = [r['net_flip_per_1k'] for r in run_results]
        sel_pct_values = [r['sel_pct'] for r in run_results]
        
        mean_delta_r1 = np.mean(delta_r1_values)
        std_delta_r1 = np.std(delta_r1_values)
        mean_net_flip = np.mean(net_flip_values)
        std_net_flip = np.std(net_flip_values)
        mean_sel_pct = np.mean(sel_pct_values)
        
        # Paired t-test vs base (delta_r1)
        if variant_name == 'Base (no refinement)':
            p_value = None
        else:
            # Base has delta_r1 = 0 for all runs
            base_delta_r1 = np.zeros(n_runs)
            t_stat, p_value = ttest_rel(delta_r1_values, base_delta_r1)
        
        all_results[variant_name] = {
            'sel_pct': mean_sel_pct,
            'net_flip_per_1k': f"{mean_net_flip:.1f}±{std_net_flip:.1f}",
            'delta_r1': f"{mean_delta_r1:.4f}±{std_delta_r1:.4f}",
            'p_value': p_value,
            'runs': run_results,
        }
        
        p_str = f"{p_value:.2e}" if p_value is not None else "-"
        sig_str = "**" if (p_value is not None and p_value < 0.01) else ("*" if (p_value is not None and p_value < 0.05) else "")
        print(f"  Mean: ΔR@1={mean_delta_r1:+.4f}±{std_delta_r1:.4f}, "
              f"NF/1k={mean_net_flip:+.1f}±{std_net_flip:.1f}, "
              f"p={p_str}{sig_str}")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Selection Strategy':<35} {'Sel%':<8} {'NetFlip/1k':<15} {'ΔR@1':<20} {'p-value':<12}")
    print("-" * 100)
    for variant_name, result in all_results.items():
        p_str = f"{result['p_value']:.2e}" if result['p_value'] is not None else "-"
        sig_str = "**" if (result['p_value'] is not None and result['p_value'] < 0.01) else ("*" if (result['p_value'] is not None and result['p_value'] < 0.05) else "")
        print(f"{variant_name:<35} {result['sel_pct']:>6.1f}% {result['net_flip_per_1k']:>15} {result['delta_r1']:>20} {p_str:>10}{sig_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table 10: Statistical significance")
    parser.add_argument("--dataset", type=str, default="coco_captions")
    parser.add_argument("--direction", type=str, default="t2i", choices=["i2t", "t2i"])
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--output", type=str, default="results/table10_statistical_significance.json")
    parser.add_argument("--embeddings_root", type=str, default=None, 
                       help="Root directory containing embeddings_* folders (e.g., /path/to/UAMR)")
    parser.add_argument("--n_runs", type=int, default=5, 
                       help="Number of random orderings to test")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    run_table10_experiment(
        args.dataset,
        args.direction,
        args.backbone,
        args.output,
        embeddings_root=args.embeddings_root,
        n_runs=args.n_runs
    )
