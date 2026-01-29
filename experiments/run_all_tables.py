#!/usr/bin/env python3
"""
Run All Paper Tables for Control-TTR.

This script orchestrates running all paper ablation experiments.

Usage:
    # Run all tables for a single configuration
    python experiments/run_all_tables.py \
        --dataset coco_captions \
        --direction i2t \
        --backbone clip \
        --embeddings_root /path/to/UAMR \
        --output_dir results/

    # Run specific tables only
    python experiments/run_all_tables.py \
        --dataset coco_captions \
        --direction t2i \
        --backbone clip \
        --tables 1 5 6 \
        --embeddings_root /path/to/UAMR

    # Run all tables for multiple configurations
    python experiments/run_all_tables.py \
        --dataset coco_captions flickr30k \
        --direction i2t t2i \
        --backbone clip siglip blip \
        --embeddings_root /path/to/UAMR
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import table runners
from experiments.paper_ablations.table1_layers import run_table1_experiment
from experiments.paper_ablations.table2_stage1a_1b import run_table2_experiment
from experiments.paper_ablations.table3_stage1b_robustness import run_table3_experiment
from experiments.paper_ablations.table4_multiplicity import run_table4_experiment
from experiments.paper_ablations.table5_budget_sweep import run_table5_experiment
from experiments.paper_ablations.table6_feature_weights import run_table6_experiment
from experiments.paper_ablations.table7_topk_ablation import run_table7_experiment
from experiments.paper_ablations.table8_bcg_components import run_table8_experiment
from experiments.paper_ablations.table9_lambda_sweep import run_table9_experiment
from experiments.paper_ablations.table10_statistical_significance import run_table10_experiment
from experiments.paper_ablations.table11_computational_cost import run_table11_experiment
from experiments.paper_ablations.table12_stage1b_spectral_rate import run_table12_experiment
from experiments.paper_ablations.table13_composability import run_table13_experiment
from experiments.paper_ablations.table14_geometry_analysis import run_table14_experiment


# Table registry
TABLE_REGISTRY = {
    1: ('Layer-wise Ablation', run_table1_experiment),
    2: ('Stage 1a+1b Control', run_table2_experiment),
    3: ('Stage 1b Robustness', run_table3_experiment),
    4: ('Multiplicity Intervention', run_table4_experiment),
    5: ('Budget Sweep', run_table5_experiment),
    6: ('Feature Weights', run_table6_experiment),
    7: ('TopK Ablation', run_table7_experiment),
    8: ('BCG Components', run_table8_experiment),
    9: ('Lambda Sweep', run_table9_experiment),
    10: ('Statistical Significance', run_table10_experiment),
    11: ('Computational Cost', run_table11_experiment),
    12: ('Stage 1b Spectral Rate', run_table12_experiment),
    13: ('Composability', run_table13_experiment),
    14: ('Geometry Analysis', run_table14_experiment),
}


def run_single_table(
    table_num: int,
    dataset: str,
    direction: str,
    backbone: str,
    output_dir: Path,
    embeddings_root: Optional[str] = None,
) -> dict:
    """Run a single table experiment."""
    if table_num not in TABLE_REGISTRY:
        raise ValueError(f"Unknown table: {table_num}. Available: {list(TABLE_REGISTRY.keys())}")
    
    table_name, run_func = TABLE_REGISTRY[table_num]
    output_file = output_dir / f"table{table_num}_{dataset}_{direction}_{backbone}.json"
    
    print(f"\n{'='*80}")
    print(f"TABLE {table_num}: {table_name}")
    print(f"Config: {dataset} | {direction} | {backbone}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    try:
        run_func(
            dataset=dataset,
            direction=direction,
            backbone=backbone,
            output_path=str(output_file),
            embeddings_root=embeddings_root,
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            'table': table_num,
            'name': table_name,
            'dataset': dataset,
            'direction': direction,
            'backbone': backbone,
            'output': str(output_file),
            'status': 'success',
            'elapsed_seconds': elapsed,
        }
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n❌ TABLE {table_num} FAILED: {error_msg}")
        traceback.print_exc()
        
        return {
            'table': table_num,
            'name': table_name,
            'dataset': dataset,
            'direction': direction,
            'backbone': backbone,
            'status': 'failed',
            'error': error_msg,
            'elapsed_seconds': elapsed,
        }


def run_all_tables(
    datasets: List[str],
    directions: List[str],
    backbones: List[str],
    tables: Optional[List[int]] = None,
    output_dir: str = 'results',
    embeddings_root: Optional[str] = None,
) -> dict:
    """Run all specified table experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if tables is None:
        tables = list(TABLE_REGISTRY.keys())
    
    all_results = []
    total_start = datetime.now()
    
    print(f"\n{'#'*80}")
    print(f"CONTROL-TTR: Running Paper Ablations")
    print(f"{'#'*80}")
    print(f"Datasets:   {datasets}")
    print(f"Directions: {directions}")
    print(f"Backbones:  {backbones}")
    print(f"Tables:     {tables}")
    print(f"Output dir: {output_path}")
    print(f"Embeddings: {embeddings_root or 'auto-detect'}")
    print(f"{'#'*80}\n")
    
    for dataset in datasets:
        for direction in directions:
            for backbone in backbones:
                for table_num in tables:
                    result = run_single_table(
                        table_num=table_num,
                        dataset=dataset,
                        direction=direction,
                        backbone=backbone,
                        output_dir=output_path,
                        embeddings_root=embeddings_root,
                    )
                    all_results.append(result)
    
    total_elapsed = (datetime.now() - total_start).total_seconds()
    
    # Summary
    n_success = sum(1 for r in all_results if r['status'] == 'success')
    n_failed = sum(1 for r in all_results if r['status'] == 'failed')
    
    print(f"\n{'#'*80}")
    print(f"SUMMARY")
    print(f"{'#'*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful:        {n_success}")
    print(f"Failed:            {n_failed}")
    print(f"Total time:        {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'#'*80}\n")
    
    if n_failed > 0:
        print("FAILED EXPERIMENTS:")
        for r in all_results:
            if r['status'] == 'failed':
                print(f"  - Table {r['table']}: {r['dataset']}/{r['direction']}/{r['backbone']}")
                print(f"    Error: {r.get('error', 'Unknown')}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(all_results),
        'successful': n_success,
        'failed': n_failed,
        'total_seconds': total_elapsed,
        'results': all_results,
    }
    
    summary_file = output_path / 'run_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run all paper tables for Control-TTR")
    
    parser.add_argument("--datasets", nargs='+', default=['coco_captions'],
                       choices=['coco_captions', 'flickr30k'],
                       help="Datasets to evaluate")
    parser.add_argument("--directions", nargs='+', default=['i2t'],
                       choices=['i2t', 't2i'],
                       help="Retrieval directions")
    parser.add_argument("--backbones", nargs='+', default=['clip'],
                       choices=['clip', 'siglip', 'blip'],
                       help="Backbone models")
    parser.add_argument("--tables", nargs='+', type=int, default=None,
                       help="Table numbers to run (default: all 1-14)")
    parser.add_argument("--output_dir", type=str, default='results',
                       help="Output directory for results")
    parser.add_argument("--embeddings_root", type=str, default=None,
                       help="Root directory containing embeddings_* folders")
    
    args = parser.parse_args()
    
    run_all_tables(
        datasets=args.datasets,
        directions=args.directions,
        backbones=args.backbones,
        tables=args.tables,
        output_dir=args.output_dir,
        embeddings_root=args.embeddings_root,
    )


if __name__ == "__main__":
    main()
