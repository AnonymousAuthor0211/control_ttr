"""
Individual table scripts for paper ablations.

Each table script can be run independently:
    python experiments/paper_ablations/table1_layers.py --help

Or all tables can be run via the orchestrator:
    python experiments/run_all_tables.py --help
"""

# Table function imports for programmatic access
from .table1_layers import run_table1_experiment
from .table2_stage1a_1b import run_table2_experiment
from .table3_stage1b_robustness import run_table3_experiment
from .table4_multiplicity import run_table4_experiment
from .table5_budget_sweep import run_table5_experiment
from .table6_feature_weights import run_table6_experiment
from .table7_topk_ablation import run_table7_experiment
from .table8_bcg_components import run_table8_experiment
from .table9_lambda_sweep import run_table9_experiment
from .table10_statistical_significance import run_table10_experiment
from .table11_computational_cost import run_table11_experiment
from .table12_stage1b_spectral_rate import run_table12_experiment
from .table13_composability import run_table13_experiment
from .table14_geometry_analysis import run_table14_experiment

__all__ = [
    'run_table1_experiment',
    'run_table2_experiment',
    'run_table3_experiment',
    'run_table4_experiment',
    'run_table5_experiment',
    'run_table6_experiment',
    'run_table7_experiment',
    'run_table8_experiment',
    'run_table9_experiment',
    'run_table10_experiment',
    'run_table11_experiment',
    'run_table12_experiment',
    'run_table13_experiment',
    'run_table14_experiment',
]
