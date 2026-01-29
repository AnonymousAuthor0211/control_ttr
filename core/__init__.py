"""
Core algorithms for Control-TTR.

This package contains self-contained implementations of:
- Data loading utilities
- Feature computation (RTS, CCS, CURV, etc.)
- Spectral operations (CPU and GPU-accelerated)
- BCG query selection
- FDR validation
- Evaluation metrics
- Production pipeline
"""

__version__ = "1.0.0"

# Core modules
from .data_loading import load_embeddings, build_gt_mapping, load_rts_distractor_bank
from .features import compute_polarity_scores, compute_rsc_scores, compute_margin_scores
from .spectral_ops import spectral_refinement, build_mutual_knn_graph
from .bcg_selection import select_queries_bcg
from .fdr_validation import validate_flip_fdr, storey_bh_robust
from .metrics import compute_base_r1, compute_flip_outcomes, compute_complementarity_table
from .production_pipeline import run_production_pipeline, run_variant_production

# GPU-accelerated operations
from .gpu_ops import (
    batch_spectral_smooth,
    compute_rts_batched,
    compute_ccs_batched,
    compute_curv_batched,
    compute_dhc_batched,
    compute_rtps_batched,
    batch_lfa_refinement,
    compute_bcg_scores,
    compute_flips_from_arrays,
    PROD_PARAMS,
)

__all__ = [
    # Data loading
    'load_embeddings', 'build_gt_mapping', 'load_rts_distractor_bank',
    # Features
    'compute_polarity_scores', 'compute_rsc_scores', 'compute_margin_scores',
    # Spectral
    'spectral_refinement', 'build_mutual_knn_graph',
    # BCG
    'select_queries_bcg',
    # FDR
    'validate_flip_fdr', 'storey_bh_robust',
    # Metrics
    'compute_base_r1', 'compute_flip_outcomes', 'compute_complementarity_table',
    # Pipeline
    'run_production_pipeline', 'run_variant_production',
    # GPU ops
    'batch_spectral_smooth', 'compute_rts_batched', 'compute_ccs_batched',
    'compute_curv_batched', 'compute_dhc_batched', 'compute_rtps_batched',
    'batch_lfa_refinement', 'compute_bcg_scores', 'compute_flips_from_arrays',
    'PROD_PARAMS',
]
