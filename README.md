# Control-TTR: Training-Free Retrieval Refinement

**Standalone Package for External Review**

This folder contains all necessary code to replicate the experiments for the BCG (Bidirectional Consistency Gating) training-free retrieval refinement system.

## Environment Setup

### Option 1: Using environment.yml (Recommended)

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate
conda activate control-ttr
```

### Option 2: Manual Conda Setup

```bash
# Create conda environment
conda create -n control-ttr python=3.10 -y
conda activate control-ttr

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install open_clip for LAION/SigLIP
pip install open_clip_torch transformers timm

# Optional: For CLAP audio embeddings
pip install laion-clap
```


### Running in Kubernetes (JupyterHub)

If using kubectl to access a JupyterHub pod:

```bash
# Interactive shell
kubectl exec -it jupyter-<username> -n jupyterhub -- /bin/bash

# Inside the pod:
cd workspace/control_ttr
source /opt/conda/etc/profile.d/conda.sh
conda activate control-ttr

# Run tests
python run_tests.py --mode smoke --embeddings_root /home/workspace/
```

Or run commands directly if you connect using a pod:

```bash
kubectl exec -it jupyter-<username> -n jupyterhub -- /bin/bash -c \
    "cd workspace/control_ttr && \
     source /opt/conda/etc/profile.d/conda.sh && \
     conda activate control-ttr && \
     python run_tests.py --mode smoke --embeddings_root /home/workspace/"
```

## Quick Start

```bash
# 1. Activate environment
conda activate controlttr

# 2. Verify installation
python run_tests.py --mode imports

# 3. Run smoke test (requires embeddings)
python run_tests.py --mode smoke --embeddings_root /path/to/

# 4. Quick GPU test on Flickr30k
python -c "from core.gpu_pipeline import run_quick_test; run_quick_test('flickr30k', 't2i', 'clip', '/path/to/', 'layer')"
```

## Structure

```
control_ttr/
├── README.md                    # This file
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment (recommended)
├── run_tests.py                 # Test runner
├── fetch_datasets.py            # Dataset downloader
├── cache_embeddings.py          # Embedding caching script
├── .gitignore                   # Git ignore rules
│
├── core/                        # Core algorithms (standalone)
│   ├── __init__.py
│   ├── data_loading.py          # Embedding loading utilities
│   ├── features.py              # Feature computation (RTS, CCS, CURV, DHC, RTPS)
│   ├── spectral_ops.py          # Spectral smoothing operations
│   ├── bcg_selection.py         # BCG query selection (Stage 1a)
│   ├── fdr_validation.py        # FDR validation (Layer 3)
│   ├── metrics.py               # Evaluation metrics (R@1, flips)
│   ├── production_pipeline.py   # Full BCG pipeline wrapper
│   ├── spectral_refinement_standalone.py  # Core refinement algorithm
│   ├── gpu_ops.py               # GPU-accelerated batched operations
│   └── gpu_pipeline.py          # Fast GPU experiment runners
│
├── embedders/                   # Embedding models
│   ├── __init__.py
│   ├── clip_embedder.py         # CLIP ViT-L/14@336px
│   ├── siglip_embedder.py       # SigLIP
│   ├── laion_embedder.py        # LAION CLIP
│   ├── blip_embedder.py         # BLIP
│   └── clap_embedder.py         # CLAP (audio-text)
│
├── experiments/                 # Paper experiments
│   ├── run_all_tables.py        # Master script for all tables
│   ├── paper_ablations/         # Individual table scripts (1-14)
│   └── utils/                   # Shared experiment utilities
│
└── results/                     # Output directory
```

## Downloading Datasets

First, download the required datasets:

```bash
# Download all image-text datasets (COCO, Flickr30k)
python fetch_datasets.py --dataset all --output-dir ./datasets

# Or download individually
python fetch_datasets.py --dataset coco --output-dir ./datasets
python fetch_datasets.py --dataset flickr30k --output-dir ./datasets

# For audio-text experiments
python fetch_datasets.py --dataset audiocaps --output-dir ./datasets
python fetch_datasets.py --dataset clotho --output-dir ./datasets

# Convert to experiment format (creates train/train_calib/val/test splits)
python fetch_datasets.py --convert-to-experiment --input-dir ./datasets --output-dir ./datasets/dataset_experiment
```

## Caching Embeddings

After downloading datasets, cache embeddings:

```bash
# CLIP embeddings for COCO and Flickr30k
python cache_embeddings.py --model_type clip --datasets coco_captions,flickr30k

# SigLIP embeddings
python cache_embeddings.py --model_type siglip --datasets flickr30k

# CLAP embeddings for audio datasets
python cache_embeddings.py --model_type clap --datasets audiocaps,clotho --data_root ./datasets
```

## Running Experiments

### Fast GPU-Accelerated Pipeline

```python
from core.gpu_pipeline import run_quick_test

# Layer ablation (Table 1)
run_quick_test('flickr30k', 't2i', 'clip', '/path/to/', 'layer')

# Budget sweep (Table 5)
run_quick_test('coco_captions', 't2i', 'clip', '/path/to/', 'budget')

# Feature ablation (Table 3/6)
run_quick_test('flickr30k', 't2i', 'clip', '/path/to/', 'feature')

# K sweep (Table 7)
run_quick_test('flickr30k', 't2i', 'clip', '/path/to/', 'k')

# Lambda sweep (Table 9)
run_quick_test('flickr30k', 't2i', 'clip', '/path/to/', 'lambda')
```

### Individual Table Scripts

```bash
# Table 1: Layer-wise ablation
python experiments/paper_ablations/table1_layers.py \
    --dataset flickr30k --direction t2i --backbone clip

# Table 5: Budget sweep
python experiments/paper_ablations/table5_budget_sweep.py \
    --dataset coco_captions --direction t2i --backbone clip
```

## Paper Tables

| Table | Description | Script |
|-------|-------------|--------|
| 1 | Layer-wise ablation | `table1_layers.py` |
| 2 | Stage 1a + 1b | `table2_stage1a_1b.py` |
| 3 | Stage 1b robustness | `table3_stage1b_robustness.py` |
| 4 | Multiplicity (m>1) | `table4_multiplicity.py` |
| 5 | Budget sweep | `table5_budget_sweep.py` |
| 6 | Feature weights | `table6_feature_weights.py` |
| 7 | Top-K ablation | `table7_topk_ablation.py` |
| 8 | BCG components | `table8_bcg_components.py` |
| 9 | Lambda sweep | `table9_lambda_sweep.py` |
| 10 | Cross-backbone | `table10_cross_backbone.py` |
| 11 | Cross-dataset | `table11_cross_dataset.py` |
| 12 | Cross-modality | `table12_cross_modality.py` |
| 13 | Computational cost | `table13_computational_cost.py` |
| 14 | Complementarity | `table14_complementarity.py` |

## Key Results

**COCO T2I (CLIP)**:
- Base R@1: 37.10%
- Features-only: **+1.15%** (67% purity)
- LFA (Spectral+Features): +0.43% (60% purity)
- BCG @ 20%: +0.36% (64% purity)

**Flickr30k T2I (CLIP)**:
- Base R@1: 66.88%
- Features-only: **+1.38%** (70% purity)
- LFA: +0.42% (62% purity)
- BCG @ 20%: +0.34% (63% purity)

## Architecture

The system uses a multi-layer architecture:
1. **Layer 1a (Coarse Selection)**: Polarity + RSC rank-sum fusion
2. **Layer 1b (Fine Veto)**: Quality gates (Polarity, Sharpness, Consensus, LS-Margin)
3. **Layer 2 (LFA Refinement)**: Spectral smoothing + Feature boosting
4. **Layer 3 (FDR Validation)**: False Discovery Rate control [ WIRED IN : DEFAULT DISABLED ]

## License

For research purposes only.
