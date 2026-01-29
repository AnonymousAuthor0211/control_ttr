"""
Data loading utilities for Control-TTR.

This module handles loading embeddings and building ground truth mappings.
All paths are relative to the control_ttr/ directory.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_embedding_dir(backbone: str, embeddings_root: Optional[str] = None) -> Path:
    """
    Get the embedding directory path.
    
    Args:
        backbone: Backbone name (e.g., 'clip', 'siglip', 'blip')
        embeddings_root: Root directory containing embeddings_* folders.
                        If None, auto-detects.
                        NOTE: Should be the PARENT directory (e.g., /path/to/UAMR),
                        NOT the embeddings_clip directory itself.
        
    Returns:
        Path to embeddings directory (e.g., embeddings_root/embeddings_clip/)
    """
    if embeddings_root is None:
        # Assume embeddings are in parent directory or current directory
        base = Path(__file__).parent.parent.parent
        # Try parent/UAMR first, then parent, then current
        for candidate in [base / "UAMR", base.parent, base]:
            embed_dir = candidate / f"embeddings_{backbone}"
            if embed_dir.exists():
                return embed_dir
        # If not found, return expected path
        return base / f"embeddings_{backbone}"
    else:
        embeddings_root_path = Path(embeddings_root)
        
        # Check if user provided the embeddings_clip directory directly
        # (e.g., ../embeddings_clip/ instead of ../)
        if embeddings_root_path.name.startswith('embeddings_'):
            # User provided embeddings_clip/ directly, use it as-is
            return embeddings_root_path
        else:
            # User provided parent directory, append embeddings_{backbone}
            return embeddings_root_path / f"embeddings_{backbone}"


def load_embeddings(
    dataset: str,
    direction: str,
    backbone: str = 'clip',
    embeddings_root: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load query and gallery embeddings for a dataset/direction.
    
    Args:
        dataset: Dataset name (e.g., 'coco_captions', 'flickr30k')
        direction: Retrieval direction ('i2t', 't2i', 'a2t', 't2a')
        backbone: Backbone name (e.g., 'clip', 'siglip', 'blip')
        embeddings_root: Root directory containing embeddings_* folders.
                        If None, auto-detects.
        
    Returns:
        (query_emb, gallery_emb, query_ids, gallery_ids)
    """
    embed_dir = get_embedding_dir(backbone, embeddings_root)
    
    # Map direction to modalities
    if direction == 'i2t':
        query_mod, gallery_mod = 'image', 'text'
    elif direction == 't2i':
        query_mod, gallery_mod = 'text', 'image'
    elif direction == 'a2t':
        query_mod, gallery_mod = 'audio', 'text'
    elif direction == 't2a':
        query_mod, gallery_mod = 'text', 'audio'
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Load embeddings
    query_file = embed_dir / f"{dataset}_test_{query_mod}.npz"
    gallery_file = embed_dir / f"{dataset}_test_{gallery_mod}.npz"
    
    if not query_file.exists() or not gallery_file.exists():
        raise FileNotFoundError(
            f"Embeddings not found:\n"
            f"  Query: {query_file}\n"
            f"  Gallery: {gallery_file}\n"
            f"  Embedding dir: {embed_dir}\n"
            f"  Make sure embeddings_{backbone}/ folder exists with {dataset}_test_*.npz files"
        )
    
    query_data = np.load(query_file)
    gallery_data = np.load(gallery_file)
    
    query_emb = torch.tensor(query_data['embeddings'], dtype=torch.float32)
    gallery_emb = torch.tensor(gallery_data['embeddings'], dtype=torch.float32)
    query_ids = query_data['ids']
    gallery_ids = gallery_data['ids']
    
    return query_emb, gallery_emb, query_ids, gallery_ids


def load_rts_distractor_bank(
    dataset: str,
    direction: str,
    backbone: str = 'clip',
    embeddings_root: Optional[str] = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load TRAIN embeddings for RTS distractor bank (deployment-safe).
    
    RTS (Round-Trip Specificity) measures how specific a candidate is to the query
    vs. a background query distribution. Using train embeddings instead of test
    queries ensures no data leakage.
    
    Args:
        dataset: Dataset name (e.g., 'coco_captions', 'clotho')
        direction: 'i2t', 't2i', 'a2t', 't2a'
        backbone: Backbone name (e.g., 'clip', 'siglip', 'clap')
        embeddings_root: Root directory containing embeddings_* folders.
                        If None, auto-detects.
        
    Returns:
        (image_bank, text_bank): Train embeddings for RTS distractor bank.
        One will be None depending on direction.
    """
    embed_dir = get_embedding_dir(backbone, embeddings_root)
    image_bank = None
    text_bank = None
    is_audio = direction in ['a2t', 't2a']
    
    # For I2T/A2T: query is image/audio, distractors are other images/audios (from train)
    # For T2I/T2A: query is text, distractors are other texts (from train)
    
    if direction in ['i2t', 'a2t']:
        if is_audio:
            train_path = embed_dir / f"{dataset}_train_audio.npz"
        else:
            train_path = embed_dir / f"{dataset}_train_image.npz"
        
        if train_path.exists():
            data = np.load(train_path, allow_pickle=True)
            image_bank = torch.tensor(data['embeddings'].astype(np.float32))
            print(f"  ✅ Loaded RTS distractor bank (TRAIN): {train_path.name}, shape={image_bank.shape}")
        else:
            print(f"  ⚠️ RTS train bank not found: {train_path}, will use test queries as fallback")
        
    elif direction in ['t2i', 't2a']:
        train_path = embed_dir / f"{dataset}_train_text.npz"
        
        if train_path.exists():
            data = np.load(train_path, allow_pickle=True)
            text_bank = torch.tensor(data['embeddings'].astype(np.float32))
            print(f"  ✅ Loaded RTS distractor bank (TRAIN): {train_path.name}, shape={text_bank.shape}")
        else:
            print(f"  ⚠️ RTS train bank not found: {train_path}, will use test queries as fallback")
    
    return image_bank, text_bank


def build_gt_mapping(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    direction: str
) -> Dict[str, List[int]]:
    """
    Build ground truth mapping from query IDs to gallery indices.
    
    Args:
        query_ids: Query identifiers [N]
        gallery_ids: Gallery identifiers [M]
        direction: Retrieval direction ('i2t', 't2i', 'a2t', 't2a')
        
    Returns:
        Dictionary mapping query_id -> list of gallery indices
    """
    gt_mapping = {}
    
    # Build gallery index for fast lookup
    gallery_ids_list = [str(gid) for gid in gallery_ids]
    
    # For I2T: query is image ID, gallery items are text with _capX suffix
    # For T2I: query is text with _capX suffix, gallery is image ID
    
    if direction in ['i2t', 'a2t']:
        # Query = image/audio ID, Gallery = text with _capX
        # Build reverse index: base_id -> list of gallery indices
        base_to_gallery = {}
        for j, gid in enumerate(gallery_ids_list):
            if '_cap' in gid:
                base_id = gid.rsplit('_cap', 1)[0]
            else:
                base_id = gid
            if base_id not in base_to_gallery:
                base_to_gallery[base_id] = []
            base_to_gallery[base_id].append(j)
        
        for i, qid in enumerate(query_ids):
            qid_str = str(qid)
            if qid_str in base_to_gallery:
                gt_mapping[qid_str] = base_to_gallery[qid_str]
    else:
        # Query = text with _capX, Gallery = image/audio ID
        # Build gallery index
        gallery_id_to_idx = {gid: j for j, gid in enumerate(gallery_ids_list)}
        
        for i, qid in enumerate(query_ids):
            qid_str = str(qid)
            if '_cap' in qid_str:
                base_id = qid_str.rsplit('_cap', 1)[0]
            else:
                base_id = qid_str
            
            if base_id in gallery_id_to_idx:
                gt_mapping[qid_str] = [gallery_id_to_idx[base_id]]
    
    return gt_mapping
