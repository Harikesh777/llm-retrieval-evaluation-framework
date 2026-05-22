import os
import hashlib
import numpy as np
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "embeddings")

def _get_cache_path(model_name: str, texts: list, prefix: str = "doc") -> str:
    """Generate a deterministic file path based on the model and hash of all texts."""
    full_text = "".join(texts)
    text_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
    
    filename = f"{model_name}_{prefix}_{text_hash}.npy"
    return os.path.join(CACHE_DIR, filename)

def cache_exists(model_name: str, texts: list, prefix: str = "doc") -> bool:
    """Check if cached embeddings exist for the given model and texts."""
    path = _get_cache_path(model_name, texts, prefix)
    return os.path.exists(path)

def save_embeddings(model_name: str, texts: list, embeddings: np.ndarray, prefix: str = "doc"):
    """Save embeddings to cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    path = _get_cache_path(model_name, texts, prefix)
    np.save(path, embeddings)

def load_embeddings(model_name: str, texts: list, prefix: str = "doc") -> np.ndarray:
    """Load embeddings from cache."""
    path = _get_cache_path(model_name, texts, prefix)
    if os.path.exists(path):
        return np.load(path)
    return None
