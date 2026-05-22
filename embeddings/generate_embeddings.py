import os
import pickle
import hashlib
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = 'all-MiniLM-L6-v2'
CACHE_DIR = '../data/processed/embeddings_cache'

class EmbeddingGenerator:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
    def _get_cache_path(self, text: str) -> str:
        """Generate a deterministic file path for caching based on text hash."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) into normalized numpy arrays.
        Utilizes local file caching to avoid recomputing embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        emb = pickle.load(f)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Error loading cache for text index {i}: {e}. Will recompute.")
                    texts_to_compute.append(text)
                    indices_to_compute.append(i)
                    embeddings.append(None) # Placeholder
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
                embeddings.append(None) # Placeholder
                
        if texts_to_compute:
            logger.info(f"Computing embeddings for {len(texts_to_compute)} new texts...")
            new_embeddings = self.model.encode(
                texts_to_compute, 
                batch_size=batch_size, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            for idx, emb, text in zip(indices_to_compute, new_embeddings, texts_to_compute):
                embeddings[idx] = emb
                cache_path = self._get_cache_path(text)
                with open(cache_path, 'wb') as f:
                    pickle.dump(emb, f)
                    
        return np.array(embeddings)
        
    def encode_chunks(self, chunks: List[dict]) -> np.ndarray:
        """Encode a list of chunk dictionaries."""
        texts = [chunk["text"] for chunk in chunks]
        return self.encode(texts)
        
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode a list of query strings."""
        return self.encode(queries)
