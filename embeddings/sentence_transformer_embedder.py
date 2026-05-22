import time
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder implementation using the sentence-transformers library.
    Handles E5 model specific prefixes automatically.
    """
    def __init__(self, model_name: str, hf_repo_id: str):
        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        
        logger.info(f"Loading SentenceTransformer model: {self.hf_repo_id}...")
        self.model = SentenceTransformer(self.hf_repo_id)
        
        self.requires_prefix = "e5" in self.model_name.lower()

    def _apply_prefix(self, texts: List[str], prefix: str) -> List[str]:
        if self.requires_prefix:
            return [f"{prefix} {text}" for text in texts]
        return texts

    def encode_documents(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        processed_texts = self._apply_prefix(texts, "passage:")
        
        start_time = time.perf_counter()
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embeddings)

    def encode_queries(self, queries: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(queries, str):
            queries = [queries]
            
        processed_queries = self._apply_prefix(queries, "query:")
        
        embeddings = self.model.encode(
            processed_queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        return self.model.get_embedding_dimension()
