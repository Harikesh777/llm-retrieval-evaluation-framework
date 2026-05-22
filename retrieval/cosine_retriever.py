import time
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class CosineRetriever(BaseRetriever):
    """
    Baseline exact retriever using scikit-learn cosine_similarity.
    """
    
    def __init__(self):
        super().__init__()
        self.doc_embeddings = None
        self.chunks = None

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        start_time = time.perf_counter()
        
        self.doc_embeddings = embeddings
        self.chunks = chunks
        
        end_time = time.perf_counter()
        self.stats["indexing_time_ms"] = (end_time - start_time) * 1000.0
        
        mem_mb = self.doc_embeddings.nbytes / (1024 * 1024)
        self.stats["memory_usage_mb"] = mem_mb
        
        logger.info(f"CosineRetriever: Indexed {len(chunks)} items in {self.stats['indexing_time_ms']:.2f} ms")

    def retrieve(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        if self.doc_embeddings is None or len(self.doc_embeddings) == 0:
            return [], [], []
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        chunk_ids = []
        out_scores = []
        out_chunks = []
        
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk_ids.append(chunk.get("chunk_id"))
            out_scores.append(float(scores[idx]))
            out_chunks.append(chunk)
            
        return chunk_ids, out_scores, out_chunks
