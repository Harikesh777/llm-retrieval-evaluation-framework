import time
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from .base_retriever import BaseRetriever
from .faiss_flat_retriever import FAISSFlatRetriever
from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    Combines FAISS Flat exact search with BM25 sparse search.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha (float): Weight for the dense score (0.0 to 1.0).
                           alpha=1.0 is pure dense, alpha=0.0 is pure sparse.
        """
        super().__init__()
        self.alpha = alpha
        self.dense_retriever = FAISSFlatRetriever()
        self.sparse_retriever = BM25Retriever()

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        start_time = time.perf_counter()
        
        self.dense_retriever.build_index(embeddings, chunks)
        self.sparse_retriever.build_index(embeddings, chunks)
        
        self.chunks = chunks
        
        end_time = time.perf_counter()
        self.stats["indexing_time_ms"] = (end_time - start_time) * 1000.0
        self.stats["memory_usage_mb"] = self.dense_retriever.stats["memory_usage_mb"] + self.sparse_retriever.stats["memory_usage_mb"]
        
        logger.info(f"HybridRetriever: Indexed {len(chunks)} items in {self.stats['indexing_time_ms']:.2f} ms")

    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_val = min(scores)
        max_val = max(scores)
        if max_val == min_val:
            return [1.0] * len(scores) if max_val > 0 else [0.0] * len(scores)
        return [(s - min_val) / (max_val - min_val) for s in scores]

    def retrieve(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        candidate_k = max(top_k * 2, 20)
        
        dense_ids, dense_scores, dense_chunks = self.dense_retriever.retrieve(query_embedding, query_text, top_k=candidate_k)
        sparse_ids, sparse_scores, sparse_chunks = self.sparse_retriever.retrieve(query_embedding, query_text, top_k=candidate_k)
        
        norm_dense_scores = self._min_max_normalize(dense_scores)
        norm_sparse_scores = self._min_max_normalize(sparse_scores)
        
        dense_map = dict(zip(dense_ids, norm_dense_scores))
        sparse_map = dict(zip(sparse_ids, norm_sparse_scores))
        
        chunk_map = {}
        for c, c_id in zip(dense_chunks, dense_ids):
            chunk_map[c_id] = c
        for c, c_id in zip(sparse_chunks, sparse_ids):
            chunk_map[c_id] = c
            
        all_ids = set(dense_ids) | set(sparse_ids)
        
        final_results = []
        for c_id in all_ids:
            d_score = dense_map.get(c_id, 0.0)
            s_score = sparse_map.get(c_id, 0.0)
            
            final_score = (self.alpha * d_score) + ((1.0 - self.alpha) * s_score)
            final_results.append((c_id, final_score, chunk_map[c_id]))
            
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        top_results = final_results[:top_k]
        
        ret_ids = [r[0] for r in top_results]
        ret_scores = [r[1] for r in top_results]
        ret_chunks = [r[2] for r in top_results]
        
        return ret_ids, ret_scores, ret_chunks
