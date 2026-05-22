import time
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
import logging
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class FAISSHNSWRetriever(BaseRetriever):
    """
    Approximate Nearest Neighbor retriever using FAISS IndexHNSWFlat.
    """
    
    def __init__(self, m: int = 32, ef_search: int = 64, ef_construction: int = 64):
        super().__init__()
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.index = None
        self.chunks = None

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        start_time = time.perf_counter()
        
        dim = embeddings.shape[1]
        
        self.index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        embeddings_f32 = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_f32)
        
        self.index.add(embeddings_f32)
        self.chunks = chunks
        
        end_time = time.perf_counter()
        self.stats["indexing_time_ms"] = (end_time - start_time) * 1000.0
        
        n = embeddings.shape[0]
        base_mem = n * dim * 4
        graph_mem = n * self.m * 4 * 2 # Bidirectional approx
        mem_mb = (base_mem + graph_mem) / (1024 * 1024)
        self.stats["memory_usage_mb"] = mem_mb
        
        logger.info(f"FAISSHNSWRetriever: Indexed {self.index.ntotal} items in {self.stats['indexing_time_ms']:.2f} ms")

    def retrieve(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        q_emb_f32 = query_embedding.astype('float32')
        faiss.normalize_L2(q_emb_f32)
            
        distances, indices = self.index.search(q_emb_f32, top_k)
        
        chunk_ids = []
        scores = []
        out_chunks = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk_ids.append(chunk.get("chunk_id"))
                scores.append(float(dist))
                out_chunks.append(chunk)
                
        return chunk_ids, scores, out_chunks
