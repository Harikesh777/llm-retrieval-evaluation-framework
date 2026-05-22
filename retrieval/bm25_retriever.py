import time
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
import logging
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    import re
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

class BM25Retriever(BaseRetriever):
    """
    Sparse Lexical Retriever using BM25Okapi.
    Ignores dense embeddings entirely.
    """
    
    def __init__(self):
        super().__init__()
        self.bm25 = None
        self.chunks = None

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        start_time = time.perf_counter()
        
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        end_time = time.perf_counter()
        self.stats["indexing_time_ms"] = (end_time - start_time) * 1000.0
        
        mem_mb = (len(chunks) / 10000.0) * 1.5 
        self.stats["memory_usage_mb"] = mem_mb
        
        logger.info(f"BM25Retriever: Indexed {len(chunks)} items in {self.stats['indexing_time_ms']:.2f} ms")

    def retrieve(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        if self.bm25 is None or not query_text:
            return [], [], []
            
        tokenized_query = tokenize(query_text)
        if not tokenized_query:
            return [], [], []
            
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        chunk_ids = []
        scores = []
        out_chunks = []
        
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score > 0: # BM25 can return 0 if no overlap
                chunk = self.chunks[idx]
                chunk_ids.append(chunk.get("chunk_id"))
                scores.append(score)
                out_chunks.append(chunk)
                
        return chunk_ids, scores, out_chunks
