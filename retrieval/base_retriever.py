from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval algorithms in the evaluation pipeline.
    """
    
    def __init__(self):
        self.stats = {
            "indexing_time_ms": 0.0,
            "memory_usage_mb": 0.0
        }
        
    @abstractmethod
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Build the retrieval index using provided embeddings and chunk metadata.
        
        Args:
            embeddings (np.ndarray): The dense document embeddings. Shape (N, dim).
            chunks (List[Dict[str, Any]]): The list of corresponding chunk dictionaries.
        """
        pass

    @abstractmethod
    def retrieve(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Retrieve the top-k most relevant chunks for a given query.
        
        Args:
            query_embedding (np.ndarray): The dense query embedding. Shape (1, dim) or (dim,).
            query_text (str): The raw text of the query (useful for sparse retrievers like BM25).
            top_k (int): The number of results to return.
            
        Returns:
            Tuple containing:
            - List of chunk IDs
            - List of similarity scores
            - List of chunk metadata/dictionaries
        """
        pass

    def get_stats(self) -> Dict[str, float]:
        """
        Retrieve performance statistics for the retriever (e.g. indexing time, memory footprint).
        
        Returns:
            Dict[str, float]: The statistics dict.
        """
        return self.stats
