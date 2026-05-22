import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedding_dim: int):
        """
        Initialize the FAISS index for retrieval.
        Using IndexFlatIP for Inner Product. Since embeddings are L2 normalized,
        Inner Product is equivalent to Cosine Similarity.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = [] # Keep a reference to chunks corresponding to vectors
        
    def add_chunks(self, doc_embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Add chunks and their embeddings to the index.
        """
        if len(doc_embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
            
        self.index.add(doc_embeddings.astype('float32'))
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[str]]:
        """
        Retrieve Top-K chunks for a given query embedding.
        
        Args:
            query_embedding (np.ndarray): The embedding vector of the query. shape (1, dim) or (dim,)
            k (int): Number of top chunks to retrieve.
            
        Returns:
            Tuple containing:
            - List of chunk IDs
            - List of similarity scores
            - List of chunk texts
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty. Returning empty results.")
            return [], [], []
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        chunk_ids = []
        scores = []
        texts = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk_ids.append(chunk.get("chunk_id"))
                scores.append(float(dist))
                texts.append(chunk.get("text"))
                
        return chunk_ids, scores, texts

def retrieve(query_embedding: np.ndarray, doc_embeddings: np.ndarray, k: int = 5) -> Tuple[List[int], List[float], List[str]]:
    """
    Stateless retrieval function matching the requested signature.
    Assuming doc_embeddings maps 1:1 with an external chunk list.
    Since the prompt asked to return chunk text, we'll need to pass chunks or assume it returns indices.
    To strictly match requirements and be useful, we will return the indices here, and the caller can map them.
    Wait, the requirement says "Return: chunk IDs, similarity scores, chunk text" but signature is `retrieve(query_embedding, doc_embeddings, k=5)`.
    Let's adjust it slightly to take `chunks` as an argument to fulfill the return requirements.
    """
    pass # Will implement a helper function below instead

def stateless_retrieve(query_embedding: np.ndarray, doc_embeddings: np.ndarray, chunks: List[Dict[str, Any]], k: int = 5) -> Tuple[List[str], List[float], List[str]]:
    """
    Stateless version of retrieve.
    """
    if len(doc_embeddings) == 0:
        return [], [], []
        
    dim = doc_embeddings.shape[1]
    retriever = Retriever(embedding_dim=dim)
    retriever.add_chunks(doc_embeddings, chunks)
    return retriever.retrieve(query_embedding, k=k)
