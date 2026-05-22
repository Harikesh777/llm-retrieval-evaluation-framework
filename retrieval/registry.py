from .cosine_retriever import CosineRetriever
from .faiss_flat_retriever import FAISSFlatRetriever
from .faiss_ivf_retriever import FAISSIVFRetriever
from .faiss_hnsw_retriever import FAISSHNSWRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever

RETRIEVAL_REGISTRY = {
    "cosine_flat": CosineRetriever,
    "faiss_flat": FAISSFlatRetriever,
    "faiss_ivf": FAISSIVFRetriever,
    "faiss_hnsw": FAISSHNSWRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever
}
