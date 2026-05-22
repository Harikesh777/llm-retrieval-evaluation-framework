from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models in the evaluation pipeline.
    """
    
    @abstractmethod
    def encode_documents(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode document texts into embeddings.
        
        Args:
            texts (Union[str, List[str]]): The text or list of texts to encode.
            batch_size (int): The batch size to use during encoding.
            
        Returns:
            np.ndarray: A numpy array of embeddings for the documents.
        """
        pass

    @abstractmethod
    def encode_queries(self, queries: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode queries into embeddings.
        
        Args:
            queries (Union[str, List[str]]): The query or list of queries to encode.
            batch_size (int): The batch size to use during encoding.
            
        Returns:
            np.ndarray: A numpy array of embeddings for the queries.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Retrieve the dimension size of the embedding vectors produced by this model.
        
        Returns:
            int: The embedding dimension size.
        """
        pass
