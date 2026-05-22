from typing import List, Dict, Any

class BaseChunker:
    """
    Base class for all chunkers in the evaluation pipeline.
    """
    def __init__(self, name: str):
        self.name = name

    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        """
        Splits text into chunks.
        
        Args:
            text (str): The raw text to chunk.
            metadata (dict, optional): Metadata of the original document.
            
        Returns:
            List[Dict]: A list of chunks. Each chunk must be a dict with:
            - chunk_id: str
            - text: str
            - metadata: dict
        """
        raise NotImplementedError("Chunkers must implement the `chunk` method.")

    def _generate_chunk_id(self, doc_id: str, index: int) -> str:
        """
        Generates a unique deterministic ID for the chunk.
        
        Args:
            doc_id (str): The document ID.
            index (int): The index of the chunk in the document.
            
        Returns:
            str: The unique chunk ID.
        """
        return f"{self.name}_{doc_id}_{index}"
