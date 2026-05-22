from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker

class FixedSizeChunker(BaseChunker):
    """
    Chunks text by fixed character sizes without overlap.
    """
    def __init__(self, size: int, overlap: int = 0):
        super().__init__(name=f"fixed_{size}_{overlap}")
        self.size = size
        self.overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.size,
            chunk_overlap=self.overlap
        )

    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = []
        if not text.strip():
            return chunks
            
        splits = self.splitter.split_text(text)
        
        for idx, split in enumerate(splits):
            chunk_id = self._generate_chunk_id(doc_id, idx)
            chunks.append({
                "chunk_id": chunk_id,
                "text": split,
                "metadata": metadata.copy()
            })
            
        return chunks
