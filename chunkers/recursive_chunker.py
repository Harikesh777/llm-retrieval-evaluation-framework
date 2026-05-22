from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker

class RecursiveChunker(BaseChunker):
    """
    Cascades from paragraphs -> sentences -> words.
    Uses RecursiveCharacterTextSplitter internally.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0):
        super().__init__(name=f"recursive_{chunk_size}_{chunk_overlap}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
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
