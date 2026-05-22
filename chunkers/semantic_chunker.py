import re
from typing import List, Dict, Any
from .base import BaseChunker

class SemanticChunker(BaseChunker):
    """
    Chunk text using a semantic-aware approach (paragraph and sentence splitting).
    Combines small paragraphs into larger ones until min_length is reached.
    """
    def __init__(self, min_length: int = 100):
        super().__init__(name=f"semantic")
        self.min_length = min_length
        
    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = []
        if not text.strip():
            return chunks
            
        paragraphs = re.split(r'\n\n+', text)
        current_chunk_text = ""
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk_text) < self.min_length:
                if current_chunk_text:
                    current_chunk_text += " " + para
                else:
                    current_chunk_text = para
            else:
                chunks.append({
                    "chunk_id": self._generate_chunk_id(doc_id, chunk_idx),
                    "text": current_chunk_text,
                    "metadata": metadata.copy()
                })
                chunk_idx += 1
                current_chunk_text = para
                
        if current_chunk_text:
            if len(current_chunk_text) < self.min_length and chunks:
                chunks[-1]["text"] += " " + current_chunk_text
            else:
                chunks.append({
                    "chunk_id": self._generate_chunk_id(doc_id, chunk_idx),
                    "text": current_chunk_text,
                    "metadata": metadata.copy()
                })
                
        return chunks
