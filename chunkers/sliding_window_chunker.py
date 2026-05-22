import re
from typing import List, Dict, Any
from .base import BaseChunker

class SlidingWindowChunker(BaseChunker):
    """
    Overlapping sentence windows.
    window_size = number of sentences
    stride = step size
    """
    def __init__(self, window_size: int, stride: int):
        super().__init__(name=f"sliding_{window_size}_{stride}")
        self.window_size = window_size
        self.stride = stride
        self._sent_tokenize = self._get_tokenizer()

    def _get_tokenizer(self):
        try:
            import nltk
            return nltk.sent_tokenize
        except ImportError:
            return lambda text: [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = []
        if not text.strip():
            return chunks
            
        sentences = self._sent_tokenize(text)
        splits = []
        
        if not sentences:
            return chunks

        for i in range(0, len(sentences), self.stride):
            window = sentences[i:i + self.window_size]
            splits.append(" ".join(window))
            
            if i + self.window_size >= len(sentences):
                break
                
        for idx, split in enumerate(splits):
            chunk_id = self._generate_chunk_id(doc_id, idx)
            chunks.append({
                "chunk_id": chunk_id,
                "text": split,
                "metadata": metadata.copy()
            })
            
        return chunks
