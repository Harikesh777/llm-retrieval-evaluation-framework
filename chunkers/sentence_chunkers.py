import re
from typing import List, Dict, Any
from .base import BaseChunker

class SentenceChunker(BaseChunker):
    """
    Groups N sentences per chunk.
    Uses nltk.sent_tokenize if available, else regex fallback.
    """
    def __init__(self, num_sentences: int):
        super().__init__(name=f"sentence_{num_sentences}")
        self.num_sentences = num_sentences
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
        for i in range(0, len(sentences), self.num_sentences):
            splits.append(" ".join(sentences[i:i + self.num_sentences]))
            
        for idx, split in enumerate(splits):
            chunk_id = self._generate_chunk_id(doc_id, idx)
            chunks.append({
                "chunk_id": chunk_id,
                "text": split,
                "metadata": metadata.copy()
            })
            
        return chunks
