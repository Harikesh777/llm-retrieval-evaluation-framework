from typing import List, Dict, Any
from langchain_text_splitters import TokenTextSplitter
from .base import BaseChunker

class TokenChunker(BaseChunker):
    """
    Chunks text strictly by token limits.
    """
    def __init__(self, token_size: int):
        super().__init__(name=f"token_{token_size}")
        self.token_size = token_size
        try:
            import tiktoken
            self.splitter = TokenTextSplitter(
                chunk_size=self.token_size,
                chunk_overlap=0
            )
        except ImportError:
            self.splitter = None

    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = []
        if not text.strip():
            return chunks

        if self.splitter:
            splits = self.splitter.split_text(text)
        else:
            words = text.split()
            splits = []
            for i in range(0, len(words), self.token_size):
                splits.append(" ".join(words[i:i + self.token_size]))

        for idx, split in enumerate(splits):
            chunk_id = self._generate_chunk_id(doc_id, idx)
            chunks.append({
                "chunk_id": chunk_id,
                "text": split,
                "metadata": metadata.copy()
            })
            
        return chunks
