import re
from typing import List, Dict, Any
from .base import BaseChunker

class HeadingBasedChunker(BaseChunker):
    """
    Splits text by markdown headings (#, ##, ###).
    """
    def __init__(self):
        super().__init__(name="heading_chunker")
        
    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = []
        if not text.strip():
            return chunks

        splits = re.split(r'(?m)^(#{1,6}\s+.*$)', text)
        
        current_heading = None
        current_text = ""
        chunk_idx = 0
        
        for item in splits:
            item = item.strip()
            if not item:
                continue
                
            if re.match(r'^#{1,6}\s+', item):
                if current_text:
                    chunk_meta = metadata.copy()
                    chunk_meta["section_type"] = "heading"
                    if current_heading:
                        chunk_meta["heading"] = current_heading
                        
                    chunks.append({
                        "chunk_id": self._generate_chunk_id(doc_id, chunk_idx),
                        "text": current_text.strip(),
                        "metadata": chunk_meta
                    })
                    chunk_idx += 1
                current_heading = item
                current_text = item + "\n"
            else:
                current_text += item + "\n"
                
        if current_text.strip():
            chunk_meta = metadata.copy()
            chunk_meta["section_type"] = "heading"
            if current_heading:
                chunk_meta["heading"] = current_heading
                
            chunks.append({
                "chunk_id": self._generate_chunk_id(doc_id, chunk_idx),
                "text": current_text.strip(),
                "metadata": chunk_meta
            })
            
        return chunks


class ParagraphBasedChunker(BaseChunker):
    """
    Splits text by paragraphs (double newlines).
    """
    def __init__(self):
        super().__init__(name="paragraph_chunker")
        
    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = []
        if not text.strip():
            return chunks
            
        splits = re.split(r'\n\s*\n', text)
        
        for idx, split in enumerate(splits):
            split = split.strip()
            if not split:
                continue
                
            chunk_meta = metadata.copy()
            chunk_meta["section_type"] = "paragraph"
            
            chunks.append({
                "chunk_id": self._generate_chunk_id(doc_id, idx),
                "text": split,
                "metadata": chunk_meta
            })
            
        return chunks


class MarkdownChunker(BaseChunker):
    """
    Advanced structure-aware chunking for markdown.
    For this implementation, it uses HeadingBasedChunker but can be extended.
    """
    def __init__(self):
        super().__init__(name="markdown_chunker")
        self.internal_chunker = HeadingBasedChunker()
        
    def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        chunks = self.internal_chunker.chunk(text, metadata)
        for i, chunk in enumerate(chunks):
            doc_id = metadata.get("doc_id", "unknown") if metadata else "unknown"
            chunk["chunk_id"] = self._generate_chunk_id(doc_id, i)
            chunk["metadata"]["section_type"] = "markdown"
        return chunks
