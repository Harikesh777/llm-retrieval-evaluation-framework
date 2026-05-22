from .fixed_chunkers import FixedSizeChunker
from .token_chunkers import TokenChunker
from .document_chunkers import HeadingBasedChunker, ParagraphBasedChunker, MarkdownChunker
from .recursive_chunker import RecursiveChunker
from .sentence_chunkers import SentenceChunker
from .sliding_window_chunker import SlidingWindowChunker
from .semantic_chunker import SemanticChunker

CHUNKER_REGISTRY = {
    "semantic": SemanticChunker(100),

    "fixed_100_0": FixedSizeChunker(100),
    "fixed_200_0": FixedSizeChunker(200, 0),
    "fixed_300_0": FixedSizeChunker(300),
    "fixed_500_0": FixedSizeChunker(500),
    "fixed_800_0": FixedSizeChunker(800),
    "fixed_1000_0": FixedSizeChunker(1000),
    
    "fixed_200_50": FixedSizeChunker(200, 50),
    "fixed_300_50": FixedSizeChunker(300, 50),
    "fixed_500_100": FixedSizeChunker(500, 100),
    "fixed_800_100": FixedSizeChunker(800, 100),
    "fixed_1000_200": FixedSizeChunker(1000, 200),
    
    "token_128": TokenChunker(128),
    "token_256": TokenChunker(256),
    "token_512": TokenChunker(512),
    "token_1024": TokenChunker(1024),
    
    "heading_chunker": HeadingBasedChunker(),
    "paragraph_chunker": ParagraphBasedChunker(),
    "markdown_chunker": MarkdownChunker(),
    
    "recursive_300_0": RecursiveChunker(300, 0),
    "recursive_500_0": RecursiveChunker(500, 0),
    "recursive_1000_0": RecursiveChunker(1000, 0),
    
    "sentence_3": SentenceChunker(3),
    "sentence_5": SentenceChunker(5),
    "sentence_10": SentenceChunker(10),
    
    "sliding_5_1": SlidingWindowChunker(5, 1),
    "sliding_5_2": SlidingWindowChunker(5, 2),
    "sliding_10_2": SlidingWindowChunker(10, 2),
}

def get_chunker(name: str):
    """Retrieve a chunker instance by name."""
    if name not in CHUNKER_REGISTRY:
        raise ValueError(f"Chunker '{name}' not found in registry.")
    return CHUNKER_REGISTRY[name]
