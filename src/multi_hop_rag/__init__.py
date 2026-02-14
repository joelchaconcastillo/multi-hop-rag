"""Multi-hop RAG system with ChromaDB, HuggingFace embeddings, and OpenAI."""

__version__ = "0.1.0"

from .config import Settings
from .pipeline import MultiHopRAGPipeline, setup_logging
from .chunking import HierarchicalChunker, DocumentChunk
from .embedding import HuggingFaceEmbedder
from .indexing import ChromaStore
from .retrieval import MultiHopRetriever
from .graph import MultiHopRAGGraph, RAGState

__all__ = [
    "Settings",
    "MultiHopRAGPipeline",
    "setup_logging",
    "HierarchicalChunker",
    "DocumentChunk",
    "HuggingFaceEmbedder",
    "ChromaStore",
    "MultiHopRetriever",
    "MultiHopRAGGraph",
    "RAGState",
]
