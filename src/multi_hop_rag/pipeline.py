"""Main pipeline for multi-hop RAG system."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import Settings
from .chunking import HierarchicalChunker, DocumentChunk
from .embedding import HuggingFaceEmbedder
from .indexing import ChromaStore
from .retrieval import MultiHopRetriever
from .graph import MultiHopRAGGraph

logger = logging.getLogger(__name__)


class MultiHopRAGPipeline:
    """End-to-end multi-hop RAG pipeline."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the RAG pipeline.

        Args:
            settings: Optional settings (will load from env if not provided)
        """
        if settings is None:
            settings = Settings()
        
        self.settings = settings
        
        # Initialize components
        logger.info("Initializing multi-hop RAG pipeline")
        
        self.chunker = HierarchicalChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        
        self.embedder = HuggingFaceEmbedder(
            api_key=settings.huggingface_api_key,
            model_name=settings.huggingface_embedding_model,
        )
        
        self.chroma_store = ChromaStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
        )
        
        self.retriever = MultiHopRetriever(
            chroma_store=self.chroma_store,
            embedder=self.embedder,
            top_k=settings.top_k_retrieval,
            max_hops=settings.max_hops,
        )
        
        self.rag_graph = MultiHopRAGGraph(
            retriever=self.retriever,
            settings=settings,
        )
        
        logger.info("Pipeline initialized successfully")

    def index_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Index a single document.

        Args:
            document_path: Path to the document file
            metadata: Optional metadata to attach to all chunks

        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing document: {document_path}")
        
        # Chunk the document
        chunks = self.chunker.chunk_document(document_path, metadata)
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Add to vector store
        self.chroma_store.add_documents(chunks, embeddings)
        
        logger.info(f"Indexed {len(chunks)} chunks from {document_path}")
        return len(chunks)

    def index_documents(
        self,
        document_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Index multiple documents.

        Args:
            document_paths: List of paths to document files
            metadata: Optional metadata to attach to all chunks

        Returns:
            Total number of chunks indexed
        """
        total_chunks = 0
        for doc_path in document_paths:
            try:
                chunks = self.index_document(doc_path, metadata)
                total_chunks += chunks
            except Exception as e:
                logger.error(f"Failed to index {doc_path}: {str(e)}")
        
        logger.info(f"Indexed {total_chunks} total chunks from {len(document_paths)} documents")
        return total_chunks

    def index_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Index raw text.

        Args:
            text: Text to index
            metadata: Optional metadata to attach to all chunks

        Returns:
            Number of chunks indexed
        """
        logger.info("Indexing raw text")
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text, metadata)
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Add to vector store
        self.chroma_store.add_documents(chunks, embeddings)
        
        logger.info(f"Indexed {len(chunks)} chunks from text")
        return len(chunks)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question

        Returns:
            Dictionary with answer and metadata
        """
        return self.rag_graph.query(question)

    async def aquery(self, question: str) -> Dict[str, Any]:
        """
        Async query to the RAG system.

        Args:
            question: User question

        Returns:
            Dictionary with answer and metadata
        """
        return await self.rag_graph.aquery(question)

    def reset(self) -> None:
        """Reset the vector store (delete all indexed documents)."""
        logger.warning("Resetting vector store - all indexed documents will be deleted")
        self.chroma_store.reset_collection()
        logger.info("Vector store reset successfully")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed documents.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": self.chroma_store.count(),
            "collection_name": self.settings.chroma_collection_name,
            "embedding_model": self.settings.huggingface_embedding_model,
            "llm_model": self.settings.openai_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "max_hops": self.settings.max_hops,
            "top_k_retrieval": self.settings.top_k_retrieval,
        }


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
