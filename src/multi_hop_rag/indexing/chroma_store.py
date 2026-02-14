"""ChromaDB vector store for document indexing and retrieval."""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

from ..chunking import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages ChromaDB vector store for document chunks."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "multi_hop_rag",
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize ChromaDB store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedding_function: Optional embedding function (if None, must provide embeddings)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Multi-hop RAG document store"}
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors corresponding to chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            logger.warning("No chunks to add")
            return

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # Convert metadata values to serializable types
            batch_metadatas = [
                self._sanitize_metadata(meta) for meta in batch_metadatas
            ]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_metadatas,
            )

            logger.debug(f"Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

        logger.info(f"Added {len(chunks)} chunks to collection '{self.collection_name}'")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure all values are ChromaDB-compatible.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, Path):
                sanitized[key] = str(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Query results containing ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        # Flatten results (remove outer list dimension)
        flattened_results = {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

        logger.debug(f"Query returned {len(flattened_results['ids'])} results")
        return flattened_results

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            Documents, metadatas, and embeddings
        """
        results = self.collection.get(ids=ids)
        logger.debug(f"Retrieved {len(results['ids'])} documents by ID")
        return results

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    def reset_collection(self) -> None:
        """Reset the collection (delete and recreate)."""
        self.delete_collection()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Multi-hop RAG document store"}
        )
        logger.info(f"Reset collection: {self.collection_name}")

    def count(self) -> int:
        """Get the number of documents in the collection."""
        count = self.collection.count()
        logger.debug(f"Collection contains {count} documents")
        return count

    def get_all_documents(self) -> Dict[str, Any]:
        """
        Get all documents in the collection.

        Returns:
            All documents with their IDs, metadatas, and embeddings
        """
        results = self.collection.get()
        logger.debug(f"Retrieved all {len(results['ids'])} documents")
        return results
