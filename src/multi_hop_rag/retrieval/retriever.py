"""Multi-hop retrieval implementation."""

import logging
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass

from ..indexing import ChromaStore
from ..embedding import HuggingFaceEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    document: str
    metadata: Dict[str, Any]
    distance: float
    hop: int


class MultiHopRetriever:
    """Performs multi-hop retrieval from ChromaDB."""

    def __init__(
        self,
        chroma_store: ChromaStore,
        embedder: HuggingFaceEmbedder,
        top_k: int = 5,
        max_hops: int = 3,
    ):
        """
        Initialize multi-hop retriever.

        Args:
            chroma_store: ChromaDB vector store
            embedder: HuggingFace embedder for query encoding
            top_k: Number of documents to retrieve per hop
            max_hops: Maximum number of hops to perform
        """
        self.chroma_store = chroma_store
        self.embedder = embedder
        self.top_k = top_k
        self.max_hops = max_hops

    def retrieve(
        self,
        query: str,
        num_hops: Optional[int] = None,
        initial_metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform multi-hop retrieval.

        Args:
            query: The query string
            num_hops: Number of hops to perform (defaults to max_hops)
            initial_metadata_filter: Optional metadata filter for first hop

        Returns:
            List of RetrievalResult objects from all hops
        """
        if num_hops is None:
            num_hops = self.max_hops

        all_results = []
        seen_ids: Set[str] = set()

        # First hop: query-based retrieval
        logger.info(f"Starting {num_hops}-hop retrieval for query: {query}")
        query_embedding = self.embedder.embed_query(query)
        
        hop_results = self.chroma_store.query(
            query_embedding=query_embedding,
            n_results=self.top_k,
            where=initial_metadata_filter,
        )

        # Process first hop results
        for i, doc_id in enumerate(hop_results["ids"]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(
                    RetrievalResult(
                        document=hop_results["documents"][i],
                        metadata=hop_results["metadatas"][i],
                        distance=hop_results["distances"][i],
                        hop=1,
                    )
                )

        logger.info(f"Hop 1: Retrieved {len(hop_results['ids'])} documents")

        # Subsequent hops: expand search based on retrieved documents
        for hop_num in range(2, num_hops + 1):
            if not all_results:
                logger.warning(f"No results to expand from at hop {hop_num}")
                break

            # Get documents from previous hop
            prev_hop_docs = [r for r in all_results if r.hop == hop_num - 1]
            
            if not prev_hop_docs:
                logger.warning(f"No documents from hop {hop_num - 1} to expand")
                break

            # Extract key terms and metadata from previous hop
            expansion_queries = self._generate_expansion_queries(prev_hop_docs)

            hop_results_ids = set()
            for expansion_query in expansion_queries:
                expansion_embedding = self.embedder.embed_query(expansion_query)
                
                expanded_results = self.chroma_store.query(
                    query_embedding=expansion_embedding,
                    n_results=self.top_k,
                )

                # Add new results
                for i, doc_id in enumerate(expanded_results["ids"]):
                    if doc_id not in seen_ids and doc_id not in hop_results_ids:
                        seen_ids.add(doc_id)
                        hop_results_ids.add(doc_id)
                        all_results.append(
                            RetrievalResult(
                                document=expanded_results["documents"][i],
                                metadata=expanded_results["metadatas"][i],
                                distance=expanded_results["distances"][i],
                                hop=hop_num,
                            )
                        )

            logger.info(f"Hop {hop_num}: Retrieved {len(hop_results_ids)} new documents")

        logger.info(f"Total retrieved: {len(all_results)} documents across {num_hops} hops")
        return all_results

    def _generate_expansion_queries(self, documents: List[RetrievalResult]) -> List[str]:
        """
        Generate expansion queries from retrieved documents.

        Args:
            documents: Documents from previous hop

        Returns:
            List of expansion query strings
        """
        expansion_queries = []

        for doc in documents:
            # Extract section references from metadata
            metadata = doc.metadata
            
            # Try to find references to other sections
            section = metadata.get("section")
            if section:
                expansion_queries.append(f"Related to section {section}")

            # Add key phrases from the document (first 200 chars)
            if doc.document:
                snippet = doc.document[:200].strip()
                if snippet:
                    expansion_queries.append(snippet)

        # Deduplicate and limit
        expansion_queries = list(set(expansion_queries))[:3]
        
        if not expansion_queries:
            # Fallback: use original documents as queries
            expansion_queries = [doc.document[:200] for doc in documents[:2]]

        return expansion_queries

    def retrieve_with_context(
        self,
        query: str,
        num_hops: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents with additional context.

        Args:
            query: The query string
            num_hops: Number of hops to perform

        Returns:
            Dictionary with results organized by hop
        """
        results = self.retrieve(query, num_hops)

        # Organize by hop
        organized = {
            "query": query,
            "total_documents": len(results),
            "hops": {},
        }

        for result in results:
            hop_key = f"hop_{result.hop}"
            if hop_key not in organized["hops"]:
                organized["hops"][hop_key] = []
            
            organized["hops"][hop_key].append({
                "document": result.document,
                "metadata": result.metadata,
                "distance": result.distance,
            })

        return organized
