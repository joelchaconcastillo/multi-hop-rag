"""Multi-hop retrieval implementation."""

import logging
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass

from ..indexing import ChromaStore
from ..embedding import HuggingFaceEmbedder
from ..categorization import CategoryScorer, DEFAULT_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    document: str
    metadata: Dict[str, Any]
    distance: float
    hop: int
    score: float


class MultiHopRetriever:
    """Performs multi-hop retrieval from ChromaDB."""

    def __init__(
        self,
        chroma_store: ChromaStore,
        embedder: HuggingFaceEmbedder,
        top_k: int = 5,
        max_hops: int = 3,
        categorizer: Optional[CategoryScorer] = None,
        similarity_weight: float = 0.7,
        rerank_fetch_multiplier: int = 3,
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
        self.categorizer = categorizer or CategoryScorer(DEFAULT_CATEGORIES)
        self.similarity_weight = similarity_weight
        self.rerank_fetch_multiplier = max(1, rerank_fetch_multiplier)

    def search(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search with category-aware reranking."""
        fetch_k = max(n_results, n_results * self.rerank_fetch_multiplier)

        query_embedding = self.embedder.embed_query(query)
        raw_results = self.chroma_store.query(
            query_embedding=query_embedding,
            n_results=fetch_k,
            where=metadata_filter,
        )

        return self._rerank_results(query, raw_results, n_results)

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
        hop_results = self.search(
            query=query,
            n_results=self.top_k,
            metadata_filter=initial_metadata_filter,
        )

        for doc in hop_results:
            doc_id = doc["id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(
                    RetrievalResult(
                        document=doc["content"],
                        metadata=doc["metadata"],
                        distance=doc["distance"],
                        hop=1,
                        score=doc["score"],
                    )
                )

        logger.info(f"Hop 1: Retrieved {len(hop_results)} documents")

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
                expanded_results = self.search(
                    query=expansion_query,
                    n_results=self.top_k,
                )

                for doc in expanded_results:
                    doc_id = doc["id"]
                    if doc_id not in seen_ids and doc_id not in hop_results_ids:
                        seen_ids.add(doc_id)
                        hop_results_ids.add(doc_id)
                        all_results.append(
                            RetrievalResult(
                                document=doc["content"],
                                metadata=doc["metadata"],
                                distance=doc["distance"],
                                hop=hop_num,
                                score=doc["score"],
                            )
                        )

            logger.info(f"Hop {hop_num}: Retrieved {len(hop_results_ids)} new documents")

        logger.info(f"Total retrieved: {len(all_results)} documents across {num_hops} hops")
        return all_results

    def _rerank_results(
        self,
        query: str,
        results: Dict[str, Any],
        n_results: int,
    ) -> List[Dict[str, Any]]:
        query_weights = self.categorizer.score_text(query)
        scored: List[Dict[str, Any]] = []

        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            content = results["documents"][i]
            distance = results["distances"][i]

            doc_weights = self.categorizer.parse_weights(metadata.get("category_weights"))
            if not doc_weights:
                doc_weights = self.categorizer.score_text(content)

            category_score = self._dot_score(query_weights, doc_weights)
            similarity = 1.0 / (1.0 + float(distance))
            combined = (self.similarity_weight * similarity) + (
                (1.0 - self.similarity_weight) * category_score
            )

            scored.append({
                "id": results["ids"][i],
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "similarity": similarity,
                "category_score": category_score,
                "score": combined,
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:n_results]

    def _dot_score(
        self,
        query_weights: Dict[str, float],
        doc_weights: Dict[str, float],
    ) -> float:
        if not query_weights or not doc_weights:
            return 0.0
        score = 0.0
        for category, query_weight in query_weights.items():
            score += query_weight * doc_weights.get(category, 0.0)
        return score

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
                "score": result.score,
            })

        return organized
