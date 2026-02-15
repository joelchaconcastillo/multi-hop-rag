"""LangGraph implementation of 3-hop multi-hop RAG."""

import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval import MultiHopRetriever
from ..config import Settings
from ..prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    build_decomposition_prompt,
    build_answer_prompt,
)

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for the multi-hop RAG graph."""

    # Input
    question: str
    
    # Query decomposition
    decomposed_queries: List[str]
    
    # Retrieval results by hop
    hop_1_results: List[Dict[str, Any]]
    hop_2_results: List[Dict[str, Any]]
    hop_3_results: List[Dict[str, Any]]
    
    # All retrieved documents
    all_documents: Annotated[List[Dict[str, Any]], add]
    
    # Final answer
    answer: str
    
    # Metadata
    current_hop: int
    total_documents_retrieved: int


class MultiHopRAGGraph:
    """LangGraph-based multi-hop RAG system."""

    def __init__(
        self,
        retriever: MultiHopRetriever,
        settings: Settings,
    ):
        """
        Initialize the multi-hop RAG graph.

        Args:
            retriever: Multi-hop retriever instance
            settings: Application settings
        """
        self.retriever = retriever
        self.settings = settings
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("decompose_query", self._decompose_query)
        workflow.add_node("retrieve_hop_1", self._retrieve_hop_1)
        workflow.add_node("retrieve_hop_2", self._retrieve_hop_2)
        workflow.add_node("retrieve_hop_3", self._retrieve_hop_3)
        workflow.add_node("generate_answer", self._generate_answer)

        # Add edges
        workflow.set_entry_point("decompose_query")
        workflow.add_edge("decompose_query", "retrieve_hop_1")
        workflow.add_edge("retrieve_hop_1", "retrieve_hop_2")
        workflow.add_edge("retrieve_hop_2", "retrieve_hop_3")
        workflow.add_edge("retrieve_hop_3", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _log_block(self, title: str, lines: List[str]) -> None:
        """Log a visual block for hop inputs/outputs."""
        border = "=" * 80
        block = "\n".join([
            border,
            title,
            "-" * 80,
            *lines,
            border,
        ])
        logger.info("\n%s", block)

    def _summarize_docs(self, docs: List[Dict[str, Any]], limit: int = 5) -> List[str]:
        """Summarize retrieved documents for logging."""
        summary = []
        for idx, doc in enumerate(docs[:limit], start=1):
            section = doc.get("metadata", {}).get("section", "N/A")
            distance = doc.get("distance", "N/A")
            snippet = doc.get("content", "").replace("\n", " ")
            snippet = (snippet[:120] + "...") if len(snippet) > 120 else snippet
            summary.append(
                f"{idx}. section={section} distance={distance} snippet={snippet}"
            )
        if not summary:
            summary.append("(no documents retrieved)")
        return summary

    def _is_complex_question(self, question: str) -> bool:
        """Heuristic check for whether a question is complex."""
        lowered = question.lower()
        keyword_hits = any(
            token in lowered
            for token in (" and ", " or ", "compare", "relationship", "how", "why", "impact")
        )
        word_count = len(question.split())
        return keyword_hits or word_count >= 12

    def _collect_section_refs(
        self,
        docs: List[Dict[str, Any]],
        limit: int = 6,
    ) -> List[str]:
        """Collect unique section references from retrieved docs."""
        sections: List[str] = []
        for doc in docs:
            section = doc.get("metadata", {}).get("section")
            if section and section not in sections:
                sections.append(str(section))
            if len(sections) >= limit:
                break
        return sections

    def _build_stage_query(
        self,
        question: str,
        focus: str,
        subquery: Optional[str] = None,
        docs: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a stage query that keeps subqueries but follows the 3-step strategy."""
        sections_text = "N/A"
        if docs:
            sections = self._collect_section_refs(docs)
            sections_text = ", ".join(sections) if sections else "N/A"
        subquery_text = subquery or question
        return (
            f"{question}\n"
            f"Focus: {focus}\n"
            f"Guiding sub-query: {subquery_text}\n"
            f"Relevant sections: {sections_text}"
        )

    def _decompose_query(self, state: RAGState) -> RAGState:
        """
        Decompose the user query into sub-queries for multi-hop retrieval.

        Args:
            state: Current graph state

        Returns:
            Updated state with decomposed queries
        """
        logger.info("Decomposing query into sub-queries")
        
        question = state["question"]
        
        prompt = build_decomposition_prompt(question)

        messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        content = response.content
        
        # Parse sub-queries from response
        queries = []
        for line in content.split('\n'):
            line = line.strip()
            # Remove numbering
            if line and any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, 10)):
                query = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if query:
                    queries.append(query)
        
        # If no queries were parsed, use the original question
        if not queries:
            queries = [question]
        
        logger.info(f"Decomposed into {len(queries)} sub-queries")
        self._log_block(
            "DECOMPOSITION",
            [
                f"Question: {question}",
                f"Sub-queries: {queries}",
            ],
        )
        
        return {
            **state,
            "decomposed_queries": queries,
            "current_hop": 1,
            "all_documents": [],
        }

    def _retrieve_hop_1(self, state: RAGState) -> RAGState:
        """
        Perform first hop retrieval.

        Args:
            state: Current graph state

        Returns:
            Updated state with hop 1 results
        """
        logger.info("Performing hop 1 retrieval")
        
        subquery = state["decomposed_queries"][0] if state["decomposed_queries"] else None
        query = self._build_stage_query(
            question=state["question"],
            focus="Identify relevant sections",
            subquery=subquery,
        )
        self._log_block(
            "HOP 1 INPUT",
            [
                f"Query: {query}",
            ],
        )
        # Retrieve documents
        query_embedding = self.retriever.embedder.embed_query(query)
        results = self.retriever.chroma_store.query(
            query_embedding=query_embedding,
            n_results=self.settings.top_k_retrieval,
        )
        
        # Format results
        hop_1_docs = []
        for i in range(len(results["ids"])):
            hop_1_docs.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i],
            })
        
        logger.info(f"Hop 1: Retrieved {len(hop_1_docs)} documents")
        self._log_block(
            "HOP 1 OUTPUT",
            [f"Documents: {len(hop_1_docs)}"] + self._summarize_docs(hop_1_docs),
        )
        
        return {
            **state,
            "hop_1_results": hop_1_docs,
            "all_documents": hop_1_docs,
            "current_hop": 2,
        }

    def _retrieve_hop_2(self, state: RAGState) -> RAGState:
        """
        Perform second hop retrieval based on first hop results.

        Args:
            state: Current graph state

        Returns:
            Updated state with hop 2 results
        """
        logger.info("Performing hop 2 retrieval")
        
        # Generate expansion query from hop 1 results
        hop_1_docs = state["hop_1_results"]
        
        if not hop_1_docs:
            logger.warning("No hop 1 results to expand from")
            return {
                **state,
                "hop_2_results": [],
                "current_hop": 3,
            }
        
        subquery = state["decomposed_queries"][1] if len(state["decomposed_queries"]) > 1 else None
        query = self._build_stage_query(
            question=state["question"],
            focus="Find specific requirements or rules referenced by these sections",
            subquery=subquery,
            docs=hop_1_docs,
        )
        self._log_block(
            "HOP 2 INPUT",
            [
                f"Query: {query}",
            ],
        )
        # Retrieve documents
        query_embedding = self.retriever.embedder.embed_query(query)
        results = self.retriever.chroma_store.query(
            query_embedding=query_embedding,
            n_results=self.settings.top_k_retrieval,
        )
        
        # Format and filter out duplicates
        hop_1_ids = {doc["id"] for doc in hop_1_docs}
        hop_2_docs = []
        for i in range(len(results["ids"])):
            doc_id = results["ids"][i]
            if doc_id not in hop_1_ids:
                hop_2_docs.append({
                    "id": doc_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "distance": results["distances"][i],
                })
        
        logger.info(f"Hop 2: Retrieved {len(hop_2_docs)} new documents")
        self._log_block(
            "HOP 2 OUTPUT",
            [f"Documents: {len(hop_2_docs)}"] + self._summarize_docs(hop_2_docs),
        )
        
        return {
            **state,
            "hop_2_results": hop_2_docs,
            "all_documents": state["all_documents"] + hop_2_docs,
            "current_hop": 3,
        }

    def _retrieve_hop_3(self, state: RAGState) -> RAGState:
        """
        Perform third hop retrieval based on previous hops.

        Args:
            state: Current graph state

        Returns:
            Updated state with hop 3 results
        """
        logger.info("Performing hop 3 retrieval")
        
        hop_1_docs = state["hop_1_results"]
        hop_2_docs = state["hop_2_results"]
        
        if not hop_1_docs and not hop_2_docs:
            logger.warning("No previous results to expand from")
            return {
                **state,
                "hop_3_results": [],
            }
        
        recent_docs = hop_2_docs if hop_2_docs else hop_1_docs
        subquery = state["decomposed_queries"][2] if len(state["decomposed_queries"]) > 2 else None
        query = self._build_stage_query(
            question=state["question"],
            focus="Expand related requirements, exceptions, definitions",
            subquery=subquery,
            docs=recent_docs,
        )
        
        self._log_block(
            "HOP 3 INPUT",
            [
                f"Query: {query}",
            ],
        )

        # Retrieve documents
        query_embedding = self.retriever.embedder.embed_query(query)
        results = self.retriever.chroma_store.query(
            query_embedding=query_embedding,
            n_results=self.settings.top_k_retrieval,
        )
        
        # Format and filter out duplicates
        existing_ids = {doc["id"] for doc in state["all_documents"]}
        hop_3_docs = []
        for i in range(len(results["ids"])):
            doc_id = results["ids"][i]
            if doc_id not in existing_ids:
                hop_3_docs.append({
                    "id": doc_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "distance": results["distances"][i],
                })
        
        logger.info(f"Hop 3: Retrieved {len(hop_3_docs)} new documents")
        self._log_block(
            "HOP 3 OUTPUT",
            [f"Documents: {len(hop_3_docs)}"] + self._summarize_docs(hop_3_docs),
        )
        
        return {
            **state,
            "hop_3_results": hop_3_docs,
            "all_documents": state["all_documents"] + hop_3_docs,
        }

    def _generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate final answer using retrieved documents.

        Args:
            state: Current graph state

        Returns:
            Updated state with final answer
        """
        logger.info("Generating final answer")
        
        question = state["question"]
        all_docs = state["all_documents"]
        
        # Prepare context from all retrieved documents
        context_parts = []
        for i, doc in enumerate(all_docs[:10]):  # Limit to top 10 documents
            section = doc["metadata"].get("section", "N/A")
            context_parts.append(f"[Document {i+1}] (Section: {section})\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        prompt = build_answer_prompt(question, context)

        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        logger.info("Answer generated successfully")
        
        return {
            **state,
            "answer": answer,
            "total_documents_retrieved": len(all_docs),
        }

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run the multi-hop RAG pipeline.

        Args:
            question: User question

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "decomposed_queries": [],
            "hop_1_results": [],
            "hop_2_results": [],
            "hop_3_results": [],
            "all_documents": [],
            "answer": "",
            "current_hop": 0,
            "total_documents_retrieved": 0,
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Prepare response
        response = {
            "question": question,
            "answer": final_state["answer"],
            "decomposed_queries": final_state["decomposed_queries"],
            "total_documents_retrieved": final_state["total_documents_retrieved"],
            "documents_by_hop": {
                "hop_1": len(final_state["hop_1_results"]),
                "hop_2": len(final_state["hop_2_results"]),
                "hop_3": len(final_state["hop_3_results"]),
            },
            "retrieved_documents": final_state["all_documents"],
        }
        
        logger.info(f"Query completed. Retrieved {final_state['total_documents_retrieved']} documents")
        
        return response

    async def aquery(self, question: str) -> Dict[str, Any]:
        """
        Async version of query method.

        Args:
            question: User question

        Returns:
            Dictionary with answer and metadata
        """
        # For now, just call the sync version
        # LangGraph supports async natively, but we'll keep it simple
        return self.query(question)
