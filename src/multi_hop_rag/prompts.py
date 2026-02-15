"""Prompt templates for the multi-hop RAG system."""

from __future__ import annotations

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a helpful assistant that decomposes complex questions into simpler sub-queries."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on provided documents. "
    "Always cite your sources."
)


def build_decomposition_prompt(question: str) -> str:
    """Build the prompt for query decomposition."""
    return (
        "Given the following question, break it down into 1-3 sub-queries that can help "
        "answer it through multi-hop retrieval.\n"
        "Each sub-query should focus on a specific aspect or piece of information needed.\n\n"
        f"Question: {question}\n\n"
        "Provide the sub-queries as a numbered list, one per line.\n"
    )


def build_answer_prompt(question: str, context: str) -> str:
    """Build the prompt for final answer generation."""
    return (
        "Based on the following retrieved documents, answer the question comprehensively.\n"
        "If the documents don't contain enough information to answer the question, say so.\n\n"
        f"Question: {question}\n\n"
        "Retrieved Documents:\n"
        f"{context}\n\n"
        "Provide a detailed answer based on the information in the documents.\n"
        "Reference specific sections or documents when relevant.\n"
    )
