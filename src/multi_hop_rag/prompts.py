"""Prompt templates for the multi-hop RAG system."""

from __future__ import annotations

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a helpful assistant that decomposes complex questions into simpler sub-queries."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on provided documents. "
    "Always cite your sources."
)

SYNTHESIZE_SYSTEM_PROMPT = (
    "You are a helpful assistant that synthesizes key terms, definitions, "
    "requirements, and exceptions from retrieved documents."
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


def build_synthesis_prompt(question: str, context: str) -> str:
    """Build the prompt for hop 3 synthesis."""
    return (
        "Synthesize the most important terms, definitions, and requirements from the "
        "retrieved documents. Focus on what is essential for answering the question. "
        "Use short bullet points when possible.\n\n"
        f"Question: {question}\n\n"
        "Retrieved Documents:\n"
        f"{context}\n\n"
        "Provide a concise synthesis.\n"
    )
