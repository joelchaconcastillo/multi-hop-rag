"""Configuration management for the multi-hop RAG system."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"

    # HuggingFace Configuration
    huggingface_api_key: str
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB Configuration
    chroma_persist_directory: Path = Path("./chroma_db")
    chroma_collection_name: str = "multi_hop_rag"

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RAG Configuration
    max_hops: int = 3
    top_k_retrieval: int = 5

    # Categorization and reranking
    rerank_similarity_weight: float = 0.7
    rerank_fetch_multiplier: int = 3
    category_smoothing: float = 0.05

    # Logging
    log_level: str = "INFO"

    def __init__(self, **kwargs):
        """Initialize settings and validate API keys."""
        super().__init__(**kwargs)
        
        # Create persist directory if it doesn't exist
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)

    def validate_api_keys(self) -> None:
        """Validate that required API keys are set."""
        if not self.openai_api_key or self.openai_api_key == "your-openai-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY must be set in environment variables or .env file"
            )
        if not self.huggingface_api_key or self.huggingface_api_key == "your-huggingface-api-key-here":
            raise ValueError(
                "HUGGINGFACE_API_KEY must be set in environment variables or .env file"
            )


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
