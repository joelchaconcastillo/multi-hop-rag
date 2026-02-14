"""Tests for configuration management."""

import os
import pytest
from pathlib import Path
from multi_hop_rag.config import Settings


def test_settings_with_env_vars(monkeypatch):
    """Test settings loaded from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-hf-key")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    
    settings = Settings()
    
    assert settings.openai_api_key == "test-openai-key"
    assert settings.huggingface_api_key == "test-hf-key"
    assert settings.chunk_size == 500


def test_settings_defaults(monkeypatch):
    """Test default settings values."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
    
    settings = Settings()
    
    assert settings.openai_model == "gpt-4-turbo-preview"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.max_hops == 3
    assert settings.top_k_retrieval == 5


def test_settings_chroma_directory_creation(monkeypatch, tmp_path):
    """Test that ChromaDB directory is created."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
    
    test_dir = tmp_path / "test_chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(test_dir))
    
    settings = Settings()
    
    assert settings.chroma_persist_directory.exists()
    assert settings.chroma_persist_directory == test_dir


def test_validate_api_keys_failure():
    """Test API key validation failure."""
    settings = Settings(
        openai_api_key="your-openai-api-key-here",
        huggingface_api_key="valid-key"
    )
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
        settings.validate_api_keys()


def test_validate_api_keys_success():
    """Test API key validation success."""
    settings = Settings(
        openai_api_key="valid-openai-key",
        huggingface_api_key="valid-hf-key"
    )
    
    # Should not raise
    settings.validate_api_keys()
