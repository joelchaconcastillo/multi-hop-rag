"""Tests for the pipeline integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from multi_hop_rag import MultiHopRAGPipeline, Settings


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock settings for testing."""
    return Settings(
        openai_api_key="test-openai-key",
        huggingface_api_key="test-hf-key",
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="test_collection",
        chunk_size=500,
        chunk_overlap=100,
    )


@patch("multi_hop_rag.pipeline.HuggingFaceEmbedder")
@patch("multi_hop_rag.pipeline.ChromaStore")
@patch("multi_hop_rag.pipeline.MultiHopRAGGraph")
def test_pipeline_initialization(mock_graph, mock_store, mock_embedder, mock_settings):
    """Test pipeline initialization."""
    pipeline = MultiHopRAGPipeline(settings=mock_settings)
    
    assert pipeline.settings == mock_settings
    assert pipeline.chunker is not None
    assert mock_embedder.called
    assert mock_store.called
    assert mock_graph.called


@patch("multi_hop_rag.pipeline.HuggingFaceEmbedder")
@patch("multi_hop_rag.pipeline.ChromaStore")
@patch("multi_hop_rag.pipeline.MultiHopRAGGraph")
def test_index_text(mock_graph, mock_store, mock_embedder, mock_settings):
    """Test indexing text."""
    # Setup mocks
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_embedder.return_value = mock_embedder_instance
    
    mock_store_instance = MagicMock()
    mock_store.return_value = mock_store_instance
    
    pipeline = MultiHopRAGPipeline(settings=mock_settings)
    
    # Index text
    text = "Test content for indexing."
    chunks_count = pipeline.index_text(text)
    
    assert chunks_count > 0
    assert mock_embedder_instance.embed_texts.called
    assert mock_store_instance.add_documents.called


@patch("multi_hop_rag.pipeline.HuggingFaceEmbedder")
@patch("multi_hop_rag.pipeline.ChromaStore")
@patch("multi_hop_rag.pipeline.MultiHopRAGGraph")
def test_query(mock_graph, mock_store, mock_embedder, mock_settings):
    """Test querying the pipeline."""
    # Setup mocks
    mock_graph_instance = MagicMock()
    mock_graph_instance.query.return_value = {
        "question": "test",
        "answer": "test answer",
        "total_documents_retrieved": 5
    }
    mock_graph.return_value = mock_graph_instance
    
    pipeline = MultiHopRAGPipeline(settings=mock_settings)
    
    # Query
    response = pipeline.query("test question")
    
    assert response["answer"] == "test answer"
    assert mock_graph_instance.query.called


@patch("multi_hop_rag.pipeline.HuggingFaceEmbedder")
@patch("multi_hop_rag.pipeline.ChromaStore")
@patch("multi_hop_rag.pipeline.MultiHopRAGGraph")
def test_get_stats(mock_graph, mock_store, mock_embedder, mock_settings):
    """Test getting pipeline statistics."""
    mock_store_instance = MagicMock()
    mock_store_instance.count.return_value = 10
    mock_store.return_value = mock_store_instance
    
    pipeline = MultiHopRAGPipeline(settings=mock_settings)
    
    stats = pipeline.get_stats()
    
    assert stats["total_documents"] == 10
    assert stats["collection_name"] == "test_collection"
    assert stats["chunk_size"] == 500
    assert stats["max_hops"] == 3
