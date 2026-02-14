"""HuggingFace embedding service using API calls."""

import logging
from typing import List, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    """Generate embeddings using HuggingFace Inference API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the HuggingFace embedder.

        Args:
            api_key: HuggingFace API key
            model_name: Name of the embedding model to use
            batch_size: Number of texts to embed in each batch
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize the HuggingFace client
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Embedded batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}")

        logger.info(f"Generated embeddings for {len(texts)} texts")
        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        for attempt in range(self.max_retries):
            try:
                embeddings = []
                for text in texts:
                    # Use feature extraction endpoint
                    embedding = self.client.feature_extraction(
                        text=text,
                        model=self.model_name
                    )
                    
                    # Handle different response formats
                    if isinstance(embedding, list):
                        # If it's already a list, check if it's a nested list
                        if embedding and isinstance(embedding[0], list):
                            # Take the mean of token embeddings or first embedding
                            embedding = self._mean_pooling(embedding)
                        embeddings.append(embedding)
                    else:
                        raise ValueError(f"Unexpected embedding format: {type(embedding)}")
                
                return embeddings
                
            except Exception as e:
                logger.warning(
                    f"Embedding attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                    raise

    def _mean_pooling(self, token_embeddings: List[List[float]]) -> List[float]:
        """
        Apply mean pooling to token embeddings.

        Args:
            token_embeddings: List of token embedding vectors

        Returns:
            Single averaged embedding vector
        """
        if not token_embeddings:
            raise ValueError("Empty token embeddings")
        
        # Calculate mean across all tokens
        embedding_dim = len(token_embeddings[0])
        mean_embedding = [0.0] * embedding_dim
        
        for token_emb in token_embeddings:
            for i, val in enumerate(token_emb):
                mean_embedding[i] += val
        
        # Divide by number of tokens
        mean_embedding = [val / len(token_embeddings) for val in mean_embedding]
        
        return mean_embedding

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings from this model.

        Returns:
            Embedding dimension
        """
        # Generate a test embedding to determine dimension
        test_embedding = self.embed_query("test")
        return len(test_embedding)
