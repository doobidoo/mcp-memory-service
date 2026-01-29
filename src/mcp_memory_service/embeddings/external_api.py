"""External embedding API adapter for OpenAI-compatible endpoints.

This module provides an adapter that mimics the SentenceTransformer interface
but calls an external OpenAI-compatible embedding API instead of running
models locally. This is useful for:

- Using hosted embedding services (OpenAI, Cohere, etc.)
- Using local inference servers (vLLM, Ollama, text-embeddings-inference)
- Sharing embedding computation across multiple MCP instances
- Using embedding models that aren't available in SentenceTransformers

Configuration:
    Set these environment variables:
    - MCP_EXTERNAL_EMBEDDING_URL: API endpoint (e.g., http://localhost:8890/v1/embeddings)
    - MCP_EXTERNAL_EMBEDDING_MODEL: Model name (e.g., nomic-embed-text, text-embedding-3-small)
    - MCP_EXTERNAL_EMBEDDING_API_KEY: Optional API key for authenticated endpoints

Example usage with vLLM:
    # Start vLLM with embedding model
    vllm serve nomic-ai/nomic-embed-text-v1.5 --port 8890

    # Configure MCP Memory Service
    export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings
    export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

Example usage with Ollama:
    # Pull and run embedding model
    ollama pull nomic-embed-text

    # Configure MCP Memory Service (use /v1/embeddings for OpenAI compatibility)
    export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:11434/v1/embeddings
    export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
"""

import logging
import os
from typing import List, Optional, Union

import numpy as np
import requests

logger = logging.getLogger(__name__)


class ExternalEmbeddingModel:
    """Adapter that mimics SentenceTransformer interface but calls external API.

    This class provides compatibility with code expecting a SentenceTransformer
    model by implementing the same interface (encode, get_sentence_embedding_dimension).

    Attributes:
        api_url: The embedding API endpoint URL.
        model_name: The model identifier to pass to the API.
        embedding_dimension: The dimensionality of the embeddings (auto-detected).
        api_key: Optional API key for authenticated endpoints.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """Initialize the external embedding model adapter.

        Args:
            api_url: The embedding API endpoint. Defaults to MCP_EXTERNAL_EMBEDDING_URL
                     environment variable or http://localhost:8890/v1/embeddings.
            model_name: The model to use. Defaults to MCP_EXTERNAL_EMBEDDING_MODEL
                        environment variable or 'nomic-embed-text'.
            api_key: API key for authenticated endpoints. Defaults to
                     MCP_EXTERNAL_EMBEDDING_API_KEY environment variable.
            timeout: Request timeout in seconds. Default 30.

        Raises:
            ConnectionError: If the API is unreachable or returns an error.
        """
        self.api_url = api_url or os.getenv(
            'MCP_EXTERNAL_EMBEDDING_URL',
            'http://localhost:8890/v1/embeddings'
        )
        self.model_name = model_name or os.getenv(
            'MCP_EXTERNAL_EMBEDDING_MODEL',
            'nomic-embed-text'
        )
        self.api_key = api_key or os.getenv('MCP_EXTERNAL_EMBEDDING_API_KEY')
        self.timeout = timeout
        self.embedding_dimension = 768  # Default, will be updated on connection

        # Verify connection and detect embedding dimension
        self._verify_connection()

    def _get_headers(self) -> dict:
        """Get request headers including optional authentication."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _verify_connection(self) -> None:
        """Verify the external API is reachable and detect embedding dimension.

        Raises:
            ConnectionError: If the API is unreachable or returns an error.
        """
        try:
            response = requests.post(
                self.api_url,
                headers=self._get_headers(),
                json={'input': 'test', 'model': self.model_name},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    self.embedding_dimension = len(data['data'][0]['embedding'])
                    logger.info(
                        f"External embedding API connected: {self.api_url}, "
                        f"model: {self.model_name}, dims: {self.embedding_dimension}"
                    )
                    return
                raise ConnectionError("API response missing embedding data")

            # Try to get error message from response
            try:
                error_json = response.json()
                # Handle different error formats from various APIs
                error_detail = (
                    error_json.get('error', {}).get('message') or
                    error_json.get('detail') or
                    error_json.get('message') or
                    response.text
                )
            except requests.exceptions.JSONDecodeError:
                error_detail = response.text
            except Exception as e:
                error_detail = f"{response.text} (parse error: {e})"

            raise ConnectionError(
                f"API returned status {response.status_code}: {error_detail}"
            )

        except requests.exceptions.Timeout:
            raise ConnectionError(
                f"Connection to {self.api_url} timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to external embedding API at {self.api_url}: {e}"
            )

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings using external API.

        This method mimics the SentenceTransformer.encode() interface for
        compatibility with existing code.

        Args:
            sentences: Single sentence or list of sentences to embed.
            convert_to_numpy: Whether to return numpy array (default True).
            batch_size: Batch size for processing (for large inputs).
            show_progress_bar: Ignored (for interface compatibility).
            **kwargs: Additional arguments (ignored, for compatibility).

        Returns:
            numpy.ndarray of shape (n_sentences, embedding_dimension) if
            convert_to_numpy is True, otherwise list of lists.

        Raises:
            RuntimeError: If the API request fails.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        # Process in batches for large inputs
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            try:
                response = requests.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json={'input': batch, 'model': self.model_name},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                # Extract embeddings in order (API may return out of order)
                batch_embeddings = [None] * len(batch)
                seen_indices = set()
                for item in data['data']:
                    if 'index' not in item:
                        raise RuntimeError(f"API response from {self.api_url} is missing the 'index' field.")

                    idx = item['index']
                    if not (0 <= idx < len(batch)):
                        logger.warning(f"API returned out-of-bounds index {idx} for batch size {len(batch)}")
                        continue

                    if idx in seen_indices:
                        raise RuntimeError(f"API returned duplicate index {idx} in the same batch.")
                    seen_indices.add(idx)

                    batch_embeddings[idx] = item['embedding']

                # Verify all embeddings were returned
                if None in batch_embeddings:
                    missing_indices = [i for i, emb in enumerate(batch_embeddings) if emb is None]
                    raise RuntimeError(f"API did not return embeddings for indices: {missing_indices}")

                all_embeddings.extend(batch_embeddings)

            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Failed to generate embedding via external API: {e}"
                )
            except (KeyError, IndexError) as e:
                raise RuntimeError(
                    f"Unexpected API response format: {e}"
                )

        if convert_to_numpy:
            return np.array(all_embeddings, dtype=np.float32)
        return all_embeddings

    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension.

        Returns:
            The dimensionality of the embeddings produced by this model.
        """
        return self.embedding_dimension


def get_external_embedding_model(
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> ExternalEmbeddingModel:
    """Factory function to create external embedding model.

    Args:
        api_url: The embedding API endpoint URL.
        model_name: The model identifier to pass to the API.
        api_key: Optional API key for authenticated endpoints.

    Returns:
        ExternalEmbeddingModel instance configured with the given parameters.

    Raises:
        ConnectionError: If the API is unreachable.
    """
    return ExternalEmbeddingModel(
        api_url=api_url,
        model_name=model_name,
        api_key=api_key
    )
