"""
ONNX-based embedding generation for MCP Memory Service.
Provides PyTorch-free embedding generation using ONNX Runtime.
Based on ONNXMiniLM_L6_V2 implementation.
"""

import hashlib
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

# Try to import tokenizers
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logger.warning("Tokenizers not available. Install with: pip install tokenizers")


def _verify_sha256(fname: str, expected_sha256: str) -> bool:
    """Verify SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(fname, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_sha256


class ONNXEmbeddingModel:
    """
    ONNX-based embedding model that provides PyTorch-free embeddings.
    Compatible with all-MiniLM-L6-v2 model.
    """
    
    MODEL_NAME = "all-MiniLM-L6-v2"
    DOWNLOAD_PATH = Path.home() / ".cache" / "mcp_memory" / "onnx_models" / MODEL_NAME
    EXTRACTED_FOLDER_NAME = "onnx"
    ARCHIVE_FILENAME = "onnx.tar.gz"
    MODEL_DOWNLOAD_URL = (
        "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
    )
    _MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", preferred_providers: Optional[List[str]] = None):
        """
        Initialize ONNX embedding model.
        
        Args:
            model_name: Name of the model (currently only all-MiniLM-L6-v2 supported)
            preferred_providers: List of ONNX execution providers in order of preference
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is required but not installed. Install with: pip install onnxruntime")
        
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("Tokenizers is required but not installed. Install with: pip install tokenizers")
        
        self.model_name = model_name
        self._preferred_providers = preferred_providers or ['CPUExecutionProvider']
        self._model = None
        self._tokenizer = None
        
        # Download model if needed
        self._download_model_if_needed()
        
        # Initialize the model
        self._init_model()
    
    def _download_model_if_needed(self):
        """Download and extract ONNX model if not present."""
        if not self.DOWNLOAD_PATH.exists():
            self.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
        
        archive_path = self.DOWNLOAD_PATH / self.ARCHIVE_FILENAME
        extracted_path = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME
        
        # Check if model is already extracted
        if extracted_path.exists() and (extracted_path / "model.onnx").exists():
            logger.info(f"ONNX model already available at {extracted_path}")
            return
        
        # Download if not present or invalid
        if not archive_path.exists() or not _verify_sha256(str(archive_path), self._MODEL_SHA256):
            logger.info(f"Downloading ONNX model from {self.MODEL_DOWNLOAD_URL}")
            try:
                import httpx
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(self.MODEL_DOWNLOAD_URL)
                    response.raise_for_status()
                    with open(archive_path, "wb") as f:
                        f.write(response.content)
                logger.info(f"Model downloaded to {archive_path}")
            except Exception as e:
                logger.error(f"Failed to download ONNX model: {e}")
                raise RuntimeError(f"Could not download ONNX model: {e}")
        
        # Extract the archive
        logger.info(f"Extracting model to {extracted_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.DOWNLOAD_PATH)
        
        # Verify extraction
        if not (extracted_path / "model.onnx").exists():
            raise RuntimeError(f"Model extraction failed - model.onnx not found in {extracted_path}")
        
        logger.info("ONNX model ready for use")
    
    def _init_model(self):
        """Initialize ONNX model and tokenizer."""
        model_path = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "model.onnx"
        tokenizer_path = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "tokenizer.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        # Initialize ONNX session
        logger.info(f"Loading ONNX model with providers: {self._preferred_providers}")
        self._model = ort.InferenceSession(
            str(model_path),
            providers=self._preferred_providers
        )
        
        # Initialize tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Get model info
        self.embedding_dimension = self._model.get_outputs()[0].shape[-1]
        logger.info(f"ONNX model loaded. Embedding dimension: {self.embedding_dimension}")
    
    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts using ONNX model.
        
        Args:
            texts: Single text or list of texts to encode
            convert_to_numpy: Whether to return numpy array (always True for compatibility)
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        encoded = self._tokenizer.encode_batch(texts)
        
        # Prepare inputs for ONNX model
        max_length = max(len(enc.ids) for enc in encoded)
        
        # Pad sequences
        input_ids = np.zeros((len(texts), max_length), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_length), dtype=np.int64)
        token_type_ids = np.zeros((len(texts), max_length), dtype=np.int64)
        
        for i, enc in enumerate(encoded):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask
            token_type_ids[i, :length] = enc.type_ids
        
        # Run inference
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        
        outputs = self._model.run(None, ort_inputs)
        
        # Extract embeddings (using mean pooling)
        last_hidden_states = outputs[0]
        
        # Mean pooling with attention mask
        input_mask_expanded = attention_mask[..., np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(last_hidden_states * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    @property
    def device(self):
        """Return device info for compatibility."""
        return "cpu"  # ONNX runtime handles device selection internally


def get_onnx_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[ONNXEmbeddingModel]:
    """
    Get ONNX embedding model if available.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        ONNXEmbeddingModel instance or None if ONNX is not available
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX Runtime not available")
        return None
    
    if not TOKENIZERS_AVAILABLE:
        logger.warning("Tokenizers not available")
        return None
    
    try:
        # Detect best available providers
        available_providers = ort.get_available_providers()
        preferred_providers = []
        
        # Prefer GPU providers if available
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        if 'DirectMLExecutionProvider' in available_providers:
            preferred_providers.append('DirectMLExecutionProvider')
        if 'CoreMLExecutionProvider' in available_providers:
            preferred_providers.append('CoreMLExecutionProvider')
        
        # Always include CPU as fallback
        preferred_providers.append('CPUExecutionProvider')
        
        logger.info(f"Creating ONNX model with providers: {preferred_providers}")
        return ONNXEmbeddingModel(model_name, preferred_providers)
    
    except Exception as e:
        logger.error(f"Failed to create ONNX embedding model: {e}")
        return None