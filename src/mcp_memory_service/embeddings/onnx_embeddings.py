"""
ONNX-based embedding generation for MCP Memory Service.
Provides PyTorch-free embedding generation using ONNX Runtime.
Based on ONNXMiniLM_L6_V2 implementation.
"""

import hashlib
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ..compat import _sanitize_log_value

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


def _safe_tar_extract(tar: tarfile.TarFile, path) -> None:
    """Safely extract a tar archive, preventing path traversal attacks."""
    abs_path = os.path.realpath(str(path))
    members = tar.getmembers()
    for member in members:
        member_path = os.path.realpath(os.path.join(abs_path, member.name))
        if not member_path.startswith(abs_path + os.sep) and member_path != abs_path:
            raise ValueError(f"Attempted path traversal in tar file: {member.name}")
    # Extract each member individually after validation to avoid tarslip
    for member in members:
        tar.extract(member, path, set_attrs=False)


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
            model_name: Name of the model. 'all-MiniLM-L6-v2' uses the bundled
                Chroma S3 archive. Any other name is fetched from the Hugging
                Face Hub as 'onnx-community/<model>-ONNX' (or the exact repo set
                via MCP_ONNX_MODEL_REPO), enabling e.g. multilingual models
                without torch. See issue: ONNX honor MCP_EMBEDDING_MODEL.
            preferred_providers: List of ONNX execution providers in order of preference
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is required but not installed. Install with: pip install onnxruntime")
        
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("Tokenizers is required but not installed. Install with: pip install tokenizers")
        
        self.model_name = model_name or self.MODEL_NAME
        self._preferred_providers = preferred_providers or ['CPUExecutionProvider']
        self._model = None
        self._tokenizer = None

        # Decide the loading strategy. The default model keeps the original
        # S3 tar.gz path (fully backward compatible). A non-default model is
        # resolved from the Hugging Face Hub instead.
        base = self.model_name.split('/')[-1]
        self._is_default_model = (base == self.MODEL_NAME)
        if self._is_default_model:
            self._hf_repo = None
            self._model_dir = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME
        else:
            # onnx-community publishes pre-exported ONNX for common sentence
            # transformers. Allow an explicit override for other repos/layouts.
            self._hf_repo = os.environ.get(
                'MCP_ONNX_MODEL_REPO', f"onnx-community/{base}-ONNX"
            )
            self._model_dir = (
                Path.home() / ".cache" / "mcp_memory" / "onnx_models" / base
            )

        # Download model if needed
        self._download_model_if_needed()
        
        # Initialize the model
        self._init_model()
    
    def _download_model_if_needed(self):
        """Download and extract ONNX model if not present."""
        # Custom (non-default) model: resolve from the Hugging Face Hub.
        if not self._is_default_model:
            self._download_from_hf_if_needed()
            return

        if not self.DOWNLOAD_PATH.exists():
            self.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
        
        archive_path = self.DOWNLOAD_PATH / self.ARCHIVE_FILENAME
        extracted_path = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME
        
        # Check if model is already extracted
        if extracted_path.exists() and (extracted_path / "model.onnx").exists():
            logger.info(f"ONNX model already available at {extracted_path}")
            return

        # Offline guard: allow deployments (and the test harness) to opt out of the
        # network fetch. When downloads are disabled and the model is not already
        # cached, raise so the caller can fall back to another backend instead of
        # blocking on a ~80MB download. See issue #162.
        if os.environ.get('MCP_MEMORY_ONNX_ALLOW_DOWNLOAD', '1').lower() in ('0', 'false', 'no'):
            raise RuntimeError(
                "ONNX model is not cached and downloads are disabled "
                "(MCP_MEMORY_ONNX_ALLOW_DOWNLOAD=0)."
            )

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
            _safe_tar_extract(tar, self.DOWNLOAD_PATH)
        
        # Verify extraction
        if not (extracted_path / "model.onnx").exists():
            raise RuntimeError(f"Model extraction failed - model.onnx not found in {extracted_path}")
        
        logger.info("ONNX model ready for use")
    
    def _download_from_hf_if_needed(self):
        """Fetch a pre-exported ONNX model from the Hugging Face Hub.

        Downloads only the ONNX weights + tokenizer (no torch) into
        ``self._model_dir``. Layout on the Hub is typically ``onnx/model.onnx``
        (quantized variants exist) plus ``tokenizer.json``/``config.json`` at the
        repo root. Resolved files are located later by ``_init_model``.
        """
        self._model_dir.mkdir(parents=True, exist_ok=True)
        # Already present?
        if self._find_onnx_file() and (self._model_dir / "tokenizer.json").exists():
            logger.info(f"ONNX model already available at {self._model_dir}")
            return

        if os.environ.get('MCP_MEMORY_ONNX_ALLOW_DOWNLOAD', '1').lower() in ('0', 'false', 'no'):
            raise RuntimeError(
                "ONNX model is not cached and downloads are disabled "
                "(MCP_MEMORY_ONNX_ALLOW_DOWNLOAD=0)."
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to fetch a custom ONNX embedding "
                "model. Install with: pip install huggingface_hub"
            ) from e

        logger.info(f"Downloading ONNX model '{self.model_name}' from HF repo {self._hf_repo}")
        try:
            snapshot_download(
                repo_id=self._hf_repo,
                local_dir=str(self._model_dir),
                allow_patterns=["model.onnx", "onnx/model.onnx", "tokenizer.json",
                                "tokenizer_config.json", "config.json",
                                "special_tokens_map.json"],
            )
        except Exception as e:
            logger.error(f"Failed to download ONNX model from {self._hf_repo}: {e}")
            raise RuntimeError(f"Could not download ONNX model {self._hf_repo}: {e}")

        if not self._find_onnx_file():
            raise RuntimeError(f"No .onnx file found in downloaded repo {self._hf_repo}")
        logger.info("ONNX model ready for use")

    def _resolve_model_path(self):
        """Return the model.onnx path for the active model (default or custom)."""
        if self._is_default_model:
            return self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "model.onnx"
        return self._find_onnx_file()

    def _find_onnx_file(self):
        """Locate the full-precision model.onnx within the custom model dir.

        Prefers the canonical ``model.onnx`` (repo root or ``onnx/``). Quantized
        variants (model_quantized/int8/uint8/...) are deliberately NOT auto-
        selected, as they change the embedding numerics; callers that want them
        should point MCP_ONNX_MODEL_REPO at a repo whose canonical file is that
        variant.
        """
        for cand in (self._model_dir / "model.onnx",
                     self._model_dir / "onnx" / "model.onnx"):
            if cand.exists():
                return cand
        return None

    def _init_model(self):
        """Initialize ONNX model and tokenizer."""
        if self._is_default_model:
            model_path = self._resolve_model_path()
            tokenizer_path = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "tokenizer.json"
        else:
            model_path = self._resolve_model_path()
            tokenizer_path = self._model_dir / "tokenizer.json"

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found for '{self.model_name}'")

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        # Initialize ONNX session
        logger.info(
            "Loading ONNX model with providers: %s",
            _sanitize_log_value(self._preferred_providers),
        )
        self._model = ort.InferenceSession(
            str(model_path),
            providers=self._preferred_providers
        )
        
        # Initialize tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Get model info
        self.embedding_dimension = self._model.get_outputs()[0].shape[-1]
        # Not every exported model takes token_type_ids (some drop it). Record
        # the accepted input names so encode() only feeds supported tensors.
        self._input_names = {i.name for i in self._model.get_inputs()}
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
        
        # Run inference — only feed inputs the model actually declares.
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        accepted = getattr(self, "_input_names", None)
        if accepted is not None:
            ort_inputs = {k: v for k, v in ort_inputs.items() if k in accepted}
        
        session = self._model
        try:
            outputs = session.run(None, ort_inputs)
        except Exception as exc:
            if "CoreMLExecutionProvider" not in session.get_providers():
                raise
            # CPU in the provider list handles unsupported nodes, not CoreML
            # runtime failures. Rebuild without CoreML and retry this batch once.
            logger.warning(
                "CoreML inference failed; retrying with CPUExecutionProvider: %s",
                _sanitize_log_value(exc),
            )
            model_path = self._resolve_model_path()
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._model = session
            self._preferred_providers = ["CPUExecutionProvider"]
            outputs = session.run(None, ort_inputs)
        
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


def _get_preferred_providers() -> list[str]:
    """Use an explicit provider pin, or prefer available accelerators."""
    available = ort.get_available_providers()
    configured = os.environ.get("MCP_MEMORY_ONNX_PROVIDERS", "").strip()
    if configured:
        providers = [provider.strip() for provider in configured.split(",")]
        if set(providers) - set(available):
            raise ValueError(
                "MCP_MEMORY_ONNX_PROVIDERS must contain comma-separated available "
                f"provider names. Available: {available}"
            )
        return providers
    return [
        provider
        for provider in (
            "CUDAExecutionProvider",
            "DirectMLExecutionProvider",
            "CoreMLExecutionProvider",
        )
        if provider in available
    ] + ["CPUExecutionProvider"]


def get_onnx_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[ONNXEmbeddingModel]:
    """
    Get ONNX embedding model if available.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        ONNXEmbeddingModel instance or None if ONNX is not available

    Raises:
        ValueError: If MCP_MEMORY_ONNX_PROVIDERS contains unavailable provider names.
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX Runtime not available")
        return None
    
    if not TOKENIZERS_AVAILABLE:
        logger.warning("Tokenizers not available")
        return None
    
    preferred_providers = _get_preferred_providers()
    try:
        logger.info(
            "Creating ONNX model with providers: %s",
            _sanitize_log_value(preferred_providers),
        )
        return ONNXEmbeddingModel(model_name, preferred_providers)
    
    except Exception as e:
        logger.error("Failed to create ONNX embedding model: %s", _sanitize_log_value(e))
        return None
