"""
ONNX-based cross-encoder model for quality scoring.
Uses ms-marco-MiniLM-L-6-v2 model for relevance scoring.

Exports the model from transformers to ONNX format on first use.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Setup offline mode before importing transformers
from ..offline_mode import setup_offline_mode
setup_offline_mode()

logger = logging.getLogger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

# Try to import transformers for model export
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoConfig
    import torch
    from torch import nn
    from huggingface_hub import PyTorchModelHubMixin
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch huggingface-hub")


# Custom model class for NVIDIA DeBERTa quality classifier
if TRANSFORMERS_AVAILABLE:
    class QualityModel(nn.Module, PyTorchModelHubMixin):
        """Custom model class for NVIDIA DeBERTa quality classifier."""
        def __init__(self, config):
            super(QualityModel, self).__init__()

            # Load base model with snapshot path detection
            base_model_name = config["base_model"]
            cache_dir = Path.home() / '.cache/huggingface/hub' / f'models--{base_model_name.replace("/", "--")}/snapshots'
            if cache_dir.exists():
                snapshots = list(cache_dir.glob('*'))
                if snapshots:
                    base_model_path = str(snapshots[0])
                    logger.info(f"Loading base model from cached snapshot: {base_model_path}")
                    self.model = AutoModel.from_pretrained(base_model_path)
                else:
                    self.model = AutoModel.from_pretrained(base_model_name)
            else:
                self.model = AutoModel.from_pretrained(base_model_name)

            self.dropout = nn.Dropout(config["fc_dropout"])
            self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

        def forward(self, input_ids, attention_mask):
            features = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            dropped = self.dropout(features)
            outputs = self.fc(dropped)
            # Return raw logits (not softmax) - softmax will be applied during inference
            return outputs[:, 0, :]


class ONNXRankerModel:
    """
    ONNX-based model for quality scoring with multi-model support.
    Supports both classifier (DeBERTa) and cross-encoder (MS-MARCO) types.

    On first use, exports the model from transformers to ONNX format.
    """

    ONNX_MODEL_FILE = "model.onnx"

    def __init__(self, model_name: str = "nvidia-quality-classifier-deberta", device: str = "auto", preferred_providers: Optional[list] = None):
        """
        Initialize ONNX ranker model.

        Args:
            model_name: Model to use (default: 'nvidia-quality-classifier-deberta')
            device: Device to use ('auto', 'cpu', 'cuda', 'mps', 'directml')
            preferred_providers: List of ONNX execution providers in order of preference
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is required but not installed. Install with: pip install onnxruntime")

        # Import config validation here to avoid circular imports
        from .config import validate_model_selection

        self.model_name = model_name
        self.model_config = validate_model_selection(model_name)
        self.MODEL_PATH = Path.home() / ".cache" / "mcp_memory" / "onnx_models" / model_name

        self.device = device
        self._preferred_providers = preferred_providers or self._detect_providers(device)
        self._model = None
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()  # Protects tokenizer state mutations in batch ops

        # Check if ONNX model already exists (no transformers needed!)
        onnx_path = self.MODEL_PATH / self.ONNX_MODEL_FILE
        if not onnx_path.exists():
            # Only require transformers if we need to export
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    f"ONNX model not found at {onnx_path}. "
                    "Either download pre-exported model or install transformers+torch for export."
                )
            # Export model to ONNX
            self._ensure_onnx_model()

        # Initialize the model (works without transformers!)
        self._init_model()

    def _detect_providers(self, device: str) -> list:
        """Detect best available ONNX execution providers based on device preference."""
        available_providers = ort.get_available_providers()
        preferred_providers = []

        if device == "auto":
            # Prefer GPU providers if available
            if 'CUDAExecutionProvider' in available_providers:
                preferred_providers.append('CUDAExecutionProvider')
            if 'DirectMLExecutionProvider' in available_providers:
                preferred_providers.append('DirectMLExecutionProvider')
            if 'CoreMLExecutionProvider' in available_providers:
                preferred_providers.append('CoreMLExecutionProvider')
        elif device == "cuda" and 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        elif device == "mps" and 'CoreMLExecutionProvider' in available_providers:
            preferred_providers.append('CoreMLExecutionProvider')
        elif device == "directml" and 'DirectMLExecutionProvider' in available_providers:
            preferred_providers.append('DirectMLExecutionProvider')

        # Always include CPU as fallback
        preferred_providers.append('CPUExecutionProvider')

        return preferred_providers

    def _ensure_onnx_model(self):
        """Export transformers model to ONNX format if not already present."""
        onnx_path = self.MODEL_PATH / self.ONNX_MODEL_FILE

        if onnx_path.exists():
            logger.info(f"ONNX model already available at {onnx_path}")
            return

        # Create directory
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)

        hf_model_name = self.model_config['hf_name']
        logger.info(f"Exporting {hf_model_name} to ONNX format...")

        # Helper function to find cached snapshot path
        def get_snapshot_path(model_id):
            """Find the snapshot path for a cached HuggingFace model."""
            cache_dir = Path.home() / '.cache/huggingface/hub' / f'models--{model_id.replace("/", "--")}/snapshots'
            if cache_dir.exists():
                snapshots = list(cache_dir.glob('*'))
                if snapshots:
                    return str(snapshots[0])
            return None

        # Load transformers model (try local_files_only first for offline mode)
        try:
            logger.info(f"Attempting to load tokenizer for {hf_model_name} (local_files_only=True)...")

            # Try to find cached snapshot path first
            snapshot_path = get_snapshot_path(hf_model_name)
            load_path = snapshot_path if snapshot_path else hf_model_name

            if snapshot_path:
                logger.info(f"Found cached snapshot at: {snapshot_path}")
                tokenizer = AutoTokenizer.from_pretrained(load_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name, local_files_only=True)
            logger.info("✓ Tokenizer loaded successfully")

            # Use custom QualityModel class for NVIDIA DeBERTa
            if self.model_config['type'] == 'classifier' and 'nvidia' in hf_model_name.lower():
                logger.info("Loading NVIDIA DeBERTa with custom QualityModel class...")
                # Load config and create QualityModel instance
                logger.info("Loading config...")
                config_dict = AutoConfig.from_pretrained(load_path).to_dict()
                logger.info(f"✓ Config loaded: {list(config_dict.keys())[:5]}...")
                model = QualityModel(config=config_dict)
                logger.info("✓ Model instance created")
                # Load weights
                import safetensors.torch
                logger.info("Loading model weights from safetensors...")
                if snapshot_path:
                    model_file = Path(snapshot_path) / "model.safetensors"
                else:
                    from huggingface_hub import hf_hub_download
                    model_file = hf_hub_download(repo_id=hf_model_name, filename="model.safetensors", local_files_only=True)
                logger.info(f"✓ Model path: {model_file}")
                state_dict = safetensors.torch.load_file(str(model_file))
                model.load_state_dict(state_dict, strict=False)
                logger.info("✓ Weights loaded successfully")
            else:
                if snapshot_path:
                    model = AutoModelForSequenceClassification.from_pretrained(load_path)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, local_files_only=True)
        except Exception as e:
            # Fall back to online mode if not cached
            logger.info(f"Local loading failed: {e}")
            logger.info("Falling back to online mode...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

            # Use custom QualityModel class for NVIDIA DeBERTa
            if self.model_config['type'] == 'classifier' and 'nvidia' in hf_model_name.lower():
                # Load config and create QualityModel instance
                from huggingface_hub import hf_hub_download
                config_dict = AutoConfig.from_pretrained(hf_model_name).to_dict()
                model = QualityModel(config=config_dict)
                # Load weights
                import safetensors.torch
                model_path = hf_hub_download(repo_id=hf_model_name, filename="model.safetensors")
                state_dict = safetensors.torch.load_file(model_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)

        model.eval()
        logger.info("Casting model to float32 for ONNX export compatibility.")
        model.float()  # Cast to float32 to prevent mixed fp16/fp32 ONNX export errors

        # Create dummy inputs based on model type
        if self.model_config['type'] == 'classifier':
            # Classifier (DeBERTa): single text input
            dummy_text = "This is a sample memory content for quality evaluation."
            inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
            input_names = ["input_ids", "attention_mask"]
            model_inputs = (inputs["input_ids"], inputs["attention_mask"])

        elif self.model_config['type'] == 'cross-encoder':
            # Cross-encoder (MS-MARCO): query [SEP] document format
            dummy_text = "query [SEP] document"
            inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            model_inputs = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

        else:
            raise ValueError(f"Unsupported model type: {self.model_config['type']}")

        # Export to ONNX
        with torch.no_grad():
            dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
            dynamic_axes["logits"] = {0: "batch"}

            torch.onnx.export(
                model,
                model_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True
            )

        # Save tokenizer config for loading
        tokenizer.save_pretrained(str(self.MODEL_PATH))

        logger.info(f"ONNX model exported to {onnx_path}")
        logger.info(f"Model type: {self.model_config['type']}, Inputs: {input_names}")

    def _init_model(self):
        """Initialize ONNX model and tokenizer."""
        onnx_path = self.MODEL_PATH / self.ONNX_MODEL_FILE

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        # Initialize ONNX session
        logger.info(f"Loading ONNX ranker model with providers: {self._preferred_providers}")
        self._model = ort.InferenceSession(
            str(onnx_path),
            providers=self._preferred_providers
        )

        # Initialize tokenizer - try tokenizers package first (no transformers needed!)
        tokenizer_json_path = self.MODEL_PATH / "tokenizer.json"
        if tokenizer_json_path.exists():
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
                self._tokenizer.enable_truncation(max_length=512)
                self._use_fast_tokenizer = True
                logger.info("Loaded tokenizer using tokenizers package (no transformers)")
            except ImportError:
                # Fall back to transformers if tokenizers not available
                if TRANSFORMERS_AVAILABLE:
                    self._tokenizer = AutoTokenizer.from_pretrained(str(self.MODEL_PATH), local_files_only=True)
                    self._use_fast_tokenizer = False
                else:
                    raise ImportError("Neither tokenizers nor transformers available for tokenizer loading")
        elif TRANSFORMERS_AVAILABLE:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.MODEL_PATH), local_files_only=True)
            self._use_fast_tokenizer = False
        else:
            raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_json_path} and transformers not available")

        logger.info(f"ONNX ranker model loaded. Active provider: {self._model.get_providers()[0]}")

    def score_quality(self, query: str, memory_content: str) -> float:
        """
        Score the quality/relevance of a memory.

        For classifiers (DeBERTa): Evaluates absolute quality (ignores query, uses memory only)
        For cross-encoders (MS-MARCO): Evaluates query-document relevance

        Args:
            query: Search query (used only for cross-encoders, ignored for classifiers)
            memory_content: Memory content to score

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not memory_content:
            return 0.0

        try:
            # Prepare input based on model type
            if self.model_config['type'] == 'classifier':
                # DeBERTa: Evaluate memory content only (absolute quality)
                text_input = memory_content

                if self._use_fast_tokenizer:
                    # Use tokenizers package
                    encoded = self._tokenizer.encode(text_input)
                    input_ids = np.array([encoded.ids], dtype=np.int64)
                    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
                else:
                    # Use transformers AutoTokenizer
                    inputs = self._tokenizer(text_input, padding=True, truncation=True, max_length=512, return_tensors="np")
                    input_ids = inputs["input_ids"].astype(np.int64)
                    attention_mask = inputs["attention_mask"].astype(np.int64)

                ort_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

            elif self.model_config['type'] == 'cross-encoder':
                # MS-MARCO: Evaluate query-document relevance
                if not query:
                    return 0.0

                if self._use_fast_tokenizer:
                    # Use tokenizers package's ability to encode pairs, which correctly handles special tokens and token type IDs.
                    self._tokenizer.enable_truncation(max_length=512)
                    self._tokenizer.enable_padding(length=512)
                    encoded = self._tokenizer.encode(query, pair=memory_content)

                    input_ids = np.array([encoded.ids], dtype=np.int64)
                    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
                    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)
                else:
                    # Use transformers AutoTokenizer
                    inputs = self._tokenizer(query, memory_content, padding=True, truncation=True, max_length=512, return_tensors="np")
                    input_ids = inputs["input_ids"].astype(np.int64)
                    attention_mask = inputs["attention_mask"].astype(np.int64)
                    token_type_ids = inputs["token_type_ids"].astype(np.int64)

                ort_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }

            else:
                logger.error(f"Unsupported model type: {self.model_config['type']}")
                return 0.5

            # Run inference
            outputs = self._model.run(None, ort_inputs)
            logits = outputs[0][0]  # Shape: (num_classes,)

            # Convert logits to score based on model type
            if self.model_config['type'] == 'classifier':
                # DeBERTa: 3-class classification
                # Apply softmax to get probabilities
                exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
                probs = exp_logits / exp_logits.sum()

                # Map to 0-1 scale with proper spacing
                # NVIDIA DeBERTa label order: 0=High, 1=Medium, 2=Low
                # So class_values should be [1.0, 0.5, 0.0]
                class_values = np.array([1.0, 0.5, 0.0])
                score = float(np.dot(probs, class_values))

            elif self.model_config['type'] == 'cross-encoder':
                # MS-MARCO: Binary classification with sigmoid
                logit = float(logits[0] if len(logits.shape) > 0 and logits.shape[0] > 1 else logits)
                score = 1.0 / (1.0 + np.exp(-logit))

            return np.clip(score, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error scoring quality with {self.model_name}: {e}")
            return 0.5  # Return neutral score on error

    # Minimum batch size to use batched ONNX inference on GPU.
    # Below this threshold, sequential single-item GPU calls are faster
    # because small batches incur padding + tensor transfer overhead
    # without enough items to amortize it. Benchmarked on RTX 5050:
    #   GPU sequential: 5.2ms/item, GPU batch@10: 15.3ms/item,
    #   GPU batch@32: 0.7ms/item. Crossover is ~16 items.
    MIN_GPU_BATCH_SIZE = int(os.environ.get("MCP_QUALITY_MIN_GPU_BATCH", "16"))

    def score_quality_batch(
        self, pairs: List[Tuple[str, str]], max_batch_size: int = 32
    ) -> List[float]:
        """
        Score multiple (query, memory_content) pairs in batched inference calls.

        Processes items in chunks of max_batch_size to cap memory usage.
        On GPU with small batches (< MIN_GPU_BATCH_SIZE), dispatches to
        sequential single-item calls which are faster due to lower overhead.
        Falls back to sequential score_quality() on per-item error.

        Args:
            pairs: List of (query, memory_content) tuples
            max_batch_size: Maximum items per ONNX inference call

        Returns:
            List of quality scores between 0.0 and 1.0
        """
        if not pairs:
            return []

        # On GPU, small batches are slower than sequential single-item calls
        # due to padding + tensor transfer overhead exceeding parallelism gains.
        is_gpu = self._model.get_providers()[0] in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')
        if is_gpu and len(pairs) < self.MIN_GPU_BATCH_SIZE:
            logger.debug(
                f"Batch size {len(pairs)} < GPU threshold {self.MIN_GPU_BATCH_SIZE}, "
                f"using sequential GPU calls"
            )
            return [self.score_quality(q, c) for q, c in pairs]

        all_scores: List[float] = []

        # Process in chunks
        for chunk_start in range(0, len(pairs), max_batch_size):
            chunk = pairs[chunk_start:chunk_start + max_batch_size]
            try:
                chunk_scores = self._score_batch_chunk(chunk)
                all_scores.extend(chunk_scores)
            except Exception as e:
                logger.warning(
                    f"Batch chunk failed ({len(chunk)} items), falling back to sequential: {e}"
                )
                for query, content in chunk:
                    all_scores.append(self.score_quality(query, content))

        return all_scores

    def _score_batch_chunk(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score a single chunk of pairs in one ONNX inference call."""
        if self.model_config['type'] == 'classifier':
            return self._score_classifier_batch(pairs)
        elif self.model_config['type'] == 'cross-encoder':
            return self._score_cross_encoder_batch(pairs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_config['type']}")

    def _score_classifier_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Batch score for classifier models (DeBERTa). Uses memory_content only."""
        texts = [content for _, content in pairs]
        texts = [t if t else " " for t in texts]  # Avoid empty strings

        if self._use_fast_tokenizer:
            encoded_list = self._tokenizer.encode_batch(texts)
            max_len = max(len(enc.ids) for enc in encoded_list)
            batch_size = len(texts)

            input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
            attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

            for i, enc in enumerate(encoded_list):
                length = len(enc.ids)
                input_ids[i, :length] = enc.ids
                attention_mask[i, :length] = enc.attention_mask
        else:
            inputs = self._tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="np"
            )
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self._model.run(None, ort_inputs)
        logits = outputs[0]  # Shape: (batch_size, num_classes)

        # Vectorized softmax + weighted score
        # DeBERTa: 3-class — [High=1.0, Medium=0.5, Low=0.0]
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        class_values = np.array([1.0, 0.5, 0.0])
        scores = probs @ class_values
        return np.clip(scores, 0.0, 1.0).tolist()

    def _score_cross_encoder_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Batch score for cross-encoder models (MS-MARCO). Uses query+content pairs."""
        if self._use_fast_tokenizer:
            # Lock protects tokenizer state (no_padding/enable_padding) from
            # concurrent callers (e.g., interleaved single-item score_quality).
            with self._tokenizer_lock:
                self._tokenizer.enable_truncation(max_length=512)
                self._tokenizer.no_padding()

                encoded_list = self._tokenizer.encode_batch(
                    [(q if q else " ", c if c else " ") for q, c in pairs]
                )

                # Re-enable padding before releasing lock
                self._tokenizer.enable_padding(length=512)

            max_len = max(len(enc.ids) for enc in encoded_list)
            batch_size = len(pairs)

            input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
            attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
            token_type_ids = np.zeros((batch_size, max_len), dtype=np.int64)

            for i, enc in enumerate(encoded_list):
                length = len(enc.ids)
                input_ids[i, :length] = enc.ids
                attention_mask[i, :length] = enc.attention_mask
                token_type_ids[i, :length] = enc.type_ids
        else:
            queries = [q if q else " " for q, _ in pairs]
            contents = [c if c else " " for _, c in pairs]
            inputs = self._tokenizer(
                queries, contents, padding=True, truncation=True, max_length=512, return_tensors="np"
            )
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            token_type_ids = inputs["token_type_ids"].astype(np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        outputs = self._model.run(None, ort_inputs)
        logits = outputs[0]  # Shape: (batch_size, num_classes) or (batch_size,)

        # Vectorized sigmoid
        if len(logits.shape) > 1 and logits.shape[1] > 1:
            raw = logits[:, 0]
        else:
            raw = logits.flatten()
        scores = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(scores, 0.0, 1.0).tolist()


def get_onnx_ranker_model(model_name: str = None, device: str = "auto") -> Optional[ONNXRankerModel]:
    """
    Get ONNX ranker model if available.

    Args:
        model_name: Model to use (default from config if None)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps', 'directml')

    Returns:
        ONNXRankerModel instance or None if ONNX is not available
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX Runtime not available")
        return None

    # Use config default if not specified
    if model_name is None:
        from .config import QualityConfig
        config = QualityConfig.from_env()
        model_name = config.local_model

    # Check if ONNX model exists - if so, we don't need transformers!
    model_path = Path.home() / ".cache" / "mcp_memory" / "onnx_models" / model_name
    onnx_path = model_path / "model.onnx"

    if not onnx_path.exists() and not TRANSFORMERS_AVAILABLE:
        logger.warning(f"ONNX model not found at {onnx_path} and transformers not available for export")
        return None

    try:
        return ONNXRankerModel(model_name=model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to create ONNX ranker for {model_name}: {e}")
        return None
