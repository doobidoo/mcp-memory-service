"""Configuration for quality scoring system."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityConfig:
    """Configuration for quality scoring system with local-first defaults."""

    # System control
    enabled: bool = True

    # AI provider selection
    # Options: 'local' (ONNX only), 'openai-compatible', 'groq', 'gemini',
    #          'auto' (try all), 'none' (implicit only)
    ai_provider: str = 'local'

    # Local ONNX model settings
    local_model: str = 'nvidia-quality-classifier-deberta'
    local_device: str = 'auto'  # auto|cpu|cuda|mps|directml

    # Cloud API settings (optional, user opt-in)
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # OpenAI-compatible endpoint settings (LiteLLM proxy, Ollama, vLLM, MLX-LM, etc.)
    openai_compat_base_url: Optional[str] = None   # e.g. http://localhost:11434/v1
    openai_compat_model: Optional[str] = None      # e.g. qwen2.5:7b-instruct
    openai_compat_api_key: Optional[str] = None    # optional; use "ollama" for Ollama

    # Quality boost — search-side reranking (see config/quality.py, consumed by
    # the storage backends). Kept here because the store-time blend below used
    # to piggyback on these two fields.
    boost_enabled: bool = False
    boost_weight: float = 0.3

    # Store-time blend of implicit signals into the AI score (QualityScorer).
    # Separate from the search-side knobs above: there, a higher weight means
    # "trust the quality score more"; here it means "trust the classifier less".
    # Both defaults are inherited from the boost settings when the dedicated
    # variables are unset, so existing deployments keep their behaviour (#179).
    implicit_blend_enabled: bool = False
    implicit_weight: float = 0.3

    # Fallback scoring (DeBERTa primary, MS-MARCO rescue for technical content)
    # NOTE: Fallback mode not recommended - MS-MARCO is query-relevance model, not quality classifier
    fallback_enabled: bool = False
    deberta_threshold: float = 0.4  # Lowered from 0.6 to accept more technical content (v8.50.0)
    ms_marco_threshold: float = 0.7  # If DeBERTa low AND MS-MARCO >= this, use MS-MARCO (rescue)

    @classmethod
    def from_env(cls) -> 'QualityConfig':
        """Load configuration from environment variables.

        ``MCP_QUALITY_IMPLICIT_BLEND_ENABLED`` and ``MCP_QUALITY_IMPLICIT_WEIGHT``
        control the store-time blend. When unset they fall back to the boost
        settings, which is what drove the blend before #179 — so nobody's stored
        scores shift because of this change.
        """
        boost_enabled = os.getenv('MCP_QUALITY_BOOST_ENABLED', 'false').lower() == 'true'
        boost_weight = float(os.getenv('MCP_QUALITY_BOOST_WEIGHT', '0.3'))

        blend_raw = os.getenv('MCP_QUALITY_IMPLICIT_BLEND_ENABLED')
        implicit_blend_enabled = (
            blend_raw.lower() == 'true' if blend_raw is not None else boost_enabled
        )
        implicit_weight = float(os.getenv('MCP_QUALITY_IMPLICIT_WEIGHT', str(boost_weight)))

        return cls(
            enabled=os.getenv('MCP_QUALITY_SYSTEM_ENABLED', 'true').lower() == 'true',
            ai_provider=os.getenv('MCP_QUALITY_AI_PROVIDER', 'local'),
            local_model=os.getenv('MCP_QUALITY_LOCAL_MODEL', 'ms-marco-MiniLM-L-6-v2'),
            local_device=os.getenv('MCP_QUALITY_LOCAL_DEVICE', 'auto'),
            groq_api_key=os.getenv('GROQ_API_KEY'),
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            openai_compat_base_url=os.getenv('MCP_QUALITY_AI_BASE_URL'),
            openai_compat_model=os.getenv('MCP_QUALITY_AI_MODEL'),
            openai_compat_api_key=os.getenv('MCP_QUALITY_AI_API_KEY'),
            boost_enabled=boost_enabled,
            boost_weight=boost_weight,
            implicit_blend_enabled=implicit_blend_enabled,
            implicit_weight=implicit_weight,
            fallback_enabled=os.getenv('MCP_QUALITY_FALLBACK_ENABLED', 'false').lower() == 'true',
            deberta_threshold=float(os.getenv('MCP_QUALITY_DEBERTA_THRESHOLD', '0.6')),
            ms_marco_threshold=float(os.getenv('MCP_QUALITY_MSMARCO_THRESHOLD', '0.7'))
        )

    def validate(self) -> bool:
        """Validate configuration settings."""
        valid_providers = ['local', 'openai-compatible', 'groq', 'gemini', 'auto', 'none']
        if self.ai_provider not in valid_providers:
            raise ValueError(f"Invalid ai_provider: {self.ai_provider}")

        if self.local_device not in ['auto', 'cpu', 'cuda', 'mps', 'directml']:
            raise ValueError(f"Invalid local_device: {self.local_device}")

        if not 0.0 <= self.boost_weight <= 1.0:
            raise ValueError(f"boost_weight must be between 0.0 and 1.0, got {self.boost_weight}")

        if not 0.0 <= self.implicit_weight <= 1.0:
            raise ValueError(f"implicit_weight must be between 0.0 and 1.0, got {self.implicit_weight}")

        # Validate fallback thresholds
        if not 0.0 <= self.deberta_threshold <= 1.0:
            raise ValueError(f"deberta_threshold must be between 0.0 and 1.0, got {self.deberta_threshold}")

        if not 0.0 <= self.ms_marco_threshold <= 1.0:
            raise ValueError(f"ms_marco_threshold must be between 0.0 and 1.0, got {self.ms_marco_threshold}")

        # Validate multi-model support for fallback
        if self.fallback_enabled:
            models = [m.strip() for m in self.local_model.split(',')]
            if len(models) < 2:
                raise ValueError(
                    "Fallback mode requires at least 2 models in local_model (comma-separated), "
                    f"got: {self.local_model}"
                )
            # Validate each model exists
            for model in models:
                if model not in SUPPORTED_MODELS:
                    raise ValueError(
                        f"Unknown model '{model}' in local_model. "
                        f"Supported models: {list(SUPPORTED_MODELS.keys())}"
                    )

        # Warn if cloud providers selected but no API keys
        if self.ai_provider == 'groq' and not self.groq_api_key:
            raise ValueError("Groq provider selected but GROQ_API_KEY not set")

        if self.ai_provider == 'gemini' and not self.gemini_api_key:
            raise ValueError("Gemini provider selected but GEMINI_API_KEY not set")

        # openai-compatible requires base_url and model
        if self.ai_provider == 'openai-compatible':
            if not self.openai_compat_base_url:
                raise ValueError(
                    "openai-compatible provider selected but MCP_QUALITY_AI_BASE_URL is not set"
                )
            if not self.openai_compat_model:
                raise ValueError(
                    "openai-compatible provider selected but MCP_QUALITY_AI_MODEL is not set"
                )

        return True

    @property
    def use_local_only(self) -> bool:
        """Check if using local-only scoring."""
        return self.ai_provider in ['local', 'none']

    @property
    def can_use_openai_compatible(self) -> bool:
        """Check if an OpenAI-compatible endpoint is configured."""
        return (
            self.ai_provider == 'openai-compatible'
            and bool(self.openai_compat_base_url)
            and bool(self.openai_compat_model)
        )

    @property
    def can_use_groq(self) -> bool:
        """Check if Groq API is available."""
        return self.groq_api_key is not None and self.ai_provider in ['groq', 'auto']

    @property
    def can_use_gemini(self) -> bool:
        """Check if Gemini API is available."""
        return self.gemini_api_key is not None and self.ai_provider in ['gemini', 'auto']


# Model registry for supported ONNX models
SUPPORTED_MODELS = {
    'nvidia-quality-classifier-deberta': {
        'hf_name': 'nvidia/quality-classifier-deberta',
        'type': 'classifier',  # 3-class classification (Low/Medium/High)
        'size_mb': 450,
        'inputs': ['input_ids', 'attention_mask'],  # No token_type_ids
        'output_classes': ['low', 'medium', 'high'],
        'description': 'Absolute quality assessment (recommended, eliminates self-matching bias)'
    },
    'ms-marco-MiniLM-L-6-v2': {
        'hf_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'type': 'cross-encoder',  # Query-document relevance ranking
        'size_mb': 23,
        'inputs': ['input_ids', 'attention_mask', 'token_type_ids'],
        'output_classes': None,  # Continuous score via sigmoid
        'description': 'Legacy relevance ranker (has self-matching bias, use for relative ranking only)'
    }
}


def validate_model_selection(model_name: str) -> dict:
    """
    Validate model selection and return model config.

    Args:
        model_name: Name of the model to validate

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}"
        )
    return SUPPORTED_MODELS[model_name]


# --- Quality model composition (#1312) -------------------------------------
# quality_score is a single materialized "effective" value written from two
# origin fields kept separately in metadata: computed_quality (machine) and
# user_rating (human -1/0/+1). Consumers that read quality_score need no change;
# retention (forgetting) reads computed_quality directly so a human down-vote
# de-ranks search without changing the keep/forget verdict.

# Human rating -1/0/+1 mapped to a search-facing effective value. A thumbs-down
# de-ranks hard (0.25) without pinning to an absolute floor (0.0); a single
# thumbs-up (0.9) does not claim more certainty than the scorer's top band.
USER_RATING_TO_QUALITY = {-1: 0.25, 0: 0.5, 1: 0.9}


def effective_quality(computed=None, user_rating=None) -> float:
    """Compose the materialized effective quality_score from the two origins.

    Human rating wins when present (mapped via USER_RATING_TO_QUALITY); otherwise
    the machine computed_quality; otherwise the 0.5 default. computed_quality is
    never consumed here in a way that erases it — callers keep it as its own field.
    """
    if user_rating is not None:
        try:
            return USER_RATING_TO_QUALITY[int(user_rating)]
        except (KeyError, ValueError, TypeError):
            pass
    if computed is not None:
        return float(computed)
    return 0.5
