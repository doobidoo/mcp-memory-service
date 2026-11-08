"""Real-model tests for the ONNX transformers NLI backend.

The suite skips unless the default model is already in the local HF cache —
CI installs no ML extras and must never download weights. Where the cache
exists (the [nli] environment), these prove the real load-and-classify path
end to end through NLIClassifier's public API.
"""

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from mcp_memory_service.reasoning.nli import NLIClassifier  # noqa: E402


def _model_cached() -> bool:
    import os
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == "cross-encoder/nli-deberta-v3-small":
            return True
    return os.environ.get("MCP_NLI_MODEL") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not _model_cached(), reason="model weights not in local HF cache")
async def test_transformers_backend_classifies_real_pair():
    classifier = NLIClassifier(backend="transformers")
    result = await classifier.classify(
        "The cat sat on the mat.", "The cat did not sit on the mat."
    )
    assert result.label == "contradiction"
    assert result.confidence > 0.8


@pytest.mark.asyncio
@pytest.mark.skipif(not _model_cached(), reason="model weights not in local HF cache")
async def test_transformers_backend_batch():
    classifier = NLIClassifier(backend="transformers")
    results = await classifier.classify_batch([
        ("The cat sat on the mat.", "The cat did not sit on the mat."),
        ("Never commit directly to main.", "The cache was never warmed after deploy."),
    ])
    assert [r.label for r in results] == ["contradiction", "neutral"]


@pytest.mark.asyncio
@pytest.mark.skipif(not _model_cached(), reason="model weights not in local HF cache")
async def test_transformers_backend_ceiling_is_one():
    classifier = NLIClassifier(backend="transformers")
    assert classifier.max_achievable_confidence() == 1.0
