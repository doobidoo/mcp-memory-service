"""CoreML failure recovery with real ONNX inference and sqlite-vec storage.

Only the unavailable CoreML hardware boundary is fault-injected. The tiny ONNX
graph casts input_ids and attention_mask to float, unsqueezes them, and
concatenates them into [batch, sequence, 2] hidden states (IR 8, opset 13).
Embedding pooling, CPU inference, and database operations all run normally.
The serialized graph avoids a model download or an extra onnx test dependency.
"""

import base64
from pathlib import Path

import numpy as np
import pytest

from mcp_memory_service.embeddings.onnx_embeddings import (
    ONNXEmbeddingModel,
    get_onnx_embedding_model,
)
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

ort = pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"
MODEL = (
    "CAg6lwQKLQoJaW5wdXRfaWRzEg9pbnB1dF9pZHNfZmxvYXQiBENhc3QqCQoCdG8YAaABAgo2Cg9p"
    "bnB1dF9pZHNfZmxvYXQKBGF4ZXMSEmlucHV0X2lkc19leHBhbmRlZCIJVW5zcXVlZXplCjcKDmF0"
    "dGVudGlvbl9tYXNrEhRhdHRlbnRpb25fbWFza19mbG9hdCIEQ2FzdCoJCgJ0bxgBoAECCkAKFGF0"
    "dGVudGlvbl9tYXNrX2Zsb2F0CgRheGVzEhdhdHRlbnRpb25fbWFza19leHBhbmRlZCIJVW5zcXVl"
    "ZXplClUKEmlucHV0X2lkc19leHBhbmRlZAoXYXR0ZW50aW9uX21hc2tfZXhwYW5kZWQSEWxhc3Rf"
    "aGlkZGVuX3N0YXRlIgZDb25jYXQqCwoEYXhpcxgCoAECEg90ZXN0X2VtYmVkZGluZ3MqDQgBEAc6"
    "AQJCBGF4ZXNaKAoJaW5wdXRfaWRzEhsKGQgHEhUKBxIFYmF0Y2gKChIIc2VxdWVuY2VaLQoOYXR0"
    "ZW50aW9uX21hc2sSGwoZCAcSFQoHEgViYXRjaAoKEghzZXF1ZW5jZVotCg50b2tlbl90eXBlX2lk"
    "cxIbChkIBxIVCgcSBWJhdGNoCgoSCHNlcXVlbmNlYjQKEWxhc3RfaGlkZGVuX3N0YXRlEh8KHQgB"
    "EhkKBxIFYmF0Y2gKChIIc2VxdWVuY2UKAggCQgIQDQ=="
)


@pytest.fixture
def onnx_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    model_dir = tmp_path / "onnx"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(base64.b64decode(MODEL))
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "short": 1, "long": 2}, "[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    monkeypatch.setattr(ONNXEmbeddingModel, "DOWNLOAD_PATH", tmp_path)
    monkeypatch.setenv("MCP_MEMORY_USE_ONNX", "1")
    monkeypatch.setenv("MCP_MEMORY_ALLOW_HASH_EMBEDDINGS", "0")
    monkeypatch.delenv("MCP_MEMORY_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("MCP_EXTERNAL_EMBEDDING_URL", raising=False)
    return model_dir


@pytest.fixture
def failing_coreml(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    create_session = ort.InferenceSession
    attempts = []

    def create(path: str, providers: list[str]) -> ort.InferenceSession:
        attempts.append(list(providers))
        session = create_session(path, providers=[CPU])
        if COREML in providers:
            monkeypatch.setattr(session, "get_providers", lambda: list(providers))

            def fail(*args: object, **kwargs: object) -> None:
                raise RuntimeError("CoreML dynamic sequence resize error -6\nretry")

            monkeypatch.setattr(session, "run", fail)
        return session

    monkeypatch.setattr(ort, "get_available_providers", lambda: [COREML, CPU])
    monkeypatch.setattr(ort, "InferenceSession", create)
    return attempts


@pytest.mark.asyncio
async def test_coreml_failure_stores_and_retrieves(
    onnx_cache: Path,
    failing_coreml: list[list[str]],
    temp_db_path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/coreml.db")
    await storage.initialize()
    try:
        assert isinstance(storage.embedding_model, ONNXEmbeddingModel)
        for content in ("short", "long short long"):
            memory = Memory(
                content=content, content_hash=generate_content_hash(content)
            )
            success, message = await storage.store(memory)
            assert success, message
        results = await storage.retrieve("short long", n_results=2)
        assert {result.memory.content for result in results} == {
            "short",
            "long short long",
        }
        assert storage.embedding_dimension == 2
        assert storage.embedding_model._model.get_providers() == [CPU]
        assert failing_coreml == [[COREML, CPU], [CPU]]
        warnings = [
            r.getMessage() for r in caplog.records if "CoreML" in r.getMessage()
        ]
        assert any("CPUExecutionProvider" in message for message in warnings)
        assert all("\n" not in message for message in warnings)
    finally:
        await storage.close()


def test_fallback_preserves_batched_embeddings(
    onnx_cache: Path, failing_coreml: list[list[str]]
) -> None:
    reference = ONNXEmbeddingModel(preferred_providers=[CPU])
    model = get_onnx_embedding_model()
    assert model is not None
    texts = ["short", "long short long"]
    np.testing.assert_allclose(model.encode(texts), reference.encode(texts))
    np.testing.assert_allclose(model.encode("short"), reference.encode("short"))
    assert failing_coreml == [[CPU], [COREML, CPU], [CPU]]


@pytest.mark.parametrize("providers", [CPU, f"  {CPU}  "])
def test_cpu_pin_skips_coreml(
    onnx_cache: Path,
    failing_coreml: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    providers: str,
) -> None:
    monkeypatch.setenv("MCP_MEMORY_ONNX_PROVIDERS", providers)
    model = get_onnx_embedding_model()
    assert model is not None
    assert model._model.get_providers() == [CPU]
    assert np.isfinite(model.encode(["short", "long short"])).all()
    assert failing_coreml == [[CPU]]


@pytest.mark.parametrize("providers", ["MissingExecutionProvider", f"{CPU},", ","])
def test_invalid_provider_pin_is_rejected(
    onnx_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    providers: str,
) -> None:
    monkeypatch.setenv("MCP_MEMORY_ONNX_PROVIDERS", providers)
    with pytest.raises(ValueError, match="MCP_MEMORY_ONNX_PROVIDERS"):
        get_onnx_embedding_model()


@pytest.mark.asyncio
@pytest.mark.parametrize("providers", ["MissingExecutionProvider", f"{CPU},", ","])
async def test_invalid_provider_pin_stops_storage_initialization(
    onnx_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    providers: str,
    temp_db_path: str,
) -> None:
    monkeypatch.setenv("MCP_MEMORY_ONNX_PROVIDERS", providers)
    monkeypatch.setenv("MCP_MEMORY_ALLOW_HASH_EMBEDDINGS", "1")
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/invalid-provider.db")
    try:
        with pytest.raises(RuntimeError, match="MCP_MEMORY_ONNX_PROVIDERS"):
            await storage.initialize()
    finally:
        await storage.close()


def test_cpu_errors_are_not_retried(
    onnx_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ONNXEmbeddingModel(preferred_providers=[CPU])
    session = model._model

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CPU inference failed")

    monkeypatch.setattr(session, "run", fail)
    with pytest.raises(RuntimeError, match="CPU inference failed"):
        model.encode("short")
    assert model._model is session
