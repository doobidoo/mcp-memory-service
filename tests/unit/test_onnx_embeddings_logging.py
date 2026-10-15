"""
Log-injection tests for embeddings/onnx_embeddings.py (#1146).

The ONNX loader logs the model name and Hub repo id it reads from the
environment, and the text of a failed download. A newline in any of them must
not reach the log as a line break.
"""

import logging
import sys
import types
from pathlib import Path

import pytest

from mcp_memory_service.embeddings import onnx_embeddings
from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel

FORGED = "FORGED model verified"
LOGGER = "mcp_memory_service.embeddings.onnx_embeddings"


@pytest.fixture
def loader_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Let the constructor run without onnxruntime, and keep every path under tmp."""
    monkeypatch.setattr(onnx_embeddings, "ONNX_AVAILABLE", True)
    monkeypatch.setattr(onnx_embeddings, "TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(ONNXEmbeddingModel, "DOWNLOAD_PATH", tmp_path / "onnx_models" / "default")
    monkeypatch.setenv("MCP_MEMORY_ONNX_ALLOW_DOWNLOAD", "1")
    return tmp_path


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages), messages
    assert not any(f"\n{FORGED}" in m for m in messages), messages


def test_hub_download_error_log_does_not_carry_newlines(loader_env, monkeypatch, caplog):
    monkeypatch.setenv("MCP_ONNX_MODEL_REPO", f"org/repo\n{FORGED}")

    def snapshot_download(**_kwargs):
        raise RuntimeError(f"hub down\n{FORGED}")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER), pytest.raises(RuntimeError):
        ONNXEmbeddingModel("custom-model")

    _assert_clean(caplog, "Downloading ONNX model 'custom-model' from HF repo org/repo")
    _assert_clean(caplog, "Failed to download ONNX model from org/repo")
    _assert_clean(caplog, "hub down")


def test_archive_download_error_log_does_not_carry_newlines(loader_env, monkeypatch, caplog):
    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get(self, _url):
            raise RuntimeError(f"archive host down\n{FORGED}")

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=_Client))

    with caplog.at_level(logging.DEBUG, logger=LOGGER), pytest.raises(RuntimeError):
        ONNXEmbeddingModel()

    _assert_clean(caplog, "Downloading ONNX model from https://")
    _assert_clean(caplog, "Failed to download ONNX model: archive host down")


def test_lazy_scan_flags_hub_repo_and_model_dir():
    """The two names this module adds to EXTERNAL_NAMES keep a later unwrap red."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_log_injection_guard import _lazy_findings

    bare = (
        "logger.info('Downloading from %s', self._hf_repo)\n"
        "logger.info('Cache at %s', self._model_dir)\n"
    )
    assert len(_lazy_findings(bare)) == 2
