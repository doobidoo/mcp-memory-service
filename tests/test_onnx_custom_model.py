"""Tests for ONNX backend honoring a custom embedding model.

The ONNX backend used to be hardcoded to all-MiniLM-L6-v2 (Chroma S3), so any
other MCP_EMBEDDING_MODEL was silently ignored under USE_ONNX. These tests pin
the routing logic that decides between the default S3 path and a Hugging Face
Hub download for a custom model.

They exercise the routing decision only (no network / no model load), so they
run anywhere ONNX Runtime + tokenizers are importable.
"""

import pytest

try:
    import onnxruntime  # noqa: F401  (probe)
    import tokenizers  # noqa: F401  (probe)
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE, reason="Requires onnxruntime + tokenizers"
)


def _make_without_io(monkeypatch, model_name, env=None):
    """Instantiate ONNXEmbeddingModel with download + init stubbed out.

    We only want to assert the routing attributes computed in __init__, not
    perform any network access or ONNX session creation.
    """
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    for key, val in (env or {}).items():
        monkeypatch.setenv(key, val)

    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_init_model", lambda self: None)
    return mod.ONNXEmbeddingModel(model_name=model_name)


def test_default_model_uses_s3_path(monkeypatch):
    """all-MiniLM-L6-v2 keeps the original bundled S3 archive path."""
    m = _make_without_io(monkeypatch, "all-MiniLM-L6-v2")
    assert m._is_default_model is True
    assert m._hf_repo is None


def test_custom_model_resolves_to_onnx_community(monkeypatch):
    """A non-default model resolves to onnx-community/<base>-ONNX on the Hub."""
    m = _make_without_io(monkeypatch, "paraphrase-multilingual-MiniLM-L12-v2")
    assert m._is_default_model is False
    assert m._hf_repo == "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
    # custom model must NOT reuse the default MiniLM cache dir
    assert m._model_dir.name == "paraphrase-multilingual-MiniLM-L12-v2"


def test_custom_model_repo_override(monkeypatch):
    """MCP_ONNX_MODEL_REPO overrides the default HF repo resolution."""
    m = _make_without_io(
        monkeypatch,
        "some-model",
        env={"MCP_ONNX_MODEL_REPO": "myorg/some-model-onnx"},
    )
    assert m._is_default_model is False
    assert m._hf_repo == "myorg/some-model-onnx"


def test_org_prefixed_model_name_uses_basename(monkeypatch):
    """A org/name model uses only the basename for repo + cache resolution."""
    m = _make_without_io(monkeypatch, "sentence-transformers/all-MiniLM-L6-v2")
    # basename is the default model -> S3 path
    assert m._is_default_model is True


def test_none_model_name_falls_back_to_default(monkeypatch):
    """A None model_name normalises to the default (no crash downstream)."""
    m = _make_without_io(monkeypatch, None)
    assert m.model_name == "all-MiniLM-L6-v2"
    assert m._is_default_model is True


def test_resolve_model_path_default_vs_custom(monkeypatch, tmp_path):
    """_resolve_model_path returns the S3 layout for default, custom dir otherwise."""
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    default = _make_without_io(monkeypatch, "all-MiniLM-L6-v2")
    assert default._resolve_model_path() == (
        default.DOWNLOAD_PATH / default.EXTRACTED_FOLDER_NAME / "model.onnx"
    )

    custom = _make_without_io(monkeypatch, "paraphrase-multilingual-MiniLM-L12-v2")
    # point the custom dir at a tmp layout with onnx/model.onnx
    custom._model_dir = tmp_path
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"stub")
    assert custom._resolve_model_path() == tmp_path / "onnx" / "model.onnx"


def test_find_onnx_file_ignores_quantized_variants(monkeypatch, tmp_path):
    """Only the canonical model.onnx is selected, not model_quantized.onnx."""
    m = _make_without_io(monkeypatch, "some-multilingual-model")
    m._model_dir = tmp_path
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model_quantized.onnx").write_bytes(b"q")
    # no canonical model.onnx yet -> None (won't silently pick the quantized one)
    assert m._find_onnx_file() is None
    (onnx_dir / "model.onnx").write_bytes(b"full")
    assert m._find_onnx_file() == onnx_dir / "model.onnx"


@pytest.mark.parametrize("bad_name", [
    "bad name with spaces",   # space in base
    "weird$model",            # shell/path metachar
    "..",                     # base resolves to parent-dir token
    "model;rm -rf",           # command-ish separators
])
def test_custom_model_name_is_validated(monkeypatch, bad_name):
    """A non-default model name whose base (after the last '/') is not a plain
    identifier is rejected before it becomes a filesystem path or Hub repo id.

    Note: a leading path like ``a/b/base`` is reduced to ``base`` first, so this
    guards the residual component that actually reaches the path/repo builders.
    """
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_init_model", lambda self: None)
    with pytest.raises(ValueError):
        mod.ONNXEmbeddingModel(model_name=bad_name)


def test_default_model_name_never_validated(monkeypatch):
    """The default model skips validation entirely (backward compatible)."""
    m = _make_without_io(monkeypatch, "all-MiniLM-L6-v2")
    assert m._is_default_model is True


def test_resolve_max_length_xlm_roberta_and_bert(tmp_path):
    """_resolve_max_length accounts for RoBERTa/XLM-R padding offset (#2073)."""
    import json
    from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel

    # 1. No config files -> 512 fallback
    assert ONNXEmbeddingModel._resolve_max_length(tmp_path) == 512

    # 2. XLM-RoBERTa (e.g. intfloat/multilingual-e5-small): 514 - (1 + 1) == 512
    #    even when tokenizer_config.json uses HF's VERY_LARGE_INTEGER sentinel.
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "xlm-roberta",
        "max_position_embeddings": 514,
        "pad_token_id": 1,
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "model_max_length": 1000000000000000019884624838656,
    }))
    assert ONNXEmbeddingModel._resolve_max_length(tmp_path) == 512

    # 3. Long-context XLM-RoBERTa (e.g. bge-m3): 8194 - 2 == 8192
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "xlm-roberta",
        "max_position_embeddings": 8194,
        "pad_token_id": 1,
    }))
    assert ONNXEmbeddingModel._resolve_max_length(tmp_path) == 8192

    # 4. BERT model with tighter tokenizer_config model_max_length=256
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "bert",
        "max_position_embeddings": 512,
        "pad_token_id": 0,
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "model_max_length": 256,
    }))
    assert ONNXEmbeddingModel._resolve_max_length(tmp_path) == 256


def test_init_model_enables_truncation_when_tokenizer_truncation_is_null(monkeypatch, tmp_path):
    """Hub tokenizers without built-in truncation get truncated to model limit (#2073)."""
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from mcp_memory_service.embeddings import onnx_embeddings as mod

    vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    assert tok.truncation is None

    model_dir = tmp_path / "multilingual-e5-small"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"stub-onnx")
    tok.save(str(model_dir / "tokenizer.json"))
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "xlm-roberta",
        "max_position_embeddings": 514,
        "pad_token_id": 1,
    }))

    class _StubIO:
        def __init__(self, name, shape=None):
            self.name = name
            self.shape = shape or [1, 384]

    class _StubSession:
        def __init__(self, path, providers=None):
            self.path = path

        def get_outputs(self):
            return [_StubIO("last_hidden_state", [1, 512, 384])]

        def get_inputs(self):
            return [_StubIO("input_ids"), _StubIO("attention_mask")]

    monkeypatch.setattr(mod.ort, "InferenceSession", _StubSession)
    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_download_model_if_needed", lambda self: None)

    orig_init = mod.ONNXEmbeddingModel._init_model

    def _patched_init(self):
        self._model_dir = model_dir
        orig_init(self)

    monkeypatch.setattr(mod.ONNXEmbeddingModel, "_init_model", _patched_init)

    instance = mod.ONNXEmbeddingModel(model_name="multilingual-e5-small")
    assert instance._tokenizer.truncation is not None
    assert instance._tokenizer.truncation["max_length"] == 512

    long_text = " ".join(["hello"] * 700)
    encoded = instance._tokenizer.encode(long_text)
    assert len(encoded.ids) == 512


