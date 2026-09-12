"""Regression tests for issue #1101: pin the legacy ONNX exporter explicitly.

torch 2.9 changed the default of ``torch.onnx.export``'s ``dynamo`` parameter
to True, which routes the call through the new ``torch.export``-based exporter.
``_ensure_onnx_model`` passes the legacy argument set (``dynamic_axes``,
``opset_version``, ...) and the DeBERTa export fails under the new exporter
(confirmed on torch 2.13; ``dynamo=False`` makes it work).

The pin has to be conditional: ``dynamo`` only exists as a parameter from
torch 2.5, and pyproject allows ``torch>=2.0.0`` — passing it unconditionally
would trade a failure on new torch for a TypeError on old torch.

These tests need no ml extras: torch is replaced by a fake module whose
``onnx.export`` records what it was called with, so they run everywhere and
exercise both signature generations.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from mcp_memory_service.quality import onnx_ranker
from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel


def _fake_torch(export_fn):
    return SimpleNamespace(no_grad=contextlib.nullcontext, onnx=SimpleNamespace(export=export_fn))


def _ranker_for_export(tmp_path):
    """A bare instance pointed at an empty dir, so _ensure_onnx_model exports."""
    ranker = object.__new__(ONNXRankerModel)
    ranker.MODEL_PATH = tmp_path / "model"
    ranker.model_config = {"type": "classifier", "hf_name": "some/quality-model"}
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    ranker._load_source_model = MagicMock(return_value=(tokenizer, MagicMock()))
    return ranker


class TestDynamoPin:
    def test_modern_torch_gets_dynamo_false(self, tmp_path, monkeypatch):
        """On torch >= 2.5 the export must pin dynamo=False explicitly.

        From 2.9 the default is True and the legacy argument set fails on
        DeBERTa; relying on the default is exactly the bug of #1101.
        """
        recorded = {}

        def export(model, args, f, *, input_names, output_names, dynamic_axes,
                   opset_version, do_constant_folding, export_params, dynamo=True):
            recorded["dynamo"] = dynamo

        monkeypatch.setattr(onnx_ranker, "torch", _fake_torch(export), raising=False)

        _ranker_for_export(tmp_path)._ensure_onnx_model()

        assert recorded["dynamo"] is False, (
            "torch.onnx.export was left on its default dynamo setting; on "
            "torch >= 2.9 that default is True and the DeBERTa export fails "
            "with this legacy argument set (issue #1101)."
        )

    def test_legacy_torch_does_not_receive_the_keyword(self, tmp_path, monkeypatch):
        """On torch < 2.5 there is no ``dynamo`` parameter to pass.

        The fake signature is strict, exactly like the real old one: passing
        ``dynamo`` unconditionally raises TypeError here, which is what an
        unguarded pin would do to every torch 2.0-2.4 install.
        """
        called = {}

        def export(model, args, f, *, input_names, output_names, dynamic_axes,
                   opset_version, do_constant_folding, export_params):
            called["ok"] = True

        monkeypatch.setattr(onnx_ranker, "torch", _fake_torch(export), raising=False)

        _ranker_for_export(tmp_path)._ensure_onnx_model()

        assert called.get("ok"), "export was never reached on a legacy-signature torch"
