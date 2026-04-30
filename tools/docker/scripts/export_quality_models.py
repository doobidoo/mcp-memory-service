"""
Pre-export quality scoring models to ONNX format at Docker build time.

Run during the builder stage of Dockerfile.quality-cpu. Triggers the existing
ONNX export path in quality/onnx_ranker.py for both supported models, writing
artifacts to ~/.cache/mcp_memory/onnx_models/<model_name>/model.onnx.

The runtime stage copies those artifacts and uses onnxruntime only — no
torch/transformers/huggingface_hub at runtime.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    "ms-marco-MiniLM-L-6-v2",
    "nvidia-quality-classifier-deberta",
]


def export_model(model_name: str) -> bool:
    """Export a single model to ONNX. Returns True on success."""
    from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

    onnx_path = (
        Path.home()
        / ".cache"
        / "mcp_memory"
        / "onnx_models"
        / model_name
        / "model.onnx"
    )

    if onnx_path.exists():
        logger.info(f"[{model_name}] Already exported at {onnx_path} — skipping.")
        return True

    logger.info(f"[{model_name}] Starting ONNX export...")
    try:
        # Instantiating ONNXRankerModel calls _ensure_onnx_model() which
        # performs the torch.onnx.export if model.onnx is absent.
        model = ONNXRankerModel(model_name=model_name, device="cpu")
        model._ensure_onnx_model()
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"[{model_name}] Export complete: {onnx_path} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"[{model_name}] Export finished but file not found at {onnx_path}")
            return False
    except Exception as exc:
        logger.error(f"[{model_name}] Export failed: {exc}", exc_info=True)
        return False


def main() -> int:
    failures = []
    for model_name in MODELS:
        if not export_model(model_name):
            failures.append(model_name)

    if failures:
        logger.error(f"Failed to export: {failures}")
        return 1

    logger.info("All quality models exported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
