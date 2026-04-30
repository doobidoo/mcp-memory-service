"""
Quantize the deberta quality classifier ONNX model at Docker build time.

The fp32 export of nvidia-quality-classifier-deberta is ~702 MB of external
weights, which dominates the :quality-cpu image size delta vs :latest. This
script produces fp16 and dynamic-int8 variants, benchmarks each against the
fp32 baseline (size, Pearson correlation, latency), and replaces the fp32
artifact with the winner.

Decision rule:
  * Correlation with fp32 must be >= MIN_CORRELATION (default 0.98).
  * Among passing variants, pick the smallest.
  * Tie-break by latency.
  * If no variant passes, keep fp32 and exit non-zero (fail the build only
    when --strict is passed; otherwise warn and keep fp32).

ms-marco-MiniLM-L-6-v2 is intentionally skipped — it's already ~80 MB.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia-quality-classifier-deberta"
MIN_CORRELATION = 0.98
NUM_SAMPLES = 100
MAX_SEQ_LEN = 256


# Synthetic but realistic memory-like texts. Kept inline so the script has no
# external dataset dependency at build time.
SAMPLE_TEXTS = [
    "Fixed bug in auth middleware where session tokens were not refreshed on rotation.",
    "Reminder: meeting with the platform team on Thursday at 2pm to discuss the migration plan.",
    "User prefers terse responses without trailing summaries.",
    "Decision: standardize on hybrid storage backend for all production deployments.",
    "Refactored consolidation/graph.py to extract relationship inference into its own module.",
    "Cloudflare token rotated 2026-04-15 — remember to update .env on production hosts.",
    "Performance: SQLite-vec reads measured at 5ms median, 12ms p99 over 10k memories.",
    "Note: the dashboard JS lacks automated test coverage; verify visually before merging.",
    "TODO: remove the legacy /api/v1 endpoint after the v11 release.",
    "Investigation: the flaky concurrent SQLite test is caused by busy_timeout interactions.",
    "User authored issue #793 about quantizing the deberta ONNX model.",
    "PR #773 introduced Reciprocal Rank Fusion for hybrid search.",
    "OAuth public clients use PKCE without a client_secret per OAuth 2.1.",
    "Hybrid sync owner setting: only the HTTP server should sync to Cloudflare.",
    "Memory consolidation runs daily at 2am UTC via APScheduler.",
    "The :quality-cpu image is currently 2.44 GB and needs to drop below 1 GB.",
    "Henry's email is henry.krupp@gmail.com per the user-info memory.",
    "Test cleanup deleted production memories on 2026-02-08; PR #438 added safeguards.",
    "Always tag memories with 'mcp-memory-service' as the first tag.",
    "Use github-release-manager agent for all version bumps and releases.",
] * 5  # 100 samples


@dataclass
class Variant:
    name: str
    path: Path
    size_bytes: int
    correlation: float
    mean_latency_ms: float


def _ensure_quantization_deps() -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import onnxconverter_common  # noqa: F401
        import tokenizers  # noqa: F401
        from onnxruntime.quantization import quantize_dynamic, QuantType  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"Missing quantization dependency ({exc}). Install onnx, onnxruntime, "
            "onnxconverter-common, and tokenizers in the builder stage."
        )


def _model_dir(model_name: str) -> Path:
    return Path.home() / ".cache" / "mcp_memory" / "onnx_models" / model_name


def _model_size_bytes(model_path: Path) -> int:
    """Total size of the .onnx file plus its external-data sidecars.

    Sidecars must share the same stem prefix as the model file (e.g.
    `model.fp16.onnx` -> `model.fp16.onnx_data` / `model.fp16.onnx_data_*`).
    Files belonging to other variants in the same directory are ignored, which
    is what allows variants to coexist for benchmarking without polluting each
    other's measured size.
    """
    total = model_path.stat().st_size
    prefix = model_path.name + "_"  # e.g. "model.fp16.onnx_"
    for sibling in model_path.parent.iterdir():
        if sibling == model_path or not sibling.is_file():
            continue
        if sibling.name.startswith(prefix):
            total += sibling.stat().st_size
    return total


def _onnx_artifact_files(model_path: Path) -> list[Path]:
    """All files that belong to a given ONNX artifact (.onnx + sidecars).

    Used for cleanup after a variant is rejected or replaced. Mirrors the
    sibling-detection rule in `_model_size_bytes` to stay consistent.
    """
    files = [model_path]
    prefix = model_path.name + "_"
    for sibling in model_path.parent.iterdir():
        if sibling == model_path or not sibling.is_file():
            continue
        if sibling.name.startswith(prefix):
            files.append(sibling)
    return files


def _quantize_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    logger.info(f"[fp16] Converting {src.name} -> {dst.name}")
    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(dst), save_as_external_data=False)


def _quantize_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    logger.info(f"[int8] Dynamic-quantizing {src.name} -> {dst.name}")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        # MatMul + Gather cover the bulk of DeBERTa weight volume.
        op_types_to_quantize=["MatMul", "Gather"],
    )


def _tokenize_samples(model_dir: Path, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize sample texts using the saved tokenizer.json. Returns padded input_ids + attention_mask."""
    from tokenizers import Tokenizer

    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(
            f"tokenizer.json not found at {tokenizer_json} — was the model exported correctly?"
        )

    tok = Tokenizer.from_file(str(tokenizer_json))
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    tok.enable_padding(length=MAX_SEQ_LEN)

    encodings = tok.encode_batch(texts)
    input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
    return input_ids, attention_mask


def _run_inference(onnx_path: Path, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_name = sess.get_outputs()[0].name
    # Run per-sample to keep peak memory low and to simulate realistic single-text scoring.
    logits = []
    for i in range(input_ids.shape[0]):
        out = sess.run(
            [output_name],
            {
                "input_ids": input_ids[i : i + 1],
                "attention_mask": attention_mask[i : i + 1],
            },
        )
        logits.append(out[0].squeeze())
    return np.array(logits)


def _benchmark(
    name: str,
    onnx_path: Path,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    fp32_logits: Optional[np.ndarray],
) -> Variant:
    import onnxruntime as ort

    size_bytes = _model_size_bytes(onnx_path)

    # Latency: warmup 3, measure 10
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_name = sess.get_outputs()[0].name
    feed = {"input_ids": input_ids[:1], "attention_mask": attention_mask[:1]}
    for _ in range(3):
        sess.run([output_name], feed)
    samples = []
    for _ in range(10):
        t0 = time.perf_counter()
        sess.run([output_name], feed)
        samples.append((time.perf_counter() - t0) * 1000)
    mean_latency_ms = float(np.mean(samples))

    # Correlation against fp32
    logits = _run_inference(onnx_path, input_ids, attention_mask)
    if fp32_logits is None:
        correlation = 1.0
    else:
        # Reduce multi-class logits to scalar by taking the argmax-class probability,
        # which is what the quality scorer actually consumes.
        fp32_scalar = _to_scalar(fp32_logits)
        var_scalar = _to_scalar(logits)
        correlation = float(np.corrcoef(fp32_scalar, var_scalar)[0, 1])

    logger.info(
        f"[{name}] size={size_bytes / 1e6:.1f} MB | corr={correlation:.4f} | latency={mean_latency_ms:.1f} ms"
    )
    return Variant(
        name=name,
        path=onnx_path,
        size_bytes=size_bytes,
        correlation=correlation,
        mean_latency_ms=mean_latency_ms,
    )


def _to_scalar(logits: np.ndarray) -> np.ndarray:
    """Reduce per-sample logits to the same scalar score the production code uses.

    Mirrors `quality/onnx_ranker.py::score_quality` for the DeBERTa classifier:
    softmax over 3 classes (label order: 0=High, 1=Medium, 2=Low) and dot
    product with [1.0, 0.5, 0.0]. Using the same reduction here means the
    correlation we measure is the correlation that actually matters at runtime.
    """
    if logits.ndim == 1:
        # Already scalar (binary / cross-encoder). Apply sigmoid to match runtime.
        return 1.0 / (1.0 + np.exp(-logits))
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    class_values = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    if probs.shape[-1] != class_values.shape[0]:
        # Defensive: unknown classifier shape — fall back to top-class probability
        # so we still produce a usable correlation rather than crashing the build.
        return probs.max(axis=-1)
    return probs @ class_values


def _pick_winner(variants: list[Variant], min_corr: float) -> Optional[Variant]:
    passing = [v for v in variants if v.correlation >= min_corr]
    if not passing:
        return None
    # Smallest size first; latency tie-break.
    passing.sort(key=lambda v: (v.size_bytes, v.mean_latency_ms))
    return passing[0]


def quantize_model(model_name: str, mode: str, min_correlation: float, strict: bool) -> int:
    _ensure_quantization_deps()

    model_dir = _model_dir(model_name)
    fp32_path = model_dir / "model.onnx"
    if not fp32_path.exists():
        logger.error(f"fp32 model not found at {fp32_path} — run export_quality_models.py first.")
        return 1

    fp16_path = model_dir / "model.fp16.onnx"
    int8_path = model_dir / "model.int8.onnx"

    input_ids, attention_mask = _tokenize_samples(model_dir, SAMPLE_TEXTS[:NUM_SAMPLES])
    fp32_logits = _run_inference(fp32_path, input_ids, attention_mask)

    variants: list[Variant] = [
        _benchmark("fp32", fp32_path, input_ids, attention_mask, fp32_logits=None)
    ]

    if mode in ("fp16", "best"):
        try:
            _quantize_fp16(fp32_path, fp16_path)
            variants.append(_benchmark("fp16", fp16_path, input_ids, attention_mask, fp32_logits))
        except Exception as exc:
            logger.warning(f"fp16 quantization failed: {exc}", exc_info=True)

    if mode in ("int8", "best"):
        try:
            _quantize_int8(fp32_path, int8_path)
            variants.append(_benchmark("int8", int8_path, input_ids, attention_mask, fp32_logits))
        except Exception as exc:
            logger.warning(f"int8 quantization failed: {exc}", exc_info=True)

    logger.info("=" * 60)
    logger.info(f"{'variant':>8} | {'size (MB)':>10} | {'corr':>6} | {'latency':>8}")
    logger.info("-" * 60)
    for v in variants:
        logger.info(
            f"{v.name:>8} | {v.size_bytes / 1e6:>10.1f} | {v.correlation:>6.4f} | {v.mean_latency_ms:>6.1f}ms"
        )
    logger.info("=" * 60)

    candidates = [v for v in variants if v.name != "fp32"]
    winner = _pick_winner(candidates, min_correlation)

    if winner is None:
        msg = f"No quantized variant met correlation >= {min_correlation}; keeping fp32."
        if strict:
            logger.error(msg + " (--strict): failing build.")
            return 2
        logger.warning(msg)
        # Clean up rejected variants (file + their sidecars).
        for v in candidates:
            for f in _onnx_artifact_files(v.path):
                f.unlink(missing_ok=True)
        return 0

    logger.info(f"Winner: {winner.name} (size {winner.size_bytes / 1e6:.1f} MB, corr {winner.correlation:.4f}).")

    # Delete the fp32 artifact set (model.onnx + every sidecar like model.onnx_data).
    # This is the critical step: without it the ~700 MB external-weights file
    # would remain in the image even though model.onnx itself was replaced.
    for f in _onnx_artifact_files(fp32_path):
        f.unlink(missing_ok=True)

    # Delete losing variants (file + their sidecars).
    for v in candidates:
        if v.name != winner.name:
            for f in _onnx_artifact_files(v.path):
                f.unlink(missing_ok=True)

    # Promote the winner into model.onnx. fp16/int8 variants are single-file,
    # so a plain rename is sufficient — there are no sidecars to move.
    shutil.move(str(winner.path), str(fp32_path))

    final_size = _model_size_bytes(fp32_path)
    logger.info(f"Final model.onnx: {final_size / 1e6:.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to quantize.")
    parser.add_argument(
        "--mode",
        choices=["fp16", "int8", "best"],
        default="best",
        help="Quantization mode. 'best' tries fp16 + int8 and picks the winner.",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=MIN_CORRELATION,
        help="Minimum Pearson correlation with fp32 baseline (default: 0.98).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit if no variant meets the correlation threshold.",
    )
    args = parser.parse_args()

    return quantize_model(
        model_name=args.model,
        mode=args.mode,
        min_correlation=args.min_correlation,
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())
