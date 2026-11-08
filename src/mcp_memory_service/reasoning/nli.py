"""NLI-based contradiction detection (Phase 3, RFC #732).

Provides NLIClassifier with three backends:

- ``heuristic`` (default, no ML deps): regex/keyword patterns from
  ``nli_patterns/*.yaml``
- ``transformers``: local cross-encoder NLI model, per
  ``specs/phase3-nli-contradiction.md`` — default
  ``cross-encoder/nli-deberta-v3-small``. Runs through ONNX Runtime with
  ``tokenizers`` and ``huggingface_hub`` only (``pip install .[nli]``);
  torch and transformers are not on the inference path. Any load or
  inference failure degrades to the heuristic for that call, bounded to one
  warning, like the LLM backend.
- ``cascade``/``llm``: LLM via the harvest provider chain.

Also provides detect_contradictions_nli, the 4-stage pipeline function.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from .nli_patterns import load_nli_patterns
from ..config.locale import get_active_locales
from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

# Highest non-neutral confidence the heuristic backend can ever emit (see
# _heuristic_classify). Read via NLIClassifier.max_achievable_confidence() to
# detect a quarantine gate no heuristic contradiction could meet (issue #1216),
# and kept honest by test_heuristic_ceiling_matches_constant. Was 0.6 and never
# read anywhere; corrected to the value the code actually returns.
HEURISTIC_MAX_CONFIDENCE = 0.55

# Flat confidence the LLM/cascade backend assigns to any non-neutral label.
LLM_NONNEUTRAL_CONFIDENCE = 0.9

# Load patterns from YAML locale files (replaces hardcoded _NEGATION_PAIRS / _VERSION_RE)
_PATTERNS = load_nli_patterns(tuple(get_active_locales()))

# Module-level ONNX session cache (see NLIClassifier._get_onnx_runtime):
# one load per (model, onnx file) per process, shared across classifier
# instances, so per-detection construction does not re-pay the session load.
_ONNX_RUNTIME_CACHE: dict = {}
_ONNX_RUNTIME_LOCK = asyncio.Lock()


@dataclass
class NLIResult:
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float  # 0.0-1.0


_NLI_LABELS = ("entailment", "contradiction", "neutral")


def _parse_nli_label(response: str):
    """Parse an LLM NLI answer into one of the three labels, or None.

    Rules (see review on PR #1215):
    - Anchor on the FIRST token (word-boundary matching over the whole answer
      would misread "there is no contradiction" / "not a contradiction" as
      ``contradiction``; the first-token anchor sends those to the heuristic).
    - Strip an optional leading ``classification:`` prefix, so a prefixed
      answer like "Classification: contradiction" still parses.
    - Reject the whole answer (return None) if a SECOND distinct label appears
      anywhere in it — a hedged answer like "contradiction, but really neutral"
      is unparseable and must fall back to the heuristic, not be acted on at
      0.9 confidence.
    - Any unknown/garbled first token also returns None.

    Returns the label string, or None when the caller should fall back to the
    heuristic classifier.
    """
    if not response or not response.strip():
        return None

    text = response.strip().lower()

    # Reject if more than one distinct label is mentioned anywhere (hedging).
    mentioned = {lbl for lbl in _NLI_LABELS if re.search(rf"\b{lbl}\b", text)}
    if len(mentioned) > 1:
        return None

    # Strip an optional leading "classification:" prefix before anchoring.
    text = re.sub(r"^\s*classification\s*:\s*", "", text)

    tokens = text.split()
    if not tokens:
        return None
    first = re.sub(r"[^a-z]", "", tokens[0])
    if first not in _NLI_LABELS:
        return None
    return first


class NLIClassifier:
    """NLI-based contradiction detection with multiple backends."""

    def __init__(self, backend: str = "auto"):
        if backend == "auto":
            # An empty/unset variable means the default, not an unknown backend:
            # `MCP_NLI_BACKEND= python -m ...` must not silently disable NLI.
            backend = (os.environ.get("MCP_NLI_BACKEND") or "heuristic").strip() or "heuristic"
        self.backend = backend
        self._warned_unimplemented = False
        # Phase 2 state (R10, R12)
        self._rewriter = None
        self._llm_available = None
        self._warned_degraded = False
        # transformers backend state (RFC #732 Phase 3): (tokenizer, session,
        # id2label) once loaded. True once any call degraded to the heuristic.
        self._onnx_runtime = None
        self.degraded = False

    async def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify relationship between two texts."""
        if self.backend == "heuristic":
            return self._heuristic_classify(premise, hypothesis)
        if self.backend == "transformers":
            return (await self._transformers_classify([(premise, hypothesis)]))[0]
        if self.backend in ("cascade", "llm"):
            return await self._llm_classify(premise, hypothesis)
        if not self._warned_unimplemented:
            logger.warning(
                "NLI backend '%s' is not implemented; returning neutral. "
                "Only 'heuristic', 'transformers' and 'cascade' are supported.",
                self.backend,
            )
            self._warned_unimplemented = True
        return NLIResult(label="neutral", confidence=0.0)

    def max_achievable_confidence(self) -> float:
        """Highest non-neutral confidence this backend can ever return.

        Lets callers detect a contradiction gate that no result could meet
        (issue #1216): the default ``heuristic`` backend tops out at
        ``HEURISTIC_MAX_CONFIDENCE``, below the default 0.7 quarantine gate, so
        quarantine-on-store is inert unless the gate is lowered or the
        ``cascade`` backend is configured. An unimplemented backend returns
        neutral 0.0, so its ceiling is 0.0.
        """
        if self.backend == "heuristic":
            return HEURISTIC_MAX_CONFIDENCE
        if self.backend == "transformers":
            # ONNX softmax probabilities top out at ~1.0.
            return 1.0
        if self.backend in ("cascade", "llm"):
            return LLM_NONNEUTRAL_CONFIDENCE
        return 0.0

    async def _llm_classify(self, premise: str, hypothesis: str) -> NLIResult:
        """LLM-based NLI via the harvest provider chain (cascade fallback).

        Reuses HarvestRewriter._call_llm, which resolves the configured
        provider chain (HARVEST_LLM_PROVIDERS) with fallback. Any failure
        degrades gracefully to the heuristic classifier.
        """
        # R10: Resolve provider config once per run
        try:
            rewriter = self._get_rewriter()
            
            # Check if LLM is configured once per run (cache result)
            if self._llm_available is None:
                try:
                    self._llm_available = rewriter.is_configured
                except Exception:
                    # R11: Exception during is_configured check -> not available
                    self._llm_available = False
            
            # R11: If no provider configured, use heuristic directly
            if not self._llm_available:
                return self._heuristic_classify(premise, hypothesis)
            
            prompt = (
                "Classify the relationship between Statement A and Statement B.\n"
                "Answer with EXACTLY one word: entailment, contradiction, or neutral.\n\n"
                f"Statement A: {premise[:500]}\n"
                f"Statement B: {hypothesis[:500]}\n\n"
                "Classification:"
            )
            timeout = float(os.environ.get("MCP_NLI_LLM_TIMEOUT", "30"))
            # _call_llm returns (response, provider_name, model) so provenance
            # travels with the call; we only need the response text here.
            response, _provider, _model = await rewriter._call_llm(prompt, timeout)
            label = _parse_nli_label(response)
            if label is None:
                # R12: Empty/unparseable response triggers bounded warning
                if not self._warned_degraded:
                    self._warned_degraded = True
                    sanitized_reason = _sanitize_log_value("unparseable or empty response from LLM")
                    self._warn_once(sanitized_reason)
                # R14: Preserve exact heuristic values
                return self._heuristic_classify(premise, hypothesis)
            return NLIResult(
                label=label,
                confidence=LLM_NONNEUTRAL_CONFIDENCE if label != "neutral" else 0.3,
            )
        except Exception as e:
            # R12, R13: Exception triggers bounded warning with sanitization
            if not self._warned_degraded:
                self._warned_degraded = True
                sanitized_reason = _sanitize_log_value(f"LLM call failed: {e}")
                self._warn_once(sanitized_reason)
            # R14: Preserve exact heuristic values
            return self._heuristic_classify(premise, hypothesis)

    async def classify_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Batch classification.

        The transformers backend runs the whole list through the model in one
        ONNX session call; the other backends classify pair by pair.
        """
        if not pairs:
            return []
        if self.backend == "transformers":
            # Chunked so a whole-store export cannot arrive as one padded
            # tensor; 32 keeps the batch memory flat regardless of input size.
            results = []
            for i in range(0, len(pairs), 32):
                results.extend(await self._transformers_classify(pairs[i:i + 32]))
            return results
        return [await self.classify(p, h) for p, h in pairs]

    async def _transformers_classify(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Local cross-encoder NLI through ONNX Runtime (RFC #732 Phase 3).

        The weights ship as ONNX files in the model repo (fp32 default;
        ``MCP_NLI_ONNX_FILE`` picks a quantized variant), the tokenizer loads
        from ``tokenizer.json`` directly, so the runtime needs
        ``onnxruntime`` + ``tokenizers`` + ``huggingface_hub`` — never torch.
        Load and inference run in worker threads; both are synchronous. Any
        failure degrades to the heuristic for every pair in this call, the
        same contract the LLM backend offers, and sets ``self.degraded`` so
        provenance (edge metadata, harness reports) can name the backend
        that actually produced the verdict.
        """
        try:
            tokenizer, session, id2label = await self._get_onnx_runtime()

            def _run():
                import numpy as np

                encodings = tokenizer.encode_batch([list(p) for p in pairs])
                feed = {
                    name: np.array(
                        [getattr(e, attr) for e in encodings], dtype=np.int64
                    )
                    for name, attr in (
                        ("input_ids", "ids"),
                        ("attention_mask", "attention_mask"),
                        ("token_type_ids", "type_ids"),
                    )
                    if name in {i.name for i in session.get_inputs()}
                }
                logits = session.run(None, feed)[0]
                results = []
                for row in logits:
                    row_max = row.max()
                    exp = np.exp(row - row_max)
                    probs = exp / exp.sum()
                    idx = int(probs.argmax())
                    results.append(
                        {"label": id2label.get(idx, str(idx)), "score": float(probs[idx])}
                    )
                return results

            results = await asyncio.to_thread(_run)
            return [
                self._nli_result_from_scored_label(out, fallback_pair=pair)
                for out, pair in zip(results, pairs)
            ]
        except Exception as e:
            self.degraded = True
            if not self._warned_degraded:
                self._warned_degraded = True
                self._warn_once(_sanitize_log_value(f"transformers inference failed: {e}"))
            return [self._heuristic_classify(p, h) for p, h in pairs]

    async def _get_onnx_runtime(self):
        """Lazy (tokenizer, session, id2label) for the model, cached per model.

        The cache is module-level: callers construct a fresh NLIClassifier
        per detection (the quarantine path, detect_contradictions_nli), and
        each instance used to pay the full ONNX session load again. Keyed by
        (repo, onnx file) so switching ``MCP_NLI_ONNX_FILE`` to a quantized
        build loads that instead. ``MCP_NLI_MODEL`` names the HF repo
        (default per the spec), ``MCP_NLI_ONNX_FILE`` the ONNX file inside
        it — ``onnx/model.onnx`` fp32 by default. Offline environments reuse
        the hub cache.
        """
        cache_key = (
            os.environ.get("MCP_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"),
            os.environ.get("MCP_NLI_ONNX_FILE", "onnx/model.onnx"),
        )
        model_name, onnx_file = cache_key
        if self._onnx_runtime is None:
            async with _ONNX_RUNTIME_LOCK:
                if cache_key not in _ONNX_RUNTIME_CACHE:
                    def _load():
                        import json

                        import onnxruntime as ort
                        from huggingface_hub import (
                            hf_hub_download, try_to_load_from_cache,
                        )
                        from tokenizers import Tokenizer

                        def _fetch(fname):
                            # Cached copy first: hf_hub_download HEAD-checks the
                            # Hub even on a cache hit, which turns a flaky
                            # connection into a failed load of a file we
                            # already have. Delete the cache to refresh.
                            cached = try_to_load_from_cache(model_name, fname)
                            if isinstance(cached, str):
                                return cached
                            return hf_hub_download(model_name, fname)

                        model_path = _fetch(onnx_file)
                        tokenizer = Tokenizer.from_file(_fetch("tokenizer.json"))
                        with open(_fetch("config.json"), encoding="utf-8") as fh:
                            config = json.load(fh)
                        id2label = {int(k): v for k, v in config["id2label"].items()}
                        session = ort.InferenceSession(
                            model_path, providers=["CPUExecutionProvider"]
                        )
                        tokenizer.enable_truncation(max_length=512)
                        tokenizer.enable_padding()
                        return tokenizer, session, id2label

                    _ONNX_RUNTIME_CACHE[cache_key] = await asyncio.to_thread(_load)
                    logger.info(
                        "NLI transformers backend loaded (ONNX): %s/%s",
                        model_name, onnx_file,
                    )
            self._onnx_runtime = _ONNX_RUNTIME_CACHE[cache_key]
        return self._onnx_runtime

    def _nli_result_from_scored_label(self, out: dict, fallback_pair: Tuple[str, str]) -> NLIResult:
        """Map one inference output to NLIResult.

        A label that is not one of the three NLI labels is treated like a
        parse failure: the pair falls back to the heuristic rather than being
        acted on at model confidence.
        """
        label = str(out.get("label", "")).lower()
        confidence = float(out.get("score", 0.0))
        if label in _NLI_LABELS:
            return NLIResult(label=label, confidence=confidence)
        return self._heuristic_classify(*fallback_pair)

    def _get_rewriter(self):
        """R10: Lazy initialization of HarvestRewriter to resolve config once per run."""
        if self._rewriter is None:
            from ..harvest.rewriter import HarvestRewriter
            self._rewriter = HarvestRewriter()
        return self._rewriter

    def _warn_once(self, sanitized_reason: str):
        """R12: Emit one degradation warning per run (reason already sanitized).

        Only the warning is suppressed after the first; later pairs still attempt
        the provider and fall back per-pair, so the message describes this pair,
        not the whole remaining run.
        """
        logger.warning(
            "NLI LLM degradation: %s; falling back to heuristic for this pair "
            "(further degradations this run are not repeated)",
            sanitized_reason
        )

    def _heuristic_classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Keyword/pattern-based fallback classification."""
        # Check version conflicts using loaded patterns
        for version_re in _PATTERNS['version_patterns']:
            pv = version_re.findall(premise)
            hv = version_re.findall(hypothesis)
            if pv and hv:
                for p_name, p_ver in pv:
                    for h_name, h_ver in hv:
                        if p_name.lower() == h_name.lower() and p_ver != h_ver:
                            return NLIResult(label="contradiction", confidence=HEURISTIC_MAX_CONFIDENCE)

        # Check antonym/negation pairs from loaded patterns
        for pat_a, pat_b in _PATTERNS['negation_pairs']:
            if pat_b is None:
                # General negation: one has "not", other doesn't
                a_has = pat_a.search(premise)
                b_has = pat_a.search(hypothesis)
                if bool(a_has) != bool(b_has):
                    return NLIResult(label="contradiction", confidence=0.5)
            else:
                if (pat_a.search(premise) and pat_b.search(hypothesis)) or \
                   (pat_b.search(premise) and pat_a.search(hypothesis)):
                    return NLIResult(label="contradiction", confidence=HEURISTIC_MAX_CONFIDENCE)

        return NLIResult(label="neutral", confidence=0.3)


async def _classify_band(
    classifier, source_content, band_hashes, band_memories, confidence_threshold, result
):
    """Stage 3 helper: run NLI over the similarity band and collect contradictions.

    Extracted from detect_contradictions_nli to keep that function under the
    complexity gate (see review on PR #1215). Mutates ``result['nli_calls']``
    and returns the list of (hash, mem_b_data, nli_result) contradictions.
    """
    contradictions = []
    for h in band_hashes:
        mem_b_data = band_memories[h]
        nli_result = await classifier.classify(source_content, mem_b_data["content"])
        result["nli_calls"] += 1
        if nli_result.label == "contradiction" and nli_result.confidence >= confidence_threshold:
            contradictions.append((h, mem_b_data, nli_result))
    return contradictions


async def detect_contradictions_nli(
    storage, memory_hash: str = None, dry_run: bool = True
) -> dict:
    """
    NLI-enhanced contradiction detection — 4-stage pipeline.

    Stage 1: Entity overlap gate
    Stage 2: Embedding pre-filter (similarity band 0.4-0.75)
    Stage 3: NLI classification
    Stage 4: Conflict registration

    Args:
        storage: MemoryStorage instance (has get_by_hash, search_memories).
                 Graph ops accessed via get_graph_storage() helper.

    Security: This function writes to the graph (store_association) when dry_run=False.
    It must only be called from write-scoped contexts (e.g., handle_store_memory).
    """
    result = {"pairs_detected": 0, "nli_calls": 0, "conflicts_registered": 0}

    # Master kill-switch
    if not os.environ.get("MCP_NLI_ENABLED", "false").lower() in ("1", "true", "yes"):
        return result

    if memory_hash is None:
        return result

    # Get the source memory
    mem_a = await storage.get_by_hash(memory_hash)
    if mem_a is None:
        return result

    entities_a = (mem_a.metadata or {}).get("entities", [])
    if not entities_a:
        return result

    # Get graph storage for entity lookups and edge creation
    try:
        from mcp_memory_service.server.handlers.graph import get_graph_storage
        graph = await get_graph_storage()
    except Exception:
        graph = None

    if not graph:
        return result

    # Stage 1: Entity overlap gate
    candidate_hashes = set()
    for entity in entities_a:
        try:
            hashes = await graph.find_memories_by_entity(entity)
            if hashes:
                candidate_hashes.update(hashes)
        except Exception:
            continue
    candidate_hashes.discard(memory_hash)

    if not candidate_hashes:
        return result

    # Stage 2: Embedding pre-filter — get candidates in similarity band
    try:
        search_result = await storage.search_memories(query=mem_a.content, limit=20)
    except Exception:
        search_result = {}

    band_hashes = set()
    band_memories = {}
    memories_list = search_result.get("memories", []) if isinstance(search_result, dict) else []
    for m in memories_list:
        h = m.get("content_hash")
        score = m.get("similarity_score", 0)
        if h in candidate_hashes and 0.4 <= score <= 0.75:
            band_hashes.add(h)
            band_memories[h] = m

    if not band_hashes:
        return result

    # Stage 3: NLI classification
    classifier = NLIClassifier(backend="auto")
    confidence_threshold = float(os.environ.get("MCP_NLI_CONFIDENCE_THRESHOLD", "0.4"))
    contradictions = await _classify_band(
        classifier, mem_a.content, band_hashes, band_memories, confidence_threshold, result
    )
    result["pairs_detected"] = len(contradictions)

    # Stage 4: Register conflicts (unless dry_run)
    if not dry_run and contradictions:
        for h, mem_b_data, nli_res in contradictions:
            try:
                # Provenance names the backend that actually classified —
                # a degraded transformers run is heuristic evidence, not
                # model evidence.
                backend_used = "heuristic" if classifier.degraded else classifier.backend
                await graph.store_association(
                    source_hash=memory_hash,
                    target_hash=h,
                    similarity=nli_res.confidence,
                    connection_types=["contradiction"],
                    relationship_type="contradicts",
                    metadata={"confidence": nli_res.confidence, "method": f"nli_{backend_used}"},
                )
                result["conflicts_registered"] += 1
            except Exception as e:
                logger.debug("Failed to register conflict: %s", _sanitize_log_value(e))

    return result
