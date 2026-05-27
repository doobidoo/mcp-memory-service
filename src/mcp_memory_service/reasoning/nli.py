"""NLI-based contradiction detection (Phase 3, RFC #732).

Provides NLIClassifier with heuristic fallback (no ML deps required)
and detect_contradictions_nli 4-stage pipeline function.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

HEURISTIC_MAX_CONFIDENCE = 0.6

# Contradiction patterns for heuristic backend
_NEGATION_PAIRS = [
    (re.compile(r'\bdisabled?\b', re.I), re.compile(r'\benabled?\b', re.I)),
    (re.compile(r'\bremoved?\b', re.I), re.compile(r'\badded?\b', re.I)),
    (re.compile(r'\bnever\b', re.I), re.compile(r'\balways\b', re.I)),
    (re.compile(r'\bfalse\b', re.I), re.compile(r'\btrue\b', re.I)),
    (re.compile(r'\bstopped\b', re.I), re.compile(r'\brunning\b', re.I)),
    (re.compile(r'\bnot\b', re.I), None),  # general negation
]

_VERSION_RE = re.compile(r'(\w[\w.-]*)\s+(?:version\s+(?:is\s+)?|is\s+v?|=\s*v?)(\d+[\d.]+)', re.I)


@dataclass
class NLIResult:
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float  # 0.0-1.0


class NLIClassifier:
    """NLI-based contradiction detection with multiple backends."""

    def __init__(self, backend: str = "auto"):
        if backend == "auto":
            backend = os.environ.get("MCP_NLI_BACKEND", "heuristic")
        self.backend = backend

    async def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify relationship between two texts."""
        if self.backend == "heuristic":
            return self._heuristic_classify(premise, hypothesis)
        return NLIResult(label="neutral", confidence=0.0)

    async def classify_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Batch classification."""
        return [await self.classify(p, h) for p, h in pairs]

    def _heuristic_classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Keyword/pattern-based fallback classification."""
        # Check version conflicts
        pv = _VERSION_RE.findall(premise)
        hv = _VERSION_RE.findall(hypothesis)
        if pv and hv:
            for p_name, p_ver in pv:
                for h_name, h_ver in hv:
                    if p_name.lower() == h_name.lower() and p_ver != h_ver:
                        return NLIResult(label="contradiction", confidence=0.55)

        # Check antonym/negation pairs
        for pat_a, pat_b in _NEGATION_PAIRS:
            if pat_b is None:
                # General negation: one has "not", other doesn't
                a_has = pat_a.search(premise)
                b_has = pat_a.search(hypothesis)
                if bool(a_has) != bool(b_has):
                    return NLIResult(label="contradiction", confidence=0.5)
            else:
                if (pat_a.search(premise) and pat_b.search(hypothesis)) or \
                   (pat_b.search(premise) and pat_a.search(hypothesis)):
                    return NLIResult(label="contradiction", confidence=0.55)

        return NLIResult(label="neutral", confidence=0.3)


async def detect_contradictions_nli(
    storage, memory_hash: str = None, dry_run: bool = True
) -> dict:
    """
    NLI-enhanced contradiction detection — 4-stage pipeline.

    Stage 1: Entity overlap gate
    Stage 2: Embedding pre-filter (similarity band 0.4-0.75)
    Stage 3: NLI classification
    Stage 4: Conflict registration
    """
    result = {"pairs_detected": 0, "nli_calls": 0, "conflicts_registered": 0}

    if memory_hash is None:
        return result

    # Get the source memory
    mem_a = await storage.get_memory(memory_hash)
    if mem_a is None:
        return result

    entities_a = mem_a.metadata.get("entities", []) if mem_a.metadata else []
    if not entities_a:
        return result

    # Stage 1: Entity overlap gate
    candidate_hashes = set()
    for entity in entities_a:
        hashes = await storage.find_memories_by_entity(entity)
        if hashes:
            candidate_hashes.update(hashes)
    candidate_hashes.discard(memory_hash)

    if not candidate_hashes:
        return result

    # Stage 2: Embedding pre-filter — get candidates in similarity band
    search_result = await storage.search_memories(mem_a.content)
    band_hashes = set()
    band_memories = {}
    if search_result and "memories" in search_result:
        for m in search_result["memories"]:
            h = m.get("content_hash")
            score = m.get("similarity_score", 0)
            if h in candidate_hashes and 0.4 <= score <= 0.75:
                band_hashes.add(h)
                band_memories[h] = m

    if not band_hashes:
        return result

    # Stage 3: NLI classification
    classifier = NLIClassifier(backend="heuristic")
    threshold = float(os.environ.get("MCP_NLI_CONFIDENCE_THRESHOLD", "0.7"))

    contradictions = []
    for h in band_hashes:
        mem_b_data = band_memories[h]
        nli_result = await classifier.classify(mem_a.content, mem_b_data["content"])
        result["nli_calls"] += 1

        if nli_result.label == "contradiction" and nli_result.confidence > 0.0:
            contradictions.append((h, mem_b_data, nli_result))

    result["pairs_detected"] = len(contradictions)

    # Stage 4: Register conflicts (unless dry_run)
    if not dry_run and contradictions:
        for h, mem_b_data, nli_res in contradictions:
            try:
                await storage.add_graph_edge(
                    memory_hash, h, "contradicts",
                    {"confidence": nli_res.confidence, "method": "nli_heuristic"}
                )
                result["conflicts_registered"] += 1
            except Exception as e:
                logger.debug(f"Failed to register conflict: {e}")

    return result
