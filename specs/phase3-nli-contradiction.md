# Phase 3: NLI-Based Contradiction Detection

## Context

RFC #732. Phases 1a, 1b, 2, 4 merged (v10.66.0). This is the final phase.

The current contradiction detection (`consolidation/contradictions.py`) uses embedding similarity band (0.4-0.75) only — 23% false positive rate in production. Phase 3 replaces this with NLI (Natural Language Inference) for high-precision classification.

## Architecture — 4-Stage Pipeline

```
memory_store / maintain cycle
    ↓
[1] Entity overlap gate (fast, no model)
    → memories sharing ≥1 entity with the new memory
    ↓
[2] Embedding pre-filter (existing code, no changes)
    → candidates in 0.4-0.75 similarity band
    ↓
[3] NLI model (NEW — this phase)
    → classify each pair: entailment / contradiction / neutral
    ↓
[4] Conflict registration (existing: memory_conflicts + superseded_by)
    → only NLI-confirmed contradictions get registered
```

**Key insight**: Stages 1+2 reduce NLI calls by ~90%. NLI is expensive — only run on pre-filtered candidates.

## Files to Create/Modify

### NEW: `src/mcp_memory_service/reasoning/nli.py`

NLI classifier. Must support multiple backends:

```python
class NLIResult:
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float  # 0.0-1.0

class NLIClassifier:
    """NLI-based contradiction detection."""
    
    def __init__(self, backend: str = "auto"):
        # Backends (in priority order):
        # 1. "transformers" — local HuggingFace model (cross-encoder/nli-deberta-v3-small)
        # 2. "api" — external API (future, not implemented now)
        # 3. "heuristic" — regex/keyword fallback (no model needed)
        pass
    
    async def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify relationship between two texts."""
        pass
    
    async def classify_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Batch classification for efficiency."""
        pass
```

**Model choice**: `cross-encoder/nli-deberta-v3-small` (22M params, fast, good accuracy).
- Optional dependency: `transformers` + `torch` (CPU only).
- If not installed → fall back to heuristic backend.
- Env var: `MCP_NLI_BACKEND=transformers|heuristic|auto` (default: auto)
- Env var: `MCP_NLI_MODEL=cross-encoder/nli-deberta-v3-small`
- Env var: `MCP_NLI_CONFIDENCE_THRESHOLD=0.7` (below this → treat as neutral)

### MODIFY: `src/mcp_memory_service/consolidation/contradictions.py`

Replace the current embedding-only detection with the 4-stage pipeline:

```python
async def detect_contradictions_nli(storage, memory_hash: str = None, dry_run: bool = True) -> dict:
    """
    NLI-enhanced contradiction detection.
    
    If memory_hash provided: check ONE memory against its entity-overlapping neighbors.
    If memory_hash is None: batch scan (maintain cycle).
    """
    # Stage 1: Entity overlap gate
    # Stage 2: Embedding pre-filter (existing search_memories in 0.4-0.75 band)
    # Stage 3: NLI classification
    # Stage 4: Register conflicts (only if NLI says "contradiction" with confidence >= threshold)
```

Keep the old `detect_contradictions()` function as-is (backward compat). Add the new one alongside.

### MODIFY: `src/mcp_memory_service/server_impl.py`

In `handle_memory_store`: after storing, if `MCP_NLI_ON_STORE=true`, call `detect_contradictions_nli(storage, memory_hash=new_hash)`.

In `handle_memory_resolve`: accept `hashes` as list (batch resolve per doobidoo's feedback).

### NEW: `tests/reasoning/test_nli.py`

Tests for the NLI classifier:
- Test heuristic backend (always available, no model needed)
- Test with mock transformers (patch the model)
- Test the 4-stage pipeline end-to-end with fixtures
- Test batch classification
- Test confidence threshold filtering

### NEW: `tests/reasoning/test_contradiction_pipeline.py`

Integration tests for the full pipeline:
- Two memories with same entity + contradicting content → detected
- Two memories with same entity + neutral content → NOT detected
- Two memories with NO shared entity → skipped (never reaches NLI)
- Batch scan via maintain cycle

## Heuristic Backend (fallback when no model)

Simple keyword/pattern-based detection for when transformers is not installed:

```python
CONTRADICTION_PATTERNS = [
    (r'\bnot\b.*\b(is|was|are|were)\b', r'\b(is|was|are|were)\b'),  # negation
    (r'\bnever\b', r'\balways\b'),
    (r'\bdisabled?\b', r'\benabled?\b'),
    (r'\bremoved?\b', r'\badded?\b'),
    (r'\bfalse\b', r'\btrue\b'),
    # version conflicts: "X is 1.0" vs "X is 2.0"
    (r'(\w+)\s+(?:is|was|=)\s+v?(\d+\.\d+)', 'version_conflict'),
]
```

Confidence for heuristic: max 0.6 (never as confident as NLI model).

## Configuration (env vars)

| Var | Default | Description |
|-----|---------|-------------|
| MCP_NLI_ENABLED | false | Master switch for NLI pipeline |
| MCP_NLI_ON_STORE | false | Check on every memory_store |
| MCP_NLI_BACKEND | auto | transformers, heuristic, or auto |
| MCP_NLI_MODEL | cross-encoder/nli-deberta-v3-small | HuggingFace model |
| MCP_NLI_CONFIDENCE_THRESHOLD | 0.7 | Min confidence to register conflict |
| MCP_NLI_MAX_CANDIDATES | 20 | Max pairs to send to NLI per memory |

## Constraints

- `transformers` and `torch` are OPTIONAL dependencies (extras: `pip install .[nli]`)
- If not installed, heuristic backend is used automatically
- Zero breaking changes to existing API
- All new code in `reasoning/` directory (our CODEOWNERS domain)
- Tests must pass WITHOUT transformers installed (heuristic fallback)
- The pipeline function must be importable from `reasoning/` (not buried in consolidation/)

## Acceptance Criteria

1. `pytest tests/reasoning/` passes (all new tests)
2. `pytest tests/` passes (no regressions)
3. Heuristic backend works without any ML dependencies
4. Pipeline correctly skips memories with no entity overlap (Stage 1 gate)
5. Pipeline correctly skips memories outside similarity band (Stage 2)
6. NLI-confirmed contradictions are registered in memory_conflicts
7. `memory_resolve` accepts batch (list of hashes)
8. Env vars control all behavior (disabled by default)
