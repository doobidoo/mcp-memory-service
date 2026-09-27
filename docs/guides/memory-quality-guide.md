# Memory Quality System Guide

> **Version**: 8.50.0
> **Status**: Production Ready
> **Feature**: Memento-Inspired Quality System (Issue #260)
> **Evaluation**: [Quality System Evaluation Report](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Quality-System-Evaluation)
>
> **⚠️ IMPORTANT (v8.50.0)**: Fallback mode is **not recommended** due to MS-MARCO architectural limitations (query-document relevance model cannot assess absolute quality). See [Recommended Configuration](#recommended-configuration-v850) for the simpler, more robust approach.

## Recommended Configuration (v8.50.0)

### Primary Recommendation: Implicit Signals Only

**✅ For Technical Corpora (RECOMMENDED)**

If your memory base contains:
- Technical fragments, file paths, references
- Code snippets, CLI output, configuration
- Task lists, tickets, abbreviations
- Short notes and checklists

```bash
# Disable AI quality scoring (avoids prose bias)
export MCP_QUALITY_AI_PROVIDER=none

# Quality based on implicit signals only
export MCP_QUALITY_SYSTEM_ENABLED=true
export MCP_QUALITY_BOOST_ENABLED=false
```

**Why Implicit Signals Work Better**:
- ✅ **Access patterns = true quality** - Frequently-used memories are valuable
- ✅ **No prose bias** - Technical fragments treated fairly
- ✅ **Self-learning** - Quality improves based on actual usage
- ✅ **Simpler** - No model loading, zero latency
- ✅ **Works offline** - No external dependencies

**Implicit Signal Components**:
```python
quality_score = (
    access_frequency × 0.40 +  # How often retrieved
    recency × 0.30 +           # When last accessed
    retrieval_ranking × 0.30   # Average position in results
)
```

**Real-World Test Results** (v8.50.0):
- DeBERTa average score: 0.209 (median: 0.165)
- Only 4% of memories scored ≥ 0.6 (DeBERTa's "good" threshold)
- Manual review: 72% classified as "low quality" were valid technical references
- Examples misclassified: file paths, Terraform configs, technical abbreviations

**Conclusion**: DeBERTa trained on Wikipedia/news systematically under-scores technical content.

---

### Alternative: DeBERTa for Prose-Heavy Corpora

**For Long-Form Documentation Only**

If your memory base contains primarily:
- Blog posts, articles, tutorials
- Narrative meeting notes
- Long-form documentation with prose paragraphs

```bash
# AI quality scoring with DeBERTa
export MCP_QUALITY_AI_PROVIDER=local
export MCP_QUALITY_LOCAL_MODEL=nvidia-quality-classifier-deberta

# Lower threshold for technical tolerance
export MCP_QUALITY_DEBERTA_THRESHOLD=0.4  # Or 0.3 for more leniency

# Combine AI + implicit signals
export MCP_QUALITY_BOOST_ENABLED=true
export MCP_QUALITY_BOOST_WEIGHT=0.3
```

**When to Use AI Scoring**:
| Content Type | Use AI? | Reason |
|--------------|---------|--------|
| Narrative documentation | ✅ Yes | DeBERTa trained for prose |
| Blog posts, articles | ✅ Yes | Complete paragraphs |
| Meeting notes (narrative) | ✅ Yes | Story-like structure |
| Technical fragments | ❌ No | Prose bias under-scores |
| File paths, references | ❌ No | Not prose |
| Code snippets, CLI output | ❌ No | Technical, not narrative |
| Task lists, tickets | ❌ No | Fragmented |

**Expected Distribution** (prose-heavy corpus with 0.4 threshold):
- Low quality (< 0.4): ~20% filtered
- Medium (0.4-0.6): ~40% accepted
- Good (0.6-0.8): ~30% accepted
- Excellent (≥ 0.8): ~10% accepted

## Overview

The **Memory Quality System** transforms MCP Memory Service from static storage to a learning memory system. It automatically evaluates memory quality using AI-driven scoring and uses these scores to improve retrieval precision, consolidation efficiency, and overall system intelligence.

### Key Benefits

- ✅ **40-70% improvement** in retrieval precision (top-5 useful rate: 50% → 70-85%)
- ✅ **Zero cost** with local SLM (privacy-preserving, offline-capable)
- ✅ **Smarter consolidation** - Preserve high-quality memories longer
- ✅ **Quality-boosted search** - Prioritize best memories in results
- ✅ **Automatic learning** - System improves from usage patterns

## How It Works

### Multi-Tier AI Scoring (Local-First)

The system evaluates memory quality (0.0-1.0 score) using a multi-tier fallback chain:

| Tier | Provider | Cost | Latency | Privacy | Default |
|------|----------|------|---------|---------|---------|
| **1** | **Local SLM (ONNX)** | **$0** | **50-100ms** | ✅ Full | ✅ Yes |
| 2 | Groq API | ~$0.30/mo | 900ms | ❌ External | ❌ Opt-in |
| 3 | Gemini API | ~$0.40/mo | 2000ms | ❌ External | ❌ Opt-in |
| 4 | Implicit Signals | $0 | 10ms | ✅ Full | Fallback |

**Default setup**: Local SLM only (zero cost, full privacy, no external API calls)

### Quality Score Components

```
quality_score = (
    local_slm_score × 0.50 +      # Cross-encoder evaluation
    implicit_signals × 0.50        # Usage patterns
)

implicit_signals = (
    access_frequency × 0.40 +      # How often retrieved
    recency × 0.30 +              # When last accessed
    retrieval_ranking × 0.30      # Average position in results
)
```

## 🎯 Local SLM Models (Tier 1 - Primary)

**Update (v8.49.0)**: DeBERTa quality classifier is now the default model, eliminating self-matching bias and providing absolute quality assessment.

### Model Options

The quality system supports multiple ONNX models for different use cases:

#### **NVIDIA DeBERTa Quality Classifier** (Default, Recommended)

**Model**: `nvidia-quality-classifier-deberta` (450MB)
**Architecture**: 3-class text classifier (Low/Medium/High quality)
**Type**: Absolute quality assessment (query-independent)

**Key Features**:
- ✅ **Eliminates self-matching bias** - No query needed, evaluates content directly
- ✅ **Absolute quality scores** - Assesses inherent memory quality
- ✅ **Uniform distribution** - More realistic quality spread (mean: 0.60-0.70)
- ✅ **Fewer false positives** - <5% perfect 1.0 scores (vs 20% with MS-MARCO)

**Performance**:
- CPU: 80-150ms per evaluation
- GPU (CUDA/MPS/DirectML): 20-40ms per evaluation
- Model size: 450MB (downloaded once, cached locally)

**Scoring Process**:
1. Tokenize memory content (no query needed)
2. Run DeBERTa inference (local, private)
3. Get 3-class probabilities: P(low), P(medium), P(high)
4. Calculate weighted score: `0.0×P(low) + 0.5×P(medium) + 1.0×P(high)`

**Usage**:
```bash
# Default model (no configuration needed)
export MCP_QUALITY_LOCAL_MODEL=nvidia-quality-classifier-deberta  # Already default in v8.49.0+

# Quality boost works better with DeBERTa (more accurate scores)
export MCP_QUALITY_BOOST_ENABLED=true
export MCP_QUALITY_BOOST_WEIGHT=0.3
```

**Quality Metrics** (Expected vs MS-MARCO):
| Metric | MS-MARCO (Old) | DeBERTa (New) | Improvement |
|--------|---------------|---------------|-------------|
| Mean Score | 0.469 | 0.60-0.70 | +28-49% |
| Perfect 1.0 Scores | 15-20% | <5% | -60% false positives |
| High Quality (≥0.7) | 32.2% | 40-50% | More accurate |
| Distribution | Bimodal | Uniform | Better spread |

#### **MS-MARCO Cross-Encoder** (Legacy, Backward Compatible)

**Model**: `ms-marco-MiniLM-L-6-v2` (23MB)
**Architecture**: Cross-encoder (query-document relevance ranking)
**Type**: Relative relevance assessment (query-dependent)

**Use Cases**:
- Legacy systems requiring MS-MARCO compatibility
- Relative ranking within search results
- When disk space is extremely limited (<100MB)

**Known Limitations**:
- ⚠️ **Self-matching bias** - Tag-based queries match content (~25% false positives)
- ⚠️ **Bimodal distribution** - Scores cluster at extremes (0.0 or 1.0)
- ⚠️ **Requires meaningful queries** - Empty queries return 0.0

**Performance**:
- CPU: 50-100ms per evaluation
- GPU (CUDA/MPS/DirectML): 10-20ms per evaluation
- Model size: 23MB

**Usage** (opt-in for legacy compatibility):
```bash
# Override to use MS-MARCO (not recommended)
export MCP_QUALITY_LOCAL_MODEL=ms-marco-MiniLM-L-6-v2
```

#### **Fallback Quality Scoring** (Best of Both Worlds) 🆕 v8.50.0

**Model**: DeBERTa + MS-MARCO hybrid approach
**Architecture**: Threshold-based fallback for optimal quality assessment
**Type**: Absolute quality with technical content rescue

**Problem Solved**:
DeBERTa has systematic bias toward prose/narrative content over technical documentation:
- **Prose content**: 0.78-0.92 scores (excellent)
- **Technical content**: 0.48-0.60 scores (undervalued)
- **Impact**: 95% of technical memories undervalued in technical databases

**Solution - Threshold-Based Fallback**:
```python
# Always score with DeBERTa first
deberta_score = score_with_deberta(content)

# If DeBERTa confident (high score), use it
if deberta_score >= DEBERTA_THRESHOLD:
    return deberta_score  # MS-MARCO not consulted (fast path)

# DeBERTa scored low - check if MS-MARCO thinks it's good (technical content)
ms_marco_score = score_with_ms_marco(content)
if ms_marco_score >= MS_MARCO_THRESHOLD:
    return ms_marco_score  # Rescue technical content

# Both agree it's low quality
return deberta_score
```

**Key Features**:
- ✅ **Signal Preservation** - No averaging dilution, uses strongest signal
- ✅ **DeBERTa Primary** - Eliminates self-matching bias for prose
- ✅ **MS-MARCO Rescue** - Catches undervalued technical content
- ✅ **Performance Optimized** - MS-MARCO only runs when DeBERTa scores low (~60% of memories)
- ✅ **Simple Configuration** - Just thresholds, no complex weights

**Expected Results**:
| Content Type | DeBERTa-Only | Fallback Mode | Improvement |
|--------------|--------------|---------------|-------------|
| **Technical content** | 0.48-0.60 | 0.70-0.80 | +45-65% ✅ |
| **Prose content** | 0.78-0.92 | 0.78-0.92 | No change ✅ |
| **High quality (≥0.7)** | 0.4% (17 memories) | 20-30% (800-1200) | 50-75x ✅ |

**Performance**:
| Scenario | DeBERTa-Only | Fallback | Time |
|----------|--------------|----------|------|
| **High prose** (~40% of memories) | DeBERTa only | DeBERTa only | 115ms ⚡ |
| **Low technical** (~35% of memories) | DeBERTa only | DeBERTa + MS-MARCO rescue | 155ms |
| **Low garbage** (~25% of memories) | DeBERTa only | DeBERTa + MS-MARCO both low | 155ms |
| **Average** | 115ms | 139ms | +21% acceptable overhead |

**Configuration**:
```bash
# Enable fallback mode (requires both models)
export MCP_QUALITY_FALLBACK_ENABLED=true
export MCP_QUALITY_LOCAL_MODEL="nvidia-quality-classifier-deberta,ms-marco-MiniLM-L-6-v2"

# Thresholds (recommended defaults)
export MCP_QUALITY_DEBERTA_THRESHOLD=0.6   # DeBERTa confidence threshold
export MCP_QUALITY_MSMARCO_THRESHOLD=0.7   # MS-MARCO rescue threshold

# Quality boost works well with fallback
export MCP_QUALITY_BOOST_ENABLED=true
export MCP_QUALITY_BOOST_WEIGHT=0.3
```

**Configuration Tuning**:
```bash
# Technical-heavy systems (more MS-MARCO rescues)
export MCP_QUALITY_DEBERTA_THRESHOLD=0.5   # Lower threshold → more rescues
export MCP_QUALITY_MSMARCO_THRESHOLD=0.6   # Lower bar for technical content

# Quality-focused (stricter standards)
export MCP_QUALITY_DEBERTA_THRESHOLD=0.7   # Higher threshold → DeBERTa must be very confident
export MCP_QUALITY_MSMARCO_THRESHOLD=0.8   # Higher bar for rescue

# Balanced (recommended default)
export MCP_QUALITY_DEBERTA_THRESHOLD=0.6
export MCP_QUALITY_MSMARCO_THRESHOLD=0.7
```

**Decision Tracking**:
Fallback mode stores detailed decision metadata:
```json
{
  "quality_score": 0.78,
  "quality_provider": "fallback_deberta-msmarco",
  "quality_components": {
    "decision": "ms_marco_rescue",
    "deberta_score": 0.52,
    "ms_marco_score": 0.78,
    "final_score": 0.78
  }
}
```

**Decision Types**:
- `deberta_confident`: DeBERTa ≥ threshold, used DeBERTa (fast path, ~40%)
- `ms_marco_rescue`: DeBERTa low, MS-MARCO ≥ threshold, used MS-MARCO (~35%)
- `both_low`: Both below thresholds, used DeBERTa (~25%)
- `deberta_only`: MS-MARCO not loaded, DeBERTa only
- `ms_marco_failed`: MS-MARCO error, used DeBERTa

**Re-scoring Existing Memories**:
```bash
# Dry-run (preview changes)
python scripts/quality/rescore_fallback.py

# Execute re-scoring
python scripts/quality/rescore_fallback.py --execute

# Custom thresholds
python scripts/quality/rescore_fallback.py --execute \
  --deberta-threshold 0.5 \
  --msmarco-threshold 0.6

# Expected time: 3-5 minutes for 4,000-5,000 memories
# Creates backup automatically in ~/backups/mcp-memory-service/
```

**When to Use Fallback Mode**:
| Use Case | Recommended Mode |
|----------|-----------------|
| **Technical documentation** | ✅ Fallback (rescues undervalued content) |
| **Mixed content** (code + prose) | ✅ Fallback (best of both) |
| **Prose-only** (creative writing) | DeBERTa-only (faster, same quality) |
| **Strict quality** (academic) | DeBERTa-only (no rescue for low scores) |
| **Legacy compatibility** | MS-MARCO-only (if required) |

### GPU Acceleration (Automatic)

Both models support automatic GPU detection:
- **CUDA** (NVIDIA GPUs)
- **CoreML/MPS** (Apple Silicon M1/M2/M3)
- **DirectML** (Windows DirectX)
- **ROCm** (AMD GPUs on Linux)
- **CPU** (fallback, always works)

```bash
# GPU detection is automatic
export MCP_QUALITY_LOCAL_DEVICE=auto  # Default (auto-detects best device)

# Or force specific device
export MCP_QUALITY_LOCAL_DEVICE=cpu    # Force CPU
export MCP_QUALITY_LOCAL_DEVICE=cuda   # Force CUDA (NVIDIA)
export MCP_QUALITY_LOCAL_DEVICE=mps    # Force MPS (Apple Silicon)
```

### Migration from MS-MARCO to DeBERTa

If you're upgrading from v8.48.x or earlier and want to re-evaluate existing memories:

```bash
# Export DeBERTa model (one-time, downloads 450MB)
python scripts/quality/export_deberta_onnx.py

# Re-evaluate all memories with DeBERTa
python scripts/quality/migrate_to_deberta.py

# Verify new distribution
curl -ks https://127.0.0.1:8000/api/quality/distribution | python3 -m json.tool
```

**Migration preserves**:
- Original MS-MARCO scores (in `quality_migration` metadata)
- Access patterns and timestamps
- User ratings and feedback

**Migration time**: ~10-20 minutes for 4,000-5,000 memories (depends on CPU/GPU)

## Installation & Setup

### 1. Basic Setup (Local SLM Only)

**Zero configuration required** - The quality system works out of the box with local SLM:

```bash
# Install MCP Memory Service (if not already installed)
pip install mcp-memory-service

# Quality system is enabled by default with local SLM
# No API keys needed, no external calls
```

### 2. Optional: Cloud APIs (Opt-In)

If you want cloud-based scoring (Groq or Gemini):

```bash
# Enable Groq API (fast, cheap)
export GROQ_API_KEY="your-groq-api-key"
export MCP_QUALITY_AI_PROVIDER=groq  # or "auto" to try all tiers

# Enable Gemini API (Google)
export GEMINI_API_KEY="your-gemini-api-key"
export MCP_QUALITY_AI_PROVIDER=gemini
```

### 3. Homelab: score with your own LLM

`MCP_QUALITY_AI_PROVIDER=openai-compatible` points scoring at any OpenAI-compatible
`/v1` endpoint — Ollama, vLLM, or a LiteLLM proxy — instead of ONNX or a cloud API. No
API key leaves the machine, no per-call cost.

```bash
export MCP_QUALITY_AI_PROVIDER=openai-compatible
export MCP_QUALITY_AI_BASE_URL=http://localhost:11434/v1   # Ollama
export MCP_QUALITY_AI_MODEL=qwen2.5:7b-instruct
# export MCP_QUALITY_AI_API_KEY=ollama                     # optional
```

Both `MCP_QUALITY_AI_BASE_URL` and `MCP_QUALITY_AI_MODEL` are required with this
provider; the service refuses to start without them. Recommended models:
`qwen2.5:7b-instruct` (Ollama), `mlx-community/Qwen2.5-7B-Instruct-4bit` (MLX), or any
instruct model behind a LiteLLM proxy. If the endpoint fails, scoring falls back to
implicit signals automatically.

### 4. Configuration Options

```bash
# Quality System Core
export MCP_QUALITY_SYSTEM_ENABLED=true         # Default: true
export MCP_QUALITY_AI_PROVIDER=local           # local|openai-compatible|groq|gemini|auto|none

# Local SLM Configuration (Tier 1)
export MCP_QUALITY_LOCAL_MODEL=ms-marco-MiniLM-L-6-v2  # Model name
export MCP_QUALITY_LOCAL_DEVICE=auto           # auto|cpu|cuda|mps|directml

# Quality-Boosted Search (Opt-In) — affects search ranking only
export MCP_QUALITY_BOOST_ENABLED=false         # Default: false (opt-in)
export MCP_QUALITY_BOOST_WEIGHT=0.3            # 0.0-1.0 (30% quality, 70% semantic)

# Store-time blend of implicit signals into the saved score.
# Both default to the MCP_QUALITY_BOOST_* values above when unset, which is how
# this worked before it was split out. Note the direction is the opposite of the
# search weight: a higher value means LESS weight on the classifier, and a fresh
# memory has weak implicit signals by construction, so a high value caps the
# score of newly stored memories regardless of content.
export MCP_QUALITY_IMPLICIT_BLEND_ENABLED=false  # Default: follows BOOST_ENABLED
export MCP_QUALITY_IMPLICIT_WEIGHT=0.3           # 0.0-1.0 implicit share of the stored score

# Quality-Based Retention (Consolidation)
export MCP_QUALITY_RETENTION_HIGH=365          # Days for quality ≥0.7
export MCP_QUALITY_RETENTION_MEDIUM=180        # Days for quality 0.5-0.7
export MCP_QUALITY_RETENTION_LOW_MIN=30        # Min days for quality <0.5
export MCP_QUALITY_RETENTION_LOW_MAX=90        # Max days for quality <0.5
```

## Using the Quality System

### 1. Automatic Quality Scoring

Quality scores are calculated automatically when memories are retrieved:

```bash
# Normal retrieval - quality scoring happens in background
claude /memory-recall "what did I work on yesterday"

# Quality score is updated in metadata (non-blocking)
```

### 2. Manual Rating (Optional)

Override AI scores with manual ratings:

```bash
# Rate a memory (MCP tool)
memory_quality(
    content_hash="abc123...",
    rating=1,  # -1 (bad), 0 (neutral), 1 (good)
    feedback="This was very helpful!"
)

# Manual ratings weighted 60%, AI scores weighted 40%
```

**HTTP API**:
```bash
curl -X POST http://127.0.0.1:8000/api/quality/memories/{hash}/rate \
  -H "Content-Type: application/json" \
  -d '{"rating": 1, "feedback": "Helpful!"}'
```

### 3. Quality-Boosted Search

Enable quality-based reranking for better results:

**Method 1: Global Configuration**
```bash
export MCP_QUALITY_BOOST_ENABLED=true
claude /memory-recall "search query"  # Uses quality boost
```

**Method 2: Per-Query (MCP Tool)**
```bash
# Search with quality boost (MCP tool)
retrieve_with_quality_boost(
    query="search query",
    n_results=10,
    quality_weight=0.3  # 30% quality, 70% semantic
)
```

**Algorithm**:
1. Over-fetch 3× candidates (30 results for top 10)
2. Rerank by: `0.7 × semantic_similarity + 0.3 × quality_score`
3. Return top N results

**Performance**: <100ms total (50ms semantic search + 20ms reranking + 30ms quality scoring)

### 4. View Quality Metrics

**MCP Tool**:
```bash
get_memory_quality(content_hash="abc123...")

# Returns:
# - quality_score: Current composite score (0.0-1.0)
# - quality_provider: Which tier scored it (ONNXRankerModel, etc.)
# - access_count: Number of retrievals
# - last_accessed_at: Last access timestamp
# - ai_scores: Historical AI evaluation scores
# - user_rating: Manual rating if present
```

**HTTP API**:
```bash
curl http://127.0.0.1:8000/api/quality/memories/{hash}
```

### 5. Quality Analytics

**MCP Tool**:
```bash
analyze_quality_distribution(min_quality=0.0, max_quality=1.0)

# Returns:
# - total_memories: Total count
# - high_quality_count: Score ≥0.7
# - medium_quality_count: 0.5 ≤ score < 0.7
# - low_quality_count: Score < 0.5
# - average_score: Mean quality score
# - provider_breakdown: Count by provider
# - top_10_memories: Highest scoring
# - bottom_10_memories: Lowest scoring
```

**Dashboard** (http://127.0.0.1:8000/):
- Quality badges on all memory cards (color-coded by tier)
- Analytics view with distribution charts
- Provider breakdown pie chart
- Top/bottom performers lists

## Quality-Based Memory Management

### 1. Quality-Based Forgetting (Consolidation)

High-quality memories are preserved longer during consolidation:

| Quality Tier | Score Range | Retention Period |
|--------------|-------------|------------------|
| **High** | ≥0.7 | 365 days inactive |
| **Medium** | 0.5-0.7 | 180 days inactive |
| **Low** | <0.5 | 30-90 days inactive (scaled by score) |

**How it works**:
- Weekly consolidation scans inactive memories
- Applies quality-based thresholds
- Archives low-quality memories sooner
- Preserves high-quality memories longer

### 2. Quality-Weighted Decay

High-quality memories decay slower in relevance scoring:

```
decay_multiplier = 1.0 + (quality_score × 0.5)
# High quality (0.9): 1.45× multiplier
# Medium quality (0.5): 1.25× multiplier
# Low quality (0.2): 1.10× multiplier

final_relevance = base_relevance × decay_multiplier
```

**Effect**: High-quality memories stay relevant longer in search results.

## Privacy & Cost

### Privacy Modes

| Mode | Configuration | Privacy | Cost |
|------|---------------|---------|------|
| **Local Only** | `MCP_QUALITY_AI_PROVIDER=local` | ✅ Full (no external calls) | $0 |
| **Hybrid** | `MCP_QUALITY_AI_PROVIDER=auto` | ⚠️ Cloud fallback | ~$0.30/mo |
| **Cloud** | `MCP_QUALITY_AI_PROVIDER=groq` | ❌ External API | ~$0.30/mo |
| **Implicit Only** | `MCP_QUALITY_AI_PROVIDER=none` | ✅ Full (no AI) | $0 |

### Cost Comparison (3500 memories, 100 retrievals/day)

| Provider | Monthly Cost | Notes |
|----------|--------------|-------|
| **Local SLM** | **$0** | Free forever, runs locally |
| Groq (Kimi K2) | ~$0.30-0.50 | Fast, good quality |
| Gemini Flash | ~$0.40-0.60 | Slower, free tier available |
| Implicit Only | $0 | No AI scoring, usage patterns only |

**Recommendation**: Use default local SLM (zero cost, full privacy, fast).

## Performance Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Local SLM Scoring (CPU)** | 50-100ms | Per memory evaluation |
| **Local SLM Scoring (GPU)** | 10-20ms | With CUDA/MPS/DirectML |
| **Quality-Boosted Search** | <100ms | Over-fetch + rerank |
| **Implicit Signals** | <10ms | Always fast |
| **Quality Metadata Update** | <5ms | Storage backend write |

**Target Metrics**:
- Quality calculation overhead: <10ms
- Search latency with boost: <100ms total
- No user-facing blocking (async scoring)

## Troubleshooting

### Local SLM Not Working

**Symptom**: `quality_provider: ImplicitSignalsEvaluator` (should be `ONNXRankerModel`)

**Fixes**:
1. Check ONNX Runtime installed:
   ```bash
   pip install onnxruntime
   ```

2. Check model downloaded:
   ```bash
   ls ~/.cache/mcp_memory/onnx_models/ms-marco-MiniLM-L-6-v2/
   # Should contain: model.onnx, tokenizer.json
   ```

   If `MCP_QUALITY_ONNX_MODEL_DIR` is set, look there instead — it replaces the
   `~/.cache/mcp_memory/onnx_models` parent and the loader appends the model
   name. In a container that is normally the case; see
   [`tools/docker/README.md`](../../tools/docker/README.md).

3. Check logs for errors:
   ```bash
   tail -f logs/mcp_memory_service.log | grep quality
   ```

### Quality Scores Always 0.5

**Symptom**: All memories have `quality_score: 0.5` (neutral default)

**Cause**: Quality scoring not triggered yet (memories haven't been retrieved)

**Fix**: Retrieve memories to trigger scoring:
```bash
claude /memory-recall "any search query"
# Quality scoring happens in background after retrieval
```

### GPU Not Detected

**Symptom**: Local SLM uses CPU despite having GPU

**Fixes**:
1. Install GPU-enabled ONNX Runtime:
   ```bash
   # NVIDIA CUDA
   pip install onnxruntime-gpu

   # DirectML (Windows)
   pip install onnxruntime-directml
   ```

2. Force device selection:
   ```bash
   export MCP_QUALITY_LOCAL_DEVICE=cuda  # or mps, directml
   ```

### Quality Boost Not Working

**Symptom**: Search results don't show quality reranking

**Checks**:
1. Verify enabled:
   ```bash
   echo $MCP_QUALITY_BOOST_ENABLED  # Should be "true"
   ```

2. Use explicit MCP tool:
   ```bash
   retrieve_with_quality_boost(query="test", quality_weight=0.5)
   ```

3. Check debug info in results:
   ```python
   result.debug_info['reranked']  # Should be True
   result.debug_info['quality_score']  # Should exist
   ```

## Best Practices

### 1. Start with Defaults (Updated for v8.48.3)

Use local SLM (default) for:
- Zero cost
- Full privacy
- Offline capability
- **Good for relative ranking** (not absolute quality assessment)

**Important**: Local SLM scores should be **combined with implicit signals** (access patterns, recency) for best results. Do not rely on ONNX scores alone.

### 2. Enable Quality Boost Gradually

```bash
# Week 1: Collect quality scores (boost disabled)
export MCP_QUALITY_BOOST_ENABLED=false

# Week 2: Test with low weight
export MCP_QUALITY_BOOST_ENABLED=true
export MCP_QUALITY_BOOST_WEIGHT=0.2  # 20% quality

# Week 3+: Increase if helpful
export MCP_QUALITY_BOOST_WEIGHT=0.3  # 30% quality (recommended)
```

### 3. Monitor Quality Distribution (With Caution)

Check analytics regularly, but interpret with awareness of limitations:
```bash
analyze_quality_distribution()

# Current observed distribution (v8.48.3):
# - High quality (≥0.7): 32.2% (includes ~25% false positives from self-matching)
# - Medium quality (0.5-0.7): 27.4%
# - Low quality (<0.5): 40.4%

# Ideal distribution (future with hybrid scoring):
# - High quality (≥0.7): 20-30% of memories
# - Medium quality (0.5-0.7): 50-60%
# - Low quality (<0.5): 10-20%
```

**Warning**: High scores may not always indicate high quality due to self-matching bias. Validate manually before making retention decisions.

### 4. Manual Rating for Edge Cases

Rate important memories manually:
```bash
# After finding a very helpful memory
memory_quality(content_hash="abc123...", rating=1, feedback="Critical info!")

# After finding unhelpful memory
memory_quality(content_hash="def456...", rating=-1, feedback="Outdated")
```

Manual ratings weighted 60%, AI scores 40%.

### 5. Periodic Review (Updated for v8.48.3)

Monthly checklist:
- [ ] Check quality distribution (analytics dashboard)
- [ ] **Manually validate** top 10 performers (verify genuinely helpful, not just self-matching bias)
- [ ] **Manually validate** bottom 10 before deletion (may be low-quality or just poor matches)
- [ ] Verify provider breakdown (target: 75%+ local SLM)
- [ ] Check average quality score (current: 0.469, target with hybrid: 0.6+)
- [ ] **Review Issue #268** progress on quality improvements

## Advanced Configuration

### Custom Retention Policy

```bash
# Conservative: Preserve longer
export MCP_QUALITY_RETENTION_HIGH=730       # 2 years for high quality
export MCP_QUALITY_RETENTION_MEDIUM=365     # 1 year for medium
export MCP_QUALITY_RETENTION_LOW_MIN=90     # 90 days minimum for low

# Aggressive: Archive sooner
export MCP_QUALITY_RETENTION_HIGH=180       # 6 months for high
export MCP_QUALITY_RETENTION_MEDIUM=90      # 3 months for medium
export MCP_QUALITY_RETENTION_LOW_MIN=14     # 2 weeks minimum for low
```

### Custom Quality Boost Weight

```bash
# Semantic-first (default)
export MCP_QUALITY_BOOST_WEIGHT=0.3  # 30% quality, 70% semantic

# Balanced
export MCP_QUALITY_BOOST_WEIGHT=0.5  # 50% quality, 50% semantic

# Quality-first
export MCP_QUALITY_BOOST_WEIGHT=0.7  # 70% quality, 30% semantic
```

**Recommendation**: Start with 0.3, increase if quality boost improves results.

### Hybrid Cloud Strategy

Use local SLM primarily, cloud APIs as fallback:

```bash
export MCP_QUALITY_AI_PROVIDER=auto  # Try all available tiers
export GROQ_API_KEY="your-key"       # Groq as Tier 2 fallback
```

**Behavior**:
1. Try local SLM (99% success rate)
2. If fails, try Groq API
3. If fails, try Gemini API
4. Ultimate fallback: Implicit signals only

## Success Metrics (Phase 1 Targets)

From Issue #260 and #261 roadmap:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Retrieval Precision** | >70% useful (top-5) | Up from ~50% baseline |
| **Quality Coverage** | >30% memories scored | Within 3 months |
| **Quality Distribution** | 20-30% high-quality | Pareto principle |
| **Search Latency** | <100ms with boost | SQLite-vec backend |
| **Monthly Cost** | <$0.50 or $0 | Groq API or local SLM |
| **Local SLM Usage** | >95% of scoring | Tier 1 success rate |

## FAQ

### Q: Do I need API keys for the quality system?

**A**: No! The default local SLM works with zero configuration, no API keys, and no external calls.

### Q: How much does it cost?

**A**: $0 with the default local SLM. Optional cloud APIs cost ~$0.30-0.50/month for typical usage.

### Q: Does quality scoring slow down searches?

**A**: No. Scoring happens asynchronously in the background. Quality-boosted search adds <20ms overhead.

### Q: Can I disable the quality system?

**A**: Yes, set `MCP_QUALITY_SYSTEM_ENABLED=false`. System works normally without quality scores.

### Q: How accurate is the local SLM?

**A**: 80%+ correlation with human quality ratings. Good enough for ranking and retention decisions.

### Q: What if the local SLM fails to download?

**A**: System falls back to implicit signals (access patterns). No failures, degraded gracefully.

### Q: Can I use my own quality scoring model?

**A**: Yes! Implement the `QualityEvaluator` interface and configure via `MCP_QUALITY_AI_PROVIDER`.

### Q: Does this work offline?

**A**: Yes! Local SLM works fully offline. No internet required for quality scoring.

## Related Documentation

- [Issue #260](https://github.com/doobidoo/mcp-memory-service/issues/260) - Quality System Specification
- [Issue #261](https://github.com/doobidoo/mcp-memory-service/issues/261) - Roadmap (Quality → Agentic RAG)
- [Consolidation Guide](./memory-consolidation-guide.md) - Detailed consolidation documentation

## Changelog

**v8.48.3** (2025-12-08):
- **Documentation Update**: Added comprehensive ONNX limitations section
- Documented self-matching bias and bimodal distribution issues
- Updated best practices with manual validation recommendations
- Added performance benchmarks from evaluation (4,762 memories)
- Linked to [Quality System Evaluation Report](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Quality-System-Evaluation)
- Created [Issue #268](https://github.com/doobidoo/mcp-memory-service/issues/268) for Phase 2 improvements

**v8.45.0** (2025-01-XX):
- Initial release of Memory Quality System
- Local SLM (ONNX) as primary tier
- Quality-based forgetting in consolidation
- Quality-boosted search with reranking
- Dashboard UI with quality badges and analytics
- Comprehensive MCP tools and HTTP API

---

**Need help?** Open an issue at https://github.com/doobidoo/mcp-memory-service/issues

**Quality System Improvements**: See [Issue #268](https://github.com/doobidoo/mcp-memory-service/issues/268) for planned enhancements
