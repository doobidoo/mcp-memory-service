# Memory Quality System - Detailed Reference

**Quick Summary for CLAUDE.md**: See main file for architecture overview. This file contains implementation details, configuration options, and troubleshooting.

## Complete Configuration Options

```bash
# Quality System (Local-First Defaults)
MCP_QUALITY_SYSTEM_ENABLED=true         # Default: enabled
MCP_QUALITY_AI_PROVIDER=local           # local|openai-compatible|groq|gemini|auto|none
MCP_QUALITY_LOCAL_MODEL=nvidia-quality-classifier-deberta  # Default v8.49.0+
MCP_QUALITY_LOCAL_DEVICE=auto           # auto|cpu|cuda|mps|directml

# OpenAI-compatible provider (homelab / self-hosted LLM — no cloud key required)
# MCP_QUALITY_AI_PROVIDER=openai-compatible
# MCP_QUALITY_AI_BASE_URL=http://localhost:11434/v1   # required
# MCP_QUALITY_AI_MODEL=qwen2.5:7b-instruct            # required
# MCP_QUALITY_AI_API_KEY=ollama                       # optional

# Legacy model (backward compatible, not recommended)
# MCP_QUALITY_LOCAL_MODEL=ms-marco-MiniLM-L-6-v2

# Quality-Boosted Search (Recommended with DeBERTa)
MCP_QUALITY_BOOST_ENABLED=true          # More accurate with DeBERTa
MCP_QUALITY_BOOST_WEIGHT=0.3            # 0.3 = 30% quality, 70% semantic

# Quality-Based Retention
MCP_QUALITY_RETENTION_HIGH=365          # Days for quality ≥0.7
MCP_QUALITY_RETENTION_MEDIUM=180        # Days for 0.5-0.7
MCP_QUALITY_RETENTION_LOW_MIN=30        # Min days for <0.5
```

## MCP Tools

- `rate_memory(content_hash, rating, feedback)` - Manual quality rating (-1/0/1)
- `get_memory_quality(content_hash)` - Retrieve quality metrics
- `analyze_quality_distribution(min_quality, max_quality)` - System-wide analytics
- `retrieve_with_quality_boost(query, n_results, quality_weight)` - Quality-boosted search

## Migration from MS-MARCO to DeBERTa

**Why Migrate:**
- ✅ Eliminates self-matching bias (no query needed)
- ✅ Uniform distribution (mean 0.60-0.70 vs 0.469)
- ✅ Fewer false positives (<5% perfect scores vs 20%)
- ✅ Absolute quality assessment vs relative ranking

**Migration Guide**: See [docs/guides/memory-quality-guide.md](../../docs/guides/memory-quality-guide.md#migration-from-ms-marco-to-deberta)

## Success Metrics (Phase 1 - v8.48.3)

**Achieved:**
- ✅ <100ms search latency with quality boost (45ms avg, +17% overhead)
- ✅ $0 monthly cost (local SLM default)
- ✅ 75% local SLM usage (3,570 of 4,762 memories)
- ✅ 95% quality score coverage

**Challenges:**
- ⚠️ Average score 0.469 (target: 0.6+)
- ⚠️ Self-matching bias ~25%
- ⚠️ Quality boost minimal ranking improvement (0-3%)

**Next Phase**: See [Issue #268](https://github.com/doobidoo/mcp-memory-service/issues/268)

## Troubleshooting

See [docs/guides/memory-quality-guide.md](../../docs/guides/memory-quality-guide.md) for:
- Model download issues
- Performance tuning
- Quality score interpretation
- User feedback integration
