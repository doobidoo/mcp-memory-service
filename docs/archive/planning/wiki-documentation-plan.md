# Wiki Documentation Plan - Memory Quality System Era (v8.45+)

## Overview

This document outlines the comprehensive wiki documentation needed to cover the major improvements introduced between v8.45.0 and v8.48.0, representing the **Memory Quality System Era** - a transformative upgrade to MCP Memory Service.

## Timeline Summary

- **v8.45.0** (Dec 5, 2025): Memory Quality System Launch
- **v8.45.1-v8.45.3** (Dec 5-6): Quality System stabilization and infrastructure
- **v8.46.0-v8.46.3** (Dec 6): Quality + Hooks integration
- **v8.47.0-v8.47.1** (Dec 6-7): Association-based quality boost and ONNX improvements
- **v8.48.0** (Dec 7): CSV-based metadata compression

**Total**: 11 releases in 3 days representing ~40 major features/fixes

## Proposed Wiki Structure

### 1. **Memory Quality System Guide** (New Page)
**URL**: `wiki/Memory-Quality-System-Guide`

**Sections**:
- Overview and Philosophy (Memento-inspired design)
- Architecture
  - Multi-tier provider system (Local SLM → Groq → Gemini → Implicit)
  - Local-first design with ONNX (ms-marco-MiniLM-L-6-v2)
  - Zero-cost, privacy-preserving evaluation
- Features
  - Automatic quality scoring (0.0-1.0)
  - Quality-based forgetting and retention tiers
  - Quality-weighted decay
  - Quality-boosted search (opt-in)
  - Association-based quality boost (v8.47.0)
- Configuration
  - Environment variables (10 config options)
  - Provider selection (local/groq/gemini/auto)
  - Device selection (CPU/CUDA/MPS/DirectML/ROCm)
  - Quality boost settings
- Platform Support
  - Windows (CUDA, DirectML)
  - macOS (MPS)
  - Linux (CUDA, ROCm)
- Performance Benchmarks
  - Latency: 50-100ms CPU, 10-20ms GPU
  - Accuracy metrics
  - Cost comparison ($0 local vs cloud APIs)
- Usage Examples
  - MCP tools (rate_memory, get_memory_quality, analyze_quality_distribution, retrieve_with_quality_boost)
  - HTTP API endpoints (4 new REST endpoints)
  - Dashboard UI (quality badges, analytics view)
- Troubleshooting
  - ONNX model loading issues
  - Quality score persistence (v8.46.3 fix)
  - Windows-specific considerations

### 2. **Quality + Hooks Integration Guide** (New Page)
**URL**: `wiki/Quality-Hooks-Integration`

**Sections**:
- Integration Overview (3-phase approach from v8.46.0)
  - Phase 1: Hooks read backendQuality from metadata (20% scoring weight)
  - Phase 2: Session-end hook triggers async quality evaluation
  - Phase 3: Quality-boosted search integration
- Hook Scoring Weights
  - timeDecay: 20%
  - tagRelevance: 30%
  - contentRelevance: 10%
  - contentQuality: 20%
  - backendQuality: 20%
- Quality Evaluation Endpoint
  - POST `/api/quality/memories/{hash}/evaluate`
  - Performance: ~355ms with ONNX ranker
  - Non-blocking with 10s timeout
- Implementation Details
  - calculateBackendQuality() in memory-scorer.js
  - triggerQualityEvaluation() in session-end.js
  - queryMemories() qualityBoost option
- Cross-Platform Considerations
  - Windows session-start hook crash fix (v8.46.2)
  - Windows installer encoding fix (v8.46.1)

### 3. **Metadata Compression System** (New Page)
**URL**: `wiki/Metadata-Compression-System`

**Sections**:
- Problem Statement
  - Cloudflare D1 10KB metadata limit
  - Quality/consolidation metadata size explosion
  - Sync failure impact (400 Bad Request errors)
- Solution: CSV-Based Compression (Phase 1)
  - Architecture (metadata_codec.py)
  - CSV encoding/decoding implementation
  - Provider code mapping (onnx_local → ox, etc.)
  - 78% size reduction (732B → 159B typical)
- Metadata Validation
  - Pre-sync size checks (<9.5KB threshold)
  - Prevents API failures before they occur
  - Implementation in hybrid.py (lines 547-559)
- Transparent Operation
  - Automatic compression on write
  - Automatic decompression on read
  - Zero user-facing impact
  - Backward compatibility guaranteed
- Quality Metadata Optimizations
  - ai_scores history limited (10 → 3 most recent)
  - quality_components removed from sync (debug-only)
  - Cloudflare-specific field suppression
- Performance Impact
  - <1ms overhead per operation
  - 100% sync success rate (0 failures)
  - 3,750 ONNX-scored memories verified
- Future Roadmap (3-phase plan)
  - Phase 1: CSV compression (COMPLETE)
  - Phase 2: Binary encoding with struct/msgpack (85-90% target)
  - Phase 3: Reference-based deduplication
- Verification
  - verify_compression.sh script
  - Round-trip accuracy testing
  - Compression ratio measurement

### 4. **ONNX Quality Evaluation Deep Dive** (New Page)
**URL**: `wiki/ONNX-Quality-Evaluation`

**Sections**:
- ONNX Ranker Overview
  - Model: ms-marco-MiniLM-L-6-v2 cross-encoder
  - Size: 23MB
  - Performance: 7-16ms per memory (CPU)
- Critical Understanding: Cross-Encoder Design
  - Scores query-memory relevance, NOT absolute quality
  - Requires meaningful query-memory pairs
  - Self-matching queries produce artificially high scores
- ONNX Self-Match Bug (v8.47.1)
  - Problem: Using memory content as its own query
  - Result: Artificially inflated scores (~1.0 for all)
  - Fix: Generate queries from tags/metadata (what memory is *about*)
  - Realistic distribution: avg 0.468 (42.9% high, 3.2% medium, 53.9% low)
- Association Pollution (v8.47.1)
  - Problem: System-generated associations/clusters scored
  - Fix: Filter type='association' and type='compressed_cluster'
  - Impact: 948 system-generated memories excluded
- Model Export and Loading (v8.45.3)
  - Dynamic export from transformers to ONNX on first use
  - Offline mode support (local_files_only=True)
  - Air-gapped environment compatibility
  - Export location: ~/.cache/mcp_memory/onnx_models/
- Bulk Evaluation
  - scripts/quality/bulk_evaluate_onnx.py
  - Association filtering
  - Sync monitoring with queue size reporting
  - Progress reporting and sync completion waiting
- Reset and Recovery
  - scripts/quality/reset_onnx_scores.py
  - Reset all scores to implicit defaults (0.5)
  - Pauses sync during reset
  - Use case: Recover from bad evaluation

### 5. **Hybrid Backend Enhancements** (Update Existing Page)
**URL**: `wiki/Hybrid-Backend-Guide` (update)

**New Sections to Add**:
- Sync Queue Overflow Handling (v8.47.1)
  - Problem: Queue capacity (1,000) overwhelmed by bulk operations (4,478 updates)
  - Result: 278 Cloudflare sync failures (27.8% rate)
  - Fix: Queue size increased to 2,000 (`MCP_HYBRID_QUEUE_SIZE`)
  - Fix: Batch size increased to 100 (`MCP_HYBRID_BATCH_SIZE`)
  - 5-second timeout with fallback to immediate sync on queue full
  - wait_for_sync_completion() method for monitoring
  - Result: 0% sync failure rate
- Metadata Normalization for Cloudflare (v8.46.3)
  - _normalize_metadata_for_cloudflare() helper function
  - Separates top-level keys from custom metadata fields
  - Wraps custom fields in 'metadata' key
  - Idempotent operation
- Quality Score Persistence (v8.46.3)
  - Fixed scores not persisting to Cloudflare
  - Metadata structure expectations (wrapped vs top-level)
  - Quality API metadata handling
  - SyncOperation preserve_timestamps flag
- Sync Pause/Resume Enhancements (v8.47.1)
  - _sync_paused flag prevents enqueuing during pause
  - Fixed race condition where operations enqueued while paused
  - Ensures operations not lost during consolidation/bulk updates
- CSV Metadata Compression Integration (v8.48.0)
  - compress_metadata_for_sync() in sync pipeline
  - decompress_metadata_from_sync() on retrieval
  - Size validation before Cloudflare API calls
  - 4 memory construction points in cloudflare.py

### 6. **Consolidation System Updates** (Update Existing Page)
**URL**: `wiki/Memory-Consolidation-System-Guide` (update)

**New Sections to Add**:
- Association-Based Quality Boost (v8.47.0)
  - Well-connected memories (≥5 connections) get 20% quality boost
  - Network effect: frequently referenced = likely more valuable
  - Configuration:
    - MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED (default: true)
    - MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST (default: 5, range: 1-100)
    - MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR (default: 1.2, range: 1.0-2.0)
  - Quality scores capped at 1.0
  - Full metadata audit trail:
    - quality_boost_applied (boolean)
    - quality_boost_date (ISO timestamp)
    - quality_boost_reason ("association_connections")
    - quality_boost_connection_count
    - original_quality_before_boost
  - Impact: ~4% relevance increase, potential retention tier promotion
  - 5 comprehensive test cases (100% pass rate)
- Batch Update Optimization (v8.47.1)
  - Problem: Sequential update_memory() calls during consolidation
  - Fix: Collect updates and use single update_memories_batch() transaction
  - Impact: 50-100x speedup for relevance score updates
  - Location: consolidation/consolidator.py
- Quality-Weighted Decay
  - High-quality memories decay 3x slower than low-quality
  - Integration with exponential decay calculation
  - Quality boost applied before quality multiplier calculation

### 7. **Dashboard UI Enhancements** (New Page)
**URL**: `wiki/Dashboard-UI-Guide`

**Sections**:
- Quality System UI Features (v8.45.0)
  - Quality badges on memory cards (color-coded: green/yellow/red/gray)
  - Analytics view with distribution charts
  - Bar chart for quality distribution (high/medium/low counts)
  - Pie chart for provider breakdown (local/groq/gemini/implicit)
  - Top/bottom performers lists
  - Settings panel for quality configuration
  - i18n support (English + Chinese)
- Dark Mode Improvements (v8.45.2)
  - Form controls and select elements dark mode fix
  - Global .form-control and .form-select overrides
  - Quality tab chart contrast improvements
  - Chart.js dark mode support (dynamic color configuration)
  - Quality distribution and provider chart updates
  - .view-btn hover states
- Performance Benchmarks
  - 25ms page load
  - <100ms search operations
  - <2s for all UI operations
- Accessibility
  - Mobile responsive (768px, 1024px breakpoints)
  - WCAG compliance considerations

## Implementation Timeline

### Week 1: Core Documentation
- [ ] Create "Memory Quality System Guide" (Priority: CRITICAL)
- [ ] Create "ONNX Quality Evaluation Deep Dive" (Priority: HIGH)
- [ ] Create "Metadata Compression System" (Priority: HIGH)

### Week 2: Integration Documentation
- [ ] Create "Quality + Hooks Integration Guide" (Priority: MEDIUM)
- [ ] Update "Hybrid Backend Guide" (Priority: MEDIUM)
- [ ] Update "Memory Consolidation System Guide" (Priority: MEDIUM)

### Week 3: UI and Polish
- [ ] Create "Dashboard UI Guide" (Priority: LOW)
- [ ] Add cross-references between all pages
- [ ] Create visual diagrams (architecture, flow charts)
- [ ] Add screenshots from dashboard

## Cross-Reference Strategy

Each wiki page should link to related pages:
- Memory Quality System Guide → ONNX Deep Dive, Quality+Hooks Integration
- ONNX Deep Dive → Memory Quality System Guide, Troubleshooting
- Metadata Compression → Hybrid Backend Guide
- Quality+Hooks Integration → Memory Quality System Guide, Hooks Quick Reference
- Hybrid Backend Guide → Metadata Compression, Consolidation Guide
- Consolidation Guide → Memory Quality System Guide, Association Quality Boost

## Visual Assets Needed

1. **Architecture Diagrams**:
   - Multi-tier quality provider system flowchart
   - Metadata compression pipeline (encode → validate → sync → decode)
   - Quality-boosted search reranking algorithm
   - Association-based quality boost decision tree

2. **Screenshots**:
   - Dashboard quality badges (light + dark mode)
   - Analytics view (distribution + provider charts)
   - Quality settings panel
   - Top/bottom performers lists

3. **Comparison Tables**:
   - Provider comparison (local vs Groq vs Gemini)
   - Performance benchmarks (CPU vs GPU, different platforms)
   - Compression ratio comparison (Phase 1/2/3)
   - Quality retention tiers (high/medium/low)

## Success Metrics

- **Documentation Coverage**: 100% of v8.45+ features documented
- **Cross-References**: Every page linked to ≥2 related pages
- **Visual Assets**: ≥1 diagram or screenshot per page
- **User Feedback**: <5% of support requests related to documented features
- **Community Engagement**: Wiki pages referenced in ≥10 GitHub issues/discussions

## Notes

- All existing documentation files referenced in CLAUDE.md should be preserved
- Wiki pages should be created in GitHub wiki, not in docs/ folder
- Each wiki page should have "Last Updated" timestamp
- Include version compatibility notes (e.g., "Available since v8.45.0")
- Add "See Also" sections for related topics
