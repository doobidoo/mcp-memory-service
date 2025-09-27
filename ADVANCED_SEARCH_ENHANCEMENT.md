# Advanced Hybrid Search Enhancement Specification

**Version**: 1.0  
**Date**: 2025-09-20  
**Status**: Design Phase  
**Priority**: High Enhancement  

## Executive Summary

This document specifies an enterprise-grade hybrid search enhancement that combines semantic vector search with traditional keyword search, content consolidation, and intelligent relationship mapping. The enhancement transforms the MCP Memory Service from a basic search tool into an intelligent knowledge consolidation system.

## Current State Analysis

### Existing Search Capabilities
- **Semantic Search**: Vector-based similarity using sentence transformers
- **Tag Search**: Filter by tags with AND/OR operations  
- **Time Search**: Natural language time-based filtering
- **Similar Search**: Find memories similar to a known content hash

### Current Limitations
1. **Single Search Mode**: Only one search method per query
2. **No Content Relationships**: Results are isolated, no contextual connections
3. **Limited Query Intelligence**: No query expansion or intent detection
4. **Basic Ranking**: Simple similarity scores without multi-signal ranking
5. **No Consolidation**: Cannot automatically group related content

## Enhancement Objectives

### Primary Goals
1. **Hybrid Search**: Combine semantic and keyword search for optimal recall and precision
2. **Content Consolidation**: Automatically group related memories into coherent topics
3. **Intelligent Ranking**: Multi-signal ranking using semantic, keyword, recency, and metadata signals
4. **Relationship Mapping**: Build connections between memories (solutions, context, timeline)
5. **Query Enhancement**: Intelligent query expansion and filter suggestion

### Enterprise Features
- **Project Consolidation**: Automatically gather all content about specific projects
- **Timeline Intelligence**: Build chronological narratives from memory fragments
- **Solution Mapping**: Connect problems with their solutions automatically
- **Context Enrichment**: Include supporting documentation and background information

## Technical Architecture

### 1. Service Layer Enhancement

**New MemoryService Methods:**

```python
class MemoryService:
    # Core hybrid search
    async def enhanced_search(
        self, query: str, search_mode: str = "hybrid",
        consolidate_related: bool = True, **kwargs
    ) -> Dict[str, Any]:

    # Content relationship building
    async def build_content_relationships(
        self, memories: List[Memory]
    ) -> Dict[str, Any]:

    # Query intelligence
    async def intelligent_query_expansion(
        self, query: str, user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:

    # Project consolidation
    async def consolidate_project_content(
        self, project_identifier: str, depth: str = "deep"
    ) -> Dict[str, Any]:
```

### 2. Storage Layer Enhancement

**Required Storage Backend Updates:**

```python
# Add to MemoryStorage base class
async def keyword_search(
    self, query: str, n_results: int = 10
) -> List[MemoryQueryResult]:

async def combined_search(
    self, semantic_query: str, keyword_query: str,
    weights: Dict[str, float]
) -> List[MemoryQueryResult]:

async def get_related_memories(
    self, memory: Memory, relationship_types: List[str]
) -> Dict[str, List[Memory]]:
```

**Implementation by Backend:**
- **SQLite-Vec**: FTS5 full-text search + BM25 scoring
- **ChromaDB**: Native hybrid search capabilities
- **Cloudflare**: Vectorize + D1 full-text search combination

### 3. API Enhancement

**New REST Endpoints:**

```http
POST /api/search/advanced          # Main hybrid search endpoint
POST /api/search/consolidate       # Content consolidation
GET  /api/projects/{id}/overview   # Project content consolidation
POST /api/search/intelligence      # Query analysis and enhancement
```

**Enhanced MCP Tools:**

```python
{
    "name": "advanced_memory_search",
    "description": "Enterprise hybrid search with consolidation",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for finding relevant memories"
            },
            "search_mode": {
                "type": "string",
                "enum": ["hybrid", "semantic", "keyword", "auto"],
                "description": "Search mode to use",
                "default": "auto"
            },
            "consolidate_related": {
                "type": "boolean",
                "description": "Whether to consolidate related memories in results",
                "default": false
            },
            "filters": {
                "type": "object",
                "description": "Additional search filters",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by specific tags"
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Filter by memory type"
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string", "format": "date-time"},
                            "end": {"type": "string", "format": "date-time"}
                        }
                    }
                },
                "additionalProperties": false
            }
        },
        "required": ["query"],
        "additionalProperties": false
    }
}
```

## Implementation Plan

### Phase 1: Core Hybrid Search (4-6 weeks)

**Week 1-2: Storage Layer**
- [ ] Implement `keyword_search()` for all storage backends
- [ ] Add BM25 scoring for SQLite-Vec using FTS5
- [ ] Create `combined_search()` method with score fusion
- [ ] Add comprehensive testing for keyword search

**Week 3-4: Service Layer**
- [ ] Implement `enhanced_search()` method
- [ ] Add score fusion algorithms (RRF, weighted combination)
- [ ] Create query analysis and expansion logic
- [ ] Add search mode auto-detection

**Week 5-6: API Integration**
- [ ] Create `/api/search/advanced` endpoint
- [ ] Update MCP tools with hybrid search capability
- [ ] Add comprehensive API testing
- [ ] Update documentation and examples

### Phase 2: Content Relationships (3-4 weeks)

**Week 1-2: Relationship Detection**
- [ ] Implement semantic clustering algorithms
- [ ] Add timeline relationship detection
- [ ] Create solution-problem mapping logic
- [ ] Build relationship scoring system

**Week 3-4: Consolidation Features**
- [ ] Implement `build_content_relationships()`
- [ ] Add automatic content grouping
- [ ] Create consolidation summary generation
- [ ] Add relationship visualization data

### Phase 3: Intelligence Features (3-4 weeks)

**Week 1-2: Query Intelligence**
- [ ] Implement query expansion using embeddings
- [ ] Add entity extraction and intent classification
- [ ] Create automatic filter suggestion
- [ ] Build user context learning

**Week 3-4: Project Consolidation**
- [ ] Implement `consolidate_project_content()`
- [ ] Add multi-pass search strategies
- [ ] Create project timeline generation
- [ ] Build project overview dashboards

### Phase 4: Enterprise Features (2-3 weeks)

**Week 1-2: Advanced Ranking**
- [ ] Implement multi-signal ranking
- [ ] Add recency and popularity signals
- [ ] Create personalization features
- [ ] Add A/B testing framework

**Week 3: Production Optimization**
- [ ] Performance optimization and caching
- [ ] Scalability testing
- [ ] Production deployment preparation
- [ ] User training and documentation

## API Specification

### Advanced Search Request

```json
{
    "query": "project Alpha deployment issues",
    "search_mode": "hybrid",
    "n_results": 15,
    "consolidate_related": true,
    "include_context": true,
    "filters": {
        "memory_types": ["task", "decision", "note"],
        "tags": ["project-alpha"],
        "time_range": "last month",
        "metadata_filters": {
            "priority": ["high", "critical"],
            "status": ["in-progress", "completed"]
        }
    },
    "ranking_options": {
        "semantic_weight": 0.6,
        "keyword_weight": 0.3,
        "recency_weight": 0.1,
        "boost_exact_matches": true
    }
}
```

### Advanced Search Response

```json
{
    "results": [
        {
            "primary_memory": {
                "content": "Memory content...",
                "content_hash": "abc123",
                "tags": ["project-alpha", "deployment"],
                "memory_type": "task",
                "metadata": {"priority": "critical"}
            },
            "similarity_score": 0.95,
            "relevance_reason": "Exact keyword match + semantic similarity",
            "consolidation": {
                "related_memories": [],
                "topic_cluster": "project-alpha-deployment",
                "consolidation_summary": "Brief summary..."
            }
        }
    ],
    "consolidated_topics": [],
    "search_intelligence": {
        "query_analysis": {},
        "recommendations": []
    },
    "performance_metrics": {
        "total_processing_time_ms": 45,
        "semantic_search_time_ms": 25,
        "keyword_search_time_ms": 8,
        "consolidation_time_ms": 12
    }
}
```

## Technical Requirements

### Performance Targets
- **Search Response Time**: < 100ms for hybrid search
- **Consolidation Time**: < 200ms for related content grouping
- **Memory Usage**: < 500MB additional RAM for caching
- **Scalability**: Support 100K+ memories with sub-second response

### Storage Requirements
- **FTS Index Storage**: +20-30% of original database size
- **Relationship Cache**: +10-15% for relationship mappings
- **Query Cache**: 100MB for frequent query caching

### Compatibility
- **Backward Compatibility**: All existing APIs remain functional
- **Storage Backend**: All three backends (SQLite-Vec, ChromaDB, Cloudflare)
- **Client Support**: Web dashboard, MCP tools, Claude Code hooks

## Quality Assurance

### Testing Strategy
1. **Unit Tests**: All new service methods with comprehensive coverage
2. **Integration Tests**: End-to-end search workflows
3. **Performance Tests**: Load testing with large datasets
4. **User Acceptance Tests**: Real-world search scenarios

### Success Metrics
- **Search Relevance**: 90%+ user satisfaction with search results
- **Response Time**: 95th percentile < 200ms
- **Consolidation Accuracy**: 85%+ correctly grouped related content
- **User Adoption**: 80%+ of users prefer hybrid over basic search

## Deployment Strategy

### Rollout Plan
1. **Alpha Testing**: Internal testing with development team (1 week)
2. **Beta Release**: Limited user group with feedback collection (2 weeks)
3. **Gradual Rollout**: 25% → 50% → 100% user adoption
4. **Feature Flags**: Toggle hybrid search on/off per user/environment

### Risk Mitigation
- **Performance Monitoring**: Real-time metrics and alerting
- **Fallback Mechanism**: Automatic fallback to basic search on errors
- **Resource Limits**: Memory and CPU usage monitoring
- **Data Integrity**: Comprehensive backup and recovery procedures

## Future Enhancements

### Phase 5: Machine Learning Integration
- **Learning to Rank**: Personalized ranking based on user behavior
- **Query Understanding**: NLP models for better intent detection
- **Recommendation Engine**: Suggest related searches and content

### Phase 6: Advanced Analytics
- **Search Analytics**: Detailed search performance dashboards
- **Content Analytics**: Memory usage patterns and insights
- **User Behavior**: Search pattern analysis and optimization

## Dependencies

### External Libraries
- **Full-Text Search**: `sqlite-fts5`, `elasticsearch-py` (optional)
- **NLP Processing**: `spacy`, `nltk` (for query enhancement)
- **Ranking Algorithms**: `scikit-learn` (for ML-based ranking)
- **Caching**: `redis` (optional, for distributed caching)

### Internal Dependencies
- **Storage Layer**: Requires enhancement to all storage backends
- **Service Layer**: Built on existing MemoryService foundation
- **Web Layer**: Requires new API endpoints and dashboard updates

## Conclusion

This Advanced Hybrid Search Enhancement will transform the MCP Memory Service into an enterprise-grade knowledge management system. The phased approach ensures minimal disruption while delivering significant value at each milestone.

The combination of hybrid search, content consolidation, and intelligent relationship mapping addresses the key limitations of the current system and provides the foundation for future AI-powered enhancements.

**Next Steps:**
1. Review and approve this specification
2. Create detailed technical design documents for Phase 1
3. Set up development environment and begin implementation
4. Establish testing infrastructure and success metrics tracking

---

**Document Prepared By**: Claude Code  
**Review Required**: Development Team, Product Owner  
**Approval Required**: Technical Lead, Project Stakeholder