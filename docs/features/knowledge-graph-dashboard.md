# Knowledge Graph Dashboard Guide

**Version:** v9.2.0+
**Last Updated:** January 18, 2026

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Relationship Types](#relationship-types)
- [Memory Types](#memory-types)
- [Accessing the Dashboard](#accessing-the-dashboard)
- [Using the Graph Visualization](#using-the-graph-visualization)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Future Roadmap](#future-roadmap)

## Overview

The Knowledge Graph Dashboard provides an interactive visualization of your memory relationships using D3.js v7.9.0. It allows you to explore the connections between memories, understand causal chains, and discover hidden patterns in your knowledge base.

**Key Capabilities:**
- **Visual Exploration**: Interactive force-directed graph with zoom, pan, and drag
- **Typed Relationships**: 6 semantic relationship types for rich knowledge graphs
- **Analytics**: Bar chart showing distribution of relationship types
- **Multi-Language**: Full UI localization in 7 languages
- **Dark Mode**: Seamless integration with dashboard theme
- **Performance**: Tested with 2,730 relationships

## Features

### Interactive Force-Directed Graph

The graph visualization uses D3.js force simulation to automatically layout memories and their relationships.

**Node Representation:**
- **Color**: Memory type (observation, decision, learning, error, pattern)
- **Size**: Fixed size for consistent visualization
- **Label**: Memory content (truncated, hover for full text)

**Edge Representation:**
- **Color**: Relationship type (causes, fixes, contradicts, supports, follows, related)
- **Direction**: Arrows indicate asymmetric relationships
- **Style**: Solid lines for all relationship types

**Interactive Controls:**
- **Zoom**: Mouse wheel or pinch gesture (mobile)
- **Pan**: Click and drag background
- **Drag Nodes**: Click and drag any node to explore connections
- **Hover Tooltips**: Shows memory details (content, type, created date)

### Relationship Type Distribution Chart

A Chart.js bar chart displays the breakdown of relationship types in your knowledge graph.

**Chart Features:**
- **Real-time Data**: Updates with graph data
- **Color-Coded Bars**: Matches relationship type colors
- **Dark Mode Support**: Adapts to dashboard theme
- **Counts**: Shows exact count for each relationship type

### Multi-Language Support

The dashboard UI is fully localized in 7 languages:

| Language | Code | Translation Keys |
|----------|------|------------------|
| English | en | 22 keys |
| Chinese | zh | 22 keys |
| German | de | 22 keys |
| Spanish | es | 22 keys |
| French | fr | 22 keys |
| Japanese | ja | 22 keys |
| Korean | ko | 22 keys |

**Total:** 154 translation keys added in v9.2.0

## Relationship Types

The Knowledge Graph supports 6 semantic relationship types:

### Asymmetric Relationships

**Direction matters:** A→B ≠ B→A

| Type | Description | Example | Direction |
|------|-------------|---------|-----------|
| **causes** | A causes or leads to B | "Bug in auth" causes "Login failure" | One-way |
| **fixes** | A fixes or resolves B | "Mutex lock" fixes "Race condition" | One-way |
| **supports** | A provides evidence for B | "Benchmark" supports "Performance claim" | One-way |
| **follows** | A follows B in sequence | "Step 2" follows "Step 1" | One-way |

### Symmetric Relationships

**No direction:** A↔B = B↔A

| Type | Description | Example | Direction |
|------|-------------|---------|-----------|
| **contradicts** | A contradicts or conflicts with B | "Use React" contradicts "Use Vue" | Two-way |
| **related** | A is related to B (general connection) | "OAuth guide" related to "Auth setup" | Two-way |

### Visual Encoding

**In the graph:**
- **Asymmetric relationships**: Displayed with arrow indicating direction
- **Symmetric relationships**: Displayed without arrows (bidirectional)
- **Color**: Each relationship type has a unique color for easy identification

## Memory Types

Memories are color-coded by type in the graph:

| Type | Description | Color | Use Case |
|------|-------------|-------|----------|
| **observation** | General observations, facts, discoveries | Blue | Project facts, code observations |
| **decision** | Decisions, planning, architecture choices | Green | Design decisions, roadmap items |
| **learning** | Learnings, insights, patterns discovered | Yellow | Lessons learned, best practices |
| **error** | Errors, failures, debugging information | Red | Bug reports, error logs |
| **pattern** | Patterns, trends, recurring behaviors | Purple | Code patterns, user behaviors |

**Other Types:**
- `discovery`, `context`, `task`, `code_edit`, etc. are displayed with default styling

## Accessing the Dashboard

### Starting the HTTP Server

The Knowledge Graph Dashboard is part of the HTTP dashboard server:

```bash
# Start the HTTP dashboard server (background)
memory launch
```

**Default URL:** http://localhost:8000

### Navigation

1. Open http://localhost:8000 in your browser
2. Click **Analytics** in the top navigation bar
3. Select **Knowledge Graph** from the dropdown or sidebar

**Direct URL:** http://localhost:8000/analytics/knowledge-graph

## Using the Graph Visualization

### Basic Interactions

**Zoom:**
- **Mouse Wheel**: Scroll up to zoom in, scroll down to zoom out
- **Touch**: Pinch gesture on mobile/tablet
- **Limits**: 0.1x to 10x zoom range

**Pan:**
- **Mouse**: Click and drag on the background (not on nodes)
- **Touch**: Single finger drag on background

**Drag Nodes:**
- **Mouse**: Click and drag any node to reposition
- **Release**: Node continues moving with simulation forces
- **Pin**: Node stays in place while you explore

**Hover:**
- **Mouse**: Hover over any node to see tooltip with:
  - Full memory content
  - Memory type
  - Created date
  - Content hash

### Exploring Connections

**Finding Related Memories:**
1. Drag a node to the center of the view
2. Observe connected nodes via edges
3. Hover on connected nodes to read details
4. Drag connected nodes to explore further

**Following Causal Chains:**
1. Identify a starting node (e.g., an error)
2. Follow **causes** edges (red arrows) to find root causes
3. Follow **fixes** edges (green arrows) to find solutions
4. Zoom in on specific areas of interest

**Detecting Contradictions:**
1. Look for **contradicts** edges (orange lines)
2. Examine both nodes to understand the conflict
3. Use this to identify inconsistencies in your knowledge base

### Relationship Type Chart

**Interpreting the Chart:**
- **X-axis**: Relationship type names
- **Y-axis**: Count of relationships
- **Bars**: Color-matched to relationship types in graph
- **Hover**: Shows exact count

**Use Cases:**
- Identify which relationship types you use most
- Detect imbalanced knowledge graphs (too many "related", not enough "causes")
- Track knowledge graph growth over time

## API Endpoints

The Knowledge Graph Dashboard uses two new API endpoints:

### GET /api/analytics/relationship-types

Get the distribution of relationship types in the graph.

**Request:**
```bash
curl http://localhost:8000/api/analytics/relationship-types
```

**Response:**
```json
{
  "causes": 245,
  "fixes": 180,
  "contradicts": 32,
  "supports": 156,
  "follows": 89,
  "related": 2028
}
```

**Use Case:** Populate the relationship type distribution chart

### GET /api/analytics/graph-visualization

Get graph data optimized for D3.js force-directed layout.

**Request:**
```bash
curl http://localhost:8000/api/analytics/graph-visualization
```

**Response:**
```json
{
  "nodes": [
    {
      "id": "abc123...",
      "content": "Implemented JWT authentication",
      "type": "decision",
      "created_at": "2026-01-15T10:30:00Z"
    },
    {
      "id": "def456...",
      "content": "Fixed token expiration bug",
      "type": "error",
      "created_at": "2026-01-16T14:20:00Z"
    }
  ],
  "links": [
    {
      "source": "def456...",
      "target": "abc123...",
      "relationship_type": "fixes"
    }
  ]
}
```

**Fields:**
- **nodes**: Array of memory objects with id, content, type, created_at
- **links**: Array of relationship objects with source, target, relationship_type

**Use Case:** Render the force-directed graph in D3.js

### Authentication

If API key authentication is enabled:

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/analytics/graph-visualization
```

## Troubleshooting

### Graph Not Loading

**Problem:** Graph shows "No relationships found" or empty screen

**Solutions:**
1. **Check for relationships**: Use MCP tool `get_memory_subgraph` to verify relationships exist
   ```bash
   # Via Claude Code/Desktop
   "Show me the subgraph for memory abc123..."
   ```
2. **Verify backend**: Graph is SQLite-only in v9.2.0, ensure you're using `sqlite_vec` or `hybrid` backend
3. **Check migrations**: Run `python scripts/migration/add_relationship_type_column.py` to add `relationship_type` column
4. **Restart server**: `memory restart`

### Slow Performance

**Problem:** Graph is laggy or unresponsive with many nodes

**Solutions:**
1. **Reduce node count**: Filter by memory type or date range (planned for v9.3.0)
2. **Check browser**: Use modern browser (Chrome 90+, Firefox 88+, Safari 14+)
3. **Close other tabs**: D3.js is CPU-intensive, reduce browser load
4. **Lower quality**: Reduce zoom level to render fewer labels

**Performance Benchmarks:**
- **100 nodes**: Smooth, <50ms frame time
- **500 nodes**: Smooth, ~100ms frame time
- **2,730 nodes**: Tested, works but may have occasional lag on older hardware

### Relationship Types Showing as "related"

**Problem:** All relationships show as "related" type

**Cause:** Migration did not run or existing relationships created before v9.0.0

**Solution:**
1. Run migration: `python scripts/migration/add_relationship_type_column.py`
2. Update existing relationships via MCP tools:
   ```python
   # Via Python API
   from mcp_memory_service.storage.factory import create_storage

   storage = create_storage("sqlite_vec")
   storage.update_relationship_type(
       source_hash="abc123...",
       target_hash="def456...",
       relationship_type="causes"
   )
   ```

### Dark Mode Issues

**Problem:** Chart or graph not adapting to dark mode

**Solutions:**
1. **Refresh page**: Force CSS reload with Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)
2. **Clear cache**: Browser cache may have old styles
3. **Check theme setting**: Ensure dashboard theme is set correctly in settings

### Translation Missing

**Problem:** UI shows English instead of selected language

**Cause:** Language not fully implemented or browser language detection issue

**Solution:**
1. **Check language support**: Only 7 languages supported (en, zh, de, es, fr, ja, ko)
2. **Manual selection**: Select language in dashboard settings
3. **Fallback**: Falls back to English if translation missing

## Performance

### Tested Configurations

**Hardware:**
- **2,730 relationships** on MacBook Pro (M1, 16GB RAM)
- **Graph render time**: ~2 seconds initial load
- **Interaction latency**: <50ms for zoom/pan/drag
- **Chart render time**: <100ms

**Browser Compatibility:**
- ✅ Chrome 90+ (recommended)
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### Optimization Tips

**For large graphs (1000+ relationships):**
1. **Limit node rendering**: Use D3.js `scaleExtent` to prevent over-zooming
2. **Reduce labels**: Only show labels when zoomed in
3. **Cluster nodes**: Group related memories (planned for v9.3.0)
4. **Pagination**: Load graph in chunks (planned for v9.3.0)

**Current implementation:**
- All relationships loaded at once
- No server-side filtering (coming in v9.3.0)
- Client-side rendering only

## Future Roadmap

### v9.3.0 (Planned)

**Hybrid Backend Graph Sync:**
- Sync graph relationships to Cloudflare D1
- Enable multi-device graph visualization
- Background sync for hybrid backend

**Server-Side Filtering:**
- Filter by memory type
- Filter by relationship type
- Filter by date range
- Filter by tag

**Enhanced Visualization:**
- Node clustering for large graphs
- Hierarchical layout option
- Timeline view for temporal relationships
- 3D graph visualization (experimental)

### v9.4.0 (Future)

**Advanced Analytics:**
- Centrality analysis (identify key memories)
- Community detection (find memory clusters)
- Path analysis (shortest paths, all paths)
- Influence tracking (memory impact over time)

**Export and Integration:**
- Export graph as PNG/SVG
- Export data as GraphML/GEXF
- Integration with external graph tools (Gephi, Neo4j)
- Embedding in external dashboards

### v9.5.0 (Future)

**AI-Powered Insights:**
- Automatic relationship suggestion
- Anomaly detection (missing connections)
- Knowledge gap identification
- Semantic similarity visualization

## Related Documentation

- [Troubleshooting Guide](../troubleshooting/) - General troubleshooting

## Support

For issues, questions, or feature requests:

- **GitHub Issues**: https://github.com/doobidoo/mcp-memory-service/issues
- **Discussions**: https://github.com/doobidoo/mcp-memory-service/discussions
- **Wiki**: https://github.com/doobidoo/mcp-memory-service/wiki

---

**Version History:**
- v9.2.0 (2026-01-18) - Initial Knowledge Graph Dashboard release
