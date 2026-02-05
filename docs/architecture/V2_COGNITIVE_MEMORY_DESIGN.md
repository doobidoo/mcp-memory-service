# MCP Memory Service v2: Cognitive Memory Architecture

> Design document for biologically-inspired memory consolidation
> Based on: arxiv:2512.23343, shodh-memory, Cowan's working memory model
> Author: Stev3 (via research synthesis)
> Date: 2026-02-02

---

## Executive Summary

Current mcp-memory-service (v8.x) uses flat vector search with no decay, no relationship tracking, and no consolidation. This design proposes a cognitive architecture based on neuroscience research that implements:

- **Three-tier memory** (Cowan's model)
- **Hebbian learning** (co-access strengthening)
- **Hybrid decay** (exponential → power-law)
- **Long-term potentiation** (10+ accesses = permanent)
- **Memory consolidation** (replay during "sleep" cycles)
- **Knowledge graph** (spreading activation)
- **Hybrid retrieval** (vector + graph + temporal)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MCP / API LAYER                                      │
│                    store | retrieve | search_by_tag | ...                      │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY CORE (Python/Rust)                            │
│                                                                                │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────────┐  │
│  │ SENSORY BUFFER  │   │ WORKING MEMORY  │   │    LONG-TERM MEMORY         │  │
│  │   ~7 items      │──▶│   ~4 chunks     │──▶│      unlimited              │  │
│  │ decay: <1s      │   │ decay: minutes  │   │   decay: power-law          │  │
│  │                 │   │                 │   │                             │  │
│  │ Ring buffer     │   │ LRU cache       │   │  ┌─────────┐ ┌───────────┐  │  │
│  │ Immediate       │   │ Active context  │   │  │ VECTORS │ │ KNOWLEDGE │  │  │
│  │ input queue     │   │ Current task    │   │  │ (Qdrant)│ │   GRAPH   │  │  │
│  └─────────────────┘   └─────────────────┘   │  └────┬────┘ └─────┬─────┘  │  │
│          │                     │              │       │           │        │  │
│          └─────── attention ───┴───────── consolidation ──────────┘        │  │
│                                                                                │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                            │
┌───────────────────────────────────────────┼───────────────────────────────────┐
│                    RETRIEVAL SUBSYSTEM    │                                    │
│                                           ▼                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                         │
│  │ VECTOR INDEX │  │   TEMPORAL   │  │    GRAPH     │                         │
│  │    (HNSW)    │  │    INDEX     │  │  (Hebbian)   │                         │
│  │              │  │              │  │              │                         │
│  │  similarity  │  │    decay     │  │  spreading   │                         │
│  │   scoring    │  │   weights    │  │  activation  │                         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                         │
│         │                 │                 │                                  │
│         └─────────────────┼─────────────────┘                                  │
│                           ▼                                                    │
│                  ┌─────────────────┐                                           │
│                  │ HYBRID RANKING  │                                           │
│                  │ α·vec + β·time  │                                           │
│                  │    + γ·graph    │                                           │
│                  └─────────────────┘                                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
┌─────────────────────┐      ┌─────────────────────────┐      ┌────────────────────┐
│ HEBBIAN CONSOLIDATOR│      │   INTERFERENCE ENGINE   │      │  DECAY PROCESSOR   │
│                     │      │                         │      │                    │
│ co-access patterns  │◀────▶│ similar memories        │      │ exponential (0-3d) │
│ edge.weight += η·Δw │      │ compete for retention   │      │ power-law (3d+)    │
│                     │      │ old decays when new     │      │ LTP check (10+)    │
└─────────────────────┘      └─────────────────────────┘      └────────────────────┘
```

---

## 1. Three-Tier Memory (Cowan's Model)

### 1.1 Sensory Buffer

```python
@dataclass
class SensoryBuffer:
    """Immediate input queue - raw content before processing"""
    capacity: int = 7  # Miller's magic number
    decay_ms: int = 1000  # 1 second TTL
    items: deque[SensoryItem] = field(default_factory=deque)

    def push(self, content: str, metadata: dict) -> None:
        """Add to buffer, evict oldest if at capacity"""
        if len(self.items) >= self.capacity:
            self.items.popleft()
        self.items.append(SensoryItem(
            content=content,
            metadata=metadata,
            timestamp=time.time()
        ))

    def flush_to_working(self) -> list[SensoryItem]:
        """Transfer non-expired items to working memory"""
        now = time.time()
        valid = [i for i in self.items if (now - i.timestamp) * 1000 < self.decay_ms]
        self.items.clear()
        return valid
```

**Use case:** Captures raw input before the agent decides what's important. Prevents flooding LTM with noise.

### 1.2 Working Memory

```python
@dataclass
class WorkingMemory:
    """Active manipulation space - current task context"""
    capacity: int = 4  # Cowan's limit
    decay_minutes: int = 30
    chunks: dict[str, WorkingChunk] = field(default_factory=dict)

    def activate(self, memory_id: str, chunk: WorkingChunk) -> None:
        """Bring a memory into active working set"""
        if len(self.chunks) >= self.capacity:
            self._evict_least_recent()
        self.chunks[memory_id] = chunk
        chunk.last_access = time.time()

    def consolidate_to_ltm(self) -> list[ConsolidationCandidate]:
        """Identify chunks ready for long-term storage"""
        candidates = []
        for mid, chunk in self.chunks.items():
            if chunk.access_count >= 2:  # Accessed multiple times = important
                candidates.append(ConsolidationCandidate(
                    memory_id=mid,
                    content=chunk.content,
                    importance=self._calculate_importance(chunk)
                ))
        return candidates
```

**Use case:** Holds the 4 most relevant pieces for current task. Automatic consolidation when accessed repeatedly.

### 1.3 Long-Term Memory

Implemented via Qdrant + Knowledge Graph (see sections below).

---

## 2. Hybrid Decay Model

Based on cognitive research showing memories follow:
- **Exponential decay** during consolidation phase (0-3 days)
- **Power-law decay** for long-term (heavy tail, never truly zero)

```python
def calculate_strength(memory: Memory, now: float) -> float:
    """
    Hybrid decay: exponential for first 3 days, then power-law.
    Potentiated memories (10+ accesses) decay 10x slower.
    """
    age_days = (now - memory.created_at) / 86400

    # Base decay parameters
    exp_half_life = 1.0  # days
    power_exponent = 0.3
    consolidation_boundary = 3.0  # days

    # Potentiation multiplier
    if memory.access_count >= 10:
        decay_multiplier = 0.1  # 10x slower decay
    else:
        decay_multiplier = 1.0

    if age_days <= consolidation_boundary:
        # Exponential phase (consolidation)
        strength = math.exp(-decay_multiplier * age_days / exp_half_life)
    else:
        # Power-law phase (long-term)
        # Match exponential value at boundary for continuity
        boundary_strength = math.exp(-decay_multiplier * consolidation_boundary / exp_half_life)
        adjusted_age = age_days - consolidation_boundary + 1
        strength = boundary_strength * (adjusted_age ** (-power_exponent * decay_multiplier))

    return max(strength, 0.01)  # Never truly zero
```

### Decay Curves

```
Strength
 100% |*
      | *
  70% | * <- Potentiated (10+ accesses)
      |  *____
  50% |   * \____
      |    * \________
  30% |     * \___________
      |      \____
  10% |----------------------------------------\__
      |                   Normal decay           *
   1% |--------------------------------------------*-
      +----+----+----+----+----+----+----+----+----+--->
       3d   7d  14d  30d  60d  90d 180d 365d
                           Time

      [====] Exponential    [----] Power-law
           (0-3 days)         (3+ days)
```

---

## 3. Hebbian Learning

> "Neurons that fire together, wire together"

Track co-access patterns to strengthen associations between memories.

```python
@dataclass
class HebbianEdge:
    source_id: str
    target_id: str
    weight: float = 0.0
    co_access_count: int = 0
    last_strengthened: float = 0.0

class HebbianNetwork:
    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.01):
        self.edges: dict[tuple[str, str], HebbianEdge] = {}
        self.η = learning_rate  # Strengthening rate
        self.δ = decay_rate     # Edge decay rate

    def record_co_access(self, memory_ids: list[str]) -> None:
        """When memories are retrieved together, strengthen their connections"""
        for i, m1 in enumerate(memory_ids):
            for m2 in memory_ids[i+1:]:
                key = tuple(sorted([m1, m2]))
                if key not in self.edges:
                    self.edges[key] = HebbianEdge(source_id=m1, target_id=m2)

                edge = self.edges[key]
                edge.co_access_count += 1

                # Hebbian update: Δw = η * (pre * post)
                # Both active = both 1, so Δw = η
                edge.weight = min(1.0, edge.weight + self.η)
                edge.last_strengthened = time.time()

    def get_associated(self, memory_id: str, min_weight: float = 0.3) -> list[str]:
        """Get memories strongly associated with this one"""
        associated = []
        for (m1, m2), edge in self.edges.items():
            if edge.weight >= min_weight:
                if m1 == memory_id:
                    associated.append((m2, edge.weight))
                elif m2 == memory_id:
                    associated.append((m1, edge.weight))
        return sorted(associated, key=lambda x: -x[1])
```

---

## 4. Long-Term Potentiation (LTP)

Memories accessed 10+ times become "potentiated" — effectively permanent.

```python
@dataclass
class Memory:
    id: str
    content: str
    embedding: np.ndarray
    created_at: float
    updated_at: float
    access_count: int = 0
    is_potentiated: bool = False
    strength: float = 1.0

    POTENTIATION_THRESHOLD = 10

    def record_access(self) -> None:
        self.access_count += 1
        self.updated_at = time.time()

        if not self.is_potentiated and self.access_count >= self.POTENTIATION_THRESHOLD:
            self.is_potentiated = True
            # Log potentiation event for observability
            logger.info(f"Memory {self.id} potentiated after {self.access_count} accesses")
```

---

## 5. Memory Consolidation (Replay)

During maintenance cycles (the agent's "sleep"), replay important memories to:
1. Strengthen frequently accessed patterns
2. Prune low-value entries
3. Form new associations
4. Update decay-adjusted strengths

```python
class ConsolidationService:
    def __init__(self, memory_store: MemoryStore, hebbian: HebbianNetwork):
        self.store = memory_store
        self.hebbian = hebbian

    async def run_consolidation_cycle(self) -> ConsolidationReport:
        """
        Hippocampal replay simulation.
        Call during low-activity periods (cron job, heartbeat).
        """
        report = ConsolidationReport()
        now = time.time()

        # 1. Replay high-importance memories (strengthens them)
        important = await self.store.get_by_importance(min_importance=0.7, limit=100)
        for memory in important:
            memory.record_access()  # Replay = implicit access
            report.replayed += 1

        # 2. Prune low-strength, non-potentiated memories
        weak = await self.store.get_weak_memories(max_strength=0.05, exclude_potentiated=True)
        for memory in weak:
            if memory.access_count < 3:  # Never meaningfully used
                await self.store.archive(memory.id)
                report.pruned += 1

        # 3. Decay Hebbian edges that haven't been reinforced
        stale_edges = [
            key for key, edge in self.hebbian.edges.items()
            if (now - edge.last_strengthened) > 86400 * 7  # 7 days
        ]
        for key in stale_edges:
            self.hebbian.edges[key].weight *= (1 - self.hebbian.δ)
            if self.hebbian.edges[key].weight < 0.05:
                del self.hebbian.edges[key]
                report.edges_pruned += 1

        # 4. Discover new associations via embedding clustering
        clusters = await self._cluster_recent_memories(days=7)
        for cluster in clusters:
            if len(cluster) >= 2:
                self.hebbian.record_co_access(cluster)
                report.new_associations += len(cluster) * (len(cluster) - 1) // 2

        return report
```

---

## 6. Knowledge Graph Layer

Add relationship tracking on top of vector search.

```python
@dataclass
class MemoryNode:
    memory_id: str
    entity_type: str  # "concept", "person", "event", "fact", etc.
    tags: list[str]

@dataclass
class MemoryRelation:
    source_id: str
    target_id: str
    relation_type: str  # "related_to", "caused_by", "part_of", "contradicts", etc.
    weight: float
    created_at: float

class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, MemoryNode] = {}
        self.relations: list[MemoryRelation] = []
        self.adjacency: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

    def spreading_activation(
        self,
        seed_ids: list[str],
        activation: float = 1.0,
        decay: float = 0.5,
        max_hops: int = 3
    ) -> dict[str, float]:
        """
        Spread activation from seed nodes through the graph.
        Returns activation levels for all reached nodes.
        """
        activations = {sid: activation for sid in seed_ids}
        frontier = list(seed_ids)

        for hop in range(max_hops):
            next_frontier = []
            current_activation = activation * (decay ** hop)

            for node_id in frontier:
                for target_id, relation_type, weight in self.adjacency.get(node_id, []):
                    spread = current_activation * weight
                    if target_id not in activations:
                        activations[target_id] = spread
                        next_frontier.append(target_id)
                    else:
                        activations[target_id] = max(activations[target_id], spread)

            frontier = next_frontier
            if not frontier:
                break

        return activations
```

---

## 7. Hybrid Retrieval

Combine vector similarity, temporal decay, and graph activation.

```python
class HybridRetriever:
    def __init__(
        self,
        vector_store: Qdrant,
        graph: KnowledgeGraph,
        hebbian: HebbianNetwork,
        # Tunable weights
        α_vector: float = 0.5,
        β_temporal: float = 0.2,
        γ_graph: float = 0.2,
        δ_hebbian: float = 0.1,
    ):
        self.vector_store = vector_store
        self.graph = graph
        self.hebbian = hebbian
        self.α = α_vector
        self.β = β_temporal
        self.γ = γ_graph
        self.δ = δ_hebbian

    async def retrieve(
        self,
        query: str,
        context_ids: list[str] = None,
        limit: int = 10
    ) -> list[ScoredMemory]:
        """
        Hybrid retrieval combining multiple signals.
        """
        now = time.time()

        # 1. Vector similarity search
        query_embedding = await self.embed(query)
        vector_results = await self.vector_store.search(query_embedding, limit=limit*3)

        # 2. Calculate temporal decay for each result
        # 3. Get graph activation if context provided
        # 4. Get Hebbian associations if context provided

        scored = []
        graph_activations = {}
        hebbian_boosts = {}

        if context_ids:
            graph_activations = self.graph.spreading_activation(context_ids)
            for cid in context_ids:
                for assoc_id, weight in self.hebbian.get_associated(cid):
                    hebbian_boosts[assoc_id] = max(hebbian_boosts.get(assoc_id, 0), weight)

        for result in vector_results:
            memory = result.memory

            # Vector score (already normalized 0-1)
            vec_score = result.score

            # Temporal score (decay-adjusted)
            temp_score = calculate_strength(memory, now)

            # Graph score (spreading activation)
            graph_score = graph_activations.get(memory.id, 0)

            # Hebbian score (association strength)
            hebb_score = hebbian_boosts.get(memory.id, 0)

            # Combined score
            final_score = (
                self.α * vec_score +
                self.β * temp_score +
                self.γ * graph_score +
                self.δ * hebb_score
            )

            scored.append(ScoredMemory(memory=memory, score=final_score))

        # Sort and return top results
        scored.sort(key=lambda x: -x.score)
        return scored[:limit]
```

---

## 8. API Changes

### New Endpoints

```python
# Consolidation trigger (for cron/manual)
POST /consolidate
Response: ConsolidationReport

# Get memory with full metadata (including strength, potentiation)
GET /memories/{id}/full
Response: MemoryFull

# Record co-access (for Hebbian learning)
POST /co-access
Body: {"memory_ids": ["id1", "id2", ...]}

# Get associated memories (graph/Hebbian)
GET /memories/{id}/associated
Response: [{"id": "...", "relation": "...", "strength": 0.8}, ...]
```

### Modified Endpoints

```python
# retrieve_memory now includes decay-adjusted scoring
POST /retrieve
Body: {
    "query": "...",
    "context_ids": ["recent_memory_1"],  # NEW: for spreading activation
    "min_strength": 0.1,  # NEW: filter by decay-adjusted strength
}

# store_memory now accepts relationship hints
POST /store
Body: {
    "content": "...",
    "tags": [...],
    "related_to": ["existing_memory_id"],  # NEW: explicit relations
    "entity_type": "concept",  # NEW: for knowledge graph
}
```

---

## 9. Data Schema Changes

### Memory Table

```sql
-- Add columns to existing memories table
ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0;
ALTER TABLE memories ADD COLUMN is_potentiated BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN strength FLOAT DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN entity_type VARCHAR(50);

-- Create index for strength-based queries
CREATE INDEX idx_memories_strength ON memories(strength);
CREATE INDEX idx_memories_potentiated ON memories(is_potentiated);
```

### New Tables

```sql
-- Hebbian edges
CREATE TABLE hebbian_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories(id),
    target_id TEXT NOT NULL REFERENCES memories(id),
    weight FLOAT DEFAULT 0.0,
    co_access_count INTEGER DEFAULT 0,
    last_strengthened TIMESTAMP,
    UNIQUE(source_id, target_id)
);

-- Knowledge graph relations
CREATE TABLE memory_relations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories(id),
    target_id TEXT NOT NULL REFERENCES memories(id),
    relation_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_relations_source ON memory_relations(source_id);
CREATE INDEX idx_relations_target ON memory_relations(target_id);
```

---

## 10. Migration Path

### Phase 1: Schema Migration (v9.0)
- Add new columns (backward compatible)
- Create new tables
- Initialize access_count from existing usage patterns (if logged)
- Set initial strength = 1.0 for all existing memories

### Phase 2: Decay Processing (v9.1)
- Implement decay calculation
- Add background job for periodic strength updates
- Update retrieval to include strength in ranking

### Phase 3: Hebbian Learning (v9.2)
- Implement co-access tracking
- Build Hebbian network from existing retrieval patterns
- Add associated memories endpoint

### Phase 4: Consolidation (v9.3)
- Implement consolidation service
- Add cron job / maintenance cycle trigger
- Implement pruning with archive (not delete)

### Phase 5: Knowledge Graph (v10.0)
- Implement graph layer
- Add entity extraction (LLM-powered or rule-based)
- Implement spreading activation retrieval

---

## 11. Performance Considerations

| Operation | Current | With Cognitive Layer | Mitigation |
|-----------|---------|---------------------|------------|
| Store | O(1) | O(E) for edge updates | Batch edge updates |
| Retrieve | O(log N) | O(log N + E) | Pre-compute graph scores |
| Consolidation | N/A | O(N) | Run during idle time |
| Decay update | N/A | O(N) | Lazy evaluation on read |

### Optimizations

1. **Lazy decay calculation** — Don't update strength in DB on every read; calculate on retrieval
2. **Batch Hebbian updates** — Queue co-access events, process in batches
3. **Graph caching** — Cache spreading activation results for hot queries
4. **Tiered consolidation** — More frequent for recent memories, less for old

---

## 12. Comparison with Current Architecture

| Feature | v8.x (Current) | v10 (Cognitive) |
|---------|----------------|-----------------|
| Memory model | Flat | Three-tier (Cowan's) |
| Decay | None | Hybrid (exp + power-law) |
| Associations | None | Hebbian + Knowledge Graph |
| Retrieval | Vector only | Vector + Temporal + Graph |
| Consolidation | Removed | Restored (replay-based) |
| Potentiation | None | 10+ accesses = permanent |
| Interference | None | Competing memories |

---

## References

1. **arxiv:2512.23343** — "AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents" (Dec 2025)
2. **shodh-memory** — https://shodh-memory.com (Rust-based cognitive memory)
3. **Cowan (2001)** — The magical number 4 in short-term memory
4. **Ebbinghaus (1885)** — Memory: A Contribution to Experimental Psychology
5. **Hebb (1949)** — The Organization of Behavior (fire together, wire together)

---

*This document is a living design. Implementation details may evolve during development.*
