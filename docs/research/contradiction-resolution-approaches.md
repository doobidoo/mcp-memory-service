# Contradiction Resolution Approaches

**Source discussion:** [Issue #732 — Bridging Advanced Reasoning with MCP Memory Lifecycle](https://github.com/doobidoo/mcp-memory-service/issues/732#issuecomment-4506467185) · [Issue #983 — incremental consolidation](https://github.com/doobidoo/mcp-memory-service/issues/983)

**Purpose:** Compare how agent memory systems detect, represent, and resolve contradictions — without prescribing a single winner. Use this as reference material when extending MCP Memory Service's lifecycle and reasoning layers.

---

## 1. Problem framing

Contradiction handling in long-running agent memory spans three distinct concerns:

| Concern | Question |
|---------|----------|
| **Detection** | How do we notice two memories conflict? |
| **Representation** | What happens to the old memory — delete, invalidate, flag, or rewire edges? |
| **Propagation** | If A contradicts B and B supports C, does C become suspect? |

Most production systems today optimize **detection cost on the ingest hot path** and defer **expensive semantic audit** to offline passes. The propagation question (transitive closure) is often deferred until scale forces it.

---

## 2. Comparative overview

| Approach | Detection | Invalidation model | Graph propagation | Hot-path cost | Offline cost | Best when |
|----------|-----------|-------------------|-------------------|---------------|--------------|-----------|
| **Point invalidation (bi-temporal)** | Title/summary heuristics or same-key supersession | Set `invalid_at` on superseded row; new row gets new `valid_at` | None — downstream reads filter `invalid_at IS NULL` | Low (ms, bounded search) | Medium (semantic scan catches drift gates missed) | Single-operator fleets, <100k atoms, explicit audit trail |
| **Transitive closure invalidation** | Graph delta + semantic checks | Mark node + all reachable dependents stale/invalid | Eager recompute on write | High on write (closure walk) | Lower audit need if closure is correct | Dense typed graphs where derived facts must not outlive premises |
| **Lazy invalidation flags** | Same as point or closure, but propagation deferred | Set `stale=true` on node; dependents flagged on read or periodic job | Lazy — recompute on next query or batch sweep | Low on write | Periodic batch (minutes–hours) | 100k+ atoms, write-heavy, acceptable eventual consistency |
| **Semantic-only audit (no hard gate)** | Embedding neighborhood + LLM pair classify | Advisory only — human or consolidate job decides | None unless consolidate merges | Zero on ingest | High (O(atoms × neighbors × LLM)) | Research systems, consolidation pipelines |
| **Entity-profile supersession** | Compare new observation to entity's current profile state | Profile version increments; old facts archived under prior version | Scoped to entity subgraph | Medium (profile lookup + embed compare) | Medium (profile rebuild on consolidate) | User/project-centric memory (Honcho-style peers) |

---

## 3. Detection mechanisms (detailed)

### 3.1 Lexical / structural gates (hot path)

**Mechanism:** On ingest, search for candidates with similar title or key (ILIKE, exact title, token Jaccard). If title matches but summary diverges beyond a threshold → contradiction.

| System | Gate location | Typical signals |
|--------|---------------|-----------------|
| **Willow 2.0** | `mem_check` / `kb_ingest` | Title ILIKE (≤20 hits), exact title → REDUNDANT; same title + summary Jaccard < 0.3 → CONTRADICTION |
| **MCP Memory Service** | `memory_conflicts`, `memory_cleanup` | Exact dedup + contradiction detection tools (see service docs) |
| **Generic pattern** | Pre-write hook | Key collision + content divergence |

**Tradeoffs:**
- ✅ Bounded cost — does not scale with total corpus if search is capped/indexed
- ✅ Deterministic, debuggable
- ❌ Misses paraphrase conflicts (same meaning, different title)
- ❌ Same-title assumption breaks when titles are auto-generated or vague

### 3.2 Semantic neighborhood scan (offline)

**Mechanism:** For each atom, embed summary → find k nearest neighbors → LLM or cross-encoder classifies pair as TENSION / REDUNDANT / COMPATIBLE.

| System | Trigger | Scope |
|--------|---------|-------|
| **Willow 2.0** | `tension_scan` (on-demand) or `dream_run` (≥24h + ≥5 sessions) | Recent `hypothesis`/`observed` tier, limit 60 atoms, cap 30 tension pairs |
| **Honcho (conceptual)** | Representation updates over peer dialogue | Entity-centric, latent state extraction |
| **MCP Memory RFC #732** | Proposed Phase 4 — temporal meaning drift | Historical decision embed vs new memory embed |

**Tradeoffs:**
- ✅ Catches paraphrase and cross-title conflicts
- ❌ Expensive — not suitable for per-ingest unless heavily capped
- ❌ LLM classification can false-positive; needs human ratification or consolidate step

### 3.3 Graph-structural triggers

**Mechanism:** Invalidate or flag when an edge type changes (e.g. `supports` → `contradicts`) or when transitive closure detects A→B→C but A and C are semantically incompatible.

| System | Status |
|--------|--------|
| **MCP Memory Service** | Typed edges exist; transitive reasoning placeholders in `SemanticReasoner` (RFC #732 Phase 1) |
| **Willow 2.0** | SOIL edges exist; ingest gates do **not** traverse them for contradiction |

**Tradeoffs:**
- ✅ Correct propagation when graph semantics are explicit
- ❌ Requires edge typing discipline at ingest — rare in practice today
- ❌ Closure maintenance cost grows with graph density

---

## 4. Invalidation models (detailed)

### 4.1 Point invalidation (bi-temporal rows)

When a contradiction is accepted at ingest:

1. Existing atom(s) with matching key get `invalid_at = now()`
2. New atom is inserted with `valid_at = now()`, `invalid_at = NULL`
3. Queries default to `WHERE invalid_at IS NULL`

**No transitive sweep.** Downstream agents re-query; anything that cached the old atom may be stale until next retrieval.

**Willow 2.0 worked example** (`knowledge_close` in Postgres):

```
CONTRADICTION detected → UPDATE knowledge SET invalid_at = $now WHERE id IN (...matched ids...)
→ INSERT new atom with superseding summary
```

At ~150k atoms on a single-operator fleet, point invalidation + periodic tension_scan has been sufficient. At 100k+, expect either lazy flags on dependents or a scheduled closure recompute — not because point invalidation is wrong, but because read-side re-query becomes the bottleneck if agents cache aggressively.

### 4.2 Transitive closure invalidation

When atom B is invalidated, all atoms reachable via `derived_from`, `supports`, or similar edges are marked invalid or `stale`.

**When to choose:** Derived facts must not survive contradictory premises (compliance, financial ledgers, formal reasoning chains).

**Cost:** O(reachable subgraph) per invalidation event. Requires maintained edge index and explicit edge semantics.

**Gap in most MCP memory systems today:** edges are often inferred post-hoc or advisory; closure without typed `supports`/`contradicts` edges produces false cascades.

### 4.3 Lazy invalidation flags

Middle ground:

1. Point-invalidate the direct conflict
2. Set `stale=true` on 1-hop neighbors (or enqueue for batch)
3. Background job or next read recomputes validity

Honcho-style **entity profiles** fit here: the profile is the materialized view; individual memories can be invalidated while the profile version increments lazily.

---

## 5. Worked example: Willow 2.0 split architecture

Willow deliberately separates **cheap ingest gates** from **expensive semantic audit**.

```
┌─────────────────────────────────────────────────────────────┐
│ INGEST HOT PATH (mem_check / kb_ingest)                     │
│  • ILIKE title search, cap 20 hits                          │
│  • REDUNDANT: exact/near title match                        │
│  • CONTRADICTION: same title + summary Jaccard < 0.3        │
│  • On CONTRADICTION: knowledge_close(old) → write new       │
│  Cost: ~ms, no embed, no LLM                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ OFFLINE AUDIT (tension_scan / dream_run)                    │
│  • Pull hypothesis/observed atoms (limit 60)                │
│  • pgvector neighbors (nomic-embed, top-4)                  │
│  • mistral:7b pair classify: TENSION|REDUNDANT|COMPATIBLE   │
│  Cost: O(atoms × neighbors × LLM) — intentionally capped  │
└─────────────────────────────────────────────────────────────┘
```

**What Willow does not do (yet):**
- Transitive closure over SOIL/KB edges on contradiction
- Explicit edge typing at ingest (`supports`, `contradicts`, `supersedes`)
- Automated synthesis/merge of redundant clusters (see consolidation gap below)

These are tier choices, not design failures — defer until graph density or agent count forces the complexity.

Reference implementation: Willow 2.0 (`memory_gate.py`, `tension_scan` in `sap_mcp.py`).

---

## 6. MCP Memory Service mapping (RFC #732 + #983)

| RFC #732 proposal | Suggested approach tier | Notes |
|-------------------|------------------------|-------|
| **Temporal contradiction detection** | Semantic neighborhood + bi-temporal invalidation | Aligns Phase 4 with offline scan, not hot-path embed |
| **Transitive closure reasoning** | Eager closure **or** lazy flags — pick per edge type | Phase 1: implement path-finding; defer closure invalidation until edges are typed at ingest |
| **Entity-centric grouping** | Entity profile supersession | Contradictions scoped to entity subgraph, not global KB |
| **Automated insight cards** | Post-consolidate synthesis | New memories with `derived_from` edges back to sources |

### Consolidation gap (production feedback)

Sources: [#732 discussion](https://github.com/doobidoo/mcp-memory-service/issues/732#issuecomment-4506467185) (original gap) and [#983](https://github.com/doobidoo/mcp-memory-service/issues/983) (refined proposal after codebase audit).

Operators running sqlite_vec at 3000+ memories with parallel sessions report **fragmentation** — many near-duplicate topic memories accumulating faster than manual curation scales.

**What already exists:** `memory_consolidate` is a shipped MCP tool. `DreamInspiredConsolidator` already covers clustering, synthesis, and lineage edges under the existing time horizons (`daily` / `weekly` / `monthly` / `quarterly` / `yearly`).

**The actual delta (#983)** is not a new tool — it is an extension of the existing surface:

```text
memory_consolidate(action="run", time_horizon="incremental")
```

Where `incremental` means: process only memories created since the last consolidation timestamp; skip full clustering; run lightweight dedup + contradiction check + relative-date normalization on recent items; **skip the forgetting/archival phase** (archival stays on monthly+ horizons, not per-session); bounded to <10s so it is safe as a Stop-hook trigger.

Five-point decomposition (conceptual — maps filhocf's original spec to current vs delta):

| Step | Status | Notes |
|------|--------|-------|
| 1. **Cluster** | ✅ Implemented | `DreamInspiredConsolidator` semantic clustering |
| 2. **Synthesize** | ✅ Implemented | Merge cluster into consolidated memory |
| 3. **Lineage** | ✅ Implemented | Graph edges back to sources (not hard-delete) |
| 4. **Incremental mode** | ❌ Delta (#983) | New `time_horizon="incremental"` — recent memories only, hook-callable |
| 5. **Full mode** | ✅ Implemented | Existing daily→yearly horizons for heavier passes |

Additional incremental-mode requirements from #983:

- **Hook-callable** — safe to invoke from session Stop hooks without blocking the agent
- **Relative-date normalization** — e.g. `"yesterday"` → `"2026-05-20"` during the pass

Point invalidation + periodic consolidate remains the natural pairing: gates prevent obvious conflicts on write; consolidate resolves drift and redundancy offline.

---

## 7. Decision guide

Use this when choosing mechanisms for MCP Memory Service phases:

| If your priority is… | Prefer… | Defer… |
|---------------------|---------|--------|
| Fast ingest, small fleet | Lexical gates + bi-temporal point invalidation | Transitive closure |
| Paraphrase detection | Offline semantic scan + consolidate | Per-ingest embedding compare |
| Correct derived-fact propagation | Typed edges + closure (or lazy flags) | Keyword-only conflict tools |
| Multi-session topic fragmentation | `memory_consolidate(time_horizon="incremental")` on existing tool | New tool or eager closure on every write |
| Audit trail / compliance | Bi-temporal rows (never hard-delete) | Silent overwrite |
| 100k+ atoms | Lazy flags + indexed title search (trigram/GIN) | Unbounded ILIKE hot path |

---

## 8. Open questions for RFC #732

1. **Edge typing contract** — Which edge types participate in closure? (`supports`, `contradicts`, `supersedes`, `derived_from`, …)
2. **Invalidation default** — On CONTRADICTION, should dependents be eagerly invalidated, lazily flagged, or left to re-query?
3. **Consolidate vs conflict** — Should incremental consolidation consume `memory_conflicts` output, or run its own lightweight dedup pass?
4. **Ratification** — Who confirms LLM-classified TENSION pairs before invalidation? (Human, auto after N sessions, never auto?)
5. **Entity scope** — Are contradictions global or scoped to entity/profile?

---

## 9. References

- [MCP Memory Service #732 — Bridging Advanced Reasoning with MCP Memory Lifecycle](https://github.com/doobidoo/mcp-memory-service/issues/732)
- [MCP Memory Service #983 — incremental consolidation time_horizon](https://github.com/doobidoo/mcp-memory-service/issues/983)
- [Honcho](https://github.com/plastic-labs/honcho) — entity-centric peers, representation layers, contradiction resolution over dialogue
- [Willow 2.0](https://github.com/rudi193-cmd/willow-2.0) — REDUNDANT/CONTRADICTION ingest gates + offline `tension_scan` (worked example in §5)
- [Bi-temporal data models](https://en.wikipedia.org/wiki/Bitemporal_modeling) — valid-time vs transaction-time (Willow uses valid-time via `valid_at`/`invalid_at`)

---

*Contributed from the [#732 discussion](https://github.com/doobidoo/mcp-memory-service/issues/732). System-neutral framing; Willow 2.0 cited as one concrete deployment, not the target architecture for MCP Memory Service.*
