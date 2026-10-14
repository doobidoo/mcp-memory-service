# Spec: §2 Belief Store (Incremental Approach)

> **⚠️ STATUS CORRIGIDO (14/out/2026):** esta RFC foi marcada "implemented" por engano. O
> código real (`BeliefService.derive_beliefs`, #33) DIVERGIU desta spec desde o dia 1:
> usa uma tabela `beliefs` própria + clusterização verbatim, NÃO o `_promote_to_belief` /
> `memory_type=belief` aqui descritos (que nunca foram construídos). Ver
> `docs/rfc/planned/rfc-belief-synthesis.md` para o estado real e o arco de correção.
>
> **O QUE AINDA VALE RESGATAR desta RFC (design bom, nunca implementado):**
> 1. **Subtipos de belief** (§1: convention/preference/avoidance/tool_pattern/causal) —
>    ausentes do ontology.py; são a tipologia conceitual que a SÍNTESE deve classificar.
> 2. **Scope classifier global vs project:xxx** (§4) — ausente; beliefs vivos têm scope
>    vazio (100/100). Ataca o rich-get-richer: global não deve competir com project na
>    mesma query. Eixo de relevância nunca ligado.
> 3. **Promoção emerging→active** (§3) — gradação por confirmação, MAS com a correção
>    medida 14/out: exigir ≥2-3 sessões mata o conhecimento dito-uma-vez (Recall@pool 0.30);
>    precisa do singular path para decision/learning.
> **O QUE É MORTO:** _promote_to_belief (nunca existiu), memory_type=belief (código usa
> tabela própria — decisão de arquitetura reaberta), §5-§7 bootstrap (tema separado).

**Date:** 2026-06-01
**Branch:** `feat/rfc1-s2-belief-store` (create from `main`)
**Repo:** `~/git/mcp-memory-service`
**Refs:** RFC #1 (Codeberg issue #1), doobidoo review 29/mai/2026, CdIA research

---

## Design Decision

**NOT a separate table.** Beliefs are memories with `memory_type="belief"` and structured metadata.

Rationale:
- Reuses existing storage, search, quality, and graph infrastructure
- No schema migration needed (metadata is JSON)
- Bootstrap formatter already filters by type
- Aligns with doobidoo's concern about ownership/complexity (simpler = mergeable)

---

## 1. Ontology Addition

In `src/mcp_memory_service/models/ontology.py`, add to TAXONOMY:

```python
"belief": [
    "convention",      # "Use pnpm not npm"
    "preference",      # "User prefers concise answers"
    "avoidance",       # "NEVER use strReplace on MEMORY.md"
    "tool_pattern",    # "Prefer ripgrep over find"
    "causal",          # "RSC reduces bundle size"
],
```

---

## 2. Belief Metadata Schema

Beliefs stored as regular memories with structured metadata:

```python
{
    "memory_type": "belief",
    "metadata": {
        "tags": "belief,convention,global",
        "belief_status": "active",       # emerging | active | quarantine | archived
        "confidence": 0.85,
        "scope": "global",               # global | project:mir | project:rer
        "evidence_hashes": ["abc123", "def456"],  # observations that support this
        "first_seen": "2026-05-20",
        "last_confirmed": "2026-06-01",
        "frustration_score": 0.0,        # for avoidances
        "source_type": "distill"         # distill | user_correction | commit_session_legacy
    }
}
```

---

## 3. Promotion Logic (in distill)

Add to `handle_memory_distill` or a new `_promote_to_belief()` helper:

```python
async def _promote_to_belief(self, insight_hash: str, evidence_hashes: list, scope: str):
    """Promote a distilled insight to belief if multi-session confirmed."""
    # Check: does this insight appear in ≥2 different sessions?
    # If yes: store as memory_type="belief" with belief_status="emerging"
    # If already emerging and confirmed again: promote to "active"
    # If user_correction: promote immediately to "active" (confidence 0.9)
```

Criteria:
- **Normal path:** ≥2 sessions confirming → `emerging`. ≥3 sessions → `active`.
- **User correction:** immediate `active` (confidence 0.9)
- **Contradiction detected:** both beliefs → `quarantine`

---

## 4. Scope Classifier (in distill/rewriter)

Add scope classification to the LLM rewriter prompt:

```
Additionally, classify the scope:
- "global" if this is about agent behavior, communication style, or universal patterns
- "project:{name}" if this is specific to a codebase, API, or domain (e.g., "project:mir", "project:mcp-memory-service")

Output format: {"insight": "...", "scope": "global|project:xxx", "belief_subtype": "convention|preference|avoidance|tool_pattern|causal"}
```

For non-LLM path (heuristic fallback):
- Contains project-specific terms (MIR, SNCR, Roma, SICAR) → `project:mir`
- Contains tool/library names without project context → `global`
- Contains "NEVER", "ALWAYS", communication patterns → `global`

---

## 5. Bootstrap Formatter Changes

In `src/mcp_memory_service/bootstrap/formatter.py`:

```python
# Current: fetches all memories with high confidence
# New: prefer beliefs over observations, filter by scope

async def _get_profile_entries(self, agent_id: str, project_id: str = None):
    # 1. Fetch beliefs (memory_type=belief, belief_status=active)
    # 2. Filter by scope: global + project:{current_project}
    # 3. Sort by confidence DESC
    # 4. Fall back to observations only if <5 beliefs exist
```

---

## 6. MCP_BOOTSTRAP_ENABLED=false (doobidoo requirement)

In config:
```python
MCP_BOOTSTRAP_ENABLED = safe_get_bool_env("MCP_BOOTSTRAP_ENABLED", False)
```

In `handle_get_bootstrap_profile`:
```python
if not MCP_BOOTSTRAP_ENABLED:
    return [types.TextContent(type="text", text='{"enabled": false, "message": "Bootstrap profile disabled. Set MCP_BOOTSTRAP_ENABLED=true to enable."}')]
```

---

## 7. Multi-Agent Isolation (doobidoo concern #4)

Add `MCP_BOOTSTRAP_MIN_SESSIONS` (default: 3):
- Agent with <3 sessions gets only its OWN beliefs
- Cross-agent merge only after threshold reached

In `handle_get_bootstrap_profile`:
```python
session_count = await self._get_session_count(agent_id)
if session_count < MCP_BOOTSTRAP_MIN_SESSIONS:
    # Only return beliefs from this agent_id
    filter_agent = agent_id
else:
    # Merge beliefs from all requested agent_ids
    filter_agent = None
```

---

## Files Affected

- `src/mcp_memory_service/models/ontology.py` — add belief subtypes (~5 lines)
- `src/mcp_memory_service/harvest/rewriter.py` — add scope classification to prompt (~20 lines)
- `src/mcp_memory_service/harvest/harvester.py` — add `_promote_to_belief` logic (~40 lines)
- `src/mcp_memory_service/bootstrap/formatter.py` — prefer beliefs, filter scope (~30 lines)
- `src/mcp_memory_service/server_impl.py` — MCP_BOOTSTRAP_ENABLED check + min_sessions (~15 lines)
- `src/mcp_memory_service/config.py` — 2 new env vars (~4 lines)
- `tests/test_belief_store.py` — new test file (~150 lines)
- `tests/test_bootstrap_profile.py` — update existing tests (~20 lines)

**Total: ~280 lines of code + ~150 lines of tests**

---

## Validation

```bash
.venv/bin/python -m pytest tests/ -x -q --timeout=120
```

Additionally, integration test:
1. Store 3 observations across "different sessions" (mock session_id)
2. Run distill → verify belief created with status=emerging
3. Store 1 more confirming observation → verify promotion to active
4. Run bootstrap → verify belief appears in profile
5. Store contradicting observation → verify quarantine

---

## Commit Strategy

1. `feat(ontology): add belief subtypes to taxonomy`
2. `feat(belief): scope classifier in distill rewriter`
3. `feat(belief): promotion logic (observation → belief)`
4. `feat(bootstrap): prefer beliefs, filter by scope, MCP_BOOTSTRAP_ENABLED`
5. `feat(bootstrap): multi-agent isolation (min_sessions threshold)`

---

## Addresses doobidoo's Concerns

| Concern | How addressed |
|---------|--------------|
| #2 Bootstrap too aggressive | `MCP_BOOTSTRAP_ENABLED=false` by default |
| #3 LLM fallback | Scope classifier has heuristic fallback (no LLM needed) |
| #4 Multi-agent contamination | `MCP_BOOTSTRAP_MIN_SESSIONS=3` isolation |
| #5 Ownership P2 | We (filhocf) own it — spec + implementation |
