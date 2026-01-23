# MCP Memory Service - Tool Optimization Plan

## Ausgangslage

**Problem:** 34 Tools → Best Practice empfiehlt 5-15 Tools pro Server

**Analyse-Ergebnis:**
- 5 Retrieve-Varianten (sollten 1 sein)
- 6 Delete-Varianten (sollten 1 sein)
- 7 Consolidation-Tools (sollten 1 sein)
- Inkonsistente Naming Convention
- Freie Dict-Parameter statt flacher Argumente

## Ziel

Reduktion von 34 auf 12 Tools durch Konsolidierung:

| Neuer Name | Ersetzt | Anzahl |
|------------|---------|--------|
| `memory_store` | store_memory | 1→1 |
| `memory_search` | retrieve_*, recall_*, exact_match | 5→1 |
| `memory_list` | list_memories, search_by_tag | 2→1 |
| `memory_delete` | delete_* | 6→1 |
| `memory_update` | update_memory_metadata | 1→1 |
| `memory_health` | check_database_health | 1→1 |
| `memory_stats` | get_cache_stats | 1→1 |
| `memory_consolidate` | consolidate_*, scheduler_*, pause/resume | 7→1 |
| `memory_cleanup` | cleanup_duplicates | 1→1 |
| `memory_ingest` | ingest_document, ingest_directory | 2→1 |
| `memory_quality` | rate_memory, get_memory_quality, analyze_* | 3→1 |
| `memory_graph` | find_connected, shortest_path, subgraph | 3→1 |

## Phasen-Übersicht

| Phase | Task-Datei | Fokus | Geschätzte Dauer |
|-------|------------|-------|------------------|
| 1 | `phase1-delete-consolidation.md` | 6 Delete → 1 | ~10 min |
| 2 | `phase2-search-consolidation.md` | 5 Search → 1 | ~15 min |
| 3 | `phase3-consolidation-consolidation.md` | 7 Consolidation → 1 | ~10 min |
| 4 | `phase4-naming-migration.md` | Naming Convention | ~5 min |
| 5 | `phase5-deprecation-layer.md` | Backwards Compatibility | ~15 min |
| 6 | `phase6-tests-validation.md` | Tests + QA | ~20 min |

**Gesamtdauer:** ~75 Minuten

## Ausführung

Jede Phase kann mit dem `amp-bridge` Agent ausgeführt werden:

```bash
# Beispiel für Phase 1
cat .claude/tasks/tool-optimization/phase1-delete-consolidation.md | amp --execute
```

Oder manuell in Claude Code durcharbeiten.

## Referenzen

- Best Practices: `memory:retrieve_memory "MCP Server Optimierung"`
- Amp-Bridge Agent: `.claude/agents/amp-bridge.md`
- Aktueller Server: `src/mcp_memory_service/server_impl.py`
- Memory Service: `src/mcp_memory_service/services/memory_service.py`
