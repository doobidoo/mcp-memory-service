# Tool Optimization - Execution Guide

## Quick Start

```bash
# Alle Tasks anzeigen
ls -la .claude/tasks/tool-optimization/

# README mit Übersicht
cat .claude/tasks/tool-optimization/README.md
```

## Ausführungsreihenfolge

### Mit amp-bridge Agent

Für jede Phase:
1. Task-Datei lesen
2. Amp Prompts extrahieren
3. Mit `amp --execute` ausführen

```bash
# Beispiel Phase 1
cat .claude/tasks/tool-optimization/phase1-delete-consolidation.md

# Amp Task ausführen (die Prompts aus der Datei kopieren)
amp --execute --dangerously-allow-all << 'EOF'
[Amp Prompt hier einfügen]
EOF
```

### Manuell mit Claude Code

1. Phase-Datei öffnen
2. Tasks durcharbeiten
3. Checkliste abhaken
4. Nächste Phase

## Phasen

| # | Datei | Beschreibung |
|---|-------|-------------|
| 1 | `phase1-delete-consolidation.md` | 6 Delete-Tools → 1 |
| 2 | `phase2-search-consolidation.md` | 5 Search-Tools → 1 |
| 3 | `phase3-consolidation-consolidation.md` | 7 Consolidation-Tools → 1 |
| 4 | `phase4-naming-migration.md` | Naming Convention + Merges |
| 5 | `phase5-deprecation-layer.md` | Backwards Compatibility |
| 6 | `phase6-tests-validation.md` | Tests + Finale Prüfung |

## Erwartetes Ergebnis

**Vorher:** 34 Tools
**Nachher:** 12 Tools

```
memory_store
memory_search
memory_list
memory_delete
memory_update
memory_health
memory_stats
memory_consolidate
memory_cleanup
memory_ingest
memory_quality
memory_graph
```

## Referenzen

- Best Practices: Phil Schmid MCP Guidelines
- Amp Agent: `.claude/agents/amp-bridge.md`
- Memory: `memory:retrieve_memory "MCP Server Optimierung"`
