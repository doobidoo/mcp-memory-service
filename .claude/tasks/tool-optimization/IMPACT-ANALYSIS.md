# Auswirkungsanalyse: Tool-Optimierung

## Übersicht

Die Tool-Optimierung (34 → 12 Tools) hat Auswirkungen auf drei Bereiche:

| Bereich | Impact | Beschreibung |
|---------|--------|--------------|
| **MCP Handler Tests** | 🔴 HIGH | 66+ Aufrufe auf alte Handler-Namen |
| **MemoryService Tests** | 🟡 MEDIUM | Backend-Methoden werden konsolidiert |
| **Dashboard/HTTP API** | 🟢 LOW | Nutzt REST API, nicht MCP Tools |

---

## 1. Betroffene Tests

### Integration Tests (KRITISCH)

**`tests/integration/test_server_handlers.py`**
- `handle_store_memory` → `handle_memory_store`
- `handle_retrieve_memory` → `handle_memory_search`
- `handle_search_by_tag` → `handle_memory_list`

**`tests/integration/test_all_memory_handlers.py`**
- `handle_delete_memory` → `handle_memory_delete`
- `handle_delete_by_tag` → `handle_memory_delete`
- `handle_delete_by_tags` → `handle_memory_delete`
- `handle_delete_by_all_tags` → `handle_memory_delete`
- `handle_retrieve_with_quality_boost` → `handle_memory_search`
- `handle_recall_memory` → `handle_memory_search`
- `handle_recall_by_timeframe` → `handle_memory_search`
- `handle_delete_by_timeframe` → `handle_memory_delete`
- `handle_delete_before_date` → `handle_memory_delete`

### Unit Tests

**`tests/unit/test_memory_service.py`**
- `store_memory()` Methode
- `delete_memory()` Methode
- Backend-Aufrufe bleiben gleich (Storage-Schicht ändert sich nicht)

**`tests/unit/test_fastapi_dependencies.py`**
- Referenziert `store_memory`, `delete_memory` in Dependency-Tests

**`tests/conftest.py`**
- `delete_by_tag()` für Test-Cleanup → muss auf neues API mappen

---

## 2. Dashboard / HTTP API

### Keine direkten Änderungen nötig

Das Dashboard nutzt die HTTP REST API:
- `/api/memories` (CRUD)
- `/api/search` (Suche)
- `/api/consolidation` (Konsolidierung)
- `/api/analytics` (Statistiken)

Diese Endpunkte rufen intern `MemoryService` auf, aber die Methoden-Namen des Service ändern sich nicht für die HTTP-Schicht.

### Mögliche Verbesserungen (Optional)

Die HTTP-API könnte um neue Unified-Endpunkte erweitert werden:
- `POST /api/memories/search` mit mode, time_expr, quality_boost
- `DELETE /api/memories/bulk` mit flexiblen Filtern

---

## 3. Strategie

### Option A: Deprecation Layer (Empfohlen)
1. Alte Handler bleiben funktionsfähig
2. Neue Handler werden hinzugefügt
3. Tests laufen weiterhin
4. Migration kann schrittweise erfolgen

### Option B: Vollständige Migration
1. Alle Handler sofort umbenennen
2. Alle Tests anpassen
3. Breaking Change in einer Version

**Empfehlung: Option A** - Die in Phase 5 erstellte `compat.py` ermöglicht beide Varianten.

---

## 4. Aufgaben

| Phase | Task | Datei |
|-------|------|-------|
| 7 | Test-Migration für Handler | `phase7-test-handler-migration.md` |
| 8 | Unit-Test-Anpassungen | `phase8-unit-test-updates.md` |
| 9 | HTTP API Erweiterungen (Optional) | `phase9-http-api-enhancements.md` |

---

## Nächste Schritte

1. **Sofort:** Phase 5 (Deprecation Layer) abschliessen
2. **Dann:** Phase 7 (Test-Handler-Migration) durchführen
3. **Optional:** Phase 9 (HTTP API) für Dashboard-Verbesserungen
