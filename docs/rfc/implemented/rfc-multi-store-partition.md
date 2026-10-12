# RFC: Multi-Store Partition

**Status:** Implemented-with-open-debt  
**Date:** 2026-10-12 (Retroativa)  
**Branch:** Implementado via PRs #57, #62, #68, #1395, #1469  
**Repo:** `~/git/mcp-memory-service`  

---

## Status: IMPLEMENTED com dívida técnica aberta

Feature **operacional** no código desde commit `53745ac0`. Schema, interface e wiring implementados. **Dívida aberta:** bugs #1465/#1466 (delete e search não-semânticos vazam entre stores).

---

## Contexto e Motivação

### Problema Original

O memory-service armazenava todas as memórias em um namespace único, misturando:
- Memórias operacionais de sessão (checkpoints, decisões, learnings)
- Documentação ingerida (PDFs, wikis, manuais)
- Dados específicos por projeto (MIR, RER, hackathons)
- Compartilhamento entre domínios (docs-shared vs projeto-específico)

Resultado: poluição semântica, vazamento de contexto entre projetos, e dificuldade de gerenciar ciclos de vida diferentes (memória de sessão é volátil, documentação é persistente).

### Motivação da Partição

1. **Separação de domínios:** memórias operacionais vs documentação técnica
2. **Isolamento por projeto:** MIR não deve "ver" memórias RER em buscas semânticas
3. **Gestão de ciclo de vida:** diferentes políticas de retenção por store
4. **Performance:** particionar busca vetorial por store reduz espaço de busca

---

## Design Implementado

### Schema: Coluna `store` na Tabela `memories`

```sql
ALTER TABLE memories ADD COLUMN store TEXT DEFAULT 'default'
```

**Arquivo:** `src/mcp_memory_service/storage/mixins/migrations.py:323`

A coluna `store` funciona como chave de partição lógica. Memórias sem store explícito ficam em `'default'`.

### Partição Vetorial: `vec0` com `partition key`

```sql
CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    content_embedding FLOAT[384] distance_metric=cosine,
    store TEXT partition key
)
```

**Arquivo:** `src/mcp_memory_service/storage/mixins/migrations.py:284`

O `partition key` do sqlite-vec (vec0) particiona fisicamente as embeddings por store, otimizando buscas semânticas.

### Interface: Parâmetro `store` em Storage Base

Todos os métodos críticos receberam parâmetro `store: Optional[str]`:

- `store()` - **L87:** `store: Optional[str] = "default"`
- `retrieve()` - **L138:** `store: Optional[str] = None`
- `get_by_hash()` - **L439:** `store: Optional[str] = None` (a verificar)
- `delete_memories()` - **L495/503:** `store: Optional[str] = None` (a verificar)
- `search_memories()` - **L1129/1143:** `store: Optional[str] = "default"`
- `count_all_memories()` - **L971:** `store: Optional[str] = None` (a verificar)
- `get_all_tags_with_counts()` - **L882:** `store: Optional[str] = None` (a verificar)
- `get_memory_timestamps()` - **L1039:** `store: Optional[str] = None` (a verificar)

**Arquivo:** `src/mcp_memory_service/storage/base.py` (linhas confirmadas por leitura direta)

### Stores em Uso

Baseado em evidência de commits e uso operacional:
- `default` - memórias operacionais de sessão
- `docs` - documentação específica por projeto  
- `docs-shared` - documentação compartilhada entre projetos
- `mir` - memórias específicas do projeto MIR
- `hackathon` - contexto de hackathons e eventos

---

## O que Funciona (Scope Correto)

1. **Store/retrieve:** memórias são salvas no store correto e recuperadas com filtro por store
2. **Busca semântica:** `search_memories` com `mode="semantic"` respeita parâmetro `store`, consulta só o partition correto
3. **Busca ranked:** buscas com ranking semântico honram filtro por store
4. **Migrations:** banco existente migra sem corrupção, store NULL vira `'default'` automaticamente

---

## Dívida Aberta (Bugs Reportados)

### Bug #1465: `delete_memories` aceita `store` mas não propaga

**Descrição:** O método `delete_memories` aceita parâmetro `store` na interface, mas os branches internos (delete por hash, por tags, por intervalo de tempo) não propagam o filtro. Resultado: delete cruza stores inadvertidamente.

**Raiz:** implementação incompleta dos ramos não-semânticos.

### Bug #1466: `search_memories` vaza stores em modos não-semânticos

**Descrição:** `search_memories` só honra parâmetro `store` em `mode="semantic"` e `mode="ranked"`. Os modos `hybrid`, `quality_boost`, `exact`, `tag-only` e `time-only` vazam memórias de outros stores.

**Raiz:** os ramos não-semânticos não propagam parâmetros de filtragem uniformemente.

### Raiz Comum

Ambos os bugs decorrem do mesmo padrão: **a interface declara suporte completo ao parâmetro `store`, mas os ramos não-semânticos/não-primários não propagam filtros uniformemente**. Mesmo padrão observado no INC-5 (re-rank sem propagação de parâmetros).

**Impacto:** delete e busca "vazam" entre stores quando não usam path semântico principal.

---

## Decisões de Arquitetura

### Por que coluna (não tabela separada)

1. **Reuso de infraestrutura:** todas as features existentes (quality rating, superseding, graph associations, harvesting) funcionam transparentemente
2. **Atomic operations:** store+retrieve+update na mesma transação SQLite
3. **Partition key nativo:** sqlite-vec otimiza fisicamente por store sem overhead

### `store='default'` como fallback

Memórias sem store explícito ficam em `'default'`, preservando backward compatibility. Store `NULL` não é permitido - migração força `'default'`.

### Propagação de parâmetros incompleta (dívida)

A interface foi expandida uniformemente, mas a implementação seguiu **path semântico first**: semântica e ranking funcionam, outros modos ficaram para depois. Dívida consciente para delivery incremental.

---

## Histórico de Implementação

| Commit | PR | Descrição |
|--------|-----|-----------|
| `53745ac0` | #57 | **Phase 1:** store partition key em vec0 + coluna memories |
| `e6549460` | #57/#62 | **Phase 2:** multi-store partition key + store param wiring |
| `02e3ba31` | #68 | **Hotfix:** multi-store vec0 migration aborts on existing DB |
| `61f89db4` | #1395 | **Scope:** memory/search/analytics by store |
| *commit* | #1469 | **Context:** store path honra MCP_MEMORY_INCLUDE_HOSTNAME |

### Arco de Implementação: #57 → #62 → #68 → #1395 → #1469

- **#57:** fundação (schema + vec0 partition)
- **#62:** wiring da interface (store params)
- **#68:** fix de migração (falha em banco existente)
- **#1395:** scope completo (analytics + search)
- **#1469:** context path (hostname no store)

---

## Evidência no Código

- **Schema:** `storage/mixins/migrations.py:323` (`ALTER TABLE`) + `migrations.py:284` (`vec0 partition key`)
- **Interface:** `storage/base.py:87,138,1129` (store params confirmados por leitura)
- **Stores ativos:** grep por `store.*TEXT` confirma uso em `default`, `docs`, `docs-shared`, `mir`, `hackathon`
- **Dívida:** issues #1465/#1466 (delete e search não-semânticos vazam stores)

---

## Próximos Passos (Dívida)

1. **Fix #1465:** propagar `store` filter em todos os ramos de `delete_memories`
2. **Fix #1466:** propagar `store` filter em modos `hybrid`/`exact`/`tag-only`/`time-only`
3. **Teste E2E:** validar isolamento completo entre stores
4. **Documentação:** usage guide para escolha de store por caso de uso

---

**Implementadores:** filhocf (Zero), Claudio Ferreira Filho  
**Reviewers:** Kenzokono (issues #1465/#1466), doobidoo (upstream)  
**Status final:** OPERACIONAL com bugs conhecidos e limitados