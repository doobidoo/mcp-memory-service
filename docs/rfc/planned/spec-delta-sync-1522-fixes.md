# SPEC — delta-sync audit fixes (#1522): tombstone, bootstrap caller, embedding_pending

**Status:** #3 and #4 MERGED upstream (`#1530`); #2 UNBLOCKED — Henry sign-off 2026-10-16 (issuecomment-6204519097), go-ahead for producer+apply+tests+ADR-0012 v2 in one PR · **Updated:** 2026-10-16 · **Fork-only · delta-sync (#1345)**
**Issue:** #1522 (doobidoo audit). **G0:** seven, evidence-based (code read post-v12 merge; re-confirmed 2026-10-15 against upstream/main f6bec0a6 / v12.1.0).
**Scope of this SPEC:** finding #2 only (the one still open). #1 fixed by `da8772c9` (#1523). #3 (bootstrap caller) and #4 (embedding_pending) fixed by `#1530`. #5 (sync_events unbounded) is doc-only per ADR-0009.
**ADR:** #2 needs **ADR-0012 v2** — the first version said "apply honors the tombstone" assuming that clause was unambiguous, but it is NOT: a legitimate same-hash re-create and a clock-ahead resurrection are the same physical event sequence, so an unconditional tombstone-preserve breaks legitimate re-create. ADR-0012 v2 must add an **intentionality signal on the event**. Henry resolved the two open calls on 2026-10-16: `recreate_of` on the **payload** naming the **delete's `event_id`**; **absence = revive**; **no `schema_version` bump** (would fail-stop v1 peers); REST/force path split to **#1560**. See the RECONCILED section below.

> ⚠️ CORRECTION (2026-10-15): an earlier revision of this SPEC claimed REQ-2.1 (preserve the
> tombstone whenever one exists) was confined to `apply.py` and that "a genuine re-create has
> different content → different hash, so there is no conflict". **Both claims were wrong.** The
> repo supports re-creating the SAME hash after a delete (`test_old_edit_does_not_restore_recreated_memory_field`
> passes today and requires the memory to come back ALIVE). That test and the anti-resurrection
> RED test (`test_create_does_not_resurrect_tombstone`, xfail) are the same `delete@hlc20 →
> create same-hash@hlc30` sequence with opposite expected outcomes. HLC cannot tell them apart;
> the fix therefore needs a producer-side signal and is NOT confined to apply. See the revised
> finding #2 below.

## Problem (per finding, confirmed in code)

- **#2 tombstone resurrection (corruption) — STILL OPEN.** `resolver._winner_key` orders by HLC first (ADR-0010); "delete beats update" only breaks an HLC tie. So a `create` with HLC strictly greater than a `delete` wins the resolver. `apply._materialize_event` (ramo `op in (create, update)`) then runs `INSERT OR REPLACE` with `deleted_at` **hard-coded NULL** and no tombstone check → the deleted memory is resurrected.
  - **Why it is not a confined apply fix (the correction):** the same `delete@hlc20 → create same-hash@hlc30` sequence is BOTH a malicious clock-ahead resurrection (must stay dead — `test_create_does_not_resurrect_tombstone`, xfail) AND a legitimate re-create (must come alive — `test_old_edit_does_not_restore_recreated_memory_field`, passes). No field already on the event (`op`, `event_id`, `agent_id`, HLC, `created_at`) distinguishes them. An unconditional "preserve tombstone" closes the hole but breaks legitimate re-create; doing nothing keeps the bug. It is an impossibility result without extra information.
  - **The missing bit exists at the producer:** `StoreMixin.store()` calls `_purge_tombstone(content_hash)` immediately before writing a re-create, so the producer KNOWS the hash was deleted — but the emitted `create` payload carries nothing of it. The fix must carry an intentionality signal from the producer (`store.py`) into apply. **Not confined to apply.py.** Pending ADR-0012 v2 + Henry's call on field format and back-compat.
- **#3 bootstrap unreachable — FIXED upstream (`#1530`).** `sync_from_peer` now calls `bootstrap_from_peer` on first sync. Kept here for history; no code owed.
- **#4 embedding_pending never cleared — FIXED upstream (`#1530`, also #1521).** Re-embed under the matching model now clears the flag with its store partition. Kept here for history; no code owed.

## Requirements (EARS)

### #2 — apply honors the durable tombstone (RECONCILED — Henry sign-off 2026-10-16)

> **Decision gate: CLEARED.** Henry answered the two open calls on #1522
> (comment `2026-10-16T08:26:54Z`, issuecomment-6204519097): (1) the signal is a
> **payload** key `recreate_of` referencing the **delete's `event_id`** (not `deleted_at`,
> which carries the very clock skew this fix fights); (2) **absence = revive**, and
> **no `schema_version` bump** (a bump fail-stops v1 peers at `apply_remote_event`
> `event_schema_version > KNOWN_SCHEMA_VERSION`, apply.py:132 — it would stall every
> un-updated client). An optional payload key is additive. Scope caveat: the REST
> store/force path is a separate item (**#1560**), not covered here.
> Henry: *"Go ahead with producer, apply, both tests and the ADR revision in one PR.
> Once it passes, the xfail on `test_create_does_not_resurrect_tombstone` should come off."*

#### The reconciled contradiction (answers the central question)

The apparent conflict was: Henry says **absence = revive**, yet the old INV-2a
(`test_create_does_not_resurrect_tombstone`, xfail) asserts a **bare create** (no
`recreate_of`) with HLC > delete **stays dead**. These are **directly opposite** and
**cannot both hold** — a bare create cannot simultaneously revive (Henry) and stay dead
(old INV-2a). They are the SAME physical sequence (`delete@hlc20 → create same-hash,
new event_id, @hlc30`) with no distinguishing field (filhocf, #1522
2026-10-15T23:36:02Z; apply.py has no per-op discriminator — the resolver orders by HLC
only, resolver evidence at apply.py:254 `reduce_events`).

**Resolution: hypothesis (c) is correct.** Not (a) — apply cannot tell "peer that should
have sent `recreate_of`" from "old peer"; both emit a bare create and Henry explicitly
chose to revive on absence rather than guess. Not (b) — HLC/replay cannot separate the
cases (that is the impossibility result; a genuine re-create and a clock-ahead
resurrection share the same HLC ordering). The discriminator is **solely the presence of
a valid `recreate_of`**. Under that rule, the only coherent reading of "once it passes"
is: a bare create over a tombstone **revives** (absence = revive). Therefore the old
INV-2a — which demands a bare create stay dead — **encodes the pre-decision strict
reading and is now WRONG**. It must be rewritten (see "Tests" below). The anti-resurrection
guarantee Henry still wants is honored in the case **he actually described**: the
clock-ahead resurrection he names is the **REST/force path** (#1560), explicitly carved
out of this PR — not the sync-path bare create.

#### Deterministic apply rule (create over a tombstone)

Given an incoming `create` event `E` for `content_hash = H`, that has **won the resolver**
(apply.py:254–267; a loser is recorded, not materialized), with payload field
`recreate_of` (optional). Let `T` = the local row for `H` with `deleted_at IS NOT NULL`
(the durable tombstone), if any. Let `D_cur` = `event_id` of the latest local `delete`
event for `H` in `sync_events` ordered by `(hlc_physical, hlc_logical)` DESC
(`_recreate_supersedes_tombstone`, apply.py:590–612).

```
materialize_create(E, H):
    if no live-or-tombstoned constraint (T is None):          # INV-2c: fresh hash
        INSERT OR REPLACE (deleted_at = NULL)                 # normal create
        return MATERIALIZED

    # T exists: H currently carries a durable tombstone.
    r = E.payload.get("recreate_of")                          # may be absent (v1 peer)

    if r is None:                                             # ABSENCE
        INSERT OR REPLACE (deleted_at = NULL)                 # == today's behavior: REVIVE
        return MATERIALIZED                                   # (Henry: absence = revive; INV-2b legit re-create)

    # r present: honor it ONLY if it names the CURRENT tombstone.
    if r == D_cur:                                           # valid supersession
        INSERT OR REPLACE (deleted_at = NULL)                 # REVIVE
        return MATERIALIZED
    else:                                                     # stale/unrelated delete named
        leave row untouched (tombstone preserved)             # DO NOT revive
        return PROCESSED (cursor advances; not a materialization failure)
```

Key facts this rule pins down:
- **Which event_id it names:** `recreate_of` names the **delete's `event_id`**, matched
  against `D_cur` (the latest local delete for `H`). Henry's rationale: an `event_id`
  names exactly one delete; `deleted_at` is wall-clock and skew-prone.
- **The HLC:** irrelevant to the materialization decision (REQ-2.2). The resolver already
  used HLC to pick the winner; apply does NOT re-derive intent from HLC. A create that lost
  the resolver never reaches this rule.
- **Absence revives.** This is the single line that was inverted in the first
  implementation (apply.py:669–677 currently does the opposite: "no `recreate_of` ⇒
  preserve tombstone"), which broke the two legitimate-re-create regressions.
- **Delete is terminal (REQ-2.3):** an incoming `delete` just writes the tombstone
  (apply.py:631–634); this rule only gates `op == create`.
- **Back-compat (REQ-2.5, Henry):** a v1 apply that does not read `recreate_of` ignores
  the key and revives (its existing behavior) — additive, no fail-stop
  (`test_v1_apply_ignores_unknown_recreate_of_key`).

#### What this changes in the current branch

The helper `_recreate_supersedes_tombstone` (apply.py:590) is **correct and kept** — it is
exactly the `r == D_cur` check. The guard block at **apply.py:664–684 is INVERTED and must
flip**: today `if not recreate_of or not _recreate_supersedes_tombstone(...) → preserve
tombstone`. The reconciled rule is: **preserve ONLY when `recreate_of` is present AND names
a stale/unrelated delete**; absent `recreate_of` ⇒ revive; present-and-current ⇒ revive.
(Design only — no code written here.)

#### Requirements (EARS) — #2, finalized

- REQ-2.1 (final): WHEN `_materialize_event` processes a resolver-winning `create` whose
  `content_hash` carries a durable local tombstone, THE apply step SHALL revive the memory
  UNLESS the event carries a `recreate_of` that names a delete OTHER than the current
  tombstone's delete event, in which case it SHALL preserve the tombstone.
- REQ-2.2: THE decision SHALL be independent of HLC and of the resolver outcome beyond
  "this create won"; it SHALL hold even when the create's HLC is strictly greater than the
  delete's.
- REQ-2.3: WHEN the incoming event is a `delete`, THE apply step SHALL write/keep the
  tombstone (deletes are terminal, not gated by this rule).
- REQ-2.4: THE producer (`store.py`) SHALL stamp `recreate_of = <event_id of the delete it
  saw>` on the create payload WHEN `_purge_tombstone(content_hash)` found and cleared a
  tombstone at store time; apply (`apply.py`) SHALL consume it. The resolver SHALL NOT change.
- REQ-2.5 (back-compat, Henry): WHEN a `create` carries no `recreate_of` (older client),
  THE apply step SHALL revive (today's behavior). THERE SHALL be NO `schema_version` bump;
  the key is additive and a v1 apply SHALL ignore it without error or stall.
- INV-2a (REWRITTEN — see Tests): a `create` carrying `recreate_of` that names a
  **stale/unrelated** delete (not the current tombstone) SHALL leave the memory deleted.
  (Replaces the old "bare create stays dead", which contradicted Henry's absence=revive.)
- INV-2b (legitimate re-create): a `create` carrying `recreate_of = <current delete
  event_id>` SHALL revive; AND a bare `create` (absence) SHALL ALSO revive (zero regression).
- INV-2c (no false block): a first `create` of a hash with NO prior tombstone SHALL
  materialize normally.

#### Tests — stays / changes / rewrite

| Test (file) | Verdict | Why |
|---|---|---|
| `test_create_does_not_resurrect_tombstone` (tests/sync/test_1522_audit_fixes.py:112, **xfail**) | **REWRITE + drop xfail** | Its step-3 bare create asserts "stay deleted" — the exact INV-2a strict reading Henry **rejected** (absence=revive). As written it would FAIL after the fix (the memory revives). Rewrite so the resurrection case it protects uses a **stale `recreate_of`** (names a delete ≠ current tombstone) — that is the case that now stays dead. Then remove `@pytest.mark.xfail`. Henry's "once it passes, the xfail should come off" presupposes a test whose "stays dead" case is the *stale-signal* case, not the bare case. |
| `test_recreate_signal_discriminates_stale_vs_valid` (test_1522_audit_fixes.py) | **KEEP** | The discriminating test (reconciled 2026-10-16). Same physical sequence (create → delete → later create, HLC > delete) yields outcomes driven only by `recreate_of`: Case A — a **bare** create (absence of signal) **REVIVES** (Henry: absence = revive); Case B — a create naming the **current** delete event_id **REVIVES** (INV-2b); Case C — a create naming a **stale/unrelated** delete **STAYS DELETED** (INV-2a). The genuine discriminator is "stale vs current `recreate_of`", not "bare vs signalled" — bare and valid share the revive outcome, only a stale signal preserves the tombstone. |
| `test_recreate_of_names_stale_delete_does_not_revive` (test_1522_audit_fixes.py) | **KEEP** | This is the real INV-2a: `recreate_of` naming an unrelated delete does NOT revive. Already correct. |
| `test_v1_apply_ignores_unknown_recreate_of_key` (test_1522_audit_fixes.py) | **KEEP** | Encodes REQ-2.5 back-compat exactly (Henry's explicit ask). |
| `test_legit_create_without_tombstone_materializes` (test_1522_audit_fixes.py:172) | **KEEP** | INV-2c, unaffected. |
| `test_old_edit_does_not_restore_recreated_memory_field` (tests/test_delta_sync_apply_regressions.py:260) | **KEEP, must go green** | Bare recreate (`c2`, no `recreate_of`) at hlc30 over a tombstone MUST revive → `deleted_at IS None`. Broke under the first (inverted) impl; the reconciled "absence=revive" restores it. |
| `test_old_metadata_key_does_not_return_after_recreation` (regressions.py:331) | **KEEP, must go green** | Same shape (`c2` bare recreate, empty metadata) — must revive and drop the dead incarnation's metadata. Restored by absence=revive. |

**Net:** 2 regressions turn green via the one-line inversion of the apply guard (absence ⇒
revive). 1 xfail is rewritten + un-xfailed around the *stale-signal* resurrection case. 1
discriminator test flips its bare-create case. 3 tests unchanged.

### #3 — orchestrator calls bootstrap on first sync

- REQ-3.1: WHEN `sync_from_peer` finds no `sync_cursor` row for `peer_id` (first sync with that peer) AND the peer adapter exposes `get_baseline`, THE orchestrator SHALL call `bootstrap_from_peer` before the pull loop, then continue the pull loop from the parked watermark.
- REQ-3.2: WHEN a `sync_cursor` row already exists, THE orchestrator SHALL NOT bootstrap (normal incremental pull only).
- REQ-3.3: IF the peer adapter does not expose `get_baseline`, THE orchestrator SHALL fall back to the current behavior (pull from cursor 0) without error (back-compat with peers that cannot serve a baseline).
- INV-3a: bootstrap SHALL run at most once per peer (guarded by cursor existence); a second `sync_from_peer` SHALL NOT re-bootstrap.

### #4 — clear embedding_pending when the model matches

- REQ-4.1: WHEN a memory with `embedding_pending=1` is (re-)embedded under the local model — whether by a maintenance re-embed or a sync-time hook — THE system SHALL set `embedding_pending=0` in the same transaction that writes the vector.
- REQ-4.2: THE maintenance script `repair_missing_embeddings_onnx.py` SHALL clear `embedding_pending=0` for each row it successfully embeds.
- REQ-4.3: WHEN `_materialize_event` stores a pulled create whose `embedding_model` now MATCHES the local `embedding_model_name`, THE apply step SHALL embed and store the vector with `embedding_pending=0` (this path already exists at apply.py:632 via `can_embed`; the gap is only the backfill of rows stored pending earlier under a mismatch).
- INV-4a: after a re-embed of a pending row under the matching model, the row SHALL be returned by a store-scoped semantic search (previously invisible → now visible).
- INV-4b: a row still genuinely pending (model still mismatched, no vector) SHALL remain excluded from search (the fix clears only rows that got their vector).

## Out of scope
- #1 (push allow-list): fixed by `da8772c9`/#1523.
- #3 (bootstrap caller), #4 (embedding_pending): fixed by `#1530` — REQs below kept for history.
- #5 (sync_events compaction): ADR-0009 leaves it out; doc-only.
- Resolver changes: none (resolver stays correct; the fix is at producer + apply).
- ~~Re-create semantics (same hash intentionally brought back): out — a durable tombstone is terminal; a genuine new memory has different content → different hash.~~ **STRUCK (wrong):** the repo supports same-hash re-create after delete; that is exactly the case that makes #2 need an intentionality signal. In scope for ADR-0012 v2.

## Acceptance criteria
- INV-2a passes: a RED test resurrects a delete today; after the fix the memory stays deleted. prove-fix: without the apply change, the test fails.
- INV-3a passes: a fresh spoke (no cursor) installs the baseline; a second sync does not re-bootstrap.
- INV-4a passes: a pending row becomes searchable after re-embed under the matching model; `repair_missing_embeddings_onnx.py` clears the flag.
- Neighbor suite green: `tests/` for sync (apply/resolver/orchestrator/bootstrap) + retrieve/hybrid store-scoped, no regression.

---

## P1 convergence — a decisão revive-vs-preserva precisa ser função do CONJUNTO de eventos (Greptile, apply.py:674)

> **G0 arquitetura (seven), 2026-10-16.** Análise read-only do worktree `mms-pr-recreate-of`.
> O achado do Greptile é **REAL e procedente**. O fix do `recreate_of` (ADR-0012 v2) resolveu
> *qual create é legítimo*, mas **não** tornou a materialização convergente: o estado final
> ainda depende da ORDEM de chegada dos eventos. Isto viola a invariante de convergência do
> delta-sync (RFC §8.2: o estado final é função determinística do CONJUNTO de eventos, não da
> ordem). Esta subseção fixa a regra determinística corrigida, prova a convergência para
> qualquer ordem, e lista os testes que faltam. **Design only — nenhuma linha de produção escrita aqui.**

### 1. Diagnóstico — por que o estado atual NÃO converge

O cenário do achado (3 eventos para o mesmo `content_hash H`):

| Evento | HLC | Papel |
|--------|-----|-------|
| `C0` create "vivo" | (assuma anterior) | a memória existiu viva |
| `D1` delete | **HLC20** | o tombstone *legítimo e atual* |
| `C2` re-create | **HLC30** | `recreate_of` nomeia um delete **STALE** (um delete ≠ `D1`, ex.: um delete antigo de uma encarnação anterior) |

`C2` vence o resolver (HLC30 > HLC20) em TODO peer — isso é convergente (resolver.py:`_winner_key`/
`reduce_events`, puro). O problema está **depois do resolver**, no apply. Há **duas** fontes de
não-convergência, ambas estruturais:

**Fonte 1 — a decisão lê o ESTADO MUTÁVEL da linha, não o conjunto de eventos.**
O guard em `apply._materialize_event`, ramo `create` (apply.py:664-684), decide olhando
`SELECT deleted_at FROM memories WHERE content_hash = H` (apply.py:666-668): *"a linha está
deletada AGORA?"*. "Agora" depende de qual evento chegou primeiro.

- **Ordem A (delete antes do create):** quando `C2` é aplicado, `D1` já materializou o
  tombstone → `tomb[0] is not None` → o guard roda → `recreate_of` é stale →
  `_recreate_supersedes_tombstone` retorna False → **PRESERVA** (apply.py:677-684). Estado final: **MORTO**. ✔ correto.
- **Ordem B (create antes do delete):** quando `C2` é aplicado, `D1` ainda não chegou → a linha
  não tem tombstone (`tomb is None` ou `deleted_at IS NULL`) → o guard **nem roda** →
  `INSERT OR REPLACE ... deleted_at = NULL` (apply.py:~700) → materializa **VIVO**. Depois `D1`
  chega, **PERDE** o resolver (HLC20 < HLC30), cai no ramo *"Remote event lost conflict
  resolution"* (apply.py:285) → gravado em `sync_events` mas **nunca re-materializa** → o
  tombstone nunca é imposto. Estado final: **VIVO**. ✘ divergente.

Dois peers com o MESMO conjunto `{C0, D1, C2}` convergem para estados DIFERENTES. Re-pull não
repara: a segunda vez que cada evento chega bate na guarda de idempotência (apply.py:~150,
`pre_existing == payload` → "Duplicate event, no rewrite") e pula o write. **A divergência é permanente.**

**Fonte 2 — `D_cur` ("último delete LOCAL conhecido") é função da ordem.**
`_recreate_supersedes_tombstone` (apply.py:590-612) resolve `D_cur` com
`SELECT event_id FROM sync_events WHERE content_hash=H AND op='delete' ORDER BY hlc DESC LIMIT 1`.
Isso é o último delete **que já chegou neste peer** — não o último delete do conjunto global.
Se o delete de maior HLC ainda não chegou, `D_cur` aponta para um delete mais antigo, e um
`recreate_of` que seria stale contra o delete global pode parecer válido contra o `D_cur` parcial
(e vice-versa). Mesmo sem a Fonte 1, isto já quebra convergência sob chegada parcial.

**Conclusão do diagnóstico:** o resolver é convergente; o apply **não é**, porque materializa a
partir do estado presente da linha e de um `D_cur` parcial, ambos dependentes da ordem. A pergunta
central ("como tornar a decisão convergente?") se responde tornando a decisão um **fold puro sobre
o conjunto de eventos do hash** — exatamente o que o resolver já faz para ordenação, e o que
`_materialize_update_metadata` já faz para metadados (apply.py:`_load_live_events`/`_resolve_md_fields`,
que recomputam a encarnação viva a partir de TODOS os eventos, por isso convergem mesmo quando o
evento corrente não vence — ver comentário apply.py:261-266).

### 2. Resposta à pergunta central + avaliação das opções (a)/(b)/(c)

**A decisão revive-vs-preserva NÃO deve olhar "a linha está deletada agora?". Deve ser computada
como função pura do conjunto de sync_events para o hash — idêntico em todo peer que tenha o mesmo
conjunto, independente de ordem.** Isto é a opção **(a) na sua forma correta**, combinada com a
necessidade de **(b)** para fechar a chegada parcial. Veredito por opção:

- **(a) "comparar `recreate_of` com TODOS os deletes registrados em sync_events para o hash;
  revive sse nomeia o delete de MAIOR HLC conhecido"** — **direção CERTA** (função do conjunto),
  mas a ressalva que você mesmo levantou é fatal se ficar só nisso: *"e se o delete de maior HLC
  ainda não chegou?"*. Enquanto o conjunto local for parcial, "maior HLC **conhecido**" ≠ "maior
  HLC **global**", e a decisão ainda oscila. (a) sozinha converge só no estado FINAL (todos os
  eventos presentes), não durante chegadas parciais.
- **(b) "um delete que perde o resolver deve ainda poder impor o tombstone se o create não o
  supersede legitimamente"** — **necessária** e é o que conserta a Fonte 1 na Ordem B. Hoje o
  ramo lost-conflict (apply.py:285) é um no-op de materialização; ele precisa **re-derivar o
  estado vivo/morto do hash a partir do conjunto** sempre que QUALQUER evento daquele hash chega,
  exatamente como `update_metadata` já re-materializa incondicionalmente (apply.py:261-266).
- **(c)** — a resposta é a **síntese (a)+(b)**: materialização do estado vivo/morto vira um
  **recompute determinístico sobre o conjunto**, disparado por TODO evento do hash (create OU
  delete, vença ou perca o resolver), e a validade do `recreate_of` é avaliada contra o **conjunto
  de deletes conhecido** (não contra o estado da linha). Com isso, dado o mesmo conjunto, o estado
  é o mesmo — e cada chegada parcial produz o melhor estado determinístico possível para aquele
  subconjunto, convergindo monotonicamente conforme o conjunto completa.

### 3. Regra determinística corrigida (pseudocódigo)

Modela-se o estado do hash `H` como uma **função pura `materialize_liveness(events(H))`** sobre
TODOS os eventos de `H` em `sync_events` (o conjunto é idêntico em todo peer que os tenha; a
ordem de inserção é irrelevante). Substitui o "olhe `deleted_at` agora".

```
# Executado em apply, para QUALQUER evento recebido de hash H (create OU delete),
# APÓS inserir o evento em sync_events (idempotente) e rodar o resolver.
# NÃO lê memories.deleted_at para decidir; deriva a liveness do CONJUNTO.

def recompute_liveness(events_H):                     # events_H = todos os sync_events de H
    creates = [e for e in events_H if e.op in ('create','update')]
    deletes = [e for e in events_H if e.op == 'delete']

    if not creates:
        return DEAD if deletes else ABSENT            # só delete(s) → morto; nada → inexistente

    # (i) eleger o create vencedor pelo MESMO _winner_key do resolver (determinístico,
    #     já usado em _load_live_events). Esta é a "encarnação viva candidata".
    Cwin = min(creates, key=_winner_key)              # menor key vence (resolver.py)

    # (ii) deletes que são logicamente POSTERIORES-ou-iguais ao create vencedor.
    #     ADR-0012: na ordenação, delete vence update em empate de HLC (_winner_key põe
    #     delete à frente). Portanto "delete domina o create" sse _winner_key(delete) <= _winner_key(Cwin).
    dominating = [d for d in deletes if _winner_key(d) <= _winner_key(Cwin)]

    if not dominating:
        # nenhum delete domina o create vencedor → VIVO. (create é o evento de maior ordem)
        return LIVE(Cwin)

    # (iii) há delete(s) dominando: a memória está morta A MENOS QUE o create vencedor
    #     carregue um recreate_of que nomeie legitimamente o delete dominante de MAIOR ordem.
    Dtop = min(dominating, key=_winner_key)           # o delete dominante "mais forte"
    r = Cwin.payload.get("recreate_of")
    if r is not None and r == Dtop.event_id:
        return LIVE(Cwin)                             # supersessão legítima (INV-2b)
    else:
        return DEAD                                   # ausência OU recreate_of stale → morto
```

> **⚠️ Nota de semântica vs ADR-0012 v2 (absence = revive):** o pseudocódigo acima aplica
> `recreate_of` contra o **delete dominante do conjunto** (`Dtop`), não contra o estado da linha.
> No cenário do achado `C2@HLC30` **vence** `D1@HLC20` no `_winner_key` (HLC30 > HLC20), logo
> `D1` **não domina** `C2` → ramo (ii) retorna `LIVE`. **Isto é "absence = revive" do Henry,
> mas também reviveria o re-create com recreate_of stale** — ou seja, converge para VIVO nas duas
> ordens, o que **conserta a divergência**, mas muda o resultado da Ordem A (que hoje preserva o
> tombstone). **Há uma decisão de produto embutida aqui que precisa do Henry** (ver §6 "Decisão
> pendente"): a convergência exige escolher UM estado para `{C0,D1,C2 stale}` — ou VIVO (regra
> acima, alinhada a "create de maior HLC vence") ou MORTO (regra alternativa abaixo). O que NÃO é
> admissível é o estado atual (VIVO ou MORTO conforme a ordem).

**Regra alternativa (se o produto exigir que recreate_of stale fique MORTO mesmo vencendo HLC):**
trocar o ramo (iii) para que um `recreate_of` **presente e stale** force `DEAD` independentemente
da ordem HLC, e **ausência** reviva (back-compat v1). Isso mantém a semântica ADR-0012 v2 da Ordem A
e a estende determinísticamente à Ordem B:

```
    # (iii') variante "stale fica morto" — alinhada ao ADR-0012 v2 Ordem A
    r = Cwin.payload.get("recreate_of")
    if r is None:               return LIVE(Cwin)     # ausência = revive (Henry, INV-2b/REQ-2.5)
    # r presente: só revive se nomeia o delete dominante de maior ordem do CONJUNTO
    known_deletes = {d.event_id for d in deletes}
    if r in known_deletes and r == min(dominating or deletes, key=_winner_key).event_id:
        return LIVE(Cwin)                             # supersessão válida
    if r not in known_deletes:
        return UNDECIDED_DEAD                          # o delete nomeado ainda não chegou → trata como morto por ora (monotônico; revê quando chegar)
    return DEAD                                        # recreate_of stale → morto (INV-2a), em QUALQUER ordem
```

Ambas as variantes são **função pura do conjunto** → convergentes. A diferença é só *qual* estado
final `{C0,D1,C2-stale}` assume. A recomendação arquitetural é a **variante (iii')**: preserva a
decisão de design do ADR-0012 v2 (stale não ressuscita) e a torna order-independent, que é
exatamente o buraco do Greptile. O `UNDECIDED_DEAD` sob chegada parcial é **seguro e monotônico**:
tratar como morto até o delete nomeado chegar nunca ressuscita indevidamente; quando o delete
chega, `recompute_liveness` roda de novo e assume o estado final correto.

### 4. Onde fica o fix — apply (ambos os ramos), resolver intocado

| Peça | Muda? | O quê |
|------|-------|-------|
| `resolver.py` (`_winner_key`, `resolve`, `reduce_events`) | **NÃO** | Já é puro/convergente. O fix reusa `_winner_key` como ordem canônica. |
| `apply._materialize_event` ramo `create` (apply.py:664-684) | **SIM** | Parar de ler `memories.deleted_at` para decidir. Chamar `recompute_liveness(events(H))` e gravar `deleted_at` conforme o resultado (NULL se LIVE, timestamp do delete dominante se DEAD). |
| `apply` ramo lost-conflict (apply.py:285) + ramo `delete` (apply.py:631-634) | **SIM** | Todo evento de `H` (inclusive um delete que PERDE o resolver, e um create que perde) deve disparar `recompute_liveness` e re-gravar a liveness — como `update_metadata` já faz incondicionalmente (apply.py:261-266). É isto que fecha a Ordem B (Fonte 1) e a chegada parcial. |
| `_recreate_supersedes_tombstone` (apply.py:590-612) | **SUBSTITUÍDO** | Deixa de olhar "último delete local" isolado; a validação de `recreate_of` passa a ocorrer dentro de `recompute_liveness` contra o conjunto de deletes conhecido (`Dtop`/`known_deletes`). |
| `storage/mixins/store.py` produtor `recreate_of` | **NÃO** | O produtor continua carimbando `recreate_of = event_id do delete que purgou`. Correto e suficiente. |

**Veredito: o fix fica inteiramente no APPLY (ambos os ramos de materialização), reusando o
`_winner_key` do resolver. O resolver NÃO muda.** É a mesma arquitetura já adotada para
`update_metadata`: materialização = fold determinístico sobre o conjunto, disparado por todo evento.

### 5. Casos de teste de CONVERGÊNCIA que FALTAM

Os testes atuais (`test_1522_audit_fixes.py`, `test_delta_sync_apply_regressions.py`) exercitam
**uma ordem** por cenário. Falta a classe de testes que prova *mesmo conjunto → mesmo estado
final, para TODA ordem*. Todos sobre o mesmo conjunto base `{C0 create, D1 delete@HLC20,
C2 re-create@HLC30}`:

| Teste (novo) | Conjunto | Ordens aplicadas | Estado final esperado |
|---|---|---|---|
| `test_converge_recreate_stale__all_orders` | `{C0, D1, C2(recreate_of=STALE)}` | as 6 permutações de chegada (C0·D1·C2, C0·C2·D1, D1·C0·C2, D1·C2·C0, C2·C0·D1, C2·D1·C0) | **idêntico nas 6** (MORTO sob variante iii'; VIVO sob iii — fixar conforme Henry) |
| `test_converge_recreate_valid__all_orders` | `{C0, D1, C2(recreate_of=D1.event_id)}` | as 6 permutações | **VIVO** nas 6 (supersessão válida, INV-2b) |
| `test_converge_bare_recreate__all_orders` | `{C0, D1, C2(sem recreate_of)}` | as 6 permutações | **VIVO** nas 6 (absence = revive, Henry/REQ-2.5) |
| `test_converge_delete_loses_resolver_still_tombstones` | `{C2@HLC30 vivo, D1@HLC20}` aplicados **create-primeiro** | C2 depois D1 | **o resultado é o mesmo que D1-primeiro** (fecha Fonte 1 Ordem B: delete que perde resolver ainda recomputa liveness) |
| `test_converge_partial_arrival_monotonic` | chegada incremental `{C2}`, depois `+D1`, depois `+C0` | subconjuntos crescentes | cada passo = estado determinístico do subconjunto; `UNDECIDED_DEAD` nunca ressuscita; estado final = o do conjunto completo |
| `test_converge_two_deletes_pick_dominant` | `{C0, D_old, D1@HLC20(dominante), C2(recreate_of=D_old STALE)}` | ordens variadas | **MORTO** (recreate_of nomeia o delete NÃO-dominante → stale; prova a Fonte 2: `D_cur` parcial não engana mais) |
| `test_converge_repull_is_idempotent_noop` | qualquer dos acima, re-aplicar o conjunto inteiro 2× | — | 2ª passada não muda o estado (idempotência não mascara divergência) |

Padrão de implementação: helper paramétrico que, dado o conjunto de eventos, aplica cada
permutação em um storage limpo e afirma `deleted_at`-final idêntico (property-based via
`itertools.permutations`; para conjuntos maiores, amostra de ordens + o par crítico
create-primeiro/delete-primeiro, que é o que o resolver já faz na brute-force de 11664 pares da
Fase 2 — reusar essa técnica).

### 6. Decisão pendente (produto, para Henry) e o que seven NÃO pôde verificar

- **Decisão de produto (bloqueia a escolha iii vs iii'):** para `{C0, D1@HLC20, C2 re-create@HLC30
  com recreate_of STALE}`, qual é o estado final canônico — **VIVO** (create de maior HLC vence, o
  recreate_of stale é ignorado; simples, mas ressuscita contra a intenção do ADR-0012 v2 Ordem A)
  ou **MORTO** (recreate_of stale nunca ressuscita, em qualquer ordem; preserva a semântica v2 e é
  a recomendação arquitetural, variante iii')? **A convergência exige escolher UM.** O estado atual
  do PR escolhe os DOIS conforme a ordem — esse é o bug. Recomendo **iii' (MORTO)**: mantém a
  decisão já assinada pelo Henry no ADR-0012 v2 e só fecha o buraco de ordem. Precisa do aval dele
  porque muda o resultado observável da Ordem A... não: a Ordem A já dá MORTO hoje; iii' só faz a
  Ordem B também dar MORTO. iii faria ambas darem VIVO (mudaria a Ordem A). Por isso iii' é a de
  menor surpresa.
- **Verificado por seven (evidência):** apply.py:590-612 (`_recreate_supersedes_tombstone` lê
  último delete local), apply.py:664-684 (guard lê `deleted_at` da linha), apply.py:285 (lost-conflict
  é no-op), apply.py:261-266 (update_metadata já recomputa incondicionalmente — o padrão a seguir),
  resolver.py:`_winner_key`/`reduce_events` (puro, convergente), ADR-0012 v2 (absence=revive, REST
  path #1560 fora de escopo), ADR-0010 (ordem canônica `_winner_key`).
- **NÃO verificado por seven:** (1) se existe teste de convergência multi-ordem hoje — busca por
  nome não achou; não é prova exaustiva de ausência. (2) O schema exato de `memories` quanto a
  gravar `deleted_at` com o timestamp do delete dominante (o ramo delete usa
  `payload.get("deleted_at", time.time())`, apply.py:631-634 — a regra DEAD deve reusar o
  `deleted_at` do delete dominante do conjunto, não `time.time()`, senão o próprio `deleted_at`
  vira não-determinístico entre peers). **Esse ponto é um sub-achado:** ao gravar DEAD por
  recompute, usar o `deleted_at` do payload do delete dominante (determinístico), nunca `time.time()`.

---

## Ponto 2 Henry — producer stamps always, 3-state (G0 seven, 2026-10-16)

> **Henry (doobidoo, AUTOR do mcp-memory-service + ADR-0012) fez request-changes no PR #1562**
> (`issuecomment-6214779309`). Ponto 2, nas palavras dele: *"An updated peer that never saw the
> delete has no tombstone, so `_recreate_of_for` returns None and its create goes out bare.
> That's the same as a v1 peer, so it revives. … To close the real case, absence must mean
> 'old peer' and nothing else. An updated producer should stamp the field on* **every** *create:
> the delete's `event_id` when it re-creates over a tombstone, and an explicit empty value
> (`recreate_of: null` or `""`) when it saw none."*
>
> **Decisão adotada (Henry, confirmada pelo Claudio):** modelo de **3 estados** — chave AUSENTE
> = v1 → revive; chave PRESENTE-E-VAZIA = produtor v2 que nunca viu delete → **delete vence**;
> chave nomeia delete dominante = revive; chave nomeia outro delete (stale) = delete vence.
> Sem `schema_version` bump. **Análise read-only; nenhuma linha de produção escrita aqui.**

### 0. O furo exato no código atual (evidência)

O furo que o Henry aponta é literal no produtor de hoje:

- `store.py:245-246` (método `store`) e `store.py:390-391` (método `store_batch`):
  `if _recreate_of is not None: payload['recreate_of'] = _recreate_of`.
- `_recreate_of_for` (`store.py:60-114`) retorna `None` quando **não há tombstone** para o hash
  (`store.py:70`, docstring: *"Returns None when the hash has no tombstone (a fresh create carries
  no recreate_of)"*; guarda em `store.py:85-87` `if tomb is None: return None`).

Logo, um **peer v2 atualizado que nunca viu o delete** (sem tombstone local) emite um create com a
chave `recreate_of` **AUSENTE** — **byte-a-byte idêntico** ao create de um **peer v1**. O apply não
tem como distinguir os dois; sob "ausência = revive", o caso CONCORRENTE REAL (peer novo cria sem
ter visto o delete de OUTRO peer) **ainda ressuscita**. O PR atual só fecha o caso de `recreate_of`
**STALE** (o raro). Isto confirma o ponto 2 do Henry: o design atual **não** fecha o furo.

### 1. (A PERGUNTA SUTIL) 'nunca viu delete' vs 'fresh create de conteúdo totalmente novo' — como o produtor distingue, e o impacto no INV-2c

**Resposta direta: o produtor NÃO precisa distinguir os dois no momento do carimbo — e não deve
tentar. Henry diz literalmente "stamp the field on *every* create". Portanto um fresh create de
conteúdo inédito TAMBÉM carrega o valor vazio.** O produtor carimba exatamente UMA de duas coisas:

| Situação no produtor | `_recreate_of_for(hash)` | O que o produtor carimba |
|---|---|---|
| Re-cria SOBRE tombstone (viu o delete) | retorna `event_id` do delete dominante | `recreate_of = <event_id>` |
| Qualquer outro create (nunca viu delete) — **inclui hash INÉDITO** | retorna `None` | `recreate_of = <VAZIO EXPLÍCITO>` |

O produtor **não tem** informação para separar "nunca viu delete de um hash que existe em outro
peer" de "hash totalmente novo" — nos **dois** casos não há tombstone local, e é por isso que
`_recreate_of_for` devolve `None` igual. Tentar distinguir exigiria conhecimento global (saber que
outro peer deletou esse hash) que o produtor, por construção offline-first, **não tem**. A beleza
da regra do Henry é que essa distinção é **desnecessária no produtor**: ela é resolvida no
apply/recompute, que SIM tem o conjunto de eventos do hash.

**Por que carimbar vazio no fresh create é seguro — impacto no INV-2c (analisado):** o vazio
explícito é **inerte** num hash que nunca foi deletado. `recompute_liveness` (apply.py:663+) só
consulta `recreate_of` **depois** de encontrar deletes dominando o create vencedor
(`_liveness_bare`/`_liveness_with_signal`, apply.py:711-732). Para um hash inédito, `deletes == []`
→ o fold retorna `_LIVE, None` **antes** de olhar a chave (apply.py:687, ramo `if not deletes:
return _LIVE, None`). Ou seja: **o vazio num fresh create nunca chega ao ramo de decisão de
signal**. Portanto:

- **INV-2c (fresh hash materializa normalmente) NÃO muda de resultado** — continua VIVO. O vazio
  não bloqueia nada: a existência de deletes para o hash é a pré-condição para a chave sequer ser
  lida.
- **O que muda no INV-2c é só a FORMA do evento emitido**, não o comportamento: o create de um hash
  inédito passa a carregar `recreate_of: <vazio>` no payload em vez de nenhuma chave. É preciso um
  **teste novo** afirmando isso (ver §4) — e o INV-2c revisado deve declarar explicitamente: *"um
  fresh create v2 carrega o vazio explícito E ainda assim materializa VIVO"*.

**Recomendação (decidido): usar `""` (string vazia), NÃO `null`.** Justificativa de evidência —
o payload faz round-trip por `json.dumps(payload)` na gravação (`base.py:487`) e `json.loads(pj)`
na leitura (apply `_load_hash_events`, apply.py:645-660; e `payload.get(...)`). O problema fatal do
`null`:

- Chave **ausente** → `payload.get("recreate_of")` devolve `None`.
- Chave presente com **JSON `null`** → `json.loads` produz `None` → `payload.get("recreate_of")`
  devolve **`None` também**.

**`null` e ausência colapsam no MESMO valor Python (`None`)** depois do round-trip. Com `.get()`
puro é **impossível** distinguir "v1 (ausente)" de "v2-nunca-viu (null)" — que é exatamente a
distinção de 3 estados que o Henry exige. `""` (string vazia) sobrevive distinto: ausente→`None`,
vazio→`""`, nomeado→`"<event_id>"`. **Três valores Python distintos, zero ambiguidade.** (Alternativa
só se quisermos manter `null`: trocar `.get()` por `"recreate_of" in payload`, o que detecta
presença-de-chave independente do valor — mas é mais frágil e exige disciplina em todo ponto de
leitura. `""` é a escolha de menor superfície de erro.)

### 2. recompute_liveness com 3 estados — pseudocódigo + prova de convergência

A base já está implementada (apply.py:663-757, variante iii' do §P1). O ponto 2 do Henry
**refina o ramo `_liveness_bare`** para separar AUSENTE de VAZIO. Hoje `_liveness_bare`
(apply.py:721-732) é chamado quando `recreate_of is None` — mas `None` hoje mistura "ausente" e
"null". Com o sentinel `""`, a lógica passa a 3 estados explícitos:

```
# SENTINEL de distinção (string vazia sobrevive ao round-trip JSON; null NÃO).
_EMPTY = ""

def classify_recreate_of(payload) -> AUSENTE | VAZIO | NOMEADO(eid):
    if "recreate_of" not in payload:            # chave AUSENTE no dict
        return AUSENTE                           # peer v1
    v = payload["recreate_of"]
    if v == _EMPTY or v is None:                 # "" (v2) — null tolerado como vazio por robustez
        return VAZIO                             # produtor v2 que NUNCA viu o delete
    return NOMEADO(v)                            # nomeia um delete

def recompute_liveness(events):                  # fold puro sobre TODOS os sync_events do hash
    creates = [e for e in events if e.op in ("create","update")]
    deletes = [e for e in events if e.op == "delete"]
    if not creates:
        return (DEAD, Dtop.payload.deleted_at) if deletes else (ABSENT, None)
    cwin = min(creates, key=winner_key)          # encarnação viva candidata (ordem do resolver)
    if not deletes:
        return (LIVE, None)                      # hash nunca deletado → VIVO (INV-2c; vazio inerte)
    dtop = min(deletes, key=winner_key)          # delete dominante do CONJUNTO (P2: _winner_key, não HLC-only)

    kind = classify_recreate_of(cwin.payload)
    if kind is AUSENTE:                           # ── ESTADO 1: v1 → REVIVE (back-compat)
        return (LIVE, None)
    if kind is VAZIO:                             # ── ESTADO 2: v2 nunca viu delete → DELETE VENCE
        return (DEAD, dtop.payload.deleted_at)    #    fecha o caso CONCORRENTE REAL do Henry
    # kind is NOMEADO(eid):                       # ── ESTADO 3: supersessão
    if kind.eid == dtop.event_id:                 #    nomeia o delete dominante → REVIVE (INV-2b)
        return (LIVE, None)
    return (DEAD, dtop.payload.deleted_at)         #    nomeia OUTRO delete (stale) → DELETE VENCE (INV-2a)
```

> **Mudança vs a variante iii' já no arquivo:** antes, "ausência = revive" era uma só folha
> (`_liveness_bare` tratava qualquer `None` como revive). Agora o ramo `None`-colapsado é
> **bifurcado** em AUSENTE (revive) vs VAZIO (morre). Essa é toda a diferença que o ponto 2 do
> Henry introduz. Os ramos NOMEADO ficam idênticos. **`_liveness_bare` (apply.py:721) deixa de ser
> chamado para o caso vazio** — a decisão vazio→DEAD não depende de dominação HLC (`d.key() <=
> cwin_key`); o delete vence SEMPRE que existe delete no conjunto, mesmo com o create de HLC maior.
> É exatamente isto que fecha a Ordem B do concorrente real.

**Prova de convergência (qualquer ordem).** Seja `S` o conjunto de sync_events do hash `H`.
`recompute_liveness` lê `S` inteiro e computa `creates`, `deletes`, `cwin = min(creates, winner_key)`,
`dtop = min(deletes, winner_key)` e `classify_recreate_of(cwin.payload)` — **todas funções puras de
`S`**, nenhuma lê `memories.deleted_at` nem a ordem de inserção. `winner_key` é total e determinístico
(resolver.py, ADR-0010/P2). Logo `cwin`, `dtop` e a classificação são idênticos em todo peer que
possua `S`, independente da ordem em que os eventos chegaram. O resultado `(estado, deleted_at)` é
função de `S` apenas; `deleted_at` em DEAD vem do **payload de `dtop`** (não `time.time()`), então
também é idêntico entre peers. `_reconcile_liveness` (apply.py:733+) roda em **TODO** evento do hash
(create/delete, vença ou perca o resolver — apply.py:782 ramo delete; apply.py:793+ ramo create),
reimpondo o estado a cada chegada; a idempotência de `sync_events` (UNIQUE agent_id+event_id,
base.py `INSERT OR IGNORE`) garante que re-pull não altera `S`. ∴ dado o mesmo `S`, todo peer
converge para o mesmo `deleted_at` — **em qualquer ordem de chegada. QED.**

Chegada parcial é **monotônica e segura**: enquanto o delete concorrente ainda não chegou, o
conjunto local tem só o create com `recreate_of` VAZIO e nenhum delete → `if not deletes: return
LIVE` → vive provisoriamente (não há informação de delete ainda — correto). Quando o delete chega,
`deletes != []` → o ramo VAZIO dispara → DEAD. Nunca ressuscita indevidamente; só transita
LIVE→DEAD ao ganhar informação, nunca o inverso espúrio.

### 3. Ausente vs vazio na prática — sobrevive ao round-trip no sync_events (confirmado)

O payload é um `dict` Python serializado com `json.dumps` na gravação (`base.py:487`
`json.dumps(payload)` dentro do `INSERT ... INTO sync_events`) e desserializado com `json.loads` na
leitura (apply `_load_hash_events`, apply.py:645-660). Round-trip verificado por caso:

| Caso | No produtor (dict) | Em `sync_events.payload` (texto) | Após `json.loads` | `"recreate_of" in payload` | `payload.get("recreate_of")` |
|---|---|---|---|---|---|
| **v1 (ausente)** | chave não existe | `{...}` (sem a chave) | sem a chave | **False** | `None` |
| **v2 vazio `""`** | `payload['recreate_of']=""` | `{...,"recreate_of":""}` | `""` | True | **`""`** ✅ distinto |
| **v2 vazio `null`** ⚠️ | `payload['recreate_of']=None` | `{...,"recreate_of":null}` | `None` | True | `None` ⚠️ **colide com ausente via `.get()`** |
| **v2 nomeado** | `payload['recreate_of']="<eid>"` | `{...,"recreate_of":"<eid>"}` | `"<eid>"` | True | `"<eid>"` |

**Conclusão:** com `""` como sentinel, `payload.get("recreate_of")` já basta e os 3 estados são
inequívocos pós round-trip (`None` / `""` / `"<eid>"`). Com `null`, seria **obrigatório** usar
`"recreate_of" in payload` (teste de presença-de-chave) em TODO leitor, porque `.get()` não
distingue — risco de regressão silenciosa se algum leitor usar `.get()`. **Por isso a recomendação
é `""`.** (Robustez defensiva: o `classify_recreate_of` acima trata `null` recebido de um peer como
VAZIO também, para o caso de outra implementação do produtor escolher `null` — mas o NOSSO produtor
emite `""`.) A chave é **aditiva**: um apply v1, que nunca lê `recreate_of`, ignora `""` tão
facilmente quanto ignora a ausência — zero impacto nele. **Sem `schema_version` bump.**

### 4. Testes novos / alterados

Inclui o **caso CONCORRENTE REAL** (peer novo cria bare-mas-com-vazio vs delete de outro peer, as
duas ordens → delete vence) e o **INV-2c revisado**:

| Teste | Novo/Alterado | Conjunto / forma | Resultado esperado |
|---|---|---|---|
| `test_concurrent_v2_empty_recreate_of_vs_peer_delete__both_orders` | **NOVO (o caso do Henry)** | `C2(recreate_of="")` de um peer + `D_other` delete de OUTRO peer, para o mesmo hash. Aplicar nas **duas ordens** (create-primeiro e delete-primeiro) | **DEAD nas duas** — o VAZIO faz o delete vencer independente de ordem/HLC. É o furo que o PR atual não fecha. |
| `test_producer_bare_create_emits_empty_recreate_of` | **NOVO (produtor, via `store()`)** | `store()` de um hash INÉDITO (sem tombstone), ler o create emitido de `sync_events` | payload contém `recreate_of == ""` (não ausente). Prova "stamp on every create". |
| `test_producer_bare_create_emits_empty_recreate_of__batch` | **NOVO (produtor, via `store_batch()`)** | idem via `store_batch()` (store.py:390-391) | idem `recreate_of == ""`. Cobre os DOIS métodos (ask do Henry no ponto 4). |
| `test_producer_recreate_over_tombstone_emits_delete_event_id` | **NOVO (produtor)** | `store()` → `delete` → `store()` do mesmo hash; ler o 2º create | `recreate_of == <event_id do delete>` (não vazio). |
| `test_inv2c_fresh_create_with_empty_recreate_of_materializes_live` | **ALTERADO (INV-2c revisado)** | fresh hash, create com `recreate_of=""`, SEM delete no conjunto | **LIVE** — o vazio é inerte sem deletes ((apply.py:687 ramo `if not deletes: return LIVE`)). Prova que o vazio não bloqueia fresh create. |
| `test_v2_empty_recreate_of_with_delete_stays_dead__all_orders` | **NOVO (3-estados)** | `{C0, D1, C2(recreate_of="")}` nas 6 permutações | **DEAD nas 6** (estado 2: vazio + delete → morre). |
| `test_v1_absent_recreate_of_with_delete_revives__all_orders` | **NOVO (back-compat)** | `{C0, D1, C2(sem a chave)}` nas 6 permutações | **LIVE nas 6** (estado 1: ausência = revive). **Distingue** explicitamente de vazio. |
| `test_recreate_signal_discriminates_stale_vs_valid` (test_1522_audit_fixes.py:436) | **ALTERADO** | acrescentar 4º caso: AUSENTE→revive vs VAZIO→morre (hoje só cobre bare/valid/stale, com "bare"=None ambíguo) | distinguir os 4 estados |
| `test_create_does_not_resurrect_tombstone` (test_1522_audit_fixes.py:104, xfail) | **REESCREVER + drop xfail** | reusar o caso VAZIO (v2 nunca viu delete) como "stays dead" em vez de bare | DEAD; drop `@pytest.mark.xfail` |
| `test_v1_apply_ignores_unknown_recreate_of_key` (test_1522_audit_fixes.py) | **RENOMEAR ou rodar contra `origin/main`** | (ponto 4 Henry: hoje roda contra o apply NOVO, não prova nada sobre v1) | rodar o apply de `origin/main` OU renomear para refletir que testa o apply atual tolerando a chave |

### 5. Compatibilidade — v1 revive, v2 sempre carimba; zero-regressão nos recreate legítimos

- **v1 (sem a chave) → REVIVE.** Mantido: `classify_recreate_of` retorna AUSENTE → `(LIVE, None)`.
  Um apply v1 (que nunca lê `recreate_of`) ignora a chave `""` dos peers v2 sem erro — a chave é
  payload aditivo, **sem `schema_version` bump** (confirmado: `schema_version = 1` fixo em
  base.py:434; o gate de fail-stop `event_schema_version > KNOWN_SCHEMA_VERSION` em
  apply.py:~132 **não** dispara).
- **v2 sempre carimba:** produtor emite `recreate_of = <event_id>` (re-create sobre tombstone) ou
  `recreate_of = ""` (qualquer outro create). Nunca mais emite create "bare" sem a chave.
- **Zero-regressão nos 2 testes de recreate legítimo** (ambos usam create `c2` **bare** hoje):
  - `test_old_edit_does_not_restore_recreated_memory_field` (regressions.py:260): o `c2` bare
    (ausente) sobre tombstone deve REVIVER → `deleted_at IS None`. Com a bifurcação, **ausente**
    continua caindo em estado 1 → LIVE. **Verde, sem mudança.** ⚠️ Ressalva: se futuramente esses
    fixtures forem regenerados por um produtor v2 real, o `c2` passará a carregar `""` (vazio) e o
    resultado VIRARIA DEAD. Como são **fixtures hand-built** que simulam um peer v1 (create sem
    chave), eles permanecem válidos como teste de back-compat v1 — mas o teste deve ganhar um
    **comentário** deixando claro que `c2` representa um peer v1 (bare) e NÃO um produtor v2.
  - `test_old_metadata_key_does_not_return_after_recreation` (regressions.py:331): mesma forma
    (`c2` bare, metadata vazia) → REVIVE e descarta metadata da encarnação morta. **Verde pela mesma
    razão** (ausência = revive), mesma ressalva de comentário.
  - `test_legit_create_without_tombstone_materializes` (test_1522_audit_fixes.py:172): INV-2c, fresh
    hash. **Verde** — sem deletes, o fold retorna LIVE antes de olhar a chave.

**Sub-achado (herdado do §P1, segue válido):** ao gravar DEAD por recompute, usar o `deleted_at` do
**payload do delete dominante** (`dtop.payload.deleted_at`), nunca `time.time()` — senão o próprio
`deleted_at` fica não-determinístico entre peers (apply.py:760 já tem o fallback com `warning`,
mantê-lo como observável, não como caminho normal).

### Resumo do que muda vs o PR atual

1. **Produtor (`store.py:245-246` e `:390-391`):** trocar `if _recreate_of is not None:
   payload['recreate_of'] = _recreate_of` por **sempre** carimbar — `payload['recreate_of'] =
   _recreate_of if _recreate_of is not None else ""`. (Design; não implementado aqui.)
2. **Apply (`recompute_liveness`, apply.py:663+):** bifurcar o ramo hoje colapsado em `None`
   (AUSENTE vs VAZIO via `classify_recreate_of`), VAZIO→DEAD.
3. **Sentinel:** `""` (string vazia), por sobreviver ao round-trip JSON de forma distinta de
   ausente (null NÃO sobrevive via `.get()`).
4. **Changelog (ponto 5 Henry):** só agora a 1ª frase ("fecha o create concorrente de um peer que
   nunca viu o delete") fica **verdadeira** — antes do ponto 2 ela prometia proteção que o PR não
   dava; narrar isto.

## Regra FINAL de liveness — P1-A ordem-HLC + P1-B bootstrap (G0 seven, 2026-10-16)

Dois P1 NOVOS do Greptile no commit `a1e77f25` (rodada 2), ambos em `recompute_liveness`,
ambos confirmados no código do worktree. Esta subseção CONSOLIDA os 3 estados (§2) + a
comparação de ordem create-vs-delete (P1-A) + o casamento robusto a bootstrap (P1-B) numa
única regra determinística FINAL, e resolve o teste flipado. **Não implementado — G0.**

### 0. Os dois furos, confirmados no código

**P1-A — `apply.py:710` (`_liveness_with_signal`), thread Greptile 2026-10-16T12:18:26Z
"Losing creates undo winning deletes".** O ramo NOMEADO revive por *name-match puro*:

```python
# apply.py:710-713 (atual, com o furo)
if recreate_of == dtop.event_id:
    return _LIVE, None                        # ← revive SEM comparar ordem create-vs-delete
return _DEAD, dtop.payload.get("deleted_at")
```

`cwin = min(creates, key=winner_key)` (apply.py:684) é eleito **independentemente dos
deletes**; depois classificado. Logo um `create@HLC30` cujo `recreate_of` nomeia
`delete@HLC40` cai em NOMEADO, `recreate_of == dtop.event_id` é verdadeiro, e revive —
**apesar de o delete dominar o resolver** (`delete@HLC40.key() < create@HLC30.key()`). A
memória fica visível embora o delete vença. **Confirmado: o ramo NOMEADO não tem gate de
ordem.** (O ramo VAZIO e o `_liveness_bare` por acaso já se protegem — VAZIO sempre morre
se há delete; `_liveness_bare` checa `d.key() <= cwin_key` — mas o NOMEADO não.)

**P1-B — `apply.py:710` + `bootstrap.py:29-37,160` "Joined peers miss re-created memories".**
O produtor carimba `recreate_of = <event_id REAL do delete local>` (`store.py:60-98`
`_recreate_of_for`; `store.py:251,398`). Mas o bootstrap REESCREVE o event_id do delete:

```python
# bootstrap.py:29-37
def _deterministic_event_id(content_hash, updated_at, op):
    return "bl-" + hashlib.sha256(raw).hexdigest()[:32]       # ← id SINTÉTICO

# bootstrap.py:137-139,160  — o delete do baseline recebe event_id = "bl-..."
if deleted_at is not None:
    op = "delete"; payload = {"content_hash": ..., "deleted_at": deleted_at}
...
"event_id": _deterministic_event_id(content_hash, updated_at, op),   # bl-...
```

`install_baseline` (bootstrap.py:171) aplica esses eventos e **parka o cursor além do
histórico** (watermark), então o peer nunca verá o delete ORIGINAL — só o `bl-...`. Quando a
origem faz `store()` e emite `recreate_of = <event_id ORIGINAL do delete>`, no peer
`dtop.event_id == "bl-..."` ≠ `<original>` → NOMEADO cai no `else` → **DEAD indevido**. Um
re-create legítimo fica deletado no peer que bootstrapou. **Confirmado: o casamento por
event_id cru é frágil a qualquer reescrita de id (bootstrap é a primeira; compaction futura
seria outra).**

### 1. A chave do P1-A: "superar o delete" sob `_winner_key`

`_winner_key` (resolver.py:42-70) é ordem total, **menor vence**, e no componente 2 usa
`0 if op=='delete' else 1` — **em HLC igual, o delete vence o create**. Portanto:

> **`create` SUPERA `delete`  ⟺  `cwin.key() < dtop.key()`  ⟺  o create tem HLC
> ESTRITAMENTE maior que o delete** (em HLC igual o delete ganha pelo componente 2). Isto é
> exatamente a definição do Greptile: *"a legitimate re-create is POSTERIOR to the delete it
> supersedes."* Um re-create nasce DEPOIS do delete; se não nasceu depois, não é re-create —
> é um create perdedor (ou concorrente) tentando desfazer um delete vencedor.

### 2. A chave do P1-B: casar por IDENTIDADE ESTÁVEL, não por event_id cru

O `recreate_of` precisa casar o delete nomeado mesmo quando o event_id foi reescrito. Duas
opções (Greptile sugere ambas):

- **Opção A — preservar a identidade original no baseline.** `bootstrap._deterministic_event_id`
  deixa de inventar `bl-...` para deletes e **reusa o event_id ORIGINAL do delete** (lido de
  `sync_events` na origem do baseline). Então no peer `dtop.event_id == <original>` e o
  name-match volta a funcionar. **Prós:** o casamento continua por event_id (simples, uma só
  regra em apply). **Contras:** o baseline hoje é gerado de `memories` (snapshot de linhas),
  NÃO de `sync_events` — `_row_to_baseline_event` (bootstrap.py:112) só tem a linha, não o
  event_id do delete que a produziu. Exigiria o baseline carregar/juntar o event_id real do
  delete por hash (um JOIN a mais no `_read_snapshot`), e ainda assim só resolve o bootstrap —
  qualquer OUTRA reescrita futura de id (compaction/GC de sync_events) reabre o furo.

- **Opção B (RECOMENDADA) — casar o `recreate_of` por âncora de identidade HLC do delete,
  com fallback ao event_id.** O produtor passa a carimbar, no create, uma âncora estável do
  delete que ele supersede: `recreate_of_hlc = (delete.hlc_physical, delete.hlc_logical,
  delete.agent_id)` além do `recreate_of` (event_id, mantido por retrocompat). O apply casa o
  delete nomeado assim: primeiro por event_id (caminho atual, cobre peers sem reescrita);
  se não casar, por `(hlc_physical, hlc_logical, agent_id)` contra os deletes do conjunto
  (cobre o delete reescrito como `bl-...`, cujo **HLC o baseline PRESERVA** —
  `bootstrap.py:132-139` usa `hlc_by_hash` real do delete, não zera). **Prós:** robusto a
  QUALQUER reescrita de event_id (bootstrap, compaction), não só a atual; `bootstrap.py` NÃO
  precisa mudar a geração de id. **Contras:** mais um campo no payload (aditivo, inerte em
  v1); o produtor precisa ler o HLC do delete (já lê no `_recreate_of_for`, store.py:88-92 —
  o SELECT já traz `hlc_physical, hlc_logical, agent_id`; é só carimbar).

> **Decisão seven: Opção B.** É a única que fecha o furo de forma durável (a Opção A só tapa o
> bootstrap e deixa o problema estrutural — event_id cru como chave de identidade — de pé). O
> baseline JÁ preserva o HLC do delete (`hlc_by_hash`, bootstrap.py:132), então a âncora HLC
> existe no peer; falta só o produtor carimbá-la e o apply casá-la. `bootstrap.py` fica
> **intocado** (não precisa mudar a reescrita `bl-...`). Isto também mantém a convergência: o
> HLC é parte do `_winner_key`, é imutável e idêntico em todo peer.

### 3. Regra determinística FINAL (pseudocódigo) — 3 estados + P1-A + P1-B

```python
# ── PRODUTOR (store.py:245-251 e :394-398) ────────────────────────────────────
# Carimba SEMPRE recreate_of (3-estados, §2), e AGORA também a âncora HLC do delete (P1-B).
def stamp_recreate_of(payload, content_hash):
    d = latest_dominant_delete(content_hash)          # _recreate_of_for, por _winner_key (P2)
    if d is None:                                      # hash sem tombstone
        payload['recreate_of'] = ""                    # VAZIO (v2 que não re-cria sobre delete)
    else:                                              # re-create sobre tombstone
        payload['recreate_of'] = d.event_id            # NOMEADO por event_id (retrocompat)
        payload['recreate_of_hlc'] = [d.hlc_physical,  # P1-B: âncora ESTÁVEL do delete
                                      d.hlc_logical, d.agent_id]

# ── CLASSIFICAÇÃO (apply.classify_recreate_of) ───────────────────────────────
def classify_recreate_of(payload):
    if "recreate_of" not in payload:   return AUSENTE, None, None      # v1
    v = payload["recreate_of"]
    if v == "" or v is None:           return VAZIO, "", None          # v2 nunca viu delete
    return NOMEADO, v, payload.get("recreate_of_hlc")                  # v2 re-create (eid + âncora)

# ── CASAMENTO ROBUSTO A BOOTSTRAP (P1-B) ─────────────────────────────────────
def names_delete(recreate_of_eid, recreate_of_hlc, dtop, deletes):
    # 1) match por event_id (peers sem reescrita de id)
    if recreate_of_eid == dtop.event_id:                 return True
    # 2) fallback por âncora HLC+agent (cobre o delete reescrito como bl-... pelo bootstrap)
    if recreate_of_hlc is not None:
        ph, lg, ag = recreate_of_hlc
        for d in deletes:
            if (d.hlc_physical, d.hlc_logical, d.agent_id) == (ph, lg, ag):
                return d is dtop        # só vale se o delete nomeado É o dominante do conjunto
    return False

# ── FOLD DE LIVENESS (apply.recompute_liveness) — função pura do CONJUNTO S ───
def recompute_liveness(events):
    creates = [e for e in events if e.op in ("create","update")]
    deletes = [e for e in events if e.op == "delete"]
    if not creates:
        return (DEAD, Dtop.payload.deleted_at) if deletes else (ABSENT, None)
    cwin = min(creates, key=winner_key)               # encarnação viva candidata (resolver)
    if not deletes:
        return (LIVE, None)                            # nunca deletado → VIVO (INV-2c; vazio inerte)
    dtop = min(deletes, key=winner_key)                # delete dominante (P2: _winner_key, não HLC-only)

    kind, eid, hlc = classify_recreate_of(cwin.payload)

    if kind is AUSENTE:                                # ESTADO 1 — v1 back-compat
        # (inalterado) revive SE nenhum delete o domina sob _winner_key
        return (LIVE, None) if cwin.key() < dtop.key() else (DEAD, dtop.payload.deleted_at)

    if kind is VAZIO:                                  # ESTADO 2 — v2 nunca viu delete
        return (DEAD, dtop.payload.deleted_at)         # delete vence SEMPRE (fecha o concorrente real)

    # kind is NOMEADO ─────────────────────────────── ESTADO 3 — supersessão
    if not names_delete(eid, hlc, dtop, deletes):      # nomeia OUTRO delete (stale) → morto (INV-2a)
        return (DEAD, dtop.payload.deleted_at)
    # P1-A: nomeia o delete DOMINANTE, mas só revive se o create TAMBÉM o supera na ordem total.
    if cwin.key() < dtop.key():                        # create POSTERIOR ao delete (HLC estritamente maior)
        return (LIVE, None)                            # re-create legítimo (INV-2b, agora com gate de ordem)
    return (DEAD, dtop.payload.deleted_at)             # P1-A: create perdedor nomeando delete vencedor → morto
```

**Diferença vs o código atual (apply.py:700-765):**

| Ramo | Hoje (worktree) | FINAL | Motivo |
|---|---|---|---|
| NOMEADO (`_liveness_with_signal`, apply.py:704-713) | `if recreate_of == dtop.event_id: return LIVE` | `if names_delete(...) and cwin.key() < dtop.key(): return LIVE` else `DEAD` | **P1-A** (gate de ordem) + **P1-B** (casamento por âncora) |
| AUSENTE (`_liveness_bare`, apply.py:716-733) | revive se nenhum delete domina (`d.key() <= cwin_key`) | **igual em efeito** (reescrito como `cwin.key() < dtop.key()` — equivalente: se `cwin` supera o `dtop`, supera todos) | sem mudança de comportamento; só uniformiza a forma |
| VAZIO (apply.py:699) | `return DEAD` | **igual** | sem mudança |
| `_liveness_with_signal` recebe só `dtop` | assinatura `(recreate_of, dtop)` | passar **`deletes`** também (para o fallback HLC do `names_delete`) | P1-B precisa varrer o conjunto |

**O que muda em cada arquivo:**

1. **`store.py:245-251` e `store.py:394-398`** (produtor, os DOIS métodos): além de
   `payload['recreate_of']`, carimbar `payload['recreate_of_hlc'] = [hlc_physical, hlc_logical,
   agent_id]` do delete dominante quando NOMEADO. O `_recreate_of_for` (store.py:60-98) já
   lê esses campos (SELECT em store.py:88-90) — hoje só retorna o event_id; passar a devolver
   a tripla também. **Aditivo, sem `schema_version` bump.**
2. **`apply.py` `classify_recreate_of` (apply.py:~655)**: retornar a tripla `(kind, eid, hlc)`
   em vez de `(kind, value)` — ler `payload.get("recreate_of_hlc")`.
3. **`apply.py` `_liveness_with_signal` (apply.py:704-713)**: receber `deletes` além de `dtop`;
   trocar o name-match puro pela chamada `names_delete(...)` **E** o gate `cwin.key() <
   dtop.key()` (P1-A). Precisa de `cwin` na assinatura (hoje só recebe `recreate_of, dtop`).
4. **`apply.py` `_liveness_with_deletes` (apply.py:695-700)**: repassar `cwin` e `deletes` ao
   ramo NOMEADO.
5. **`bootstrap.py`: NÃO muda.** A âncora HLC do delete já é preservada pelo baseline
   (`hlc_by_hash`, bootstrap.py:132-139). A reescrita `bl-...` do event_id fica como está — o
   fallback por HLC a contorna. (Se um dia preferirmos a Opção A, aí `bootstrap.py` mudaria;
   não é o caminho escolhido.)

### 4. Prova de convergência (qualquer ordem, inclui P1-A e P1-B)

Seja `S` o conjunto de `sync_events` do hash `H`. `recompute_liveness` computa
`creates, deletes, cwin=min(creates,winner_key), dtop=min(deletes,winner_key)`, a
classificação de `cwin.payload`, e os predicados `names_delete(...)` e `cwin.key() <
dtop.key()` — **todas funções puras de `S`**. `winner_key` é ordem total determinística
(resolver.py:42-70). A âncora `recreate_of_hlc` é parte imutável de `cwin.payload` (viaja no
evento, não é recomputada). `names_delete` compara contra `(hlc,agent_id)` dos deletes de `S`
— idênticos em todo peer que possua `S`, mesmo que o event_id do delete tenha sido reescrito
como `bl-...` num peer (o HLC+agent do delete é preservado pelo baseline, bootstrap.py:132).
Logo, dado o mesmo `S`, `(estado, deleted_at)` é idêntico em todo peer, **independente da
ordem de chegada e independente de o peer ter bootstrapado**. `deleted_at` em DEAD vem sempre
do `payload` de `dtop` (determinístico, não `time.time()`). `_reconcile_liveness` roda em TODO
evento do hash (apply.py:736+), reimpondo o estado; idempotência de `sync_events` (UNIQUE
agent_id+event_id) garante que re-pull não altera `S`. ∴ convergência total. **QED.**

Monotonicidade de chegada parcial (sem regressão espúria):

- **P1-A:** enquanto só o `create@HLC30(recreate_of=Dx)` chegou e o `delete@HLC40` ainda não,
  `deletes==[]` → `LIVE` provisório (não há delete conhecido — correto). Quando `delete@HLC40`
  chega, `dtop=delete@HLC40`, o create NÃO o supera (`cwin.key() > dtop.key()`) → `DEAD`.
  Transita LIVE→DEAD ao ganhar informação; nunca o inverso espúrio.
- **P1-B:** no peer bootstrapado, o delete existe como `bl-...` (HLC preservado). Quando o
  `create(recreate_of=<eid original>, recreate_of_hlc=[HLC do delete])` chega da origem,
  `names_delete` casa pelo fallback HLC → se o create supera o delete (`cwin.key() <
  dtop.key()`), revive; senão morre. Mesmo resultado que num peer que nunca bootstrapou e viu
  o delete original. **Bootstrap deixa de divergir.**

### 5. O teste flipado — VEREDITO: o flip está CORRETO (manter), com uma ressalva de nomenclatura

`test_imported_tombstone_keeps_remote_hlc_against_newer_peer_create`
(tests/test_sync_event_log.py:1370). Cenário: `peer.store()` cria `C@HLC base+25`; `primary`
importa `delete@HLC base+20`. Create HLC (25) **>** delete HLC (20). O flip mudou a asserção
final de `winner[0] is None` (vivo) para `winner[0] is not None` (morto).

**Decisão: o flip é CORRETO.** Justificativa, cruzando com P1-A:

1. O create foi emitido por `peer.store()` — um produtor v2 REAL que **nunca viu** o delete do
   `primary` (o delete foi importado DEPOIS, por reconciliação). Logo o create carrega
   `recreate_of=""` (VAZIO), não um nome. Cai no **ESTADO 2 (VAZIO)** → `DEAD`, **independente
   do HLC**. Este é precisamente o furo concorrente que o modelo 3-estados fecha: um peer que
   re-cria sem ter visto o delete de outro peer não ressuscita a memória. ✅

2. **Não colide com P1-A, confirma-o.** P1-A trata do ramo NOMEADO (create que NOMEIA o delete
   dominante mas tem HLC menor → morto). Aqui o create é VAZIO, não NOMEADO — mas a direção é a
   mesma: *HLC maior sozinho NÃO basta para reviver sobre um delete*. Só um **NOMEADO que
   supera o delete em ordem** revive. A pergunta do manager ("se o create tem HLC>delete E
   nomeia/vazio, qual ganha?"): **VAZIO → delete ganha sempre; NOMEADO → o create ganha só se
   `cwin.key()<dtop.key()` E nomear o `dtop`.** O teste é o caso VAZIO, delete ganha. Coerente.

3. O docstring do teste diz *"a later peer create must not be outranked by an imported
   historical delete"* — essa premissa descreve o mundo **pré-3-estados** (LWW puro por HLC).
   Sob o 3-estados do Henry, "um peer que criou sem ver o delete" perde para o delete por
   DESIGN, não por HLC. **A semântica mudou por decisão de produto endossada pelo Henry
   (Ponto 2), não por bug.** O flip reflete essa decisão.

**Ressalva (dívida de clareza, NÃO reverter):** o NOME e o docstring do teste agora
**contradizem** o que ele asserta (o nome promete "keeps remote hlc ... against newer peer
create" com sentido de "o create mais novo prevalece", mas o corpo asserta o oposto). Isto
engana a próxima sessão. **Ação:** renomear para algo como
`test_imported_tombstone_wins_over_v2_empty_recreate_even_with_newer_hlc` e reescrever o
docstring para explicitar a semântica 3-estados. As asserções de preservação de HLC (linhas
~1355-1368) ficam — continuam válidas e são ortogonais à liveness. **O flip da asserção final
fica; o nome/docstring é que precisam alinhar.**

### 6. Testes novos (além dos já listados em §4 da seção Ponto 2)

| Teste | Conjunto / forma | Resultado esperado | Fura hoje? |
|---|---|---|---|
| `test_named_recreate_older_than_delete_stays_dead__both_orders` | `{D@HLC40, C@HLC30(recreate_of=D.event_id)}`, as 2 ordens | **DEAD** nas 2 — NOMEIA o dtop mas NÃO o supera (`cwin.key() > dtop.key()`). | **SIM (P1-A)** — hoje revive |
| `test_named_recreate_newer_than_delete_revives__both_orders` | `{D@HLC30, C@HLC40(recreate_of=D.event_id)}`, as 2 ordens | **LIVE** nas 2 — nomeia o dtop E o supera. | Não (já verde) — guard anti-regressão do gate |
| `test_named_recreate_ties_delete_hlc_stays_dead` | `{D@HLC30(agentA), C@HLC30(recreate_of=D.event_id)}` HLC IGUAL | **DEAD** — em HLC igual o delete vence (`_winner_key` comp.2); o create NÃO supera. | **SIM (P1-A, borda)** |
| `test_bootstrapped_peer_revives_on_recreate_via_hlc_anchor` | origem: `C0 → D1 → C2(recreate_of=D1.eid, recreate_of_hlc=[D1.hlc])`, C2 supera D1. Peer BOOTSTRAPA (D1 vira `bl-...`), depois recebe C2. | **LIVE** no peer — `names_delete` casa D1 pelo fallback HLC apesar do `bl-...`. | **SIM (P1-B)** — hoje DEAD indevido |
| `test_bootstrapped_peer_keeps_dead_on_stale_recreate` | idem mas `recreate_of_hlc` aponta um delete que NÃO é o dtop | **DEAD** — âncora casa um delete não-dominante → `names_delete` retorna False. | Guard: o fallback HLC não vira brecha de revive de stale |
| `test_bootstrapped_peer_keeps_dead_on_named_older_recreate` | peer bootstrapado; C2 nomeia D1 (via HLC) mas C2 NÃO supera D1 | **DEAD** — P1-A vale mesmo no caminho bootstrap. | Guard: P1-A + P1-B juntos |
| `test_producer_stamps_recreate_of_hlc_on_recreate` | `store()→delete→store()`, ler o 2º create | payload contém `recreate_of_hlc == [hlc do delete]` além de `recreate_of`. | Prova o lado produtor da P1-B |
| `test_producer_omits_recreate_of_hlc_on_empty` | `store()` de hash inédito | payload tem `recreate_of==""` e **sem** `recreate_of_hlc`. | Guard: âncora só no NOMEADO |
| `test_imported_tombstone_wins_over_v2_empty_recreate_even_with_newer_hlc` | (RENOMEADO do flipado) | **DEAD** | O flip; manter |

Convergência deve ser provada nas **6 permutações** para os conjuntos de 3 eventos
(`{C0,D1,C2}`), como já faz `test_converge_*` em §5 da seção §P1.

### 7. O que seven NÃO pôde verificar

- Não executei a suíte — os "fura hoje?" são derivados por leitura do fold, não por run.
- `recreate_of_hlc` como `list` no JSON vira `list` no `json.loads` (tupla vira list); o
  `names_delete` compara contra `tuple` — o pseudocódigo normaliza (`(ph,lg,ag)`), mas a
  implementação deve garantir a coerção (list→tuple) ao ler o payload. Ponto de atenção p/ o reg.
- Opção A (preservar event_id no baseline) foi **descartada** por análise, não por teste; se o
  reg preferir A, o JOIN em `_read_snapshot` para trazer o event_id real do delete é o custo —
  mas continua sem cobrir reescritas futuras de id.
