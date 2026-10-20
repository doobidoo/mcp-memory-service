# ESTADO — mcp-memory-service (Claudio)

## ⚡ ATUALIZAÇÃO 19/out/2026 (leia antes da árvore)

- **ACERVO — índice uniforme em produção.** Os vetores estavam partidos em três espaços
  (35,8% `all-MiniLM`, 64,2% `paraphrase`, ~2.100 órfãos) e isso inviabilizava qualquer
  medição de recuperação. Re-embeddado uniforme com `paraphrase-multilingual`: régua por
  documento **@10 0,567 · @30 0,675 · @100 0,750** (antes 0,008 / 0,017 / 0,017). Banco
  1.195 → 910,5 MB. **Toda medição de recall passa a ser lida contra esta régua.**
  Detalhe operacional em `diario-de-gates.md` §E11.
- **Worktrees:** `docs/_fork/worktrees.md` é o registro canônico (inclui `lab-recall`,
  `pr-agent-id`, `pr-fix-a`, `pr-fix-b`).
- **Bugs e apoio pontual NÃO entram neste mapa** (regra de curadoria mantida): autoria
  `X-Agent-ID`, importer do `state.db`, visibilidade cross-store e migrador `vec0` vivem em
  `diario-de-gates.md` §E11–§E13 e na memória. O migrador já subiu como PR #1589.

> Para você abrir e entender onde estamos sem reconstruir contexto.
> Árvore primeiro (visão), notas depois (detalhe). Atualizado: 2026-10-19 (bloco acima); a árvore dos arcos foi medida em 2026-10-14 — exceto o ramo UIAA/hermes, medido em 19/out — (fork sincronizado v12.0.0; arco novo INBOX INTER-AGENTE = RFC + discussion #1529; delta-sync reabriu 1 ponto de design = ADR-0012 v2 tombstone via auditoria #1522 do Henry. NOTA DE CURADORIA: bugs/apoio pontual ao Henry — #1522 #3/#4, #1521, #1518, PRs #1530/#1531 — NÃO entram aqui; vivem no task-orch + memória. Este mapa é só dos NOSSOS ARCOS). Antes (13/out): venv-ONNX blindado; INC2+INC6 done; INC4 reposicionado (arco 40e6ecfb). F4 upstream TRAVADA (CLIENT não 100% feliz — loop-lento+ranking+fragmentos-lixo).

```
mcp-memory-service (fork = linha viva · 0 atrás do upstream · 312 à frente · 09/out)
│  Legenda: ✅ MERGED/feito · 🟢 feito fork-only · 🟡 parcial/design · 🔴 a fazer
│
├─ ⭐ ARCO DELTA-SYNC (#1345) ..... sync nativo máquina↔hub que APOSENTA o Insync · ✅ 5/5 FASES MERGED — ARCO FECHADO
│   │   RFC: rfc-delta-sync (v0.4 colab ducanhnguyen223). Multi-writer mesh (vs #1304 hub-and-spoke).
│   ├─ Fase 1 event-log local (mesma tx) ............... ✅ MERGED #1478 · ADR-0007/8/9
│   ├─ Fase 2 HLC + resolver determinístico ............ ✅ MERGED #1485 · ADR-0010..0013
│   ├─ Fase 3 embedding consistency guardrail .......... ✅ MERGED #1487 · ADR-0014..0017
│   ├─ Fase 4 pull+push transport+orchestration+sched .. ✅ MERGED #1489 · ADR-0018..0028 · PULL+PUSH provados a quente
│   │     └─ loop review 3 rodadas (Greptile 9 P1 materialização + CodeQL + follow-up Henry metadata convergence)
│   └─ Fase 5 bootstrap + event version negotiation .... ✅ MERGED #1494 (e9c9ffc6) — FECHA O ARCO · spec: "Closes the delta-sync arc"
│         bootstrap de spoke novo (§9.3) + version negotiation (§9.3 rule 6) = pré-req p/ aposentar Insync. DEFERIDO (RFC §5, trabalho futuro): per-spoke CRYPTO identity — a topologia hub+known-spokes da 4c é suficiente por ora.
│         operacional pós-arco: ligar schedule de regime nas 3 máquinas + aposentar crons Insync (não é mais código, é operação).
│   └─ operacional pós-Fase 5: schedule de regime (15m), ligar nas 3 máquinas, aposentar crons Insync
│   └─ 🔧 FOLLOW-UP 14/out (auditoria #1522 do Henry): #3 bootstrap-no-1º-sync + #4 embedding_pending/store corrigidos (PR #1530, apoio — fora deste mapa). MAS 1 achado reabre DESIGN NOSSO: #2 tombstone não honrado no apply (ADR-0012 §8.1c nunca implementado no apply) → ARCO/item ADR-0012 v2 (8c9b1345, 🔴): HLC sozinho não distingue ressurreição-maliciosa de re-create legítimo → precisa SINAL DE INTENCIONALIDADE no evento. Precede G1 arch (seven). Teste xfail no vivo.
│
├─ ⭐ #1304 HYBRID SELF-HOSTED SECONDARY (http) ........ ✅ FECHADO (4 fases MERGED) · raiz que destravou o delta-sync
│   │   SPEC: ✅ rfc-hybrid-http-secondary (EARS R1-R16, Fases 1-4). Topologia validada em PROD: sirdata em hybrid-online com hub VPS (07/out).
│   ├─ Fase 1 (R1-R5) list_content_hashes + bulk endpoint ........................ ✅ MERGED PR #1470
│   ├─ Fase 2 (R6-R9+R9b) RemoteHTTPStorage + auth + pull real ................... ✅ MERGED PR #1471
│   ├─ Fase 3 (R10-R13) desacoplar hybrid.py do CF (capability-gating) ........... ✅ MERGED PR #1474
│   ├─ Fase 4 (R14-R16) model-match startup check (fail-closed modelo hub≠local) . ✅ MERGED PR #1476
│   ├─ fix: hybrid+http não exige creds Cloudflare (storage.py:164 guard) ........ ✅ MERGED #1480 (08/out, Henry)
│   └─ 🧪 hybrid-online TESTADO no sirdata (07/out) e REVERTIDO p/ sqlite_vec. Provou conexão REST+push ao hub VPS; achou o bug CF-guard (#1480). Decisão 4/out: sync que aposenta Insync = DELTA-SYNC (#1345), não hybrid (hub=hub-and-spoke, tentativa anterior). remote_http do #1304 será o transporte da Fase 4 do delta-sync. Crons religados.
│
├─ ✅ ARCO RATING / QUALITY ............................. FECHADO
│   ├─ quality-model (computed vs user_rating) ......... ✅ MERGED PR #1349
│   ├─ retention_periods (ontologia, upstream) ......... ✅ MERGED PR #1368 — desmascarou o decay
│   ├─ MCP_DECAY_ENABLED (timkjr) ...................... ✅ MERGED PR #1391 — furo lateral fechado
│   └─ supersession orphan (fix coluna + list_orphans) . ✅ MERGED PR #1404 (nosso #1352)
│
├─ 🟢 ARCO LEARNING LOOP ............... o NORTE (aprender, não só colher) · WorkItem 62046f1e
│   │   Meta (Claudio): o CONJUNTO destilação+injeção+feedback. "Ao final, preciso de tudo."
│   │   Correção de rumo (3/out): L4 feedback = RFC-MM-01 do CLAUDIO (jul), Henry endossou #1286.
│   │
│   ├─ D0 limpar beliefs ................................... ✅ já resolvido (noise filter existe; task dc1c7756 cancelada)
│   ├─ D1 destilação (L2) .................................. ⚠️ REABERTO 14/out — NÃO destila: derive_beliefs COPIA o representante do cluster VERBATIM (314/314 beliefs = cópia literal de uma memória-fonte, medido). conf 0.87-0.98 engana (é da fórmula, não de síntese). RFC §2 presumiu síntese no rewriter LLM a montante; o caminho real não passa por ele → síntese caiu num vão. É A DOENÇA do learning loop (ranking afina leme de barco sem motor). mem 0ab1aff0+25a0766. ARCO SÍNTESE abaixo.
│   ├─ L1 telemetria de proveito (usage_events) ............ 🟢 FORK 3/out (cc412cfe) · migration 014 no banco VIVO · opt-out MCP_USAGE_TELEMETRY
│   ├─ L2 injeção proativa (tool memory_context) ........... 🟢 FORK 3/out (f752dd13) · injeta por TEMA · E2E banco vivo 24k · opt-out MCP_CONTEXT_INJECTION_ENABLED
│   ├─ D2 feedback SÍNCRONO (rating→belief) ................ ↩️ REVERTIDO 3/out · letra morta (rating manual que ninguém faz; a RFC-MM-01 já previa)
│   ├─ L4 REAL = terminar RFC-MM-01 (feedback PASSIVO) ..... 🟢 FORK 5/out (reconciliado 6/out) · WIRING ATIVO: job quality_recompute (MCP_QUALITY_RECOMPUTE_SCHEDULE=6h, dry-run=true ADR-0006: agent_id=None contamina) + persist_quality_scores delega (clamp+effective_quality PRESERVA user_rating #1312) · tool get_assertiveness_metrics (ADR-0005) · 2944 tests pass
│   ├─ L4-unblock agent_id no retrieve/feedback/injection .. 🟢 FORK 6/out · _resolve_telemetry_agent_id (MCP_AGENT_ID, null-safe RFC#1100) nos 3 call sites de usage_events · derive_signals já particionava · 5 tests (REQ-A5 anti-contaminação) · banco vivo: retrieval grava agent_id=zero. PERSISTÊNCIA ainda dry-run: histórico 16965 eventos None contamina 96% até janela 14d limpar (ou expurgo) — decisão Claudio
│   ├─ ENRIQUECIMENTO 6/out (backfill + host) .............. 🟢 FORK · (a) backfill: 22534 memórias NULL→agent_id=zero (guri==zero), kiro/unknown preservados, backup+integridade OK · (b) host automático: RFC rfc-mcp-hostname-stamping, _resolve_hostname nos 3 call sites MCP (store_memory+2×session, G5 P2 fechado), flag MCP_MEMORY_INCLUDE_HOSTNAME=true ligada, E2E quente: metadata.hostname=DNBSCDC289.741 · host = CANDIDATO PR UPSTREAM (assimetria: flag só na Web API)
│   ├─ N1 janela 1 semana — medir proveito REAL ............ ✅ MEDIDO 10/out · ACHADO: injection_coverage=0.0 era BUG ESTRUTURAL, não falta de volume (belief_hash nunca == content_hash). L3 push CONFIRMADO disparando (E2E memory_search via /mcp).
│   ├─ 🔧 FIX PROVENIÊNCIA (10/out, 63ddfaaa, 3 hosts) ..... 🟢 FORK · injeção grava source_hashes (=derived_from do belief) no evento injection; _event_belief_hashes correlaciona por eles. injection_coverage 0.0→0.013 no vivo (1º ciclo inject→use). É Context Utilization / Chunk Attribution (pesquisa 10/out). test_injection_provenance_coverage.py (4).
│   ├─ ⭐ FRENTES B/C/E (10/out) ........................... 🟢 TODAS IMPL-FORK · ARC e3dc542d · gate+delegação completo · fork-only (falta só empacotar upstream = 85febdca)
│   │     ├─ B sinal-negativo (forgetting de belief inútil)  🟢 FORK @abbea491 · injetado ≥N sem uso → confiança −X%/ciclo → sai do top-k. usage_events + metadata JSON (zero DDL).
│   │     ├─ C freshness (versão nova vence na injeção) .... 🟢 FORK @5b8c6d1f · v2>v1 no top-k. timestamp boost em context_injection.py:274 (zero DDL).
│   │     └─ E eval adversarial + MRR ...................... 🟢 FORK @708dcd38 · script offline determinístico: MRR(vs 0.4140)+taxa_freshness+taxa_forgetting. Loop mensurável E2E discrimina feature on/off.
│   ├─ ⚠️ INC-5 "ligar score no RETRIEVAL" (RFC-MM-01 §5, item 76522321) . 🔴 INERTE — achado 12/out: retrieve_with_quality_boost (base.py:161) usa quality_score que NÃO discrimina (INC-0: 89% em 1.0 = no-op). NUNCA usou o utility_prior. ASSIMETRIA: injeção (re-rank A) aprende (util), mas retrieval (memory_search composite) não. NUANCE: util vive no BELIEF, retrieval busca MEMÓRIA (sem util) → pauta de design (mem 222f9138), não fix trivial. NÃO confundir com o re-rank A da injeção (esse está no vivo).
│   ├─ ⭐ ARCO LEARNING AGÊNTICO (ARC 07bcf182) · A+dreaming+D1/D2/D3+util-modular NO VIVO (sirdata) · fases:
│   │     ├─ F1 A re-rank .............. ✅ done · bc51c4a5+727d9757 · MCP_INJECT_RERANK_A · generic-wins 2/5→0/5
│   │     ├─ F2 D chunk-attribution .... ✅ done · D1 utility_prior passivo (b9a1d43e) Beta em metadata, NUNCA toca confidence, correlação por source_hash
│   │     ├─ F3 Dreaming Inc1 singular . ✅ done · 2d5b42c8+0ed9576e+fe735300 · MCP_BELIEF_SINGULAR 0.55 · 242 derivados byte-idênticos
│   │     ├─ F5 laço A↔D + D2 3-sinais . ✅ done · 0e720178(A consome util)+c9ba2a14(D2: never_used NEUTRO+assimetria W_NEG=3+clamp 0.1)
│   │     ├─ F6 D3 negative real ....... ✅ done · b5f0544e · re-query ≤300s após injeção → negative; _is_requery DRY
│   │     ├─ F7 util MODULAR ........... ✅ done 12/out (R2, 6abb063c) · aditivo→multiplicativo (rel*(1+λ*util_surplus), neutro=0 boost) · E2E: off-topic@3 0.083→0.0, MRR 0.658→0.683 · benchmark eval/rerank_bench.py (R1) + Recall@pool=100% (R0)
│   │     ├─ F8 R4 embedding no belief . ✅ done 12/out (2b82c1bc) · migration 021 + hook _create_belief + fusão lex+cosine (MCP_INJECT_RERANK_ALPHA=0.5) + backfill 2731 no vivo · E2E: MRR 0.725→0.750 · NO VIVO sirdata
│   │     ├─ F9 R1 derive por ontologia  ✅ done 13/out DNBSCDC289 (77e100ef..30e35232, item 2561825a) · _get_memories_by_ontology: base types curados {decision,learning,error,observation} via get_parent_type + store='default' + EXCLUSÃO RAG {document,reference} (defense-in-depth: resolvem p/ observation mas são doc ingerido) + opt-in MCP_BELIEF_ONTOLOGY_FILTERING + cap MCP_BELIEF_BATCH_CAP · GATE rok→reg→tuvok PASS + fix testes inertes + fix RAG (sabotagem) · 128 belief tests, 0 regressão · MEDIDO no vivo: corpus 17275→15585 (−1690 RAG) ouro operacional · ACHADO: multi-store JÁ funciona (default 18284/mir 4509/docs-shared 2154/hackathon 735); 1690 RAG vazaram p/ default em jun-jul (dívida histórica de ingestão, store=COLUNA não metadata) — exclusão por tipo cobre em qualquer store, não precisa mover dados · PENDENTE p/ vivo: ligar flag + re-rodar veredito CLIENT
│   │     ├─ F10 wire utility_prior recompute 🟢 FORK 14/out DNBSCDC289 (branch feat/wire-utility-recompute-scheduler, commit 4978ea3d) · ACHADO por medição: recompute_utility_priors era CÓDIGO MORTO (0 call-sites) → utility_prior NULL em 314/314 beliefs active → re-rank util_mult=1.0 p/ TODOS (frente D inteira no-op no vivo). FIX: _schedule_utility_recompute_job + _run_utility_recompute no ConsolidationScheduler, opt-in MCP_UTILITY_RECOMPUTE_SCHEDULE, espelha quality_recompute. Só escreve utility_prior (nunca confidence). MEDIDO no vivo: 1 run moveu 118 beliefs NULL→discriminante (82@0.60, 7@0.56, 27 neutro). 8 testes (mirror harvest, mutation-provado). tuvok APPROVED (pegou log órfão 'Quality recompute failed' duplicado → corrigido; diff puramente aditivo, quality byte-intacto). fork-only (depende da frente D, não existe no upstream). PENDENTE: ligar flag no env + mergear no fork main.
│   │     └─ F4 UPSTREAM ............... 🔴 queue TRAVADA · só com OK Claudio + corroboração Henry · ⚠️ VEREDITO CLIENT 13/out (R1 no vivo, mem 00f02b0c): corpus LIMPO (R1 resolveu — 2000 ouro, zero RAG) mas AINDA NÃO 100% feliz. A dor NÃO é mais CORPUS; migrou para: (a) LOOP LENTO (derive offline 03:00/dom → aprendizado de hoje não vira belief hoje) — ATACADO em parte por F10 (util agora recomputável por schedule) + INC6 (gatilho importância); (b) RANKING na query real (top relevante-por-palavra, não por tema); (c) FRAGMENTOS-LIXO destilados ('TYPE:','convention','Causa Raiz Identificada:') — ATACADO por PR #1533 (len>500 só é ruído se dump real). F4 NÃO abre até CLIENT feliz. Bug ONNX do venv corrigido (mistake 4e6323c5).
│   │     ⚠️ DESAMBIGUAÇÃO "D2": o D2 feito = frente D 3-sinais (c9ba2a14, VIVO). NÃO confundir com o "D2 feedback SÍNCRONO (rating→belief)" do plano ANTIGO, revertido 3/out (letra morta — rating manual que ninguém faz; virou o feedback PASSIVO).
│   │     📋 FILA (consenso 9 IAs, pesquisa ranking-util-modular-embedding-belief-2026-10, mem 12699e27): R3 propensity-por-rank no update Beta · R4 embedding no BELIEF (o que faz MRR saltar, tabela beliefs sem coluna hoje) · R5 fusão lex+sem · R6 FTS dual-index · R7 utility CONTEXTUAL por domínio (6/9: causa-raiz do rich-get-richer).
│   │     🔑 ACHADO 12/out (mem 9ff8c7b8): derive_beliefs consome SÓ memory_type=observation → decision/learning/checkpoint NUNCA viram belief. Beliefs têm 2 fontes (memory_store + harvest) via o store de memórias. Design aberto: só observation→belief é intencional ou gap?
│   │
│   ├─ ✅ BLOQUEADOR CRÔNICO RESOLVIDO 13/out: venv perdia onnxruntime (gates com uv removem [sqlite]) → R4 inerte → 'por palavra'. BLINDADO em 2 camadas: (1) ExecStartPre ensure-onnx-venv.sh na unit (auto-reinstala [sqlite] no restart, provado); (2) nota de proteção no Gate 3 do dev-workflow (não uv sync cru; manager repete no prompt do subagent). Mistake 4e6323c5/3d1b91f8. R4 medido no vivo pós-blindagem: relevância top-1 0.25→0.51-0.57 (~2×).
│   ├─ 💤 DREAMING-LOOP (item FASE 3 = 03c1d773, guarda-chuva) · o aprendizado "SONO" (offline: consolida/associa/agrupa por tema). Decomposto 13/out nos INC 2-7 do CONSOLIDADO (pesquisa dreaming-loop-navegacao-2026-10, 9 IAs). INC1=R1(F9)=✅. Ordem por dependência:
│   │     ├─ INC 2 consolidate(theme=...) ..... ✅ DONE (522d9cf6..13374daa) · theme filtra candidatos antes das 6 fases · 0 dep
│   │     ├─ INC 3 grafo heterogêneo tipado ... ✅ JÁ SATISFEITO · schema memory_graph já tem relationship_type populado (migration 9) — sem ALTER necessário. Era pré-req de 4/5/7.
│   │     ├─ INC 4 associações→injeção ........ ❌ CANCELADO/REPOSICIONADO 13/out · revertido 2× (subtrativo; tag backup/inc4-subtractive-20261013, stash dropado 6c8595af no reflog) · MEDIDO: R4 estável já resolve 'por tema PONTO-A-PONTO' (0.25→~0.55); o eixo de NAVEGAÇÃO ASSOCIATIVA multi-hop virou ARCO PRÓPRIO (40e6ecfb, ver abaixo) com pré-req qualidade-de-belief
│   │     ├─ INC 5 loop sono→acordado ......... 🔴 queue (ed46db92) · reweighting Hebbiano das arestas via usage_events (EMA+half-life) · dep INC3(ok)+ex-INC4(agora arco 40e6ecfb)
│   │     ├─ INC 6 gatilho por importância .... ✅ DONE (106b9576) · reflection-trigger por tema COMPLEMENTO ao cron (sinais existentes, não LLM) · ⭐ ataca LOOP LENTO · LIGADO NO VIVO 13/out (MCP_IMPORTANCE_TRIGGER_ENABLED=true no env DNBSCDC289, threshold 10/cooldown 30min defaults) · fix manager: reset de importância só após sucesso do task async
│   │     └─ INC 7 (opc) memory_explore + reflection LLM-gated 🔴 queue (f138b8d3, LOW) · dep INC3(ok) · só se os outros provarem valor
│   │     ✅ VEREDITO RE-MEDIDO 14/out (venv blindado + INC6 no vivo): INC6/scheduler HEALTHY (engasgo era rede caída, não design); storage DEGRADED→HEALTHY (11 missing embeddings reembedados). GARGALO REAL achado por medição: metade do top-k da injeção era META-BELIEF [GAP]/[PATTERN] → RESOLVIDO com MCP_INJECT_EXCLUDE_TYPES (medido 13→0, commit bae8a9aa, no vivo). Proposto ao upstream: discussion #1532 (learning loop awake/dreaming).
│   │     ➡️ PRÓXIMO (reordenado pela medição): NÃO é INC5/INC7. É QUALIDADE DE BELIEF (f0e2c306) — fragmentos 'TYPE:'/'convention' antigos ainda vazam; 6 degenerados óbvios expurgados (inactive) 14/out, expurgo em massa NÃO (14 falsos-positivos medidos no _is_noise). ACHADO p/ Henry: _is_noise agressivo demais p/ uso retroativo. INC5/INC7 depois. F4 upstream aguarda reação Henry em #1532.
│   │
│   ├─ 🟢 ARCO CONSUMIDOR DE GRAFO NA INJEÇÃO (40e6ecfb, ex-INC4 reposicionado 13/out) · navegação por tema/entidade na busca · SEPARADO do dreaming (que foca consolidação)
│   │     │   ACHADO (seven, código): memory_graph tem 3 arestas: (A) memória↔memória, (B) memória→entidade has_entity=agrupamento-por-tema estilo Obsidian, (C) entidade↔entidade SPO. Vídeo do Henry (/graph-visualization D3) desenha SÓ (A), EXCLUI has_entity de propósito (#256). Busca/injeção NÃO navega NENHUM grafo (~528k edges sem consumidor). memory_explore/detail usam has_entity+shares_entity, fora da busca.
│   │     ├─ PRÉ-REQ qualidade de belief (f0e2c306, HIGH) ... ✅ DONE 14/out (PR #1533: len>500 só é ruído se dump real, entropy<0.4 OR words<10; +497 memórias acionáveis resgatadas, medido)
│   │     ├─ RFC+SPEC consumidor ADITIVO (a74c0e1f) ......... ✅ DONE 14/out · docs/rfc/planned/rfc-learning-loop-graph-proximity.md (EARS R1-R7, CA1-CA3, G0 seven) · INVARIANTE append-only honrado
│   │     └─ Impl+gate (9fe4abd0) ........................... ✅ DONE 14/out (merge f45e9bbe) · gate G0(seven)→G1(RFC)→G3(rok)→G4(reg)→G5(tuvok 2 rodadas) · termo aditivo graph_proximity no _apply_rerank_a, opt-in MCP_INJECT_GRAPH_PROXIMITY+W_GRAPH (default 0.0=R4 byte-idêntico) · reusa get_graph_storage()+find_memories_by_entity+EntityExtractor (sem DDL) · 11 testes, 0 regressão · G5 pegou 4 bloqueadores (gate morto storage.graph→get_graph_storage #219; complexidade; CA1 baseline anchor por mutation; domain_extractors) TODOS corrigidos · ACHADO: minha verificação de manager passou código-morto (storage.graph), o tuvok independente pegou — review externo é obrigatório · ⚠️ ANTES DE LIGAR a flag: medir recall de entidades em prosa + CA3 MRR + tunar peso (seção 'Before enabling' no RFC)
│   │
│   ├─ ⭐⭐ ARCO SÍNTESE DE BELIEF (a DOENÇA, não o sintoma) · 🔴 NOVO 14/out · O NORTE REAL do learning loop
│   │     │   DESCOBERTA (seven + 3 medições banco vivo): o maquinário todo (re-rank A, util, grafo, NER, cosine) é RANKING. Mas o loop não quebra no ranking — quebra ANTES: derive_beliefs NÃO sintetiza, COPIA (314/314 verbatim). "belief = memória com crachá". Afinar ranking = leme perfeito em barco sem motor.
│   │     │   AUTORIA: NOSSO (belief_service.py criado #33 filhocf 03/jun; 13 commits Claudio). RFC docs/rfc/implemented/rfc-s2-belief-store.md marcada 'implemented' MAS presumiu síntese no rewriter LLM a montante (§3 Promotion + §4 Scope Classifier in distill/rewriter) — o derive_beliefs real pega observação crua SEM passar pelo rewriter → síntese caiu num VÃO de concepção. mem 25a0766+0ab1aff0.
│   │     │   A PERGUNTA (definição, não feature): "o que É um belief, QUEM o escreve abstrato, COMO sabemos que foi útil sem o agente reportar?" Hoje: belief=cópia verbatim; útil=reuso circular (injected_then_used=0).
│   │     ├─ PRÉ-REVISÃO RFC §2 (Claudio pediu 14/out): revisar rfc-s2-belief-store.md — o que falta p/ ser FUNCIONAL (síntese real), melhorias, mover de 'implemented' p/ 'planned/reaberto' honestamente (Gate +1: 'implemented' sem efeito é dívida)
│   │     ├─ G-1 VIABILIDADE: medir Recall@pool (existe belief que responde a query de tema? seven: se <0.8 prova que é corpus não ranking); auditar origem (15/314 beliefs têm fonte decision/learning; 578 observation dominam); verbatim-ratio (314/314 hoje)
│   │     ├─ Síntese real: derive_beliefs produz CLAIM abstrato do cluster — (a) ligar ao rewriter LLM que a RFC presumiu, OU (b) passo de síntese explícito (template/LLM + representante por centralidade, não o 1º do loop). belief_service.py:413-521 (_create/group_content), _group_observations:687-749
│   │     ├─ Incluir decision/learning/mistake na derivação (hoje só observation vira belief — mem 9ff8c7b8)
│   │     └─ Definir 'útil' com fonte de verdade real (injected_then_used=0 estruturalmente — proxy de reuso é circular/rich-get-richer)
│   │     ⚠️ INC5/INC7 do dreaming e 'ligar flag do grafo' ficam ABAIXO disto na prioridade — ranking sobre corpus-sem-síntese reencena rich-get-richer. Síntese PRIMEIRO.
│   │
│   └─ 📦 COMOs resgatados (protótipo 2e1978d5 = os 3 jobs da trilogia; branch validacao, 93 testes):
│       ├─ rfc-mm-02 fact-extraction (L2) ............. 🟢 resgatado · job scheduler + migration · spec: 6 req
│       ├─ rfc-mm-03 gap-detection (L4) ............... 🟢 resgatado · migration + handler · spec: 0 EARS (completar)
│       ├─ rfc-mm-01 feedback-loop (L4) ............... 🟢 resgatado = É o "L4 REAL" acima (passivo, da RFC do Claudio)
│       └─ p/ virar PR: completar spec ⚠️ (doc payload window-tools #1286 · EARS mm-03 · changelog) → 3 PRs → responder #1286
│
├─ 🟢 ARCO STORE PARTITION (multi-store) ... feature NOSSA · RFC retroativa docs/rfc/implemented/rfc-multi-store-partition.md
│   │   Coluna `store TEXT DEFAULT 'default'` em memories+vec0 · scope de search/delete/analytics por store · stores vivos: default, docs, docs-shared, mir, hackathon.
│   ├─ Feature base (#57→#62→#68→#1395→#1469) ............. ✅ MERGED upstream · store/retrieve/semantic-search scoped OK
│   └─ 🐞 DÍVIDA ABERTA (bugs do Kenzokono, issues OPEN 06/out) · RED commitado 0c37921e (5 testes provam o leak):
│       ├─ #1465 memory_delete ignora store (delete por hash/tag/tempo CRUZA partições — perda de dados) 🔴 PR candidato (fix de bug, não "muda ranking")
│       └─ #1466 memory_search(store=) vaza outros stores em hybrid/quality_boost/exact/tag-only/time-only 🔴 (só semantic/ranked honram) · MESMA RAIZ do INC-5 (ramos não-semânticos de base.py divergiram)
│
├─ ⭐ ARCO HARVEST DE SESSÕES (11/out, ARC 323e1654) ... learning de COLHEITAS (par do learning de memórias) · RFC rfc-harvest-sessoes
│   │   Dor: sessões .jsonl (~/.kiro/sessions/cli, 1336, ~900MB) = fonte mais rica, massivamente sub-colhida. Raiz = não depender da disciplina do agente.
│   ├─ FASE 0 planejamento + RFC + decisão arquitetura (A→B, C descartado) . ✅ DONE (G0 seven)
│   ├─ FASE 1 fix do tracker (667→19 regressão) ......... 🟢 FORK commit c5f3a3a4 · NO VIVO · persist_tracker (merge monotônico+anti-shrink+store-antes-delete+lock módulo) · gate 4 rodadas (tuvok reprovou guard invertido) · 6/6 · tracker reconstruído 19→1299 no vivo
│   ├─ FASE 1b fix do triage cego ao formato ............ ⚠️ WORKAROUND (MCP_HARVEST_TRIAGE=0) · FIX CÓDIGO PENDENTE (item f52de007) · triage.py só lê formato v4, descartava 100% do legado — causa-raiz real do furo
│   ├─ FASE 2 backfill das 788 pendentes (546 out) ...... 🔨 EM WORK · tracker reconstruído feito · dry-run comprova candidatos · FALTA harvest real com LLM (longo, cap anti-flood)
│   ├─ FASE 3 religar scheduler (MCP_HARVEST_SCHEDULE) .. 🔴 depende FASE 2
│   ├─ FASE 4 resumo-por-sessão (session-summary, dec.B)  🔴 depende FASE 3
│   └─ 🔧 INCIDENTE resolvido: serviço SEM backend embedding (onnxruntime/tokenizers ausentes no venv) recusava escrita (health 200 enganoso) → uv pip install · mistake 347f7009
│
├─ 🟢 METODOLOGIA DE CURADORIA DE FORK (3/out) ......... nasceu do erro: reimplementei a trilogia que já existia
│   ├─ metodologia-fork.md (3 rituais) + skill fork-curation + ref AGENTS.md . 🟢 permanente
│   ├─ LEDGER-feats.md + ORFAOS.md (índice vivo) ....... 🟢 invariante: nenhuma feat sem rastro
│   ├─ Inventário completo (29 refs órfãos, 1-a-1) ..... ✅ só 3 órfãos-reais; nenhum 2º protótipo escondido
│   ├─ delta-sync §8 (invariantes #1345) PORTADO ....... 🟢 @ef2555ac (era órfão em branch)
│   ├─ 19 branches-lixo + 3 worktrees deletadas ........ ✅ fork limpo (3 branches: main, trilogia, audit)
│   └─ limpeza cross-host (WI a0d5abf2) ................ 🔴 DNBSCDC289 + socrates (branches são por-máquina)
│
├─ 🟡 ARCO INGESTÃO MULTI-AGENTE ....... REDESENHO (foco) · RFC #1393 na discussion
│   │   "1 serviço, N agentes/clientes · regras por agente, plugáveis (YAML)"
│   │
│   ├─ Peças JÁ FEITAS (encaixam nas camadas):
│   │   ├─ I0 coverage instrument ......................... ✅ MERGED PR #1350
│   │   ├─ I0-lang idioma no Phase 0 ...................... ✅ MERGED PR #1379 (split extracted/dropped)
│   │   ├─ IA discovery workspace ......................... ✅ MERGED PR #1378 (find_sessions+resolve+guard)
│   │   ├─ IB parser SQLite/Crew .......................... ✅ MERGED PR #1379 (conversations_v2 read-only)
│   │   └─ I1 ToolResults CLI {kind} (78e29b05) ........... 🟢 FORK (msg+bloco; thinking=redacted, só contado)
│   │
│   ├─ C1 registro/descoberta de N fontes ............... 🔴 design (absorve source-identity)
│   │      declarativo + auto-descoberta assistida · lê sidecar identidade (agent_id.name/workspacePaths)
│   ├─ C2 perfil de parsing por agente (YAML) ........... 🟢 PR1 FEITO 5/out (5308efda) — harvest/agents/kiro.yaml + loader, parser lê do profile, byte-idêntico (arch provou). Gate G0-G5. Triagem já plugada (02/out). Vira PR upstream.
│   ├─ C3 extração/qualidade sinal-ruído ................ 🟡 design (absorve design-extraction) · ⚠️ BUG 11/out: triage.py (a peça de C3 já plugada) só lê formato v4 'payload' → descartava 100% das sessões CLI formato legado 'kind'. O PARSER lê os dois (C2 ok); o TRIAGE não. Workaround MCP_HARVEST_TRIAGE=0; fix no arco Harvest FASE 1b (f52de007). Lição: "parse OK" ≠ "triage OK".
│   │      heurísticas O(n) 1º (95% prosa / 4% json medido) → LLM só gated · honra locale
│   │
│   ├─ 📦 3 FONTES Kiro (WI d500bdfb — guarda-chuva "bring your agent"):
│   │   ├─ CLI (cli/*.jsonl + nested v4) ................ ✅ parse+discovery OK
│   │   ├─ IDE (workspace-sessions) ..................... 🟡 neste host JÁ é v4 payload-wrapped (= CLI, só path difere; parse ok). history[] = formato ANTIGO (outros hosts). discovery do path IDE = evolução (WI ad09b221)
│   │   └─ Crew (memory.db episodic/semantic) ........... 🔴 schema próprio, parse_sqlite lê só conversations_v2 (CLI). Importar via memory_store = evolução (WI 0a0fb0f9)
│   │
│   └─ 🐛 BUG CANAL MCP (WI e5ac16db) ................... tool memory_consolidate action=harvest dá sessions:1 com sessions=20. Arch PROVOU: NÃO é código (handler real dá 20). Stale no canal (mcp-proxy/cliente Kiro cache). FIX: reiniciar CLIENTE Kiro. Não é bug do produto.
│   │
│   ├─ triage.py (score valor, 2 eixos, 193 testes) ..... 🟢 PRONTO mas NÃO PLUGADO no harvester
│   └─ ⚠️ FASE 0 (fecha a dor, NÃO depende do Henry): plugar triage + Kiro→YAML + colher 11 OURO curados
│
├─ 🟡 ARCO PORTABILIDADE ............... design aceito pelo Henry (5 camadas)
│   ├─ discussion #1364 (5 camadas) ................... ✅ Henry aceitou o modelo
│   ├─ wiki Memory-Portability-Map ...................... ✅ no ar (mapa por harness)
│   ├─ "bring your memory": conversor mem0 .............. 🟡 PR #1401 (Harbor404, fecha nosso #1390)
│   ├─ "use in any agent" — lado SERVIDOR ............... → é o arco ingestão multi-agente (acima)
│   └─ 🟡 UIAA — lado CLIENTE (harness) ................. `rfc-hermes-first-class` v0.1 (19/out)
│       └─ hermes ...................................... híbrido: MEMORY.md p/ hoje · mms p/ histórico
│           ├─ D1 convivência .......................... ✅ HÍBRIDO — provider desligado É o estado correto
│           ├─ D2 provider plugin ...................... 🟡 adiado · moraria em $HERMES_HOME/plugins/mms/
│           ├─ D3 diagnóstico e degradação ............. 🔴 `mms-doctor` (6 checks + reparo seguro, ≤10 s)
│           ├─ D4 autoria X-Agent-ID ................... ✅ MERGED #1598 (d6d3fd47, 19/out 19:15Z) · ✅ integrado no fork (c675458c)
│           ├─ D5 importer do state.db ................. 🔴 construído e NUNCA aplicado · dry-run 77/113 sessões
│           ├─ D6 hooks TUI/CLI ........................ ✅ funciona desde 0.22.2 (3094cde513, 07/set)
│           ├─ D7 instalador de 1ª classe .............. 🔴 a entregar · checklist 8 itens · `scripts/hermes/`
│           └─ D8 porta/transporte ..................... 🟡 3202 (SSE legado) vs default 8000 — decisão ABERTA
│
└─ 🟡 ARCO HUB MULTI-AGENTE ............ agent_id feito; delta-sync 4/5 fases mergeadas
    ├─ agent_id F1/F2 (autoria no store) .............. ✅ MERGED PR #1278/#1297
    ├─ SPEC-hub F0-F8 ................................. 🟡 spec pronta
    ├─ delta-sync (#1345) ............................. ✅ 4/5 MERGED (ver arco DELTA-SYNC no topo) · Fase 5 fecha
    └─ F3-F8: estrela, consolidação nas pontas, NLI cross-agent  🔴 a fazer

   ⭐ ARCO INBOX INTER-AGENTE (14/out) ... entrega confiável de mensagem entre agentes · NOSSA proposta ao projeto · RFC fork-only docs/rfc/planned/rfc-inter-agent-inbox.md
    │   TESE: heartbeat/disciplina periódica do agente é não-confiável → gatilho de entrega tem que ser SERVER-SIDE. Evidência (4 casos nossos): checkpoint-gate (falhou), A3-pull (falhou), session-registry heartbeat (falhou parcial), learning-scheduler server-side (funciona). Endereçamento 3 níveis (host/agent/session) p/ N instâncias do mesmo agente.
    ├─ RFC escrita + revisada (seven) ................. 🟢 commit e2fd1cf0 (fork main)
    ├─ Discussion #1529 (Ideas, upstream) ............. 🟢 postada + ping #1526 · aguardando reação do Henry
    └─ 3 etapas: (1) pull tag [= #1526/#1527 do Henry] · (2) endereçamento dirigido + injeção ativa · (3) gatilho server-side push [o alvo real, SSE address-filtered] 🔴 depende da reação do Henry na discussion
```

## Onde estamos, em uma frase
Fechamos o arco de rating; descobrimos que os 3 arcos de harvest/identidade/portabilidade são **o mesmo problema** — colher bem de N agentes exige regras por agente **plugáveis (YAML)**, não hardcoded. Estamos redesenhando isso como um arco só (ingestão multi-agente) antes de escrever mais código.

## 📍 ONDE ESTAMOS / PRA ONDE VAMOS (atualizado 3/out noite)

**Entregue hoje (fork main, pushado, aplicado no banco vivo):**
- 🟢 **Telemetria de proveito** (`usage_events` + `get_usage_metrics`, migration 014) — o serviço observa o próprio uso. Opt-out `MCP_USAGE_TELEMETRY`.
- 🟢 **Injeção proativa** (tool `memory_context`) — injeta beliefs por TEMA, não top-N cego. Opt-out `MCP_CONTEXT_INJECTION_ENABLED`.
- 🟢 **Metodologia de curadoria de fork** — `metodologia-fork.md` + `LEDGER-feats.md` + `ORFAOS.md` + skill `fork-curation` + ref no AGENTS.md. Nasceu de um erro real (reimplementei a trilogia que já existia).
- 🟢 **Inventário completo de material oculto** — 29 refs órfãos varridos 1-a-1. Só 3 órfãos-reais (trilogia resgatada, delta-sync §8 portado, audit macro-map). 26 refs-lixo deletados (local+remoto). Fork limpo.
- ↩️ **Revertido:** feedback síncrono (rating→belief) — LETRA MORTA (depende de rating manual que ninguém faz; a RFC-MM-01 já previa).

**Correção de rumo:** o L4 feedback é a **RFC-MM-01 do Claudio** (jul, feedback PASSIVO "zero disciplina"), que o Henry endossou na #1286 — não "modelo do Henry". A telemetria de hoje materializa ela.

**Pra onde vamos (ordem):**
1. 🔨 **Terminar a RFC-MM-01** fork-only sobre a telemetria: captar reaccess + job de recálculo de quality_score + wiring real no fluxo + acceptance (quality 0.5→0.65, bootstrap noise <10%). Arco 62046f1e.
2. **Janela 1 semana** (N1, ~10/out) medindo proveito real → decide N2 (push via hook) e N3 (feedback positivo).
3. **Trilogia** (branch `validacao/prototipo-trilogia`, 93 testes): completar spec p/ PR (doc payload window-tools #1286, EARS mm-03, changelog) → 3 PRs + responder #1286 (draft EN+PT-BR).
4. **Limpeza cross-host** (WorkItem a0d5abf2): worktrees sirdata + branches-lixo DNBSCDC289/socrates.

## 📋 BACKLOG COMPLETO — todas as RFCs/ideias (para enxergar tudo)
> Detalhe e evidência de código em `docs/rfc/_index.md` (47 RFCs: 23 implemented, 24 planned). Aqui = visão de uma linha por ideia, agrupada. Regra: toda RFC/arco/ideia nova entra AQUI no mesmo passo.

**🟢 ARCO LEARNING LOOP (o NORTE)** — ver árvore acima. RFCs: `rfc-learning-loop` (guarda-chuva) · specs IMPLEMENTADAS (fork, antes órfãs de rastreio): `rfc-learning-loop-a-rerank` (F1), `spec-learning-loop-b-negative-signal` (B), `spec-learning-loop-c-freshness` (C), `spec-learning-loop-e-eval-mrr` (E), `rfc-dreaming-loop-inc1-distill-specific` (F3) · roadmap ranking R3-R7 em mem 12699e27 (R4=item FASE 8).
Alimentado por (os COMOs, hoje design/0-EARS): `rfc-mm-01-feedback-loop`, `rfc-mm-02-fact-extraction`, `rfc-mm-03-gap-detection`, `rfc-fact-extraction`, `rfc-self-service-memory-intelligence`, `rfc-autolearn-autodream`, `rfc-server-side-lifecycle`.

**🟡 ARCO INGESTÃO MULTI-AGENTE** — ver árvore. RFCs: `rfc-ingestao-multi-agente` (guarda-chuva), `spec-fase0-kiro-yaml-triagem` (Fase 0 pronta), `rfc-pipeline-harvest-quality` ✅impl.

**🟡 ARCO HUB MULTI-AGENTE** — ver árvore. RFCs: `rfc-hub-memoria-centralizada` (SPEC F0-F8), `rfc-delta-sync` (#1345), `rfc-agent-id-multi-agent` ✅impl, `rfc-sync-multi-agente` ✅impl.

**🟡 ARCO PORTABILIDADE** — ver árvore. RFCs: `rfc-memory-portability` (5 camadas, #1364), `rfc-importers` (mem0/letta/zep, #1390).

**⚪ BACKLOG DE IDEIAS (sem arco ainda — ranking de prioridade a definir):**
- `rfc-query-intent` — 🟡 PAUSADA (WorkItem 9fa7e0ef, hold): teste do modelo de gerência; 4 arquivos não-commitados; valor marginal (ver ironia learning-loop). Decidir retomar/arquivar.
- `rfc-working-memory` — contexto quente auto-gerenciado (tier de memória recente). Conecta com injeção/L3.
- `rfc-ranking-upgrades` — Weibull decay + polyphonic recall (melhora recuperação).
- `rfc-memory-hygiene` — noise audit & safe cleanup (limpeza segura do acervo).
- `rfc-persona-tier` — identidade portável auto-gerada (conecta learning-loop + portabilidade).
- `rfc-skill-auto-generation` — pipeline de auto-geração de skill (padrão→skill).
- `rfc-embedding-quantization` — int8/binary vectors (performance/espaço).
- `rfc-multimodal-memory` — imagem/vídeo/áudio → texto recallável.
- `rfc-onboarding-discoverable` — guias de uso MCP agent-native (discovery protocol).
- `rfc-kiro-headless-llm-provider` — Kiro CLI headless como LLM provider (destilação sem API externa).
- `rfc-structural-improvements` — melhorias estruturais diversas.

**✅ INFRA JÁ IMPLEMENTADA (base, não precisa ação):** belief-store, nli-cascade, anti-hallucination, handler/tool-registry/routing, split-config, schema-versioning, multi-store-NER (NER por domínio ≠ store-partition), quality-model. Detalhe no `_index.md`.
**🟢 STORE PARTITION (multi-store, feature de DADOS — distinta do NER):** RFC retroativa `rfc-multi-store-partition` · base MERGED (#57/#62/#68/#1395/#1469) · DÍVIDA aberta #1465/#1466 (ver arco acima). NÃO confundir com `rfc-multi-store-domain-ner` (que é roteamento de NER por domínio).

## ◄ RE-ENQUADRAMENTO (02/out): o NORTE é APRENDER, não colher
> Pergunta do Claudio: "de que adianta colher memórias se não vira CONHECIMENTO? Como me fazer APRENDER?"
> Nova RFC guarda-chuva: **`docs/rfc/planned/rfc-learning-loop.md`** (do colhedor ao aprendiz).

O produto NÃO é "mais memórias colhidas" — é o **ciclo de aprendizado fechado**:
```
colher(limpo) → DESTILAR[L2] → VALIDAR/feedback[L4] → INJETAR proativo[L3] → agir melhor → recalibra ⟲
```
- **L1 Memória** ✅ · **L2 Destilação** ❌ REABERTO 14/out: NÃO destila (314/314 beliefs verbatim, derive_beliefs copia o representante cru; conf alta é da fórmula, não síntese) — é A DOENÇA · **L3 Injeção** 🟡 por-tema (memory_context, one-shot) · **L4 Feedback** 🟡 1º passo fechado (negative-use).
- **Colher (Fase 0) é PRÉ-REQUISITO, não o objetivo** — enche o reservatório limpo. O motor é destilar+validar+injetar.
- **Gargalo real = SINAL DE USO** (o serviço é cego pro agente; não sabe se a injeção ajudou). Priorizar **negative learning** (contradição = sinal forte e barato) antes do positivo.
- **É do SERVIÇO, não do agente** (agente é efêmero; serviço é o substrato persistente). Não é fine-tuning — o aprendizado vive no harness, efeito indistinguível de aprender no contexto.
- **Pesquisa JÁ EXISTE** (não reinventar): CdIA `padroes/mcp-memory-autolearn-rfc.md` (11 sistemas + 8 fases) + plano self-improvement. Reler autolearn-rfc §2.2/§2.4 antes de implementar.
- **Caminho mínimo:** (1) ~~limpar beliefs~~ ✅ já resolvido → (2) destilação ✅ (belief store já destila) → (3) feedback-loop ✅ **1º passo 3/out: negative-use** (rating −1 derruba confiança; E2E −47,8%) → (4) injeção proativa ✅ **3/out: tool `memory_context` por tema** (E2E banco vivo) + (5) **telemetria de proveito ✅ 3/out** (usage_events, migration 014 no vivo). **PRÓXIMO: janela 1 semana medindo proveito real (N1), depois push automático (N2) e feedback positivo/automático (N3).**


## Feito recentemente (2/out)
- **Fork sincronizado + serviço resgatado:** merge upstream/main (7 commits; nosso #1404/#1352 supersession mergeado pelo Henry). Serviço systemd em 11.14.0, 11 OURO colhidas.
- **2 bugs achados e corrigidos POR GATE (reg + tuvok), não na mão:**
  - **Bug rewriter `TYPE:` leak** (real, upstream também) → issue #1417 (CLOSED) + **PR #1418 ✅ MERGED** (02/out, Henry). O LLM ecoava o placeholder literal `TYPE:` para dentro do content + degenerados; `_unleak_type` + prompt `<type>:`.
  - **Bug triagem nunca ativada** (fork-only): `_apply_triage` plugado mas nenhum call-site passava a flag → feature morta. Corrigido (`harvest_config_from_env` nos 4 entry points, default OFF). NÃO vira PR de bug avulso — a triagem não existe no upstream, então entra junto no PR de FEAT.
    - ⚠️ **RECONCILIAÇÃO 11/out:** a triagem ativada (MCP_HARVEST_TRIAGE=1) tinha um 2º bug mais grave — `triage.py score_session` só lê o formato v4 `payload`, mas as sessões reais são formato LEGADO `kind` → dava score 0/empty p/ TODAS → descartava 100% das sessões ANTES do parser. ERA a causa-raiz real de 61% das sessões nunca colhidas. Workaround: MCP_HARVEST_TRIAGE=0. Fix de código pendente (arco Harvest FASE 1b, item f52de007). As "11 OURO colhidas" de 2/out foram com um acervo-fixture curado (layout 2-níveis), não as sessões reais do dia-a-dia.
- **11 OURO gravadas** no banco vivo via `memory_consolidate action=harvest` (in-process, integridade intacta), triagem ativada (`MCP_HARVEST_TRIAGE=1`), rewriter já corrigido → 0 vazamento.
- **Devolutiva do Henry (29/set) nos RFCs do arco** (ver "Em andamento").

## Em andamento
- **Arco ingestão multi-agente — a bola está do NOSSO lado (Henry respondeu via #1346, não via #1393).**
  A discussion #1393 (nosso guarda-chuva, 30/set) ainda só tem o nosso comentário. MAS o Henry respondeu o mesmo arco na **issue #1346 (design-extraction, 29/set)** com pedido concreto antes de qualquer extractor:
  1. **Tornar o coverage visível** — `coverage_report()` (#1350) existe mas nenhum call-site em `src/` o lê; pôr no resultado do harvest + log. "Phase 0 isn't done" até isso. **← próximo PR que ele pediu.**
  2. **Nota de compatibilidade** — quantas memórias a mais o harvest escreve (extractor + path v4 tool_result do #1366), tags `harvest:mode:*`, como desligar.
  3. **Números do kill switch** — taxa de adoção + threshold + data de revisão.
  4. **Legacy Kiro `ToolResult`** ainda dropado vs v4 mantido — unificar ou documentar o porquê.
  Ordem dele: "cheapest first", PR do ponto 1 + respostas 2/3 como update do RFC. → Em vez de "ping" na #1393, entregar o ponto 1 e aí comentar linkando.
- **#1345 delta-sync:** Fase 1 do #1304 MERGEADA (#1470) destravou a raiz (list_content_hashes). Falta Fase 2 (remote_http) virar PR. PR #4 recuperado em recovered/pr4-delta-sync-v04.
- **PR #1418 (rewriter):** ✅ MERGED pelo Henry (02/out). 1 PR de bug pode coexistir com outros (regra 2/out: 1-por-vez só p/ feat).

## A DOR (o que falta de verdade, em 1 frase)
A colheita do Kiro **lê os formatos** (feito) mas **NÃO filtra valor**: hoje colheria "hello v3", testes e dumps junto com o conhecimento. A triagem que resolve isso (`harvest/triage.py` — score calibrável, 2 eixos, 11 testes verdes) **já existe no código mas NÃO está plugada no harvester**. Fechar a dor = plugar a triagem + colher o acervo bom.

## FASE 0 — tarefas concretas (fecha a dor; NÃO depende do Henry — refactor seguro)
- [x] **T1. Plugar `triage.py` no harvester** ✅ FEITO (c1d52109, opt-in MCP_HARVEST_TRIAGE, sync+async, 194 testes) — antes de colher/gravar, descartar sessões `teste`/`vazia`/`truncada`/`dump` (score < limiar). Hoje o `find_sessions` colhe tudo que o parser reconhece.
- [ ] **T2. Extrair regras do Kiro → `harvest/agents/kiro.yaml`** — maps hardcoded (KIRO_KIND_MAP, markers, cutoff, globs) viram YAML declarativo (camada C2). Refactor dado→config.
- [ ] **T3. Golden test** — `coverage_report()` byte-idêntico antes/depois (prova que não quebrou os 3 formatos já suportados).
- [ ] **T4. COLHER o acervo curado** — AÇÃO DEDICADA (demora/LLM): rodar harvest_and_store use_llm no acervo. Dry-run deu 11 sessões/23 cand heurístico; LLM refina o resíduo intra-sessão. Decisão Claudio: colher via LLM (não heurístico cru). — rodar o harvest com triagem nos 11 OURO+CONVERSA já triados (`~/local-data/kiro-harvest-curado/`). **É aqui que a dor morre:** conhecimento real do Kiro entra na memória, sem o lixo. dry-run 1º.
- Peça pronta: `src/mcp_memory_service/harvest/triage.py` + `tests/harvest/test_triage.py` (193 verdes). Fixture: `~/local-data/kiro-harvest-curado-v2/curated/{uuid}/messages.jsonl` (11 sessões, layout 2-níveis que o find_sessions acha).

## Depende do Henry (paralelo, não bloqueia a Fase 0)
1. **Discussion #1393** — Henry avaliar o formato em camadas + Kiro→YAML como 1º incremento (RFC ingestão). Não respondeu ainda.
2. **PR #1404** (fix #1352) — CI/Greptile + merge do Henry.
3. Se #1393 ok → materializar specs das outras camadas (C1 multi-fonte, C3 LLM-extractor).

## Decisões abertas (esperando você)
- **Escopo da Fase 0:** (a) fork-only — plugar triagem + colher os 11 OURO (fecha a dor HOJE); ou (b) + `kiro.yaml` + golden test visando PR upstream (generaliza, mais trabalho). A dor fecha com (a); (b) é (a) empacotada p/ o Henry.

## Onde está o quê
- **RFCs:** `docs/rfc/planned/` (futuro) e `docs/rfc/implemented/` (feito). Índice: `docs/rfc/_index.md`.
- **Radar de terceiros:** `docs/_fork/radar-terceiros.md` (PRs/issues de terceiros na nossa área)
- **Acompanhamento (fork-only):** `docs/_fork/` — roadmap, pilha de PRs, este ESTADO.md, ARCOS.md (versão densa p/ o agente), `worktrees.md` (registro canônico dos worktrees, um por sessão).
- **Decisões de arquitetura:** `docs/adr/` (ADR-0001 adotar · 0002 sync=delta-sync · 0003 aprendizado=uso real · 0004 working-memory reusa memory_context · 0005 métrica assertividade proposta) + `docs/rfc/README.md` (árvore de arcos).
- **Metodologia de fork:** `docs/_fork/metodologia-fork.md` (3 rituais) + `LEDGER-feats.md` (estado de cada feat) + `ORFAOS.md` (detector de trabalho não-relandado). Consultar ANTES de sync/merge/implementar feat.
- **Estudos/pesquisas/guias:** ficam no CdIA (`conhecimentos-de-ia/ferramentas/mcp/memory-service/estudos|guias`).
