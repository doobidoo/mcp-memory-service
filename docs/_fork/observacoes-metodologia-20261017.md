# Observações de método e ferramentas — sessão 2026-10-17

> Registro VIVO, fork-only, do EXPERIMENTO paralelo: enquanto construímos o
> acoplamento mms ↔ Hermes (produto), estamos testando a metodologia SDD, os
> gates G0-G6, a delegação via LIA e o próprio mms como memória.
> Regra: observação vira lição acionável — não desabafo. Cada item tem
> evidência e uma consequência prática.

## A. O que funcionou

| # | Observação | Evidência | Consequência |
|---|---|---|---|
| A1 | Delegação via LIA funcionou ponta a ponta | subagente arquiteto: Qwen/Qwen3.8-27B, 21 chamadas, 152 s, resposta estruturada com arquivo:linha | Manter Qwen como padrão de execução; LIA é infra utilizável |
| A2 | Gate 0 (thinking outsourcing) produziu o que eu não teria produzido sozinha | G0 revelou que plugins do mms são reativos (on_store) e que não há hook data-source — derrubou o caminho (1) da issue #1095 | Manter a delegação de investigação; eu não devo grepar código |
| A3 | Spec em EARS + prosa força precisão | 8 requisitos, cada um com critério verificável; critérios CA1-CA7 numéricos | Adotar como padrão para todo incremento |
| A4 | Medir antes de especificar mudou a spec | tool = 89% do volume (17,9 M chars); sem essa medição, a spec ingeriria 18 MB de ruído | Medição entra no G0 como entregável, não como opcional |
| A5 | O mms respondeu perguntas históricas com filtro temporal | after/before mudou o conjunto retornado (08/out → 02/out) | Usar janela temporal em toda pergunta histórica |

## B. O que falhou ou atritou

| # | Observação | Evidência | Consequência |
|---|---|---|---|
| B1 | G0 mapeou CÓDIGO, não RUNTIME | a spec inicial propunha POST /api/sessions; teste real: /api/health, /api/sessions, /api/memories → 404; só 3202 (MCP) escuta | Adicionar verificação de runtime ao G0 quando a feature depender de endpoint |
| B2 | Documentação de estado divergente no repo | ESTADO.md diz delta-sync Fase 5 MERGED #1494; ARCOS.md e LEDGER dizem "próximo/DESIGN" | Antes de qualquer frente nova, reconciliar ARCOS/LEDGER com ESTADO |
| B3 | Skill `workflow-memoria-checkpoint` desatualizada | cita porta 8000; hoje 3202; não menciona store/agent_id | Atualizar a skill no mesmo passo de qualquer mudança de porta/contrato |
| B4 | `memory_context` estoura o orçamento no payload | budget 300/1.200/4.000 → content OK (996/4.680/15.839) mas resposta total 18.405/73.301/224.845 chars (~15x) | Bloqueia injeção automática por turno; feature candidata (payload enxuto) |
| B5 | `max_response_chars` degrada mal | 1500 → "Showing 0 of 3 results" | Não usar esse parâmetro; bug candidato |
| B6 | Store próprio tranca a memória fora das buscas | busca sem `store` retorna apenas o default; store=mir devolve outro conteúdo | Memória do Hermes vai em `default` + tag `source:hermes` + agent_id |
| B7 | Config com linhas possivelmente mal posicionadas | `provider: lia` / `model:` aparecem fora do bloco `delegation` na leitura por `sed`, mas `hermes config get` lê corretamente | Não editar config.yaml à mão; sempre `hermes config set`; investigar o YAML depois |
| B8 | Meu comportamento inicial violou o §8.1 | eu mesma rodei greps no código antes de delegar | Autocorreção: a partir do G0, investigação é delegada |

## C. Decisões metodológicas tomadas hoje

| # | Decisão | Razão |
|---|---|---|
| C1 | Dois eixos: sessão pode misturar temas; MEMÓRIA não (endereço explícito) | Claudio trabalha vários temas por sessão e 3-6 CLIs simultâneos |
| C2 | Profiles só para identidades de máquina (zero/chakotay/trip); projetos via `hermes project` | Profile duplica skills e fragmenta memória |
| C3 | Papéis de execução = delegate + toolsets recortados + modelo por papel | Contexto é orçamento: mms com 30 tools = ~10.2k tokens por turno |
| C4 | Fork-first com números; upstream só depois de medido | Prática do Claudio; documentação no repo é a contrapartida |
| C5 | Não criar RFC nova: revisar as duas existentes + spec onde não havia nada | Evitar duplicação (já custou trilogia e quality-recompute) |

## D. Pendências de método (para a conversa final)

1. Pinar modelo por papel na delegação (claude-sonnet-5 em review; amazonia-a em texto PT-BR).
2. Definir se o subagente de execução recebe mms em modo leitura (e como recortar tools).
3. Reconciliar ARCOS/LEDGER/ESTADO antes da próxima frente.
4. Atualizar `workflow-memoria-checkpoint` (porta 3202, store/agent_id, troca de tema).
5. Transformar B4/B5 em pedidos de melhoria com critério de aceitação (pauta CLIENT).
