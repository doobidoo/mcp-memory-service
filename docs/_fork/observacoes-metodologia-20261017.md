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
| A6 | G5 adversário achou o que 12 testes verdes não acharam | reviewer confirmou por execução 3 bloqueadores: delta vazio (perda de edição), falha no lote (duplicata), 200+JSON-RPC error lido como sucesso | Review independente é obrigatório antes de qualquer escrita no acervo; teste verde não substitui auditoria adversarial |
| A7 | Subagente reportou um teste que passou de cara, investigou e o manteve como guarda | T19 (read-only em WAL): a implementação já satisfazia CA7 naquele cenário; em vez de forçar RED falso, o rok explicou e manteve o teste como regressão | Instruir explicitamente: se um teste novo passar de imediato, investigar e reportar — nunca inflar RED |
| A8 | O ciclo completo G0→G6 funcionou e se pagou | 1 investigação delegada, 3 ciclos RED/GREEN, 2 passagens de review; 22 testes verdes e volume em 4,86% do lido (era 46,8%) | O gate não é burocracia: cada rodada achou um defeito real (CA3 violado, 3 bloqueadores, 3 ressalvas) |

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
| B9 | Edição concorrente de artefato compartilhado | ao tentar atualizar `workflow-memoria-checkpoint` (porta 8000 / troca de tema), ela JÁ estava atualizada com o modelo de dois eixos — feita por outra sessão em paralelo | Definir convenção de colisão para skills/config entre instâncias; verificar antes de editar (ler o arquivo, não a versão em cache) |
| B10 | Testes verdes com critério de aceitação violado | G4 passou 9/9 e ruff limpo; o dry-run REAL mostrou 9,33M de 19,94M chars (46,8%) contra CA3 ≤10% — o mock do transporte escondia o defeito | Rodar o caminho real com dados reais ANTES de declarar G4 cumprido; teste verde não é critério atendido |
| B11 | Vazamento de payload de tool SÓ apareceu em produção | 48 das 88 memórias importadas começam com `[tool]` e carregam até 2.000 chars de saída crua. Causa: o resumo corta por "primeira linha", e a saída real de `read_file` é um JSON de UMA linha — logo a "primeira linha" é o arquivo inteiro. As fixtures dos testes tinham quebras de linha | Fixtures devem incluir o formato REAL (linha única e longa); todo resumo de tool precisa de teto em caracteres, não só corte por linha. Corrigir e limpar o que já foi gravado |
| B12 | Métrica errada quase gerou falso positivo (erro meu) | após corrigir B11, contei "memórias >260 chars começando com [tool]" e concluí que o fix falhara. Errado: o chunk do serviço (1.500 chars) começa com um turno de tool e segue com turnos de assistant. Medindo o TURNO gerado: 571 chamadas, máx 227 chars, zero acima de 260 | Definir a métrica ANTES de concluir: medir a unidade de origem (turno), não o artefato derivado (chunk). Verificar antes de reportar bug |

## C. Decisões metodológicas tomadas hoje

| # | Decisão | Razão |
|---|---|---|
| C1 | Dois eixos: sessão pode misturar temas; MEMÓRIA não (endereço explícito) | Claudio trabalha vários temas por sessão e 3-6 CLIs simultâneos |
| C2 | Profiles só para identidades de máquina (zero/chakotay/trip); projetos via `hermes project` | Profile duplica skills e fragmenta memória |
| C3 | Papéis de execução = delegate + toolsets recortados + modelo por papel | Contexto é orçamento: mms com 30 tools = ~10.2k tokens por turno |
| C4 | Fork-first com números; upstream só depois de medido | Prática do Claudio; documentação no repo é a contrapartida |
| C5 | Não criar RFC nova: revisar as duas existentes + spec onde não havia nada | Evitar duplicação (já custou trilogia e quality-recompute) |
| C6 | Calibração de custo: incremento trivial (≤50 linhas, 1 arquivo) usa UM worker em TDD interno (RED mostrado antes do GREEN); lógica nova ou >1 arquivo usa dois workers (rok/reg); review independente é sempre separado | Cada ciclo de dois workers custou 2-10 min; para 3 linhas isso é ritual, não rigor. O valor está no RED falhar antes, não em quem digita |

## D. Pendências de método (para a conversa final)

1. Pinar modelo por papel na delegação (claude-sonnet-5 em review; amazonia-a em texto PT-BR).
2. Definir se o subagente de execução recebe mms em modo leitura (e como recortar tools).
3. Reconciliar ARCOS/LEDGER/ESTADO antes da próxima frente.
4. Atualizar `workflow-memoria-checkpoint` (porta 3202, store/agent_id, troca de tema).
5. Transformar B4/B5 em pedidos de melhoria com critério de aceitação (pauta CLIENT).
