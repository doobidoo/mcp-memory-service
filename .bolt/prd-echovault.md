\# PRD — \*\*Fork & Extend \`doobidoo/mcp‑memory‑service\`\*\*    
\#\#\# “EchoVault MCP v2.0 (Neon \+ Qdrant \+ Cloudflare R2) — \*Bolt.new‑First\* Guide (Claude 4 Opus/Sonnet)”

| Version | Author | Last Edited | Status |  
|---------|--------|-------------|--------|  
| 1.1‑bolt | Bjorn Hansen | 2025‑05‑25 | \*\*Import into Bolt.new\*\* |

\---

\#\# 0 · Context & Problem Statement (Why we’re building this)

1\. \*\*Current state\*\* — \`doobidoo/mcp-memory-service\` gives Claude Desktop long‑term memory via ChromaDB, but:  
   \* Data grows past Chroma’s in‑memory comfort zone.  
   \* No built‑in blob off‑load → large diffs balloon DB size.  
   \* No telemetry → blind to cost / latency.  
2\. \*\*Immediate need\*\* — support multi‑agent workflow (Claude Desktop, Cursor, Jules) while staying in \*\*\\$0 free‑tier\*\* budgets.  
3\. \*\*Solution\*\* — EchoVault MCP layers durable, cheap storage (Neon \+ R2), fast ANN search (Qdrant Free), and observability on \*\*top\*\* of the existing codebase \*\*without rewriting its proven retrieval logic\*\*.

\---

\#\# 1 · Guiding Principles for Bolt

\* \*\*Overlay, don’t rewrite:\*\* re‑use classes & API contracts already present in \`doobidoo/mcp-memory-service\`; extend via new modules or subclassing only when current implementation is insufficient.  
\* \*\*Minimal‑surface diffs:\*\* each Build‑Mode request may touch \*\*≤ 2 files\*\*; never mass‑replace existing packages.  
\* \*\*Fail fast with tests:\*\* acceptance tests (section 2\) must remain green after every build run.  
\* \*\*Claude 4 Opus/Sonnet Thinking:\*\* all reasoning & code generation should assume Claude‑style function‑calling and chain‑of‑thought.

\---

\#\# 2 · Success Criteria (Bolt tests)

| ID | KPI / Test | Pass Condition |  
|----|------------|----------------|  
| T‑1 | \`pytest tests/test\_write\_then\_query.py\` | round‑trip \< 500 ms, uses Neon & pgvector |  
| T‑2 | \`pytest tests/test\_blob\_offload.py\` | payload \> 32 KB stored in R2; row has \`payload\_url\` |  
| T‑3 | \`GET /metrics\` | Prometheus-formatted response |   
| T‑4 | \`scripts/summarise\_old\_events.py\` | reduces rows \> 30 d to ≤ 10 % original count |

\---

\#\# 3 · Things \*\*NOT\*\* to do  ➡️  \*\*Forbidden Work\*\*

\* \*\*Do NOT\*\* touch or regenerate existing \`tag\_storage\`, \`time\_based\_recall\`, or backup helpers—they’re stable.    
\* \*\*Do NOT\*\* change REST endpoint signatures.    
\* \*\*Do NOT\*\* swap out sentence‑transformers or Chroma unless instructed—Chroma may remain for tiny-memory fallback.    
\* \*\*Do NOT\*\* introduce new web frameworks (keep FastAPI).    
\* \*\*Do NOT\*\* exceed free‑tier quotas (Neon 0.5 GB, Qdrant 1 GB, R2 10 GB).    
Bolt must refuse any build request that violates the above.

\---

\#\# 4 · Architecture Overlay

\`\`\`text  
Existing  modules (unchanged)  
┌─────────────────────────┐  
│  memory\_wrapper.py      │  
│  tag\_storage.py         │  
│  ...                    │  
└─────────────────────────┘  
         ▲  ▲  
         │  │  
   New overlay modules  
┌───────────────────────────────┐  
│ neon\_client.py     (asyncpg)  │  
│ vector\_store.py    (mirror)   │  
│ blob\_store.py      (R2)       │  
│ otel\_prom.py  (tracing/metrics)│  
└───────────────────────────────┘  
5 · Bolt Project Scaffold  
bash  
Copy  
Edit  
.bolt/  
 ├── prompt          \# system prompt for Claude 4  
 ├── ignore          \# venv, tests/fixtures  
 └── env.example  
PRD.md               \# \<‑‑ this file  
5.1 .bolt/prompt (auto‑injected)  
“You are Claude 4 Opus/Sonnet extending an existing FastAPI service called EchoVault MCP.  
Follow PRD.md. Touch ≤ 2 files per Build‑Mode run.  
NEVER rewrite modules listed in §3 Forbidden Work.”

6 · Functional Overlay Requirements  
6.1 memory.write additions  
Neon insert via neon\_client.insert\_event.

Dual‑write vector via vector\_store.upsert.

Blob off‑load via blob\_store.save\_if\_large.

OTEL span emitted via otel\_prom.trace\_write.

6.2 memory.query additions  
ANN search order: Qdrant → pgvector → (fallback) existing Chroma.

Blob dereference if payload\_url present.

OTEL span latency\_ms.

7 · Overlay Tasks (Bolt checklist)  
\#	New file / diff	Description  
1	requirements\_overlay.txt	add asyncpg, pgvector, qdrant-client, boto3, otel libs  
2	neon\_client.py	async wrapper \+ Alembic migration  
3	vector\_store.py	mirror pattern with toggle USE\_QDRANT  
4	blob\_store.py	R2 presign helper  
5	modify api/routes.py	wire in overlay calls  
6	otel\_prom.py	tracing \+ metrics  
7	scripts/summarise\_old\_events.py	nightly summariser  
8	tests/ additions	implement T‑1 … T‑4

Bolt must tick tasks sequentially.

8 · User Actions (Bjorn’s to‑do)  
GitHub fork ready ✅ (you have bjorndavidhansen/mcp-memory-service).

Sign up

Neon Free Tier — enable pgvector extension.

Qdrant Cloud Free 1 GB cluster — copy API key & URL.

Cloudflare R2 — create bucket echovault-events; generate access keys.

Populate .bolt/.env with credentials (use env.example).

Import repo into Bolt.new — select Discussion Mode.

Ask Bolt: “Read PRD.md. Output checklist from §7.” and proceed.

9 · ENV Template  
env  
Copy  
Edit  
\# Core  
JWT\_SECRET=dev‑secret  
BLOB\_THRESHOLD=32768

\# Neon  
NEON\_DSN=postgresql://…  
NEON\_POOL\_SIZE=5

\# Qdrant (optional)  
USE\_QDRANT=true  
QDRANT\_URL=https://…  
QDRANT\_API\_KEY=…

\# Cloudflare R2  
R2\_ENDPOINT=https://\<acct\>.r2.cloudflarestorage.com  
R2\_ACCESS\_KEY\_ID=…  
R2\_SECRET\_ACCESS\_KEY=…  
R2\_BUCKET=echovault-events

\# Observability  
OTEL\_EXPORTER\_OTLP\_ENDPOINT=http://jaeger:4317  
PROMETHEUS\_METRICS=true  
✅ Ready for Bolt.new  
Push this PRD.md and .bolt/ scaffold to main, then import the repo into Bolt.new and start with Discussion Task \#7‑checklist.  
Remember: extend, don’t rebuild. Good luck\!  
