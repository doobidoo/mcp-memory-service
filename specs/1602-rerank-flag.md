# 1602: Optional ONNX Reranker — Scoped First PR

## Context

Upstream issue #1602 (doobidoo/mcp-memory-service, author filhocf, OPEN). Scope agreed in issue
comments on 21/out (doobidoo acceptance 20/out 15:44Z; filhocf counter-scope 21/out 00:40Z). This spec
is the G0 artifact for that scope.

Baseline measurement (issue body): paired bootstrap, 120 cases, document-level recall, index
`paraphrase-multilingual-MiniLM-L12-v2`, CPU 16 threads, model `Xenova/bge-reranker-base`
`onnx/model_int8.onnx` (266 MB, ~2.06 GB RSS, 53 ms/pair):

| candidates | recall@5 | recall@10 | recall@30 | p50 |
|------------|---------|-----------|-----------|-----|
| baseline   | 0.4917  | 0.5667    | 0.6750    | 0.19–0.28 s |
| 40         | 0.6417  | 0.6667    | 0.6750    | 2.098 s |
| 100        | 0.7083  | 0.7333    | 0.7417    | 5.314 s |

## Scope (agreed with doobidoo)

1. A `rerank` boolean flag on `memory_search` — per-call opt-in.
2. Env flags: `MCP_RERANK_ENABLED` (default off), `MCP_RERANK_CANDIDATES` (default 40),
   `MCP_RERANK_MODEL`.
3. No-op unless `MCP_RERANK_ENABLED` is set; nothing downloaded or loaded when the flag is
   absent or the env var is off.
4. Model resolved through the existing `quality/onnx_ranker.py` loader — one ONNX path,
   not two.
5. Fail-open: load/inference failure silently degrades to unreranked order, logged once.
6. Not on any default search path; `memory_context` and proactive injection are out of
   scope for this PR (they do not exist yet — doobidoo's correction stands).

## EARS requirements

- WHEN `rerank` is absent or false, THE SYSTEM SHALL return results identical to the
  current golden behavior (byte-identical ordering).
- WHEN `MCP_RERANK_ENABLED` is unset/off, THE SYSTEM SHALL treat `rerank=true` as no-op
  (no model load, no download, no timing change).
- WHEN `rerank=true` and the feature is enabled, THE SYSTEM SHALL rerank the top
  `MCP_RERANK_CANDIDATES` (default 40) results using the model named in `MCP_RERANK_MODEL`,
  resolved through the existing loader.
- WHEN model load or inference raises, THE SYSTEM SHALL return the unreranked order
  and log once per process lifetime, never return an error response.
- THE SYSTEM SHALL resolve the model via the existing `quality/onnx_ranker.py`
  loader.
- WHILE concurrent `memory_search` calls run, THE SYSTEM SHALL remain thread-safe
  (no shared mutable state; safe alongside `quality_boost`).
- WHEN `rerank=true` is exercised, THE SYSTEM SHALL add no dependency beyond
  `onnxruntime`/`tokenizers` (no torch at runtime).

## Acceptance criteria (issue #1602, mapped)

| AC | requirement | result |
|----|-------------|--------|
| 1  | Off by default; identical behavior when off | GREEN — flag absent/false = passthrough by identity; env off = no-op (loader never called) |
| 2  | 40 candidates → +≥8 p.p. recall@10 on eval set | measured on the 120-case ruler, document-level: **bge int8 +10.0 p.p.**, IC95 [5.0 · 15.8], 12 cases better, 0 worse; **ms-marco int8 −8.3 p.p. — refuted** |
| 3  | p50 ≤ 2.5 s/query on CPU | measured: bge int8 p50 2.10 s @40c (52.5 ms/pair); ms-marco 476 ms but regresses |
| 4  | No torch dependency added | GREEN — pyproject untouched |
| 5  | Load failure degrades to baseline, never to error | GREEN — fail-open probe: empty cache → one warning, baseline order |

## ms-marco zero-new-model baseline (doobidoo's request) — REFUTED

Measured 21/out on the 120-case ruler, 40 candidates, document-level
(`eval/lab_rerank_ms_marco_40c.json`, artifact versioned in the bench repo):

| metric | base | ms-marco int8 | delta |
|--------|------|---------------|-------|
| recall@5 | 0.4917 | 0.4083 | −8.3 p.p. |
| recall@10 | 0.5667 | 0.4833 | −8.3 p.p. |
| recall@30 | 0.6750 | 0.6583 | −1.7 p.p. |
| p50 | ~0.2 s | 476 ms | fine |

The existing registered cross-encoder is English-only and reorders Portuguese
queries worse than the embedding index. Conclusion: the only model that delivers
the measured gain on this corpus is `Xenova/bge-reranker-base` int8 — and it must be
registered in `SUPPORTED_MODELS` (`quality/config.py`) for `MCP_RERANK_MODEL` to
resolve. Without that registration, the flag fails open forever and the feature
cannot produce its measured gain. Registration of the bge-int8 entry is therefore
blocking for this PR (config data only — no new inference path).

## Test surface

- Flag off: existing golden snapshot tests must pass unchanged.
- Flag on without env: no-op (identical ordering, no model instantiation).
- Flag on with env + model missing: fail-open (baseline order, one log line).
- Flag on with env + model present: reranked order, `MCP_RERANK_CANDIDATES` honored.
- Concurrency: parallel calls with/without flag, no state leak.

## Publication gate

Branch `feat/1602-rerank-flag`. No push, no PR, no upstream comment without explicit
acceptance of the final text.
