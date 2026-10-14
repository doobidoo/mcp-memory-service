- **Graph-proximity signal in context injection (opt-in).** The injection re-rank
  (`_apply_rerank_a`) now accepts an additive `graph_proximity` term: beliefs whose source
  memories share entities with the query's entities are surfaced on their theme relevance,
  before the utility term can amplify generic high-reuse beliefs (rich-get-richer measured
  14/out). Opt-in via `MCP_INJECT_GRAPH_PROXIMITY` + weight `MCP_INJECT_RERANK_A_W_GRAPH`
  (default 0.0 → score byte-identical to R4 when off). Reuses the existing graph read
  primitives (`find_memories_by_entity`, `EntityExtractor`); no DDL, no write to the graph;
  degrades to neutral (0) when no entities resolve or no graph storage is available. Arc
  40e6ecfb; see docs/rfc/planned/rfc-learning-loop-graph-proximity.md.
