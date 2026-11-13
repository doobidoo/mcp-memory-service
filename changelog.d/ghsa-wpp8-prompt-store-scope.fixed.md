- **MCP prompts refuse store-restricted tokens (GHSA-wpp8-2qcw-x4pc).**
  `prompts/get` applied no scope check, so a `read:<store>` token could read
  other stores' memories through `knowledge_export` and `memory_analysis`
  (tag search spans every store) and the `default` store through the other
  prompts. Prompts now require a global `read` scope, like MCP resources.
