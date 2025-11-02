---
name: Evidence & Storage Conventions
description: Declare artifact locations and maintain evidence checklists
---

# Evidence & Storage Conventions

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
## Evidence Requirements

- Wishes must declare where artefacts live; there is no default `qa/` directory. Capture metrics inline in the wish (e.g., tables under a **Metrics** section) or in clearly named companion files.
- Every wish must complete the **Evidence Checklist** block before implementation begins, spelling out validation commands, artefact locations, and approval checkpoints.
- External tracker IDs live in the wish markdown (for example a **Tracking** section with `Forge task: FORGE-123`).
- Background agent outputs are summarised in the wish context ledger; raw logs can be viewed with `mcp__genie__view` with sessionId parameter.

## Testing & Evaluation

- Evaluation tooling is optional. If a project adds its own evaluator agent, the review or plan workflow can reference it; otherwise, evaluation steps default to manual validation.
- Typical metrics: `{{METRICS}}` such as latency or quality. Domain-specific metrics should be added per project in the wish/forge plan.
- Validation hooks should be captured in wishes/forge plans (e.g., `pnpm test`, `cargo test`, metrics scripts).
