**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: install
description: Code Collective install workflow — prepare product docs and wire up Code agents

---

# Code Install Workflow

Purpose: Initialize the project’s product docs and connect Code collective agents and workflows.

## Phases

1) Discovery
- Detect repository state (fresh vs existing codebase)
- Identify domain, constraints, and intended tech stack
- Choose path: Analyze Existing • New Repo Interview • Hybrid

2) Implementation
- Create/update product docs:
  - `@.genie/product/mission.md`
  - `@.genie/product/mission-lite.md`
  - `@.genie/product/tech-stack.md`
  - `@.genie/product/roadmap.md`
  - `@.genie/product/environment.md`
- Calibrate Code agents by adding a short "Project Notes" section inside relevant `.genie/code/agents/*` or `.genie/skills/*` docs (no `custom/` folder)
- Initialize `.genie/CONTEXT.md` and add `.genie/CONTEXT.md` to `.gitignore`
- Keep edits under `.genie/` (no app code changes here)

3) Verification
- Validate cross-references and required sections in product docs
- Exercise MCP tools: `mcp__genie__list_agents` and a sample Code agent invocation
- Capture a Done Report and hand off to `code/wish` for the first scoped feature

## Context Auto-Loading
@.genie/product/mission.md
@.genie/product/tech-stack.md
@.genie/product/environment.md
@.genie/product/roadmap.md
@README.md
@package.json

## Modes

Mode 1: Codebase Analysis
- Map structure, languages/frameworks, dependencies
- Identify architecture patterns and external integrations
- Summarize implementation progress and testing approach

Mode 2: New Repository Interview
Use a concise Q&A to capture missing product identity and technical intent.

Mode 3: Hybrid
Analyze what exists, interview for the rest, reconcile discrepancies.

Mode 4: Bootstrap Guardrails (No Code Changes)
- Only write to `.genie/`
- Defer app scaffolding to a `code/wish` → `code/forge` cycle

## Outputs

Product docs populated with stable headings so downstream tools can parse consistently. Example sections:
- mission: Pitch, Users, Problem, Key Features
- tech-stack: Core Technologies, Architecture, Dependencies, Infrastructure
- environment: Required/Optional vars + Setup instructions
- roadmap: Phase 0 (completed), Phase 1 goals and measurable criteria

User context file:
- `.genie/CONTEXT.md` created and git-ignored

Done Report:
- `.genie/wishes/<slug>/reports/done-install-code-<timestamp>.md`

## Success Criteria
- Product docs complete and coherent
- Context file present and ignored
- Code agents discoverable via MCP/CLI
- Clear next step: `code/wish` → `code/forge` → `code/review`

## Safety
- Do not modify application code
- Keep changes minimal, targeted, and reviewable

