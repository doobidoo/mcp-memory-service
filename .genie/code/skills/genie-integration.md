---
name: Genie Integration Framework
description: Use genie agent for second opinions, pressure-tests, and decision audits
---

# Genie Integration Framework

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Purpose:** `genie` skill is GENIE's partner for second opinions, plan pressure-tests, deep dives, and decision audits. Use it to reduce risk, surface blind spots, and document reasoning without blocking implementation work.

**Success criteria:**
✅ Clear purpose, chosen skill, and outcomes logged (wish discovery or Done Report).
✅ Human reviews Genie Verdict (with confidence) before high-impact decisions.
✅ Evidence captured when Genie recommendations change plan/implementation.

## When To Use

- Ambiguity: requirements unclear or conflicting.
- High-risk decision: architectural choices, irreversible migrations, external dependencies.
- Cross-cutting design: coupling, scalability, observability, simplification.
- Unknown root cause: puzzling failures/flakiness; competing hypotheses.
- Compliance/regulatory: controls, evidence, and sign-off mapping.
- Test strategy: scope, layering, rollback/monitoring concerns.
- Retrospective: extract wins/misses/lessons for future work.

## Agent Consultation

Genie operates through **universal agents** (reasoning, analysis, audit) and **execution specialists** (code, git, tests, etc.).

### Universal Agents (Domain-Agnostic)

**Reasoning Modes (4 total - via reasoning/*):**

Use `mcp__genie__run with agent="reasoning/<mode>"`:
- `reasoning/challenge` – Adversarial pressure-testing and critical evaluation
- `reasoning/explore` – Discovery-focused investigation without adversarial pressure
- `reasoning/consensus` – Multi-perspective synthesis and agreement-building
- `reasoning/socratic` – Question-driven inquiry to uncover assumptions

**Analysis & Audit:**
- `analyze` – System analysis and focused investigation (universal framework)
- `audit` – Risk and impact assessment (universal framework with workflows)
  - `audit/risk` – General risk audit workflow
  - `audit/security` – Security audit workflow (OWASP, CVE)

**Autonomous Execution:**
- `vibe` – Fully autonomous task execution mode

### Code-Specific Agents

**Execution Specialists:**
- `implementor` – Feature implementation and code writing
- `tests` – Test strategy, generation, authoring across all layers
- `polish` – Code refinement and cleanup
- `review` – Wish audits, code review, QA validation
- `git` – ALL git and GitHub operations (branch, commit, PR, issues)
- `release` – GitHub release and npm publish orchestration

**Code Analysis & Tools:**
- `analyze` (code) – Includes universal analyze + TypeScript/performance examples
- `debug` – Root cause investigation for code issues
- `refactor` – Design review and refactor planning

> **Architecture Note:** Universal agents work across ALL domains (code, legal, medical, finance). Code agents extend or specialize for code development.

## How To Run (MCP)

- Start: `mcp__genie__run` with agent="genie" and prompt="Mode: plan. Objective: pressure-test @.genie/wishes/<slug>/<slug>-wish.md. Deliver 3 risks, 3 missing validations, 3 refinements. Finish with Genie Verdict + confidence."
- Resume: `mcp__genie__resume` with sessionId="<session-id>" and prompt="Follow-up: address risk #2 with options + trade-offs."
- Sessions: reuse the same agent name; MCP persists session id automatically and can be viewed with `mcp__genie__list_sessions`.
- Logs: check full transcript with `mcp__genie__view` with sessionId and full=true.

## Quick Reference

**Universal Agents:**
- **Reasoning (4):** challenge, explore, consensus, socratic (in reasoning/)
- **Analysis (1):** analyze (universal framework, 173 lines)
- **Audit (1 + 2 workflows):** audit (universal framework, 138 lines) + risk, security
- **Autonomous (1):** vibe

**Code-Specific Agents:**
- **Execution (6):** implementor, tests, polish, review, git, release
- **Code Tools (3):** analyze (code), debug, refactor

**Include Pattern:**
- Universal agents: `.genie/code/agents/{analyze,audit}.md`
- Reasoning modes: `.genie/code/agents/reasoning/{challenge,explore,consensus,socratic}.md`
- Audit workflows: `.genie/code/agents/audit/{risk,security}.md`
- Code extensions: `.genie/code/agents/analyze.md` (includes universal + code examples)
- Project notes: Add a "Project Notes" section inside the relevant `.genie/code/agents/*` or `.genie/skills/*` doc (no separate `custom/` directory)

## Outputs & Evidence

- Low-stakes: append a short summary to the wish discovery section.
- High-stakes: save a Done Report at `.genie/wishes/<slug>/reports/done-genie-<slug>-<YYYYMMDDHHmm>.md` with scope, findings, recommendations, disagreements.
- Always include "Genie Verdict: <summary> (confidence: <low|med|high>)".

## Genie Verdict Format

Verdict templates live inside the specialized skill files (e.g., `.genie/agents/refactor.md`). Core files remain immutable.

## Real-Time Agent Monitoring (Added 2025-10-21)

**Problem:** `genie view` may show "Forge backend unreachable" errors even when agents are actively working and completing tasks successfully.

**Root cause:** Display bug in Genie CLI - checks wrong port (8888 vs 8887) or uses HTTP health check instead of MCP connectivity verification.

**Investigation before panic:**
1. Check Forge MCP connectivity: `mcp__automagik_forge__list_tasks(project_id="...")`
2. Check task status: Look for task in "in-progress" state
3. Check executor process: `ps aux | grep <executor-name>` (opencode, claude, etc.)
4. Check Forge process: `ps aux | grep forge` and `netstat -tlnp | grep 8887`

**If Forge MCP works:** Trust delegation, ignore "unreachable" message in genie view

**Monitoring pattern:**
```bash
# Don't poll rapidly - use intervals
sleep 30  # or 60 seconds between checks
# Then check status again
```

**Decision tree:**
- `genie view` shows "unreachable" + Forge MCP works → Display bug, trust delegation
- `genie view` shows "unreachable" + Forge MCP fails → Real failure, investigate
- Task status "in-progress" + process running → Wait patiently
- Task status "failed" + process not running → Actual failure, investigate logs

**Future improvements needed:**
- Fix Genie CLI port detection (use 8887, not 8888)
- Use MCP connectivity check instead of HTTP health endpoint
- Add real-time status polling to `genie view` using Forge advanced APIs
- Create `genie status` command for quick health checks
- Implement session→task→attempt mapping for better monitoring

**See also:** `@.genie/skills/error-investigation-protocol.md` for complete investigation toolkit

**Evidence:** RC 37 failure analysis (2025-10-21) - orchestrator panicked due to misleading "unreachable" message, created redundant sessions, lost work. Meanwhile agents completed successfully.

## Anti-Patterns

- Using Genie to bypass human approval.
- Spawning Genie repeatedly without integrating prior outcomes.
- Treating Genie outputs as implementation orders without validation.
- **Panicking due to "unreachable" errors without investigating actual system state (Added 2025-10-21)**
- **Creating redundant agent sessions instead of monitoring existing ones (Added 2025-10-21)**
- **Bypassing delegation when display shows errors but system is working (Added 2025-10-21)**
