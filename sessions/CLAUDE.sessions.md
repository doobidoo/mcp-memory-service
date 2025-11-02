# CLAUDE.sessions.md

This file provides guidance for working in a repository with cc-sessions installed.

## Collaboration Philosophy

**Core Principles**:
- **Investigate patterns** - Look for existing examples, understand established conventions, don't reinvent what already exists
- **Confirm approach** - Explain your reasoning, show what you found in the codebase, get consensus before proceeding
- **State your case if you disagree** - Present multiple viewpoints when architectural decisions have trade-offs
- When working on highly standardized tasks: Provide SOTA (State of the Art) best practices
- When working on paradigm-breaking approaches: Generate "opinion" through rigorous deductive reasoning from available evidence

## Code Philosophy

### Locality of Behavior
- Keep related code close together rather than over-abstracting
- Code that relates to a process should be near that process
- Functions that serve as interfaces to data structures should live with those structures

### Solve Today's Problems
- Deal with local problems that exist today
- Avoid excessive abstraction for hypothetical future problems

### Minimal Abstraction
- Prefer simple function calls over complex inheritance hierarchies
- Just calling a function is cleaner than complex inheritance scenarios

### Readability > Cleverness
- Code should be obvious and easy to follow
- Same structure in every file reduces cognitive load

## Using Context-Gathering Agent

When you need to answer a question that would require reading many files (polluting your context window), use the **context-gathering** agent instead.

**During task work**: If the answer needs to persist throughout the task duration, the agent can write to the task file's context manifest.

**For one-off questions**: Prompt the agent to respond directly:
```
"Research how authentication works in this codebase and respond with a summary of the flow"
```

The agent operates in its own context window and can read extensively without affecting your token budget.

## DAIC Mode System

This repository uses Discussion-Alignment-Implementation-Check (DAIC):

- **Discussion Mode** (default): Edit/Write/MultiEdit tools are blocked. Focus on discussing approach.
- **Implementation Mode**: Tools are available. Execute approved todos only.

**Only the user can activate implementation mode** using their configured trigger phrases.

When implementation is complete, return to discussion mode:
```bash
sessions mode discussion
```

## Workflow Protocols

The system has automated protocols for task creation, startup, completion, and context compaction. When loaded, follow the protocol instructions. The user activates these with trigger phrases - you don't need to manage this.
