# Claude Code Commands

Custom slash commands for mcp-memory-service development.

## Available Commands

### `/release`

Full release cycle automation: list open PRs, merge approved, verify CI, check landing page, bump version, create tag + GitHub release, clean branches, save summary to MCP Memory.

### `/triage`

Security alert triage & remediation: scan Dependabot/CodeQL/secret scanning alerts, classify severity + complexity, create fix branches, implement fixes, open PRs, save patterns to MCP Memory.

### `/merge-sweep`

Merge round for open PRs: check CI per head SHA, open review threads, conflicts and Dependabot bounds, classify each PR (merge / quick-fix / decision / skip), present one table, and merge one by one only after approval.

### `/refactor-function` (PoC)

Automated function complexity reduction using multi-agent workflow.

**Status**: Proof of Concept (simulated agents)

**Usage**:
```bash
# Quick test
/refactor-function --dry-run

# Interactive (would work with real agents)
# 1. Select function in editor
# 2. Run: /refactor-function
# 3. Review changes
# 4. Apply if approved
```

**Workflow**:
```
Select Function → Baseline Analysis → Refactor → Validate → Review → Apply
```

**Current Implementation** (PoC):
- ✅ CLI interface working
- ✅ Workflow orchestration
- ✅ User confirmation flow
- ⚠️ Simulated agent responses (placeholder)
- ❌ Not yet integrated with real agents (amp-bridge, code-quality-guard)

**Next Steps for Production**:
1. Integrate with Task agent calls
2. Read actual file content via Read tool
3. Apply changes via Edit tool
4. Create commits via Bash tool
5. Add Claude Code settings.json hook

**Proven Track Record**:
- Based on Issue #340 workflow (45.5% complexity reduction)
- 2-3x faster than manual refactoring
- 100% validation success rate

## Installation

Commands are located in `.claude/commands/` and executable:

```bash
# Make executable (already done)
chmod +x .claude/commands/refactor-function

# Test PoC
.claude/commands/refactor-function --dry-run

# With options
.claude/commands/refactor-function --target-complexity 6
```

## Configuration (Future)

Add to Claude Code `settings.json`:

```json
{
  "commands": {
    "refactor-function": {
      "path": ".claude/commands/refactor-function",
      "description": "Reduce function complexity automatically",
      "requiresSelection": true,
      "agents": ["amp-bridge", "code-quality-guard"]
    }
  }
}
```

## Development

### Command Structure
```
.claude/commands/
├── README.md                    # This file
├── refactor-function            # Executable Python script (PoC)
└── refactor-function.md         # Full specification
```

### Adding New Commands

1. Create executable script in `.claude/commands/<name>`
2. Add specification in `.claude/commands/<name>.md`
3. Update this README
4. Test with `--dry-run` flag
5. Document in commit message

### Testing

```bash
# Unit test individual functions
python3 -c "from refactor-function import RefactorCommand; ..."

# Integration test full workflow
.claude/commands/refactor-function --dry-run

# With real selection (future)
echo '{"file": "...", "function": "..."}' | .claude/commands/refactor-function
```

## Related Documentation

- **Command Specification**: `refactor-function.md` - Full design doc
- **Issue #340**: Original workflow that inspired this command
- **Agent Docs**: `.claude/directives/agents.md` - Agent usage guide

## Future Commands (Ideas)

- `/complexity-report` - Analyze file/module complexity
- `/extract-method` - Manual Extract Method refactoring
- `/add-tests` - Generate tests for function
- `/simplify-conditionals` - Reduce boolean complexity
- `/remove-duplication` - DRY principle enforcement

## Contributing

New command ideas? Open an issue or PR with:
1. Use case description
2. Example workflow
3. Expected input/output
4. Agent requirements

## Credits

Inspired by proven workflow from Issue #340:
- 3 functions refactored in 2 hours
- 45.5% total complexity reduction
- Multi-agent orchestration (amp-bridge + code-quality-guard)
