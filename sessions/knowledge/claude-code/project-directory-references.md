# $CLAUDE_PROJECT_DIR Environment Variable

## Overview
`$CLAUDE_PROJECT_DIR` is an environment variable set by Claude Code at runtime that corresponds to the directory from which the `claude` command was invoked to start the Claude Code session.

## Key Behaviors
- **Set at startup**: Claude Code sets this variable to the directory where `claude` was run
- **Available in runtime**: The variable is available during Claude Code's internal execution (hooks, statusline, etc.)
- **Not in interactive shells**: The variable is NOT available in interactive Bash tool shells
- **Critical for hooks**: All hook invocations depend on this variable being set correctly

## Usage in Sessions System
The sessions system heavily relies on `$CLAUDE_PROJECT_DIR`:
- All hooks are invoked as `$CLAUDE_PROJECT_DIR/.claude/hooks/[hook-name].py`
- The statusline script is invoked as `$CLAUDE_PROJECT_DIR/.claude/statusline-script.sh`
- Without this variable set correctly, the entire sessions system would fail

## Important Implications
- **Project root assumption**: The sessions system assumes Claude Code is always started from the project root
- **Subdirectory starts break assumptions**: Starting `claude` from a subdirectory will cause state files to be looked for in the wrong location
- **Fix for statusline**: Using `$CLAUDE_PROJECT_DIR` instead of `$cwd` in statusline script ensures correct state file locations regardless of current working directory

## Testing Observations
- When `claude` is run from project root, statusline correctly shows state even when navigating to subdirectories
- When `claude` is run from a subdirectory (e.g., cc-sessions/), statusline looks for state files relative to that subdirectory and fails to find them