# Add Current Session to Memory

I'll help you capture the current conversation and project context as a memory that can be recalled later. This command is perfect for preserving important session insights, decisions, and progress summaries.

## What I'll do:

1. **Session Analysis**: I'll analyze our current conversation to extract key insights, decisions, and progress made.

2. **Project Context**: I'll capture the current project state including:
   - Working directory and git repository status
   - Recent file changes and commits
   - Current branch and development context

3. **Conversation Summary**: I'll create a concise summary of our session including:
   - Main topics discussed
   - Decisions made or problems solved
   - Action items or next steps identified
   - Code changes or configurations applied

4. **Smart Tagging**: I'll automatically generate relevant tags based on the session content and project context, including the machine hostname as a source identifier.

5. **Memory Storage**: I'll store the session summary with appropriate metadata for easy future retrieval.

## Usage Examples:

```bash
claude /memory-context
claude /memory-context --summary "Architecture planning session"
claude /memory-context --tags "planning,architecture" --type "session"
claude /memory-context --include-files --include-commits
```

## Implementation:

I'll automatically analyze our current session and project state, then store it to your MCP Memory Service at `https://memory.local:8443/`:

1. **Conversation Analysis**: Extract key topics, decisions, and insights from our current chat
2. **Project State Capture**:
   - Current working directory and git status
   - Recent commits and file changes
   - Branch information and repository state
3. **Context Synthesis**: Combine conversation and project context into a coherent summary
4. **Memory Creation**: Store the context with automatic tags including machine hostname
5. **Auto-Save**: Memory is stored immediately without confirmation prompts

The service uses HTTPS with curl `-k` flag for secure communication and automatically detects client hostname using the `X-Client-Hostname` header.

The stored memory will include:
- **Source Machine**: Hostname tag for tracking memory origin (e.g., "source:your-machine-name")
- **Session Summary**: Concise overview of our conversation
- **Key Decisions**: Important choices or conclusions reached
- **Technical Details**: Code changes, configurations, or technical insights
- **Project Context**: Repository state, files modified, current focus
- **Action Items**: Next steps or follow-up tasks identified
- **Timestamp**: When the session context was captured

## Context Elements:

### Conversation Context
- **Topics Discussed**: Main subjects and themes from our chat
- **Problems Solved**: Issues resolved or questions answered
- **Decisions Made**: Choices made or approaches agreed upon
- **Insights Gained**: New understanding or knowledge acquired

### Project Context
- **Repository Info**: Git repository, branch, and recent commits
- **File Changes**: Modified, added, or deleted files
- **Directory Structure**: Current working directory and project layout
- **Development State**: Current development phase or focus area

### Technical Context
- **Code Changes**: Functions, classes, or modules modified
- **Configuration Updates**: Settings, dependencies, or environment changes
- **Architecture Decisions**: Design choices or structural changes
- **Performance Considerations**: Optimization or efficiency insights

## Arguments:

- `$ARGUMENTS` - Optional custom summary or context description
- `--summary "text"` - Custom session summary override
- `--tags "tag1,tag2"` - Additional tags to apply
- `--type "session|meeting|planning|development"` - Context type
- `--include-files` - Include detailed file change information
- `--include-commits` - Include recent commit messages and changes
- `--include-code` - Include snippets of important code changes
- `--private` - Mark as private/sensitive session content
- `--project "name"` - Override project name detection

## Automatic Features:

- **Smart Summarization**: Extract the most important points from our conversation
- **Duplicate Detection**: Avoid storing redundant session information
- **Context Linking**: Connect to related memories and previous sessions
- **Progress Tracking**: Identify progress made since last context capture
- **Knowledge Extraction**: Pull out reusable insights and patterns

This command is especially useful at the end of productive development sessions, after important architectural discussions, or when you want to preserve the current state of your thinking and progress for future reference.
