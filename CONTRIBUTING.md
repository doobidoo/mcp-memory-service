# Contributing to MCP Memory Service

Thank you for your interest in contributing to MCP Memory Service! 🎉

This project provides semantic memory and persistent storage for AI assistants through the Model Context Protocol. We welcome contributions of all kinds - from bug fixes and features to documentation and testing.

## Where development happens

All development happens on **GitHub**: https://github.com/doobidoo/mcp-memory-service

That is where issues, pull requests, CI, and releases live. Open your issue or PR there.

From June to September 2026 the project was hosted on Codeberg. That repository at
https://codeberg.org/doobidoo/mcp-memory-service stays readable so older links, issue
numbers, and pull request references still resolve, but it is an archive: it receives
no further pushes and runs no CI.

Vulnerabilities are the one thing that does not belong in a public issue - see
[Handling a security vulnerability](#handling-a-security-vulnerability).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Security-Sensitive Changes](#security-sensitive-changes)
- [Autonomous Agents & AI-Generated PRs](#autonomous-agents--ai-generated-prs)
- [Reporting Issues](#reporting-issues)
- [Community & Support](#community--support)
- [Recognition](#recognition)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive criticism and collaborative problem-solving
- Respect differing viewpoints and experiences
- Avoid harassment, discrimination, or inappropriate behavior

## Ways to Contribute

### 🐛 Bug Reports
Help us identify and fix issues by reporting bugs with detailed information.

### ✨ Feature Requests
Suggest new features or improvements to existing functionality.

### 📝 Documentation
Improve README, Wiki pages, code comments, or API documentation.

### 🧪 Testing
Write tests, improve test coverage, or help with manual testing.

### 💻 Code Contributions
Fix bugs, implement features, or improve performance.

### 🌍 Translations
Help make the project accessible to more users (future goal).

### 💬 Community Support
Answer questions in Issues or help other users.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Platform-specific requirements:
  - **macOS**: Homebrew Python recommended for SQLite extension support
  - **Windows**: Visual Studio Build Tools for some dependencies
  - **Linux**: Build essentials package

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcp-memory-service.git
   cd mcp-memory-service
   ```

3. **Install dependencies**:
   ```bash
   python install.py
   ```
   This will automatically detect your platform and install appropriate dependencies.

4. **Verify installation**:
   ```bash
   python scripts/verify_environment.py
   ```

5. **Run the service**:
   ```bash
   # HTTP server (background, dashboard + REST API, recommended)
   memory launch

   # OR MCP stdio server (for Claude Desktop integration)
   memory server
   ```

6. **Test with MCP Inspector** (optional, stdio child process):
   ```bash
   npx @modelcontextprotocol/inspector memory server
   ```

### Alternative: Docker Setup

For a containerized environment:
```bash
docker-compose up -d  # For MCP mode
docker-compose -f docker-compose.http.yml up -d  # For HTTP API mode
```

## Development Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Use descriptive branch names:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for test improvements
- `refactor/` for code refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add/update tests as needed
- Update documentation if applicable
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_server.py

# Run with coverage
pytest --cov=mcp_memory_service tests/
```

### 4. Commit Your Changes

Use semantic commit messages:
```bash
git commit -m "feat: add memory export functionality"
git commit -m "fix: resolve timezone handling in memory search"
git commit -m "docs: update installation guide for Windows"
git commit -m "test: add coverage for storage backends"
```

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 5. Push to Your Fork

```bash
git push origin your-branch-name
```

### 6. Create a Pull Request

Open a PR from your fork to the main repository with:
- Clear title describing the change
- Description of what and why
- Reference to any related issues
- Screenshots/examples if applicable

## Coding Standards

### Python Style Guide

- Follow PEP 8 with these modifications:
  - Line length: 88 characters (Black formatter default)
  - Use double quotes for strings
- Use type hints for all function signatures
- Write descriptive variable and function names
- Add docstrings to all public functions/classes (Google style)

### Code Organization

```python
# Import order
import standard_library
import third_party_libraries
from mcp_memory_service import local_modules

# Type hints
from typing import Optional, List, Dict, Any

# Async functions
async def process_memory(content: str) -> Dict[str, Any]:
    """Process and store memory content.

    Args:
        content: The memory content to process

    Returns:
        Dictionary containing memory metadata
    """
    # Implementation
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Never silently fail

```python
try:
    result = await storage.store(memory)
except StorageError as e:
    logger.error(f"Failed to store memory: {e}")
    raise MemoryServiceError(f"Storage operation failed: {e}") from e
```

## Testing Requirements

### Writing Tests

- Place tests in `tests/` directory (it mirrors `src/`), name the file `test_*.py`
- A change under `src/` comes with a test that is **red without the change**. CI
  checks this (`tests-prove-fix` runs your added or changed tests against the base
  branch's `src/` and fails if they pass there). Run both sides yourself and paste
  them into the PR:
  ```bash
  git stash -- src/ && .venv/bin/pytest tests/<your file> -q; git stash pop   # must fail
  .venv/bin/pytest tests/<your file> -q                                        # must pass
  ```
- Drive the real code path. Use the sqlite-vec storage through the `temp_db_path`
  fixture, the real tool registry, the real handler. Mock only network and
  external services (Cloudflare, Groq, Gemini). A mock of the function under test
  proves nothing about it, and a mock that mirrors the bug is green on both sides.
- Pure removals need no test; the check recognises a deletion-only diff.
- Include both positive and negative cases, and the error path

Example test, run against the real storage (the fixtures come from `tests/conftest.py`):
```python
import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_store_then_retrieve(temp_db_path, unique_content):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/test.db")
    await storage.initialize()
    try:
        content = unique_content("memory about authentication")
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["auth"],
        )
        success, message = await storage.store(memory)
        assert success, message

        results = await storage.retrieve("authentication", n_results=1)
        assert results[0].memory.content == content
    finally:
        await storage.close()
```

### Test Coverage

- Focus on the behavior the change adds or fixes, then its edge cases
- Test error handling scenarios
- `tests/integration` starts the HTTP app; `tests/consolidation` runs the
  maintenance pipeline. Both run in CI, so a test there is not optional coverage

## Documentation

### Code Documentation

- Add docstrings to all public APIs
- Include type hints
- Provide usage examples in docstrings
- Keep comments concise and relevant

### Project Documentation

When adding features or making significant changes:

1. Update README.md if needed
2. Add/update Wiki pages for detailed guides
3. Update CHANGELOG.md following Keep a Changelog format
4. Update AGENTS.md or CLAUDE.md if development workflow changes

**Advanced Workflow Automation**:
- See [Context Provider Workflow Automation](https://github.com/doobidoo/mcp-memory-service/wiki/Context-Provider-Workflow-Automation) for automating development workflows with intelligent patterns

### API Documentation

- Document new MCP tools in `docs/api/tools.md`
- Include parameter descriptions and examples
- Note any breaking changes

## Submitting Changes

### Pull Request Guidelines

1. **PR Title**: Use semantic format (e.g., "feat: add batch memory operations")

2. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation
   Why these changes are needed

   ## Changes
   - List of specific changes
   - Breaking changes (if any)

   ## Testing
   - How you tested the changes
   - Test coverage added

   ## Screenshots
   (if applicable)

   ## Related Issues
   Fixes #123
   ```

3. **PR Checklist**:
   - [ ] Tests pass locally (`bash scripts/pr/pre_pr_check.sh`)
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] No sensitive data exposed
   - [ ] **If adding/modifying MCP tools**: tool is added to `valid_actions` in `server/handlers/graph.py` (if applicable) and write-scope enforcement is correct (see [Security-Sensitive Changes](#security-sensitive-changes))
   - [ ] **I am a human contributor**, or I have disclosed that this PR was generated by an automated agent (see [Autonomous Agents & AI-Generated PRs](#autonomous-agents--ai-generated-prs))

### Review Process

- PRs require at least one review
- Address review feedback promptly
- Keep discussions focused and constructive
- Be patient - reviews may take a few days

## Security-Sensitive Changes

The following areas require extra care. PRs touching them receive additional scrutiny and may require a second maintainer review.

### Protected paths

| Path | Risk |
|------|------|
| `src/mcp_memory_service/web/api/mcp.py` | HTTP MCP transport, OAuth scope enforcement |
| `src/mcp_memory_service/server_impl.py` | MCP tool registration, write-scope gate |
| `src/mcp_memory_service/web/oauth/` | OAuth 2.1 implementation |
| `src/mcp_memory_service/storage/` | Data integrity, production database |
| `.github/workflows/` | CI/CD pipeline, supply-chain security |

### Required for any PR that adds or modifies an MCP tool

1. **Scope enforcement**: State explicitly whether the new tool is read-only or write. Read-only tools must set `readOnlyHint=True` in annotations. Write tools must be reachable only with a write-scoped token.
2. **`valid_actions` registration**: If the tool is routed through `handle_memory_graph()`, it must be added to the `valid_actions` list at the top of that function — otherwise the new branches are dead code.
3. **Security section in PR description**: Add a `## Security` section explaining (a) what data the tool reads/writes, (b) how scope is enforced, and (c) any edge cases (None inputs, malformed arguments, concurrent access).

### Handling a security vulnerability

Do **not** open a public issue. Report it through GitHub's private [Security Advisories](https://github.com/doobidoo/mcp-memory-service/security/advisories/new).

Issues and pull requests are public from the moment you open them, which is fine for
everything except an unfixed vulnerability. Security Advisories are the confidential
channel for that: the report stays private until a fix ships, and you are credited in
the advisory. See [SECURITY.md](SECURITY.md) for what to include.

## Autonomous Agents & AI-Generated PRs

We welcome contributions from AI coding assistants and autonomous agents. However, transparency is required.

For the project's own position — authorship, copyright, and the safeguards every
change passes before it reaches `main` — see [AUTHORSHIP.md](AUTHORSHIP.md).

### Disclosure requirement

If a PR was generated, drafted, or submitted by an automated agent (an AI system acting without direct human review of each change), the PR description **must** include:

```
## Agent Disclosure
This PR was generated by [agent name/system]. The submitting account is operated by [human name / organization].
Human review of this PR: [yes — describe what was reviewed / no — fully automated]
```

PRs that appear to be automated but carry no disclosure may be closed pending clarification.

### Why this matters

- Automated agents can miss context-specific security constraints (scope gates, invariants, production safeguards) that are not visible from the code alone.
- Disclosure lets maintainers calibrate review depth accordingly.
- It is not a barrier to contribution — filhocf's fully automated RFC #732 series is a model example of high-quality agent contributions. Disclosure + quality work = fast merge.

### What happens without disclosure

1. Maintainer asks for clarification in a comment.
2. If no response within 7 days, PR is closed. Re-opening with disclosure is welcome.

## Reporting Issues

### Language

All issues, pull requests, and discussions should be written in **English**. This keeps the project accessible to our international community of contributors and ensures problems are searchable for others hitting the same issue. If English isn't your first language, tools like DeepL or Google Translate work well — we genuinely appreciate the effort.

Contributions opened in other languages may be translated by a maintainer, or redirected with a polite request to resubmit in English.

### Bug Reports

When reporting bugs, include:

1. **Environment**:
   - OS and version
   - Python version
   - MCP Memory Service version
   - Installation method (pip, Docker, source)

2. **Steps to Reproduce**:
   - Minimal code example
   - Exact commands run
   - Configuration used

3. **Expected vs Actual Behavior**:
   - What you expected to happen
   - What actually happened
   - Error messages/stack traces

4. **Additional Context**:
   - Screenshots if applicable
   - Relevant log output
   - Related issues

### Feature Requests

For feature requests, describe:

- The problem you're trying to solve
- Your proposed solution
- Alternative approaches considered
- Potential impact on existing functionality

## Community & Support

### Getting Help

- **Documentation**: Check the [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki) first
- **Issues**: Search existing [issues](https://github.com/doobidoo/mcp-memory-service/issues?q=is%3Aissue) before creating new ones
- **Questions**: [Discussions](https://github.com/doobidoo/mcp-memory-service/discussions) for open-ended questions, [issues](https://github.com/doobidoo/mcp-memory-service/issues) for anything with a reproduction
- **Response Time**: Maintainers typically respond within 2-3 days

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, and general questions
- **GitHub Pull Requests**: Code contributions and reviews
- **GitHub Security Advisories**: Confidential vulnerability reports only

### For AI Agents

If you're an AI coding assistant, also check:
- [AGENTS.md](AGENTS.md) - Generic AI agent instructions
- [CLAUDE.md](CLAUDE.md) - Claude-specific guidelines
- [Context Provider Workflow Automation](https://github.com/doobidoo/mcp-memory-service/wiki/Context-Provider-Workflow-Automation) - Automate development workflows with intelligent patterns

## Recognition

We value all contributions! Contributors are:

- Listed in release notes for their contributions
- Mentioned in CHANGELOG.md entries
- Credited in commit messages when providing fixes/solutions
- Welcome to add themselves to a CONTRIBUTORS file (future)

### Types of Recognition

- 🐛 Bug reporters who provide detailed, reproducible issues
- 💻 Code contributors who submit PRs
- 📝 Documentation improvers
- 🧪 Test writers and reviewers
- 💬 Community helpers who support other users
- 🎨 UI/UX improvers (for dashboard contributions)

---

Thank you for contributing to MCP Memory Service! Your efforts help make AI assistants more capable and useful for everyone. 🚀

If you have questions not covered here, please open an [issue on GitHub](https://github.com/doobidoo/mcp-memory-service/issues) or check our [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki).