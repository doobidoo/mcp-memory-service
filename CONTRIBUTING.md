# Contributing to MCP Memory Service

Thank you for your interest in contributing to MCP Memory Service! 🎉

This project provides semantic memory and persistent storage for AI assistants through the Model Context Protocol. We welcome contributions of all kinds - from bug fixes and features to documentation and testing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
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
Answer questions in Issues, Discussions, or help other users.

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
   uv run memory server
   ```

6. **Test with MCP Inspector** (optional):
   ```bash
   npx @modelcontextprotocol/inspector uv run memory server
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

- Place tests in `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies

Example test:
```python
import pytest
from mcp_memory_service.storage import SqliteVecStorage

@pytest.mark.asyncio
async def test_store_memory_success():
    """Test successful memory storage."""
    storage = SqliteVecStorage(":memory:")
    result = await storage.store("test content", tags=["test"])
    assert result is not None
    assert "hash" in result
```

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths and edge cases
- Test error handling scenarios
- Include integration tests where appropriate

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
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] No sensitive data exposed

### Review Process

- PRs require at least one review
- Address review feedback promptly
- Keep discussions focused and constructive
- Be patient - reviews may take a few days

## Reporting Issues

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
- **Issues**: Search existing [issues](https://github.com/doobidoo/mcp-memory-service/issues) before creating new ones
- **Discussions**: Use [GitHub Discussions](https://github.com/doobidoo/mcp-memory-service/discussions) for questions
- **Response Time**: Maintainers typically respond within 2-3 days

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code contributions and reviews

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

If you have questions not covered here, please open a [Discussion](https://github.com/doobidoo/mcp-memory-service/discussions) or check our [Wiki](https://github.com/doobidoo/mcp-memory-service/wiki).
