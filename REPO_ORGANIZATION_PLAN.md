# Repository Organization Plan

This document outlines a plan to reorganize the repository structure to make it cleaner, more maintainable, and easier to navigate.

## Current Issues

1. Too many files at the root level
2. Multiple related files with similar purposes scattered across the repository
3. Inconsistent naming conventions
4. Test files mixed with source code
5. Multiple README and documentation files with overlapping content
6. Experimental scripts mixed with production code

## Proposed Directory Structure

```
mcp-memory-service/
├── .github/                      # GitHub-specific files
├── docs/                         # Documentation
│   ├── guides/                   # User guides
│   ├── integration/              # Integration documentation
│   │   └── homebrew/             # Homebrew PyTorch integration docs
│   └── development/              # Developer documentation
├── scripts/                      # Utility scripts
│   ├── installation/             # Installation scripts
│   ├── maintenance/              # Maintenance scripts
│   ├── migration/                # Data migration scripts
│   └── testing/                  # Testing helper scripts
├── src/                          # Source code
│   └── mcp_memory_service/       # Main package
│       ├── models/               # Data models
│       ├── storage/              # Storage implementations
│       │   ├── chroma/           # ChromaDB implementation
│       │   └── sqlite_vec/       # SQLite-vec implementation
│       ├── integrations/         # External integrations
│       │   └── homebrew/         # Homebrew PyTorch integration
│       └── utils/                # Utility modules
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── performance/              # Performance tests
├── tools/                        # Development tools
│   ├── docker/                   # Docker configuration
│   └── deployment/               # Deployment utilities
├── examples/                     # Example configurations and usage
└── archive/                      # Archived code (consider removing)
```

## File Reorganization Plan

### 1. Documentation Consolidation

- Merge related README files
- Move all documentation to the `docs/` directory
- Organize documentation by purpose (user guides, development, integration)

### 2. Homebrew Integration Reorganization

Move Homebrew PyTorch integration files:
- `homebrew_pytorch_embeddings.py` → `src/mcp_memory_service/integrations/homebrew/embeddings.py`
- `homebrew_server.py` → `src/mcp_memory_service/integrations/homebrew/server.py`
- `run_with_homebrew.sh` → `scripts/run/run_with_homebrew.sh`
- All Homebrew integration documentation → `docs/integration/homebrew/`

### 3. Test Organization

- Move all test files to appropriate subdirectories in `tests/`
- Organize by test type (unit, integration, performance)
- Create test fixtures directory for shared test resources

### 4. Script Organization

- Move installation scripts to `scripts/installation/`
- Move maintenance scripts to `scripts/maintenance/`
- Move migration scripts to `scripts/migration/`

### 5. Docker Configuration

- Move all Docker-related files to `tools/docker/`
- Consolidate docker-compose files

### 6. Archive or Remove Outdated Files

- Review files in `archive/` directory
- Remove or clearly mark experimental code
- Document archived code with README explaining status

## Implementation Steps

1. Create the new directory structure
2. Move files to their new locations
3. Update imports and references in the codebase
4. Update documentation to reflect the new structure
5. Update CI/CD configurations if applicable
6. Create clear README files in each directory explaining its purpose

## Benefits

1. Easier navigation and discovery of code
2. Clearer separation of concerns
3. More maintainable codebase
4. Better onboarding experience for new contributors
5. Reduced cognitive load when working on the project
6. More professional presentation of the project

## Post-Migration Tasks

1. Verify all tests pass in the new structure
2. Update installation instructions
3. Update import paths in documentation
4. Consider creating an index or map document for the repository