# Memory Type Taxonomy (Updated Nov 2025)

Database consolidated from 342 fragmented types to 128 organized types. Use these **24 core types** for all new memories.

## Content Types
- `note` - General notes, observations, summaries
- `reference` - Reference materials, knowledge base entries
- `document` - Formal documents, code snippets
- `guide` - How-to guides, tutorials, troubleshooting guides

## Activity Types
- `session` - Work sessions, development sessions
- `implementation` - Implementation work, integrations
- `analysis` - Analysis, reports, investigations
- `troubleshooting` - Problem-solving, debugging
- `test` - Testing activities, validation

## Artifact Types
- `fix` - Bug fixes, corrections
- `feature` - New features, enhancements
- `release` - Releases, release notes
- `deployment` - Deployments, deployment records

## Progress Types
- `milestone` - Milestones, completions, achievements
- `status` - Status updates, progress reports

## Infrastructure Types
- `configuration` - Configurations, setups, settings
- `infrastructure` - Infrastructure changes, system updates
- `process` - Processes, workflows, procedures
- `security` - Security-related memories
- `architecture` - Architecture decisions, design patterns

## Other Types
- `documentation` - Documentation artifacts
- `solution` - Solutions, resolutions
- `achievement` - Accomplishments, successes

## Usage Guidelines

### Avoid Creating New Type Variations

**DON'T** create variations like:
- `bug-fix`, `bugfix`, `technical-fix` → Use `fix`
- `technical-solution`, `project-solution` → Use `solution`
- `project-implementation` → Use `implementation`
- `technical-note` → Use `note`

### Avoid Redundant Prefixes

Remove unnecessary qualifiers:
- `project-*` → Use base type
- `technical-*` → Use base type
- `development-*` → Use base type

### Cleanup Commands

```bash
# Preview type consolidation
python scripts/maintenance/consolidate_memory_types.py --dry-run

# Execute type consolidation
python scripts/maintenance/consolidate_memory_types.py

# Check type distribution
python scripts/maintenance/check_memory_types.py

# Assign types to untyped memories
python scripts/maintenance/assign_memory_types.py --dry-run
python scripts/maintenance/assign_memory_types.py
```

## Consolidation Rules

The consolidation script applies these transformations:

1. **Fix variants** → `fix`: bug-fix, bugfix, technical-fix, etc.
2. **Implementation variants** → `implementation`: integrations, project-implementation, etc.
3. **Solution variants** → `solution`: technical-solution, project-solution, etc.
4. **Note variants** → `note`: technical-note, development-note, etc.
5. **Remove redundant prefixes**: project-, technical-, development-

## Benefits of Standardization

- Improved search and retrieval accuracy
- Better tag-based filtering
- Reduced database fragmentation
- Easier memory type analytics
- Consistent memory organization
