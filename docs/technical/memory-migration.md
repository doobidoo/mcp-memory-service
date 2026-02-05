# Memory Migration Technical Documentation

This document provides technical details about the memory migration process in the MCP Memory Service.

## Overview

The memory migration process allows transferring memories between different ChromaDB instances, supporting both local and remote migrations. The process is handled by the `mcp-migration.py` script, which provides a robust and efficient way to move memories while maintaining data integrity.

## Migration Types

### 1. Local to Remote Migration
- Source: Local ChromaDB instance
- Target: Remote ChromaDB server
- Use case: Moving memories from a development environment to production

### 2. Remote to Local Migration
- Source: Remote ChromaDB server
- Target: Local ChromaDB instance
- Use case: Creating local backups or development environments

## Technical Implementation

### Environment Verification
Before starting the migration, the script performs environment verification:
- Checks Python version compatibility
- Verifies required packages are installed
- Validates ChromaDB paths and configurations
- Ensures network connectivity for remote migrations

### Migration Process
1. **Connection Setup**
   - Establishes connections to both source and target ChromaDB instances
   - Verifies collection existence and creates if necessary
   - Sets up embedding functions for consistent vectorization

2. **Data Transfer**
   - Implements batch processing (default batch size: 10)
   - Includes delay between batches to prevent overwhelming the target
   - Handles duplicate detection to avoid data redundancy
   - Maintains metadata and document relationships

3. **Verification**
   - Validates successful transfer by comparing record counts
   - Checks for data integrity
   - Provides detailed logging of the migration process

## Error Handling

The migration script includes comprehensive error handling for:
- Connection failures
- Collection access issues
- Data transfer interruptions
- Configuration errors
- Environment incompatibilities

## Performance Considerations

- **Batch Size**: Default 10 records per batch
- **Delay**: 1 second between batches
- **Memory Usage**: Optimized for minimal memory footprint
- **Network**: Handles connection timeouts and retries

## Configuration Options

### Source Configuration
```json
{
    "type": "local|remote",
    "config": {
        "path": "/path/to/chroma",  // for local
        "host": "remote-host",      // for remote
        "port": 8000               // for remote
    }
}
```

### Target Configuration
```json
{
    "type": "local|remote",
    "config": {
        "path": "/path/to/chroma",  // for local
        "host": "remote-host",      // for remote
        "port": 8000               // for remote
    }
}
```

## Best Practices

1. **Pre-Migration**
   - Verify environment compatibility
   - Ensure sufficient disk space
   - Check network connectivity for remote migrations
   - Backup existing data

2. **During Migration**
   - Monitor progress through logs
   - Avoid interrupting the process
   - Check for error messages

3. **Post-Migration**
   - Verify data integrity
   - Check collection statistics
   - Validate memory access

## Troubleshooting

Common issues and solutions:

1. **Connection Failures**
   - Verify network connectivity
   - Check firewall settings
   - Validate host and port configurations

2. **Data Transfer Issues**
   - Check disk space
   - Verify collection permissions
   - Monitor system resources

3. **Environment Issues**
   - Run environment verification
   - Check package versions
   - Validate Python environment

## Example Usage

### Command Line
```bash
# Local to Remote Migration
python scripts/mcp-migration.py \
    --source-type local \
    --source-config /path/to/local/chroma \
    --target-type remote \
    --target-config '{"host": "remote-host", "port": 8000}'

# Remote to Local Migration
python scripts/mcp-migration.py \
    --source-type remote \
    --source-config '{"host": "remote-host", "port": 8000}' \
    --target-type local \
    --target-config /path/to/local/chroma
```

### Programmatic Usage
```python
from scripts.mcp_migration import migrate_memories

# Local to Remote Migration
migrate_memories(
    source_type='local',
    source_config='/path/to/local/chroma',
    target_type='remote',
    target_config={'host': 'remote-host', 'port': 8000}
)

# Remote to Local Migration
migrate_memories(
    source_type='remote',
    source_config={'host': 'remote-host', 'port': 8000},
    target_type='local',
    target_config='/path/to/local/chroma'
)
```
