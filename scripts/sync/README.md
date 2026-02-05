# Database Synchronization Scripts

This directory contains scripts for synchronizing SQLite-vec databases across multiple machines using JSON export/import and Litestream replication.

## Overview

The synchronization system enables you to:
- Export memories from one machine to JSON format
- Import memories from multiple JSON files into a central database
- Set up real-time replication using Litestream
- Maintain consistent memory databases across multiple devices

## Scripts

### export_memories.py

Export memories from a local SQLite-vec database to JSON format.

**Usage:**
```bash
# Basic export
python export_memories.py

# Export from specific database
python export_memories.py --db-path /path/to/sqlite_vec.db

# Export to specific file
python export_memories.py --output my_export.json

# Export with embedding vectors (large file size)
python export_memories.py --include-embeddings

# Export only specific tags
python export_memories.py --filter-tags claude-code,architecture
```

**Features:**
- Preserves original timestamps and metadata
- Adds source machine tracking
- Supports tag filtering
- Optional embedding vector export
- Cross-platform compatibility

### import_memories.py

Import memories from JSON export files into a central database.

**Usage:**
```bash
# Import single file
python import_memories.py windows_export.json

# Import multiple files
python import_memories.py windows_export.json macbook_export.json

# Dry run analysis
python import_memories.py --dry-run exports/*.json

# Import to specific database
python import_memories.py --db-path /path/to/central.db exports/*.json
```

**Features:**
- Intelligent deduplication based on content hash
- Preserves original timestamps
- Adds source tracking tags
- Conflict detection and resolution
- Comprehensive import statistics

## Typical Workflow

### Phase 1: Initial Consolidation

1. **Export from each machine:**
   ```bash
   # On Windows PC
   python export_memories.py --output windows_memories.json

   # On MacBook
   python export_memories.py --output macbook_memories.json
   ```

2. **Transfer files to central server:**
   ```bash
   scp windows_memories.json central-server:/tmp/
   scp macbook_memories.json central-server:/tmp/
   ```

3. **Import on central server:**
   ```bash
   # Analyze first
   python import_memories.py --dry-run /tmp/*.json

   # Import for real
   python import_memories.py /tmp/windows_memories.json /tmp/macbook_memories.json
   ```

### Phase 2: Set up Litestream

After consolidating all memories into the central database, set up Litestream for ongoing synchronization.

## JSON Export Format

The export format preserves all memory data:

```json
{
  "export_metadata": {
    "source_machine": "machine-name",
    "export_timestamp": "2025-08-12T10:30:00Z",
    "total_memories": 450,
    "database_path": "/path/to/sqlite_vec.db",
    "platform": "Windows",
    "exporter_version": "5.0.0"
  },
  "memories": [
    {
      "content": "Memory content here",
      "content_hash": "sha256hash",
      "tags": ["tag1", "tag2"],
      "created_at": 1673545200.0,
      "updated_at": 1673545200.0,
      "memory_type": "note",
      "metadata": {},
      "export_source": "machine-name"
    }
  ]
}
```

## Deduplication Strategy

Memories are deduplicated based on content hash:
- Same content hash = duplicate (skipped)
- Different content hash = unique (imported)
- Original timestamps are preserved
- Source machine tags are added for tracking

## Error Handling

Both scripts include comprehensive error handling:
- JSON format validation
- Database connectivity checks
- File existence verification
- Transaction rollback on failures
- Detailed error logging

## Performance Considerations

- **Export**: ~1000 memories/second
- **Import**: ~500 memories/second with deduplication
- **File Size**: ~1KB per memory (without embeddings)
- **Memory Usage**: Processes files in streaming fashion

## Troubleshooting

### Common Issues

1. **Database not found:**
   ```bash
   # Check default location
   python -c "from mcp_memory_service.config import get_storage_path; print(get_storage_path())"
   ```

2. **Permission errors:**
   ```bash
   # Ensure database directory is writable
   chmod 755 ~/.local/share/mcp-memory/
   ```

3. **JSON format errors:**
   ```bash
   # Validate JSON file
   python -m json.tool export.json > /dev/null
   ```

### Logging

Enable verbose logging for debugging:
```bash
python export_memories.py --verbose
python import_memories.py --verbose
```

## Next Steps

After using these scripts for initial consolidation:

1. Set up Litestream for real-time sync
2. Configure replica nodes
3. Implement monitoring and alerting
4. Schedule regular backups

See the main documentation for complete Litestream setup instructions.
