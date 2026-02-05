# Memory Metadata Enhancement API

## Overview

The Memory Metadata Enhancement API provides efficient memory metadata updates without requiring complete memory recreation. This addresses the core limitation identified in Issue #10 where updating memory metadata required deleting and recreating entire memory entries.

## API Method

### `update_memory_metadata`

Updates memory metadata while preserving the original memory content, embeddings, and optionally timestamps.

**Signature:**
```python
async def update_memory_metadata(
    content_hash: str,
    updates: Dict[str, Any],
    preserve_timestamps: bool = True
) -> Tuple[bool, str]
```

**Parameters:**
- `content_hash` (string, required): The content hash of the memory to update
- `updates` (object, required): Dictionary of metadata fields to update
- `preserve_timestamps` (boolean, optional): Whether to preserve original created_at timestamp (default: true)

**Returns:**
- `success` (boolean): Whether the update was successful
- `message` (string): Summary of updated fields or error message

## Supported Update Fields

### Core Metadata Fields

1. **tags** (array of strings)
   - Replaces existing tags completely
   - Example: `"tags": ["important", "reference", "new-tag"]`

2. **memory_type** (string)
   - Updates the memory type classification
   - Example: `"memory_type": "reminder"`

3. **metadata** (object)
   - Merges with existing custom metadata
   - Example: `"metadata": {"priority": "high", "due_date": "2024-01-15"}`

### Custom Fields

Any other fields not in the protected list can be updated directly:
- `"priority": "urgent"`
- `"status": "active"`
- `"category": "work"`
- Custom application-specific fields

### Protected Fields

These fields cannot be modified through this API:
- `content` - Memory content is immutable
- `content_hash` - Content hash is immutable
- `embedding` - Embeddings are preserved automatically
- `created_at` / `created_at_iso` - Preserved unless `preserve_timestamps=false`
- Internal timestamp fields (`timestamp`, `timestamp_float`, `timestamp_str`)

## Usage Examples

### Example 1: Add Tags to Memory

```json
{
  "content_hash": "abc123def456...",
  "updates": {
    "tags": ["important", "reference", "project-alpha"]
  }
}
```

### Example 2: Update Memory Type and Custom Metadata

```json
{
  "content_hash": "abc123def456...",
  "updates": {
    "memory_type": "reminder",
    "metadata": {
      "priority": "high",
      "due_date": "2024-01-15",
      "assignee": "john.doe@example.com"
    }
  }
}
```

### Example 3: Update Custom Fields Directly

```json
{
  "content_hash": "abc123def456...",
  "updates": {
    "priority": "urgent",
    "status": "active",
    "category": "work",
    "last_reviewed": "2024-01-10"
  }
}
```

### Example 4: Update with Timestamp Reset

```json
{
  "content_hash": "abc123def456...",
  "updates": {
    "tags": ["archived", "completed"]
  },
  "preserve_timestamps": false
}
```

## Timestamp Behavior

### Default Behavior (preserve_timestamps=true)

- `created_at` and `created_at_iso` are preserved from original memory
- `updated_at` and `updated_at_iso` are set to current time
- Legacy timestamp fields are updated for backward compatibility

### Reset Behavior (preserve_timestamps=false)

- All timestamp fields are set to current time
- Useful for marking memories as "refreshed" or "re-activated"

## Implementation Details

### Storage Layer

The API is implemented in the storage abstraction layer:

1. **Base Storage Interface** (`storage/base.py`)
   - Abstract method definition
   - Consistent interface across storage backends

2. **ChromaDB Implementation** (`storage/chroma.py`)
   - Efficient upsert operation preserving embeddings
   - Metadata merging with validation
   - Timestamp synchronization

3. **Future Storage Backends**
   - sqlite-vec implementation will follow same interface
   - Other storage backends can implement consistently

### MCP Protocol Integration

The API is exposed via the MCP protocol:

1. **Tool Registration** - Available as `update_memory_metadata` tool
2. **Input Validation** - Comprehensive parameter validation
3. **Error Handling** - Clear error messages for debugging
4. **Logging** - Detailed operation logging for monitoring

## Performance Benefits

### Efficiency Gains

1. **No Content Re-processing**
   - Original content remains unchanged
   - No need to regenerate embeddings
   - Preserves vector database relationships

2. **Minimal Network Transfer**
   - Only metadata changes are transmitted
   - Reduced bandwidth usage
   - Faster operation completion

3. **Database Optimization**
   - Single update operation vs delete+insert
   - Maintains database indices and relationships
   - Reduces transaction overhead

### Resource Savings

- **Memory Usage**: No need to load full memory content
- **CPU Usage**: No embedding regeneration required
- **Storage I/O**: Minimal database operations
- **Network**: Reduced data transfer

## Error Handling

### Common Error Scenarios

1. **Memory Not Found**
   ```
   Error: Memory with hash abc123... not found
   ```

2. **Invalid Updates Format**
   ```
   Error: updates must be a dictionary
   ```

3. **Invalid Tags Format**
   ```
   Error: Tags must be provided as a list of strings
   ```

4. **Storage Not Initialized**
   ```
   Error: Collection not initialized, cannot update memory metadata
   ```

### Error Recovery

- Detailed error messages for debugging
- Transaction rollback on failures
- Original memory remains unchanged on errors
- Logging for troubleshooting

## Migration and Compatibility

### Backward Compatibility

- Existing memories work without modification
- Legacy timestamp fields are maintained
- No breaking changes to existing APIs

### Migration Strategy

1. **Immediate Availability** - API available immediately after deployment
2. **Gradual Adoption** - Can be adopted incrementally
3. **Fallback Support** - Original store/delete pattern still works
4. **Validation** - Comprehensive testing before production use

## Use Cases

### Memory Organization

1. **Tag Management**
   - Add organizational tags over time
   - Categorize memories as understanding improves
   - Apply bulk tagging for organization

2. **Priority Updates**
   - Mark memories as high/low priority
   - Update urgency as contexts change
   - Implement memory lifecycle management

3. **Status Tracking**
   - Track memory processing status
   - Mark memories as reviewed/processed
   - Implement workflow states

### Advanced Features

1. **Memory Linking**
   - Add relationship metadata
   - Create memory hierarchies
   - Implement reference systems

2. **Time-to-Live Management**
   - Add expiration metadata
   - Implement memory aging
   - Schedule automatic cleanup

3. **Access Control**
   - Add ownership metadata
   - Implement sharing controls
   - Track access permissions

## Testing and Validation

### Unit Tests

- Comprehensive test coverage for all update scenarios
- Error condition testing
- Timestamp behavior validation
- Metadata merging verification

### Integration Tests

- End-to-end MCP protocol testing
- Storage backend compatibility testing
- Performance benchmarking
- Cross-platform validation

### Performance Testing

- Large dataset updates
- Concurrent update operations
- Memory usage monitoring
- Response time measurement

## Future Enhancements

### Planned Improvements

1. **Batch Updates** - Update multiple memories in single operation
2. **Conditional Updates** - Update only if conditions are met
3. **Metadata Validation** - Schema validation for metadata fields
4. **Update History** - Track metadata change history
5. **Selective Updates** - Update only specific metadata fields

### Storage Backend Support

- sqlite-vec implementation (Issue #40)
- Other vector database backends
- Consistent API across all backends
- Performance optimization per backend

## Conclusion

The Memory Metadata Enhancement API provides a robust, efficient solution for memory metadata management. It enables sophisticated memory organization features while maintaining excellent performance and backward compatibility.

This implementation forms the foundation for advanced memory management features like re-tagging systems (Issue #45) and memory consolidation (Issue #11).
