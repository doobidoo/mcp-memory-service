# Tag Storage Procedure

## File Structure Overview
```
mcp_memory_service/
├── tests/
│   └── test_tag_storage.py    # Integration tests
├── scripts/
│   ├── validate_memories.py   # Validation script
│   └── migrate_tags.py        # Migration script
```

## Execution Steps

1. **Run Initial Validation**
   ```bash
   python scripts/validate_memories.py
   ```
   - Generates validation report of current state

2. **Run Integration Tests**
   ```bash
   python tests/test_tag_storage.py
   ```
   - Verifies functionality

3. **Execute Migration**
   ```bash
   python scripts/migrate_tags.py
   ```
   The script will:
   - Create a backup automatically
   - Run validation check
   - Ask for confirmation before proceeding
   - Perform migration
   - Verify the migration

4. **Post-Migration Validation**
   ```bash
   python scripts/validate_memories.py
   ```
   - Confirms successful migration

## Monitoring Requirements
- Keep backup files for at least 7 days
- Monitor logs for any tag-related errors
- Run validation script daily for the first week
- Check search functionality with various tag formats

## Rollback Process
If issues are detected, use:
```bash
python scripts/migrate_tags.py --rollback
```
