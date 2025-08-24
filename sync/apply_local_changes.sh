#!/bin/bash
# Apply staged changes to the main database with intelligent conflict resolution

MAIN_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db"
STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"
CONFLICT_LOG="/Users/hkr/Library/Application Support/mcp-memory/sync_conflicts.log"

echo "$(date): Applying staged changes to main database..."

if [ ! -f "$MAIN_DB" ]; then
    echo "$(date): ERROR: Main database not found at $MAIN_DB"
    exit 1
fi

if [ ! -f "$STAGING_DB" ]; then
    echo "$(date): No staging database found - nothing to apply"
    exit 0
fi

# Get count of staged changes
STAGED_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'none';" 2>/dev/null || echo "0")

if [ "$STAGED_COUNT" -eq 0 ]; then
    echo "$(date): No staged changes to apply"
    exit 0
fi

echo "$(date): Found $STAGED_COUNT staged changes to apply"

# Create backup before applying changes
BACKUP_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_pre_apply.db"
cp "$MAIN_DB" "$BACKUP_PATH"
echo "$(date): Created backup at $BACKUP_PATH"

# Initialize conflict log
echo "$(date): Starting application of staged changes" >> "$CONFLICT_LOG"

# Apply changes with conflict detection
APPLIED_COUNT=0
CONFLICT_COUNT=0
SKIPPED_COUNT=0

# Process each staged change
sqlite3 "$STAGING_DB" "
SELECT id, content, content_hash, tags, metadata, memory_type, 
       operation, staged_at, original_created_at, source_machine
FROM staged_memories 
WHERE conflict_status = 'none'
ORDER BY staged_at ASC;
" | while IFS='|' read -r id content content_hash tags metadata memory_type operation staged_at created_at source_machine; do

    # Escape single quotes for SQL
    content_escaped=$(echo "$content" | sed "s/'/''/g")
    tags_escaped=$(echo "$tags" | sed "s/'/''/g") 
    metadata_escaped=$(echo "$metadata" | sed "s/'/''/g")
    
    case "$operation" in
        "INSERT")
            # Check if content already exists in main database (by hash)
            EXISTING_COUNT=$(sqlite3 "$MAIN_DB" "
                SELECT COUNT(*) FROM memories 
                WHERE content = '$content_escaped' 
                   OR (content_hash IS NOT NULL AND content_hash = '$content_hash');
            " 2>/dev/null || echo "0")
            
            if [ "$EXISTING_COUNT" -gt 0 ]; then
                echo "$(date): CONFLICT: Content already exists (hash: ${content_hash:0:8}...)"
                echo "$(date): CONFLICT: ${content:0:80}..." >> "$CONFLICT_LOG"
                
                # Mark as conflict in staging
                sqlite3 "$STAGING_DB" "
                UPDATE staged_memories 
                SET conflict_status = 'detected' 
                WHERE id = '$id';
                "
                CONFLICT_COUNT=$((CONFLICT_COUNT + 1))
            else
                # Insert new memory
                # Note: This assumes your main database has a 'memories' table
                # Adjust the INSERT statement based on your actual schema
                INSERT_RESULT=$(sqlite3 "$MAIN_DB" "
                INSERT INTO memories (content, content_hash, tags, metadata, memory_type, created_at, updated_at)
                VALUES (
                    '$content_escaped',
                    '$content_hash', 
                    '$tags_escaped',
                    '$metadata_escaped',
                    '$memory_type',
                    COALESCE('$created_at', datetime('now')),
                    datetime('now')
                );
                " 2>&1)
                
                if [ $? -eq 0 ]; then
                    echo "$(date): Applied: ${content:0:50}..."
                    APPLIED_COUNT=$((APPLIED_COUNT + 1))
                    
                    # Remove from staging on successful application
                    sqlite3 "$STAGING_DB" "DELETE FROM staged_memories WHERE id = '$id';"
                else
                    echo "$(date): ERROR applying change: $INSERT_RESULT"
                    echo "$(date): ERROR: ${content:0:80}... - $INSERT_RESULT" >> "$CONFLICT_LOG"
                    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                fi
            fi
            ;;
            
        "UPDATE")
            # For updates, try to find the record and update it
            # This is more complex and depends on your schema
            echo "$(date): UPDATE operation not yet implemented for: ${content:0:50}..."
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            ;;
            
        "DELETE")
            # For deletes, remove the record if it exists
            echo "$(date): DELETE operation not yet implemented for ID: $id"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            ;;
            
        *)
            echo "$(date): Unknown operation: $operation"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            ;;
    esac
done

# Update counters (need to read from temp files since we're in a subshell)
# For now, let's get final counts from the databases
FINAL_STAGED_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'none';" 2>/dev/null || echo "0")
FINAL_CONFLICT_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'detected';" 2>/dev/null || echo "0")

PROCESSED_COUNT=$((STAGED_COUNT - FINAL_STAGED_COUNT))

echo "$(date): Application completed"
echo "$(date): Changes processed: $PROCESSED_COUNT"
echo "$(date): Conflicts detected: $FINAL_CONFLICT_COUNT"
echo "$(date): Remaining staged: $FINAL_STAGED_COUNT"

# Update sync status
sqlite3 "$STAGING_DB" "
UPDATE sync_status 
SET value = datetime('now'), updated_at = CURRENT_TIMESTAMP 
WHERE key = 'last_local_sync';
"

if [ "$FINAL_CONFLICT_COUNT" -gt 0 ]; then
    echo "$(date): WARNING: $FINAL_CONFLICT_COUNT conflicts detected"
    echo "$(date): Check conflict log: $CONFLICT_LOG"
    echo "$(date): Use ./resolve_conflicts.sh to handle conflicts"
fi

if [ "$FINAL_STAGED_COUNT" -gt 0 ]; then
    echo "$(date): NOTE: $FINAL_STAGED_COUNT changes still staged (may need manual review)"
fi

# Keep backup if there were issues
if [ "$FINAL_CONFLICT_COUNT" -gt 0 ] || [ "$FINAL_STAGED_COUNT" -gt 0 ]; then
    echo "$(date): Backup preserved due to conflicts/remaining changes"
else
    rm -f "$BACKUP_PATH"
    echo "$(date): Backup removed (clean application)"
fi

echo "$(date): Staged changes application completed"