#!/usr/bin/env python3
# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced SQLite migration script to fix timestamp formats in ChromaDB.
This improved version populates all timestamp columns with appropriate values.
"""
import sqlite3
import logging
import os
import sys
import platform
from pathlib import Path
import json
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("sqlite_migration")

def find_claude_chroma_db():
    """
    Finds the Claude desktop ChromaDB storage location based on the operating system.
    """
    system = platform.system()
    home = Path.home()
    
    # List of potential ChromaDB locations for Claude desktop
    possible_locations = []
    
    if system == "Darwin":  # macOS
        # Standard iCloud Drive location
        icloud_path = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "AI" / "claude-memory" / "chroma_db"
        possible_locations.append(icloud_path)
        
        # Local AppData location
        local_path = home / "Library" / "Application Support" / "Claude" / "claude-memory" / "chroma_db"
        possible_locations.append(local_path)
        
    elif system == "Windows":
        # Standard Windows location
        appdata_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Claude" / "claude-memory" / "chroma_db"
        possible_locations.append(appdata_path)
        
        # OneDrive potential path
        onedrive_path = home / "OneDrive" / "Documents" / "Claude" / "claude-memory" / "chroma_db"
        possible_locations.append(onedrive_path)
        
    elif system == "Linux":
        # Standard Linux location
        linux_path = home / ".config" / "Claude" / "claude-memory" / "chroma_db"
        possible_locations.append(linux_path)
    
    # Try to find config file that might tell us the location
    config_locations = []
    
    if system == "Darwin":
        config_locations.append(home / "Library" / "Application Support" / "Claude" / "config.json")
    elif system == "Windows":
        config_locations.append(Path(os.environ.get("APPDATA", "")) / "Claude" / "config.json")
    elif system == "Linux":
        config_locations.append(home / ".config" / "Claude" / "config.json")
    
    # Check if config file exists and try to read DB path from it
    for config_path in config_locations:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'memoryStoragePath' in config:
                        mem_path = Path(config['memoryStoragePath']) / "chroma_db"
                        possible_locations.insert(0, mem_path)  # Prioritize this path
                        logger.info(f"Found memory path in config: {mem_path}")
            except Exception as e:
                logger.warning(f"Error reading config file {config_path}: {e}")
    
    # Check all possible locations
    for location in possible_locations:
        db_path = location / "chroma.sqlite3"
        if db_path.exists():
            logger.info(f"Found ChromaDB at: {db_path}")
            return str(db_path)
    
    logger.error("Could not find Claude's ChromaDB storage location")
    return None

def get_table_schema(cursor, table_name):
    """
    Retrieves the schema of a table to understand available columns.
    """
    # Allowlist of valid table names to prevent SQL injection
    ALLOWED_TABLES = {'embedding_metadata', 'memories', 'tags', 'associations', 'quality_scores'}

    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' not in allowed list: {ALLOWED_TABLES}")

    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return {col[1]: col for col in columns}  # returns dict with column name as key

def timestamp_to_all_types(timestamp_value):
    """
    Convert a timestamp to all possible formats:
    - integer (unix timestamp)
    - float (unix timestamp with milliseconds)
    - string (ISO format)
    """
    # Handle different input types
    timestamp_int = None
    
    if isinstance(timestamp_value, int):
        timestamp_int = timestamp_value
    elif isinstance(timestamp_value, float):
        timestamp_int = int(timestamp_value)
    elif isinstance(timestamp_value, str):
        try:
            # Try to parse as float first
            timestamp_int = int(float(timestamp_value))
        except ValueError:
            # Try to parse as ISO date
            try:
                dt = datetime.datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                timestamp_int = int(dt.timestamp())
            except ValueError:
                # Try different date formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                    try:
                        dt = datetime.datetime.strptime(timestamp_value, fmt)
                        timestamp_int = int(dt.timestamp())
                        break
                    except ValueError:
                        continue
    
    if timestamp_int is None:
        raise ValueError(f"Could not convert timestamp value: {timestamp_value}")
    
    # Generate all formats
    timestamp_float = float(timestamp_int)
    
    # ISO format string representation
    dt = datetime.datetime.fromtimestamp(timestamp_int, tz=datetime.timezone.utc)
    timestamp_str = dt.isoformat().replace('+00:00', 'Z')
    
    return {
        'int': timestamp_int,
        'float': timestamp_float,
        'string': timestamp_str
    }

def migrate_timestamps_in_sqlite(db_path):
    """
    Enhanced migration that identifies timestamp data across all columns
    and populates all columns with consistent type values.
    """
    logger.info(f"Connecting to SQLite database at {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return False
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the embedding_metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_metadata'")
        if not cursor.fetchone():
            logger.error("Table embedding_metadata not found in database.")
            conn.close()
            return False
        
        # Get table schema to understand column structure
        schema = get_table_schema(cursor, "embedding_metadata")
        logger.info(f"Table schema: {schema}")
        
        # Find all timestamp entries from any column
        logger.info("Identifying all timestamp entries...")
        cursor.execute("""
            SELECT id, key, string_value, int_value, float_value 
            FROM embedding_metadata 
            WHERE key = 'timestamp'
        """)
        all_rows = cursor.fetchall()
        
        if not all_rows:
            logger.warning("No timestamp entries found in the database.")
            conn.close()
            return True
        
        logger.info(f"Found {len(all_rows)} timestamp entries to process")
        
        # Process each timestamp row
        processed_count = 0
        failed_count = 0
        
        for row in all_rows:
            id_val, key, string_val, int_val, float_val = row
            source_value = None
            source_type = None
            
            # Find which column has a non-NULL value
            if int_val is not None:
                source_value = int_val
                source_type = 'int'
            elif float_val is not None:
                source_value = float_val
                source_type = 'float'
            elif string_val is not None:
                source_value = string_val
                source_type = 'string'
            
            if source_value is None:
                logger.warning(f"Row ID {id_val} has no timestamp value in any column")
                failed_count += 1
                continue
            
            try:
                # Convert to all types
                logger.info(f"Processing ID {id_val}: {source_type} value {source_value}")
                all_formats = timestamp_to_all_types(source_value)
                
                # Update the row with all formats
                cursor.execute("""
                    UPDATE embedding_metadata 
                    SET int_value = ?, float_value = ?, string_value = ? 
                    WHERE id = ? AND KEY ='timestamp'
                """, (all_formats['int'], all_formats['float'], all_formats['string'], id_val))
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing timestamp for ID {id_val}: {e}")
                failed_count += 1
        
        # Commit all changes
        conn.commit()
        
        # Verify the changes
        cursor.execute("""
            SELECT COUNT(*) 
            FROM embedding_metadata 
            WHERE key = 'timestamp' AND (int_value IS NULL OR float_value IS NULL OR string_value IS NULL)
        """)
        incomplete = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM embedding_metadata 
            WHERE key = 'timestamp' AND int_value IS NOT NULL AND float_value IS NOT NULL AND string_value IS NOT NULL
        """)
        complete = cursor.fetchone()[0]
        
        logger.info(f"Migration summary:")
        logger.info(f"  - {processed_count} timestamp entries processed successfully")
        logger.info(f"  - {failed_count} timestamp entries failed to process")
        logger.info(f"  - {complete} timestamp entries now have values in all columns")
        logger.info(f"  - {incomplete} timestamp entries still have NULL values in some columns")
        
        # Show some examples of remaining problematic entries if any
        if incomplete > 0:
            cursor.execute("""
                SELECT id, key, string_value, int_value, float_value 
                FROM embedding_metadata 
                WHERE key = 'timestamp' AND (int_value IS NULL OR float_value IS NULL OR string_value IS NULL)
                LIMIT 5
            """)
            problem_rows = cursor.fetchall()
            logger.info(f"Examples of incomplete entries: {problem_rows}")
        
        conn.close()
        return incomplete == 0
    
    except Exception as e:
        logger.error(f"Error during SQLite migration: {e}")
        return False

def main():
    # Check if a database path was provided as a command-line argument
    if len(sys.argv) >= 2:
        db_path = sys.argv[1]
    else:
        # Try to automatically find the ChromaDB location
        db_path = find_claude_chroma_db()
        if not db_path:
            print("Could not automatically find Claude's ChromaDB location.")
            print("Please provide the path as a command-line argument:")
            print("python migrate_timestamps.py /path/to/chroma.sqlite3")
            sys.exit(1)
    
    print(f"Using database: {db_path}")
    success = migrate_timestamps_in_sqlite(db_path)
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("All timestamps now have consistent values in all columns (int_value, float_value, and string_value).")
        sys.exit(0)
    else:
        print("\n⚠️ Migration completed with issues. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()