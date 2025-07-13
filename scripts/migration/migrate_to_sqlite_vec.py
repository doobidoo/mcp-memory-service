#!/usr/bin/env python3
"""
Simple migration script to help users migrate from ChromaDB to sqlite-vec.
This provides an easy way to switch to the lighter sqlite-vec backend.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from migrate_storage import MigrationTool

async def main():
    """Simple migration from ChromaDB to sqlite-vec with sensible defaults."""
    print("🔄 MCP Memory Service - Migrate to SQLite-vec")
    print("=" * 50)
    
    # Get default paths
    home = Path.home()
    if sys.platform == 'darwin':  # macOS
        base_dir = home / 'Library' / 'Application Support' / 'mcp-memory'
    elif sys.platform == 'win32':  # Windows
        base_dir = Path(os.getenv('LOCALAPPDATA', '')) / 'mcp-memory'
    else:  # Linux
        base_dir = home / '.local' / 'share' / 'mcp-memory'
    
    chroma_path = base_dir / 'chroma_db'
    sqlite_path = base_dir / 'sqlite_vec.db'
    backup_path = base_dir / 'migration_backup.json'
    
    print(f"📁 Source (ChromaDB): {chroma_path}")
    print(f"📁 Target (SQLite-vec): {sqlite_path}")
    print(f"💾 Backup: {backup_path}")
    print()
    
    # Check if source exists
    if not chroma_path.exists():
        print(f"❌ ChromaDB path not found: {chroma_path}")
        print("💡 Make sure you have some memories stored first.")
        return 1
    
    # Check if target already exists
    if sqlite_path.exists():
        response = input(f"⚠️  SQLite-vec database already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Migration cancelled")
            return 1
    
    # Confirm migration
    print("🚀 Ready to migrate!")
    print("   This will:")
    print("   - Export all memories from ChromaDB")
    print("   - Create a backup file")
    print("   - Import memories to SQLite-vec")
    print()
    
    response = input("Continue? (Y/n): ")
    if response.lower() == 'n':
        print("❌ Migration cancelled")
        return 1
    
    # Perform migration
    migration_tool = MigrationTool()
    
    try:
        success = await migration_tool.migrate(
            from_backend='chroma',
            to_backend='sqlite_vec',
            source_path=str(chroma_path),
            target_path=str(sqlite_path),
            create_backup=True,
            backup_path=str(backup_path)
        )
        
        if success:
            print()
            print("✅ Migration completed successfully!")
            print()
            print("📝 Next steps:")
            print("   1. Set environment variable: MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            print("   2. Restart your MCP client (Claude Desktop)")
            print("   3. Test that your memories are accessible")
            print()
            print("🔧 Environment variable examples:")
            print("   # Bash/Zsh:")
            print("   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            print()
            print("   # Windows Command Prompt:")
            print("   set MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
            print()
            print("   # Windows PowerShell:")
            print("   $env:MCP_MEMORY_STORAGE_BACKEND='sqlite_vec'")
            print()
            print(f"💾 Backup available at: {backup_path}")
            return 0
        else:
            print("❌ Migration failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))