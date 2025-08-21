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
Migration Validation Script for MCP Memory Service

This script validates that a migration from ChromaDB to SQLite-vec was successful
by checking data integrity, required fields, and identifying any corruption.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from mcp_memory_service.storage.chroma import ChromaMemoryStorage
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.utils.hashing import generate_content_hash
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: MCP modules not available. Running in standalone mode.")


class ValidationReport:
    """Container for validation results."""
    
    def __init__(self):
        self.total_memories = 0
        self.valid_memories = 0
        self.issues = []
        self.warnings = []
        self.tag_issues = []
        self.hash_mismatches = []
        self.missing_fields = []
        self.timestamp_issues = []
        self.encoding_issues = []
        
    def add_issue(self, issue: str):
        """Add a critical issue."""
        self.issues.append(issue)
    
    def add_warning(self, warning: str):
        """Add a warning."""
        self.warnings.append(warning)
    
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.issues) == 0
    
    def print_report(self):
        """Print the validation report."""
        print("\n" + "="*60)
        print("MIGRATION VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 Statistics:")
        print(f"  Total memories: {self.total_memories}")
        print(f"  Valid memories: {self.valid_memories}")
        print(f"  Validation rate: {self.valid_memories/self.total_memories*100:.1f}%" if self.total_memories > 0 else "N/A")
        
        if self.issues:
            print(f"\n❌ Critical Issues ({len(self.issues)}):")
            for i, issue in enumerate(self.issues[:10], 1):
                print(f"  {i}. {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        if self.tag_issues:
            print(f"\n🏷️  Tag Issues ({len(self.tag_issues)}):")
            for i, issue in enumerate(self.tag_issues[:5], 1):
                print(f"  {i}. {issue}")
            if len(self.tag_issues) > 5:
                print(f"  ... and {len(self.tag_issues) - 5} more")
        
        if self.hash_mismatches:
            print(f"\n🔑 Hash Mismatches ({len(self.hash_mismatches)}):")
            for i, mismatch in enumerate(self.hash_mismatches[:5], 1):
                print(f"  {i}. {mismatch}")
            if len(self.hash_mismatches) > 5:
                print(f"  ... and {len(self.hash_mismatches) - 5} more")
        
        if self.timestamp_issues:
            print(f"\n🕐 Timestamp Issues ({len(self.timestamp_issues)}):")
            for i, issue in enumerate(self.timestamp_issues[:5], 1):
                print(f"  {i}. {issue}")
            if len(self.timestamp_issues) > 5:
                print(f"  ... and {len(self.timestamp_issues) - 5} more")
        
        # Final verdict
        print("\n" + "="*60)
        if self.is_valid():
            print("✅ VALIDATION PASSED")
            if self.warnings:
                print(f"   (with {len(self.warnings)} warnings)")
        else:
            print("❌ VALIDATION FAILED")
            print(f"   Found {len(self.issues)} critical issues")
        print("="*60)


class MigrationValidator:
    """Tool for validating migrated data."""
    
    def __init__(self, sqlite_path: str, chroma_path: Optional[str] = None):
        self.sqlite_path = sqlite_path
        self.chroma_path = chroma_path
        self.report = ValidationReport()
    
    def validate_content_hash(self, content: str, stored_hash: str) -> bool:
        """Validate that content hash is correct."""
        if not stored_hash:
            return False
        
        # Generate expected hash
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        return expected_hash == stored_hash
    
    def validate_tags(self, tags_str: str) -> Tuple[bool, List[str]]:
        """Validate tag format and return cleaned tags."""
        if not tags_str:
            return True, []
        
        try:
            # Tags should be comma-separated
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            
            # Check for common corruption patterns
            issues = []
            for tag in tags:
                if '\n' in tag or '\r' in tag:
                    issues.append(f"Newline in tag: {repr(tag)}")
                if len(tag) > 100:
                    issues.append(f"Tag too long: {tag[:50]}...")
                if tag.startswith('[') or tag.endswith(']'):
                    issues.append(f"Array syntax in tag: {tag}")
            
            return len(issues) == 0, tags
            
        except Exception as e:
            return False, []
    
    def validate_timestamp(self, timestamp: Any, field_name: str) -> bool:
        """Validate timestamp value."""
        if timestamp is None:
            return False
        
        try:
            if isinstance(timestamp, (int, float)):
                # Check if timestamp is reasonable (between 2000 and 2100)
                if 946684800 <= float(timestamp) <= 4102444800:
                    return True
            return False
        except:
            return False
    
    def validate_metadata(self, metadata_str: str) -> Tuple[bool, Dict]:
        """Validate metadata JSON."""
        if not metadata_str:
            return True, {}
        
        try:
            metadata = json.loads(metadata_str)
            if not isinstance(metadata, dict):
                return False, {}
            return True, metadata
        except json.JSONDecodeError:
            return False, {}
    
    async def validate_sqlite_database(self) -> bool:
        """Validate the SQLite-vec database."""
        print(f"Validating SQLite database: {self.sqlite_path}")
        
        if not Path(self.sqlite_path).exists():
            self.report.add_issue(f"Database file not found: {self.sqlite_path}")
            return False
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            
            # Check if tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            
            if 'memories' not in table_names:
                self.report.add_issue("Missing 'memories' table")
                return False
            
            # Check schema
            schema = conn.execute("PRAGMA table_info(memories)").fetchall()
            column_names = [col[1] for col in schema]
            
            required_columns = [
                'content_hash', 'content', 'tags', 'memory_type',
                'metadata', 'created_at', 'updated_at'
            ]
            
            for col in required_columns:
                if col not in column_names:
                    self.report.add_issue(f"Missing required column: {col}")
            
            # Validate each memory
            cursor = conn.execute("""
                SELECT id, content_hash, content, tags, memory_type,
                       metadata, created_at, updated_at
                FROM memories
                ORDER BY id
            """)
            
            for row in cursor:
                memory_id, content_hash, content, tags, memory_type = row[:5]
                metadata, created_at, updated_at = row[5:]
                
                self.report.total_memories += 1
                memory_valid = True
                
                # Validate content and hash
                if not content:
                    self.report.missing_fields.append(f"Memory {memory_id}: missing content")
                    memory_valid = False
                elif not content_hash:
                    self.report.missing_fields.append(f"Memory {memory_id}: missing content_hash")
                    memory_valid = False
                elif not self.validate_content_hash(content, content_hash):
                    self.report.hash_mismatches.append(
                        f"Memory {memory_id}: hash mismatch (hash: {content_hash[:8]}...)"
                    )
                    memory_valid = False
                
                # Validate tags
                if tags:
                    valid_tags, cleaned_tags = self.validate_tags(tags)
                    if not valid_tags:
                        self.report.tag_issues.append(
                            f"Memory {memory_id}: malformed tags: {tags[:50]}..."
                        )
                
                # Validate timestamps
                if not self.validate_timestamp(created_at, 'created_at'):
                    self.report.timestamp_issues.append(
                        f"Memory {memory_id}: invalid created_at: {created_at}"
                    )
                    memory_valid = False
                
                if not self.validate_timestamp(updated_at, 'updated_at'):
                    self.report.timestamp_issues.append(
                        f"Memory {memory_id}: invalid updated_at: {updated_at}"
                    )
                
                # Validate metadata
                if metadata:
                    valid_meta, meta_dict = self.validate_metadata(metadata)
                    if not valid_meta:
                        self.report.warnings.append(
                            f"Memory {memory_id}: invalid metadata JSON"
                        )
                
                # Check for encoding issues
                try:
                    content.encode('utf-8')
                except UnicodeEncodeError:
                    self.report.encoding_issues.append(
                        f"Memory {memory_id}: encoding issues in content"
                    )
                    memory_valid = False
                
                if memory_valid:
                    self.report.valid_memories += 1
            
            conn.close()
            
            # Check for critical issues
            if self.report.total_memories == 0:
                self.report.add_issue("No memories found in database")
            elif self.report.valid_memories < self.report.total_memories * 0.5:
                self.report.add_issue(
                    f"Less than 50% of memories are valid ({self.report.valid_memories}/{self.report.total_memories})"
                )
            
            return True
            
        except Exception as e:
            self.report.add_issue(f"Database validation error: {e}")
            return False
    
    async def compare_with_chroma(self) -> bool:
        """Compare migrated data with original ChromaDB."""
        if not self.chroma_path or not IMPORTS_AVAILABLE:
            print("Skipping ChromaDB comparison (not available)")
            return True
        
        print(f"Comparing with ChromaDB: {self.chroma_path}")
        
        try:
            # Load ChromaDB memories
            chroma_storage = ChromaMemoryStorage(path=self.chroma_path)
            collection = chroma_storage.collection
            
            if not collection:
                self.report.add_warning("Could not access ChromaDB collection")
                return True
            
            # Get count from ChromaDB
            chroma_results = collection.get()
            chroma_count = len(chroma_results.get('ids', []))
            
            print(f"  ChromaDB memories: {chroma_count}")
            print(f"  SQLite memories: {self.report.total_memories}")
            
            if chroma_count > 0:
                migration_rate = self.report.total_memories / chroma_count * 100
                if migration_rate < 95:
                    self.report.add_warning(
                        f"Only {migration_rate:.1f}% of ChromaDB memories were migrated"
                    )
                elif migration_rate > 105:
                    self.report.add_warning(
                        f"SQLite has {migration_rate:.1f}% of ChromaDB count (possible duplicates)"
                    )
            
            return True
            
        except Exception as e:
            self.report.add_warning(f"Could not compare with ChromaDB: {e}")
            return True
    
    async def run(self) -> bool:
        """Run the validation process."""
        print("\n" + "="*60)
        print("MCP Memory Service - Migration Validator")
        print("="*60)
        
        # Validate SQLite database
        if not await self.validate_sqlite_database():
            self.report.print_report()
            return False
        
        # Compare with ChromaDB if available
        if self.chroma_path:
            await self.compare_with_chroma()
        
        # Print report
        self.report.print_report()
        
        return self.report.is_valid()


def find_databases() -> Tuple[Optional[str], Optional[str]]:
    """Try to find SQLite and ChromaDB databases."""
    sqlite_path = None
    chroma_path = None
    
    # Check environment variables
    sqlite_path = os.environ.get('MCP_MEMORY_SQLITE_PATH')
    chroma_path = os.environ.get('MCP_MEMORY_CHROMA_PATH')
    
    # Check default locations
    home = Path.home()
    if sys.platform == 'darwin':  # macOS
        default_base = home / 'Library' / 'Application Support' / 'mcp-memory'
    elif sys.platform == 'win32':  # Windows
        default_base = Path(os.getenv('LOCALAPPDATA', '')) / 'mcp-memory'
    else:  # Linux
        default_base = home / '.local' / 'share' / 'mcp-memory'
    
    if not sqlite_path:
        # Try to find SQLite database
        possible_sqlite = [
            default_base / 'sqlite_vec.db',
            default_base / 'sqlite_vec_migrated.db',
            default_base / 'memory_migrated.db',
            Path.cwd() / 'sqlite_vec.db',
        ]
        
        for path in possible_sqlite:
            if path.exists():
                sqlite_path = str(path)
                print(f"Found SQLite database: {path}")
                break
    
    if not chroma_path:
        # Try to find ChromaDB
        possible_chroma = [
            home / '.mcp_memory_chroma',
            default_base / 'chroma_db',
            Path.cwd() / 'chroma_db',
        ]
        
        for path in possible_chroma:
            if path.exists():
                chroma_path = str(path)
                print(f"Found ChromaDB: {path}")
                break
    
    return sqlite_path, chroma_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ChromaDB to SQLite-vec migration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'sqlite_path',
        nargs='?',
        help='Path to SQLite-vec database (default: auto-detect)'
    )
    parser.add_argument(
        '--chroma-path',
        help='Path to original ChromaDB for comparison'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with ChromaDB (requires ChromaDB path)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix common issues (experimental)'
    )
    
    args = parser.parse_args()
    
    # Find databases
    if args.sqlite_path:
        sqlite_path = args.sqlite_path
    else:
        sqlite_path, detected_chroma = find_databases()
        if not args.chroma_path and detected_chroma:
            args.chroma_path = detected_chroma
    
    if not sqlite_path:
        print("Error: Could not find SQLite database")
        print("Please specify the path or set MCP_MEMORY_SQLITE_PATH")
        return 1
    
    # Run validation
    validator = MigrationValidator(sqlite_path, args.chroma_path)
    success = asyncio.run(validator.run())
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())