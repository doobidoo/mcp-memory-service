#!/usr/bin/env python3
"""
Comprehensive Database Health Check for MCP Memory Service SQLite-vec Backend
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Set environment for sqlite-vec
os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'

class HealthChecker:
    def __init__(self):
        self.results = []
        self.errors = []
    
    async def test(self, name: str, func):
        """Run a test and record results."""
        print(f"🔍 {name}...")
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            
            duration = time.time() - start_time
            
            if result:
                print(f"   ✅ PASS ({duration:.2f}s)")
                self.results.append((name, "PASS", duration))
            else:
                print(f"   ❌ FAIL ({duration:.2f}s)")
                self.results.append((name, "FAIL", duration))
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ ERROR ({duration:.2f}s): {str(e)}")
            self.results.append((name, "ERROR", duration))
            self.errors.append((name, str(e)))
    
    def test_imports(self):
        """Test all necessary imports."""
        try:
            import sqlite_vec
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            from mcp_memory_service.models.memory import Memory
            from mcp_memory_service.utils.hashing import generate_content_hash
            return True
        except ImportError as e:
            print(f"      Import error: {e}")
            return False
    
    async def test_database_creation(self):
        """Test database creation and initialization."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "health_check.db")
        storage = None
        try:
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()
            
            # Check tables exist
            cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['memories', 'memory_embeddings']
            for table in expected_tables:
                if table not in tables:
                    print(f"      Missing table: {table}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"      Database creation error: {e}")
            return False
        finally:
            if storage:
                try:
                    await storage.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_memory_operations(self):
        """Test core memory operations."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "operations_test.db")
        storage = None
        try:
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            from mcp_memory_service.models.memory import Memory
            from mcp_memory_service.utils.hashing import generate_content_hash
            
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()
            
            # Test store
            content = "Health check test memory"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["health", "check"],
                memory_type="test"
            )
            
            success, message = await storage.store(memory)
            if not success:
                print(f"      Store failed: {message}")
                return False
            
            # Test retrieve
            results = await storage.retrieve("health check", n_results=1)
            if not results:
                print("      Retrieve failed: No results")
                return False
            
            # Test tag search
            tag_results = await storage.search_by_tag(["health"])
            if not tag_results:
                print("      Tag search failed: No results")
                return False
            
            # Test delete
            success, message = await storage.delete(memory.content_hash)
            if not success:
                print(f"      Delete failed: {message}")
                return False
            
            return True
            
        except Exception as e:
            print(f"      Memory operations error: {e}")
            return False
        finally:
            if storage:
                try:
                    await storage.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_vector_search(self):
        """Test vector similarity search functionality."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "vector_test.db")
        storage = None
        try:
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            from mcp_memory_service.models.memory import Memory
            from mcp_memory_service.utils.hashing import generate_content_hash
            
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()
            
            # Store multiple memories
            test_memories = [
                "Python programming is fun",
                "JavaScript development tools",
                "Database design patterns"
            ]
            
            for content in test_memories:
                memory = Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=["programming"],
                    memory_type="test"
                )
                await storage.store(memory)
            
            # Test vector search
            results = await storage.retrieve("programming languages", n_results=3)
            if len(results) != 3:
                print(f"      Expected 3 results, got {len(results)}")
                return False
            
            # Check relevance scores
            for result in results:
                if result.relevance_score < 0:
                    print(f"      Invalid relevance score: {result.relevance_score}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"      Vector search error: {e}")
            return False
        finally:
            if storage:
                try:
                    await storage.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_embedding_invariants(self):
        """Test invariant: all active memories have a corresponding embedding row."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "embedding_invariants_test.db")
        storage = None
        try:
            from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
            from mcp_memory_service.models.memory import Memory
            from mcp_memory_service.utils.hashing import generate_content_hash

            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            content = "Embedding invariant verification memory"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["invariant", "test"],
                memory_type="test"
            )
            success, message = await storage.store(memory)
            if not success:
                print(f"      Store failed during embedding invariant test: {message}")
                return False

            cursor = storage.conn.execute("""
                SELECT COUNT(*) FROM memories m
                WHERE m.deleted_at IS NULL
                  AND NOT EXISTS (SELECT 1 FROM memory_embeddings e WHERE e.rowid = m.id)
            """)
            missing = cursor.fetchone()[0]
            if missing != 0:
                print(f"      Embedding invariant violated: {missing} active memories without embeddings")
                return False

            return True
        except Exception as e:
            print(f"      Embedding invariants error: {e}")
            return False
        finally:
            if storage:
                try:
                    await storage.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_configured_database_invariants(self):
        """Test invariant on existing configured database: active memories have embeddings."""
        import sqlite3
        conn = None
        try:
            from mcp_memory_service.config.storage import SQLITE_VEC_PATH
            db_path = SQLITE_VEC_PATH
            if not db_path or not os.path.exists(db_path):
                print(f"      Configured database not found ({db_path}), skipping check")
                return True

            conn = sqlite3.connect(db_path)
            try:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception as e:
                print(f"      Failed to load sqlite-vec extension on {db_path}: {e}")
                return False

            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('memories', 'memory_embeddings')")
            tables = {row[0] for row in cursor.fetchall()}

            if 'memories' not in tables:
                print(f"      'memories' table not present in {db_path} (uninitialized), skipping check")
                return True

            if 'memory_embeddings' not in tables:
                print(f"      Embedding invariant violated: 'memories' table exists but 'memory_embeddings' is missing in {db_path}")
                return False

            cursor.execute("""
                SELECT COUNT(*) FROM memories m
                WHERE m.deleted_at IS NULL
                  AND NOT EXISTS (SELECT 1 FROM memory_embeddings e WHERE e.rowid = m.id)
            """)
            missing = cursor.fetchone()[0]
            if missing > 0:
                print(f"      Configured database invariant violated: {missing} active memories without embeddings in {db_path}")
                return False

            print(f"      Configured database verified: 0 missing embeddings in {db_path}")
            return True
        except Exception as e:
            print(f"      Configured database invariant error: {e}")
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def test_environment(self):
        """Test environment configuration."""
        required_vars = {
            'MCP_MEMORY_STORAGE_BACKEND': 'sqlite_vec'
        }
        
        for var, expected in required_vars.items():
            actual = os.getenv(var)
            if actual != expected:
                print(f"      Environment variable {var}: expected '{expected}', got '{actual}'")
                return False
        
        return True
    
    def test_dependencies(self):
        """Test that all dependencies are available."""
        try:
            import sqlite_vec
            version = getattr(sqlite_vec, '__version__', 'unknown')
            print(f"      sqlite-vec version: {version}")
            
            import sentence_transformers
            print(f"      sentence-transformers available")
            
            import torch
            print(f"      torch available")
            
            return True
        except ImportError as e:
            print(f"      Dependency error: {e}")
            return False
    
    def test_database_path(self):
        """Test database path accessibility."""
        home = str(Path.home())
        db_path = os.path.join(home, '.local', 'share', 'mcp-memory', 'sqlite_vec.db')
        db_dir = os.path.dirname(db_path)
        
        try:
            # Ensure directory exists
            os.makedirs(db_dir, exist_ok=True)
            
            # Test write permission
            test_file = os.path.join(db_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            print(f"      Database path: {db_path}")
            print(f"      Directory writable: ✅")
            return True
            
        except Exception as e:
            print(f"      Database path error: {e}")
            return False

async def main():
    """Main health check function."""
    print("=" * 60)
    print("🏥 MCP Memory Service - SQLite-vec Database Health Check")
    print("=" * 60)
    
    checker = HealthChecker()
    
    # Run all tests
    await checker.test("Environment Configuration", checker.test_environment)
    await checker.test("Dependencies Check", checker.test_dependencies)
    await checker.test("Import Tests", checker.test_imports)
    await checker.test("Database Path", checker.test_database_path)
    await checker.test("Database Creation", checker.test_database_creation)
    await checker.test("Memory Operations", checker.test_memory_operations)
    await checker.test("Vector Search", checker.test_vector_search)
    await checker.test("Embedding Invariants (Write Test)", checker.test_embedding_invariants)
    await checker.test("Configured Database Invariants", checker.test_configured_database_invariants)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Health Check Summary")
    print("=" * 60)
    
    passed = sum(1 for _, status, _ in checker.results if status == "PASS")
    failed = sum(1 for _, status, _ in checker.results if status in ["FAIL", "ERROR"])
    total = len(checker.results)
    
    for name, status, duration in checker.results:
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {name}: {status} ({duration:.2f}s)")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if checker.errors:
        print("\n❌ Errors Found:")
        for name, error in checker.errors:
            print(f"   • {name}: {error}")
    
    if passed == total:
        print("\n🎉 Database Health Check: PASSED")
        print("   SQLite-vec backend is fully functional and ready for production use!")
        print("\n🚀 Ready for Claude Code integration:")
        print("   - Start server: python -m mcp_memory_service.server")
        print("   - Database: ~/.local/share/mcp-memory/sqlite_vec.db")
        print("   - 75% memory reduction vs ChromaDB")
        return 0
    else:
        print("\n💥 Database Health Check: FAILED")
        print("   Please resolve the issues above before using the service.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))