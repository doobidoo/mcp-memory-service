#!/usr/bin/env python3
"""
Check the status of SQLite-vec database and identify issues.
"""

import os
import sqlite3
import sys


def check_sqlite_vec_status(db_path):
    """Check the status of the SQLite-vec database."""
    print(f"Checking SQLite-vec database: {db_path}")
    print("=" * 60)

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    issues_found = []

    try:
        # Check basic tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        print(f"Tables: {', '.join(tables)}")

        if "memories" not in tables:
            issues_found.append("Missing 'memories' table")
        else:
            cursor.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]
            print(f"📝 Memories: {memory_count}")

        if "memory_embeddings" not in tables:
            issues_found.append("Missing 'memory_embeddings' table")
        else:
            # Try to access the embeddings table
            try:
                cursor.execute("SELECT COUNT(*) FROM memory_embeddings")
                embedding_count = cursor.fetchone()[0]
                print(f"🧠 Embeddings: {embedding_count}")

                # Check if counts match
                if "memories" in tables:
                    if memory_count != embedding_count:
                        issues_found.append(f"Count mismatch: {memory_count} memories vs {embedding_count} embeddings")

            except Exception as e:
                if "no such module: vec0" in str(e):
                    issues_found.append("sqlite-vec extension not loaded - cannot access embeddings")
                else:
                    issues_found.append(f"Cannot access embeddings table: {e}")

        # Check if extension loading is possible
        try:
            conn.enable_load_extension(True)
            extension_support = True
        except:
            extension_support = False
            issues_found.append("Extension loading not supported")

        print(f"Extension loading: {'✅ Supported' if extension_support else '❌ Not supported'}")

        # Try to load sqlite-vec
        if extension_support:
            try:
                # This will fail if sqlite-vec is not installed
                import sqlite_vec

                sqlite_vec.load(conn)
                print("✅ sqlite-vec extension loaded successfully")

                # Now try to access embeddings
                try:
                    cursor.execute("SELECT COUNT(*) FROM memory_embeddings")
                    embedding_count = cursor.fetchone()[0]
                    print(f"✅ Can now access embeddings: {embedding_count}")

                    # Test a simple search
                    if embedding_count > 0:
                        cursor.execute("SELECT * FROM memory_embeddings LIMIT 1")
                        cursor.fetchone()
                        print("✅ Embedding data accessible")

                except Exception as e:
                    issues_found.append(f"Still cannot access embeddings after loading extension: {e}")

            except ImportError:
                issues_found.append("sqlite-vec Python module not installed")
            except Exception as e:
                issues_found.append(f"Failed to load sqlite-vec extension: {e}")

    except Exception as e:
        issues_found.append(f"Database error: {e}")

    finally:
        conn.close()

    print("\n" + "=" * 60)
    if issues_found:
        print("⚠️  Issues Found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")

        print("\nRecommendations:")
        if "sqlite-vec Python module not installed" in str(issues_found):
            print("  • Install sqlite-vec: uv pip install sqlite-vec")
        if "sentence-transformers" in str(issues_found) or "embedding" in str(issues_found).lower():
            print("  • Install sentence-transformers: uv pip install sentence-transformers torch")
        if "Count mismatch" in str(issues_found):
            print("  • Run repair script to regenerate missing embeddings")
        if "cannot access embeddings" in str(issues_found).lower():
            print("  • Database may need migration to fix schema issues")

    else:
        print("✅ No issues found - database appears healthy!")

    return len(issues_found) == 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_sqlite_vec_status.py <database_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    healthy = check_sqlite_vec_status(db_path)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
