#!/usr/bin/env python3
"""
Simple analysis script to examine SQLite-vec database without dependencies.
"""

import os
import re
import sqlite3
import sys


def analyze_database(db_path):
    """Analyze the database structure and content."""
    print(f"Analyzing database: {db_path}")
    print("=" * 60)

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables found: {', '.join(tables)}")
        print()

        # Analyze memories table
        if "memories" in tables:
            cursor.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]
            print(f"📝 Memories: {memory_count}")

            if memory_count > 0:
                # Sample some memories
                cursor.execute("SELECT content, tags, memory_type FROM memories LIMIT 3")
                samples = cursor.fetchall()
                print("   Sample memories:")
                for i, (content, tags, mem_type) in enumerate(samples, 1):
                    print(f"   {i}. [{mem_type or 'general'}] {content[:50]}..." + (f" (tags: {tags})" if tags else ""))

        # Analyze embeddings table
        if "memory_embeddings" in tables:
            try:
                cursor.execute("SELECT COUNT(*) FROM memory_embeddings")
                embedding_count = cursor.fetchone()[0]
                print(f"🧠 Embeddings: {embedding_count}")

                # Check schema to get dimension
                cursor.execute("""
                    SELECT sql FROM sqlite_master
                    WHERE type='table' AND name='memory_embeddings'
                """)
                schema = cursor.fetchone()
                if schema:
                    print(f"   Schema: {schema[0]}")
                    match = re.search(r"FLOAT\[(\d+)\]", schema[0])
                    if match:
                        dimension = int(match.group(1))
                        print(f"   Dimension: {dimension}")

            except Exception as e:
                print(f"🧠 Embeddings: Error accessing table - {e}")

        else:
            print("🧠 Embeddings: Table not found")

        # Check for mismatches
        if "memories" in tables and "memory_embeddings" in tables:
            cursor.execute("SELECT COUNT(*) FROM memories")
            mem_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM memory_embeddings")
            emb_count = cursor.fetchone()[0]

            print()
            if mem_count == emb_count:
                print("✅ Memory and embedding counts match")
            else:
                print(f"⚠️  Mismatch: {mem_count} memories vs {emb_count} embeddings")

                # Find memories without embeddings
                cursor.execute("""
                    SELECT COUNT(*) FROM memories m
                    WHERE NOT EXISTS (
                        SELECT 1 FROM memory_embeddings e WHERE e.rowid = m.id
                    )
                """)
                missing = cursor.fetchone()[0]
                if missing > 0:
                    print(f"   → {missing} memories missing embeddings")

                # Find orphaned embeddings
                cursor.execute("""
                    SELECT COUNT(*) FROM memory_embeddings e
                    WHERE NOT EXISTS (
                        SELECT 1 FROM memories m WHERE m.id = e.rowid
                    )
                """)
                orphaned = cursor.fetchone()[0]
                if orphaned > 0:
                    print(f"   → {orphaned} orphaned embeddings")

        # Check for extension loading capability
        print()
        try:
            conn.enable_load_extension(True)
            print("✅ Extension loading enabled")
        except:
            print("❌ Extension loading not available")

    except Exception as e:
        print(f"❌ Error analyzing database: {e}")

    finally:
        conn.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_sqlite_vec_db.py <database_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    analyze_database(db_path)


if __name__ == "__main__":
    main()
