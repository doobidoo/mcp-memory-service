#\!/usr/bin/env python3
"""
Test script to verify memory service with Homebrew PyTorch embeddings.
"""
import os
import sys
import json
import sqlite3
import asyncio
import time
import subprocess
from datetime import datetime
import importlib.util
import pathlib

# Set environment variables for testing
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

# Get the Homebrew Python path
result = subprocess.run(
    ['brew', '--prefix', 'pytorch'],
    capture_output=True,
    text=True,
    check=True
)
pytorch_prefix = result.stdout.strip()
homebrew_python_path = f"{pytorch_prefix}/libexec/bin/python3"
homebrew_site_packages = subprocess.check_output(
    [homebrew_python_path, "-c", "import site; print(site.getsitepackages()[0])"],
    text=True
).strip()

print(f"Using Homebrew Python: {homebrew_python_path}")
print(f"Homebrew site-packages: {homebrew_site_packages}")

# Import sentence_transformers from Homebrew Python
sys.path.append(homebrew_site_packages)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    import sentence_transformers
    print(f"sentence-transformers version: {sentence_transformers.__version__}")
    
    # Create a test encoder
    model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L3-v2')
    print(f"Model loaded: {model}")
    
    # Test encoding
    test_text = "This is a test sentence for encoding with Homebrew PyTorch"
    embeddings = model.encode([test_text])
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Now try to connect to the SQLite database and run a test
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    print(f"Connecting to SQLite database at: {db_path}")
    
    # Check if the database exists
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)
        print(f"Created directory: {db_dir}")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Check if memories table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
    tables = cursor.fetchall()
    
    if not tables:
        print("Creating memories table")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                memory_type TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL,
                created_at_iso TEXT,
                updated_at_iso TEXT
            )
        ''')
        conn.commit()
    
    # Store a test memory
    test_content = f"Test memory created at {datetime.now().isoformat()} with Homebrew PyTorch"
    
    # Generate a simple hash
    import hashlib
    content_hash = hashlib.sha256(test_content.encode()).hexdigest()
    
    # Prepare metadata
    tags = "test,homebrew-pytorch,embedding-test"
    metadata = json.dumps({"source": "test_script", "type": "test"})
    timestamp = time.time()
    timestamp_iso = datetime.utcfromtimestamp(timestamp).isoformat() + "Z"
    
    # Insert into database
    try:
        conn.execute('''
            INSERT INTO memories (
                content_hash, content, tags, memory_type,
                metadata, created_at, updated_at, created_at_iso, updated_at_iso
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content_hash,
            test_content,
            tags,
            "note",
            metadata,
            timestamp,
            timestamp,
            timestamp_iso,
            timestamp_iso
        ))
        conn.commit()
        print(f"Stored test memory: {content_hash}")
    except sqlite3.IntegrityError:
        print(f"Memory with hash {content_hash} already exists")
    
    # Count memories
    cursor = conn.execute('SELECT COUNT(*) FROM memories')
    memory_count = cursor.fetchone()[0]
    print(f"Database contains {memory_count} memories")
    
    # Test retrieving memories by tag
    cursor = conn.execute("SELECT content FROM memories WHERE tags LIKE '%test%' ORDER BY created_at DESC LIMIT 5")
    test_memories = cursor.fetchall()
    print(f"Found {len(test_memories)} memories with 'test' tag")
    for i, memory in enumerate(test_memories):
        print(f"  Memory {i+1}: {memory[0][:60]}...")
    
    # Close connection
    conn.close()
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

print("Test completed successfully\!")
