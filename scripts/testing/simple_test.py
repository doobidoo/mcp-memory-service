#\!/usr/bin/env python3
"""
Simple test to use Homebrew Python's sentence-transformers
"""
import os
import sys
import subprocess

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

print(f"Using Homebrew Python: {homebrew_python_path}")

# Run a simple test with the Homebrew Python
test_script = """
import torch
import sentence_transformers
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"sentence-transformers: {sentence_transformers.__version__}")

# Load a model
model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L3-v2')
print(f"Model loaded: {model}")

# Encode a test sentence
test_text = "This is a test sentence for encoding with Homebrew PyTorch"
embedding = model.encode([test_text])
print(f"Embedding shape: {embedding.shape}")
print("Test successful\!")
"""

# Run the test with Homebrew Python
result = subprocess.run(
    [homebrew_python_path, "-c", test_script],
    capture_output=True,
    text=True
)

print("=== STDOUT ===")
print(result.stdout)

if result.stderr:
    print("=== STDERR ===")
    print(result.stderr)
