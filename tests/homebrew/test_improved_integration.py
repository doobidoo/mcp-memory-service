#\!/usr/bin/env python3
"""
Test script to verify the improved Homebrew PyTorch integration.
"""
import os
import sys
import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure environment variables
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"
os.environ["MCP_MEMORY_USE_HOMEBREW_PYTORCH"] = "1"

# Import the module
try:
    from mcp_memory_service.homebrew_integration import get_homebrew_model, patch_storage
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.utils.hashing import generate_content_hash
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

async def main():
    print("=== MCP Memory Service Improved Homebrew PyTorch Integration Test ===")
    
    # First, test the Homebrew model directly
    print("\n1. Testing Homebrew PyTorch model...")
    model = get_homebrew_model("paraphrase-MiniLM-L3-v2")
    
    if model and model.initialized:
        print(f"✅ Model loaded successfully: {model.model_name}")
        print(f"✅ Model dimension: {model.dimension}")
        
        # Test encoding
        test_texts = [
            "This is a test sentence for encoding.",
            "Let's test if the Homebrew PyTorch integration works properly."
        ]
        
        embeddings = model.encode(test_texts)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    else:
        print("❌ Failed to load Homebrew PyTorch model")
        sys.exit(1)
    
    # Now test the storage integration
    print("\n2. Testing storage integration...")
    
    # Initialize the storage
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    print(f"Using SQLite-vec database at: {db_path}")
    
    storage = SqliteVecMemoryStorage(db_path)
    
    # Patch the storage to use Homebrew PyTorch
    success = patch_storage(storage)
    print(f"{'✅' if success else '❌'} Storage patched: {success}")
    
    # Initialize the storage
    await storage.initialize()
    
    # Verify that the model is available
    if hasattr(storage, 'model') and storage.model:
        print("✅ Storage has model attribute")
        
        # Check if it's our HomebrewPyTorchModel
        homebrew_model = hasattr(storage.model, 'homebrew_python')
        print(f"{'✅' if homebrew_model else '❌'} Using Homebrew PyTorch model: {homebrew_model}")
    else:
        print("❌ Storage does not have model attribute")
    
    # Get database stats
    print("\n3. Database Stats")
    stats = storage.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Store a test memory
    print("\n4. Creating Test Memory")
    test_content = f"Improved integration test memory created at {datetime.now().isoformat()}"
    
    test_memory = Memory(
        content=test_content,
        content_hash=generate_content_hash(test_content),
        tags=["improved-test", "homebrew-pytorch"],
        memory_type="note",
        metadata={"source": "improved_test_script"}
    )
    print(f"Memory content: {test_memory.content}")
    
    success, message = await storage.store(test_memory)
    print(f"{'✅' if success else '❌'} Store success: {success}")
    print(f"Message: {message}")
    
    # Retrieve by tag
    print("\n5. Retrieving by Tag")
    memories = await storage.search_by_tag(["improved-test"])
    
    if memories:
        print(f"✅ Found {len(memories)} memories with tag 'improved-test'")
        for i, memory in enumerate(memories):
            print(f"  Memory {i+1}: {memory.content[:60]}...")
    else:
        print("❌ No memories found with tag 'improved-test'")
    
    # Try semantic search
    print("\n6. Semantic Search")
    results = await storage.retrieve("improved integration homebrew pytorch", n_results=5)
    
    if results:
        print(f"✅ Found {len(results)} memories via semantic search")
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Content: {result.memory.content[:60]}...")
            print(f"    Score: {result.relevance_score}")
    else:
        print("❌ No memories found via semantic search")
    
    # Test the debug utilities
    print("\n7. Testing Debug Utilities")
    try:
        from mcp_memory_service.utils.debug import check_embedding_model
        result = check_embedding_model(storage)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error in debug utilities: {e}")
    
    print("\n=== Test Complete ===")
    storage.close()

if __name__ == "__main__":
    asyncio.run(main())
