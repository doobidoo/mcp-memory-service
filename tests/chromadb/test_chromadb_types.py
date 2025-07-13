"""
Test ChromaDB metadata type support
"""
import chromadb
import time

# Test what types ChromaDB supports in metadata
client = chromadb.Client()
collection = client.create_collection("test_metadata_types")

test_timestamp = time.time()

# Test different metadata formats
test_cases = [
    ("int_timestamp", {"timestamp": int(test_timestamp)}),
    ("float_timestamp", {"timestamp": test_timestamp}),
    ("str_timestamp", {"timestamp": str(test_timestamp)}),
    ("millisecond_int", {"timestamp_ms": int(test_timestamp * 1000)}),
    ("iso_string", {"timestamp_iso": "2025-06-24T12:00:00Z"}),
]

print("Testing ChromaDB metadata type support:\n")

for name, metadata in test_cases:
    try:
        collection.add(
            documents=[f"Test document for {name}"],
            ids=[name],
            metadatas=[metadata]
        )
        print(f"✓ {name}: {metadata} - SUCCESS")
        
        # Try to query with the metadata
        result = collection.get(ids=[name])
        print(f"  Retrieved: {result['metadatas'][0]}")
        
    except Exception as e:
        print(f"✗ {name}: {metadata} - FAILED: {e}")

print("\nTesting numeric filtering with different types:")

# Clear collection
client.delete_collection("test_metadata_types")
collection = client.create_collection("test_metadata_types")

# Add documents with different timestamp formats
now = time.time()
docs = [
    {"id": "1", "timestamp": int(now - 86400), "timestamp_float": now - 86400},  # 1 day ago
    {"id": "2", "timestamp": int(now - 3600), "timestamp_float": now - 3600},    # 1 hour ago
    {"id": "3", "timestamp": int(now), "timestamp_float": now},                  # now
]

for doc in docs:
    collection.add(
        documents=[f"Document {doc['id']}"],
        ids=[doc['id']],
        metadatas=[doc]
    )

# Test filtering
print(f"\nCurrent timestamp: {int(now)}")
print("Testing where clause with $gte on int timestamp:")
results = collection.get(where={"timestamp": {"$gte": int(now - 7200)}})  # 2 hours ago
print(f"  Found {len(results['ids'])} documents: {results['ids']}")

# Try float filtering if supported
try:
    print("\nTesting where clause with $gte on float timestamp:")
    results = collection.get(where={"timestamp_float": {"$gte": now - 7200}})
    print(f"  Found {len(results['ids'])} documents: {results['ids']}")
except Exception as e:
    print(f"  Float filtering failed: {e}")
