import asyncio
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
import json
import hashlib
from dataclasses import dataclass

@dataclass
class Memory:
    content: str
    content_hash: str
    tags: List[str]
    memory_type: str
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MemoryQueryResult:
    memory: Memory
    similarity: Optional[float] = None

class ChromaMemoryStorage:
    def __init__(self, path: str = "chroma_test_db"):
        self.path = path
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        self.client = chromadb.EphemeralClient()  # Use EphemeralClient for testing
        self.collection = self.client.create_collection(
            name="memory_collection",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )

    def _format_metadata_for_chroma(self, memory: Memory) -> Dict[str, Any]:
        """Format metadata for ChromaDB storage."""
        metadata = {
            "type": memory.memory_type,
            "content_hash": memory.content_hash,
            "tags": json.dumps(memory.tags) if memory.tags else "[]",
            "timestamp": memory.timestamp
        }
        metadata.update(memory.metadata)
        return metadata

    async def store(self, memory: Memory) -> Tuple[bool, Optional[str]]:
        try:
            existing = self.collection.get(where={"content_hash": memory.content_hash})
            if existing["ids"]:
                return False, "Duplicate content detected."

            metadata = self._format_metadata_for_chroma(memory)
            self.collection.add(
                documents=[memory.content],
                metadatas=[metadata],
                ids=[memory.content_hash]
            )
            return True, None
        except Exception as e:
            return False, str(e)

    async def recall(self, query: Optional[str] = None, n_results: int = 5,
                    start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[MemoryQueryResult]:
        try:
            where_clause = {}
            if start_time is not None and end_time is not None:
                where_clause = {
                    "$and": [
                        {"timestamp": {"$gte": start_time}},
                        {"timestamp": {"$lte": end_time}}
                    ]
                }

            if query:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause,
                    include=["metadatas", "documents", "distances"]
                )
            else:
                results = self.collection.get(
                    where=where_clause,
                    limit=n_results,
                    include=["metadatas", "documents"]
                )

            memory_results = []
            
            # Get the appropriate lists based on whether we used query or get
            documents = results["documents"]
            if isinstance(documents, str):
                documents = [documents]
            
            metadatas = results["metadatas"]
            if isinstance(metadatas, dict):
                metadatas = [metadatas]
            
            distances = results.get("distances", [None] * len(documents))
            if isinstance(distances, float):
                distances = [distances]

            for i, doc in enumerate(documents):
                metadata = metadatas[i]
                try:
                    retrieved_tags = json.loads(metadata.get("tags", "[]"))
                except json.JSONDecodeError:
                    retrieved_tags = []

                memory = Memory(
                    content=doc,
                    content_hash=metadata["content_hash"],
                    tags=retrieved_tags,
                    memory_type=metadata.get("type", ""),
                    metadata={k: v for k, v in metadata.items() 
                            if k not in ["type", "content_hash", "tags", "timestamp"]},
                    timestamp=metadata.get("timestamp")
                )
                
                similarity = distances[i] if i < len(distances) else None
                memory_results.append(MemoryQueryResult(memory, similarity))

            return memory_results

        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []

async def test_memory_service():
    storage = ChromaMemoryStorage()
    
    # Generate content hash
    def generate_content_hash(content: str, metadata: Dict[str, Any]) -> str:
        combined = content + json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    # Create test memories
    test_memories = [
        {"content": "AWS EC2 setup guide with detailed instructions", 
         "tags": ["aws", "setup"], 
         "type": "documentation",
         "timestamp": 1673913600},
        {"content": "Python coding best practices and patterns", 
         "tags": ["python", "coding"], 
         "type": "guide",
         "timestamp": 1673913700},
        {"content": "Database backup procedures for AWS services", 
         "tags": ["aws", "database", "backup"], 
         "type": "procedure",
         "timestamp": 1673913800}
    ]
    
    # Store memories
    for mem_data in test_memories:
        content_hash = generate_content_hash(mem_data["content"], mem_data)
        memory = Memory(
            content=mem_data["content"],
            content_hash=content_hash,
            tags=mem_data["tags"],
            memory_type=mem_data["type"],
            timestamp=mem_data["timestamp"]
        )
        success, error = await storage.store(memory)
        print(f"Stored memory: {success}, Error: {error}")
    
    # Test recall with different queries
    print("\nTesting recall with 'AWS' query:")
    results = await storage.recall(query="AWS", n_results=2)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result.memory.content}")
        print(f"Tags: {result.memory.tags}")
        print(f"Similarity: {result.similarity}")
    
    # Test recall with time range
    print("\nTesting recall with time range:")
    time_results = await storage.recall(
        start_time=1673913600,
        end_time=1673913700,
        n_results=2
    )
    for i, result in enumerate(time_results):
        print(f"\nTime-based Result {i+1}:")
        print(f"Content: {result.memory.content}")
        print(f"Tags: {result.memory.tags}")
        print(f"Timestamp: {result.memory.timestamp}")

# Run the test
asyncio.run(test_memory_service())
