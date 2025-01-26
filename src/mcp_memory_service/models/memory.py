"""Memory-related data models."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

@dataclass
class Memory:
    """Represents a single memory entry."""
    content: str
    content_hash: str
    tags: List[str] = field(default_factory=list)
    memory_type: Optional[str] = None
    # timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    timestamp: float = field(default_factory=lambda: float(f"{time.time():.6f}"))  # Ensure microsecond precision
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary format for storage."""
        return {
            "content": self.content,
            "content_hash": self.content_hash,
            "tags": self.tags,
            # "tags_str": ",".join(self.tags) if self.tags else "",
            "type": self.memory_type,
            "timestamp": self.timestamp,
            **self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> 'Memory':
        """Create a Memory instance from dictionary data."""
        tags = data.get("tags_str", "").split(",") if data.get("tags_str") else []
        return cls(
            content=data["content"],
            content_hash=data["content_hash"],
            tags=[tag for tag in tags if tag],  # Filter out empty tags
            memory_type=data.get("type"),
            timestamp=float(f"{float(data['timestamp']):.6f}") if "timestamp" in data else float(f"{time.time():.6f}"),
            metadata={k: v for k, v in data.items() if k not in 
                     ["content", "content_hash", "tags_str", "type", "timestamp"]},
            embedding=embedding
        )

@dataclass
class MemoryQueryResult:
    """Represents a memory query result with relevance score and debug information."""
    memory: Memory
    similarity: float = 0.0
    # relevance_score: float
    debug_info: Dict[str, Any] = field(default_factory=dict)