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

"""Debug utilities for memory service."""

from typing import Any

from ..models.memory import Memory, MemoryQueryResult


def _get_embedding_model(storage):
    """
    Get the embedding model from storage, handling different backend attribute names.

    Different backends use different attribute names (e.g., 'model', 'embedding_model').
    """
    if hasattr(storage, "model") and storage.model is not None:
        return storage.model
    elif hasattr(storage, "embedding_model") and storage.embedding_model is not None:
        return storage.embedding_model
    else:
        raise AttributeError(f"Storage backend {type(storage).__name__} has no embedding model attribute")


def get_raw_embedding(storage, content: str) -> dict[str, Any]:
    """Get raw embedding vector for content."""
    try:
        model = _get_embedding_model(storage)
        embedding = model.encode(content).tolist()
        return {"status": "success", "embedding": embedding, "dimension": len(embedding)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_embedding_model(storage) -> dict[str, Any]:
    """Check if embedding model is loaded and working."""
    try:
        model = _get_embedding_model(storage)
        test_embedding = model.encode("test").tolist()

        # Try to get model name, handling different model types
        model_name = "unknown"
        if hasattr(model, "_model_card_vars"):
            model_name = model._model_card_vars.get("modelname", "unknown")
        elif hasattr(storage, "embedding_model_name"):
            model_name = storage.embedding_model_name

        return {"status": "healthy", "model_loaded": True, "model_name": model_name, "embedding_dimension": len(test_embedding)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def debug_retrieve_memory(
    storage, query: str, n_results: int = 5, similarity_threshold: float = 0.0
) -> list[MemoryQueryResult]:
    """Retrieve memories with enhanced debug information including raw similarity scores."""
    try:
        # Use the storage's retrieve method which handles embedding generation and search
        results = await storage.retrieve(query, n_results)

        # Filter by similarity threshold and enhance debug info
        filtered_results = []
        for result in results:
            if result.relevance_score >= similarity_threshold:
                # Enhance debug info with additional details
                enhanced_debug_info = result.debug_info.copy() if result.debug_info else {}
                enhanced_debug_info.update(
                    {
                        "raw_similarity": result.relevance_score,
                        "backend": type(storage).__name__,
                        "query": query,
                        "similarity_threshold": similarity_threshold,
                    }
                )

                filtered_results.append(
                    MemoryQueryResult(
                        memory=result.memory, relevance_score=result.relevance_score, debug_info=enhanced_debug_info
                    )
                )

        return filtered_results
    except Exception:
        # Return empty list on error to match original behavior
        return []


async def exact_match_retrieve(storage, content: str) -> list[Memory]:
    """Retrieve memories using exact content match."""
    try:
        results = storage.collection.get(where={"content": content})

        memories = []
        for i in range(len(results["ids"])):
            memory = Memory.from_dict(
                {"content": results["documents"][i], **results["metadatas"][i]},
                embedding=results["embeddings"][i] if "embeddings" in results else None,
            )
            memories.append(memory)

        return memories
    except Exception:
        return []
