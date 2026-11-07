"""Keep the Cloudflare memory-store helpers within the repository's budget."""

import ast
from pathlib import Path

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage


def _decision_complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    nodes = list(ast.walk(function))
    complexity = 1 + sum(
        isinstance(
            node,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp),
        )
        for node in nodes
    )
    complexity += sum(
        len(node.values) - 1 for node in nodes if isinstance(node, ast.BoolOp)
    )
    complexity += sum(
        1 + len(node.ifs)
        for node in nodes
        if isinstance(node, ast.comprehension)
    )
    return complexity


def test_cloudflare_memory_store_helpers_stay_within_complexity_budget():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src/mcp_memory_service/storage/cloudflare.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    target_names = {
        "_memory_r2_key",
        "_memory_storage_ids",
        "_memory_vector_metadata",
        "_stage_memory_artifacts",
        "_store_memory",
    }
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in target_names
    }

    assert functions.keys() >= target_names
    over_budget = {
        name: _decision_complexity(function)
        for name, function in functions.items()
        if _decision_complexity(function) > 8
    }
    assert not over_budget, f"Cloudflare store helper complexity exceeds 8: {over_budget}"


@pytest.mark.asyncio
async def test_restore_staging_failure_cleans_partial_remote_artifacts(monkeypatch):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.r2_bucket = "test-bucket"
    storage.large_content_threshold = 1
    stored_r2 = []
    deleted_vectors = []
    deleted_r2 = []

    async def generate_embedding(_content):
        return [0.1, 0.2]

    async def store_r2(key, _content):
        stored_r2.append(key)

    async def fail_vectorize(*_args):
        raise RuntimeError("Vectorize unavailable")

    async def delete_vector(vector_id):
        deleted_vectors.append(vector_id)

    async def delete_r2(key):
        deleted_r2.append(key)

    monkeypatch.setattr(storage, "_generate_embedding", generate_embedding)
    monkeypatch.setattr(storage, "_store_r2_content", store_r2)
    monkeypatch.setattr(storage, "_store_vectorize_vector", fail_vectorize)
    monkeypatch.setattr(storage, "_delete_vectorize_vector", delete_vector)
    monkeypatch.setattr(storage, "_delete_r2_content", delete_r2)

    memory = Memory(
        content="staged restore with a vector failure",
        content_hash="stable-content-hash",
        created_at=100.0,
        updated_at=101.0,
        tags=["__test__"],
    )
    success, message = await storage._store_memory(
        memory, expected_deleted_at=99.0
    )

    assert success is False
    assert message == "Storage failed: Vectorize unavailable"
    assert len(stored_r2) == len(deleted_r2) == 1
    assert stored_r2[0] == deleted_r2[0]
    assert len(deleted_vectors) == 1
    assert deleted_vectors[0].startswith("restore-")
