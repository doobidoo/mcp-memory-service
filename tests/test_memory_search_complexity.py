"""Keep the memory-search handler within the complexity budget (#1484)."""

import ast
from pathlib import Path


def test_handle_memory_search_stays_within_complexity_budget():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/mcp_memory_service/server/handlers/memory.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    handler = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "handle_memory_search"
    )

    nodes = list(ast.walk(handler))
    # McCabe decision points for the branch constructs used by this handler.
    complexity = 1 + sum(
        isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                          ast.ExceptHandler, ast.IfExp))
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

    assert complexity <= 8, (
        f"handle_memory_search complexity is {complexity}; the budget is 8"
    )
