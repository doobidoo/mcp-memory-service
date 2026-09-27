"""Guard direct consolidation storage calls against backend drift (#1319)."""

import ast
import importlib
import inspect
from pathlib import Path

from mcp_memory_service.consolidation.consolidator import StorageProtocol
from mcp_memory_service.storage.base import MemoryStorage


CONSOLIDATION_DIR = (
    Path(__file__).parents[2] / "src" / "mcp_memory_service" / "consolidation"
)
BACKEND_MODULES = (
    "mcp_memory_service.storage.sqlite_vec",
    "mcp_memory_service.storage.hybrid",
    "mcp_memory_service.storage.cloudflare",
    "mcp_memory_service.storage.milvus",
)

# These calls are guarded by SyncPauseContext.is_hybrid. All other direct calls
# must be part of the protocol and available on every concrete backend.
OPTIONAL_STORAGE_CALLS = {"pause_sync", "resume_sync"}


def _direct_storage_calls() -> set[str]:
    calls = set()
    for path in CONSOLIDATION_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            is_storage = isinstance(receiver, ast.Name) and receiver.id == "storage"
            is_self_storage = (
                isinstance(receiver, ast.Attribute)
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
                and receiver.attr == "storage"
            )
            if is_storage or is_self_storage:
                calls.add(node.func.attr)
    return calls


def _backend_classes():
    result = []
    for module_name in BACKEND_MODULES:
        module = importlib.import_module(module_name)
        result.extend(
            (name, value)
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and issubclass(value, MemoryStorage)
            and value is not MemoryStorage
            and value.__module__ == module_name
        )
    return result


def test_direct_consolidation_storage_calls_match_all_backends():
    calls = _direct_storage_calls() - OPTIONAL_STORAGE_CALLS
    protocol_missing = sorted(name for name in calls if not hasattr(StorageProtocol, name))
    assert not protocol_missing, (
        "StorageProtocol is missing direct consolidation calls: "
        f"{protocol_missing}"
    )

    backend_missing = {
        class_name: sorted(name for name in calls if not hasattr(backend, name))
        for class_name, backend in _backend_classes()
    }
    backend_missing = {
        class_name: missing for class_name, missing in backend_missing.items() if missing
    }
    assert not backend_missing, (
        "Consolidation calls are not implemented by every storage backend: "
        f"{backend_missing}"
    )
