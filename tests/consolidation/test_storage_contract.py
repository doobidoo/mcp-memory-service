"""Guard direct consolidation storage calls against backend drift (#1319)."""

import ast
import importlib
import inspect
from pathlib import Path
from collections import defaultdict

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
KNOWN_INHERITED_FALLBACKS = {
    # Cloudflare D1 has no superseded_by column yet. Keep this limitation
    # explicit instead of treating MemoryStorage's no-op as an implementation.
    ("CloudflareStorage", "mark_superseded_batch"),
}


def _direct_storage_calls() -> dict[str, set[str]]:
    calls = defaultdict(set)
    for path in CONSOLIDATION_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            is_storage = isinstance(receiver, ast.Name) and receiver.id == "storage"
            is_storage_attribute = (
                isinstance(receiver, ast.Attribute) and receiver.attr == "storage"
            )
            if is_storage or is_storage_attribute:
                calls[node.func.attr].update(
                    keyword.arg for keyword in node.keywords if keyword.arg is not None
                )
    return calls


def _missing_keywords(method, keywords: set[str]) -> list[str]:
    if not keywords:
        return []
    parameters = inspect.signature(method).parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return []
    accepted = {parameter.name for parameter in parameters}
    return sorted(keywords - accepted)


def _inherits_base_noop(backend, method_name: str) -> bool:
    method = inspect.getattr_static(backend, method_name, None)
    base_method = inspect.getattr_static(MemoryStorage, method_name, None)
    return method is base_method


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
    calls = {
        name: keywords
        for name, keywords in _direct_storage_calls().items()
        if name not in OPTIONAL_STORAGE_CALLS
    }
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

    # Most base methods are deliberate generic implementations. Supersession is
    # different: the base implementation is an empty no-op and consolidation
    # relies on it changing retrieval state.
    inherited_noops = {
        class_name: ["mark_superseded_batch"]
        for class_name, backend in _backend_classes()
        if "mark_superseded_batch" in calls
        and _inherits_base_noop(backend, "mark_superseded_batch")
        and (class_name, "mark_superseded_batch") not in KNOWN_INHERITED_FALLBACKS
    }
    inherited_noops = {
        class_name: missing for class_name, missing in inherited_noops.items() if missing
    }
    assert not inherited_noops, (
        "Consolidation calls resolve to MemoryStorage no-ops: "
        f"{inherited_noops}"
    )

    keyword_mismatch = {}
    for method_name, keywords in calls.items():
        missing = _missing_keywords(getattr(StorageProtocol, method_name), keywords)
        if missing:
            keyword_mismatch["StorageProtocol"] = {method_name: missing}
        for class_name, backend in _backend_classes():
            missing = _missing_keywords(getattr(backend, method_name), keywords)
            if missing:
                keyword_mismatch.setdefault(class_name, {})[method_name] = missing
    assert not keyword_mismatch, (
        "Storage methods reject keywords used by consolidation: "
        f"{keyword_mismatch}"
    )
