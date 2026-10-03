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

"""Interface-conformance guard for the storage contract the web API relies on.

Two guards live here, both added after a backend silently drifted from what its
callers assume.

Part one, the `store` keyword (issue #133), described below. Part two, the set
of methods the web API calls on `storage` without a `hasattr` guard (issue
#213): Milvus never implemented `get_all_tags_with_counts`, so the unguarded
call in web/api/memories.py::get_tags raised AttributeError, which that handler
converts into HTTP 501 — the error the Browse tab showed. The `store` guard
could not catch it: it only compares signatures of methods that exist.

Part one — the multi-store `store` parameter (issue #133).

The multi-store partition key (commit 53745ac0, #57 Phase 1) added a `store`
keyword argument to a set of storage methods and to every call site in the
service/handler layer. Python does not check that each backend's override kept
its signature in sync, so the Milvus backend shipped with `count_all_memories`,
`search_memories`, and `store` still on the old signature -> `count_all_memories()
got an unexpected keyword argument 'store'` at runtime (issue #133).

This test is the practical equivalent of the compile-time check the reporter
asked for: it inspects each backend class (no live backend needed) and asserts
that every method in the multi-store contract accepts a `store` keyword, so the
signatures can never silently drift apart again.

Part two — methods the web API calls unguarded (issue #213).

Some `storage.<method>()` call sites in web/api/ are wrapped in
`if hasattr(storage, ...)` (or bail out early when it is missing), which makes
the method optional. The rest are not, which makes the method mandatory for
every backend the web server can be pointed at. WEB_API_REQUIRED_METHODS is that
second list: each entry must be declared on MemoryStorage and overridden by all
four backends, so a missing method fails here instead of on a dashboard tab.
"""

import importlib
import inspect

import pytest

from mcp_memory_service.storage.base import MemoryStorage

# Methods that callers invoke with `store=` (service + server/handlers layer).
# Any backend override of these MUST accept a `store` keyword argument.
STORE_CONTRACT_METHODS = [
    "store",
    "get_all_memories",
    "count_all_memories",
    "search_memories",
    "delete_memories",
    # Added with the HTTP store-scope work (#1106): web/api now passes store=
    # to these as well. Milvus.retrieve was missing the keyword entirely, so a
    # by-time search with a semantic query raised TypeError -> HTTP 500.
    "retrieve",
    "recall",
    "get_by_hash",
    "get_memory_timestamps",
    "get_all_tags_with_counts",
    "get_graph_visualization_data",
]

# Methods that web/api/*.py calls on `storage` WITHOUT a hasattr guard, i.e.
# every backend must implement them or the endpoint fails at runtime (issue
# #213: Milvus lacked get_all_tags_with_counts -> AttributeError -> HTTP 501 on
# /api/tags; recall and get_largest_memories were missing the same way).
#
# Deliberately excluded, because their call sites DO guard:
#   - get_sync_status / force_sync / pause_sync / resume_sync (web/api/sync.py
#     returns early when the attribute is absent — hybrid-only by design)
#   - cleanup_duplicates, count_memories_by_tag, count_untagged_memories,
#     delete_by_tag, get_type_counts, get_initial_sync_status (hasattr blocks in
#     manage.py / analytics.py / health.py)
# Add an entry here whenever a new unguarded storage call enters web/api/.
#
# Every entry must be declared on MemoryStorage, and every backend must override
# it — except for the feature areas in BASE_DEFAULT_ALLOWED below, where the
# base default ("this backend has no graph/conflict support") is the intended
# answer rather than a placeholder.
WEB_API_REQUIRED_METHODS = [
    "count_all_memories",
    "delete",
    "delete_by_tags",
    "get_all_memories",
    "get_all_tags_with_counts",
    "get_by_hash",
    "get_conflicts",
    "get_graph_visualization_data",
    "get_largest_memories",
    "get_memories_by_time_range",
    "get_memory_timestamps",
    "get_recent_memories",
    "get_relationship_type_distribution",
    "get_stats",
    "recall",
    "resolve_conflict",
    "retrieve",
    "search_by_tag",
    "search_by_tags",
    "store",
    "update_memory_metadata",
]

# Optional feature areas: the base implementation returns an empty result, and
# backends without graph or conflict-detection support inherit it on purpose.
# The endpoint then reports "nothing" instead of raising, which is the intended
# degradation — unlike the methods above, where an empty answer would be wrong.
BASE_DEFAULT_ALLOWED = {
    "get_conflicts",
    "resolve_conflict",
    "get_graph_visualization_data",
    "get_relationship_type_distribution",
}

# Milvus parity methods (issue #1201). These are not guaranteed on every
# backend: older backends may intentionally inherit a base fallback, and the
# web layer still has SQLite-only compatibility paths for untagged memories and
# type counts. Milvus, however, is expected to implement all of them, and the
# calls below mirror every shape used by web/api and the MCP handlers. Keeping
# the call shapes here prevents a rebase from silently changing a keyword name
# or dropping a parameter that production callers pass.
MILVUS_PARITY_CALL_SHAPES = {
    "count_untagged_memories": [
        ((), {}),
    ],
    "delete_untagged_memories": [
        ((), {}),
    ],
    "get_type_counts": [
        ((), {}),
    ],
    "get_relationship_type_distribution": [
        ((), {}),
    ],
    "get_graph_visualization_data": [
        ((), {}),
        ((10, 2), {}),
        ((), {"limit": 10, "min_connections": 2}),
    ],
    "update_memory_versioned": [
        (("old-hash", "new content"), {}),
        (
            ("old-hash", "new content"),
            {
                "new_tags": ["updated"],
                "new_memory_type": "decision",
                "reason": "source changed",
            },
        ),
    ],
}

# The reasoning service treats GraphStorage as a duck-typed protocol. These
# methods are called by the MCP graph handlers and two-phase aggregation code,
# so MilvusGraphStorage must stay signature-compatible with the SQLite
# GraphStorage implementation instead of growing a private variant.
GRAPH_STORAGE_PARITY_METHODS = [
    "store_entity_link",
    "list_entities",
    "find_memories_by_entity",
    "get_entities_for_memory",
    "get_entity_profile",
    "transitive_closure",
    "common_neighbors",
]

# Backend modules to check. Each must be importable WITHOUT its optional heavy
# deps (pymilvus, etc.) so this guard runs in the ML-free CI image.
BACKEND_MODULES = [
    "mcp_memory_service.storage.sqlite_vec",
    "mcp_memory_service.storage.milvus",
    "mcp_memory_service.storage.cloudflare",
    "mcp_memory_service.storage.hybrid",
]


def _concrete_backend_classes():
    """Yield (class_name, class) for each MemoryStorage subclass defined in a
    backend module. Modules that cannot be imported at all are skipped with a
    marker so the guard still covers every backend that IS importable."""
    found = []
    for mod_name in BACKEND_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # pragma: no cover - import guard
            found.append((mod_name, None, exc))
            continue
        for name, obj in vars(mod).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, MemoryStorage)
                and obj is not MemoryStorage
                and obj.__module__ == mod.__name__
            ):
                found.append((f"{name}", obj, None))
    return found


BACKENDS = _concrete_backend_classes()


def test_backend_modules_all_importable():
    """Every backend module imports without its optional heavy deps present."""
    failed = [(m, repr(e)) for (m, cls, e) in BACKENDS if e is not None]
    assert not failed, f"backend module(s) failed to import: {failed}"


@pytest.mark.parametrize("method_name", STORE_CONTRACT_METHODS)
def test_all_backends_accept_store_kwarg(method_name):
    """Each backend override of a multi-store contract method accepts `store`."""
    offenders = []
    for name, cls, err in BACKENDS:
        if cls is None:
            continue
        method = getattr(cls, method_name, None)
        assert method is not None, f"{name} is missing {method_name}()"
        params = inspect.signature(method).parameters
        accepts_store = (
            "store" in params
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        )
        if not accepts_store:
            offenders.append(name)
    assert not offenders, (
        f"{method_name}() is missing the multi-store `store` keyword on: "
        f"{offenders}. Callers pass store=... to this method (issue #133)."
    )


@pytest.mark.parametrize("method_name", STORE_CONTRACT_METHODS)
def test_base_interface_declares_store_kwarg(method_name):
    """The abstract interface must declare `store` too, not just the backends.

    Checking only concrete backends leaves the declaration free to understate the
    contract. That is exactly what had happened: all four backends and all four
    call sites (services/memory_service.py, server/handlers/documents.py,
    utils/document_processing.py) passed and accepted `store` on `store()`, while
    MemoryStorage.store() still stopped at skip_semantic_dedup — so a new backend
    written against the declaration would have reproduced #133 on day one.
    """
    method = getattr(MemoryStorage, method_name, None)
    assert method is not None, f"MemoryStorage is missing {method_name}()"
    params = inspect.signature(method).parameters
    assert "store" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), (
        f"MemoryStorage.{method_name}() does not declare the multi-store `store` "
        f"keyword, but callers pass it and every backend accepts it (issue #133)."
    )


@pytest.mark.parametrize("method_name", WEB_API_REQUIRED_METHODS)
def test_base_interface_declares_web_api_methods(method_name):
    """The abstract interface declares every method the web API calls unguarded.

    Without the declaration the method is an implicit interface that exists only
    in the backends that happen to have it — which is how Milvus shipped without
    get_all_tags_with_counts (issue #213).
    """
    assert getattr(MemoryStorage, method_name, None) is not None, (
        f"MemoryStorage does not declare {method_name}(), but web/api/ calls it "
        f"on the injected storage without a hasattr guard (issue #213)."
    )


@pytest.mark.parametrize("method_name", WEB_API_REQUIRED_METHODS)
def test_all_backends_implement_web_api_methods(method_name):
    """Every backend overrides them — inheriting the base default is not enough.

    The defaults on MemoryStorage return empty results so a partially
    implemented backend degrades instead of crashing. A backend that ships with
    the default still answers the endpoint with nothing, which is the silent
    version of the same bug, so the guard requires a real override — except for
    BASE_DEFAULT_ALLOWED, where inheriting is the documented behaviour.
    """
    if method_name in BASE_DEFAULT_ALLOWED:
        pytest.skip(f"{method_name}: base default is the intended fallback")
    base_method = getattr(MemoryStorage, method_name, None)
    offenders = []
    for name, cls, err in BACKENDS:
        if cls is None:
            continue
        method = getattr(cls, method_name, None)
        if method is None or (base_method is not None and method is base_method):
            offenders.append(name)
    assert not offenders, (
        f"{method_name}() is not implemented on: {offenders}. web/api/ calls it "
        f"without a hasattr guard, so those backends fail the endpoint at "
        f"runtime (issue #213)."
    )


def _backend_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _public_signature(method):
    """Return a class method's signature without the leading ``self``."""
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        signature = signature.replace(parameters=parameters[1:])
    return signature


def _signature_shape(method):
    """Comparable parameter shape, intentionally ignoring annotations."""
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    ]


@pytest.mark.parametrize("method_name", MILVUS_PARITY_CALL_SHAPES)
def test_milvus_parity_methods_accept_web_api_call_shapes(method_name):
    """Milvus implements the method and accepts every production call shape."""
    milvus = _backend_class(
        "mcp_memory_service.storage.milvus", "MilvusMemoryStorage"
    )
    method = milvus.__dict__.get(method_name)
    assert method is not None, (
        f"MilvusMemoryStorage does not override {method_name}(); issue #1201 "
        f"requires a real Milvus implementation, not a base fallback"
    )
    assert inspect.iscoroutinefunction(method), (
        f"MilvusMemoryStorage.{method_name}() must be async to match its callers"
    )

    signature = _public_signature(method)
    for args, kwargs in MILVUS_PARITY_CALL_SHAPES[method_name]:
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise AssertionError(
                f"MilvusMemoryStorage.{method_name}{signature} does not accept "
                f"the web/MCP call shape args={args!r}, kwargs={kwargs!r}: {exc}"
            ) from exc


@pytest.mark.parametrize("method_name", GRAPH_STORAGE_PARITY_METHODS)
def test_milvus_graph_storage_matches_graph_storage_signature(method_name):
    """MilvusGraphStorage implements the same duck-typed GraphStorage protocol."""
    from mcp_memory_service.storage.graph import GraphStorage

    sqlite_method = getattr(GraphStorage, method_name, None)
    assert sqlite_method is not None, f"GraphStorage is missing {method_name}()"

    milvus_graph = _backend_class(
        "mcp_memory_service.storage.milvus_graph", "MilvusGraphStorage"
    )
    milvus_method = getattr(milvus_graph, method_name, None)
    assert milvus_method is not None, (
        f"MilvusGraphStorage is missing {method_name}(); issue #1201 requires "
        f"parity with GraphStorage"
    )
    assert inspect.iscoroutinefunction(milvus_method), (
        f"MilvusGraphStorage.{method_name}() must be async to match GraphStorage"
    )
    assert _signature_shape(milvus_method) == _signature_shape(sqlite_method), (
        f"MilvusGraphStorage.{method_name}{inspect.signature(milvus_method)} "
        f"does not match GraphStorage.{method_name}{inspect.signature(sqlite_method)}"
    )
