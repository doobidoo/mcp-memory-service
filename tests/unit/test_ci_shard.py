"""The CI shard split must cover every test exactly once and keep files whole."""

import os
from types import SimpleNamespace

import pytest

from tests.ci_shard import parse_shard, shard_of

NODEIDS = [
    f"tests/{area}/test_{n}.py::test_case_{k}"
    for area in ("unit", "storage", "sync", "consolidation")
    for n in range(40)
    for k in range(3)
]


@pytest.mark.parametrize("total", [1, 2, 3])
def test_every_test_lands_in_exactly_one_shard(total):
    owners = [shard_of(nodeid, total) for nodeid in NODEIDS]
    assert all(1 <= owner <= total for owner in owners)
    # Each shard gets something, so a split never silently runs nothing.
    assert set(owners) == set(range(1, total + 1))


def test_a_file_never_straddles_two_shards():
    by_file = {}
    for nodeid in NODEIDS:
        by_file.setdefault(nodeid.split("::")[0], set()).add(shard_of(nodeid, 2))
    assert all(len(shards) == 1 for shards in by_file.values())


def test_mapping_is_pinned_not_randomized():
    # xdist workers must agree on the selection, so the mapping may not depend on
    # the process (hash() would). Pinned: crc32(b"tests/unit/test_ci_shard.py") is
    # 4139539040.
    assert shard_of("tests/unit/test_ci_shard.py::test_x", 1000) == 41


@pytest.mark.parametrize("spec", ["", None])
def test_unset_means_no_sharding(spec):
    assert parse_shard(spec) is None


@pytest.mark.parametrize("spec", ["2", "0/2", "3/2", "a/b", "1/0", "1/2/3"])
def test_malformed_spec_is_rejected(spec):
    with pytest.raises(ValueError):
        parse_shard(spec)


def test_valid_spec_parses():
    assert parse_shard("2/3") == (2, 3)


# --- The collection hook itself ---------------------------------------------
# These call the pytest_collection_modifyitems that tests/conftest.py registered
# in this very session, so they break if the hook stops filtering, is removed, or
# stops reporting what it drops.


@pytest.fixture
def conftest_hook(request):
    suffix = os.path.join("tests", "conftest.py")
    for impl in request.config.pluginmanager.hook.pytest_collection_modifyitems.get_hookimpls():
        if getattr(impl.plugin, "__file__", "").endswith(suffix):
            return impl.function
    pytest.fail("tests/conftest.py registers no pytest_collection_modifyitems hook")


def _collect(hook, monkeypatch, spec):
    """Run the hook over fake items; return (kept, deselected) node ids."""
    if spec is None:
        monkeypatch.delenv("MCP_TEST_SHARD", raising=False)
    else:
        monkeypatch.setenv("MCP_TEST_SHARD", spec)
    items = [SimpleNamespace(nodeid=nodeid) for nodeid in NODEIDS]
    deselected = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    hook(config=config, items=items)
    return [i.nodeid for i in items], [i.nodeid for i in deselected]


def test_hook_leaves_the_suite_alone_without_a_shard(conftest_hook, monkeypatch):
    kept, deselected = _collect(conftest_hook, monkeypatch, None)
    assert kept == NODEIDS
    assert deselected == []


def test_hook_shards_partition_the_suite(conftest_hook, monkeypatch):
    kept1, dropped1 = _collect(conftest_hook, monkeypatch, "1/2")
    kept2, dropped2 = _collect(conftest_hook, monkeypatch, "2/2")
    assert kept1 and kept2, "a shard selected nothing"
    assert not set(kept1) & set(kept2), "a test runs in both shards"
    assert sorted(kept1 + kept2) == sorted(NODEIDS), "a test runs in no shard"
    # What a shard drops is reported as deselected, and is exactly the other shard.
    assert sorted(dropped1) == sorted(kept2)
    assert sorted(dropped2) == sorted(kept1)


def test_hook_rejects_a_malformed_shard(conftest_hook, monkeypatch):
    with pytest.raises(pytest.UsageError):
        _collect(conftest_hook, monkeypatch, "3/2")
