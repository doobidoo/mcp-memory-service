"""The CI shard split must cover every test exactly once and keep files whole."""

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
