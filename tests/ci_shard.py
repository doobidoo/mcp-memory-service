"""Split the test suite into CI shards by test file.

CI runs a long job as several parallel shards, each on its own runner. A shard
is chosen with ``MCP_TEST_SHARD=<index>/<total>`` (1-based, e.g. ``1/2``);
tests/conftest.py deselects every test outside that shard after collection.

Whole files stay together, so module-scoped fixtures and module state never
straddle two runners. The file-to-shard mapping uses crc32, not ``hash()``:
string hashing is randomized per process, and every xdist worker has to
collect exactly the same set of tests.
"""

import zlib
from typing import Optional, Tuple


def parse_shard(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """Return (index, total) for a spec like ``"1/2"``, or None when unset."""
    if not spec:
        return None
    try:
        index, total = (int(part) for part in spec.split("/"))
    except ValueError:
        raise ValueError(f"MCP_TEST_SHARD must look like '1/2', got {spec!r}") from None
    if total < 1 or not 1 <= index <= total:
        raise ValueError(f"MCP_TEST_SHARD index must be within 1..total, got {spec!r}")
    return index, total


def shard_of(nodeid: str, total: int) -> int:
    """Return the 1-based shard that owns the test file of ``nodeid``."""
    path = nodeid.split("::", 1)[0]
    return zlib.crc32(path.encode("utf-8")) % total + 1
