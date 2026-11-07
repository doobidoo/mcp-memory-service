"""Importing the storage package must not load the Milvus backend.

Every deployment imports ``mcp_memory_service.storage`` (the web app does it
through ``web/dependencies.py``). When that import pulled in ``storage/milvus``
eagerly, a sqlite_vec or hybrid install without pymilvus logged
"pymilvus not available" at WARNING on every start, for a backend it never
selected. Each check runs in a fresh interpreter, because by the time pytest
gets here other tests have already imported the module.
"""

import subprocess
import sys


def _run(*lines: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "\n".join(lines)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_storage_package_import_does_not_load_milvus():
    result = _run(
        "import sys",
        "import mcp_memory_service.storage",
        'print("mcp_memory_service.storage.milvus" in sys.modules)',
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    assert "pymilvus not available" not in result.stderr


def test_milvus_class_still_reachable_from_package():
    result = _run(
        "from mcp_memory_service.storage import MilvusMemoryStorage",
        "from mcp_memory_service.storage.milvus import MilvusMemoryStorage as direct",
        "print(MilvusMemoryStorage is direct)",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_milvus_module_import_without_pymilvus_is_silent():
    # Block pymilvus so the check holds whether or not it is installed here.
    result = _run(
        "import sys",
        'sys.modules["pymilvus"] = None',
        "import logging",
        "logging.basicConfig(level=logging.WARNING)",
        "import mcp_memory_service.storage.milvus as m",
        "print(m.PYMILVUS_AVAILABLE)",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    assert "pymilvus not available" not in result.stderr
