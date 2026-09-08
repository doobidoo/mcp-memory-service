"""Verify scripts/ do not import through the src. prefix.

Issue #1172: scripts that import via `src.mcp_memory_service` load a shadow
copy of the package, bypassing any patches or singleton guards in the test
suite.  The fix is to import via `mcp_memory_service` directly.
"""

import subprocess
import sys


def test_no_src_prefix_imports_in_scripts():
    """No script in scripts/ should import through the src. prefix."""
    result = subprocess.run(
        [sys.executable, "-c", r"""
import pathlib, re, sys

pattern = re.compile(r'^\s*(?:from\s+src\.|import\s+src\.)', re.MULTILINE)
hits = []
for p in sorted(pathlib.Path('scripts').rglob('*.py')):
    text = p.read_text(errors='replace')
    for m in pattern.finditer(text):
        hits.append(f'{p}:{text[:m.start()].count(chr(10))+1}: {m.group().strip()}')

if hits:
    print('\n'.join(hits))
    sys.exit(1)
"""],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0, (
        f"scripts/ still contain src. prefix imports:\n{result.stdout}"
    )
