"""Regression tests for scripts/ci/check_md_links.py."""

from __future__ import annotations

import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "check_md_links.py"
_spec = importlib.util.spec_from_file_location("check_md_links", SCRIPT)
links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(links)


def test_extract_skips_code_and_keeps_line_numbers():
    text = (
        "intro [a](docs/a.md)\n"
        "```bash\n"
        "[not](a-link.md)\n"
        "```\n"
        "inline `[nor](this.md)` here\n"
        "[ref]: ../b.md\n"
        '<a href="c.md">c</a> ![img](img/x.png)\n'
        "- <https://pypi.org/project/x/>\n"
    )
    assert links.extract_links(text) == [
        (1, "docs/a.md"),
        (6, "../b.md"),
        (7, "c.md"),
        (7, "img/x.png"),
        (8, "https://pypi.org/project/x/"),
    ]


@pytest.mark.parametrize("target, expected", [
    ("../x.md#part", "x.md"),
    ("/docs/y.md", "docs/y.md"),
    ("https://github.com/doobidoo/mcp-memory-service/blob/main/src/z.py#L3", "src/z.py"),
    ("https://github.com/doobidoo/mcp-memory-service/tree/main/docs", "docs"),
])
def test_internal_target_resolves(target, expected):
    path = links.internal_target("sub/README.md", target)
    assert path.resolve() == (links.REPO_ROOT / expected).resolve()


@pytest.mark.parametrize("target", [
    "https://example.org/x", "mailto:a@b.c", "#anchor",
    "https://github.com/doobidoo/mcp-memory-service/issues/1",
])
def test_internal_target_skips_external_and_anchors(target):
    assert links.internal_target("README.md", target) is None


def test_check_internal_reports_only_missing_targets(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text("x")
    (tmp_path / "README.md").write_text(
        "[ok](docs/ok.md)\n[gone](docs/gone.md)\n[web](https://example.org)\n")
    monkeypatch.setattr(links, "REPO_ROOT", tmp_path)
    assert links.check_internal(["README.md"]) == ["README.md:2: docs/gone.md"]


def test_deleted_but_unstaged_source_is_skipped(tmp_path, monkeypatch):
    # git ls-files still lists a file deleted from the working tree until the
    # deletion is staged; it has no links left to check.
    monkeypatch.setattr(links, "REPO_ROOT", tmp_path)
    assert links.check_internal(["gone.md"]) == []
    assert links.external_links(["gone.md"]) == {}


def test_external_links_skip_own_repo_local_auth_only_and_placeholders(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text(
        "[a](https://pypi.org/x#y) [b](https://pypi.org/x)\n"
        "[own](https://github.com/doobidoo/mcp-memory-service/blob/main/README.md)\n"
        "[local](http://localhost:8000/api) [ph](https://your-server/x)\n"
        "[alert](https://github.com/doobidoo/mcp-memory-service/security/code-scanning/428)\n"
        "[cb](https://codeberg.org/doobidoo/mcp-memory-service/issues/12)\n")
    monkeypatch.setattr(links, "REPO_ROOT", tmp_path)
    assert links.external_links(["README.md"]) == {
        "https://pypi.org/x": ["README.md:1", "README.md:1"],
    }


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        code = int(self.path.strip("/"))
        self.send_response(code)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("code, verdict", [
    (200, "ok"), (404, "broken"), (410, "broken"),
    (403, "warn"), (429, "warn"), (503, "warn"),
])
def test_probe_classifies_status(server, code, verdict):
    assert links.probe(f"{server}/{code}") == (verdict, code)


def test_probe_connection_refused_is_broken():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    srv.server_close()
    verdict, _ = links.probe(f"http://127.0.0.1:{port}/")
    assert verdict == "broken"
