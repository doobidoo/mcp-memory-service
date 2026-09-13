import importlib.util
import plistlib
from pathlib import Path
from unittest.mock import MagicMock

from mcp_memory_service.cli import lifecycle

REPO_ROOT = Path(__file__).parents[2]


def test_self_signed_certificate_includes_configured_sans(
    tmp_path: Path, monkeypatch
) -> None:
    socket = MagicMock()
    socket.getsockname.return_value = ("192.0.2.20", 12345)
    socket_factory = MagicMock(return_value=socket)
    monkeypatch.setattr("socket.socket", socket_factory)
    run = MagicMock()
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    cert, key = lifecycle.generate_self_signed_certificate(
        cert_dir=tmp_path,
        additional_ips="192.0.2.10, 192.0.2.10",
        additional_hostnames="memory.example.test, localhost",
    )

    assert cert == str(tmp_path / "cert.pem")
    assert key == str(tmp_path / "key.pem")
    request_command = run.call_args_list[-1].args[0]
    san_argument = request_command[request_command.index("-addext") + 1]
    assert "IP:192.0.2.20" in san_argument
    assert san_argument.count("IP:192.0.2.10") == 1
    assert "DNS:memory.example.test" in san_argument
    assert san_argument.count("DNS:localhost") == 1
    socket.close.assert_called_once()


def test_legacy_script_uses_packaged_certificate_generator() -> None:
    script_path = REPO_ROOT / "scripts" / "server" / "run_http_server.py"
    spec = importlib.util.spec_from_file_location("run_http_server", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.generate_self_signed_certificate
        is lifecycle.generate_self_signed_certificate
    )


def test_documented_launchd_service_uses_lifecycle_cli() -> None:
    """Validate the canonical launchd recipe rather than a machine-specific copy."""
    guide = (REPO_ROOT / "docs" / "http-server-management.md").read_text()
    template = guide.split("```xml\n", 1)[1].split("\n```", 1)[0]
    config = plistlib.loads(template.encode())

    assert config["WorkingDirectory"] == "/path/to/repository"
    assert config["ProgramArguments"][0].endswith("/.venv/bin/memory")
    assert config["ProgramArguments"][-2:] == [
        "launch",
        "--foreground",
    ]


def test_default_certificate_uses_user_runtime_directory(tmp_path, monkeypatch):
    user_dir = tmp_path / "user-state"
    monkeypatch.setattr(lifecycle, "_data_dir", lambda: user_dir)
    monkeypatch.setattr(lifecycle, "_local_certificate_ip", lambda: None)
    monkeypatch.setattr(lifecycle.subprocess, "run", MagicMock())
    cert, key = lifecycle.generate_self_signed_certificate()
    assert Path(cert).parent == user_dir / "certs"
    assert Path(key).parent == user_dir / "certs"
