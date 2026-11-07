"""The release trust check must reject actual incident roots, not their labels."""

import base64
import importlib.util
import os
import ssl
from pathlib import Path

import certifi
import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/verify_tls_trust.py"

# Published incident root: https://emudhra.we-have-a-problem.com/roots.json
COMPROMISED_ROOT = """-----BEGIN CERTIFICATE-----
MIICTjCCAdOgAwIBAgIKPPYHqWhwDtqLhDAKBggqhkjOPQQDAzBrMQswCQYDVQQG
EwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNo
bm9sb2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0g
RzMwHhcNMTgwMjE4MTgzMDAwWhcNNDMwMjE4MTgzMDAwWjBrMQswCQYDVQQGEwJJ
TjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNobm9s
b2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0gRzMw
djAQBgcqhkjOPQIBBgUrgQQAIgNiAAQjpQy4LRL1KPOxst3iAhKAnjlfSU2fySU0
WXTsuwYc58Byr+iuL+FBVIcUqEqy6HyC5ltqtdyzdc6LBtCGI79G1Y4PPwT01xyS
fvalY8L1X44uT6EYGQIrMgqCZH0Wk9GjQjBAMB0GA1UdDgQWBBR8XQKEE9TMipuB
zhccLikenEhjQjAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zAKBggq
hkjOPQQDAwNpADBmAjEAvvNhzwIQHWSVB7gYboiFBS+DCBeQyh+KTOgNG3qxrdWB
CUfvO6wIBHxcmbHtRwfSAjEAnbpV/KlK6O3t5nYBQnvI+GDZjVGLVTv7jHvrZQnD
+JbNR6iC8hZVdyR+EhCVBCyj
-----END CERTIFICATE-----
"""

AAA_ROOT = """-----BEGIN CERTIFICATE-----
MIIEMjCCAxqgAwIBAgIBATANBgkqhkiG9w0BAQUFADB7MQswCQYDVQQGEwJHQjEb
MBkGA1UECAwSR3JlYXRlciBNYW5jaGVzdGVyMRAwDgYDVQQHDAdTYWxmb3JkMRow
GAYDVQQKDBFDb21vZG8gQ0EgTGltaXRlZDEhMB8GA1UEAwwYQUFBIENlcnRpZmlj
YXRlIFNlcnZpY2VzMB4XDTA0MDEwMTAwMDAwMFoXDTI4MTIzMTIzNTk1OVowezEL
MAkGA1UEBhMCR0IxGzAZBgNVBAgMEkdyZWF0ZXIgTWFuY2hlc3RlcjEQMA4GA1UE
BwwHU2FsZm9yZDEaMBgGA1UECgwRQ29tb2RvIENBIExpbWl0ZWQxITAfBgNVBAMM
GEFBQSBDZXJ0aWZpY2F0ZSBTZXJ2aWNlczCCASIwDQYJKoZIhvcNAQEBBQADggEP
ADCCAQoCggEBAL5AnfRu4ep2hxxNRUSOvkbIgwadwSr+GB+O5AL686tdUIoWMQua
BtDFcCLNSS1UY8y2bmhGC1Pqy0wkwLxyTurxFa70VJoSCsN6sjNg4tqJVfMiWPPe
3M/vg4aijJRPn2jymJBGhCfHdr/jzDUsi14HZGWCwEiwqJH5YZ92IFCokcdmtet4
YgNW8IoaE+oxox6gmf049vYnMlhvB/VruPsUK6+3qszWY19zjNoFmag4qMsXeDZR
rOme9Hg6jc8P2ULimAyrL58OAd7vn5lJ8S3frHRNG5i1R8XlKdH5kBjHYpy+g8cm
ez6KJcfA3Z3mNWgQIJ2P2N7Sw4ScDV7oL8kCAwEAAaOBwDCBvTAdBgNVHQ4EFgQU
oBEKIz6W8Qfs4q8p74Klf9AwpLQwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQF
MAMBAf8wewYDVR0fBHQwcjA4oDagNIYyaHR0cDovL2NybC5jb21vZG9jYS5jb20v
QUFBQ2VydGlmaWNhdGVTZXJ2aWNlcy5jcmwwNqA0oDKGMGh0dHA6Ly9jcmwuY29t
b2RvLm5ldC9BQUFDZXJ0aWZpY2F0ZVNlcnZpY2VzLmNybDANBgkqhkiG9w0BAQUF
AAOCAQEACFb8AvCb6P+k+tZ7xkSAzk/ExfYAWMymtrwUSWgEdujm7l3sAg9g1o1Q
GE8mTgHj5rCl7r+8dFRBv/38ErjHT1r0iWAFf2C3BUrz9vHCv8S5dIa2LX1rzNLz
Rt0vxuBqw8M0Ayx9lt1awg6nCpnBBYurDC/zXDrPbDdVCYfeU0BsWO/8tqtlbgT2
G9w84FoVxp7Z8VlIMCFlA2zs6SFz7JsDoeA3raAVGI/6ugLOpyypEBMs1OUIJqsi
l2D4kF501KKaU73yqWjgom7C12yxow+ev+to51byrvLjKzg6CYG1a4XXvi3tPxq3
smPi9WIsgtRqAEFQ8TmDn5XpNpaYbg==
-----END CERTIFICATE-----
"""


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("verify_tls_trust", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejects_real_incident_root_without_subject_comments(check, tmp_path):
    bundle = tmp_path / "roots.pem"
    bundle.write_text(COMPROMISED_ROOT)
    count, rejected = check.inspect_bundle(bundle)
    assert count == 1
    assert rejected == {"emSign ECC Root CA - G3"}


def test_rejects_same_root_with_different_pem_line_wrapping(check, tmp_path):
    der = ssl.PEM_cert_to_DER_cert(COMPROMISED_ROOT)
    payload = base64.b64encode(der).decode("ascii")
    bundle = tmp_path / "rewrapped.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n" + payload + "\n-----END CERTIFICATE-----\n")
    assert check.inspect_bundle(bundle)[1] == {"emSign ECC Root CA - G3"}


def test_healthy_certificate_is_not_rejected_for_comment_name(check, tmp_path):
    pem = Path(certifi.where()).read_text()
    first = check.PEM_CERTIFICATE.search(pem).group()
    bundle = tmp_path / "safe.pem"
    bundle.write_text("# emSign ECC Root CA - G3\n" + first)
    assert check.inspect_bundle(bundle) == (1, set())


@pytest.mark.parametrize("content", ["", "not a CA bundle", "-----BEGIN CERTIFICATE-----\n!\n-----END CERTIFICATE-----"])
def test_missing_or_invalid_certificates_fail_closed(check, tmp_path, content):
    bundle = tmp_path / "invalid.pem"
    bundle.write_text(content)
    with pytest.raises(ValueError):
        check.inspect_bundle(bundle)


def test_cli_fails_for_bad_bundle(check, tmp_path, capsys):
    bundle = tmp_path / "roots.pem"
    bundle.write_text(COMPROMISED_ROOT)
    assert check.main(["--bundle", str(bundle)]) == 1
    assert "emSign ECC Root CA - G3" in capsys.readouterr().out


def test_rejects_real_cross_sign_root(check, tmp_path):
    bundle = tmp_path / "aaa.pem"
    bundle.write_text(AAA_ROOT)
    assert check.inspect_bundle(bundle) == (1, {"AAA Certificate Services"})


def test_bundle_only_does_not_require_installed_packages(check, tmp_path, monkeypatch):
    pem = Path(certifi.where()).read_text()
    bundle = tmp_path / "safe.pem"
    bundle.write_text(check.PEM_CERTIFICATE.search(pem).group())

    def unexpected_import(name):
        pytest.fail(f"bundle-only check imported {name}")

    monkeypatch.setattr(check.importlib, "import_module", unexpected_import)
    assert check.main(["--bundle", str(bundle)]) == 0


def test_required_missing_pip_fails(check, tmp_path, monkeypatch, capsys):
    pem = Path(certifi.where()).read_text()
    bundle = tmp_path / "safe.pem"
    bundle.write_text(check.PEM_CERTIFICATE.search(pem).group())

    def missing_pip(name):
        raise ModuleNotFoundError("No module named 'pip'", name="pip")

    monkeypatch.setattr(check.importlib, "import_module", missing_pip)
    assert check.main(["--bundle", str(bundle), "--require-pip"]) == 1
    assert "FAIL pip._vendor.certifi" in capsys.readouterr().out


def test_system_checks_effective_loaded_certificates(check, tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    pem = Path(certifi.where()).read_text()
    bundle = tmp_path / "safe.pem"
    bundle.write_text(check.PEM_CERTIFICATE.search(pem).group())
    context = SimpleNamespace(get_ca_certs=lambda binary_form: [ssl.PEM_cert_to_DER_cert(AAA_ROOT)])
    monkeypatch.setattr(check.ssl, "create_default_context", lambda: context)
    assert check.main(["--bundle", str(bundle), "--system"]) == 1
    assert "FAIL system: trusts AAA Certificate Services" in capsys.readouterr().out


def test_system_checks_lazy_ca_directory_roots(check, tmp_path, monkeypatch, capsys):
    pem = Path(certifi.where()).read_text()
    bundle = tmp_path / "safe.pem"
    bundle.write_text(check.PEM_CERTIFICATE.search(pem).group())
    safe_directory = tmp_path / "empty-certs"
    safe_directory.mkdir()
    directory = tmp_path / "lazy-certs"
    directory.mkdir()
    # OpenSSL loads hashed CApath files only on demand. This is the actual
    # subject hash of the published emSign root, as returned by openssl x509.
    (directory / "14bc7599.0").write_text(COMPROMISED_ROOT)
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.setenv("SSL_CERT_DIR", os.pathsep.join([str(safe_directory), str(directory)]))

    assert check.main(["--bundle", str(bundle), "--system"]) == 1
    assert "FAIL system: trusts emSign ECC Root CA - G3" in capsys.readouterr().out
