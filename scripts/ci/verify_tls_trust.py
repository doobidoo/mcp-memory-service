#!/usr/bin/env python3
"""Reject the October 2026 eMudhra incident roots in release CA stores.

Offline check of certificate DER fingerprints, independent of PEM comments.
Source: https://emudhra.we-have-a-problem.com/roots.json
With --bundle, inspect only the supplied PEM files (suitable before pip exists).
Otherwise inspect certifi and pip's independent vendored bundle, when installed.
--require-pip makes the pip bundle mandatory; --system also checks Python's
effective default OpenSSL CA store, including SSL_CERT_FILE/SSL_CERT_DIR overrides.
This checks trust anchors, not OS revocation blocklists or live TLS connections.
"""

import argparse
import hashlib
import importlib
import os
import re
import ssl
from pathlib import Path

DISTRUSTED_ROOTS = {
    "86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B": "emSign ECC Root CA - G3",
    "D7A7A0FB5D7E2731D771E9484EBCDEF71D5F0C3E0A2948782BC83EE0EA699EF4": "AAA Certificate Services",
}
PEM_CERTIFICATE = re.compile(
    r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL,
)


def inspect_der(certificates):
    """Return certificate count and names of incident roots still trusted."""
    certificates = list(certificates)
    if not certificates:
        raise ValueError("CA store contains no certificates")
    fingerprints = {hashlib.sha256(der).hexdigest().upper() for der in certificates}
    return len(certificates), {DISTRUSTED_ROOTS[fp] for fp in fingerprints & DISTRUSTED_ROOTS.keys()}


def inspect_bundle(path):
    """Validate a PEM CA file and inspect the actual certificates it contains."""
    pem = Path(path).read_text(encoding="ascii")
    blocks = PEM_CERTIFICATE.findall(pem)
    if (not blocks or len(blocks) != pem.count("-----BEGIN CERTIFICATE-----")
            or len(blocks) != pem.count("-----END CERTIFICATE-----")):
        raise ValueError("CA bundle has missing or incomplete certificates")
    try:
        # Reject malformed base64/DER rather than treating unparsed data as safe.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cadata="\n".join(blocks))
        return inspect_der(ssl.PEM_cert_to_DER_cert(block) for block in blocks)
    except (ssl.SSLError, ValueError) as exc:
        raise ValueError(f"invalid CA bundle: {exc}") from exc


def inspect_system():
    """Inspect eager anchors and CApath roots OpenSSL otherwise loads lazily."""
    certificates = ssl.create_default_context().get_ca_certs(binary_form=True)
    count, rejected = inspect_der(certificates) if certificates else (0, set())
    paths = ssl.get_default_verify_paths()
    # get_default_verify_paths().capath can be None for a path-separated list,
    # even though OpenSSL uses every directory in SSL_CERT_DIR.
    directories = os.environ.get(paths.openssl_capath_env, paths.openssl_capath)
    for directory in (directories or "").split(os.pathsep):
        if not directory or not Path(directory).is_dir():
            continue
        for path in Path(directory).iterdir():
            # Hash symlinks/files are trust anchors; .rN files are CRLs.
            if re.fullmatch(r"[0-9a-fA-F]{8}\.\d+", path.name):
                found, blocked = inspect_bundle(path)
                count += found
                rejected.update(blocked)
    if not count:
        raise ValueError("CA store contains no certificates")
    return count, rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", default=[], metavar="PEM",
                        help="check only these files; repeat for multiple bundles")
    parser.add_argument("--require-pip", action="store_true",
                        help="also require pip's vendored CA bundle")
    parser.add_argument("--system", action="store_true",
                        help="also check Python's effective default OpenSSL CA store")
    args = parser.parse_args(argv)
    checks = [(str(path), lambda path=path: inspect_bundle(path)) for path in args.bundle]
    failed = False

    for name in ([] if args.bundle else ["certifi"]) + (["pip._vendor.certifi"] if not args.bundle or args.require_pip else []):
        try:
            module = importlib.import_module(name)
            path = module.where()
            checks.append((name, lambda path=path: inspect_bundle(path)))
        except ModuleNotFoundError as exc:
            if name == "pip._vendor.certifi" and exc.name == "pip" and not args.require_pip:
                print("SKIP pip: not installed")
            else:
                print(f"FAIL {name}: {exc}")
                failed = True

    if args.system:
        checks.append(("system", inspect_system))

    for name, inspect in checks:
        try:
            count, rejected = inspect()
            if rejected:
                print(f"FAIL {name}: trusts {', '.join(sorted(rejected))}")
                failed = True
            else:
                print(f"PASS {name}: {count} certificates; incident roots absent")
        except (OSError, UnicodeError, ValueError, ssl.SSLError) as exc:
            print(f"FAIL {name}: {exc}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
