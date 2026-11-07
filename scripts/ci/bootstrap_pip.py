#!/usr/bin/env python3
"""Upgrade pip through a checked CA file using flags supported by the old pip."""

import argparse
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from verify_tls_trust import inspect_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", required=True, type=Path, metavar="PEM")
    args = parser.parse_args(argv)
    try:
        _, rejected = inspect_bundle(args.cert)
        if rejected:
            raise ValueError(f"CA bundle trusts {', '.join(sorted(rejected))}")
        version = metadata.version("pip")
        if not re.fullmatch(r"\d+(?:\.\d+)*", version):
            raise ValueError("cannot determine a numeric installed pip version")
        installed = tuple(int(part) for part in version.split("."))
        command = [sys.executable, "-m", "pip", "install", "--upgrade"]
        # pip <24.2 uses only the supplied CA file by default. Later pip unions
        # system and vendored roots unless legacy-certs disables that behavior.
        if (24, 2) <= installed < (26, 2, 2):
            command.append("--use-deprecated=legacy-certs")
        command.extend(["--cert", str(args.cert), "pip>=26.2.2"])
        return subprocess.run(command, check=False).returncode
    except (OSError, UnicodeError, ValueError, metadata.PackageNotFoundError) as exc:
        message = re.sub(r"[\x00-\x1f\x7f]", "", str(exc))
        print(f"FAIL bootstrap pip: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
