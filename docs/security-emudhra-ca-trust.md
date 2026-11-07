# eMudhra certificate trust update

Eight eMudhra intermediate CA keys were compromised in October 2026. Clients
using older trust stores can accept forged HTTPS certificates. This is a trust
store incident; it does not establish that this project's source or credentials
were compromised.

Update each independent trust store used by your installation:

- `certifi>=2026.10.19` for application clients that use certifi.
- `pip>=26.2.2` and `uv>=0.13.4` for the package managers' bundled trust stores.
- The operating system's security update for `ca-certificates`, and any custom
  bundles selected with `SSL_CERT_FILE`, `SSL_CERT_DIR`, or `PIP_CERT`.

Updating the application or certifi alone does not update pip, uv, or the operating
system. Bootstrap these updates through a trust store you have already updated;
an old package manager can still trust the compromised chains during its own
upgrade. Do not disable certificate validation to install the updates.

Rebuild Docker images with `--pull --no-cache` to refresh the base image and system
certificates. The maintained Dockerfiles check system trust before bootstrapping
pip, and check application, pip, and system trust after installing dependencies.
The build fails if an affected root remains trusted.

See the [incident report and certificate-chain tests](https://emudhra.we-have-a-problem.com/),
the [uv advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62),
the [certifi advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2),
and the [pip release notes](https://pip.pypa.io/en/stable/news/).
