# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Single source of truth for OAuth redirect-URI handling.

Three concerns, one module:

* ``validate_registration_redirect_uris`` — scheme allowlist enforced once, at
  client registration (RFC 7591). This is where untrusted, attacker-controlled
  URIs enter the system.
* ``resolve_redirect_uri`` — at ``/authorize``/``/token`` time, match the
  requested URI against the values the client already registered and return the
  trusted stored value (or the runtime loopback URI for native apps).
* ``build_redirect_url`` — assemble the final redirect URL from an
  already-resolved URI.

Because every URI used to build a redirect has already passed the registration
allowlist and the authorize-time match, ``build_redirect_url`` does not need a
second scheme denylist — that pass was dead code (it could never be reached
with a dangerous scheme) and has been removed.
"""

from typing import List, Optional
from urllib.parse import urlencode, urlparse, ParseResult

from fastapi import HTTPException, status

from .limits import MAX_REDIRECT_URI_LEN

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Schemes accepted at registration. Whitelist approach: anything not listed is
# rejected. http is additionally constrained to loopback hosts below.
_ALLOWED_REGISTRATION_SCHEMES = {
    "https",
    "http",
    # IDE deep-link schemes for OAuth callback in editor extensions
    "cursor",
    "vscode",
    "vscode-insiders",
    # Native app custom schemes (common patterns)
    "com.example.app",
    "myapp",
}


def _invalid_uri(description: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "invalid_redirect_uri", "error_description": description},
    )


def reject_ambiguous_redirect_uri(uri: str) -> None:
    """Refuse a redirect URI that a browser reads differently from ``urlsplit``.

    Browsers treat a backslash in an http(s) URL as a slash and drop tabs and
    newlines, so ``https://evil.example\\@claude.ai/cb`` parses here with the
    host ``claude.ai`` and navigates to ``evil.example``. No legitimate
    redirect URI carries a backslash, whitespace or a control character.
    """
    if any(ch == "\\" or ch <= " " or ch == "\x7f" for ch in uri):
        raise _invalid_uri(
            "redirect URI must not contain backslashes, whitespace or control characters"
        )


def validate_registration_redirect_uris(redirect_uris: Optional[List[str]]) -> None:
    """
    Validate redirect URIs supplied at client registration (RFC 7591).

    Uses proper URL parsing (not string matching) to prevent bypass attacks and
    enforces a scheme whitelist. http is restricted to loopback hosts; https
    accepts any host; listed custom schemes are allowed for native/IDE clients.
    """
    if not redirect_uris:
        return

    for uri in redirect_uris:
        uri_str = str(uri).strip()

        if not uri_str:
            raise _invalid_uri("Empty redirect URI not allowed")

        if len(uri_str) > MAX_REDIRECT_URI_LEN:
            raise _invalid_uri(
                f"redirect URI exceeds maximum length of {MAX_REDIRECT_URI_LEN} characters"
            )

        try:
            parsed: ParseResult = urlparse(uri_str)
        except ValueError as exc:
            raise _invalid_uri(f"Invalid URL format: {uri_str}. Error: {exc}")

        scheme = parsed.scheme.lower()
        if not scheme:
            raise _invalid_uri(f"Missing scheme in redirect URI: {uri_str}")

        if scheme == "http":
            hostname = parsed.hostname
            if not parsed.netloc or not hostname:
                raise _invalid_uri(f"HTTP URI missing host: {uri_str}")
            if hostname.lower() not in _LOOPBACK_HOSTS:
                raise _invalid_uri(
                    "HTTP redirect URIs must use localhost, 127.0.0.1, or ::1. "
                    f"Got: {hostname}"
                )
        elif scheme == "https":
            if not parsed.netloc:
                raise _invalid_uri(f"HTTPS URI missing host: {uri_str}")
        elif scheme not in _ALLOWED_REGISTRATION_SCHEMES:
            raise _invalid_uri(
                f"Unsupported scheme '{parsed.scheme}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_REGISTRATION_SCHEMES))}"
            )


def _is_loopback_http_redirect(parsed: ParseResult) -> bool:
    """Return True when the parsed URI is an HTTP loopback redirect."""
    return (
        parsed.scheme.lower() == "http"
        and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS
    )


def loopback_redirect_matches(registered_uri: str, requested_uri: str) -> bool:
    """
    Match native-app loopback redirects while allowing runtime-assigned ports.

    RFC 8252 recommends loopback redirects for native apps and requires OAuth
    servers to tolerate ephemeral localhost ports. Keep scheme/path strict while
    allowing host aliases inside the loopback set and any runtime port.
    """
    registered = urlparse(registered_uri)
    requested = urlparse(requested_uri)

    if not (
        _is_loopback_http_redirect(registered) and _is_loopback_http_redirect(requested)
    ):
        return False

    return (
        registered.path == requested.path
        and registered.params == requested.params
        and registered.query == requested.query
        and registered.fragment == requested.fragment
    )


def resolve_registered_redirect_uri(
    registered_uris: List[str], redirect_uri: Optional[str]
) -> str:
    """
    Resolve a requested redirect URI against a client's registered set.

    Returns the trusted stored value on exact match, the runtime URI for
    native-app loopback callbacks (ephemeral ports), or the first registered
    URI when the request omits one. Raises 400 otherwise.
    """
    if not redirect_uri:
        if not registered_uris:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "redirect_uri is required when client has no registered redirect URIs",
                },
            )
        return registered_uris[0]

    if len(redirect_uri) > MAX_REDIRECT_URI_LEN:
        raise _invalid_uri(
            f"redirect_uri exceeds maximum length of {MAX_REDIRECT_URI_LEN} characters"
        )

    # Exact match — return the stored (trusted) value, not the user-supplied one.
    for registered_uri in registered_uris:
        if registered_uri == redirect_uri:
            return registered_uri

    # Native-app loopback redirects use runtime-assigned ports.
    for registered_uri in registered_uris:
        if loopback_redirect_matches(registered_uri, redirect_uri):
            return redirect_uri

    raise _invalid_uri("redirect_uri not registered for this client")


def build_redirect_url(redirect_uri: str, params: dict[str, str]) -> str:
    """Build a redirect URL from an already-resolved (trusted) redirect URI."""
    return f"{redirect_uri}?{urlencode(params)}"
