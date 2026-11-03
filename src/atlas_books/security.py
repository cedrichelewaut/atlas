"""Hardening for the local Atlas server.

Atlas has no accounts: it is meant to run on your own machine. These checks stop *other websites* you visit from
driving it through your browser (CSRF, DNS rebinding) and keep external tools explicit:

- Host check: when bound to a loopback address, requests must be addressed to localhost / 127.0.0.1 / ::1
  (plus anything in ATLAS_ALLOWED_HOSTS). Blocks DNS-rebinding pages.
- Every state-changing request (POST, PUT, PATCH, DELETE) must be same-origin: a foreign Origin header or
  `Sec-Fetch-Site: cross-site` is rejected.
- POSTs to /api/* must be `Content-Type: application/json` (HTML forms and `no-cors` fetches cannot send that) and carry
  the per-launch token in the `X-Atlas-Token` header. The page gets the token from the HTML it is served; external
  agents read it from the startup log or from `$ATLAS_HOME/api_token` (mode 0600). ATLAS_TOKEN pins a fixed token.
"""

from __future__ import annotations

import ipaddress
import json
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

HEADER = "x-atlas-token"
TOKEN = os.environ.get("ATLAS_TOKEN") or secrets.token_urlsafe(24)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def is_loopback(host: str | None) -> bool:
    h = (host or "").strip("[]").lower()
    if h in LOOPBACK_NAMES or h.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def is_private(host: str | None) -> bool:
    """Loopback, RFC 1918 / link-local / ULA addresses and *.local names: traffic that stays on your network."""
    h = (host or "").strip("[]").lower()
    if is_loopback(h) or h.endswith((".local", ".lan", ".internal")) or h == "host.docker.internal":
        return True
    try:
        ip = ipaddress.ip_address(h)
        return ip.is_private or ip.is_link_local
    except ValueError:
        return False


def bind_host() -> str:
    return os.environ.get("ATLAS_BIND_HOST", "127.0.0.1")


def allowed_hosts() -> set[str] | None:
    """Host names the server answers to, or None for "any" (bound to a non-loopback address without an explicit list)."""
    extra = {h.strip().lower() for h in os.environ.get("ATLAS_ALLOWED_HOSTS", "").split(",") if h.strip()}
    b = bind_host()
    if is_loopback(b):
        return LOOPBACK_NAMES | extra
    if extra:
        return LOOPBACK_NAMES | extra | ({b.lower()} if b not in ("0.0.0.0", "::") else set())
    return None


def _hostname(host_header: str) -> str:
    h = host_header.strip().lower()
    if h.startswith("["):
        return h[1 : h.index("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def write_token_file(home: Path) -> Path:
    p = home / "api_token"
    fd = os.open(str(p) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(TOKEN + "\n")
    os.replace(str(p) + ".tmp", p)
    os.chmod(p, 0o600)
    return p


def check_llm_url(base: str) -> tuple[str | None, str | None]:
    """Validate an OpenAI-compatible base URL. Returns (error, warning); error=None means it can be used."""
    base = (base or "").strip()
    try:
        u = urlsplit(base)
    except ValueError:
        return "base URL is not a valid URL", None
    if u.scheme not in ("http", "https"):
        return "base URL must start with http:// or https://", None
    if not u.hostname:
        return "base URL needs a host name", None
    if u.username or u.password:
        return "put the API key in the key field, not in the URL", None
    if u.query or u.fragment:
        return "base URL must not contain ?query or #fragment", None
    warn = None
    if not is_private(u.hostname):
        warn = f"{u.hostname} is not on this machine or your network: Atlas AI will send your chat, the current view and your reading profile to it."
        if u.scheme == "http":
            warn += " The connection is unencrypted (http://)."
    return None, warn


class Guard:
    """Pure ASGI middleware (keeps server-sent events streaming) applying the checks in the module docstring."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        method = scope.get("method", "GET").upper()
        path = scope.get("path", "")
        host = h.get("host", "")
        allowed = allowed_hosts()
        if allowed is not None and _hostname(host) not in allowed:
            return await _deny(send, 400, f"Host {host!r} is not allowed (set ATLAS_ALLOWED_HOSTS to add one)")
        if method not in SAFE_METHODS:
            origin = h.get("origin")
            if origin is not None and (origin == "null" or urlsplit(origin).netloc.lower() != host.lower()):
                return await _deny(send, 403, "cross-origin request blocked")
            if h.get("sec-fetch-site") == "cross-site":
                return await _deny(send, 403, "cross-site request blocked")
            if path.startswith("/api/"):
                if not h.get("content-type", "").split(";")[0].strip().lower() == "application/json":
                    return await _deny(send, 415, "POST bodies must be Content-Type: application/json")
                if not secrets.compare_digest(h.get(HEADER, ""), TOKEN):
                    return await _deny(
                        send,
                        403,
                        "missing or wrong X-Atlas-Token header (the token is printed when Atlas starts and saved in "
                        "$ATLAS_HOME/api_token; reload the page after a restart)",
                        stale=True,
                    )
        return await self.app(scope, receive, send)


async def _deny(send, status: int, detail: str, stale: bool = False):
    body = json.dumps({"detail": detail} | ({"stale": True} if stale else {})).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
