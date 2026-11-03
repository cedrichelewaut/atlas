"""Shared fixtures: a live `atlas demo` server on a free port with an isolated ATLAS_HOME."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest

# In-process tests (units) use an isolated home and the bundled sample, never ~/.atlas or a dataset you imported.
os.environ["ATLAS_HOME"] = tempfile.mkdtemp(prefix="atlas-test-home-")
os.environ["AGENT_PROVIDER"] = "none"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
os.environ.pop("ATLAS_DATA", None)

TOKEN = "test-token-0123456789abcdef"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="session")
def atlas_env(tmp_path_factory):
    e = dict(os.environ)
    e["ATLAS_HOME"] = str(tmp_path_factory.mktemp("atlas_home"))
    e["ATLAS_TOKEN"] = TOKEN
    e["AGENT_PROVIDER"] = "none"  # rules/command mode only: tests never call an LLM
    e["OLLAMA_HOST"] = "http://127.0.0.1:9"  # nothing listens there
    e.pop("ATLAS_DATA", None)
    return e


@pytest.fixture(scope="session")
def server(atlas_env):
    """Base URL of a running demo server (bundled Gutenberg sample)."""
    port = free_port()
    p = subprocess.Popen(
        [sys.executable, "-m", "atlas_books.cli", "demo", "--port", str(port), "--no-browser"],
        env=atlas_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(base + "/api/config", timeout=2).read()
                break
            except Exception:
                if p.poll() is not None:
                    pytest.fail("server exited: " + (p.stdout.read() if p.stdout else ""))
                time.sleep(0.5)
        else:
            pytest.fail("server did not start")
        yield base
    finally:
        p.terminate()
        p.wait(10)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_opener = urllib.request.build_opener(NoRedirect)


def request(base, path, method="GET", body=None, headers=None, raw=None, timeout=60):
    """(status, headers, body bytes) without following redirects or raising on HTTP errors."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    rq = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with _opener.open(rq, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def api_post(base, path, body=None):
    """A POST the way the bundled page sends it: JSON + the per-launch token."""
    return request(
        base, path, "POST", body if body is not None else {}, {"Content-Type": "application/json", "X-Atlas-Token": TOKEN}
    )
