"""The local server refuses requests another website could make through your browser (see security.py)."""

import json

import pytest

from atlas_books import security
from conftest import TOKEN, api_post, request


def _profile(base):
    return json.loads(request(base, "/api/profile")[2])["text"]


def test_page_carries_token_and_is_not_cached(server):
    st, h, body = request(server, "/")
    assert st == 200 and f'<meta name="atlas-token" content="{TOKEN}">'.encode() in body
    assert h.get("cache-control") == "no-store"


def test_good_post_works(server):
    st, _, body = api_post(server, "/api/profile", {"text": "likes sea stories"})
    assert st == 200, body
    assert _profile(server) == "likes sea stories"


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Content-Type": "text/plain", "X-Atlas-Token": TOKEN}, 415),  # what a <form> or no-cors fetch can send
        ({"Content-Type": "application/x-www-form-urlencoded", "X-Atlas-Token": TOKEN}, 415),
        ({"Content-Type": "application/json"}, 403),  # no token
        ({"Content-Type": "application/json", "X-Atlas-Token": "guess"}, 403),
        ({"Content-Type": "application/json", "X-Atlas-Token": TOKEN, "Origin": "https://evil.example"}, 403),
        ({"Content-Type": "application/json", "X-Atlas-Token": TOKEN, "Origin": "null"}, 403),
        ({"Content-Type": "application/json", "X-Atlas-Token": TOKEN, "Sec-Fetch-Site": "cross-site"}, 403),
    ],
)
def test_forged_posts_are_refused(server, headers, status):
    before = _profile(server)
    for path in ("/api/profile", "/api/chat", "/api/tools/set_view", "/api/librarian/model", "/api/state"):
        st, _, _ = request(server, path, "POST", {"text": "pwned", "messages": [{"role": "user", "content": "reset"}]}, headers)
        assert st == status, (path, headers, st)
    assert _profile(server) == before


def test_dns_rebinding_host_is_refused(server):
    st, _, _ = request(server, "/api/state", headers={"Host": "attacker.example"})
    assert st == 400
    assert request(server, "/api/state", headers={"Host": "localhost"})[0] == 200


def test_same_origin_post_with_origin_header_is_fine(server):
    host = server.split("://", 1)[1]
    st, _, _ = request(
        server,
        "/api/tools/get_view_state",
        "POST",
        {},
        {"Content-Type": "application/json", "X-Atlas-Token": TOKEN, "Origin": server, "Host": host},
    )
    assert st == 200


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/v1",
        "http://user:pw@example.com/v1",
        "javascript:alert(1)",
        "https://",
        "gopher://x",
    ],
)
def test_bad_model_urls_are_rejected(server, url):
    st, _, body = api_post(server, "/api/librarian/model", {"kind": "openai", "base_url": url, "model": "m"})
    assert st == 400, body


def test_llm_url_check():
    assert security.check_llm_url("http://127.0.0.1:11434/v1") == (None, None)
    assert security.check_llm_url("http://192.168.1.20:8000/v1") == (None, None)
    err, warn = security.check_llm_url("https://api.example.com/v1")
    assert err is None and "api.example.com" in warn
    assert "unencrypted" in security.check_llm_url("http://api.example.com/v1")[1]
    assert security.check_llm_url("file:///etc/passwd")[0]
    assert security.check_llm_url("http://a:b@example.com")[0]


def test_host_helpers(monkeypatch):
    assert security.is_loopback("127.0.0.1") and security.is_loopback("[::1]") and security.is_loopback("localhost")
    assert not security.is_loopback("0.0.0.0")
    monkeypatch.setenv("ATLAS_BIND_HOST", "0.0.0.0")
    monkeypatch.delenv("ATLAS_ALLOWED_HOSTS", raising=False)
    assert security.allowed_hosts() is None
    monkeypatch.setenv("ATLAS_ALLOWED_HOSTS", "books.lan")
    assert "books.lan" in security.allowed_hosts()
    monkeypatch.setenv("ATLAS_BIND_HOST", "127.0.0.1")
    assert security.allowed_hosts() >= {"localhost", "127.0.0.1", "books.lan"}


def test_token_file_written(server, atlas_env):
    from pathlib import Path

    p = Path(atlas_env["ATLAS_HOME"]) / "api_token"
    assert p.read_text().strip() == TOKEN
    assert p.stat().st_mode & 0o777 == 0o600


def test_non_loopback_bind_warns(atlas_env, tmp_path):
    import subprocess
    import sys

    from conftest import free_port

    env = dict(atlas_env, ATLAS_HOME=str(tmp_path))
    p = subprocess.Popen(
        [sys.executable, "-m", "atlas_books.cli", "demo", "--host", "0.0.0.0", "--port", str(free_port()), "--no-browser"],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = p.stderr.readline()
    finally:
        p.terminate()
        p.wait(10)
    assert line.startswith("WARNING: Atlas is listening on 0.0.0.0")
