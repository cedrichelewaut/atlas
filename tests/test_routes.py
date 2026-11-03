"""Every GET route of the app answers (the list comes from the live server's OpenAPI schema, so a new route without a
sample request here fails the test)."""

import json
import socket
import urllib.parse

from conftest import request

SAMPLES = {
    "/": "/",
    "/api/config": "/api/config?h=700",
    "/assets/wordmark-{v}.png": "/assets/wordmark-light.png",
    "/assets/fv-light.css": "/assets/fv-light.css",
    "/assets/{name}": "/assets/InterVariable.woff2",
    "/api/state": "/api/state",
    "/api/top": "/api/top?limit=5",
    "/api/librarian/model": "/api/librarian/model",
    "/api/profile": "/api/profile",
    "/api/tools": "/api/tools",
    "/api/map/overlay": "/api/map/overlay?x0=-100&x1=100&y0=-100&y1=100",
    "/api/perf": "/api/perf",
    "/api/map/meta": "/api/map/meta",
    "/api/map/points.bin": "/api/map/points.bin",
    "/api/map/bin": "/api/map/bin?x0=-100&x1=100&y0=-100&y1=100",
    "/go/book/{i}": "/go/book/0",
    "/api/book/{i}": "/api/book/0",
    "/api/search": "/api/search?q=dracula",
    "/api/librarian/parse": "/api/librarian/parse?q=" + urllib.parse.quote("books like Dracula after 1890"),
    "/api/events": None,  # server-sent events: checked separately below
}


def test_every_get_route_answers(server):
    st, _, body = request(server, "/openapi.json")
    assert st == 200
    paths = json.loads(body)["paths"]
    gets = {p for p, ops in paths.items() if "get" in ops}
    missing = gets - set(SAMPLES)
    assert not missing, f"add a sample request for {sorted(missing)}"
    for route, url in SAMPLES.items():
        if url is None:
            continue
        st, h, body = request(server, url)
        expect = 302 if route.startswith("/go/") else 200
        assert st == expect, (route, st, body[:300])
        if route == "/go/book/{i}":
            assert h["location"].startswith("https://www.gutenberg.org/")


def test_parse_endpoint(server):
    st, _, body = request(server, "/api/librarian/parse?q=" + urllib.parse.quote("books like Dracula after 1890"))
    d = json.loads(body)
    assert st == 200 and d["recognised"] and d["like"] == "Dracula" and d["filters"]["year"][0] == 1891


def test_events_stream_says_hello(server):
    u = urllib.parse.urlparse(server)
    s = socket.create_connection((u.hostname, u.port), timeout=10)
    s.sendall(f"GET /api/events HTTP/1.1\r\nHost: {u.netloc}\r\nAccept: text/event-stream\r\n\r\n".encode())
    buf = b""
    while b"data: " not in buf or b"\n\n" not in buf.split(b"data: ", 1)[1]:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    assert b"200" in buf.split(b"\r\n", 1)[0]
    assert b'"type": "hello"' in buf


def test_page_examples_come_from_the_dataset(server):
    body = request(server, "/")[2].decode()
    assert "{{" not in body and "Martian" not in body and "Connect Goodreads" not in body
    assert "zoom to science fiction" in body.lower()  # the Gutenberg sample has no ratings: no "4★+" example
    cfg = json.loads(request(server, "/api/config")[2])
    assert cfg["has_rating"] is False


def test_system_prompt_names_the_dataset():
    from atlas_books import agent

    p = agent.system_prompt("2,920 public-domain novels from Project Gutenberg", has_rating=False)
    assert "Project Gutenberg" in p and "Goodreads" not in p and "has no ratings" in p
    assert "482k" not in agent.SYSTEM_TEMPLATE
