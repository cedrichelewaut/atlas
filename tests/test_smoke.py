"""Smoke tests: import every input format, then serve the bundled sample and hit the API."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import pytest

HERE = Path(__file__).parent
FIX = HERE / "fixtures"


def _atlas(*args, env):
    return subprocess.run([sys.executable, "-m", "atlas_books.cli", *args], env=env, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = dict(os.environ)
    e["ATLAS_HOME"] = str(tmp_path_factory.mktemp("atlas_home"))
    return e


@pytest.mark.parametrize(
    "fname,expect", [("goodreads_export_synthetic.csv", 60), ("generic_semicolon.csv", 118), ("books.json", 199)]
)
def test_import_formats(env, fname, expect):
    r = _atlas("import", str(FIX / fname), "--no-switch", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    name = fname.rsplit(".", 1)[0].replace("_", "-")
    meta = json.loads((Path(env["ATLAS_HOME"]) / "datasets" / name / "meta.json").read_text())
    assert meta["n"] == expect


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_serve_sample(env):
    port = _free_port()
    p = subprocess.Popen([sys.executable, "-m", "atlas_books.cli", "demo", "--port", str(port), "--no-browser"], env=env)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                cfg = json.load(urllib.request.urlopen(base + "/api/config", timeout=2))
                break
            except Exception:
                time.sleep(1)
        else:
            pytest.fail("server did not start")
        assert cfg["total"] > 2000 and "Gutenberg" in cfg["subtitle"]
        assert b"Made with" in urllib.request.urlopen(base + "/").read()
        rq = urllib.request.Request(
            base + "/api/chat",
            json.dumps({"messages": [{"role": "user", "content": "books like Dracula"}]}).encode(),
            {"Content-Type": "application/json"},
        )
        reply = json.load(urllib.request.urlopen(rq, timeout=60))["reply"]
        assert "Dracula" in reply and "Map updated" in reply  # rules mode works without any LLM
        top = json.load(urllib.request.urlopen(base + "/api/top"))["books"]
        assert top and top[0]["url"].startswith("https://www.gutenberg.org/")
    finally:
        p.terminate()
        p.wait(10)
