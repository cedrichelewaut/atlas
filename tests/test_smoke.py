"""Smoke tests: import every input format, then serve the bundled sample and hit the API."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import api_post, request

HERE = Path(__file__).parent
FIX = HERE / "fixtures"


def _atlas(*args, env):
    return subprocess.run([sys.executable, "-m", "atlas_books.cli", *args], env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize(
    "fname,expect", [("goodreads_export_synthetic.csv", 60), ("generic_semicolon.csv", 118), ("books.json", 199)]
)
def test_import_formats(atlas_env, fname, expect):
    r = _atlas("import", str(FIX / fname), "--no-switch", env=atlas_env)
    assert r.returncode == 0, r.stderr[-2000:]
    name = fname.rsplit(".", 1)[0].replace("_", "-")
    meta = json.loads((Path(atlas_env["ATLAS_HOME"]) / "datasets" / name / "meta.json").read_text())
    assert meta["n"] == expect


def test_serve_sample(server):
    cfg = json.loads(request(server, "/api/config")[2])
    assert cfg["total"] > 2000 and "Gutenberg" in cfg["subtitle"]
    assert b"Made with" in request(server, "/")[2]
    st, _, body = api_post(server, "/api/chat", {"messages": [{"role": "user", "content": "books like Dracula"}]})
    reply = json.loads(body)["reply"]
    assert "Dracula" in reply and "Map updated" in reply  # rules mode works without any LLM
    top = json.loads(request(server, "/api/top")[2])["books"]
    assert top and top[0]["url"].startswith("https://www.gutenberg.org/")
