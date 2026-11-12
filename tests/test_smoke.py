"""Smoke tests: import every input format, then serve the fixture dataset and hit the API."""

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
    assert "n_jobs value" not in r.stderr  # UMAP's seed/parallelism warning is silenced
    name = fname.rsplit(".", 1)[0].replace("_", "-")
    meta = json.loads((Path(atlas_env["ATLAS_HOME"]) / "datasets" / name / "meta.json").read_text())
    assert meta["n"] == expect


def test_serve_dataset(server):
    cfg = json.loads(request(server, "/api/config")[2])
    assert cfg["total"] == 199 and "books.json" in cfg["subtitle"]
    assert b"Made with" in request(server, "/")[2]
    st, _, body = api_post(server, "/api/chat", {"messages": [{"role": "user", "content": "zoom out"}]})
    assert "Map updated" in json.loads(body)["reply"]  # rules mode works without any LLM
    top = json.loads(request(server, "/api/top")[2])["books"]
    assert top and top[0]["url"].startswith("https://")


def test_busy_port_falls_back(atlas_env):
    import socket

    from conftest import free_port

    port = free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    blocker.listen()
    p = subprocess.Popen(
        [sys.executable, "-m", "atlas_books.cli", "serve", "--port", str(port), "--no-browser"],
        env=atlas_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        first, second = p.stdout.readline(), p.stdout.readline()
    finally:
        p.terminate()
        p.wait(10)
        blocker.close()
    assert f"port {port} is already in use" in first and f"using {port + 1}" in first, first
    assert second.startswith(f"Atlas: serving testdata at http://127.0.0.1:{port + 1}/"), second


def test_embedding_model_cached_under_atlas_home(atlas_env):
    # test_import_formats ran first in this module and downloaded the model into $ATLAS_HOME/models
    models = Path(atlas_env["ATLAS_HOME"]) / "models"
    assert models.is_dir() and any(models.iterdir())
