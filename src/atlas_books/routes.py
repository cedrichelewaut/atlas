"""HTTP routes: the page, the shared-state API, Atlas AI chat/tools, map data for the WebGL layer."""

import os
import re
import json
import time
import asyncio
import threading
import uuid
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, RedirectResponse, Response
import flexviz as _fv
from flexviz import mount_into
from . import store, paths
from . import agent, rec, librarian
from .state import (
    ACK,
    B,
    CLIENTS,
    CORPUS,
    FIG,
    FOCUS,
    HAS_RATING,
    HERE,
    HIGHLIGHT,
    M,
    X0,
    X1,
    Y0,
    Y1,
    _log,
    bookrow,
    books_in_view,
    dash_url,
    describe,
    filt,
    filters_of,
    gr_url,
    mention_links,
    set_state,
    view_state,
)
from .tools import (
    _cur_vp,
    author_lookup,
    call_tool,
    recs_from_chat,
    search,
    t_recommend_answer,
    t_recommend_start,
    title_lookup,
    tool_list,
    top_rated,
    topic_lookup,
)

# ---------- HTTP ----------
app = FastAPI(title="Atlas")
mount_into(app, "/flexviz")


@app.get("/", response_class=HTMLResponse)
def index():
    """index.html with dataset-aware example prompts (the originals assume a large modern fiction catalogue)."""
    h = open(f"{HERE}/index.html").read()
    top = B.sort("ratings_count", descending=True, nulls_last=True).head(1)
    if top.height:
        ex = re.sub(r"\s*[(:;].*$", "", top["title"][0]).strip()[:40] or top["title"][0][:40]
        h = h.replace("The Martian", ex.replace('"', "").replace("'", ""))
    if not HAS_RATING:
        h = h.replace("only 4★+ sci-fi after 2010", "zoom to science fiction").replace(
            "Only highly rated sci-fi after 2010", "Zoom to science fiction"
        )
    if not B["pages"].is_not_null().any():
        h = h.replace("under 300 pages", "before 1900").replace("Under 300 pages", "Before 1900")
    return h


@app.get("/api/config")
def config(h: int = 760):
    return {
        "dash_url": dash_url(h),
        "figures": FIG,
        "llm": agent.status(),
        "total": B.height,
        "subtitle": CORPUS,
        "has_rating": bool(HAS_RATING),
        "attribution": store.META.get("attribution", ""),
    }


@app.get("/assets/wordmark-{v}.png")
def wordmark(v: str):
    if v not in ("dark", "light"):
        raise HTTPException(404)
    return FileResponse(os.path.join(os.path.dirname(_fv.__file__), "adapters", f"wordmark-{v}.png"))


@app.get("/assets/fv-light.css")
def fv_light():
    """FlexViz's own light design tokens (first :root block of its theme.css), re-declared unconditionally so an OS dark mode can't override them."""
    import re
    from fastapi.responses import PlainTextResponse

    css = open(os.path.join(os.path.dirname(_fv.__file__), "adapters", "js", "theme.css")).read()
    block = re.search(r"^:root\s*\{.*?\n\}", css, re.S | re.M).group(0)
    block = re.sub(r"--fv-brand-image:[^;]*;", "--fv-brand-image: url(/assets/wordmark-light.png);", block)
    return PlainTextResponse(block, media_type="text/css")


@app.get("/assets/{name}")
def asset(name: str):
    p = os.path.join(HERE, "assets", os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/state")
def get_state():
    return view_state()


@app.post("/api/state")
async def post_state(req: Request):
    """Body: {"state": {...FlexViz state patch...}} (raw) or {"tool": name, "args": {...}}."""
    b = await req.json()
    if "tool" in b:
        return await call_tool(b["tool"], b.get("args"))
    return await set_state(b["state"], by=b.get("by", "external"))


@app.post("/api/browser_state")
async def browser_state(req: Request):
    """The page posts flexvizState({compact:true}) whenever its revision changes."""
    b = await req.json()
    if _stale(b.get("state") or {}):
        return {"stale": True, "boot": BOOT}
    M["browser_seen"] = time.time()
    if b.get("ack") and b["ack"] in ACK:
        ACK[b["ack"]].set()
    st = b["state"]
    changed = json.dumps(st, sort_keys=True) != json.dumps(M["state"], sort_keys=True)
    M["client_state"] = b.get("client_state", {})
    M["fv_revision"] = b.get("revision", 0)
    if changed:
        M["state"] = st
        M["rev"] += 1
        if not b.get("ack"):
            M["by"] = "human"
            M["at"] = time.time()
            _log("human", describe(st))
    n, rows = books_in_view(M["state"], 50)
    C, top = top_rated(M["state"], 40)
    return {
        "books_in_view": n,
        "books": rows,
        "top_rated": top,
        "focus": FOCUS["book"],
        "by": M["by"],
        "description": describe(M["state"]),
        "filters": filters_of(M["state"]),
    }


@app.get("/api/top")
def api_top(limit: int = 30):
    C, rows = top_rated(M["state"], min(limit, 100))
    return {"books": rows, "mean": C}


PROFILE = str(paths.home() / "user_profile.md")


# ---- Librarian model picker: local Ollama models or an OpenAI-compatible endpoint ----
def _model_state(extra=None):
    tags = agent.ollama_tags(True)
    return {
        "current": agent.public_conf(),
        "provider": agent.provider(),
        "ollama": {"reachable": tags is not None, "host": agent.OLLAMA, "models": tags or []},
        "default_local": os.environ.get("AGENT_MODEL", "ministral-3:8b"),
        "rec_model": agent.REC_MODEL(),
    } | (extra or {})


@app.get("/api/librarian/model")
def lib_model_get():
    return _model_state()


@app.post("/api/librarian/model")
async def lib_model_set(req: Request):
    b = await req.json()
    kind = b.get("kind")
    if kind == "default":  # back to the server default (local Ollama, AGENT_MODEL); no test call
        agent.save_conf({"kind": "ollama", "model": None, "base_url": None, "api_key": None})
        return _model_state({"saved": True})
    if kind == "ollama":
        d = {"kind": "ollama", "model": str(b.get("model") or ""), "base_url": None, "api_key": None}
    elif kind == "openai":
        base = str(b.get("base_url") or "").strip().rstrip("/")
        model = str(b.get("model") or "").strip()
        if not re.match(r"^https?://[^\s/]+", base):
            raise HTTPException(400, "base URL must start with http:// or https://")
        if not model:
            raise HTTPException(400, "model name is required")
        key = b.get("api_key")
        if not key and agent.CONF["kind"] == "openai" and agent.CONF["base_url"] == base:
            key = agent.CONF["api_key"]  # keep the saved key
        d = {"kind": "openai", "model": model, "base_url": base, "api_key": (str(key).strip() or None) if key else None}
    else:
        raise HTTPException(400, "kind must be 'ollama' or 'openai'")
    t = await agent.test_model(d)
    if t["ok"]:
        agent.save_conf(d)
    return _model_state({"test": t, "saved": t["ok"]})


@app.post("/api/librarian/model/test")
async def lib_model_test():
    return _model_state(
        {
            "test": await agent.test_model(
                dict(agent.CONF, model=agent.OLLAMA_MODEL() if agent.CONF["kind"] == "ollama" else agent.CONF["model"])
            )
        }
    )


@app.get("/api/profile")
def get_profile():
    return {
        "text": open(PROFILE).read() if os.path.exists(PROFILE) else "",
        "goodreads": {"connected": False, "status": "coming soon"},
    }


@app.post("/api/profile")
async def post_profile(req: Request):
    b = await req.json()
    t = str(b.get("text", ""))[:8000]
    tmp = PROFILE + ".tmp"
    open(tmp, "w").write(t)
    os.replace(tmp, PROFILE)
    _log("human", "updated reading profile")
    return {"ok": True, "chars": len(t), "saved_at": time.strftime("%H:%M:%S")}


@app.get("/api/events")
async def events(req: Request):
    q = asyncio.Queue()
    CLIENTS.add(q)

    async def gen():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'boot': BOOT, 'state': M['state'], 'focus': FOCUS['book'], 'recs': HIGHLIGHT['recs']})}\n\n"
            while True:
                if await req.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), 15)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            CLIENTS.discard(q)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/api/tools")
def tools():
    return {"tools": tool_list()}


@app.post("/api/tools/{name}")
async def tool(name: str, req: Request):
    try:
        args = await req.json()
    except Exception:
        args = {}
    return await call_tool(name, args)


# ---------- map overlay data (page-side layer on top of the FlexViz hist2d) ----------
def _view(x0, x1, y0, y1):
    st = {
        "viewport": {f"{FIG['map']}/x": {"min": x0, "max": x1}, f"{FIG['map']}/y": {"min": y0, "max": y1}},
        "selections": M["state"].get("selections") or [],
    }
    return B.filter(filt(st))


def _lab(c):
    return c.split(" ", 1)[-1].replace("-", " ")


@app.get("/api/map/overlay")
def map_overlay(x0: float, x1: float, y0: float, y1: float, all_max: int = 2500, top: int = 600, points: int = 1):
    v = _view(x0, x1, y0, y1)
    n = v.height
    pts = v.sort("ratings_count", descending=True).head((all_max if n <= all_max else top) if points else 0)
    P = {
        k: pts[c].to_list()
        for k, c in [
            ("i", "i"),
            ("x", "x"),
            ("y", "y"),
            ("t", "title"),
            ("a", "author"),
            ("yr", "year"),
            ("r", "rating"),
            ("n", "ratings_count"),
            ("g", "genre"),
            ("c", "cluster"),
        ]
    }
    P["x"] = [round(float(x), 4) for x in P["x"]]
    P["y"] = [round(float(y), 4) for y in P["y"]]
    P["r"] = [None if r is None else round(float(r), 2) for r in P["r"]]
    L = (
        v.group_by("cluster")
        .agg(pl.len().alias("n"), pl.col("x").median().alias("x"), pl.col("y").median().alias("y"))
        .sort("n", descending=True)
    )
    L = L.filter(pl.col("n") >= max(5, int(n * 0.004)))
    labels = []  # cluster labels removed with the topic clusters
    return {"in_view": n, "all": n <= all_max, "full": (x1 - x0) >= 0.8 * (X1 - X0), "points": P, "labels": labels}


# all books as compact columns for the client-side WebGL layer (sent once, cached by the browser)
def _points_blob():
    o = B.sort(["ratings_count", "i"], nulls_last=False)  # low-count first -> popular drawn on top
    _top = lambda c: sorted(
        B[c].drop_nulls().value_counts(sort=True).head(254)[c].to_list()
    )  # u8 codes: 254 most common, 255 = other
    G = _top("genre")
    C = _top("cluster")
    gi = {g: k for k, g in enumerate(G)}
    ci = {c: k for k, c in enumerate(C)}
    cols = [
        o["i"].cast(pl.UInt32).to_numpy(),
        o["x"].cast(pl.Float32).to_numpy(),
        o["y"].cast(pl.Float32).to_numpy(),
        o["year"].fill_null(-1).cast(pl.Int16).to_numpy(),
        (o["rating"].fill_null(0) * 1000).round().cast(pl.UInt16).to_numpy(),
        (o["log10_ratings"].fill_null(0) * 1000).round().cast(pl.UInt16).to_numpy(),
        np.array([gi.get(g, 255) for g in o["genre"].to_list()], dtype=np.uint8),
        np.array([ci.get(c, 255) for c in o["cluster"].to_list()], dtype=np.uint8),
        np.zeros((4 - (20 * o.height) % 4) % 4, dtype=np.uint8),
        o["ratings_count"].fill_null(0).cast(pl.UInt32).to_numpy(),
        o["pages"].fill_null(0).clip(0, 65535).cast(pl.UInt16).to_numpy(),
    ]
    blob = b"".join(np.ascontiguousarray(c).tobytes() for c in cols)
    import hashlib

    meta = {
        "n": o.height,
        "layout": [
            "i:u32",
            "x:f32",
            "y:f32",
            "year:i16",
            "rating:u16/1000",
            "log10_ratings:u16/1000",
            "genre:u8",
            "cluster:u8",
            "<pad to 4>",
            "ratings_count:u32",
            "pages:u16 (0=unknown)",
        ],
        "genres": G,
        "clusters": C,
        "full": {"x": [X0, X1], "y": [Y0, Y1]},
        "version": hashlib.md5(blob).hexdigest()[:12],
        "bytes": len(blob),
    }
    return meta, blob


_PB = {}


def points_blob():
    if not _PB:
        _PB["meta"], _PB["blob"] = _points_blob()
    return _PB["meta"], _PB["blob"]


PERF = []
BOOT = uuid.uuid4().hex[:12]


def _stale(st):
    """A tab loaded from an older server run references figure uids that no longer exist; ignore it so it cannot clobber the shared state."""
    ok = set(FIG.values())
    if any(s.get("source_figure_uid") not in ok for s in st.get("selections") or []):
        return True
    return any(k.split("/")[0] not in ok for k in (st.get("viewport") or {}))


@app.post("/api/perf")
async def perf_post(req: Request):
    d = await req.json()
    d["t"] = time.strftime("%H:%M:%S")
    PERF.append(d)
    del PERF[:-20]
    return {"ok": True}


@app.get("/api/perf")
def perf_get():
    return PERF


@app.get("/api/map/meta")
def map_meta():
    return points_blob()[0]


@app.get("/api/map/points.bin")
def map_points(v: str = ""):
    meta, blob = points_blob()
    return Response(
        blob,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable" if v == meta["version"] else "no-cache",
            "ETag": meta["version"],
        },
    )


@app.get("/api/map/bin")
def map_bin(x0: float, x1: float, y0: float, y1: float):
    v = _view(x0, x1, y0, y1)
    top = v.sort("ratings_count", descending=True).head(3)
    cl = v.group_by("cluster").len().sort("len", descending=True).head(1)["cluster"].to_list()
    return {
        "count": v.height,
        "cluster": _lab(cl[0]) if cl else None,
        "top": [
            {"title": r["title"], "author": r["author"], "rating": None if r["rating"] is None else round(r["rating"], 2)}
            for r in top.iter_rows(named=True)
        ],
    }


@app.get("/go/book/{i}")
def go_book(i: int):
    u = gr_url(i)
    if not u:
        raise HTTPException(404, "no link for this book")
    return RedirectResponse(u, status_code=302)


@app.get("/api/book/{i}")
def api_book(i: int):
    if not 0 <= i < B.height:
        raise HTTPException(404, "no such book")
    r = bookrow(B.row(i, named=True))
    e = rec._extra([i]).get(i, {})
    d = (e.get("desc") or "").strip()
    r["blurb"] = (d[:200].rsplit(" ", 1)[0] + "…") if len(d) > 200 else d
    r["pages"] = e.get("pages")
    return r


@app.get("/api/search")
def api_search(q: str, limit: int = 8):
    return search(q, min(limit, 20))


@app.post("/api/rec/start")
async def rec_start(req: Request):
    try:
        b = await req.json()
    except Exception:
        b = {}
    return await t_recommend_start((b or {}).get("query", ""))


@app.post("/api/rec/answer")
async def rec_answer(req: Request):
    b = await req.json()
    return await t_recommend_answer(b.get("session"), b.get("option"))


def _H():
    return {
        "title": title_lookup,
        "author": author_lookup,
        "topic": topic_lookup,
        "filters": lambda: filters_of(M["state"]),
        "top": lambda n=5: top_rated(M["state"], n)[1],
        "focus": lambda: FOCUS["book"],
        "full": (X0, X1, Y0, Y1),
        "cur_vp": _cur_vp,
    }


@app.get("/api/librarian/parse")
def librarian_parse(q: str):
    p = librarian.parse(q, _H())
    return {
        "recognised": librarian.recognised(p),
        "filters": p.f,
        "remove": sorted(p.remove),
        "clear": p.clear,
        "view": p.view,
        "zoom": p.zoom,
        "pan": p.pan,
        "focus": p.focus and p.focus["title"],
        "like": p.like and p.like["title"],
        "author": p.author,
        "topic": p.topic,
        "ask": p.ask,
        "wizard": p.wizard,
        "leftover": p.t.strip(),
    }


@app.post("/api/chat")
async def chat(req: Request):
    b = await req.json()
    log = {}

    async def ct(name, args):
        res = await call_tool(name, args)
        if name in ("similar_books", "top_rated_in_view") and isinstance(res, dict):
            log[name] = res
        if name.startswith("recommend_"):
            log["wizard"] = True
        return res

    r = await agent.chat(
        b.get("messages", []), tool_list(), ct, view_state, H=_H(), model=b.get("model"), rules=b.get("rules", True) is not False
    )
    try:
        msgs = b.get("messages") or []
        await recs_from_chat(msgs[-1].get("content", "") if msgs else "", r.get("reply", ""), r.get("via"), log)
    except Exception as e:
        print("recs_from_chat", e)
    r["recommended"] = HIGHLIGHT["recs"]
    try:
        r["links"] = mention_links(r.get("reply", ""))
    except Exception:
        r["links"] = {}
    return r


threading.Thread(target=points_blob, daemon=True).start()  # warm the WebGL point blob
