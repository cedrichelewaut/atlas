"""Shared state of one Atlas server: the loaded dataset, the FlexViz dashboard spec and the mirror of the browser's
FlexViz state that the human and Atlas AI both change (zoom, slider selections, focus, recommendation highlights)."""

import re
import json
import time
import asyncio
import numpy as np
import polars as pl
from flexviz import Dashboard, LayoutSpec, ToolbarConfig, GridItem
from flexviz.server import register_source
from . import store, paths
from . import agent, librarian

HERE = str(paths.WEB)
ROOT = HERE
SRC = "books"
SOURCE = "fv_books"
DATA = store.DATA
B = store.books()  # i,title,author,year,x,y,cluster_label,ratings_count,cluster,genre,rating,url
B = B.with_columns(
    (pl.col("ratings_count").fill_null(0) + 1).log10().cast(pl.Float32).alias("log10_ratings")
)  # log10(ratings+1) for the ratings-count chart
B = B.join(pl.read_parquet(f"{DATA}/extra.parquet", columns=["i", "pages"]), on="i", how="left", suffix="_x").sort("i")
if "pages_x" in B.columns:
    B = B.with_columns(pl.coalesce("pages", "pages_x").cast(pl.Int32).alias("pages")).drop("pages_x")
HAS_RATING = B["rating"].is_not_null().any()
FV = B.select("x", "y", "year", "rating", "log10_ratings", "pages")
register_source(SOURCE, FV.lazy(), cache=True)
LOWER = B["title"].str.to_lowercase()
URLS = B["url"].to_list()
CORPUS = store.describe(B.height)
_yr = B["year"].drop_nulls()
if _yr.len():
    librarian.YMIN, librarian.YMAX = int(_yr.quantile(0.01)), int(_yr.max())
agent.SYSTEM = agent.system_prompt(CORPUS, bool(HAS_RATING))


def book_url(i):
    if not 0 <= int(i) < len(URLS):
        return None
    return URLS[int(i)]


_GL = {}


def mention_links(text, limit=20):
    """Map *Title* / **Title** spans in a reply to book URLs (exact title match, most-rated edition)."""
    out = {}
    for t in re.findall(r"\*\*([^*\n]{2,140})\*\*|\*([^*\n]{2,140})\*", text or ""):
        t = (t[0] or t[1]).strip()
        if not t or t in out or len(out) >= limit:
            continue
        hit = B.filter(LOWER == t.lower())
        if hit.height == 0:
            if "ns" not in _GL:
                _GL["ns"] = LOWER.str.replace(r"\s*\([^)]*#[^)]*\)$", "")  # title without '(Series, #n)'
            hit = B.filter(_GL["ns"] == t.lower())
        if hit.height:
            out[t] = book_url(hit.sort("ratings_count", descending=True)["i"][0])
    return {k: v for k, v in out.items() if v}


X0, X1 = [float(v) for v in np.percentile(B["x"].to_numpy(), [0.3, 99.7])]
Y0, Y1 = [float(v) for v in np.percentile(B["y"].to_numpy(), [0.3, 99.7])]

# ---------- FlexViz dashboard (the only visualisation) ----------
FONT = 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
INK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"family": FONT, "color": "#4a5162", "size": 11},
    hoverlabel={"bgcolor": "#ffffff", "bordercolor": "rgba(15,23,42,.15)", "font": {"family": FONT, "color": "#141824"}},
    xaxis={
        "gridcolor": "rgba(15,23,42,0.06)",
        "zerolinecolor": "rgba(15,23,42,0.10)",
        "linecolor": "rgba(15,23,42,0.15)",
        "tickfont": {"color": "#6b7280"},
        "title": {"font": {"color": "#6b7280"}},
    },
    yaxis={
        "gridcolor": "rgba(15,23,42,0.06)",
        "zerolinecolor": "rgba(15,23,42,0.10)",
        "linecolor": "rgba(15,23,42,0.15)",
        "tickfont": {"color": "#6b7280"},
        "title": {"font": {"color": "#6b7280"}},
    },
)


def th(**kw):
    out = json.loads(json.dumps(INK))
    for k, v in kw.items():
        out[k] = (out.get(k, {}) | v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


d = Dashboard(FV.lazy(), cache=True)
TITLE = {"font": {"family": FONT, "color": "#141824", "size": 13}, "x": 0.02, "xanchor": "left"}


def fig(t, **kw):
    return d.add_figure(title=t, **th(**kw))


fig(
    "Description map  ·  books with similar blurbs sit together  ·  hover for details",
    margin={"l": 10, "r": 6, "t": 34, "b": 10},
    xaxis={"title": {"text": ""}, "showticklabels": False, "showgrid": False, "zeroline": False, "showline": False, "ticks": ""},
    yaxis={"title": {"text": ""}, "showticklabels": False, "showgrid": False, "zeroline": False, "showline": False, "ticks": ""},
).add_histogram2d(x="x", y="y", x_bins=80, y_bins=80, color_norm="log", color_scale="Viridis")
fig(
    "First published", margin={"l": 44, "t": 34, "b": 28, "r": 10}, xaxis={"title": {"text": ""}}, yaxis={"title": {"text": ""}}
).add_histogram(x="year", bins=40, color="#14b8a6")
fig(
    "Average rating", margin={"l": 44, "t": 34, "b": 28, "r": 10}, xaxis={"title": {"text": ""}}, yaxis={"title": {"text": ""}}
).add_histogram(x="rating", bins=30, color="#f59e0b")
fig(
    "Pages", margin={"l": 44, "t": 34, "b": 28, "r": 10}, xaxis={"title": {"text": ""}}, yaxis={"title": {"text": ""}}
).add_histogram(x="pages", bins=40, color="#64748b")
fig(
    "Number of ratings (log scale)",
    margin={"l": 44, "t": 34, "b": 28, "r": 10},
    xaxis={
        "title": {"text": ""},
        "tickmode": "array",
        "tickvals": [0, 1, 2, 3, 4, 5, 6, 7],
        "ticktext": ["1", "10", "100", "1k", "10k", "100k", "1M", "10M"],
    },
    yaxis={"title": {"text": ""}},
).add_histogram(x="log10_ratings", bins=36, color="#0ea5e9")
U = [f.uid for f in d.to_spec(source_name=SOURCE).figures]
FIG = dict(zip(["map", "year", "rating", "pages", "votes"], U))


def dash_url(px_h=None):
    # only the map is laid out; the year/rating/pages/ratings histograms stay in the spec (hidden) as the selection sources
    # for the page-side range sliders, so every filter lives in the one FlexViz state
    rows = max(4, min(14, int(((px_h or 600) - 40) // 80)))
    lay = LayoutSpec(
        draggable=False,
        grid_items=[GridItem(fig_uid=FIG["map"], x=0, y=0, w=12, h=rows)],
        toolbar=ToolbarConfig(
            show_share=False, show_export=False, show_import=False, show_grid=False, show_lock_all_axes=False, show_cfmode=False
        ),
    )
    return d.share_url(server_url="/flexviz", source_name=SOURCE, cache=True, layout=lay)


DASH_URL = dash_url()
COLFIG = {
    "x": FIG["map"],
    "y": FIG["map"],
    "year": FIG["year"],
    "rating": FIG["rating"],
    "log10_ratings": FIG["votes"],
    "pages": FIG["pages"],
}
FILTERS = {"year": "year", "rating": "rating", "ratings_count": "log10_ratings", "pages": "pages"}  # slider name -> state column


def filters_of(state):
    """The slider view of the shared state: {year:[a,b]|None, rating, ratings_count, pages, region}."""
    out = {k: None for k in FILTERS} | {"region": None}
    for sel in state.get("selections") or []:
        for p in sel.get("predicates", []):
            cl = {c["column"]: c for c in p.get("clauses", []) if c.get("range") is not None}
            if "x" in cl and "y" in cl:
                out["region"] = [*cl["x"]["range"], *cl["y"]["range"]]
                continue
            for k, col in FILTERS.items():
                if col in cl:
                    r = [float(v) for v in cl[col]["range"]]
                    out[k] = [round(10 ** r[0] - 1), round(10 ** r[1] - 1)] if k == "ratings_count" else [round(v, 2) for v in r]
    return out


# ---------- shared state: mirror of the browser's FlexViz state ----------
INIT = {"viewport": {f"{FIG['map']}/x": {"min": X0, "max": X1}, f"{FIG['map']}/y": {"min": Y0, "max": Y1}}, "selections": []}
M = {
    "state": json.loads(json.dumps(INIT)),
    "client_state": {},
    "fv_revision": 0,
    "rev": 0,
    "by": "init",
    "at": time.time(),
    "browser_seen": 0.0,
}
LOG = []
CLIENTS = set()
PATCH_ID = [0]
ACK = {}
FOCUS = {"book": None}


def _log(by, what):
    LOG.append({"t": time.strftime("%H:%M:%S"), "by": by, "what": what})
    del LOG[:-40]


def describe(state):
    out = []
    for k, v in (state.get("viewport") or {}).items():
        if v:
            out.append(f"zoom {k.split('/')[1]}=[{v['min']:.2f},{v['max']:.2f}]" if k.startswith(FIG["map"]) else f"zoom {k}")
    for s in state.get("selections") or []:
        for p in s.get("predicates", []):
            for c in p.get("clauses", []):
                if c["column"] == "log10_ratings" and c.get("range"):
                    out.append(
                        f"select ratings received∈[{max(0, round(10 ** float(c['range'][0]) - 1)):,}–{round(10 ** float(c['range'][1]) - 1):,}]"
                    )
                    continue
                if c.get("range"):
                    lo, hi = (float(x) for x in c["range"])
                    f = lambda v: f"{round(v, 2):g}"
                    out.append(f"select {c['column']}∈[{f(lo)}–{f(hi)}]" if hi < 1e5 else f"select {c['column']}≥{f(lo)}")
                    continue
                out.append(f"select {c['column']}∈{c.get('values')}")
    return "; ".join(out) or "full map, no selection"


def filt(state, ignore_viewport=False):
    e = pl.lit(True)
    vp = state.get("viewport") or {}
    if not ignore_viewport:
        for ax in "xy":
            r = vp.get(f"{FIG['map']}/{ax}")
            if r:
                e = e & pl.col(ax).is_between(r["min"], r["max"])
    for s in state.get("selections") or []:
        pe = pl.lit(False)
        for p in s.get("predicates", []):
            ce = pl.lit(True)
            for c in p.get("clauses", []):
                col = c["column"]
                if c.get("values") is not None:
                    ce = ce & pl.col(col).cast(pl.Utf8).is_in([str(v) for v in c["values"]])
                elif c.get("range") is not None:
                    ce = ce & pl.col(col).is_between(float(c["range"][0]), float(c["range"][1]))
            pe = pe | ce
        e = e & pe
    return e


def bookrow(r):
    return {
        "i": r["i"],
        "title": r["title"],
        "author": r["author"],
        "year": r["year"],
        "rating": None if r["rating"] is None else round(r["rating"], 2),
        "ratings_count": r["ratings_count"],
        "pages": r.get("pages"),
        "x": round(r["x"], 6),
        "y": round(r["y"], 6),
        "url": book_url(r["i"]),
    }  # full precision: the focus ring must sit exactly on the dot at any zoom


def books_in_view(state, limit=20, offset=0):
    v = B.filter(filt(state))
    top = v.sort("ratings_count", descending=True).slice(offset, limit)
    return v.height, [bookrow(r) for r in top.iter_rows(named=True)]


def summary(state, limit=15):
    v = B.filter(filt(state))
    n = v.height
    return {
        "books_in_view": n,
        "of_total": B.height,
        "description": describe(state),
        "top_books": [bookrow(r) for r in v.sort("ratings_count", descending=True).head(limit).iter_rows(named=True)],
        "filters": filters_of(state),
        "years": None
        if n == 0
        else {"median": v["year"].median(), "p10": v["year"].quantile(0.1), "p90": v["year"].quantile(0.9)},
        "avg_rating": None if n == 0 else (round(v["rating"].mean(), 2) if v["rating"].mean() is not None else None),
        "focus": FOCUS["book"],
        "recommended": [f"{r['rank']}. {r['title']}" for r in HIGHLIGHT["recs"]],
    }


def view_state():
    return {
        "figures": FIG,
        "state": M["state"],
        "fv_revision": M["fv_revision"],
        "rev": M["rev"],
        "last_change_by": M["by"],
        "browser_connected": len(CLIENTS) > 0,
        "recent_changes": LOG[-8:],
        "summary": summary(M["state"], 10),
        "highlights": HIGHLIGHT,
    }


LOOP = {}


async def _broadcast(msg):
    for q in list(CLIENTS):
        q.put_nowait(msg)


async def set_state(patch, by="agent", what=None, wait=4.0):
    """Merge a state patch one level deep (FlexViz semantics), push to the browser, wait for it to apply."""
    for k, v in patch.items():
        M["state"][k] = v
    M["rev"] += 1
    M["by"] = by
    M["at"] = time.time()
    _log(by, what or describe(patch))
    PATCH_ID[0] += 1
    pid = PATCH_ID[0]
    applied = False
    if CLIENTS:
        ev = asyncio.Event()
        ACK[pid] = ev
        await _broadcast({"type": "apply", "id": pid, "patch": {"state": patch}, "by": by})
        try:
            await asyncio.wait_for(ev.wait(), wait)
            applied = True
        except asyncio.TimeoutError:
            pass
        ACK.pop(pid, None)
    return {
        "applied_in_browser": applied,
        "browser_connected": bool(CLIENTS),
        "state": M["state"],
        "summary": summary(M["state"], 10),
    }


# recommendation highlights (shared state): every recommended book gets a yellow ring + "rank title" tag
HIGHLIGHT = {"recs": [], "by": None, "at": None}
