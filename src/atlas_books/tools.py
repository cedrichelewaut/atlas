"""Atlas AI tools (also served at /api/tools for external agents): search, similar books, zoom/filter the shared view,
topics, authors, top rated, the recommendation wizard and recommendation highlights."""

import re
import time
import threading
import numpy as np
import polars as pl
from fastapi import HTTPException
from . import store
from . import agent, rec
from .state import (
    B,
    COLFIG,
    DATA,
    FIG,
    FILTERS,
    FOCUS,
    HAS_RATING,
    HIGHLIGHT,
    INIT,
    LOWER,
    M,
    X0,
    X1,
    Y0,
    Y1,
    _broadcast,
    _log,
    bookrow,
    books_in_view,
    describe,
    filt,
    filters_of,
    book_url,
    set_state,
    view_state,
)

# ---------- embeddings (lazy) ----------
EMB = {}
ELK = threading.Lock()


def _vecs():
    """fp16 vectors if present, else int8 + per-dimension scale (portable package)."""
    n, get = store.vec_getter()
    return range(n), get


def emb():
    with ELK:
        if not EMB:
            V, get = _vecs()
            try:
                import faiss

                ix = faiss.IndexScalarQuantizer(get(0, 1).shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
                for a in range(0, len(V), 100000):
                    ix.add(get(a, a + 100000))
            except ImportError:
                ix = None  # numpy fallback below (no faiss wheel for this Python)
            EMB.update(V=V, get=get, n=len(V), ix=ix)
    return EMB


def _search(E, q, k):
    if E["ix"] is not None:
        D, I = E["ix"].search(q, k)
        return D[0], I[0]
    best = []
    for a in range(0, E["n"], 100000):
        s = E["get"](a, a + 100000) @ q[0]
        j = np.argpartition(-s, min(k, len(s) - 1))[:k]
        best += list(zip(s[j], j + a))
    best.sort(key=lambda t: -t[0])
    best = best[:k]
    return np.array([b[0] for b in best]), np.array([b[1] for b in best])


def find(title=None, i=None):
    if i is not None:
        return int(i)
    h = search(title, 1)
    if not h:
        raise HTTPException(404, f"no book matching {title!r}")
    return h[0]["i"]


def search(q, limit=8):
    ql = (q or "").lower().strip()
    if not ql:
        return []
    h = (
        B.with_columns(_l=LOWER)
        .filter(pl.col("_l").str.contains(ql, literal=True))
        .with_columns(_p=pl.col("_l").str.starts_with(ql), _n=pl.col("title").str.len_chars())
        .sort(["_p", "ratings_count", "_n"], descending=[True, True, False])
        .head(limit)
    )
    return [bookrow(r) for r in h.iter_rows(named=True)]


def similar(i, k=10):
    E = emb()
    D, I = _search(E, E["get"](i, i + 1), k + 1)
    return [bookrow(B.row(int(j), named=True)) | {"similarity": round(float(s), 3)} for j, s in zip(I, D) if j != i and j >= 0][
        :k
    ]


# ---------- tools (shared by the in-page agent and external agents) ----------
def _rng(v):
    return [float(v[0]), float(v[1])]


def _sel(col, clause):
    return {"source_figure_uid": COLFIG[col], "predicates": [{"clauses": [clause]}]}


async def t_get_view_state():
    return view_state()


async def t_list_books_in_view(limit=20, offset=0):
    n, rows = books_in_view(M["state"], min(int(limit), 100), int(offset))
    return {"books_in_view": n, "books": rows, "sorted_by": "ratings_count desc"}


async def t_search_book(title, limit=5):
    return {"matches": search(title, min(int(limit), 20))}


async def t_set_view(x_min=None, x_max=None, y_min=None, y_max=None, reset=False):
    if reset:
        vp = dict(INIT["viewport"])
        FOCUS["book"] = None
        await set_recs([], by="reset", frame=False)
    else:
        if None in (x_min, x_max, y_min, y_max):
            raise HTTPException(400, "set_view needs x_min, x_max, y_min, y_max (or reset=true)")
        x_min, x_max = sorted([float(x_min), float(x_max)])
        y_min, y_max = sorted([float(y_min), float(y_max)])
        if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
            raise HTTPException(400, "empty rectangle")
        vp = {f"{FIG['map']}/x": {"min": x_min, "max": x_max}, f"{FIG['map']}/y": {"min": y_min, "max": y_max}}
        FOCUS["book"] = None
    out = await set_state({"viewport": vp}, what="reset map zoom" if reset else None)
    if FOCUS["book"] is None:
        await _broadcast({"type": "focus", "book": None})
    return out


async def t_set_filters(year=None, rating=None, ratings_count=None, pages=None, region=None, clear=False):
    """Set slider ranges in the shared FlexViz state. Unspecified filters are kept; [] removes one; clear=true removes all."""
    if not HAS_RATING:
        rating = None  # dataset without ratings: a rating filter would hide every book
    cur = {} if clear else {k: v for k, v in filters_of(M["state"]).items() if v}
    for k, v in (("year", year), ("rating", rating), ("ratings_count", ratings_count), ("pages", pages), ("region", region)):
        if v is None:
            continue
        if not isinstance(v, (list, tuple)):
            raise HTTPException(400, f"{k} must be a list [min,max]")
        if len(v) == 0:
            cur.pop(k, None)
            continue
        if len(v) != (4 if k == "region" else 2):
            raise HTTPException(400, f"{k} must be [min,max]" if k != "region" else "region must be [x_min,x_max,y_min,y_max]")
        try:
            v = [float(x) for x in v]
        except (TypeError, ValueError):
            raise HTTPException(400, f"{k} values must be numbers")
        if k != "region":
            v = sorted(v)
        lim = {"year": (0, 2100), "rating": (1, 5), "ratings_count": (0, 1e7), "pages": (0, 1e5)}.get(k)
        if lim:
            v = [min(max(x, lim[0]), lim[1]) for x in v]
        cur[k] = v
    sels = []
    for k, v in cur.items():
        if k == "region":
            sels.append(
                {
                    "source_figure_uid": FIG["map"],
                    "predicates": [{"clauses": [{"column": "x", "range": _rng(v[:2])}, {"column": "y", "range": _rng(v[2:])}]}],
                }
            )
        elif k == "ratings_count":
            sels.append(
                _sel(
                    "log10_ratings", {"column": "log10_ratings", "range": [float(np.log10(max(float(x), 0) + 1)) for x in v[:2]]}
                )
            )
        else:
            sels.append(_sel(FILTERS[k], {"column": FILTERS[k], "range": _rng(v)}))
    out = await set_state({"selections": sels}, what="clear filters" if not sels else None)
    return out | {"filters": filters_of(M["state"])}


async def t_focus_book(title=None, i=None, radius=None, keep_selections=True):
    """Zoom to a book and its neighbourhood (~400 nearest books on the map). Slider filters are kept unless they hide the book."""
    j = find(title, i) if i is not None else (title_lookup(title, strict=False) or {"i": find(title)})["i"]
    r = B.row(j, named=True)
    FOCUS["book"] = bookrow(r)
    rad = float(radius) if radius else _neigh_r(r["x"], r["y"])
    cleared = False
    patch = {"viewport": _vp(r["x"], r["y"], rad, rad * max(ASPECT, 0.7))}
    if not keep_selections:
        patch["selections"] = []
        cleared = bool(M["state"].get("selections"))
    else:
        one = B.slice(j, 1).filter(filt({"selections": M["state"].get("selections") or []}, ignore_viewport=True))
        if one.height == 0:
            patch["selections"] = []
            cleared = True
    out = await set_state(
        patch, what=f"focus on '{r['title']}' (±{rad:.2f})" + (" and clear filters that hid it" if cleared else "")
    )
    await _broadcast({"type": "focus", "book": FOCUS["book"]})
    return {"focus": FOCUS["book"], "cleared_filters": cleared} | out


async def t_similar_books(title=None, i=None, k=10):
    j = find(title, i)
    res = similar(j, min(int(k), 50))
    st = M["state"]
    inview = set(B.filter(filt(st))["i"].to_list())
    return {
        "book": bookrow(B.row(j, named=True)),
        "similar": [r | {"in_current_view": r["i"] in inview} for r in res],
        "note": f"similarity is cosine over {store.META.get('embedding_model', 'all-MiniLM-L6-v2')} embeddings of {store.META.get('embedding_text', 'the description')}"
        + "; map position is a 2D UMAP and does not always keep neighbours together",
    }


async def t_get_user_profile():
    return agent.load_profile(B)


async def t_recommend_start(query=""):
    r = await rec.start(B, filt(M["state"]), describe(M["state"]), query=query or "")
    await _wizard_recs(r)
    return r


async def t_recommend_answer(session, option):
    r = await rec.answer(session, option)
    await _wizard_recs(r)
    return r


REC_RE = re.compile(
    r"\b(recommend\w*|suggest\w*|books? like|similar|what (?:should|can|could) i read|top[- ]?rated|best|good (?:books?|reads?)|must[- ]reads?|picks?|worth reading)\b",
    re.I,
)


def _rec_row(b, k):
    i = int(b["i"])
    r = B.row(i, named=True)
    return {
        "rank": k + 1,
        "i": i,
        "title": r["title"],
        "author": r["author"],
        "x": round(float(r["x"]), 6),
        "y": round(float(r["y"]), 6),
        "url": book_url(i),
    }


async def set_recs(books, by="assistant", frame=True):
    """Replace the highlighted recommendations (kept until the next recommendation or a reset) and frame the view on them."""
    recs = [_rec_row(b, k) for k, b in enumerate(books[:10]) if b and b.get("i") is not None]
    if not recs and not HIGHLIGHT["recs"]:
        return
    HIGHLIGHT.update(recs=recs, by=by, at=time.time())
    if recs:
        _log(by, "highlight recommendations: " + "; ".join(f"{r['rank']}. {r['title']}" for r in recs))
    await _broadcast({"type": "recs", "books": recs})
    if frame and recs:
        pts = [(r["x"], r["y"]) for r in recs] + ([(FOCUS["book"]["x"], FOCUS["book"]["y"])] if FOCUS["book"] else [])
        x0, x1, y0, y1 = _cur_vp()
        mx, my = (x1 - x0) * 0.08, (y1 - y0) * 0.12
        if not all(x0 + mx <= x <= x1 - mx and y0 + my <= y <= y1 - my for x, y in pts):
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            rx = max((max(xs) - min(xs)) / 2 * 1.3, 0.6)
            ry = max((max(ys) - min(ys)) / 2 * 1.35, 0.45)
            rx, ry = max(rx, ry / max(ASPECT, 1e-6) * 0.85), max(ry, rx * ASPECT * 0.85)
            await set_state({"viewport": _vp(cx, cy, rx, ry)}, by=by, what="frame the recommended books")


async def _wizard_recs(r):
    f = (r or {}).get("final") if isinstance(r, dict) else None
    if f and f.get("top3"):
        await set_recs(f["top3"], by="recommendation")


def mention_rows(text, limit=8):
    out = []
    for t in re.findall(r"\*\*([^*\n]{2,140})\*\*|\*([^*\n]{2,140})\*", text or ""):
        t = (t[0] or t[1]).strip()
        h = title_lookup(t) if len(t) > 2 else None
        if (
            h
            and h["title"].strip().lower() == t.lower()
            and h["i"] not in [o["i"] for o in out]
            and not (FOCUS["book"] and h["i"] == FOCUS["book"]["i"])
        ):
            out.append(h)
        if len(out) >= limit:
            break
    return out


async def recs_from_chat(user_text, reply, via, log):
    """Which books did this chat turn recommend? similar_books results, or top rated / named books when the user asked for recommendations."""
    if "wizard" in log:
        return
    picks = None
    if "similar_books" in log:
        picks = (log["similar_books"].get("similar") or [])[:5]
    elif REC_RE.search(user_text or ""):
        if "top_rated_in_view" in log:
            picks = (log["top_rated_in_view"].get("books") or [])[:5]
        else:
            picks = mention_rows(reply)[:6]
    if picks:
        await set_recs(picks, by="assistant")


# ---------- map navigation helpers: top rated, topics, authors, relative zoom ----------
NX = NY = 200
_XA = B["x"].to_numpy().astype(np.float64)
_YA = B["y"].to_numpy().astype(np.float64)
_CX = np.clip(((_XA - X0) / (X1 - X0) * NX).astype(int), 0, NX - 1)
_CY = np.clip(((_YA - Y0) / (Y1 - Y0) * NY).astype(int), 0, NY - 1)


def _integral(mask, w=None):
    H = np.zeros((NX, NY))
    np.add.at(H, (_CX[mask], _CY[mask]), 1.0 if w is None else w[mask])
    I = np.zeros((NX + 1, NY + 1))
    I[1:, 1:] = H.cumsum(0).cumsum(1)
    return H, I


def _box(I, cx, cy, r):
    a, b = max(0, cx - r), min(NX, cx + r + 1)
    c, e = max(0, cy - r), min(NY, cy + r + 1)
    return I[b, e] - I[a, e] - I[b, c] + I[a, c]


def _mask(state):
    return B.with_columns(filt(state, ignore_viewport=True).alias("_m"))["_m"].fill_null(False).to_numpy()


def _vp(cx, cy, rx, ry):
    return {
        f"{FIG['map']}/x": {"min": float(cx - rx), "max": float(cx + rx)},
        f"{FIG['map']}/y": {"min": float(cy - ry), "max": float(cy + ry)},
    }


def _cur_vp():
    vp = M["state"].get("viewport") or {}
    x = vp.get(f"{FIG['map']}/x") or {"min": X0, "max": X1}
    y = vp.get(f"{FIG['map']}/y") or {"min": Y0, "max": Y1}
    return x["min"], x["max"], y["min"], y["max"]


CELL = (X1 - X0) / NX
ASPECT = (Y1 - Y0) / (X1 - X0)


def best_window(m, base, w=None, mode="topic"):
    """Find the map window where the books in mask m concentrate. Returns (cx,cy,rx,ry,info)."""
    Ht, It = _integral(m & base, w)
    Ha, Ia = _integral(base)
    tot = It[-1, -1]
    p0 = tot / max(Ia[-1, -1], 1)
    S = Ht.copy()  # smoothed seed map
    for _ in range(3):
        S = (S + np.roll(S, 1, 0) + np.roll(S, -1, 0) + np.roll(S, 1, 1) + np.roll(S, -1, 1)) / 5
    seeds = np.dstack(np.unravel_index(np.argsort(-S, axis=None)[:12], S.shape))[0]
    if mode != "topic":
        seeds = np.vstack([np.dstack(np.unravel_index(np.argsort(-Ht, axis=None)[:6], Ht.shape))[0], seeds])
    best = None
    for cx, cy in seeds:
        for r in (4, 6, 9, 13, 18, 25, 34):
            nt = _box(It, cx, cy, r)
            na = max(_box(Ia, cx, cy, r), 1)
            if mode == "topic":
                if nt < 25:
                    continue
                lift = (nt / na) / max(p0, 1e-9)
                sc = nt * lift**2
            else:  # author: smallest window holding half their (weighted) books
                lift = (nt / na) / max(p0, 1e-9)
                sh = nt / max(tot, 1e-9)
                sc = 1.0 / r if sh >= 0.5 else sh / 100.0
                if best is not None and sc <= best[0]:
                    continue
                if sh >= 0.5:
                    best = (sc, cx, cy, r, nt, na, lift)
                    break
            if best is None or sc > best[0]:
                best = (sc, cx, cy, r, nt, na, lift)
    if best is None:
        return None
    sc, cx, cy, r, nt, na, lift = best
    rx = (r + 0.5) * CELL
    return (
        X0 + (cx + 0.5) * CELL,
        Y0 + (cy + 0.5) * (Y1 - Y0) / NY,
        rx,
        max(rx * ASPECT * 1.0, rx * 0.7),
        {"share_in_window": round(nt / max(tot, 1e-9), 3), "lift": round(float(lift), 2)},
    )


TOPICS = {
    "science fiction": (
        ["science-fiction", "sci-fi", "scifi", "sf", "space-opera"],
        ["science fiction", "sci-fi", "sci fi", "scifi", "sf", "space opera", "space"],
    ),
    "fantasy": (["fantasy", "high-fantasy", "epic-fantasy"], ["fantasy", "high fantasy", "epic fantasy", "sword and sorcery"]),
    "urban fantasy": (["urban-fantasy"], ["urban fantasy"]),
    "romance": (["romance", "contemporary-romance"], ["romance", "romances", "romantic", "love story", "love stories"]),
    "historical romance": (["historical-romance", "regency", "regency-romance"], ["historical romance", "regency"]),
    "paranormal romance": (["paranormal-romance"], ["paranormal romance"]),
    "mystery": (
        ["mystery", "mysteries", "detective"],
        ["mystery", "mysteries", "whodunit", "whodunnit", "detective", "detective stories"],
    ),
    "cozy mystery": (
        ["cozy-mystery", "cozy-mysteries", "mystery-cozy"],
        ["cozy mystery", "cozy mysteries", "cosy mystery", "cosy mysteries"],
    ),
    "thriller": (
        ["thriller", "thrillers", "suspense", "mystery-thriller"],
        ["thriller", "thrillers", "suspense", "psychological thriller", "psychological thrillers"],
    ),
    "crime": (["crime", "crime-fiction"], ["crime", "crime fiction", "noir"]),
    "horror": (["horror"], ["horror", "scary", "creepy", "spooky"]),
    "historical fiction": (["historical-fiction", "historical"], ["historical fiction", "historical", "history"]),
    "young adult": (["young-adult", "ya", "teen"], ["young adult", "ya", "teen", "teens"]),
    "middle grade": (["middle-grade"], ["middle grade"]),
    "children's": (
        ["childrens", "children", "picture-books", "children-s-books", "kids"],
        ["children's", "childrens", "children", "kids", "picture books"],
    ),
    "humor": (["humor", "humour", "funny", "comedy"], ["humor", "humour", "funny", "comedy", "comedic", "hilarious"]),
    "dystopian": (
        ["dystopian", "dystopia", "post-apocalyptic", "apocalyptic"],
        ["dystopian", "dystopia", "dystopias", "post-apocalyptic", "post apocalyptic", "apocalyptic"],
    ),
    "paranormal": (["paranormal", "supernatural"], ["paranormal", "supernatural"]),
    "vampires": (["vampires", "vampire"], ["vampire", "vampires"]),
    "werewolves": (
        ["werewolves", "shifters", "shifter", "shapeshifters"],
        ["werewolf", "werewolves", "shifters", "shapeshifters"],
    ),
    "zombies": (["zombies", "zombie"], ["zombie", "zombies"]),
    "dragons": (["dragons"], ["dragon", "dragons"]),
    "magic": (["magic"], ["magic", "wizards", "witches", "witch"]),
    "graphic novels": (
        ["graphic-novels", "graphic-novel", "comics", "comic"],
        ["graphic novel", "graphic novels", "comics", "comic books", "comic"],
    ),
    "manga": (["manga", "mangas"], ["manga"]),
    "superheroes": (["superheroes", "superhero", "marvel", "dc-comics"], ["superhero", "superheroes", "marvel", "dc"]),
    "literary fiction": (["literary-fiction"], ["literary fiction", "literary"]),
    "classics": (["classics"], ["classics", "classic"]),
    "christian fiction": (["christian-fiction", "christian", "christian-romance"], ["christian", "christian fiction", "faith"]),
    "lgbtq": (
        ["lgbt", "lgbtq", "queer", "glbt", "m-m", "lesbian", "gay"],
        ["lgbt", "lgbtq", "lgbtq+", "queer", "gay", "lesbian"],
    ),
    "erotica": (["erotica", "erotic-romance", "erotic"], ["erotica", "erotic", "steamy", "spicy"]),
    "western": (["western", "cowboys", "cowboy"], ["western", "westerns", "cowboy", "cowboys"]),
    "adventure": (["adventure", "action-adventure"], ["adventure", "adventures", "action"]),
    "steampunk": (["steampunk"], ["steampunk"]),
    "time travel": (["time-travel"], ["time travel", "time-travel"]),
    "mythology": (["mythology", "fairy-tales"], ["mythology", "myths", "fairy tales", "fairytales", "retellings"]),
    "war": (["war", "military", "wwii"], ["war", "military", "ww2", "wwii", "world war"]),
    "short stories": (
        ["short-stories", "short-story", "anthology", "short-story-collections"],
        ["short stories", "short story", "anthology", "anthologies"],
    ),
    "poetry": (["poetry"], ["poetry", "poems"]),
    "chick lit": (
        ["chick-lit", "chicklit", "womens-fiction", "women-s-fiction"],
        ["chick lit", "chick-lit", "women's fiction", "womens fiction"],
    ),
    "aliens": (["aliens", "alien"], ["alien", "aliens"]),
    "ghosts": (["ghosts"], ["ghost", "ghosts", "ghost stories", "haunted"]),
    "sports": (["sports", "sports-romance"], ["sport", "sports", "football", "baseball"]),
    "christmas": (["christmas", "holiday"], ["christmas", "holiday", "holidays"]),
    "gothic": (["gothic"], ["gothic"]),
    "coming of age": (["coming-of-age"], ["coming of age", "coming-of-age"]),
    "alternate history": (["alternate-history"], ["alternate history"]),
    "magical realism": (["magical-realism"], ["magical realism"]),
    "animals": (["animals", "dogs", "cats"], ["animals", "animal", "dogs", "dog", "cats", "cat"]),
    "mafia": (["mafia"], ["mafia", "mob"]),
    "pirates": (["pirates"], ["pirate", "pirates"]),
    "spy": (["spy", "espionage"], ["spy", "spies", "espionage"]),
    "true crime": (["true-crime"], ["true crime"]),
    "new adult": (["new-adult", "college"], ["new adult", "college"]),
}
ALIAS = {a: k for k, (t, al) in TOPICS.items() for a in al + [k]}
_TAGS = {}


def tags_col():
    if "c" not in _TAGS:
        _TAGS["c"] = pl.read_parquet(f"{DATA}/extra.parquet", columns=["i", "tags"]).sort("i")["tags"].fill_null("")
    return _TAGS["c"]


def topic_lookup(text):
    """Longest topic alias contained in text -> (topic key, alias) or None."""
    low = " " + re.sub(r"[^a-z0-9+\' -]", " ", (text or "").lower()) + " "
    for a in sorted(ALIAS, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", low):
            return ALIAS[a], a
    return None


def topic_mask(key):
    tg = TOPICS[key][0]
    return tags_col().str.contains(r"(?:^|, )(?:" + "|".join(re.escape(t) for t in tg) + r")(?:,|$)").to_numpy()


AUTH = {}


def authors():
    if "df" not in AUTH:
        a = B.group_by("author").agg(pl.col("ratings_count").sum().alias("n"), pl.len().alias("k")).drop_nulls("author")
        AUTH["df"] = a.with_columns(pl.col("author").str.to_lowercase().alias("l")).sort("n", descending=True)
    return AUTH["df"]


def author_lookup(q):
    q = re.sub(r"\s+", " ", (q or "").lower().strip(" .?!'\""))
    q = re.sub(r"'s$", "", q)
    if len(q) < 4:
        return None
    A = authors()
    h = A.filter(pl.col("l") == q)
    if h.height == 0:
        h = A.filter(pl.col("l").str.contains(r"(?:^| )" + re.escape(q) + r"(?:$| )")).filter(pl.col("n") >= 500)
    return None if h.height == 0 else h.row(0, named=True)["author"]


def _norm(t):
    return re.sub(r"^(the|a|an) ", "", re.sub(r"\s*\(.*?\)\s*", "", (t or "").lower()).strip(" .?!\"'"))


def title_lookup(q, strict=True):
    """Title -> book. Exact (series/subtitle-insensitive) matches and series matches ("Harry Potter" -> the series) beat substring hits;
    among them the most-rated wins (exact matches get a 5x boost)."""
    q = (q or "").strip(" .?!\"'“”")
    nq = _norm(q)
    if len(nq) < 3:
        return None
    best = None
    for r in {x["i"]: x for x in search(q, 25) + search(nq, 25)}.values():
        tl = r["title"].lower()
        nt = _norm(r["title"])
        exact = nt == nq
        series = bool(re.search(r"\(" + re.escape(nq) + r",?\s*#", tl))
        pref = nt.startswith(nq) and len(nq) >= max(5, 0.5 * len(nt))
        if not (exact or series or pref):
            continue
        sc = (r["ratings_count"] or 0) * (5 if exact else 1)
        if best is None or sc > best[0]:
            best = (sc, r)
    if best:
        return best[1]
    if strict:
        return None
    h = search(nq, 1)
    return h[0] if h else None


def top_rated(state, limit=30, m=1000.0):
    if not HAS_RATING:  # no ratings in this dataset (e.g. the Gutenberg sample): most popular first
        v = B.filter(filt(state)).sort("ratings_count", descending=True, nulls_last=True).head(limit)
        return None, [bookrow(r) | {"score": None} for r in v.iter_rows(named=True)]
    v = B.filter(filt(state)).filter(pl.col("rating").is_not_null())
    if v.height == 0:
        return 0, []
    C = float(v["rating"].mean())
    n = pl.col("ratings_count").fill_null(0).cast(pl.Float64)
    v = v.with_columns(((n * pl.col("rating") + m * C) / (n + m)).alias("score")).sort("score", descending=True).head(limit)
    return C, [bookrow(r) | {"score": round(r["score"], 3)} for r in v.iter_rows(named=True)]


async def t_top_rated_in_view(limit=10):
    C, rows = top_rated(M["state"], min(int(limit), 50))
    n = B.filter(filt(M["state"])).height
    if C is None:
        return {"books_in_view": n, "books": rows, "ranking": "this dataset has no ratings: most popular (ratings_count) first"}
    return {
        "books_in_view": n,
        "books": rows,
        "ranking": f"Bayesian average: (ratings_count*rating + 1000*{C:.2f})/(ratings_count+1000), i.e. rating weighted by number of ratings",
    }


async def t_zoom(factor=2.0, direction=None):
    """factor>1 zooms in, <1 zooms out (around the current centre or the focused book); direction pans by half a view."""
    x0, x1, y0, y1 = _cur_vp()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    f = float(factor or 1)
    if direction:
        dx = {"left": -1, "west": -1, "right": 1, "east": 1}.get(direction, 0)
        dy = {"up": 1, "north": 1, "down": -1, "south": -1}.get(direction, 0)
        cx += dx * rx
        cy += dy * ry
    if f > 1 and FOCUS["book"] and x0 <= FOCUS["book"]["x"] <= x1 and y0 <= FOCUS["book"]["y"] <= y1 and not direction:
        cx, cy = FOCUS["book"]["x"], FOCUS["book"]["y"]
    rx /= f
    ry /= f
    if rx * 2 >= (X1 - X0) * 0.98:
        return await t_set_view(reset=True)
    out = await set_state(
        {"viewport": _vp(cx, cy, rx, ry)},
        what=f"zoom {'in' if f > 1 else 'out' if f < 1 else ''} ×{f:g}" + (f" pan {direction}" if direction else ""),
    )
    return out


async def t_zoom_to_topic(topic):
    hit = topic_lookup(topic)
    key = hit[0] if hit else None
    if not key:
        raise HTTPException(404, f"unknown topic {topic!r}; known: {sorted(TOPICS)}")
    base = _mask(M["state"])
    w = best_window(topic_mask(key), base, mode="topic")
    if not w:
        raise HTTPException(404, f"no {key} books under the current filters")
    cx, cy, rx, ry, info = w
    FOCUS["book"] = None
    out = await set_state({"viewport": _vp(cx, cy, rx, ry)}, what=f"zoom to the {key} area")
    await _broadcast({"type": "focus", "book": None})
    return {
        "topic": key,
        **info,
        "note": "genre is not a filter; this zooms to where books tagged " + key + " concentrate on the description map",
    } | out


async def t_zoom_to_author(author):
    name = author_lookup(author)
    if not name:
        raise HTTPException(404, f"no author matching {author!r}")
    m = (B["author"] == name).fill_null(False).to_numpy()
    w = np.log10(B["ratings_count"].fill_null(0).to_numpy() + 10.0)
    win = best_window(m, np.ones(B.height, bool), w, mode="author")
    cx, cy, rx, ry, info = win
    top = B.filter(pl.col("author") == name).sort("ratings_count", descending=True).head(5)
    tb = top.row(0, named=True)
    rx = max(rx, 0.6)
    ry = max(ry, 0.5)
    if not (cx - rx <= tb["x"] <= cx + rx and cy - ry <= tb["y"] <= cy + ry):  # always show their best-known book
        x0, x1 = min(cx - rx, tb["x"] - 0.3), max(cx + rx, tb["x"] + 0.3)
        y0, y1 = min(cy - ry, tb["y"] - 0.3), max(cy + ry, tb["y"] + 0.3)
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    FOCUS["book"] = bookrow(top.row(0, named=True))
    out = await set_state({"viewport": _vp(cx, cy, rx, ry)}, what=f"zoom to books by {name}")
    await _broadcast({"type": "focus", "book": FOCUS["book"]})
    return {"author": name, "books": [bookrow(r) for r in top.iter_rows(named=True)], **info} | out


def _neigh_r(x, y):
    """Smallest square around (x,y) holding ~400 books (all books, not filtered)."""
    if "I" not in _TAGS:
        _TAGS["I"] = _integral(np.ones(B.height, bool))[1]
    I = _TAGS["I"]
    cx = int(np.clip((x - X0) / (X1 - X0) * NX, 0, NX - 1))
    cy = int(np.clip((y - Y0) / (Y1 - Y0) * NY, 0, NY - 1))
    for r in range(2, 40):
        if _box(I, cx, cy, r) >= 400:
            return max((r + 0.5) * CELL, 0.35)
    return 1.5


TOOLS = {
    "get_view_state": (
        t_get_view_state,
        "What the human currently sees: map zoom, the four range-slider filters (year, rating, ratings_count, pages; null = not set), who changed it last, recent changes, and a summary (books in view, top books by ratings_count, years, avg rating).",
        {},
    ),
    "list_books_in_view": (
        t_list_books_in_view,
        "Books inside the current map zoom AND all selections, most-rated first.",
        {"limit": {"type": "integer", "default": 20}, "offset": {"type": "integer", "default": 0}},
    ),
    "search_book": (
        t_search_book,
        "Find books by title substring (prefix matches first, then most rated). Returns i, title, author, year, rating, map x/y.",
        {"title": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
    ),
    "set_view": (
        t_set_view,
        "Zoom the description map to a rectangle in UMAP coordinates (the hist2d re-bins), or reset=true for the full map.",
        {
            "x_min": {"type": "number"},
            "x_max": {"type": "number"},
            "y_min": {"type": "number"},
            "y_max": {"type": "number"},
            "reset": {"type": "boolean", "default": False},
        },
    ),
    "set_filters": (
        t_set_filters,
        "Move the range sliders under the map (shared FlexViz state; the UI updates live). Each is [min,max]: year, rating (1-5 average), ratings_count (number of ratings / popularity, e.g. [1000,1000000]), pages (page count; books without one are excluded while set). Other filters are kept unless clear=true; pass [] to remove one. region=[x_min,x_max,y_min,y_max] selects a map rectangle.",
        {
            "year": {"type": "array", "items": {"type": "number"}},
            "rating": {"type": "array", "items": {"type": "number"}},
            "ratings_count": {"type": "array", "items": {"type": "number"}},
            "pages": {"type": "array", "items": {"type": "number"}},
            "region": {"type": "array", "items": {"type": "number"}},
            "clear": {"type": "boolean", "default": False},
        },
    ),
    "focus_book": (
        t_focus_book,
        "Find a book by title (or index i) and zoom the map to it and its neighbourhood (books with similar blurbs). Keeps the slider filters unless they would hide the book.",
        {
            "title": {"type": "string"},
            "i": {"type": "integer"},
            "radius": {
                "type": "number",
                "description": "optional half-width in map units; default = neighbourhood of ~400 books",
            },
            "keep_selections": {"type": "boolean", "default": True},
        },
    ),
    "zoom": (
        t_zoom,
        "Relative zoom/pan of the map: factor>1 zooms in (2 = twice as close), factor<1 zooms out (0.5); direction = left|right|up|down pans by half a view.",
        {"factor": {"type": "number", "default": 2.0}, "direction": {"type": "string", "enum": ["left", "right", "up", "down"]}},
    ),
    "zoom_to_topic": (
        t_zoom_to_topic,
        "Genre/topic is NOT a filter. This zooms the map to where books tagged with a topic concentrate (e.g. science fiction, fantasy, romance, mystery, thriller, horror, historical fiction, young adult, humor, dystopian, vampires, dragons...). Respects the current slider filters.",
        {"topic": {"type": "string"}},
    ),
    "zoom_to_author": (
        t_zoom_to_author,
        "Zoom the map to where an author's books sit and list their most-rated books.",
        {"author": {"type": "string"}},
    ),
    "top_rated_in_view": (
        t_top_rated_in_view,
        'Best books in the current zoom + filters, ranked by rating weighted by number of ratings (Bayesian average). Use for "best", "top rated", "what should I read here".',
        {"limit": {"type": "integer", "default": 10}},
    ),
    "similar_books": (
        t_similar_books,
        "Nearest books by description embedding (not by map position). Flags which ones are in the current view.",
        {"title": {"type": "string"}, "i": {"type": "integer"}, "k": {"type": "integer", "default": 10}},
    ),
    "recommend_start": (
        t_recommend_start,
        "Start the recommendation wizard on the books currently in view (zoom + slider filters; best ~50 by Bayesian rating, boosted by words from query). Returns the first multiple-choice question (options with id) or, if few books, the final top 3.",
        {"query": {"type": "string", "description": "what the reader is looking for, in their words"}},
    ),
    "recommend_answer": (
        t_recommend_answer,
        "Answer the current wizard question with option id (or exact label). Returns the next question or the final top 3 with summary and reason.",
        {"session": {"type": "string"}, "option": {"type": "integer"}},
    ),
    "get_user_profile": (
        t_get_user_profile,
        "The user's reading profile (user_profile.md and, if present, goodreads_library_export.csv in ATLAS_HOME matched to this catalogue).",
        {},
    ),
}


def tool_list():
    return [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": p}} for n, (f, d, p) in TOOLS.items()]


async def call_tool(name, args):
    if name not in TOOLS:
        raise HTTPException(404, f"unknown tool {name}")
    return await TOOLS[name][0](**(args or {}))
