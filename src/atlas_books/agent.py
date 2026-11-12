"""Chat agent over the shared FlexViz state. Provider via env:
  AGENT_PROVIDER = xai | openai | anthropic | none   (default: first of XAI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY that is set)
  AGENT_MODEL    = model id (defaults below)          AGENT_BASE_URL = optional OpenAI-compatible base URL override
Without a key the panel runs a small command interpreter that calls the same tools."""

import os
import json
import re
import csv
import asyncio
import urllib.request
import urllib.error
import polars as pl
from . import librarian, paths


class HTTPError(Exception):
    pass


def _post_sync(url, headers, body):
    rq = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST"
    )
    try:
        with urllib.request.urlopen(rq, timeout=240) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise HTTPError(f"LLM error {e.code}")


async def post(url, headers, body):
    return await asyncio.to_thread(_post_sync, url, headers, body)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1", "grok-4"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4.1-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "claude-sonnet-4-5"),
}
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
REC_MODEL = lambda: os.environ.get(
    "REC_MODEL", "ministral-3:3b"
)  # recommendation questions: structured JSON + validation, speed matters
# ---- Librarian model choice: chosen in the UI, kept in memory + librarian_model.json (0600). The API key never leaves the server in full. ----
CONF_PATH = str(paths.home() / "librarian_model.json")
CONF = {"kind": "ollama", "model": None, "base_url": None, "api_key": None}


def _load_conf():
    try:
        d = json.load(open(CONF_PATH))
        if d.get("kind") in ("ollama", "openai"):
            CONF.update({k: d.get(k) for k in CONF})
    except Exception:
        pass


_load_conf()


def save_conf(d):
    CONF.update({k: d.get(k) for k in CONF})
    tmp = CONF_PATH + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(CONF, f)
    os.replace(tmp, CONF_PATH)
    os.chmod(CONF_PATH, 0o600)


def key_hint(k):
    return ("••••" + k[-4:]) if k and len(k) >= 8 else ("••••" if k else None)


def public_conf():
    c = {"kind": CONF["kind"], "model": OLLAMA_MODEL() if CONF["kind"] == "ollama" else CONF["model"]}
    if CONF["kind"] == "openai":
        c |= {"base_url": CONF["base_url"], "key": key_hint(CONF["api_key"])}
    return c


OLLAMA_MODEL = lambda: (
    CONF["model"] if CONF["kind"] == "ollama" and CONF["model"] else os.environ.get("AGENT_MODEL", "ministral-3:8b")
)  # Librarian (eval: 8b 18/21 vs 3b 16/21 LLM-only)
_TG = {"t": 0, "tags": None}


def ollama_tags(fresh=False):
    import time

    if not fresh and time.time() - _TG["t"] < 20:
        return _TG["tags"]
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=1.5) as r:
            tags = [
                {"name": m["name"], "size": m.get("size"), "params": (m.get("details") or {}).get("parameter_size")}
                for m in json.loads(r.read())["models"]
            ]
    except Exception:
        tags = None
    _TG.update(t=time.time(), tags=tags)
    return tags


_OC = {"t": 0}  # kept for callers that reset the cache


def _has(m):
    t = ollama_tags()
    return bool(t) and any(x["name"] in (m, m + ":latest") for x in t)


def ollama_ok():
    """Local Ollama is reachable and has the recommendation model (used by rec.py)."""
    if _OC["t"] == 0:
        _TG["t"] = 0
        _OC["t"] = 1
    return _has(REC_MODEL())


def librarian_ollama_ok():
    return _has(OLLAMA_MODEL())


def provider():
    if CONF["kind"] == "openai" and CONF["base_url"] and CONF["model"]:
        return "custom"
    p = os.environ.get("AGENT_PROVIDER", "").lower()
    if p == "none":
        return None
    if p == "ollama":
        return "ollama" if librarian_ollama_ok() else None
    if p in DEFAULTS and os.environ.get(DEFAULTS[p][0]):
        return p
    if not p and librarian_ollama_ok():
        return "ollama"
    for k, (env, _, _) in DEFAULTS.items():
        if os.environ.get(env):
            return k
    return None


def status():
    p = provider()
    if p == "ollama":
        return {"provider": "ollama", "model": OLLAMA_MODEL(), "host": OLLAMA, "mode": "llm"}
    if p == "custom":
        return {"provider": "custom", "model": CONF["model"], "base_url": CONF["base_url"], "mode": "llm"}
    return {
        "provider": p,
        "model": (os.environ.get("AGENT_MODEL") or DEFAULTS[p][2]) if p else None,
        "mode": "llm"
        if p
        else f"commands (no LLM reachable: start Ollama at {OLLAMA} with `ollama pull {OLLAMA_MODEL()}`, or set an API key)",
    }


def _scrub(msg, key):
    msg = str(msg)
    if key:
        msg = msg.replace(key, "••••")
    return re.sub(r"(sk|xai|gsk|key)[-_][A-Za-z0-9_\-]{8,}", "••••", msg)[:240]


def _openai_post_sync(base, key, body, timeout=120):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    rq = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error")
        except Exception:
            detail = None
        if isinstance(detail, dict):
            detail = detail.get("message")
        raise HTTPError(_scrub(f"LLM error {e.code}" + (f": {detail}" if detail else ""), key))
    except urllib.error.URLError as e:
        raise HTTPError(_scrub(f"cannot reach {base}: {getattr(e, 'reason', e)}", key))


async def test_model(d):
    """One tiny call; returns {ok, latency_s, error?}. Never echoes the key."""
    import time

    t0 = time.time()
    try:
        if d["kind"] == "ollama":
            if not any(x["name"] == d["model"] for x in (ollama_tags(True) or [])):
                return {"ok": False, "error": f"{d['model']} is not installed in Ollama"}
            r = await post(
                OLLAMA + "/api/chat",
                {},
                {
                    "model": d["model"],
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": 5},
                },
            )
            if "message" not in r:
                return {"ok": False, "error": "unexpected reply from Ollama"}
        else:
            r = await asyncio.to_thread(
                _openai_post_sync,
                d["base_url"],
                d.get("api_key"),
                {"model": d["model"], "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 5},
                30,
            )
            if not (r.get("choices") or [{}])[0].get("message"):
                return {"ok": False, "error": "reply is not OpenAI chat format"}
    except Exception as e:
        return {"ok": False, "error": _scrub(getattr(e, "args", [e])[0] if getattr(e, "args", None) else e, d.get("api_key"))}
    return {"ok": True, "latency_s": round(time.time() - t0, 2)}


def load_profile(B=None):
    out = {"profile_md": None, "goodreads_export": None}
    p = str(paths.home() / "user_profile.md")
    if os.path.exists(p):
        out["profile_md"] = open(p).read()[:4000]
    g = str(paths.home() / "goodreads_library_export.csv")
    if os.path.exists(g):
        rows = list(csv.DictReader(open(g, encoding="utf-8", errors="replace")))
        read = [r for r in rows if (r.get("Exclusive Shelf") or "") == "read"]
        top = sorted(read, key=lambda r: -int(r.get("My Rating") or 0))[:30]
        info = {
            "rows": len(rows),
            "read": len(read),
            "to_read": sum(1 for r in rows if r.get("Exclusive Shelf") == "to-read"),
            "top_rated": [{"title": r["Title"], "author": r.get("Author"), "my_rating": r.get("My Rating")} for r in top],
        }
        if B is not None:
            low = B.select(pl.col("title").str.to_lowercase().str.replace(r"\s*\(.*\)$", "").alias("t"), "i")
            want = {r["Title"].lower().split(" (")[0]: r for r in rows}
            hit = low.filter(pl.col("t").is_in(list(want)))
            info["matched_in_catalogue"] = hit.height
            info["matched_i"] = hit["i"].head(200).to_list()
        out["goodreads_export"] = info
    else:
        out["goodreads_export"] = "not provided: copy goodreads_library_export.csv into ~/.atlas/ (or ATLAS_HOME)"
    return out


def _content_text(content):
    """Normalise an assistant message's content to a string; some providers (e.g. Mistral) return a list of
    content blocks (text/thinking) instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


async def _openai(p, messages, tools, call_tool, trace):
    env, base, model = DEFAULTS[p]
    base = os.environ.get("AGENT_BASE_URL", base)
    model = os.environ.get("AGENT_MODEL", model)
    T = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
        for t in tools
    ]
    msgs = [{"role": "system", "content": SYSTEM}] + messages
    for _ in range(8):
        r = await post(
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {os.environ[env]}"},
            {"model": model, "messages": msgs, "tools": T},
        )
        m = r["choices"][0]["message"]
        msgs.append({**m, "content": _content_text(m.get("content"))})
        if not m.get("tool_calls"):
            return _content_text(m.get("content"))
        for tc in m["tool_calls"]:
            args = json.loads(tc["function"]["arguments"] or "{}")
            res = await call_tool(tc["function"]["name"], args)
            trace.append({"tool": tc["function"]["name"], "args": args})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res, default=str)[:12000]})
    return "(stopped after 8 tool rounds)"


async def _anthropic(messages, tools, call_tool, trace):
    env, base, model = DEFAULTS["anthropic"]
    model = os.environ.get("AGENT_MODEL", model)
    T = [{"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]} for t in tools]
    msgs = list(messages)
    for _ in range(8):
        out = await post(
            f"{base}/messages",
            {"x-api-key": os.environ[env], "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": 1500, "system": SYSTEM, "messages": msgs, "tools": T},
        )
        msgs.append({"role": "assistant", "content": out["content"]})
        uses = [b for b in out["content"] if b["type"] == "tool_use"]
        if not uses:
            return "".join(b.get("text", "") for b in out["content"] if b["type"] == "text")
        res = []
        for u in uses:
            v = await call_tool(u["name"], u["input"])
            trace.append({"tool": u["name"], "args": u["input"]})
            res.append({"type": "tool_result", "tool_use_id": u["id"], "content": json.dumps(v, default=str)[:12000]})
        msgs.append({"role": "user", "content": res})
    return "(stopped after 8 tool rounds)"


SYSTEM_TEMPLATE = """You are Atlas AI, the concise book-discovery guide built into Atlas. If asked your name, you are Atlas AI. You and the user share ONE live map of {corpus}: every book is a dot; books with similar blurbs sit close together (dot size = number of ratings, colour = average rating). Under the map are four range sliders: year, rating (1-5), ratings_count, pages. Genre is NOT a filter.{rating_note}
Each turn you get CURRENT VIEW (zoom, filters, books in view, top rated) and USER PROFILE.
You control the map with tools; the user's screen updates immediately:
- focus_book(title): zoom to one book and its neighbourhood (use for "show X", "where is X", "books like X").
- zoom_to_topic(topic): zoom to where a genre/theme concentrates (sci-fi, fantasy, romance, mystery, horror...).
- zoom_to_author(author); zoom(factor) (>1 in, <1 out, direction pans); set_view(reset=true) for the whole map.
- set_filters(year=[a,b], rating=[a,b], ratings_count=[a,b], pages=[a,b]): other sliders are KEPT automatically. Never use clear=true unless the user explicitly asks to reset or clear all filters. Pass [] to remove one filter.
- top_rated_in_view, list_books_in_view, similar_books (only_in_view=true ranks neighbours under the current zoom and filters), search_book, get_view_state for information.
- get_book_details(i or title): blurb, tags, pages, series; for finalists only. web_lookup(query): outside facts, only when the data cannot answer.
- highlight_books(ids or titles): ring your final picks on the map, exactly as many as the user asked for.
Rules: if the user asks to see, filter, narrow or move anything, CALL THE TOOL (do not just describe it). Then answer in at most 4 sentences or 5 bullets, recommending only books from tool results or CURRENT VIEW (title, author, year, rating). No preamble."""


def system_prompt(corpus: str = "your books", has_rating: bool = True) -> str:
    """The Atlas AI system prompt for one dataset (corpus = its description from meta.json)."""
    note = (
        ""
        if has_rating
        else " This dataset has no ratings: ratings_count is its popularity measure and the rating slider is unused."
    )
    return SYSTEM_TEMPLATE.format(corpus=corpus, rating_note=note)


SYSTEM = system_prompt()


def compact_view(v):
    s = v["summary"]
    return {
        "view": s["description"],
        "last_change_by": v["last_change_by"],
        "books_in_view": s["books_in_view"],
        "top_books_in_view": [
            f"{b['title']} — {b['author']} ({b['year']}) ★{b['rating']} [{b['ratings_count']:,} ratings]"
            for b in s["top_books"][:10]
        ],
        "filters": {k: v for k, v in (s.get("filters") or {}).items() if v},
        "focus": (s["focus"] or {}).get("title"),
    }


def compact_result(name, r):
    """Small-model friendly tool results."""
    if not isinstance(r, dict):
        return r
    if "error" in r:
        return r
    bk = lambda b: (
        f"{b['title']} — {b['author']} ({b['year']}) ★{b.get('rating')}"
        + (f" sim={b['similarity']}" if "similarity" in b else "")
    )
    if name == "get_view_state":
        return compact_view(r)
    if name == "list_books_in_view":
        return {"books_in_view": r["books_in_view"], "books": [bk(b) + f" i={b['i']}" for b in r["books"]]}
    if name == "search_book":
        return {"matches": [bk(b) + f" i={b['i']}" for b in r["matches"]]}
    if name == "similar_books":  # ratings count, id and in-view flag let the model filter, rank and pick the right edition
        return {
            "book": bk(r["book"]) + f" i={r['book']['i']}",
            "similar": [
                bk(b)
                + f" [{b.get('ratings_count') or 0:,} ratings] i={b['i']}"
                + (" in_view" if b.get("in_current_view") else "")
                for b in r["similar"]
            ],
        } | ({"only_in_view": True} if r.get("only_in_view") else {})
    if name == "get_book_details":
        return {
            "book": bk(r) + f" [{r.get('ratings_count') or 0:,} ratings] i={r['i']}",
            "series": f"{r['series']['name']} #{r['series']['number']}" if r.get("series") else None,
            "pages": r.get("pages"),
            "tags": r.get("tags"),
            "description": r.get("description"),
        }
    if name == "highlight_books":
        s = r["summary"]
        return {
            "ok": True,
            "highlighted": [
                f"{b['rank']}. " + bk(b) + ("" if b.get("in_current_view") else " (hidden by filters)") for b in r["highlighted"]
            ],
            "now_in_view": {"description": s["description"], "books_in_view": s["books_in_view"]},
        }
    if name == "web_lookup":
        return {
            "query": r["query"],
            "results": [{"title": x["title"], "url": x["url"], "snippet": x["snippet"]} for x in r["results"]],
        }
    if name == "get_user_profile":
        return r
    if name == "top_rated_in_view":
        return {
            "books_in_view": r["books_in_view"],
            "top_rated": [bk(b) + f" [{b['ratings_count']:,} ratings] i={b['i']}" for b in r["books"]],
        }
    out = {"ok": True, "applied_in_browser": r.get("applied_in_browser")}
    if "focus" in r and r["focus"]:
        out["focused"] = bk(r["focus"])
    if "author" in r:
        out["author"] = r["author"]
        out["their_books"] = [bk(b) + f" i={b['i']}" for b in r.get("books", [])]
    if "topic" in r:
        out["topic"] = r["topic"]
        out["topic_lift"] = r.get("lift")
    if "summary" in r:
        out["now_in_view"] = {
            "description": r["summary"]["description"],
            "books_in_view": r["summary"]["books_in_view"],
            "top": [bk(b) for b in r["summary"]["top_books"][:5]],
        }
    return out


RESET_INTENT = re.compile(r"\b(reset|clear|start over|remove (?:all|every)|no filters|all books|everything|from scratch)\b", re.I)


def _guard(name, args, utext):
    """Semantic checks on top of the schema: ranges are [min,max] numbers, no silent clearing of other filters."""
    if name == "set_filters":
        if args.get("clear") and not RESET_INTENT.search(utext or ""):
            args.pop("clear")
        for k in ("year", "rating", "ratings_count", "pages"):
            v = args.get(k)
            if v is None:
                continue
            if not isinstance(v, list) or len(v) not in (0, 2):
                return f"{k} must be [min,max] or []"
            try:
                args[k] = sorted(float(x) for x in v)
            except (TypeError, ValueError):
                return f"{k} values must be numbers"
        if not any(args.get(k) is not None for k in ("year", "rating", "ratings_count", "pages", "region")) and not args.get(
            "clear"
        ):
            return "set_filters needs at least one of year, rating, ratings_count, pages"
    if name == "focus_book":
        if not args.get("title") and args.get("i") is None:
            return "focus_book needs title"
        if args.get("keep_selections") is False and not RESET_INTENT.search(utext or ""):
            args["keep_selections"] = True
    if name in ("get_book_details", "similar_books") and not args.get("title") and args.get("i") is None:
        return f"{name} needs title or i"
    if name == "highlight_books":
        n = len(args.get("titles") or []) + len(args.get("ids") or [])
        if n == 0:
            return "highlight_books needs titles or ids"
        if n > 10:
            return "highlight_books takes at most 10 books"
    if name == "set_view" and not args.get("reset") and any(args.get(k) is None for k in ("x_min", "x_max", "y_min", "y_max")):
        return "set_view needs x_min,x_max,y_min,y_max or reset=true; to zoom to a book use focus_book, to zoom in/out use zoom"
    if name == "zoom":
        try:
            f = float(args.get("factor", 2))
        except (TypeError, ValueError):
            return "factor must be a number"
        if not 0.05 <= f <= 20:
            return "factor must be between 0.05 and 20"
    return None


def _valid(name, args, tools):
    t = next((t for t in tools if t["name"] == name), None)
    if not t:
        return f"unknown tool {name!r}; use one of {[x['name'] for x in tools]}"
    props = t["inputSchema"].get("properties", {})
    bad = [k for k in args if k not in props]
    if bad:
        return f"unknown argument(s) {bad} for {name}; allowed: {list(props)}"
    miss = [k for k in t["inputSchema"].get("required", []) if args.get(k) in (None, "")]
    if miss:
        return f"missing required argument(s) {miss} for {name}"
    for k, v in args.items():
        ty = props[k].get("type")
        if ty == "array" and not isinstance(v, list):
            args[k] = [v] if not isinstance(v, str) or "," not in v else [x.strip() for x in v.split(",")]
        if ty in ("number", "integer") and isinstance(v, str):
            try:
                args[k] = float(v) if ty == "number" else int(float(v))
            except ValueError:
                return f"argument {k} must be a {ty}"
    return None


LLM_HIDDEN = ("get_user_profile", "recommend_start", "recommend_answer")


def llm_tools(tools):
    """Tool schemas in function-calling format, minus the wizard/profile tools the LLM loop never sees."""
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
        for t in tools
        if t["name"] not in LLM_HIDDEN
    ]


def build_messages(messages, view_state):
    """SYSTEM + last 5 history messages + one user message carrying CURRENT VIEW and USER PROFILE: exactly what the student sees."""
    prof = load_profile()
    ptxt = (prof["profile_md"] or "")[:1200]
    ctx = f"CURRENT VIEW: {json.dumps(compact_view(view_state()), ensure_ascii=False)}\nUSER PROFILE: {ptxt}"
    hist = [m for m in messages[-6:-1]]
    return (
        [{"role": "system", "content": SYSTEM}]
        + hist
        + [{"role": "user", "content": ctx + "\n\nUSER: " + messages[-1]["content"]}]
    )


def snapshot(view_state):
    """Small state summary for traces: viewport, active filters, books in view, focus and highlights."""
    v = view_state()
    s = v["summary"]
    return {
        "viewport": json.loads(json.dumps((v.get("state") or {}).get("viewport"), default=str)),
        "filters": {k: x for k, x in (s.get("filters") or {}).items() if x},
        "description": s["description"],
        "books_in_view": s["books_in_view"],
        "focus": (s.get("focus") or {}).get("title"),
        "highlights": [r["title"] for r in (v.get("highlights") or {}).get("recs") or []],
    }


def ollama_step(model=None, temperature=0.15, num_ctx=6144):
    """One model round against Ollama /api/chat; appends the assistant message (Ollama shape) to msgs."""

    async def step(msgs, T, k):
        body = {
            "model": model or OLLAMA_MODEL(),
            "messages": msgs,
            "tools": T,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        try:
            r = await post(OLLAMA + "/api/chat", {}, body)
        except HTTPError:
            await asyncio.sleep(1)
            r = await post(OLLAMA + "/api/chat", {}, body)
        m = r.get("message", {})
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content", ""), **({"tool_calls": calls} if calls else {})})
        return m, calls, r

    return step


def openai_step(base_url=None, api_key=None, model=None, temperature=0.15, extra_system=None):
    """One model round against an OpenAI-compatible endpoint (default: the one chosen in the UI). extra_system is sent to the
    model as a second system message but never stored in msgs (teacher-only hints)."""

    async def step(msgs, T, k):
        base, key, mdl = base_url or CONF["base_url"], api_key if api_key is not None else CONF["api_key"], model or CONF["model"]
        out = [
            x if x.get("role") != "tool" else {"role": "tool", "tool_call_id": x["tool_call_id"], "content": x["content"]}
            for x in msgs
        ]
        if extra_system:
            out = out[:1] + [{"role": "system", "content": extra_system}] + out[1:]
        body = {"model": mdl, "messages": out, "tools": T, "temperature": temperature}
        try:
            r = await asyncio.to_thread(_openai_post_sync, base, key, body)
        except HTTPError as e:
            if "cannot reach" in str(e):
                raise
            await asyncio.sleep(1)
            r = await asyncio.to_thread(_openai_post_sync, base, key, body)
        m = (r.get("choices") or [{}])[0].get("message") or {}
        raw = m.get("tool_calls") or []
        calls = [{"id": c.get("id") or f"call_{k}_{j}", "function": c.get("function") or {}} for j, c in enumerate(raw)]
        msgs.append(
            {
                "role": "assistant",
                "content": _content_text(m.get("content")),
                **(
                    {
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["function"].get("name"),
                                    "arguments": c["function"].get("arguments")
                                    if isinstance(c["function"].get("arguments"), str)
                                    else json.dumps(c["function"].get("arguments") or {}),
                                },
                            }
                            for c in calls
                        ]
                    }
                    if calls
                    else {}
                ),
            }
        )
        return m, calls, r

    return step


def _plain(x):
    return json.loads(json.dumps(x, default=str))


async def run_agent_loop(msgs, tools, call_tool, trace, utext, step, view_state=None, hook=None, max_rounds=6):
    """The shared Atlas AI loop (student, local models and teacher): step() asks the model and appends its message; every call
    passes _valid then _guard, runs, and the compacted result goes back as a tool message (max 3500 chars). hook(event), if
    given, receives per round a 'model' event (raw output) and per call a 'tool' event (raw and normalised args, raw and
    compacted result, state snapshot), then 'final', 'abort' or 'stop'. The hook only observes: behaviour is the same without it."""
    T = llm_tools(tools)
    errors = 0
    retried = False
    for k in range(max_rounds):
        m, calls, raw = await step(msgs, T, k)
        if hook:
            hook(
                {
                    "type": "model",
                    "round": k,
                    "content": _content_text(m.get("content")),
                    "tool_calls": _plain(calls),
                    "raw": _plain(raw),
                }
            )
        if not calls:
            txt = _content_text(m.get("content")).strip()
            if not txt and not retried:
                retried = True
                msgs.append({"role": "user", "content": "Answer the user now, briefly."})
                if hook:
                    hook({"type": "nudge", "round": k})
                continue
            if hook:
                hook({"type": "final", "round": k, "text": txt})
            return txt
        for c in calls:
            f = c.get("function", {})
            name = f.get("name")
            args = f.get("arguments") or {}
            raw_args = _plain(args)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            err = _valid(name, args, tools) or _guard(name, args, utext)
            rawres = None
            if err:
                trace.append({"tool": name, "args": args, "error": err})
                res = {"error": err + " Fix the arguments and call the tool again."}
                errors += 1
                if errors > 2:
                    if hook:
                        hook(
                            {
                                "type": "tool",
                                "round": k,
                                "call_id": c.get("id"),
                                "name": name,
                                "raw_args": raw_args,
                                "args": _plain(args),
                                "error": err,
                                "raw_result": None,
                                "result": json.dumps(res, ensure_ascii=False),
                                "state": snapshot(view_state) if view_state else None,
                            }
                        )
                        hook({"type": "abort", "round": k, "reason": "3 invalid calls"})
                    return '(I could not turn that into a valid map action; try rephrasing, e.g. "only books after 2010" or "show me <a title>".)'
            else:
                try:
                    rawres = await call_tool(name, args)
                    res = compact_result(name, rawres)
                    trace.append(
                        {"tool": name, "args": args}
                        | ({"label": rawres["focus"]["title"]} if isinstance(rawres, dict) and rawres.get("focus") else {})
                    )
                except Exception as e:
                    res = {"error": str(getattr(e, "detail", e))[:900]}
                    trace.append({"tool": name, "args": args, "error": res["error"]})
            content = json.dumps(res, ensure_ascii=False, default=str)[:3500]
            msgs.append({"role": "tool", "tool_name": name, "tool_call_id": c.get("id") or name, "content": content})
            if hook:
                hook(
                    {
                        "type": "tool",
                        "round": k,
                        "call_id": c.get("id") or name,
                        "name": name,
                        "raw_args": raw_args,
                        "args": _plain(args),
                        "error": err or (res.get("error") if isinstance(res, dict) else None),
                        "raw_result": _plain(rawres),
                        "result": content,
                        "state": snapshot(view_state) if view_state else None,
                    }
                )
    if hook:
        hook({"type": "stop", "round": max_rounds})
    return f"(stopped after {max_rounds} tool rounds)"


async def _ollama(messages, tools, call_tool, trace, view_state, model=None, backend="ollama", hook=None):
    utext = messages[-1]["content"] if messages else ""
    msgs = build_messages(messages, view_state)
    step = ollama_step(model) if backend == "ollama" else openai_step()  # custom endpoint: same prompt, validation, guards
    return await run_agent_loop(msgs, tools, call_tool, trace, utext, step, view_state, hook)


HELP = (
    "No LLM key on this server, so I understand commands: "
    "<b>focus</b> <i>title</i> · <b>similar</b> <i>title</i> · <b>view</b> (what's selected/in view) · <b>books</b> · "
    "<b>years</b> 1990-2000 · <b>rating</b> 4-5 · <b>pages</b> 0-300 · <b>ratings</b> 1000-100000 · <b>clear</b> · <b>reset</b> · <b>profile</b>"
)


def _fmt(rows, n=8):
    return "<br>".join(
        f"{k + 1}. {r['title']} — {r['author'] or ''} ({r['year'] or '?'})" + (f" ★{r['rating']}" if r.get("rating") else "")
        for k, r in enumerate(rows[:n])
    )


async def _commands(text, call_tool, trace):
    t = text.strip()
    low = t.lower()

    async def T(n, **a):
        trace.append({"tool": n, "args": a})
        return await call_tool(n, a)

    if m := re.match(r"(focus|zoom to|show|find)\s+(.+)", low):
        r = await T("focus_book", title=t[len(m.group(1)) :].strip())
        b = r["focus"]
        s = r["summary"]
        return (
            f"Zoomed to <b>{b['title']}</b> ({b['author']}, {b['year']}) at x={b['x']}, y={b['y']}. {s['books_in_view']:,} books in view; most rated: "
            + ", ".join(x["title"] for x in s["top_books"][:3])
            + f". Applied in browser: {r['applied_in_browser']}."
        )
    if m := re.match(r"(similar|like)\s+(?:to\s+)?(.+)", low):
        r = await T("similar_books", title=t.split(None, 1)[1].removeprefix("to ").strip(), k=10)
        return f"Nearest by description to <b>{r['book']['title']}</b>:<br>" + _fmt(r["similar"], 10)
    if m := re.match(r"years?\s+(\d{4})\s*[-–to ]+\s*(\d{4})", low):
        r = await T("set_filters", year=[int(m.group(1)), int(m.group(2))])
        return f"Year filter set. {r['summary']['books_in_view']:,} books in view."
    if m := re.match(r"rating\s+([\d.]+)\s*[-–to ]+\s*([\d.]+)", low):
        r = await T("set_filters", rating=[float(m.group(1)), float(m.group(2))])
        return f"Rating filter set. {r['summary']['books_in_view']:,} books in view."
    if m := re.match(r"pages?\s+(\d+)\s*[-–to ]+\s*(\d+)", low):
        r = await T("set_filters", pages=[int(m.group(1)), int(m.group(2))])
        return f"Page filter set. {r['summary']['books_in_view']:,} books in view."
    if m := re.match(r"ratings\s+(\d+)\s*[-–to ]+\s*(\d+)", low):
        r = await T("set_filters", ratings_count=[int(m.group(1)), int(m.group(2))])
        return f"Ratings-count filter set. {r['summary']['books_in_view']:,} books in view."
    if low in ("clear", "deselect"):
        r = await T("set_filters", clear=True)
        return f"Filters cleared. {r['summary']['books_in_view']:,} books in view."
    if low in ("reset", "home"):
        r = await T("set_view", reset=True)
        return f"Map reset. {r['summary']['books_in_view']:,} books in view."
    if low in ("books", "list"):
        r = await T("list_books_in_view", limit=10)
        return f"{r['books_in_view']:,} books in view, most rated:<br>" + _fmt(r["books"], 10)
    if low.startswith("profile"):
        r = await T("get_user_profile")
        return (
            "Profile: "
            + ("user_profile.md loaded" if r["profile_md"] else "none")
            + "; Goodreads export: "
            + (r["goodreads_export"] if isinstance(r["goodreads_export"], str) else f"{r['goodreads_export']['read']} read")
        )
    if (
        low in ("view", "state", "what did i select", "what do i see", "what is selected", "?")
        or "select" in low
        or "view" in low
    ):
        r = await T("get_view_state")
        s = r["summary"]
        return (
            f"You see: {s['description']} (last change by {r['last_change_by']}). {s['books_in_view']:,} books. "
            f"Filters: "
            + (", ".join(f"{k} {v[0]}–{v[1]}" for k, v in (s.get("filters") or {}).items() if v and k != "region") or "none")
            + ". Most rated: "
            + ", ".join(b["title"] for b in s["top_books"][:3])
            + "."
        )
    return HELP


MAP_TOOLS = {"set_filters", "set_view", "focus_book", "zoom", "zoom_to_topic", "zoom_to_author", "highlight_books"}


def _changes(trace):
    ch = []
    for c in trace:
        if c.get("error") or c["tool"] not in MAP_TOOLS:
            continue
        n, g = c["tool"], c.get("args") or {}
        if n == "set_filters":
            if g.get("clear"):
                ch.append("cleared all filters")
            sets = [librarian.fmt_range(k, v) for k, v in g.items() if k in ("year", "rating", "ratings_count", "pages") and v]
            rm = [k.replace("_", " ") for k, v in g.items() if k in ("year", "rating", "ratings_count", "pages") and v == []]
            if sets:
                ch.append("set " + ", ".join(sets))
            if rm:
                ch.append("removed the " + ", ".join(rm) + " filter")
        elif n == "set_view":
            ch.append("reset the map to the whole atlas" if g.get("reset") else "zoomed to a region")
        elif n == "focus_book":
            ch.append(f"zoomed to {c.get('label') or g.get('title') or 'the book'} and its neighbourhood")
        elif n == "zoom":
            ch.append(
                (
                    "zoomed in"
                    if float(g.get("factor", 2)) > 1
                    else "zoomed out"
                    if float(g.get("factor", 2)) < 1
                    else "moved the map"
                )
                + (f" ({g['direction']})" if g.get("direction") else "")
            )
        elif n == "zoom_to_topic":
            ch.append(f"zoomed to the {g.get('topic')} area")
        elif n == "zoom_to_author":
            ch.append(f"zoomed to books by {g.get('author')}")
        elif n == "highlight_books":
            k = len(g.get("titles") or []) + len(g.get("ids") or [])
            ch.append(f"highlighted {k} book{'s' if k != 1 else ''}")
    return ch


async def chat(messages, tools, call_tool, view_state, H=None, model=None, rules=True, hook=None):
    trace = []
    p = provider()
    text = messages[-1]["content"] if messages else ""
    if model and p == "ollama":
        try:
            with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as r:
                ok = model in [m["name"] for m in json.loads(r.read())["models"]]
        except Exception:
            ok = False
        if not ok:
            model = None
    elif p != "ollama":
        model = None
    used = model or status().get("model")
    if H and rules:
        P = librarian.parse(text, H)
        if librarian.recognised(P):

            async def T(n, **a):
                r = await call_tool(n, a)
                trace.append(
                    {"tool": n, "args": a} | ({"label": r["focus"]["title"]} if isinstance(r, dict) and r.get("focus") else {})
                )
                return r

            try:
                reply, wq = await librarian.run(P, T, H)
                return {
                    "reply": reply,
                    "tool_calls": trace,
                    "via": "rules",
                    "wizard": wq,
                    "mode": "rules",
                    "provider": p,
                    "model": used,
                }
            except Exception as e:
                trace.append({"tool": "rules", "args": {}, "error": str(getattr(e, "detail", e))[:200]})
    try:
        if p == "ollama":
            text = await _ollama(messages, tools, call_tool, trace, view_state, model, hook=hook)
        elif p == "custom":
            text = await _ollama(messages, tools, call_tool, trace, view_state, backend="openai", hook=hook)
        elif p == "anthropic":
            text = await _anthropic(messages, tools, call_tool, trace)
        elif p:
            text = await _openai(p, messages, tools, call_tool, trace)
        else:
            text = await _commands(messages[-1]["content"] if messages else "", call_tool, trace)
    except HTTPError as e:
        text = f"The {used} model returned an error: {_scrub(e, CONF.get('api_key'))}"
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        _OC["t"] = 0
        text = "(LLM unreachable, command mode) " + await _commands(messages[-1]["content"] if messages else "", call_tool, trace)
    except Exception as e:
        text = f"Error: {type(e).__name__}: {_scrub(e, CONF.get('api_key'))}"
    ch = _changes(trace)
    if ch and H:
        n = view_state()["summary"]["books_in_view"]
        text = librarian.confirm(ch, H, n) + "\n\n" + text
    return {
        "reply": text,
        "tool_calls": trace,
        "via": "llm" if p else "commands",
        "mode": status()["mode"],
        "provider": p,
        "model": used,
    }
