"""Tools and loop plumbing for synthetic training data: highlight_books, get_book_details, web_lookup (cache + offline replay),
similar_books(only_in_view), required arguments, compacted results, the shared agent loop with its trace hook, and the
Anthropic teacher adapter (fake client, no network). In-process on the bundled sample."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from atlas_books import agent, providers, tools
from atlas_books.state import HIGHLIGHT, view_state


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def fresh_map():
    async def reset():
        await tools.t_set_filters(clear=True)
        await tools.t_set_view(reset=True)

    run(reset())
    yield
    run(reset())


# ---------------------------------------------------------------- schemas


def test_required_in_schemas_and_enforced():
    T = {t["name"]: t for t in tools.tool_list()}
    assert T["zoom_to_topic"]["inputSchema"]["required"] == ["topic"]
    assert tools.TOOLS["web_lookup"] and "web_lookup" in tools.REQUIRED
    assert "required" not in T["focus_book"]["inputSchema"]  # title OR i: checked by the guard
    assert "missing required" in agent._valid("zoom_to_topic", {}, tools.tool_list())
    assert agent._valid("zoom_to_topic", {"topic": "horror"}, tools.tool_list()) is None


def test_new_tools_visible_to_llm_loop():
    names = {t["function"]["name"] for t in agent.llm_tools(tools.tool_list())}
    assert {"highlight_books", "get_book_details"} <= names
    assert not names & set(agent.LLM_HIDDEN)
    assert "only_in_view" in tools.TOOLS["similar_books"][2]


def test_guards_for_new_tools():
    assert agent._guard("highlight_books", {}, "") == "highlight_books needs titles or ids"
    assert "at most 10" in agent._guard("highlight_books", {"ids": list(range(11))}, "")
    assert agent._guard("get_book_details", {}, "") == "get_book_details needs title or i"
    assert agent._guard("get_book_details", {"i": 0}, "") is None


# ---------------------------------------------------------------- get_book_details


def test_get_book_details_by_title_and_id():
    d = run(tools.t_get_book_details(title="Dracula"))
    assert d["i"] == 5 and d["author"] == "Bram Stoker"
    assert isinstance(d["tags"], list) and len(d["description"]) <= 603
    assert run(tools.t_get_book_details(i=5))["title"] == "Dracula"
    c = agent.compact_result("get_book_details", d)
    assert c["book"].startswith("Dracula — Bram Stoker") and "i=5" in c["book"]
    assert set(c) == {"book", "series", "pages", "tags", "description"}


def test_get_book_details_unknown():
    with pytest.raises(HTTPException) as e:
        run(tools.t_get_book_details(title="zzqx no such book qqz"))
    assert e.value.status_code == 404


# ---------------------------------------------------------------- highlight_books


def test_highlight_books_order_and_compact():
    r = run(tools.t_highlight_books(titles=["Dracula", "Pride and Prejudice"], label="test"))
    assert [b["title"] for b in r["highlighted"]] == ["Dracula", "Pride and Prejudice"]
    assert [x["rank"] for x in HIGHLIGHT["recs"]] == [1, 2]
    assert view_state()["summary"]["recommended"] == ["1. Dracula", "2. Pride and Prejudice"]
    c = agent.compact_result("highlight_books", r)
    assert c["ok"] and c["highlighted"][0].startswith("1. Dracula — Bram Stoker")


def test_highlight_books_by_id_and_errors_keep_old_highlights():
    run(tools.t_highlight_books(ids=[5]))
    with pytest.raises(HTTPException) as e:
        run(tools.t_highlight_books(titles=["Dracula", "zzqx no such book qqz"]))
    assert e.value.status_code == 404
    assert [x["title"] for x in HIGHLIGHT["recs"]] == ["Dracula"]  # unchanged after the failed call
    with pytest.raises(HTTPException):
        run(tools.t_highlight_books(ids=list(range(11))))
    run(tools.t_set_view(reset=True))
    assert HIGHLIGHT["recs"] == []


def test_highlight_blocks_automatic_recs_after_chat():
    run(tools.t_highlight_books(ids=[5]))
    log = {"similar_books": {"similar": [{"i": 0}]}, "highlighted": True}
    run(tools.recs_from_chat("books like dracula", "", "llm", log))
    assert [x["i"] for x in HIGHLIGHT["recs"]] == [5]


# ---------------------------------------------------------------- similar_books


def test_similar_books_only_in_view_and_compact_fields():
    run(tools.t_focus_book(title="Dracula"))
    r = run(tools.t_similar_books(title="Dracula", k=5, only_in_view=True))
    assert 0 < len(r["similar"]) <= 5 and all(b["in_current_view"] for b in r["similar"])
    c = agent.compact_result("similar_books", r)
    assert c["only_in_view"] is True and c["book"].endswith("i=5")
    assert all(" ratings] i=" in s and s.endswith("in_view") for s in c["similar"])
    plain = agent.compact_result("similar_books", run(tools.t_similar_books(title="Dracula", k=5)))
    assert "only_in_view" not in plain and len(plain["similar"]) == 5


# ---------------------------------------------------------------- web_lookup


def test_web_lookup_fetches_once_then_replays(monkeypatch):
    calls = []

    def fake_search(q, n):
        calls.append(q)
        return [
            {
                "title": "Dracula",
                "url": "https://en.wikipedia.org/wiki/Dracula",
                "snippet": "Dracula is an epistolary novel, told in letters and diaries.",
                "score": 0.8,
            },
            {
                "title": "fan page",
                "url": "https://www.facebook.com/x",
                "snippet": "a long enough snippet from a social media page",
                "score": 0.9,
            },
            {
                "title": "weak",
                "url": "https://example.org/weak",
                "snippet": "a long enough snippet but with a very low score",
                "score": 0.1,
            },
        ]

    monkeypatch.setitem(tools.WEB_PROVIDERS, "tavily", fake_search)
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("ATLAS_WEB_OFFLINE", raising=False)
    a = run(tools.t_web_lookup("Dracula  narration style?"))
    b = run(tools.t_web_lookup("dracula narration style"))  # same normalised query
    assert calls == ["dracula narration style"]
    assert a["cached"] is False and b["cached"] is True and a["results"] == b["results"]
    assert [x["url"] for x in b["results"]] == ["https://en.wikipedia.org/wiki/Dracula"]  # social + low score dropped
    c = agent.compact_result("web_lookup", b)
    assert c == {"query": "dracula narration style", "results": a["results"]} and "score" not in c["results"][0]


def test_web_lookup_offered_only_with_key_or_offline(monkeypatch):
    for k in ("TAVILY_API_KEY", "ATLAS_WEB_OFFLINE", "WEB_SEARCH_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    names = lambda: {t["name"] for t in tools.tool_list()}
    assert "web_lookup" not in names() and "highlight_books" in names()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert "web_lookup" in names()
    monkeypatch.delenv("TAVILY_API_KEY")
    monkeypatch.setenv("ATLAS_WEB_OFFLINE", "1")
    assert "web_lookup" in names()


def test_web_lookup_offline_and_missing_key(monkeypatch):
    monkeypatch.setenv("ATLAS_WEB_OFFLINE", "1")
    with pytest.raises(HTTPException) as e:
        run(tools.t_web_lookup("a query nobody cached 123"))
    assert e.value.status_code == 404
    monkeypatch.delenv("ATLAS_WEB_OFFLINE")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    with pytest.raises(HTTPException) as e:
        run(tools.t_web_lookup("another uncached query 456"))
    assert e.value.status_code == 503 and "TAVILY_API_KEY" in e.value.detail
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "bing")
    with pytest.raises(HTTPException) as e:
        run(tools.t_web_lookup("another uncached query 456"))
    assert "unknown WEB_SEARCH_PROVIDER" in e.value.detail


def test_tavily_request_and_parsing(monkeypatch):
    seen = {}

    def fake_http(req):
        seen["url"], seen["auth"], seen["body"] = req.full_url, req.get_header("Authorization"), json.loads(req.data)
        return {"query": "q", "results": [{"title": "T", "url": "https://x.org", "content": "a  long\n snippet", "score": 0.9}]}

    monkeypatch.setattr(tools, "_http_json", fake_http)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert tools._tavily("red rising narrator", 5) == [
        {"title": "T", "url": "https://x.org", "snippet": "a long snippet", "score": 0.9}
    ]
    assert seen == {
        "url": "https://api.tavily.com/search",
        "auth": "Bearer tvly-test",
        "body": {"query": "red rising narrator", "max_results": 8, "search_depth": "basic"},
    }


# ---------------------------------------------------------------- shared loop + trace hook


def scripted(rounds):
    """A fake model: each round is a list of (tool, args) or a final string. Appends Ollama-shaped messages like ollama_step."""

    async def step(msgs, T, k):
        r = rounds[k]
        if isinstance(r, str):
            msgs.append({"role": "assistant", "content": r})
            return {"content": r}, [], {"round": k}
        calls = [{"id": f"c{k}{j}", "function": {"name": n, "arguments": dict(a)}} for j, (n, a) in enumerate(r)]
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        return {"content": ""}, calls, {"round": k}

    return step


ROUNDS = [
    [("focus_book", {"title": "Dracula"})],
    [("set_filters", {"year": ["1800", 1900], "clear": True}), ("no_such_tool", {})],
    "Here is **Dracula**.",
]


def run_loop(hook=None):
    run(tools.t_set_filters(clear=True))
    run(tools.t_set_view(reset=True))
    msgs = agent.build_messages([{"role": "user", "content": "show dracula, 19th century only"}], view_state)
    trace = []
    text = run(
        agent.run_agent_loop(
            msgs, tools.tool_list(), tools.call_tool, trace, "show dracula, 19th century only", scripted(ROUNDS), view_state, hook
        )
    )
    return text, trace, msgs


def test_build_messages_matches_student_prompt():
    msgs = agent.build_messages([{"role": "user", "content": "hi"}], view_state)
    assert msgs[0] == {"role": "system", "content": agent.SYSTEM} and len(msgs) == 2
    assert msgs[1]["content"].startswith("CURRENT VIEW: ") and msgs[1]["content"].endswith("\n\nUSER: hi")


def test_trace_hook_records_and_does_not_change_behaviour():
    ev = []
    text, trace, msgs = run_loop(ev.append)
    text0, trace0, msgs0 = run_loop(None)
    assert (text, trace, msgs) == (text0, trace0, msgs0)
    assert text == "Here is **Dracula**."
    assert [e["type"] for e in ev] == ["model", "tool", "model", "tool", "tool", "model", "final"]
    t1, t2, t3 = [e for e in ev if e["type"] == "tool"]
    assert t1["state"]["focus"] == "Dracula" and t1["raw_result"]["focus"]["i"] == 5
    assert json.loads(t1["result"]) == json.loads(msgs[3]["content"])  # what the model saw
    assert t2["raw_args"] == {"year": ["1800", 1900], "clear": True}  # raw model output
    assert t2["args"] == {"year": [1800.0, 1900.0]}  # after _valid/_guard: clear stripped, numbers coerced
    assert t2["state"]["filters"] == {"year": [1800.0, 1900.0]}
    assert t3["error"].startswith("unknown tool 'no_such_tool'") and t3["raw_result"] is None
    assert ev[0]["tool_calls"][0]["function"]["name"] == "focus_book"


def test_ollama_backend_goes_through_shared_loop(monkeypatch):
    replies = iter(
        [
            {"message": {"content": "", "tool_calls": [{"function": {"name": "focus_book", "arguments": {"title": "Dracula"}}}]}},
            {"message": {"content": "Zoomed to **Dracula**."}},
        ]
    )
    sent = []

    async def fake_post(url, headers, body):
        sent.append(json.loads(json.dumps(body)))
        return next(replies)

    monkeypatch.setattr(agent, "post", fake_post)
    trace, ev = [], []
    out = run(
        agent._ollama(
            [{"role": "user", "content": "show dracula"}],
            tools.tool_list(),
            tools.call_tool,
            trace,
            view_state,
            "m",
            hook=ev.append,
        )
    )
    assert out == "Zoomed to **Dracula**."
    assert trace == [{"tool": "focus_book", "args": {"title": "Dracula"}, "label": "Dracula"}]
    assert sent[0]["model"] == "m" and sent[0]["options"] == {"temperature": 0.15, "num_ctx": 6144}
    assert sent[1]["messages"][-1]["role"] == "tool" and sent[1]["messages"][-1]["tool_name"] == "focus_book"
    assert [e["type"] for e in ev] == ["model", "tool", "model", "final"]


# ---------------------------------------------------------------- Anthropic teacher adapter (fake client)


class FakeAnthropic:
    def __init__(self, outputs):
        self.outputs, self.requests = list(outputs), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=lambda o: o.__dict__)))
        content = self.outputs.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop)


def B(**kw):
    return SimpleNamespace(**kw)


def test_anthropic_teacher_shares_loop_and_hides_hint():
    fake = FakeAnthropic(
        [
            [B(type="thinking", thinking=""), B(type="tool_use", id="tu_1", name="focus_book", input={"title": "Dracula"})],
            [B(type="text", text="Zoomed to **Dracula**.")],
        ]
    )
    step = providers.teacher_step(hint="focus first", cfg={"provider": "anthropic", "model": "claude-opus-5-5"}, client=fake)
    hist = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "show dracula"},
    ]
    msgs = agent.build_messages(hist, view_state)
    trace = []
    out = run(agent.run_agent_loop(msgs, tools.tool_list(), tools.call_tool, trace, "show dracula", step, view_state))
    assert out == "Zoomed to **Dracula**." and trace[0]["tool"] == "focus_book"
    r1, r2 = fake.requests
    assert r1["model"] == "claude-opus-5-5" and r1["output_config"] == {"effort": "high"} and "temperature" not in r1
    assert r1["system"][0]["text"] == agent.SYSTEM and "focus first" in r1["system"][1]["text"]
    assert [m["role"] for m in r1["messages"]] == ["user", "assistant", "user"]
    assert {t["name"] for t in r1["tools"]} == {t["function"]["name"] for t in agent.llm_tools(tools.tool_list())}
    last = r2["messages"][-1]["content"][0]
    assert last["type"] == "tool_result" and last["tool_use_id"] == "tu_1" and last["content"] == msgs[-2]["content"]
    assert r2["messages"][-2]["content"][0]["type"] == "thinking"  # full response content goes back unchanged
    assert all("focus first" not in json.dumps(m) for m in msgs)  # canonical (training) messages never carry the hint
    assert msgs[-3]["tool_calls"][0] == {"id": "tu_1", "function": {"name": "focus_book", "arguments": {"title": "Dracula"}}}


def test_teacher_config_from_env(monkeypatch):
    monkeypatch.delenv("TEACHER_PROVIDER", raising=False)
    monkeypatch.setenv("TEACHER_EFFORT", "xhigh")
    assert providers.teacher_config() == {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "xhigh"}
    monkeypatch.setenv("TEACHER_PROVIDER", "openai")
    monkeypatch.setenv("TEACHER_MODEL", "some-model")
    c = providers.teacher_config()
    assert c["base_url"].startswith("https://") and c["key_env"] == "OPENAI_API_KEY" and c["temperature"] == 0.15
