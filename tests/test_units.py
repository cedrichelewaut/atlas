"""Unit tests for the deterministic parts: the rules parser, the embedding-text cleaner, import normalisation and the
recommendation wizard (deterministic fallback, no LLM)."""

import asyncio

import polars as pl
import pytest

from atlas_books import build, librarian, textprep

# ---------------------------------------------------------------- librarian.parse

BOOKS = {"dracula": {"i": 1, "title": "Dracula", "author": "Bram Stoker", "year": 1897, "rating": 4.0}}
H = {
    "title": lambda q, strict=True: BOOKS.get(q.lower().strip()),
    "author": lambda q: "Jules Verne" if q.lower().strip() in ("jules verne", "verne") else None,
    "topic": lambda t: next(((k, k) for k in ("science fiction", "horror") if k in t), None),
}


def P(q):
    return librarian.parse(q, H)


@pytest.mark.parametrize(
    "q,field,value",
    [
        ("only after 1990", "year", [1991, librarian.YMAX]),
        ("between 1990 and 2000", "year", [1990, 2000]),
        ("books from the 80s", "year", [1980, 1989]),
        ("before 1950", "year", [librarian.YMIN, 1949]),
        ("4+ stars", "rating", [4.0, 5.0]),
        ("only 4★+ please", "rating", [4.0, 5.0]),
        ("rated at least 3.5", "rating", [3.5, 5.0]),
        ("under 300 pages", "pages", [0, 300]),
        ("at least 10k ratings", "ratings_count", [10000.0, librarian.RMAX]),
        ("hidden gems", "ratings_count", [30, 5000]),
    ],
)
def test_parse_filters(q, field, value):
    p = P(q)
    assert librarian.recognised(p) and p.f[field] == value


def test_parse_combined_request():
    p = P("only 4★+ science fiction after 2010")
    assert p.f["rating"] == [4.0, 5.0] and p.f["year"][0] == 2011 and p.topic == "science fiction"


@pytest.mark.parametrize(
    "q,attr,value",
    [
        ("books like Dracula", "like", BOOKS["dracula"]),
        ("show me Dracula", "focus", BOOKS["dracula"]),
        ("books by Jules Verne", "author", "Jules Verne"),
        ("show me horror", "topic", "horror"),
        ("zoom out", "zoom", 0.4),
        ("zoom way out", "zoom", 0.25),
        ("pan left", "pan", "left"),
        ("reset filters", "clear", True),
        ("remove the year filter", "remove", {"year"}),
        ("recommend me something", "wizard", True),
        ("what's in view?", "ask", "view"),
    ],
)
def test_parse_actions(q, attr, value):
    assert getattr(P(q), attr) == value


def test_parse_leaves_chat_to_the_llm():
    assert not librarian.recognised(P("hello there"))
    assert not librarian.recognised(P("who wrote the first vampire novel?"))


def test_what_is_questions_are_not_map_commands():
    # "what is X" must fall through to the LLM (which can answer, or use web_lookup),
    # not zoom to a topic that happens to match X
    for q in ("what is horror", "what is science fiction", "explain fae to me", "what is a fae?"):
        p = P(q)
        assert not librarian.recognised(p) and p.topic is None and p.focus is None and p.like is None, q
    # directives with the same word still parse as map commands
    assert P("zoom to horror").topic == "horror"
    assert P("show me science fiction").topic == "science fiction"


def test_fmt_range_and_confirm():
    assert librarian.fmt_range("year", [1990, 2000]) == "years 1990–2000"
    assert librarian.fmt_range("pages", [0, 300]) == "under 300 pages"
    assert librarian.fmt_range("ratings_count", [10000, 1e7]) == "10k+ ratings"
    s = librarian.confirm(["zoomed out"], {"filters": lambda: {"year": [1990, 2000], "region": None}}, 42)
    assert s.startswith("**Map updated:** zoomed out.") and "years 1990–2000" in s and "**42**" in s


# ---------------------------------------------------------------- textprep


def test_book_text_strips_marketing():
    d = (
        "A young woman inherits a lighthouse on a remote island and uncovers her family's secret. "
        "From the New York Times bestselling author of The Silent Shore. "
        '"A gripping, unforgettable page-turner!" -- Jane Doe, author of Night Tides. '
        "Now a major motion picture."
    )
    t = textprep.book_text("The Keeper (Islands, #2)", d, tags=["to-read", "mystery", "owned", "gothic"], author="Ann Lee")
    assert t.startswith("The Keeper (Islands). Tags: mystery, gothic.")
    assert "lighthouse" in t
    for junk in ("bestselling", "gripping", "motion picture", "to-read", "owned", "#2"):
        assert junk not in t, junk


def test_book_text_never_drops_a_short_blurb():
    t = textprep.book_text("X", "An award-winning tale of two cities.")
    assert "two cities" in t


def test_tags_clean_keeps_genres_that_start_like_shelves():
    assert textprep.tags_clean(["my-books", "mystery", "mythology", "own", "owned"], None) == ["mystery", "mythology"]


def test_tags_clean_drops_shelves_and_author_names():
    assert textprep.tags_clean(["currently-reading", "2019", "verne-collection", "adventure", "Adventure"], "Jules Verne") == [
        "adventure"
    ]


# ---------------------------------------------------------------- build.normalise


def test_normalise_flexible_columns_rescale_and_dedupe():
    rows = [
        {
            "Book Title": "Dune",
            "Writer": "Frank Herbert",
            "Blurb": "Spice.",
            "Stars": "9",
            "Votes": "1,200",
            "Genres": "sci-fi; classics",
        },
        {"Book Title": "Dune", "Writer": "Frank Herbert", "Blurb": "dup", "Stars": "8", "Votes": "10", "Genres": ""},
        {"Book Title": "", "Writer": "nobody"},
        {"Book Title": "Emma", "Writer": "Jane Austen", "Blurb": "", "Stars": "0", "Votes": "", "Genres": "romance"},
    ]
    df = build.normalise(rows)
    assert df.height == 2
    dune = df.filter(pl.col("title") == "Dune").row(0, named=True)
    assert dune["rating"] == pytest.approx(4.5) and dune["ratings_count"] == 1200 and dune["tags"] == "sci-fi, classics"
    assert dune["url"].startswith("https://openlibrary.org/search?")
    emma = df.filter(pl.col("title") == "Emma").row(0, named=True)
    assert emma["rating"] is None  # 0 means "no rating"


def test_normalise_mapping_override_and_missing_title():
    df = build.normalise([{"Name": "A", "Notes": "about a boat"}], {"description": "Notes"})
    assert df.row(0, named=True)["description"] == "about a boat"
    with pytest.raises(SystemExit):
        build.normalise([{"foo": "bar"}])


def test_normalise_goodreads_export():
    rows = [
        {"Book Id": "42", "Title": "Emma", "Author": "Jane Austen", "My Rating": "5", "Average Rating": "4.02",
         "Original Publication Year": "1815", "Bookshelves": "to-read, classics", "Exclusive Shelf": "read"}
    ]  # fmt: skip
    assert build.is_goodreads_export(rows)
    r = build.normalise(rows).row(0, named=True)
    assert r["rating"] == pytest.approx(4.02) and r["year"] == 1815 and r["tags"] == "classics"
    assert r["url"] == "https://www.goodreads.com/book/show/42"


# ---------------------------------------------------------------- rec (deterministic wizard)


@pytest.fixture(scope="module")
def sample_books():
    from atlas_books import store

    return store.books()


def test_candidates_prefer_query_words(sample_books):
    from atlas_books import rec

    books, total = rec.candidates(sample_books, pl.lit(True), n=20, query="vampire")
    assert total == sample_books.height and len(books) == 20
    first = books[0]
    assert "vampire" in (first["title"] + first["tags"] + first["desc"]).lower()


def test_wizard_runs_to_three_picks_without_llm(sample_books):
    from atlas_books import rec

    async def run():
        r = await rec.start(sample_books, pl.lit(True), "test view")
        assert "question" in r and r["model"] == "deterministic"
        for _ in range(rec.MAXQ + 1):
            if "final" in r:
                return r
            r = await rec.answer(r["session"], 0)
            assert "error" not in r, r
        return r

    r = asyncio.run(run())
    top = r["final"]["top3"]
    assert len(top) == 3 and len({b["i"] for b in top}) == 3
    assert all(b["year"] is None or 0 < b["year"] < 2100 for b in top)
    assert asyncio.run(rec.answer(r["session"], 0))["error"] == "session already finished"


def test_wizard_refuses_an_empty_view(sample_books):
    from atlas_books import rec

    r = asyncio.run(rec.start(sample_books, pl.col("year") > 3000))
    assert "error" in r


# ---------------------------------------------------------------- mention link disambiguation


def test_mention_links_same_title(monkeypatch):
    """Two books called 'The Giver' (Lynn Hall 1985 vs Lois Lowry 1993): the bolded title must link to the
    book the reply is about — by author when the reply names one, else the most-rated edition."""
    from atlas_books import state

    df = pl.DataFrame(
        {
            "i": [0, 1],
            "title": ["The Giver", "The Giver (The Giver, #1)"],
            "author": ["Lynn Hall", "Lois Lowry"],
            "ratings_count": [34, 1_361_154],
        }
    )
    monkeypatch.setattr(state, "B", df)
    monkeypatch.setattr(state, "LOWER", df["title"].str.to_lowercase())
    monkeypatch.setattr(
        state, "URLS", ["https://www.goodreads.com/book/show/1856057", "https://www.goodreads.com/book/show/3636"]
    )
    assert state.mention_links("**The Giver** by Lois Lowry (1993), 4.12") == {
        "The Giver": "https://www.goodreads.com/book/show/3636"
    }
    assert state.mention_links("**The Giver** by Lynn Hall (1985)") == {
        "The Giver": "https://www.goodreads.com/book/show/1856057"
    }
    assert state.mention_links("have you read **The Giver**?") == {"The Giver": "https://www.goodreads.com/book/show/3636"}
    assert state.mention_links("**The Giver (The Giver, #1)** is great") == {
        "The Giver (The Giver, #1)": "https://www.goodreads.com/book/show/3636"
    }


# ---------------------------------------------------------------- agent content blocks


def test_content_text_normalises_provider_blocks():
    """Some providers (e.g. Mistral) return assistant content as a list of blocks, not a string."""
    from atlas_books import agent

    assert agent._content_text("hi") == "hi"
    assert agent._content_text(None) == ""
    assert (
        agent._content_text(
            [{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}, {"type": "text", "text": "Fae are faeries."}]
        )
        == "Fae are faeries."
    )


def test_chat_tolerates_list_content_from_provider(monkeypatch):
    """POST /api/chat must return a string reply even when the model answers with content blocks."""
    from atlas_books import agent

    async def fake_post(url, headers, body):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": []}, {"type": "text", "text": "Fae are faeries."}],
                    }
                }
            ]
        }

    async def noop_call(name, args):
        return {}

    monkeypatch.setattr(agent, "post", fake_post)
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    r = asyncio.run(
        agent.chat([{"role": "user", "content": "what is Fae"}], [], noop_call, lambda: {"summary": {"books_in_view": 5}})
    )
    assert r["via"] == "llm"
    assert isinstance(r["reply"], str) and "Fae are faeries." in r["reply"]
