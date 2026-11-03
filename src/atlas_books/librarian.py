"""Deterministic layer of the Librarian: parses common map requests into validated tool calls, runs them,
and writes a confirmation of what changed on the map. Anything it cannot parse goes to the LLM (agent.py)."""

import re

YMIN, YMAX = 1980, 2025
RMAX = 1e7
STOP = set(
    """a an the me my i i'm im we us you your please only just some any all books book novels novel stories story stuff things something
 anything ones one with that which who are is be to of for in on at and or but also show give find get want would like could can let lets let's
 see look looking from by more really very maybe now then here there this these those them it its around about map view zoom some kind sort""".split()
)


def _n(s, suf=None):
    v = float(s.replace(",", ""))
    return v * (1e3 if suf == "k" else 1e6 if suf == "m" else 1)


def _y(s):
    v = int(s)
    return v if 1000 < v < 3000 else None


class P:
    def __init__(s, text):
        s.raw = text.strip()
        s.t = " " + re.sub(r"\s+", " ", s.raw.lower().replace("’", "'").replace("–", "-").replace("★", " stars")) + " "
        s.f = {}
        s.remove = set()
        s.clear = False
        s.view = None
        s.zoom = None
        s.pan = None
        s.focus = None
        s.like = None
        s.author = None
        s.topic = None
        s.ask = None
        s.wizard = False
        s.notes = []

    def eat(s, pat, fn):
        hit = [False]

        def rep(m):
            r = fn(m)
            if r is False:
                return m.group(0)
            hit[0] = True
            return " "

        s.t = re.sub(pat, rep, s.t)
        return hit[0]

    def left(s):
        return [w for w in re.findall(r"[a-z0-9'+-]+", s.t) if w not in STOP]


def parse(text, H):
    p = P(text)
    t = p.t
    question = bool(
        re.match(r"\s*(is|are|does|do|did|who|whom|why|how|when|can you tell|tell me about|what's the difference)\b", t)
    )
    # ---- questions about the view ----
    if p.eat(
        r"\bwhat(?:'s| is| are)? (?:the )?(?:best|top|highest)[- ]?(?:rated|rating|ranked)?(?: books?| ones?| reads?)?(?: (?:in view|here|on (?:the )?(?:screen|map)|visible|in this (?:area|region|view|part)))?\b|\b(?:which|what) (?:books? )?(?:are|is) (?:the )?(?:best|top)\b|\b(?:list|show)(?: me)? the (?:best|top)(?:[- ]rated)?(?: books?)?(?: (?:in view|here))?\b",
        lambda m: True,
    ):
        p.ask = "top"
    if p.eat(
        r"\b(?:which|what) (?:book|one|title|novel)s?(?: here| in view| on (?:the )?(?:map|screen)| in this (?:area|view))? (?:has|have|is|are|got) (?:the )?(?:best|highest|top)(?:[- ]rated| rating| ratings| reviews| score)?\b|\b(?:best|highest)[- ]rated (?:book|one|title)s? (?:here|in view|on (?:the )?(?:map|screen))\b",
        lambda m: True,
    ):
        p.ask = "top"
    if p.eat(
        r"\bwhat(?:'s| is| are)? (?:in|on) (?:the |my )?(?:view|screen|map)\b|\bwhat(?:'s| is) (?:here|this area|this region|in view)\b|\bwhat am i (?:looking at|seeing)\b|\bwhat do i see\b|\b(?:describe|summari[sz]e) (?:the |this |my )?(?:view|map|area|region|screen)\b|\bwhere am i\b|\bwhat(?:'s| is) (?:on )?(?:screen|the map)\b",
        lambda m: True,
    ):
        p.ask = "view"
    if p.eat(
        r"\b(?:recommend(?:ation)?s?|suggest(?:ion)?s?|what should i read|find me (?:a|something|some)(?: good)?(?: book| read)?|help me (?:find|choose|pick)(?: a book)?|pick (?:a book |something )?for me|something to read)\b",
        lambda m: True,
    ):
        p.wizard = True
    # ---- resets ----
    p.eat(
        r"\b(?:remove|clear|drop|reset|delete|turn off|get rid of|undo|no|without|ignore|forget)(?: the| my| any| that)? (ratings?[- ]count|number of ratings|popularity|votes|page(?:s| count)?|length|year|date|publication(?: year| date)?|rating|stars?)(?: filter| limit| slider| constraint| range| restriction)?s?\b",
        lambda m: p.remove.add(
            {
                "page": "pages",
                "pages": "pages",
                "page count": "pages",
                "length": "pages",
                "year": "year",
                "date": "year",
                "rating": "rating",
                "star": "rating",
                "stars": "rating",
                "popularity": "ratings_count",
                "votes": "ratings_count",
                "number of ratings": "ratings_count",
            }.get(
                m.group(1).split(" ")[0] if m.group(1).startswith("publication") else m.group(1),
                "ratings_count" if "count" in m.group(1) else "year" if m.group(1).startswith("publication") else "rating",
            )
        ),
    )
    if p.eat(
        r"\b(?:reset|clear|remove|drop|delete|turn off|get rid of|undo)(?: all| all the| the| my| any)? (?:filters?|sliders?|constraints?|restrictions?)\b|\bno (?:more )?filters\b|\bclear all\b|\bunfilter\b",
        lambda m: True,
    ):
        p.clear = True
    if p.eat(
        r"\breset(?: everything| all| it all)?\s*[.!]?\s*$|\bstart (?:over|again|fresh)\b|\breset everything\b|\bfrom scratch\b|\bshow (?:me )?everything\b|\bback to (?:the )?(?:start|beginning)\b",
        lambda m: True,
    ):
        p.clear = True
        p.view = "reset"
    if p.eat(
        r"\b(?:reset|recenter|re-center)(?: the)? (?:map|view|zoom)\b|\bzoom (?:all the way|fully|completely) out\b|\b(?:show|see)(?: me)? (?:the )?(?:whole|entire|full) map\b|\bwhole map\b",
        lambda m: True,
    ):
        p.view = "reset"
    # ---- zoom / pan ----
    if p.eat(
        r"\bzoom(?: (?:way|a bit|a little|bit|further|more|right|back))* out\b|\bpull back\b|\bwider view\b|\bzoom back\b",
        lambda m: True,
    ):
        p.zoom = 0.25 if "way" in t else 0.4
    p.eat(
        r"\b(?:pan|move|go|scroll|shift|look)(?: the map| the view)?(?: (?:a bit|a little|further|more))?(?: to(?: the)?)? (left|right|up|down|north|south|east|west)\b",
        lambda m: setattr(
            p, "pan", {"north": "up", "south": "down", "east": "right", "west": "left"}.get(m.group(1), m.group(1))
        ),
    )
    if p.eat(
        r"\bzoom(?: (?:way|a bit|a little|bit|further|more|right))* in\b(?! (?:to|on)\b)|\b(?:get )?closer\b|\benlarge\b",
        lambda m: True,
    ):
        p.zoom = 4 if "way" in t else 2.5

    # ---- year ----
    def yr(lo, hi):
        p.f["year"] = [max(YMIN, lo), min(YMAX, hi)]

    p.eat(
        r"\b(?:between|from) (\d{4}) ?(?:and|to|-|until|till) ?(\d{4})\b",
        lambda m: yr(int(m.group(1)), int(m.group(2))) if _y(m.group(1)) and _y(m.group(2)) else False,
    )
    p.eat(
        r"\b(\d{4}) ?(?:-|to) ?(\d{4})\b",
        lambda m: yr(int(m.group(1)), int(m.group(2))) if _y(m.group(1)) and _y(m.group(2)) else False,
    )

    def dec(m):
        if m.group(1):
            d = int(m.group(1) + m.group(2) + "0")
        else:
            d = int(m.group(3))
            d = 1900 + d if d >= 30 else 2000 + d
        yr(d, d + 9)

    p.eat(r"\b(?:the )?(?:(19|20)(\d)0|'?(\d0))'?s\b", dec)
    p.eat(
        r"\b(after|since|post|newer than|later than|from|published after|published since|released after) (\d{4})(?: ?(?:on|onwards?|and (?:later|after|newer|beyond)|or (?:later|newer)|\+))?\b",
        lambda m: (
            yr(
                int(m.group(2))
                + (1 if m.group(1) in ("after", "post", "newer than", "later than", "published after", "released after") else 0),
                YMAX,
            )
            if _y(m.group(2))
            else False
        ),
    )
    p.eat(
        r"\b(\d{4}) ?(?:onwards?|and (?:later|newer|after)|or (?:later|newer)|\+)",
        lambda m: yr(int(m.group(1)), YMAX) if _y(m.group(1)) else False,
    )
    p.eat(
        r"\b(before|pre|prior to|older than|until|up to|earlier than|published before|no later than) (\d{4})\b",
        lambda m: (
            yr(
                YMIN,
                int(m.group(2))
                - (1 if m.group(1) in ("before", "pre", "prior to", "older than", "earlier than", "published before") else 0),
            )
            if _y(m.group(2))
            else False
        ),
    )
    p.eat(
        r"\b(?:in|published in|released in|written in|from) (\d{4})\b",
        lambda m: yr(int(m.group(1)), int(m.group(1))) if _y(m.group(1)) else False,
    )
    p.eat(r"\b(?:recent|newer|latest|new releases|recently published)\b", lambda m: yr(2013, YMAX))
    p.eat(r"\bolder (?:books?|ones|titles|novels)\b", lambda m: yr(YMIN, 1999))

    # ---- rating ----
    def rt(lo, hi=5.0, vague=False):
        if not (1 <= lo <= 5 and 1 <= hi <= 5):
            return False
        if vague and "rating" in p.f:
            return
        p.f["rating"] = sorted([lo, hi])

    p.eat(r"\b(\d(?:\.\d+)?) ?(?:-|to) ?(\d(?:\.\d+)?) stars?\b", lambda m: rt(float(m.group(1)), float(m.group(2))))
    p.eat(
        r"\b(?:rated|ratings?|rating of|scores?|stars?|average)(?: of)? (?:at least|above|over|higher than|better than|more than|>=?|minimum(?: of)?|min\.?) ?(\d(?:\.\d+)?)(?! ?(?:k|m|,\d|\d))(?: stars?)?",
        lambda m: rt(float(m.group(1))),
    )
    p.eat(r"\b(?:at least|above|over|more than|minimum(?: of)?) (\d(?:\.\d+)?) stars?\b", lambda m: rt(float(m.group(1))))
    p.eat(
        r"\b(\d(?:\.\d+)?) ?(?:\+|plus)? ?stars?(?: (?:or|and) (?:more|up|above|higher|better|over))?\b",
        lambda m: rt(float(m.group(1))),
    )
    p.eat(
        r"\b(?:rated|rating(?: of)?) (\d(?:\.\d+)?) ?(?:\+|or (?:more|higher|better|above)|and (?:up|above|higher))",
        lambda m: rt(float(m.group(1))),
    )
    p.eat(
        r"\b(?:rated|rating|ratings?)(?: of)? (?:below|under|less than|lower than|<) ?(\d(?:\.\d+)?)\b(?! ?(?:k|m|,\d|\d))",
        lambda m: rt(1.0, float(m.group(1))),
    )
    p.eat(
        r"\b(?:very highly|extremely well|exceptionally well|super highly)[- ]rated\b|\b(?:top|best)[- ]rated\b|\bmasterpieces?\b",
        lambda m: rt(4.3, vague=True),
    )
    p.eat(
        r"\b(?:highly|well|high|great|good|nicely)[- ]?(?:rated|reviewed)\b|\b(?:good|great|strong) (?:reviews|ratings)\b|\bacclaimed\b|\bwell[- ]liked\b",
        lambda m: rt(4.0, vague=True),
    )
    p.eat(r"\b(?:low|poorly|badly|lowest)[- ]rated\b|\bbad (?:reviews|ratings)\b", lambda m: rt(1.0, 3.5, vague=True))

    # ---- pages ----
    def pg(lo, hi, vague=False):
        if vague and "pages" in p.f:
            return
        p.f["pages"] = [lo, hi]

    p.eat(r"\b(\d{2,4}) ?(?:-|to) ?(\d{2,4}) pages?\b", lambda m: pg(int(m.group(1)), int(m.group(2))))
    p.eat(
        r"\b(?:under|less than|fewer than|below|max(?:imum)?(?: of)?|at most|no more than|shorter than|up to|<) ?(\d{2,4}) ?(?:pages?|pp)\b",
        lambda m: pg(0, int(m.group(1))),
    )
    p.eat(
        r"\b(?:over|more than|at least|above|longer than|min(?:imum)?(?: of)?|>) ?(\d{2,4}) ?(?:pages?|pp)\b",
        lambda m: pg(int(m.group(1)), 100000),
    )
    p.eat(
        r"\b(?:short|quick|brief|small|fast)(?: (?:and|&) [a-z-]+)?(?= (?:reads?|books?|novels?|ones?)\b)|\b(?:something|anything|only|just|make (?:it|them)|keep (?:it|them)|more|but|nice and) short(?:er)?\b(?! (?:sentence|answer|reply|summary|version|description|stor))|^\s*short(?:er)?(?: (?:please|pls))?\s*[.!?]?\s*$|\bnovellas?\b|\bquick reads?\b",
        lambda m: pg(0, 250, True),
    )
    p.eat(
        r"\b(?:long|big|chunky|thick|lengthy|huge|massive)(?= (?:reads?|books?|novels?|ones?|series)\b)|\bdoorstoppers?\b|\bepic[- ]length\b",
        lambda m: pg(500, 100000, True),
    )

    # ---- number of ratings ----
    def rc(lo, hi):
        p.f["ratings_count"] = [lo, hi]

    p.eat(
        r"\b(?:at least|over|more than|above|minimum(?: of)?|min\.?|>) ?([\d][\d.,]*) ?([km])?\+? (?:ratings|reviews|votes|readers)\b",
        lambda m: rc(_n(m.group(1), m.group(2)), RMAX),
    )
    p.eat(r"\b([\d][\d.,]*) ?([km])?\+ (?:ratings|reviews|votes|readers)\b", lambda m: rc(_n(m.group(1), m.group(2)), RMAX))
    p.eat(
        r"\b(?:under|fewer than|less than|below|at most|<) ?([\d][\d.,]*) ?([km])? (?:ratings|reviews|votes|readers)\b",
        lambda m: rc(0, _n(m.group(1), m.group(2))),
    )
    p.eat(
        r"\b(?:very popular|bestsellers?|best[- ]sellers?|blockbusters?|mega[- ]popular|hugely popular)\b",
        lambda m: rc(100000, RMAX),
    )
    p.eat(r"\b(?:popular|well[- ]known|famous|mainstream)\b", lambda m: rc(10000, RMAX))

    def gem(m):
        rc(30, 5000)
        if "gem" in m.group(0) or "underrated" in m.group(0):
            p.f.setdefault("rating", [4.0, 5.0])

    p.eat(r"\b(?:hidden gems?|gems|lesser[- ]known|obscure|underrated|under the radar|undiscovered|niche|little[- ]known)\b", gem)

    # ---- books like X / show X / books by X ----
    def ent(x):
        x = re.sub(
            r"\b(?:on|in) (?:the|this) (?:map|view|atlas)\b|\bplease\b|\bfor me\b|\blocated\b|\bon the map\b|[?!.\"“”]", " ", x
        )
        x = re.split(r"\b(?:but|and also|and then|so that|then|with|that are|which are|only)\b|,|;", x)[0]
        return re.sub(r"\s+", " ", x).strip(" '")

    def like(m):
        x = ent(m.group(1))
        b = H["title"](x)
        if b:
            p.like = b
            return True
        a = H["author"](x)
        if a:
            p.author = a
            return True
        return False

    if not re.search(r"\b(?:i'd|i would|would you|do you|i) like\b", p.t):
        p.eat(
            r"\b(?:books?|novels?|something|stuff|anything|more|ones?|titles?|reads?|similar)?\s*(?:similar to|like|in the vein of|in the style of|along the lines of|reminiscent of|comparable to)\s+(.+?)\s*$",
            like,
        )
    p.eat(r"^\s*(.+?)[- ]like (?:books?|novels?|stories|reads?)\b", like)

    def by(m):
        w = ent(m.group(1)).split()
        for k in range(min(4, len(w)), 0, -1):
            a = H["author"](" ".join(w[-k:]))
            if a:
                p.author = a
                return True
        return False

    p.eat(
        r"\b(?:books?|novels?|works?|stuff|anything|everything|something)? ?(?:by|written by|from author|from the author) ([a-z][a-z .'-]+?)\s*$",
        by,
    )
    p.eat(r"\b([a-z][a-z .-]+?)'s (?:books|novels|work|works)\b", by)

    def show(m):
        x = ent(m.group(2))
        xl = re.sub(r"^(?:me |us |the book |the novel |book |novel |where )", "", x).strip()
        xa = re.sub(
            r"^(?:all |the |some )+", "", re.sub(r"(?:'s)? (?:books?|novels?|works?|stuff|stories|titles)$", "", xl)
        ).strip()
        words = [w for w in re.findall(r"[a-z0-9'+-]+", xl) if w not in STOP]
        if not words:
            return True
        tp = H["topic"](xl)
        if tp and len(" ".join(words)) <= len(tp[1]) + 12:
            p.topic = tp[0]
            return True
        b = H["title"](xl)
        if b:
            p.focus = b
            return True
        a = H["author"](xl) or (H["author"](xa) if xa != xl else None)
        if a:
            p.author = a
            return True
        if tp:
            p.topic = tp[0]
            return True
        b = H["title"](xl) if len(xl) >= 4 else None
        return False

    p.eat(
        r"\b(show me|show|find|zoom (?:in )?(?:to|on)|take me to|go to|fly to|jump to|navigate to|where(?:'s| is| are| can i find)|locate|focus on|cent(?:er|re) on|open|bring up)\s+(.+?)\s*$",
        show,
    )
    # ---- remaining topic words (only for directives, not questions) ----
    if not (p.topic or p.like or p.focus or p.author):
        directive = (
            re.search(
                r"\b(only|just|show|find|zoom|filter|give me|i want|i'm looking|looking for|i need|go to|take me|some)\b",
                " " + p.raw.lower() + " ",
            )
            or len(p.left()) <= 3
        )
        if directive and not question:
            tp = H["topic"](p.t)
            if tp:
                p.topic = tp[0]
                p.t = p.t.replace(tp[1], " ")
    if not recognised(p) and len(p.left()) <= 4 and not question:
        x = ent(" ".join(p.left()))
        if x and (b := H["title"](x)):
            p.focus = b
    return p


def recognised(p):
    return bool(
        p.f or p.remove or p.clear or p.view or p.zoom or p.pan or p.focus or p.like or p.author or p.topic or p.ask or p.wizard
    )


def fmt_range(k, v):
    if k == "year":
        return f"years {int(v[0])}–{int(v[1])}" if v[0] != v[1] else f"year {int(v[0])}"
    if k == "rating":
        return f"rating ★{v[0]:g}–{v[1]:g}"
    if k == "pages":
        return (
            f"under {int(v[1])} pages"
            if v[0] <= 0
            else (f"{int(v[0])}+ pages" if v[1] >= 1e4 else f"{int(v[0])}–{int(v[1])} pages")
        )
    if k == "ratings_count":
        f = lambda n: f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}k" if n >= 1e3 else f"{int(n)}"
        return f"{f(v[0])}+ ratings" if v[1] >= 1e6 else f"{f(v[0])}–{f(v[1])} ratings"
    return k


def bk(b, short=False):
    s = f"*{b['title']}*" + ("" if short else f" ({b.get('author') or '?'}, {b.get('year') or '?'})")
    return (
        s
        + (f" ★{b['rating']:.2f}" if b.get("rating") else "")
        + (f" · {b['ratings_count']:,} ratings" if b.get("ratings_count") and not short else "")
    )


def confirm(changes, H, n=None):
    """One line saying what changed on the map, from the executed (validated) actions."""
    if not changes:
        return ""
    f = H["filters"]()
    active = [fmt_range(k, v) for k, v in f.items() if v and k != "region"]
    s = "**Map updated:** " + "; ".join(changes) + "."
    s += " Active filters: " + (", ".join(active) if active else "none") + "."
    if n is not None:
        s += f" **{n:,}** books in view."
    return s


async def run(p, T, H):
    """Execute a parsed request. T(name,**args) calls a tool (validated server-side). Returns (reply, wizard_query|None)."""
    ch = []
    extra = []
    n = None
    if p.clear:
        r = await T("set_filters", clear=True)
        ch.append("cleared all filters")
        n = r["summary"]["books_in_view"]
    if p.remove and not p.clear:
        r = await T("set_filters", **{k: [] for k in p.remove})
        ch.append(
            "removed the "
            + ", ".join(sorted(k.replace("_", " ") for k in p.remove))
            + " filter"
            + ("s" if len(p.remove) > 1 else "")
        )
        n = r["summary"]["books_in_view"]
    if p.f:
        r = await T("set_filters", **p.f)
        ch.append("set " + ", ".join(fmt_range(k, v) for k, v in p.f.items()) + " (other filters kept)")
        n = r["summary"]["books_in_view"]
    if p.view == "reset":
        r = await T("set_view", reset=True)
        ch.append("reset the map to the whole atlas")
        n = r["summary"]["books_in_view"]
    if p.like:
        r = await T("focus_book", i=p.like["i"])
        b = r["focus"]
        n = r["summary"]["books_in_view"]
        ch.append(
            f"zoomed to {bk(b, True)} and its neighbourhood (books with similar blurbs sit nearby)"
            + (" and cleared filters that hid it" if r.get("cleared_filters") else "")
        )
        s = await T("similar_books", i=p.like["i"], k=6)
        extra.append(
            "Closest by description: " + "; ".join(bk(x, True) + f" ({x['author']}, {x['year']})" for x in s["similar"][:5]) + "."
        )
    elif p.focus:
        r = await T("focus_book", i=p.focus["i"])
        b = r["focus"]
        n = r["summary"]["books_in_view"]
        ch.append(f"zoomed to {bk(b)}" + (" and cleared filters that hid it" if r.get("cleared_filters") else ""))
        extra.append("It is highlighted on the map; nearby dots have similar descriptions.")
    elif p.author:
        r = await T("zoom_to_author", author=p.author)
        n = r["summary"]["books_in_view"]
        ch.append(
            f"zoomed to where {r['author']}'s books sit ({round(100 * r['share_in_window'])}% of their books are in this window)"
        )
        extra.append("Their most-rated: " + "; ".join(bk(x, True) + f" ({x['year']})" for x in r["books"][:4]) + ".")
    elif p.topic:
        r = await T("zoom_to_topic", topic=p.topic)
        n = r["summary"]["books_in_view"]
        ch.append(
            f"zoomed to the {r['topic']} area of the map ({r['lift']:g}× the usual share of {r['topic']} books; genre itself is not a filter)"
        )
    if p.zoom and not (p.like or p.focus or p.author or p.topic or p.view):
        r = await T("zoom", factor=p.zoom, direction=p.pan)
        n = r["summary"]["books_in_view"]
        ch.append(("zoomed in" if p.zoom > 1 else "zoomed out") + (f" and moved {p.pan}" if p.pan else ""))
    elif p.pan and not (p.like or p.focus or p.author or p.topic or p.view):
        r = await T("zoom", factor=1, direction=p.pan)
        n = r["summary"]["books_in_view"]
        ch.append(f"moved the map {p.pan}")
    out = []
    c = confirm(ch, H, n)
    if c:
        out.append(c)
    if p.notes:
        out += p.notes
    out += extra
    if p.ask == "view" or (p.ask is None and not ch and not p.wizard):
        v = (await T("get_view_state"))["summary"]
        f = [fmt_range(k, x) for k, x in (v.get("filters") or {}).items() if x and k != "region"]
        fo = H["focus"]()
        x0, x1, y0, y1 = H["cur_vp"]()
        X0, X1, _, _ = H["full"]
        zoomed = (x1 - x0) < (X1 - X0) * 0.95
        out.append(
            f"You're looking at {'a zoomed-in part of the map' + (' around ' + bk(fo, True) if fo else '') if zoomed else 'the whole map'}"
            f" with **{v['books_in_view']:,}** books; filters: {', '.join(f) if f else 'none'}."
            + (f" Typical year {int(v['years']['median'])}, average rating ★{v['avg_rating']}." if v.get("years") else "")
        )
        top = H["top"](5)
        out.append("Top rated here: " + "; ".join(bk(b, True) + f" ({b['author']})" for b in top[:5]) + ".")
        out.append("Most rated here: " + "; ".join(f"*{b['title']}*" for b in v["top_books"][:3]) + ".")
    elif p.ask == "top":
        top = H["top"](5)
        out.append(
            "Top rated in view (rating weighted by number of ratings):\n"
            + "\n".join(f"{k + 1}. {bk(b)}" for k, b in enumerate(top))
            if top
            else "No books in view. Zoom out or clear a filter."
        )
    elif ch and not p.wizard:
        top = H["top"](3)
        if top:
            out.append("Top rated now in view: " + "; ".join(bk(b, True) + f" ({b['author']})" for b in top) + ".")
    wq = None
    if p.wizard:
        wq = p.raw
        out.append("Let's find you a book from what's in view: I'll ask a couple of quick questions.")
    return "\n\n".join(out), wq
