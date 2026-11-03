"""Embedding text for a book: title (series kept, "#n" dropped) + genres/tags + the description with marketing fluff
stripped (praise quotes, "New York Times bestselling", "from the author of", award blurbs, edition/packaging notes,
hype adjectives such as "gripping" or "unforgettable"). Library shelves such as to-read / owned / kindle / 2019 are dropped.

    book_text(title, description, tags=["vampires", "boarding-school"], genres=None, author=None)
"""

from __future__ import annotations

import re

PUBS = r"(publishers weekly|kirkus|booklist|library journal|school library journal|new york times|nytimes|usa today|washington post|los angeles times|chicago tribune|boston globe|wall street journal|the guardian|the times|entertainment weekly|people magazine|romantic times|rt book reviews|huffington post|npr|o magazine|oprah|the independent|daily mail|sunday times|times literary|the observer|bookpage|vogue|cosmopolitan|elle|time magazine|newsweek|amazon|goodreads|audible|horn book|voya|the bulletin)"
# whole sentence goes when it is mainly marketing / packaging
DROP = re.compile(
    r"(best[- ]?sell|bestseller|from the (?:author|creator|writer)s? of|by the (?:author|creator)s? of|(?:award|prize)[- ]winning|winner of|finalist for|shortlisted|longlisted|nominated for|nominee|praise for|acclaimed|critically|major motion picture|now a (?:major )?(?:film|movie|tv|netflix|hbo)|soon to be a|includes? (?:a |an )?(?:preview|excerpt|sneak peek|bonus|reading group|discussion|readers? guide)|reading group guide|discussion questions|sneak peek|bonus (?:content|story|chapter|material)|don'?t miss|look for (?:the|book|more)|also (?:available|by)|available (?:now|in|as)|first time in (?:paperback|print|ebook)|(?:paperback|hardcover|e-?book|kindle|audio(?:book)?|anniversary|collector'?s|special|deluxe|movie tie-in|tie-in|illustrated) edition|this edition|new edition|reissue|previously published|originally published|copyright|isbn|all rights reserved|fans of|perfect for (?:fans|readers)|readers who (?:loved|enjoyed)|if you (?:liked|loved|enjoyed)|will (?:love|enjoy) this|million copies|copies sold|translated into|\d+ languages|#\s?1|number one|top ten|instant (?:#|new york)|\b"
    ""
    + PUBS
    + r"\b|\bstars?\b\s*(?:review|rated)|starred review|\bfive[- ]stars?\b|\b5[- ]stars?\b|debut novel from|stunning debut|dazzling debut|brilliant debut|(?:his|her|their) (?:latest|newest) (?:novel|book|thriller|romance|work)|in (?:his|her|their) (?:first|latest|newest|new) (?:novel|book|thriller)|(?:master|queen|king) of (?:horror|suspense|the thriller|thrillers|romance|crime|mystery|fantasy|science fiction|the macabre)|the (?:acclaimed|beloved|bestselling) |readers (?:everywhere|worldwide|around the world)|book \d+ in the|book (?:one|two|three|four|five) (?:in|of) the|(?:first|second|third|fourth|fifth|final|next) (?:book|installment|instalment|volume|novel) in|series (?:starter|continues)|stand-?alone|can be read (?:as|on)|novella|word count|approx(?:imately)?\.? \d+|\d+,?\d* words|contains? (?:mature|adult|explicit)|content warning|trigger warning|\b18\+|recommended for (?:ages|readers)|intended for mature)",
    re.I,
)
QUOTE = re.compile(
    r"[\"“][^\"”]{10,400}[\"”]\s*(?:[-—–~]+\s*(?:[A-Z][\w.'’]+(?:\s+[A-Z][\w.'’&]+){0,1})(?:,\s*(?:author|bestselling|editor|reviewer|host)[^.\"“]{0,80})?|(?=\s*[-—–~,]?\s*(?:[Tt]he\s+)?(?i:"
    + PUBS
    + r")\b))"
)
FLUFF = re.compile(
    r"\b(gripping|unforgettable|riveting|compelling|captivating|breathtaking|heart[- ]?(?:pounding|stopping|wrenching|warming|breaking)|page[- ]turn(?:er|ing)|unputdownable|spellbinding|mesmerizing|mesmerising|stunning|dazzling|masterful(?:ly)?|masterpiece|tour de force|must[- ]read|edge[- ]of[- ](?:your|the)[- ]seat|nail[- ]biting|pulse[- ]pounding|electrifying|unputdownable|thrilling|addictive|irresistible|luminous|lyrical|exquisite(?:ly)?|brilliant(?:ly)?|powerful|poignant|haunting(?:ly)?|beautifully written|wonderfully|deliciously|utterly|truly|extraordinary|remarkable|unforgettably|sweeping|epic(?= novel)|highly anticipated|long[- ]awaited|eagerly awaited|beloved|acclaimed|bestselling|award[- ]winning|critically[- ]acclaimed|internationally|nationally)\b,?\s*",
    re.I,
)
JUNK = re.compile(
    r"to-?read|currently|favou?rite|fave|own|kindle|nook|library|wish|default|e-?books?|^books?|^read|tbr|audio|finished|abandon|dnf|did-not|^my-|^fiction$|^novels?$|^novel$|general|shelfari|^\d{4}|-\d{4}$|20\d\d|19\d\d|challenge|giveaway|netgalley|arc$|^arcs?|first-?reads|review|bought|purchased|borrowed|kobo|scribd|amazon|freebie|free|^series$|^re-?read|want|maybe|someday|^fic$|tbr|on-hold|unread|^paused|^p$|^ya-?lit$|^book-?club|club|^stand-?alone|^paperback|^hardcover|^signed|^physical|cover|^not-|^i-|^have|^\w$|recommended|recommendation|^5-?stars?|stars|^loved|^liked|^meh|^dropped|^gave-up|default|^adult$|^english|^usa$|^uk$|bookshelf|^shelf|^to-buy|sale|bargain|^z-|^x-|^a-|^my$|^fiction-|shelf|audible|guilty|must-?read|best-?books|ever$|faves?$|^loved?$|keeper|^y-a$|^e-|^on-|^in-|^ku$|^owned",
    re.I,
)
CLAUSE = re.compile(r"[^,.;:()]*?(?:" + DROP.pattern + r")[^,.;:()]*[,;:]?\s*", re.I)


def title_clean(t):
    t = re.sub(r"\(([^()#]*?),?\s*#\s*[\d.\-–]+\)", r"(\1)", t or "")
    t = re.sub(r"\s*#\s*[\d.\-–]+", "", t)
    return re.sub(r"\(\s*\)", "", t).strip()


def desc_clean(d):
    d = re.sub(r"\s+", " ", d or "").strip()
    d0 = d
    d = QUOTE.sub(" ", d)
    sents = re.split(r"(?<=[.!?…])\s+(?=[\"“A-Z0-9#*])", d)
    keep = []
    nd = 0
    for s in sents:
        if not DROP.search(s):
            keep.append(s)
            continue
        nd += 1
        if len(s.split()) > 25:  # long sentence: cut only the comma-delimited clause(s) with the marketing phrase
            s2 = CLAUSE.sub(" ", s).strip()
            keep.append(s2 if len(s2.split()) >= 8 and not DROP.search(s2) else s)
    d = " ".join(keep)
    d = FLUFF.sub("", d)
    d = re.sub(r"\s+([,.;:!?])", r"\1", d)
    d = re.sub(r"\s{2,}", " ", d).strip(" ,;-—")
    if len(d) < 40:
        d = re.sub(r"\s{2,}", " ", FLUFF.sub("", d0)).strip()  # never leave a book without its blurb
    return d, nd


def tags_clean(shelves, author):
    an = {w for w in re.split(r"[^a-z]+", (author or "").lower()) if len(w) > 2}
    out = []
    for s in shelves or []:
        s = s.lower().strip()
        if JUNK.search(s) or any(w in s.split("-") for w in an):
            continue
        s = s.replace("-", " ")
        if s not in out:
            out.append(s)
        if len(out) >= 10:
            break
    return out


def book_text(title: str, description: str | None, tags=None, genres=None, author: str | None = None, max_tags: int = 10) -> str:
    """Cleaned text that is embedded for a book (bge models truncate at 512 tokens, so the short head comes first)."""
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,;|]", tags) if t.strip()]
    tg = tags_clean([t.replace(" ", "-") for t in (tags or [])], author)[:max_tags]
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]
    d, _ = desc_clean(description or "")
    head = (
        title_clean(title or "")
        + "."
        + (f" Genres: {', '.join(genres)}." if genres else "")
        + (f" Tags: {', '.join(tg)}." if tg else "")
    )
    return (head + " " + d).strip()
