"""`atlas import`: turn a book list into the files the app reads.

Input  : Goodreads library export CSV, any CSV/TSV (flexible column names), JSON / JSON Lines, Parquet, or a Hugging Face dataset.
Output : <dataset dir>/
           books.parquet   i, title, author, year, x, y, rating, ratings_count, pages, genre, cluster, cluster_label, url
           extra.parquet   i, tags (", "-joined slugs), pages, desc (first 360 chars)
           vecs_i8.npy     bge-small-en-v1.5 embeddings of title + tags + cleaned description (textprep.py), int8 ...
           vecs_scale.npy  ... with a per-dimension float scale (vec = i8 * scale)
           meta.json       name, description, counts, source, attribution
Works from ~50 to ~500k books (UMAP is fitted on a 150k sample above that size and the rest is projected).
"""

from __future__ import annotations
import csv
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
import numpy as np
import polars as pl

from .textprep import book_text

TEXT_RECIPE = "title + tags + description without marketing fluff (textprep v1)"

FIELDS = {
    "title": ["title", "name", "book", "book_title", "booktitle"],
    "author": ["author", "authors", "author_name", "writer", "creator", "author_l-f"],
    "year": [
        "year",
        "original_publication_year",
        "original publication year",
        "year_published",
        "year published",
        "publication_year",
        "publication year",
        "first_published",
        "published",
        "pub_year",
        "date",
        "issued",
    ],
    "description": ["description", "desc", "blurb", "summary", "synopsis", "overview", "abstract", "about"],
    "rating": ["average_rating", "average rating", "avg_rating", "rating", "stars", "score", "my rating", "my_rating"],
    "ratings_count": [
        "ratings_count",
        "ratings count",
        "num_ratings",
        "number_of_ratings",
        "work_ratings_count",
        "ratings",
        "votes",
        "popularity",
        "downloads",
        "download_count",
        "count",
        "reviews",
    ],
    "pages": ["pages", "num_pages", "number of pages", "number_of_pages", "page_count", "pagecount", "length"],
    "url": ["url", "link", "href", "goodreads_url", "web"],
    "tags": ["tags", "genres", "genre", "subjects", "subject", "shelves", "bookshelves", "categories", "category", "keywords"],
    "isbn": ["isbn13", "isbn", "isbn_13", "isbn10"],
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- reading
def _read_any(path: str) -> list[dict]:
    p = path.lower()
    if p.endswith((".parquet", ".pq")):
        return pl.read_parquet(path).to_dicts()
    if p.endswith((".jsonl", ".ndjson")):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if p.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        if isinstance(d, dict):  # {"books":[...]} or {"data":[...]}
            d = next((v for v in d.values() if isinstance(v, list)), [d])
        return d
    raw = open(path, encoding="utf-8-sig", errors="replace").read()
    head = raw.split("\n", 1)[0]
    if p.endswith(".tsv") or head.count("\t") > max(head.count(","), head.count(";")):
        delim = "\t"
    elif head.count(";") > head.count(","):
        delim = ";"
    else:
        delim = ","
    rows = list(csv.DictReader(io.StringIO(raw), delimiter=delim))
    return [{k: v for k, v in r.items() if k is not None} for r in rows]


def read_hf(repo: str, file: str | None = None) -> list[dict]:
    """Rows from a Hugging Face *dataset* repo (public, or private with $HF_TOKEN). Reads `file`, or every
    parquet/csv/jsonl file of the 'train' split (falling back to all files of the most common format)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit(
            "`atlas import --hf` needs huggingface_hub: pipx inject atlas-books huggingface_hub  (or pip install 'atlas-books[hf]')"
        )
    tok = os.environ.get("HF_TOKEN") or None
    if file:
        files = [file]
    else:
        allf = [
            f
            for f in HfApi().list_repo_files(repo, repo_type="dataset", token=tok)
            if f.lower().endswith((".parquet", ".csv", ".tsv", ".jsonl", ".json")) and not f.startswith(".")
        ]
        if not allf:
            raise SystemExit(f"no parquet/csv/json files in dataset {repo}")
        ext = max({f.rsplit(".", 1)[1] for f in allf}, key=lambda e: (e == "parquet", sum(f.endswith(e) for f in allf)))
        files = [f for f in allf if f.endswith(ext)]
        files = [f for f in files if "train" in f] or files
    rows: list[dict] = []
    for f in sorted(files):
        rows += _read_any(hf_hub_download(repo, f, repo_type="dataset", token=tok))
        log(f"read {f}: {len(rows):,} rows so far")
    return rows


def _num(v, kind=float):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return kind(v)
    s = str(v).strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return kind(float(m.group(0)))
    except ValueError:
        return None


def _slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def _tags(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        parts = [str(x) for x in v]
    else:
        parts = re.split(r"\s*[;,|]\s*|\s+--\s+", str(v))
    out = []
    for x in parts:
        x = re.sub(r"\s*\(.*?\)$", "", x.replace("Category: ", "")).strip()
        if x and x.lower() not in ("to-read", "currently-reading", "read", "favorites", "owned"):
            out.append(_slug(x))
    return list(dict.fromkeys(t for t in out if t))[:25]


def is_goodreads_export(rows: list[dict]) -> bool:
    return bool(rows) and {"Book Id", "Title", "Exclusive Shelf"} <= set(rows[0])


def normalise(rows: list[dict], mapping: dict[str, str] | None = None) -> pl.DataFrame:
    """rows (any column names) -> one row per book with the canonical fields."""
    if not rows:
        raise SystemExit("the input has no rows")
    cols = list(rows[0].keys())
    nz = lambda c: re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
    low = {c.lower().strip(): c for c in cols}
    low |= {nz(c): c for c in cols}
    pick = {}
    for f, syn in FIELDS.items():  # 1) exact (case/punctuation-insensitive) names
        if mapping and f in mapping:
            pick[f] = mapping[f]
            continue
        for s in syn:
            if s in low or nz(s) in low:
                pick[f] = low.get(s) or low[nz(s)]
                break
    used = set(pick.values())
    for f, syn in FIELDS.items():  # 2) a synonym as a whole word inside the column name ("Book Title", "Blurb text")
        if f in pick:
            continue
        for c in cols:
            if c in used:
                continue
            toks = set(nz(c).split("_"))
            if any(
                s in toks
                for s in syn
                if " " not in s and "_" not in s and s not in ("book", "name", "date", "count", "length", "web")
            ):
                pick[f] = c
                used.add(c)
                break
    gr = is_goodreads_export(rows)
    if gr:  # Goodreads export: prefer the community average over the user's own stars
        pick.update({"rating": "Average Rating", "year": low.get("original publication year") or low.get("year published")})
        pick["tags"] = "Bookshelves"
    if "title" not in pick:
        raise SystemExit(f"could not find a title column in {cols[:15]}; use --map title=<column>")
    log("column mapping: " + ", ".join(f"{k}<-{v!r}" for k, v in pick.items() if v))
    out = []
    for r in rows:
        g = lambda f: r.get(pick[f]) if pick.get(f) else None
        title = str(g("title") or "").strip()
        if not title:
            continue
        author = g("author")
        if isinstance(author, list):
            author = ", ".join(map(str, author))
        author = str(author or "").strip()
        if re.match(r"^[^,]+, [^,]+$", author) and (pick.get("author") or "").lower() == "author_l-f":
            author = " ".join(reversed(author.split(", ")))
        y = _num(g("year"), int)
        url = str(g("url") or "").strip() or None
        if gr and r.get("Book Id"):
            url = f"https://www.goodreads.com/book/show/{r['Book Id']}"
        if not url:
            url = "https://openlibrary.org/search?" + urllib.parse.urlencode({"q": f"{title} {author}".strip()})
        tags = _tags(g("tags"))
        out.append(
            {
                "title": title,
                "author": author or None,
                "year": y if y and 0 < y < 2100 else None,
                "description": str(g("description") or "").strip(),
                "rating": _num(g("rating")),
                "ratings_count": _num(g("ratings_count"), int),
                "pages": _num(g("pages"), int),
                "url": url,
                "tags": ", ".join(tags),
            }
        )
    df = pl.DataFrame(
        out,
        schema={
            "title": pl.Utf8,
            "author": pl.Utf8,
            "year": pl.Int64,
            "description": pl.Utf8,
            "rating": pl.Float64,
            "ratings_count": pl.Int64,
            "pages": pl.Int32,
            "url": pl.Utf8,
            "tags": pl.Utf8,
        },
    )
    if df["rating"].drop_nulls().len() and df["rating"].max() > 5:  # 0-10 or 0-100 scales -> 0-5
        mx = df["rating"].max()
        df = df.with_columns((pl.col("rating") * (5 / (10 if mx <= 10 else 100))).alias("rating"))
    df = df.with_columns(pl.when(pl.col("rating") <= 0).then(None).otherwise(pl.col("rating")).alias("rating"))
    n0 = df.height
    df = (
        df.with_columns(
            pl.concat_str(
                [pl.col("title").str.to_lowercase(), pl.col("author").fill_null("").str.to_lowercase()], separator="|"
            ).alias("_k")
        )
        .sort("ratings_count", descending=True, nulls_last=True)
        .unique("_k", keep="first", maintain_order=True)
        .drop("_k")
    )
    if df.height < n0:
        log(f"dropped {n0 - df.height:,} duplicate title+author rows")
    return df


# ---------------------------------------------------------------- embedding
MODEL = os.environ.get("ATLAS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-d, 512-token context, ONNX via fastembed
MAX_TOKENS = 512


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    """Normalised embeddings. Texts are processed in length order (pad to the longest in the batch, not to 512 tokens),
    which is ~2.5x faster on CPU than fixed-length padding; the result is returned in the input order."""
    t0 = time.time()
    order = np.argsort([len(t) for t in texts], kind="stable")
    st = [texts[k] for k in order]
    try:
        from fastembed import TextEmbedding

        m = TextEmbedding(MODEL)
        tok = getattr(getattr(m, "model", None), "tokenizer", None)
        if tok is not None:
            tok.enable_truncation(max_length=MAX_TOKENS)
            pad = tok.padding or {}
            tok.enable_padding(pad_id=pad.get("pad_id", 0), pad_token=pad.get("pad_token", "[PAD]"))
        out = []
        for a in range(0, len(st), 4096):
            out.append(np.asarray(list(m.embed(st[a : a + 4096], batch_size=batch)), np.float32))
            log(f"  embedded {min(a + 4096, len(st)):,}/{len(st):,} ({time.time() - t0:.0f}s)")
        S = np.vstack(out)
    except ImportError:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("no embedding backend: pip install fastembed (default) or sentence-transformers")
        sm = SentenceTransformer(MODEL)
        sm.max_seq_length = MAX_TOKENS
        S = sm.encode(st, batch_size=batch, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    V = np.empty_like(S)
    V[order] = S
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V


def quantise(V: np.ndarray):
    sc = (np.abs(V).max(0) / 127.0 + 1e-12).astype(np.float32)
    return np.clip(np.round(V / sc), -127, 127).astype(np.int8), sc


# ---------------------------------------------------------------- layout
def layout(V: np.ndarray, seed: int = 42) -> np.ndarray:
    n = len(V)
    if n < 3:
        return np.array([[0.0, 0.0], [1.0, 0.0]][:n], np.float32)
    if n < 20:  # too few for UMAP: PCA
        X = V - V.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        return (X @ vt[:2].T).astype(np.float32) * 10
    import umap  # imported late (slow import)

    nn = int(min(15, max(2, n // 4)))
    fit_n = int(os.environ.get("ATLAS_UMAP_FIT_N", "150000"))  # above this, fit on a sample and project the rest
    t0 = time.time()
    if n <= fit_n:
        log(f"UMAP on {n:,} books (n_neighbors={nn})...")
        Y = umap.UMAP(
            n_neighbors=nn,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
            low_memory=True,
            init="spectral" if n > 50 else "random",
        ).fit_transform(V)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, fit_n, replace=False)
        log(f"UMAP: fitting on {fit_n:,} of {n:,} books, then projecting the rest...")
        U = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="cosine", random_state=seed, low_memory=True).fit(V[idx])
        Y = np.empty((n, 2), np.float32)
        Y[idx] = U.embedding_
        rest = np.setdiff1d(np.arange(n), idx)
        for a in range(0, len(rest), 50_000):
            Y[rest[a : a + 50_000]] = U.transform(V[rest[a : a + 50_000]])
            log(f"  projected {min(a + 50_000, len(rest)):,}/{len(rest):,}")
    log(f"layout done in {time.time() - t0:.0f}s")
    return np.asarray(Y, np.float32)


def kmeans2d(Y: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = Y[rng.choice(len(Y), k, replace=False)].astype(np.float64)
    for _ in range(iters):
        lab = (
            np.argmin(((Y[:, None, :] - C[None]) ** 2).sum(-1), 1)
            if len(Y) * k < 5e7
            else np.concatenate(
                [np.argmin(((Y[a : a + 200000, None, :] - C[None]) ** 2).sum(-1), 1) for a in range(0, len(Y), 200000)]
            )
        )
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = Y[m].mean(0)
    return lab


def clusters(Y: np.ndarray, tags: list[str]) -> tuple[list[str], list[str]]:
    n = len(Y)
    k = int(min(30, max(2, round(math.sqrt(n / 40))))) if n >= 6 else 1
    lab = kmeans2d(Y, k) if k > 1 else np.zeros(n, int)
    T = [t.split(", ") if t else [] for t in tags]
    glob: dict[str, int] = {}
    for ts in T:
        for t in ts:
            glob[t] = glob.get(t, 0) + 1
    names = {}
    for j in range(k):
        idx = np.where(lab == j)[0]
        cnt: dict[str, int] = {}
        for i in idx:
            for t in T[i]:
                cnt[t] = cnt.get(t, 0) + 1
        best = sorted(((c * (c / glob[t]), t) for t, c in cnt.items() if c >= max(2, 0.08 * len(idx))), reverse=True)
        names[j] = best[0][1].replace("-", " ") if best else f"cluster {j + 1}"
    return [names[j] for j in lab], [f"c{j:02d}" for j in lab]


# ---------------------------------------------------------------- main
def build(df: pl.DataFrame, out: Path, name: str, source: str, attribution: str = "", description: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    n = df.height
    log(f"building '{name}': {n:,} books -> {out}")
    desc = df["description"].fill_null("").to_list()
    has = sum(1 for d in desc if len(d) > 40)
    if has < n:
        log(f"{n - has:,} books have no (or a very short) description: embedding title + author + tags for those")
    # title + tags + description with marketing fluff stripped (textprep.py); books without a blurb still get title + tags
    texts = [
        book_text(t, d, tg, author=a)
        for d, t, a, tg in zip(desc, df["title"].to_list(), df["author"].to_list(), df["tags"].fill_null("").to_list())
    ]
    log(f"embedding {n:,} books (title + tags + cleaned description) with {MODEL} (local, max {MAX_TOKENS} tokens)...")
    V = embed(texts)
    Y = layout(V)
    labels, cl = clusters(Y, df["tags"].fill_null("").to_list())
    genre = [t.split(", ")[0].replace("-", " ") if t else None for t in df["tags"].fill_null("").to_list()]
    books = pl.DataFrame(
        {
            "i": np.arange(n, dtype=np.uint32),
            "title": df["title"],
            "author": df["author"],
            "year": df["year"],
            "x": Y[:, 0].astype(np.float32),
            "y": Y[:, 1].astype(np.float32),
            "cluster_label": labels,
            "ratings_count": df["ratings_count"].fill_null(0).cast(pl.Int32),
            "cluster": cl,
            "genre": genre,
            "rating": df["rating"].cast(pl.Float32),
            "url": df["url"],
        }
    )
    books.write_parquet(out / "books.parquet")
    pl.DataFrame(
        {
            "i": np.arange(n, dtype=np.uint32),
            "tags": df["tags"].fill_null(""),
            "pages": df["pages"],
            "desc": pl.Series([d[:360] for d in desc]),
        }
    ).write_parquet(out / "extra.parquet")
    Q, sc = quantise(V)
    np.save(out / "vecs_i8.npy", Q)
    np.save(out / "vecs_scale.npy", sc)
    meta = {
        "name": name,
        "n": n,
        "source": source,
        "attribution": attribution,
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "embedding_model": MODEL,
        "embedding_text": TEXT_RECIPE,
        "with_description": has,
        "with_rating": int(df["rating"].is_not_null().sum()),
        "description": description or f"{n:,} books",
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    log(f"done: {n:,} books, {has:,} with descriptions, {meta['with_rating']:,} with ratings")
    return out
