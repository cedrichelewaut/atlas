"""Loads one built dataset for the app (see build.py for the format)."""

import json
import os
import numpy as np
import polars as pl
from . import paths

DATA = str(paths.data_dir())
if not os.path.exists(f"{DATA}/books.parquet"):
    raise SystemExit(f"No dataset at {DATA}. Run `atlas import <file>` first, then `atlas serve`.")
META = json.load(open(f"{DATA}/meta.json")) if os.path.exists(f"{DATA}/meta.json") else {}
EXTRA = f"{DATA}/extra.parquet"


def books() -> pl.DataFrame:
    B = pl.read_parquet(f"{DATA}/books.parquet").sort("i")
    for c, t in (
        ("year", pl.Int64),
        ("rating", pl.Float64),
        ("ratings_count", pl.Int64),
        ("pages", pl.Int32),
        ("genre", pl.Utf8),
        ("cluster", pl.Utf8),
        ("cluster_label", pl.Utf8),
        ("url", pl.Utf8),
    ):
        if c not in B.columns:
            B = B.with_columns(pl.lit(None, dtype=t).alias(c))
    return B


def vec_getter():
    """(n, get(a,b) -> float32 rows a..b) over the description embeddings."""
    Q = np.load(f"{DATA}/vecs_i8.npy", mmap_mode="r")
    sc = np.load(f"{DATA}/vecs_scale.npy").astype(np.float32)
    return len(Q), (lambda a, b: np.asarray(Q[a:b], np.float32) * sc)


def describe(n: int) -> str:
    """Corpus description used in the UI subtitle and the Atlas AI system prompt."""
    return META.get("description") or f"{n:,} books"
