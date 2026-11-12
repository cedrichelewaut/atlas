# Notices and attribution

**No book data ships with Atlas** — no bundled sample, and no Goodreads data. Atlas started as a prototype on the UCSD Goodreads dataset, which is licensed for
academic, non-commercial use only; none of it (and nothing derived from it) is in this repository. The Goodreads importer
reads *your own* library export. `tests/fixtures/` contains synthetic files (public-domain titles, made-up ids and ratings).

## Third-party software and assets
- **FlexViz** (<https://flexviz.tech>, <https://github.com/flex-analytics/flexviz>): Apache-2.0, installed from PyPI (not vendored).
- **BAAI/bge-small-en-v1.5** text-embedding model (downloaded at import time by fastembed / sentence-transformers): MIT.
  Datasets built with an earlier Atlas version used **all-MiniLM-L6-v2** (Apache-2.0).
- **UMAP** (umap-learn): BSD-3-Clause. **Polars**, **FastAPI**, **Uvicorn**, **NumPy**: MIT/BSD.
- **Inter** typeface (`web/assets/InterVariable.woff2`) by Rasmus Andersson: SIL Open Font License 1.1.
