# Notices and attribution

## Bundled sample data (`src/atlas_books/sample/`)
The sample (2,920 English-language novels) comes from the **Project Gutenberg catalog**
(<https://www.gutenberg.org/ebooks/offline_catalogs.html>): title, author, subjects/bookshelves, catalog summary, download
count and ebook URL. The books are in the public domain in the USA. Project Gutenberg is a registered trademark of the
Project Gutenberg Literary Archive Foundation; this project is not affiliated with or endorsed by it. The `downloads` column
(used as `ratings_count`) is Gutenberg's download count; there are no ratings. `sample/built/` holds the map built from
that CSV with `atlas import` (MiniLM embeddings quantised to int8, UMAP coordinates).

**No Goodreads data is included.** Atlas started as a prototype on the UCSD Goodreads dataset, which is licensed for
academic, non-commercial use only; none of it (and nothing derived from it) is in this repository. The Goodreads importer
reads *your own* library export. `tests/fixtures/` contains synthetic files (public-domain titles, made-up ids and ratings).

## Third-party software and assets
- **FlexViz** (<https://flexviz.tech>, <https://github.com/flex-analytics/flexviz>): Apache-2.0, installed from PyPI (not vendored).
- **all-MiniLM-L6-v2** sentence-embedding model (downloaded at import time by fastembed / sentence-transformers): Apache-2.0.
- **UMAP** (umap-learn): BSD-3-Clause. **Polars**, **FastAPI**, **Uvicorn**, **NumPy**: MIT/BSD.
- **Inter** typeface (`web/assets/InterVariable.woff2`) by Rasmus Andersson: SIL Open Font License 1.1.
