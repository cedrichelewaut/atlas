# ADR-004: No book data ships with Atlas

- **Status:** accepted (supersedes the earlier "bundled Project Gutenberg sample" state — the sample was removed so the
  repo carries no dataset at all)
- **Context:** the prototype was built on the UCSD Goodreads research dataset, which is licensed for academic,
  non-commercial use only. Shipping any of it (or the derived vectors) in a public repo would breach that licence.
  The public-domain Project Gutenberg sample that used to ship in `src/atlas_books/sample/` was licence-clean but made
  the repo carry 5 MB of data and a second data path (`atlas demo`).
- **Decision:** nothing ships. Users import their own list — a Goodreads CSV export
  (`atlas import goodreads_library_export.csv`), any CSV/JSON/Parquet, or a Hugging Face dataset they own.
- **Consequences:** `atlas serve` requires a prior `atlas import` (clear error otherwise); tests use a small synthetic
  fixture dataset (`tests/fixtures/testdata`, 199 books, made-up ids/ratings). Goodreads exports carry no
  descriptions, so those books are placed by title + author + shelves (less precise) — documented in the README
  import table. Full attribution in `NOTICE.md`.
