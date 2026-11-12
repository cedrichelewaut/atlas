# ADR-004: No Goodreads data ships with Atlas

- **Status:** accepted
- **Context:** the prototype was built on the UCSD Goodreads research dataset, which is licensed for academic,
  non-commercial use only. Shipping any of it (or the derived vectors) in a public repo would breach that licence.
- **Decision:** nothing from that dataset is in this repo. The bundled sample is 2,920 public-domain novels from the
  Project Gutenberg catalog (summaries, subjects, download counts). Users import their own Goodreads CSV export
  (`atlas import goodreads_library_export.csv`).
- **Consequences:** the demo works out of the box with no licence concerns; Goodreads exports carry no descriptions,
  so those books are placed by title + author + shelves (less precise) — documented in the README import table.
  Full attribution in `NOTICE.md`.
