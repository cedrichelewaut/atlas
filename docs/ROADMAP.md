# Roadmap

1. **Better recommendations.** "Similar books" uses descriptions only, so it returns companions and look-alikes. Blend in
   reader co-ratings (item-item filtering) and de-duplicate editions.
2. **Personal Goodreads profile.** Highlight read / to-read / 5-star books on the map, hide what you've read, seed
   recommendations from favourites.
3. **Readable regions.** Label clusters per zoom level (c-TF-IDF over tags and blurbs) so the map isn't anonymous.
4. **Incremental imports.** `atlas import --append` that projects new books with the saved UMAP model; an Open Library importer for fresher data.
5. **Faster, more reliable LLM path.** Constrained tool calls, streaming, model warm-up, an eval in CI.
