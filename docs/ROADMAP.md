# Top 5 improvements (prioritised)

1. **Recommendations from reader co-ratings, not descriptions alone.**
   *Why:* "similar books" uses blurb embeddings only, so it surfaces companions and look-alikes (for *The Hunger Games*:
   *The World of the Hunger Games*, *Guide to The Hunger Games*) rather than what the same readers loved. Blend
   item-item collaborative filtering (implicit ALS or co-occurrence on ratings/shelves) with the embedding score, and
   de-duplicate editions and companions. *Effort:* M (needs an interactions source the licence allows: your own export
   plus an open dataset). *Impact:* high: better "books like X", and the recommendation wizard gets real taste signals.

2. **A real Goodreads profile import that personalises the map.**
   *Why:* the Profile tab is free text and the Goodreads card is a placeholder. Importing the user's export as a
   *profile* (not just as a dataset) lets Atlas highlight read / to-read / 5-star books on the big map, exclude what
   they've read, and seed recommendations from their favourites. *Effort:* S–M (parser exists; add matching by ISBN/title,
   a highlight layer and "near books I loved"). *Impact:* high: the first thing returning users want.

3. **Readable regions: cluster / topic labels on the map.**
   *Why:* labels were removed because they were noisy, so the map is anonymous until you hover. Label regions per zoom
   level with c-TF-IDF over tags + blurbs (optionally polished by the LLM offline) and place them with collision
   avoidance. *Effort:* S–M. *Impact:* medium–high: orientation, and Atlas AI can refer to areas by name.

4. **Fresher data past 2017, and incremental imports.**
   *Why:* the Goodreads research snapshot stops in 2017, so recent books are missing. Add `atlas import --append`
   (embed only new books and project them with the saved UMAP model, no full rebuild), plus an Open Library dump
   importer for a licence-clean, up-to-date catalogue with language filtering built in. *Effort:* M. *Impact:* high for
   anyone using Atlas for current reading.

5. **LLM reliability and latency.**
   *Why:* rules-first keeps common commands instant and correct (100% on the eval set), but the free-form LLM path is
   slower (seconds per tool round on an 8B model) and less exact (18/21 on the LLM-only eval). Run the eval in CI,
   use JSON-schema constrained tool calls, stream partial answers, warm the model at start-up, cache view summaries, and
   pick the smallest model that passes. *Effort:* M. *Impact:* medium–high: answers feel instant and can be trusted.
