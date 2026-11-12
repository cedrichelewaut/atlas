# ADR-003: Embeddings run locally, nothing leaves the machine by default

- **Status:** accepted
- **Context:** book descriptions are personal data; sending them to a hosted embedding API would leak the user's
  library. A map also needs to work offline and scale to ~500k books on a laptop CPU.
- **Decision:** embed with `BAAI/bge-small-en-v1.5` via fastembed/ONNX on-device (384-d, ≤512 tokens, no GPU), cached
  in `~/.atlas/models/` (~65 MB, once). Layout with UMAP (sampled fit above 150k books). `ATLAS_EMBED_MODEL` allows
  another fastembed model. Descriptions are pre-cleaned (`textprep.py`) so marketing fluff does not dominate the
  embedding.
- **Consequences:** import is private and offline after the one-time model download; quality is bounded by a small
  English model (roadmap item 1 explores blending in co-rating signals). Trade-off: first import downloads ~65 MB.
