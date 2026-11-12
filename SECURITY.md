# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub private vulnerability reporting (this repo's Security
tab → "Report a vulnerability"). Do not open a public issue. You should hear back within a few days.

## Atlas's security model

- **No login by design.** Atlas binds to `127.0.0.1` by default; `--host 0.0.0.0` prints a warning, because anyone who can reach the port can open your map, read your reading profile and use Atlas AI.
- **POSTs are token-gated.** Every mutating endpoint requires `Content-Type: application/json` and the per-launch `X-Atlas-Token` (stored mode-0600 in `$ATLAS_HOME/api_token`), plus same-origin/Host checks in `src/atlas_books/security.py`. GETs only serve read-only map data.
- **LLM endpoints are validated.** Custom base URLs are checked (scheme, host, no credentials embedded in the URL) and warn when non-local, since your chat, the current view and your reading profile are sent there. API keys stay on your machine in `~/.atlas/librarian_model.json` (mode 0600) and are never sent to the browser.
- **Local by default.** Embeddings run on-device (fastembed/ONNX, model cached in `~/.atlas/models/`); nothing leaves your machine unless you configure a remote LLM endpoint or a web-lookup provider key.
- **Scope.** Vulnerabilities in third-party dependencies (FlexViz, FastAPI, UMAP, fastembed, ...) should be reported upstream; this policy covers Atlas's own code.
