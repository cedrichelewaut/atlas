# AGENTS.md — Atlas

Conventions for anyone (human or agent) changing this repository. Vibe, Cursor, Claude Code and most modern
coding agents load this file automatically when working in this repo.

## Layout

```
src/atlas_books/
├── cli.py         `atlas` entry point: demo / import / serve / list / use
├── build.py       `atlas import` pipeline: read -> normalise -> textprep -> embed (fastembed) -> UMAP -> parquet + vecs + meta.json
├── paths.py       where things live (~/.atlas, datasets, current-dataset pointer, model cache)
├── store.py       dataset accessors (books.parquet, extra.parquet, meta.json, int8 vectors)
├── state.py       loaded dataset + FlexViz dashboard spec + mirror M of the browser's shared state (assembled at import; tests set ATLAS_HOME first)
├── routes.py      FastAPI routes: page, /api/* (state, chat, tools, map, rec, profile), SSE stream
├── tools.py       Atlas AI tools; also served at GET /api/tools and POST /api/tools/<name> for external agents
├── librarian.py   deterministic rules parser (no LLM) for common map requests
├── agent.py       LLM agent loop (Ollama / OpenAI-compatible), system prompt built from the dataset, trace hook
├── rec.py         recommendation wizard: candidates -> 2-4 multiple-choice questions -> top 3
├── providers.py   optional cloud "teacher" (Anthropic) for synthetic training data
├── security.py    ASGI guard: Host allowlist, same-origin checks, JSON-only + X-Atlas-Token on POSTs
├── textprep.py    strips marketing fluff / review quotes / library shelves from descriptions before embedding
└── web/           single-page UI (index.html + Inter font); talks to /api/* and embeds the FlexViz iframe
tests/             pytest; conftest.py boots a live `atlas demo` server on a free port with an isolated ATLAS_HOME
docs/              ARCHITECTURE.md, AGENT_GUIDE.md (the prompt given to the LLM), ROADMAP.md, adr/, screenshot.png
.githooks/         committed pre-commit hook (ruff); enable with `git config core.hooksPath .githooks`
```

## Rules

- Python >= 3.10. Match existing style; add type hints on new public functions.
- Every module keeps its top-of-file docstring (its role in one or two sentences). Public functions get a one-line
  docstring; add Args/Returns only when the signature is not self-explanatory.
- Lint and format with ruff (line-length 130, config in `pyproject.toml`): `ruff check` and `ruff format`.
  CI runs both plus pytest on Python 3.10/3.11/3.13 — keep it green.
- Every behaviour change gets/updates a test in `tests/` using the `server` / `api_post` fixtures. Tests must never
  touch the real `~/.atlas` (conftest already isolates `ATLAS_HOME`).
- Do not weaken anything in `security.py`; POSTs stay JSON + `X-Atlas-Token`, and the token never goes to the browser.
- User-facing changes update `README.md`; design decisions and trade-offs go in `docs/` (ADRs in `docs/adr/` are
  welcome); the LLM-facing prompt lives in `docs/AGENT_GUIDE.md`.
- Never commit caches or local data: `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `venv/`, `~/.atlas` contents.
- Commits: short imperative subject line, no emoji. The committed pre-commit hook (`.githooks/`) runs
  `ruff check` + `ruff format --check` on every commit once enabled (`git config core.hooksPath .githooks`).
