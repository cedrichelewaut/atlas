# Atlas

**A map of your books, arranged by what they are about, that you explore with an AI librarian.**

Every book is a dot; books with similar descriptions sit close together. Zoom, filter by year, rating or length, click a
dot to open the book, or ask *Atlas AI*: "books like Dracula", "only after 1990", "zoom to science fiction". You and the
AI share **one view**: what it does happens on your screen, and it can see what you see.

![Atlas: the bundled Project Gutenberg sample, after asking "books like Dracula"](docs/screenshot.png)

## Quickstart

```sh
pipx install --pip-args=--pre git+https://github.com/cedrichelewaut/atlas   # or: uv tool install --prerelease=allow git+https://github.com/cedrichelewaut/atlas
atlas demo                                                                   # opens the sample map at http://127.0.0.1:8010
```

Your own books:

```sh
atlas import goodreads_library_export.csv && atlas serve
```

Other ways to install: `curl -fsSL https://raw.githubusercontent.com/cedrichelewaut/atlas/main/install.sh | sh`, or
Docker: `docker build -t atlas . && docker run -p 8010:8010 atlas`.

Good to know:
- `--pre` is needed because Atlas depends on the beta [FlexViz](https://flexviz.tech) `0.1.0b5`.
- The install is about **900 MB** (UMAP, SciPy, ONNX Runtime). `atlas demo` needs nothing more.
- The first `atlas import` downloads a ~65 MB embedding model to `~/.atlas/models/` (once).
- If port 8010 is busy, Atlas uses the next free one and tells you.

## Commands

| Command | What it does |
|---|---|
| `atlas demo` | Open the bundled sample |
| `atlas import <file>` | Build a map from a book list (`--name`, `--limit N`, `--map description=MyColumn`, `--serve`) |
| `atlas serve` | Open the current map (`--dataset NAME`, `--host`, `--port`, `--no-browser`) |
| `atlas list` / `atlas use <name>` | List maps / switch to one |

Maps live in `~/.atlas/datasets/<name>/` (`ATLAS_HOME` moves them).

## Importing your books

| Format | Notes |
|---|---|
| **Goodreads export** | Goodreads → My Books → Import and export → Export library. Has no descriptions, so books are placed by title, author and shelves (less precise). |
| **CSV / TSV** | Columns are matched by name: `title`, `author`, `year`, `description`, `rating`, `ratings_count`, `pages`, `url`, `tags` (common synonyms work). Use `--map description=MyColumn` to override. |
| **JSON / JSON Lines / Parquet** | A list of objects (or `{"books": [...]}`) with the same fields. |
| **Hugging Face dataset** | `atlas import --hf user/dataset` (needs `pipx inject atlas-books huggingface_hub`). |

Each book becomes one text (title, tags, cleaned description), embedded **locally** (no GPU, nothing leaves your machine),
then laid out in 2D with UMAP. Expect about a minute for 3,000 books; it scales to ~500k.

## Atlas AI

Out of the box Atlas AI runs in **rules mode**: filters, zoom, "books like X", authors, topics, top rated and a
recommendation wizard. For free-form chat, run [Ollama](https://ollama.com) (`ollama pull ministral-3:8b`) and pick the
model in the dropdown, or connect any OpenAI-compatible endpoint (the key stays in `~/.atlas/librarian_model.json`, mode 0600).

## How it works

The browser shows a FlexViz dashboard; a FastAPI server keeps a mirror of its state. Atlas AI (rules, then optional LLM)
changes the view through the same tools an external agent could use (`GET /api/tools`, `POST /api/tools/<name>`), the
browser applies the change and confirms it. Books you look up get an orange ring; recommendations get a yellow ring.

## Development

```sh
pip install --pre -e '.[dev]' && pytest && ruff check && ruff format --check
```

## Data and licence

Code is **MIT**. The bundled sample is 2,920 public-domain novels from the Project Gutenberg catalog (no ratings, so "top
rated" uses downloads). No Goodreads data ships with Atlas. Details in [`NOTICE.md`](NOTICE.md); ideas in [`docs/ROADMAP.md`](docs/ROADMAP.md).
