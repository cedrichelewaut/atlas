#!/bin/sh
# Atlas one-line installer:  curl -fsSL https://raw.githubusercontent.com/cedrichelewaut/atlas/main/install.sh | sh
# Installs the `atlas` command with uv (preferred) or pipx, then tells you how to start. Never uses sudo.
# Needs ~900 MB of disk (UMAP/numba, scipy, ONNX runtime); the first `atlas import` also downloads a ~65 MB embedding model.
set -e
REPO="${ATLAS_REPO:-git+https://github.com/cedrichelewaut/atlas}"
say() { printf '%s\n' "$*"; }
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
BIN=""
if command -v uv >/dev/null 2>&1; then
  say "Installing Atlas with uv (about 900 MB of dependencies; a minute or two)..."; uv tool install --force --prerelease=allow "$REPO"
  BIN="$(uv tool dir --bin 2>/dev/null || true)"
elif command -v pipx >/dev/null 2>&1; then
  [ -n "$PY" ] || { say "Atlas needs Python 3.10+ (https://www.python.org/downloads/)"; exit 1; }
  say "Installing Atlas with pipx (about 900 MB of dependencies; a minute or two)..."; pipx install --force --python "$PY" --pip-args=--pre "$REPO"
  BIN="$(pipx environment --value PIPX_BIN_DIR 2>/dev/null || true)"
else
  say "Installing uv (https://docs.astral.sh/uv/) to manage Atlas's Python environment..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  say "Installing Atlas with uv (about 900 MB of dependencies; a minute or two)..."; uv tool install --force --prerelease=allow "$REPO"
  BIN="$(uv tool dir --bin 2>/dev/null || true)"
fi
if command -v ollama >/dev/null 2>&1; then
  say "Ollama found: for Atlas AI run   ollama pull ministral-3:8b"
else
  say "Optional: install Ollama (https://ollama.com) and run 'ollama pull ministral-3:8b' for the full Atlas AI;"
  say "without it Atlas AI runs in rules mode (filters, zoom, similar books, top rated)."
fi
say ""
say "Done. Try:   atlas import goodreads_library_export.csv && atlas serve"
say "       or:   atlas import mybooks.csv && atlas serve"
if ! command -v atlas >/dev/null 2>&1; then
  BIN="${BIN:-$HOME/.local/bin}"
  say ""
  say "'atlas' is installed in $BIN, which is not on your PATH in this shell. Add it (uv: 'uv tool update-shell'; pipx: 'pipx ensurepath'),"
  say "then open a new terminal, or for now run:"
  say "  export PATH=\"$BIN:\$PATH\""
  say "(or run it directly: $BIN/atlas import <your-books.csv> && $BIN/atlas serve)"
fi
