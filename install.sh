#!/bin/sh
# Atlas one-line installer:  curl -fsSL https://raw.githubusercontent.com/cedrichelewaut/atlas/main/install.sh | sh
# Installs the `atlas` command with uv (preferred) or pipx, then tells you how to start. Never uses sudo.
set -e
REPO="${ATLAS_REPO:-git+https://github.com/cedrichelewaut/atlas}"
say() { printf '%s\n' "$*"; }
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if command -v uv >/dev/null 2>&1; then
  say "Installing Atlas with uv..."; uv tool install --force --prerelease=allow "$REPO"
elif command -v pipx >/dev/null 2>&1; then
  [ -n "$PY" ] || { say "Atlas needs Python 3.10+ (https://www.python.org/downloads/)"; exit 1; }
  say "Installing Atlas with pipx..."; pipx install --force --python "$PY" --pip-args=--pre "$REPO"
else
  say "Installing uv (https://docs.astral.sh/uv/) to manage Atlas's Python environment..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv tool install --force --prerelease=allow "$REPO"
fi
if command -v ollama >/dev/null 2>&1; then
  say "Ollama found: for Atlas AI run   ollama pull ministral-3:8b"
else
  say "Optional: install Ollama (https://ollama.com) and run 'ollama pull ministral-3:8b' for the full Atlas AI;"
  say "without it Atlas AI runs in rules mode (filters, zoom, similar books, top rated)."
fi
say ""
say "Done. Try:   atlas demo                 (bundled sample)"
say "       or:   atlas import mybooks.csv && atlas serve"
say "(If 'atlas' is not found, open a new terminal or add ~/.local/bin to PATH.)"
