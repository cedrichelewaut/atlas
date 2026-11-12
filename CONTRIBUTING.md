# Contributing to Atlas

Issues and PRs are welcome. Atlas is a single-developer home project, so keep changes small and focused.

## Development setup

Requires Python 3.10+.

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install --pre -e ".[dev]"          # --pre: FlexViz is a beta on PyPI
git config core.hooksPath .githooks     # enable the committed pre-commit hook (ruff)
```

## Checks

- `ruff check && ruff format` — lint and format (line-length 130, config in `pyproject.toml`). The pre-commit hook runs both in check mode on every commit.
- `pytest -q` — the suite serves the committed fixture dataset (`tests/fixtures/testdata`) on a free port with an isolated `ATLAS_HOME`; it never touches your real `~/.atlas`. CI (`.github/workflows/ci.yml`) runs lint + the full test matrix (Python 3.10/3.11/3.13) plus an end-to-end CLI check.
- `atlas serve` — manual smoke test in the browser on your current dataset.

## Conventions

See [AGENTS.md](AGENTS.md) — Vibe and most modern coding agents load it automatically, and it states the same rules: module docstrings (PEP 257), one-line docstrings on public functions, type hints on new public code, never weaken `security.py`, tests for every behaviour change, user-facing changes in `README.md`, design decisions as ADRs in `docs/adr/`.

## Commits and PRs

- Short imperative commit subjects, no emoji. Keep commits focused; CI must stay green.
- Open a PR against `main` with a short description of the change and how you tested it.
