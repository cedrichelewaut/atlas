"""Atlas web app: FlexViz is the whole visualisation; its client state is the shared surface for a human and Atlas AI.
Run via the CLI:  atlas serve   (or: ATLAS_DATA=<dataset dir> uvicorn atlas_books.app:app --port 8010)

The code lives in state.py (dataset + shared FlexViz state), tools.py (Atlas AI tools) and routes.py (HTTP).
"""

from .routes import app

__all__ = ["app"]
