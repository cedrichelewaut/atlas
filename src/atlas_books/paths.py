"""Where Atlas keeps things.

ATLAS_HOME (default ~/.atlas)            settings + your datasets
  datasets/<name>/                       built map data (books.parquet, extra.parquet, vecs_i8.npy, vecs_scale.npy, meta.json)
  current                                name of the dataset `atlas serve` opens
  librarian_model.json                   Atlas AI model choice (0600; may hold an API key; never leaves the server in full)
  user_profile.md                        your reading profile (edited in the Profile tab)
  goodreads_library_export.csv           optional: your Goodreads export, used by the profile
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
WEB = PKG / "web"
SAMPLE_CSV = PKG / "sample" / "sample_gutenberg.csv"
SAMPLE_BUILT = PKG / "sample" / "built"


def home() -> Path:
    h = Path(os.environ.get("ATLAS_HOME", Path.home() / ".atlas")).expanduser()
    h.mkdir(parents=True, exist_ok=True)
    return h


def datasets() -> Path:
    d = home() / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_name() -> str | None:
    f = home() / "current"
    return f.read_text().strip() if f.exists() else None


def set_current(name: str) -> None:
    (home() / "current").write_text(name)


def data_dir() -> Path:
    """Dataset the server opens: $ATLAS_DATA (a directory), else ~/.atlas/current, else the bundled sample."""
    if os.environ.get("ATLAS_DATA"):
        return Path(os.environ["ATLAS_DATA"]).expanduser()
    n = current_name()
    if n and (datasets() / n / "books.parquet").exists():
        return datasets() / n
    return SAMPLE_BUILT
