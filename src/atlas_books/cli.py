"""atlas: map your books and explore them with Atlas AI.

atlas demo                         open the bundled sample (3,000 public-domain novels from Project Gutenberg)
atlas import books.csv             build a map from your own list (Goodreads export, CSV, JSON, Parquet)
atlas import --hf user/dataset     ... or from a Hugging Face dataset you have access to
atlas serve                        open the current dataset in the browser
atlas list | atlas use NAME        switch between imported datasets
"""

import argparse
import os
import re
import sys
import webbrowser
import threading
from pathlib import Path
from . import paths, __version__


def _serve(data: Path, host: str, port: int, open_browser: bool):
    os.environ["ATLAS_DATA"] = str(data)
    import uvicorn

    url = f"http://{host if host not in ('0.0.0.0', '::') else '127.0.0.1'}:{port}/"
    print(f"Atlas: serving {data.name} at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("atlas_books.app:app", host=host, port=port, log_level="warning")


def cmd_import(a):
    from . import build

    if a.hf:
        rows = build.read_hf(a.hf, a.hf_file)
        source = f"huggingface:{a.hf}"
        name = a.name or re.sub(r"[^a-z0-9]+", "-", a.hf.split("/")[-1].lower())
    else:
        if not a.file:
            sys.exit("usage: atlas import <file> | atlas import --hf <user/dataset>")
        p = Path(a.file).expanduser()
        if not p.exists():
            sys.exit(f"no such file: {p}")
        rows = build._read_any(str(p))
        source = p.name
        name = a.name or re.sub(r"[^a-z0-9]+", "-", p.stem.lower()).strip("-") or "books"
    mapping = dict(m.split("=", 1) for m in (a.map or []))
    df = build.normalise(rows, mapping)
    if a.limit:
        df = df.sort("ratings_count", descending=True, nulls_last=True).head(a.limit)
    if df.height < 5:
        sys.exit(f"only {df.height} usable rows; Atlas needs at least 5 books")
    kind = "Goodreads library" if build.is_goodreads_export(rows) else "books"
    out = build.build(
        df, paths.datasets() / name, name, source, description=f"{df.height:,} {kind} from {source} · mapped by description"
    )
    if not a.no_switch:
        paths.set_current(name)
    print(f"\nImported {df.height:,} books as '{name}'.  Next: atlas serve", flush=True)
    if a.serve:
        _serve(out, a.host, a.port, not a.no_browser)


def cmd_serve(a):
    if a.dataset:
        d = paths.datasets() / a.dataset
        if not (d / "books.parquet").exists():
            sys.exit(f"no dataset '{a.dataset}' (see: atlas list)")
    else:
        d = paths.data_dir()
    _serve(d, a.host, a.port, not a.no_browser)


def cmd_demo(a):
    _serve(paths.SAMPLE_BUILT, a.host, a.port, not a.no_browser)


def cmd_list(a):
    cur = paths.current_name()
    rows = [d for d in sorted(paths.datasets().iterdir()) if (d / "books.parquet").exists()]
    print(f"sample (bundled){'   <- current' if not cur else ''}")
    for d in rows:
        import json

        m = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
        print(f"{d.name:24s} {m.get('n', '?'):>8} books  {m.get('source', '')}{'   <- current' if d.name == cur else ''}")


def cmd_use(a):
    if a.name == "sample":
        f = paths.home() / "current"
        f.unlink(missing_ok=True)
    elif not (paths.datasets() / a.name / "books.parquet").exists():
        sys.exit(f"no dataset '{a.name}' (see: atlas list)")
    else:
        paths.set_current(a.name)
    print(f"current dataset: {a.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="atlas", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"atlas {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def net(p):
        p.add_argument("--host", default=os.environ.get("ATLAS_HOST", "127.0.0.1"))
        p.add_argument("--port", type=int, default=int(os.environ.get("ATLAS_PORT", "8010")))
        p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")

    p = sub.add_parser("import", help="build a map from a book list")
    p.add_argument("file", nargs="?")
    p.add_argument("--hf", metavar="USER/DATASET", help="read from a Hugging Face dataset (uses $HF_TOKEN if set)")
    p.add_argument("--hf-file", help="a specific file inside the HF dataset")
    p.add_argument("--name", help="dataset name (default: from the file name)")
    p.add_argument(
        "--map", action="append", metavar="FIELD=COLUMN", help="column mapping, e.g. --map description=Blurb (repeatable)"
    )
    p.add_argument("--limit", type=int, help="keep only the N most-rated books")
    p.add_argument("--no-switch", action="store_true", help="don't make it the current dataset")
    p.add_argument("--serve", action="store_true", help="start the server when done")
    net(p)
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("serve", help="open the current (or --dataset) map")
    p.add_argument("--dataset")
    net(p)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("demo", help="open the bundled sample map")
    net(p)
    p.set_defaults(fn=cmd_demo)
    sub.add_parser("list", help="list imported datasets").set_defaults(fn=cmd_list)
    p = sub.add_parser("use", help="make a dataset current")
    p.add_argument("name")
    p.set_defaults(fn=cmd_use)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
