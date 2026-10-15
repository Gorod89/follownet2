"""Plan (the default) or run the download of one raw data source of conf/sources.yaml.

    python scripts/download_data.py --list
    python scripts/download_data.py <source> [--subset NAME ...] [--dest DIR]          dry run
    python scripts/download_data.py <source> [--subset NAME ...] [--dest DIR] --yes    download

The dry run downloads nothing and sends no request: it prints the licence and the access
conditions, every file with its size, the total, the free disk space, whether the data are
already on this machine (the pipeline then reads them in place) and the command that
downloads. Sources that need an application or a manual step are never downloaded: with
--yes the script prints what to do and exits with code 2.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import sys
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import requests

from trafficloop.data import download as dl
from trafficloop.utils import REPO_ROOT, DiskSpaceError, free_bytes, require_free_space, resolve_path

REFUSED = 2  # exit code of --yes for a source this script does not download


def human(n: float, exact: bool = True) -> str:
    """Size in decimal units, as the hosts state them (1 MB = 10**6 B); '~' marks an estimate."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}" if exact else f"~{n / scale:.3g} {unit}"
    return f"{n:.0f} B" if exact else f"~{n:.0f} B"


def shown(path: Path) -> str:
    """``path`` relative to the repository root, or absolute when it is on another drive."""
    try:
        return Path(os.path.relpath(path, REPO_ROOT)).as_posix()
    except ValueError:
        return str(path)


def field(label: str, text: str) -> None:
    print(
        textwrap.fill(
            text,
            width=100,
            initial_indent=f"  {label + ':':<15}",
            subsequent_indent=" " * 17,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def print_table(sources: dict) -> None:
    rows = [("source", "access", "kind", "items", "size", "raw data", "licence", "local copy")]
    for name, spec in sources.items():
        items = dl.plan(name, sources=sources)
        local = dl.find_existing(name, sources=sources)
        rows.append(
            (
                name,
                spec["access"],
                spec["kind"],
                str(len(items)) if items else "-",
                human(dl.total_bytes(items), all(item.exact for item in items)) if items else "-",
                spec["redistribution"],
                spec["licence"],
                shown(local) if local else "-",
            )
        )
    widths = [max(len(row[k]) for row in rows) for k in range(len(rows[0]))]
    for row in rows:
        cells = (cell.rjust(w) if k in (3, 4) else cell.ljust(w) for k, (cell, w) in enumerate(zip(row, widths)))
        print("  ".join(cells).rstrip())
    raw = resolve_path(dl.RAW_ROOT)
    print(f"\nfree disk space at {shown(raw)}: {human(free_bytes(raw))}; '~': estimate")


def print_items(items: list[dl.Item]) -> None:
    width = max(len(item.path) for item in items)
    for item in items:
        origin = f"  {item.rows:,} rows of {item.table} at {item.domain}" if item.table else ""
        print(f"    {item.path:<{width}}  {human(item.bytes, item.exact):>11}{origin}")
    total = human(dl.total_bytes(items), all(item.exact for item in items))
    print(f"    {'total':<{width}}  {total:>11}")
    if any(item.table for item in items):
        print("    (a table's size estimates its CSV pages; the parquet file is smaller)")


def command(args: argparse.Namespace) -> str:
    """The command line that downloads what the dry run showed, to be run from the repository root."""
    exe = Path(sys.executable)
    python = "./" + exe.relative_to(REPO_ROOT).as_posix() if exe.is_relative_to(REPO_ROOT) else str(exe)
    words = [python, "scripts/download_data.py", args.source]
    if args.subset:
        words += ["--subset", *args.subset]
    if args.dest:
        words += ["--dest", args.dest]
    return shlex.join([*words, "--yes"])


def report(args: argparse.Namespace, spec: dict, items: list[dl.Item], dest: Path, existing: Path | None) -> None:
    print(f"{args.source}: {spec['title']}")
    field("licence", f"{spec['licence']} ({spec['licence_url']})")
    field("access", f"{spec['access']}; redistribution of the raw data: {spec['redistribution']}")
    field("landing page", spec["landing_page"])
    field("citation", spec["citation"])
    for note in spec.get("notes", ()):
        field("note", note)
    if spec["access"] != "open":
        field("instructions", spec["instructions"])
    field("target", shown(dest))
    if existing is not None:
        field("local copy", f"{shown(existing)} holds it; the pipeline can read it in place")
    else:
        field("local copy", "none in " + " or ".join(shown(d) for d in dl.local_dirs(spec)))
    if items:
        print(f"  {'files:' if spec['kind'] == 'files' else 'tables:'}")
        print_items(items)
    elif spec.get("expected"):
        field("expected", "files matching " + ", ".join(spec["expected"]) + " in the target directory")
    free = human(free_bytes(dest))
    if items:
        remaining = dl.remaining_bytes(items, dest)
        try:
            require_free_space(dest, remaining)
            verdict = "fits"
        except DiskSpaceError as exc:
            verdict = f"does NOT fit: {exc}"
        exact = all(item.exact for item in items)
        field("disk", f"{free} free; {human(remaining, exact)} still to fetch into the target; {verdict}")
    else:
        field("disk", f"{free} free")
    print()
    if spec["access"] != "open":
        print(f"This script does not download sources with access '{spec['access']}'; see the instructions above.")
        return
    print("Dry run: nothing was downloaded. To download, run from the repository root:")
    print("  " + command(args))
    if existing is not None and args.dest is None:
        print(f"(The data are already in {shown(existing)}: --yes downloads nothing unless --dest is given.)")


def fetch_table(item: dl.Item, dest: Path, http: requests.Session) -> dict:
    path = dest / item.path
    if not path.is_file():
        pages_dir = dest / f"{item.name}.pages"
        pages = dl.fetch_socrata(
            item.domain,
            item.table,
            pages_dir,
            select=item.select,
            where=item.where,
            session=http,
            row_bytes=item.bytes / item.rows,
            progress=lambda page, rows: print(f"  {page.name}: {rows:,} rows", flush=True),
        )
        print(f"  converting {len(pages)} pages to {item.path}", flush=True)
        dl.socrata_to_parquet(pages, path, dtypes=dict(item.dtypes))
        with contextlib.suppress(OSError):  # left in place when something else is inside
            pages_dir.rmdir()
    rows = pq.read_metadata(path).num_rows
    url = dl.socrata_url(item.domain, item.table)
    return dl.file_record(path, item.path, url, rows=rows, select=list(item.select), where=item.where)


def download(spec: dict, items: list[dl.Item], dest: Path) -> None:
    require_free_space(dest, dl.remaining_bytes(items, dest))
    with requests.Session() as http:
        for item in items:
            print(f"{item.path} ({human(item.bytes, item.exact)})", flush=True)
            if item.table:
                record = fetch_table(item, dest, http)
            else:
                path = dl.fetch_file(item.url, dest / item.path, item.bytes, item.sha256, http, md5=item.md5)
                record = dl.file_record(path, item.path, item.url, sha256=item.sha256)
            dl.write_sources_json(dest, [record], licence=spec["licence"], source=spec["landing_page"])
            print(f"  ok: {human(record['size'])}, sha256 {record['sha256']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="name of the source (see --list)")
    parser.add_argument("--subset", nargs="+", metavar="NAME", help="subset tags or item names (default: all items)")
    parser.add_argument("--dest", help="target directory (default: data/raw/<target>; relative to the repository root)")
    parser.add_argument("--yes", action="store_true", help="download; without it the script only reports (dry run)")
    parser.add_argument("--list", action="store_true", help="print the table of all sources")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # a Windows code page may lack characters of the registry
    sources = dl.load_sources()
    if args.list:
        print_table(sources)
        return 0
    if args.source not in sources:
        parser.error(f"name a source ({', '.join(sources)}) or use --list")
    spec = sources[args.source]
    try:
        items = dl.plan(args.source, args.subset, sources)
    except ValueError as exc:
        parser.error(str(exc))
    dest = resolve_path(args.dest) if args.dest else dl.target_dir(spec)
    existing = dl.find_existing(args.source, args.subset, sources)
    if not args.yes:
        report(args, spec, items, dest, existing)
        return 0
    if spec["access"] != "open":
        print(f"{args.source}: access '{spec['access']}'; this script does not download it.")
        field("instructions", spec["instructions"])
        field("put files in", shown(dest))
        return REFUSED
    if existing is not None and args.dest is None:
        print(f"{args.source} is already on this machine in {shown(existing)}; the pipeline reads it in place.")
        print("Nothing was downloaded (--dest DIR downloads a copy).")
        return 0
    try:
        download(spec, items, dest)
    except (OSError, ValueError) as exc:  # requests' errors are OSErrors too
        print(f"error: {exc}", file=sys.stderr)
        response = getattr(exc, "response", None)
        if response is not None and response.status_code in (401, 403):
            print(
                f"The server refused the request. Download the files by hand from {spec['landing_page']} "
                f"into {shown(dest)}; a rerun skips the files that are complete.",
                file=sys.stderr,
            )
        return 1
    print(f"done: {shown(dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
