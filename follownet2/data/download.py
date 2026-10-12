"""Fetch the raw data sources listed in ``conf/sources.yaml`` (the registry).

Open sources come in two kinds: plain files over HTTP(S), which are resumed after an
interruption and checked against the registry's sizes and check sums, and Socrata tables,
which are paged as CSV and converted to one parquet file. Sources that need an application or
a manual step carry instructions instead and are never fetched. Every function that writes
checks the free disk space first, because the volume is shared with other jobs.

Provenance is written to ``SOURCES.json`` next to the data, in the layout used by the sibling
project cf-stability, so that either project can read the other's downloads in place
(:func:`find_existing`). Progress output belongs to ``scripts/download_data.py``, not here.
"""

from __future__ import annotations

import csv
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml

from follownet2.utils import REPO_ROOT, file_sha256, read_json, require_free_space, resolve_path, write_json

SOURCES_YAML = "conf/sources.yaml"
RAW_ROOT = "data/raw"  # paths.raw of conf/config.yaml
SIBLING_RAW_ROOT = REPO_ROOT.parent / "cf-stability" / "data" / "raw"
SOURCES_JSON = "SOURCES.json"
TIMEOUT = (30, 600)  # s: connect, read (Socrata may think for minutes before a deep page)
# Bound on parquet size / CSV size for the disk-space check (zstd parquet of NGSIM: about 0.2).
PARQUET_RATIO = 0.5

T = TypeVar("T")


class IntegrityError(OSError):
    """Data on disk disagree with what was expected: size, check sum or row count."""


class ServerError(requests.HTTPError):
    """A 5xx answer. Unlike a 4xx answer it is transient and retried."""


_TRANSIENT = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError, ServerError)


@dataclass(frozen=True)
class Item:
    """One entry of a download plan: a file, or a Socrata query stored as one parquet file."""

    source: str
    name: str
    bytes: int  # exact size of a file; an estimate for a table or an ``approx`` file
    exact: bool
    subset: tuple[str, ...] = ()
    url: str | None = None  # only files of open sources have one
    sha256: str | None = None
    md5: str | None = None
    domain: str | None = None  # Socrata: host, data set id, query, expected rows, column types
    table: str | None = None
    select: tuple[str, ...] = ()
    where: str | None = None
    rows: int | None = None
    dtypes: tuple[tuple[str, str], ...] = ()

    @property
    def path(self) -> str:
        """Location under the target directory, with forward slashes."""
        return f"{self.name}.parquet" if self.table else self.name


def load_sources(path: str | Path | None = None) -> dict:
    """The registry, ``{source name: entry}``; ``conf/sources.yaml`` unless ``path`` is given."""
    path = resolve_path(SOURCES_YAML if path is None else path)
    with open(path, encoding="utf-8") as fh:
        sources = yaml.safe_load(fh)
    if not isinstance(sources, dict):
        raise ValueError(f"{path}: expected a mapping from source names to entries")
    return sources


def plan(source: str, subset: str | Iterable[str] | None = None, sources: Mapping | None = None) -> list[Item]:
    """What would be fetched for ``source``, with sizes.

    ``subset`` selects items by their ``subset`` tags or by their names; a name that matches
    nothing raises ``ValueError``. A source of kind ``none`` plans nothing.
    """
    spec = _entry(source, sources)
    if spec["kind"] == "files":
        items = [_file_item(source, entry) for entry in spec["files"]]
    elif spec["kind"] == "socrata":
        items = [_table_item(source, entry) for entry in spec["tables"]]
    else:
        items = []
    if subset is None:
        return items
    wanted = {subset} if isinstance(subset, str) else set(subset)
    known = {label for item in items for label in (item.name, *item.subset)}
    if wanted - known:
        tags = sorted({tag for item in items for tag in item.subset})
        raise ValueError(f"{source}: unknown subset {sorted(wanted - known)}; tags: {tags}, or item names")
    return [item for item in items if wanted & {item.name, *item.subset}]


def total_bytes(items: Iterable[Item]) -> int:
    return sum(item.bytes for item in items)


def target_dir(spec: Mapping) -> Path:
    """Default location of a source: ``data/raw/<target>``."""
    return resolve_path(RAW_ROOT) / spec["target"]


def remaining_bytes(items: Iterable[Item], target: str | Path) -> int:
    """Bytes still to fetch into ``target``: complete files and converted tables count zero
    and the part file of a file is subtracted, since it is resumed."""
    target = Path(target)
    remaining = 0
    for item in items:
        path = target / item.path
        if path.is_file() and (item.table or not item.exact or path.stat().st_size == item.bytes):
            continue
        part = path.with_name(path.name + ".part")
        done = part.stat().st_size if item.exact and not item.table and part.is_file() else 0
        remaining += max(item.bytes - done, 0)
    return remaining


def local_dirs(spec: Mapping) -> tuple[Path, Path]:
    """Where :func:`find_existing` looks: ``data/raw/<target>``, then the sibling checkout."""
    return target_dir(spec), SIBLING_RAW_ROOT / spec.get("sibling_dir", spec["target"])


def find_existing(source: str, subset: str | Iterable[str] | None = None, sources: Mapping | None = None) -> Path | None:
    """A directory that already holds ``source``, or None.

    Looks at ``data/raw/<target>`` and then at the sibling checkout
    ``../cf-stability/data/raw/<sibling_dir or target>``, so the pipeline can read the data in
    place instead of downloading them again. Every planned file must be there with its
    registry size, and where the directory's SOURCES.json records a sha256 it must be the
    registry's; files are not hashed here, which would take minutes. A table counts when its
    parquet file exists, a source of kind ``none`` when every ``expected`` pattern matches.
    """
    sources = load_sources() if sources is None else sources
    spec = _entry(source, sources)
    items = plan(source, subset, sources)
    for root in local_dirs(spec):
        if root.is_dir() and _holds(root, spec, items):
            return root
    return None


def fetch_file(
    url: str,
    dest: str | Path,
    expected_bytes: int | None = None,
    sha256: str | None = None,
    session: requests.Session | None = None,
    chunk: int = 1 << 20,
    *,
    md5: str | None = None,
    retries: int = 5,
    backoff: float = 2.0,
) -> Path:
    """Download ``url`` to ``dest``, verify it and return ``dest``.

    A ``dest`` that already has ``expected_bytes`` (and the check sums given) is kept without
    any request. Otherwise the bytes go to ``<dest>.part``. When the size is known, an existing
    part file is resumed with an HTTP Range request: extended if the server answers 206,
    rewritten from the start if it answers 200 (no Range support); without a known size a
    complete part file could not be told from a truncated one, so it is discarded. Size and
    check sums are verified before the part file is renamed to ``dest`` in one step, so
    ``dest`` never holds a partial or unverified file; on a mismatch the part file is removed
    and :class:`IntegrityError` raised. Connection errors, timeouts and 5xx answers are retried
    ``retries`` times after ``backoff * 2**k`` seconds; a 4xx answer raises
    ``requests.HTTPError`` at once. ``DiskSpaceError`` is raised before the first request when
    the remaining bytes do not fit.
    """
    dest = Path(dest)
    if dest.is_file() and _mismatch(dest, expected_bytes, sha256, md5) is None:
        return dest
    part = dest.with_name(dest.name + ".part")
    if part.exists() and (expected_bytes is None or part.stat().st_size > expected_bytes):
        part.unlink()
    done = part.stat().st_size if part.exists() else 0
    require_free_space(dest, (expected_bytes or 0) - done)
    if expected_bytes is None or not part.exists() or done < expected_bytes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with _session(session) as http:
            try:
                _retrying(partial(_fetch_once, http, url, part, expected_bytes, chunk), retries, backoff)
            except IntegrityError:
                part.unlink(missing_ok=True)
                raise
    problem = _mismatch(part, expected_bytes, sha256, md5)
    if problem is not None:
        part.unlink()
        raise IntegrityError(f"{url}: {problem}; the partial download was removed")
    os.replace(part, dest)
    return dest


def socrata_url(domain: str, table: str) -> str:
    """SODA endpoint of a table; ``domain`` may carry its own scheme (``http://host:port``)."""
    origin = domain if "://" in domain else f"https://{domain}"
    return f"{origin}/resource/{table}.csv"


def fetch_socrata(
    domain: str,
    table: str,
    dest_dir: str | Path,
    select: Sequence[str] | None = None,
    where: str | None = None,
    page: int = 500_000,
    order: str = ":id",
    fmt: str = "csv",
    session: requests.Session | None = None,
    *,
    row_bytes: float = 200.0,
    retries: int = 5,
    backoff: float = 2.0,
    progress: Callable[[Path, int], None] | None = None,
) -> list[Path]:
    """Download a Socrata table, or its rows matching ``where``, as CSV pages; return them.

    Pages are requested with ``$limit``/``$offset`` in the stable order ``order`` and written
    as ``<dest_dir>/<table>_<offset:010d>.csv`` (through a part file, renamed when complete).
    A page already on disk is not requested again, so an interrupted run resumes page by
    page; paging stops at the first page with fewer than ``page`` rows. Before each request
    the estimated page size, ``page * row_bytes``, must fit on the disk. Retries are those of
    :func:`fetch_file`. ``progress(page_file, rows)`` is called for every page.
    """
    if fmt != "csv":
        raise ValueError(f"page format {fmt!r} is not supported: row counts and conversion read CSV")
    dest_dir = Path(dest_dir)
    url = socrata_url(domain, table)
    query: dict[str, str | int] = {"$order": order, "$limit": page}
    if select:
        query["$select"] = ",".join(select)
    if where:
        query["$where"] = where
    pages: list[Path] = []
    with _session(session) as http:
        offset = 0
        while True:
            path = dest_dir / f"{table}_{offset:010d}.csv"
            if not path.is_file():
                require_free_space(path, int(page * row_bytes))
                dest_dir.mkdir(parents=True, exist_ok=True)
                _retrying(partial(_fetch_page, http, url, {**query, "$offset": offset}, path), retries, backoff)
            rows = _csv_rows(path)
            if rows > page:
                raise ValueError(f"{path}: {rows} rows in a page of {page}; it was written with another page size")
            pages.append(path)
            if progress is not None:
                progress(path, rows)
            if rows < page:
                return pages
            offset += page


def socrata_to_parquet(
    pages: Sequence[str | Path],
    dest: str | Path,
    dtypes: Mapping[str, str] | None = None,
    chunk_rows: int = 250_000,
) -> Path:
    """Convert CSV pages into the single parquet file ``dest``, then delete the pages.

    Pages are read ``chunk_rows`` rows at a time, so memory is bounded by one chunk. All
    chunks share one schema: the pandas types given in ``dtypes``, and for the other columns
    float64 when the first page reads them as numbers, string otherwise. float64 holds
    integers below 2**53 exactly and does not break on a blank cell in a later page. The file
    is written as ``<dest>.part`` and renamed only when its row count equals the sum of the
    page rows; only then are the pages deleted. Otherwise the part file is removed, the pages
    are kept and :class:`IntegrityError` is raised.
    """
    pages = [Path(p) for p in pages]
    if not pages:
        raise ValueError("no pages to convert")
    dest = Path(dest)
    expected = sum(_csv_rows(p) for p in pages)
    require_free_space(dest, int(PARQUET_RATIO * sum(p.stat().st_size for p in pages)))
    types = {**_column_types(pages[0]), **(dtypes or {})}
    part = dest.with_name(dest.name + ".part")
    try:
        _write_parquet(pages, part, types, chunk_rows)
        rows = pq.read_metadata(part).num_rows
        if rows != expected:
            raise IntegrityError(f"{dest}: {rows} rows in the parquet file, {expected} in the pages; the pages are kept")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    for p in pages:
        p.unlink()
    return dest


def file_record(path: str | Path, name: str, url: str, sha256: str | None = None, **extra) -> dict:
    """Provenance of one file for :func:`write_sources_json`.

    ``sha256`` is computed unless given (pass only a value verified against the file);
    ``retrieved`` is the file's modification time in UTC, i.e. when its last byte arrived.
    """
    path = Path(path)
    stat = path.stat()
    return {
        "name": name,
        "url": url,
        "size": stat.st_size,
        "sha256": sha256 or file_sha256(path),
        "retrieved": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def write_sources_json(target_dir: str | Path, records: Iterable[Mapping], *, licence: str, source: str) -> Path:
    """Merge provenance records (:func:`file_record`) into ``<target_dir>/SOURCES.json``.

    The layout is cf-stability's, so either project reads the other's file: ``files`` maps
    each file's path under ``target_dir`` to its ``url``, ``size`` (bytes) and ``sha256``,
    here also ``retrieved`` (UTC) and any extra keys of the record; ``licence``, ``source``
    (landing page) and ``updated`` (UTC) are top-level keys. A record replaces the entry of
    the same file; other entries are kept.
    """
    path = Path(target_dir) / SOURCES_JSON
    payload = read_json(path) if path.exists() else {}
    files = payload.setdefault("files", {})
    for record in records:
        entry = dict(record)
        files[entry.pop("name")] = entry
    payload.update(licence=licence, source=source, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    require_free_space(path, 1 << 20)  # the file has a few kB; a MiB is ample
    write_json(path, payload)
    return path


def _entry(source: str, sources: Mapping | None) -> Mapping:
    sources = load_sources() if sources is None else sources
    if source not in sources:
        raise KeyError(f"unknown source {source!r}; known: {', '.join(sources)}")
    return sources[source]


def _tags(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _file_item(source: str, entry: Mapping) -> Item:
    return Item(
        source=source,
        name=entry["name"],
        bytes=int(entry["bytes"]),
        exact=not entry.get("approx", False),
        subset=_tags(entry.get("subset")),
        url=entry.get("url"),
        sha256=entry.get("sha256"),
        md5=entry.get("md5"),
    )


def _table_item(source: str, entry: Mapping) -> Item:
    return Item(
        source=source,
        name=entry["name"],
        bytes=int(entry["bytes_estimate"]),
        exact=False,
        subset=_tags(entry.get("subset")),
        domain=entry["domain"],
        table=entry["id"],
        select=_tags(entry.get("select")),
        where=entry.get("where"),
        rows=int(entry["rows"]),
        dtypes=tuple((entry.get("dtypes") or {}).items()),
    )


def _holds(root: Path, spec: Mapping, items: list[Item]) -> bool:
    if not items:
        patterns = spec.get("expected", ())
        return bool(patterns) and all(next(root.rglob(pattern), None) is not None for pattern in patterns)
    sources_json = root / SOURCES_JSON
    recorded = read_json(sources_json).get("files", {}) if sources_json.is_file() else {}
    return all(_present(root / item.path, item, recorded.get(item.path, {})) for item in items)


def _present(path: Path, item: Item, record: Mapping) -> bool:
    if not path.is_file():
        return False
    if item.table or not item.exact:
        return True
    if path.stat().st_size != item.bytes:
        return False
    return not (item.sha256 and record.get("sha256") and record["sha256"] != item.sha256)


@contextmanager
def _session(session: requests.Session | None) -> Iterator[requests.Session]:
    """The caller's session, or a new one that is closed afterwards."""
    if session is not None:
        yield session
        return
    with requests.Session() as new:
        yield new


def _retrying(attempt: Callable[[], T], retries: int, backoff: float) -> T:
    """``attempt()``, repeated after transient failures (connection errors, timeouts, 5xx)
    with waits of ``backoff * 2**k`` s; the last failure and any other error propagate."""
    for k in range(retries):
        try:
            return attempt()
        except _TRANSIENT:
            time.sleep(backoff * 2**k)
    return attempt()


def _raise_for_status(resp: requests.Response) -> None:
    if resp.status_code >= 500:
        raise ServerError(f"{resp.status_code} {resp.reason} for url: {resp.url}", response=resp)
    resp.raise_for_status()


def _fetch_once(http: requests.Session, url: str, part: Path, expected_bytes: int | None, chunk: int) -> None:
    """One request: extend the part file (206) or write it from the start (anything else)."""
    done = part.stat().st_size if expected_bytes is not None and part.exists() else 0
    headers = {"Accept-Encoding": "identity"}  # sizes and ranges refer to the stored bytes
    if done:
        headers["Range"] = f"bytes={done}-"
    with http.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
        if resp.status_code == 416:  # Range Not Satisfiable: the file is shorter than the part
            raise IntegrityError(f"{url}: the server has fewer than the {done} bytes already fetched")
        _raise_for_status(resp)
        if resp.status_code != 206:
            done = 0
        length = resp.headers.get("Content-Length")
        if length is not None and "Content-Encoding" not in resp.headers:
            if expected_bytes is not None and done + int(length) != expected_bytes:
                raise IntegrityError(f"{url}: the server sends {done + int(length)} bytes, expected {expected_bytes}")
            require_free_space(part, int(length))
        with open(part, "ab" if done else "wb") as fh:
            for block in resp.iter_content(chunk):
                done += len(block)
                if expected_bytes is not None and done > expected_bytes:
                    raise IntegrityError(f"{url}: more than the expected {expected_bytes} bytes")
                fh.write(block)


def _mismatch(path: Path, expected_bytes: int | None, sha256: str | None, md5: str | None) -> str | None:
    """Why ``path`` is not the expected file, or None when it is."""
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        return f"{size} bytes instead of {expected_bytes}"
    if sha256 and (actual := file_sha256(path)) != sha256.lower():
        return f"sha256 {actual} instead of {sha256}"
    if md5:
        with open(path, "rb") as fh:
            actual = hashlib.file_digest(fh, "md5").hexdigest()
        if actual != md5.lower():
            return f"md5 {actual} instead of {md5}"
    return None


def _fetch_page(http: requests.Session, url: str, params: Mapping, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    with http.get(url, params=params, stream=True, timeout=TIMEOUT) as resp:
        _raise_for_status(resp)
        with open(part, "wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    os.replace(part, path)


def _csv_rows(path: Path) -> int:
    """Records after the header, blank lines not counted. The csv module is used rather than
    a line count because quoted fields may contain line breaks."""
    with open(path, newline="", encoding="utf-8") as fh:
        return max(sum(1 for record in csv.reader(fh) if record) - 1, 0)


def _column_types(page: Path, sample_rows: int = 10_000) -> dict[str, str]:
    head = pd.read_csv(page, nrows=sample_rows)
    return {
        column: "float64" if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype) else "string"
        for column, dtype in head.dtypes.items()
    }


def _write_parquet(pages: list[Path], part: Path, types: Mapping[str, str], chunk_rows: int) -> None:
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    try:
        for page in pages:
            # the reader is closed explicitly: Windows cannot delete a page that is still open
            with pd.read_csv(page, dtype=dict(types), chunksize=chunk_rows) as reader:
                for frame in reader:
                    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(part, schema, compression="zstd")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
