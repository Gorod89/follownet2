"""Download tooling against a local HTTP server on 127.0.0.1 (no internet), and the registry."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
import yaml

from follownet2 import utils
from follownet2.data import download as dl
from follownet2.utils import REPO_ROOT, DiskSpaceError

PAYLOAD = bytes(range(256)) * 800  # 204,800 bytes
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
MD5 = hashlib.md5(PAYLOAD).hexdigest()


class Server(ThreadingHTTPServer):
    """Serves the files under ``root`` and one fake Socrata table; records every request.

    ``ranges`` switches HTTP Range support; ``fail[path] = n`` answers 503 n times first.
    """

    def __init__(self, root: Path):
        super().__init__(("127.0.0.1", 0), Handler)
        self.root = root
        self.ranges = True
        self.fail: dict[str, int] = {}
        self.requests: list[dict] = []
        self.columns = ["id", "speed", "lane"]
        self.rows: list[list[str]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        self.server.requests.append({"path": url.path, "query": query, "range": self.headers.get("Range")})
        if self.server.fail.get(url.path, 0) > 0:
            self.server.fail[url.path] -= 1
            return self.reply(503, b"busy")
        if url.path.startswith("/resource/"):
            return self.socrata(query)
        path = self.server.root / url.path.lstrip("/")
        if not path.is_file():
            return self.reply(404, b"missing")
        data = path.read_bytes()
        wanted = self.headers.get("Range")
        if self.server.ranges and wanted:
            start = int(re.fullmatch(r"bytes=(\d+)-", wanted).group(1))
            if start >= len(data):
                return self.reply(416, b"")
            return self.reply(206, data[start:], {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
        return self.reply(200, data)

    def socrata(self, query: dict) -> None:
        """Honours $limit, $offset and $select, and quotes every field as SODA does."""
        columns = query.get("$select", ",".join(self.server.columns)).split(",")
        index = [self.server.columns.index(c) for c in columns]
        offset, limit = int(query["$offset"]), int(query["$limit"])
        lines = [",".join(f'"{c}"' for c in columns)]
        lines += [",".join(f'"{row[i]}"' for i in index) for row in self.server.rows[offset : offset + limit]]
        self.reply(200, ("\n".join(lines) + "\n").encode())

    def reply(self, status: int, body: bytes, headers: dict | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """No proxy of the environment for the local server, and ample free space by default:
    the real volume is shared with other jobs, and the disk guard is tested on its own."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(utils, "free_bytes", lambda path: 1 << 40)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    (root / "data.bin").write_bytes(PAYLOAD)
    srv = Server(root)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def http():
    with requests.Session() as session:
        yield session


def fetch(server: Server, http: requests.Session, tmp_path: Path, **kwargs) -> Path:
    kwargs = {"expected_bytes": len(PAYLOAD), "sha256": SHA256, "md5": MD5, "session": http, "backoff": 0.0, **kwargs}
    return dl.fetch_file(f"{server.url}/data.bin", tmp_path / "out" / "data.bin", **kwargs)


def write_part(tmp_path: Path, content: bytes) -> Path:
    part = tmp_path / "out" / "data.bin.part"
    part.parent.mkdir()
    part.write_bytes(content)
    return part


def test_plain_download(server, http, tmp_path):
    dest = fetch(server, http, tmp_path)
    assert dest.read_bytes() == PAYLOAD
    assert sorted(p.name for p in dest.parent.iterdir()) == ["data.bin"]
    assert [r["range"] for r in server.requests] == [None]


def test_interrupted_download_resumes_with_range(server, http, tmp_path):
    part = write_part(tmp_path, PAYLOAD[:50_000])
    dest = fetch(server, http, tmp_path)
    assert dest.read_bytes() == PAYLOAD and not part.exists()
    assert [r["range"] for r in server.requests] == ["bytes=50000-"]


def test_restart_when_the_server_ignores_range(server, http, tmp_path):
    server.ranges = False
    part = write_part(tmp_path, b"x" * 50_000)  # not a prefix: appending to it would corrupt the file
    dest = fetch(server, http, tmp_path)
    assert dest.read_bytes() == PAYLOAD and not part.exists()
    assert [r["range"] for r in server.requests] == ["bytes=50000-"]


def test_size_mismatch(server, http, tmp_path):
    with pytest.raises(dl.IntegrityError, match="bytes"):
        fetch(server, http, tmp_path, expected_bytes=len(PAYLOAD) + 1)
    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("digest", ["sha256", "md5"])
def test_checksum_mismatch_removes_the_part_file(server, http, tmp_path, digest):
    wrong = "0" * len(SHA256 if digest == "sha256" else MD5)
    with pytest.raises(dl.IntegrityError, match=digest):
        fetch(server, http, tmp_path, **{digest: wrong})
    assert list((tmp_path / "out").iterdir()) == []


def test_complete_file_is_skipped(server, http, tmp_path):
    dest = tmp_path / "out" / "data.bin"
    dest.parent.mkdir()
    dest.write_bytes(PAYLOAD)
    assert fetch(server, http, tmp_path) == dest
    assert server.requests == []


def test_retry_after_503_then_success(server, http, tmp_path):
    server.fail["/data.bin"] = 1
    assert fetch(server, http, tmp_path).read_bytes() == PAYLOAD
    assert len(server.requests) == 2


def test_503_retries_are_bounded(server, http, tmp_path):
    server.fail["/data.bin"] = 10
    with pytest.raises(dl.ServerError):
        fetch(server, http, tmp_path, retries=2)
    assert len(server.requests) == 3


def test_no_retry_on_404(server, http, tmp_path):
    with pytest.raises(requests.HTTPError) as info:
        dl.fetch_file(f"{server.url}/missing.bin", tmp_path / "missing.bin", 10, session=http, backoff=0.0)
    assert info.value.response.status_code == 404 and not isinstance(info.value, dl.ServerError)
    assert len(server.requests) == 1


def socrata_pages(server: Server, http: requests.Session, pages_dir: Path, n_rows: int, **kwargs) -> list[Path]:
    server.rows = [[str(i), f"{i / 4}", f"L{i % 3}"] for i in range(n_rows)]
    return dl.fetch_socrata(server.url, "abcd-1234", pages_dir, page=4, session=http, backoff=0.0, **kwargs)


def test_socrata_paging_and_resume(server, http, tmp_path):
    pages = socrata_pages(server, http, tmp_path / "pages", 10, select=["id", "speed"], where="speed > 0")
    assert [p.name for p in pages] == [f"abcd-1234_{k:010d}.csv" for k in (0, 4, 8)]
    assert [len(pd.read_csv(p)) for p in pages] == [4, 4, 2]
    queries = [r["query"] for r in server.requests]
    assert [(q["$offset"], q["$limit"]) for q in queries] == [("0", "4"), ("4", "4"), ("8", "4")]
    assert all(q["$order"] == ":id" and q["$select"] == "id,speed" and q["$where"] == "speed > 0" for q in queries)
    assert list(pd.read_csv(pages[0]).columns) == ["id", "speed"]
    # after an interruption only the missing page is requested again
    pages[1].unlink()
    server.requests.clear()
    again = socrata_pages(server, http, tmp_path / "pages", 10, select=["id", "speed"], where="speed > 0")
    assert again == pages
    assert [r["query"]["$offset"] for r in server.requests] == ["4"]


def test_socrata_pages_to_one_parquet_file(server, http, tmp_path):
    pages = socrata_pages(server, http, tmp_path / "pages", 8)
    assert [len(pd.read_csv(p)) for p in pages] == [4, 4, 0]  # the empty page ends the paging
    dest = dl.socrata_to_parquet(pages, tmp_path / "table.parquet", dtypes={"id": "Int64"})
    table = pq.read_table(dest)
    assert table.num_rows == 8
    assert table.schema.field("id").type == pa.int64() and table.schema.field("speed").type == pa.float64()
    lane = table.schema.field("lane").type
    assert pa.types.is_string(lane) or pa.types.is_large_string(lane)
    assert table.column("speed").to_pylist() == [i / 4 for i in range(8)]
    assert not any(p.exists() for p in pages) and not dest.with_name("table.parquet.part").exists()


def test_row_count_mismatch_keeps_the_pages(server, http, tmp_path, monkeypatch):
    pages = socrata_pages(server, http, tmp_path / "pages", 6)
    monkeypatch.setattr(dl, "_csv_rows", lambda path: 5)  # the pages would claim 10 rows
    with pytest.raises(dl.IntegrityError, match="rows"):
        dl.socrata_to_parquet(pages, tmp_path / "table.parquet")
    assert all(p.exists() for p in pages)
    assert not list(tmp_path.glob("table.parquet*"))


def test_disk_space_guard_requests_nothing(server, http, tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "free_bytes", lambda path: 1 << 20)
    with pytest.raises(DiskSpaceError):
        fetch(server, http, tmp_path)
    with pytest.raises(DiskSpaceError):
        dl.fetch_socrata(server.url, "abcd-1234", tmp_path / "pages", page=4, session=http)
    page = tmp_path / "page.csv"
    page.write_text('"id"\n"1"\n', encoding="utf-8")
    with pytest.raises(DiskSpaceError):
        dl.socrata_to_parquet([page], tmp_path / "table.parquet")
    assert server.requests == []
    assert not (tmp_path / "out").exists() and not (tmp_path / "pages").exists()
    assert page.exists() and not list(tmp_path.glob("table.parquet*"))


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_data", REPO_ROOT / "scripts" / "download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry(tmp_path, monkeypatch, server):
    """A registry with a file source and a table source served locally and an
    application-only source; the data roots point into tmp_path."""
    common = {
        "licence_url": "https://example.org/licence",
        "citation": "Demo (2026).",
        "landing_page": "https://example.org/demo",
    }
    sources = {
        "demo": {
            **common,
            "title": "Demo files",
            "licence": "CC0 1.0",
            "access": "open",
            "redistribution": "allowed",
            "kind": "files",
            "target": "demo",
            "files": [{"name": "data.bin", "url": f"{server.url}/data.bin", "bytes": len(PAYLOAD), "sha256": SHA256}],
        },
        "demo_table": {
            **common,
            "title": "Demo table",
            "licence": "US public domain",
            "access": "open",
            "redistribution": "allowed",
            "kind": "socrata",
            "target": "demo_table",
            "tables": [
                {
                    "name": "speeds",
                    "id": "abcd-1234",
                    "domain": server.url,
                    "rows": 6,
                    "bytes_estimate": 600,
                    "select": ["id", "speed"],
                    "where": "speed >= 0",
                    "dtypes": {"id": "Int64"},
                }
            ],
        },
        "restricted": {
            **common,
            "title": "Restricted drone data",
            "licence": "non-commercial use, no redistribution",
            "access": "application",
            "redistribution": "forbidden",
            "kind": "none",
            "target": "restricted",
            "instructions": "Apply on the landing page, then unpack the recordings.",
            "expected": ["*_tracks.csv"],
        },
    }
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(sources), encoding="utf-8")
    monkeypatch.setattr(dl, "SOURCES_YAML", str(path))
    monkeypatch.setattr(dl, "RAW_ROOT", str(tmp_path / "raw"))
    monkeypatch.setattr(dl, "SIBLING_RAW_ROOT", tmp_path / "sibling")
    return sources


def test_dry_run_sends_no_request(registry, server, tmp_path, capsys):
    assert load_script().main(["demo"]) == 0
    out = capsys.readouterr().out
    assert server.requests == []
    assert not (tmp_path / "raw").exists()
    for text in ("CC0 1.0", "https://example.org/licence", "data.bin", "204.80 kB", "free;", "demo --yes"):
        assert text in out


def test_application_source_is_refused(registry, server, capsys):
    assert load_script().main(["restricted", "--yes"]) == 2
    assert "Apply on the landing page" in capsys.readouterr().out
    assert server.requests == []


def test_download_records_provenance_like_cf_stability(registry, server, tmp_path):
    script = load_script()
    assert script.main(["demo", "--yes"]) == 0
    target = tmp_path / "raw" / "demo"
    assert (target / "data.bin").read_bytes() == PAYLOAD
    sources = utils.read_json(target / "SOURCES.json")
    assert sources["licence"] == "CC0 1.0" and sources["source"] == "https://example.org/demo"
    entry = sources["files"]["data.bin"]
    assert entry["sha256"] == SHA256 and entry["size"] == len(PAYLOAD) and entry["url"] == f"{server.url}/data.bin"
    assert datetime.fromisoformat(entry["retrieved"]).utcoffset() == timedelta(0)
    assert datetime.fromisoformat(sources["updated"]).utcoffset() == timedelta(0)
    # the data are now found in place: a second run requests nothing
    server.requests.clear()
    assert script.main(["demo", "--yes"]) == 0
    assert server.requests == []


def test_download_of_a_socrata_table(registry, server, tmp_path):
    server.rows = [[str(i), f"{i / 4}", f"L{i % 3}"] for i in range(6)]
    assert load_script().main(["demo_table", "--yes"]) == 0
    target = tmp_path / "raw" / "demo_table"
    assert sorted(p.name for p in target.iterdir()) == ["SOURCES.json", "speeds.parquet"]  # pages removed
    table = pq.read_table(target / "speeds.parquet")
    assert table.num_rows == 6 and table.column_names == ["id", "speed"]
    assert table.schema.field("id").type == pa.int64()
    entry = utils.read_json(target / "SOURCES.json")["files"]["speeds.parquet"]
    assert entry["rows"] == 6 and entry["where"] == "speed >= 0" and entry["select"] == ["id", "speed"]
    assert entry["url"] == f"{server.url}/resource/abcd-1234.csv"
    assert [r["query"]["$where"] for r in server.requests] == ["speed >= 0"]


def test_find_existing_checks_size_and_recorded_checksum(registry, tmp_path):
    sibling = tmp_path / "sibling" / "demo"
    sibling.mkdir(parents=True)
    (sibling / "data.bin").write_bytes(PAYLOAD)
    assert dl.find_existing("demo") == sibling
    utils.write_json(sibling / "SOURCES.json", {"files": {"data.bin": {"sha256": "0" * 64}}})
    assert dl.find_existing("demo") is None
    (sibling / "SOURCES.json").unlink()
    (sibling / "data.bin").write_bytes(PAYLOAD[:-1])
    assert dl.find_existing("demo") is None
    restricted = tmp_path / "raw" / "restricted"
    restricted.mkdir(parents=True)
    assert dl.find_existing("restricted") is None
    (restricted / "01_tracks.csv").write_text("id\n", encoding="utf-8")
    assert dl.find_existing("restricted") == restricted


REQUIRED = ("title", "licence", "licence_url", "access", "redistribution", "citation", "landing_page", "kind", "target")
SOURCES = {"follownet", "openacc", "ngsim", "waymo_hu", "tgsim", "ute", "highd", "exid", "citysim", "pneuma"}


def https(url) -> bool:
    return isinstance(url, str) and re.fullmatch(r"https://\S+", url) is not None


def test_registry_is_consistent():
    sources = dl.load_sources()
    assert set(sources) == SOURCES
    targets = [spec["target"] for spec in sources.values()]
    assert len(set(targets)) == len(targets)
    for name, spec in sources.items():
        assert all(isinstance(spec.get(key), str) and spec[key].strip() for key in REQUIRED), name
        assert spec["access"] in {"open", "application", "manual"}, name
        assert spec["redistribution"] in {"allowed", "forbidden", "unclear"}, name
        assert spec["kind"] in {"files", "socrata", "none"}, name
        assert https(spec["licence_url"]) and https(spec["landing_page"]), name
        files, tables = spec.get("files", []), spec.get("tables", [])
        assert bool(files) == (spec["kind"] == "files") and bool(tables) == (spec["kind"] == "socrata"), name
        names = [entry["name"] for entry in files + tables]
        assert len(set(names)) == len(names), name
        for f in files:
            assert type(f["bytes"]) is int and f["bytes"] > 0, f
            assert re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "0" * 64)), f
            assert re.fullmatch(r"[0-9a-f]{32}", f.get("md5", "0" * 32)), f
            assert https(f["url"]) if spec["access"] == "open" else "url" not in f, f
        for t in tables:
            assert type(t["rows"]) is int and t["rows"] > 0, t
            assert type(t["bytes_estimate"]) is int and t["bytes_estimate"] > 0, t
            assert t["select"] and t["id"] and t["domain"], t
            assert set(t.get("dtypes", {})) <= set(t["select"]), t
        if spec["access"] == "open":
            assert spec["kind"] != "none", name
        else:
            assert spec["instructions"].strip() and "://" not in spec["instructions"] and not tables, name
        if spec["access"] == "application":
            assert spec["kind"] == "none" and spec["expected"], name
    # totals of the verified inventories catch transcription errors
    assert len(dl.plan("follownet")) == 26 and dl.total_bytes(dl.plan("follownet")) == 1_468_486_648
    assert len(dl.plan("openacc")) == 158 and dl.total_bytes(dl.plan("openacc")) == 525_635_528
    assert sum(item.rows for item in dl.plan("tgsim")) == 13_785_701


def test_plan_selects_subsets():
    event_sets = dl.plan("follownet", "event_set")
    assert [item.name for item in event_sets] == [
        "HighD_car_fol_event_set.npy",
        "NGSIM_I_80_car_fol_event_set.npy",
        "SPMD_das1_car_fol_event_set.npy",
        "SPMD_das2_car_fol_event_set.npy",
        "Waymo_filter_car_fol_event_set.npy",
    ]
    assert len(dl.plan("follownet", ["highd", "lyft"])) == 7
    assert len(dl.plan("openacc", "casale")) == 12
    assert [item.path for item in dl.plan("ngsim", "i-80")] == ["ngsim_i-80_raw.parquet"]
    assert dl.plan("highd") == []
    with pytest.raises(ValueError, match="unknown subset"):
        dl.plan("follownet", "lyft_event_set")


def test_list_shows_every_source(capsys):
    assert load_script().main(["--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert {line.split()[0] for line in lines[1:11]} == SOURCES
