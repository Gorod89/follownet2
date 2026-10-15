"""Small shared helpers: repository paths, config hashing, JSON output, seeding."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Interpret a relative path as relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def to_plain(cfg: Any) -> Any:
    """Convert an OmegaConf object (or anything JSON-like) to plain containers."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return cfg


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def config_hash(cfg: Any, length: int = 12) -> str:
    """Stable hash of a config: sha256 of its canonical JSON form."""
    text = json.dumps(to_plain(cfg), sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON through a temporary file, so that a crash leaves no truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_plain(payload), indent=2, sort_keys=True, default=_json_default)
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    part.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    # utf-8-sig: files written by PowerShell start with a byte-order mark
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiskSpaceError(OSError):
    pass


def free_bytes(path: str | Path) -> int:
    """Free space of the volume that holds ``path`` (the nearest existing parent is probed)."""
    import shutil

    probe = Path(path).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


#: free space that must remain after a write, unless the caller asks for another reserve
DISK_RESERVE = 200 << 20


def require_free_space(path: str | Path, needed: int, reserve: int = DISK_RESERVE) -> int:
    """Raise :class:`DiskSpaceError` unless ``needed + reserve`` bytes are free at ``path``.

    The reserve protects other processes that write to the same volume.
    """
    free = free_bytes(path)
    if free < needed + reserve:
        raise DiskSpaceError(
            f"{path}: {free / 2**20:.0f} MiB free, {needed / 2**20:.0f} MiB needed "
            f"plus a reserve of {reserve / 2**20:.0f} MiB"
        )
    return free
