"""Build the human platoon windows of protocol P2h: ``data/scenarios/human_profiles.json``.

    python scripts/build_human_profiles.py [out=data/scenarios/human_profiles.json] [sources=ngsim,tgsim]
                                           [min_followers=4] [max_per_site=20]
                                           [openacc=data/scenarios/openacc_profiles.json]

Windows of the track sources by the rules of ``trafficloop/data/platoons.py`` (docs/protocols.md,
section 2, P2h): NGSIM and TGSIM as configured in ``conf/dataset/<name>.yaml`` (``raw_dir``,
``options``, ``reconstruction`` over the defaults of ``conf/config.yaml``), at most ``max_per_site``
windows per site, plus the AstaZero windows of the OpenACC cache whose followers are all human
(the two manual runs, names kept; the cache is built when missing). The file has the layout of the
OpenACC cache (``{dt, meta, profiles}``); ``meta`` holds the rules, the reconstruction settings, the
SHA-256 of the source files, the counts per site and one record per window. The same data give the
same file. Prints a summary per site and per window.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
from _common import show
from omegaconf import OmegaConf

from trafficloop.data import platoons
from trafficloop.data.reconstruction import ReconstructionConfig
from trafficloop.data.registry import get_loader
from trafficloop.data.schema import DT
from trafficloop.protocols.profiles import CACHE, LeaderProfile, build_openacc_cache, load_profiles, save_profiles
from trafficloop.utils import REPO_ROOT, file_sha256, read_json, resolve_path, to_plain

DEFAULTS = {
    "out": platoons.CACHE,
    "sources": "ngsim,tgsim",
    "min_followers": str(platoons.MIN_FOLLOWERS),
    "max_per_site": str(platoons.MAX_PER_SITE),
    "openacc": CACHE,
}
SOURCES = ("ngsim", "tgsim")
RULE = (
    "a leader and a chain of at least min_followers consecutive followers in one lane (the leader of follower k is "
    "vehicle k-1 at every sample at a raw net gap in [min_spacing, max_spacing), all vehicles present throughout, no "
    "lane change; followers as in the loader's events: NGSIM autos in the mainline lanes, TGSIM not automated); the "
    "longest stretch of the chain's common interval of min_window to window seconds whose leader speed is at least "
    "min_start_speed at the start and spans at least min_speed_range, ties by the largest leader-speed variance, then "
    "the earliest start; per site the windows ranked by followers, then duration, skipping every window that shares a "
    "vehicle with one chosen, at most max_per_site; reconstruction of the dataset (leader first, then every follower "
    "against its reconstructed predecessor), the speed rules checked again on the reconstructed leader"
)


def arguments(argv: Sequence[str] | None = None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            usage = " ".join(f"[{k}=...]" for k in DEFAULTS)
            raise SystemExit(f"unknown argument {item!r}; usage: build_human_profiles.py {usage}")
        args[key] = value
    sources = [s for s in args["sources"].split(",") if s]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; known: {list(SOURCES)}")
    return {
        "out": resolve_path(args["out"]),
        "sources": sources,
        "min_followers": int(args["min_followers"]),
        "max_per_site": int(args["max_per_site"]),
        "openacc": resolve_path(args["openacc"]),
    }


def dataset_settings(name: str) -> tuple[dict, ReconstructionConfig]:
    """``conf/dataset/<name>.yaml`` and its reconstruction settings over those of ``conf/config.yaml``
    (as ``pipeline.reconstruction_config``)."""
    base = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    dataset = to_plain(OmegaConf.load(REPO_ROOT / "conf" / "dataset" / f"{name}.yaml"))
    settings = {**to_plain(base.reconstruction), **(dataset.get("reconstruction") or {})}
    return dataset, ReconstructionConfig(**settings)


def source_hashes(loader, raw_dir, options: dict) -> dict[str, str]:
    """SHA-256 of the files the loader reads: NGSIM one parquet file per location, TGSIM one file
    per data set (the ``files`` option, else ``<key>.parquet``, else the CSV export, as the loader)."""
    if loader.DATASET == "ngsim":
        candidates = [[f"ngsim_{location}_raw.parquet"] for location in options.get("locations") or ("us-101", "i-80")]
    else:
        names = options.get("datasets") or [name for name, spec in loader.DATASETS.items() if spec.axis]
        files = options.get("files") or {}
        candidates = [[files[n]] if n in files else [f"{n}.parquet", loader.DATASETS[n].file] for n in names]
    found = [next((name for name in names if (raw_dir / name).is_file()), None) for names in candidates]
    return {name: file_sha256(raw_dir / name) for name in found if name is not None}


def stats(values) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if not x.size:
        return {"min": None, "median": None, "max": None}
    return {"min": round(float(x.min()), 3), "median": round(float(np.median(x)), 3), "max": round(float(x.max()), 3)}


def site_summary(profiles: Sequence[LeaderProfile]) -> dict:
    return {
        "windows": len(profiles),
        "followers": stats(len(p.followers) for p in profiles),
        "duration": stats((len(p.v) - 1) * DT for p in profiles),
        "leader_speed_range": stats(float(p.v.max() - p.v.min()) for p in profiles),
        "leader_speed_min": stats(float(p.v.min()) for p in profiles),
        "leader_speed_max": stats(float(p.v.max()) for p in profiles),
    }


def site_of(profile: LeaderProfile) -> str:
    """``<dataset>/<site>`` of a profile (the OpenACC copies: ``openacc/<campaign>``)."""
    return "/".join(profile.source.split("/")[:2])


def astazero_human(path) -> tuple[list[LeaderProfile], dict]:
    """The AstaZero windows of the OpenACC cache (built when missing) whose followers are all
    human, and the SHA-256 of their files."""
    if not path.exists():
        build_openacc_cache(path=path)
    profiles = [
        p for p in load_profiles(path)
        if p.source.startswith("openacc/AstaZero/") and p.followers and all(f.driver == platoons.HUMAN for f in p.followers)
    ]
    hashes = read_json(path).get("meta", {}).get("sha256", {})
    return profiles, {p.source: hashes.get(p.source) for p in profiles}


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    rules = {
        "window": platoons.WINDOW,
        "min_window": platoons.MIN_WINDOW,
        "min_followers": args["min_followers"],
        "min_start_speed": platoons.MIN_START_SPEED,
        "min_speed_range": platoons.MIN_SPEED_RANGE,
        "max_per_site": args["max_per_site"],
    }
    spacing = {"min_spacing": platoons.MIN_SPACING, "max_spacing": platoons.MAX_SPACING}
    profiles: list[LeaderProfile] = []
    records: list[dict] = []
    sources: dict[str, dict] = {}
    for name in args["sources"]:
        dataset, cfg = dataset_settings(name)
        loader = get_loader(dataset["loader"])
        raw_dir = resolve_path(dataset["raw_dir"])
        options = dict(dataset.get("options") or {})
        found, rows, counts = platoons.track_profiles(
            loader.load_tracks(raw_dir, **options), name, cfg, args["min_followers"], args["max_per_site"],
            rules["window"], rules["min_window"], rules["min_start_speed"], rules["min_speed_range"],
        )
        profiles += found
        records += rows
        sources[name] = {
            "raw_dir": dataset["raw_dir"],
            "options": options,
            "reconstruction": {k: v for k, v in asdict(cfg).items() if k not in ("eps", "max_iter", "dt")},
            "sha256": source_hashes(loader, raw_dir, options),
            "sites": counts,
        }
    human, hashes = astazero_human(args["openacc"])
    profiles += human
    sources["openacc"] = {
        "cache": DEFAULTS["openacc"] if args["openacc"] == resolve_path(CACHE) else str(args["openacc"]),
        "profiles": [p.name for p in human],
        "sha256": hashes,
        "note": "copied unchanged from the OpenACC cache: raw speeds and IVS gaps, not reconstructed",
    }
    by_site: dict[str, list[LeaderProfile]] = {}
    for profile in profiles:
        by_site.setdefault(site_of(profile), []).append(profile)
    summary = {site: site_summary(items) for site, items in by_site.items()}
    by_dataset = {name: sum(1 for p in profiles if p.source.startswith(name + "/")) for name in [*args["sources"], "openacc"]}
    meta = {
        "rule": RULE,
        **rules,
        **spacing,
        "driver": platoons.HUMAN,
        "gap": "s_k = x_front[k-1] - L[k-1] - x_front[k] of the reconstructed front bumpers, L the vehicle's median length",
        "sources": sources,
        "counts": {"profiles": len(profiles), "by_dataset": by_dataset, "by_site": summary},
        "windows": records,
    }
    path = save_profiles(profiles, args["out"], meta)
    counts = {f"{name}/{site}": count for name, block in sources.items() for site, count in block.get("sites", {}).items()}
    show("windows per site", [
        {
            "site": site,
            "windows": len(by_site.get(site, [])),
            "chains": count.get("chains", "-"),
            "candidates": count.get("candidates", "-"),
            "rejected": count.get("rejected", "-"),
            **(
                {
                    "followers min/med/max": "{min}/{median}/{max}".format(**summary[site]["followers"]),
                    "duration s min/med/max": "{min}/{median}/{max}".format(**summary[site]["duration"]),
                    "leader range m/s min/med/max": "{min}/{median}/{max}".format(**summary[site]["leader_speed_range"]),
                    "leader speed m/s": f"{summary[site]['leader_speed_min']['min']}..{summary[site]['leader_speed_max']['max']}",
                }
                if site in summary else {}
            ),
        }
        for site, count in {**counts, **{site: {} for site in summary if site not in counts}}.items()
    ])
    show("windows", [
        {
            "name": p.name,
            "followers": len(p.followers),
            "duration": round((len(p.v) - 1) * DT, 1),
            "v_min": round(float(p.v.min()), 2),
            "v_max": round(float(p.v.max()), 2),
            "length": round(p.vehicle_length, 2),
            "gap_min": round(min(float(f.s.min()) for f in p.followers), 2),
            "gap_max": round(max(float(f.s.max()) for f in p.followers), 2),
        }
        for p in profiles
    ])
    print(f"\n{len(profiles)} profiles written to {path}")


if __name__ == "__main__":
    main()
