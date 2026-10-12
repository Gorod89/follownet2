"""Build the leaderboard of the benchmark (docs/leaderboard.md, milestone 5).

    python scripts/make_leaderboard.py [datasets=a,b] [root=results/models] [out=results/leaderboard]
                                       [n_boot=1000] [seed=0] [date=YYYY-MM-DD]

Validates every run directory under ``root`` (the checklist gate of
``follownet2/leaderboard/checklist.py``), builds per dataset the main table with its bootstrap
intervals, the driver-leak, transfer and refused tables and the provenance footer, and writes into
``out``: ``LEADERBOARD.md`` (the datasets built), ``<dataset>.md``, ``index.html`` (one
self-contained page) and ``rank_stability_<dataset>.png`` / ``.pdf``. Prints the admitted and
refused counts per dataset. ``datasets`` defaults to every dataset with runs under ``root``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from follownet2.leaderboard.render import dataset_page, html_page, leaderboard_markdown
from follownet2.leaderboard.stability import rank_stability_figure, save_figure
from follownet2.leaderboard.tables import Board, build_leaderboard
from follownet2.utils import REPO_ROOT, resolve_path

DEFAULTS = {"datasets": "", "root": "results/models", "out": "results/leaderboard", "n_boot": "1000", "seed": "0", "date": ""}


def arguments(argv: list[str] | None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise SystemExit(f"unknown argument {item!r}; usage: make_leaderboard.py " + " ".join(f"[{k}=...]" for k in DEFAULTS))
        args[key] = value
    datasets = [d.strip() for d in args["datasets"].split(",") if d.strip()]
    return {"datasets": datasets or None, "root": resolve_path(args["root"]), "out": resolve_path(args["out"]),
            "n_boot": int(args["n_boot"]), "seed": int(args["seed"]), "date": args["date"] or None}


def _shown(path: Path) -> str:
    """``path`` relative to the repository when it lies inside it."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def report(board: Board) -> str:
    c = board.counts()
    leak = f"; driver leak {c['leak_admitted']} admitted, {c['leak_refused']} refused" if c["leak_admitted"] + c["leak_refused"] else ""
    return (f"{board.dataset}: ranked entries {c['admitted']} admitted, {c['refused']} refused{leak}; "
            f"run directories {c['runs']} validated, {c['runs_refused']} refused")


def main(argv: list[str] | None = None) -> dict[str, Board]:
    args = arguments(argv)
    root, out = args["root"], args["out"]
    started = time.time()
    try:
        boards = build_leaderboard(root, args["datasets"], args["n_boot"], args["seed"], args["date"])
    except ValueError as error:
        raise SystemExit(str(error)) from None
    out.mkdir(parents=True, exist_ok=True)
    source = _shown(root)
    names, images = {}, {}
    for dataset, board in boards.items():
        families = dict(zip(board.main["model"], board.main["family"])) if len(board.main) else {}
        figure = rank_stability_figure(dataset, board.composite_ranks, board.p1_ranks, families, board.n_boot)
        png, _ = save_figure(figure, out, dataset)
        names[dataset], images[dataset] = png.name, png.read_bytes()
        (out / f"{dataset}.md").write_text(dataset_page(board, png.name, source), encoding="utf-8")
        print(report(board), flush=True)
    (out / "LEADERBOARD.md").write_text(leaderboard_markdown(boards, names, source), encoding="utf-8")
    (out / "index.html").write_text(html_page(boards, images, source), encoding="utf-8")
    print(f"leaderboard of {len(boards)} dataset(s) written to {_shown(out)} ({time.time() - started:.0f} s)", flush=True)
    return boards


if __name__ == "__main__":
    main()
