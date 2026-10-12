"""Rank-stability plot of the leaderboard (docs/leaderboard.md, section 3).

For every admitted model of a dataset: the composite rank (bootstrap over profiles, speeds and
training seeds) and, next
to it, the whole-event P1 rank (driver bootstrap), each as the mean rank over the replicates
(dot) with its 95 % interval (whiskers); models sorted by the mean composite rank (by the mean P1
rank for a dataset without composite), families in distinct colours. Drawn with the
object-oriented matplotlib API (no pyplot, no global backend state); PNG and PDF are written by
the Agg and PDF canvases.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

#: family colours: the first three slots of a colour-blind-safe categorical palette (validated all-pairs)
FAMILY_COLOURS = {"baseline": "#2a78d6", "mechanistic": "#eb6834", "network": "#1baf7a"}
OTHER_COLOUR = "#7a7974"
INK, MUTED, GRID = "#1b1b1a", "#5d5c58", "#e2e1dc"
FONT_SIZE = 9.0
PANELS = (("composite", "composite rank (profiles, speeds, training seeds)"), ("p1", "whole-event P1 rank (driver bootstrap)"))


def rank_ticks(n: int) -> list[int]:
    """Every rank up to 12 models, else 1, the multiples of 5 and the last rank."""
    if n <= 12:
        return list(range(1, n + 1))
    ticks = [1, *range(5, n + 1, 5)]
    return ticks + [n] if n - ticks[-1] >= 2 else ticks[:-1] + [n]


def model_order(composite_ranks: pd.DataFrame | None, p1_ranks: pd.DataFrame | None) -> list[str]:
    """Models by the mean composite rank, then the models without composite by the mean P1 rank."""
    order: list[str] = []
    for frame in (composite_ranks, p1_ranks):
        if frame is not None and len(frame):
            order += [m for m in frame.sort_values("mean_rank", kind="stable").index if m not in order]
    return order


def rank_stability_figure(dataset: str, composite_ranks: pd.DataFrame | None, p1_ranks: pd.DataFrame | None,
                          families: Mapping[str, str], n_boot: int) -> Figure:
    """The figure (one panel per available rank distribution, shared model axis).

    ``composite_ranks`` and ``p1_ranks``: index model, columns ``mean_rank``, ``rank_low``,
    ``rank_high`` (:func:`~follownet2.stats.rank_distribution`); ``families``: model -> family.
    """
    frames = {"composite": composite_ranks, "p1": p1_ranks}
    panels = [(key, title, frames[key]) for key, title in PANELS if frames[key] is not None and len(frames[key])]
    order = model_order(composite_ranks, p1_ranks)
    n = max(len(order), 1)
    small = FONT_SIZE - 0.5
    fig = Figure(figsize=(1.9 + 3.3 * max(len(panels), 1), 1.45 + 0.23 * n), layout="constrained")
    fig.suptitle(f"{dataset}: rank stability over {n_boot} bootstrap replicates\n(dot: mean rank, whiskers: 95 % interval)",
                 fontsize=FONT_SIZE + 0.5, color=INK)
    if not panels:
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, "no rank distribution available", ha="center", va="center", color=MUTED, fontsize=FONT_SIZE)
        return fig
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    position = {m: i for i, m in enumerate(order)}
    for index, (ax, (_, title, frame)) in enumerate(zip(axes, panels)):
        for model, row in frame.iterrows():
            colour = FAMILY_COLOURS.get(families.get(model, ""), OTHER_COLOUR)
            y = position[model]
            ax.hlines(y, row["rank_low"], row["rank_high"], color=colour, lw=1.8, zorder=2)
            ax.plot([row["mean_rank"]], [y], "o", ms=5.5, color=colour, mec="white", mew=0.8, zorder=3)
        ax.set_xlim(0.4, len(frame) + 0.6)
        ax.set_xticks(rank_ticks(len(frame)))
        ax.set_xlabel("rank (1 = best)", color=INK, fontsize=FONT_SIZE)
        ax.set_title(f"({chr(ord('a') + index)}) {title}", loc="left", color=INK, fontsize=FONT_SIZE)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=small)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    present = [f for f in FAMILY_COLOURS if f in {families.get(m) for m in order}]
    handles = [Line2D([], [], marker="o", ls="-", lw=1.8, ms=5.5, color=FAMILY_COLOURS[f], mec="white", label=f) for f in present]
    if any(families.get(m) not in FAMILY_COLOURS for m in order):
        handles.append(Line2D([], [], marker="o", ls="-", lw=1.8, ms=5.5, color=OTHER_COLOUR, mec="white", label="other"))
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False, fontsize=small)
    return fig


def save_figure(fig: Figure, out_dir, dataset: str, dpi: int = 200) -> tuple[Path, Path]:
    """Write ``rank_stability_<dataset>.png`` (``dpi``) and ``.pdf`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"rank_stability_{dataset}.png", out_dir / f"rank_stability_{dataset}.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    return png, pdf
