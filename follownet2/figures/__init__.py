"""Figures of the report (docs/leaderboard.md, section 4).

``style``: palettes, rc settings and :func:`~follownet2.figures.style.save` (PNG at 200 dpi and
PDF); ``plots``: one function per figure, data frames in, :class:`matplotlib.figure.Figure` out.
``scripts/make_report.py`` reads the data and writes ``results/figures``.
"""

from follownet2.figures.plots import (
    concave_growth_curves,
    horizon_curves,
    persistence_ratio_bars,
    rank_bump_chart,
    split_inflation_bars,
    transfer_heatmap,
)
from follownet2.figures.style import save

__all__ = [
    "concave_growth_curves",
    "horizon_curves",
    "persistence_ratio_bars",
    "rank_bump_chart",
    "save",
    "split_inflation_bars",
    "transfer_heatmap",
]
