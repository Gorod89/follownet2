"""Shared style of the report figures (docs/leaderboard.md, section 4).

One look for every figure: 160 mm wide, 9 pt sans-serif text, hairline axes without top and right
spines, no grid except hairlines at the decades of log axes, ``constrained_layout``, PNG at 200 dpi
and PDF (TrueType fonts embedded, no creation date, so a rerun gives the same file).

Palettes (checked with the colour-vision validator of the data-viz guidelines, Machado et al.
2009 simulation, OKLab distances x100, on white):

* :data:`FAMILY_COLORS`: Okabe-Ito blue, vermillion and bluish green for network, mechanistic and
  baseline; all pairs pass (worst protan/deutan distance 11.0, normal vision 18.7, every colour at
  least 3:1 against white).
* :data:`DATASET_COLORS`: the first six slots of the reference categorical palette in their fixed
  order (adjacent pairs pass: protan/deutan 9.1, normal vision 19.6); three of them are below 3:1
  against white, so dataset-coloured marks need direct labels. No report figure shows datasets and
  families together.
* :data:`HORIZON_GREYS`: an ordinal grey ramp (light to dark: one-step, 10 s, whole event), light
  end 2.1:1 against white, dark end light enough for black whiskers; no hue, so it never reads as a
  family.
* :data:`DIVERGING_CMAP`: ColorBrewer RdBu reversed (blue below 1, red above, neutral midpoint).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

MM = 1 / 25.4  # inches per millimetre
WIDTH_MM = 160.0
FONT_SIZE = 9.0
#: smallest text size used anywhere (cell values of dense heatmaps)
MIN_FONT_SIZE = 5.5
DPI = 200

INK = "#1a1a1a"  # text, reference lines, interval whiskers
SECONDARY = "#555555"  # notes
AXIS = "#4d4d4d"  # spines and ticks
GRID = "#e3e3e3"  # hairlines at the decades of log axes, separators
MISSING = "#d9d9d9"  # cells without a value

FAMILY_ORDER = ("baseline", "mechanistic", "network")
FAMILY_COLORS = {"baseline": "#009E73", "mechanistic": "#D55E00", "network": "#0072B2"}
#: a family outside FAMILY_ORDER (never expected; keeps a figure drawable)
OTHER_COLOR = "#7f7f7f"

#: canonical dataset order (trafficloop.experiments.analysis.TRANSFER_DATASETS)
DATASET_ORDER = ("ngsim", "follownet_highd", "openacc", "waymo_hu", "follownet_ngsim", "tgsim")
DATASET_COLORS = dict(zip(DATASET_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
DATASET_LABELS = {
    "ngsim": "NGSIM", "follownet_highd": "FollowNet highD", "openacc": "OpenACC",
    "waymo_hu": "Waymo (Hu et al.)", "follownet_ngsim": "FollowNet NGSIM", "tgsim": "TGSIM",
}
DATASET_SHORT = {
    "ngsim": "NGSIM", "follownet_highd": "FN-highD", "openacc": "OpenACC",
    "waymo_hu": "Waymo-Hu", "follownet_ngsim": "FN-NGSIM", "tgsim": "TGSIM",
}

HORIZON_GREYS = ("#b3b3b3", "#7a7a7a", "#474747")
DIVERGING_CMAP = "RdBu_r"

_FONT_CANDIDATES = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans")


def _font_family() -> str:
    """First installed font of :data:`_FONT_CANDIDATES` (a concrete name, so text drawn outside
    :func:`context` keeps it); DejaVu Sans ships with matplotlib."""
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in _FONT_CANDIDATES if name in installed), "DejaVu Sans")


FONT_FAMILY = _font_family()

RC = {
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "figure.labelsize": FONT_SIZE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.unicode_minus": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "patch.linewidth": 0.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.2,
    "legend.borderpad": 0.2,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
    "figure.constrained_layout.h_pad": 2.0 * MM,
    "figure.constrained_layout.w_pad": 2.0 * MM,
    "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


@contextmanager
def context():
    """The rc settings of :data:`RC` for the duration of the block (figures are built and saved
    inside it; nothing global changes)."""
    with matplotlib.rc_context(RC):
        yield


def figure(height_mm: float, width_mm: float = WIDTH_MM) -> Figure:
    """A white figure of ``width_mm`` x ``height_mm`` with constrained layout and its own Agg canvas
    (no pyplot, so nothing has to be closed). Call it inside :func:`context`."""
    fig = Figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def family_color(family) -> str:
    return FAMILY_COLORS.get(str(family), OTHER_COLOR)


def dataset_label(dataset: str, short: bool = False) -> str:
    return (DATASET_SHORT if short else DATASET_LABELS).get(dataset, dataset)


def panel_label(ax, index: int, text: str = "") -> None:
    """``(a) text`` above the top-left corner of ``ax`` (the only title an axes gets)."""
    ax.set_title(f"({chr(ord('a') + index)}) {text}".rstrip(), loc="left", fontsize=FONT_SIZE,
                 fontweight="bold", color=INK)


def note(ax, text: str) -> None:
    """A short remark above the top-right corner of ``ax`` (secondary ink, regular weight)."""
    ax.set_title(text, loc="right", fontsize=FONT_SIZE - 1, fontweight="normal", color=SECONDARY)


def save(fig: Figure, path_without_suffix) -> list[Path]:
    """Write ``fig`` as ``<path>.png`` (200 dpi) and ``<path>.pdf``; returns both paths. The figure
    keeps its size (no tight bounding box): constrained layout already keeps every label inside."""
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with context():
        for suffix, options in ((".png", {"dpi": DPI}), (".pdf", {"metadata": {"CreationDate": None}})):
            path = base.with_name(base.name + suffix)
            fig.savefig(path, facecolor="white", **options)
            paths.append(path)
    return paths
