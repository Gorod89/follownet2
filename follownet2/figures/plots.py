"""The six figures of the report (docs/leaderboard.md, section 4).

Every function takes data frames (the CSVs of ``results/m4`` or tables of the run directories,
read by ``scripts/make_report.py``) and returns a :class:`matplotlib.figure.Figure` 160 mm wide;
:func:`follownet2.figures.style.save` writes it as PNG and PDF. Nothing here reads or writes
files. Lower is better for every error; families are coloured as in
:data:`~follownet2.figures.style.FAMILY_COLORS` in every figure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import colormaps, ticker
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath

from follownet2.figures import style

#: as in follownet2.experiments.analysis (tests/test_figures.py checks they agree)
REFERENCE = "persistence"
SUBSET = ("idm", "gipps", "mlp", "lstm", "mlp_loop", "lstm_loop", "ddpg", "stability_mlp")
MAIN_DATASETS = ("ngsim", "follownet_highd")
LEAK_DATASETS = ("ngsim", "openacc")
#: the four models of the concave-growth figure
P2_MODELS = ("idm", "ddpg", "mlp_loop", "lstm")

E2_METRICS = {
    "p0_rmse_a": "P0 acceleration, one step",
    "p1_h10_rmse_s": "P1 spacing, 10 s",
    "p1_whole_rmse_s": "P1 spacing, whole event",
}
E3_COLUMNS = {
    "p0_rmse_a": "P0, one step",
    "p1_h10_rmse_s": "P1, 10 s",
    "p1_whole_rmse_s": "P1, whole event",
    "composite": "Composite",
}
P1_HORIZONS = {"p1_h10_rmse_s": "10 s", "p1_h30_rmse_s": "30 s", "p1_whole_rmse_s": "Whole event"}

__all__ = [
    "persistence_ratio_bars", "split_inflation_bars", "rank_bump_chart", "concave_growth_curves",
    "transfer_heatmap", "horizon_curves",
]


# ----------------------------------------------------------------------------- helpers


def _require(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}; the frame has {list(frame.columns)}")


def _tick_text(value: float, _pos=None) -> str:
    """Plain numbers on log axes: 0.01, 0.5, 1, 20, 1000."""
    if not np.isfinite(value) or value <= 0:
        return ""
    return f"{value:g}" if value < 1e5 else f"{value:.0f}"


def _log_axis(axis, low: float, high: float) -> None:
    """Major ticks at the decades (and at 2 and 5 when the range is under 2.5 decades), plain
    labels, unlabelled minor ticks."""
    decades = np.log10(high / low) if high > low > 0 else 1.0
    axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,) if decades >= 2.5 else (1.0, 2.0, 5.0)))
    axis.set_major_formatter(ticker.FuncFormatter(_tick_text))
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    axis.set_minor_formatter(ticker.NullFormatter())


def _log_limits(values: np.ndarray, pad: float = 1.35, include: Sequence[float] = ()) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = np.concatenate([values[np.isfinite(values) & (values > 0)], np.asarray(include, dtype=float)])
    if values.size == 0:
        return 0.5, 2.0
    low, high = float(values.min()), float(values.max())
    if high <= low:
        low, high = low / 2, high * 2
    return low / pad, high * pad


def _families_present(families: Iterable) -> list[str]:
    present = {str(f) for f in families}
    ordered = [f for f in style.FAMILY_ORDER if f in present]
    return ordered + sorted(present - set(ordered))


def _family_patches(families: Iterable) -> list[Patch]:
    return [Patch(facecolor=style.family_color(f), label=f) for f in _families_present(families)]


def _family_lines(families: Iterable, marker: str | None = None) -> list[Line2D]:
    return [Line2D([], [], color=style.family_color(f), linewidth=1.2, marker=marker, markersize=3.2,
                   markeredgecolor="white", markeredgewidth=0.5, label=f) for f in _families_present(families)]


def _top_legend(fig: Figure, handles: Sequence, ncol: int | None = None) -> None:
    if handles:
        fig.legend(handles=list(handles), loc="outside upper center", ncol=ncol or len(handles))


def _empty(ax, text: str = "no data") -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=style.SECONDARY)


def _ranks(block: pd.DataFrame, column: str) -> pd.Series:
    """Rank 1 = lowest value; a missing value ranks last; ties are broken by the model name."""
    values = pd.to_numeric(block[column], errors="coerce").astype(float)
    order = pd.DataFrame({"missing": values.isna(), "value": values.fillna(0.0), "model": block["model"].astype(str)},
                         index=block.index).sort_values(["missing", "value", "model"], kind="stable")
    return pd.Series(np.arange(1, len(order) + 1), index=order.index).reindex(block.index)


def _spread(targets: np.ndarray, gap: float, low: float, high: float) -> np.ndarray:
    """Label positions at least ``gap`` apart, each run of crowded labels centred on the mean of
    its targets, kept inside ``[low, high]`` (1-D label placement; the order is preserved)."""
    targets = np.asarray(targets, dtype=float)
    if targets.size == 0:
        return targets
    order = np.argsort(targets, kind="stable")
    t = targets[order]
    clusters = [[i] for i in range(t.size)]
    merged = True
    while merged:  # merge neighbouring runs until the centred runs no longer overlap
        merged, out = False, [clusters[0]]
        for run in clusters[1:]:
            prev = out[-1]
            prev_top = t[prev].mean() + gap * (len(prev) - 1) / 2
            run_bottom = t[run].mean() - gap * (len(run) - 1) / 2
            if run_bottom - prev_top < gap:
                out[-1] = prev + run
                merged = True
            else:
                out.append(run)
        clusters = out
    pos = np.empty_like(t)
    for run in clusters:
        pos[run] = t[run].mean() + gap * (np.arange(len(run)) - (len(run) - 1) / 2)
    if pos[0] < low:
        pos[0] = low
        for i in range(1, pos.size):
            pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos[-1] > high:
        pos[-1] = high
        for i in range(pos.size - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    result = np.empty_like(pos)
    result[order] = pos
    return result


def _text_width_em(texts: Iterable[str]) -> float:
    """Widest ink width of ``texts`` in em of :data:`style.FONT_FAMILY`."""
    prop = FontProperties(family=style.FONT_FAMILY)
    widths = [TextPath((0, 0), t, size=1.0, prop=prop).get_extents().width for t in texts if t]
    return max(widths, default=1.0)


def _ink_on(rgba) -> str:
    """Text colour readable on the fill ``rgba`` (white on dark cells)."""
    r, g, b = mcolors.to_rgb(rgba)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else style.INK


def _grey_ramp(n: int) -> list[str]:
    if n == len(style.HORIZON_GREYS):
        return list(style.HORIZON_GREYS)
    lo, hi = np.array(mcolors.to_rgb(style.HORIZON_GREYS[0])), np.array(mcolors.to_rgb(style.HORIZON_GREYS[-1]))
    return [mcolors.to_hex(lo + (hi - lo) * t) for t in np.linspace(0.0, 1.0, max(n, 1))]


def _cell_text(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.0f}" if value >= 9.995 else f"{value:.2f}"


# ----------------------------------------------------------------------------- E1


def persistence_ratio_bars(ratios: pd.DataFrame, datasets: Sequence[str] = MAIN_DATASETS,
                           reference: str = REFERENCE) -> Figure:
    """E1 (H5.1): the P0 ratio ``RMSE_a(model) / RMSE_a(persistence)``, one panel per dataset.

    Horizontal bars from 1 on a log axis with the 95 % driver-bootstrap interval, a vertical line
    at 1 (persistence itself, ratio 1 by definition, is that line and gets no bar), models sorted
    by the ratio with the best at the top, family colours. ``ratios``: ``dataset, model, family,
    ratio, low, high`` (``results/m4/e1_ratios.csv``).
    """
    _require(ratios, ("dataset", "model", "family", "ratio", "low", "high"), "persistence_ratio_bars")
    blocks = []
    for dataset in datasets:
        block = ratios[(ratios["dataset"] == dataset) & (ratios["model"] != reference)].copy()
        for column in ("ratio", "low", "high"):
            block[column] = pd.to_numeric(block[column], errors="coerce").astype(float)
        block = block[np.isfinite(block["ratio"]) & (block["ratio"] > 0)]
        blocks.append(block.sort_values(["ratio", "model"], kind="stable").reset_index(drop=True))
    n_rows = max([len(b) for b in blocks] + [3])
    shown = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=["ratio", "low", "high", "family"])
    low, high = _log_limits(shown[["ratio", "low", "high"]].to_numpy().ravel(), include=(1.0,))
    with style.context():
        fig = style.figure(height_mm=30.0 + 4.1 * n_rows)
        axes = fig.subplots(1, max(len(datasets), 1), sharex=True, squeeze=False)[0]
        interval = None
        for i, (ax, dataset, block) in enumerate(zip(axes, datasets, blocks)):
            y = np.arange(len(block))
            ratio = block["ratio"].to_numpy()
            ax.barh(y, ratio - 1.0, left=1.0, height=0.62, color=[style.family_color(f) for f in block["family"]],
                    zorder=2)
            lo, hi = block["low"].to_numpy(), block["high"].to_numpy()
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                err = np.vstack([np.clip(ratio - lo, 0, None), np.clip(hi - ratio, 0, None)])[:, ok]
                container = ax.errorbar(ratio[ok], y[ok], xerr=err, fmt="none", ecolor=style.INK, elinewidth=0.7,
                                        capsize=1.6, capthick=0.7, zorder=3)
                interval = interval or container
            ax.axvline(1.0, color=style.INK, linewidth=0.8, zorder=4)
            ax.set_yticks(y, block["model"].astype(str).tolist())
            ax.set_ylim(n_rows - 0.5, -0.5)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_xscale("log")
            ax.grid(axis="x", which="major")
            style.panel_label(ax, i, style.dataset_label(dataset))
            if block.empty:
                _empty(ax)
        axes[0].set_xlim(low, high)
        _log_axis(axes[0].xaxis, low, high)
        fig.supxlabel("One-step acceleration RMSE relative to persistence (P0 ratio, log scale)")
        handles = _family_patches(shown["family"]) + [Line2D([], [], color=style.INK, linewidth=0.8,
                                                             label=f"{reference} = 1")]
        if interval is not None:
            interval.set_label("95 % interval")
            handles.append(interval)
        _top_legend(fig, handles)
    return fig


# ----------------------------------------------------------------------------- E2


def split_inflation_bars(paired: pd.DataFrame, datasets: Sequence[str] = LEAK_DATASETS, subset: str = "affected",
                         metrics: Mapping[str, str] | Sequence[str] = E2_METRICS) -> Figure:
    """E2 (H5.2): the paired inflation ``(clean - leaky) / clean`` in %, one panel per dataset.

    Grouped horizontal bars (one grey per metric, light to dark with the horizon) for every model,
    the models grouped by family, the 95 % interval of the paired driver bootstrap (the interval
    of ``clean - leaky`` divided by the clean error), a zero line; positive bars: the leak lowers
    the error. ``paired``: ``dataset, subset, model, family, metric, clean, inflation, low, high``
    (``results/m4/e2_paired.csv``); ``n_events`` is shown when present.
    """
    _require(paired, ("dataset", "subset", "model", "family", "metric", "clean", "inflation", "low", "high"),
             "split_inflation_bars")
    labels = dict(metrics) if isinstance(metrics, Mapping) else {m: m for m in metrics}
    data = paired[(paired["subset"] == subset) & paired["metric"].isin(labels) & paired["dataset"].isin(datasets)].copy()
    for column in ("clean", "inflation", "low", "high"):
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)
    clean = data["clean"].where(data["clean"] != 0)
    data["value"] = 100.0 * data["inflation"]
    data["err_low"], data["err_high"] = 100.0 * data["low"] / clean, 100.0 * data["high"] / clean
    family_of = data.drop_duplicates("model").set_index("model")["family"].astype(str)
    groups = [(f, [m for m in family_of.index if family_of[m] == f]) for f in _families_present(family_of)]
    # rows: a family header, then its models
    ticks, tick_labels, header_rows, row_of = [], [], [], {}
    row = 0
    for family, members in groups:
        ticks.append(row)
        tick_labels.append(family)
        header_rows.append(row)
        row += 1
        for model in members:
            row_of[model] = row
            ticks.append(row)
            tick_labels.append(model)
            row += 1
    n_rows = max(row, 1)
    greys = _grey_ramp(len(labels))
    height = 0.8 / max(len(labels), 1)
    with style.context():
        fig = style.figure(height_mm=max(60.0, 30.0 + 6.4 * n_rows))
        axes = fig.subplots(1, max(len(datasets), 1), sharey=True, squeeze=False)[0]
        for i, (ax, dataset) in enumerate(zip(axes, datasets)):
            block = data[data["dataset"] == dataset]
            values = [0.0]
            for j, metric in enumerate(labels):
                part = block[block["metric"] == metric]
                part = part[part["model"].isin(row_of)]
                y = part["model"].map(row_of).to_numpy(dtype=float) + (j - (len(labels) - 1) / 2) * height
                ax.barh(y, part["value"].to_numpy(), height=height, color=greys[j], edgecolor="white",
                        linewidth=0.5, zorder=2)
                lo, hi = part["err_low"].to_numpy(), part["err_high"].to_numpy()
                ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(part["value"].to_numpy())
                if ok.any():
                    v = part["value"].to_numpy()[ok]
                    ax.errorbar(v, y[ok], xerr=np.vstack([np.clip(v - lo[ok], 0, None), np.clip(hi[ok] - v, 0, None)]),
                                fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
                values += [*part["value"].dropna(), *lo[np.isfinite(lo)], *hi[np.isfinite(hi)]]
            ax.axvline(0.0, color=style.INK, linewidth=0.7, zorder=4)
            for header in header_rows[1:]:
                ax.axhline(header - 0.5, color=style.GRID, linewidth=0.6, zorder=1)
            span = max(values) - min(values) or 1.0
            ax.set_xlim(min(values) - 0.06 * span, max(values) + 0.06 * span)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            style.panel_label(ax, i, style.dataset_label(dataset))
            if "n_events" in block.columns and len(block):
                n_events = pd.to_numeric(block["n_events"], errors="coerce").max()
                if np.isfinite(n_events):
                    style.note(ax, f"{subset} probe events: {int(n_events):,}")
            if block.empty:
                _empty(ax)
        axes[0].set_yticks(ticks, tick_labels)
        axes[0].set_ylim(n_rows - 0.5, -0.6)
        for k, text in enumerate(axes[0].get_yticklabels()):
            if ticks[k] in header_rows:
                text.set_fontstyle("italic")
                text.set_color(style.SECONDARY)
        fig.supxlabel("Error reduction with the driver leak (% of the clean error; whiskers: 95 % interval)")
        _top_legend(fig, [Patch(facecolor=g, label=t) for g, t in zip(greys, labels.values())])
    return fig


# ----------------------------------------------------------------------------- E3


def rank_bump_chart(rankings: pd.DataFrame, datasets: Sequence[str] = MAIN_DATASETS,
                    columns: Mapping[str, str] | Sequence[str] = E3_COLUMNS) -> Figure:
    """E3 (H5.3): the rank of every model under P0, P1 10 s, P1 whole event and the composite,
    one panel per dataset (stacked, so that the names at both ends leave room for the lines), as a
    bump chart (connected lines, family colours), model names at both ends (left: P0 order,
    right: composite order). Rank 1 is the lowest value; a missing value ranks last; ties are
    broken by the model name. ``rankings``: ``dataset, model, family`` and the value columns
    (``results/m4/e3_rankings.csv``).
    """
    labels = dict(columns) if isinstance(columns, Mapping) else {c: c for c in columns}
    _require(rankings, ("dataset", "model", "family", *labels), "rank_bump_chart")
    blocks = [rankings[rankings["dataset"] == d].drop_duplicates("model").reset_index(drop=True) for d in datasets]
    n_max = max([len(b) for b in blocks] + [2])
    x = np.arange(len(labels))
    with style.context():
        fig = style.figure(height_mm=10.0 + max(len(datasets), 1) * (14.0 + 3.5 * n_max))
        axes = fig.subplots(max(len(datasets), 1), 1, squeeze=False)[:, 0]
        for i, (ax, dataset, block) in enumerate(zip(axes, datasets, blocks)):
            right = ax.twinx()
            n = len(block)
            if n:
                ranks = pd.DataFrame({c: _ranks(block, c) for c in labels})
                for k in range(n):  # the networks below, so the smaller families stay visible
                    family = block.loc[k, "family"]
                    ax.plot(x, ranks.loc[k].to_numpy(), color=style.family_color(family), linewidth=1.1, alpha=0.9,
                            marker="o", markersize=3.2, markeredgecolor="white", markeredgewidth=0.5,
                            zorder=3 if family != "network" else 2, solid_capstyle="round")
                first, last = list(labels)[0], list(labels)[-1]
                by_first = block.assign(r=ranks[first]).sort_values("r")
                by_last = block.assign(r=ranks[last]).sort_values("r")
                ax.set_yticks(by_first["r"].to_numpy(), by_first["model"].astype(str).tolist())
                right.set_yticks(by_last["r"].to_numpy(), by_last["model"].astype(str).tolist())
            else:
                _empty(ax)
                ax.set_yticks([])
                right.set_yticks([])
            for a in (ax, right):
                a.set_ylim(n_max + 0.5, 0.5)
                a.tick_params(axis="y", length=0, pad=2.0)
                for spine in a.spines.values():
                    spine.set_visible(False)
            ax.set_xticks(x, list(labels.values()))
            ax.tick_params(axis="x", length=0, pad=3.0)
            ax.set_xlim(x[0] - 0.04, x[-1] + 0.04)
            style.panel_label(ax, i, style.dataset_label(dataset))
            style.note(ax, "rank 1 (lowest error) at the top")
        _top_legend(fig, _family_lines(pd.concat(blocks)["family"] if blocks else [], marker="o"))
    return fig


# ----------------------------------------------------------------------------- P2


def concave_growth_curves(vehicles: pd.DataFrame, profiles: pd.DataFrame, models: Sequence[str] = P2_MODELS,
                          dataset: str | None = None, families: Mapping[str, str] | None = None,
                          log_y: bool = True) -> Figure:
    """P2: the speed standard deviation against the vehicle index (0 = leader) along the simulated
    platoon, one panel per model (2 x 2 for four models), one thin line per in-range OpenACC
    profile, the recorded followers of the same profiles as black markers.

    ``vehicles``: ``model, profile, index, std_v, std_v_rec`` (``p2.vehicles.parquet`` of the runs
    with the model added); ``profiles``: ``model, profile, source, in_range``
    (``p2.profiles.parquet``). The family comes from a ``family`` column of either frame or from
    ``families``. A profile is shown when it is in range and its source is an OpenACC file.
    ``log_y`` (default): one log axis shared by the panels, since unstable platoons reach speed
    deviations two decades above the recorded ones; otherwise a linear axis per panel up to the
    recorded maximum or the 95th percentile of the simulated values, with the profiles that leave
    it counted in a note. ``dataset`` names the training data in the legend.
    """
    _require(vehicles, ("model", "profile", "index", "std_v", "std_v_rec"), "concave_growth_curves (vehicles)")
    _require(profiles, ("model", "profile", "source", "in_range"), "concave_growth_curves (profiles)")
    family_of = dict(families or {})
    for frame in (profiles, vehicles):
        if "family" in frame.columns:
            for model, family in frame.drop_duplicates("model")[["model", "family"]].itertuples(index=False):
                family_of.setdefault(model, family)
    shown = profiles[profiles["in_range"].astype(bool)
                     & profiles["source"].astype(str).str.lower().str.startswith("openacc")]
    selected = {m: set(shown.loc[shown["model"] == m, "profile"].astype(str)) for m in models}
    n_cols = 2 if len(models) > 1 else 1
    n_rows = int(np.ceil(max(len(models), 1) / n_cols))
    x_max = float(pd.to_numeric(vehicles["index"], errors="coerce").max()) if len(vehicles) else 1.0
    blocks = []
    for model in models:
        block = vehicles[(vehicles["model"] == model) & vehicles["profile"].astype(str).isin(selected[model])].copy()
        block["std_v"] = pd.to_numeric(block["std_v"], errors="coerce").astype(float)
        block["std_v_rec"] = pd.to_numeric(block["std_v_rec"], errors="coerce").astype(float)
        blocks.append(block.sort_values(["profile", "index"], kind="stable"))
    with style.context():
        fig = style.figure(height_mm=22.0 + 50.0 * n_rows)
        axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=log_y, squeeze=False).ravel()
        for ax in axes[len(models):]:
            ax.set_visible(False)
        for i, (ax, model, block) in enumerate(zip(axes, models, blocks)):
            color = style.family_color(family_of.get(model))
            for _, line in block.groupby("profile", sort=True):
                ax.plot(line["index"].to_numpy(dtype=float), line["std_v"].to_numpy(), color=color, linewidth=0.7,
                        alpha=0.75, zorder=2)
            recorded = block[np.isfinite(block["std_v_rec"])]
            ax.plot(recorded["index"].to_numpy(dtype=float), recorded["std_v_rec"].to_numpy(), linestyle="none",
                    marker="o", markersize=2.6, markerfacecolor=style.INK, markeredgecolor="white",
                    markeredgewidth=0.3, zorder=4)
            simulated = block["std_v"].to_numpy()
            simulated = simulated[np.isfinite(simulated)]
            rec = recorded["std_v_rec"].to_numpy()
            if not log_y and (simulated.size or rec.size):
                top = max([float(np.quantile(simulated, 0.95))] if simulated.size else [0.0])
                top = max(top, float(rec.max()) if rec.size else 0.0)
                limit = float(ticker.MaxNLocator(nbins=5).tick_values(0.0, 1.08 * top)[-1]) if top > 0 else 1.0
                ax.set_ylim(0.0, limit)
                above = block.loc[block["std_v"] > limit, "profile"].nunique()
                if above:
                    style.note(ax, f"{above} of {block['profile'].nunique()} profiles leave the axis "
                                   f"(max {simulated.max():.0f} m/s)")
            if block.empty:
                _empty(ax, "no in-range OpenACC profile")
            ax.set_xlim(0.0, max(x_max, 1.0))
            if i + n_cols >= len(models):
                ax.xaxis.set_tick_params(labelbottom=True)
            style.panel_label(ax, i, f"{model} ({len(selected[model])} profiles)")
        fig.supxlabel("Vehicle index in the platoon (0 = leader)")
        fig.supylabel("Speed standard deviation (m/s" + (", log scale)" if log_y else ")"))
        if log_y:
            values = np.concatenate([np.concatenate([b["std_v"].to_numpy(), b["std_v_rec"].to_numpy()]) for b in blocks]
                                    or [np.array([])])
            low, high = _log_limits(values, pad=1.3)
            axes[0].set_yscale("log")
            axes[0].set_ylim(low, high)
            _log_axis(axes[0].yaxis, low, high)
            for ax in axes[: len(models)]:
                ax.grid(axis="y", which="major")
        handles = [Line2D([], [], color=style.family_color(f), linewidth=1.0, label=f"simulated, {f} model")
                   for f in _families_present(family_of.get(m) for m in models if m in family_of)]
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=3.2, markerfacecolor=style.INK,
                              markeredgecolor="white", markeredgewidth=0.3, label="recorded follower"))
        title = f"Models trained on {style.dataset_label(dataset)}; leader: in-range OpenACC profiles" if dataset \
            else "Leader: in-range OpenACC profiles"
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), title=title)
    return fig


# ----------------------------------------------------------------------------- E4


def transfer_heatmap(transfer: pd.DataFrame, models: Sequence[str] = SUBSET, horizon: str = "whole",
                     datasets: Sequence[str] | None = None, cap: float = 10.0) -> Figure:
    """E4 (P5): relative degradation ``rmse_s(A -> B) / rmse_s(B -> B)``, one panel per model of
    ``models`` with rows in the table (a 2 x 4 grid for eight), rows trained on, columns tested on,
    a diverging colour scale centred at 1 on a log scale capped at ``1 / cap`` and ``cap``, the
    value printed in every cell (``n/a`` where a pair is missing). ``transfer``: ``model,
    trained_on, tested_on, horizon, ratio`` (``results/m4/e4_transfer.csv``).
    """
    _require(transfer, ("model", "trained_on", "tested_on", "horizon", "ratio"), "transfer_heatmap")
    data = transfer[(transfer["horizon"] == horizon) & transfer["model"].isin(models)].copy()
    data["ratio"] = pd.to_numeric(data["ratio"], errors="coerce").astype(float)
    if datasets is None:
        present = set(data["trained_on"].astype(str)) | set(data["tested_on"].astype(str))
        datasets = [d for d in style.DATASET_ORDER if d in present] + sorted(present - set(style.DATASET_ORDER))
    datasets = list(datasets)
    models = [m for m in models if m in set(data["model"])]
    n_cols = min(4, max(len(models), 1))
    n_rows = int(np.ceil(max(len(models), 1) / n_cols))
    norm = mcolors.LogNorm(vmin=1.0 / cap, vmax=cap)
    cmap = colormaps[style.DIVERGING_CMAP].with_extremes(bad=style.MISSING)
    names = [style.dataset_label(d, short=True) for d in datasets]
    # square panels: the height follows from the width left after the row labels
    panel_mm = (style.WIDTH_MM - 28.0 - 4.0 * (n_cols - 1)) / n_cols
    with style.context():
        fig = style.figure(height_mm=40.0 + n_rows * (panel_mm + 6.0))
        axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
        image = None
        matrices = []
        for k, ax in enumerate(axes.ravel()):
            if k >= len(models):
                ax.set_visible(False)
                continue
            block = data[data["model"] == models[k]]
            matrix = (block.pivot_table(index="trained_on", columns="tested_on", values="ratio", aggfunc="first")
                      .reindex(index=datasets, columns=datasets).to_numpy(dtype=float)) if len(block) else \
                np.full((len(datasets), len(datasets)), np.nan)
            matrices.append((ax, matrix))
            shown = np.ma.masked_invalid(np.clip(matrix, 1.0 / cap, cap))
            image = ax.imshow(shown, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
            ax.set_xticks(np.arange(len(datasets)), names, rotation=90)
            ax.set_yticks(np.arange(len(datasets)), names)
            ax.tick_params(length=0, pad=2.0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            style.panel_label(ax, k, models[k])
        if image is not None:
            bar = fig.colorbar(image, ax=[ax for ax in axes.ravel() if ax.get_visible()], location="top", shrink=0.55,
                               aspect=45, extend="both", pad=0.02)
            ticks = [t for t in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) if 1.0 / cap <= t <= cap]
            bar.set_ticks(ticks, labels=[("\u2264" if t == 1.0 / cap else "\u2265" if t == cap else "") + _tick_text(t)
                                         for t in ticks])
            bar.ax.xaxis.set_minor_locator(ticker.NullLocator())
            bar.ax.tick_params(length=2.0, width=0.5)
            bar.outline.set_linewidth(0.4)
            what = "Whole-event" if horizon == "whole" else horizon.lstrip("h") + " s"
            bar.set_label(f"{what} spacing RMSE: transferred model / target's own model (1 = no loss)")
        fig.supxlabel("Tested on")
        fig.supylabel("Trained on")
        # cell values: the largest size (at most FONT_SIZE - 1) that fits the cells after layout
        fig.draw_without_rendering()
        texts = [_cell_text(v) for _, m in matrices for v in m.ravel()]
        if matrices:
            box = matrices[0][0].get_window_extent()
            cell = min(box.width, box.height) / max(len(datasets), 1) * 72.0 / fig.dpi
            size = float(np.clip(0.8 * cell / _text_width_em(texts), style.MIN_FONT_SIZE, style.FONT_SIZE - 1))
            for ax, matrix in matrices:
                for r in range(matrix.shape[0]):
                    for c in range(matrix.shape[1]):
                        value = matrix[r, c]
                        fill = cmap(norm(np.clip(value, 1.0 / cap, cap))) if np.isfinite(value) else style.MISSING
                        ax.text(c, r, _cell_text(value), ha="center", va="center", fontsize=size, color=_ink_on(fill))
    return fig


# ----------------------------------------------------------------------------- P1


def horizon_curves(rankings: pd.DataFrame, datasets: Sequence[str] = MAIN_DATASETS,
                   horizons: Mapping[str, str] = P1_HORIZONS, n_labels: int = 3) -> Figure:
    """P1: the spacing RMSE at 10 s, 30 s and the whole event per model (one line each, family
    colours, log axis), one panel per dataset; a horizon without any value in a dataset is left
    out of its panel (and named in a note). The ``n_labels`` lowest and highest whole-event values
    are labelled. ``rankings``: ``dataset, model, family`` and the ``horizons`` columns
    (``results/m4/e3_rankings.csv`` or the ``p1.json`` of the runs).
    """
    _require(rankings, ("dataset", "model", "family", *horizons), "horizon_curves")
    with style.context():
        fig = style.figure(height_mm=105.0)
        axes = fig.subplots(1, max(len(datasets), 1), squeeze=False)[0]
        panels = []
        for i, (ax, dataset) in enumerate(zip(axes, datasets)):
            block = rankings[rankings["dataset"] == dataset].drop_duplicates("model").reset_index(drop=True)
            values = block[list(horizons)].apply(pd.to_numeric, errors="coerce").astype(float)
            values = values.where(values > 0)
            used = [h for h in horizons if values[h].notna().any()]
            missing = [horizons[h] for h in horizons if h not in used]
            x = np.arange(len(used))
            for k in sorted(range(len(block)), key=lambda k: block.loc[k, "family"] != "network"):
                family = block.loc[k, "family"]
                row = values.loc[k, used].to_numpy(dtype=float)
                ok = np.isfinite(row)
                ax.plot(x[ok], row[ok], color=style.family_color(family), linewidth=0.9, alpha=0.85, marker="o",
                        markersize=2.6, markeredgecolor="white", markeredgewidth=0.4,
                        zorder=3 if family != "network" else 2)
            ax.set_yscale("log")
            low, high = _log_limits(values[used].to_numpy().ravel(), pad=1.25)
            ax.set_ylim(low, high)
            _log_axis(ax.yaxis, low, high)
            ax.grid(axis="y", which="major")
            ax.set_xticks(x, [horizons[h] for h in used])
            ax.set_xlim(-0.25, max(len(used) - 1, 0) + 0.85)
            ax.set_xlabel("Closed-loop horizon")
            ax.set_ylabel("Spacing RMSE (m, log scale)")
            style.panel_label(ax, i, style.dataset_label(dataset))
            if missing and len(block):
                style.note(ax, "no " + " / ".join(missing) + " windows")
            if block.empty or not used:
                _empty(ax)
            panels.append((ax, block, values, used, low, high))
        _top_legend(fig, _family_lines(rankings[rankings["dataset"].isin(datasets)]["family"], marker="o"))
        # labels of the extremes at the last horizon, spread so they never overlap
        fig.draw_without_rendering()
        for ax, block, values, used, low, high in panels:
            if not used or block.empty or n_labels <= 0:
                continue
            last = values[used[-1]]
            finite = last.dropna().sort_values()
            chosen = list(dict.fromkeys([*finite.index[:n_labels], *finite.index[-n_labels:]]))
            if not chosen:
                continue
            span = np.log10(high) - np.log10(low)
            height_pt = ax.get_window_extent().height * 72.0 / fig.dpi
            gap = (style.FONT_SIZE - 1) * 1.15 / max(height_pt, 1.0) * span
            target = np.log10(last.loc[chosen].to_numpy(dtype=float))
            placed = _spread(target, gap, np.log10(low) + gap / 2, np.log10(high) - gap / 2)
            x_last = len(used) - 1
            for k, t, p in zip(chosen, target, placed):
                ax.annotate(str(block.loc[k, "model"]), xy=(x_last, 10 ** t), xytext=(x_last + 0.14, 10 ** p),
                            textcoords="data", va="center", ha="left", fontsize=style.FONT_SIZE - 1,
                            color=style.INK, annotation_clip=False,
                            arrowprops={"arrowstyle": "-", "color": style.SECONDARY, "linewidth": 0.5,
                                        "shrinkA": 1.5, "shrinkB": 2.5} if abs(p - t) > gap / 4 else None)
    return fig
