"""Figures of the report (follownet2/figures, scripts/make_report.py; docs/leaderboard.md, sections 4-6)
on fabricated frames and a fabricated results root: every figure function returns a 160 mm wide
figure that ``style.save`` writes as PNG and PDF, the figure-specific rules hold (persistence is the
line at 1, ranks with ties and gaps, in-range OpenACC profiles only, capped colours with the true
values printed, missing horizons left out), and the loaders of the script read run directories."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from follownet2.figures import plots, style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FAMILIES = {"persistence": "baseline", "constant_speed": "baseline", "idm": "mechanistic", "gipps": "mechanistic",
            "mlp": "network", "lstm": "network"}
#: P0 ratios on the first dataset (persistence is 1 by definition; mlp beats it)
RATIOS = {"persistence": 1.0, "constant_speed": 8.0, "idm": 25.0, "gipps": 54.0, "mlp": 0.6, "lstm": 1.3}
PNG_WIDTH = style.WIDTH_MM / 25.4 * style.DPI  # 1259.8 px; matplotlib truncates to whole pixels


# ----------------------------------------------------------------------------- fabricated frames


def e1_frame() -> pd.DataFrame:
    rows = []
    for dataset, scale in (("ngsim", 1.0), ("follownet_highd", 3.0)):
        for model, family in FAMILIES.items():
            ratio = RATIOS[model] if model == "persistence" else RATIOS[model] * scale
            rows.append({"dataset": dataset, "model": model, "family": family, "rmse_a": 0.1 * ratio, "ratio": ratio,
                         "low": 0.95 * ratio, "high": 1.05 * ratio, "n_drivers": 40})
    frame = pd.DataFrame(rows)
    frame.loc[(frame["dataset"] == "follownet_highd") & (frame["model"] == "lstm"), ["low", "high"]] = np.nan
    return frame


E2_MODELS = {"idm": "mechanistic", "mlp": "network", "lstm": "network"}


def e2_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("ngsim", "openacc"):
        for subset in ("all", "affected"):
            for k, metric in enumerate(plots.E2_METRICS):
                for j, (model, family) in enumerate(E2_MODELS.items()):
                    clean = 2.0 + j + k
                    leaky = clean * (1.0 - 0.01 * (j - 1) * (k + 1))
                    diff = clean - leaky
                    rows.append({"dataset": dataset, "subset": subset, "model": model, "family": family, "metric": metric,
                                 "clean": clean, "leaky": leaky, "inflation": diff / clean, "mean_difference": diff,
                                 "low": diff - 0.02, "high": diff + 0.03, "p_bootstrap": 0.01, "n_drivers": 30,
                                 "n_events": 120 if subset == "affected" else 300, "n_folds": 5})
    return pd.DataFrame(rows)


def e3_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("ngsim", "follownet_highd"):
        for k, (model, family) in enumerate(FAMILIES.items()):
            rows.append({"dataset": dataset, "model": model, "family": family, "p0_rmse_a": 0.1 + 0.05 * k,
                         "p1_h10_rmse_s": 3.0 + (5 - k) * 0.4, "p1_h30_rmse_s": 5.0 + (5 - k) * 0.7,
                         "p1_whole_rmse_s": 6.0 + 9.0 * k, "composite": [0.5, 0.5, 0.1, 0.2, 0.7, 0.9][k]})
    frame = pd.DataFrame(rows)
    frame.loc[frame["dataset"] == "follownet_highd", "p1_h30_rmse_s"] = np.nan  # no 30 s windows
    frame.loc[(frame["dataset"] == "follownet_highd") & (frame["model"] == "gipps"), "composite"] = np.nan
    return frame


P2_PROFILES = {  # profile -> (source, in_range)
    "dyn1": ("openacc/ZalaZone/dyn1.csv", True),
    "dyn2": ("openacc/ZalaZone/dyn2.csv", True),
    "asta": ("openacc/AstaZero/asta.csv", False),
    "braking_pulse": ("synthetic", True),
}
P2_FAMILY = {"idm": "mechanistic", "ddpg": "network", "mlp_loop": "network", "lstm": "network"}


def p2_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    vehicles, profiles = [], []
    index = np.arange(21)
    for m, (model, family) in enumerate(P2_FAMILY.items()):
        for p, (profile, (source, in_range)) in enumerate(P2_PROFILES.items()):
            std = (0.5 + 0.1 * p) * np.exp((0.05 * m - 0.06) * index)
            if model == "lstm":
                std[12] = 150.0  # an unstable platoon
            rec = np.full(index.size, np.nan)
            if source.startswith("openacc"):
                rec[1:5] = 0.6 + 0.2 * np.arange(4)
            vehicles.append(pd.DataFrame({"profile": profile, "index": index, "std_v": std, "std_v_rec": rec,
                                          "model": model, "family": family}))
            for n in (10, 20):
                profiles.append({"profile": profile, "source": source, "n_vehicles": n, "in_range": in_range,
                                 "model": model, "family": family})
    return pd.concat(vehicles, ignore_index=True), pd.DataFrame(profiles)


E4_DATASETS = ("ngsim", "openacc", "tgsim")
E4_MODELS = ("idm", "mlp", "lstm")


def e4_frame() -> pd.DataFrame:
    rows = []
    for m, model in enumerate((*E4_MODELS, "persistence")):
        for a, trained in enumerate(E4_DATASETS):
            for b, tested in enumerate(E4_DATASETS):
                for horizon in ("h10", "whole"):
                    ratio = 1.0 if (a == b or model == "persistence") else 0.8 + 0.3 * (a + 2 * b + m)
                    rows.append({"model": model, "trained_on": trained, "tested_on": tested, "horizon": horizon,
                                 "rmse_s": 3.0 * ratio, "rmse_s_own": 3.0, "ratio": ratio, "low": ratio, "high": ratio,
                                 "n_drivers": 10, "note": None})
    frame = pd.DataFrame(rows)
    whole = frame["horizon"] == "whole"
    frame.loc[whole & (frame["model"] == "lstm") & (frame["trained_on"] == "ngsim") & (frame["tested_on"] == "openacc"),
              "ratio"] = 42.0
    frame.loc[whole & (frame["model"] == "mlp") & (frame["trained_on"] == "openacc") & (frame["tested_on"] == "tgsim"),
              "ratio"] = 0.02
    frame.loc[whole & (frame["model"] == "idm") & (frame["trained_on"] == "tgsim") & (frame["tested_on"] == "ngsim"),
              "ratio"] = np.nan
    return frame


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def width_mm(fig: Figure) -> float:
    return float(fig.get_size_inches()[0] * 25.4)


def panels(fig: Figure) -> list:
    """Visible data axes (no colorbars, no twins)."""
    return [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>" and ax.get_title(loc="left")]


# ----------------------------------------------------------------------------- style


def test_palettes_are_complete_and_distinct():
    assert set(style.FAMILY_COLORS) == set(style.FAMILY_ORDER)
    assert len(set(style.FAMILY_COLORS.values())) == len(style.FAMILY_COLORS)
    assert set(style.DATASET_COLORS) == set(style.DATASET_ORDER) == set(style.DATASET_LABELS) == set(style.DATASET_SHORT)
    assert len(set(style.DATASET_COLORS.values())) == 6
    luminance = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in map(mcolors.to_rgb, style.HORIZON_GREYS)]
    assert luminance == sorted(luminance, reverse=True)  # ordinal ramp, light to dark
    assert style.family_color("unknown") == style.OTHER_COLOR
    assert style.dataset_label("follownet_highd", short=True) == "FN-highD" and style.dataset_label("new") == "new"


def test_constants_match_the_analysis():
    from follownet2.experiments import analysis as an

    assert plots.SUBSET == an.SUBSET
    assert plots.MAIN_DATASETS == an.MAIN_DATASETS
    assert plots.LEAK_DATASETS == an.LEAK_DATASETS
    assert plots.REFERENCE == an.REFERENCE
    assert set(style.DATASET_ORDER) == set(an.TRANSFER_DATASETS)


def test_save_writes_png_and_pdf(tmp_path):
    with style.context():
        fig = style.figure(height_mm=50.0)
        ax = fig.subplots()
        ax.plot([0, 1], [0, 1])
        style.panel_label(ax, 1, "label")
    assert ax.get_title(loc="left") == "(b) label"
    paths = style.save(fig, tmp_path / "sub" / "figure.v1")
    assert [p.name for p in paths] == ["figure.v1.png", "figure.v1.pdf"]
    assert abs(png_size(paths[0])[0] - PNG_WIDTH) < 1
    assert paths[1].read_bytes()[:5] == b"%PDF-"
    assert b"CreationDate" not in paths[1].read_bytes()  # the same bytes on every run


# ----------------------------------------------------------------------------- every figure


def build_all() -> dict[str, Figure]:
    vehicles, profiles = p2_frames()
    return {
        "persistence_ratio_bars": plots.persistence_ratio_bars(e1_frame()),
        "split_inflation_bars": plots.split_inflation_bars(e2_frame()),
        "rank_bump_chart": plots.rank_bump_chart(e3_frame()),
        "concave_growth_curves": plots.concave_growth_curves(vehicles, profiles, dataset="ngsim"),
        "transfer_heatmap": plots.transfer_heatmap(e4_frame(), models=E4_MODELS),
        "horizon_curves": plots.horizon_curves(e3_frame()),
    }


@pytest.fixture(scope="module")
def figures() -> dict[str, Figure]:
    return build_all()


@pytest.mark.parametrize("name", ["persistence_ratio_bars", "split_inflation_bars", "rank_bump_chart",
                                  "concave_growth_curves", "transfer_heatmap", "horizon_curves"])
def test_figure_is_built_and_saved(figures, name, tmp_path):
    fig = figures[name]
    assert isinstance(fig, Figure)
    assert width_mm(fig) == pytest.approx(style.WIDTH_MM)
    assert fig.get_layout_engine() is not None  # constrained layout
    labels = [ax.get_title(loc="left") for ax in panels(fig)]
    assert labels and all(label.startswith(f"({chr(ord('a') + k)})") for k, label in enumerate(labels))
    png, pdf = style.save(fig, tmp_path / name)
    assert png.stat().st_size > 10_000 and pdf.stat().st_size > 5_000
    assert abs(png_size(png)[0] - PNG_WIDTH) < 1


def test_every_function_is_exported():
    import follownet2.figures as figures_package

    for name in plots.__all__:
        assert callable(getattr(figures_package, name))


# ----------------------------------------------------------------------------- figure rules


def test_ratio_bars_sorted_with_persistence_as_the_line(figures):
    fig = figures["persistence_ratio_bars"]
    first, second = panels(fig)
    expected = [m for m, _ in sorted(((m, r) for m, r in RATIOS.items() if m != "persistence"), key=lambda x: x[1])]
    assert [t.get_text() for t in first.get_yticklabels()] == expected
    assert len(first.patches) == len(expected)  # no bar for persistence
    widths = {t.get_text(): p.get_width() for t, p in zip(first.get_yticklabels(), first.patches)}
    assert widths["mlp"] == pytest.approx(0.6 - 1.0)  # bars start at 1, below 1 to the left
    assert first.get_xscale() == "log" and any(np.allclose(line.get_xdata(), 1.0) for line in first.lines)
    colors = {t.get_text(): mcolors.to_hex(p.get_facecolor()) for t, p in zip(first.get_yticklabels(), first.patches)}
    assert colors["idm"] == style.FAMILY_COLORS["mechanistic"].lower()
    assert colors["constant_speed"] == style.FAMILY_COLORS["baseline"].lower()
    assert "persistence = 1" in [t.get_text() for t in fig.legends[0].get_texts()]
    assert len(second.patches) == len(expected)


def test_inflation_bars_in_percent_grouped_by_family(figures):
    fig = figures["split_inflation_bars"]
    first, second = panels(fig)
    assert [t.get_text() for t in first.get_yticklabels()] == ["mechanistic", "idm", "network", "mlp", "lstm"]
    frame = e2_frame()
    expected = frame[(frame["dataset"] == "ngsim") & (frame["subset"] == "affected")]
    widths = sorted(round(p.get_width(), 9) for p in first.patches)
    assert widths == sorted(round(v, 9) for v in 100.0 * expected["inflation"])
    assert first.get_title(loc="right") == "affected probe events: 120"
    assert [t.get_text() for t in fig.legends[0].get_texts()] == list(plots.E2_METRICS.values())


def test_ranks_break_ties_by_name_and_put_missing_last():
    block = pd.DataFrame({"model": ["b", "a", "c", "d"], "v": [0.5, 0.5, np.nan, 0.1]})
    assert plots._ranks(block, "v").tolist() == [3, 2, 4, 1]


def test_bump_chart_names_both_ends(figures):
    fig = figures["rank_bump_chart"]
    ngsim = [ax for ax in fig.axes if ax.get_title(loc="left") == "(a) NGSIM"][0]
    twin = [ax for ax in fig.axes if ax is not ngsim and ax.bbox.bounds == ngsim.bbox.bounds][0]
    frame = e3_frame()
    block = frame[frame["dataset"] == "ngsim"]
    assert [t.get_text() for t in ngsim.get_yticklabels()] == block.sort_values("p0_rmse_a")["model"].tolist()
    # composite ties (persistence, constant_speed at 0.5) broken by the name
    assert [t.get_text() for t in twin.get_yticklabels()] == ["idm", "gipps", "constant_speed", "persistence", "mlp", "lstm"]
    assert len(ngsim.lines) == len(block)
    highd = [ax for ax in fig.axes if ax.get_title(loc="left") == "(b) FollowNet highD"][0]
    lines = {tuple(line.get_ydata()) for line in highd.lines}
    assert any(y[-1] == len(block) for y in lines)  # gipps without a composite ranks last


def test_concave_growth_shows_in_range_openacc_profiles_only(figures):
    fig = figures["concave_growth_curves"]
    axes = panels(fig)
    assert [ax.get_title(loc="left") for ax in axes] == [f"({c}) {m} (2 profiles)" for c, m in zip("abcd", P2_FAMILY)]
    for ax in axes:
        simulated = [line for line in ax.lines if line.get_linestyle() != "None"]
        recorded = [line for line in ax.lines if line.get_linestyle() == "None"]
        assert len(simulated) == 2  # dyn1 and dyn2; neither AstaZero (out of range) nor the pulse
        assert len(recorded[0].get_xdata()) == 2 * 4
        assert ax.get_yscale() == "log"
    assert axes[-1].get_ylim()[1] > 150.0  # the unstable platoon stays on the shared axis
    colors = {mcolors.to_hex(line.get_color()) for line in axes[0].lines if line.get_linestyle() != "None"}
    assert colors == {style.FAMILY_COLORS["mechanistic"].lower()}
    legend = fig.legends[0]
    assert legend.get_title().get_text().startswith("Models trained on NGSIM")


def test_concave_growth_linear_axis_counts_profiles_off_the_axis():
    vehicles, profiles = p2_frames()
    fig = plots.concave_growth_curves(vehicles, profiles, log_y=False)
    lstm = panels(fig)[-1]
    assert lstm.get_yscale() == "linear" and lstm.get_ylim()[1] < 150.0
    assert lstm.get_title(loc="right").startswith("2 of 2 profiles leave the axis")


def test_transfer_heatmap_caps_colours_and_prints_values(figures):
    fig = figures["transfer_heatmap"]
    axes = panels(fig)
    assert [ax.get_title(loc="left") for ax in axes] == [f"({c}) {m}" for c, m in zip("abc", E4_MODELS)]
    lstm = axes[2]
    texts = {(round(t.get_position()[1]), round(t.get_position()[0])): t.get_text() for t in lstm.texts}
    assert len(texts) == len(E4_DATASETS) ** 2
    assert texts[(0, 1)] == "42"  # capped colour, true value printed
    image = lstm.images[0]
    assert image.norm.vmin == pytest.approx(0.1) and image.norm.vmax == pytest.approx(10.0)
    assert image.get_array()[0, 1] == pytest.approx(10.0)
    mlp_texts = {(round(t.get_position()[1]), round(t.get_position()[0])): t.get_text() for t in axes[1].texts}
    assert mlp_texts[(1, 2)] == "0.02"
    idm_texts = {(round(t.get_position()[1]), round(t.get_position()[0])): t.get_text() for t in axes[0].texts}
    assert idm_texts[(2, 0)] == "n/a" and idm_texts[(0, 0)] == "1.00"
    colorbar = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(colorbar) == 1
    sizes = {t.get_fontsize() for ax in axes for t in ax.texts}
    assert len(sizes) == 1 and style.MIN_FONT_SIZE <= sizes.pop() <= style.FONT_SIZE


def test_transfer_heatmap_eight_models_in_a_two_by_four_grid():
    frame = e4_frame()
    copies = [frame[frame["model"] == "idm"].assign(model=m) for m in plots.SUBSET]
    fig = plots.transfer_heatmap(pd.concat(copies, ignore_index=True))
    axes = panels(fig)
    assert len(axes) == 8
    rows = sorted({round(ax.get_position().y0, 3) for ax in axes})
    cols = sorted({round(ax.get_position().x0, 3) for ax in axes})
    assert len(rows) == 2 and len(cols) == 4


def test_horizon_curves_leave_out_missing_horizons(figures):
    fig = figures["horizon_curves"]
    ngsim, highd = panels(fig)
    assert [t.get_text() for t in ngsim.get_xticklabels()] == ["10 s", "30 s", "Whole event"]
    assert [t.get_text() for t in highd.get_xticklabels()] == ["10 s", "Whole event"]
    assert highd.get_title(loc="right") == "no 30 s windows" and ngsim.get_title(loc="right") == ""
    assert ngsim.get_yscale() == "log" and len(ngsim.lines) == len(FAMILIES)
    labelled = {t.get_text() for t in ngsim.texts}
    assert labelled == set(FAMILIES)  # 3 lowest + 3 highest of six models


def test_spread_keeps_the_gap_inside_the_bounds():
    targets = np.array([0.50, 0.51, 0.52, 0.90, 0.05])
    placed = plots._spread(targets, 0.05, 0.0, 1.0)
    order = np.argsort(targets)
    assert np.all(np.diff(placed[order]) >= 0.05 - 1e-12)
    assert placed.min() >= 0.0 and placed.max() <= 1.0
    assert placed[:3].mean() == pytest.approx(targets[:3].mean())  # a crowded run stays centred
    assert placed[3] == pytest.approx(0.90) and placed[4] == pytest.approx(0.05)
    top = plots._spread(np.array([0.99, 0.995, 1.0]), 0.05, 0.0, 1.0)
    assert top.max() == pytest.approx(1.0) and np.all(np.diff(np.sort(top)) >= 0.05 - 1e-12)


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="ratio"):
        plots.persistence_ratio_bars(e1_frame().drop(columns="ratio"))


# ----------------------------------------------------------------------------- the script


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("make_report", REPO / "scripts" / "make_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fabricate_root(base: Path, with_e3: bool = True) -> tuple[Path, Path]:
    """``m4`` with the four CSVs and ``models`` with driver-split runs on ngsim and follownet_highd
    (P2 tables on ngsim for the four models except lstm, p1.json everywhere)."""
    m4, root = base / "m4", base / "models"
    m4.mkdir(parents=True)
    e1_frame().to_csv(m4 / "e1_ratios.csv", index=False)
    e2_frame().to_csv(m4 / "e2_paired.csv", index=False)
    e4_frame().to_csv(m4 / "e4_transfer.csv", index=False)
    if with_e3:
        e3_frame().to_csv(m4 / "e3_rankings.csv", index=False)
    vehicles, profiles = p2_frames()
    rankings = e3_frame()
    for dataset in ("ngsim", "follownet_highd"):
        for model, family in {**FAMILIES, **P2_FAMILY}.items():
            run = root / dataset / "reconstructed" / "driver-fold0" / model / "seed0"
            run.mkdir(parents=True, exist_ok=True)
            (run / "train.json").write_text(json.dumps({"model": model, "family": family}), encoding="utf-8")
            row = rankings[(rankings["dataset"] == dataset) & (rankings["model"] == model)]
            if len(row):
                values = {h: float(row.iloc[0][c]) for h, c in report_columns().items()}
                summary = {h: {"rmse_s": None if np.isnan(v) else v} for h, v in values.items()}
                (run / "p1.json").write_text(json.dumps({"summary": summary, "model": {"family": family}}),
                                             encoding="utf-8")
            if dataset == "ngsim" and model in P2_FAMILY and model != "lstm":
                keep = ["profile", "index", "std_v", "std_v_rec"]
                vehicles.loc[vehicles["model"] == model, keep].assign(area=0.0).to_parquet(run / "p2.vehicles.parquet")
                profiles.loc[profiles["model"] == model, ["profile", "source", "n_vehicles", "in_range"]].to_parquet(
                    run / "p2.profiles.parquet")
    return m4, root


def report_columns() -> dict[str, str]:
    return {"h10": "p1_h10_rmse_s", "h30": "p1_h30_rmse_s", "whole": "p1_whole_rmse_s"}


def test_script_columns_agree(report):
    assert report.P1_COLUMNS == report_columns()


def test_load_p2_reads_the_runs(report, tmp_path):
    _, root = fabricate_root(tmp_path)
    vehicles, profiles = report.load_p2(root, "ngsim")
    assert sorted(vehicles["model"].unique()) == ["ddpg", "idm", "mlp_loop"]  # lstm has no P2 tables
    assert set(vehicles.columns) >= {"model", "family", "profile", "index", "std_v", "std_v_rec"}
    assert dict(profiles.drop_duplicates("model")[["model", "family"]].itertuples(index=False)) == \
        {"idm": "mechanistic", "ddpg": "network", "mlp_loop": "network"}
    empty_v, empty_p = report.load_p2(root, "follownet_highd")
    assert empty_v.empty and empty_p.empty and "std_v" in empty_v.columns


def test_load_horizons_from_e3_or_from_p1_json(report, tmp_path):
    m4, root = fabricate_root(tmp_path / "a")
    frame, sources = report.load_horizons(m4, root)
    assert set(sources.values()) == {"e3_rankings.csv"}
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), e3_frame()[frame.columns].reset_index(drop=True))
    m4, root = fabricate_root(tmp_path / "b", with_e3=False)
    frame, sources = report.load_horizons(m4, root)
    assert set(sources.values()) == {"p1.json of the runs"}
    expected = e3_frame().set_index(["dataset", "model"])
    got = frame.set_index(["dataset", "model"]).loc[expected.index]
    for column in report_columns().values():
        np.testing.assert_allclose(got[column].to_numpy(float), expected[column].to_numpy(float))
    assert got["family"].tolist() == expected["family"].tolist()


def test_script_writes_every_figure_and_the_readme(report, tmp_path):
    m4, root = fabricate_root(tmp_path)
    out = tmp_path / "figures"
    paths = report.main([f"root={root}", f"m4={m4}", f"out={out}"])
    stems = ["e1_persistence_ratio_bars", "e2_split_inflation_bars", "e3_rank_bump_chart", "e4_transfer_heatmap",
             "p1_horizon_curves", "p2_concave_growth_curves"]
    expected = {f"{s}{suffix}" for s in stems for suffix in (".png", ".pdf")}
    assert {p.name for p in paths} == expected | {"README.md"}
    assert {p.name for p in out.iterdir()} == expected | {"README.md"}
    lines = [line for line in (out / "README.md").read_text(encoding="utf-8").splitlines() if line.startswith("- `")]
    assert sorted(line.split("`")[1] for line in lines) == sorted(expected)  # one line per file
    assert all(abs(png_size(out / f"{s}.png")[0] - PNG_WIDTH) < 1 for s in stems)
    p2_line = next(line for line in lines if line.startswith("- `p2_concave_growth_curves.png`"))
    assert "idm, ddpg, mlp_loop trained on NGSIM" in p2_line


def test_script_skips_figures_without_input(report, tmp_path, capsys):
    m4, root = tmp_path / "m4", tmp_path / "models"
    m4.mkdir()
    e1_frame().to_csv(m4 / "e1_ratios.csv", index=False)
    out = tmp_path / "figures"
    report.main([f"root={root}", f"m4={m4}", f"out={out}"])
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "e1_persistence_ratio_bars.pdf",
                                                     "e1_persistence_ratio_bars.png"]
    printed = capsys.readouterr().out
    assert "skipped (input missing)" in printed and "e4_transfer_heatmap" in printed


def test_script_rejects_unknown_arguments(report):
    with pytest.raises(SystemExit, match="unknown argument"):
        report.arguments(["colour=red"])
