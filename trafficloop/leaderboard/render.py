"""Markdown and HTML pages of the leaderboard (docs/leaderboard.md, sections 2 and 5).

Every table of a :class:`~trafficloop.leaderboard.tables.Board` is turned into a frame of display
strings (3 decimals, ``n/a`` for a missing value, ``value [low, high]`` for a value with its 95 %
interval) and written as a Markdown table or an HTML table. :func:`dataset_markdown` gives the
section of one dataset (main table, driver-leak runs, transfer table, refused entries, provenance
footer), :func:`leaderboard_markdown` the page with every dataset and :func:`html_page` the
self-contained HTML page (inline CSS, the rank-stability PNGs embedded as base64).
"""

from __future__ import annotations

import base64
import html
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from trafficloop.leaderboard.tables import Board, summary_rows, transfer_means

TITLE = "TrafficLoop leaderboard"
MISSING = "n/a"
Cell = Callable[[pd.Series], str]


# ----------------------------------------------------------------------------- cells


def _finite(value) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def num(value, digits: int = 3) -> str:
    """A value with ``digits`` decimals, ``n/a`` when missing."""
    return f"{float(value):.{digits}f}" if _finite(value) else MISSING


def count(value) -> str:
    """A count as an integer (3 decimals when it is not whole)."""
    if not _finite(value):
        return MISSING
    value = float(value)
    return str(int(round(value))) if value.is_integer() else f"{value:.3f}"


def interval(low, high, digits: int = 3) -> str:
    """``[low, high]``, ``n/a`` when a bound is missing."""
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]" if _finite(low) and _finite(high) else MISSING


def with_interval(value, low, high, digits: int = 3) -> str:
    """``value [low, high]``; the value alone when the interval is missing."""
    if not _finite(value):
        return MISSING
    return f"{num(value, digits)} {interval(low, high, digits)}" if _finite(low) and _finite(high) else num(value, digits)


def rank(value, low, high) -> str:
    """A rank (average rank on ties) with its interval at one decimal."""
    if not _finite(value):
        return MISSING
    text = f"{float(value):g}"
    return f"{text} {interval(low, high, 1)}" if _finite(low) and _finite(high) else text


def _text(value) -> str:
    return MISSING if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _ci(column: str) -> Cell:
    return lambda r: with_interval(r[column], r[f"{column}_low"], r[f"{column}_high"])


def _pair(a: str, b: str, pattern: str) -> Cell:
    def cell(r: pd.Series) -> str:
        x, y = num(r[a]), num(r[b])
        return MISSING if x == y == MISSING else pattern.format(x, y)
    return cell


#: main table: (first source column, header, cell); a display column is shown when its source column is
MAIN_CELLS: list[tuple[str, str, Cell]] = [
    ("model", "model", lambda r: _text(r["model"])),
    ("family", "family", lambda r: _text(r["family"])),
    ("p0_ratio", "P0 ratio [95% CI]", _ci("p0_ratio")),
    ("p1_h10_rmse_s", "P1 10 s RMSE_s (m) [95% CI]", _ci("p1_h10_rmse_s")),
    ("p1_h30_rmse_s", "P1 30 s RMSE_s (m) [95% CI]", _ci("p1_h30_rmse_s")),
    ("p1_whole_rmse_s", "P1 whole RMSE_s (m) [95% CI]", _ci("p1_whole_rmse_s")),
    ("p1_whole_collision_rate", "P1 whole collision rate", lambda r: num(r["p1_whole_collision_rate"])),
    ("p2_concave_growth_error", "P2 concave-growth error*", lambda r: num(r["p2_concave_growth_error"])),
    ("p2_hysteresis_error", "P2 hysteresis error*", lambda r: num(r["p2_hysteresis_error"])),
    ("p2_collisions_100", "P2 collisions_100*", lambda r: count(r["p2_collisions_100"])),
    ("p3_unstable_fraction", "P3 unstable fraction (1 m/s)", _pair("p3_unstable_fraction", "p3_unstable_fraction_large", "{} ({})")),
    ("p4_collision_rate_synthetic", "P4 collision rate synthetic* / empirical",
     _pair("p4_collision_rate_synthetic", "p4_collision_rate_empirical", "{} / {}")),
    ("composite", "composite", lambda r: num(r["composite"])),
    ("composite_rank", "composite rank [95% CI]", lambda r: rank(r["composite_rank"], r["composite_rank_low"], r["composite_rank_high"])),
    ("composite_v2", "composite v2", lambda r: num(r["composite_v2"])),
    ("rank_v2", "rank v2", lambda r: rank(r["rank_v2"], None, None)),
    ("p0_ratio_seed_min", "P0 ratio seed spread", lambda r: interval(r["p0_ratio_seed_min"], r["p0_ratio_seed_max"])),
    ("p1_whole_rmse_s_seed_min", "P1 whole seed spread (m)", lambda r: interval(r["p1_whole_rmse_s_seed_min"], r["p1_whole_rmse_s_seed_max"])),
    ("p0_ratio_fold_min", "P0 ratio fold spread", lambda r: interval(r["p0_ratio_fold_min"], r["p0_ratio_fold_max"])),
    ("p1_whole_rmse_s_fold_min", "P1 whole fold spread (m)", lambda r: interval(r["p1_whole_rmse_s_fold_min"], r["p1_whole_rmse_s_fold_max"])),
]
LEAK_CELLS: list[tuple[str, str, Cell]] = [
    ("model", "model", lambda r: _text(r["model"])),
    ("family", "family", lambda r: _text(r["family"])),
    ("label", "label", lambda r: _text(r["label"])),
    ("run", "run", lambda r: _text(r["run"])),
    ("p0_rmse_a", "P0 RMSE_a (m/s^2) [95% CI]", _ci("p0_rmse_a")),
    ("p1_h10_rmse_s", "P1 10 s RMSE_s (m) [95% CI]", _ci("p1_h10_rmse_s")),
    ("p1_h30_rmse_s", "P1 30 s RMSE_s (m) [95% CI]", _ci("p1_h30_rmse_s")),
    ("p1_whole_rmse_s", "P1 whole RMSE_s (m) [95% CI]", _ci("p1_whole_rmse_s")),
    ("p1_whole_collision_rate", "P1 whole collision rate", lambda r: num(r["p1_whole_collision_rate"])),
    ("p1_whole_rmse_s_driver", "P1 whole RMSE_s, driver split (m)", lambda r: num(r["p1_whole_rmse_s_driver"])),
]
TEXT_COLUMNS = {"model", "family", "label", "run", "role", "problems", "dataset", "ranked by", "top three", "field", "value"}


def display(frame: pd.DataFrame, cells: Sequence[tuple[str, str, Cell]]) -> pd.DataFrame:
    """Display strings of ``frame`` for the cells whose source column it has."""
    chosen = [(header, cell) for source, header, cell in cells if source in frame.columns]
    rows = [[cell(row) for _, cell in chosen] for _, row in frame.iterrows()]
    return pd.DataFrame(rows, columns=[header for header, _ in chosen])


def main_display(board: Board) -> pd.DataFrame:
    return display(board.main, MAIN_CELLS)


def leak_display(board: Board) -> pd.DataFrame:
    return display(board.leak, LEAK_CELLS)


def transfer_display(board: Board) -> tuple[pd.DataFrame, list[str]]:
    """Models by targets: the whole-event relative degradation with its interval and its mean; the
    notes of the cells without a ratio or an interval."""
    transfer = board.transfer
    if transfer.empty:
        return pd.DataFrame(), []
    targets = sorted(transfer["tested_on"].unique())
    means = transfer_means(transfer).set_index("model")
    rows = []
    for model, block in transfer.groupby("model", sort=False):
        by = block.set_index("tested_on")
        row = {"model": model, "family": _text(block["family"].iloc[0])}
        for target in targets:
            row[f"-> {target}"] = with_interval(by.at[target, "ratio"], by.at[target, "low"], by.at[target, "high"]) if target in by.index else "-"
        row["mean"] = num(means.at[model, "mean_degradation"])
        rows.append(row)
    notes = [f"{r.model} -> {r.tested_on}: {r.note}" for r in transfer.itertuples(index=False) if isinstance(r.note, str) and r.note]
    return pd.DataFrame(rows), notes


def refused_display(board: Board) -> pd.DataFrame:
    return board.refused[["run", "model", "role", "label", "problems"]].fillna("") if len(board.refused) else pd.DataFrame()


def counts_text(board: Board) -> str:
    c = board.counts()
    leak = (f"driver leak (`event` split): {c['leak_admitted']} admitted and {c['leak_refused']} refused runs, listed apart and never ranked"
            if c["leak_admitted"] + c["leak_refused"] else "no run under the `event` split")
    return (f"Ranked entries (driver split, fold 0, seed 0): **{c['admitted']} admitted, {c['refused']} refused**; {leak}; "
            f"run directories validated: {c['runs']}, refused {c['runs_refused']}.")


def main_caption(board: Board) -> str:
    main = board.main
    order = ("composite (lower is better), then by the whole-event P1 spacing error" if "composite" in main
             else "whole-event P1 spacing error (the dataset has P1 only: no P2-P4, no composite)")
    parts = [f"Sorted by the {order}. `[low, high]`: 95 % intervals over {board.n_boot} replicates (seed {board.seed}): "
             "driver bootstrap with common random numbers across models for P0 and P1 (per-event tables that reproduce their "
             "JSON summary)"]
    if "composite_rank" in main:
        parts.append(", bootstrap over the in-range P2 profiles, the P3 speeds with an equilibrium and the training seeds for "
                     "the composite rank (the rank of the seed-0 composite, average ranks on ties)")
    parts.append(". P0 ratio: RMSE_a(model) / RMSE_a(persistence).")
    if "p2_concave_growth_error" in main or "p4_collision_rate_synthetic" in main:
        parts.append(" `*`: in-range values (P2 over the OpenACC profiles inside the 1-99 % speed range of the training data and "
                     "the pulse, P4 over the synthetic scenarios inside it).")
    if "p3_unstable_fraction" in main:
        parts.append(" P3: numerical unstable fraction at 0.05 m/s (at 1 m/s in parentheses) over the speeds with an equilibrium "
                     "(n/a: none).")
    if "p0_ratio_seed_min" in main:
        parts.append(" Seed spread: min-max over the training seeds.")
    if "p0_ratio_fold_min" in main:
        parts.append(" Fold spread: min-max over the clean arms of the paired split (probe events; the P0 ratio against "
                     "persistence on the same events).")
    return "".join(parts)


def plot_caption(board: Board) -> str:
    ranks = [name for name, frame in (("composite rank", board.composite_ranks), ("whole-event P1 rank", board.p1_ranks))
             if frame is not None and len(frame)]
    if not ranks:
        return "no rank distribution available."
    return (f"{' and '.join(ranks)} of every admitted model over {board.n_boot} replicates (dot: mean rank, whiskers: 95 % "
            "interval).")


TRANSFER_CAPTION = ("Relative degradation `rmse_s(A -> B) / rmse_s(B -> B)` of the whole event (A: this dataset, B: the "
                    "target; 1 = no loss) with 95 % driver-bootstrap intervals over the test drivers of B, and its mean over "
                    "the targets with a ratio. Only admitted entries; the ratio is left out when the run trained on B is not "
                    "admitted.")
LEAK_CAPTION = ("Runs under the `event` split share drivers between training and test (checklist rule 2): they are labelled "
                "\"driver leak\", listed apart and never ranked. Intervals: driver bootstrap over their own test part; the last "
                "column repeats the whole-event P1 error of the driver-split entry.")


# ----------------------------------------------------------------------------- Markdown


def markdown_table(frame: pd.DataFrame) -> str:
    """A Markdown table of display strings (``|`` escaped), ``(none)`` without rows."""
    if frame is None or frame.empty:
        return "(none)\n"

    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(c).replace("*", "\\*") for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _bullets(lines: Sequence[str]) -> str:
    return "".join(f"* {line}\n" for line in lines)


def dataset_markdown(board: Board, level: int = 2, image: str | None = None, heading: bool = True) -> str:
    """The section of one dataset: counts, main table (with notes and the rank-stability plot),
    driver-leak runs, transfer table, refused entries and provenance footer."""
    h, sub = "#" * level, "#" * (level + 1)
    text = f"{h} {board.dataset}\n\n" if heading else ""
    text += counts_text(board) + "\n\n"
    text += f"{sub} Main table\n\n{main_caption(board)}\n\n{markdown_table(main_display(board))}"
    if board.notes:
        text += "\nNotes:\n\n" + _bullets(board.notes)
    if image:
        text += f"\n![Rank stability on {board.dataset}]({image})\n\n*Rank stability*: {plot_caption(board)}\n"
    text += f"\n{sub} Driver leak (event split)\n\n"
    text += (LEAK_CAPTION + "\n\n" + markdown_table(leak_display(board))) if len(board.leak) else "No admitted run under the `event` split.\n"
    text += f"\n{sub} Transfer\n\n"
    frame, notes = transfer_display(board)
    if frame.empty:
        text += "No admitted entry with `p1_<other>.json`.\n"
    else:
        text += TRANSFER_CAPTION + "\n\n" + markdown_table(frame)
        if notes:
            text += "\n" + _bullets(notes)
    text += f"\n{sub} Refused entries\n\n"
    refused = refused_display(board)
    text += ("Runs whose checklist is incomplete (docs/checklist.md); they never appear in a ranking.\n\n" + markdown_table(refused)
             if len(refused) else "None: every run directory of the dataset has a complete checklist.\n")
    text += f"\n{sub} Provenance\n\n" + _bullets([f"**{key}**: {value}" for key, value in board.provenance.items()])
    return text


def dataset_page(board: Board, image: str | None = None, source: str = "") -> str:
    """``<dataset>.md``: the section of one dataset as a page."""
    return (f"# {TITLE}: {board.dataset}\n\n{_generated(source, board)}\n\n"
            + dataset_markdown(board, level=1, image=image, heading=False))


def _generated(source: str, board: Board | None = None) -> str:
    date = board.provenance.get("date", "") if board is not None else ""
    where = f" from `{source}`" if source else ""
    return (f"Generated by `scripts/make_leaderboard.py`{where}{' on ' + date if date else ''}; definitions in "
            "`docs/leaderboard.md`, the checklist in `docs/checklist.md`. 3 decimals, `n/a` = not available.")


def summary_display(boards: Mapping[str, Board]) -> pd.DataFrame:
    rows = summary_rows(boards)
    if rows.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "dataset": rows["dataset"],
        "ranked entries admitted": rows["admitted"].astype(str),
        "refused": rows["refused"].astype(str),
        "driver leak (event split)": [f"{a} admitted, {r} refused" if a + r else "-" for a, r in zip(rows["leak_admitted"], rows["leak_refused"])],
        "run directories (refused)": [f"{n} ({r})" for n, r in zip(rows["runs"], rows["runs_refused"])],
        "ranked by": rows["ranked_by"],
        "top three": rows["top"],
    })


def leaderboard_markdown(boards: Mapping[str, Board], images: Mapping[str, str] | None = None, source: str = "") -> str:
    """``LEADERBOARD.md``: the summary and the section of every dataset."""
    first = next(iter(boards.values()), None)
    text = f"# {TITLE}\n\n{_generated(source, first)}\n\n## Summary\n\n{markdown_table(summary_display(boards))}\n"
    for dataset, board in boards.items():
        text += "\n" + dataset_markdown(board, level=2, image=(images or {}).get(dataset))
    return text


# ----------------------------------------------------------------------------- HTML


CSS = """
:root { color-scheme: light; --bg: #fcfcfb; --surface: #ffffff; --text: #1b1b1a; --muted: #5d5c58;
  --rule: #e2e1dc; --head: #f3f2ee; --stripe: #f8f7f4; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --bg: #161615; --surface: #1d1d1c; --text: #f2f2f0;
    --muted: #b5b4ad; --rule: #3a3a37; --head: #262624; --stripe: #222220; --accent: #5b9be8; }
}
:root[data-theme="dark"] { color-scheme: dark; --bg: #161615; --surface: #1d1d1c; --text: #f2f2f0;
  --muted: #b5b4ad; --rule: #3a3a37; --head: #262624; --stripe: #222220; --accent: #5b9be8; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
main { max-width: 1480px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; }
h2 { font-size: 1.3rem; margin: 2.6rem 0 .6rem; padding-bottom: .3rem; border-bottom: 2px solid var(--rule); }
h3 { font-size: 1.02rem; margin: 1.6rem 0 .4rem; }
p { margin: .4rem 0 .7rem; max-width: 110ch; }
.muted, .caption { color: var(--muted); }
.caption { font-size: 12.5px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .92em; }
nav { margin: .8rem 0 1.2rem; display: flex; flex-wrap: wrap; gap: .4rem .9rem; }
nav a { color: var(--accent); text-decoration: none; }
nav a:hover { text-decoration: underline; }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); margin: .4rem 0 .6rem; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; width: max-content; min-width: 100%; }
th, td { padding: 4px 9px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
th { background: var(--head); font-weight: 600; text-align: center; vertical-align: bottom; white-space: normal; min-width: 5.5em; }
td.text, th.text { text-align: left; }
td.wrap { white-space: normal; min-width: 28em; }
tbody tr:nth-child(even) td { background: var(--stripe); }
tbody tr:last-child td { border-bottom: 0; }
th:first-child, td:first-child { position: sticky; left: 0; z-index: 1; background: var(--surface); }
th:first-child { background: var(--head); }
tbody tr:nth-child(even) td:first-child { background: var(--stripe); }
ul.notes { margin: .3rem 0 .8rem; padding-left: 1.2rem; color: var(--muted); font-size: 12.5px; }
figure { margin: 1rem 0; }
figure img { display: block; max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; background: #ffffff; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: .3rem; }
footer.provenance { margin-top: 1.2rem; padding: .7rem .9rem; border: 1px solid var(--rule); border-radius: 6px;
  background: var(--surface); font-size: 12.5px; }
footer.provenance dl { display: grid; grid-template-columns: max-content 1fr; gap: .2rem 1rem; margin: .3rem 0 0; }
footer.provenance dt { font-weight: 600; }
footer.provenance dd { margin: 0; overflow-wrap: anywhere; }
.badge { display: inline-block; padding: 0 .45em; border-radius: 4px; border: 1px solid var(--rule); font-size: 11.5px; }
"""


def _inline(text: str) -> str:
    """Escaped text with `code` spans and **bold** of the Markdown captions."""
    out, code = [], False
    for i, part in enumerate(html.escape(text).split("`")):
        if i:
            code = not code
        out.append(f"<code>{part}</code>" if code else part)
    joined = "".join(out)
    pieces = joined.split("**")
    return "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(pieces))


def html_table(frame: pd.DataFrame, wrap: Sequence[str] = ()) -> str:
    """An HTML table of display strings in a horizontally scrollable box."""
    if frame is None or frame.empty:
        return '<p class="muted">(none)</p>'
    def css(column: str) -> str:
        names = (["text"] if column in TEXT_COLUMNS else []) + (["wrap"] if column in wrap else [])
        return f' class="{" ".join(names)}"' if names else ""

    head = "".join(f"<th{css(c)}>{html.escape(str(c))}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f"<td{css(c)}>{html.escape(str(v))}</td>" for c, v in zip(frame.columns, row)) + "</tr>"
                   for row in frame.itertuples(index=False))
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _notes(lines: Sequence[str]) -> str:
    return '<ul class="notes">' + "".join(f"<li>{_inline(line)}</li>" for line in lines) + "</ul>" if lines else ""


def _anchor(dataset: str) -> str:
    return "ds-" + "".join(ch if ch.isalnum() else "-" for ch in dataset)


def html_section(board: Board, image: bytes | None = None) -> str:
    """The section of one dataset: main table, plot, driver-leak runs, transfer table, refused
    entries and provenance footer."""
    parts = [f'<section id="{_anchor(board.dataset)}">', f"<h2>{html.escape(board.dataset)}</h2>", f"<p>{_inline(counts_text(board))}</p>",
             "<h3>Main table</h3>", f'<p class="caption">{_inline(main_caption(board))}</p>', html_table(main_display(board)),
             _notes(board.notes)]
    if image:
        data = base64.b64encode(image).decode("ascii")
        parts.append(f'<figure><img alt="Rank stability on {html.escape(board.dataset)}" src="data:image/png;base64,{data}">'
                     f"<figcaption>Rank stability: {html.escape(plot_caption(board))}</figcaption></figure>")
    parts.append("<h3>Driver leak (event split)</h3>")
    if len(board.leak):
        parts += [f'<p class="caption">{_inline(LEAK_CAPTION)}</p>', html_table(leak_display(board))]
    else:
        parts.append('<p class="muted">No admitted run under the <code>event</code> split.</p>')
    parts.append("<h3>Transfer</h3>")
    frame, notes = transfer_display(board)
    if frame.empty:
        parts.append('<p class="muted">No admitted entry with <code>p1_&lt;other&gt;.json</code>.</p>')
    else:
        parts += [f'<p class="caption">{_inline(TRANSFER_CAPTION)}</p>', html_table(frame), _notes(notes)]
    parts.append("<h3>Refused entries</h3>")
    refused = refused_display(board)
    if len(refused):
        parts += ['<p class="caption">Runs whose checklist is incomplete (docs/checklist.md); they never appear in a ranking.</p>',
                  html_table(refused, wrap=("problems",))]
    else:
        parts.append('<p class="muted">None: every run directory of the dataset has a complete checklist.</p>')
    items = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in board.provenance.items())
    parts.append(f'<footer class="provenance"><strong>Provenance</strong><dl>{items}</dl></footer>')
    parts.append("</section>")
    return "\n".join(p for p in parts if p)


def html_page(boards: Mapping[str, Board], images: Mapping[str, bytes] | None = None, source: str = "") -> str:
    """The self-contained page with every dataset (inline CSS, PNGs as base64, no external resource)."""
    images = images or {}
    first = next(iter(boards.values()), None)
    nav = "".join(f'<a href="#{_anchor(d)}">{html.escape(d)}</a>' for d in boards)
    sections = "\n".join(html_section(board, images.get(dataset)) for dataset, board in boards.items())
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{TITLE}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n"
        f"<h1>{TITLE}</h1>\n<p class=\"muted\">{_inline(_generated(source, first))}</p>\n"
        f"<nav>{nav}</nav>\n<h2>Summary</h2>\n{html_table(summary_display(boards))}\n{sections}\n</main>\n</body>\n</html>\n"
    )
