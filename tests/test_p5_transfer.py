from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trafficloop.protocols.p5 import collect_transfer, transfer_matrix


def _runs() -> pd.DataFrame:
    rows = []
    for model, own_a, own_b, cross_ab, cross_ba in (("idm", 6.0, 3.0, 4.5, 9.0), ("lstm", 7.0, 3.5, 7.0, 3.5)):
        rows += [
            {"model": model, "trained_on": "A", "tested_on": "A", "horizon": "whole", "rmse_s": own_a},
            {"model": model, "trained_on": "B", "tested_on": "B", "horizon": "whole", "rmse_s": own_b},
            {"model": model, "trained_on": "A", "tested_on": "B", "horizon": "whole", "rmse_s": cross_ab},
            {"model": model, "trained_on": "B", "tested_on": "A", "horizon": "whole", "rmse_s": cross_ba},
        ]
    frame = pd.DataFrame(rows)
    frame["rmse_v"] = 1.0
    frame["collision_rate"] = 0.0
    return frame


def test_transfer_matrix_is_relative_to_the_own_run():
    matrix = transfer_matrix(_runs(), "whole")
    assert matrix.loc[("idm", "A"), "A"] == pytest.approx(1.0)
    assert matrix.loc[("idm", "A"), "B"] == pytest.approx(4.5 / 3.0)
    assert matrix.loc[("idm", "B"), "A"] == pytest.approx(9.0 / 6.0)
    assert matrix.loc[("lstm", "A"), "B"] == pytest.approx(2.0)
    assert transfer_matrix(_runs(), "h10").empty  # no rows of that horizon


def test_missing_own_run_gives_nan():
    runs = _runs()
    runs = runs[~((runs["model"] == "idm") & (runs["trained_on"] == "B") & (runs["tested_on"] == "B"))]
    matrix = transfer_matrix(runs, "whole")
    assert np.isnan(matrix.loc[("idm", "A"), "B"])


def test_collect_transfer_reads_p1_files(tmp_path):
    def write(dataset, label, name, rmse):
        directory = tmp_path / dataset / "reconstructed" / "driver-fold0" / label / "seed0"
        directory.mkdir(parents=True, exist_ok=True)
        summary = {"h10": {"rmse_s": rmse, "rmse_v": 0.5, "collision_rate": 0.0, "n_events": 3},
                   "whole": {"rmse_s": 2 * rmse, "rmse_v": 0.6, "collision_rate": 0.1, "n_events": 3}}
        (directory / f"{name}.json").write_text(json.dumps({"summary": summary}))

    write("A", "idm", "p1", 1.0)
    write("A", "idm", "p1_B", 1.5)
    write("B", "idm", "p1", 2.0)
    runs = collect_transfer(tmp_path, "reconstructed", "driver", 0)
    assert len(runs) == 6 and set(runs["tested_on"]) == {"A", "B"}
    row = runs[(runs["trained_on"] == "A") & (runs["tested_on"] == "B") & (runs["horizon"] == "whole")].iloc[0]
    assert row["rmse_s"] == 3.0 and row["model"] == "idm"
    matrix = transfer_matrix(runs, "whole")
    assert matrix.loc[("idm", "A"), "B"] == pytest.approx(3.0 / 4.0)
