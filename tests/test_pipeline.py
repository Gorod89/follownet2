"""Milestone 1 pipeline end to end on a small synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from trafficloop import pipeline
from trafficloop.data.schema import EventSet, events_dir
from trafficloop.data.splits import Split, check_leakage, event_table
from trafficloop.results import CHECKLIST_FIELDS
from trafficloop.utils import REPO_ROOT, DiskSpaceError


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    overrides = [
        "dataset=synthetic",
        "dataset.options.n_drivers=24",
        "dataset.options.events_per_driver=3",
        "workers=1",
        "storage.reserve_mb=1",
        "storage.compact=true",
        "calibration.models=[idm]",
        "calibration.splits=[driver]",
        "calibration.maxiter=15",
        "calibration.per_event_sample=6",
        "calibration.global_sample=20",
        "calibration.device=cpu",
        "calibration.dtype=float64",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=overrides)
    for name in ("events", "splits", "calibration", "results"):
        cfg.paths[name] = str(root / name)  # not an override: hydra cannot parse Windows paths
    return cfg


@pytest.fixture(scope="module")
def built(cfg):
    return pipeline.build_dataset(cfg)


def test_both_versions_are_written_with_tags(cfg, built):
    assert built["raw"]["n_events"] == built["reconstructed"]["n_events"] > 50
    assert built["n_failed_reconstruction"] == 0
    for version in ("raw", "reconstructed"):
        directory = events_dir(cfg.paths.events, "synthetic", version)
        events = EventSet.from_parquet(directory)
        events.validate()
        assert events.data_version == version
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["data_version"] == version and manifest["config_hash"] == built["config_hash"]
        assert manifest["layout"] == "compact"
    raw, rec = (pipeline.load_events(cfg, v) for v in ("raw", "reconstructed"))
    assert [e.event_id for e in raw] == [e.event_id for e in rec]
    assert [e.regime for e in raw] == [e.regime for e in rec]
    assert rec[0].meta["reconstruction"]["method"] == "qp-semi-implicit-v1"


def test_splits_pass_the_leakage_check(cfg, built):
    report = pipeline.build_splits(cfg)
    assert set(report) == {"event", "driver", "site", "regime", "block"}
    assert report["driver"]["test_followers_seen_in_training"] == 0
    assert report["event"]["share_test_followers_seen"] > 0.5
    table = event_table(pipeline.load_events(cfg, "raw"))
    for kind in report:
        check_leakage(Split.load(pipeline.split_path(cfg, kind)), table)


def test_baselines_and_calibration_write_complete_results(cfg, built, monkeypatch):
    pipeline.build_splits(cfg)
    rows = pipeline.run_p0_baselines(cfg)
    persistence = [r for r in rows if r["model"] == "persistence"]
    assert len(persistence) == 10 and all(r["identity_error"] < 1e-12 for r in persistence)  # 5 splits, 2 versions
    oracle = [r for r in rows if r["model"] == "recorded_acceleration" and r["version"] == "reconstructed"]
    assert all(r["rmse_s"] < 1e-9 for r in oracle)

    rows = pipeline.run_calibration(cfg)
    assert {r["kind"] for r in rows} == {"per_event", "global/driver"}
    # a second run with the same configuration resumes: nothing is calibrated again
    monkeypatch.setattr(pipeline, "calibrate_per_event", None)
    monkeypatch.setattr(pipeline, "calibrate_global", None)
    assert pipeline.run_calibration(cfg) == rows
    cfg_other = cfg.copy()
    cfg_other.calibration.maxiter = 16
    with pytest.raises(TypeError):
        pipeline.run_calibration(cfg_other)
    for path in Path(cfg.paths.results).rglob("*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        assert set(result["checklist"]) == set(CHECKLIST_FIELDS)
        assert result["data_version"] in ("raw", "reconstructed") and len(result["config_hash"]) == 12
        assert result["checklist"]["data_version"] == result["data_version"]
        assert result["checklist"]["split_type"] == result["split"]["kind"]
        assert (path.parent / result["per_event"]).exists()
        if result["family"] == "mechanistic":
            assert result["checklist"]["mechanistic_calibration_protocol"]
            assert result["checklist"]["closed_loop_horizon"] == "whole event"


def test_writing_is_refused_when_the_disk_is_full(cfg, built, monkeypatch):
    monkeypatch.setattr("trafficloop.utils.free_bytes", lambda path: 1 << 20)
    with pytest.raises(DiskSpaceError):
        pipeline.write_events(cfg, pipeline.load_events(cfg, "raw"), "synthetic", "raw", {})
