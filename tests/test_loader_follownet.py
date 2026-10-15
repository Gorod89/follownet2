"""FollowNet loader: fabricated .npy files in the published layout, safe unpickling, 25 Hz resampling."""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
from numpy.lib import format as npy_format

from trafficloop.data.extraction import ExtractionConfig, extract_all
from trafficloop.data.loaders import follownet
from trafficloop.data.extraction import to_grid
from trafficloop.data.loaders.follownet import SafeUnpickler, last_report, load_pairs
from trafficloop.utils import resolve_path

REAL_DIR = resolve_path(follownet.DEFAULT_RAW_DIR)


def extract(pairs, cfg: ExtractionConfig | None = None) -> list:
    """``extract_all`` plus ``Event.validate()`` on every event."""
    events = extract_all(pairs, cfg or ExtractionConfig())
    for ev in events:
        ev.validate()
    return events


def event(n: int, rate_hz: float, v: float = 13.0, s0: float = 40.0, sign: float = 1.0) -> list[np.ndarray]:
    """Channels of an event: leader 12 + sin(0.5 t) m/s, follower at constant speed ``v``."""
    t = np.arange(n) / rate_hz
    v_lead = 12.0 + np.sin(0.5 * t)
    spacing = s0 + (12.0 - v) * t + 2.0 * (1.0 - np.cos(0.5 * t))  # s0 + integral of (v_lead - v)
    return [spacing, np.full(n, v), sign * (v_lead - v), v_lead]


def write_object_npy(path, events, legacy_module: bool = True) -> bytes:
    """An object array in the published format: npy 1.0 header and a protocol 3 pickle."""
    array = np.empty((len(events), 4), dtype=object)
    for i, channels in enumerate(events):
        for c in range(4):
            array[i, c] = np.asarray(channels[c], dtype=np.float64)
    payload = pickle.dumps(array, protocol=3)
    if legacy_module:  # the published files were written by numpy < 2
        payload = payload.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")
    with open(path, "wb") as fh:
        npy_format.write_array_header_1_0(fh, {"descr": "|O", "fortran_order": False, "shape": array.shape})
        fh.write(payload)
    return payload


def test_object_event_set_gives_ids_values_and_the_benchmark_sign(tmp_path):
    payload = write_object_npy(
        tmp_path / "NGSIM_I_80_car_fol_event_set.npy",
        [event(401, 10.0, v=13.0), event(301, 10.0, v=11.0), event(120, 10.0)],
    )
    assert b"numpy.core.multiarray" in payload and b"numpy._core" not in payload
    pairs = list(load_pairs(tmp_path, subset="ngsim"))
    assert [p.follower_id for p in pairs] == [f"follownet_ngsim/ngsim/e{i:06d}" for i in range(3)]
    assert pairs[1].leader_id[0] == "follownet_ngsim/ngsim/l000001" and set(pairs[1].leader_id) == {pairs[1].leader_id[0]}
    assert pairs[0].meta == {
        "source_file": "NGSIM_I_80_car_fol_event_set.npy",
        "native_rate_hz": 10.0,
        "ids": "pseudo",
        "gap_definition": follownet.GAP_FRONT,
        "source_index": 0,
    }
    events = extract(pairs)
    assert [(ev.follower_id[-7:], len(ev)) for ev in events] == [("e000000", 401), ("e000001", 301)]  # 12 s is too short
    assert all(ev.site == "ngsim" and ev.dataset == "follownet_ngsim" for ev in events)
    closing, opening = events
    expected = event(401, 10.0, v=13.0)
    np.testing.assert_allclose(closing.s, expected[0], atol=1e-9)
    np.testing.assert_allclose(closing.v_lead, expected[3])
    assert np.all(closing.v == 13.0)
    # the source stores v_lead - v; the benchmark has dv = v - v_lead: positive when closing in
    np.testing.assert_allclose(closing.dv, -expected[2], atol=1e-12)
    assert np.mean(closing.dv) > 0 and np.mean(opening.dv) < 0


def test_highd_is_resampled_from_25_hz_to_the_10_hz_grid(tmp_path):
    dense = np.stack([event(375, 25.0), event(375, 25.0, v=12.5, s0=25.0)])
    np.save(tmp_path / "HighD_car_fol_event_set.npy", dense)
    pairs = list(load_pairs(tmp_path, subset="highd"))
    for pair in pairs:
        np.testing.assert_allclose(pair.t, np.arange(150) * 0.1, atol=1e-12)  # 0 ... 14.9 s: the source ends at 14.96 s
        assert pair.meta["native_rate_hz"] == 25.0
    spacing, v, _, v_lead = event(150, 10.0)  # linear interpolation error below 0.04^2 / 8 * max|f''|
    np.testing.assert_allclose(pairs[0].x_lead - pairs[0].x_follower, spacing, atol=2e-4)
    np.testing.assert_allclose(pairs[0].v_lead, v_lead, atol=2e-4)
    np.testing.assert_allclose(pairs[0].v, v)
    # a 15 s window is 150 samples on the 10 Hz grid, first to last sample 14.9 s
    events = extract(pairs)
    assert [len(ev) for ev in events] == [150, 150] and events[0].duration == pytest.approx(14.9)
    assert all(np.mean(ev.dv) > 0 for ev in events)  # both followers are faster than the leader on average


def test_events_failing_the_channel_cross_check_are_dropped_and_counted(tmp_path):
    good, flipped = event(200, 10.0), event(200, 10.0, sign=-1.0)
    write_object_npy(tmp_path / "Waymo_filter_car_fol_event_set.npy", [good, flipped, good])
    pairs = list(load_pairs(tmp_path, subset="waymo"))
    assert [p.meta["source_index"] for p in pairs] == [0, 2]
    assert pairs[0].meta["gap_definition"] == follownet.GAP_CENTRE
    report = last_report()
    assert report["events_read"] == 3 and report["events_yielded"] == 2 and report["dropped_cross_check"] == 1
    assert report["dropped_events"] == ["Waymo_filter_car_fol_event_set.npy#1"]


def test_split_files_carry_the_legacy_split(tmp_path):
    for split, n in zip(follownet.SPLITS, (3, 1, 2)):
        write_object_npy(tmp_path / f"SPMD1_{split}_data.npy", [event(160, 10.0)] * n)
    pairs = list(load_pairs(tmp_path, subset="spmd1", files="splits"))
    assert [p.meta["legacy_split"] for p in pairs] == ["train"] * 3 + ["val"] + ["test"] * 2
    assert pairs[3].follower_id == "follownet_spmd1/spmd1/val.e000000"
    assert pairs[3].leader_id[0] == "follownet_spmd1/spmd1/val.l000000"
    assert len({p.follower_id for p in pairs}) == 6
    assert len(list(load_pairs(tmp_path, subset="spmd1", files="splits", max_groups=4))) == 4


@pytest.mark.parametrize(
    "payload",
    [
        b"cos\nsystem\n.",  # protocol 0 GLOBAL
        b"\x80\x04\x8c\x02os\x94\x8c\x06system\x94\x93\x94.",  # protocol 4 STACK_GLOBAL
        b"cbuiltins\neval\n.",
        b"cnumpy\nload\n.",
        b"cnumpy.core.multiarray\nscalar\n.",
    ],
)
def test_foreign_globals_are_refused(tmp_path, payload):
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        SafeUnpickler(io.BytesIO(payload)).load()
    with open(tmp_path / "Lyft_train_data.npy", "wb") as fh:
        npy_format.write_array_header_1_0(fh, {"descr": "|O", "fortran_order": False, "shape": (1, 4)})
        fh.write(payload)
    for split in follownet.SPLITS[1:]:
        write_object_npy(tmp_path / f"Lyft_{split}_data.npy", [event(160, 10.0)])
    with pytest.raises(pickle.UnpicklingError):
        list(load_pairs(tmp_path, subset="lyft", files="splits"))


def test_np_load_is_never_asked_to_unpickle(tmp_path, monkeypatch):
    original = np.load

    def guarded(*args, **kwargs):
        assert not kwargs.get("allow_pickle", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    np.save(tmp_path / "HighD_car_fol_event_set.npy", np.stack([event(375, 25.0)]))
    write_object_npy(tmp_path / "NGSIM_I_80_car_fol_event_set.npy", [event(200, 10.0)], legacy_module=False)
    assert len(list(load_pairs(tmp_path, subset="highd"))) == 1
    assert len(list(load_pairs(tmp_path, subset="ngsim"))) == 1


def test_missing_files_and_invalid_options(tmp_path):
    with pytest.raises(FileNotFoundError, match="HighD_car_fol_event_set.npy"):
        list(load_pairs(tmp_path, subset="highd"))
    with pytest.raises(FileNotFoundError, match="NGSIM_I_80_val_data.npy"):
        list(load_pairs(tmp_path, subset="ngsim", files="splits"))
    with pytest.raises(ValueError, match="no event set"):
        list(load_pairs(tmp_path, subset="lyft"))
    with pytest.raises(ValueError, match="unknown FollowNet subset"):
        list(load_pairs(tmp_path, subset="highd30"))
    with pytest.raises(ValueError, match="files must be"):
        list(load_pairs(tmp_path, subset="ngsim", files="all"))


def test_to_grid_places_interpolates_and_keeps_holes():
    t = np.array([10.0, 10.1, 10.2, 10.5, 10.6])  # on the grid, 0.2 s missing
    grid_t, cols = to_grid(t, {"x": np.arange(5.0)})
    np.testing.assert_allclose(grid_t, np.arange(100, 107) * 0.1)
    np.testing.assert_array_equal(np.isnan(cols["x"]), [False, False, False, True, True, False, False])
    t = np.array([0.0, 0.04, 0.09, 0.13, 0.17, 0.21, 0.45, 0.49])  # about 25 Hz with a hole of 0.24 s
    grid_t, cols = to_grid(t, {"x": 3.0 * t, "k": np.arange(8.0), "k2": np.arange(8.0)}, nearest=("k",))
    np.testing.assert_allclose(grid_t, [0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(cols["x"][:3], [0.0, 0.3, 0.6])
    assert np.all(np.isnan(cols["x"][3:])) and np.all(np.isnan(cols["k"][3:]))
    np.testing.assert_array_equal(cols["k"][:3], [0.0, 2.0, 5.0])  # nearest samples: 0.09 and 0.21
    np.testing.assert_allclose(cols["k2"][:3], [0.0, 2.25, 4.75])
    grid_t, cols = to_grid(np.array([0.3, 0.1, 0.1, 0.2]), {"x": np.array([3.0, 1.0, 9.0, 2.0])})
    np.testing.assert_allclose(cols["x"], [1.0, 2.0, 3.0])  # sorted, first of a repeated time stamp kept


@pytest.mark.data
@pytest.mark.skipif(not (REAL_DIR / "NGSIM_I_80_car_fol_event_set.npy").is_file(), reason="FollowNet files not on disk")
def test_real_event_sets():
    pairs = list(load_pairs(REAL_DIR, subset="ngsim", max_groups=40))  # the pickle is read whole (24 MB)
    report = last_report()
    assert report["events_read"] == 40 and report["dropped_cross_check"] == 0 and report["max_residual"] <= 1e-9
    assert pairs[7].follower_id == "follownet_ngsim/ngsim/e000007"
    assert len(extract(pairs)) >= 30
    if (REAL_DIR / "HighD_car_fol_event_set.npy").is_file():  # memory-mapped: 20 events are read
        pairs = list(load_pairs(REAL_DIR, subset="highd", max_groups=20))
        assert last_report()["dropped_cross_check"] == 0 and all(len(p.t) == 150 for p in pairs)
        assert len(extract(pairs, ExtractionConfig(min_duration=14.9))) == 20
