from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from follownet2 import compat
from follownet2.data.schema import ARRAY_FIELDS, EVENTS_FILE, Event, EventSet, EventValidationError


def test_parquet_round_trip(driver_events, tmp_path):
    directory = driver_events.to_parquet(tmp_path / "synthetic" / "raw", manifest={"config_hash": "abc"})
    loaded = EventSet.from_parquet(directory)
    assert [ev.event_id for ev in loaded] == [ev.event_id for ev in driver_events]
    for a, b in zip(loaded, driver_events):
        assert a.meta == b.meta and a.follower_id == b.follower_id and a.leader_id == b.leader_id
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    loaded.validate()
    assert loaded.data_version == "raw"
    assert set(pd.read_parquet(directory / EVENTS_FILE)["data_version"]) == {"raw"}


def test_compact_layout_is_lossless_and_smaller(driver_events, tmp_path):
    full = driver_events.to_parquet(tmp_path / "full")
    compact = driver_events.to_parquet(tmp_path / "compact", compact=True)
    loaded = EventSet.from_parquet(compact)
    loaded.validate()
    for a, b in zip(loaded, driver_events):
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert set(pd.read_parquet(compact / "samples.parquet").columns) == {"event_id", *ARRAY_FIELDS} - {"s", "dv"}
    size = lambda d: (d / "samples.parquet").stat().st_size  # noqa: E731
    assert size(compact) < 0.8 * size(full)


def test_version_tag_is_mandatory(clean_events):
    ev = copy.deepcopy(clean_events[0])
    ev.meta.pop("data_version")
    with pytest.raises(EventValidationError, match="data_version"):
        ev.validate()
    ev.meta["data_version"] = "smoothed"
    with pytest.raises(EventValidationError, match="data_version"):
        ev.validate()


@pytest.mark.parametrize(
    "field, change, message",
    [
        ("dv", lambda x: -x, "dv != v - v_lead"),
        ("s", lambda x: x + 1.0, "s != x_lead - x_follower"),
        ("t", lambda x: x * 2.0, "not uniform"),
        ("v", lambda x: np.where(np.arange(len(x)) == 3, np.nan, x), "non-finite"),
    ],
)
def test_validation_catches_broken_conventions(clean_events, field, change, message):
    ev = copy.deepcopy(clean_events[0])
    setattr(ev, field, change(getattr(ev, field)))
    with pytest.raises(EventValidationError, match=message):
        ev.validate()


def test_mixed_versions_are_refused(clean_events):
    a, b = copy.deepcopy(clean_events[0]), copy.deepcopy(clean_events[1])
    b.meta["data_version"] = "reconstructed"
    with pytest.raises(ValueError, match="mixes data versions"):
        _ = EventSet([a, b]).data_version


def test_wire_compatibility_with_cf_stability(driver_events, tmp_path):
    """A directory written here is read by cf-stability, and the other way round."""
    if compat.sibling("cf_stability") is None:
        pytest.skip("cf-stability checkout not found")
    import cf_stability.data.schema as theirs

    directory = driver_events.to_parquet(tmp_path / "ours")
    read_by_them = theirs.EventSet.from_parquet(directory)
    assert len(read_by_them) == len(driver_events)
    np.testing.assert_array_equal(read_by_them[0].s, driver_events[0].s)
    assert read_by_them[0].meta["data_version"] == "raw"

    written_by_them = compat.to_cf_stability(driver_events).to_parquet(tmp_path / "theirs")
    back = EventSet.from_parquet(written_by_them)
    back.validate()
    np.testing.assert_array_equal(back[3].x_follower, driver_events[3].x_follower)
    assert compat.from_cf_stability(read_by_them)[0].event_id == driver_events[0].event_id


def test_untagged_foreign_events_need_a_default_version(driver_events, tmp_path):
    stripped = []
    for ev in list(driver_events)[:3]:
        meta = {k: v for k, v in ev.meta.items() if k != "data_version"}
        stripped.append(Event(**{**ev.__dict__, "meta": meta}))
    foreign = EventSet(stripped)
    directory = tmp_path / "foreign"
    directory.mkdir()
    foreign.samples_frame().to_parquet(directory / "samples.parquet", index=False)
    foreign.events_frame().drop(columns=["data_version"]).to_parquet(directory / "events.parquet", index=False)
    with pytest.raises(EventValidationError):
        EventSet.from_parquet(directory).validate()
    EventSet.from_parquet(directory, default_version="raw").validate()
