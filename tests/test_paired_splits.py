"""Paired splits of E2 (decision R2): the clean and the leaky arm against the driver split."""

from __future__ import annotations

import copy

import pytest

from trafficloop.data.splits import LeakageError, Split, check_leakage, event_table, make_split
from trafficloop.experiments.paired import STATS, paired_partitions, paired_splits
from trafficloop.utils import read_json, write_json


@pytest.fixture(scope="module")
def table(driver_events):
    return event_table(driver_events)


@pytest.fixture(scope="module")
def driver(table):
    return make_split(table, "driver", seed=0)


@pytest.fixture(scope="module")
def arms(driver, table):
    return paired_splits(driver, table.set_index("event_id"))


def mixed_table(table):
    """Drivers with one, three and four events (the fixture gives every driver four)."""
    keep = []
    for i, (_, events) in enumerate(table.sort_values("event_id").groupby("follower_id", sort=True)):
        keep.extend(events["event_id"].iloc[: (1, 3, 4)[i % 3]])
    return table[table["event_id"].isin(keep)].reset_index(drop=True)


def test_arms_are_splits_derived_from_the_driver_split(driver, arms):
    clean, leaky = arms
    for arm, kind in zip(arms, ("paired-clean", "paired-leaky")):
        assert (arm.kind, arm.dataset, arm.seed, arm.n_folds) == (kind, driver.dataset, driver.seed, 5)
        assert arm.groups == driver.groups and arm.folds == driver.folds and arm.val == driver.val
        assert set(arm.explicit) == {str(k) for k in range(5)}
        assert sum(note.startswith("fold ") for note in arm.notes) == 5
        for fold in range(5):  # the parts of a fold are a partition of the events
            parts = arm.partition(fold)
            assert sum(len(ids) for ids in parts.values()) == len(arm.assignment(fold)) == len(driver.folds)
    assert clean.driver_disjoint and not leaky.driver_disjoint


def test_arms_share_the_probe_events_and_the_validation_part(driver, arms):
    clean, leaky = arms
    for fold in range(driver.n_folds):
        reference, a, b = driver.partition(fold), clean.partition(fold), leaky.partition(fold)
        assert a["test"] == b["test"] and set(a["test"]) < set(reference["test"])
        assert a["val"] == b["val"] == sorted(reference["val"])
        assert a["train"] == sorted(reference["train"])
        assert len(a["train"]) == len(b["train"])
        assert not set(a["test"]) & (set(a["train"]) | set(b["train"]))


def test_leak_events_of_the_test_drivers_replace_training_events(driver, arms, table):
    follower = table.set_index("event_id")["follower_id"]
    clean, leaky = arms
    for fold in range(driver.n_folds):
        reference, a, b = driver.partition(fold), clean.partition(fold), leaky.partition(fold)
        leak, replaced = set(b["train"]) - set(a["train"]), set(a["train"]) - set(b["train"])
        assert leak and len(leak) == len(replaced)
        assert set(follower.loc[sorted(leak)]) <= set(follower.loc[reference["test"]])
        assert leak | set(a["test"]) == set(reference["test"])
        assert leak == set(a["unused"]) and replaced == set(b["unused"])
        test = follower.loc[reference["test"]]
        for _, events in test.groupby(test):  # every test driver is halved
            ids = set(events.index)
            assert len(ids & leak) == len(ids) // 2 and len(ids & set(a["test"])) == len(ids) - len(ids) // 2


def test_statistics_describe_the_parts(driver, table):
    frame = table.set_index("event_id")
    for fold in range(driver.n_folds):
        out = paired_partitions(driver, frame, fold)
        stats, clean, leaky = out["stats"], out["clean"], out["leaky"]
        assert tuple(stats) == STATS
        assert stats["n_probe"] == len(clean["test"])
        assert stats["n_leak"] == stats["n_replaced"] == len(set(leaky["train"]) - set(clean["train"]))
        assert stats["n_train"] == len(clean["train"]) == len(leaky["train"])
        assert stats["n_test_drivers"] == frame.loc[driver.partition(fold)["test"], "follower_id"].nunique()
        assert stats["share_probe_with_leak"] == 1.0  # four events per driver: every test driver leaks


def test_single_event_drivers_give_probe_events_only(table):
    mixed = mixed_table(table)
    frame = mixed.set_index("event_id")
    counts = frame["follower_id"].value_counts()
    assert set(counts) == {1, 3, 4}
    split = make_split(mixed, "driver", seed=3)
    n_single = 0
    for fold in range(split.n_folds):
        out = paired_partitions(split, frame, fold)
        test = frame.loc[split.partition(fold)["test"], "follower_id"]
        leak = set(out["leaky"]["train"]) - set(out["clean"]["train"])
        probe = set(out["clean"]["test"])
        single = {e for e, f in test.items() if counts[f] == 1}
        n_single += len(single)
        assert single <= probe and not single & leak
        assert leak | probe == set(test.index) and not leak & probe
        expected = sum(len(ids) - len(ids) // 2 for ids in test.groupby(test).groups.values() if len(ids) > 1)
        assert out["stats"]["share_probe_with_leak"] == pytest.approx(expected / len(probe))
        assert out["stats"]["n_leak"] == sum(len(ids) // 2 for ids in test.groupby(test).groups.values())
    assert n_single > 0


def test_leaky_arm_fails_the_leakage_check_and_the_clean_arm_passes(arms, table):
    clean, leaky = arms
    report = check_leakage(clean, table)
    for fold, diagnostics in report["folds"].items():
        assert diagnostics["test_followers_seen_in_training"] == 0
        assert diagnostics["n_test"] == len(clean.partition(int(fold))["test"])
    for fold in range(leaky.n_folds):
        with pytest.raises(LeakageError, match="groups on both sides|follower"):
            check_leakage(leaky, table, fold=fold)


def test_arms_depend_on_the_driver_split_and_the_fold_only(driver, table):
    frame = table.set_index("event_id")
    first = paired_partitions(driver, frame, 0)
    assert paired_partitions(driver, frame.sample(frac=1.0, random_state=2), 0) == first
    assert paired_partitions(driver, table, 0) == first  # an event_id column serves as well
    reseeded = copy.deepcopy(driver)
    reseeded.seed = 1
    assert paired_partitions(reseeded, frame, 0)["leaky"]["train"] != first["leaky"]["train"]


def test_partition_returns_a_copy_of_the_explicit_parts(driver):
    parts = {"train": ["b"], "val": ["c"], "test": ["a"], "unused": ["d"]}
    split = copy.deepcopy(driver)
    split.explicit = {"0": copy.deepcopy(parts)}
    got = split.partition(0)
    assert got == parts and list(got) == ["train", "val", "test", "unused"]
    got["train"].append("x")
    assert split.explicit["0"]["train"] == ["b"]
    assert split.assignment(0) == {"a": "test", "b": "train", "c": "val", "d": "unused"}
    assert split.partition(1) == driver.partition(1)  # a fold without explicit parts is dealt


def test_split_with_explicit_parts_survives_save_and_load(arms, tmp_path):
    for arm in arms:
        path = arm.save(tmp_path / f"{arm.kind}.json")
        assert "explicit" in read_json(path)
        loaded = Split.load(path)
        assert loaded == arm
        for fold in range(arm.n_folds):
            assert loaded.partition(fold) == arm.partition(fold)
            assert loaded.assignment(fold) == arm.assignment(fold)


def test_split_files_without_explicit_parts_still_load(driver, tmp_path):
    payload = {name: value for name, value in driver.__dict__.items() if name != "explicit"}
    old = Split(**payload)
    loaded = Split.load(write_json(tmp_path / "old.json", payload))
    assert old.explicit == {} and loaded.explicit == {}
    for fold in range(driver.n_folds):
        assert old.partition(fold) == loaded.partition(fold) == driver.partition(fold)
    assert "explicit" not in read_json(driver.save(tmp_path / "driver.json"))  # the dealt kinds keep their format


def test_paired_splits_need_a_driver_split_and_its_events(driver, table):
    event = make_split(table, "event", seed=0)
    with pytest.raises(ValueError, match="driver split"):
        paired_partitions(event, table.set_index("event_id"), 0)
    with pytest.raises(ValueError, match="derived from a driver split"):
        make_split(table, "paired-clean")
    with pytest.raises(ValueError, match="not in the event table"):
        paired_partitions(driver, table.set_index("event_id").iloc[1:], 0)
