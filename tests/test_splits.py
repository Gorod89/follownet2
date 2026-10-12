"""Leakage check and determinism of the splits."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from follownet2.data.splits import (
    KINDS,
    LeakageError,
    Split,
    check_leakage,
    event_table,
    make_legacy_split,
    make_split,
)


@pytest.fixture(scope="module")
def table(driver_events):
    return event_table(driver_events)


@pytest.mark.parametrize("kind", KINDS)
def test_split_is_a_partition_and_passes_the_check(table, kind):
    split = make_split(table, kind, seed=0)
    report = check_leakage(split, table)
    for fold in range(split.n_folds):
        parts = split.partition(fold)
        ids = [e for part in parts.values() for e in part]
        assert sorted(ids) == sorted(table["event_id"])
        assert all(len(parts[name]) > 0 for name in ("train", "val", "test"))
    assert set(report["folds"]) == {str(k) for k in range(split.n_folds)}


def test_driver_split_has_no_shared_follower(table):
    split = make_split(table, "driver", seed=0)
    follower = table.set_index("event_id")["follower_id"]
    for fold in range(split.n_folds):
        parts = split.partition(fold)
        train, val, test = (set(follower.loc[parts[name]]) for name in ("train", "val", "test"))
        assert not train & test and not val & test and not train & val
    report = check_leakage(split, table)
    assert all(f["test_followers_seen_in_training"] == 0 for f in report["folds"].values())


def test_leakage_check_fails_on_a_leaky_driver_split(table):
    split = make_split(table, "driver", seed=0)
    leaky = copy.deepcopy(split)
    follower = table.set_index("event_id")["follower_id"]
    victim = split.partition(0)["test"][0]
    # move one of several events of a test driver to the development part
    siblings = [e for e in split.partition(0)["test"] if follower[e] == follower[victim]]
    assert len(siblings) > 1
    leaky.folds[victim] = 1
    with pytest.raises(LeakageError, match="follower|driver"):
        check_leakage(leaky, table, fold=0)


def test_event_split_relabelled_as_driver_split_is_caught(table):
    split = make_split(table, "event", seed=0)
    report = check_leakage(split, table)
    assert all(f["share_test_followers_seen"] > 0.5 for f in report["folds"].values())
    disguised = copy.deepcopy(split)
    disguised.kind = "driver"
    with pytest.raises(LeakageError):
        check_leakage(disguised, table)


def test_site_and_regime_splits_are_leave_one_group_out(table):
    site = make_split(table, "site", seed=0)
    assert site.n_folds == 3 and site.notes
    sites = table.set_index("event_id")["site"]
    for fold in range(site.n_folds):
        parts = site.partition(fold)
        assert len(set(sites.loc[parts["test"]])) == 1
        assert not set(sites.loc[parts["test"]]) & set(sites.loc[parts["train"] + parts["val"]])
    regime = make_split(table, "regime", seed=0)
    assert regime.n_folds == table["regime"].nunique()


def test_regime_split_leaves_out_the_followers_of_the_test_part(table):
    """Every synthetic driver is met in several regimes: the strict split must not leak them."""
    follower = table.set_index("event_id")["follower_id"]
    strict = make_split(table, "regime", seed=0, max_unused=1.0)
    assert strict.driver_disjoint and not any("not driver-disjoint" in n for n in strict.notes)
    report = check_leakage(strict, table)
    for fold in range(strict.n_folds):
        parts = strict.partition(fold)
        assert not set(follower.loc[parts["test"]]) & set(follower.loc[parts["train"] + parts["val"]])
        assert report["folds"][str(fold)]["n_unused"] == len(parts.get("unused", []))
    assert sum(len(v) for v in strict.unused.values()) > 0
    # when too much would be lost the split stays overlapping and says so
    loose = make_split(table, "regime", seed=0, max_unused=0.0)
    assert not loose.driver_disjoint and loose.unused == {}
    assert any("not driver-disjoint" in n for n in loose.notes)
    assert any(f["test_followers_seen_in_training"] > 0 for f in check_leakage(loose, table)["folds"].values())
    tampered = copy.deepcopy(loose)
    tampered.driver_disjoint = True
    with pytest.raises(LeakageError):
        check_leakage(tampered, table)


def platoon_table(n_blocks=12, per_block=20):
    """Vehicles pass one after the other; every vehicle follows the one before it."""
    rows = []
    for v in range(1, n_blocks * per_block):
        t0 = 15.0 * v
        rows.append({
            "event_id": f"toy/a/r.{v}|{v - 1}|0", "dataset": "toy", "site": "a", "group": "r",
            "follower_id": f"toy/a/r.{v}", "leader_id": f"toy/a/r.{v - 1}", "n_samples": 300,
            "t0": t0, "regime": "steady", "legacy_split": None,
        })
    return pd.DataFrame(rows)


def test_block_split_separates_platoons():
    table = platoon_table()
    driver = make_split(table, "driver", seed=0)
    overlap = check_leakage(driver, table)["folds"]["0"]["share_test_events_with_training_follower_as_leader"]
    assert overlap > 0.5  # the driver split leaves the leaders of the test events in the training part
    block = make_split(table, "block", seed=0, block_seconds=300.0)
    assert block.driver_disjoint and block.block_seconds == 300.0
    report = check_leakage(block, table)
    for fold in range(block.n_folds):
        parts = block.partition(fold)
        dev = table[table["event_id"].isin(parts["train"] + parts["val"])]
        test = table[table["event_id"].isin(parts["test"])]
        assert not (set(dev["follower_id"]) | set(dev["leader_id"])) & (set(test["follower_id"]) | set(test["leader_id"]))
        assert report["folds"][str(fold)]["share_test_events_with_training_follower_as_leader"] == 0.0
        assert 0 < report["folds"][str(fold)]["n_unused"] <= 2 * len(test) / 20 + 4  # the vehicles at the block borders
    tampered = copy.deepcopy(block)
    moved = tampered.unused["0"].pop()
    with pytest.raises(LeakageError, match="vehicles on both sides|follower"):
        check_leakage(tampered, table, fold=0)
    assert moved
    with pytest.raises(ValueError, match="group"):
        make_split(table.drop(columns=["group"]), "block")


def test_split_depends_on_the_seed_only(table, tmp_path):
    a = make_split(table, "driver", seed=11)
    b = make_split(table.sample(frac=1.0, random_state=1), "driver", seed=11)
    assert a.folds == b.folds and a.val == b.val
    assert a.folds != make_split(table, "driver", seed=12).folds
    path = a.save(tmp_path / "split.json")
    assert Split.load(path).partition(2) == a.partition(2)


def test_folds_are_balanced_in_samples(table):
    split = make_split(table, "driver", seed=0)
    report = check_leakage(split, table)
    shares = [f["share_samples_test"] for f in report["folds"].values()]
    assert max(shares) - min(shares) < 0.05
    assert all(0.10 < f["share_samples_val"] < 0.22 for f in report["folds"].values())


def test_legacy_split_follows_the_published_labels(table):
    labels = ["train", "train", "val", "test"] * (len(table) // 4) + ["train"] * (len(table) % 4)
    published = table.assign(legacy_split=labels)
    split = make_legacy_split(published)
    parts = split.partition(0)
    by_id = dict(zip(published["event_id"], labels))
    assert all(by_id[e] == name for name, ids in parts.items() for e in ids)
    assert split.n_folds == 1 and not split.driver_disjoint
    report = check_leakage(split, published)
    assert report["folds"]["0"]["n_test"] == labels.count("test")
    with pytest.raises(ValueError):
        make_legacy_split(table.assign(legacy_split="holdout"))


def test_split_and_table_must_agree(table):
    split = make_split(table, "driver", seed=0)
    with pytest.raises(LeakageError, match="disagree"):
        check_leakage(split, table.iloc[1:])
