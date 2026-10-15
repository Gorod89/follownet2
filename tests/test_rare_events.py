"""Near-miss episode mining."""

from __future__ import annotations

import numpy as np
import pytest

from trafficloop.data.batch import WARMUP, EventBatch
from trafficloop.data.extraction import PairSeries
from trafficloop.data.rare_events import EpisodeConfig, drac, mine_all, mine_episodes, time_to_collision
from trafficloop.data.schema import DT
from trafficloop.models.mechanistic import IDM
from trafficloop.sim.rollout import rollout


def platoon_pair(n=600, brake_at=None, brake=-6.0, cut_in_at=None, cut_in_gap=8.0, gap=35.0, speed=20.0):
    """Follower at constant speed behind a leader that may brake hard or be replaced by a cut-in."""
    t = np.arange(n) * DT
    acc = np.zeros(n)
    if brake_at is not None:
        acc[brake_at : brake_at + 25] = brake
    v_lead = np.clip(speed + np.cumsum(acc) * DT, 0.0, None)
    x_lead = gap + np.cumsum(v_lead) * DT
    leader = np.array(["toy/a/1"] * n, dtype=object)
    v = np.full(n, speed)
    x = np.cumsum(v) * DT
    if brake_at is not None:  # the follower reacts 1.5 s later with the same braking
        lag = np.zeros(n)
        lag[brake_at + 15 : brake_at + 40] = brake
        v = np.clip(speed + np.cumsum(lag) * DT, 0.0, None)
        x = np.cumsum(v) * DT
    if cut_in_at is not None:
        leader[cut_in_at:] = "toy/a/9"
        x_lead[cut_in_at:] = x[cut_in_at] + cut_in_gap + (x_lead[cut_in_at:] - x_lead[cut_in_at])
    return PairSeries("toy", "a", "toy/a/2", t, x, v, leader, x_lead, v_lead,
                      lane_follower=np.full(n, 1.0), meta={"position_source": "track"})


def test_safety_measures():
    s = np.array([20.0, 20.0, 20.0])
    dv = np.array([5.0, 0.0, -5.0])
    np.testing.assert_allclose(time_to_collision(s, dv), [4.0, np.inf, np.inf])
    np.testing.assert_allclose(drac(s, dv), [25.0 / 40.0, 0.0, 0.0])


def test_calm_following_has_no_episode():
    assert mine_episodes(platoon_pair()) == []


def test_hard_braking_leader_gives_one_episode():
    cfg = EpisodeConfig()
    episodes = mine_episodes(platoon_pair(brake_at=300), cfg)
    assert len(episodes) == 1
    ev = episodes[0]
    ev.validate()
    tags = ev.meta["episode"]
    assert "brake" in tags["triggers"] and tags["min_leader_acc"] < -4.0 and not tags["cut_in"]
    assert ev.event_id == "toy/a/2|1|nm0" and ev.meta["data_version"] == "raw"
    # the window holds the history given to the models before the critical instant
    assert tags["t_trigger"] >= WARMUP * DT and len(ev) <= (cfg.pre + cfg.post) / DT + 1
    assert 295 * DT <= ev.meta["t0"] + tags["t_trigger"] <= 345 * DT


def test_cut_in_gives_an_episode_with_a_gap_jump():
    episodes = mine_episodes(platoon_pair(cut_in_at=300, cut_in_gap=8.0))
    assert len(episodes) == 1
    ev = episodes[0]
    tags = ev.meta["episode"]
    assert tags["cut_in"] and "cut_in" in tags["triggers"] and tags["leader_changes"] == 1
    assert ev.leader_id == "toy/a/9"
    jump = np.diff(ev.s)
    assert jump.min() < -20.0 and np.sum(jump < -1.0) == 1
    ev.validate()
    # the replay sees the cut-in through the leader displacement
    batch = EventBatch.from_events([ev])
    result = rollout(IDM(), batch)
    k = int(np.argmin(jump)) + 1
    assert float(result.s[0, k]) < float(result.s[0, k - 1]) - 20.0
    assert float(result.a[0, k : k + 20].min()) < -2.0


def test_wide_cut_in_is_not_an_episode():
    assert mine_episodes(platoon_pair(cut_in_at=300, cut_in_gap=30.0)) == []


def test_low_ttc_is_found_and_noise_does_not_fire():
    pair = platoon_pair(n=500, gap=60.0)
    pair.v_lead[:] = 12.0  # the follower closes in at 8 m/s
    pair.x_lead = 60.0 + np.cumsum(pair.v_lead) * DT
    stop = int(np.argmax(pair.x_lead - pair.x_follower < 2.0))
    pair.leader_id[stop:] = None
    episodes = mine_episodes(pair)
    assert len(episodes) == 1
    tags = episodes[0].meta["episode"]
    assert "ttc" in tags["triggers"] and tags["min_ttc"] < 2.0 and tags["max_drac"] > 3.35

    rng = np.random.default_rng(0)
    noisy = platoon_pair()
    noisy.v = noisy.v + 0.6 * rng.standard_normal(len(noisy.v))
    noisy.v_lead = noisy.v_lead + 0.6 * rng.standard_normal(len(noisy.v))
    assert mine_episodes(noisy) == []


def test_creeping_queue_and_touching_records_are_not_near_misses():
    creeping = platoon_pair(n=400, gap=8.0, speed=2.0)
    creeping.v_lead[:] = 1.0  # closes 1 m/s at walking pace: TTC falls below 2 s
    creeping.x_lead = 8.0 + np.cumsum(creeping.v_lead) * DT
    creeping.leader_id[int(np.argmax(creeping.x_lead - creeping.x_follower < 0.5)):] = None
    assert mine_episodes(creeping) == []
    assert len(mine_episodes(creeping, EpisodeConfig(min_speed=0.0, min_gap=0.0))) == 1
    touching = platoon_pair(n=400, gap=0.4)
    touching.v_lead[:] = 19.9
    touching.x_lead = 0.4 + np.cumsum(touching.v_lead) * DT + 0.1 * DT * np.arange(400)
    assert mine_episodes(touching) == []


def test_follower_lane_change_ends_the_run_and_short_runs_are_skipped():
    pair = platoon_pair(brake_at=300)
    pair.lane_follower[290:] = 2.0
    pair.lane_follower[330:] = 3.0  # the braking happens inside a run of 4 s
    assert mine_episodes(pair) == []


def test_leaders_that_stay_for_one_or_two_samples_do_not_break_the_smoothing():
    pair = platoon_pair(brake_at=300)
    pair.leader_id[200] = "toy/a/7"
    pair.leader_id[400:402] = "toy/a/8"
    episodes = mine_episodes(pair)
    assert len(episodes) == 1 and "brake" in episodes[0].meta["episode"]["triggers"]


def test_triggers_can_be_selected_and_ids_are_unique():
    pair = platoon_pair(brake_at=300)
    assert mine_episodes(pair, EpisodeConfig(triggers=("cut_in",))) == []
    both = mine_all([pair, platoon_pair(cut_in_at=200)])
    assert len(both) == 2 and len({ev.event_id for ev in both}) == 2


@pytest.mark.parametrize("post", [6.0, 12.0])
def test_separate_events_give_separate_episodes(post):
    pair = platoon_pair(n=900, brake_at=200)
    pair.leader_id[700:] = "toy/a/9"
    # both vehicles run at 5 m/s after the braking: 3 m are a time gap of 0.6 s
    pair.x_lead[700:] = pair.x_follower[700] + 3.0 + (pair.x_lead[700:] - pair.x_lead[700])
    episodes = mine_episodes(pair, EpisodeConfig(post=post))
    assert [ev.meta["episode"]["cut_in"] for ev in episodes] == [False, True]
