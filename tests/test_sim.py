"""Invariants of the fleet simulator on a synthetic city."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import tiny_config

from evfleet.sim.engine import Simulator
from evfleet.sim.policies import OrchestratedPolicy, QueueAwarePolicy, ThresholdPolicy

POLICIES = [
    lambda: ThresholdPolicy(0.2, 0.8),
    lambda: ThresholdPolicy(0.2, 1.0),
    lambda: QueueAwarePolicy(0.2, 0.8),
    lambda: OrchestratedPolicy(),
]


def max_concurrent(intervals: list[tuple[float, float]]) -> int:
    points = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in points:
        cur += d
        best = max(best, cur)
    return best


@pytest.mark.parametrize("make_policy", POLICIES)
def test_energy_is_conserved(env, make_policy):
    res = Simulator(env, tiny_config(), make_policy()).run()
    assert abs(res.energy_residual_kwh) < 1e-6
    assert res.energy["charged_battery"] > 0  # the scenario forces charging


@pytest.mark.parametrize("make_policy", POLICIES)
def test_no_vehicle_is_stranded(env, make_policy):
    res = Simulator(env, tiny_config(), make_policy()).run()
    assert res.energy["depletions"] == 0


@pytest.mark.parametrize("make_policy", POLICIES)
def test_request_accounting_and_waits(env, make_policy):
    cfg = tiny_config()
    res = Simulator(env, cfg, make_policy()).run()
    req = res.requests
    assert not (req.served & req.lost).any()
    served = req[req.served & req.wait_s.notna()]
    assert (served.wait_s >= 0).all()
    assert (served.wait_s <= cfg.max_wait_s + 1e-6).all()
    # every request is either served or lost once its patience has expired
    assert (req.served | req.lost).all()


@pytest.mark.parametrize("make_policy", POLICIES)
def test_charger_capacity_never_exceeded(env, make_policy):
    cfg = tiny_config()
    res = Simulator(env, cfg, make_policy()).run()
    ses = res.sessions
    chargers = [n for _, n in sorted(cfg.depots.items())]
    for depot, group in ses.groupby("depot"):
        # charger held from plug-in (plug_start - overhead) to end; the 1e-6 s
        # shrink absorbs float rounding when one session starts as another ends
        intervals = list(zip(group.plug_start - 120.0 + 1e-6, group.end, strict=True))
        assert max_concurrent(intervals) <= chargers[depot]
    assert (ses.soc1 <= 1.0 + 1e-9).all() and (ses.soc0 >= 0).all()
    assert (ses.queue_wait_s >= -1e-6).all()


def test_runs_are_deterministic(env):
    a = Simulator(env, tiny_config(), QueueAwarePolicy()).run()
    b = Simulator(env, tiny_config(), QueueAwarePolicy()).run()
    pd.testing.assert_frame_equal(a.requests, b.requests)
    assert a.energy == b.energy


def test_policies_see_identical_demand(env):
    a = Simulator(env, tiny_config(seed=5), ThresholdPolicy()).run()
    b = Simulator(env, tiny_config(seed=5), OrchestratedPolicy()).run()
    np.testing.assert_array_equal(a.requests.t.to_numpy(), b.requests.t.to_numpy())
    np.testing.assert_array_equal(a.requests.origin.to_numpy(), b.requests.origin.to_numpy())


def test_outage_stops_charging_at_that_depot(env):
    cfg = tiny_config(outages=[(10, "2025-04-21 12:00", "2025-04-21 18:00")])
    res = Simulator(env, cfg, QueueAwarePolicy()).run()
    t0 = pd.Timestamp(cfg.start)
    a = (pd.Timestamp("2025-04-21 12:00") - t0).total_seconds()
    b = (pd.Timestamp("2025-04-21 18:00") - t0).total_seconds()
    depot0 = res.sessions[res.sessions.depot == 0]  # zone 10 sorts first
    overlapping = depot0[(depot0.plug_start < b) & (depot0.end > a + 1e-6)]
    # sessions that were running at the outage start end exactly at it
    assert ((overlapping.plug_start < a) & (np.isclose(overlapping.end, a))).all()
    assert abs(res.energy_residual_kwh) < 1e-6


def test_sampling_fraction_thins_demand(env):
    full = Simulator(env, tiny_config(sample_frac=1.0), ThresholdPolicy()).run()
    half = Simulator(env, tiny_config(sample_frac=0.5), ThresholdPolicy()).run()
    ratio = len(half.requests) / len(full.requests)
    assert 0.4 < ratio < 0.6


def test_bigger_fleet_serves_at_least_as_many(env):
    small = Simulator(env, tiny_config(fleet_size=12), QueueAwarePolicy()).run()
    large = Simulator(env, tiny_config(fleet_size=40), QueueAwarePolicy()).run()
    assert large.requests.served.mean() >= small.requests.served.mean()


def test_state_hours_cover_the_measured_window(env):
    cfg = tiny_config()
    res = Simulator(env, cfg, OrchestratedPolicy()).run()
    hours = (pd.Timestamp(cfg.end) - pd.Timestamp(cfg.warmup_end)).total_seconds() / 3600
    assert sum(res.state_hours.values()) == pytest.approx(hours * cfg.fleet_size, rel=1e-9)
