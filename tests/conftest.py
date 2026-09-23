"""Shared synthetic fixtures: a four-zone 'line city' that needs no downloaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evfleet.sim.engine import Environment, SimConfig
from evfleet.sim.network import N_BUCKETS, Network

ZONES = np.array([10, 20, 30, 40])


def line_network() -> Network:
    k = np.arange(len(ZONES))
    gap = np.abs(k[:, None] - k[None, :])
    base = 60.0 + 180.0 * gap  # seconds
    dist = 0.5 + 1.5 * gap  # km
    factor = np.ones(N_BUCKETS)
    factor[7:10] = 1.3  # weekday morning congestion
    return Network(ZONES, base, dist, factor, np.ones_like(base, dtype=bool))


def synthetic_trips(days: int = 2, per_hour: float = 40.0, seed: int = 0, start: str = "2025-04-21") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horizon = days * 86400
    n = rng.poisson(per_hour * horizon / 3600)
    t = np.sort(rng.uniform(0, horizon, n))
    pu = rng.choice(ZONES, n)
    do = rng.choice(ZONES, n)
    gap = np.abs(np.searchsorted(ZONES, pu) - np.searchsorted(ZONES, do))
    dist = 0.8 + 1.6 * gap + rng.exponential(0.5, n)
    dur = 120 + 200 * gap + rng.exponential(60, n)
    ts = pd.Timestamp(start) + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({"pickup_ts": ts, "dropoff_ts": ts + pd.to_timedelta(dur, unit="s"),
                         "pu": pu, "do_zone": do, "distance_km": dist, "duration_s": dur})


def tiny_environment(days: int = 2, per_hour: float = 40.0, seed: int = 0) -> Environment:
    net = line_network()
    hours = pd.date_range("2025-04-20", periods=24 * (days + 3), freq="h")
    price = pd.Series(40 + 15 * np.sin(2 * np.pi * (hours.hour - 13) / 24), index=hours)
    return Environment(
        network=net,
        trips=synthetic_trips(days, per_hour, seed),
        zone_profile=np.full((168, len(ZONES)), per_hour / len(ZONES)),
        fleet_profile=np.full(672, per_hour / 4),
        dropoff_share=np.full(len(ZONES), 1 / len(ZONES)),
        temperature=pd.Series(20.0, index=hours),
        price=price,
    )


def tiny_config(**kw) -> SimConfig:
    base = dict(fleet_size=30, depots={10: 2, 40: 2}, start="2025-04-21", warmup_end="2025-04-21 06:00",
                end="2025-04-23", sample_frac=1.0, seed=3, battery_kwh=20.0, init_soc=(0.25, 0.6))
    base.update(kw)
    return SimConfig(**base)


@pytest.fixture
def env() -> Environment:
    return tiny_environment()
