"""Build the simulation environment, site depots, run policies and summarise runs.

Everything the simulator "knows in advance" (travel times, demand profiles,
depot locations) is estimated on the training weeks, 1-21 April 2025. The
test week is 22-28 April 2025, with 21 April as warm-up.
"""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from multiprocessing import get_context

import numpy as np
import pandas as pd

from evfleet.data.download import RAW
from evfleet.data.nyiso import load_nyc_lbmp
from evfleet.data.tlc import load_trips, load_zones
from evfleet.data.weather import load_hourly
from evfleet.forecast.demand import fleet_profile, slot_counts, zone_hour_profile
from evfleet.optimize.siting import allocate_chargers, assignment_cost, p_median
from evfleet.sim.engine import Environment, SimConfig, SimResult, Simulator
from evfleet.sim.network import fit_network, time_bucket
from evfleet.sim.policies import OrchestratedPolicy, QueueAwarePolicy, ThresholdPolicy

TRAIN = ("2025-04-01", "2025-04-22")
SIM_WINDOW = ("2025-04-13", "2025-04-29")  # validation week 13-20, test week 21-28
VALIDATION = {"start": "2025-04-13", "warmup_end": "2025-04-14", "end": "2025-04-21"}
TEST = {"start": "2025-04-21", "warmup_end": "2025-04-22", "end": "2025-04-29"}

POLICIES = {
    "threshold_80": lambda: ThresholdPolicy(0.20, 0.80),
    "threshold_100": lambda: ThresholdPolicy(0.20, 1.00),
    "queue_aware": lambda: QueueAwarePolicy(0.20, 0.80),
    "orchestrated": lambda: OrchestratedPolicy(),
    # ablations: the orchestrator with one component removed
    "orch_no_topups": lambda: OrchestratedPolicy(topups=False, name="orch_no_topups"),
    "orch_no_scarce_trigger": lambda: OrchestratedPolicy(scarce_threshold=0.20, name="orch_no_scarce_trigger"),
    "orch_no_preemption": lambda: OrchestratedPolicy(preempt_soc=None, name="orch_no_preemption"),
    "orch_no_price_signal": lambda: OrchestratedPolicy(soc_ceiling=(0.55, 0.55, 0.55), name="orch_no_price_signal"),
    "orch_no_reserve": lambda: OrchestratedPolicy(reserve_per_depot=0, name="orch_no_reserve"),
    "orch_topups_only": lambda: OrchestratedPolicy(scarce_threshold=0.20, preempt_soc=None,
                                                   name="orch_topups_only"),
}


def build_environment() -> Environment:
    zones = load_zones()
    zones = zones[zones.in_service_area].sort_values("zone_id").reset_index(drop=True)
    train = load_trips(*TRAIN)
    net = fit_network(train, zones)
    weeks = (pd.Timestamp(TRAIN[1]) - pd.Timestamp(TRAIN[0])).days / 7
    zprof = zone_hour_profile(train, net.zone_ids, weeks)
    fprof = fleet_profile(slot_counts(train, *TRAIN))
    late = train[(train.dropoff_ts.dt.hour >= 23) | (train.dropoff_ts.dt.hour < 1)]
    share = late["do_zone"].map(net.index).value_counts().reindex(range(net.n), fill_value=0).to_numpy(float)
    share = (share + 1) / (share + 1).sum()
    weather = load_hourly(RAW / "open_meteo_nyc_2025-04.json")
    return Environment(
        network=net, trips=load_trips(*SIM_WINDOW), zone_profile=zprof, fleet_profile=fprof,
        dropoff_share=share, temperature=weather["temp_c"], price=load_nyc_lbmp(),
    )


def siting_inputs(env: Environment, train: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cost = demand-weighted mean travel time over the day; weight = drop-offs per zone."""
    net = env.network
    train = load_trips(*TRAIN) if train is None else train
    b = pd.Series(time_bucket(train["pickup_ts"])).value_counts(normalize=True)
    cost = sum(net.tt[k] * w for k, w in b.items())
    weights = train["do_zone"].map(net.index).value_counts().reindex(range(net.n), fill_value=0).to_numpy(float)
    return cost, weights


def depot_layout(env: Environment, p: int, total_chargers: int, method: str = "p_median",
                 min_per_site: int = 2) -> dict[int, int]:
    cost, weights = siting_inputs(env)
    if method == "p_median":
        sites, _ = p_median(cost, weights, p)
    elif method == "busiest":
        sites = sorted(np.argsort(-weights)[:p].tolist())
    else:
        raise ValueError(method)
    alloc = allocate_chargers(cost, weights, sites, total_chargers, min_per_site)
    return {int(env.network.zone_ids[s]): int(n) for s, n in zip(sites, alloc, strict=True)}


def layout_cost_minutes(env: Environment, depots: dict[int, int]) -> float:
    """Mean travel time (minutes) from a drop-off to its nearest depot."""
    cost, weights = siting_inputs(env)
    sites = [env.network.index[z] for z in depots]
    return assignment_cost(cost, weights, sites) / weights.sum() / 60.0


def summarize(res: SimResult, price: pd.Series) -> dict:
    cfg = res.config
    req = res.requests
    days = (pd.Timestamp(cfg.end) - pd.Timestamp(cfg.warmup_end)).days
    served = req[req.served]
    wait = served["wait_s"].dropna() / 60.0
    sh = res.state_hours
    fleet_h = sum(sh.values())
    km = res.km
    empty_km = km["to_pickup"] + km["to_depot"] + km["reposition"]
    ses = res.sessions
    ses_m = ses[ses["end"] > (pd.Timestamp(cfg.warmup_end) - pd.Timestamp(cfg.start)).total_seconds()]
    load = res.load_kw[res.load_kw.index >= pd.Timestamp(cfg.warmup_end)]
    kwh = load * (5 / 60)
    lbmp = price.reindex(load.index.floor("h")).to_numpy()
    cost = float((kwh.to_numpy() * lbmp / 1000.0).sum())
    grid_kwh = float(kwh.sum())
    return {
        "policy": res.policy, "seed": cfg.seed, "fleet": cfg.fleet_size, "chargers": sum(cfg.depots.values()),
        "sites": len(cfg.depots), "battery_kwh": cfg.battery_kwh, "sample_frac": cfg.sample_frac,
        "requests": len(req), "service_rate": req["served"].mean(), "lost_rate": req["lost"].mean(),
        "wait_mean_min": wait.mean(), "wait_p50_min": wait.quantile(0.5), "wait_p95_min": wait.quantile(0.95),
        "trips_per_vehicle_day": req["served"].sum() / cfg.fleet_size / days,
        "util_with_rider": sh["with_rider"] / fleet_h,
        "share_charging_or_queued": (sh["charging"] + sh["queued"] + sh["to_depot"]) / fleet_h,
        "share_idle": sh["idle"] / fleet_h,
        "empty_km_share": empty_km / (empty_km + km["with_rider"]),
        "charge_km_per_vehicle_day": km["to_depot"] / cfg.fleet_size / days,
        "sessions_per_vehicle_day": len(ses_m) / cfg.fleet_size / days,
        "queue_wait_mean_min": ses_m["queue_wait_s"].mean() / 60 if len(ses_m) else 0.0,
        "queue_wait_p95_min": ses_m["queue_wait_s"].quantile(0.95) / 60 if len(ses_m) else 0.0,
        "grid_mwh_per_day": grid_kwh / 1000 / days,
        "energy_cost_usd_per_day": cost / days,
        "avg_price_usd_mwh": cost / grid_kwh * 1000 if grid_kwh else np.nan,
        "peak_charging_mw": float(load.max()) / 1000,
        "depletions": res.energy["depletions"],
        "energy_residual_kwh": res.energy_residual_kwh,
        "runtime_s": res.runtime_s, "events": res.events,
    }


# ------------------------------------------------------------- parallel runs
_ENV: Environment | None = None


def _init_worker() -> None:
    global _ENV
    _ENV = build_environment()


def _run_one(job: tuple[str, dict]) -> dict:
    policy_name, cfg_dict = job
    d = dict(cfg_dict)
    label = d.pop("_label", "")
    env = _ENV if _ENV is not None else build_environment()
    res = Simulator(env, SimConfig(**d), POLICIES[policy_name]()).run()
    out = summarize(res, env.price)
    out["label"] = label
    return out


def run_grid(jobs: list[tuple[str, SimConfig, str]], processes: int | None = None) -> pd.DataFrame:
    """Run (policy, config, label) jobs in parallel; one environment per worker."""
    payload = []
    for policy, cfg, label in jobs:
        d = asdict(cfg)
        d["_label"] = label
        payload.append((policy, d))
    procs = processes or max(1, min(len(payload), (os.cpu_count() or 2) - 1))
    if procs == 1:
        _init_worker()
        rows = [_run_one(j) for j in payload]
    else:
        with get_context("spawn").Pool(procs, initializer=_init_worker) as pool:
            rows = pool.map(_run_one, payload, chunksize=1)
    return pd.DataFrame(rows)


def base_config(env: Environment, fleet: int, chargers: int, sites: int = 6, layout: str = "p_median",
                **kw) -> SimConfig:
    depots = depot_layout(env, sites, chargers, method=layout)
    period = kw.pop("period", TEST)
    return replace(SimConfig(fleet_size=fleet, depots=depots, **period), **kw)
