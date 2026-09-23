"""Run every fleet experiment reported in the README and save tidy results.

All policy parameters were fixed on the validation week (13-20 April 2025);
everything here except E0 runs on the held-out test week (21-28 April 2025).
Outputs: reports/results/fleet_runs.csv (one row per run) and
reports/results/fleet_paired.csv (paired differences vs threshold_80).

    python scripts/run_fleet_experiments.py [--quick]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

from evfleet.paths import RESULTS, ensure_dirs
from evfleet.sim.experiment import VALIDATION, base_config, build_environment, layout_cost_minutes, run_grid

POLICIES = ["threshold_80", "threshold_100", "queue_aware", "orchestrated"]
BASE = dict(fleet=550, chargers=36, sites=6)


def jobs(env, seeds: int, quick: bool):
    s_main = range(3 if quick else 10)
    s_side = range(2 if quick else seeds)
    out = []
    for p in POLICIES:
        for s in range(3):
            out.append((p, base_config(env, **BASE, period=VALIDATION, seed=s), "E0_validation"))
        for s in s_main:
            out.append((p, base_config(env, **BASE, seed=s), "E1_main"))
    for c in (24, 30, 36, 48, 72):
        for p in ("threshold_80", "queue_aware", "orchestrated"):
            for s in s_side:
                out.append((p, base_config(env, fleet=550, chargers=c, sites=6, seed=s), f"E2_chargers_{c}"))
    for f in (450, 500, 550, 600, 650):
        for p in ("threshold_80", "orchestrated"):
            for s in s_side:
                out.append((p, base_config(env, fleet=f, chargers=36, sites=6, seed=s), f"E3_fleet_{f}"))
    biggest = max(base_config(env, **BASE).depots.items(), key=lambda kv: kv[1])[0]
    stress = {
        "E4_cold_minus5C": dict(temp_override_c=-5.0),
        "E4_outage_biggest_depot": dict(outages=[(biggest, "2025-04-24 16:00", "2025-04-24 22:00")]),
        "E4_demand_plus20pct": dict(demand_multiplier=1.2),
    }
    for label, kw in stress.items():
        for p in POLICIES:
            for s in s_side:
                out.append((p, base_config(env, **BASE, seed=s, **kw), label))
    for label, sites, layout in [("E5_pmedian_6", 6, "p_median"), ("E5_busiest_6", 6, "busiest"),
                                 ("E5_pmedian_1", 1, "p_median"), ("E5_pmedian_3", 3, "p_median"),
                                 ("E5_pmedian_12", 12, "p_median")]:
        for p in ("queue_aware", "orchestrated"):
            for s in s_side:
                cfg = base_config(env, fleet=550, chargers=36, sites=sites, layout=layout, seed=s)
                out.append((p, cfg, label))
    for kwh in (40, 60, 75):
        for p in ("threshold_80", "orchestrated"):
            for s in s_side:
                out.append((p, base_config(env, **BASE, seed=s, battery_kwh=float(kwh)), f"E6_battery_{kwh}"))
    return out


def paired(runs: pd.DataFrame, metrics: list[str], baseline: str = "threshold_80") -> pd.DataFrame:
    """Mean paired difference (policy - baseline) across seeds with a 95 % t-interval."""
    rows = []
    for label, g in runs.groupby("label"):
        if baseline not in set(g.policy):
            continue
        base = g[g.policy == baseline].set_index("seed")
        for pol, h in g.groupby("policy"):
            if pol == baseline:
                continue
            h = h.set_index("seed")
            common = base.index.intersection(h.index)
            for m in metrics:
                d = (h.loc[common, m] - base.loc[common, m]).to_numpy(float)
                n = len(d)
                half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                rows.append({"label": label, "policy": pol, "metric": m, "n_seeds": n,
                             "baseline_mean": base.loc[common, m].mean(), "diff_mean": d.mean(),
                             "ci95_low": d.mean() - half, "ci95_high": d.mean() + half,
                             "rel_diff": d.mean() / base.loc[common, m].mean() if base.loc[common, m].mean() else np.nan})
    return pd.DataFrame(rows)


def full_scale(env) -> pd.DataFrame:
    """One seed at 100 % of demand (fleet and chargers scaled x5) as a scale check."""
    js = [(p, base_config(env, fleet=2750, chargers=180, sites=6, seed=0, sample_frac=1.0), "E7_full_scale")
          for p in ("threshold_80", "orchestrated")]
    return run_grid(js, processes=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--skip-full-scale", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    env = build_environment()
    js = jobs(env, args.seeds, args.quick)
    print(f"{len(js)} runs")
    t = time.time()
    runs = run_grid(js, processes=args.processes)
    print(f"grid wall time {time.time() - t:.0f} s")
    if not args.skip_full_scale:
        t = time.time()
        runs = pd.concat([runs, full_scale(env)], ignore_index=True)
        print(f"full-scale wall time {time.time() - t:.0f} s")
    runs.to_csv(RESULTS / "fleet_runs.csv", index=False)
    metrics = ["service_rate", "wait_mean_min", "wait_p95_min", "queue_wait_mean_min", "energy_cost_usd_per_day",
               "avg_price_usd_mwh", "peak_charging_mw", "empty_km_share", "sessions_per_vehicle_day",
               "grid_mwh_per_day"]
    paired(runs, metrics).to_csv(RESULTS / "fleet_paired.csv", index=False)
    layouts = []
    for label, sites, layout in [("pmedian_6", 6, "p_median"), ("busiest_6", 6, "busiest"),
                                 ("pmedian_1", 1, "p_median"), ("pmedian_3", 3, "p_median"),
                                 ("pmedian_12", 12, "p_median")]:
        cfg = base_config(env, fleet=550, chargers=36, sites=sites, layout=layout)
        layouts.append({"layout": label, "depots": cfg.depots,
                        "mean_minutes_to_nearest_depot": layout_cost_minutes(env, cfg.depots)})
    pd.DataFrame(layouts).to_csv(RESULTS / "fleet_layouts.csv", index=False)
    bad = runs[(runs.depletions > 0) | (runs.energy_residual_kwh.abs() > 1e-3)]
    print("runs with depletions or energy residual:", len(bad))


if __name__ == "__main__":
    main()
