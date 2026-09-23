"""Hour-of-day charging load, price and demand for the base scenario (test week).

Saves reports/results/fleet_hourly_profiles.csv, averaging seeds 0-2, used by
the load-shifting figure. Same configuration as experiment E1.

    python scripts/run_fleet_profiles.py
"""

from __future__ import annotations

import pandas as pd

from evfleet.paths import RESULTS, ensure_dirs
from evfleet.sim.engine import Simulator
from evfleet.sim.experiment import POLICIES, base_config, build_environment

BASE = dict(fleet=550, chargers=36, sites=6)


def main() -> None:
    ensure_dirs()
    env = build_environment()
    rows = []
    for policy in ("threshold_80", "threshold_100", "queue_aware", "orchestrated"):
        for seed in range(3):
            cfg = base_config(env, **BASE, seed=seed)
            res = Simulator(env, cfg, POLICIES[policy]()).run()
            load = res.load_kw[res.load_kw.index >= pd.Timestamp(cfg.warmup_end)]
            by_hour = (load / 1000).groupby(load.index.hour).mean()
            req = res.requests
            hour = ((req.t // 3600) % 24).astype(int)
            demand = req.groupby(hour).size() / 7
            lost = req.groupby(hour).lost.mean()
            for h in range(24):
                rows.append({"policy": policy, "seed": seed, "hour": h, "charging_mw": by_hour.get(h, 0.0),
                             "requests_per_day": demand.get(h, 0.0), "lost_rate": lost.get(h, 0.0)})
    df = pd.DataFrame(rows).groupby(["policy", "hour"], as_index=False).mean(numeric_only=True).drop(columns="seed")
    week = env.price[(env.price.index >= "2025-04-22") & (env.price.index < "2025-04-29")]
    price = week.groupby(week.index.hour).mean().rename("lbmp_usd_mwh")
    df = df.merge(price, left_on="hour", right_index=True)
    df.to_csv(RESULTS / "fleet_hourly_profiles.csv", index=False)
    print(df.pivot(index="hour", columns="policy", values="charging_mw").round(2).to_string())


if __name__ == "__main__":
    main()
