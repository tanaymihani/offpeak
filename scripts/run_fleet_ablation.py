"""Which parts of the orchestrator produce its gains? Remove one component at a time.

Runs on the test week with 6 paired seeds; this explains the mechanism behind the
E1 result and is not used to choose any parameter. Output:
reports/results/fleet_ablation.csv (paired differences vs threshold_80 and vs the full orchestrator).

    python scripts/run_fleet_ablation.py
"""

from __future__ import annotations

import pandas as pd

from evfleet.paths import RESULTS, ensure_dirs
from evfleet.sim.experiment import base_config, build_environment, run_grid

VARIANTS = ["threshold_80", "orchestrated", "orch_no_topups", "orch_no_scarce_trigger", "orch_no_preemption",
            "orch_no_price_signal", "orch_no_reserve", "orch_topups_only"]
METRICS = ["service_rate", "wait_mean_min", "energy_cost_usd_per_day", "avg_price_usd_mwh", "peak_charging_mw",
           "queue_wait_mean_min", "sessions_per_vehicle_day"]


def main() -> None:
    ensure_dirs()
    env = build_environment()
    jobs = [(v, base_config(env, fleet=550, chargers=36, sites=6, seed=s), "E8_ablation")
            for v in VARIANTS for s in range(6)]
    runs = run_grid(jobs, processes=5)
    runs.to_csv(RESULTS / "fleet_ablation_runs.csv", index=False)
    rows = []
    for ref in ("threshold_80", "orchestrated"):
        base = runs[runs.policy == ref].set_index("seed")
        for v in VARIANTS:
            if v == ref:
                continue
            h = runs[runs.policy == v].set_index("seed")
            for m in METRICS:
                d = h[m] - base[m]
                rows.append({"reference": ref, "variant": v, "metric": m, "variant_mean": h[m].mean(),
                             "diff_mean": d.mean(), "diff_sd": d.std(ddof=1), "n_seeds": len(d)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "fleet_ablation.csv", index=False)
    print(runs.groupby("policy")[METRICS].mean().round(4).loc[VARIANTS].to_string())
    print("depletions:", runs.depletions.sum(), "max residual:", runs.energy_residual_kwh.abs().max())


if __name__ == "__main__":
    main()
