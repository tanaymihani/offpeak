"""Compute the headline numbers quoted in the README from reports/results/*.csv.

Writes reports/results/headline_numbers.csv (name, value, source) so every
number in the README can be traced to a result file.

    python scripts/summarize_results.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evfleet.paths import RESULTS

rows: list[dict] = []


def put(name: str, value, source: str) -> None:
    rows.append({"name": name, "value": value, "source": source})


def fleet_needed(runs: pd.DataFrame, policy: str, target: float) -> float:
    r = runs[runs.label.str.startswith("E3_fleet_") & (runs.policy == policy)].copy()
    r["fleet_n"] = r.label.str.rsplit("_", n=1).str[-1].astype(int)
    curve = r.groupby("fleet_n").service_rate.mean()
    if curve.max() < target:
        return float("nan")
    return float(np.interp(target, curve.to_numpy(), curve.index.to_numpy()))


def main() -> None:
    runs = pd.read_csv(RESULTS / "fleet_runs.csv")
    paired = pd.read_csv(RESULTS / "fleet_paired.csv")
    src = "fleet_runs.csv"
    put("fleet_runs_total", len(runs) + len(pd.read_csv(RESULTS / "fleet_ablation_runs.csv")), src)
    put("fleet_runs_depletions", int(runs.depletions.sum()), src)
    put("fleet_runs_max_energy_residual_kwh", float(runs.energy_residual_kwh.abs().max()), src)

    e1 = runs[runs.label == "E1_main"]
    put("e1_requests_per_run", int(e1.requests.mean()), src)
    for pol, g in e1.groupby("policy"):
        put(f"e1_service_{pol}", g.service_rate.mean(), src)
        put(f"e1_cost_usd_day_{pol}", g.energy_cost_usd_per_day.mean(), src)
        put(f"e1_peak_mw_{pol}", g.peak_charging_mw.mean(), src)
        put(f"e1_wait_min_{pol}", g.wait_mean_min.mean(), src)
        put(f"e1_queue_min_{pol}", g.queue_wait_mean_min.mean(), src)
        put(f"e1_avg_price_{pol}", g.avg_price_usd_mwh.mean(), src)
    p = paired[paired.label == "E1_main"].set_index(["policy", "metric"])
    for pol in ("orchestrated", "queue_aware", "threshold_100"):
        for m in ("service_rate", "wait_mean_min", "energy_cost_usd_per_day", "peak_charging_mw",
                  "queue_wait_mean_min", "avg_price_usd_mwh", "empty_km_share"):
            r = p.loc[(pol, m)]
            put(f"e1_diff_{pol}_{m}", r.diff_mean, "fleet_paired.csv")
            put(f"e1_diff_{pol}_{m}_ci", f"[{r.ci95_low:.4g}, {r.ci95_high:.4g}]", "fleet_paired.csv")
            put(f"e1_rel_{pol}_{m}", r.rel_diff, "fleet_paired.csv")

    fs = runs[runs.label == "E7_full_scale"].set_index("policy")
    for pol in fs.index:
        put(f"full_scale_service_{pol}", fs.loc[pol, "service_rate"], src)
        put(f"full_scale_cost_usd_day_{pol}", fs.loc[pol, "energy_cost_usd_per_day"], src)
        put(f"full_scale_runtime_s_{pol}", fs.loc[pol, "runtime_s"], src)
        put(f"full_scale_requests_{pol}", int(fs.loc[pol, "requests"]), src)
        put(f"full_scale_fleet_{pol}", int(fs.loc[pol, "fleet"]), src)

    for target in (0.95, 0.96):
        a, b = fleet_needed(runs, "threshold_80", target), fleet_needed(runs, "orchestrated", target)
        put(f"fleet_needed_{int(target * 100)}_threshold_80", a, src)
        put(f"fleet_needed_{int(target * 100)}_orchestrated", b, src)
        put(f"fleet_saving_{int(target * 100)}_pct", 1 - b / a, src)

    e2 = runs[runs.label.str.startswith("E2_chargers_")].copy()
    e2["chargers_n"] = e2.label.str.rsplit("_", n=1).str[-1].astype(int)
    tab = e2.groupby(["policy", "chargers_n"]).service_rate.mean()
    for (pol, c), v in tab.items():
        put(f"e2_service_{pol}_{c}", v, src)

    e5 = runs[runs.label.str.startswith("E5_") & (runs.policy == "queue_aware")]
    for label, g in e5.groupby("label"):
        put(f"e5_charge_km_per_vehicle_day_{label[3:]}", g.charge_km_per_vehicle_day.mean(), src)
        put(f"e5_service_{label[3:]}", g.service_rate.mean(), src)
    lay = pd.read_csv(RESULTS / "fleet_layouts.csv").set_index("layout")
    for k in lay.index:
        put(f"layout_minutes_to_depot_{k}", lay.loc[k, "mean_minutes_to_nearest_depot"], "fleet_layouts.csv")

    seeds = sorted(runs[runs.label == "E4_cold_minus5C"].seed.unique())
    st = runs[runs.seed.isin(seeds) & (runs.label.str.startswith("E4_") | (runs.label == "E1_main"))]
    for (label, pol), g in st.groupby(["label", "policy"]):
        put(f"stress_service_{label}_{pol}", g.service_rate.mean(), src)
        put(f"stress_cost_{label}_{pol}", g.energy_cost_usd_per_day.mean(), src)

    ab = pd.read_csv(RESULTS / "fleet_ablation_runs.csv").groupby("policy")
    for pol, g in ab:
        put(f"ablation_service_{pol}", g.service_rate.mean(), "fleet_ablation_runs.csv")
        put(f"ablation_cost_{pol}", g.energy_cost_usd_per_day.mean(), "fleet_ablation_runs.csv")

    tt = pd.read_csv(RESULTS / "nyc_travel_time_validation.csv").set_index("predictor")
    for k in tt.index:
        put(f"travel_time_mae_min_{k.split(' ')[0]}", tt.loc[k, "mae_min"], "nyc_travel_time_validation.csv")
        put(f"travel_time_trips_{k.split(' ')[0]}", int(tt.loc[k, "trips"]), "nyc_travel_time_validation.csv")
    fc = pd.read_csv(RESULTS / "nyc_demand_forecast_eval.csv").set_index("method")
    for k in fc.index:
        put(f"forecast_mape_{k.split(' ')[0]}", fc.loc[k, "mape"], "nyc_demand_forecast_eval.csv")

    # ---------------------------------------------------------------- Palo Alto
    q = pd.read_csv(RESULTS / "pa_data_quality.csv")
    put("pa_sessions_raw", int(q.rows.sum()), "pa_data_quality.csv")
    put("pa_sessions_kept", int(q.loc[q.outcome == "kept", "rows"].sum()), "pa_data_quality.csv")
    for target in ("dwell", "energy"):
        pt = pd.read_csv(RESULTS / f"pa_{target}_point.csv").set_index("model")
        for m in pt.index:
            put(f"pa_{target}_mae_{m}", pt.loc[m, "mae"], f"pa_{target}_point.csv")
        iv = pd.read_csv(RESULTS / f"pa_{target}_intervals.csv").set_index("method")
        for m in iv.index:
            put(f"pa_{target}_coverage_{m}", iv.loc[m, "coverage"], f"pa_{target}_intervals.csv")
            put(f"pa_{target}_width_{m}", iv.loc[m, "mean_width"], f"pa_{target}_intervals.csv")
    sh = pd.read_csv(RESULTS / "pa_dwell_shift.csv")
    late = sh[sh.month >= "2020-07"]
    put("pa_shift_static_cov_jul_dec_2020", late.static_cqr_coverage.mean(), "pa_dwell_shift.csv")
    put("pa_shift_rolling_cov_jul_dec_2020", late.rolling_cqr_coverage.mean(), "pa_dwell_shift.csv")
    put("pa_shift_static_cov_min", sh.static_cqr_coverage.min(), "pa_dwell_shift.csv")
    put("pa_shift_rolling_cov_min", sh.rolling_cqr_coverage.min(), "pa_dwell_shift.csv")
    seg = pd.read_csv(RESULTS / "pa_dwell_segments.csv").set_index("segment")
    put("pa_dwell_coverage_anonymous", seg.loc["anonymous", "coverage"], "pa_dwell_segments.csv")

    eff = pd.read_csv(RESULTS / "pa_fee_effects.csv")
    for _, r in eff.iterrows():
        bw = f"_{int(r.bandwidth_days)}" if r.design == "its" else ""
        put(f"fee_{r.design}{bw}_{r.outcome}", r.pct_effect, "pa_fee_effects.csv")
        put(f"fee_{r.design}{bw}_{r.outcome}_ci", f"[{r.pct_ci_low:.3f}, {r.pct_ci_high:.3f}]", "pa_fee_effects.csv")
    fs2 = pd.read_csv(RESULTS / "pa_segments_fee_shift.csv").set_index("segment")
    for s_name, r in fs2.iterrows():
        put(f"segment_fee_change_{s_name}", float(np.exp(r.did_log_change) - 1), "pa_segments_fee_shift.csv")
    sel = pd.read_csv(RESULTS / "pa_segments_selection.csv")
    best = sel.loc[sel.silhouette.idxmax()]
    put("segments_k", int(best.k), "pa_segments_selection.csv")
    put("segments_silhouette", best.silhouette, "pa_segments_selection.csv")
    put("segments_stability_ari", best.stability_ari, "pa_segments_selection.csv")
    co = pd.read_csv(RESULTS / "pa_fee_cohorts.csv")
    for _, r in co.iterrows():
        put(f"cohort_{r.treated_year}_{r.metric}_did", r.did, "pa_fee_cohorts.csv")
        put(f"cohort_{r.treated_year}_{r.metric}_ci", f"[{r.ci_low:.3f}, {r.ci_high:.3f}]", "pa_fee_cohorts.csv")
    cg = pd.read_csv(RESULTS / "pa_site_congestion.csv", index_col=0)
    for site in cg.index[:3]:
        put(f"site_full_share_weekday_9_17_{site}", cg.loc[site, "weekday_9_17_share_full"], "pa_site_congestion.csv")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "headline_numbers.csv", index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
