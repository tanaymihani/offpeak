"""Palo Alto charging-session study plus the fleet model's input validations.

Writes tidy CSVs to reports/results/:
  pa_dwell_*.csv, pa_energy_*.csv   prediction accuracy, intervals, segments, shift, importance
  pa_segments*.csv                  unsupervised segmentation and its change around the fee
  pa_fee_*.csv                      causal analysis of the 1 Aug 2017 fee
  pa_site_congestion*.csv           utilization / all-ports-busy by site and hour
  nyc_travel_time_validation.csv    travel-time model vs held-out real trips
  nyc_demand_forecast_eval.csv      demand forecaster vs baselines on held-out days

    python scripts/run_session_analysis.py
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from evfleet.data import palo_alto
from evfleet.data.tlc import load_trips, load_zones
from evfleet.forecast.demand import evaluate as eval_forecast
from evfleet.forecast.demand import slot_counts
from evfleet.paths import RESULTS, ensure_dirs
from evfleet.sessions import causal, models, segments, utilization
from evfleet.sim.network import fit_network, validate


def timed(label):
    class _T:
        def __enter__(self):
            self.t = time.time()

        def __exit__(self, *a):
            print(f"{label}: {time.time() - self.t:.0f} s")

    return _T()


STEPS = ["features", "models", "segments", "causal", "utilization", "fleet_inputs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=STEPS, default=STEPS)
    steps = set(ap.parse_args().only)
    ensure_dirs()
    if "features" in steps:
        with timed("palo alto features"):
            palo_alto.build()
    feats = palo_alto.load_features()

    if "models" in steps:
        with timed("prediction models"):
            for target in ("dwell", "energy"):
                out = models.evaluate(feats, target)
                for name, df in out.items():
                    if name == "test_pred":
                        df.sample(min(5000, len(df)), random_state=0).to_csv(
                            RESULTS / f"pa_{target}_test_sample.csv", index=False
                        )
                    else:
                        df.to_csv(RESULTS / f"pa_{target}_{name}.csv", index=False)

    if "segments" in steps:
        with timed("segmentation"):
            _, _, labels, selection, _ = segments.fit_segments(feats)
            prof = segments.describe(feats, labels)
            prof.to_csv(RESULTS / "pa_segments.csv")
            selection.to_csv(RESULTS / "pa_segments_selection.csv", index=False)
            _, did = segments.fee_shift(feats, labels, prof["name"].to_dict())
            did.to_csv(RESULTS / "pa_segments_fee_shift.csv", index=False)
            by_year = (
                feats.assign(segment=pd.Series(labels, index=feats.index).map(prof["name"]))
                .groupby([feats.start_ts.dt.year, "segment"])
                .size()
                .unstack(fill_value=0)
            )
            by_year.div(by_year.sum(axis=1), axis=0).to_csv(RESULTS / "pa_segments_by_year.csv")

    if "causal" in steps:
        with timed("causal analysis"):
            for name, df in causal.run_all(feats).items():
                df.to_csv(RESULTS / f"pa_fee_{name}.csv", index=name == "daily_panel")

    if "utilization" in steps:
        with timed("utilization"):
            summary, by_hour = utilization.congestion_table(feats)
            summary.to_csv(RESULTS / "pa_site_congestion.csv")
            by_hour.to_csv(RESULTS / "pa_site_full_by_hour.csv")

    if "fleet_inputs" in steps:
        with timed("fleet input validation"):
            zones = load_zones()
            zones = zones[zones.in_service_area]
            train, hold = load_trips("2025-04-01", "2025-04-22"), load_trips("2025-04-22", "2025-05-01")
            validate(fit_network(train, zones), hold).to_csv(
                RESULTS / "nyc_travel_time_validation.csv", index=False
            )
            eval_forecast(
                slot_counts(train, "2025-04-01", "2025-04-22"), slot_counts(hold, "2025-04-22", "2025-05-01")
            ).to_csv(RESULTS / "nyc_demand_forecast_eval.csv", index=False)


if __name__ == "__main__":
    main()
