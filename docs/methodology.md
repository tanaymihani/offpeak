# Methodology, assumptions and limitations

This document explains how I produced each result, what I assumed, and what the results do not show. Every number in the README comes from a CSV file in `reports/results/`, and each of those files is written by a script in `scripts/`.

## 1. Data

| Source | What I use it for | Licence or terms |
|---|---|---|
| City of Palo Alto, *EV Charging Station Usage, July 2011 to December 2020* (ChargePoint export, 259,415 sessions) | Session analysis, prediction, segmentation, fee analysis, congestion | Open Data Commons PDDL |
| NYC TLC yellow-taxi trip records, April 2025 (3.97 million trips) | Ride requests for the simulator and the travel-time model | NYC Open Data terms of use |
| NYC TLC taxi zone shapefile | Zone centroids, areas and maps | NYC Open Data terms of use |
| NYISO day-ahead zonal LBMP, April 2025, N.Y.C. zone | Hourly electricity prices | NYISO public market data |
| Open-Meteo historical weather (hourly temperature and precipitation) | Weather features and the vehicle heating and cooling load | CC BY 4.0 |

`python -m evfleet.data.download` downloads every file and records its URL, size, SHA-256 and download time in `data/raw/MANIFEST.json`. It never overwrites a raw file, and I do not commit raw data. The cleaning is written in SQL (`sql/`), and every raw row gets exactly one outcome, either kept or one named drop reason. That way the exclusions add up to the raw row count (`reports/results/pa_data_quality.csv` and `nyc_trips_data_quality.csv`).

## 2. Charging-session study (Palo Alto)

**Features as of plug-in time** (`sql/session_features.sql`). A driver's history only uses earlier sessions. If any earlier session of the same driver had not ended yet, its duration and energy were not known at plug-in, so I set all history features to missing for that row (0.4% of rows). I leave out the session's own fee because it is billed per kWh at the end, so it would leak the energy target. The only fee-related feature is whether the session falls before or after the fee started. `tests/test_features.py` checks this with a future-intervention test: changing a later session must not change any earlier features.

**Splits in time order.** Training is 2011 to 2018, calibration is January to June 2019, the test set is July to December 2019, and 2020 is a separate shift test because of the COVID shelter-in-place.

**Models.** I compare three baselines (the global median, the median by site, weekend and hour, and the driver's own median history) with histogram gradient-boosted quantile regression at the 10th, 50th and 90th percentiles, on log connection time and on log(1 + energy). I fixed the hyperparameters in advance and did not search over them. The 80% intervals use conformalized quantile regression (CQR), calibrated on the calibration half-year and evaluated on the test half-year. Conformal methods do not guarantee coverage for each subgroup, so I report coverage by segment. For 2020, a nightly recalibration uses the latest 3,000 sessions whose outcome was known before midnight, since a stay is only observed once the car leaves.

**Segmentation.** K-means on standardized arrival time (sine and cosine of the hour), log connection time, log energy and log idle time. K = 5 has the highest silhouette score for K from 3 to 8 (0.29, so the segments overlap) and is stable across seeds (adjusted Rand index 0.997). I tried a Gaussian mixture first and dropped it: its BIC kept improving up to the largest K I tried, and the binary and point-mass features produced degenerate components. The segment names come from the centroid statistics through a fixed rule (`evfleet.sessions.segments.name_segment`).

**Fee analysis.** Every station went from free to $0.23/kWh on 1 August 2017, so there is no untreated station. I report three designs side by side.

1. *Interrupted time series* on a daily panel of stations that operated throughout, with separate linear trends before and after, day-of-week, holiday and weather controls, and Newey-West standard errors with 14 lags. I use windows of 60, 90, 120 and 180 days on each side.
2. *Year-over-year difference-in-differences.* I compare each day in 2017 with the same weekday 364 days earlier, and the effect is the change in that growth rate after August. The placebo repeats the same analysis for 2016 against 2015, with a fake fee date of 1 August 2016.
3. *Driver cohorts.* Among drivers who charged between February and July, I look at the share who still charged between August and December, and at how often and how much the remaining drivers charged, comparing 2017 with 2016 (placebo: 2016 with 2015) with bootstrap intervals over drivers. This design fails its placebo, because the fake 2016 effect on how often the remaining drivers charged is as large as the 2017 one. I report it but do not interpret it.

The year-over-year placebo is not zero either (for example +10% for sessions per day). Year-over-year growth can shift that much without any intervention, so I read the estimates with that design uncertainty in mind, not only their standard errors.

The main assumption is that without the fee, 2017 would have followed the 2016 seasonal pattern around a smooth trend. The fee priced these public stations only. Drivers may have moved to workplace or home charging that the data cannot see, so the effects are on demand at the city's stations, not on EV charging overall. Other changes in 2017, such as the mix of EVs or new private chargers, are not controlled separately.

**Congestion** (`evfleet.sessions.utilization`). I sample the number of connected cars at each site every 15 minutes in 2019. A site is full when that number reaches the ports in service. Ports come and go during the year, so for each month I count only the station ports that had at least one session that month. A port that stopped working partway through a month still counts for that month, so the share of time full is slightly understated. Drivers who arrive at a full site leave no record, so the share of time full is a lower bound on the demand a site could not serve.

## 3. Robotaxi fleet simulator (NYC)

**Service area.** The 66 Manhattan taxi zones, without the harbour islands. I started with the airports included and removed them. Airport pickups arrive in waves 30 to 50 minutes' drive from the fleet, so even with long-range staging, 42 to 48% of airport pickups went unserved in validation runs with the threshold baseline, while 98 to 100% of Manhattan pickups were served in the same runs. Airport staging is its own problem, and it would have dominated the service numbers.

**Demand.** Every cleaned Manhattan-internal yellow-taxi trip is a ride request at its recorded pickup time, origin and destination, with its recorded duration and distance. The experiments use a 20% random sample of demand, with the fleet and chargers sized for it. Each seed draws a different sample, and all policies see the same sample for a given seed, so the comparisons are paired. One full-scale run (100% of demand, five times the fleet and chargers) checks that the conclusions do not depend on the sampling. Taxi trips are the demand that was served, not all the demand there was, because unserved hails are not in the data.

**Travel times for empty driving.** A log-additive model, log time = zone-pair effect + weekday/weekend by hour congestion effect, fitted by median polish on the training weeks (1 to 21 April 2025). Zone pairs with too few trips fall back to a regression on log centroid distance. I validate it on 0.95 million held-out trips from 22 to 30 April (`reports/results/nyc_travel_time_validation.csv`). Riders' own trips use their recorded duration and distance.

**Energy** (`evfleet.sim.energy`). Energy per km is a + b v² + (P_aux + P_hvac(T)) / v, where v is the average speed and T the outside temperature. The rolling and stop-and-go term is a = 0.095 kWh/km, b comes from an aerodynamic drag area of 0.55 m², the always-on computer and auxiliary load is 0.35 kW, a heat pump runs outside an 18 to 24°C band, and a parked car that is awake draws 0.25 kW. These are my assumptions for a compact electric robotaxi, not a fit to telemetry, because no public telemetry exists. The temperature is the real hourly temperature for the simulated days.

**Charging** (`evfleet.sim.charging`). 40 kWh usable battery, 150 kW DC chargers, and a piecewise-linear taper in C-rate (2.5C up to 50% charge, 1.2C at 80%, 0.2C at 100%), with 92% grid-to-battery efficiency and a 2-minute plug overhead. I integrate the charging time exactly from the curve and use its inverse when a car is unplugged early.

**Dispatch and rebalancing** are the same for every policy. A request goes to the nearest idle car that can reach the rider within 10 minutes and still has enough energy for the trip, a drive to the nearest charger and a 5% reserve. Unmatched riders wait up to 10 minutes. Every 15 minutes, idle cars are moved toward forecast demand with a minimum-cost assignment.

**Policies** (`evfleet.sim.policies`). A threshold rule that charges from 20% to 80%, the same rule charging to 100%, queue-aware depot choice (earliest predicted plug-in time, first come first served), and an orchestrated policy. The orchestrated policy tops up surplus idle cars in demand troughs and cheap day-ahead hours, keeps one charger per depot free for low-battery arrivals, lowers its trigger when cars are scarce, assigns cars to chargers with the Hungarian algorithm, and can unplug a part-charged car for a waiting rider.

**How I tuned it.** I chose the policy parameters and the base scenario (550 cars, 36 chargers, 6 p-median sites) on the validation week (13 to 20 April). The reported results are on the held-out test week (22 to 28 April, with 21 April as warm-up). The demand forecaster, the travel-time model and the siting were fitted on 1 to 21 April only.

**Siting.** I solve the p-median problem exactly as a MILP (SciPy with HiGHS), weighting zones by drop-offs, and split the chargers in proportion to the demand each site serves.

**Correctness checks** (`tests/test_sim.py`, run in CI). Energy conservation to 1e-6 kWh per run, no car running out of charge, request accounting, waits within patience, charger capacity never exceeded, determinism, identical demand across policies, outage behaviour and time accounting. Every experiment run also reports its energy-ledger error and how many cars ran out of charge.

**Statistics.** Each seed is an independent replicate. I pair the differences from the baseline by seed and report means with 95% t-intervals. All seeds use the same real week of demand, so the intervals describe simulation variability for that week, not variation from week to week.

### What the simulator does not capture

- Road-network routing, and congestion caused by the fleet itself. Travel times are taken from taxi data.
- Ride pooling, rider choice, how prices affect demand, and cancellations other than a fixed patience.
- Real vehicle telemetry, battery wear, and the effect of temperature on charging speed.
- Demand charges and distribution tariffs. Energy cost uses the wholesale day-ahead price only, which is why I report peak charging load separately.
- Charger faults other than the scripted outage, and per-site power limits.

## 4. Reproducing

```bash
make setup      # virtual environment and package
make data       # about 160 MB download, hashed manifest, processed tables
make sessions   # Palo Alto study and fleet input checks (about 3 minutes)
make fleet      # fleet experiments (10 to 15 minutes on 6 cores)
python scripts/run_fleet_ablation.py
python scripts/run_fleet_profiles.py
make figures
python scripts/summarize_results.py
make test
```
