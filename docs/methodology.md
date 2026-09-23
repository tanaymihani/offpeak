# Methodology, assumptions and limitations

This document records how each result was produced, what was assumed, and what the
results do **not** show. Numbers quoted in the README trace to CSV files in
`reports/results/`, which are produced by the scripts in `scripts/`.

## 1. Data

| Source | Use | Licence / terms |
|---|---|---|
| City of Palo Alto, *EV Charging Station Usage, Jul 2011 - Dec 2020* (ChargePoint export, 259,415 sessions) | Session analytics, prediction, segmentation, fee analysis | Open Data Commons PDDL |
| NYC TLC yellow-taxi trip records, April 2025 (3.97 M trips) | Ride requests for the fleet simulator, travel-time model | NYC Open Data terms of use |
| NYC TLC taxi-zone shapefile | Zone geometry (centroids, areas, maps) | NYC Open Data terms of use |
| NYISO day-ahead zonal LBMP, April 2025, zone N.Y.C. | Hourly electricity price signal | NYISO public market data |
| Open-Meteo historical weather (hourly temperature, precipitation) | Weather covariates; vehicle HVAC load | CC BY 4.0 |

`python -m evfleet.data.download` fetches every file, records URL, size, SHA-256 and
download time in `data/raw/MANIFEST.json`, and never overwrites an existing raw file.
Raw data are not committed. Cleaning is written in SQL (`sql/`) and every raw row gets
exactly one outcome, so exclusions are fully accounted for
(`reports/results/pa_data_quality.csv`, `nyc_trips_data_quality.csv`).

## 2. Charging-session study (Palo Alto)

**Features are computed as of plug-in time** (`sql/session_features.sql`). A driver's
history uses only earlier sessions; if any earlier session of that driver had not yet
ended, its duration and energy were not yet observable, so all history features are
nulled for that row (0.4 % of rows). The session's own fee is excluded because it is
billed per kWh at the end and would leak the energy target; only the pricing regime is
used. `tests/test_features.py` checks this with a future-intervention test: editing a
later session must not change earlier features.

**Splits are chronological:** train 2011-2018, calibration Jan-Jun 2019, test Jul-Dec
2019, and 2020 as a separate distribution-shift test (COVID shelter-in-place).

**Models.** Three baselines (global median; site x weekend x hour median; the driver's
own median history) and histogram gradient-boosted quantile regression (q = 0.1, 0.5,
0.9) on log connection time and log(1 + energy). Hyper-parameters were fixed a priori,
not searched. 80 % intervals use conformalized quantile regression (CQR) calibrated on
the calibration half-year, evaluated on the test half-year, and broken down by segment
(conditional coverage is not guaranteed by conformal methods and is therefore reported).
For 2020 a nightly recalibration uses the latest 3,000 sessions whose outcome was known
before midnight - a dwell time is only observable once the car leaves.

**Segmentation.** K-means on standardized arrival time (sin/cos), log connection time,
log energy and log idle time; K = 5 maximises the silhouette score over K = 3..8
(0.29, i.e. soft segments) and is stable across seeds (adjusted Rand index 0.997).
A Gaussian mixture was tried first and rejected: its BIC kept improving up to the
largest K tried and binary / point-mass features produced degenerate components.
Segment names are generated from centroid statistics by a fixed rule
(`evfleet.sessions.segments.name_segment`).

**Fee analysis.** Every station switched from free to $0.23/kWh on 1 Aug 2017, so there
is no untreated station. Three designs are reported side by side:

1. *Interrupted time series* on a daily panel of stations operating throughout, with
   separate pre/post linear trends, day-of-week, holiday and weather controls and
   Newey-West standard errors (14 lags); bandwidths 60, 90, 120 and 180 days.
2. *Year-over-year difference-in-differences*: each 2017 day is compared with the same
   weekday 364 days earlier; the effect is the post-August change in that growth rate.
   The **placebo** repeats the analysis for 2016 vs 2015 with a fake 1 Aug 2016 date.
3. *Driver cohorts*: of drivers active Feb-Jul, the share still charging Aug-Dec, and
   stayers' change in frequency and kWh per session, 2017 vs 2016 (placebo 2016 vs 2015),
   with bootstrap intervals over drivers. **This design fails its placebo** (the 2016
   "effect" on stayers' frequency is as large as the 2017 one), so it is reported but not
   interpreted.

The year-over-year placebo itself is not zero (e.g. +10 % for sessions per day), which
shows year-over-year growth can shift by that much without any intervention; the point
estimates should be read with that design uncertainty, not only their HAC intervals.

*Assumption:* without the fee, 2017 would have followed 2016's seasonal path around a
smooth trend. *Limits:* the fee prices these public stations only; drivers may have moved
to workplace or home charging that the data cannot see, so effects are on demand at city
stations, not total EV charging. Other changes in 2017 (EV mix, new private chargers) are
not separately controlled. Sessions turned away by full sites are never recorded.

## 3. Robotaxi fleet simulator (NYC)

**Service area:** the 66 Manhattan taxi zones (harbour islands excluded). Airports were
tried first and removed: airport pickups arrive in waves 30-50 minutes' drive from the
fleet, so even with long-range staging 42-48 % of them went unserved in validation
runs with the threshold baseline (98-100 % of Manhattan pickups were served in the same
runs). That is a separate staging problem and would have dominated the service metric.

**Demand:** every cleaned Manhattan-internal yellow-taxi trip is a ride request at its
recorded pickup time, origin and destination, with its recorded duration and distance.
Experiments use a **20 % random replica** of demand (fleet and chargers are sized for
it); each seed draws a different replica, and all policies see the identical draw for
a given seed (paired design). One full-scale run (100 % demand, 5x fleet and chargers)
checks that conclusions do not depend on the replica. Realized taxi trips are *served*
demand, not latent demand - unserved hails are not in the data.

**Travel times for empty driving** come from a log-additive model
`log t = pair effect + weekday/weekend-by-hour congestion effect`, fitted by median
polish on the training weeks (1-21 April 2025); sparse pairs fall back to a regression
on log centroid distance. It is validated on 0.95 M held-out trips (22-30 April) in
`reports/results/nyc_travel_time_validation.csv`.

**Energy** (`evfleet.sim.energy`): `e(v, T) = a + b v^2 + (P_aux + P_hvac(T)) / v`
kWh/km, with rolling/urban losses a = 0.095 kWh/km, aerodynamic b from CdA = 0.55 m^2,
0.35 kW always-on compute/auxiliary load, heat-pump HVAC outside an 18-24 degC band, and
0.25 kW while parked awake. These are **representative assumptions for a compact
electric robotaxi, not a fit to telemetry** (none is public). Weather is the real hourly
temperature for the simulated days.

**Charging** (`evfleet.sim.charging`): 40 kWh usable battery, 150 kW DC chargers, a
piecewise-linear C-rate taper (2.5C to 50 % SoC, 1.2C at 80 %, 0.2C at 100 %), 92 %
grid-to-battery efficiency and a 2-minute plug overhead. Time-to-charge is integrated
exactly from the curve; unplugging early uses its inverse.

**Dispatch and rebalancing** are identical for every policy: nearest idle vehicle that
can reach the rider within 10 minutes and still has energy for the trip, a drive to the
nearest charger and a 5 % reserve; unmatched riders wait up to 10 minutes; every 15
minutes idle vehicles are rebalanced toward forecast demand by a minimum-cost assignment.

**Policies** (`evfleet.sim.policies`): threshold 20 -> 80 %; threshold 20 -> 100 %;
queue-aware depot choice (earliest predicted plug-in time, FIFO-aware); and an
orchestrator that tops up surplus idle vehicles in demand troughs and cheap day-ahead
hours, keeps one charger per depot free for low-battery arrivals, lowers its trigger
when vehicles are scarce, assigns vehicles to chargers by the Hungarian algorithm, and
may unplug partly charged vehicles for waiting riders.

**Tuning discipline:** policy parameters and the base scenario (550 vehicles, 36
chargers, 6 p-median sites) were chosen on the validation week (13-20 April); the
reported results are the held-out test week (22-28 April, with 21 April as warm-up).
The demand forecaster, travel-time model and siting were fitted on 1-21 April only.

**Siting** solves the p-median problem exactly as a MILP (SciPy/HiGHS), weighting zones
by drop-offs; chargers are split in proportion to the demand each site serves.

**Correctness checks** (`tests/test_sim.py`, run in CI): energy conservation to 1e-6 kWh
per run, no stranded vehicles, request accounting, waits within patience, charger
capacity never exceeded, determinism, identical demand across policies, outage
behaviour, and time accounting. Every experiment run also reports its energy residual
and depletion count.

**Statistics:** each seed is an independent replicate; differences vs the baseline are
paired by seed and reported as means with 95 % t-intervals. Seeds share the same week of
real demand, so intervals describe simulation variability for that week, not
week-to-week variation.

### What the simulator does not capture

- Road-network routing and congestion caused by the fleet itself (travel times are
  exogenous, from taxi data).
- Pooling/ride-sharing, rider choice, pricing effects on demand, cancellations beyond a
  fixed patience.
- Real vehicle telemetry, battery degradation, thermal effects on charging speed.
- Demand charges and distribution tariffs: energy cost uses the wholesale day-ahead
  price only; peak charging load is reported separately because utilities bill it.
- Charger faults other than the scripted outage; per-site power limits.

## 4. Reproducing

```bash
make setup      # virtual environment + package
make data       # ~160 MB download, hashed manifest, processed tables
make sessions   # Palo Alto study + fleet input validation (~3 min)
make fleet      # fleet experiments (~10-15 min on 6 cores)
python scripts/run_fleet_ablation.py
python scripts/run_fleet_profiles.py
make figures
python scripts/summarize_results.py
make test
```
