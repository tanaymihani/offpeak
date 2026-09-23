# OffPeak

[![ci](https://github.com/tanaymihani/offpeak/actions/workflows/ci.yml/badge.svg)](https://github.com/tanaymihani/offpeak/actions/workflows/ci.yml)

When, where and how should an electric fleet charge? I studied this question in two parts, both built on public data.

1. **Charging sessions.** 259k real sessions from the City of Palo Alto's public chargers, 2011 to 2020. I predict at plug-in how long a car will stay, with prediction intervals that I check on held-out data. I also group sessions into behaviour types, measure how often each site is full, and estimate what happened when the city started charging $0.23/kWh.
2. **Robotaxi fleet.** An event-driven simulator of an electric robotaxi fleet serving 3.08 million real Manhattan taxi trips from April 2025, with real NYISO wholesale electricity prices and real hourly weather. I use it to compare charging policies, place chargers with an exact optimization model, and stress-test the results.

Everything runs on a laptop CPU from free data. Every number below is produced by a script in [`scripts/`](scripts) and saved in [`reports/results/`](reports/results), and the simulator is covered by invariant tests that run in CI.

## Main results

### Fleet charging (NYC, held-out test week, 10 paired seeds)

- Charging at the right time mattered more than anything else I tested. A policy that tops cars up when demand is low and keeps them on the road at peaks served 96.1% of riders, against 92.7% for the common rule of charging at 20% and stopping at 80%. That is 3.4 percentage points more (95% CI 3.0 to 3.8). Mean pickup wait was 0.5 minutes shorter and energy cost was 15% lower. The trade-off is a 16% higher peak charging load, because the policy fills every charger overnight.
- The gap holds at full scale: with 745k requests and 2,750 cars, service goes from 94.0% to 97.4%. One 8-day full-scale run takes about 65 seconds.
- To reach 96% service, the orchestrated policy needs about 545 cars instead of about 609, roughly 10% fewer.
- With the orchestrated policy, 24 chargers served more riders (96.1%) than 72 chargers did under the threshold rule (93.4%).
- The ablation shows where the gains come from: charging in demand troughs, unplugging part-charged cars when riders are waiting, and lowering the charging trigger when cars are scarce. The price signal added nothing measurable, because in New York the cheapest wholesale hours are already the low-demand hours. On a grid with a lot of solar, where midday power is cheapest, this could be different. I have not tested that.
- For siting, the exact p-median layout cut the average drive to a charger from 9.1 to 6.5 minutes compared with putting chargers in the six busiest zones, and cut charger-trip distance by 25%. It barely changed service.

### Charging sessions (Palo Alto)

- At plug-in, gradient-boosted quantile regression predicts how long a car will stay with a mean absolute error of 52 minutes. The three baselines I compared it with were at 56 to 73 minutes. The 80% conformal (CQR) intervals covered 80.6% of held-out sessions.
- During COVID in 2020, intervals calibrated once on 2019 fell to 73 to 77% coverage. Recalibrating every night on sessions that had already ended (a stay is only known once the car leaves) brought coverage back to about 80% from July.
- After the $0.23/kWh fee started, sessions fell 29 to 43% and idle time per session (plugged in but not charging) fell 50 to 65%, depending on the design. Sessions that were mostly parking fell 69%. The same design run on 2016, when there was no fee, gives estimates between -15% and +10%, so I read these results as the right direction and rough size, not as precise numbers.
- Hamilton and Webster are the congested sites: every port is taken for 25% and 20% of weekday working hours. Every other site is full 5% of that time or less, so those two are where more ports would help first.

---

## Fleet study: charging a robotaxi fleet in Manhattan

<p align="center"><img src="reports/figures/fleet_load_shift.png" width="720" alt="Fleet charging load by hour of day for the threshold rule and the orchestrated policy"></p>

Under the threshold rule, cars go to charge as soon as they reach 20%. That mostly happens in the afternoon and evening, which is exactly when riders and electricity prices peak. The orchestrated policy moves almost all charging to between 11pm and 5am.

### Setup

| | |
|---|---|
| Demand | Every cleaned Manhattan-internal yellow-taxi trip (NYC TLC, April 2025) becomes a ride request with its real time, origin, destination, duration and distance. Experiments use a 20% random sample of demand, and one full-scale run checks the conclusions. |
| Travel times for empty driving | A log-additive model (zone-pair effect times a weekday/weekend by hour congestion factor) fit by median polish on 1 to 21 April. On 952,699 held-out trips the mean absolute error is 3.3 minutes, against 3.7 without time of day and 5.1 with a constant speed. |
| Vehicles | 40 kWh usable battery. Energy use covers rolling and stop-and-go losses, aerodynamic drag, always-on computers, and a heat-pump HVAC load driven by the real hourly temperature. These are my assumptions, not telemetry. |
| Charging | 150 kW DC chargers with a lithium-ion taper curve (going from 80% to 100% takes longer than going from 20% to 80%), 92% efficiency and first-come queues. |
| Prices | NYISO day-ahead zonal LBMP for the N.Y.C. zone, hourly. |
| Dispatch | The nearest idle car that can reach the rider within 10 minutes and has enough energy for the trip, a drive to a charger and a 5% reserve. Riders wait up to 10 minutes. Every 15 minutes, idle cars are moved toward forecast demand with a minimum-cost assignment. |
| Tuning | I fixed the policy parameters and the scenario (550 cars, 36 chargers, 6 sites) on a validation week (13 to 20 April). All results are on the held-out test week (22 to 28 April). Each seed is an independent sample of demand, shared by all policies. |

### Policies

| Policy | Rule |
|---|---|
| Threshold, 20% to 80% | Charge below 20% at the nearest charger and stop at 80%. |
| Threshold, 20% to 100% | Same trigger, but charge to full. |
| Queue-aware | Same trigger, but pick the charger with the earliest predicted plug-in time, counting the cars already charging, waiting or on their way. |
| Orchestrated | Every 5 minutes it forecasts the next hour of demand, keeps a buffer of idle cars, and sends the extra low-battery cars to free chargers using the Hungarian algorithm. It allows higher top-ups in cheap day-ahead hours, keeps one charger per depot free for low-battery arrivals, lowers the trigger when cars are scarce, and can unplug a part-charged car for a waiting rider. |

### Results

<p align="center"><img src="reports/figures/fleet_policy_comparison.png" width="900" alt="Paired differences against the threshold baseline"></p>

| Test week, 10 seeds | Riders served | Mean pickup wait | Mean charger queue | Energy cost per day | Average price paid | Peak charging load |
|---|---:|---:|---:|---:|---:|---:|
| Threshold, 20% to 80% | 92.7% | 6.89 min | 5.0 min | $538 | $35.6/MWh | 3.12 MW |
| Threshold, 20% to 100% | 91.6% | 7.00 min | 26.6 min | $485 | $33.9/MWh | 2.19 MW |
| Queue-aware | 92.9% | 6.88 min | 2.3 min | $535 | $35.3/MWh | 3.13 MW |
| Orchestrated | **96.1%** | **6.40 min** | 17.5 min | **$457** | **$28.8/MWh** | 3.61 MW |

Costs are wholesale energy only, for the 20% sample. Queue-aware depot choice halves the charger queue, but its service gain (0.15 points) is not distinguishable from zero. Charging to 100% is worse for service and queues, although its fewer, longer sessions lower the peak.

<table>
<tr>
<td><img src="reports/figures/fleet_charger_scarcity.png" alt="Service against the number of chargers"></td>
<td><img src="reports/figures/fleet_fleet_size.png" alt="Service against fleet size"></td>
</tr>
</table>

**Ablation** (test week, 6 paired seeds). I used this only to explain where the gains come from, not to choose anything.

| Variant | Riders served | Energy cost per day |
|---|---:|---:|
| Threshold, 20% to 80% | 93.0% | $546 |
| Orchestrated (full) | 96.2% | $458 |
| without trough top-ups | 95.1% | $528 |
| without unplugging cars for riders | 95.8% | $451 |
| without the lower trigger when cars are scarce | 95.9% | $481 |
| without the price signal | 96.1% | $458 |
| without the reserved charger per depot | 96.2% | $453 |
| top-ups only (no unplugging, no scarce trigger) | 94.4% | $465 |

**Stress tests** (seeds 0 to 3). The orchestrated policy keeps its service lead in every scenario. In a week at -5°C, energy use rises for every policy and the orchestrated policy loses its cost advantage (about $924 against $884 a day), because more of its charging has to happen in expensive hours.

<p align="center"><img src="reports/figures/fleet_stress.png" width="640" alt="Stress tests"></p>

**Siting.** I solve the p-median problem exactly as a MILP (SciPy with HiGHS), weighting zones by where cars drop riders off, and compare it with a common heuristic.

<p align="center"><img src="reports/figures/fleet_siting_map.png" width="620" alt="p-median and busiest-zone charger layouts"></p>

| Layout (36 chargers) | Mean drive to a charger | Charger-trip km per car per day | Riders served (queue-aware) |
|---|---:|---:|---:|
| 1 site | 12.9 min | 4.5 | 93.3% |
| 3 sites, p-median | 8.1 min | 3.3 | 93.2% |
| 6 busiest drop-off zones | 9.1 min | 3.3 | 93.3% |
| 6 sites, p-median | **6.5 min** | **2.5** | 93.1% |
| 12 sites, p-median | 5.6 min | 2.1 | 92.8% |

Siting mostly changes how far cars drive empty to reach a charger, which costs energy and wear, rather than whether riders get served. Service differs by less than 0.5 points across layouts (4 seeds each). Twelve small sites serve slightly fewer riders. That fits the idea that small sites pool their queues less well, but these runs do not prove it.

---

## Session study: 259k charging sessions in Palo Alto

<p align="center"><img src="reports/figures/pa_demand_timeline.png" width="720" alt="Weekly charging sessions, 2011 to 2020"></p>

### Predicting how long a car will stay

I compute every feature as of plug-in time with SQL window functions. A driver's history only uses sessions that had already ended, and a test checks that changing a later session never changes an earlier row. I leave out the session's own fee because it is billed per kWh at the end, so it would leak the energy target. The splits are in time order: train on 2011 to 2018, calibrate on January to June 2019, test on July to December 2019, and use 2020 as a shift test.

<p align="center"><img src="reports/figures/pa_dwell_prediction.png" width="900" alt="Prediction error and interval coverage by segment"></p>

| Held-out July to December 2019 (24k sessions) | Connection time MAE | Energy MAE |
|---|---:|---:|
| Global median | 73 min | 5.4 kWh |
| Site by weekend by hour median | 66 min | 5.2 kWh |
| Driver's own history | 56 min | 4.0 kWh |
| Gradient-boosted quantile regression | **52 min** | **3.6 kWh** |

Raw quantile regression covered 78.5% with its 80% intervals. After conformal calibration (CQR), coverage is 80.6% for connection time and 79.5% for energy. Coverage stays close to 80% across sites and times of day, but anonymous drivers only get 74.6%. Conformal guarantees hold for the population as a whole, not for each group, so I measure the groups separately.

<p align="center"><img src="reports/figures/pa_conformal_shift.png" width="720" alt="Interval coverage during 2020"></p>

### What the $0.23/kWh fee did

Every station went from free to paid on 1 August 2017, so there is no untreated station to compare with. I report three designs side by side. The first is an interrupted time series with separate trends, weather and calendar controls and Newey-West standard errors. The second is a year-over-year difference-in-differences against 2016. The third is the same year-over-year design run as a placebo, with a fake fee in August 2016.

<p align="center"><img src="reports/figures/pa_fee_effects.png" width="780" alt="Estimated fee effects for each design"></p>

<p align="center"><img src="reports/figures/pa_fee_event_study.png" width="900" alt="Monthly event study"></p>

| Outcome | Time series (120 days each side) | Year-over-year DiD | Placebo (2016) |
|---|---:|---:|---:|
| Sessions per day | -30% | -43% | +10% |
| Energy per day | -43% | -51% | +2% |
| Unique drivers per day | -30% | -41% | +7% |
| kWh per session | -19% | -14% | -7% |
| Connection time per session | -30% | -23% | -6% |
| Idle time per session | **-65%** | **-50%** | -15% |

The fee was charged per kWh, but idle time fell the most. The segments explain this: the fee mostly removed people who used the chargers as parking.

<p align="center"><img src="reports/figures/pa_segments.png" width="900" alt="Session segments and how each changed after the fee"></p>

The segments are k-means clusters on arrival time, connection time, energy and idle time. K = 5 has the best silhouette score (0.29, so the groups overlap) and the result is stable across seeds (adjusted Rand index 0.997).

One part did not work. I tried to split the drop into drivers who stopped coming and drivers who charged less, using driver cohorts. That design failed its own placebo: the fake 2016 effect on how often the remaining drivers charged (-20%) was as large as the real 2017 one (-18%). The numbers are in the results files, but I do not interpret them. Retention of existing drivers did not change in a way I could detect (-2.4 points, CI -4.9 to +0.2).

### Where drivers find no free port

<p align="center"><img src="reports/figures/pa_site_congestion.png" width="760" alt="Share of time each site has no free port, by hour"></p>

When every port is taken, a driver who wanted to charge leaves no record in the data. So the share of time a site is full is a lower bound on the demand it could not serve. Ports also come and go during the year, so for each month I only count the ports that had at least one session.

---

## Engineering

- **Data pipeline.** `python -m evfleet.data.download` downloads every source and records its URL, size, SHA-256 and download time in a manifest, and it never overwrites raw files. Cleaning is SQL ([`sql/`](sql)) run with DuckDB on Parquet. Every raw row is either kept or gets one named reason for being dropped, so nothing disappears silently. 99.96% of the Palo Alto rows are kept, and the NYC exclusions are listed in [`reports/results/nyc_trips_data_quality.csv`](reports/results/nyc_trips_data_quality.csv).
- **Simulator checks.** [`tests/test_sim.py`](tests/test_sim.py) checks energy conservation to 1e-6 kWh, that no car runs out of charge, request accounting, waits within patience, charger capacity, determinism, identical demand across policies, outages and time accounting. Across all 314 experiment runs no car ran out of charge and the largest energy-ledger error was 3e-6 kWh. These checks caught real bugs while I was building the simulator: idle drain on parked cars that was never re-checked, a charger choice that could exceed a car's remaining range, and time lost in the end-of-run accounting. I fixed all of them before running the reported experiments.
- **Statistics.** An exact finite-sample conformal quantile (it returns an unbounded interval when there is too little calibration data), Newey-West standard errors, paired seeds with t-intervals, and placebo tests. All of these are unit-tested.
- **Optimization.** The p-median MILP is checked against brute force. The Hungarian algorithm assigns cars to charger slots and to rebalancing targets.
- **Tests and CI.** 54 tests. Lint and tests run in GitHub Actions on Python 3.11 and 3.12 using synthetic data, so CI does not need to download anything.

## Limitations

The full list is in [`docs/methodology.md`](docs/methodology.md). The main ones:

- Taxi trips are the demand that was served, not all the demand there was. Airports are outside the service area because airport pickups need their own staging logic.
- Travel times come from taxi data, so the fleet does not create congestion of its own, and there is no road-level routing or ride pooling.
- The vehicle energy and charging curves are my assumptions, not telemetry. The cold-weather result depends on the heating assumption.
- Energy cost uses wholesale day-ahead prices only. Utilities also bill for peak demand, which is why I report peak load separately.
- All fleet results come from one real week of demand. The seeds are samples of that week, not of every city, season or fleet.
- The fee study measures demand at these public stations. Drivers may have moved to chargers that the data cannot see.

## Reproducing the results

```bash
git clone https://github.com/tanaymihani/offpeak && cd offpeak
make setup        # virtual environment and package (Python 3.11 or newer)
make data         # about 160 MB of public data, a hashed manifest and processed tables
make sessions     # Palo Alto study and fleet input checks (about 3 minutes)
make fleet        # 266 fleet runs (about 10 minutes on 6 cores)
.venv/bin/python scripts/run_fleet_ablation.py
.venv/bin/python scripts/run_fleet_profiles.py
make figures
.venv/bin/python scripts/summarize_results.py   # writes reports/results/headline_numbers.csv
make test
```

## Layout

```
src/evfleet/
  data/        downloads and manifest, Palo Alto and TLC builders, NYISO prices, weather
  sessions/    plug-in models, conformal intervals, segmentation, fee analysis, congestion
  forecast/    short-horizon demand forecaster
  optimize/    p-median siting (MILP)
  sim/         travel times, energy, charging curve, event-driven engine, policies, experiments
sql/           cleaning and feature SQL (DuckDB)
scripts/       one script per study, each writing to reports/results
reports/       results (CSV) and figures (PNG)
tests/         unit and invariant tests on synthetic data
docs/          methodology, assumptions and limitations
```

## Data sources

| Source | Licence or terms |
|---|---|
| [City of Palo Alto, EV Charging Station Usage, July 2011 to December 2020](https://data.paloalto.gov/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/) | Open Data Commons PDDL |
| [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (yellow taxi, April 2025) and the taxi zone shapefile | NYC Open Data terms of use |
| [NYISO](https://www.nyiso.com/energy-market-operational-data) day-ahead zonal LBMP, April 2025 | NYISO public market data |
| [Open-Meteo](https://open-meteo.com/) historical weather API | CC BY 4.0 |

The scripts download the raw data. I do not redistribute it here.

## License

The code is under the MIT License. See [LICENSE](LICENSE).
