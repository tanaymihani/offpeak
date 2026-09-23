"""Event-driven simulator of an electric robotaxi fleet with shared DC fast chargers.

Ride requests are real NYC yellow-taxi trips (optionally subsampled). Each
request is served by the nearest idle vehicle that can reach the rider within
the rider's patience *and* has enough energy for pickup + trip + reaching a
charger + a reserve. Unmatched requests wait (still eligible for vehicles that
free up) until their patience expires, after which they are lost.

Vehicles consume energy on every leg (``EnergyModel``) and while idle; they
charge at depots with a finite number of chargers and a first-come queue
(``ChargeCurve``). When to charge, where, and to what level is delegated to a
``ChargingPolicy``; everything else (dispatch, rebalancing, demand) is shared,
so policy comparisons on the same seed are paired.

Energy is debited when a leg starts and credited when charging stops, and a
full ledger is kept so the conservation identity

    initial + charged - consumed + depletion_shortfall == final

can be checked after every run (``SimResult.energy_residual_kwh``).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evfleet.forecast.demand import SLOT_MIN, LevelAdjustedProfile
from evfleet.sim.charging import ChargeCurve
from evfleet.sim.energy import EnergyModel
from evfleet.sim.network import Network, bucket_of

# vehicle states
IDLE, TO_PICKUP, WITH_RIDER, TO_DEPOT, QUEUED, CHARGING, REPOSITION = range(7)
STATE_NAMES = ["idle", "to_pickup", "with_rider", "to_depot", "queued", "charging", "reposition"]

# event kinds (ordering within a timestamp is by insertion sequence)
EV_REQUEST, EV_PICKUP, EV_DROPOFF, EV_AT_DEPOT, EV_CHARGED, EV_REPOS_DONE, EV_EXPIRE, EV_TICK, EV_OUTAGE = range(9)

BOARDING_S = 30.0  # rider boarding at pickup


@dataclass
class SimConfig:
    fleet_size: int
    depots: dict[int, int]  # zone_id -> number of chargers
    start: str = "2025-04-21"
    warmup_end: str = "2025-04-22"  # metrics are collected from here
    end: str = "2025-04-29"
    sample_frac: float = 0.2
    seed: int = 0
    battery_kwh: float = 40.0
    charger_kw: float = 150.0
    max_wait_s: float = 600.0
    reserve_soc: float = 0.05
    tick_s: float = 300.0
    rebalance: bool = True
    rebalance_every_s: float = 900.0
    rebalance_max_tt_s: float = 720.0
    rebalance_min_soc: float = 0.30
    temp_override_c: float | None = None  # constant temperature scenario
    demand_multiplier: float = 1.0
    outages: list[tuple[int, str, str]] = field(default_factory=list)  # (zone_id, start, end)
    init_soc: tuple[float, float] = (0.4, 0.9)


@dataclass
class Environment:
    """Exogenous inputs shared by all runs (fit on training weeks only)."""

    network: Network
    trips: pd.DataFrame  # candidate requests covering [start, end)
    zone_profile: np.ndarray  # (168, Z) expected requests per hour-of-week and zone
    fleet_profile: np.ndarray  # (672,) expected requests per 15-min slot-of-week
    dropoff_share: np.ndarray  # (Z,) where vehicles tend to be (initial placement)
    temperature: pd.Series  # hourly degC, local time
    price: pd.Series  # hourly $/MWh day-ahead LBMP, local time
    energy: EnergyModel = field(default_factory=EnergyModel)


class Vehicle:
    __slots__ = ("vid", "state", "zone", "soc", "t_state", "token", "req", "depot",
                 "target", "plug_start", "soc_at_plug", "charge_end", "queued_at")

    def __init__(self, vid: int, zone: int, soc: float, t: float):
        self.vid, self.state, self.zone, self.soc, self.t_state = vid, IDLE, zone, soc, t
        self.token = 0
        self.req = -1
        self.depot = -1
        self.target = 0.0
        self.plug_start = 0.0
        self.soc_at_plug = 0.0
        self.charge_end = 0.0
        self.queued_at = 0.0


class Depot:
    __slots__ = ("did", "zone", "chargers", "charging", "queue", "incoming", "online")

    def __init__(self, did: int, zone: int, chargers: int):
        self.did, self.zone, self.chargers = did, zone, chargers
        self.charging: set[int] = set()
        self.queue: deque[int] = deque()
        self.incoming: dict[int, float] = {}  # vid -> arrival time
        self.online = True

    @property
    def free(self) -> int:
        return self.chargers - len(self.charging) if self.online else 0


@dataclass
class SimResult:
    config: SimConfig
    policy: str
    requests: pd.DataFrame
    sessions: pd.DataFrame
    state_hours: dict[str, float]
    km: dict[str, float]
    energy: dict[str, float]
    load_kw: pd.Series  # fleet charging load (grid side) at 5-minute resolution
    runtime_s: float
    events: int

    @property
    def energy_residual_kwh(self) -> float:
        e = self.energy
        return e["initial"] + e["charged_battery"] - e["consumed_total"] + e["depletion_shortfall"] - e["final"]


class Simulator:
    def __init__(self, env: Environment, cfg: SimConfig, policy):
        self.env, self.cfg, self.policy = env, cfg, policy
        self.net = env.network
        self.energy_model = env.energy
        self.curve = ChargeCurve(capacity_kwh=cfg.battery_kwh, charger_kw=cfg.charger_kw)
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = pd.Timestamp(cfg.start)
        self.t_warm = (pd.Timestamp(cfg.warmup_end) - self.t0).total_seconds()
        self.t_end = (pd.Timestamp(cfg.end) - self.t0).total_seconds()
        self._events: list = []
        self._seq = 0
        self.n_events = 0

        self._prepare_requests()
        self._prepare_exogenous()
        self._place_fleet()
        self.depots = [Depot(k, self.net.index[z], n) for k, (z, n) in enumerate(sorted(cfg.depots.items()))]
        self.depot_by_zone = {d.zone: d for d in self.depots}
        self.forecaster = LevelAdjustedProfile(env.fleet_profile, scale=cfg.sample_frac * cfg.demand_multiplier)
        self._slot_arrivals = 0
        self._current_slot = None
        self._depot_energy_cache: dict[tuple[int, int], float] = {}

        # ledgers
        self.consumed = {"drive_rider": 0.0, "drive_empty": 0.0, "idle": 0.0}
        self.charged_battery = 0.0
        self.depletion_shortfall = 0.0
        self.depletions = 0
        self.state_time = np.zeros(len(STATE_NAMES))
        self.km = {"with_rider": 0.0, "to_pickup": 0.0, "to_depot": 0.0, "reposition": 0.0}
        self.sessions: list[tuple] = []

    # ------------------------------------------------------------------ setup
    def _prepare_requests(self) -> None:
        cfg, trips = self.cfg, self.env.trips
        mask = (trips["pickup_ts"] >= self.t0) & (trips["pickup_ts"] < pd.Timestamp(cfg.end))
        tr = trips.loc[mask]
        p = cfg.sample_frac * cfg.demand_multiplier
        if p < 1.0:
            tr = tr.loc[self.rng.random(len(tr)) < p]
        elif p > 1.0:  # surge: add duplicates of a random subset with jittered times
            extra = tr.loc[self.rng.random(len(tr)) < (p - 1.0)].copy()
            extra["pickup_ts"] += pd.to_timedelta(self.rng.uniform(-300, 300, len(extra)), unit="s")
            tr = pd.concat([tr, extra]).sort_values("pickup_ts")
            tr = tr[(tr["pickup_ts"] >= self.t0) & (tr["pickup_ts"] < pd.Timestamp(cfg.end))]
        idx = self.net.index
        self.r_t = (tr["pickup_ts"] - self.t0).dt.total_seconds().to_numpy()
        self.r_o = tr["pu"].map(idx).to_numpy()
        self.r_d = tr["do_zone"].map(idx).to_numpy()
        self.r_km = tr["distance_km"].to_numpy(dtype=float)
        self.r_dur = tr["duration_s"].to_numpy(dtype=float)
        n = len(self.r_t)
        self.r_status = np.zeros(n, dtype=np.int8)  # 0 pending/new, 1 assigned, 2 lost
        self.r_pickup = np.full(n, np.nan)
        self.r_vehicle = np.full(n, -1, dtype=np.int64)
        self.pending_by_zone: list[deque[int]] = [deque() for _ in range(self.net.n)]

    def _prepare_exogenous(self) -> None:
        hours = pd.date_range(self.t0, pd.Timestamp(self.cfg.end), freq="h")
        temp = self.env.temperature.reindex(hours).ffill().bfill()
        if self.cfg.temp_override_c is not None:
            temp[:] = self.cfg.temp_override_c
        self.temp_by_hour = temp.to_numpy()
        price = self.env.price.reindex(hours).ffill().bfill()
        self.price_by_hour = price.to_numpy()  # $/MWh
        # Day-ahead prices are published the day before, so a policy may use
        # the full day's price ranks when deciding to charge (causal).
        days = hours.normalize()
        self.price_rank_by_hour = price.groupby(days).rank(pct=True).to_numpy()
        self.dow_by_hour = hours.dayofweek.to_numpy()
        self.hod_by_hour = hours.hour.to_numpy()

    def _place_fleet(self) -> None:
        cfg = self.cfg
        zones = self.rng.choice(self.net.n, size=cfg.fleet_size, p=self.env.dropoff_share)
        socs = self.rng.uniform(*cfg.init_soc, size=cfg.fleet_size)
        self.vehicles = [Vehicle(k, int(z), float(s), 0.0) for k, (z, s) in enumerate(zip(zones, socs, strict=True))]
        self.idle_by_zone: list[set[int]] = [set() for _ in range(self.net.n)]
        for v in self.vehicles:
            self.idle_by_zone[v.zone].add(v.vid)
        self.initial_energy = sum(v.soc for v in self.vehicles) * cfg.battery_kwh

    # ----------------------------------------------------------------- helpers
    def push(self, t: float, kind: int, obj: int, token: int = 0) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, obj, token))

    def hour_index(self, t: float) -> int:
        return min(int(t // 3600), len(self.temp_by_hour) - 1)

    def bucket(self, t: float) -> int:
        h = self.hour_index(t)
        return bucket_of(int(self.dow_by_hour[h]), int(self.hod_by_hour[h]))

    def temp(self, t: float) -> float:
        return float(self.temp_by_hour[self.hour_index(t)])

    def can_reach(self, v: Vehicle, d: Depot, t: float, margin_soc: float = 0.02) -> bool:
        """Whether the vehicle can drive to depot ``d`` keeping ``margin_soc`` in reserve."""
        sec, km = self.travel(v.zone, d.zone, t)
        return v.soc - self.leg_energy(km, sec, t) / self.cfg.battery_kwh >= margin_soc

    def charging_power_kw(self, t: float) -> float:
        """Grid-side power currently drawn by all plugged-in vehicles."""
        total = 0.0
        for d in self.depots:
            for vid in d.charging:
                v = self.vehicles[vid]
                if t >= v.plug_start:
                    total += float(self.curve.power_kw(self.current_soc(v, t))) / self.curve.efficiency
        return total

    def current_slot_of_week(self, t: float) -> int:
        h = self.hour_index(t)
        minute = int((t % 3600) // 60)
        return int(self.dow_by_hour[h]) * 96 + int(self.hod_by_hour[h]) * 4 + minute // SLOT_MIN

    def price_rank(self, t: float) -> float:
        return float(self.price_rank_by_hour[self.hour_index(t)])

    def travel(self, i: int, j: int, t: float) -> tuple[float, float]:
        """(seconds, km) of empty driving from zone i to zone j departing at t."""
        return self.net.tt[self.bucket(t), i, j], self.net.dist_km[i, j]

    def leg_energy(self, km: float, seconds: float, t: float) -> float:
        return self.energy_model.leg_kwh(km, seconds, self.temp(t))

    def _account_state(self, v: Vehicle, t: float) -> None:
        """Close the vehicle's current state interval [t_state, t) in the time ledger."""
        a, b = max(v.t_state, self.t_warm), min(t, self.t_end)
        if b > a:
            self.state_time[v.state] += b - a

    def set_state(self, v: Vehicle, state: int, t: float) -> None:
        if v.state == IDLE:
            self.idle_by_zone[v.zone].discard(v.vid)
        if v.state in (IDLE, QUEUED):  # parked and awake: auxiliary drain
            self.debit(v, self.energy_model.idle_kwh(t - v.t_state), "idle")
        self._account_state(v, t)
        v.state, v.t_state = state, t
        if state == IDLE:
            self.idle_by_zone[v.zone].add(v.vid)

    def debit(self, v: Vehicle, kwh: float, kind: str) -> None:
        self.consumed[kind] += kwh
        new = v.soc - kwh / self.cfg.battery_kwh
        if new < 0:
            self.depletions += 1
            self.depletion_shortfall += -new * self.cfg.battery_kwh
            new = 0.0
        v.soc = new

    def drive(self, v: Vehicle, dest: int, t: float, kind: str) -> float:
        """Start an empty leg to ``dest``; returns arrival time."""
        sec, km = self.travel(v.zone, dest, t)
        self.debit(v, self.leg_energy(km, sec, t), "drive_empty")
        if t >= self.t_warm:
            self.km[kind] += km
        v.zone = dest
        return t + sec

    def energy_to_nearest_depot(self, zone: int, t: float) -> float:
        key = (self.hour_index(t), zone)
        hit = self._depot_energy_cache.get(key)
        if hit is not None:
            return hit
        best, km = math.inf, 0.0
        b = self.bucket(t)
        for d in self.depots:
            if d.online:
                sec = self.net.tt[b, zone, d.zone]
                if sec < best:
                    best, km = sec, self.net.dist_km[zone, d.zone]
        value = 0.0 if best is math.inf else self.leg_energy(km, best, t)
        self._depot_energy_cache[key] = value
        return value

    def trip_need_soc(self, r: int, t: float) -> float:
        """SoC needed for the rider trip plus reaching a charger afterwards."""
        return (self.leg_energy(self.r_km[r], self.r_dur[r], t)
                + self.energy_to_nearest_depot(self.r_d[r], t)) / self.cfg.battery_kwh

    def pickup_need_soc(self, sec: float, km: float, t: float) -> float:
        return self.leg_energy(km, sec, t) / self.cfg.battery_kwh

    def current_soc(self, v: Vehicle, t: float) -> float:
        if v.state == CHARGING and t > v.plug_start:
            return self.curve.soc_after(v.soc_at_plug, t - v.plug_start)
        return v.soc

    # ---------------------------------------------------------------- dispatch
    def try_assign(self, r: int, t: float) -> bool:
        """Nearest feasible idle (or policy-preemptible charging) vehicle for request r."""
        b = self.bucket(t)
        o = self.r_o[r]
        remaining = self.r_t[r] + self.cfg.max_wait_s - t
        preempt = getattr(self.policy, "preempt_soc", None)
        tt_col = self.net.tt[b, :, o]
        trip_need = self.trip_need_soc(r, t)
        floor = self.cfg.reserve_soc
        for z in self.net.order_to[b, o]:
            sec = tt_col[z]
            if sec > remaining:
                break
            need = trip_need + self.pickup_need_soc(sec, self.net.dist_km[z, o], t)
            # energy need depends only on the zone, so the fullest vehicle decides
            best, best_soc = None, -1.0
            for vid in self.idle_by_zone[z]:
                s = self.vehicles[vid].soc
                if s > best_soc:
                    best, best_soc = self.vehicles[vid], s
            if best is not None and best_soc - need < floor:
                best = None
            if best is None and preempt is not None and z in self.depot_by_zone:
                cand, cand_soc = None, -1.0
                for vid in self.depot_by_zone[z].charging:
                    s = self.current_soc(self.vehicles[vid], t)
                    if s >= preempt and s > cand_soc:
                        cand, cand_soc = self.vehicles[vid], s
                if cand is not None and cand_soc - need >= floor:
                    self.unplug(cand, t)
                    best = cand
            if best is not None:
                self._dispatch(best, r, t)
                return True
        return False

    def _dispatch(self, v: Vehicle, r: int, t: float) -> None:
        self.set_state(v, TO_PICKUP, t)
        v.req = r
        self.r_status[r] = 1
        self.r_vehicle[r] = v.vid
        arrive = self.drive(v, self.r_o[r], t, "to_pickup")
        self.push(arrive, EV_PICKUP, v.vid, v.token)

    def serve_pending(self, v: Vehicle, t: float) -> bool:
        """Give a newly available vehicle to the earliest reachable waiting rider."""
        b = self.bucket(t)
        row = self.net.tt[b, v.zone]
        chosen, chosen_t = -1, math.inf
        for z in self.net.order_from[b, v.zone]:
            sec = row[z]
            if sec > self.cfg.max_wait_s:
                break
            q = self.pending_by_zone[z]
            while q and self.r_status[q[0]] != 0:
                q.popleft()  # lazy deletion of assigned/lost requests
            pickup = self.pickup_need_soc(sec, self.net.dist_km[v.zone, z], t)
            for r in q:  # arrival order, so the first feasible one is this zone's earliest
                if self.r_status[r] != 0 or self.r_t[r] + self.cfg.max_wait_s - t < sec:
                    continue
                if self.r_t[r] >= chosen_t:
                    break
                if v.soc - pickup - self.trip_need_soc(r, t) >= self.cfg.reserve_soc:
                    chosen, chosen_t = r, self.r_t[r]
                    break
        if chosen < 0:
            return False
        self._dispatch(v, chosen, t)
        return True

    # ---------------------------------------------------------------- charging
    def expected_start(self, d: Depot, arrival: float, exclude: int = -1) -> float:
        """Earliest time a charger frees for a vehicle arriving at ``arrival`` (FIFO)."""
        if not d.online:
            return math.inf
        free_at = [max(arrival, self.vehicles[vid].charge_end) for vid in d.charging]
        free_at += [arrival] * (d.chargers - len(d.charging))
        heapq.heapify(free_at)
        ahead = [(self.vehicles[vid].queued_at, vid) for vid in d.queue]
        ahead += [(ta, vid) for vid, ta in d.incoming.items() if ta <= arrival and vid != exclude]
        for ta, vid in sorted(ahead):
            u = self.vehicles[vid]
            start = max(heapq.heappop(free_at), ta)
            dur = self.curve.plug_overhead_s + self.curve.duration_s(u.soc, u.target or 0.8)
            heapq.heappush(free_at, start + dur)
        return max(heapq.heappop(free_at), arrival)

    def send_to_charge(self, v: Vehicle, depot: Depot, target: float, t: float) -> None:
        self.set_state(v, TO_DEPOT, t)
        v.depot, v.target = depot.did, target
        arrive = self.drive(v, depot.zone, t, "to_depot")
        depot.incoming[v.vid] = arrive
        self.push(arrive, EV_AT_DEPOT, v.vid, v.token)

    def _at_depot(self, v: Vehicle, t: float) -> None:
        d = self.depots[v.depot]
        d.incoming.pop(v.vid, None)
        if not d.online:
            alt = self.policy.choose_depot(self, v, t)
            if alt is None:
                self.vehicle_available(v, t)
                return
            self.send_to_charge(v, alt, v.target, t)
            return
        if len(d.charging) < d.chargers:
            self._start_charge(v, d, t)
        else:
            self.set_state(v, QUEUED, t)
            v.queued_at = t
            d.queue.append(v.vid)

    def _start_charge(self, v: Vehicle, d: Depot, t: float) -> None:
        arrived = v.queued_at if v.state == QUEUED else t
        self.set_state(v, CHARGING, t)
        d.charging.add(v.vid)
        v.plug_start = t + self.curve.plug_overhead_s
        v.soc_at_plug = v.soc
        v.charge_end = v.plug_start + self.curve.duration_s(v.soc, v.target)
        v.queued_at = arrived
        self.push(v.charge_end, EV_CHARGED, v.vid, v.token)

    def _finish_charge(self, v: Vehicle, t: float) -> None:
        d = self.depots[v.depot]
        soc1 = self.current_soc(v, t)
        gained = (soc1 - v.soc_at_plug) * self.cfg.battery_kwh
        self.charged_battery += gained
        self.sessions.append((v.vid, d.did, v.queued_at, v.plug_start, t, v.soc_at_plug, soc1))
        v.soc = soc1
        d.charging.discard(v.vid)
        v.token += 1  # invalidates any pending EV_CHARGED
        self.set_state(v, IDLE, t)
        self._start_next(d, t)

    def unplug(self, v: Vehicle, t: float) -> None:
        """Stop charging early (demand preemption or outage)."""
        self._finish_charge(v, t)

    def _start_next(self, d: Depot, t: float) -> None:
        while d.queue and len(d.charging) < d.chargers and d.online:
            u = self.vehicles[d.queue.popleft()]
            if u.state == QUEUED:
                self._start_charge(u, d, t)

    # ----------------------------------------------------------- availability
    def vehicle_available(self, v: Vehicle, t: float) -> None:
        """Vehicle is free at v.zone (after drop-off, charge or repositioning)."""
        if v.state != IDLE:
            self.set_state(v, IDLE, t)
        decision = self.policy.on_available(self, v, t)
        if decision is not None:
            depot, target = decision
            self.send_to_charge(v, depot, target, t)
            return
        self.serve_pending(v, t)

    # -------------------------------------------------------------- rebalance
    def rebalance(self, t: float) -> None:
        h = self.hour_index(t)
        how = int(self.dow_by_hour[h]) * 24 + int(self.hod_by_hour[h])
        scale = self.cfg.sample_frac * self.cfg.demand_multiplier * self.forecaster.ratio
        demand = self.env.zone_profile[how] * scale  # expected requests per zone next hour
        idle = [(vid, z) for z in range(self.net.n) for vid in self.idle_by_zone[z]]
        movable = [(vid, z) for vid, z in idle if self.vehicles[vid].soc >= self.cfg.rebalance_min_soc]
        if len(movable) < 2 or demand.sum() <= 0:
            return
        counts = np.bincount([z for _, z in idle], minlength=self.net.n).astype(float)
        target = demand / demand.sum() * len(idle)
        deficit = np.floor(target - counts).astype(int)
        slots = [z for z in range(self.net.n) for _ in range(max(deficit[z], 0))]
        surplus = counts - np.ceil(target)
        donors = []
        for vid, z in movable:
            if surplus[z] >= 1:
                donors.append((vid, z))
                surplus[z] -= 1
        if not donors or not slots:
            return
        b = self.bucket(t)
        cost = self.net.tt[b][np.ix_([z for _, z in donors], slots)]
        rows, cols = linear_sum_assignment(cost)
        for rr, cc in zip(rows, cols, strict=True):
            if cost[rr, cc] > self.cfg.rebalance_max_tt_s:
                continue
            v = self.vehicles[donors[rr][0]]
            self.set_state(v, REPOSITION, t)
            arrive = self.drive(v, slots[cc], t, "reposition")
            self.push(arrive, EV_REPOS_DONE, v.vid, v.token)

    # -------------------------------------------------------------------- run
    def _tick(self, t: float) -> None:
        slot = int(t // (SLOT_MIN * 60))
        if self._current_slot is None:
            self._current_slot = slot
        if slot != self._current_slot:  # close the finished slot in the forecaster
            ts = self.t0 + pd.Timedelta(seconds=self._current_slot * SLOT_MIN * 60)
            sow = ts.dayofweek * 96 + ts.hour * 4 + ts.minute // SLOT_MIN
            self.forecaster.update(int(sow), float(self._slot_arrivals))
            self._slot_arrivals = 0
            self._current_slot = slot
        self._settle_idle(t)
        self.policy.on_tick(self, t)
        if self.cfg.rebalance and t % self.cfg.rebalance_every_s < 1e-6:
            self.rebalance(t)

    def _settle_idle(self, t: float) -> None:
        """Book idle drain so far and let the policy re-check parked vehicles.

        Without this a vehicle that is never dispatched would drain below its
        charging trigger unnoticed, and dispatch would see a stale SoC.
        """
        parked = [self.vehicles[vid] for z in range(self.net.n) for vid in self.idle_by_zone[z]]
        for v in parked:
            self.debit(v, self.energy_model.idle_kwh(t - v.t_state), "idle")
            self._account_state(v, t)
            v.t_state = t
        for v in parked:
            if v.state == IDLE:
                decision = self.policy.on_available(self, v, t)
                if decision is not None:
                    self.send_to_charge(v, decision[0], decision[1], t)

    def _outage(self, did: int, t: float, start: bool) -> None:
        d = self.depots[did]
        self._depot_energy_cache.clear()
        if start:
            d.online = False
            for vid in list(d.charging):
                self.unplug(self.vehicles[vid], t)
                self.vehicle_available(self.vehicles[vid], t)
            while d.queue:
                v = self.vehicles[d.queue.popleft()]
                if v.state == QUEUED:
                    v.token += 1
                    alt = self.policy.choose_depot(self, v, t)
                    if alt is None:
                        self.vehicle_available(v, t)
                    else:
                        self.send_to_charge(v, alt, v.target, t)
        else:
            d.online = True
            self._start_next(d, t)

    def run(self) -> SimResult:
        import time

        wall = time.perf_counter()
        for r, tr in enumerate(self.r_t):
            self.push(tr, EV_REQUEST, r)
        t = 0.0
        while t < self.t_end:
            self.push(t, EV_TICK, 0)
            t += self.cfg.tick_s
        for zone_id, a, b in self.cfg.outages:
            did = self.depot_by_zone[self.net.index[zone_id]].did
            self.push((pd.Timestamp(a) - self.t0).total_seconds(), EV_OUTAGE, did, 1)
            self.push((pd.Timestamp(b) - self.t0).total_seconds(), EV_OUTAGE, did, 0)

        while self._events:
            t, _, kind, obj, token = heapq.heappop(self._events)
            if t >= self.t_end:
                break
            self.n_events += 1
            if kind == EV_REQUEST:
                self._slot_arrivals += 1
                if not self.try_assign(obj, t):
                    self.pending_by_zone[self.r_o[obj]].append(obj)
                    self.push(self.r_t[obj] + self.cfg.max_wait_s, EV_EXPIRE, obj)
            elif kind == EV_EXPIRE:
                if self.r_status[obj] == 0:
                    self.r_status[obj] = 2
            elif kind == EV_TICK:
                self._tick(t)
            elif kind == EV_OUTAGE:
                self._outage(obj, t, start=bool(token))
            else:
                v = self.vehicles[obj]
                if token != v.token:
                    continue  # stale (e.g. charging interrupted)
                if kind == EV_PICKUP:
                    r = v.req
                    self.r_pickup[r] = t
                    self.set_state(v, WITH_RIDER, t)
                    self.debit(v, self.leg_energy(self.r_km[r], self.r_dur[r], t), "drive_rider")
                    if t >= self.t_warm:
                        self.km["with_rider"] += self.r_km[r]
                    v.zone = self.r_d[r]
                    self.push(t + BOARDING_S + self.r_dur[r], EV_DROPOFF, v.vid, v.token)
                elif kind == EV_DROPOFF:
                    self.vehicle_available(v, t)
                elif kind == EV_AT_DEPOT:
                    self._at_depot(v, t)
                elif kind == EV_CHARGED:
                    self._finish_charge(v, t)
                    self.vehicle_available(v, t)
                elif kind == EV_REPOS_DONE:
                    self.vehicle_available(v, t)
        return self._finalize(time.perf_counter() - wall)

    # --------------------------------------------------------------- results
    def _finalize(self, runtime: float) -> SimResult:
        t = self.t_end
        for v in self.vehicles:
            if v.state in (IDLE, QUEUED):
                self.debit(v, self.energy_model.idle_kwh(t - v.t_state), "idle")
            elif v.state == CHARGING:
                d = self.depots[v.depot]
                soc1 = self.current_soc(v, t)
                self.charged_battery += (soc1 - v.soc_at_plug) * self.cfg.battery_kwh
                self.sessions.append((v.vid, d.did, v.queued_at, v.plug_start, t, v.soc_at_plug, soc1))
                v.soc = soc1
            self._account_state(v, t)

        req = pd.DataFrame({
            "t": self.r_t, "origin": self.net.zone_ids[self.r_o], "dest": self.net.zone_ids[self.r_d],
            "km": self.r_km, "served": self.r_status == 1, "lost": self.r_status == 2,
            "wait_s": self.r_pickup - self.r_t,
        })
        req = req[(req.t >= self.t_warm) & (req.t < self.t_end - self.cfg.max_wait_s)]
        ses = pd.DataFrame(self.sessions, columns=["vid", "depot", "arrived", "plug_start", "end", "soc0", "soc1"])
        ses["queue_wait_s"] = (ses["plug_start"] - self.curve.plug_overhead_s) - ses["arrived"]
        ses["battery_kwh"] = (ses["soc1"] - ses["soc0"]) * self.cfg.battery_kwh
        ses["grid_kwh"] = ses["battery_kwh"] / self.curve.efficiency
        load = self._load_profile(ses)
        final = sum(v.soc for v in self.vehicles) * self.cfg.battery_kwh
        return SimResult(
            config=self.cfg, policy=self.policy.name, requests=req, sessions=ses,
            state_hours={n: h / 3600 for n, h in zip(STATE_NAMES, self.state_time, strict=True)},
            km=dict(self.km),
            energy={"initial": self.initial_energy, "charged_battery": self.charged_battery,
                    "consumed_total": sum(self.consumed.values()), **{f"consumed_{k}": x for k, x in self.consumed.items()},
                    "depletion_shortfall": self.depletion_shortfall, "depletions": self.depletions, "final": final},
            load_kw=load, runtime_s=runtime, events=self.n_events,
        )

    def _load_profile(self, ses: pd.DataFrame, step: float = 300.0) -> pd.Series:
        """Exact grid-side charging energy per 5-minute bin (from the charge curve)."""
        n = int(math.ceil(self.t_end / step))
        energy = np.zeros(n)
        for s0, start, end in zip(ses["soc0"], ses["plug_start"], ses["end"], strict=True):
            if end <= start:
                continue
            k0, k1 = int(start // step), int(min(end, self.t_end - 1e-9) // step)
            prev = s0
            for k in range(k0, k1 + 1):
                edge = min((k + 1) * step, end)
                soc = self.curve.soc_after(s0, edge - start)
                energy[k] += (soc - prev) * self.cfg.battery_kwh / self.curve.efficiency
                prev = soc
        idx = self.t0 + pd.to_timedelta(np.arange(n) * step, unit="s")
        return pd.Series(energy / (step / 3600.0), index=idx, name="grid_kw")
