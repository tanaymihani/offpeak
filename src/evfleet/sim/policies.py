"""Charging policies: when a vehicle charges, where, and to what level.

All policies share the simulator's dispatch, rebalancing and demand, so any
difference in outcomes on the same seed is attributable to charging decisions.
Parameters below were fixed on the validation week, never on the test week.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from evfleet.sim.engine import IDLE, Depot, Simulator, Vehicle


class ThresholdPolicy:
    """Charge when SoC drops below ``threshold``; go to the nearest charger; stop at ``target``."""

    preempt_soc: float | None = None

    def __init__(self, threshold: float = 0.20, target: float = 0.80, name: str | None = None):
        self.threshold, self.target = threshold, target
        self.name = name or f"threshold_{round(target * 100)}"

    def on_available(self, sim: Simulator, v: Vehicle, t: float):
        if v.soc < self.threshold:
            depot = self.choose_depot(sim, v, t)
            if depot is not None:
                return depot, self.target
        return None

    def on_tick(self, sim: Simulator, t: float) -> None:
        return None

    def choose_depot(self, sim: Simulator, v: Vehicle, t: float) -> Depot | None:
        b = sim.bucket(t)
        online = [d for d in sim.depots if d.online]
        if not online:
            return None
        return min(online, key=lambda d: sim.net.tt[b, v.zone, d.zone])


class QueueAwarePolicy(ThresholdPolicy):
    """Same trigger, but pick the depot with the earliest *expected plug-in time*.

    Expected plug-in accounts for travel, vehicles charging (known end times),
    the queue and vehicles already driving there (FIFO by arrival).
    """

    def __init__(self, threshold: float = 0.20, target: float = 0.80):
        super().__init__(threshold, target, name="queue_aware")

    def choose_depot(self, sim: Simulator, v: Vehicle, t: float) -> Depot | None:
        b = sim.bucket(t)
        best, best_start = None, math.inf
        for d in sim.depots:
            if not d.online or not sim.can_reach(v, d, t):
                continue
            arrival = t + sim.net.tt[b, v.zone, d.zone]
            start = sim.expected_start(d, arrival, exclude=v.vid)
            if start < best_start:
                best, best_start = d, start
        # nothing reachable with margin: fall back to the nearest charger
        return best if best is not None else ThresholdPolicy.choose_depot(self, sim, v, t)


class OrchestratedPolicy(QueueAwarePolicy):
    """Forecast- and price-aware orchestration on top of queue-aware charging.

    Every tick it (1) forecasts requests over the next hour, (2) keeps an idle
    buffer sized to that forecast, (3) sends *surplus* idle vehicles to free
    chargers, lowest SoC first, with an opportunistic SoC ceiling that is higher
    in cheap day-ahead-price hours and lower in expensive ones, and (4) matches
    those vehicles to charger slots by minimum total travel time (Hungarian
    algorithm). When vehicles are scarce the charging trigger is lowered so
    vehicles keep serving, and partly charged vehicles may be unplugged to serve
    waiting riders (``preempt_soc``). Energy feasibility is still enforced by
    dispatch, so a lower trigger cannot strand a vehicle.
    """

    def __init__(self, threshold: float = 0.20, scarce_threshold: float = 0.12, target: float = 0.80,
                 buffer_share: float = 0.06, buffer_minutes: float = 15.0,
                 soc_ceiling: tuple[float, float, float] = (0.70, 0.55, 0.35),
                 preempt_soc: float | None = 0.55, reserve_per_depot: int = 1,
                 power_cap_kw: float | None = None, topups: bool = True, name: str = "orchestrated"):
        super().__init__(threshold, target)
        self.name = name
        self.scarce_threshold = scarce_threshold
        self.buffer_share, self.buffer_minutes = buffer_share, buffer_minutes
        self.soc_ceiling = soc_ceiling  # (cheap, normal, expensive) hours
        self.preempt_soc = preempt_soc
        self.reserve_per_depot = reserve_per_depot  # chargers kept free for low-SoC arrivals
        self.power_cap_kw = power_cap_kw  # fleet charging power above which top-ups pause
        self.topups = topups  # opportunistic charging of surplus idle vehicles (ablation switch)
        self._spare = 0.0

    def idle_buffer(self, sim: Simulator, t: float) -> float:
        slot = sim.current_slot_of_week(t)
        per_slot_next_hour = sim.forecaster.forecast(slot, horizon=4) / 4.0
        demand_buffer = per_slot_next_hour * self.buffer_minutes / 15.0
        return max(self.buffer_share * sim.cfg.fleet_size, demand_buffer)

    def on_available(self, sim: Simulator, v: Vehicle, t: float):
        trigger = self.scarce_threshold if self._spare < 0 else self.threshold
        if v.soc < trigger:
            depot = self.choose_depot(sim, v, t)
            if depot is not None:
                return depot, self.target
        return None

    def on_tick(self, sim: Simulator, t: float) -> None:
        idle = [sim.vehicles[vid] for z in range(sim.net.n) for vid in sim.idle_by_zone[z]]
        self._spare = len(idle) - self.idle_buffer(sim, t)
        if self._spare < 1 or not self.topups:
            return
        rank = sim.price_rank(t)
        ceiling = self.soc_ceiling[0] if rank <= 1 / 3 else self.soc_ceiling[2] if rank >= 0.75 else self.soc_ceiling[1]
        candidates = sorted((v for v in idle if v.soc < ceiling), key=lambda v: v.soc)
        slots = []
        for d in sim.depots:
            if d.online:
                slots += [d] * max(d.free - len(d.incoming) - self.reserve_per_depot, 0)
        k = min(int(self._spare), len(candidates), len(slots))
        if self.power_cap_kw is not None:
            load = sim.charging_power_kw(t)
            k = min(k, max(int((self.power_cap_kw - load) // sim.cfg.charger_kw), 0))
        if k <= 0:
            return
        chosen = candidates[:k]
        b = sim.bucket(t)
        cost = np.array([[sim.net.tt[b, v.zone, d.zone] for d in slots] for v in chosen])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            v = chosen[r]
            if v.state == IDLE:
                sim.send_to_charge(v, slots[c], self.target, t)
