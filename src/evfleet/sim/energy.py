"""Vehicle energy consumption model.

Per-kilometre battery energy at average speed ``v`` (km/h) and ambient
temperature ``T`` (degC)::

    e(v, T) = a  +  b * v^2  +  (P_aux + P_hvac(T)) / v        [kWh/km]

* ``a``  rolling resistance plus urban stop-and-go losses (speed independent);
* ``b``  aerodynamic drag, 0.5 * rho * CdA / drivetrain efficiency;
* ``P_aux``  always-on compute, sensors and low-voltage loads while driving;
* ``P_hvac(T)``  cabin heating/cooling outside an 18-24 degC comfort band.

These are representative, stated assumptions for a compact electric robotaxi,
not a fit to vehicle telemetry (none is public). At 13 km/h and 20 degC the
model gives about 0.12 kWh/km; it is used identically for every policy, so
policy comparisons are paired under the same assumption, and the cold-weather
stress test shows sensitivity to the HVAC term.
"""

from __future__ import annotations

from dataclasses import dataclass

RHO_AIR = 1.2  # kg/m^3


@dataclass(frozen=True)
class EnergyModel:
    a_kwh_per_km: float = 0.095
    cda_m2: float = 0.55
    drivetrain_eff: float = 0.88
    aux_kw: float = 0.35
    idle_kw: float = 0.25  # parked and awake, waiting for dispatch
    comfort_low_c: float = 18.0
    comfort_high_c: float = 24.0
    heat_kw_per_c: float = 0.07
    heat_max_kw: float = 3.5
    cool_kw_per_c: float = 0.10
    cool_max_kw: float = 2.5

    @property
    def b_kwh_per_km_kmh2(self) -> float:
        # 0.5 rho CdA v^2 [N] with v in m/s, times 1000 m, in kWh, per (km/h)^2
        return 0.5 * RHO_AIR * self.cda_m2 / self.drivetrain_eff * 1000 / 3.6e6 / 3.6**2

    def hvac_kw(self, temp_c: float) -> float:
        if temp_c < self.comfort_low_c:
            return min(self.heat_max_kw, self.heat_kw_per_c * (self.comfort_low_c - temp_c))
        if temp_c > self.comfort_high_c:
            return min(self.cool_max_kw, self.cool_kw_per_c * (temp_c - self.comfort_high_c))
        return 0.0

    def leg_kwh(self, distance_km: float, duration_s: float, temp_c: float) -> float:
        """Battery energy for a driving leg (time-based loads use the actual duration)."""
        hours = max(duration_s, 1.0) / 3600.0
        v = distance_km / hours
        motion = distance_km * (self.a_kwh_per_km + self.b_kwh_per_km_kmh2 * v * v)
        return motion + (self.aux_kw + self.hvac_kw(temp_c)) * hours

    def per_km(self, speed_kmh: float, temp_c: float) -> float:
        return self.leg_kwh(1.0, 3600.0 / speed_kmh, temp_c)

    def idle_kwh(self, seconds: float) -> float:
        return self.idle_kw * max(seconds, 0.0) / 3600.0
