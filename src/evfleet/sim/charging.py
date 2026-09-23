"""DC fast-charging curve: power tapers as the battery fills.

Accepted power is ``min(charger_kw, c_rate(soc) * capacity)`` with a
piecewise-linear C-rate curve (representative of lithium-ion fast charging:
flat up to about 50%, tapering steeply above 80%). Time-to-charge is tabulated once
by integrating ``capacity / P(soc)`` on a fine SoC grid, so

* ``duration_s(s0, s1)`` is a table difference, and
* ``soc_after(s0, dt)`` (used when a vehicle is unplugged early) is the
  inverse of the same monotone table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SOC_PTS = (0.0, 0.1, 0.5, 0.8, 0.9, 1.0)
C_RATE_PTS = (1.5, 2.5, 2.5, 1.2, 0.6, 0.2)


@dataclass(frozen=True)
class ChargeCurve:
    capacity_kwh: float = 40.0
    charger_kw: float = 150.0
    efficiency: float = 0.92  # grid -> battery
    plug_overhead_s: float = 120.0  # connect, handshake, disconnect
    soc_pts: tuple[float, ...] = SOC_PTS
    c_rate_pts: tuple[float, ...] = C_RATE_PTS
    grid_points: int = 4001
    _soc: np.ndarray = field(init=False, repr=False, compare=False)
    _t: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        soc = np.linspace(0.0, 1.0, self.grid_points)
        p = self.power_kw(soc)
        inv = self.capacity_kwh / p * 3600.0  # seconds per unit SoC
        t = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(soc))])
        object.__setattr__(self, "_soc", soc)
        object.__setattr__(self, "_t", t)

    def power_kw(self, soc):
        c = np.interp(soc, self.soc_pts, self.c_rate_pts)
        return np.minimum(self.charger_kw, c * self.capacity_kwh)

    def time_from_empty(self, soc: float) -> float:
        return float(np.interp(soc, self._soc, self._t))

    def duration_s(self, soc0: float, soc1: float) -> float:
        """Seconds of active charging from ``soc0`` to ``soc1`` (no plug overhead)."""
        if soc1 <= soc0:
            return 0.0
        return self.time_from_empty(soc1) - self.time_from_empty(soc0)

    def soc_after(self, soc0: float, seconds: float) -> float:
        """State of charge after charging ``seconds`` starting from ``soc0``."""
        if seconds <= 0:
            return soc0
        return float(np.interp(self.time_from_empty(soc0) + seconds, self._t, self._soc))

    def grid_kwh(self, soc0: float, soc1: float) -> float:
        return max(soc1 - soc0, 0.0) * self.capacity_kwh / self.efficiency
