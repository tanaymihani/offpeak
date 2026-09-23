"""Short-horizon ride-demand forecasting used by the charging orchestrator.

The forecaster is deliberately simple and causal: a slot-of-week seasonal
profile estimated on training weeks, multiplied by an exponentially smoothed
ratio of recently observed demand to the profile (a level correction). It is
evaluated against a seasonal-naive forecast and the raw profile on held-out
days before being used inside the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SLOT_MIN = 15
SLOTS_PER_WEEK = 7 * 24 * 60 // SLOT_MIN  # 672


def slot_of_week(ts: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return (ts.dayofweek * 96 + ts.hour * 4 + ts.minute // SLOT_MIN).to_numpy()


def slot_counts(trips: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Fleet-wide request counts per 15-minute slot over [start, end)."""
    idx = pd.date_range(start, end, freq=f"{SLOT_MIN}min", inclusive="left")
    counts = trips.set_index("pickup_ts").resample(f"{SLOT_MIN}min").size()
    return counts.reindex(idx, fill_value=0).astype(float)


def fleet_profile(train_counts: pd.Series) -> np.ndarray:
    """Mean requests per slot-of-week across the training weeks."""
    sow = slot_of_week(train_counts.index)
    prof = pd.Series(train_counts.to_numpy()).groupby(sow).mean()
    out = np.zeros(SLOTS_PER_WEEK)
    out[prof.index.to_numpy()] = prof.to_numpy()
    return out


def zone_hour_profile(train: pd.DataFrame, zone_ids: np.ndarray, weeks: float) -> np.ndarray:
    """Mean requests per (hour-of-week, zone) from training trips. Shape (168, Z)."""
    index = {int(z): k for k, z in enumerate(zone_ids)}
    how = (train["pickup_ts"].dt.dayofweek * 24 + train["pickup_ts"].dt.hour).to_numpy()
    zi = train["pu"].map(index).to_numpy()
    out = np.zeros((168, len(zone_ids)))
    np.add.at(out, (how, zi), 1.0)
    return out / weeks


@dataclass
class LevelAdjustedProfile:
    """Seasonal profile x smoothed observed/expected ratio."""

    profile: np.ndarray  # (672,) expected requests per slot at scale 1
    scale: float = 1.0  # e.g. demand sampling fraction in the simulator
    alpha: float = 0.3
    clip: tuple[float, float] = (0.5, 2.0)
    ratio: float = field(default=1.0, init=False)

    def expected(self, sow: int) -> float:
        return self.scale * self.profile[sow % SLOTS_PER_WEEK]

    def update(self, sow: int, observed: float) -> None:
        exp = self.expected(sow)
        if exp <= 0:
            return
        r = float(np.clip(observed / exp, *self.clip))
        self.ratio = (1 - self.alpha) * self.ratio + self.alpha * r

    def forecast(self, sow: int, horizon: int = 4) -> float:
        """Expected requests over the ``horizon`` slots starting at ``sow``."""
        return self.ratio * sum(self.expected(sow + h) for h in range(horizon))


def evaluate(train_counts: pd.Series, test_counts: pd.Series, horizon: int = 4) -> pd.DataFrame:
    """One-hour-ahead forecast accuracy on held-out slots.

    At the start of each test slot, every method forecasts the total over the
    next ``horizon`` slots using only data before that slot.
    """
    prof = fleet_profile(train_counts)
    history = pd.concat([train_counts, test_counts])
    idx = test_counts.index
    sow = slot_of_week(idx)
    n = len(idx) - horizon + 1
    actual = np.array([test_counts.iloc[t : t + horizon].sum() for t in range(n)])

    week = SLOTS_PER_WEEK
    pos0 = len(train_counts)
    hist_vals = history.to_numpy()
    naive = np.array([hist_vals[pos0 + t - week : pos0 + t - week + horizon].sum() for t in range(n)])
    profile_only = np.array([sum(prof[(sow[t] + h) % week] for h in range(horizon)) for t in range(n)])

    model = LevelAdjustedProfile(prof)
    # warm the level ratio on the final training day
    for s, c in zip(slot_of_week(train_counts.index[-96:]), train_counts.to_numpy()[-96:], strict=True):
        model.update(int(s), float(c))
    adjusted = np.empty(n)
    for t in range(n):
        adjusted[t] = model.forecast(int(sow[t]), horizon)
        model.update(int(sow[t]), float(test_counts.iloc[t]))

    rows = []
    for name, pred in [("seasonal_naive_1wk", naive), ("profile_mean", profile_only),
                       ("profile_x_level (used)", adjusted)]:
        err = pred - actual
        rows.append({"method": name, "forecasts": n, "mae": np.mean(np.abs(err)),
                     "mape": np.mean(np.abs(err) / np.maximum(actual, 1)),
                     "bias": np.mean(err)})
    return pd.DataFrame(rows)
