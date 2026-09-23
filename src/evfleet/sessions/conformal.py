"""Conformalized quantile regression (CQR) and label-maturity-aware recalibration.

Split conformal: with calibration nonconformity scores s_1..s_n and miscoverage
alpha, the interval correction is the k-th smallest score with
k = ceil((n + 1)(1 - alpha)). If k > n the correction is +inf, because there is
not enough calibration data for that coverage level. Under exchangeability this
gives marginal coverage >= 1 - alpha. Charging sessions over time are not
exchangeable, so I measure coverage on held-out time periods instead of assuming it.

For CQR the score of a calibration point with lower/upper quantile predictions
(lo, hi) is max(lo - y, y - hi) (Romano, Patterson & Candes, 2019).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def cqr_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(lo) - y, np.asarray(y) - hi)


def cqr_interval(lo: np.ndarray, hi: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(lo) - q, np.asarray(hi) + q


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def rolling_cqr(frame: pd.DataFrame, alpha: float, window: int = 3000,
                warm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Nightly recalibration using only *matured* labels.

    ``frame`` has columns start_ts, end_ts, y, lo, hi (raw quantile predictions).
    Each day, the correction is recomputed from the most recent ``window``
    sessions whose outcome was known by midnight (end_ts <= day start); a
    session's dwell time only becomes observable when it ends. ``warm``
    (same columns) seeds the window with earlier matured sessions.
    """
    pool = frame if warm is None else pd.concat([warm, frame], ignore_index=True)
    pool = pool.assign(score=cqr_scores(pool["y"], pool["lo"], pool["hi"])).sort_values("end_ts")
    ends = pool["end_ts"].to_numpy()
    scores = pool["score"].to_numpy()
    out = frame.copy()
    q = np.empty(len(frame))
    days = frame["start_ts"].dt.normalize()
    for day, idx in frame.groupby(days).groups.items():
        k = np.searchsorted(ends, np.datetime64(day), side="right")
        recent = scores[max(0, k - window) : k]
        q[frame.index.get_indexer(idx)] = conformal_quantile(recent, alpha)
    out["q"] = q
    out["lo_adj"], out["hi_adj"] = out["lo"] - q, out["hi"] + q
    return out
