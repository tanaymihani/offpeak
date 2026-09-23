import math

import numpy as np
import pandas as pd
import pytest

from evfleet.sessions.causal import ols_hac
from evfleet.sessions.conformal import conformal_quantile, coverage, cqr_scores, rolling_cqr


def test_conformal_quantile_order_statistic():
    scores = np.arange(1, 10, dtype=float)  # n = 9
    # k = ceil(10 * 0.8) = 8 -> 8th smallest
    assert conformal_quantile(scores, 0.2) == 8.0
    # k = ceil(10 * 0.95) = 10 > n -> unbounded
    assert conformal_quantile(scores, 0.05) == math.inf
    assert conformal_quantile(np.array([]), 0.2) == math.inf


def test_split_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        y_cal, y_test = rng.normal(size=200), rng.normal(size=500)
        lo, hi = -0.5, 0.5  # deliberately too narrow quantile predictions
        q = conformal_quantile(cqr_scores(y_cal, lo, hi), 0.2)
        covs.append(coverage(y_test, lo - q, hi + q))
    assert np.mean(covs) == pytest.approx(0.8, abs=0.01)
    assert np.mean(covs) >= 0.795


def test_rolling_cqr_uses_only_matured_labels():
    days = pd.date_range("2020-01-01", periods=3, freq="D")
    # the calibration point ends *after* the day starts, so it cannot be used that day
    frame = pd.DataFrame({"start_ts": days + pd.Timedelta(hours=9), "end_ts": days + pd.Timedelta(hours=40),
                          "y": [0.0, 0.0, 0.0], "lo": [-1.0, -1.0, -1.0], "hi": [1.0, 1.0, 1.0]})
    out = rolling_cqr(frame, alpha=0.5, window=10)
    assert math.isinf(out.q.iloc[0])  # nothing matured by 2020-01-01 00:00
    assert math.isinf(out.q.iloc[1])  # first session ends 2020-01-02 16:00
    assert np.isfinite(out.q.iloc[2])


def test_hac_with_zero_lags_is_white_hc0_times_dof():
    rng = np.random.default_rng(1)
    n = 400
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = 1 + 2 * x[:, 1] + rng.normal(size=n) * (1 + np.abs(x[:, 1]))
    beta, se = ols_hac(y, x, lags=0)
    e = y - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    hc0 = xtx_inv @ (x.T * e**2) @ x @ xtx_inv * n / (n - 2)
    assert np.allclose(se, np.sqrt(np.diag(hc0)))
    assert beta[1] == pytest.approx(2, abs=0.2)


def test_hac_widens_errors_under_autocorrelation():
    rng = np.random.default_rng(2)
    n = 2000
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + rng.normal()
    x = np.column_stack([np.ones(n), np.cumsum(rng.normal(size=n)) / 30])
    y = x @ np.array([0.0, 1.0]) + u
    _, se0 = ols_hac(y, x, lags=0)
    _, se20 = ols_hac(y, x, lags=20)
    assert se20[0] > 1.5 * se0[0]
