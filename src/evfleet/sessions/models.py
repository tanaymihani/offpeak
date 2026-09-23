"""Predict, at plug-in time, how long a car will stay and how much energy it will take.

Use cases: tell a driver when their car will be done, tell the next driver when a
port is likely to free up, and give operators a load forecast per session.

Chronological splits (no shuffling):
    train  2011-07 .. 2018-12
    calib  2019-01 .. 2019-06   (conformal calibration only)
    test   2019-07 .. 2019-12   (held out, pre-COVID)
    shift  2020-01 .. 2020-12   (COVID distribution shift; stress test)

Models: three baselines, then gradient-boosted quantile regression
(q = 0.1, 0.5, 0.9) on log targets, with CQR for calibrated 80 % intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from evfleet.sessions.conformal import conformal_quantile, coverage, cqr_interval, cqr_scores, rolling_cqr

SPLITS = {
    "train": (None, "2019-01-01"),
    "calib": ("2019-01-01", "2019-07-01"),
    "test": ("2019-07-01", "2020-01-01"),
    "shift": ("2020-01-01", "2021-01-01"),
}
ALPHA = 0.2  # 80 % intervals
QUANTILES = (0.1, 0.5, 0.9)

FEATURES = [
    "hour_sin", "hour_cos", "dow", "weekend", "month", "holiday", "site_code", "level1", "local_driver",
    "anonymous", "fee_regime", "temp_c", "precip_mm", "site_active_at_start", "log_u_prior_n",
    "u_prior_mean_log_dwell", "u_prior_median_dwell", "u_prior_mean_energy", "u_prior_mean_idle",
    "u_prior_mean_start_hour", "hour_vs_user_usual", "u_prior_same_site_share", "u_last_dwell",
    "u_last_energy", "log_u_hours_since_last",
]
CATEGORICAL = ["site_code", "dow"]
TARGETS = {"dwell": "connected_h", "energy": "energy_kwh"}


def split(feats: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = SPLITS[name]
    m = pd.Series(True, index=feats.index)
    if a:
        m &= feats["start_ts"] >= pd.Timestamp(a)
    if b:
        m &= feats["start_ts"] < pd.Timestamp(b)
    return feats.loc[m]


def design(feats: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=feats.index)
    ang = 2 * np.pi * feats["start_hour"] / 24
    x["hour_sin"], x["hour_cos"] = np.sin(ang), np.cos(ang)
    x["dow"] = feats["dow"].astype(int) - 1
    x["weekend"] = (feats["dow"] >= 6).astype(int)
    x["month"] = feats["month"].astype(int)
    x["holiday"] = feats["holiday"].astype(int)
    code = {s: k for k, s in enumerate(sites)}
    x["site_code"] = feats["site"].map(code).fillna(len(sites)).astype(int)
    x["level1"] = (feats["port_type"] == "Level 1").astype(int)
    x["local_driver"] = feats["local_driver"].fillna(False).astype(int)
    x["anonymous"] = feats["anonymous"].astype(int)
    x["fee_regime"] = feats["fee_regime"].astype(int)
    for c in ["temp_c", "precip_mm", "site_active_at_start", "u_prior_mean_log_dwell", "u_prior_median_dwell",
              "u_prior_mean_energy", "u_prior_mean_idle", "u_prior_mean_start_hour", "u_prior_same_site_share",
              "u_last_dwell", "u_last_energy"]:
        x[c] = feats[c].astype(float)
    x["log_u_prior_n"] = np.log1p(feats["u_prior_n"].astype(float))
    x["log_u_hours_since_last"] = np.log1p(feats["u_hours_since_last"].astype(float))
    diff = feats["start_hour"] - feats["u_prior_mean_start_hour"]
    x["hour_vs_user_usual"] = diff.astype(float)
    return x[FEATURES]


def log_target(y: pd.Series, target: str) -> np.ndarray:
    return np.log(y.to_numpy(float)) if target == "dwell" else np.log1p(y.to_numpy(float))


def inverse(z: np.ndarray, target: str) -> np.ndarray:
    return np.exp(z) if target == "dwell" else np.expm1(z)


@dataclass
class QuantileModel:
    target: str
    models: dict[float, HistGradientBoostingRegressor]
    sites: list[str]

    def predict(self, feats: pd.DataFrame) -> dict[float, np.ndarray]:
        x = design(feats, self.sites)
        preds = {q: m.predict(x) for q, m in self.models.items()}
        # enforce non-crossing quantiles
        stack = np.sort(np.vstack([preds[q] for q in QUANTILES]), axis=0)
        return {q: stack[k] for k, q in enumerate(QUANTILES)}


def fit_quantile_model(train: pd.DataFrame, target: str, seed: int = 0) -> QuantileModel:
    sites = sorted(train["site"].unique())
    x = design(train, sites)
    y = log_target(train[TARGETS[target]], target)
    cat = [FEATURES.index(c) for c in CATEGORICAL]
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0, categorical_features=cat, random_state=seed,
        )
        models[q] = m.fit(x, y)
    return QuantileModel(target, models, sites)


# ------------------------------------------------------------------ baselines
def baselines(train: pd.DataFrame, evalset: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    col = TARGETS[target]
    glob = train[col].median()
    key_tr = train["site"] + "|" + (train["dow"] >= 6).astype(str) + "|" + train["start_hour"].astype(int).astype(str)
    key_ev = evalset["site"] + "|" + (evalset["dow"] >= 6).astype(str) + "|" + evalset["start_hour"].astype(int).astype(str)
    lookup = train.groupby(key_tr)[col].median()
    site_hour = key_ev.map(lookup).fillna(glob).to_numpy()
    hist_col = "u_prior_median_dwell" if target == "dwell" else "u_prior_mean_energy"
    enough = (evalset["u_prior_n"] >= 5) & evalset[hist_col].notna()
    user = np.where(enough, evalset[hist_col], site_hour)
    return {"global_median": np.full(len(evalset), glob), "site_x_weekend_x_hour_median": site_hour,
            "user_history_else_site_hour": user}


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {"mae": float(np.mean(np.abs(err))), "median_ae": float(np.median(np.abs(err))),
            "bias": float(np.mean(err))}


def interval_row(name: str, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {"method": name, "coverage": coverage(y, lo, hi), "mean_width": float(np.mean(hi - lo)),
            "median_width": float(np.median(hi - lo))}


def evaluate(feats: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    train, calib, test, shift = (split(feats, k).reset_index(drop=True) for k in SPLITS)
    col = TARGETS[target]
    model = fit_quantile_model(train, target)

    # point accuracy on the held-out test half-year (original units)
    y_test = test[col].to_numpy(float)
    pred_test = model.predict(test)
    rows = []
    for name, pred in baselines(train, test, target).items():
        rows.append({"model": name, **point_metrics(y_test, pred)})
    rows.append({"model": "gbm_quantile_median", **point_metrics(y_test, inverse(pred_test[0.5], target))})
    point = pd.DataFrame(rows)

    # intervals: raw quantile regression vs CQR calibrated on calib (log scale, then mapped back)
    pred_cal = model.predict(calib)
    z_cal = log_target(calib[col], target)
    q_hat = conformal_quantile(cqr_scores(z_cal, pred_cal[0.1], pred_cal[0.9]), ALPHA)
    lo_c, hi_c = cqr_interval(pred_test[0.1], pred_test[0.9], q_hat)
    intervals = pd.DataFrame([
        interval_row("raw_quantile_10_90", y_test, inverse(pred_test[0.1], target), inverse(pred_test[0.9], target)),
        interval_row("cqr_calibrated", y_test, inverse(lo_c, target), inverse(hi_c, target)),
    ])
    intervals["target_coverage"] = 1 - ALPHA
    intervals["q_hat_log"] = [np.nan, q_hat]

    # conditional coverage by segment on the test period
    t = test.assign(covered=(y_test >= inverse(lo_c, target)) & (y_test <= inverse(hi_c, target)),
                    width=inverse(hi_c, target) - inverse(lo_c, target))
    t["user_segment"] = np.select([t.anonymous, t.u_prior_n < 5], ["anonymous", "new_user(<5 prior)"], "returning")
    seg = pd.concat([
        t.groupby("user_segment").agg(n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean"))
         .reset_index().rename(columns={"user_segment": "segment"}).assign(dimension="user"),
        t.groupby("site").agg(n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean"))
         .reset_index().rename(columns={"site": "segment"}).assign(dimension="site"),
        t.assign(daypart=pd.cut(t.start_hour, [0, 6, 10, 15, 19, 24], right=False,
                                labels=["night", "morning", "midday", "evening", "late"]))
         .groupby("daypart", observed=True).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                                mean_width=("width", "mean"))
         .reset_index().rename(columns={"daypart": "segment"}).assign(dimension="arrival_time"),
    ])

    # COVID shift: static calibration vs nightly recalibration on matured labels
    pred_shift = model.predict(shift)
    z_shift = log_target(shift[col], target)
    frame = pd.DataFrame({"start_ts": shift.start_ts, "end_ts": shift.end_ts, "y": z_shift,
                          "lo": pred_shift[0.1], "hi": pred_shift[0.9]})
    warm_parts = []
    for part, pred in ((calib, pred_cal), (test, pred_test)):
        warm_parts.append(pd.DataFrame({"start_ts": part.start_ts, "end_ts": part.end_ts,
                                        "y": log_target(part[col], target), "lo": pred[0.1], "hi": pred[0.9]}))
    rolled = rolling_cqr(frame, ALPHA, window=3000, warm=pd.concat(warm_parts, ignore_index=True))
    month = shift["start_ts"].dt.to_period("M").astype(str)
    static_cov = (z_shift >= pred_shift[0.1] - q_hat) & (z_shift <= pred_shift[0.9] + q_hat)
    roll_cov = (z_shift >= rolled["lo_adj"]) & (z_shift <= rolled["hi_adj"])
    shift_tbl = pd.DataFrame({"month": month, "static": static_cov, "rolling": roll_cov,
                              "abs_err_log": np.abs(pred_shift[0.5] - z_shift)})
    shift_tbl = shift_tbl.groupby("month").agg(n=("static", "size"), static_cqr_coverage=("static", "mean"),
                                               rolling_cqr_coverage=("rolling", "mean"),
                                               median_abs_err_log=("abs_err_log", "median")).reset_index()

    # permutation importance of the median model on a test subsample
    sample = test.sample(min(20000, len(test)), random_state=0)
    imp = permutation_importance(model.models[0.5], design(sample, model.sites), log_target(sample[col], target),
                                 scoring="neg_mean_absolute_error", n_repeats=3, random_state=0)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
                  .sort_values("importance", ascending=False))

    sizes = pd.DataFrame([{"split": k, "sessions": len(v)} for k, v in
                          zip(SPLITS, (train, calib, test, shift), strict=True)])
    return {"point": point, "intervals": intervals, "segments": seg, "shift": shift_tbl,
            "importance": importance, "sizes": sizes,
            "test_pred": pd.DataFrame({"y": y_test, "p50": inverse(pred_test[0.5], target),
                                       "lo": inverse(lo_c, target), "hi": inverse(hi_c, target),
                                       "start_hour": test.start_hour, "site": test.site})}
