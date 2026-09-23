"""What happened when Palo Alto started charging $0.23/kWh on 1 August 2017?

Every station switched from free to paid on the same day, so there is no
untreated comparison station. Three complementary quasi-experimental designs:

1. Interrupted time series (regression discontinuity in time) on a daily panel
   of stations operating throughout, with separate pre/post trends,
   day-of-week, holiday and weather controls, and Newey-West (HAC) standard
   errors for serial correlation; bandwidth sensitivity 60-180 days.
2. Year-over-year difference-in-differences: each day in 2017 is compared with
   the same weekday 364 days earlier, and the post-August change in that
   year-over-year growth is the effect. A placebo runs the identical analysis on
   2016 vs 2015 with a fake 1 August 2016 fee date, where the effect should be ~0.
3. User cohorts: of the drivers who charged in Feb-Jul, how many kept charging
   in Aug-Dec, and how did the stayers' behaviour change - 2017 vs 2016, with
   2016 vs 2015 as a placebo, and bootstrap intervals over drivers.

Identifying assumption (2, 3): absent the fee, 2017 would have followed 2016's
seasonal pattern around a smooth growth trend. The fee is a price on *these*
public stations; drivers may have shifted to other chargers we cannot observe,
so effects are on demand at city stations, not on total EV charging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEE_DATE = pd.Timestamp("2017-08-01")
OUTCOMES = {
    "sessions": "sessions per day",
    "energy_kwh": "energy per day (kWh)",
    "users": "unique drivers per day",
    "mean_energy": "energy per session (kWh)",
    "mean_connected_h": "connection time per session (h)",
    "mean_idle_h": "idle (plugged, not charging) time per session (h)",
}


# ---------------------------------------------------------------- inference
def ols_hac(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett kernel) standard errors."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    xe = x * (y - x @ beta)[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    n, k = x.shape
    cov = xtx_inv @ s @ xtx_inv * n / (n - k)
    return beta, np.sqrt(np.diag(cov))


# -------------------------------------------------------------------- panel
def balanced_stations(feats: pd.DataFrame, start: str, end: str) -> list[str]:
    """Stations with sessions within 30 days of both ends of [start, end)."""
    g = feats.groupby("station")["start_ts"].agg(["min", "max"])
    ok = (g["min"] <= pd.Timestamp(start) + pd.Timedelta(days=30)) & (g["max"] >= pd.Timestamp(end) - pd.Timedelta(days=30))
    return sorted(g.index[ok])


def daily_panel(feats: pd.DataFrame, stations: list[str], start: str, end: str) -> pd.DataFrame:
    f = feats[feats.station.isin(stations) & (feats.start_ts >= start) & (feats.start_ts < end)]
    day = f.start_ts.dt.normalize()
    agg = f.groupby(day).agg(
        sessions=("raw_row", "size"), energy_kwh=("energy_kwh", "sum"), users=("user_id", "nunique"),
        mean_energy=("energy_kwh", "mean"), mean_connected_h=("connected_h", "mean"), mean_idle_h=("idle_h", "mean"),
        temp_c=("temp_c", "mean"), precip_mm=("precip_mm", "mean"), holiday=("holiday", "max"),
    )
    idx = pd.date_range(start, end, freq="D", inclusive="left")
    agg = agg.reindex(idx)
    agg["dow"] = agg.index.dayofweek
    agg["holiday"] = agg["holiday"].fillna(False).astype(bool)
    return agg


def _controls(p: pd.DataFrame) -> list[np.ndarray]:
    cols = [(p["dow"] == d).astype(float).to_numpy() for d in range(1, 7)]
    cols.append(p["holiday"].astype(float).to_numpy())
    cols.append(p["temp_c"].fillna(p["temp_c"].mean()).to_numpy())
    cols.append(p["precip_mm"].fillna(0).to_numpy())
    return cols


# ---------------------------------------------------------------- design 1
def its(panel: pd.DataFrame, outcome: str, bandwidth_days: int, date: pd.Timestamp = FEE_DATE,
        lags: int = 14) -> dict:
    p = panel[(panel.index >= date - pd.Timedelta(days=bandwidth_days)) &
              (panel.index < date + pd.Timedelta(days=bandwidth_days))].dropna(subset=[outcome])
    p = p[p[outcome] > 0]
    rel = (p.index - date).days.to_numpy(float) / 30.0  # months
    post = (rel >= 0).astype(float)
    x = np.column_stack([np.ones(len(p)), post, rel, rel * post, *_controls(p)])
    beta, se = ols_hac(np.log(p[outcome].to_numpy(float)), x, lags)
    return _effect_row("its", outcome, beta[1], se[1], len(p), bandwidth_days=bandwidth_days)


# ---------------------------------------------------------------- design 2
def yoy_frame(panel: pd.DataFrame, year: int, outcome: str) -> pd.DataFrame:
    cur = panel[(panel.index >= f"{year}-02-01") & (panel.index < f"{year + 1}-01-01")]
    prev_idx = cur.index - pd.Timedelta(days=364)
    prev = panel.reindex(prev_idx)
    ok = (cur[outcome].to_numpy() > 0) & (prev[outcome].to_numpy() > 0)
    d = pd.DataFrame({
        "delta": np.log(cur[outcome].to_numpy(float)) - np.log(prev[outcome].to_numpy(float)),
        "hol": cur["holiday"].astype(float).to_numpy() - prev["holiday"].astype(float).to_numpy(),
        "dtemp": cur["temp_c"].to_numpy() - prev["temp_c"].to_numpy(),
        "dprecip": cur["precip_mm"].to_numpy() - prev["precip_mm"].to_numpy(),
        "dow": cur["dow"].to_numpy(),
    }, index=cur.index)
    return d[ok].dropna()


def yoy_did(panel: pd.DataFrame, outcome: str, year: int = 2017, lags: int = 14) -> dict:
    d = yoy_frame(panel, year, outcome)
    post = (d.index >= pd.Timestamp(f"{year}-08-01")).astype(float)
    x = np.column_stack([np.ones(len(d)), post, d["hol"], d["dtemp"].fillna(0), d["dprecip"].fillna(0)])
    beta, se = ols_hac(d["delta"].to_numpy(), x, lags)
    design = "yoy_did" if year == FEE_DATE.year else f"yoy_placebo_{year}"
    return _effect_row(design, outcome, beta[1], se[1], len(d))


def yoy_event_study(panel: pd.DataFrame, outcome: str, year: int) -> pd.DataFrame:
    """Monthly mean YoY log change, re-centred on the Feb-Jul mean (for plotting)."""
    d = yoy_frame(panel, year, outcome)
    base = d.loc[d.index < f"{year}-08-01", "delta"].mean()
    m = d.groupby(d.index.month)["delta"].agg(["mean", "std", "size"])
    m["effect"] = m["mean"] - base
    m["se"] = m["std"] / np.sqrt(m["size"])
    return m.reset_index(names="month").assign(year=year, outcome=outcome)


def _effect_row(design: str, outcome: str, b: float, se: float, n: int, **extra) -> dict:
    return {"design": design, "outcome": outcome, "log_effect": b, "se": se, "n_days": n,
            "pct_effect": np.exp(b) - 1, "pct_ci_low": np.exp(b - 1.96 * se) - 1,
            "pct_ci_high": np.exp(b + 1.96 * se) - 1, **extra}


# ---------------------------------------------------------------- design 3
def cohort(feats: pd.DataFrame, year: int) -> pd.DataFrame:
    f = feats[feats.user_id.notna()]
    pre = f[(f.start_ts >= f"{year}-02-01") & (f.start_ts < f"{year}-08-01")]
    post = f[(f.start_ts >= f"{year}-08-01") & (f.start_ts < f"{year + 1}-01-01")]
    users = pd.DataFrame(index=pd.Index(pre.user_id.unique(), name="user_id"))
    users["pre_sessions_pm"] = pre.groupby("user_id").size() / 6.0
    users["post_sessions_pm"] = post.groupby("user_id").size().reindex(users.index, fill_value=0) / 5.0
    users["pre_kwh"] = pre.groupby("user_id").energy_kwh.mean()
    users["post_kwh"] = post.groupby("user_id").energy_kwh.mean().reindex(users.index)
    users["pre_idle_h"] = pre.groupby("user_id").idle_h.mean()
    users["post_idle_h"] = post.groupby("user_id").idle_h.mean().reindex(users.index)
    users["retained"] = users.post_sessions_pm > 0
    return users


def cohort_stats(users: pd.DataFrame) -> dict:
    kept = users[users.retained]
    return {
        "retention": users.retained.mean(),
        "stayer_log_change_sessions": np.log(kept.post_sessions_pm / kept.pre_sessions_pm).mean(),
        "stayer_change_kwh": (kept.post_kwh - kept.pre_kwh).mean(),
        "stayer_change_idle_h": (kept.post_idle_h - kept.pre_idle_h).mean(),
    }


def cohort_did(feats: pd.DataFrame, treated: int, control: int, reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    """(treated-year change) - (control-year change) with a driver bootstrap."""
    a, b = cohort(feats, treated), cohort(feats, control)
    sa, sb = cohort_stats(a), cohort_stats(b)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in sa}
    for _ in range(reps):
        ra = cohort_stats(a.iloc[rng.integers(0, len(a), len(a))])
        rb = cohort_stats(b.iloc[rng.integers(0, len(b), len(b))])
        for k in sa:
            boot[k].append(ra[k] - rb[k])
    rows = []
    for k in sa:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append({"treated_year": treated, "control_year": control, "metric": k, "treated": sa[k],
                     "control": sb[k], "did": sa[k] - sb[k], "ci_low": lo, "ci_high": hi,
                     "n_treated": len(a), "n_control": len(b)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ runner
def run_all(feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    its_st = balanced_stations(feats, "2017-02-01", "2018-02-01")
    its_panel = daily_panel(feats, its_st, "2017-01-01", "2018-03-01")
    yoy_st = balanced_stations(feats, "2016-01-01", "2018-01-01")
    yoy_panel = daily_panel(feats, yoy_st, "2015-01-01", "2018-01-01")
    plc_st = balanced_stations(feats, "2015-01-01", "2017-01-01")
    plc_panel = daily_panel(feats, plc_st, "2014-01-01", "2017-01-01")

    rows = []
    for o in OUTCOMES:
        for bw in (60, 90, 120, 180):
            rows.append(its(its_panel, o, bw))
        rows.append(yoy_did(yoy_panel, o, 2017))
        rows.append(yoy_did(plc_panel, o, 2016))
    effects = pd.DataFrame(rows)
    effects["stations"] = effects["design"].map(
        {"its": len(its_st), "yoy_did": len(yoy_st), "yoy_placebo_2016": len(plc_st)})

    events = pd.concat([yoy_event_study(yoy_panel, o, 2017) for o in OUTCOMES] +
                       [yoy_event_study(plc_panel, o, 2016) for o in OUTCOMES], ignore_index=True)
    cohorts = pd.concat([cohort_did(feats, 2017, 2016), cohort_did(feats, 2016, 2015)], ignore_index=True)
    daily = yoy_panel.assign(stations=len(yoy_st))
    return {"effects": effects, "event_study": events, "cohorts": cohorts, "daily_panel": daily}
