"""Regenerate every figure in reports/figures from reports/results (and processed data for maps).

    python scripts/make_figures.py
"""

from __future__ import annotations

import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from evfleet.data.palo_alto import load_features  # noqa: E402
from evfleet.data.tlc import load_trips, load_zone_shapes, load_zones  # noqa: E402
from evfleet.paths import FIGURES, RESULTS, ensure_dirs  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89", "#e7e6e2"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
POLICY_COLOR = {"orchestrated": BLUE, "threshold_80": ORANGE, "queue_aware": AQUA, "threshold_100": YELLOW}
POLICY_LABEL = {"orchestrated": "Orchestrated", "threshold_80": "Threshold 20% -> 80%",
                "queue_aware": "Queue-aware depot choice", "threshold_100": "Threshold 20% -> 100%"}
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", name)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


# ---------------------------------------------------------------- Palo Alto
def pa_timeline(feats: pd.DataFrame) -> None:
    weekly = feats.set_index("start_ts").resample("W").size()
    weekly = weekly[(weekly.index >= "2011-08-07") & (weekly.index <= "2020-12-27")]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(weekly.index, weekly.to_numpy(), color=BLUE, lw=1.6)
    for date, text, frac in [("2017-08-01", "Fee introduced\n$0.23/kWh", 0.25), ("2020-03-17", "Shelter-in-place", 0.97)]:
        ax.axvline(pd.Timestamp(date), color=MUTED, lw=1)
        ax.annotate(text, (pd.Timestamp(date), weekly.max() * frac), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="top", color=INK2, fontsize=9)
    ax.set_title("Palo Alto public charging sessions per week, 2011-2020")
    ax.set_ylabel("sessions / week")
    ax.set_ylim(0, None)
    save(fig, "pa_demand_timeline.png")


def pa_fee_event_study() -> None:
    ev = read("pa_fee_event_study.csv")
    panels = [("sessions", "Sessions per day"), ("mean_energy", "Energy per session"),
              ("mean_idle_h", "Idle time per session")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    months = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax, (outcome, title) in zip(axes, panels, strict=True):
        for year, color, label, dx in [(2016, MUTED, "2016 vs 2015 (placebo, no fee)", -0.12),
                                       (2017, BLUE, "2017 vs 2016 (fee from 1 Aug)", 0.12)]:
            d = ev[(ev.outcome == outcome) & (ev.year == year)].sort_values("month")
            x = d.month.to_numpy() + dx
            y = np.exp(d.effect) - 1
            lo, hi = np.exp(d.effect - 1.96 * d.se) - 1, np.exp(d.effect + 1.96 * d.se) - 1
            ax.vlines(x, lo * 100, hi * 100, color=color, lw=1.4)
            ax.plot(x, y * 100, "o", color=color, ms=5, mec=SURFACE, mew=1.5, label=label)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvspan(7.5, 12.5, color=BLUE, alpha=0.05, lw=0)
        ax.set_xticks(range(2, 13), months, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("change vs Feb-Jul, % (year-over-year)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, -0.08), ncol=2, fontsize=8.5)
    fig.suptitle("Event study: year-over-year change by month, relative to the pre-fee months", x=0.01, ha="left",
                 fontsize=11.5, fontweight="semibold", y=1.03)
    fig.tight_layout()
    save(fig, "pa_fee_event_study.png")


def pa_fee_effects() -> None:
    eff = read("pa_fee_effects.csv")
    eff = eff[(eff.design != "its") | (eff.bandwidth_days == 120)]
    labels = {"sessions": "Sessions / day", "energy_kwh": "Energy / day", "users": "Unique drivers / day",
              "mean_energy": "kWh / session", "mean_connected_h": "Connection time / session",
              "mean_idle_h": "Idle time / session"}
    designs = [("yoy_did", "Year-over-year DiD (2017 vs 2016)", BLUE, -0.22),
               ("its", "Interrupted time series (+/-120 days)", ORANGE, 0.0),
               ("yoy_placebo_2016", "Placebo: fake fee on 1 Aug 2016", MUTED, 0.22)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    order = list(labels)
    for design, name, color, dy in designs:
        d = eff[eff.design == design].set_index("outcome").reindex(order)
        y = np.arange(len(order)) + dy
        ax.hlines(y, d.pct_ci_low * 100, d.pct_ci_high * 100, color=color, lw=1.6)
        ax.plot(d.pct_effect * 100, y, "o", color=color, ms=6, mec=SURFACE, mew=1.5, label=name)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order)), [labels[o] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("estimated effect of the fee, % (95% CI, HAC standard errors)")
    ax.set_title("Effect of introducing a $0.23/kWh fee at Palo Alto public chargers")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=8.5)
    save(fig, "pa_fee_effects.png")


def pa_prediction() -> None:
    point = read("pa_dwell_point.csv")
    seg = read("pa_dwell_segments.csv")
    names = {"global_median": "Global median", "site_x_weekend_x_hour_median": "Site x weekend x hour median",
             "user_history_else_site_hour": "Driver's own history", "gbm_quantile_median": "Gradient boosting (median)"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.45})
    p = point.set_index("model").reindex(list(names))
    colors = [MUTED] * 3 + [BLUE]
    y = np.arange(len(p))
    a.barh(y, p["mae"] * 60, height=0.55, color=colors)
    for yy, v in zip(y, p["mae"] * 60, strict=True):
        a.text(v + 1.5, yy, f"{v:.0f} min", va="center", color=INK, fontsize=9)
    a.set_yticks(y, [names[m] for m in p.index])
    a.invert_yaxis()
    a.set_xlabel("mean absolute error of connection time (minutes)")
    a.set_title("Connection-time error, Jul-Dec 2019")
    a.set_xlim(0, p["mae"].max() * 60 * 1.25)
    a.grid(axis="y", visible=False)

    s = seg[seg.dimension.isin(["user", "arrival_time"])].copy()
    order = ["returning", "new_user(<5 prior)", "anonymous", "night", "morning", "midday", "evening", "late"]
    s = s.set_index("segment").reindex(order).dropna(subset=["coverage"])
    x = np.arange(len(s))
    b.bar(x, s.coverage * 100, width=0.55, color=BLUE)
    b.axhline(80, color=INK2, lw=1)
    b.text(len(s) - 0.45, 80, " target\n 80%", ha="left", va="center", color=INK2, fontsize=8.5)
    for xx, v in zip(x, s.coverage * 100, strict=True):
        b.text(xx, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5)
    short = {"returning": "returning", "new_user(<5 prior)": "new", "anonymous": "anon."}
    ticks = [short.get(o, o) for o in s.index]
    b.set_xticks(x, [f"{t}\nn={n / 1000:.1f}k" for t, n in zip(ticks, s.n, strict=True)], fontsize=8)
    n_user = sum(o in short for o in s.index)
    for lo, hi, text in [(0, n_user - 1, "driver history"), (n_user, len(s) - 1, "arrival time")]:
        b.annotate(text, ((lo + hi) / 2, 0), xytext=(0, -38), textcoords="offset points", ha="center",
                   xycoords=("data", "axes fraction"), fontsize=8.5, color=INK2)
    b.set_xlim(-0.6, len(s) - 0.4)
    b.set_ylim(0, 100)
    b.set_ylabel("coverage of 80% interval, %")
    b.set_title("80% conformal (CQR) interval coverage by segment")
    b.grid(axis="x", visible=False)
    save(fig, "pa_dwell_prediction.png")


def pa_shift() -> None:
    sh = read("pa_dwell_shift.csv")
    x = pd.to_datetime(sh.month)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.plot(x, sh.static_cqr_coverage * 100, "-o", color=ORANGE, ms=5, mec=SURFACE, label="Static: calibrated once on Jan-Jun 2019")
    ax.plot(x, sh.rolling_cqr_coverage * 100, "-o", color=BLUE, ms=5, mec=SURFACE,
            label="Nightly recalibration on sessions that have ended")
    ax.axhline(80, color=INK2, lw=1)
    ax.axvline(pd.Timestamp("2020-03-17"), color=MUTED, lw=1)
    ax.text(pd.Timestamp("2020-03-20"), 52, "shelter-in-place", color=INK2, fontsize=8.5)
    ax.set_ylim(50, 100)
    ax.set_ylabel("coverage of 80% interval, %")
    ax.set_title("COVID shift (2020): static intervals under-cover; nightly recalibration recovers by July")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "pa_conformal_shift.png")


def pa_segments() -> None:
    prof = read("pa_segments.csv")
    shift = read("pa_segments_fee_shift.csv").set_index("segment")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    a.scatter(prof.typical_arrival_hour, prof.median_connected_h, s=prof.share * 4000, color=BLUE, alpha=0.8,
              edgecolor=SURFACE, linewidth=2)
    for _, r in prof.iterrows():
        left = r.typical_arrival_hour < 12 or r.median_connected_h > 4
        a.annotate(f"{r['name']}\n{r.share:.0%} of sessions, {r.median_energy_kwh:.1f} kWh",
                   (r.typical_arrival_hour, r.median_connected_h), xytext=(-14 if left else 14, 0),
                   textcoords="offset points", fontsize=7.5, va="center", ha="right" if left else "left", color=INK)
    a.set_yscale("log")
    a.set_ylim(0.25, 9)
    a.set_yticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
    a.minorticks_off()
    a.set_xlim(0, 24)
    a.set_xticks(range(0, 25, 3))
    a.set_xlabel("typical arrival hour")
    a.set_ylabel("median connection time (h, log scale)")
    a.set_title("Session segments (k-means, K = 5 by silhouette)")
    s = shift.reindex(prof["name"])
    y = np.arange(len(s))
    vals = (np.exp(s.did_log_change) - 1) * 100
    b.barh(y, vals, height=0.55, color=[BLUE if v < 0 else ORANGE for v in vals])
    for yy, v in zip(y, vals, strict=True):
        b.text(v + (1.5 if v >= 0 else -1.5), yy, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
               fontsize=8.5)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_yticks(y, s.index, fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("change in sessions/month, % (2017 vs 2016)")
    b.set_title("Which kinds of sessions the fee removed")
    b.set_xlim(min(vals.min() * 1.3, -10), max(vals.max() * 1.3, 10))
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "pa_segments.png")


def pa_congestion() -> None:
    by_hour = pd.read_csv(RESULTS / "pa_site_full_by_hour.csv", index_col=0)
    summary = pd.read_csv(RESULTS / "pa_site_congestion.csv", index_col=0)
    sites = summary.index.tolist()
    m = by_hour[sites].T * 100
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=max(10, float(m.to_numpy().max())))
    def site_label(name: str) -> str:
        pretty = " ".join(w if w in {"MPL"} else w.title() for w in name.split())
        return f"{pretty} ({int(summary.loc[name, 'ports'])} ports)"

    ax.set_yticks(range(len(sites)), [site_label(s) for s in sites])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("hour of day (weekdays, 2019)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("% of time every port is occupied", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Where drivers find no free port: share of time each site is full")
    save(fig, "pa_site_congestion.png")


# -------------------------------------------------------------------- fleet
def fleet_policy_comparison() -> None:
    pr = read("fleet_paired.csv")
    pr = pr[pr.label == "E1_main"]
    panels = [("service_rate", "Riders served", 100, "percentage points"),
              ("wait_mean_min", "Mean pickup wait", 1, "minutes"),
              ("energy_cost_usd_per_day", "Energy cost", None, "% of baseline"),
              ("peak_charging_mw", "Peak charging load", None, "% of baseline")]
    pols = ["queue_aware", "threshold_100", "orchestrated"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    for ax, (metric, title, scale, unit) in zip(axes, panels, strict=True):
        d = pr[pr.metric == metric].set_index("policy").reindex(pols)
        if scale is None:
            mean, lo, hi = (d[c] / d.baseline_mean * 100 for c in ("diff_mean", "ci95_low", "ci95_high"))
        else:
            mean, lo, hi = (d[c] * scale for c in ("diff_mean", "ci95_low", "ci95_high"))
        y = np.arange(len(pols))
        for yy, p, m, a, b in zip(y, pols, mean, lo, hi, strict=True):
            ax.hlines(yy, a, b, color=POLICY_COLOR[p], lw=2)
            ax.plot(m, yy, "o", color=POLICY_COLOR[p], ms=7, mec=SURFACE, mew=1.5)
            ax.annotate(f"{m:+.2f}" if abs(m) < 10 else f"{m:+.0f}", (m, yy), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_ylim(-0.6, len(pols) - 0.4)
    axes[0].set_yticks(range(len(pols)), [POLICY_LABEL[p] for p in pols])
    axes[0].invert_yaxis()
    fig.suptitle("Change vs the threshold 20%->80% baseline, test week, 10 paired seeds (95% CI)", x=0.01,
                 ha="left", fontsize=11.5, fontweight="semibold", y=1.06)
    save(fig, "fleet_policy_comparison.png")


def fleet_load_shift() -> None:
    hp = read("fleet_hourly_profiles.csv")
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.4]})
    base = hp[hp.policy == "threshold_80"].sort_values("hour")
    axes[0].plot(base.hour, base.requests_per_day, color=MUTED)
    axes[0].set_ylabel("requests / hour")
    axes[0].set_title("Ride demand (20% replica of Manhattan yellow-taxi trips)")
    axes[1].plot(base.hour, base.lbmp_usd_mwh, color=MUTED)
    axes[1].set_ylabel("$/MWh")
    axes[1].set_title("NYC day-ahead wholesale electricity price (NYISO zone J), test-week mean")
    for p in ("threshold_80", "orchestrated"):
        d = hp[hp.policy == p].sort_values("hour")
        axes[2].plot(d.hour, d.charging_mw, color=POLICY_COLOR[p], label=POLICY_LABEL[p])
        axes[2].plot(d.hour.iloc[-1], d.charging_mw.iloc[-1], "o", color=POLICY_COLOR[p], ms=6, mec=SURFACE)
    axes[2].set_ylabel("MW")
    axes[2].set_title("Fleet charging load by hour of day")
    axes[2].legend(loc="upper center", fontsize=8.5)
    axes[2].set_xticks(range(0, 24, 2))
    axes[2].set_xlabel("hour of day")
    for ax in axes:
        ax.set_ylim(0, None)
    fig.tight_layout()
    save(fig, "fleet_load_shift.png")


def _sweep(label_prefix: str, key: str) -> pd.DataFrame:
    runs = read("fleet_runs.csv")
    r = runs[runs.label.str.startswith(label_prefix)].copy()
    r[key] = r.label.str.rsplit("_", n=1).str[-1].astype(int)
    return r


def fleet_scarcity() -> None:
    r = _sweep("E2_chargers_", "chargers_n")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    for p in ("threshold_80", "queue_aware", "orchestrated"):
        g = r[r.policy == p].groupby("chargers_n")
        m = g.service_rate.mean() * 100
        a.plot(m.index, m.to_numpy(), "-o", color=POLICY_COLOR[p], ms=5, mec=SURFACE, label=POLICY_LABEL[p])
        c = g.queue_wait_mean_min.mean()
        b.plot(c.index, c.to_numpy(), "-o", color=POLICY_COLOR[p], ms=5, mec=SURFACE, label=POLICY_LABEL[p])
    a.set_xlabel("DC fast chargers for 550 vehicles")
    a.set_ylabel("riders served, %")
    a.set_title("Service level vs charging infrastructure")
    a.legend(fontsize=8.5, loc="lower right")
    b.set_xlabel("DC fast chargers for 550 vehicles")
    b.set_ylabel("minutes")
    b.set_title("Mean wait for a free charger")
    b.set_ylim(0, None)
    fig.tight_layout()
    save(fig, "fleet_charger_scarcity.png")


def fleet_size() -> None:
    r = _sweep("E3_fleet_", "fleet_n")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for p in ("threshold_80", "orchestrated"):
        m = r[r.policy == p].groupby("fleet_n").service_rate.mean() * 100
        ax.plot(m.index, m.to_numpy(), "-o", color=POLICY_COLOR[p], ms=5, mec=SURFACE, label=POLICY_LABEL[p])
    ax.axhline(98, color=MUTED, lw=1)
    ax.text(r.fleet_n.min(), 98.1, "98% service", color=INK2, fontsize=8.5, va="bottom")
    ax.set_xlabel("fleet size (vehicles, 20% demand replica)")
    ax.set_ylabel("riders served, %")
    ax.set_title("Fleet size needed for a service level")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "fleet_fleet_size.png")


def fleet_stress() -> None:
    runs = read("fleet_runs.csv")
    scen = [("E1_main", "Base week"), ("E4_cold_minus5C", "Cold snap (-5 C all week)"),
            ("E4_outage_biggest_depot", "Largest depot offline 16-22h"), ("E4_demand_plus20pct", "Demand +20%")]
    pols = ["threshold_100", "threshold_80", "queue_aware", "orchestrated"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    seeds = sorted(runs[runs.label == "E4_cold_minus5C"].seed.unique())
    runs = runs[runs.seed.isin(seeds)]
    for k, (label, _) in enumerate(scen):
        for j, p in enumerate(pols):
            v = runs[(runs.label == label) & (runs.policy == p)].service_rate.mean() * 100
            ax.plot(v, k + (j - 1.5) * 0.16, "o", color=POLICY_COLOR[p], ms=7, mec=SURFACE, mew=1.5,
                    label=POLICY_LABEL[p] if k == 0 else None)
    ax.set_yticks(range(len(scen)), [s for _, s in scen])
    ax.invert_yaxis()
    ax.set_xlabel(f"riders served, % (mean of seeds {seeds[0]}-{seeds[-1]}, identical demand per seed)")
    ax.set_title("Stress tests on the test week")
    ax.legend(fontsize=8, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.42))
    save(fig, "fleet_stress.png")


def fleet_siting_map() -> None:
    zones = load_zones()
    service = zones[zones.in_service_area]
    shapes = {s.location_id: s for s in load_zone_shapes() if s.location_id in set(service.zone_id)}
    train = load_trips("2025-04-01", "2025-04-22")
    drops = train.do_zone.value_counts()
    layouts = read("fleet_layouts.csv").set_index("layout")
    cent = service.set_index("zone_id")[["cx_km", "cy_km"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 7.2))
    logd = np.log10(drops.reindex(list(shapes), fill_value=1).clip(lower=1))
    vmin, vmax = logd.min(), logd.max()
    for ax, (key, title) in zip(axes, [("pmedian_6", "p-median optimum, 6 sites"),
                                       ("busiest_6", "Heuristic: 6 busiest zones")], strict=True):
        for zid, s in shapes.items():
            color = SEQ(0.08 + 0.84 * (logd[zid] - vmin) / (vmax - vmin))
            for ring in s.rings:
                ax.add_patch(Polygon(ring, closed=True, facecolor=color, edgecolor=SURFACE, linewidth=0.6))
        depots = ast.literal_eval(layouts.loc[key, "depots"])
        for zid, n in depots.items():
            x, y = cent.loc[zid]
            ax.plot(x, y, "o", ms=4 + n, color=ORANGE, mec=SURFACE, mew=2)
            ax.annotate(str(n), (x, y), ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        mins = layouts.loc[key, "mean_minutes_to_nearest_depot"]
        ax.set_title(f"{title}\nmean drive to a charger: {mins:.1f} min", fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
    fig.suptitle("Charging depot siting in Manhattan (zones shaded by drop-offs; dots sized by chargers)",
                 x=0.02, ha="left", fontsize=11.5, fontweight="semibold")
    save(fig, "fleet_siting_map.png")


def main() -> None:
    ensure_dirs()
    feats = load_features()
    pa_timeline(feats)
    pa_fee_event_study()
    pa_fee_effects()
    pa_prediction()
    pa_shift()
    pa_segments()
    pa_congestion()
    fleet_policy_comparison()
    fleet_load_shift()
    fleet_scarcity()
    fleet_size()
    fleet_stress()
    fleet_siting_map()


if __name__ == "__main__":
    main()
