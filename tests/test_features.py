"""The plug-in-time features must not look into the future."""

import csv

import pandas as pd
import pytest

from evfleet.data.palo_alto import run_pipeline

HEADER = ["Station Name", "MAC Address", "Org Name", "Start Date", "Start Time Zone", "End Date", "End Time Zone",
          "Transaction Date (Pacific Time)", "Total Duration (hh:mm:ss)", "Charging Time (hh:mm:ss)", "Energy (kWh)",
          "GHG Savings (kg)", "Gasoline Savings (gallons)", "Port Type", "Port Number", "Plug Type", "EVSE ID",
          "Address 1", "City", "State/Province", "Postal Code", "Country", "Latitude", "Longitude", "Currency", "Fee",
          "Ended By", "Plug In Event Id", "Driver Postal Code", "User ID", "County", "System S/N", "Model Number"]


def row(start, dur, chg, kwh, user, station="PALO ALTO CA / BRYANT #1", tz="PDT", fee="0"):
    r = dict.fromkeys(HEADER, "")
    r.update({"Station Name": station, "Start Date": start, "Start Time Zone": tz, "Total Duration (hh:mm:ss)": dur,
              "Charging Time (hh:mm:ss)": chg, "Energy (kWh)": kwh, "Port Type": "Level 2", "Port Number": "1",
              "Plug Type": "J1772", "Fee": fee, "User ID": user, "Driver Postal Code": "94301"})
    return r


def features(tmp_path, rows):
    path = tmp_path / "raw.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    con = run_pipeline(path)
    feats = con.execute("SELECT * FROM session_features ORDER BY start_ts").df()
    sessions = con.execute("SELECT * FROM pa_sessions ORDER BY raw_row").df()
    return feats, sessions


BASE = [
    row("6/3/2019 8:00", "2:00:00", "1:30:00", "6.0", "u1"),
    row("6/4/2019 8:10", "3:00:00", "2:00:00", "9.0", "u1"),
    row("6/5/2019 8:05", "1:00:00", "1:00:00", "3.0", "u1"),
    row("6/5/2019 9:00", "4:00:00", "3:00:00", "12.0", "u2", station="PALO ALTO CA / BRYANT #2"),
]


def test_history_uses_only_earlier_sessions(tmp_path):
    f, _ = features(tmp_path, BASE)
    u1 = f[f.user_id == "u1"].reset_index(drop=True)
    assert u1.u_prior_n.tolist() == [0, 1, 2]
    assert pd.isna(u1.u_prior_median_dwell[0])
    assert u1.u_prior_median_dwell[1] == pytest.approx(2.0)
    assert u1.u_prior_mean_energy[2] == pytest.approx(7.5)
    assert u1.u_last_dwell[2] == pytest.approx(3.0)
    # u2 arrives while u1's third session (08:05-09:05) is still plugged in at BRYANT
    assert f.loc[f.user_id == "u2", "site_active_at_start"].item() == 1


def test_changing_the_future_does_not_change_the_past(tmp_path):
    f1, _ = features(tmp_path, BASE)
    altered = BASE[:2] + [row("6/5/2019 8:05", "9:00:00", "1:00:00", "30.0", "u1")] + BASE[3:]
    f2, _ = features(tmp_path, altered)
    cols = [c for c in f1.columns if c.startswith("u_")]
    early = f1.start_ts < pd.Timestamp("2019-06-05")
    pd.testing.assert_frame_equal(f1.loc[early, cols].reset_index(drop=True), f2.loc[early, cols].reset_index(drop=True))


def test_overlapping_history_is_masked(tmp_path):
    rows = [row("6/3/2019 8:00", "5:00:00", "2:00:00", "6.0", "u1"),
            row("6/3/2019 9:00", "1:00:00", "1:00:00", "3.0", "u1", station="PALO ALTO CA / HIGH #1")]
    f, _ = features(tmp_path, rows)
    second = f.iloc[1]
    assert bool(second.history_overlap) and pd.isna(second.u_prior_median_dwell)


def test_quality_rules(tmp_path):
    rows = BASE + [row("6/6/2019 8:00", "0:30:00", "0:40:00", "1.0", "u3"),   # charging > connected
                   row("6/6/2019 8:00", "1:00:00", "0:30:00", "1.0", "u4", tz="UTC"),
                   BASE[0]]  # exact duplicate
    _, s = features(tmp_path, rows)
    reasons = s.drop_reason.value_counts().to_dict()
    assert reasons == {"charging_exceeds_connected": 1, "non_pacific_timezone": 1, "duplicate_row": 1}
