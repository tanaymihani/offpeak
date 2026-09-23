import numpy as np
import pandas as pd
import pytest

from evfleet.data.tlc import polygon_area_centroid
from evfleet.forecast.demand import LevelAdjustedProfile, slot_of_week
from evfleet.sim.network import fit_network, time_bucket


def test_polygon_with_hole():
    # clockwise outer square (shapefile convention) with a counter-clockwise hole
    outer = np.array([[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]], float)
    hole = np.array([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]], float)
    area, cx, cy = polygon_area_centroid([outer, hole])
    assert area == pytest.approx(15.0)
    assert cx == pytest.approx((16 * 2 - 1 * 1.5) / 15)
    assert cy == pytest.approx(cx)


def test_network_recovers_known_structure():
    rng = np.random.default_rng(0)
    zones = pd.DataFrame({"zone_id": [1, 2, 3], "cx_km": [0.0, 1.0, 3.0], "cy_km": [0.0, 0.0, 0.0],
                          "area_km2": [1.0, 1.0, 1.0]})
    base = {(1, 2): 300, (2, 1): 300, (1, 1): 120, (2, 2): 120}
    n = 4000
    pu = rng.choice([1, 2], n)
    do = rng.choice([1, 2], n)
    ts = pd.Timestamp("2025-04-07") + pd.to_timedelta(rng.uniform(0, 5 * 86400, n), unit="s")  # weekdays
    rush = np.isin(ts.hour, [8, 17])
    dur = np.array([base[(a, b)] for a, b in zip(pu, do, strict=True)]) * np.where(rush, 1.5, 1.0)
    dur = dur * np.exp(rng.normal(0, 0.05, n))
    trips = pd.DataFrame({"pickup_ts": ts, "pu": pu, "do_zone": do, "duration_s": dur,
                          "distance_km": 1.0 + (pu != do)})
    net = fit_network(trips, zones, min_trips=5)
    b_rush, b_calm = time_bucket(pd.DatetimeIndex([pd.Timestamp("2025-04-07 08:30"), pd.Timestamp("2025-04-07 11:00")]))
    ratio = net.factor[b_rush] / net.factor[b_calm]
    assert ratio == pytest.approx(1.5, rel=0.05)
    k = net.index
    assert net.base_s[k[1], k[2]] / net.base_s[k[1], k[1]] == pytest.approx(2.5, rel=0.05)
    # zone 3 has no trips: its pairs come from the distance regression and are finite
    assert np.isfinite(net.base_s).all() and not net.observed_pair[k[3]].any()
    assert (net.tt[:, 0, :][:, net.order_from[0, 0]] == np.sort(net.tt[:, 0, :], axis=1)).all()


def test_level_adjusted_profile_tracks_a_level_shift():
    model = LevelAdjustedProfile(np.full(672, 10.0), alpha=0.3)
    for s in range(20):
        model.update(s, 15.0)  # demand runs 50 % above profile
    assert model.forecast(20, horizon=4) == pytest.approx(60.0, rel=0.01)


def test_slot_of_week():
    ts = pd.DatetimeIndex(["2025-04-21 00:00", "2025-04-21 00:14", "2025-04-27 23:59"])  # Monday .. Sunday
    assert slot_of_week(ts).tolist() == [0, 0, 671]
