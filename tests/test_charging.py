import numpy as np
import pytest

from evfleet.sim.charging import ChargeCurve


def test_constant_power_segment_matches_closed_form():
    c = ChargeCurve(capacity_kwh=40.0, charger_kw=150.0)
    # between 10% and 50% SoC the C-rate is flat at 2.5C = 100 kW
    assert c.duration_s(0.1, 0.5) == pytest.approx(0.4 * 40.0 / 100.0 * 3600.0, rel=1e-6)


def test_charger_power_cap_binds():
    c = ChargeCurve(capacity_kwh=100.0, charger_kw=50.0)  # vehicle could take 250 kW
    assert c.duration_s(0.1, 0.5) == pytest.approx(0.4 * 100.0 / 50.0 * 3600.0, rel=1e-6)


def test_soc_after_inverts_duration():
    c = ChargeCurve()
    for s0, s1 in [(0.05, 0.3), (0.2, 0.8), (0.6, 0.95), (0.1, 1.0)]:
        assert c.soc_after(s0, c.duration_s(s0, s1)) == pytest.approx(s1, abs=1e-6)


def test_taper_makes_the_last_20_percent_slow():
    c = ChargeCurve()
    assert c.duration_s(0.8, 1.0) > c.duration_s(0.2, 0.8)


def test_durations_are_additive_and_nonnegative():
    c = ChargeCurve()
    assert c.duration_s(0.2, 0.5) + c.duration_s(0.5, 0.9) == pytest.approx(c.duration_s(0.2, 0.9))
    assert c.duration_s(0.7, 0.3) == 0.0
    assert c.soc_after(0.99, 1e9) == pytest.approx(1.0)


def test_grid_energy_includes_losses():
    c = ChargeCurve(capacity_kwh=40.0, efficiency=0.9)
    assert c.grid_kwh(0.2, 0.8) == pytest.approx(24.0 / 0.9)
    assert np.all(np.diff(c._t) > 0)
