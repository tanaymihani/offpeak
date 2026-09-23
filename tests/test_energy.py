import pytest

from evfleet.sim.energy import EnergyModel


def test_hvac_is_zero_in_comfort_band_and_capped():
    e = EnergyModel()
    assert e.hvac_kw(20.0) == 0.0
    assert e.hvac_kw(10.0) == pytest.approx(0.07 * 8)
    assert e.hvac_kw(-100.0) == e.heat_max_kw
    assert e.hvac_kw(30.0) == pytest.approx(0.10 * 6)


def test_consumption_is_u_shaped_in_speed():
    e = EnergyModel()
    slow, mid, fast = e.per_km(8, 20), e.per_km(40, 20), e.per_km(110, 20)
    assert mid < slow and mid < fast


def test_cold_increases_energy_and_time_loads_scale_with_duration():
    e = EnergyModel()
    assert e.leg_kwh(5, 1200, -5) > e.leg_kwh(5, 1200, 20)
    # same distance, twice the time: the time-based loads double
    extra = e.leg_kwh(5, 2400, 20) - e.leg_kwh(5, 1200, 20)
    aero = 5 * e.b_kwh_per_km_kmh2 * ((5 / (2400 / 3600)) ** 2 - (5 / (1200 / 3600)) ** 2)
    assert extra == pytest.approx(e.aux_kw * 1200 / 3600 + aero)


def test_idle_drain():
    e = EnergyModel(idle_kw=0.25)
    assert e.idle_kwh(3600) == pytest.approx(0.25)
    assert e.idle_kwh(-5) == 0.0
