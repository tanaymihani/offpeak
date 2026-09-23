import numpy as np
import pytest

from evfleet.optimize.siting import allocate_chargers, brute_force_p_median, p_median


@pytest.mark.parametrize("seed", range(6))
def test_milp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 10, (9, 2))
    cost = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    w = rng.uniform(1, 5, 9)
    p = 1 + seed % 4
    _, milp_val = p_median(cost, w, p)
    _, brute_val = brute_force_p_median(cost, w, p)
    assert milp_val == pytest.approx(brute_val, rel=1e-9)


def test_p_equal_n_opens_everything_at_zero_cost():
    cost = np.array([[0.0, 3.0], [3.0, 0.0]])
    sites, val = p_median(cost, np.ones(2), 2)
    assert sites == [0, 1] and val == 0.0


def test_allocation_sums_to_total_and_respects_minimum():
    rng = np.random.default_rng(0)
    cost = rng.uniform(0, 10, (20, 20))
    w = rng.uniform(0, 5, 20)
    alloc = allocate_chargers(cost, w, [2, 7, 11], total=23, min_per_site=3)
    assert sum(alloc) == 23 and min(alloc) >= 3


def test_allocation_rejects_impossible_minimum():
    with pytest.raises(ValueError):
        allocate_chargers(np.ones((3, 3)), np.ones(3), [0, 1], total=3, min_per_site=2)
