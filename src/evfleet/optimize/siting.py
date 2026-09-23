"""Charging-depot siting as a p-median problem, solved exactly as a MILP.

Choose ``p`` depot zones to minimise the demand-weighted travel time from where
vehicles become free (drop-off locations) to their assigned depot::

    min  sum_ij w_i c_ij x_ij
    s.t. sum_j x_ij = 1          for every demand zone i
         x_ij <= y_j             assign only to open depots
         sum_j y_j = p
         y_j in {0,1}, 0 <= x_ij <= 1

With ``y`` integral the assignment ``x`` is integral at the optimum, so only
``y`` needs integrality. Chargers are then split across the chosen sites in
proportion to the demand each serves (largest-remainder rounding with a
per-site minimum).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def p_median(cost: np.ndarray, weights: np.ndarray, p: int, time_limit: float = 60.0) -> tuple[list[int], float]:
    """Return (sorted open site indices, objective) for the p-median problem."""
    n_dem, n_site = cost.shape
    if not 1 <= p <= n_site:
        raise ValueError("p must be between 1 and the number of candidate sites")
    nx = n_dem * n_site
    # variable order: x_ij (row-major), then y_j
    c = np.concatenate([(weights[:, None] * cost).ravel(), np.zeros(n_site)])

    rows, cols, vals = [], [], []
    # sum_j x_ij = 1
    for i in range(n_dem):
        for j in range(n_site):
            rows.append(i)
            cols.append(i * n_site + j)
            vals.append(1.0)
    r = n_dem
    # x_ij - y_j <= 0
    for i in range(n_dem):
        for j in range(n_site):
            rows += [r, r]
            cols += [i * n_site + j, nx + j]
            vals += [1.0, -1.0]
            r += 1
    # sum_j y_j = p
    for j in range(n_site):
        rows.append(r)
        cols.append(nx + j)
        vals.append(1.0)
    r += 1
    a = coo_matrix((vals, (rows, cols)), shape=(r, nx + n_site)).tocsr()
    lb = np.concatenate([np.ones(n_dem), np.full(n_dem * n_site, -np.inf), [p]])
    ub = np.concatenate([np.ones(n_dem), np.zeros(n_dem * n_site), [p]])
    integrality = np.concatenate([np.zeros(nx), np.ones(n_site)])
    res = milp(c, constraints=LinearConstraint(a, lb, ub), integrality=integrality,
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if not res.success:
        raise RuntimeError(f"p-median MILP failed: {res.message}")
    y = res.x[nx:]
    sites = sorted(int(j) for j in np.flatnonzero(y > 0.5))
    return sites, float(assignment_cost(cost, weights, sites))


def assignment_cost(cost: np.ndarray, weights: np.ndarray, sites: list[int]) -> float:
    return float((weights * cost[:, sites].min(axis=1)).sum())


def brute_force_p_median(cost: np.ndarray, weights: np.ndarray, p: int) -> tuple[list[int], float]:
    """Exhaustive search (for testing on small instances)."""
    best, best_val = None, np.inf
    for combo in combinations(range(cost.shape[1]), p):
        val = assignment_cost(cost, weights, list(combo))
        if val < best_val:
            best, best_val = list(combo), val
    return best, best_val


def allocate_chargers(cost: np.ndarray, weights: np.ndarray, sites: list[int], total: int,
                      min_per_site: int = 2) -> list[int]:
    """Split ``total`` chargers across ``sites`` in proportion to the demand they serve."""
    if total < min_per_site * len(sites):
        raise ValueError("not enough chargers for the per-site minimum")
    nearest = np.asarray(sites)[cost[:, sites].argmin(axis=1)]
    served = np.array([weights[nearest == s].sum() for s in sites], dtype=float)
    share = served / served.sum() if served.sum() > 0 else np.full(len(sites), 1 / len(sites))
    spare = total - min_per_site * len(sites)
    raw = share * spare
    alloc = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - alloc))[: spare - alloc.sum()]:
        alloc[k] += 1
    return (alloc + min_per_site).tolist()
