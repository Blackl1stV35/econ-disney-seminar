"""Synthetic test panels for validating the estimators.

The panels are test data with a known treatment effect.  They are generated from
a documented data-generating process and carry the *structure* of the panel of
the original study: nine units, the same study blocks, the same years with gaps, a known
treatment effect, differential trends and very few clusters.

Structure of the panel file
---------------------------
Hong Kong block (units 2, 3, 4, 7, 8, 9; ``N = 129``)
    years 1998 to 2019 for units 2, 8 and 9; Ireland (3) starts in 1999;
    New Zealand (4) and Singapore (7) end in 2018.
Shanghai block (units 1, 5, 6; ``N = 48``)
    Shanghai (5) 2001 to 2019; Beijing (1) from 2005 and Shenzhen (6) from 2006,
    both to 2019, so that the window 2008 to 2019 is balanced (``N = 36``).

Data-generating process
-----------------------
Donor outcomes: ``y_jt = mu_j + beta_j (t - t0) + f_t + e_jt`` with a common
AR(1) factor ``f_t`` and AR(1) idiosyncratic errors.  The treated unit's
untreated outcome is a fixed convex combination of the donors plus white noise,
``y_1t(0) = sum_j w_j y_jt + eta_t``, and the treated outcome adds a constant
effect ``tau`` from ``t0`` on.  The slope heterogeneity ``trend_spread`` sets how
far the donor trends differ: with 0 the trends are parallel, so a two-way
fixed-effects DiD is unbiased; with a positive value the treated trend differs
from the donor average (a pre-trend) while the convex combination still
reproduces it, so the synthetic control remains valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "UNIT_NAMES",
    "STRUCTURE",
    "SimTruth",
    "simulate_panel",
    "simulate_factor_panel",
]

UNIT_NAMES = {
    1: "Beijing",
    2: "Hong Kong",
    3: "Ireland",
    4: "New Zealand",
    5: "Shanghai",
    6: "Shenzhen",
    7: "Singapore",
    8: "South Korea",
    9: "Switzerland",
}

#: first and last year of observation per unit in the original panel structure
STRUCTURE = {
    "Hong Kong": {2: (1998, 2019), 3: (1999, 2019), 4: (1998, 2018), 7: (1998, 2018), 8: (1998, 2019), 9: (1998, 2019)},
    "Shanghai": {5: (2001, 2019), 1: (2005, 2019), 6: (2006, 2019)},
}
TREATED_UNIT = {"Hong Kong": 2, "Shanghai": 5}
TREATMENT_YEAR = {"Hong Kong": 2005, "Shanghai": 2016}


@dataclass
class SimTruth:
    """Parameters of the data-generating process (the known truth)."""

    tau: dict
    weights: dict
    trend_spread: float
    seed: int
    note: str = "SIMULATED: parameters of the data-generating process, not estimates from the panel file"


def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    e = rng.normal(0.0, sd, size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def simulate_panel(
    seed: int = 20260925,
    tau_hk: float = 5.0,
    tau_sh: float = 1.0,
    trend_spread: float = 0.0,
    noise: float = 0.12,
) -> tuple[pd.DataFrame, SimTruth]:
    """Simulated analogue of the panel in ``data/raw/disney_did_panel.csv``.

    Returns a data frame with columns ``unit_id, year, receipts_pct_gdp, treated,
    post, treated_post, study_case, rel_year`` (177 rows) and the known truth.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1998, 2020)
    T = years.size
    blocks = {
        "Hong Kong": dict(
            donors=[3, 4, 7, 8, 9],
            mu={3: 2.0, 4: 4.6, 7: 3.4, 8: 1.4, 9: 3.1},
            slope_dir={3: -0.5, 4: 1.0, 7: 1.5, 8: -0.8, 9: 0.2},
            w={3: 0.0, 4: 0.5, 7: 0.3, 8: 0.0, 9: 0.2},
            tau=tau_hk,
        ),
        "Shanghai": dict(
            donors=[1, 6],
            mu={1: 2.2, 6: 1.6},
            slope_dir={1: 0.6, 6: -0.6},
            w={1: 0.6, 6: 0.4},
            tau=tau_sh,
        ),
    }
    frames = []
    for case, spec in blocks.items():
        t0 = TREATMENT_YEAR[case]
        f = _ar1(rng, T, 0.6, 0.20) + 0.05 * (years - t0)
        paths = {}
        for j in spec["donors"]:
            slope = spec["slope_dir"][j] * trend_spread
            paths[j] = spec["mu"][j] + slope * (years - t0) + f + _ar1(rng, T, 0.4, noise)
        counter = sum(spec["w"][j] * paths[j] for j in spec["donors"]) + rng.normal(0.0, 0.04, size=T)
        tr = TREATED_UNIT[case]
        paths[tr] = counter + spec["tau"] * (years >= t0)
        for u, (a, b) in STRUCTURE[case].items():
            m = (years >= a) & (years <= b)
            frames.append(
                pd.DataFrame(
                    {
                        "unit_id": u,
                        "year": years[m],
                        "receipts_pct_gdp": paths[u][m],
                        "treated": int(u == tr),
                        "post": (years[m] >= t0).astype(int),
                        "study_case": case,
                        "rel_year": years[m] - t0,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    df["treated_post"] = df["treated"] * df["post"]
    df = df[["unit_id", "year", "receipts_pct_gdp", "treated", "post", "treated_post", "study_case", "rel_year"]]
    df = df.sort_values(["unit_id", "year"]).reset_index(drop=True)
    truth = SimTruth(
        tau={"Hong Kong": tau_hk, "Shanghai": tau_sh},
        weights={case: blocks[case]["w"] for case in blocks},
        trend_spread=trend_spread,
        seed=seed,
    )
    return df, truth


def simulate_factor_panel(
    n_donors: int = 96,
    T: int = 22,
    T0: int = 14,
    n_factors: int = 3,
    n_active: int = 4,
    tau: float = 2.0,
    noise: float = 0.05,
    seed: int = 7,
) -> dict:
    """Interactive fixed-effects panel for exercising the synthetic control at scale.

    The treated unit's loadings are a convex combination of ``n_active`` donor
    loadings, so the untreated outcome lies in the convex hull of the donors up
    to ``noise``.  Returns a dictionary with ``Y`` (``T x (1 + n_donors)``, the
    treated unit in column 0), ``pre`` (boolean), the true ``weights`` over
    donors and the effect ``tau``.
    """
    rng = np.random.default_rng(seed)
    F = np.cumsum(rng.normal(0, 0.4, size=(T, n_factors)), axis=0) + rng.normal(0, 0.2, size=(T, n_factors))
    L = rng.normal(0, 1.0, size=(n_donors, n_factors))
    mu = rng.normal(3.0, 1.0, size=n_donors)
    Y0 = mu[None, :] + F @ L.T + rng.normal(0, noise, size=(T, n_donors))
    active = rng.choice(n_donors, size=n_active, replace=False)
    w = np.zeros(n_donors)
    w[active] = rng.dirichlet(np.ones(n_active) * 2.0)
    y1 = Y0 @ w + rng.normal(0, noise / 2, size=T)
    pre = np.arange(T) < T0
    y1 = y1 + tau * (~pre)
    return {"Y": np.column_stack([y1, Y0]), "pre": pre, "weights": w, "tau": tau, "T0": T0}
