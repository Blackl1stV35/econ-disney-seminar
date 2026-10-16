"""Hand-built panels for the difference-in-differences, synthetic control and replication tests.

The study panel carries the structure of the dataset: nine units in two study
blocks, the same years and gaps, one treated unit per block, a known treatment
effect from the opening year on and, in the heterogeneous variant, differential
donor trends.  The factor panel has the factor structure used by the synthetic
control tests.

Outcomes are closed-form arithmetic of the unit, the year and the explicit
effect.  The only random ingredient is a noise array, drawn with a seed that the
caller passes, so every panel is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["YEARS", "OBSERVED", "TREATED", "OPENING", "DONORS", "StudyTruth", "study_panel", "factor_panel"]

#: Calendar years of the study window.  Every unit is written for all of them and masked to its years of observation.
YEARS = np.arange(1998, 2020)

#: Years of observation (first, last) of each unit, by study block.
OBSERVED = {
    "Hong Kong": {2: (1998, 2019), 3: (1999, 2019), 4: (1998, 2018), 7: (1998, 2018), 8: (1998, 2019), 9: (1998, 2019)},
    "Shanghai": {5: (2001, 2019), 1: (2005, 2019), 6: (2006, 2019)},
}
#: Treated unit of each block.
TREATED = {"Hong Kong": 2, "Shanghai": 5}
#: Opening year of each block (first treated period).
OPENING = {"Hong Kong": 2005, "Shanghai": 2016}
#: Donors of each block: level ``mu``, slope per year at a trend spread of one, and weight in the counterfactual.
DONORS = {
    "Hong Kong": {
        3: dict(mu=2.0, slope=-0.5, weight=0.0),
        4: dict(mu=4.6, slope=1.0, weight=0.5),
        7: dict(mu=3.4, slope=1.5, weight=0.3),
        8: dict(mu=1.4, slope=-0.8, weight=0.0),
        9: dict(mu=3.1, slope=0.2, weight=0.2),
    },
    "Shanghai": {
        1: dict(mu=2.2, slope=0.6, weight=0.6),
        6: dict(mu=1.6, slope=-0.6, weight=0.4),
    },
}


@dataclass
class StudyTruth:
    """Parameters of the construction: the known effects, the counterfactual weights and the trend spread."""

    tau: dict
    weights: dict
    trend_spread: float
    noise_seed: int


def _common_year_effect(case: str) -> np.ndarray:
    """Year effect shared by the units of a block: a sinusoid plus a drift from the opening year."""
    return 0.25 * np.sin(0.8 * (YEARS - 1998)) + 0.05 * (YEARS - OPENING[case])


def _ar1_rows(innovations: np.ndarray, rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) series along the columns, one row per series, with standard deviation ``sd``."""
    scale = sd * np.sqrt(1.0 - rho**2)
    u = innovations * scale
    x = np.empty_like(u)
    x[:, 0] = u[:, 0] / np.sqrt(1.0 - rho**2)
    for i in range(1, u.shape[1]):
        x[:, i] = rho * x[:, i - 1] + u[:, i]
    return x


def study_panel(trend_spread: float = 0.0, tau_hk: float = 5.0, tau_sh: float = 1.0, noise: float = 0.12, noise_seed: int = 11):
    """Panel with the structure of the dataset: 177 rows, nine units, two study blocks.

    Donor outcomes are ``y_jt = mu_j + slope_j * trend_spread * (t - t0) + f_t + e_jt``, where ``f_t`` is the
    year effect shared by the block and ``e_jt`` is an AR(1) error (coefficient 0.4) built from a seeded noise
    array.  The untreated outcome of the treated unit is the convex combination of the donors with the weights
    of :data:`DONORS` plus a small noise term, and the treated outcome adds the effect from the opening year on.
    With ``trend_spread = 0`` the trends are parallel, so the two-way estimator with unit and year effects is
    unbiased; with a positive spread the treated trend differs from the donor average while the convex combination still
    reproduces the treated path.

    Returns
    -------
    df : DataFrame
        Columns ``unit_id, year, receipts_pct_gdp, treated, post, treated_post, study_case, rel_year``.
    truth : StudyTruth
        The known effects, the counterfactual weights and the parameters of the construction.
    """
    rng = np.random.default_rng(noise_seed)
    eps = _ar1_rows(rng.normal(size=(10, YEARS.size)), 0.4, noise)  # one row per unit id
    eta = rng.normal(0.0, 0.04, size=(2, YEARS.size))  # one row per block
    tau_of = {"Hong Kong": tau_hk, "Shanghai": tau_sh}
    frames = []
    for b, case in enumerate(("Hong Kong", "Shanghai")):
        t0 = OPENING[case]
        f = _common_year_effect(case)
        paths = {}
        for j, spec in DONORS[case].items():
            paths[j] = spec["mu"] + spec["slope"] * trend_spread * (YEARS - t0) + f + eps[j]
        counterfactual = sum(spec["weight"] * paths[j] for j, spec in DONORS[case].items()) + eta[b]
        tr = TREATED[case]
        paths[tr] = counterfactual + tau_of[case] * (YEARS >= t0)
        for u, (first, last) in OBSERVED[case].items():
            m = (YEARS >= first) & (YEARS <= last)
            frames.append(
                pd.DataFrame(
                    {
                        "unit_id": u,
                        "year": YEARS[m],
                        "receipts_pct_gdp": paths[u][m],
                        "treated": int(u == tr),
                        "post": (YEARS[m] >= t0).astype(int),
                        "study_case": case,
                        "rel_year": YEARS[m] - t0,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    df["treated_post"] = df["treated"] * df["post"]
    df = df[["unit_id", "year", "receipts_pct_gdp", "treated", "post", "treated_post", "study_case", "rel_year"]]
    df = df.sort_values(["unit_id", "year"]).reset_index(drop=True)
    truth = StudyTruth(
        tau=dict(tau_of),
        weights={case: {j: spec["weight"] for j, spec in DONORS[case].items()} for case in DONORS},
        trend_spread=trend_spread,
        noise_seed=noise_seed,
    )
    return df, truth


def factor_panel(
    n_donors: int = 96,
    T: int = 22,
    T0: int = 14,
    n_active: int = 4,
    tau: float = 2.0,
    noise: float = 0.05,
    noise_seed: int = 7,
) -> dict:
    """Panel with an interactive factor structure for the synthetic control tests.

    The donor outcomes are ``mu_j + (F L')_tj + e_tj`` with three factors ``F`` and loadings ``L`` written in
    closed form, and ``e`` a seeded noise array.  The treated unit has the loadings and the level of a convex
    combination of ``n_active`` donors, placed at equally spaced positions of the donor list with decreasing
    weights, so its untreated outcome lies in the convex hull of the donors up to noise.  The treated outcome
    adds ``tau`` from period ``T0`` on.

    Returns
    -------
    dict
        ``Y`` (``T x (1 + n_donors)``, the treated unit in column 0), ``pre`` (boolean, first ``T0`` periods),
        ``weights`` (the true weights over donors), ``tau`` and ``T0``.
    """
    rng = np.random.default_rng(noise_seed)
    t = np.arange(T)[:, None]
    k = np.arange(3)[None, :]
    j = np.arange(n_donors)[:, None]
    F = np.cumsum(0.4 * np.cos(0.9 * t + 1.3 * k), axis=0) + 0.2 * np.sin(0.7 * t + 2.1 * k)
    L = np.cos(1.1 * j + 0.8 * k) + 0.5 * np.sin(0.45 * j * (k + 1))
    mu = 3.0 + 0.8 * np.sin(0.45 * np.arange(n_donors) + 0.3)
    Y0 = mu[None, :] + F @ L.T + rng.normal(0.0, noise, size=(T, n_donors))
    active = np.linspace(0, n_donors - 1, n_active).round().astype(int)
    weights = np.arange(n_active, 0, -1, dtype=float)
    weights /= weights.sum()
    w = np.zeros(n_donors)
    w[active] = weights
    y1 = Y0 @ w + rng.normal(0.0, noise / 2, size=T)
    pre = np.arange(T) < T0
    y1 = y1 + tau * (~pre)
    return {"Y": np.column_stack([y1, Y0]), "pre": pre, "weights": w, "tau": tau, "T0": T0}
