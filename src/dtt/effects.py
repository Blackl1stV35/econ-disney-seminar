"""Synthetic-control effect of each case, with placebo-in-space inference.

For one case or treatment episode the outcome of the host economy (receipts as a
percentage of GDP) is compared with a synthetic control built from the donor
pool. The weights are fitted on every pre-opening year with an observed outcome.
The effect is the mean gap between the actual and the synthetic outcome over the
post-opening window, in percentage points of GDP (``att_pp``) and relative to the
mean synthetic level (``att_rel_pct``). The window starts in the opening year and
ends at the earliest of ``opening_year + post_horizon - 1``, the last year of the
estimation window and the year before the next episode of the same economy (see
:func:`dtt.cases.post_window_end`). All statistics, including the placebo
statistics, use this window.

Inference follows the placebo-in-space design of Abadie, Diamond and Hainmueller
(2010): every donor is treated in turn, with the remaining donors as its pool,
and the effect of the case is ranked against the placebo effects. The estimators
are those of :mod:`dtt.scm`.

Comparable placebos
-------------------
The gap of a placebo whose pre-opening fit is much worse than that of a typical
placebo reflects its misfit and not the chance of an effect of the size found for
the case. A placebo is therefore comparable when its pre-opening RMSPE is at most
``placebo_fit_factor`` times the median placebo pre-opening RMSPE (default
:data:`PLACEBO_FIT_FACTOR`, the multiple that the case itself must meet for
``pre_fit_ok``; the median is taken as at least :data:`EXACT_FIT_TOLERANCE`), and
the ``min(J, 5)`` best-fitting placebos of ``J`` donors are always comparable.
The placebo standard deviations (``placebo_sd_pp`` and ``placebo_sd_rel_pct``) and
the rank p-value of the mean gap (``p_gap_rank``) use the comparable placebos
only. The rank p-value of the RMSPE ratio (``p_ratio_rank``) uses every placebo
with a defined ratio, because the ratio does not depend on the scale of the
misfit. ``placebo_fit_factor=None`` keeps all placebos. The effects table reports
the number of placebos (``n_placebos``) and of comparable placebos
(``n_placebos_comparable``), and the placebo table carries the boolean column
``comparable``.

Exact fits
----------
The synthetic control fits the pre-opening years exactly when the largest
absolute gap in a pre-opening year is at most ``fit_tolerance``, which is the
larger of :data:`EXACT_FIT_TOLERANCE` and ``exact_fit_relative_tolerance`` times
the median placebo pre-opening RMSPE. With fewer pre-opening years than donors the
weights of an exact fit are not unique. The table then reports the smallest and
the largest effect that any admissible weight vector gives (``att_pp_min`` and
``att_pp_max``; the raw bounds of the two linear programs are ``att_pp_lp_min``
and ``att_pp_lp_max``) and does not accept the fit as a good one (``pre_fit_ok``).
This range is a range over weights that reproduce the pre-opening years within the
tolerance. It is not a confidence interval and not a sampling interval, and it
contains no sampling uncertainty. For an exact fit the ratio of the post-opening
to the pre-opening RMSPE is undefined (``rmspe_ratio`` and ``p_ratio_rank`` are
NaN), and a placebo whose own pre-opening RMSPE is at most ``fit_tolerance`` has
no ratio and is left out of the ratio rank.

Non-unique placebo fits
-----------------------
When the fit of a placebo is exact and there are fewer pre-opening years than other
donors, its minimiser is not unique. The weights are then those of the rule
described in :mod:`dtt.scm` (the vertex of the set of minimisers at which the
active-set path of :func:`dtt.scm.qp_simplex` stops, not the minimiser of smallest
norm), and the effect of such a placebo depends on that rule. The same solver
supplies the weights of every method (the direct weights are the starting point
of ``method="ridge"`` and the inner problem of ``method="nested"``), so the rule
applies to all of them. The table reports the number of these placebos as
``n_placebos_exact``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from . import scm
from .cases import POST_HORIZON, check_feasibility, donor_pool, outcome_matrix, post_window_end, resolve_case, resolve_next_start

__all__ = [
    "METHODS",
    "NESTED_STARTS",
    "GUARD_INTERVAL",
    "EXACT_FIT_TOLERANCE",
    "EXACT_FIT_RELATIVE_TOLERANCE",
    "PLACEBO_FIT_FACTOR",
    "MIN_COMPARABLE_PLACEBOS",
    "CaseEffect",
    "rank_p_value",
    "estimate_case",
    "estimate_all",
    "write_effects",
]

#: Weight methods of :func:`estimate_case`.
METHODS: tuple[str, ...] = ("direct", "ridge", "nested")

#: Number of starting points of the nested optimisation, for the case and its placebos.
NESTED_STARTS = 4

#: Number of donor fits between two calls of the guard of :func:`estimate_case`.
GUARD_INTERVAL = 10

#: Smallest tolerance, in units of the outcome, on the absolute gap in every pre-opening year of a fit that counts as exact.
EXACT_FIT_TOLERANCE = 1.0e-6

#: Default share of the median placebo pre-opening RMSPE that the absolute gap of an exact fit may reach in a pre-opening year.
EXACT_FIT_RELATIVE_TOLERANCE = 0.05

#: Default multiple of the median placebo pre-opening RMSPE up to which a placebo is comparable.
PLACEBO_FIT_FACTOR = 2.0

#: Number of best-fitting placebos that are always comparable (all placebos when there are fewer).
MIN_COMPARABLE_PLACEBOS = 5

#: Relative difference below which two placebo pre-opening RMSPE values count as tied.
_TIE_TOLERANCE = 1.0e-9

_EFFECT_COLUMNS: tuple[str, ...] = (
    "case_id",
    "iso3",
    "economy_group",
    "opening_year",
    "n_pre",
    "n_post",
    "window_end",
    "post_horizon_used",
    "n_donors",
    "method",
    "att_pp",
    "att_rel_pct",
    "att_pp_min",
    "att_pp_max",
    "att_pp_lp_min",
    "att_pp_lp_max",
    "rmspe_pre",
    "rmspe_post",
    "rmspe_ratio",
    "p_gap_rank",
    "p_ratio_rank",
    "placebo_sd_pp",
    "placebo_sd_rel_pct",
    "n_placebos",
    "n_placebos_comparable",
    "n_placebos_exact",
    "fit_exact",
    "fit_tolerance",
    "pre_fit_ok",
    "did_att_pp",
)
_PLACEBO_COLUMNS: tuple[str, ...] = (
    "case_id",
    "donor_iso3",
    "placebo_att_pp",
    "placebo_att_rel_pct",
    "placebo_rmspe_pre",
    "comparable",
)
_FEASIBILITY_KEYS = frozenset({"min_pre", "min_post", "min_donors"})
_SHARED_KEYS = frozenset({"outcome", "first_year", "last_year", "buffer", "post_horizon"})
_ESTIMATE_KEYS = frozenset({"method", "ridge", "seed", "n_starts"})
_MIN_PRE_FIT = 2


@dataclass(eq=False)
class CaseEffect:
    """Synthetic-control result of one case.

    Attributes
    ----------
    case_id, iso3 : str
        Identifier and economy of the case.
    opening_year : int
        Year of the opening.
    n_pre, n_post : int
        Years with an observed outcome before the opening and in the
        post-opening window.
    window_end : int
        Last year of the post-opening window.
    post_horizon_used : int
        Length of the post-opening window in years, ``window_end - opening_year + 1``.
    n_donors : int
        Number of donors.
    years, actual, synthetic, gap : numpy.ndarray
        Year, observed outcome, synthetic outcome and their difference for every
        year from the first year of the estimation window to ``window_end``;
        ``actual`` and ``gap`` are NaN in years without an observed outcome.
    weights : pandas.Series
        Donor weights, indexed by donor economy code.
    att_pp : float
        Mean gap from the opening year on, in percentage points of GDP.
    att_rel_pct : float
        ``att_pp`` divided by the mean synthetic outcome from the opening year on,
        times 100; NaN when that mean is not positive.
    rmspe_pre, rmspe_post, rmspe_ratio : float
        Root mean squared gap before and from the opening, and their ratio. The
        ratio is NaN for an exact fit (``fit_exact``), because the pre-opening
        RMSPE is then rounding error, and NaN when both RMSPE values are zero.
    p_gap_rank, p_ratio_rank : float
        Placebo-in-space rank p-values of the absolute mean gap and of the
        post-to-pre RMSPE ratio (see :func:`rank_p_value`). ``p_gap_rank`` ranks
        the case among the comparable placebos (``placebo_comparable``).
        ``p_ratio_rank`` ranks the case among all placebos that have a ratio, that
        is all placebos whose pre-opening RMSPE exceeds ``fit_tolerance``; it is
        NaN when ``rmspe_ratio`` is NaN.
    placebo_att_pp, placebo_att_rel_pct, placebo_rmspe_pre : numpy.ndarray
        Effect in percentage points, relative effect in percent and pre-opening
        RMSPE of the placebo of every donor, in the order of ``weights``.
    did_att_pp : float
        Mean change of the case from before to after the opening minus the mean
        change of the equally weighted donors, with the post-opening window as
        the after period.
    method : str
        Weight method used.
    outcome : str
        Name of the outcome column.
    placebo_gaps : numpy.ndarray
        Gap of every donor placebo, shape (years, donors), NaN in years without
        an observed outcome of the case.
    fit_exact : bool
        Whether the gap is at most ``fit_tolerance`` in absolute value in every
        pre-opening year.
    fit_tolerance : float
        The larger of :data:`EXACT_FIT_TOLERANCE` and
        ``exact_fit_relative_tolerance`` times the median placebo pre-opening
        RMSPE.
    exact_fit_relative_tolerance : float
        The multiple of the median placebo pre-opening RMSPE in ``fit_tolerance``.
    att_pp_min, att_pp_max : float
        Smallest and largest mean post-opening gap over all weight vectors that
        are non-negative, sum to one and keep the gap within ``fit_tolerance`` in
        absolute value in every pre-opening year. This is a range over weights
        that reproduce the pre-opening years; it is not a confidence interval and
        not a sampling interval, and it contains no sampling uncertainty. When the
        weights of the case are non-negative the interval is widened if necessary
        to contain ``att_pp``. Both equal ``att_pp`` when ``fit_exact`` is false;
        they are NaN when the fit is exact and the linear programs have no
        optimal solution.
    att_pp_lp_min, att_pp_lp_max : float
        The bounds of the two linear programs before any widening; NaN when
        ``fit_exact`` is false or a program has no optimal solution.
    pre_fit_ok : bool
        True when ``rmspe_pre`` is at most ``pre_fit_factor`` times the median
        placebo pre-opening RMSPE and the fit is not exact with fewer
        pre-opening years than donors.
    pre_fit_factor : float
        The factor of the rule for ``pre_fit_ok``.
    placebo_comparable : numpy.ndarray
        Boolean array in the order of ``weights``: whether the placebo of the
        donor is comparable, that is, whether its pre-opening RMSPE is at most
        ``placebo_fit_factor`` times the median placebo pre-opening RMSPE (the
        median taken as at least :data:`EXACT_FIT_TOLERANCE`) or it is among the
        ``min(J, 5)`` best-fitting placebos. All entries are true when
        ``placebo_fit_factor`` is None.
    placebo_fit_factor : float or None
        The multiple of the median placebo RMSPE that defines comparability;
        None keeps all placebos.
    placebo_sd_pp, placebo_sd_rel_pct : float
        Sample standard deviation of the finite effects, in percentage points and
        in percent of the synthetic level, over the comparable placebos; NaN with
        fewer than two finite values.
    n_placebos, n_placebos_comparable, n_placebos_exact : int
        Number of placebos, of comparable placebos, and of placebos whose
        pre-opening RMSPE is at most ``fit_tolerance`` (properties).
    """

    case_id: str
    iso3: str
    opening_year: int
    n_pre: int
    n_post: int
    window_end: int
    post_horizon_used: int
    n_donors: int
    years: np.ndarray = field(repr=False)
    actual: np.ndarray = field(repr=False)
    synthetic: np.ndarray = field(repr=False)
    gap: np.ndarray = field(repr=False)
    weights: pd.Series = field(repr=False)
    att_pp: float
    att_rel_pct: float
    rmspe_pre: float
    rmspe_post: float
    rmspe_ratio: float
    p_gap_rank: float
    p_ratio_rank: float
    placebo_att_pp: np.ndarray = field(repr=False)
    placebo_att_rel_pct: np.ndarray = field(repr=False)
    placebo_rmspe_pre: np.ndarray = field(repr=False)
    did_att_pp: float
    method: str = "direct"
    outcome: str = "receipts_pct_gdp"
    placebo_gaps: np.ndarray | None = field(default=None, repr=False)
    fit_exact: bool = False
    att_pp_min: float = float("nan")
    att_pp_max: float = float("nan")
    pre_fit_ok: bool = True
    pre_fit_factor: float = 2.0
    placebo_comparable: np.ndarray | None = field(default=None, repr=False)
    placebo_fit_factor: float | None = PLACEBO_FIT_FACTOR
    placebo_sd_pp: float = float("nan")
    placebo_sd_rel_pct: float = float("nan")
    fit_tolerance: float = EXACT_FIT_TOLERANCE
    exact_fit_relative_tolerance: float = EXACT_FIT_RELATIVE_TOLERANCE
    att_pp_lp_min: float = float("nan")
    att_pp_lp_max: float = float("nan")

    @property
    def n_placebos(self) -> int:
        """Number of placebos, one per donor."""
        return int(np.size(self.placebo_att_pp))

    @property
    def n_placebos_comparable(self) -> int:
        """Number of comparable placebos; every placebo when ``placebo_comparable`` is not set."""
        if self.placebo_comparable is None:
            return self.n_placebos
        return int(np.count_nonzero(self.placebo_comparable))

    @property
    def n_placebos_exact(self) -> int:
        """Number of placebos whose pre-opening RMSPE is at most ``fit_tolerance``."""
        return int(np.count_nonzero(np.asarray(self.placebo_rmspe_pre, dtype=float) <= self.fit_tolerance))


def _relative(effect: float, level: float) -> float:
    """Effect as a percentage of a level; NaN when the level is not finite or not positive."""
    if not np.isfinite(level) or level <= 0.0:
        return float("nan")
    return 100.0 * effect / level


def _rmspe(gap: np.ndarray) -> float:
    """Root mean squared value of a gap vector."""
    return float(np.sqrt(np.mean(gap**2)))


def _ratio(numerator: float, denominator: float) -> float:
    """Ratio of two root mean squared gaps: NaN for 0/0, infinite for a positive numerator over zero."""
    if np.isnan(numerator) or np.isnan(denominator):
        return float("nan")
    if denominator > 0.0:
        return numerator / denominator
    return float("nan") if numerator == 0.0 else float("inf")


def _sample_sd(values: np.ndarray) -> float:
    """Sample standard deviation of the finite entries; NaN with fewer than two."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")


def rank_p_value(statistic: float, placebo_statistics: Sequence[float] | np.ndarray) -> float:
    """Placebo-in-space rank p-value of a statistic.

    The p-value is ``(1 + m) / (1 + n)``, where ``n`` is the number of placebo
    statistics that are not NaN and ``m`` the number of them that are at least as
    large as ``statistic``. A placebo with a statistic equal to ``statistic``
    therefore counts as at least as extreme, and placebos with a NaN statistic
    are left out. The p-value lies in ``(0, 1]``. A statistic that is undefined
    for a unit, such as the ratio of the post-opening to the pre-opening RMSPE of
    a unit with an exact pre-opening fit, is passed as NaN and takes no part in the
    ranking.

    Parameters
    ----------
    statistic : float
        Statistic of the case; large values are extreme.
    placebo_statistics : array_like of float
        Statistics of the placebos; NaN entries are left out.

    Returns
    -------
    float
        The p-value, NaN when ``statistic`` is NaN.
    """
    value = float(statistic)
    if np.isnan(value):
        return float("nan")
    placebo = np.asarray(placebo_statistics, dtype=float).ravel()
    placebo = placebo[~np.isnan(placebo)]
    return float(1 + np.count_nonzero(placebo >= value)) / float(1 + placebo.size)


def _identification_range(
    y1: np.ndarray,
    Y0: np.ndarray,
    pre: np.ndarray,
    tolerance: float = EXACT_FIT_TOLERANCE,
) -> tuple[float, float]:
    """Smallest and largest mean post-opening gap over the weights that fit the pre-opening years.

    Two linear programs run over the non-negative weight vectors that sum to one
    and keep the absolute pre-opening gap within ``tolerance`` in every year. The
    bounds are those of the programs, with no widening; both are NaN when a
    program has no optimal solution.
    """
    post = ~pre
    n_donors = Y0.shape[1]
    level = Y0[post].mean(axis=0)
    A_ub = np.vstack([Y0[pre], -Y0[pre]])
    b_ub = np.r_[y1[pre] + tolerance, tolerance - y1[pre]]
    A_eq = np.ones((1, n_donors))
    b_eq = np.ones(1)
    lowest = optimize.linprog(level, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    highest = optimize.linprog(-level, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if lowest.status != 0 or highest.status != 0:
        return float("nan"), float("nan")
    observed = float(y1[post].mean())
    return observed + float(highest.fun), observed - float(lowest.fun)


def _comparable_placebos(placebo_pre: np.ndarray, factor: float | None) -> np.ndarray:
    """Boolean array marking the placebos whose pre-opening fit is comparable with that of a typical placebo.

    A placebo is comparable when its pre-opening RMSPE is at most ``factor`` times
    the median placebo RMSPE, the median taken as at least
    :data:`EXACT_FIT_TOLERANCE`, or when it is among the
    :data:`MIN_COMPARABLE_PLACEBOS` best-fitting placebos. Placebos tied with the
    last of them, to a relative difference of ``1e-9``, stay as well, so that the
    result does not depend on rounding error in the RMSPE values. ``factor=None``
    marks every placebo.
    """
    if factor is None:
        return np.ones(placebo_pre.size, dtype=bool)
    cutoff = factor * max(float(np.median(placebo_pre)), EXACT_FIT_TOLERANCE)
    kept = min(placebo_pre.size, MIN_COMPARABLE_PLACEBOS)
    best = float(np.sort(placebo_pre)[kept - 1])
    tied = best + _TIE_TOLERANCE * max(best, EXACT_FIT_TOLERANCE)
    return placebo_pre <= max(cutoff, tied)


def _weights(
    method: str,
    y_pre: np.ndarray,
    X_pre: np.ndarray,
    ridge: float,
    n_starts: int,
    seed: int,
) -> np.ndarray:
    """Donor weights fitted on the pre-opening outcome of one unit.

    ``y_pre`` has one entry per pre-opening year and ``X_pre`` one row per year
    and one column per donor.
    """
    if method == "nested":
        return scm.fit_nested(y_pre, X_pre, y_pre, X_pre, n_starts=n_starts, seed=seed).w
    base = scm.fit_direct(y_pre, X_pre).w
    if method == "ridge":
        return scm.ridge_augment_weights(y_pre, X_pre, base, ridge)
    return base


def _unit_summary(y: np.ndarray, synthetic: np.ndarray, pre: np.ndarray) -> dict[str, float]:
    """Effect, relative effect and RMSPEs of one unit given its synthetic series."""
    gap = y - synthetic
    post = ~pre
    att = float(gap[post].mean())
    rmspe_pre = _rmspe(gap[pre])
    rmspe_post = _rmspe(gap[post])
    return {
        "att": att,
        "rel": _relative(att, float(synthetic[post].mean())),
        "pre": rmspe_pre,
        "post": rmspe_post,
        "ratio": _ratio(rmspe_post, rmspe_pre),
    }


def _checked_number(value: Any, name: str, positive: bool) -> float:
    """Return ``value`` as a float; raise ValueError unless it is a finite real number, positive or non-negative as required."""
    kind = "positive" if positive else "non-negative"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a {kind} finite number, got {value!r}")
    number = float(value)
    if not (np.isfinite(number) and (number > 0.0 if positive else number >= 0.0)):
        raise ValueError(f"{name} must be a {kind} finite number, got {value!r}")
    return number


def _validate_factor(pre_fit_factor: float) -> float:
    """Return the factor of the pre-fit rule as a float; raise ValueError unless it is a positive finite number."""
    return _checked_number(pre_fit_factor, "pre_fit_factor", positive=True)


def _validate_placebo_factor(placebo_fit_factor: float | None) -> float | None:
    """Return the placebo factor as a float, or None; raise ValueError unless it is None or a positive finite number."""
    if placebo_fit_factor is None:
        return None
    return _checked_number(placebo_fit_factor, "placebo_fit_factor", positive=True)


def _validate_relative_tolerance(exact_fit_relative_tolerance: float) -> float:
    """Return the relative tolerance as a float; raise ValueError unless it is a non-negative finite number."""
    return _checked_number(exact_fit_relative_tolerance, "exact_fit_relative_tolerance", positive=False)


def estimate_case(
    panel: pd.DataFrame,
    case: pd.Series | Mapping[str, Any],
    donors: list[str],
    outcome: str = "receipts_pct_gdp",
    first_year: int = 1995,
    last_year: int = 2019,
    method: str = "direct",
    ridge: float = 0.0,
    seed: int = 0,
    guard: Callable[[], None] | None = None,
    n_starts: int = NESTED_STARTS,
    post_horizon: int | None = POST_HORIZON,
    pre_fit_factor: float = 2.0,
    placebo_fit_factor: float | None = PLACEBO_FIT_FACTOR,
    exact_fit_relative_tolerance: float = EXACT_FIT_RELATIVE_TOLERANCE,
) -> CaseEffect:
    """Estimate the synthetic-control effect of one case or episode.

    The weights are fitted on all pre-opening years from ``first_year`` in which
    the case economy has an observed outcome. The effect and every other
    statistic refer to the post-opening window, which runs from the opening year
    to the earliest of ``opening_year + post_horizon - 1``, ``last_year`` and the
    year before ``next_start_year``. Donors must have a complete outcome series
    from ``first_year`` to the end of that window. Placebo-in-space inference
    treats every donor in turn with the other donors as its pool, using the same
    method and the same years as for the case.

    Comparable placebos. A placebo is comparable when its pre-opening RMSPE is at
    most ``placebo_fit_factor`` times the median placebo pre-opening RMSPE (the
    median taken as at least :data:`EXACT_FIT_TOLERANCE`), and the ``min(J, 5)``
    best-fitting placebos of ``J`` donors are always comparable. The placebo
    standard deviations (``placebo_sd_pp``, ``placebo_sd_rel_pct``) and the rank
    p-value of the mean gap (``p_gap_rank``) use the comparable placebos only; the
    rank p-value of the RMSPE ratio (``p_ratio_rank``) uses every placebo that has
    a ratio. ``placebo_fit_factor=None`` keeps all placebos.

    Exact fits. The fit is exact when the gap is at most ``fit_tolerance`` in
    absolute value in every pre-opening year, where ``fit_tolerance`` is the
    larger of :data:`EXACT_FIT_TOLERANCE` and ``exact_fit_relative_tolerance``
    times the median placebo pre-opening RMSPE. With fewer pre-opening years than
    donors the weights of an exact fit are not unique; ``att_pp_min`` and
    ``att_pp_max`` are then the smallest and the largest mean post-opening gap
    over all non-negative weight vectors that sum to one and keep every
    pre-opening gap within ``fit_tolerance`` (two linear programs solved with
    :func:`scipy.optimize.linprog`), widened if necessary to contain ``att_pp``
    when the weights of the case are non-negative; ``att_pp_lp_min`` and
    ``att_pp_lp_max`` are the bounds of the programs. This is a range over
    weights that reproduce the pre-opening years within the tolerance. It is not
    a confidence interval and not a sampling interval, and it contains no
    sampling uncertainty. For a fit that is not exact, ``att_pp_min`` and
    ``att_pp_max`` equal ``att_pp`` and the raw bounds are NaN. For an exact fit
    ``rmspe_ratio`` and ``p_ratio_rank`` are NaN, and a placebo whose
    pre-opening RMSPE is at most ``fit_tolerance`` has no ratio and takes no part
    in the ratio rank.

    Non-unique placebo fits. When the fit of a placebo is exact and there are
    fewer pre-opening years than other donors, its minimiser is not unique. The
    weights are those of the rule of :func:`dtt.scm.fit_direct`: the vertex of the
    set of minimisers at which the active-set path of :func:`dtt.scm.qp_simplex`
    stops, with the smallest index winning a tie, and not the minimiser of
    smallest norm. The same solver supplies the weights of the other methods (the
    direct weights are the starting point of ``"ridge"`` and the inner problem of
    ``"nested"``). The effect of such a placebo depends on that rule, and so do
    the statistics that use it (the placebo standard deviations and ``p_gap_rank``
    when the placebo is comparable). ``CaseEffect.n_placebos_exact`` counts these
    placebos.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel from :func:`dtt.panel.build_global_panel`.
    case : pandas.Series or mapping
        Catalogue or episode row with the keys ``case_id``, ``iso3`` and
        ``opening_year``, and optionally ``next_start_year`` (start year of the
        next episode of the same economy; missing means none).
    donors : list of str
        Donor economy codes, for example from :func:`dtt.cases.donor_pool`; at
        least two, none equal to the economy of the case.
    outcome : str, default "receipts_pct_gdp"
        Panel column holding the outcome.
    first_year, last_year : int
        First and last year of the estimation window.
    method : {"direct", "ridge", "nested"}
        ``"direct"`` minimises the pre-opening squared gap over the simplex with
        :func:`dtt.scm.fit_direct`. ``"ridge"`` augments the direct weights by
        ridge regression of the remaining imbalance with
        :func:`dtt.scm.ridge_augment_weights`; the weights then sum to one but may
        be negative. ``"nested"`` runs :func:`dtt.scm.fit_nested` with the
        pre-opening outcomes as predictors.
    ridge : float, default 0.0
        Penalty of the ridge augmentation; required to be positive for
        ``method="ridge"`` and not used by the other methods.
    seed : int, default 0
        Seed of the nested optimisation.
    guard : callable, optional
        Function without arguments, called after every ``GUARD_INTERVAL``-th
        completed donor placebo fit (``J // 10`` calls for ``J`` donors); it may
        block, for example to let the machine cool down.
    n_starts : int, default ``NESTED_STARTS``
        Starting points of the nested optimisation, used for the case and for
        every placebo.
    post_horizon : int, optional, default 5
        Largest number of post-opening years, counted from the opening year;
        None removes this cap; the window then ends at ``last_year`` or just
        before the next episode.
    pre_fit_factor : float, default 2.0
        The pre-opening fit is accepted (``pre_fit_ok``) when ``rmspe_pre`` is at
        most this multiple of the median placebo pre-opening RMSPE and the fit is
        not exact with fewer pre-opening years than donors.
    placebo_fit_factor : float or None, default 2.0
        Multiple of the median placebo pre-opening RMSPE up to which a placebo is
        comparable; None keeps all placebos.
    exact_fit_relative_tolerance : float, default 0.05
        Multiple of the median placebo pre-opening RMSPE in ``fit_tolerance``; 0.0
        gives the absolute tolerance :data:`EXACT_FIT_TOLERANCE` only.

    Returns
    -------
    CaseEffect
        Effect, fit, placebo distribution and inference of the case.

    Raises
    ------
    ValueError
        If the method is unknown, the ridge penalty is not positive for
        ``method="ridge"``, ``pre_fit_factor`` or ``placebo_fit_factor`` is not a
        positive finite number (``placebo_fit_factor`` may also be None),
        ``exact_fit_relative_tolerance`` is not a non-negative finite number,
        ``post_horizon`` is neither None nor a positive whole number, the economy
        or a donor is not in the panel, a donor has missing outcome years in the
        window, there are fewer than two donors, or the case has fewer than two
        pre-opening or no post-opening years with data.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got '{method}'")
    if method == "ridge" and not ridge > 0.0:
        raise ValueError("method 'ridge' needs a positive ridge penalty")
    factor = _validate_factor(pre_fit_factor)
    placebo_factor = _validate_placebo_factor(placebo_fit_factor)
    relative_tolerance = _validate_relative_tolerance(exact_fit_relative_tolerance)
    case_id, iso3, opening = resolve_case(case)
    window_end = post_window_end(opening, last_year, post_horizon, resolve_next_start(case))
    donors = [str(d) for d in donors]
    if len(donors) < 2:
        raise ValueError("at least two donors are needed")
    if len(set(donors)) != len(donors):
        raise ValueError("donor codes must be unique")
    if iso3 in donors:
        raise ValueError(f"the economy of the case ({iso3}) is in the donor list")

    wide = outcome_matrix(panel, outcome, first_year, last_year)
    wide = wide.loc[wide.index <= window_end]
    if iso3 not in wide.columns:
        raise ValueError(f"economy '{iso3}' is not in the panel")
    lacking = [d for d in donors if d not in wide.columns]
    if lacking:
        raise ValueError(f"donor(s) {lacking[:5]} are not in the panel")
    incomplete = [d for d in donors if wide[d].isna().any()]
    if incomplete:
        raise ValueError(f"donor(s) {incomplete[:5]} have missing outcome years in the window up to {window_end}")

    years_all = wide.index.to_numpy()
    treated_all = wide[iso3].to_numpy(dtype=float)
    donor_all = wide[donors].to_numpy(dtype=float)
    valid = np.isfinite(treated_all)
    pre = (years_all < opening)[valid]
    n_pre, n_post = int(pre.sum()), int((~pre).sum())
    if n_pre < _MIN_PRE_FIT or n_post < 1:
        raise ValueError(f"case '{case_id}' has {n_pre} pre-opening and {n_post} post-opening years with data")
    y1 = treated_all[valid]
    Y0 = donor_all[valid]
    J = len(donors)

    w = _weights(method, y1[pre], Y0[pre], ridge, n_starts, seed)
    synthetic_r = Y0 @ w
    main = _unit_summary(y1, synthetic_r, pre)

    equal_weight = Y0.mean(axis=1)
    did = float((y1[~pre].mean() - y1[pre].mean()) - (equal_weight[~pre].mean() - equal_weight[pre].mean()))

    placebo_att = np.empty(J)
    placebo_rel = np.empty(J)
    placebo_pre = np.empty(J)
    placebo_ratio = np.empty(J)
    placebo_gap_r = np.empty((y1.size, J))
    Y0_pre = Y0[pre]
    for k in range(J):
        pool = np.delete(np.arange(J), k)
        wk = _weights(method, Y0_pre[:, k], Y0_pre[:, pool], ridge, n_starts, seed)
        synth_k = Y0[:, pool] @ wk
        s = _unit_summary(Y0[:, k], synth_k, pre)
        placebo_att[k], placebo_rel[k], placebo_pre[k], placebo_ratio[k] = s["att"], s["rel"], s["pre"], s["ratio"]
        placebo_gap_r[:, k] = Y0[:, k] - synth_k
        if guard is not None and (k + 1) % GUARD_INTERVAL == 0:
            guard()

    fit_tolerance = max(EXACT_FIT_TOLERANCE, relative_tolerance * float(np.median(placebo_pre)))
    fit_exact = bool(np.max(np.abs((y1 - synthetic_r)[pre])) <= fit_tolerance)
    comparable = _comparable_placebos(placebo_pre, placebo_factor)
    ratio = float("nan") if fit_exact else main["ratio"]
    placebo_ratio = np.where(placebo_pre <= fit_tolerance, np.nan, placebo_ratio)

    p_gap = rank_p_value(abs(main["att"]), np.abs(placebo_att[comparable]))
    p_ratio = rank_p_value(ratio, placebo_ratio)

    att_min = att_max = main["att"]
    lp_min = lp_max = float("nan")
    if fit_exact:
        lp_min, lp_max = _identification_range(y1, Y0, pre, fit_tolerance)
        att_min, att_max = lp_min, lp_max
        if w.min() >= -1.0e-12 and not (np.isnan(lp_min) or np.isnan(lp_max)):
            att_min, att_max = min(lp_min, main["att"]), max(lp_max, main["att"])
    pre_fit_ok = bool(main["pre"] <= factor * float(np.median(placebo_pre))) and not (fit_exact and n_pre < J)

    synthetic_full = donor_all @ w
    gap_full = treated_all - synthetic_full
    placebo_gaps = np.full((years_all.size, J), np.nan)
    placebo_gaps[valid] = placebo_gap_r
    return CaseEffect(
        case_id=case_id,
        iso3=iso3,
        opening_year=opening,
        n_pre=n_pre,
        n_post=n_post,
        window_end=window_end,
        post_horizon_used=window_end - opening + 1,
        n_donors=J,
        years=years_all.astype(np.int64),
        actual=treated_all,
        synthetic=synthetic_full,
        gap=gap_full,
        weights=pd.Series(w, index=pd.Index(donors, name="donor_iso3"), name="weight"),
        att_pp=main["att"],
        att_rel_pct=main["rel"],
        rmspe_pre=main["pre"],
        rmspe_post=main["post"],
        rmspe_ratio=ratio,
        p_gap_rank=p_gap,
        p_ratio_rank=p_ratio,
        placebo_att_pp=placebo_att,
        placebo_att_rel_pct=placebo_rel,
        placebo_rmspe_pre=placebo_pre,
        did_att_pp=did,
        method=method,
        outcome=outcome,
        placebo_gaps=placebo_gaps,
        fit_exact=fit_exact,
        att_pp_min=att_min,
        att_pp_max=att_max,
        pre_fit_ok=pre_fit_ok,
        pre_fit_factor=factor,
        placebo_comparable=comparable,
        placebo_fit_factor=placebo_factor,
        placebo_sd_pp=_sample_sd(placebo_att[comparable]),
        placebo_sd_rel_pct=_sample_sd(placebo_rel[comparable]),
        fit_tolerance=fit_tolerance,
        exact_fit_relative_tolerance=relative_tolerance,
        att_pp_lp_min=lp_min,
        att_pp_lp_max=lp_max,
    )


def _effect_row(effect: CaseEffect) -> dict[str, Any]:
    """One row of the effects table."""
    return {
        "case_id": effect.case_id,
        "iso3": effect.iso3,
        "economy_group": effect.iso3,
        "opening_year": effect.opening_year,
        "n_pre": effect.n_pre,
        "n_post": effect.n_post,
        "window_end": effect.window_end,
        "post_horizon_used": effect.post_horizon_used,
        "n_donors": effect.n_donors,
        "method": effect.method,
        "att_pp": effect.att_pp,
        "att_rel_pct": effect.att_rel_pct,
        "att_pp_min": effect.att_pp_min,
        "att_pp_max": effect.att_pp_max,
        "att_pp_lp_min": effect.att_pp_lp_min,
        "att_pp_lp_max": effect.att_pp_lp_max,
        "rmspe_pre": effect.rmspe_pre,
        "rmspe_post": effect.rmspe_post,
        "rmspe_ratio": effect.rmspe_ratio,
        "p_gap_rank": effect.p_gap_rank,
        "p_ratio_rank": effect.p_ratio_rank,
        "placebo_sd_pp": effect.placebo_sd_pp,
        "placebo_sd_rel_pct": effect.placebo_sd_rel_pct,
        "n_placebos": effect.n_placebos,
        "n_placebos_comparable": effect.n_placebos_comparable,
        "n_placebos_exact": effect.n_placebos_exact,
        "fit_exact": effect.fit_exact,
        "fit_tolerance": effect.fit_tolerance,
        "pre_fit_ok": effect.pre_fit_ok,
        "did_att_pp": effect.did_att_pp,
    }


def estimate_all(
    panel: pd.DataFrame,
    cases: pd.DataFrame,
    feasibility: pd.DataFrame | None = None,
    guard: Callable[[], None] | None = None,
    catalogue: pd.DataFrame | None = None,
    pre_fit_factor: float = 2.0,
    placebo_fit_factor: float | None = PLACEBO_FIT_FACTOR,
    exact_fit_relative_tolerance: float = EXACT_FIT_RELATIVE_TOLERANCE,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, CaseEffect]]:
    """Estimate the effect of every feasible case or episode.

    Each case is estimated with :func:`estimate_case`. The placebo standard
    deviations and the rank p-value of the mean gap use the comparable placebos
    of the case: a placebo is comparable when its pre-opening RMSPE is at most
    ``placebo_fit_factor`` times the median placebo pre-opening RMSPE (the median
    taken as at least :data:`EXACT_FIT_TOLERANCE`), and the ``min(J, 5)``
    best-fitting placebos of ``J`` donors always are; ``placebo_fit_factor=None``
    keeps all placebos. The rank p-value of the RMSPE ratio uses every placebo
    that has a ratio. The fit of a case is exact when its largest absolute
    pre-opening gap is at most ``fit_tolerance``, the larger of
    :data:`EXACT_FIT_TOLERANCE` and ``exact_fit_relative_tolerance`` times the
    median placebo pre-opening RMSPE. The range ``att_pp_min`` to ``att_pp_max`` of
    an exact fit is a range over weights that reproduce the pre-opening years
    within that tolerance; it is not a confidence interval and not a sampling
    interval, and it contains no sampling uncertainty. When the fit of a placebo is
    exact and there are fewer pre-opening years than other donors, the weights of
    that placebo are not unique and its effect depends on the rule of
    :func:`dtt.scm.fit_direct`; ``n_placebos_exact`` counts these placebos.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel from :func:`dtt.panel.build_global_panel`.
    cases : pandas.DataFrame
        Catalogue from :func:`dtt.cases.load_cases` or episode table from
        :func:`dtt.cases.build_episodes`, with the columns ``case_id``, ``iso3``
        and ``opening_year``; the column ``next_start_year`` limits the
        post-opening window when present.
    feasibility : pandas.DataFrame, optional
        Output of :func:`dtt.cases.check_feasibility`; computed with the keyword
        arguments below when omitted. Only cases with ``feasible`` true are
        estimated.
    guard : callable, optional
        Function without arguments, called once before each feasible case is
        estimated and, inside :func:`estimate_case`, after every tenth donor
        placebo fit of that case; a case with ``J`` donors triggers ``1 + J // 10``
        calls. It may block, for example to let the machine cool down.
    catalogue : pandas.DataFrame, optional
        Full catalogue whose openings exclude economies from the donor pools; see
        :func:`dtt.cases.donor_pool`.
    pre_fit_factor : float, default 2.0
        Factor of the pre-opening fit rule of :func:`estimate_case`.
    placebo_fit_factor : float or None, default 2.0
        Multiple of the median placebo pre-opening RMSPE up to which a placebo is
        comparable; None keeps all placebos.
    exact_fit_relative_tolerance : float, default 0.05
        Multiple of the median placebo pre-opening RMSPE in ``fit_tolerance``; 0.0
        gives the absolute tolerance :data:`EXACT_FIT_TOLERANCE` only.
    **kwargs
        ``outcome``, ``first_year``, ``last_year``, ``buffer`` and
        ``post_horizon`` (used for the feasibility check, the donor pools and the
        estimates), ``min_pre``, ``min_post`` and ``min_donors`` (feasibility
        check), and ``method``, ``ridge``, ``seed`` and ``n_starts`` (see
        :func:`estimate_case`).

    Returns
    -------
    effects : pandas.DataFrame
        One row per feasible case with ``case_id``, ``iso3``, ``economy_group``
        (equal to ``iso3``, for clustering), ``opening_year``, ``n_pre``,
        ``n_post``, ``window_end``, ``post_horizon_used``, ``n_donors``,
        ``method``, ``att_pp``, ``att_rel_pct``, ``att_pp_min`` and
        ``att_pp_max`` (range of the effect over the weights of an exact fit,
        widened to contain ``att_pp`` when the weights are non-negative; both
        equal ``att_pp`` otherwise), ``att_pp_lp_min`` and ``att_pp_lp_max`` (the
        bounds of the two linear programs before any widening; NaN unless the fit
        is exact), ``rmspe_pre``, ``rmspe_post``, ``rmspe_ratio`` (NaN for an
        exact fit), ``p_gap_rank``, ``p_ratio_rank`` (NaN for an exact fit),
        ``placebo_sd_pp`` and ``placebo_sd_rel_pct`` (sample standard deviations of
        the finite effects of the comparable placebos), ``n_placebos``,
        ``n_placebos_comparable`` and ``n_placebos_exact`` (number of placebos, of
        comparable placebos, and of placebos whose pre-opening RMSPE is at most
        ``fit_tolerance``), ``fit_exact`` (bool; see :func:`estimate_case`),
        ``fit_tolerance``, ``pre_fit_ok`` (bool; ``rmspe_pre`` at most
        ``pre_fit_factor`` times the median placebo pre-opening RMSPE, and false
        for an exact fit with fewer pre-opening years than donors) and
        ``did_att_pp``.
    placebos : pandas.DataFrame
        One row per case and donor with ``case_id``, ``donor_iso3``,
        ``placebo_att_pp``, ``placebo_att_rel_pct``, ``placebo_rmspe_pre`` and the
        boolean ``comparable``.
    results : dict of str to CaseEffect
        The full result of every estimated case, keyed by ``case_id``.

    Raises
    ------
    TypeError
        If an unknown keyword argument is passed.
    ValueError
        If ``cases`` lacks a required column or repeats a ``case_id``,
        ``feasibility`` lacks the columns ``case_id`` and ``feasible``,
        ``pre_fit_factor`` or ``placebo_fit_factor`` is not a positive finite
        number (``placebo_fit_factor`` may also be None), or
        ``exact_fit_relative_tolerance`` is not a non-negative finite number.
    """
    unknown = set(kwargs) - _FEASIBILITY_KEYS - _SHARED_KEYS - _ESTIMATE_KEYS
    if unknown:
        raise TypeError(f"estimate_all got unexpected keyword argument(s) {sorted(unknown)}")
    factor = _validate_factor(pre_fit_factor)
    placebo_factor = _validate_placebo_factor(placebo_fit_factor)
    relative_tolerance = _validate_relative_tolerance(exact_fit_relative_tolerance)
    lacking = [c for c in ("case_id", "iso3", "opening_year") if c not in cases.columns]
    if lacking:
        raise ValueError(f"cases lack column(s) {lacking}")
    if cases["case_id"].duplicated().any():
        raise ValueError("case_id values must be unique")
    outcome = kwargs.get("outcome", "receipts_pct_gdp")
    first_year = kwargs.get("first_year", 1995)
    last_year = kwargs.get("last_year", 2019)
    buffer = kwargs.get("buffer", 5)
    post_horizon = kwargs.get("post_horizon", POST_HORIZON)
    estimate_args = {k: kwargs[k] for k in _ESTIMATE_KEYS if k in kwargs}

    if feasibility is None:
        feasibility = check_feasibility(
            cases,
            panel,
            outcome=outcome,
            first_year=first_year,
            last_year=last_year,
            buffer=buffer,
            post_horizon=post_horizon,
            catalogue=catalogue,
            **{k: kwargs[k] for k in _FEASIBILITY_KEYS if k in kwargs},
        )
    absent = [c for c in ("case_id", "feasible") if c not in feasibility.columns]
    if absent:
        raise ValueError(f"feasibility lacks column(s) {absent}")
    feasible_ids = set(feasibility.loc[feasibility["feasible"].astype(bool), "case_id"])

    results: dict[str, CaseEffect] = {}
    effect_rows: list[dict[str, Any]] = []
    placebo_frames: list[pd.DataFrame] = []
    for _, row in cases.iterrows():
        if row["case_id"] not in feasible_ids:
            continue
        if guard is not None:
            guard()
        donors = donor_pool(
            panel, row, cases, outcome=outcome, first_year=first_year, last_year=last_year, buffer=buffer, catalogue=catalogue
        )
        effect = estimate_case(
            panel,
            row,
            donors,
            outcome=outcome,
            first_year=first_year,
            last_year=last_year,
            guard=guard,
            post_horizon=post_horizon,
            pre_fit_factor=factor,
            placebo_fit_factor=placebo_factor,
            exact_fit_relative_tolerance=relative_tolerance,
            **estimate_args,
        )
        results[effect.case_id] = effect
        effect_rows.append(_effect_row(effect))
        placebo_frames.append(
            pd.DataFrame(
                {
                    "case_id": effect.case_id,
                    "donor_iso3": list(effect.weights.index),
                    "placebo_att_pp": effect.placebo_att_pp,
                    "placebo_att_rel_pct": effect.placebo_att_rel_pct,
                    "placebo_rmspe_pre": effect.placebo_rmspe_pre,
                    "comparable": effect.placebo_comparable,
                }
            )
        )

    effects = pd.DataFrame(effect_rows, columns=list(_EFFECT_COLUMNS))
    placebos = (
        pd.concat(placebo_frames, ignore_index=True)
        if placebo_frames
        else pd.DataFrame(columns=list(_PLACEBO_COLUMNS))
    )
    if not effects.empty:
        for column in ("opening_year", "n_pre", "n_post", "window_end", "post_horizon_used", "n_donors"):
            effects[column] = effects[column].astype(np.int64)
        for column in ("n_placebos", "n_placebos_comparable", "n_placebos_exact"):
            effects[column] = effects[column].astype(np.int64)
        effects["fit_exact"] = effects["fit_exact"].astype(bool)
        effects["pre_fit_ok"] = effects["pre_fit_ok"].astype(bool)
    placebos["comparable"] = placebos["comparable"].astype(bool)
    return effects, placebos[list(_PLACEBO_COLUMNS)], results


def write_effects(effects: pd.DataFrame, placebos: pd.DataFrame, directory: str | Path) -> tuple[Path, Path]:
    """Write the effects and the placebo table as csv files.

    Parameters
    ----------
    effects : pandas.DataFrame
        Effects table from :func:`estimate_all`.
    placebos : pandas.DataFrame
        Placebo table from :func:`estimate_all`.
    directory : str or pathlib.Path
        Output directory, created if needed.

    Returns
    -------
    tuple of pathlib.Path
        Paths of ``case_effects.csv`` and ``case_placebos.csv``.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    effects_path = out / "case_effects.csv"
    placebos_path = out / "case_placebos.csv"
    effects.to_csv(effects_path, index=False)
    placebos.to_csv(placebos_path, index=False)
    return effects_path, placebos_path
