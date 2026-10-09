"""Thailand impact scaling and the rule that selects the primary route.

Functions
---------
load_thailand_baseline
    Read the Thailand baseline series into one row per year.
scale_effect
    Convert an effect estimate into US$ and THB billion per year.
impact_table
    Quantile table of the scaled effect for chosen baseline years.
minimum_detectable_effect, detection_probability
    Detection threshold and detection probability from null draws.
decide_route
    Apply the decision rule that picks the primary route.
route_comparison
    Stack the impact tables of several routes and flag the primary one.
"""
from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

__all__ = [
    "DEFAULT_ROUTE_RULES",
    "load_thailand_baseline",
    "scale_effect",
    "impact_table",
    "minimum_detectable_effect",
    "detection_probability",
    "RouteDecision",
    "decide_route",
    "route_comparison",
]

_BASELINE_VARIABLES = MappingProxyType(
    {
        "gdp_current_usd_bn": "gdp_usd_bn",
        "intl_tourism_receipts_usd_bn": "receipts_usd_bn",
        "receipts_pct_gdp": "receipts_pct_gdp",
        "intl_arrivals_million": "arrivals_million",
        "fx_thb_per_usd": "fx_thb_per_usd",
    }
)

DEFAULT_ROUTE_RULES = MappingProxyType(
    {"rmse_ratio_equal": 0.95, "rmse_ratio_uniform": 0.95, "min_cases": 10}
)

_DEFAULT_BASELINE_YEARS = (2019, 2024)
_ROUTE_METHODS = ("ot_weighted", "equal", "ot_uniform")
_MAX_RMSE_RATIO = 1.5


# ----------------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------------
def _as_years(years: Any) -> list[int]:
    """Convert a year or a sequence of years to a list of integers.

    Parameters
    ----------
    years : int or sequence of int
        Years; whole-valued floats and NumPy integers are accepted.

    Returns
    -------
    list of int
        The years, in the given order.

    Raises
    ------
    ValueError
        If ``years`` is empty, or holds a boolean, a string or a number that is
        not a finite whole number.
    """
    values = [years] if np.ndim(years) == 0 else list(years)
    if not values:
        raise ValueError("years must contain at least one year.")
    out = []
    for value in values:
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise ValueError("years must be whole numbers.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("years must be whole numbers.") from exc
        if not np.isfinite(number) or number != np.floor(number):
            raise ValueError("years must be whole numbers.")
        out.append(int(number))
    return out


def load_thailand_baseline(
    path: str | PathLike[str], years: int | Sequence[int] | None = None
) -> pd.DataFrame:
    """Read the Thailand baseline series from the long-format CSV file.

    The file has the columns ``year``, ``variable``, ``value``, ``unit``,
    ``source``, ``url`` and ``note``.  Only the variables
    ``gdp_current_usd_bn``, ``intl_tourism_receipts_usd_bn``,
    ``receipts_pct_gdp``, ``intl_arrivals_million`` and ``fx_thb_per_usd`` are
    used; every other variable, including those with the prefix ``alt_``, is
    ignored.  Every value present for these variables must be finite and
    positive, each variable must have a value in at least one year, and each
    year listed in ``years`` must have all five variables.

    Parameters
    ----------
    path : str or path-like
        Location of the long-format CSV file, in UTF-8 with or without a byte
        order mark.
    years : int or sequence of int, optional
        Years that the file must contain with all five variables.  The default
        applies this requirement to those of the years 2019 and 2024 that the
        file contains.

    Returns
    -------
    pandas.DataFrame
        One row per year, indexed by the integer year (named ``year``) in
        ascending order, with the float columns ``gdp_usd_bn``,
        ``receipts_usd_bn``, ``receipts_pct_gdp``, ``arrivals_million`` and
        ``fx_thb_per_usd``.  A variable that has no value for a year outside
        ``years`` gives a missing value in that cell.

    Raises
    ------
    ValueError
        If a required column or variable is absent, a year or value cannot be
        read as a number, a year and variable pair occurs more than once, a
        value is not finite and positive, or a year in ``years`` is absent or
        lacks a variable.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = [c for c in ("year", "variable", "value") if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"The baseline file has no column named {missing_columns}.")
    rows = raw.loc[raw["variable"].isin(list(_BASELINE_VARIABLES)), ["year", "variable", "value"]]
    rows = rows.copy()
    file_years = pd.to_numeric(rows["year"], errors="raise")
    if not bool(np.all(np.mod(file_years.to_numpy(dtype=float), 1.0) == 0.0)):
        raise ValueError("The year column must contain whole numbers.")
    rows["year"] = file_years.astype(int)
    rows["value"] = pd.to_numeric(rows["value"], errors="raise").astype(float)
    if rows.duplicated(subset=["year", "variable"]).any():
        raise ValueError("A year and variable pair occurs more than once in the baseline file.")
    present = rows.loc[rows["value"].notna()]
    absent = [name for name in _BASELINE_VARIABLES if name not in set(present["variable"])]
    if absent:
        raise ValueError(f"The baseline file has no values for the variables {absent}.")
    invalid = present.loc[~(np.isfinite(present["value"]) & (present["value"] > 0.0))]
    if len(invalid):
        first = invalid.iloc[0]
        raise ValueError(
            f"The baseline value of {first['variable']} in {int(first['year'])} must be finite "
            f"and positive; found {first['value']}."
        )
    wide = rows.pivot(index="year", columns="variable", values="value")
    wide = wide.rename(columns=dict(_BASELINE_VARIABLES))[list(_BASELINE_VARIABLES.values())]
    wide = wide.sort_index().astype(float)
    wide.index = wide.index.astype(int)
    wide.index.name = "year"
    wide.columns.name = None
    if years is None:
        required = [year for year in _DEFAULT_BASELINE_YEARS if year in wide.index]
    else:
        required = _as_years(years)
    for year in required:
        if year not in wide.index:
            raise ValueError(f"The baseline file has no rows for the year {year}.")
        lacking = [name for name in wide.columns if pd.isna(wide.loc[year, name])]
        if lacking:
            raise ValueError(f"The baseline file has no value for {lacking} in {year}.")
    return wide


def _baseline_by_year(baseline: pd.DataFrame) -> pd.DataFrame:
    """Index the baseline table by whole-number year.

    Parameters
    ----------
    baseline : pandas.DataFrame
        Baseline table, indexed by year or with a ``year`` column.

    Returns
    -------
    pandas.DataFrame
        The table indexed by integer year.

    Raises
    ------
    ValueError
        If a year is not a whole number or occurs in more than one row.
    """
    table = baseline.set_index("year") if "year" in baseline.columns else baseline
    labels = pd.to_numeric(pd.Series(table.index), errors="coerce").to_numpy(dtype=float)
    if not bool(np.all(np.isfinite(labels) & (labels == np.floor(labels)))):
        raise ValueError("The baseline index must hold whole-number years.")
    table = table.copy()
    table.index = pd.Index(labels.astype(int), name="year")
    if not table.index.is_unique:
        repeated = sorted(set(table.index[table.index.duplicated()]))
        raise ValueError(f"The baseline table has more than one row for the years {repeated}.")
    return table


# ----------------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------------
def _unwrap(values: Any) -> float | np.ndarray:
    """Convert a zero-dimensional result to a float.

    Parameters
    ----------
    values : array_like
        Scalar or array.

    Returns
    -------
    float or ndarray
        A float for a scalar and a float array otherwise.
    """
    arr = np.asarray(values, dtype=float)
    return float(arr) if arr.ndim == 0 else arr


def _usd_amount(
    effect: ArrayLike, baseline_row: Mapping[str, Any] | pd.Series, scaling: str
) -> np.ndarray:
    """Scale an effect to US$ billion per year for one baseline year.

    Parameters
    ----------
    effect : array_like
        Effect in percentage points of GDP (``scaling="absolute"``) or in
        percent of the counterfactual receipts (``scaling="relative"``).
    baseline_row : mapping or Series
        Baseline year with ``gdp_usd_bn`` and ``receipts_usd_bn``.
    scaling : {"absolute", "relative"}
        Scaling rule.

    Returns
    -------
    ndarray
        ``effect / 100`` times the year's GDP or receipts in US$ billion.
    """
    base = baseline_row["gdp_usd_bn" if scaling == "absolute" else "receipts_usd_bn"]
    return np.asarray(effect, dtype=float) / 100.0 * float(base)


def scale_effect(
    att_pp: ArrayLike | None,
    att_rel_pct: ArrayLike | None,
    baseline_row: Mapping[str, Any] | pd.Series,
) -> dict[str, Any]:
    """Convert an effect estimate into economic quantities for one baseline year.

    The absolute scaling multiplies the effect in percentage points of GDP by
    the year's GDP: ``att_pp / 100 * gdp_usd_bn`` in US$ billion per year.  The
    relative scaling multiplies the effect in percent of the counterfactual
    receipts by the year's receipts: ``att_rel_pct / 100 * receipts_usd_bn``.
    Each US$ amount is converted to THB billion with ``fx_thb_per_usd``.

    Parameters
    ----------
    att_pp : float, array_like or None
        Effect in percentage points of GDP.  ``None`` skips the absolute scaling.
    att_rel_pct : float, array_like or None
        Effect in percent of the counterfactual receipts.  ``None`` skips the
        relative scaling.
    baseline_row : mapping or Series
        One baseline year with the entries ``gdp_usd_bn``, ``receipts_usd_bn``
        and ``fx_thb_per_usd``.

    Returns
    -------
    dict
        ``absolute_usd_bn``, ``absolute_thb_bn``, ``relative_usd_bn`` and
        ``relative_thb_bn`` (``None`` for a scaling that was skipped), and
        ``ratio_relative_to_absolute``, the relative US$ amount divided by the
        absolute US$ amount (``None`` unless both were computed, and NaN where
        the absolute amount is zero).  Scalar inputs give floats and array
        inputs give arrays.
    """
    out: dict[str, Any] = {
        "absolute_usd_bn": None,
        "absolute_thb_bn": None,
        "relative_usd_bn": None,
        "relative_thb_bn": None,
        "ratio_relative_to_absolute": None,
    }
    fx = float(baseline_row["fx_thb_per_usd"])
    absolute = relative = None
    if att_pp is not None:
        absolute = _usd_amount(att_pp, baseline_row, "absolute")
        out["absolute_usd_bn"] = _unwrap(absolute)
        out["absolute_thb_bn"] = _unwrap(absolute * fx)
    if att_rel_pct is not None:
        relative = _usd_amount(att_rel_pct, baseline_row, "relative")
        out["relative_usd_bn"] = _unwrap(relative)
        out["relative_thb_bn"] = _unwrap(relative * fx)
    if absolute is not None and relative is not None:
        nonzero = absolute != 0.0
        ratio = np.where(nonzero, relative / np.where(nonzero, absolute, 1.0), np.nan)
        out["ratio_relative_to_absolute"] = _unwrap(ratio)
    return out


def _quantile_label(prob: float) -> str:
    """Column label of a quantile.

    Parameters
    ----------
    prob : float
        Probability in [0, 1].

    Returns
    -------
    str
        ``q`` followed by the percentage, for example ``q5`` for 0.05.
    """
    return f"q{round(100.0 * float(prob), 6):g}"


def _as_draws(values: Any, name: str) -> np.ndarray:
    """Return ``values`` as a finite, non-empty one-dimensional float array.

    Parameters
    ----------
    values : array_like
        Input values.
    name : str
        Name of the argument, used in error messages.

    Returns
    -------
    ndarray, shape (n,)
        The values as floats.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _percent_of(value: float, base: float) -> float:
    """Express ``value`` as a percentage of ``base``.

    Parameters
    ----------
    value : float
        Numerator.
    base : float
        Denominator.

    Returns
    -------
    float
        ``100 * value / base``; NaN or infinity when ``base`` is zero or missing.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(100.0 * np.float64(value) / np.float64(base))


def impact_table(
    draws_pp: ArrayLike | None,
    draws_rel: ArrayLike | None,
    baseline: pd.DataFrame,
    years: int | Sequence[int] = (2019, 2024),
    route: str = "",
    probs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """Quantiles of the scaled effect for chosen baseline years.

    For each baseline year the draws of the effect in percentage points of GDP
    are scaled with the absolute rule of :func:`scale_effect` and the draws in
    percent of receipts with the relative rule, both in US$ billion per year.

    Parameters
    ----------
    draws_pp : array_like or None
        Draws of the effect in percentage points of GDP; ``None`` omits the
        absolute scaling.
    draws_rel : array_like or None
        Draws of the effect in percent of the counterfactual receipts; ``None``
        omits the relative scaling.
    baseline : pandas.DataFrame
        Output of :func:`load_thailand_baseline`, indexed by year, or a table
        with a ``year`` column.  Only the columns ``gdp_usd_bn`` and
        ``receipts_usd_bn`` are used; they must be finite and positive in each
        requested year.
    years : int or sequence of int, default (2019, 2024)
        Baseline years; whole-valued floats are accepted.
    route : str, default ""
        Label stored in the ``route`` column.
    probs : sequence of float, default (0.05, 0.25, 0.5, 0.75, 0.95)
        Probabilities of the reported quantiles.

    Returns
    -------
    pandas.DataFrame
        One row per baseline year and scaling.  Columns: ``route``,
        ``baseline_year``, ``scaling`` (``absolute`` or ``relative``), one column
        per probability named ``q`` followed by the percentage (``q5``, ``q25``,
        ``q50``, ``q75``, ``q95`` for the default) in US$ billion per year,
        ``median_share_receipts_pct`` and ``median_share_gdp_pct`` (the median
        as a percentage of that year's baseline receipts and of its GDP).

    Raises
    ------
    ValueError
        If no draws are given, a draw or probability is invalid, a year is not
        a whole number or is absent from the baseline, the baseline has more
        than one row for a year, or its GDP or receipts in a requested year are
        not finite and positive.
    """
    if draws_pp is None and draws_rel is None:
        raise ValueError("At least one of draws_pp and draws_rel must be given.")
    pp = None if draws_pp is None else _as_draws(draws_pp, "draws_pp")
    rel = None if draws_rel is None else _as_draws(draws_rel, "draws_rel")
    prob_arr = np.asarray(probs, dtype=float)
    if prob_arr.ndim != 1 or prob_arr.size == 0 or np.any((prob_arr < 0.0) | (prob_arr > 1.0)):
        raise ValueError("probs must be a non-empty sequence of probabilities in [0, 1].")
    labels = [_quantile_label(p) for p in prob_arr]
    if len(set(labels)) != len(labels):
        raise ValueError("probs must not contain duplicate values.")
    table = _baseline_by_year(baseline)
    needed = ["gdp_usd_bn", "receipts_usd_bn"]
    lacking = [c for c in needed if c not in table.columns]
    if lacking:
        raise ValueError(f"The baseline table has no column named {lacking}.")
    year_list = _as_years(years)

    rows = []
    for year in year_list:
        if year not in table.index:
            raise ValueError(f"The baseline table has no row for the year {year}.")
        row = table.loc[year]
        for column in needed:
            value = float(row[column])
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"The baseline value of {column} in {year} must be finite and positive; "
                    f"found {value}."
                )
        for scaling, draws in (("absolute", pp), ("relative", rel)):
            if draws is None:
                continue
            scaled = _usd_amount(draws, row, scaling)
            median = float(np.median(scaled))
            record: dict[str, Any] = {
                "route": route,
                "baseline_year": year,
                "scaling": scaling,
            }
            record.update(zip(labels, (float(q) for q in np.quantile(scaled, prob_arr))))
            record["median_share_receipts_pct"] = _percent_of(median, row["receipts_usd_bn"])
            record["median_share_gdp_pct"] = _percent_of(median, row["gdp_usd_bn"])
            rows.append(record)
    columns = [
        "route",
        "baseline_year",
        "scaling",
        *labels,
        "median_share_receipts_pct",
        "median_share_gdp_pct",
    ]
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def minimum_detectable_effect(
    null_draws: ArrayLike, alpha: float = 0.05, power: float = 0.8
) -> dict[str, float]:
    """Detection threshold and minimum detectable effect from null draws.

    An effect is detected when the absolute value of the observed effect
    exceeds ``threshold``.  An observed effect is the true effect plus a draw
    from the null distribution, whose shape is taken from ``null_draws`` as it
    is (empirical quantiles with linear interpolation), without assuming
    normality or a centre at zero.

    Parameters
    ----------
    null_draws : array_like
        Draws of the effect under the null hypothesis, for example the
        transported placebo effects; at least two.
    alpha : float, default 0.05
        Size of the test.
    power : float, default 0.8
        Target power.

    Returns
    -------
    dict
        ``threshold``
            The ``1 - alpha`` quantile of the absolute null draws.
        ``mde``
            The threshold plus the larger of the ``power`` quantile of the null
            draws and the negative of their ``1 - power`` quantile; for a null
            symmetric about zero both equal the ``power`` quantile of the null.
            An effect of size ``mde`` of either sign, added to the null draws,
            has an absolute value above the threshold in at least the share
            ``power`` of the draws, up to the resolution of one draw.
        ``mde_normal``
            The normal-theory value for a null centred at zero: the threshold
            plus ``z_power`` times the sample standard deviation of the null
            draws, where ``z_power`` is the standard normal quantile at
            ``power``.

    Raises
    ------
    ValueError
        If ``null_draws`` has fewer than two values or a value that is not
        finite, or ``alpha`` or ``power`` is outside (0, 1).
    """
    draws = _as_draws(null_draws, "null_draws")
    if draws.size < 2:
        raise ValueError("null_draws must contain at least two values.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1.")
    threshold = float(np.quantile(np.abs(draws), 1.0 - alpha))
    reach = max(float(np.quantile(draws, power)), -float(np.quantile(draws, 1.0 - power)))
    mde_normal = threshold + float(stats.norm.ppf(power)) * float(np.std(draws, ddof=1))
    return {"threshold": threshold, "mde": threshold + reach, "mde_normal": mde_normal}


def detection_probability(
    effect_draws: ArrayLike, threshold: float, null_draws: ArrayLike | None = None
) -> float:
    """Share of effect draws whose absolute value exceeds a threshold.

    Without ``null_draws`` the effect draws are observed effects.  With
    ``null_draws`` they are true effects and each is observed with an
    independent draw from the empirical null distribution, so the result is the
    probability that the absolute value of an effect plus a null draw exceeds
    the threshold.

    Parameters
    ----------
    effect_draws : float or array_like
        Draws of the effect.
    threshold : float
        Detection threshold.
    null_draws : array_like, optional
        Draws of the effect under the null hypothesis.

    Returns
    -------
    float
        Without ``null_draws``, the proportion of draws with
        ``abs(draw) > threshold``.  With ``null_draws``, the average over the
        effect draws of the proportion of null draws ``n`` with
        ``abs(draw + n) > threshold``.  The value lies in [0, 1].
    """
    draws = _as_draws(np.atleast_1d(np.asarray(effect_draws, dtype=float)), "effect_draws")
    threshold = float(threshold)
    if np.isnan(threshold):
        raise ValueError("threshold must not be NaN.")
    if null_draws is None:
        return float(np.mean(np.abs(draws) > threshold))
    null = np.sort(_as_draws(null_draws, "null_draws"))
    above = null.size - np.searchsorted(null, threshold - draws, side="right")
    below = np.searchsorted(null, -threshold - draws, side="left")
    return float(np.mean(np.minimum(above + below, null.size)) / null.size)


# ----------------------------------------------------------------------------
# Route decision
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteDecision:
    """Outcome of the route decision rule.

    Attributes
    ----------
    primary : str
        ``ot_importance`` or ``ambient``.
    passed : dict
        Criterion name to whether the criterion is met: ``importance_usable``,
        ``loco_rmse`` and ``enough_cases``.
    details : str
        Plain-language account of each criterion and of the decision.
    """

    primary: str
    passed: dict[str, bool]
    details: str


def _usable_flag(importance_diagnostics: Mapping[str, Any]) -> bool:
    """Read the ``usable`` flag of the importance diagnostics.

    Parameters
    ----------
    importance_diagnostics : mapping
        Diagnostics with a boolean entry ``usable``.

    Returns
    -------
    bool
        The value of ``usable``.

    Raises
    ------
    ValueError
        If the diagnostics have no entry ``usable`` or its value is not a Python
        or NumPy boolean.
    """
    try:
        value = importance_diagnostics["usable"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "importance_diagnostics must be a mapping with an entry named 'usable'."
        ) from exc
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("importance_diagnostics['usable'] must be a boolean.")
    return bool(value)


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real number that is not a boolean.

    Parameters
    ----------
    value : object
        Value to test.

    Returns
    -------
    bool
        True for Python and NumPy integers and floats.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _loco_rmse(loco_summary: pd.DataFrame) -> dict[str, float]:
    """RMSE of each compared method in the leave-one-group-out summary.

    Parameters
    ----------
    loco_summary : pandas.DataFrame
        Summary indexed by method name with a column ``rmse``.

    Returns
    -------
    dict
        The RMSE of ``ot_weighted``, ``equal`` and ``ot_uniform``.

    Raises
    ------
    ValueError
        If the summary is not a DataFrame with a column ``rmse`` and one row per
        method, lacks one of the three methods, or holds an RMSE that is not a
        finite positive number.
    """
    if not isinstance(loco_summary, pd.DataFrame):
        raise ValueError("loco_summary must be a DataFrame indexed by method name.")
    if "rmse" not in loco_summary.columns:
        raise ValueError("loco_summary must have a column named 'rmse'.")
    if not loco_summary.index.is_unique:
        raise ValueError("loco_summary must have one row per method.")
    absent = [method for method in _ROUTE_METHODS if method not in loco_summary.index]
    if absent:
        raise ValueError(
            f"loco_summary has no row for the methods {absent}; its index must hold the "
            "method names."
        )
    out = {}
    for method in _ROUTE_METHODS:
        value = loco_summary.loc[method, "rmse"]
        if not _is_number(value):
            raise ValueError(f"The rmse of {method} must be a number.")
        number = float(value)
        if not (np.isfinite(number) and number > 0.0):
            raise ValueError(f"The rmse of {method} must be finite and positive; found {number}.")
        out[method] = number
    return out


def _resolve_rules(rules: Mapping[str, float] | None) -> dict[str, float]:
    """Merge user thresholds into the defaults and validate them.

    Parameters
    ----------
    rules : mapping or None
        Overrides for ``rmse_ratio_equal``, ``rmse_ratio_uniform`` and
        ``min_cases``.

    Returns
    -------
    dict
        The three thresholds.

    Raises
    ------
    ValueError
        If ``rules`` is not a mapping, a name is unknown, a value is not a
        number, an RMSE ratio is outside (0, 1.5], or ``min_cases`` is negative
        or not finite.
    """
    resolved = dict(DEFAULT_ROUTE_RULES)
    if rules is not None:
        if not isinstance(rules, Mapping):
            raise ValueError("rules must be a mapping from threshold names to numbers.")
        unknown = sorted(set(rules) - set(resolved))
        if unknown:
            raise ValueError(f"Unknown rule names {unknown}; allowed names are {sorted(resolved)}.")
        for key, value in rules.items():
            if not _is_number(value):
                raise ValueError(f"The threshold {key} must be a number.")
            resolved[key] = float(value)
    for key in ("rmse_ratio_equal", "rmse_ratio_uniform"):
        if not (np.isfinite(resolved[key]) and 0.0 < resolved[key] <= _MAX_RMSE_RATIO):
            raise ValueError(f"The threshold {key} must lie in (0, {_MAX_RMSE_RATIO:g}].")
    if not (np.isfinite(resolved["min_cases"]) and resolved["min_cases"] >= 0.0):
        raise ValueError("The threshold min_cases must be a non-negative finite number.")
    return resolved


def _check_n_cases(n_cases: Any) -> int:
    """Validate the number of cases.

    Parameters
    ----------
    n_cases : int
        Number of cases.

    Returns
    -------
    int
        The number of cases.

    Raises
    ------
    ValueError
        If ``n_cases`` is not a non-negative integer (a boolean is not).
    """
    if not isinstance(n_cases, numbers.Integral) or isinstance(n_cases, (bool, np.bool_)):
        raise ValueError("n_cases must be a non-negative integer.")
    if n_cases < 0:
        raise ValueError("n_cases must be a non-negative integer.")
    return int(n_cases)


def decide_route(
    importance_diagnostics: Mapping[str, Any],
    loco_summary: pd.DataFrame,
    n_cases: int,
    rules: Mapping[str, float] | None = None,
) -> RouteDecision:
    """Choose the primary route for the Thailand estimate.

    The primary route is ``ot_importance`` if and only if all three criteria
    hold, and ``ambient`` otherwise:

    1. ``importance_diagnostics["usable"]`` is true.
    2. The leave-one-economy-out RMSE of ``ot_weighted`` is at most 0.95 times
       the RMSE of ``equal`` and at most 0.95 times the RMSE of ``ot_uniform``.
    3. ``n_cases`` is at least 10.

    Parameters
    ----------
    importance_diagnostics : mapping
        Diagnostics of the importance model.  Only the entry ``usable`` is read;
        it must be a boolean.
    loco_summary : pandas.DataFrame
        Leave-one-economy-out summary indexed by method name, with a column
        ``rmse``.  The rows ``ot_weighted``, ``equal`` and ``ot_uniform`` must
        be present with a finite positive RMSE; other rows are ignored.
    n_cases : int
        Number of cases; a non-negative integer.
    rules : mapping, optional
        Overrides for the thresholds: ``rmse_ratio_equal`` (default 0.95) and
        ``rmse_ratio_uniform`` (default 0.95), both in (0, 1.5], and
        ``min_cases`` (default 10), a non-negative number.

    Returns
    -------
    RouteDecision
        The primary route, the three criteria in ``passed``
        (``importance_usable``, ``loco_rmse``, ``enough_cases``) and a text
        that reports each criterion with its numbers.

    Raises
    ------
    ValueError
        If ``importance_diagnostics`` has no boolean entry ``usable``, the
        summary lacks one of the three methods or holds an RMSE that is not
        finite and positive, ``n_cases`` is not a non-negative integer, or a
        rule name or value is invalid.
    """
    thresholds = _resolve_rules(rules)
    usable = _usable_flag(importance_diagnostics)
    rmse = _loco_rmse(loco_summary)
    n_cases = _check_n_cases(n_cases)
    r_weighted, r_equal, r_uniform = (rmse[method] for method in _ROUTE_METHODS)
    ok_equal = r_weighted <= thresholds["rmse_ratio_equal"] * r_equal
    ok_uniform = r_weighted <= thresholds["rmse_ratio_uniform"] * r_uniform
    rmse_ok = bool(ok_equal and ok_uniform)
    enough = bool(n_cases >= thresholds["min_cases"])
    passed = {"importance_usable": usable, "loco_rmse": rmse_ok, "enough_cases": enough}
    primary = "ot_importance" if all(passed.values()) else "ambient"

    parts = [
        "The importance diagnostics report that the importance model is usable."
        if usable
        else "The importance diagnostics do not report a usable importance model.",
        f"The leave-one-economy-out RMSE of ot_weighted is {r_weighted:.4g}, which is "
        f"{r_weighted / r_equal:.3f} times the RMSE of equal ({r_equal:.4g}) and "
        f"{r_weighted / r_uniform:.3f} times the RMSE of ot_uniform ({r_uniform:.4g}); "
        f"the limits are {thresholds['rmse_ratio_equal']:g} and "
        f"{thresholds['rmse_ratio_uniform']:g} times, so the RMSE criterion is "
        f"{'met' if rmse_ok else 'not met'}.",
        f"There are {n_cases} cases and at least {thresholds['min_cases']:g} are required, so the "
        f"case-count criterion is {'met' if enough else 'not met'}.",
        "All three criteria are met, so the primary route is ot_importance."
        if primary == "ot_importance"
        else "At least one criterion is not met, so the primary route is ambient.",
    ]
    return RouteDecision(primary=primary, passed=passed, details=" ".join(parts))


def route_comparison(tables: Mapping[str, pd.DataFrame], decision: RouteDecision) -> pd.DataFrame:
    """Stack the impact tables of several routes and flag the primary route.

    Parameters
    ----------
    tables : mapping of str to pandas.DataFrame
        Route name (for example ``ot_importance`` or ``ambient``) to the output
        of :func:`impact_table` for that route.
    decision : RouteDecision
        Output of :func:`decide_route`.

    Returns
    -------
    pandas.DataFrame
        The rows of all tables, in the order of ``tables``, with ``route`` set
        to the mapping key and a boolean column ``primary`` that is true for the
        rows of the route named in ``decision.primary``.
    """
    if len(tables) == 0:
        raise ValueError("tables must contain at least one route.")
    if decision.primary not in tables:
        raise ValueError(f"tables has no entry for the primary route {decision.primary!r}.")
    frames = []
    for name, table in tables.items():
        frame = table.copy()
        frame["route"] = name
        frame["primary"] = bool(name == decision.primary)
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)
    others = [c for c in stacked.columns if c not in ("route", "primary")]
    return stacked[["route", *others, "primary"]]
