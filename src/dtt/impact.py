"""Thailand impact scaling and the rule that selects the primary route.

Functions
---------
load_thailand_baseline
    Read the Thailand baseline series into one row per year.
to_panel_basis
    Put the baseline receipts on the basis of the panel series that measures
    the effects.
scale_effect
    Convert an effect estimate into US$ and THB billion per year.
impact_table
    Quantile table of the scaled effect for chosen baseline years.
source_receipts_share, rescale_null_pp
    Receipts share of the sources behind the target, and the placebo null in
    percentage points of GDP rescaled to the receipts share of the target.
minimum_detectable_effect, detection_probability
    Detection threshold and detection probability from null draws.
decide_route
    Apply the decision rule that picks the primary route.
select_ambient
    Select the ambient estimator from leave-one-economy-out RMSE values.
route_comparison
    Stack the impact tables of several routes and flag the primary one.

Scalings and their nulls
------------------------
An effect is converted to US$ in one of two ways.  The absolute scaling takes an
effect in percentage points of GDP and multiplies it by the GDP of the baseline
year.  The relative scaling takes an effect in percent of the counterfactual
receipts and multiplies it by the receipts of the baseline year.  Each scaling
has its own placebo null for the detection limits.

* The null of the relative scaling consists of placebo effects in percent of
  receipts.  It is free of the level of receipts and is used as it is.
* The null of the absolute scaling consists of placebo effects in percentage
  points of GDP.  It is a mixture over the source economies, and the noise of a
  placebo effect in percentage points of GDP is proportional to the level of
  receipts relative to GDP in the economy that produced it.  It is multiplied by
  the receipts share of GDP of the target divided by the usage-weighted receipts
  share of the sources (:func:`source_receipts_share` and
  :func:`rescale_null_pp`) before it enters the detection limits.

The effects are measured on the receipts series of the panel, which covers
travel and passenger transport.  The Thailand baseline holds the balance of
payments travel credits.  :func:`to_panel_basis` puts the baseline receipts and
the receipts share of GDP on the basis of the panel, so that the relative
scaling applies a percent of receipts to the base in which it was measured and
the receipts share of the target is on the basis of the cases.
"""
from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    "to_panel_basis",
    "scale_effect",
    "impact_table",
    "source_receipts_share",
    "rescale_null_pp",
    "minimum_detectable_effect",
    "detection_probability",
    "RouteDecision",
    "decide_route",
    "select_ambient",
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

# ``min_cases`` is 20 because the importance verdict has no power to separate a
# usable model from an unusable one below about 20 episodes.
DEFAULT_ROUTE_RULES = MappingProxyType(
    {"rmse_ratio_equal": 0.95, "rmse_ratio_uniform": 0.95, "min_cases": 20}
)

_DEFAULT_BASELINE_YEARS = (2019, 2024)
_ROUTE_METHODS = ("ot_weighted", "equal", "ot_uniform")
_NEIGHBOUR_METHODS = ("nn1", "nn3")
_MAX_RMSE_RATIO = 1.5
# Relative tolerance of every comparison of two RMSE values, so that a ratio that
# equals its limit in exact arithmetic is not decided by rounding noise.
_RATIO_TOL = 1e-12
_BASIS_COLUMNS = ("receipts_usd_bn_baseline", "receipts_basis", "basis_ratio")


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


def _panel_receipts(panel_receipts_usd_bn: Any) -> pd.Series:
    """Panel receipts as a float series indexed by whole-number year.

    Parameters
    ----------
    panel_receipts_usd_bn : Series or mapping
        Receipts in US$ billion by year; missing values are dropped.

    Returns
    -------
    pandas.Series
        The receipts that the panel has, indexed by integer year (named
        ``year``), with finite positive values.

    Raises
    ------
    ValueError
        If the input is not a Series or mapping, a year is not a whole number
        or occurs twice, or a value is not finite and positive.
    """
    if isinstance(panel_receipts_usd_bn, Mapping):
        panel_receipts_usd_bn = pd.Series(dict(panel_receipts_usd_bn), dtype=float)
    if not isinstance(panel_receipts_usd_bn, pd.Series):
        raise ValueError("panel_receipts_usd_bn must be a Series indexed by year.")
    labels = pd.to_numeric(pd.Series(panel_receipts_usd_bn.index), errors="coerce")
    labels = labels.to_numpy(dtype=float)
    if not bool(np.all(np.isfinite(labels) & (labels == np.floor(labels)))):
        raise ValueError("The index of panel_receipts_usd_bn must hold whole-number years.")
    panel = pd.Series(
        panel_receipts_usd_bn.to_numpy(dtype=float),
        index=pd.Index(labels.astype(int), name="year"),
        dtype=float,
    )
    if not panel.index.is_unique:
        raise ValueError("panel_receipts_usd_bn has more than one value for a year.")
    panel = panel.dropna()
    if not bool(np.all(np.isfinite(panel.to_numpy()) & (panel.to_numpy() > 0.0))):
        raise ValueError("panel_receipts_usd_bn must be finite and positive where present.")
    return panel


def to_panel_basis(
    baseline: pd.DataFrame,
    panel_receipts_usd_bn: pd.Series,
    ratio_years: Sequence[int] = range(2015, 2020),
) -> pd.DataFrame:
    """Put the baseline receipts on the basis of the panel series.

    The effects are measured on the receipts series of the panel, which covers
    travel and passenger transport, whereas the baseline holds the balance of
    payments travel credits and is smaller.  For a year in which the panel has a
    value, the panel value replaces the baseline receipts.  For any other year
    the baseline receipts are multiplied by the ratio of panel receipts to
    baseline receipts, the mean of the yearly ratios over ``ratio_years`` in
    which both exist.  The receipts share of GDP is recomputed from the new
    receipts and the unchanged GDP, as ``100 * receipts / GDP``.

    Parameters
    ----------
    baseline : pandas.DataFrame
        Output of :func:`load_thailand_baseline`, indexed by year, or a table
        with a ``year`` column.  The columns ``gdp_usd_bn`` and
        ``receipts_usd_bn`` are required; values that are present must be finite
        and positive.
    panel_receipts_usd_bn : pandas.Series
        Receipts of the target economy in the panel, in US$ billion, indexed by
        year.  Missing values are treated as years without a panel value, and
        years outside the baseline are ignored.
    ratio_years : sequence of int, default range(2015, 2020)
        Years whose ratios of panel to baseline receipts are averaged.

    Returns
    -------
    pandas.DataFrame
        A copy of ``baseline`` in which ``receipts_usd_bn`` and
        ``receipts_pct_gdp`` are on the panel basis.  Added columns:
        ``receipts_usd_bn_baseline`` (the original receipts),
        ``receipts_basis`` (``panel`` where the panel value was used and
        ``ratio`` where the ratio was applied) and ``basis_ratio`` (the ratio in
        use, NaN where the panel value was used).  Every other column and the
        index are kept.

    Raises
    ------
    ValueError
        If a required column is absent, the baseline already has one of the added
        columns, a value or year is invalid, or no year of ``ratio_years`` has
        both a panel and a baseline value.
    """
    table = _baseline_by_year(baseline)
    lacking = [c for c in ("gdp_usd_bn", "receipts_usd_bn") if c not in table.columns]
    if lacking:
        raise ValueError(f"The baseline table has no column named {lacking}.")
    present = [c for c in _BASIS_COLUMNS if c in baseline.columns]
    if present:
        raise ValueError(f"The baseline table already has the columns {present}.")
    receipts = table["receipts_usd_bn"].to_numpy(dtype=float)
    gdp = table["gdp_usd_bn"].to_numpy(dtype=float)
    for name, values in (("receipts_usd_bn", receipts), ("gdp_usd_bn", gdp)):
        known = values[~np.isnan(values)]
        if not bool(np.all(np.isfinite(known) & (known > 0.0))):
            raise ValueError(f"The baseline values of {name} must be finite and positive.")
    panel = _panel_receipts(panel_receipts_usd_bn)

    years = table.index.to_numpy()
    yearly = []
    for year in sorted(set(_as_years(ratio_years))):
        rows = np.flatnonzero(years == year)
        if rows.size and year in panel.index and not np.isnan(receipts[rows[0]]):
            yearly.append(float(panel.loc[year]) / float(receipts[rows[0]]))
    if not yearly:
        raise ValueError(
            "No year of ratio_years has both a panel value and a baseline value of the receipts."
        )
    ratio = float(np.mean(yearly))

    in_panel = np.isin(years, panel.index.to_numpy())
    panel_values = panel.reindex(pd.Index(years)).to_numpy(dtype=float)
    new_receipts = np.where(in_panel, panel_values, receipts * ratio)
    out = baseline.copy()
    out["receipts_usd_bn"] = new_receipts
    out["receipts_pct_gdp"] = 100.0 * new_receipts / gdp
    out["receipts_usd_bn_baseline"] = receipts
    out["receipts_basis"] = ["panel" if flag else "ratio" for flag in in_panel]
    out["basis_ratio"] = np.where(in_panel, np.nan, ratio)
    return out


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
def _positive_finite(value: Any, name: str) -> float:
    """Return ``value`` as a float after checking that it is finite and positive.

    Parameters
    ----------
    value : float
        Number to check; a boolean is not accepted.
    name : str
        Name of the argument, used in error messages.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    ValueError
        If ``value`` is not a real number, or is not finite and positive.
    """
    if not _is_number(value):
        raise ValueError(f"{name} must be a number.")
    number = float(value)
    if not (np.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be finite and positive; found {number}.")
    return number


def source_receipts_share(shares: ArrayLike | pd.Series, usage: ArrayLike | pd.Series) -> float:
    """Usage-weighted mean receipts share of the sources behind the target.

    The share of a source is its tourism receipts in percent of its GDP, and the
    usage of a source is the weight that the transport puts on it (for example
    the mass of its episodes in the transported effect).  The result is
    ``sum(usage * shares) / sum(usage)``.  Sources with zero usage do not enter,
    so their shares may be missing.  The usages are used as given: their sum and
    their products with the shares must stay below the largest float.

    Parameters
    ----------
    shares : array_like or Series, shape (m,)
        Receipts share of GDP of each source, in percent.  A share used in the
        mean must be finite and positive.
    usage : array_like or Series, shape (m,)
        Non-negative weight of each source, with a positive sum.  When both
        ``shares`` and ``usage`` are Series they are matched by label and must
        name the same sources; otherwise they are matched by position.

    Returns
    -------
    float
        The usage-weighted mean receipts share, in percent of GDP.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional and of equal length, a usage is
        negative or not finite, the usages sum to zero or to a value that is not
        finite, a share with positive usage is not finite and positive, two
        Series name different sources, or the weighted mean is not finite.
    """
    if isinstance(shares, pd.Series) and isinstance(usage, pd.Series):
        if not (shares.index.is_unique and usage.index.is_unique):
            raise ValueError("shares and usage must have one entry per source.")
        if set(shares.index) != set(usage.index):
            raise ValueError("shares and usage must name the same sources.")
        usage = usage.reindex(shares.index)
    s = np.asarray(shares, dtype=float)
    u = np.asarray(usage, dtype=float)
    if s.ndim != 1 or u.ndim != 1 or s.size == 0:
        raise ValueError("shares and usage must be non-empty one-dimensional arrays.")
    if s.shape != u.shape:
        raise ValueError("shares and usage must have the same length.")
    if not bool(np.all(np.isfinite(u) & (u >= 0.0))):
        raise ValueError("usage must be finite and non-negative.")
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(u))
    if not total > 0.0:
        raise ValueError("usage must have a positive sum.")
    if not np.isfinite(total):
        raise ValueError("usage must have a finite sum.")
    used = u > 0.0
    if not bool(np.all(np.isfinite(s[used]) & (s[used] > 0.0))):
        raise ValueError("shares of the sources in use must be finite and positive.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.sum(u[used] * s[used]) / total)
    if not np.isfinite(mean):
        raise ValueError(
            "The usage-weighted mean of the shares is not finite: usage times share overflows."
        )
    return mean


def rescale_null_pp(
    null_pp: ArrayLike, source_receipts_share: float, target_receipts_share: float
) -> tuple[np.ndarray, float]:
    """Put a placebo null in percentage points of GDP on the scale of the target.

    A placebo effect in percentage points of GDP is a fraction of the receipts
    of the economy that produced it, so its noise is proportional to the receipts
    share of GDP of that economy.  The null of the transported effect is a
    mixture over the source economies; multiplying its draws by the receipts
    share of the target over the receipts share of the sources gives the null
    that a placebo run on the target would have.  The result is the tuple
    ``(draws, factor)``.

    Parameters
    ----------
    null_pp : array_like, shape (n,)
        Draws of the placebo effect in percentage points of GDP.
    source_receipts_share : float
        Receipts share of GDP of the sources, in percent; see
        :func:`source_receipts_share`.  Positive and finite.
    target_receipts_share : float
        Receipts share of GDP of the target, in percent, on the same basis as
        the sources.  Positive and finite.

    Returns
    -------
    tuple of (ndarray, float)
        ``draws``, the draws multiplied by the factor as a new array of shape
        (n,), and ``factor``, ``target_receipts_share / source_receipts_share``.

    Raises
    ------
    ValueError
        If the draws are empty or not finite, a share is not a finite positive
        number, the factor overflows or underflows to a value that is not finite
        and positive, or a rescaled draw is not finite.
    """
    draws = _as_draws(null_pp, "null_pp")
    source = _positive_finite(source_receipts_share, "source_receipts_share")
    target = _positive_finite(target_receipts_share, "target_receipts_share")
    factor = target / source
    if not (np.isfinite(factor) and factor > 0.0):
        raise ValueError(
            "The factor target_receipts_share / source_receipts_share must be finite and "
            f"positive; found {factor}."
        )
    with np.errstate(over="ignore"):
        rescaled = draws * factor
    if not bool(np.all(np.isfinite(rescaled))):
        raise ValueError(
            "The rescaled null is not finite: the draws times the factor "
            f"{factor:g} overflow."
        )
    return rescaled, float(factor)


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
        Criterion name to whether the criterion is met, in this order:
        ``importance_usable``, ``loco_rmse``, ``enough_cases`` and
        ``target_supported``.
    details : str
        Plain-language account of each criterion, of the comparisons that are
        only reported, and of the decision.
    info : dict, optional
        Comparisons that are reported and not used by the rule; empty when none
        was computed.  ``rmse_ratio_to_best_ambient`` and ``best_ambient``
        (the ratio of the ``ot_weighted`` RMSE to the smallest ambient RMSE, and
        the name of that estimator) are present when ambient RMSE values were
        given.  ``rmse_ratio_to_best_neighbour`` and ``best_neighbour`` (the
        same against the smaller of the ``nn1`` and ``nn3`` RMSE) are present
        when a neighbour RMSE was given or found in the summary.  Such a ratio
        compares like with like only when both RMSE values are computed on the
        same economies.  ``n_ot`` and ``n_ambient`` (the numbers of cases behind
        the RMSE of ``ot_weighted`` and behind the ambient RMSE values) are
        present when they were given.
    """

    primary: str
    passed: dict[str, bool]
    details: str
    info: dict[str, Any] = field(default_factory=dict)


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


def _rmse_entries(entries: Any, name: str, allow_empty: bool = False) -> dict[str, float]:
    """Read a mapping of estimator names to RMSE values.

    Parameters
    ----------
    entries : mapping or Series
        Estimator name to leave-one-economy-out RMSE; a Series must have one
        entry per label.
    name : str
        Name of the argument, used in error messages.
    allow_empty : bool, default False
        Whether an empty mapping is accepted.

    Returns
    -------
    dict
        The RMSE of each estimator as a float, in the order of ``entries``.

    Raises
    ------
    ValueError
        If ``entries`` is not a mapping or Series, is a Series whose index
        repeats a label, is empty and ``allow_empty`` is false, a name is not a
        string, or an RMSE is not a finite positive number.
    """
    if not isinstance(entries, (Mapping, pd.Series)):
        raise ValueError(f"{name} must be a mapping from estimator names to RMSE values.")
    if isinstance(entries, pd.Series) and not entries.index.is_unique:
        repeated = sorted({str(label) for label in entries.index[entries.index.duplicated()]})
        raise ValueError(
            f"{name} must have one RMSE per estimator; the labels {repeated} occur more than once."
        )
    out: dict[str, float] = {}
    for key, value in entries.items():
        if not isinstance(key, str):
            raise ValueError(f"The names in {name} must be strings.")
        if not _is_number(value):
            raise ValueError(f"The rmse of {key} in {name} must be a number.")
        number = float(value)
        if not (np.isfinite(number) and number > 0.0):
            raise ValueError(
                f"The rmse of {key} in {name} must be finite and positive; found {number}."
            )
        out[key] = number
    if not out and not allow_empty:
        raise ValueError(f"{name} must contain at least one estimator.")
    return out


def _reported_rmse(
    loco_summary: pd.DataFrame, ambient_rmse: Any
) -> tuple[dict[str, float], dict[str, float], set[str]]:
    """RMSE values that are reported next to the rule and not used by it.

    Parameters
    ----------
    loco_summary : pandas.DataFrame
        Validated summary indexed by method name with a column ``rmse``.  The
        rows ``nn1`` and ``nn3`` are read when present.
    ambient_rmse : mapping, Series or None
        Estimator name to RMSE.  The entries ``nn1`` and ``nn3`` are neighbour
        predictors and replace the rows of the summary; every other entry is an
        ambient estimator.

    Returns
    -------
    ambient : dict
        RMSE of the ambient estimators, in the given order.
    neighbours : dict
        RMSE of the neighbour predictors that are available.
    replaced : set of str
        Names of the neighbour predictors whose RMSE comes from ``ambient_rmse``
        and not from the summary.

    Raises
    ------
    ValueError
        If a neighbour row of the summary or an entry of ``ambient_rmse`` is not
        a finite positive number.
    """
    neighbours: dict[str, float] = {}
    for method in _NEIGHBOUR_METHODS:
        if method in loco_summary.index:
            value = loco_summary.loc[method, "rmse"]
            if not _is_number(value):
                raise ValueError(f"The rmse of {method} must be a number.")
            number = float(value)
            if not (np.isfinite(number) and number > 0.0):
                raise ValueError(
                    f"The rmse of {method} must be finite and positive; found {number}."
                )
            neighbours[method] = number
    ambient: dict[str, float] = {}
    replaced: set[str] = set()
    if ambient_rmse is not None:
        for key, number in _rmse_entries(ambient_rmse, "ambient_rmse", allow_empty=True).items():
            if key in _NEIGHBOUR_METHODS:
                neighbours[key] = number
                replaced.add(key)
            else:
                ambient[key] = number
    return ambient, neighbours, replaced


def _smallest(entries: Mapping[str, float]) -> str:
    """Name of the entry with the smallest value; the first of equal values.

    Parameters
    ----------
    entries : mapping
        Name to value; not empty.

    Returns
    -------
    str
        The name of the smallest value.  Values that differ by less than the
        relative tolerance of the RMSE comparisons count as equal.
    """
    best = next(iter(entries))
    for name, value in entries.items():
        if value < entries[best] * (1.0 - _RATIO_TOL):
            best = name
    return best


def _check_support(support: Any) -> tuple[bool, float | None, list[str]]:
    """Read the target support result.

    Parameters
    ----------
    support : mapping or Series
        Result of the support check of the transport, with the boolean entry
        ``supported`` and, optionally, ``ess`` (a number) and ``reasons`` (a
        string or a sequence of strings).

    Returns
    -------
    supported : bool
        The value of ``supported``.
    ess : float or None
        The effective number of sources, or None when not given.
    reasons : list of str
        The reasons, empty when none were given.

    Raises
    ------
    ValueError
        If ``support`` is missing, is not a mapping or Series with a boolean
        entry ``supported``, or holds an ``ess`` or ``reasons`` entry of another
        type.
    """
    if support is None:
        raise ValueError("support is required: pass the result of transport.target_support")
    if not isinstance(support, (Mapping, pd.Series)):
        raise ValueError("support must be a mapping with a boolean entry named 'supported'.")
    if "supported" not in support:
        raise ValueError("support must be a mapping with a boolean entry named 'supported'.")
    flag = support["supported"]
    if not isinstance(flag, (bool, np.bool_)):
        raise ValueError("support['supported'] must be a boolean.")
    ess = None
    if "ess" in support and support["ess"] is not None:
        if not _is_number(support["ess"]):
            raise ValueError("support['ess'] must be a number.")
        ess = float(support["ess"])
    reasons: list[str] = []
    if "reasons" in support and support["reasons"] is not None:
        raw = support["reasons"]
        items = [raw] if isinstance(raw, str) else raw
        if isinstance(items, (Mapping, bytes)) or not isinstance(items, Sequence):
            raise ValueError("support['reasons'] must be a string or a sequence of strings.")
        if not all(isinstance(item, str) for item in items):
            raise ValueError("support['reasons'] must be a string or a sequence of strings.")
        reasons = [item.strip() for item in items if item.strip()]
    return bool(flag), ess, reasons


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


def _check_sample_size(value: Any, name: str) -> int | None:
    """Validate an optional sample size.

    Parameters
    ----------
    value : int or None
        Number of cases behind an RMSE, or None when not given.
    name : str
        Name of the argument, used in the error message.

    Returns
    -------
    int or None
        The sample size, or None when ``value`` is None.

    Raises
    ------
    ValueError
        If ``value`` is not None and not a positive integer (a boolean is not).
    """
    if value is None:
        return None
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _cases(count: int) -> str:
    """Phrase a number of cases.

    Parameters
    ----------
    count : int
        Number of cases.

    Returns
    -------
    str
        The number followed by ``case`` or ``cases``.
    """
    return f"{count} case" if count == 1 else f"{count} cases"


def decide_route(
    importance_diagnostics: Mapping[str, Any],
    loco_summary: pd.DataFrame,
    n_cases: int,
    rules: Mapping[str, float] | None = None,
    support: Mapping[str, Any] | None = None,
    ambient_rmse: Mapping[str, float] | None = None,
    *,
    n_ot: int | None = None,
    n_ambient: int | None = None,
) -> RouteDecision:
    """Choose the primary route for the Thailand estimate.

    The primary route is ``ot_importance`` if and only if all four criteria
    hold, and ``ambient`` otherwise.  The thresholds are those of ``rules`` and
    default to the values below:

    1. ``importance_diagnostics["usable"]`` is true.
    2. The leave-one-economy-out RMSE of ``ot_weighted`` is at most 0.95 times
       the RMSE of ``equal`` and at most 0.95 times the RMSE of ``ot_uniform``.
    3. ``n_cases`` is at least 20.  The importance verdict has no power to tell
       a usable model from an unusable one below about 20 episodes.
    4. ``support["supported"]`` is true: the covariates of the target lie within
       the reach of the sources of the transport.  A barycentric transport
       cannot exceed the largest effect among its sources, so for a target
       outside that reach the transported effect is bounded by the data however
       far out the target lies.

    ``n_cases`` counts episodes.  The independent units are the economies: the
    validation of criterion 2 leaves out whole economies, so a sample of many
    episodes from few economies carries less evidence than its episode count
    suggests.

    The RMSE of the ambient estimators and of the nearest-neighbour predictors
    ``nn1`` and ``nn3`` is not a criterion.  It is compared with the RMSE of
    ``ot_weighted`` and the comparison is stored in ``RouteDecision.info`` and
    stated in the text of the decision, which says that it is reported and not
    used by the rule.

    A ratio of two RMSE values compares like with like only when both are
    computed on the same economies.  The transport and the ambient estimators
    can be validated on different samples, for example when the ambient
    estimators need complete covariates, and then the ratio mixes the difficulty
    of the sample with the quality of the estimator.  The text of the decision
    says so, and ``n_ot`` and ``n_ambient`` record the two sample sizes when they
    are given.  The rows ``nn1`` and ``nn3`` of ``loco_summary`` come from the
    validation of ``ot_weighted``; entries of the same names in ``ambient_rmse``
    replace them and carry the same condition.

    Parameters
    ----------
    importance_diagnostics : mapping
        Diagnostics of the importance model.  Only the entry ``usable`` is read;
        it must be a boolean.
    loco_summary : pandas.DataFrame
        Leave-one-economy-out summary indexed by method name, with a column
        ``rmse``.  The rows ``ot_weighted``, ``equal`` and ``ot_uniform`` must
        be present with a finite positive RMSE.  The rows ``nn1`` and ``nn3`` are
        read when present, with the same requirement; other rows are ignored.
    n_cases : int
        Number of episodes; a non-negative integer.
    rules : mapping, optional
        Overrides for the thresholds: ``rmse_ratio_equal`` (default 0.95) and
        ``rmse_ratio_uniform`` (default 0.95), both in (0, 1.5], and
        ``min_cases`` (default 20), a non-negative number.
    support : mapping
        Result of ``dtt.transport.target_support``: the boolean ``supported``
        and, optionally, ``ess`` (the effective number of sources, a number) and
        ``reasons`` (a string or a list of strings).  Required; there is no
        default that lets the criterion pass.  A Series is accepted as a mapping.
    ambient_rmse : mapping, optional
        Estimator name to leave-one-economy-out RMSE of the ambient estimators,
        for example ``{"A1": 0.35, "A2": 0.34, "A3": 0.32}``.  The entries
        ``nn1`` and ``nn3`` are read as neighbour predictors and replace the rows
        of ``loco_summary`` with these names.  A Series must have one entry per
        label.  Not a criterion.
    n_ot : int, optional
        Number of cases (episodes) that the RMSE of ``ot_weighted`` is computed
        on; a positive integer.  Keyword only.
    n_ambient : int, optional
        Number of cases (episodes) that the RMSE values in ``ambient_rmse`` are
        computed on; a positive integer.  Keyword only.

    Returns
    -------
    RouteDecision
        The primary route, the four criteria in ``passed``
        (``importance_usable``, ``loco_rmse``, ``enough_cases``,
        ``target_supported``), a text that reports each criterion with its
        numbers, and ``info`` with ``rmse_ratio_to_best_ambient``,
        ``best_ambient``, ``rmse_ratio_to_best_neighbour`` and
        ``best_neighbour`` for the comparisons that could be made, followed by
        ``n_ot`` and ``n_ambient`` for the sample sizes that were given.

    Raises
    ------
    ValueError
        If ``support`` is missing or has no boolean entry ``supported``,
        ``importance_diagnostics`` has no boolean entry ``usable``, the summary
        lacks one of the three methods or holds an RMSE that is not finite and
        positive, ``ambient_rmse`` is not a mapping of finite positive RMSE
        values or is a Series with a repeated label, ``n_cases`` is not a
        non-negative integer, ``n_ot`` or ``n_ambient`` is not a positive
        integer, or a rule name or value is invalid.
    """
    thresholds = _resolve_rules(rules)
    usable = _usable_flag(importance_diagnostics)
    rmse = _loco_rmse(loco_summary)
    n_cases = _check_n_cases(n_cases)
    supported, ess, reasons = _check_support(support)
    ambient, neighbours, replaced = _reported_rmse(loco_summary, ambient_rmse)
    n_ot = _check_sample_size(n_ot, "n_ot")
    n_ambient = _check_sample_size(n_ambient, "n_ambient")
    r_weighted, r_equal, r_uniform = (rmse[method] for method in _ROUTE_METHODS)
    slack = 1.0 + _RATIO_TOL
    ok_equal = r_weighted <= thresholds["rmse_ratio_equal"] * r_equal * slack
    ok_uniform = r_weighted <= thresholds["rmse_ratio_uniform"] * r_uniform * slack
    rmse_ok = bool(ok_equal and ok_uniform)
    enough = bool(n_cases >= thresholds["min_cases"])
    passed = {
        "importance_usable": usable,
        "loco_rmse": rmse_ok,
        "enough_cases": enough,
        "target_supported": supported,
    }
    primary = "ot_importance" if all(passed.values()) else "ambient"

    info: dict[str, Any] = {}
    reported = []
    same_economies = (
        "The ratio compares like with like only when both RMSE values are computed on the same "
        "economies."
    )
    if ambient:
        best = _smallest(ambient)
        info["rmse_ratio_to_best_ambient"] = float(r_weighted / ambient[best])
        info["best_ambient"] = best
        reported.append(
            f"The leave-one-economy-out RMSE of ot_weighted is {r_weighted / ambient[best]:.3f} "
            f"times the RMSE of the best ambient estimator {best} ({ambient[best]:.4g}); this is "
            f"reported and is not used by the rule. {same_economies}"
        )
    if neighbours:
        best = _smallest(neighbours)
        info["rmse_ratio_to_best_neighbour"] = float(r_weighted / neighbours[best])
        info["best_neighbour"] = best
        reported.append(
            f"The leave-one-economy-out RMSE of ot_weighted is {r_weighted / neighbours[best]:.3f} "
            f"times the RMSE of the best neighbour predictor {best} ({neighbours[best]:.4g}); this "
            "is reported and is not used by the rule."
            + (f" {same_economies}" if best in replaced else "")
        )
    if n_ot is not None:
        info["n_ot"] = n_ot
    if n_ambient is not None:
        info["n_ambient"] = n_ambient
    if n_ot is not None and n_ambient is not None:
        differ = (
            ", so the two are not computed on the same cases." if n_ot != n_ambient else "."
        )
        reported.append(
            f"The RMSE of ot_weighted is computed on {_cases(n_ot)} and the ambient RMSE values "
            f"on {_cases(n_ambient)}{differ}"
        )
    elif n_ot is not None:
        reported.append(f"The RMSE of ot_weighted is computed on {_cases(n_ot)}.")
    elif n_ambient is not None:
        reported.append(f"The ambient RMSE values are computed on {_cases(n_ambient)}.")

    ess_text = "" if ess is None else f" (effective number of sources {ess:.3g})"
    if supported:
        support_text = (
            f"The target support check reports that the target is supported{ess_text}, so the "
            "support criterion is met."
        )
    else:
        why = "; ".join(reason.rstrip(". ") for reason in reasons)
        support_text = (
            f"The target support check does not report that the target is supported{ess_text}"
            + (f": {why}" if why else "")
            + ", so the support criterion is not met."
        )
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
        support_text,
        *reported,
        "All four criteria are met, so the primary route is ot_importance."
        if primary == "ot_importance"
        else "At least one criterion is not met, so the primary route is ambient.",
    ]
    return RouteDecision(primary=primary, passed=passed, details=" ".join(parts), info=info)


def select_ambient(
    rmse: Mapping[str, float],
    baseline: str = "A1",
    ratio: float = 0.95,
    *,
    close_within: float = 0.05,
) -> dict[str, Any]:
    """Select the ambient estimator from leave-one-economy-out RMSE values.

    Every estimator other than ``baseline`` whose RMSE is at most ``ratio`` times
    the RMSE of ``baseline`` qualifies.  The qualifying estimator with the
    smallest RMSE is selected, the first one listed when RMSE values are equal,
    and ``baseline`` is selected when none qualifies.  Comparisons use a relative
    tolerance of ``1e-12``, so a ratio that equals its limit in exact arithmetic
    qualifies and RMSE values that differ only by rounding count as equal.

    The result also reports how firm the selection is.  The runner up is the
    estimator with the smallest RMSE among those not selected, and the margin
    is its RMSE relative to the selected RMSE minus one.  The selection is
    close when the margin is within ``close_within`` in either direction, or when
    the RMSE ratio of any estimator other than ``baseline`` lies within
    ``close_within`` of the limit ``ratio`` in relative terms, so that a small
    change in the RMSE values could change which estimator qualifies.  A distance
    that equals ``close_within`` in exact arithmetic counts as within it: the
    comparison allows the same tolerance of ``1e-12``, added to ``close_within``.

    Parameters
    ----------
    rmse : mapping or Series
        Estimator name to leave-one-economy-out RMSE; finite and positive.  The
        order of the entries decides ties.  A Series must have one entry per
        label.
    baseline : str, default "A1"
        Name of the estimator that the others have to beat; it must be in
        ``rmse``.
    ratio : float, default 0.95
        Largest RMSE ratio to the baseline at which an estimator qualifies; in
        (0, 1.5].
    close_within : float, default 0.05
        Relative distance that makes a selection close; non-negative.

    Returns
    -------
    dict
        ``selected``
            Name of the selected estimator.
        ``qualifying``
            Names of the qualifying estimators, in the order of ``rmse``.
        ``ratios``
            Each estimator's RMSE over the RMSE of ``baseline``.
        ``margin``
            ``rmse[runner_up] / rmse[selected] - 1``: positive when the selected
            estimator has the smaller RMSE, negative when the runner up has the
            smaller RMSE but does not qualify, and 0 when ``rmse`` holds only
            the baseline.
        ``runner_up``
            Name of the best other estimator, or None when ``rmse`` holds only
            the baseline.
        ``close``
            True when the selection is close in the sense above.

    Raises
    ------
    ValueError
        If ``rmse`` is not a mapping of finite positive numbers or is a Series
        with a repeated label, ``baseline`` is not one of its names, or
        ``ratio`` or ``close_within`` is out of range.
    """
    values = _rmse_entries(rmse, "rmse")
    if not isinstance(baseline, str) or baseline not in values:
        raise ValueError(f"baseline {baseline!r} is not one of the estimators in rmse.")
    limit = _positive_finite(ratio, "ratio")
    if limit > _MAX_RMSE_RATIO:
        raise ValueError(f"ratio must lie in (0, {_MAX_RMSE_RATIO:g}].")
    if not _is_number(close_within) or not (
        np.isfinite(close_within) and float(close_within) >= 0.0
    ):
        raise ValueError("close_within must be a non-negative finite number.")
    near = float(close_within) + _RATIO_TOL

    reference = values[baseline]
    ratios = {name: value / reference for name, value in values.items()}
    qualifying = [
        name for name in values if name != baseline and ratios[name] <= limit * (1.0 + _RATIO_TOL)
    ]
    selected = _smallest({name: values[name] for name in qualifying}) if qualifying else baseline
    others = {name: value for name, value in values.items() if name != selected}
    if others:
        runner_up: str | None = _smallest(others)
        margin = float(values[runner_up] / values[selected] - 1.0)
    else:
        runner_up, margin = None, 0.0
    near_runner_up = runner_up is not None and abs(margin) <= near
    near_limit = any(
        abs(ratios[name] / limit - 1.0) <= near for name in values if name != baseline
    )
    return {
        "selected": selected,
        "qualifying": qualifying,
        "ratios": ratios,
        "margin": margin,
        "runner_up": runner_up,
        "close": bool(near_runner_up or near_limit),
    }


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
