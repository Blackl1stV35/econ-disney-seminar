"""Thailand impact scaling and the rule that selects the primary route.

Functions
---------
load_thailand_baseline
    Read the Thailand baseline series into one row per year.
to_panel_basis
    Express the baseline receipts on the basis of the receipts series of the panel.
scale_effect
    Convert an effect estimate into US$ and THB billion per year.
impact_table
    Quantile table of the scaled effect for chosen baseline years.
minimum_detectable_effect, detection_probability
    Detection threshold and detection probability from null draws.
source_receipts_share, rescale_null_pp
    Move a null in percentage points of GDP to the receipts share of the target.
decide_route
    Apply the decision rule that picks the primary route.
select_ambient
    Select the ambient estimator by its leave-one-economy-out RMSE and report how firm it is.
route_comparison
    Stack the impact tables of several routes and flag the primary one.

Notes
-----
Scale of the null.  An observed effect is compared with a null distribution
that must be on the scale of the effect.  The null of the relative scaling, in
percent of the counterfactual receipts, is scale free: a placebo effect measured
against the receipts does not depend on how large the receipts are, and it is
used as it is.  The null of the absolute scaling, in percentage points of GDP,
grows with the level of the receipts share of GDP.  The draws are therefore
multiplied by the ratio of the receipts share of the target to the receipts
share of the sources (``rescale_null_pp``, with the source share from
``source_receipts_share``) before the minimum detectable effect and the
detection probability are computed.

Basis of the receipts.  The effects of the cases are measured on the receipts
series of the panel.  The relative scaling applies a percent of receipts, and
the receipts share of GDP of the target is compared with those of the cases, so
the Thailand receipts must be on the same basis (``to_panel_basis``).

Rule of the route.  The primary route is the importance route only if the
importance model is usable, the validation favours the weighted transport, the
sample has enough episodes and the target lies within the support of the
sources.  The comparison of the weighted transport with the ambient estimators
and with the nearest-neighbour predictors is reported next to the decision and
does not enter it.
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
    "minimum_detectable_effect",
    "detection_probability",
    "source_receipts_share",
    "rescale_null_pp",
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

# The importance verdict rests on a permutation p-value and a split-half correlation, which have
# little power with fewer than about 20 episodes; the route rule therefore requires 20 episodes.
DEFAULT_ROUTE_RULES = MappingProxyType(
    {"rmse_ratio_equal": 0.95, "rmse_ratio_uniform": 0.95, "min_cases": 20}
)

_DEFAULT_BASELINE_YEARS = (2019, 2024)
_ROUTE_METHODS = ("ot_weighted", "equal", "ot_uniform")
_NEIGHBOUR_METHODS = ("nn1", "nn3")
_MAX_RMSE_RATIO = 1.5
# Relative tolerance of the comparisons between RMSE values: near ties and RMSE ratios against
# their limits.  An RMSE is a sum of squares whose last digits depend on the math kernel, so a
# comparison that holds with equality in exact arithmetic must not depend on them.
_RMSE_TOLERANCE = 1e-9
# Relative distance that makes two estimators, or an estimator and the qualification limit, close.
_CLOSE_SHARE = 0.05


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


def _panel_receipts_by_year(values: Any) -> pd.Series:
    """Index the receipts of the panel by whole-number year.

    Parameters
    ----------
    values : pandas.Series or mapping
        Receipts in US$ billion by year.  Missing values mean that the panel has
        no value for the year.

    Returns
    -------
    pandas.Series
        Float values indexed by integer year (named ``year``).

    Raises
    ------
    ValueError
        If ``values`` is not a Series or a mapping, a year is not a whole number
        or occurs twice, a value cannot be read as a number, or a value that is
        present is not finite and positive.
    """
    if isinstance(values, Mapping):
        series = pd.Series(list(values.values()), index=list(values.keys()), dtype="object")
    elif isinstance(values, pd.Series):
        series = values
    else:
        raise ValueError("panel_receipts_usd_bn must be a Series or a mapping indexed by year.")
    labels = pd.to_numeric(pd.Series(series.index), errors="coerce").to_numpy(dtype=float)
    if not bool(np.all(np.isfinite(labels) & (labels == np.floor(labels)))):
        raise ValueError("The panel receipts must be indexed by whole-number years.")
    numbers = pd.to_numeric(pd.Series(series.to_numpy()), errors="raise").to_numpy(dtype=float)
    out = pd.Series(numbers, index=pd.Index(labels.astype(int), name="year"), dtype=float)
    if not out.index.is_unique:
        raise ValueError("The panel receipts have more than one value for a year.")
    present = out.to_numpy()[~np.isnan(out.to_numpy())]
    if not bool(np.all(np.isfinite(present) & (present > 0.0))):
        raise ValueError("The panel receipts must be finite and positive where they are given.")
    return out


def to_panel_basis(
    baseline: pd.DataFrame,
    panel_receipts_usd_bn: pd.Series | Mapping[Any, float],
    ratio_years: int | Sequence[int] = range(2015, 2020),
) -> pd.DataFrame:
    """Express the baseline receipts on the basis of the receipts series of the panel.

    The effects of the cases are measured on the receipts series of the panel,
    which counts travel and passenger transport receipts.  The Thailand baseline
    holds the travel credits of the balance of payments, a smaller quantity.  The
    relative scaling applies a percent of receipts, and the receipts share of GDP
    is a feature of the target that the cases have on the basis of the panel, so
    both need the receipts on that basis.  For a year in which the panel has a
    value, that value is used.  For every other year the baseline value is
    multiplied by the ratio of the panel receipts to the baseline receipts, the
    mean of the yearly ratios over the years of ``ratio_years`` in which both
    values exist.

    Parameters
    ----------
    baseline : pandas.DataFrame
        Output of :func:`load_thailand_baseline`, indexed by year (a table with a
        ``year`` column is accepted and returned indexed by year).  The columns
        ``gdp_usd_bn`` and ``receipts_usd_bn`` are required.
    panel_receipts_usd_bn : pandas.Series or mapping
        Receipts of the target economy in the panel, in US$ billion, by year.
        Years that the panel does not have may be absent or missing; years that
        the baseline does not have are not used.
    ratio_years : int or sequence of int, default range(2015, 2020)
        Years from which the ratio of the panel to the baseline is taken.  A year
        counts only if the panel has a value and the baseline has a positive
        value for it.

    Returns
    -------
    pandas.DataFrame
        A copy of the baseline, with ``receipts_usd_bn`` on the panel basis and
        ``receipts_pct_gdp`` recomputed as 100 times those receipts over
        ``gdp_usd_bn``.  The index and all other columns are kept.  Three columns
        are added: ``receipts_usd_bn_baseline`` (the receipts of the input),
        ``receipts_basis`` (``panel`` where the panel value was used, ``ratio``
        otherwise) and ``basis_ratio`` (the ratio that was applied, NaN where the
        panel value was used).  A year that has neither a panel value nor a
        baseline value keeps a missing value; a GDP that is missing or not
        positive gives a missing receipts share.

    Raises
    ------
    ValueError
        If the baseline lacks ``gdp_usd_bn`` or ``receipts_usd_bn``, has a year
        that is not a whole number or occurs twice, or already has the column
        ``receipts_basis``; if the panel receipts are not a Series or a mapping
        indexed by whole-number years, or hold a value that is not finite and
        positive; or if no year of ``ratio_years`` has both a panel value and a
        positive baseline value.
    """
    table = _baseline_by_year(baseline)
    lacking = [c for c in ("gdp_usd_bn", "receipts_usd_bn") if c not in table.columns]
    if lacking:
        raise ValueError(f"The baseline table has no column named {lacking}.")
    if "receipts_basis" in table.columns:
        raise ValueError("The baseline already has a column named receipts_basis.")
    panel = _panel_receipts_by_year(panel_receipts_usd_bn)
    years = list(dict.fromkeys(_as_years(ratio_years)))

    original = table["receipts_usd_bn"].to_numpy(dtype=float)
    gdp = table["gdp_usd_bn"].to_numpy(dtype=float)
    on_panel = panel.reindex(table.index).to_numpy(dtype=float)
    has_panel = np.isfinite(on_panel)

    ratios = []
    for year in years:
        if year in table.index and year in panel.index:
            base_value = float(table.loc[year, "receipts_usd_bn"])
            panel_value = float(panel.loc[year])
            if np.isfinite(base_value) and base_value > 0.0 and np.isfinite(panel_value):
                ratios.append(panel_value / base_value)
    if not ratios:
        raise ValueError(
            f"None of the years {years} has both a panel value and a positive baseline value, "
            "so the ratio of the panel receipts to the baseline receipts is undefined."
        )
    ratio = float(np.mean(ratios))

    receipts = np.where(has_panel, on_panel, original * ratio)
    valid_gdp = np.isfinite(gdp) & (gdp > 0.0)
    share = np.where(valid_gdp, 100.0 * receipts / np.where(valid_gdp, gdp, 1.0), np.nan)
    out = table.copy()
    out["receipts_usd_bn"] = receipts
    out["receipts_pct_gdp"] = share
    out["receipts_usd_bn_baseline"] = original
    out["receipts_basis"] = np.where(has_panel, "panel", "ratio")
    out["basis_ratio"] = np.where(has_panel, np.nan, ratio)
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


def _positive_scalar(value: Any, name: str) -> float:
    """Validate a positive finite number.

    Parameters
    ----------
    value : object
        The value to check.
    name : str
        Name of the argument, used in error messages.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    ValueError
        If ``value`` is not a real number (a boolean is not), or is not finite
        and positive.
    """
    if not _is_number(value):
        raise ValueError(f"{name} must be a number.")
    number = float(value)
    if not (np.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be finite and positive; found {number}.")
    return number


def source_receipts_share(shares: ArrayLike, usage: ArrayLike) -> float:
    """Usage-weighted mean receipts share of the sources behind the target.

    The weights are the mass that the transport takes from each source, so the
    result is the receipts share of GDP of the economies whose placebo effects
    make up the null of the target.

    Parameters
    ----------
    shares : array_like, shape (n,)
        Receipts share of GDP of each source, in percent; finite and positive.
    usage : array_like, shape (n,)
        Mass that the transport takes from each source; finite and non-negative
        with a positive sum.  When both arguments are Series their indexes must
        be equal.

    Returns
    -------
    float
        ``sum(usage * shares) / sum(usage)``, in the unit of ``shares``.

    Raises
    ------
    ValueError
        If an argument is empty, not one-dimensional or not finite, the lengths
        differ, two Series have different indexes, a share is not positive, a
        usage is negative, or the usage sums to zero.
    """
    share_values = _as_draws(shares, "shares")
    weights = _as_draws(usage, "usage")
    if share_values.shape != weights.shape:
        raise ValueError("shares and usage must have the same length.")
    if (
        isinstance(shares, pd.Series)
        and isinstance(usage, pd.Series)
        and not shares.index.equals(usage.index)
    ):
        raise ValueError("shares and usage must have the same index when both are Series.")
    if np.any(share_values <= 0.0):
        raise ValueError("shares must be positive.")
    if np.any(weights < 0.0):
        raise ValueError("usage must be non-negative.")
    total = float(np.sum(weights))
    if not total > 0.0:
        raise ValueError("usage must have a positive sum.")
    return float(np.sum(weights * share_values) / total)


def rescale_null_pp(
    null_pp: ArrayLike, source_receipts_share: float, target_receipts_share: float
) -> tuple[np.ndarray, float]:
    """Move a null in percentage points of GDP to the receipts share of the target.

    A placebo effect in percentage points of GDP is a change in receipts as a
    share of GDP, and its spread grows with the level of that share.  The null
    draws come from the placebo effects of the source economies, so they are
    multiplied by the ratio of the receipts share of the target to the receipts
    share of the sources, which assumes that the spread is proportional to the
    share.  The null of the relative scaling, in percent of the counterfactual
    receipts, does not depend on the level of the receipts and is not rescaled.

    Parameters
    ----------
    null_pp : array_like, shape (n,)
        Draws of the effect under the null hypothesis, in percentage points of
        GDP.
    source_receipts_share : float
        Receipts share of GDP of the sources, in percent; positive and finite.
        :func:`source_receipts_share` computes it from the usage of the sources.
    target_receipts_share : float
        Receipts share of GDP of the target, in percent; positive and finite,
        on the same basis as the share of the sources.

    Returns
    -------
    draws : ndarray, shape (n,)
        The draws multiplied by ``factor``; a new array.
    factor : float
        ``target_receipts_share / source_receipts_share``.

    Raises
    ------
    ValueError
        If the draws are empty, not one-dimensional or not finite, or a share is
        not a finite positive number.
    """
    draws = _as_draws(null_pp, "null_pp")
    source = _positive_scalar(source_receipts_share, "source_receipts_share")
    target = _positive_scalar(target_receipts_share, "target_receipts_share")
    factor = target / source
    return draws * factor, float(factor)


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
        Plain-language account of each criterion and of the decision.
    info : dict, default empty
        Comparisons that are reported and not used by the rule.  An entry is
        present only when its inputs were given.  ``rmse_ratio_to_best_ambient``
        is the leave-one-economy-out RMSE of ``ot_weighted`` divided by the
        smallest RMSE of the ambient estimators and ``best_ambient`` names that
        estimator.  ``rmse_ratio_to_best_neighbour`` is the same ratio to the
        smaller RMSE of the nearest-neighbour predictors ``nn1`` and ``nn3``
        and ``best_neighbour`` names the predictor.
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


def _positive_rmse(value: Any, label: str) -> float:
    """Validate one RMSE value.

    Parameters
    ----------
    value : object
        The value to check.
    label : str
        Name of the method or estimator, used in error messages.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    ValueError
        If ``value`` is not a number, or is not finite and positive.
    """
    if not _is_number(value):
        raise ValueError(f"The rmse of {label} must be a number.")
    number = float(value)
    if not (np.isfinite(number) and number > 0.0):
        raise ValueError(f"The rmse of {label} must be finite and positive; found {number}.")
    return number


def _read_rmse_mapping(values: Any, argument: str) -> dict[str, float]:
    """Read a mapping of estimator names to RMSE values.

    Parameters
    ----------
    values : mapping or pandas.Series
        Estimator name to RMSE.
    argument : str
        Name of the argument, used in error messages.

    Returns
    -------
    dict
        The names, in the given order, with their RMSE as floats.

    Raises
    ------
    ValueError
        If ``values`` is not a mapping or a Series, is empty, has a name that is
        not a string or occurs twice, or holds an RMSE that is not a finite
        positive number.
    """
    if not isinstance(values, (Mapping, pd.Series)):
        raise ValueError(f"{argument} must be a mapping from estimator names to RMSE values.")
    items = list(values.items())
    if not items:
        raise ValueError(f"{argument} must contain at least one estimator.")
    out: dict[str, float] = {}
    for name, value in items:
        if not isinstance(name, str):
            raise ValueError(f"The names in {argument} must be strings.")
        if name in out:
            raise ValueError(f"{argument} has more than one entry named {name!r}.")
        out[name] = _positive_rmse(value, name)
    return out


def _smallest(values: Mapping[str, float]) -> str:
    """Name of the smallest value, resolving near ties by the order of the mapping.

    Parameters
    ----------
    values : mapping of str to float
        Positive values in the order of the mapping.

    Returns
    -------
    str
        The first name, in the order of the mapping, whose value is within the
        relative tolerance ``_RMSE_TOLERANCE`` of the smallest value.
    """
    cutoff = min(values.values()) * (1.0 + _RMSE_TOLERANCE)
    return next(name for name, value in values.items() if value <= cutoff)


def _loco_rmse(loco_summary: pd.DataFrame) -> dict[str, float]:
    """RMSE of each compared method in the leave-one-group-out summary.

    Parameters
    ----------
    loco_summary : pandas.DataFrame
        Summary indexed by method name with a column ``rmse``.

    Returns
    -------
    dict
        The RMSE of ``ot_weighted``, ``equal`` and ``ot_uniform``, followed by
        that of ``nn1`` and ``nn3`` for those of the two that have a row.

    Raises
    ------
    ValueError
        If the summary is not a DataFrame with a column ``rmse`` and one row per
        method, lacks one of the three methods, or holds an RMSE that is not a
        finite positive number for one of the three methods or for ``nn1`` or
        ``nn3`` when they are present.
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
    methods = [*_ROUTE_METHODS, *(m for m in _NEIGHBOUR_METHODS if m in loco_summary.index)]
    return {method: _positive_rmse(loco_summary.loc[method, "rmse"], method) for method in methods}


def _read_support(support: Any) -> tuple[bool, float | None, list[str]]:
    """Read the result of the target-support check.

    Parameters
    ----------
    support : mapping
        Mapping with the boolean entry ``supported`` and, optionally, ``ess``
        (a number) and ``reasons`` (a string or a sequence of strings).

    Returns
    -------
    supported : bool
        Whether the target is supported.
    ess : float or None
        The effective number of sources, or None when the entry is absent or
        None.
    reasons : list of str
        The reasons, each ending in punctuation; empty when none are given.

    Raises
    ------
    ValueError
        If ``support`` is None, is not a mapping, has no boolean entry
        ``supported``, or holds an ``ess`` that is not a number or ``reasons``
        that are not strings.
    """
    if support is None:
        raise ValueError("support is required: pass the result of transport.target_support")
    if not isinstance(support, Mapping):
        raise ValueError("support must be a mapping with a boolean entry named 'supported'.")
    if "supported" not in support:
        raise ValueError("support must have an entry named 'supported'.")
    flag = support["supported"]
    if not isinstance(flag, (bool, np.bool_)):
        raise ValueError("support['supported'] must be a boolean.")
    ess = support.get("ess")
    if ess is not None and not _is_number(ess):
        raise ValueError("support['ess'] must be a number.")
    raw = support.get("reasons")
    if raw is None:
        items: list[Any] = []
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError("support['reasons'] must be a string or a sequence of strings.")
    if not all(isinstance(item, str) for item in items):
        raise ValueError("support['reasons'] must be a string or a sequence of strings.")
    reasons = []
    for item in items:
        text = item.strip()
        if text:
            reasons.append(text if text.endswith((".", "!", "?")) else text + ".")
    return bool(flag), None if ess is None else float(ess), reasons


def _support_sentence(supported: bool, ess: float | None, reasons: Sequence[str]) -> str:
    """Plain-language account of the target-support criterion.

    Parameters
    ----------
    supported : bool
        Whether the target is supported.
    ess : float or None
        Effective number of sources, if reported.
    reasons : sequence of str
        Reasons reported by the support check.

    Returns
    -------
    str
        One or more sentences that state the result with ``ess`` and the reasons.
    """
    verdict = "supported" if supported else "not supported"
    text = f"The support check reports that the target is {verdict} by the sources"
    if ess is not None:
        text += f" (effective number of sources {ess:.3g})"
    text += "."
    if reasons:
        text += " Reasons reported: " + " ".join(reasons)
    return text + f" The target-support criterion is {'met' if supported else 'not met'}."


def _reported_comparisons(
    weighted: float,
    ambient: Mapping[str, float] | None,
    neighbours: Mapping[str, float],
) -> tuple[dict[str, Any], list[str]]:
    """Ratios of the RMSE of ``ot_weighted`` to the best competing predictors.

    Parameters
    ----------
    weighted : float
        RMSE of ``ot_weighted``.
    ambient : mapping of str to float or None
        RMSE of the ambient estimators, if given.
    neighbours : mapping of str to float
        RMSE of the nearest-neighbour predictors that the summary holds.

    Returns
    -------
    info : dict
        The entries of :attr:`RouteDecision.info` that can be computed.
    sentences : list of str
        One sentence for each ratio, which states that it is reported and not
        used by the rule.
    """
    info: dict[str, Any] = {}
    sentences: list[str] = []
    if ambient:
        best = _smallest(ambient)
        ratio = weighted / ambient[best]
        info["rmse_ratio_to_best_ambient"] = float(ratio)
        info["best_ambient"] = best
        sentences.append(
            f"The leave-one-economy-out RMSE of ot_weighted is {ratio:.3f} times the smallest "
            f"RMSE among the ambient estimators, which is {best} with {ambient[best]:.4g}; this "
            "comparison is reported and is not used by the rule."
        )
    if neighbours:
        best = _smallest(neighbours)
        ratio = weighted / neighbours[best]
        info["rmse_ratio_to_best_neighbour"] = float(ratio)
        info["best_neighbour"] = best
        sentences.append(
            f"The RMSE of ot_weighted is {ratio:.3f} times the smallest RMSE among the "
            f"nearest-neighbour predictors, which is {best} with {neighbours[best]:.4g}; this "
            "comparison is reported and is not used by the rule."
        )
    return info, sentences


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
    support: Mapping[str, Any] | None = None,
    ambient_rmse: Mapping[str, float] | pd.Series | None = None,
) -> RouteDecision:
    """Choose the primary route for the Thailand estimate.

    The primary route is ``ot_importance`` if and only if all four criteria
    hold, and ``ambient`` otherwise:

    1. ``importance_diagnostics["usable"]`` is true.
    2. The leave-one-economy-out RMSE of ``ot_weighted`` is at most 0.95 times
       the RMSE of ``equal`` and at most 0.95 times the RMSE of ``ot_uniform``.
    3. ``n_cases`` is at least 20.
    4. ``support["supported"]`` is true: the target lies within the support of
       the sources, so that the transported effect can reach it.

    The fourth criterion is needed because the validation predicts each case at
    its own features, that is, by interpolation among the cases, and says
    nothing about a target outside the cloud of the sources.  There a transport
    of the source effects cannot return an effect larger than the largest
    effect of a source, however far out the target lies.

    ``n_cases`` counts episodes.  Cases of one economy are not independent, and
    the validation of criterion 2 leaves out whole economies, so the economies
    are the independent units and the number of episodes overstates the number
    of independent observations.

    The comparison of ``ot_weighted`` with the ambient estimators and with the
    nearest-neighbour predictors is not a criterion.  It is stated in the
    ``details`` and stored in ``info`` so that the reader sees how the chosen
    route compares, and it does not enter the decision.

    Parameters
    ----------
    importance_diagnostics : mapping
        Diagnostics of the importance model.  Only the entry ``usable`` is read;
        it must be a boolean.
    loco_summary : pandas.DataFrame
        Leave-one-economy-out summary indexed by method name, with a column
        ``rmse``.  The rows ``ot_weighted``, ``equal`` and ``ot_uniform`` must
        be present with a finite positive RMSE.  The rows ``nn1`` and ``nn3``
        are used for ``info`` when present, with a finite positive RMSE; other
        rows are ignored.
    n_cases : int
        Number of episodes; a non-negative integer.
    rules : mapping, optional
        Overrides for the thresholds: ``rmse_ratio_equal`` (default 0.95) and
        ``rmse_ratio_uniform`` (default 0.95), both in (0, 1.5], and
        ``min_cases`` (default 20), a non-negative number.  The ratios are
        compared with a relative tolerance of 1e-9, so a ratio that equals its
        limit up to rounding passes.
    support : mapping
        Result of the target-support check, for example the output of
        ``dtt.transport.target_support``.  Required.  It must have the boolean
        entry ``supported``; the entries ``ess`` (effective number of sources, a
        number) and ``reasons`` (a string or a sequence of strings) are stated
        in the ``details`` when present.  Other entries are ignored.
    ambient_rmse : mapping or pandas.Series, optional
        Leave-one-economy-out RMSE of the ambient estimators by name, each a
        finite positive number.  Used for ``info`` only.

    Returns
    -------
    RouteDecision
        The primary route, the four criteria in ``passed`` (``importance_usable``,
        ``loco_rmse``, ``enough_cases``, ``target_supported``), a text that
        reports each criterion with its numbers and the support result with
        ``ess`` and the reasons, and ``info``.  ``info`` holds, when the inputs
        are given, ``rmse_ratio_to_best_ambient`` (RMSE of ``ot_weighted``
        divided by the smallest RMSE in ``ambient_rmse``), ``best_ambient``,
        ``rmse_ratio_to_best_neighbour`` (divided by the smaller RMSE of ``nn1``
        and ``nn3`` in ``loco_summary``) and ``best_neighbour``.

    Raises
    ------
    ValueError
        If ``support`` is missing ("support is required: pass the result of
        transport.target_support"), is not a mapping or has no boolean entry
        ``supported``; if ``importance_diagnostics`` has no boolean entry
        ``usable``, the summary lacks one of the three methods or holds an RMSE
        that is not finite and positive, ``n_cases`` is not a non-negative
        integer, ``ambient_rmse`` is not a non-empty mapping of finite positive
        numbers, or a rule name or value is invalid.
    """
    supported, ess, reasons = _read_support(support)
    thresholds = _resolve_rules(rules)
    usable = _usable_flag(importance_diagnostics)
    rmse = _loco_rmse(loco_summary)
    n_cases = _check_n_cases(n_cases)
    ambient = None if ambient_rmse is None else _read_rmse_mapping(ambient_rmse, "ambient_rmse")
    r_weighted, r_equal, r_uniform = (rmse[method] for method in _ROUTE_METHODS)
    slack = 1.0 + _RMSE_TOLERANCE
    ok_equal = r_weighted / r_equal <= thresholds["rmse_ratio_equal"] * slack
    ok_uniform = r_weighted / r_uniform <= thresholds["rmse_ratio_uniform"] * slack
    rmse_ok = bool(ok_equal and ok_uniform)
    enough = bool(n_cases >= thresholds["min_cases"])
    passed = {
        "importance_usable": usable,
        "loco_rmse": rmse_ok,
        "enough_cases": enough,
        "target_supported": supported,
    }
    primary = "ot_importance" if all(passed.values()) else "ambient"
    neighbours = {m: rmse[m] for m in _NEIGHBOUR_METHODS if m in rmse}
    info, comparisons = _reported_comparisons(r_weighted, ambient, neighbours)

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
        _support_sentence(supported, ess, reasons),
        *comparisons,
        "All four criteria are met, so the primary route is ot_importance."
        if primary == "ot_importance"
        else "At least one criterion is not met, so the primary route is ambient.",
    ]
    return RouteDecision(primary=primary, passed=passed, details=" ".join(parts), info=info)


def select_ambient(
    rmse: Mapping[str, float] | pd.Series,
    baseline: str = "A1",
    ratio: float = 0.95,
) -> dict[str, Any]:
    """Select the ambient estimator from its leave-one-economy-out RMSE.

    The baseline estimator stands unless another estimator qualifies.  An
    estimator qualifies when its RMSE, divided by the RMSE of the baseline, is at
    most ``ratio``; the baseline itself never qualifies.  If nothing qualifies,
    the baseline is selected.  Otherwise the qualifying estimator with the
    smallest RMSE is selected, and an exact tie goes to the estimator that comes
    first in ``rmse``.

    The selection comes with a description of how firm it is.  ``margin`` is the
    relative RMSE gap between the selected estimator and the best other
    estimator, ``(rmse_other - rmse_selected) / rmse_selected``.  It is zero when
    no other estimator is given, and it is negative when the baseline is selected
    although another estimator has a smaller RMSE that does not qualify.
    ``close`` is true when that runner-up lies within 5 percent of the selected
    RMSE, or when an estimator other than the baseline has a ratio to the
    baseline that lies within 5 percent of ``ratio``, that is, between 0.95 and
    1.05 times ``ratio``.  A close choice is one that a small change of the
    estimates could reverse.

    The comparisons with the qualification limit and the closeness bounds use a
    relative tolerance of 1e-9, so that a ratio that equals the limit in exact
    arithmetic gives the same decision on every math kernel.  The RMSE values come
    from a validation that leaves out whole economies, so the economies are the
    independent units behind the comparison.

    Parameters
    ----------
    rmse : mapping or pandas.Series
        Leave-one-economy-out RMSE by estimator name, each a finite positive
        number.  The names must be unique strings and include ``baseline``.
    baseline : str, default "A1"
        Name of the baseline estimator, which is selected when no other
        estimator qualifies.
    ratio : float, default 0.95
        Qualification limit in (0, 1]: an estimator qualifies when its RMSE is at
        most ``ratio`` times the RMSE of the baseline.

    Returns
    -------
    dict
        ``selected``, the name of the selected estimator; ``qualifying``, the
        qualifying estimators as a list in the order of ``rmse``; ``ratios``, the
        RMSE of every estimator divided by the RMSE of the baseline; ``margin``,
        the relative gap described above as a float; ``runner_up``, the best
        other estimator, or None when there is none; and ``close``, a boolean.

    Raises
    ------
    ValueError
        If ``rmse`` is not a mapping or Series of finite positive RMSE values
        with unique string names, ``baseline`` is not one of the names, or
        ``ratio`` is not in (0, 1].
    """
    values = _read_rmse_mapping(rmse, "rmse")
    if not isinstance(baseline, str) or baseline not in values:
        raise ValueError(f"The baseline {baseline!r} has no RMSE in rmse.")
    limit = _positive_scalar(ratio, "ratio")
    if limit > 1.0:
        raise ValueError("ratio must lie in (0, 1].")
    base = values[baseline]
    ratios = {name: value / base for name, value in values.items()}
    qualifying = [
        name
        for name in values
        if name != baseline and ratios[name] <= limit * (1.0 + _RMSE_TOLERANCE)
    ]
    selected = _smallest({name: values[name] for name in qualifying}) if qualifying else baseline
    others = {name: value for name, value in values.items() if name != selected}
    if not others:
        return {
            "selected": selected,
            "qualifying": qualifying,
            "ratios": ratios,
            "margin": 0.0,
            "runner_up": None,
            "close": False,
        }
    runner_up = _smallest(others)
    margin = (others[runner_up] - values[selected]) / values[selected]
    close_bound = _CLOSE_SHARE + _RMSE_TOLERANCE
    close = abs(margin) <= close_bound or any(
        abs(ratios[name] / limit - 1.0) <= close_bound for name in values if name != baseline
    )
    return {
        "selected": selected,
        "qualifying": qualifying,
        "ratios": ratios,
        "margin": float(margin),
        "runner_up": runner_up,
        "close": bool(close),
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
