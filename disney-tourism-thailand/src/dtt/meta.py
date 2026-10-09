"""Random-effects and Bayesian hierarchical meta-analysis of case effects.

The functions take the estimated effect of each case, ``tau_hat``, together
with its standard error ``se``, and provide the fallback route used when the
feature-importance route is not supported.

Functions
---------
dersimonian_laird
    Random-effects pooling with the method-of-moments between-case variance.
reml
    Random-effects pooling with the between-case variance estimated by
    restricted maximum likelihood.
hksj_interval
    Hartung-Knapp-Sidik-Jonkman confidence or prediction interval.
bayes_normal_hierarchical
    Bayesian normal hierarchical model with the posterior of the between-case
    standard deviation evaluated on a grid.
meta_regression
    Mixed-effects weighted regression of case effects on case features.
extrapolation
    Position of a new case against the range of the case features.
scaling_law
    Weighted regression of case effects through the origin on a size measure.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import integrate, optimize, stats

__all__ = [
    "dersimonian_laird",
    "reml",
    "hksj_interval",
    "BayesMeta",
    "bayes_normal_hierarchical",
    "meta_regression",
    "extrapolation",
    "scaling_law",
]

_SUMMARY_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_SUMMARY_COLUMNS = ("mean", "sd", "q5", "q25", "q50", "q75", "q95")
# A slope whose absolute value is at most this multiple of the largest absolute effect is
# rounding noise and has no meaningful ratio to another slope.
_SLOPE_FLOOR = 1e-12
# Relative tolerance of the range check of a new case: a multiple of the larger absolute
# end of the range of the cases, so that the position does not change with the unit.
_RANGE_TOL = 1e-9
_SINGULAR_DESIGN = (
    "The regression has a singular design (collinear features or too few cases for the "
    "number of features); use a positive ridge or fewer features."
)


class _IntervalWarning(UserWarning, RuntimeWarning):
    """Warning that interval limits are NaN because the degrees of freedom are below one."""


# ----------------------------------------------------------------------------
# Input checks
# ----------------------------------------------------------------------------
def _as_vector(values: Any, name: str) -> np.ndarray:
    """Return ``values`` as a finite one-dimensional float array.

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
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _check_effects(tau_hat: Any, se: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate case effects and standard errors.

    Parameters
    ----------
    tau_hat : array_like
        Estimated effect of each case.
    se : array_like
        Standard error of each estimated effect; must be strictly positive and
        have the length of ``tau_hat``.

    Returns
    -------
    tuple of ndarray
        The effects and the standard errors as float arrays.
    """
    y = _as_vector(tau_hat, "tau_hat")
    s = _as_vector(se, "se")
    if y.shape != s.shape:
        raise ValueError("tau_hat and se must have the same length.")
    if np.any(s <= 0.0):
        raise ValueError("se must be strictly positive.")
    return y, s


def _check_level(level: float) -> float:
    """Validate a coverage level.

    Parameters
    ----------
    level : float
        Coverage; must lie strictly between 0 and 1.

    Returns
    -------
    float
        The level as a float.
    """
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1.")
    return level


def _prepare_design(X: Any, n_cases: int) -> tuple[list[str], np.ndarray]:
    """Convert a feature table to column names and a float design matrix.

    Parameters
    ----------
    X : array_like, Series or DataFrame
        Case features; a one-dimensional input is a single column.
    n_cases : int
        Required number of rows.

    Returns
    -------
    names : list of str
        Column names (``x0``, ``x1``, ... when ``X`` has no labels).
    ndarray, shape (n_cases, p)
        The finite feature values.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        names = [f"x{j}" for j in range(arr.shape[1])] if arr.ndim == 2 else []
    if arr.ndim != 2 or arr.shape[0] != n_cases:
        raise ValueError("X must be a two-dimensional array with one row per case.")
    if arr.shape[1] == 0:
        raise ValueError("X must have at least one column.")
    if len(set(names)) != len(names):
        raise ValueError("The column names of X must be unique.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values.")
    return names, arr


def _has_labels(index: pd.Index) -> bool:
    """Whether a pandas index holds names rather than positions.

    Parameters
    ----------
    index : pandas.Index
        Column labels of a frame or index of a series.

    Returns
    -------
    bool
        True if at least one label is a string.
    """
    return any(isinstance(label, str) for label in index)


def _align_new_cases(x_new: Any, names: list[str]) -> tuple[np.ndarray, bool]:
    """Feature rows of new cases in the column order of the fitted features.

    A mapping, a DataFrame and a Series with string labels are matched to the
    features by name: every feature name must be present and further names are
    ignored.  Any other input is read in the column order of the fitted features.

    Parameters
    ----------
    x_new : array_like, Series, DataFrame or mapping
        Features of one or several new cases.
    names : list of str
        Feature names of the fitted model.

    Returns
    -------
    ndarray, shape (n, p)
        The feature rows.
    bool
        True when ``x_new`` describes one case without a row dimension.
    """
    single = False
    if isinstance(x_new, Mapping):
        entries = {str(key): value for key, value in x_new.items()}
        single = all(np.ndim(value) == 0 for value in entries.values())
        frame: pd.DataFrame | None = pd.DataFrame([entries] if single else entries)
    elif isinstance(x_new, pd.DataFrame) and _has_labels(x_new.columns):
        frame = x_new
    elif isinstance(x_new, pd.Series) and _has_labels(x_new.index):
        frame, single = x_new.to_frame().T, True
    else:
        frame = None
    if frame is None:
        arr = np.asarray(x_new, dtype=float)
        single = arr.ndim <= 1
        arr = np.atleast_2d(arr)
    else:
        labels = [str(label) for label in frame.columns]
        if len(set(labels)) != len(labels):
            raise ValueError("The names of x_new must be unique.")
        absent = [name for name in names if name not in labels]
        if absent:
            raise ValueError(f"x_new has no value for the features {absent}.")
        arr = frame.iloc[:, [labels.index(name) for name in names]].to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError("x_new must have one value per column of X.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x_new must contain only finite values.")
    return arr, single


def _align_new_size(size_new: Any, name: str) -> np.ndarray:
    """Sizes of new cases, matched by name to the size measure when labelled.

    A mapping, a DataFrame with string column names and a Series with string
    labels must contain the name of the size measure; further names are
    ignored.  Any other input holds the sizes themselves.

    Parameters
    ----------
    size_new : float, array_like, Series, DataFrame or mapping
        Size of one new case or of several.
    name : str
        Name of the size measure.

    Returns
    -------
    ndarray
        The sizes as floats; zero-dimensional for a single size.
    """
    if isinstance(size_new, Mapping):
        entries = {str(key): value for key, value in size_new.items()}
        if name not in entries:
            raise ValueError(f"size_new has no value for the size measure {name!r}.")
        return np.asarray(entries[name], dtype=float)
    if isinstance(size_new, pd.DataFrame):
        if not _has_labels(size_new.columns):
            if size_new.shape[1] != 1:
                raise ValueError("size_new must have one column.")
            return size_new.iloc[:, 0].to_numpy(dtype=float)
        labels = [str(label) for label in size_new.columns]
        if name not in labels:
            raise ValueError(f"size_new has no value for the size measure {name!r}.")
        return size_new.iloc[:, labels.index(name)].to_numpy(dtype=float)
    if isinstance(size_new, pd.Series) and _has_labels(size_new.index):
        labels = [str(label) for label in size_new.index]
        if name not in labels:
            raise ValueError(f"size_new has no value for the size measure {name!r}.")
        return np.asarray(size_new.iloc[labels.index(name)], dtype=float)
    return np.asarray(size_new, dtype=float)


def _t_half_width(level: float, df: int, scale: Any, label: str) -> np.ndarray:
    """Half-width of a Student interval.

    Parameters
    ----------
    level : float
        Coverage of the interval.
    df : int
        Degrees of freedom of the Student quantile.
    scale : float or ndarray
        Standard error of the predicted quantity.
    label : str
        Name of the interval, used in the warning.

    Returns
    -------
    ndarray
        The Student quantile at ``0.5 + level / 2`` with ``df`` degrees of
        freedom times ``scale``; NaN, with a warning, when ``df`` is below one.
    """
    scale = np.asarray(scale, dtype=float)
    if df < 1:
        warnings.warn(
            f"The {label} has {df} degrees of freedom; its limits are NaN.",
            _IntervalWarning,
            stacklevel=3,
        )
        return np.full(scale.shape, np.nan)
    return float(stats.t.ppf(0.5 + level / 2.0, df)) * scale


# ----------------------------------------------------------------------------
# Heterogeneity and pooling helpers
# ----------------------------------------------------------------------------
def _heterogeneity(y: np.ndarray, s: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    """Fixed-effect heterogeneity statistics.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    s : ndarray, shape (k,)
        Standard errors.

    Returns
    -------
    q : float
        Cochran's Q.
    df : int
        Degrees of freedom ``k - 1``.
    i2 : float
        ``max(0, (Q - df) / Q)``, or zero when ``Q`` is zero.
    w : ndarray, shape (k,)
        Inverse-variance weights ``1 / s**2``.
    """
    w = 1.0 / s**2
    mu_fixed = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu_fixed) ** 2))
    df = y.size - 1
    i2 = max(0.0, (q - df) / q) if q > 0.0 else 0.0
    return q, df, float(i2), w


def _pooled_summary(y: np.ndarray, s: np.ndarray, tau2: float) -> dict[str, float | int]:
    """Random-effects pooled mean and heterogeneity summary for a given ``tau2``.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    s : ndarray, shape (k,)
        Standard errors.
    tau2 : float
        Between-case variance.

    Returns
    -------
    dict
        ``mu``, ``se_mu``, ``tau2``, ``Q``, ``I2`` and ``df``.
    """
    q, df, i2, _ = _heterogeneity(y, s)
    w_star = 1.0 / (s**2 + tau2)
    mu = float(np.sum(w_star * y) / np.sum(w_star))
    se_mu = float(np.sqrt(1.0 / np.sum(w_star)))
    return {
        "mu": mu,
        "se_mu": se_mu,
        "tau2": float(tau2),
        "Q": q,
        "I2": i2,
        "df": int(df),
    }


# ----------------------------------------------------------------------------
# Restricted maximum likelihood
# ----------------------------------------------------------------------------
def _neg2_restricted_loglik(
    tau2: float, y: np.ndarray, v: np.ndarray, design: np.ndarray, penalty: np.ndarray
) -> float:
    """Minus twice the restricted log-likelihood, up to an additive constant.

    The model is ``y ~ N(design @ theta, diag(v + tau2))`` with a flat prior on
    the unpenalised coefficients and a Gaussian prior with precision
    ``penalty[j]`` on coefficient ``j``.

    Parameters
    ----------
    tau2 : float
        Between-case variance, non-negative.
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    design : ndarray, shape (k, m)
        Fixed-effects design matrix.
    penalty : ndarray, shape (m,)
        Prior precision of each coefficient; zero for a flat prior.

    Returns
    -------
    float
        The objective to be minimised over ``tau2``.

    Raises
    ------
    ValueError
        If the penalised weighted cross-product matrix is singular.
    """
    total = v + tau2
    w = 1.0 / total
    xtw = design.T * w
    info = xtw @ design
    info[np.diag_indices_from(info)] += penalty
    try:
        theta = np.linalg.solve(info, xtw @ y)
    except np.linalg.LinAlgError as exc:
        raise ValueError(_SINGULAR_DESIGN) from exc
    resid = y - design @ theta
    sse = float(np.sum(w * resid**2) + np.sum(penalty * theta**2))
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0.0 or not np.isfinite(logdet):
        raise ValueError(_SINGULAR_DESIGN)
    return float(np.sum(np.log(total)) + logdet + sse)


def _reml_tau2(y: np.ndarray, v: np.ndarray, design: np.ndarray, penalty: np.ndarray) -> float:
    """Maximise the restricted likelihood over ``tau2 >= 0``.

    A geometric grid locates the maximum and a bounded Brent search from
    :func:`scipy.optimize.minimize_scalar` refines it.  The value zero is
    returned when the likelihood is maximal on the boundary or when the
    design leaves no residual degrees of freedom.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    design : ndarray, shape (k, m)
        Fixed-effects design matrix.
    penalty : ndarray, shape (m,)
        Prior precision of each coefficient; zero for a flat prior.

    Returns
    -------
    float
        The restricted maximum likelihood estimate of ``tau2``.
    """
    k, p = design.shape
    if k <= p and not np.any(penalty > 0.0):
        return 0.0
    scale = max(float(np.mean(v)), float(np.var(y)))

    def objective(t: float) -> float:
        return _neg2_restricted_loglik(t, y, v, design, penalty)

    upper = 100.0 * scale
    grid = np.zeros(1)
    values = np.zeros(1)
    best = 0
    for _ in range(4):
        grid = np.concatenate(([0.0], np.geomspace(1e-9 * scale, upper, 90)))
        values = np.array([objective(float(t)) for t in grid])
        best = int(np.argmin(values))
        if best < grid.size - 1:
            break
        upper *= 100.0
    best_t, best_f = float(grid[best]), float(values[best])
    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, grid.size - 1)])
    if hi > lo:
        res = optimize.minimize_scalar(
            objective,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12 * scale, "maxiter": 500},
        )
        if float(res.fun) < best_f:
            best_t, best_f = float(res.x), float(res.fun)
    return best_t


# ----------------------------------------------------------------------------
# Random-effects pooling
# ----------------------------------------------------------------------------
def dersimonian_laird(tau_hat: ArrayLike, se: ArrayLike) -> dict[str, float | int]:
    """Pool case effects with the DerSimonian-Laird random-effects estimator.

    The between-case variance is the truncated method-of-moments estimate
    ``max(0, (Q - df) / C)`` with ``C = sum(w) - sum(w**2) / sum(w)`` and
    ``w = 1 / se**2``.  The pooled mean uses the weights ``1 / (se**2 + tau2)``
    and its standard error is the square root of the inverse of their sum.

    Parameters
    ----------
    tau_hat : array_like, shape (k,)
        Estimated effect of each case.
    se : array_like, shape (k,)
        Standard error of each estimated effect; strictly positive.

    Returns
    -------
    dict
        ``mu`` pooled mean, ``se_mu`` its standard error, ``tau2`` between-case
        variance, ``Q`` Cochran's heterogeneity statistic, ``I2`` the share of
        total variation due to heterogeneity as a proportion in [0, 1], and
        ``df`` the degrees of freedom ``k - 1``.
    """
    y, s = _check_effects(tau_hat, se)
    q, df, _, w = _heterogeneity(y, s)
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0.0 else 0.0
    return _pooled_summary(y, s, tau2)


def reml(tau_hat: ArrayLike, se: ArrayLike) -> dict[str, float | int]:
    """Pool case effects with the between-case variance estimated by REML.

    The variance ``tau2 >= 0`` maximises the restricted log-likelihood of the
    model ``tau_hat_i ~ N(mu, se_i**2 + tau2)``, found with
    :mod:`scipy.optimize`.  The pooled mean uses the weights
    ``1 / (se**2 + tau2)``.

    Parameters
    ----------
    tau_hat : array_like, shape (k,)
        Estimated effect of each case.
    se : array_like, shape (k,)
        Standard error of each estimated effect; strictly positive.

    Returns
    -------
    dict
        The keys of :func:`dersimonian_laird`: ``mu``, ``se_mu``, ``tau2``,
        ``Q``, ``I2`` and ``df``.  ``Q``, ``I2`` and ``df`` are the fixed-effect
        heterogeneity statistics.
    """
    y, s = _check_effects(tau_hat, se)
    tau2 = _reml_tau2(y, s**2, np.ones((y.size, 1)), np.zeros(1))
    return _pooled_summary(y, s, tau2)


def hksj_interval(
    tau_hat: ArrayLike, se: ArrayLike, level: float = 0.9, new_study: bool = True
) -> dict[str, float]:
    """Hartung-Knapp-Sidik-Jonkman interval for the mean or for a new case.

    With ``tau2`` estimated by REML and ``w = 1 / (se**2 + tau2)``, the pooled
    mean is ``mu = sum(w * tau_hat) / sum(w)``.  The variance of the mean is
    ``max(1, q) / sum(w)`` with ``q = sum(w * (tau_hat - mu)**2) / (k - 1)``, so
    that the interval is never narrower than the random-effects interval with
    the same quantile; a ``q`` of zero, as for identical effects, gives the
    random-effects variance ``1 / sum(w)``.  The confidence interval for the mean
    is ``mu`` plus and minus ``t * sqrt(var)`` with the two-sided Student
    quantile at ``k - 1`` degrees of freedom.  The prediction interval for the
    effect of a new case uses the variance ``var + tau2`` and the Student
    quantile at ``k - 2`` degrees of freedom; for ``k = 2`` it has no degrees of
    freedom, its limits are NaN and a warning is issued.  The REML estimate of
    ``tau2`` enters both intervals as a fixed value: its sampling uncertainty is
    not propagated.

    Parameters
    ----------
    tau_hat : array_like, shape (k,)
        Estimated effect of each case; ``k >= 2``.
    se : array_like, shape (k,)
        Standard error of each estimated effect; strictly positive.
    level : float, default 0.9
        Coverage of the interval.
    new_study : bool, default True
        Return the prediction interval for a new case instead of the
        confidence interval for the mean.

    Returns
    -------
    dict
        ``mean`` the pooled mean, ``lo`` and ``hi`` the interval limits.
    """
    y, s = _check_effects(tau_hat, se)
    level = _check_level(level)
    k = y.size
    if k < 2:
        raise ValueError("At least two cases are required.")
    tau2 = _reml_tau2(y, s**2, np.ones((k, 1)), np.zeros(1))
    w = 1.0 / (s**2 + tau2)
    mean = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mean) ** 2) / (k - 1))
    var = max(1.0, q) / float(np.sum(w)) + (tau2 if new_study else 0.0)
    df = k - 2 if new_study else k - 1
    half = float(_t_half_width(level, df, np.sqrt(var), "prediction interval"))
    return {"mean": mean, "lo": mean - half, "hi": mean + half}


# ----------------------------------------------------------------------------
# Bayesian normal hierarchical model
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class BayesMeta:
    """Posterior of the normal hierarchical meta-analysis.

    Attributes
    ----------
    draws_mu : ndarray, shape (n_draws,)
        Posterior draws of the mean effect ``mu``.
    draws_tau : ndarray, shape (n_draws,)
        Posterior draws of the between-case standard deviation ``tau``.
    draws_pred : ndarray, shape (n_draws,)
        Posterior predictive draws of the effect of a new case, ``mu + tau * z``
        with ``z`` standard normal.
    shrunk_effects : ndarray, shape (k,)
        Posterior mean of the true effect of each case.
    summary : pandas.DataFrame
        Rows ``mu``, ``tau`` and ``predictive``; columns ``mean``, ``sd`` and the
        quantiles ``q5``, ``q25``, ``q50``, ``q75``, ``q95``.
    prob_pred_positive : float
        Posterior probability that the predictive effect exceeds zero.
    tau_grid : ndarray, shape (n_grid,)
        Grid on which the posterior of ``tau`` was evaluated.
    tau_density : ndarray, shape (n_grid,)
        Normalised posterior density of ``tau`` on ``tau_grid``.
    tau_prior_scale : float
        Scale of the half-normal prior on ``tau`` that was used.
    mu_prior : tuple of float
        Mean and standard deviation of the normal prior on ``mu`` that was used.
    """

    draws_mu: np.ndarray
    draws_tau: np.ndarray
    draws_pred: np.ndarray
    shrunk_effects: np.ndarray
    summary: pd.DataFrame
    prob_pred_positive: float
    tau_grid: np.ndarray
    tau_density: np.ndarray
    tau_prior_scale: float
    mu_prior: tuple[float, float]

    def predictive_quantiles(self, qs: ArrayLike) -> np.ndarray:
        """Quantiles of the posterior predictive effect of a new case.

        Parameters
        ----------
        qs : float or array_like
            Probabilities in [0, 1].

        Returns
        -------
        float or ndarray
            The quantiles of ``draws_pred``, with the shape of ``qs``.
        """
        return np.quantile(self.draws_pred, qs)


def _log_marginal_tau(
    tau: np.ndarray, y: np.ndarray, v: np.ndarray, m0: float, s0: float
) -> np.ndarray:
    """Log marginal likelihood of the data given ``tau``, up to a constant.

    The common mean ``mu`` is integrated out under the prior ``N(m0, s0**2)``.

    Parameters
    ----------
    tau : ndarray, shape (g,)
        Values of the between-case standard deviation.
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    m0, s0 : float
        Mean and standard deviation of the normal prior on ``mu``.

    Returns
    -------
    ndarray, shape (g,)
        The log marginal likelihood at each value of ``tau``.
    """
    total = v[None, :] + np.square(tau)[:, None]
    w = 1.0 / total
    sw = w.sum(axis=1)
    ybar = (w * y).sum(axis=1) / sw
    ssw = (w * np.square(y - ybar[:, None])).sum(axis=1)
    a = 1.0 + s0**2 * sw
    quad = ssw + sw * np.square(ybar - m0) / a
    return -0.5 * (np.log(total).sum(axis=1) + np.log(a) + quad)


def _tau_upper_limit(
    y: np.ndarray, v: np.ndarray, m0: float, s0: float, prior_scale: float
) -> float:
    """Upper end of the grid for ``tau``.

    The limit is twice the largest value on a doubling ladder at which the log
    posterior is within 25 of its maximum over the ladder.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    m0, s0 : float
        Mean and standard deviation of the normal prior on ``mu``.
    prior_scale : float
        Scale of the half-normal prior on ``tau``.

    Returns
    -------
    float
        The upper limit of the grid.
    """
    spread = float(np.std(y)) if y.size > 1 else 0.0
    scale = max(prior_scale, float(np.sqrt(v.max())), spread)
    ladder = scale * np.power(2.0, np.arange(-24, 12))
    logpost = _log_marginal_tau(ladder, y, v, m0, s0) - 0.5 * np.square(ladder / prior_scale)
    inside = np.flatnonzero(logpost > logpost.max() - 25.0)
    return 2.0 * float(ladder[inside.max()])


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoid-rule weights for nodes that may be unequally spaced.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Increasing nodes.

    Returns
    -------
    ndarray, shape (n,)
        Weights such that ``weights @ f(x)`` is the trapezoid integral of ``f``.
    """
    gaps = np.diff(x)
    weights = np.zeros(x.size)
    weights[:-1] += 0.5 * gaps
    weights[1:] += 0.5 * gaps
    return weights


def bayes_normal_hierarchical(
    tau_hat: ArrayLike,
    se: ArrayLike,
    mu_prior: tuple[float, float] | None = None,
    tau_prior_scale: float | None = None,
    n_grid: int = 2000,
    n_draws: int = 20000,
    seed: int | np.random.Generator = 0,
) -> BayesMeta:
    """Bayesian normal hierarchical meta-analysis of case effects.

    The model is ``tau_hat_i ~ N(theta_i, se_i**2)``, ``theta_i ~ N(mu, tau**2)``,
    ``mu ~ N(mu_prior[0], mu_prior[1]**2)`` and ``tau ~ half-normal`` with scale
    ``tau_prior_scale``.  The marginal posterior of ``tau`` is evaluated on a
    grid of ``n_grid`` points ``upper * u**2`` with ``u`` equally spaced on
    [0, 1], where ``upper`` is the point beyond which the posterior density is
    negligible.  Draws of ``tau`` come from the inverse of the piecewise-linear
    posterior distribution function on the grid.  Given ``tau``, ``mu`` has an
    analytic normal conditional posterior, from which ``mu`` is drawn.  The
    posterior mean of each case effect is computed by quadrature over the grid.

    Parameters
    ----------
    tau_hat : array_like, shape (k,)
        Estimated effect of each case.
    se : array_like, shape (k,)
        Standard error of each estimated effect; strictly positive.
    mu_prior : tuple of float, optional
        Mean and standard deviation of the normal prior on ``mu``.  The default
        has mean 0 and standard deviation ``5 * max(sd, median(se))``, where
        ``sd`` is the sample standard deviation of ``tau_hat`` (zero when
        ``k = 1``), so the prior is wide on the scale of the data.  The prior
        in use is returned in the field ``mu_prior`` of the result.
    tau_prior_scale : float, optional
        Scale of the half-normal prior on ``tau``.  The default is the sample
        standard deviation of ``tau_hat``, which requires ``k >= 2``.
    n_grid : int, default 2000
        Number of grid points for ``tau``.
    n_draws : int, default 20000
        Number of posterior draws.
    seed : int or numpy.random.Generator, default 0
        Seed of the random generator, or a generator to draw from.

    Returns
    -------
    BayesMeta
        Draws of ``mu``, ``tau`` and the predictive effect of a new case, the
        posterior means of the case effects, a summary table, the posterior
        probability that the predictive effect is positive and the priors in
        use.
    """
    y, s = _check_effects(tau_hat, se)
    v = s**2
    k = y.size
    if mu_prior is None:
        spread = float(np.std(y, ddof=1)) if k > 1 else 0.0
        m0, s0 = 0.0, 5.0 * max(spread, float(np.median(s)))
    else:
        if len(mu_prior) != 2:
            raise ValueError("mu_prior must contain a mean and a standard deviation.")
        m0, s0 = float(mu_prior[0]), float(mu_prior[1])
    if not (np.isfinite(m0) and np.isfinite(s0) and s0 > 0.0):
        raise ValueError("mu_prior must be a finite mean and a positive standard deviation.")
    if tau_prior_scale is None:
        if k < 2:
            raise ValueError("tau_prior_scale is required when only one case is supplied.")
        scale_a = float(np.std(y, ddof=1))
    else:
        scale_a = float(tau_prior_scale)
    if not (np.isfinite(scale_a) and scale_a > 0.0):
        raise ValueError("The half-normal prior scale for tau must be positive and finite.")
    n_grid, n_draws = int(n_grid), int(n_draws)
    if n_grid < 3:
        raise ValueError("n_grid must be at least 3.")
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")

    upper = _tau_upper_limit(y, v, m0, s0, scale_a)
    grid = upper * np.linspace(0.0, 1.0, n_grid) ** 2
    logpost = _log_marginal_tau(grid, y, v, m0, s0) - 0.5 * np.square(grid / scale_a)
    dens = np.exp(logpost - logpost.max())
    dens /= integrate.trapezoid(dens, grid)
    cdf = integrate.cumulative_trapezoid(dens, grid, initial=0.0)
    cdf /= cdf[-1]

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    draws_tau = np.interp(rng.uniform(size=n_draws), cdf, grid)
    total = v[None, :] + np.square(draws_tau)[:, None]
    w = 1.0 / total
    precision = 1.0 / s0**2 + w.sum(axis=1)
    cond_mean = (m0 / s0**2 + (w * y).sum(axis=1)) / precision
    draws_mu = cond_mean + rng.standard_normal(n_draws) / np.sqrt(precision)
    draws_pred = draws_mu + draws_tau * rng.standard_normal(n_draws)

    mass = dens * _trapezoid_weights(grid)
    mass /= mass.sum()
    total_g = v[None, :] + np.square(grid)[:, None]
    w_g = 1.0 / total_g
    precision_g = 1.0 / s0**2 + w_g.sum(axis=1)
    cond_mean_g = (m0 / s0**2 + (w_g * y).sum(axis=1)) / precision_g
    pull = v[None, :] / total_g
    shrunk = mass @ ((1.0 - pull) * y[None, :] + pull * cond_mean_g[:, None])

    rows = []
    for draws in (draws_mu, draws_tau, draws_pred):
        quantiles = np.quantile(draws, _SUMMARY_QUANTILES)
        rows.append(
            [float(np.mean(draws)), float(np.std(draws, ddof=1)) if draws.size > 1 else np.nan]
            + [float(q) for q in quantiles]
        )
    summary = pd.DataFrame(
        rows, index=["mu", "tau", "predictive"], columns=list(_SUMMARY_COLUMNS), dtype=float
    )
    return BayesMeta(
        draws_mu=draws_mu,
        draws_tau=draws_tau,
        draws_pred=draws_pred,
        shrunk_effects=shrunk,
        summary=summary,
        prob_pred_positive=float(np.mean(draws_pred > 0.0)),
        tau_grid=grid,
        tau_density=dens,
        tau_prior_scale=scale_a,
        mu_prior=(m0, s0),
    )


# ----------------------------------------------------------------------------
# Meta-regression
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _MixedFit:
    """Fitted mixed-effects regression on standardised features.

    Attributes
    ----------
    center : ndarray, shape (p,)
        Column means of the features.
    scale : ndarray, shape (p,)
        Column standard deviations, one for the columns left out of the fit.
    keep : ndarray of bool, shape (p,)
        Columns that enter the fit.
    theta : ndarray, shape (m,)
        Intercept and standardised slopes, ``m = 1 + keep.sum()``.
    cov : ndarray, shape (m, m)
        Covariance matrix of ``theta``.
    tau2 : float
        Between-case variance.
    """

    center: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    theta: np.ndarray
    cov: np.ndarray
    tau2: float

    @property
    def n_coef(self) -> int:
        """Number of fitted coefficients, including the intercept."""
        return int(self.theta.size)

    def design(self, X: np.ndarray) -> np.ndarray:
        """Standardise the rows of ``X`` and prepend the intercept column.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Feature rows on the original scale.

        Returns
        -------
        ndarray, shape (n, m)
            The design matrix of the fitted model.
        """
        z = (X[:, self.keep] - self.center[self.keep]) / self.scale[self.keep]
        return np.column_stack([np.ones(X.shape[0]), z])


def _fit_mixed(y: np.ndarray, v: np.ndarray, X: np.ndarray, ridge: float) -> _MixedFit:
    """Fit ``y = a + z @ b + u + e`` with REML ``tau2`` and a ridge penalty on ``b``.

    The columns of ``X`` are centred and scaled to unit population standard
    deviation; constant columns are left out of the fit.  The penalty added to
    the slope block of the weighted cross-product matrix is ``ridge`` times the
    mean of ``1 / v``.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    X : ndarray, shape (k, p)
        Case features.
    ridge : float
        Non-negative ridge weight.

    Returns
    -------
    _MixedFit
        The fitted model.

    Raises
    ------
    ValueError
        If ``ridge`` is zero and the standardised design is singular, that is,
        its smallest singular value is below ``1e-8`` times the largest.
    """
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    keep = scale > 1e-12 * np.maximum(1.0, np.abs(center))
    scale = np.where(keep, scale, 1.0)
    z = (X[:, keep] - center[keep]) / scale[keep]
    design = np.column_stack([np.ones(y.size), z])
    if ridge == 0.0:
        singular_values = np.linalg.svd(design, compute_uv=False)
        if np.sum(singular_values > 1e-8 * singular_values[0]) < design.shape[1]:
            raise ValueError(_SINGULAR_DESIGN)
    penalty = np.concatenate(([0.0], np.full(z.shape[1], ridge * float(np.mean(1.0 / v)))))
    tau2 = _reml_tau2(y, v, design, penalty)
    w = 1.0 / (v + tau2)
    xtw = design.T * w
    info = xtw @ design
    info[np.diag_indices_from(info)] += penalty
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:
        raise ValueError(_SINGULAR_DESIGN) from exc
    theta = cov @ (xtw @ y)
    return _MixedFit(center=center, scale=scale, keep=keep, theta=theta, cov=cov, tau2=tau2)


def meta_regression(
    tau_hat: ArrayLike,
    se: ArrayLike,
    X: ArrayLike | pd.DataFrame,
    x_new: ArrayLike | None = None,
    ridge: float = 0.0,
    level: float = 0.9,
) -> dict[str, Any]:
    """Mixed-effects weighted regression of case effects on case features.

    The model is ``tau_hat_i = a + z_i @ b + u_i + e_i`` with ``u_i ~ N(0, tau2)``
    and ``e_i ~ N(0, se_i**2)``, where ``z_i`` is row ``i`` of ``X`` with each
    column centred and divided by its population standard deviation.  The
    between-case variance ``tau2`` maximises the restricted likelihood.  The
    optional ridge penalty acts on the standardised slopes ``b``: the matrix
    added to the weighted cross-product matrix is ``ridge * mean(1 / se**2)``
    times the identity on the slope block, and the restricted likelihood
    includes the same penalty.  The coefficient covariance is the inverse of
    the penalised weighted cross-product matrix.  Constant columns of ``X``
    are left out of the fit and receive the coefficient zero and a missing
    standard error.

    The features are standardised before the penalty is applied, so the
    penalty does not depend on the units of the features; it depends on the
    standard errors but not on ``tau2``.  The prediction for a new case with
    features ``x_new`` has variance ``tau2`` plus the variance of the fitted
    value, and its interval uses the Student quantile with ``k - p - 1``
    degrees of freedom, where ``p`` is the number of fitted features; when
    this is below one the limits are NaN and a warning is issued.  Leave-one-out
    errors refit the whole model, including standardisation and ``tau2``,
    without case ``i`` and predict case ``i``.

    The ridge penalty shrinks the standardised slopes towards zero, and the
    prediction for a case whose features lie outside the range of the cases is
    an extrapolation of the fitted line, so the shrinkage of a slope moves the
    prediction further the further the new case lies from the feature means.
    ``shrinkage`` reports each slope with and without the penalty, and
    :func:`extrapolation` reports where a new case lies against the range of the
    cases.

    Parameters
    ----------
    tau_hat : array_like, shape (k,)
        Estimated effect of each case; ``k >= 2``.
    se : array_like, shape (k,)
        Standard error of each estimated effect; strictly positive.
    X : array_like or DataFrame, shape (k, p)
        Case features, one row per case.
    x_new : array_like, Series, DataFrame or mapping, optional
        Features of one new case, shape (p,) or a scalar when ``p`` is 1, or of
        several new cases, shape (n, p).  A DataFrame, a Series with string
        labels and a mapping are matched to the columns of ``X`` by name: every
        column name of ``X`` must occur and further names are ignored.  Any
        other input is read in the column order of ``X``.
    ridge : float, default 0.0
        Non-negative ridge weight on the standardised slopes.  It must be
        positive when the number of cases does not exceed the number of fitted
        coefficients.
    level : float, default 0.9
        Coverage of the prediction interval.

    Returns
    -------
    dict
        ``coef`` DataFrame indexed by ``intercept`` and the feature names with
        columns ``estimate`` and ``se`` for the standardised slopes (the
        intercept is the predicted effect at the feature means); ``coef_raw``
        the same table for the original units of ``X`` (the intercept is the
        predicted effect at zero features); ``tau2`` the between-case variance;
        ``prediction`` ``None`` when ``x_new`` is ``None``, otherwise a dict
        with ``mean``, ``lo``, ``hi`` and ``se_pred``, floats for one new case
        given as a vector, a Series or a mapping of scalars and arrays
        otherwise; ``loo_pred`` the leave-one-out predictions; ``loo_errors``
        the observed effects minus ``loo_pred``; ``loo_var`` ``se**2 + tau2``
        with ``tau2`` from the refit without the case; ``loo_rmse`` the root
        mean square of ``loo_errors``; ``df`` the degrees of freedom
        ``k - p - 1`` of the Student quantile; ``shrinkage`` a DataFrame indexed
        by the fitted features (those that are not constant) with the columns
        ``ridge_slope`` (the standardised slope of this fit),
        ``unpenalised_slope`` (the standardised slope of the same model with
        ``ridge=0``, NaN when that model is singular or has no residual degree
        of freedom) and ``ratio`` (``ridge_slope`` over ``unpenalised_slope``,
        NaN where the unpenalised slope is NaN or is rounding noise, that is, its
        absolute value is at most ``1e-12`` times the largest absolute effect).
        When ``ridge`` is zero the two slopes are equal, so the ratio is 1 for
        every feature whose slope is not rounding noise and NaN for the others.

    Raises
    ------
    ValueError
        If the design is singular: ``ridge`` is zero and the features are
        collinear or too numerous, or a leave-one-out fit loses rank.
    """
    y, s = _check_effects(tau_hat, se)
    level = _check_level(level)
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a non-negative finite number.")
    if y.size < 2:
        raise ValueError("At least two cases are required.")
    names, Xm = _prepare_design(X, y.size)
    k, p = Xm.shape
    v = s**2
    fit = _fit_mixed(y, v, Xm, ridge)
    if ridge == 0.0 and k < fit.n_coef + 1:
        raise ValueError(
            "At least one residual degree of freedom is required when ridge is zero; "
            "supply a positive ridge."
        )
    df = k - fit.n_coef

    terms = ["intercept"] + names
    est = np.zeros(p + 1)
    se_est = np.full(p + 1, np.nan)
    kept = np.concatenate(([True], fit.keep))
    est[kept] = fit.theta
    se_est[kept] = np.sqrt(np.diag(fit.cov))
    coef = pd.DataFrame({"estimate": est, "se": se_est}, index=terms)

    transform = np.zeros((fit.n_coef, fit.n_coef))
    transform[0, 0] = 1.0
    transform[0, 1:] = -fit.center[fit.keep] / fit.scale[fit.keep]
    transform[1:, 1:] = np.diag(1.0 / fit.scale[fit.keep])
    raw_theta = transform @ fit.theta
    raw_se = np.sqrt(np.diag(transform @ fit.cov @ transform.T))
    est_raw = np.zeros(p + 1)
    se_raw = np.full(p + 1, np.nan)
    est_raw[kept] = raw_theta
    se_raw[kept] = raw_se
    coef_raw = pd.DataFrame({"estimate": est_raw, "se": se_raw}, index=terms)
    shrinkage = _shrinkage_table(y, v, Xm, ridge, fit, names)

    prediction: dict[str, Any] | None = None
    if x_new is not None:
        xn, single = _align_new_cases(x_new, names)
        dn = fit.design(xn)
        mean = dn @ fit.theta
        var_fit = np.einsum("ij,jk,ik->i", dn, fit.cov, dn)
        se_pred = np.sqrt(fit.tau2 + var_fit)
        half = _t_half_width(level, df, se_pred, "prediction interval")
        prediction = {"mean": mean, "lo": mean - half, "hi": mean + half, "se_pred": se_pred}
        if single:
            prediction = {key: float(val[0]) for key, val in prediction.items()}

    loo_pred = np.empty(k)
    loo_tau2 = np.empty(k)
    for i in range(k):
        rows = np.arange(k) != i
        try:
            sub = _fit_mixed(y[rows], v[rows], Xm[rows], ridge)
        except ValueError as exc:
            raise ValueError(
                f"Leaving out case {i} leaves a singular design; use a positive ridge or "
                "fewer features."
            ) from exc
        loo_pred[i] = float((sub.design(Xm[i : i + 1]) @ sub.theta)[0])
        loo_tau2[i] = sub.tau2
    loo_errors = y - loo_pred
    return {
        "coef": coef,
        "coef_raw": coef_raw,
        "tau2": float(fit.tau2),
        "prediction": prediction,
        "loo_pred": loo_pred,
        "loo_errors": loo_errors,
        "loo_var": v + loo_tau2,
        "loo_rmse": float(np.sqrt(np.mean(loo_errors**2))),
        "df": int(df),
        "shrinkage": shrinkage,
    }


def _shrinkage_table(
    y: np.ndarray, v: np.ndarray, X: np.ndarray, ridge: float, fit: _MixedFit, names: list[str]
) -> pd.DataFrame:
    """Standardised slopes with and without the ridge penalty.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    X : ndarray, shape (k, p)
        Case features.
    ridge : float
        Ridge weight of the fit in ``fit``.
    fit : _MixedFit
        The fit with weight ``ridge``.
    names : list of str
        Feature names.

    Returns
    -------
    pandas.DataFrame
        Indexed by the features that enter the fit, with the columns
        ``ridge_slope``, ``unpenalised_slope`` and ``ratio``.  The unpenalised
        slope is NaN when the model with ``ridge=0`` is singular or has no
        residual degree of freedom; the ratio is NaN where the unpenalised slope
        is NaN or its absolute value is at most ``1e-12`` times the largest
        absolute effect.
    """
    ridge_slope = np.asarray(fit.theta[1:], dtype=float)
    plain = np.full(ridge_slope.size, np.nan)
    if ridge == 0.0:
        plain = ridge_slope.copy()
    elif y.size >= fit.n_coef + 1:
        try:
            plain = np.asarray(_fit_mixed(y, v, X, 0.0).theta[1:], dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            pass
    ratio = np.full(ridge_slope.size, np.nan)
    floor = _SLOPE_FLOOR * float(np.max(np.abs(y)))
    usable = np.isfinite(plain) & (np.abs(plain) > floor)
    ratio[usable] = ridge_slope[usable] / plain[usable]
    index = [name for name, flag in zip(names, fit.keep) if flag]
    return pd.DataFrame(
        {"ridge_slope": ridge_slope, "unpenalised_slope": plain, "ratio": ratio}, index=index
    )


def extrapolation(X: ArrayLike | pd.DataFrame, x_new: Any) -> pd.DataFrame:
    """Position of a new case against the range of the case features.

    A prediction for a case whose feature lies outside the range of the cases is
    an extrapolation of the fitted line, and the ridge penalty on the slope
    changes such a prediction more than one inside the range.

    Parameters
    ----------
    X : array_like or DataFrame, shape (k, p)
        Case features, one row per case.
    x_new : array_like, Series, DataFrame or mapping
        Features of one new case.  A mapping, a Series with string labels and a
        DataFrame are matched to the columns of ``X`` by name (further names are
        ignored); any other input is read in the column order of ``X``.

    Returns
    -------
    pandas.DataFrame
        One row per feature of ``X`` with the columns ``x_new`` (the value of the
        new case), ``case_min`` and ``case_max`` (the range of the cases),
        ``position`` (``below`` when the value is under the smallest case value,
        ``above`` when it is over the largest, ``inside`` otherwise) and
        ``ratio_to_max`` (the value over the largest case value for a feature
        that is non-negative in every case and positive in at least one; NaN for
        any other feature).  The position allows a tolerance of ``1e-9`` times
        the larger of the absolute smallest and largest case value of the
        feature, so it does not change when a feature is multiplied by a positive
        constant.  A feature that is zero in every case has no tolerance.

    Raises
    ------
    ValueError
        If ``X`` or ``x_new`` is not finite or has the wrong shape, ``x_new`` is
        missing a feature, or ``x_new`` describes more than one case.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        n_cases = X.shape[0]
    else:
        raw = np.asarray(X, dtype=float)
        if raw.ndim not in (1, 2):
            raise ValueError("X must be a two-dimensional array with one row per case.")
        n_cases = raw.shape[0]
    names, cases = _prepare_design(X, n_cases)
    new, _ = _align_new_cases(x_new, names)
    if new.shape[0] != 1:
        raise ValueError("x_new must describe exactly one new case.")
    target = new[0]
    low = cases.min(axis=0)
    high = cases.max(axis=0)
    tol = _RANGE_TOL * np.maximum(np.abs(low), np.abs(high))
    below = target < low - tol
    above = target > high + tol
    position = np.where(below, "below", np.where(above, "above", "inside"))
    ratio = np.full(len(names), np.nan)
    positive = (low >= 0.0) & (high > 0.0)
    ratio[positive] = target[positive] / high[positive]
    return pd.DataFrame(
        {
            "x_new": target,
            "case_min": low,
            "case_max": high,
            "position": position.tolist(),
            "ratio_to_max": ratio,
        },
        index=names,
    )


# ----------------------------------------------------------------------------
# Scaling law through the origin
# ----------------------------------------------------------------------------
def _fit_origin(y: np.ndarray, v: np.ndarray, size: np.ndarray) -> tuple[float, float, float]:
    """Weighted slope through the origin with a REML between-case variance.

    Parameters
    ----------
    y : ndarray, shape (k,)
        Case effects.
    v : ndarray, shape (k,)
        Squared standard errors.
    size : ndarray, shape (k,)
        Size measure of each case.

    Returns
    -------
    beta : float
        The slope.
    var_beta : float
        Variance of the slope.
    tau2 : float
        Between-case variance.
    """
    tau2 = _reml_tau2(y, v, size[:, None], np.zeros(1))
    w = 1.0 / (v + tau2)
    sxx = float(np.sum(w * size**2))
    beta = float(np.sum(w * size * y) / sxx)
    return beta, 1.0 / sxx, tau2


def scaling_law(
    tau_hat: ArrayLike,
    se: ArrayLike,
    size: ArrayLike,
    size_new: ArrayLike,
    level: float = 0.9,
) -> dict[str, Any]:
    """Fit an effect proportional to a size measure and predict a new case.

    The model is ``tau_hat_i = beta * size_i + u_i + e_i`` through the origin,
    with ``u_i ~ N(0, tau2)`` and ``e_i ~ N(0, se_i**2)``.  The between-case
    variance ``tau2`` maximises the restricted likelihood and the slope is the
    weighted least-squares estimate with weights ``1 / (se**2 + tau2)``.  The
    prediction for a new case of size ``size_new`` is ``beta * size_new`` with
    variance ``tau2 + size_new**2 * var(beta)``; its interval uses the Student
    quantile with ``k - 2`` degrees of freedom, and for ``k = 2`` the limits are
    NaN and a warning is issued.  Leave-one-out errors refit the model,
    including ``tau2``, without case ``i`` and predict case ``i``.

    Parameters
    ----------
    tau_hat : array_like, shape (k,)
        Estimated effect of each case; ``k >= 2``.
    se : array_like, shape (k,)
        Standard error of each estimated effect; strictly positive.
    size : array_like or Series, shape (k,)
        Size measure of each case, for example relative investment.  The name
        of a Series is the name of the size measure; any other input is named
        ``size``.
    size_new : float, array_like, Series, DataFrame or mapping
        Size of the new case, or of several new cases.  A DataFrame, a mapping
        and a Series with string labels are matched to the size measure by
        name: the name must occur and further names are ignored.  Any other
        input holds the sizes themselves.
    level : float, default 0.9
        Coverage of the prediction interval.

    Returns
    -------
    dict
        ``beta`` the slope, ``se_beta`` its standard error, ``tau2`` the
        between-case variance, ``prediction`` a dict with ``mean``, ``lo``,
        ``hi`` and ``se_pred`` (floats for a single size, arrays otherwise),
        ``loo_pred`` the leave-one-out predictions, ``loo_errors`` the observed
        effects minus ``loo_pred``, ``loo_var`` ``se**2 + tau2`` with ``tau2``
        from the refit without the case, ``loo_rmse`` the root mean square of
        ``loo_errors``, and ``df`` the degrees of freedom ``k - 2`` of the
        Student quantile.
    """
    y, s = _check_effects(tau_hat, se)
    name = str(size.name) if isinstance(size, pd.Series) and size.name is not None else "size"
    x = _as_vector(size, "size")
    if x.shape != y.shape:
        raise ValueError("size must have the same length as tau_hat.")
    level = _check_level(level)
    k = y.size
    if k < 2:
        raise ValueError("At least two cases are required.")
    if not np.any(x != 0.0):
        raise ValueError("size must not be identically zero.")
    xn = _align_new_size(size_new, name)
    if not np.all(np.isfinite(xn)):
        raise ValueError("size_new must contain only finite values.")
    v = s**2
    beta, var_beta, tau2 = _fit_origin(y, v, x)
    df = k - 2
    mean = beta * xn
    se_pred = np.sqrt(tau2 + np.square(xn) * var_beta)
    half = _t_half_width(level, df, se_pred, "prediction interval")
    prediction: dict[str, Any] = {
        "mean": mean,
        "lo": mean - half,
        "hi": mean + half,
        "se_pred": se_pred,
    }
    if xn.ndim == 0:
        prediction = {key: float(val) for key, val in prediction.items()}

    loo_pred = np.empty(k)
    loo_tau2 = np.empty(k)
    for i in range(k):
        rows = np.arange(k) != i
        if not np.any(x[rows] != 0.0):
            raise ValueError("Leaving out one case leaves only zero sizes.")
        beta_i, _, loo_tau2[i] = _fit_origin(y[rows], v[rows], x[rows])
        loo_pred[i] = beta_i * x[i]
    loo_errors = y - loo_pred
    return {
        "beta": float(beta),
        "se_beta": float(np.sqrt(var_beta)),
        "tau2": float(tau2),
        "prediction": prediction,
        "loo_pred": loo_pred,
        "loo_errors": loo_errors,
        "loo_var": v + loo_tau2,
        "loo_rmse": float(np.sqrt(np.mean(loo_errors**2))),
        "df": int(df),
    }
