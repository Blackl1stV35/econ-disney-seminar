"""Optimal transport of estimated case effects to a target cloud.

The module moves per-case effect estimates, and the placebo noise attached to
them, from source economies to the feature cloud of a target economy.  The
ground cost is a feature-importance-weighted squared distance, and sources that
resemble the target on the heavily weighted features receive the most mass.
Everything is implemented with NumPy and SciPy.

Notation
--------
Sources are indexed by ``i`` (``n`` of them), target points by ``j`` (``m`` of
them) and features by ``k`` (``d`` of them).  ``a`` and ``b`` are the
non-negative masses of the sources and of the target points, ``tau`` holds the
source effects and ``w`` the feature weights.

Ground cost
    ``C_ij = sum_k w_k (zs_ik - zt_jk)^2``, with ``w`` rescaled to sum to ``d``.

Transport problem
    :func:`sinkhorn_plan` solves, over non-negative matrices ``pi``,

    ``min <C, pi> + eps KL(pi | a b') + rho_s KL(pi 1 | a) + rho_t KL(pi' 1 | b)``

    with ``KL(p | q) = sum p log(p / q) - p + q``.  A marginal whose ``rho`` is
    ``None`` is enforced exactly and its penalty term is dropped.  The solution
    has the form ``pi_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)`` and is found
    by log-domain scaling iterations whose update for a relaxed marginal has
    exponent ``rho / (rho + eps)`` (Chizat, Peyre, Schmitzer and Vialard).

Small regularisation
    The scaling iterations contract by the factor ``rho / (rho + eps)`` per
    iteration when a marginal is relaxed, so they need about ``(rho + eps) /
    eps`` iterations per factor ``e`` of accuracy.  When ``eps`` is small next
    to ``rho``, for example when the feature weights are almost concentrated on
    a feature with few values and :func:`select_eps` gives a small positive
    value, the iterations can stop at ``max_iter`` before the marginal residual
    is below ``tol``.  The solvers then solve the same problem (same ``eps``,
    ``rho`` and masses) from scratch by a log-domain rescue: Newton iterations
    with Levenberg damping on the dual objective in the log-scalings of the
    smaller side of the plan, the log-scalings of the other side being
    eliminated exactly by their log-sum-exp update, with a continuation in
    ``eps`` that starts at the largest cost and shrinks by a factor of four
    per stage until the requested value.  The plan is evaluated from the
    log-scalings, and the costs are shifted by their row and column minima, so
    the rescue converges for ``eps`` far below the reach of the scaling
    iterations.  A scaling iteration that reaches the tolerance is returned
    unchanged, bit for bit, and the rescue is never run for it.
    :class:`Plan` reports a rescued plan
    in ``log_domain``, :class:`LocoResult` and :func:`bootstrap_transport`
    count rescued solves in ``n_log_domain``, and ``n_nonconverged`` counts only
    the solves that do not reach the tolerance even after the rescue.  The
    rescue needs ``eps`` large enough for the exponents ``C / eps`` to be
    resolved in double precision (the marginal residual cannot fall below about
    ``1e-16 * max(C) / eps``); below that the solve is reported as not
    converged.  The balanced iterations of :func:`sinkhorn_divergence` have no
    rescue.

Transported effect
    ``effect_j = sum_i pi_ij tau_i / sum_i pi_ij`` for each target point, and
    the target effect is the ``b``-weighted mean of these values.

Limits of the method
    The leave-one-economy-out errors of :func:`loco_validation` are the errors
    of cases predicted at their own features, so they measure interpolation
    among the cases.  The transported effect of a target point is a weighted
    mean of source effects, so the transport cannot predict an effect larger
    than the largest source effect (or smaller than the smallest one): a target
    that is costlier, larger or more concentrated than every case is an
    extrapolation that the method cannot make.  The intervals of this module
    describe the target only inside the support checked by
    :func:`target_support`, which reports the effective number of sources
    behind the target and whether the target lies within the range of the
    sources on the features that carry the weight.

Three intervals
    The module produces three different intervals, each for its own purpose.

    * The bootstrap interval of the transported average effect,
      ``bootstrap_transport(...)["draws"]`` and its ``percentiles``.  It
      describes the sampling uncertainty of the transported average effect
      (resampled cases and measurement noise).  It is not an interval for the
      effect observed in a new study and it is not validated.
    * The predictive draws of :func:`bootstrap_transport`
      (``predictive_draws``).  They add a between-case deviation and the
      measurement noise to the bootstrap draws, from a normal model.  They are
      not validated against held-out cases.
    * The conformal interval of the observed effect of a new study at the
      target, :func:`loco_interval` (also ``loco_predictive_draws(...)["interval"]``).
      It is the estimate plus and minus an order statistic of the absolute
      leave-one-economy-out errors, whose coverage of the effect observed in a
      new study at the target is validated by construction under
      exchangeability of the errors.  The ``percentiles`` of
      :func:`loco_predictive_draws` are descriptive and carry no such guarantee.

Numerical tolerances
    Decisions that rounding noise could flip use explicit tolerances.  A mass
    or a feature weight of at most ``1e-12`` times the largest one is zero,
    pairwise distances that agree to a relative ``1e-9`` are ties, a squared
    distance of at most ``1e-18`` times the largest squared weighted
    coordinate of the rows is zero (so the choices of ``eps`` and of the kernel
    bandwidth do not depend on the units of the features), RMSEs that agree to
    a relative ``1e-9`` of the larger of the largest RMSE and the largest
    absolute effect share a rank, and counts of permutation statistics at least
    as large as the observed one allow a relative ``1e-9``.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import optimize, sparse
from scipy.spatial.distance import cdist

__all__ = [
    "robust_standardise",
    "weighted_sq_cost",
    "select_eps",
    "Plan",
    "sinkhorn_plan",
    "transported_effects",
    "source_usage",
    "effective_sample_size",
    "target_effect",
    "transported_mixture",
    "weighted_quantile",
    "wasserstein2",
    "sinkhorn_divergence",
    "overlap_permutation_test",
    "leave_group_out_indices",
    "LocoResult",
    "loco_validation",
    "loco_interval",
    "loco_predictive_draws",
    "target_support",
    "bootstrap_transport",
]

# Relative size below which a mass or a feature weight is rounding noise.
_NOISE = 1e-12
# Relative tolerance of comparisons whose exact-arithmetic outcome is an equality.
_TIE_TOL = 1e-9
# Squared weighted distances at most this large, relative to the largest squared weighted coordinate of the
# rows compared, are rounding noise and count as zero.  The reference is the size of the coordinates, which is
# what the rounding error of a difference depends on, so the rule does not depend on the units of the features.
_DIST_NOISE = 1e-18
_MAD_TO_SD = 1.4826
_LOCO_METHODS = ("ot_weighted", "ot_uniform", "equal", "nn1", "nn3", "kernel")
_WEIGHTED_METHODS = ("ot_weighted", "nn1", "nn3", "kernel")
_PERCENTILES = (5, 25, 50, 75, 95)
# Log-domain rescue: the largest ratios between the regularisation strengths of successive stages of the continuation,
# tried in turn until the final stage reaches the tolerance, the largest number of stages, the marginal residual at
# which an intermediate stage is accepted, the largest numbers of Newton steps of an intermediate and of the final
# stage, and the largest change of one log-scaling in one step.
_RESCUE_RATIOS = (0.25, 0.5, 0.75)
_RESCUE_MAX_STAGES = 60
_RESCUE_STAGE_TOL = 1e-5
_RESCUE_STAGE_STEPS = 40
_RESCUE_FINAL_STEPS = 60
_RESCUE_STEP_CAP = 30.0
# Near the solution (marginal residual below _RESCUE_WIDE_ERR) the coordinates that carry mass, as opposed to those
# with at most _RESCUE_PASSIVE times the total mass, may first try a step of up to _RESCUE_WIDE_CAP, because flat
# directions of the dual objective need steps far beyond _RESCUE_STEP_CAP.
_RESCUE_WIDE_ERR = 1e-3
_RESCUE_WIDE_CAP = 1e6
_RESCUE_PASSIVE = 1e-10
_RATIO_NOTE = (
    "The intervals of the RMSE ratios are descriptive: they describe the cases at hand "
    "and are not a test of the route rule."
)


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
def _numeric(x: Any, name: str) -> np.ndarray:
    """Return ``x`` as a float array, raising a ``ValueError`` that names ``name`` when it is not numeric."""
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _as_frame(X: Any, name: str) -> pd.DataFrame:
    """Return ``X`` as a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        raise TypeError(f"{name} must be a DataFrame or a two-dimensional array")
    arr = _numeric(X, name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a DataFrame or a two-dimensional array")
    return pd.DataFrame(arr)


def _as_matrix(Z: Any, name: str, allow_row: bool = True) -> tuple[np.ndarray, pd.Index | None, pd.Index | None]:
    """Return a float matrix with the column names and row labels of a DataFrame (else ``None``).

    A one-dimensional input is taken as a single row when ``allow_row`` is true.
    """
    if isinstance(Z, pd.DataFrame):
        try:
            return Z.to_numpy(dtype=float), Z.columns, Z.index
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    arr = _numeric(Z, name)
    if arr.ndim == 1 and allow_row:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a DataFrame or a two-dimensional array")
    return arr, None, None


def _pair_matrices(
    Zs: Any, Zt: Any, names: tuple[str, str] = ("Zs", "Zt")
) -> tuple[np.ndarray, np.ndarray, pd.Index | None, pd.Index | None, pd.Index | None]:
    """Return the matrices of two feature tables with ``Zt`` columns ordered as those of ``Zs``.

    ``names`` are the argument names of the two tables used in error messages.
    """
    A, cols, idx_a = _as_matrix(Zs, names[0])
    if cols is not None and isinstance(Zt, pd.DataFrame):
        missing = [c for c in cols if c not in Zt.columns]
        if missing:
            raise KeyError(f"{names[1]} lacks the features {missing}")
        B, _, idx_b = _as_matrix(Zt[list(cols)], names[1])
    else:
        B, _, idx_b = _as_matrix(Zt, names[1])
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"{names[0]} has {A.shape[1]} features but {names[1]} has {B.shape[1]}")
    return A, B, cols, idx_a, idx_b


def _by_label(x: Any, labels: pd.Index | None) -> Any:
    """Return ``x``, reordered to follow ``labels`` when it is a Series whose index is a permutation of ``labels``.

    Any other input, including a Series with other labels, is returned unchanged and used by position.
    """
    if labels is None or not isinstance(x, pd.Series) or x.index.equals(labels):
        return x
    if x.index.is_unique and labels.is_unique and len(x) == len(labels) and set(x.index) == set(labels):
        return x.reindex(labels)
    return x


def _rescaled_weights(w: Any, d: int, names: pd.Index | None = None, name: str = "w") -> np.ndarray:
    """Validate feature weights and rescale them to sum to ``d``; ``name`` labels the weights in error messages.

    A weight of at most ``1e-12`` times the largest weight is rounding noise and is set to zero.
    """
    if isinstance(w, pd.Series) and names is not None:
        missing = [c for c in names if c not in w.index]
        if missing:
            raise KeyError(f"{name} lacks weights for the features {missing}")
        arr = _numeric(w.reindex(names), name)
    else:
        arr = _numeric(w, name)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if arr.size != d:
        raise ValueError(f"{name} has {arr.size} entries but there are {d} features")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"{name} sums to zero")
    arr = np.where(arr > _NOISE * arr.max(), arr, 0.0)
    return arr * (d / float(arr.sum()))


def _vector(x: Any, n: int, name: str, finite: bool = False, positive: bool = False) -> np.ndarray:
    """Return a one-dimensional float vector of length ``n``.

    The vector must be finite when ``finite`` is true and strictly positive and
    finite when ``positive`` is true.
    """
    arr = _numeric(x, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size != n:
        raise ValueError(f"{name} has {arr.size} entries, expected {n}")
    if (finite or positive) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{name} must be positive")
    return arr


def _mass_vector(x: Any, n: int, name: str, allow_zero_total: bool = False) -> np.ndarray:
    """Return non-negative masses of length ``n``; uniform ``1/n`` when ``x`` is ``None``.

    The total must be positive unless ``allow_zero_total`` is true.  A mass of at
    most ``1e-12`` times the largest mass is rounding noise and is set to zero.
    """
    if x is None:
        return np.full(n, 1.0 / n)
    arr = _vector(x, n, name)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    if arr.sum() <= 0 and not allow_zero_total:
        raise ValueError(f"{name} has zero total mass")
    if arr.size and arr.max() > 0:
        arr = np.where(arr > _NOISE * arr.max(), arr, 0.0)
    return arr


def _rho(value: float | None, name: str) -> float | None:
    """Return the penalty strength of a marginal, with ``None`` for an exactly enforced marginal."""
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or value <= 0:
        raise ValueError(f"{name} must be positive or None")
    return None if np.isinf(value) else value


def _check_eps(eps: float) -> float:
    """Return ``eps`` as a float after checking that it is positive and finite."""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    return eps


def _default_eps(Z: np.ndarray, wt: np.ndarray) -> tuple[float, bool]:
    """Return the default regularisation strength of the rows of ``Z`` and whether a fallback was used.

    The value is :func:`select_eps` at the median.  The flag is true when the
    median of the pairwise weighted distances is zero, so that the median of the
    positive distances, or 1, was used instead.  Fewer than two rows raise a
    ``ValueError``.
    """
    if Z.shape[0] < 2:
        raise ValueError("eps must be given when fewer than two sources are available")
    return _select_eps(Z, _rescaled_weights(wt, Z.shape[1]), 0.5)


def _per_case(items: Any, labels: pd.Index, name: str) -> list[Any]:
    """Return ``items`` as a list ordered like ``labels``; a mapping is looked up by label."""
    if isinstance(items, Mapping):
        missing = [lab for lab in labels if lab not in items]
        if missing:
            raise KeyError(f"{name} lacks the entries {missing}")
        return [items[lab] for lab in labels]
    out = list(items)
    if len(out) != len(labels):
        raise ValueError(f"{name} has {len(out)} entries, expected {len(labels)}")
    return out


def _group_codes(groups: Any, n: int | None = None, name: str = "groups") -> np.ndarray:
    """Return integer codes ``0, 1, ...`` of one-dimensional group labels, numbered in order of first appearance.

    Parameters
    ----------
    groups : array-like
        One scalar group label per case; missing labels are not allowed.
    n : int, optional
        Required number of labels.
    name : str
        Argument name used in error messages.

    Returns
    -------
    ndarray of int, shape (len(groups),)
        Code of the group of every case.
    """
    arr = np.asarray(groups)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of labels")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} has {arr.size} entries, expected {n}")
    if pd.isna(arr).any():
        raise ValueError(f"{name} must not contain missing values")
    codes, _ = pd.factorize(arr)
    return codes.astype(np.int64)


def _check_guard(guard: Callable[[], Any] | None) -> None:
    """Raise a ``TypeError`` unless ``guard`` is ``None`` or a callable."""
    if guard is not None and not callable(guard):
        raise TypeError("guard must be None or a callable that takes no arguments")


def _sq_cost(A: np.ndarray, B: np.ndarray, wt: np.ndarray, names: tuple[str, str] = ("Zs", "Zt")) -> np.ndarray:
    """Weighted squared distances between the rows of ``A`` and ``B`` for rescaled weights ``wt``.

    ``names`` are the argument names of ``A`` and ``B`` used in the error message
    raised when a feature with positive weight is not finite.
    """
    active = wt > 0
    sw = np.sqrt(wt[active])
    A_, B_ = A[:, active], B[:, active]
    if not np.isfinite(A_).all():
        raise ValueError(f"{names[0]} must be finite in the features with positive weight")
    if not np.isfinite(B_).all():
        raise ValueError(f"{names[1]} must be finite in the features with positive weight")
    return cdist(A_ * sw, B_ * sw, "sqeuclidean")


def _coordinate_size(wt: np.ndarray, *tables: np.ndarray) -> float:
    """Largest squared weighted coordinate ``w_k z_ik^2`` over the rows of ``tables`` and the features with weight.

    The rounding errors of differences between coordinates are proportional to this size.  The tables must
    be finite in the features with positive weight.
    """
    active = wt > 0
    size = 0.0
    for table in tables:
        if table.size and active.any():
            size = max(size, float(np.max(wt[active] * table[:, active] ** 2)))
    return size


# ----------------------------------------------------------------------------
# Standardisation and ground cost
# ----------------------------------------------------------------------------
def _median_and_scale(values: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Median and scaled median absolute deviation of every column, ignoring missing values."""
    centre = values.median()
    return centre, _MAD_TO_SD * (values - centre).abs().median()


def _unusable_scale(centre: pd.Series, scale: pd.Series) -> pd.Series:
    """Flag the scales that are missing, zero or at most ``1e-12`` times the absolute value of the centre."""
    return ~np.isfinite(scale) | (scale <= 1e-12 * centre.abs())


def robust_standardise(
    Xs: pd.DataFrame,
    Xt: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    clip: float | None = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Centre and scale features with a median and a scaled median absolute deviation.

    Each feature is centred at the median of ``reference`` and divided by
    ``1.4826`` times the median absolute deviation of ``reference``.  When that
    scale is missing, zero or at most ``1e-12`` times the absolute value of the
    centre, the feature takes the scale of the pooled sources ``Xs``, computed in
    the same way from the median and the deviations of ``Xs``; when that scale
    is unusable as well, the scale is 1.  A feature whose reference median is
    missing is centred at the median of ``Xs``.  Standardised values are clipped
    to the interval ``[-clip, clip]``.  Missing values in ``Xs`` and ``Xt`` stay
    missing; missing values in ``reference`` are ignored by the median.
    Another table is put on the same scale by passing it as ``Xs`` with the
    same ``reference``.

    Parameters
    ----------
    Xs : DataFrame
        Raw features of the sources, one row per source.
    Xt : DataFrame
        Raw features of the target points, one row per point.
    reference : DataFrame, optional
        Table whose columns define the centre and the scale, for example the
        global distribution of economy-years.  When ``None`` the rows of ``Xs``
        are used.
    columns : sequence of str, optional
        Features to use, in the order they are returned.  Defaults to the
        columns of ``Xs``.
    clip : float or None
        Positive bound of the standardised values; ``None`` leaves them unclipped.

    Returns
    -------
    Zs : DataFrame
        Standardised sources with the index of ``Xs`` and the selected columns.
    Zt : DataFrame
        Standardised target points with the index of ``Xt`` and the selected columns.
    scale : Series
        Scale applied to each feature, indexed by feature name.
    """
    Xs = _as_frame(Xs, "Xs")
    Xt = _as_frame(Xt, "Xt")
    cols = list(Xs.columns) if columns is None else list(columns)
    if len(set(cols)) != len(cols):
        raise ValueError("columns must not contain duplicates")
    ref = Xs if reference is None else _as_frame(reference, "reference")
    for label, table in (("Xs", Xs), ("Xt", Xt), ("reference", ref)):
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise KeyError(f"{label} lacks the columns {missing}")
    if clip is not None:
        clip = float(clip)
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be positive and finite, or None")
    tables = {}
    for label, table in (("Xs", Xs), ("Xt", Xt), ("reference", ref)):
        try:
            tables[label] = table[cols].astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if np.isinf(tables[label].to_numpy()).any():
            raise ValueError(f"{label} must not contain infinite values")
    if len(ref) == 0:
        raise ValueError("reference needs at least one row")
    centre, scale = _median_and_scale(tables["reference"])
    pooled_centre, pooled_scale = _median_and_scale(tables["Xs"])
    centre = centre.where(np.isfinite(centre), pooled_centre)
    scale = scale.mask(_unusable_scale(centre, scale), pooled_scale)
    scale = scale.mask(_unusable_scale(centre, scale), 1.0)
    scale.name = "scale"
    Zs = (tables["Xs"] - centre) / scale
    Zt = (tables["Xt"] - centre) / scale
    if clip is not None:
        Zs, Zt = Zs.clip(-clip, clip), Zt.clip(-clip, clip)
    return Zs, Zt, scale


def weighted_sq_cost(Zs: pd.DataFrame | ArrayLike, Zt: pd.DataFrame | ArrayLike, w: ArrayLike) -> np.ndarray:
    """Weighted squared distances between source rows and target rows.

    The entry ``(i, j)`` is ``sum_k w_k (zs_ik - zt_jk)^2``.  The weights must
    be non-negative and are rescaled internally to sum to ``d``, the number of
    features including those with zero weight, so that uniform weights give the
    plain squared Euclidean distance and a common factor in ``w`` has no effect.
    A feature with zero weight does not influence the result and may contain
    missing values; a weight of at most ``1e-12`` times the largest weight is
    rounding noise and counts as zero.  When both inputs are DataFrames the columns of ``Zt`` are
    matched to those of ``Zs`` by name, and a Series ``w`` is matched to the
    columns by name.

    Parameters
    ----------
    Zs : DataFrame or array, shape (n, d)
        Standardised features of the sources.
    Zt : DataFrame or array, shape (m, d)
        Standardised features of the target points.
    w : array-like or Series, length d
        Non-negative feature weights with a positive sum.

    Returns
    -------
    ndarray, shape (n, m)
        Cost matrix.
    """
    A, B, cols, _, _ = _pair_matrices(Zs, Zt)
    wt = _rescaled_weights(w, A.shape[1], cols)
    return _sq_cost(A, B, wt)


def select_eps(Z: pd.DataFrame | ArrayLike, w: ArrayLike, quantile: float = 0.5) -> float:
    """Entropic regularisation strength from the spread of a feature cloud.

    The value is ``0.1 * q``, where ``q`` is the requested quantile of the
    weighted squared distances ``C_ij = sum_k w_k (z_ik - z_jk)^2`` over all
    pairs of distinct rows ``i < j`` of ``Z``, with ``w`` rescaled to sum to the
    number of features as in :func:`weighted_sq_cost`.

    Many pairs coincide when the weight is concentrated on a feature that takes
    few values, for example a 0/1 indicator.  Two fallbacks keep the value
    positive.  When the requested quantile of the distances is zero, ``q`` is
    the same quantile of the positive distances.  When no pairwise distance is
    positive, all rows coincide in the weighted features and the function
    returns 1.0.  A distance of at most ``1e-18`` times the largest squared
    weighted coordinate ``w_k z_ik^2`` of the rows is rounding noise and counts
    as zero, so that multiplying the features by ``c`` multiplies the value by
    ``c^2`` (apart from the constant 1.0 of the second fallback).  The value is
    unchanged by the fallbacks whenever the requested quantile is positive.

    Weights that are close to, but not exactly, concentrated on a feature with
    few values give a small positive quantile of the distances and so a small
    ``eps``, with no fallback.  The Sinkhorn iterations of such a small ``eps``
    may not reach the tolerance within their iteration limit.  The solvers then
    solve the same problem by the log-domain rescue of :func:`sinkhorn_plan`,
    which converges for small ``eps``, and the callers report those solves as
    ``n_log_domain``; ``n_nonconverged`` counts the solves that do not reach the
    tolerance even after the rescue.

    Parameters
    ----------
    Z : DataFrame or array, shape (n, d)
        Standardised features, at least two rows.
    w : array-like or Series, length d
        Non-negative feature weights.
    quantile : float
        Quantile of the pairwise distances, between 0 and 1.

    Returns
    -------
    float
        The regularisation strength ``eps``, always positive.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie between 0 and 1")
    A, cols, _ = _as_matrix(Z, "Z")
    if A.shape[0] < 2:
        raise ValueError("Z needs at least two rows")
    wt = _rescaled_weights(w, A.shape[1], cols)
    return _select_eps(A, wt, quantile)[0]


def _select_eps(A: np.ndarray, wt: np.ndarray, quantile: float) -> tuple[float, bool]:
    """Regularisation strength of :func:`select_eps` for rescaled weights, and whether a fallback was used.

    Parameters
    ----------
    A : ndarray, shape (n, d)
        Features, at least two rows.
    wt : ndarray, shape (d,)
        Weights rescaled to sum to ``d``.
    quantile : float
        Quantile of the pairwise distances, between 0 and 1.

    Returns
    -------
    eps : float
        The value, positive.
    fallback : bool
        Whether the quantile of all distances was zero, so that the positive
        distances (or the constant 1) determined the value.
    """
    n = A.shape[0]
    dist = _sq_cost(A, A, wt, ("Z", "Z"))[np.triu_indices(n, k=1)]
    dist = np.where(dist <= _DIST_NOISE * _coordinate_size(wt, A), 0.0, dist)
    level = float(np.quantile(dist, quantile))
    if level > 0.0:
        return 0.1 * level, False
    positive = dist[dist > 0.0]
    if positive.size == 0:
        return 1.0, True
    return 0.1 * float(np.quantile(positive, quantile)), True


# ----------------------------------------------------------------------------
# Sinkhorn iterations
# ----------------------------------------------------------------------------
@dataclass(eq=False)
class Plan:
    """A transport plan between ``n`` sources and ``m`` target points.

    Attributes
    ----------
    pi : ndarray, shape (n, m)
        Transported mass from each source (rows) to each target point (columns).
    converged : bool
        Whether ``marginal_error`` fell below the tolerance.
    n_iter : int
        Number of scaling iterations performed.  When ``log_domain`` is true
        these are the iterations that did not reach the tolerance and preceded
        the rescue.
    marginal_error : float
        Largest L1 residual of the two marginal optimality conditions, divided
        by the total mass of the plan, so that it does not depend on the scale of
        the masses.  For an enforced marginal the residual is the distance
        between the plan marginal and the prescribed mass.  For a relaxed
        marginal it is the distance between the plan marginal and ``mass *
        exp(-f / rho)``, where ``f`` is the dual potential of that side.
    eps : float
        Entropic regularisation strength.
    rho_source : float or None
        Strength of the penalty on the source marginal, ``None`` when enforced.
    rho_target : float or None
        Strength of the penalty on the target marginal, ``None`` when enforced.
    source_index : Index, optional
        Labels of the sources, used by the functions that return pandas objects.
    target_index : Index, optional
        Labels of the target points.
    log_domain : bool
        True when the scaling iterations did not reach the tolerance within
        ``max_iter`` and the plan comes from the log-domain rescue (see the
        module header and :func:`sinkhorn_plan`), which reached it.  False for
        a plan of the scaling iterations, also for one that did not converge.
    mass : float
        Total transported mass (read-only property).
    """

    pi: np.ndarray = field(repr=False)
    converged: bool
    n_iter: int
    marginal_error: float
    eps: float
    rho_source: float | None
    rho_target: float | None
    source_index: pd.Index | None = field(default=None, repr=False)
    target_index: pd.Index | None = field(default=None, repr=False)
    log_domain: bool = False

    @property
    def mass(self) -> float:
        """Total transported mass."""
        return float(self.pi.sum())


def _sinkhorn_core(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    rho_s: float | None,
    rho_t: float | None,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Log-domain scaling iterations for strictly positive masses.

    Each iteration updates the log-scaling of the target side and then
    evaluates the marginal residuals of the state formed with the log-scaling
    of the source side from the previous iteration.  The source-side update
    follows when the relative residual is not below ``tol``.

    Parameters
    ----------
    a, b : ndarray
        Strictly positive source and target masses.
    C : ndarray, shape (len(a), len(b))
        Cost matrix.
    eps : float
        Regularisation strength.
    rho_s, rho_t : float or None
        Penalty strengths of the source and target marginals, ``None`` when enforced.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on the marginal residual relative to the total mass of the plan.

    Returns
    -------
    phi, psi : ndarray
        Log-scalings, so that the plan equals ``exp(log a + log b - C / eps + phi + psi)``.
    pi : ndarray
        The plan of the returned state.
    n_iter : int
        Number of iterations performed.
    err : float
        Marginal residual of the returned state divided by its total mass, as
        defined for ``Plan.marginal_error``; infinite when the state has no mass.
    """
    log_a = np.log(a)
    log_b = np.log(b)
    logK = log_a[:, None] + log_b[None, :] - C / eps
    fi_s = 1.0 if rho_s is None else rho_s / (rho_s + eps)
    fi_t = 1.0 if rho_t is None else rho_t / (rho_t + eps)
    phi = np.zeros(a.size)
    psi = np.zeros(b.size)
    err = np.inf
    n_iter = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for n_iter in range(1, max_iter + 1):
            A = logK + phi[:, None]
            top = A.max(axis=0)
            lse_c = top + np.log(np.exp(A - top).sum(axis=0))
            psi = fi_t * (log_b - lse_c)
            B = logK + psi
            top = B.max(axis=1)
            lse_r = top + np.log(np.exp(B - top[:, None]).sum(axis=1))
            row = np.exp(phi + lse_r)
            col = np.exp(psi + lse_c)
            row_ref = a if rho_s is None else a * np.exp(-(eps / rho_s) * phi)
            col_ref = b if rho_t is None else b * np.exp(-(eps / rho_t) * psi)
            total = row.sum()
            err = max(np.abs(row - row_ref).sum(), np.abs(col - col_ref).sum()) / total if total > 0 else np.inf
            if err < tol or n_iter == max_iter:
                break
            phi = fi_s * (log_a - lse_r)
        pi = np.exp(logK + phi[:, None] + psi[None, :])
    return phi, psi, pi, n_iter, float(err)


def _lower(err: float, best: float) -> bool:
    """Whether the residual ``err`` is smaller than ``best``; any finite residual beats a missing one."""
    return bool(err < best) or (bool(np.isnan(best)) and not bool(np.isnan(err)))


def _log_domain_rescue(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    rho_s: float | None,
    rho_t: float | None,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Plan of the transport problem of :func:`sinkhorn_plan` by Newton iterations on the dual, for small ``eps``.

    With the log-scalings ``phi`` of the sources and ``psi`` of the targets the
    plan is ``pi = exp(log a + log b - C / eps + phi + psi)`` and the dual
    objective divided by ``eps`` is ``T_s(phi) + T_t(psi) - sum_ij pi_ij``.
    ``T_s(phi)`` is ``sum_i a_i phi_i`` when the source marginal is enforced
    and ``-(rho_s / eps) sum_i a_i exp(-eps phi_i / rho_s)`` when it is
    relaxed, and likewise ``T_t(psi)``.  The objective is concave, and its
    gradient is the marginal residual of the optimality conditions that
    :func:`_sinkhorn_core` iterates on.

    The log-scalings of the larger side of the plan are eliminated: for given
    log-scalings of the smaller side they are the exact maximiser, which is the
    log-sum-exp scaling update.  The reduced objective of the smaller side has
    the marginal residual of that side as its gradient and a negative Hessian
    that is the sum of the Laplacian of the coupling ``pi' diag(1 / pi 1) pi``,
    a non-negative multiple of that coupling when the larger side is relaxed,
    and the diagonal of the penalty of the smaller side when it is relaxed.
    All three are formed without subtracting nearly equal numbers.  Newton
    steps with a Levenberg damping that grows until the dual objective does not
    fall (within rounding), and with every coordinate limited to
    ``_RESCUE_STEP_CAP``, update the log-scalings.

    The costs are shifted by their row minima and then their column minima, and
    the log-scalings are measured from the shifts divided by ``eps``, so the
    exponents that are evaluated stay of moderate size.  The problem is solved
    for a decreasing sequence of regularisation strengths that starts at the
    larger of ``eps`` and the largest cost, shrinks geometrically by a factor of
    at most ``_RESCUE_RATIOS[0]`` and ends at ``eps``; each stage starts from the
    dual potentials ``eps * psi`` (in cost units) of the previous one.  When the
    final stage does not reach ``tol``, the continuation is repeated with the
    next, finer, ratio of ``_RESCUE_RATIOS``, and the best result is returned.

    Parameters
    ----------
    a, b : ndarray
        Strictly positive source and target masses.
    C : ndarray, shape (len(a), len(b))
        Cost matrix.
    eps : float
        Regularisation strength.
    rho_s, rho_t : float or None
        Penalty strengths of the source and target marginals, ``None`` when enforced.
    tol : float
        Tolerance on the marginal residual relative to the total mass of the plan.

    Returns
    -------
    pi : ndarray, shape (len(a), len(b))
        The plan of the best state found.
    err : float
        Its marginal residual divided by its total mass, as defined for ``Plan.marginal_error``.
    n_steps : int
        Number of Newton steps taken in all stages and continuations.
    """
    if C.shape[1] > C.shape[0]:
        pi, err, n_steps = _log_domain_rescue(b, a, C.T, eps, rho_t, rho_s, tol)
        return pi.T, err, n_steps
    m = C.shape[1]
    log_a, log_b = np.log(a), np.log(b)
    row_min = C.min(axis=1)
    col_min = (C - row_min[:, None]).min(axis=0)
    shifted = C - row_min[:, None] - col_min[None, :]
    first = max(eps, float(C.max()))

    def newton(e: float, y: np.ndarray, accept: float, max_steps: int, polish: bool) -> tuple[dict[str, Any], int]:
        """Newton iterations at strength ``e`` from the column log-scalings ``y``; the best state and the steps taken."""
        kernel = log_a[:, None] + log_b[None, :] - shifted / e
        ks = 0.0 if rho_s is None else e / rho_s
        kt = 0.0 if rho_t is None else e / rho_t

        def state(yy: np.ndarray) -> dict[str, Any]:
            """Row log-scalings, plan, dual objective and marginal residual at the column log-scalings ``yy``."""
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                grid = kernel + yy[None, :]
                top = grid.max(axis=1)
                lse = top + np.log(np.exp(grid - top[:, None]).sum(axis=1))
                if rho_s is None:
                    xx = log_a - lse
                    row_ref = a
                    dual = float(a @ xx)
                    size = abs(dual)
                else:
                    xx = (rho_s / (rho_s + e)) * (log_a - row_min / rho_s - lse)
                    row_ref = np.exp(log_a - row_min / rho_s - ks * xx)
                    penalty = float(row_ref.sum()) / ks
                    dual = -penalty
                    size = penalty
                if rho_t is None:
                    col_ref = b
                    linear = float(b @ yy)
                    dual += linear
                    size += abs(linear)
                else:
                    col_ref = np.exp(log_b - col_min / rho_t - kt * yy)
                    penalty = float(col_ref.sum()) / kt
                    dual -= penalty
                    size += penalty
                pi = np.exp(grid + xx[:, None])
                row, col = pi.sum(axis=1), pi.sum(axis=0)
                total = float(row.sum())
                dual -= total
                size += total
                err = max(np.abs(row - row_ref).sum(), np.abs(col - col_ref).sum()) / total if total > 0 else np.inf
            return {
                "y": yy, "pi": pi, "row": row, "col": col, "col_ref": col_ref, "dual": dual, "size": size, "err": float(err)
            }

        current = best = state(y)
        taken = settled = 0
        damping = 1e-10
        while True:
            if _lower(current["err"], best["err"]):
                best = current
            if current["err"] < accept:
                settled += 1
                if not polish or current["err"] < 1e-3 * accept or settled > 2:
                    break
            if taken >= max_steps:
                break
            pi, col, col_ref = current["pi"], current["col"], current["col_ref"]
            residual = col_ref - col
            share = np.divide(pi, current["row"][:, None], out=np.zeros_like(pi), where=current["row"][:, None] > 0)
            coupling = pi.T @ share
            off = coupling - np.diag(np.diag(coupling))
            curvature = np.diag(kt * col_ref + off.sum(axis=1)) - off + (ks / (1.0 + ks)) * coupling
            scale = col + kt * col_ref
            live = scale > 0
            dead = np.flatnonzero(~live)
            passive = scale <= _RESCUE_PASSIVE * float(col.sum())
            accepted = False
            for _ in range(14):
                system = curvature + damping * np.diag(scale)
                system[dead, :] = 0.0
                system[:, dead] = 0.0
                system[dead, dead] = 1.0
                try:
                    step = np.linalg.solve(system, np.where(live, residual, 0.0))
                except np.linalg.LinAlgError:
                    step = None
                if step is not None and np.isfinite(step).all():
                    candidates = [np.clip(step, -_RESCUE_STEP_CAP, _RESCUE_STEP_CAP)]
                    if current["err"] < _RESCUE_WIDE_ERR:
                        wide = np.where(passive, candidates[0], np.clip(step, -_RESCUE_WIDE_CAP, _RESCUE_WIDE_CAP))
                        candidates.insert(0, wide)
                    for candidate in candidates:
                        trial = state(current["y"] + candidate)
                        if np.isfinite(trial["dual"]) and trial["dual"] >= current["dual"] - 1e-13 * current["size"]:
                            accepted = True
                            break
                    if accepted:
                        break
                damping *= 100.0
            if not accepted:
                break
            current = trial
            damping = max(0.1 * damping, 1e-12)
            taken += 1
        return best, taken

    n_steps = 0
    result = None
    for factor in _RESCUE_RATIOS:
        strengths = [first]
        ratio = min(factor, (eps / first) ** (1.0 / _RESCUE_MAX_STAGES))
        while strengths[-1] > eps:
            strengths.append(max(eps, strengths[-1] * ratio))
        y = np.zeros(m)
        for k, e in enumerate(strengths):
            last = k == len(strengths) - 1
            best, taken = newton(
                e, y, tol if last else _RESCUE_STAGE_TOL, _RESCUE_FINAL_STEPS if last else _RESCUE_STAGE_STEPS, last
            )
            n_steps += taken
            if not last:
                y = e * best["y"] / strengths[k + 1]
        if result is None or _lower(best["err"], result["err"]):
            result = best
        if best["err"] < tol:
            break
    return result["pi"], result["err"], n_steps


def _solve(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    rho_s: float | None,
    rho_t: float | None,
    max_iter: int,
    tol: float,
    rescue: bool = True,
) -> tuple[np.ndarray, bool, int, float, bool]:
    """Solve on the support of ``a`` and ``b``.

    The scaling iterations of :func:`_sinkhorn_core` run first.  When they do
    not reach ``tol`` within ``max_iter`` iterations, or produce non-finite
    values, and ``rescue`` is true, the same problem is solved by
    :func:`_log_domain_rescue` and the plan with the smaller marginal residual
    is returned.  Iterations that reach ``tol`` are returned as they are.

    Parameters
    ----------
    a, b : ndarray
        Non-negative source and target masses.
    C : ndarray, shape (len(a), len(b))
        Cost matrix.
    eps : float
        Regularisation strength.
    rho_s, rho_t : float or None
        Penalty strengths of the source and target marginals, ``None`` when enforced.
    max_iter : int
        Maximum number of scaling iterations.
    tol : float
        Tolerance on the marginal residual relative to the total mass of the plan.
    rescue : bool
        Whether to run the log-domain rescue when the iterations do not converge.

    Returns
    -------
    pi : ndarray, shape (len(a), len(b))
        The plan, zero outside the support.
    converged : bool
        Whether the relative residual fell below ``tol``.
    n_iter : int
        Number of scaling iterations performed.
    err : float
        Marginal residual of the returned plan relative to its total mass.
    rescued : bool
        Whether the returned plan comes from the rescue and reached ``tol``.
    """
    ia = np.flatnonzero(a > 0)
    jb = np.flatnonzero(b > 0)
    a_s, b_s = a[ia], b[jb]
    if rho_s is None and rho_t is None:
        b_s = b_s * (a_s.sum() / b_s.sum())
    C_s = C[np.ix_(ia, jb)]
    _, _, pi_s, n_iter, err = _sinkhorn_core(a_s, b_s, C_s, eps, rho_s, rho_t, max_iter, tol)
    finite = bool(np.isfinite(pi_s).all())
    rescued = False
    if rescue and not (finite and err < tol):
        pi_r, err_r, _ = _log_domain_rescue(a_s, b_s, C_s, eps, rho_s, rho_t, tol)
        if bool(np.isfinite(pi_r).all()) and (not finite or err_r < err):
            pi_s, err, finite = pi_r, err_r, True
            rescued = bool(err_r < tol)
    if not finite:
        raise ValueError("the scaling iterations produced non-finite values; increase eps or rho")
    if ia.size == a.size and jb.size == b.size:
        return pi_s, bool(err < tol), n_iter, err, rescued
    pi = np.zeros(C.shape)
    pi[np.ix_(ia, jb)] = pi_s
    return pi, bool(err < tol), n_iter, err, rescued


def sinkhorn_plan(
    a: ArrayLike,
    b: ArrayLike,
    C: ArrayLike,
    eps: float,
    rho_source: float | None = None,
    rho_target: float | None = None,
    max_iter: int = 5000,
    tol: float = 1e-9,
    rescue: bool = True,
) -> Plan:
    """Entropic optimal transport plan by log-domain Sinkhorn iterations, rescued for small ``eps``.

    The plan minimises ``<C, pi> + eps KL(pi | a b') + rho_s KL(pi 1 | a) +
    rho_t KL(pi' 1 | b)`` over non-negative matrices, with
    ``KL(p | q) = sum p log(p / q) - p + q``.  A marginal with ``rho = None``
    (or ``inf``) is enforced exactly instead of penalised.  A relaxed marginal
    uses the scaling update with exponent ``rho / (rho + eps)``.  Any
    combination of enforced and relaxed marginals is supported.  Entries of
    ``a`` or ``b`` equal to zero, or at most ``1e-12`` times the largest entry
    (rounding noise), receive no mass.  When both marginals are
    enforced their total masses must agree to a relative 1e-6, and ``b`` is
    rescaled to the total mass of ``a``.

    Small ``eps``.  A relaxed marginal makes the scaling iterations contract by
    the factor ``rho / (rho + eps)`` per iteration, which is close to one when
    ``eps`` is small next to ``rho``, so the iterations can end at ``max_iter``
    with a marginal residual above ``tol``.  When ``rescue`` is true and that
    happens, the same problem (the same ``eps``, ``rho`` and masses) is solved
    from scratch by Newton iterations on the dual objective in the log domain
    with a continuation in ``eps`` (see the module header), and the plan with
    the smaller marginal residual is returned.  The returned ``Plan`` has
    ``log_domain`` true when that plan comes from the rescue and reached
    ``tol``, and ``converged`` false only when neither the iterations nor the
    rescue reached ``tol``.  Iterations that reach ``tol`` are returned
    unchanged, bit for bit, whatever ``rescue`` is.  The rescue is accurate
    while the exponents ``C / eps`` are resolved in double precision, which
    bounds the attainable marginal residual below by about ``1e-16 * max(C) /
    eps``.

    Parameters
    ----------
    a : array-like, length n, or None
        Masses of the sources, uniform with total mass one when ``None``.  A
        Series supplies the source labels.
    b : array-like, length m, or None
        Masses of the target points, uniform with total mass one when ``None``.
        A Series supplies the target labels.
    C : array-like, shape (n, m)
        Finite ground cost.  A DataFrame supplies labels when ``a`` and ``b`` have none.
    eps : float
        Entropic regularisation strength, positive.
    rho_source : float, optional
        Strength of the KL penalty on the source marginal, ``None`` to enforce it.
    rho_target : float, optional
        Strength of the KL penalty on the target marginal, ``None`` to enforce it.
    max_iter : int
        Maximum number of scaling iterations before the rescue.
    tol : float
        Tolerance on ``Plan.marginal_error``, the marginal residual relative to
        the total mass of the plan.
    rescue : bool
        Whether to run the log-domain rescue when the scaling iterations do not
        reach ``tol`` within ``max_iter`` iterations.  With ``False`` the plan
        of the scaling iterations is returned whatever its residual.

    Returns
    -------
    Plan
        The plan and its convergence information.
    """
    C_arr = _numeric(C, "C")
    if C_arr.ndim != 2:
        raise ValueError("C must be two-dimensional")
    if not np.isfinite(C_arr).all():
        raise ValueError("C must be finite")
    a_arr = _mass_vector(a, C_arr.shape[0], "a")
    b_arr = _mass_vector(b, C_arr.shape[1], "b")
    eps = _check_eps(eps)
    rho_s = _rho(rho_source, "rho_source")
    rho_t = _rho(rho_target, "rho_target")
    if int(max_iter) < 1 or not tol > 0:
        raise ValueError("max_iter must be at least 1 and tol positive")
    if rho_s is None and rho_t is None:
        gap = abs(a_arr.sum() - b_arr.sum())
        if gap > 1e-6 * max(a_arr.sum(), b_arr.sum()):
            raise ValueError("balanced transport needs a and b with equal total mass")
    pi, converged, n_iter, err, rescued = _solve(
        a_arr, b_arr, C_arr, eps, rho_s, rho_t, int(max_iter), float(tol), rescue=bool(rescue)
    )
    if isinstance(a, pd.Series):
        source_index = a.index
    elif isinstance(C, pd.DataFrame):
        source_index = C.index
    else:
        source_index = None
    if isinstance(b, pd.Series):
        target_index = b.index
    elif isinstance(C, pd.DataFrame):
        target_index = C.columns
    else:
        target_index = None
    return Plan(
        pi=pi,
        converged=converged,
        n_iter=n_iter,
        marginal_error=err,
        eps=eps,
        rho_source=rho_s,
        rho_target=rho_t,
        source_index=source_index,
        target_index=target_index,
        log_domain=rescued,
    )


# ----------------------------------------------------------------------------
# Transported effects and noise
# ----------------------------------------------------------------------------
def _barycentric(pi: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass received by each target point and its barycentric effect.

    Parameters
    ----------
    pi : ndarray, shape (n, m)
        Plan.
    tau : ndarray, shape (n,)
        Source effects.

    Returns
    -------
    mass : ndarray, shape (m,)
        Column sums of the plan.
    effect : ndarray, shape (m,)
        ``sum_i pi_ij tau_i / sum_i pi_ij`` over the sources with positive usage,
        missing where the column sum is zero.
    """
    mass = pi.sum(axis=0)
    used = pi.sum(axis=1) > 0
    num = tau[used] @ pi[used]
    effect = np.full(mass.shape, np.nan)
    np.divide(num, mass, out=effect, where=mass > 0)
    return mass, effect


def _source_effects(tau: ArrayLike, pi: np.ndarray) -> np.ndarray:
    """Return ``tau`` as a vector with one entry per row of ``pi``.

    The entries of the sources that send mass must be finite; the others are
    never used.
    """
    arr = _vector(tau, pi.shape[0], "tau")
    if not np.all(np.isfinite(arr[pi.sum(axis=1) > 0])):
        raise ValueError("tau must be finite for every source that sends mass")
    return arr


def transported_effects(plan: Plan, tau: ArrayLike) -> pd.DataFrame:
    """Mass received and barycentric effect at each target point.

    The barycentric effect of target point ``j`` is
    ``sum_i pi_ij tau_i / sum_i pi_ij``.  It is missing when the point receives
    no mass.  Sources that send no mass are ignored.

    Parameters
    ----------
    plan : Plan
        Transport plan.
    tau : array-like, length n
        Effect of each source.

    Returns
    -------
    DataFrame
        One row per target point with the columns ``mass_received`` and ``effect``.
    """
    m = plan.pi.shape[1]
    mass, effect = _barycentric(plan.pi, _source_effects(tau, plan.pi))
    index = plan.target_index if plan.target_index is not None else pd.RangeIndex(m)
    return pd.DataFrame({"mass_received": mass, "effect": effect}, index=index)


def source_usage(plan: Plan) -> pd.Series:
    """Mass sent by each source, the row sums of the plan.

    Parameters
    ----------
    plan : Plan
        Transport plan.

    Returns
    -------
    Series
        Usage of each source, named ``usage``.
    """
    index = plan.source_index if plan.source_index is not None else pd.RangeIndex(plan.pi.shape[0])
    return pd.Series(plan.pi.sum(axis=1), index=index, name="usage")


def effective_sample_size(weights: ArrayLike) -> float:
    """Effective sample size ``(sum w)^2 / sum w^2`` of non-negative weights.

    The result does not depend on the scale of the weights, however large or
    small they are.

    Parameters
    ----------
    weights : array-like
        Non-negative weights.

    Returns
    -------
    float
        The effective sample size, 0 when all weights are zero.
    """
    w = _numeric(weights, "weights").ravel()
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    top = float(w.max()) if w.size else 0.0
    if top == 0.0:
        return 0.0
    scaled = w / top
    return float(scaled.sum() ** 2 / (scaled**2).sum())


def target_effect(plan: Plan, tau: ArrayLike, b: ArrayLike | None = None) -> float:
    """The ``b``-weighted mean of the barycentric effects over the target points.

    Target points with zero weight are ignored.  The result is missing when a
    point with positive weight receives no mass.

    Parameters
    ----------
    plan : Plan
        Transport plan.
    tau : array-like, length n
        Effect of each source.
    b : array-like, length m, optional
        Weights of the target points, uniform when ``None``.

    Returns
    -------
    float
        Transported effect for the target cloud.
    """
    m = plan.pi.shape[1]
    _, effect = _barycentric(plan.pi, _source_effects(tau, plan.pi))
    weights = _mass_vector(b, m, "b")
    use = weights > 0
    return float((weights[use] * effect[use]).sum() / weights[use].sum())


def weighted_quantile(values: ArrayLike, probs: ArrayLike, q: ArrayLike) -> float | np.ndarray:
    """Quantile of a weighted empirical distribution.

    The sorted values are placed at the cumulative probability of the mass
    below them plus half of their own mass, and the quantile is obtained by
    linear interpolation between these positions, with the extreme values
    returned beyond the first and last position.  With equal weights this is
    the Hazen quantile.  Points with zero weight or a missing value are dropped;
    infinite values are not allowed.

    Parameters
    ----------
    values : array-like, one-dimensional
        Support points.
    probs : array-like, one-dimensional
        Non-negative weights of the support points, not necessarily normalised.
    q : float or array-like
        Probability levels between 0 and 1.

    Returns
    -------
    float or ndarray
        Quantile for each level in ``q``; a float when ``q`` is a scalar.
    """
    v = _numeric(values, "values")
    p = _numeric(probs, "probs")
    if v.ndim != 1 or p.ndim != 1:
        raise ValueError("values and probs must be one-dimensional")
    if v.size != p.size:
        raise ValueError("values and probs must have the same length")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probs must be finite and non-negative")
    if np.isinf(v).any():
        raise ValueError("values must not contain infinite entries")
    keep = (p > 0) & np.isfinite(v)
    v, p = v[keep], p[keep]
    if v.size == 0:
        raise ValueError("no support point with positive weight")
    q_arr = _numeric(q, "q")
    if not np.all((q_arr >= 0) & (q_arr <= 1)):
        raise ValueError("q must lie between 0 and 1")
    order = np.argsort(v, kind="stable")
    v, p = v[order], p[order]
    cum = np.cumsum(p)
    position = (cum - 0.5 * p) / cum[-1]
    out = np.interp(q_arr, position, v)
    return float(out) if out.ndim == 0 else out


def transported_mixture(plan: Plan, draws_by_source: Sequence[ArrayLike] | Mapping[Any, ArrayLike]) -> tuple[np.ndarray, np.ndarray]:
    """Weighted empirical mixture of the per-source noise draws.

    Source ``i`` enters the mixture with probability proportional to
    :func:`source_usage`, spread evenly over its own draws.  Missing draws are
    ignored and infinite draws are not allowed; sources with no usage or no
    draw are left out.

    Parameters
    ----------
    plan : Plan
        Transport plan.
    draws_by_source : sequence of array-like, or mapping
        Draws (for example placebo effects) of each source, in source order, or
        a mapping from the source labels of the plan (integers when the plan has
        no labels) to the draws.

    Returns
    -------
    values : ndarray
        Pooled draws in increasing order.
    probs : ndarray
        Probability of each draw, summing to one.
    """
    usage = source_usage(plan)
    arrays = _per_case(draws_by_source, usage.index, "draws_by_source")
    values: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for u, draws in zip(usage.to_numpy(), arrays):
        d = _numeric(draws, "draws_by_source").ravel()
        if np.isinf(d).any():
            raise ValueError("draws_by_source must not contain infinite entries")
        d = d[np.isfinite(d)]
        if u > 0 and d.size > 0:
            values.append(d)
            probs.append(np.full(d.size, u / d.size))
    if not values:
        raise ValueError("no source with positive usage has finite draws")
    all_values = np.concatenate(values)
    all_probs = np.concatenate(probs)
    all_probs = all_probs / all_probs.sum()
    order = np.argsort(all_values, kind="stable")
    return all_values[order], all_probs[order]


# ----------------------------------------------------------------------------
# Distances between measures and overlap
# ----------------------------------------------------------------------------
def _exact_ot(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    """Optimal value of the balanced transport linear program.

    Parameters
    ----------
    a, b : ndarray
        Source and target masses with equal totals.
    C : ndarray, shape (len(a), len(b))
        Cost matrix.

    Returns
    -------
    float
        Minimum of ``<C, pi>`` over couplings ``pi`` of ``a`` and ``b``, from HiGHS.
    """
    n, m = C.shape
    rows = sparse.kron(sparse.identity(n), np.ones((1, m)), format="csr")
    cols = sparse.kron(np.ones((1, n)), sparse.identity(m), format="csr")
    res = optimize.linprog(
        C.ravel(),
        A_eq=sparse.vstack([rows, cols], format="csr"),
        b_eq=np.concatenate([a, b]),
        bounds=(0, None),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"the transport linear program failed: {res.message}")
    return float(res.fun)


def wasserstein2(a: ArrayLike | None, b: ArrayLike | None, Za: pd.DataFrame | ArrayLike, Zb: pd.DataFrame | ArrayLike, w: ArrayLike) -> float:
    """Exact 2-Wasserstein distance between two weighted clouds.

    The square root of the optimal value of the balanced transport linear
    program with the cost of :func:`weighted_sq_cost`, solved with
    ``scipy.optimize.linprog`` (HiGHS).  Both measures are normalised to unit
    mass, so the distance does not depend on the scale of ``a`` and ``b`` and
    their totals need not agree.  The result is NaN when ``a`` or ``b`` has
    zero total mass.  The cost is dense, so the routine is meant for problems
    with up to a few hundred points on each side.

    Parameters
    ----------
    a : array-like, length n, or None
        Non-negative masses of the points of ``Za``, uniform when ``None``.
    b : array-like, length m, or None
        Non-negative masses of the points of ``Zb``, uniform when ``None``.
    Za : DataFrame or array, shape (n, d)
        First cloud.
    Zb : DataFrame or array, shape (m, d)
        Second cloud.
    w : array-like, length d
        Non-negative feature weights.

    Returns
    -------
    float
        The distance, NaN when a measure has no mass.
    """
    A, B, cols, _, _ = _pair_matrices(Za, Zb, ("Za", "Zb"))
    C = _sq_cost(A, B, _rescaled_weights(w, A.shape[1], cols), ("Za", "Zb"))
    a_arr = _mass_vector(a, C.shape[0], "a", allow_zero_total=True)
    b_arr = _mass_vector(b, C.shape[1], "b", allow_zero_total=True)
    if a_arr.sum() <= 0 or b_arr.sum() <= 0:
        return float("nan")
    return float(np.sqrt(max(_exact_ot(a_arr / a_arr.sum(), b_arr / b_arr.sum(), C), 0.0)))


def _entropic_ot(a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float, max_iter: int, tol: float) -> tuple[float, bool]:
    """Entropic transport value between two probability vectors.

    Parameters
    ----------
    a, b : ndarray
        Probability vectors.
    C : ndarray, shape (len(a), len(b))
        Cost matrix.
    eps : float
        Regularisation strength.
    max_iter, tol : int, float
        Iteration budget and marginal tolerance.

    Returns
    -------
    value : float
        ``min <C, pi> + eps KL(pi | a b')`` over couplings, evaluated from the dual potentials.
    converged : bool
        Whether the marginal residual fell below ``tol``.
    """
    ia = np.flatnonzero(a > 0)
    jb = np.flatnonzero(b > 0)
    a_s, b_s = a[ia], b[jb]
    phi, psi, pi, _, err = _sinkhorn_core(a_s, b_s, C[np.ix_(ia, jb)], eps, None, None, max_iter, tol)
    return float(eps * (a_s @ phi + b_s @ psi) - eps * (pi.sum() - 1.0)), bool(err < tol)


def _entropic_ot_self(a: np.ndarray, C: np.ndarray, eps: float, max_iter: int, tol: float) -> tuple[float, bool]:
    """Entropic transport value of a probability vector coupled with itself.

    The cost must be symmetric.  The two dual potentials coincide, and the
    iteration replaces the potential by the average of itself and its Sinkhorn
    update.

    Parameters
    ----------
    a : ndarray
        Probability vector.
    C : ndarray, shape (len(a), len(a))
        Symmetric cost matrix.
    eps : float
        Regularisation strength.
    max_iter, tol : int, float
        Iteration budget and marginal tolerance.

    Returns
    -------
    value : float
        Same quantity as :func:`_entropic_ot` with both measures equal to ``a``.
    converged : bool
        Whether the marginal residual fell below ``tol``.
    """
    ia = np.flatnonzero(a > 0)
    a_s = a[ia]
    log_a = np.log(a_s)
    logK = log_a[:, None] + log_a[None, :] - C[np.ix_(ia, ia)] / eps
    phi = np.zeros(a_s.size)
    converged = False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            B = logK + phi
            top = B.max(axis=1)
            lse = top + np.log(np.exp(B - top[:, None]).sum(axis=1))
            if np.abs(np.exp(phi + lse) - a_s).sum() < tol:
                converged = True
                break
            phi = 0.5 * (phi + log_a - lse)
        mass = np.exp(logK + phi[:, None] + phi[None, :]).sum()
    return float(2.0 * eps * (a_s @ phi) - eps * (mass - 1.0)), converged


def sinkhorn_divergence(
    a: ArrayLike,
    b: ArrayLike,
    C_ab: ArrayLike,
    C_aa: ArrayLike,
    C_bb: ArrayLike,
    eps: float,
    max_iter: int = 20000,
    tol: float = 1e-12,
) -> float:
    """Sinkhorn divergence between two weighted clouds.

    The value is ``OT_eps(a, b) - 0.5 OT_eps(a, a) - 0.5 OT_eps(b, b)`` with
    ``OT_eps(a, b) = min <C, pi> + eps KL(pi | a b')`` over couplings of the
    two measures, evaluated from the dual potentials of balanced Sinkhorn
    iterations.  The masses ``a`` and ``b`` are normalised to sum to one.

    Parameters
    ----------
    a : array-like, length n, or None
        Masses of the first cloud, uniform when ``None``.
    b : array-like, length m, or None
        Masses of the second cloud, uniform when ``None``.
    C_ab : array-like, shape (n, m)
        Cost between the points of the two clouds.
    C_aa : array-like, shape (n, n)
        Cost between the points of the first cloud.
    C_bb : array-like, shape (m, m)
        Cost between the points of the second cloud.
    eps : float
        Entropic regularisation strength, positive.
    max_iter : int
        Maximum number of iterations of each of the three problems.
    tol : float
        Marginal tolerance of each of the three problems.

    Returns
    -------
    float
        The divergence.  A ``RuntimeWarning`` is issued when one of the three
        problems does not reach the tolerance within ``max_iter`` iterations.
    """
    cross_cost = _numeric(C_ab, "C_ab")
    if cross_cost.ndim != 2:
        raise ValueError("C_ab must be two-dimensional")
    a_arr = _mass_vector(a, cross_cost.shape[0], "a")
    b_arr = _mass_vector(b, cross_cost.shape[1], "b")
    a_arr = a_arr / a_arr.sum()
    b_arr = b_arr / b_arr.sum()
    n, m = a_arr.size, b_arr.size
    mats = []
    for name, mat, shape in (("C_ab", cross_cost, (n, m)), ("C_aa", C_aa, (n, n)), ("C_bb", C_bb, (m, m))):
        arr = _numeric(mat, name)
        if arr.shape != shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        mats.append(arr)
    eps = _check_eps(eps)
    coincide = (
        n == m
        and np.array_equal(a_arr, b_arr)
        and np.array_equal(mats[0], mats[1])
        and np.allclose(mats[0], mats[0].T, rtol=1e-10, atol=1e-12)
    )
    if coincide:
        cross, ok = _entropic_ot_self(a_arr, mats[0], eps, int(max_iter), float(tol))
    else:
        cross, ok = _entropic_ot(a_arr, b_arr, mats[0], eps, int(max_iter), float(tol))
    flags = [ok]
    terms = []
    for mass, mat in ((a_arr, mats[1]), (b_arr, mats[2])):
        if np.allclose(mat, mat.T, rtol=1e-10, atol=1e-12):
            value, ok = _entropic_ot_self(mass, mat, eps, int(max_iter), float(tol))
        else:
            value, ok = _entropic_ot(mass, mass, mat, eps, int(max_iter), float(tol))
        terms.append(value)
        flags.append(ok)
    if not all(flags):
        warnings.warn(
            "the Sinkhorn iterations of the divergence did not reach the tolerance; increase max_iter or eps",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(cross - 0.5 * terms[0] - 0.5 * terms[1])


def overlap_permutation_test(
    Zs: pd.DataFrame | ArrayLike,
    Zt: pd.DataFrame | ArrayLike,
    w: ArrayLike,
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
    n_perm: int = 499,
    seed: int | np.random.Generator = 0,
    pool: pd.DataFrame | ArrayLike | None = None,
) -> dict[str, Any]:
    """Permutation test of the overlap between a target cloud and the sources.

    The statistic is the ``b``-weighted mean over the target points of the
    weighted squared distance to the nearest source with positive weight.  A
    large value means that the target lies far from the sources.

    With ``pool`` given, the null distribution is the statistic of target
    clouds of the same size drawn at random from the rows of ``pool`` (without
    replacement when the pool is large enough), with the target weights
    attached in order.  With ``pool`` equal to ``None``, the rows of the
    sources and of the target are pooled and split at random into pseudo
    sources and a pseudo target of the original sizes, which gives the exact
    permutation test of the hypothesis that all rows come from one
    distribution.  The p-value is ``(1 + #{null >= statistic}) / (n_perm + 1)``,
    where a null value counts when it is at least the statistic less a relative
    ``1e-9`` (``1e-9`` times the larger of 1 and the statistic), so that null
    values equal to the statistic in exact arithmetic are counted whatever the
    rounding.

    The test uses the features it is given.  A feature left out of ``Zs`` and
    ``Zt`` (or carrying zero weight) does not enter the statistic, so a target
    that is far from the sources on that feature only is not detected.

    Parameters
    ----------
    Zs : DataFrame or array, shape (n, d)
        Standardised sources.
    Zt : DataFrame or array, shape (m, d)
        Standardised target points.
    w : array-like, length d
        Non-negative feature weights.
    a : array-like, length n, optional
        Source masses; sources with zero mass are ignored.
    b : array-like, length m, optional
        Weights of the target points, uniform when ``None``.
    n_perm : int
        Number of permutations.
    seed : int or numpy.random.Generator
        Seed of the random generator.
    pool : DataFrame or array, shape (p, d), optional
        Standardised global reference to draw null target clouds from.

    Returns
    -------
    dict
        ``statistic`` (float), ``p_value`` (float), ``null`` (ndarray of the
        ``n_perm`` null statistics) and ``n_perm`` (int).
    """
    if int(n_perm) < 1:
        raise ValueError("n_perm must be at least 1")
    A, B, cols, _, _ = _pair_matrices(Zs, Zt)
    n, m = A.shape[0], B.shape[0]
    wt = _rescaled_weights(w, A.shape[1], cols)
    a_vec = _mass_vector(a, n, "a")
    b_vec = _mass_vector(b, m, "b")
    b_norm = b_vec / b_vec.sum()
    A = A[a_vec > 0]
    if A.shape[0] == 0:
        raise ValueError("no source has positive mass")
    rng = np.random.default_rng(seed)
    statistic = float(b_norm @ _sq_cost(A, B, wt).min(axis=0))
    null = np.empty(int(n_perm))
    if pool is None:
        stacked = np.vstack([A, B])
        D = _sq_cost(stacked, stacked, wt)
        for s in range(null.size):
            perm = rng.permutation(stacked.shape[0])
            null[s] = b_norm @ D[np.ix_(perm[:m], perm[m:])].min(axis=1)
    else:
        if isinstance(pool, pd.DataFrame) and cols is not None:
            missing = [c for c in cols if c not in pool.columns]
            if missing:
                raise KeyError(f"pool lacks the features {missing}")
            P = pool[list(cols)].to_numpy(dtype=float)
        else:
            P, _, _ = _as_matrix(pool, "pool")
        if P.shape[1] != A.shape[1]:
            raise ValueError(f"pool has {P.shape[1]} features, expected {A.shape[1]}")
        nearest = _sq_cost(P, A, wt, ("pool", "Zs")).min(axis=1)
        for s in range(null.size):
            draw = rng.choice(P.shape[0], size=m, replace=P.shape[0] < m)
            null[s] = b_norm @ nearest[draw]
    at_least = null >= statistic - _TIE_TOL * max(1.0, abs(statistic))
    p_value = float((1 + np.count_nonzero(at_least)) / (null.size + 1))
    return {"statistic": statistic, "p_value": p_value, "null": null, "n_perm": int(null.size)}


# ----------------------------------------------------------------------------
# Support of the target
# ----------------------------------------------------------------------------
def _nonnegative(value: Any, name: str, upper: float | None = None) -> float:
    """Return ``value`` as a finite float that is at least 0 (and at most ``upper`` when given)."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not np.isfinite(out) or out < 0 or (upper is not None and out > upper):
        bound = "" if upper is None else f" and at most {upper:g}"
        raise ValueError(f"{name} must be finite, at least 0{bound}")
    return out


def target_support(
    Xs: pd.DataFrame | ArrayLike,
    Xt: pd.DataFrame | ArrayLike,
    w: ArrayLike,
    plan: Plan,
    reference: pd.DataFrame | None = None,
    min_ess: float = 2.0,
    min_weight: float = 0.10,
    range_tolerance: float = 0.10,
    clip: float | None = 5.0,
) -> dict[str, Any]:
    """Check that a target lies inside the support of the sources of a transport plan.

    A barycentric transport cannot exceed the largest observed effect of the
    sources, so a target that is costlier, larger or more concentrated than
    every case is an extrapolation that the method cannot make.  Two checks
    describe how far the target is from what the sources can represent.

    The first is the mass.  The usage of source ``i`` is the row sum ``u_i`` of
    the plan, and the effective number of sources behind the target is
    ``ess = (sum u)^2 / sum u^2`` (:func:`effective_sample_size` of the usage).
    It equals the number of sources with positive usage when all share the mass
    equally and approaches 1 when one source carries almost all of it.

    The second is the range of every feature.  A feature is *outside* when the
    target lies beyond ``[min - range_tolerance * range, max + range_tolerance
    * range]``, where ``min``, ``max`` and ``range`` are taken over the sources
    with positive usage (a usage above ``1e-12`` of the total).  For a target
    cloud with several rows the target value is the median of the rows.  The
    test is made on the raw feature values, so it is not hidden by the clipping
    of standardised values (the tables are standardised internally with
    :func:`robust_standardise`, without clipping, only to report the standardised
    value and whether it would be clipped).  A feature on which the sources
    have no range (relative range at most ``1e-9``) is outside when the target
    differs from the common value by more than ``1e-9`` times the larger of 1
    and the absolute value.  Comparisons use a slack of relative ``1e-9`` so
    that a target on the edge of the band is inside whatever the rounding.

    The target is *supported* when the effective number of sources is at least
    ``min_ess`` and no feature with a normalised weight of at least
    ``min_weight`` is outside.  A feature with a smaller weight may be outside
    without affecting the result, because it hardly enters the ground cost.
    The conformal and bootstrap intervals of this module describe the target
    only when it is supported.

    Parameters
    ----------
    Xs : DataFrame or array, shape (n, d)
        Raw (unstandardised) features of the sources, one row per row of the plan.
    Xt : DataFrame or array, shape (m, d)
        Raw features of the target points; a one-dimensional array is one
        point.  Two DataFrames are matched by column name and the columns of
        ``Xs`` are used.  When either table is an array, the columns are
        matched by position and carry the names of ``Xs`` (``0, 1, ...`` when
        ``Xs`` is an array).
    w : array-like or Series, length d
        Non-negative feature weights of the ground cost; normalised here to sum to one.
    plan : Plan
        Plan from the sources to the target, for example from :func:`sinkhorn_plan`.
    reference : DataFrame, optional
        Table whose median and scaled deviation standardise the features, as in
        :func:`robust_standardise`; the rows of ``Xs`` when ``None``.
    min_ess : float
        Smallest effective number of sources for which the target is supported.
    min_weight : float
        Smallest normalised weight of a feature whose range test must pass, between 0 and 1.
    range_tolerance : float
        Margin around the range of the sources, as a share of that range.
    clip : float or None
        Bound beyond which a standardised value counts as clipped in the
        notebooks; ``None`` flags no feature.

    Returns
    -------
    dict
        ``ess`` (effective number of sources), ``max_source_share`` (largest
        share of the total usage), ``n_sources`` (sources with positive usage),
        ``features`` (DataFrame with one row per feature and the columns
        ``weight``, ``target``, ``source_min``, ``source_max``, ``outside``,
        ``excess``, ``z_target`` and ``clipped``), ``weighted_outside_share``
        (sum of the weights of the outside features), ``ess_ok``, ``range_ok``,
        ``supported`` and ``reasons`` (plain sentences, empty when supported).
        ``excess`` is the distance from the target to the nearest edge of the
        band in units of the range of the sources (0 when inside; in units of
        the robust scale of the feature when the sources have no range; missing
        when the target or the sources have no value).  ``z_target`` is the
        robust standardised value without clipping and ``clipped`` is
        ``abs(z_target) > clip``.
    """
    if not isinstance(plan, Plan):
        raise TypeError("plan must be a Plan")
    min_ess = _nonnegative(min_ess, "min_ess")
    min_weight = _nonnegative(min_weight, "min_weight", upper=1.0)
    range_tolerance = _nonnegative(range_tolerance, "range_tolerance")
    if clip is not None:
        clip = float(clip)
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be positive and finite, or None")
    Xs_f = _as_frame(Xs, "Xs")
    if isinstance(Xs, pd.DataFrame) and isinstance(Xt, pd.DataFrame):
        Xt_f = Xt
    else:
        rows, _, _ = _as_matrix(Xt, "Xt")
        if rows.shape[1] != Xs_f.shape[1]:
            raise ValueError(f"Xs has {Xs_f.shape[1]} features but Xt has {rows.shape[1]}")
        Xt_f = pd.DataFrame(rows, columns=Xs_f.columns)
    cols = list(Xs_f.columns)
    n = len(Xs_f)
    if n == 0 or len(Xt_f) == 0:
        raise ValueError("Xs and Xt need at least one row")
    if plan.pi.shape[0] != n:
        raise ValueError(f"plan has {plan.pi.shape[0]} sources but Xs has {n} rows")
    wt = _rescaled_weights(w, len(cols), Xs_f.columns, "w")
    weight = wt / float(wt.sum())
    Zs, Zt, scale = robust_standardise(Xs_f, Xt_f, reference=reference, clip=None)

    usage = plan.pi.sum(axis=1)
    total = float(usage.sum())
    used = usage > _NOISE * total if total > 0 else np.zeros(n, dtype=bool)
    ess = effective_sample_size(usage)
    max_share = float(usage.max() / total) if total > 0 else float("nan")

    source_values = Xs_f[cols].astype(float).to_numpy()[used]
    target_rows = Xt_f[cols].astype(float).to_numpy()
    active = wt > 0
    if not np.isfinite(target_rows[:, active]).all():
        raise ValueError("Xt must be finite in the features with positive weight")
    if not np.isfinite(source_values[:, active]).all():
        raise ValueError("Xs must be finite in the features with positive weight for the sources with positive usage")
    source_frame = pd.DataFrame(source_values, columns=cols)
    lo = source_frame.min().to_numpy()
    hi = source_frame.max().to_numpy()
    target = pd.DataFrame(target_rows, columns=cols).median().to_numpy()
    span = hi - lo
    magnitude = np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
    slack = _TIE_TOL * magnitude
    flat = span <= slack
    with np.errstate(invalid="ignore", divide="ignore"):
        beyond = np.maximum((lo - range_tolerance * span) - target, target - (hi + range_tolerance * span))
        known = np.isfinite(beyond)
        outside = known & (beyond > slack)
        unit = np.where(flat, scale.to_numpy(dtype=float), span)
        excess = np.where(known, np.where(outside, beyond / unit, 0.0), np.nan)
    z_target = Zt.median().to_numpy(dtype=float)
    clipped = np.zeros(len(cols), dtype=bool) if clip is None else np.abs(z_target) > clip
    features = pd.DataFrame(
        {
            "weight": weight,
            "target": target,
            "source_min": lo,
            "source_max": hi,
            "outside": outside,
            "excess": excess,
            "z_target": z_target,
            "clipped": clipped,
        },
        index=pd.Index(cols, name="feature"),
    )

    ess_ok = bool(ess >= min_ess * (1.0 - _TIE_TOL))
    heavy = weight >= min_weight - _TIE_TOL
    failing = np.flatnonzero(outside & heavy)
    range_ok = failing.size == 0
    reasons: list[str] = []
    if not ess_ok:
        share = "" if not np.isfinite(max_share) else f"; the largest source carries {100.0 * max_share:.0f} percent of the mass"
        reasons.append(
            f"The effective number of sources behind the target is {ess:.2f}, below the minimum of {min_ess:g}{share}."
        )
    for k in failing[np.argsort(-weight[failing], kind="stable")]:
        if flat[k]:
            where = f"the sources have the single value {lo[k]:g} and the target has {target[k]:g}"
        else:
            where = f"{excess[k]:.2f} source ranges beyond the allowed band of the sources"
        reasons.append(
            f"The target lies outside the range of the sources on {cols[k]} (weight {weight[k]:.2f}): {where}, "
            f"standardised value {z_target[k]:.1f}."
        )
    return {
        "ess": float(ess),
        "max_source_share": max_share,
        "n_sources": int(used.sum()),
        "features": features,
        "weighted_outside_share": float(weight[outside].sum()),
        "ess_ok": ess_ok,
        "range_ok": range_ok,
        "supported": bool(ess_ok and range_ok),
        "reasons": reasons,
    }


# ----------------------------------------------------------------------------
# Leave-one-case-out and leave-one-group-out validation
# ----------------------------------------------------------------------------
def leave_group_out_indices(groups: ArrayLike) -> list[np.ndarray]:
    """Positions of the sources of every case when whole groups are left out.

    Parameters
    ----------
    groups : array-like, length n
        Group label of each case, for example the code of its economy.  Labels
        are scalars and may not be missing.

    Returns
    -------
    list of ndarray
        Element ``i`` holds, in increasing order, the positions of all cases
        whose group label differs from that of case ``i``.  The case itself and
        every other case of its group are absent.  The array is empty when all
        cases share one group.  When every case has its own label the element
        is the positions of all other cases.
    """
    codes = _group_codes(groups)
    return [np.flatnonzero(codes != code) for code in codes]


@dataclass(eq=False)
class LocoResult:
    """Result of :func:`loco_validation`.

    Attributes
    ----------
    table : DataFrame
        One row per case and method with the columns ``case``, ``method``,
        ``tau`` (observed effect), ``se`` (its standard error), ``prediction``,
        ``error`` (prediction minus observed effect), ``group`` (group label of
        the case, which is the case label when no groups were given) and
        ``n_sources_used`` (number of cases with positive mass outside the
        group of the case, which are the sources of every method).
    summary : DataFrame
        One row per method with ``rmse``, ``mae``, ``mean_error``,
        ``correlation`` (between prediction and observed effect), ``rank`` (1 for
        the smallest RMSE; RMSEs that differ by at most ``1e-9`` times the
        larger of the largest RMSE and the largest absolute observed effect are
        tied and share the smaller rank, so that methods that are all exact up
        to rounding share rank 1) and ``rmse_adj`` (square root of the mean
        squared error less the mean squared standard error, floored at zero).
    ratios : DataFrame
        One row per RMSE ratio (``ot_weighted / equal`` and ``ot_weighted /
        ot_uniform`` when both methods were run) with the sample ``ratio``,
        paired-bootstrap percentile intervals ``lo80``, ``hi80``, ``lo95``,
        ``hi95`` and ``share_below_one``, the share of resamples with a ratio
        below one.  The intervals are descriptive: they describe the cases at
        hand and are not a test of the route rule.  The frame carries this
        statement in ``ratios.attrs["note"]``.
    eps : dict
        Regularisation strength of each optimal-transport method: the value
        given to :func:`loco_validation`, or else the median over the folds of
        the value selected in each fold.
    rho_source : float or None
        Source-side penalty strength of the optimal-transport methods.
    n_boot : int
        Number of bootstrap resamples of cases, or of groups when groups were given.
    n_nonconverged : int
        Number of Sinkhorn solves that did not reach the tolerance, also after
        the log-domain rescue of :func:`sinkhorn_plan`.  A solve is counted
        here only when the scaling iterations and the rescue both stopped with
        a marginal residual above the tolerance.
    n_eps_fallback : int
        Number of fold solves of the optimal-transport methods in which the
        default regularisation strength came from a fallback of
        :func:`select_eps` (the median of the pairwise weighted distances of the
        fold was zero, so the median of the positive distances, or 1, was
        used).  It is zero when ``eps`` is given.
    n_log_domain : int
        Number of fold solves of the optimal-transport methods in which the
        scaling iterations did not reach the tolerance within their iteration
        limit and the log-domain rescue of :func:`sinkhorn_plan` solved the same
        problem to the tolerance.  Weights that are close to, but not exactly,
        concentrated on a feature with few values give a small default ``eps``
        (see :func:`select_eps`) for which the scaling iterations are slow, so
        the folds of such weights are counted here and not in
        ``n_nonconverged``.  The field is last, so that positional construction
        with the other fields still works.
    ratio_note : str
        Statement that the intervals of ``ratios`` are descriptive.

    Methods
    -------
    predictions()
        Predictions with one row per case and one column per method.
    errors()
        Prediction errors with one row per case and one column per method.
    """

    table: pd.DataFrame
    summary: pd.DataFrame
    ratios: pd.DataFrame
    eps: dict[str, float]
    rho_source: float | None
    n_boot: int
    n_nonconverged: int
    ratio_note: str = _RATIO_NOTE
    n_eps_fallback: int = 0
    n_log_domain: int = 0

    def _wide(self, column: str) -> pd.DataFrame:
        """Return ``column`` of the table with one row per case and one column per method."""
        cases = list(dict.fromkeys(self.table["case"]))
        methods = list(dict.fromkeys(self.table["method"]))
        wide = self.table.pivot(index="case", columns="method", values=column)
        wide.columns.name = None
        return wide.reindex(index=cases, columns=methods)

    def predictions(self) -> pd.DataFrame:
        """Predictions with one row per case and one column per method."""
        return self._wide("prediction")

    def errors(self) -> pd.DataFrame:
        """Prediction errors with one row per case and one column per method."""
        return self._wide("error")


def _loco_nn(C: np.ndarray, tau: np.ndarray, k: int, noise: float = 0.0) -> np.ndarray:
    """Mean effect of the ``k`` nearest sources at each target point.

    Parameters
    ----------
    C : ndarray, shape (n, m)
        Weighted squared distances from the sources to the target points.
    tau : ndarray, shape (n,)
        Source effects.
    k : int
        Number of neighbours, reduced to ``n`` when there are fewer sources.
    noise : float
        Squared distance at or below which a difference between two distances
        is rounding noise (see :func:`_coordinate_size`).

    Returns
    -------
    ndarray, shape (m,)
        Prediction at each target point.  Distances that agree to a relative
        ``1e-9``, or to within ``noise``, are ties, and ties are broken by
        source order.
    """
    k = min(k, C.shape[0])
    nearest = np.empty((k, C.shape[1]), dtype=np.intp)
    for j in range(C.shape[1]):
        order = np.argsort(C[:, j], kind="stable")
        ranked = C[order, j]
        new_block = np.diff(ranked) > _TIE_TOL * np.abs(ranked[1:]) + noise
        block = np.concatenate(([0], np.cumsum(new_block)))
        nearest[:, j] = order[np.lexsort((order, block))][:k]
    return tau[nearest].mean(axis=0)


def _kernel_bandwidth(dist: np.ndarray, size: float) -> float:
    """Bandwidth of the Gaussian kernel from the pairwise distances between the sources of a fold.

    The bandwidth is the median of ``dist``.  A distance of at most ``1e-9`` times ``sqrt(size)`` is rounding
    noise and counts as zero.  When the median is zero (more than half of the pairs coincide, as for weights
    concentrated on a 0/1 indicator) the bandwidth is the median of the positive distances.  When no distance
    is positive the sources coincide and every bandwidth that is large next to the noise gives them equal
    weights, so ``sqrt(size)`` is used (1 when all coordinates are zero).

    Parameters
    ----------
    dist : ndarray
        Weighted distances (not squared) between pairs of sources, or from the sources to the target points
        when there is one source.
    size : float
        Largest squared weighted coordinate of the sources (see :func:`_coordinate_size`).

    Returns
    -------
    float
        Positive bandwidth in distance units.
    """
    noise = float(np.sqrt(_DIST_NOISE * size))
    median = float(np.median(dist))
    if median > noise:
        return median
    positive = dist[dist > noise]
    if positive.size:
        return float(np.median(positive))
    return float(np.sqrt(size)) if size > 0.0 else 1.0


def _loco_kernel(C: np.ndarray, tau: np.ndarray, bandwidth: float) -> np.ndarray:
    """Nadaraya-Watson Gaussian-kernel mean effect at each target point.

    Parameters
    ----------
    C : ndarray, shape (n, m)
        Weighted squared distances from the sources to the target points.
    tau : ndarray, shape (n,)
        Source effects.
    bandwidth : float
        Positive bandwidth in distance units; the kernel weight is ``exp(-C / (2 bandwidth^2))``.

    Returns
    -------
    ndarray, shape (m,)
        Kernel-weighted mean effect at each target point.
    """
    logw = -C / (2.0 * bandwidth**2)
    weights = np.exp(logw - logw.max(axis=0))
    return (weights * tau[:, None]).sum(axis=0) / weights.sum(axis=0)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two vectors, missing when either is constant.

    A vector whose range is at most ``1e-12`` times its largest absolute value is constant.
    """
    if x.size < 2 or np.ptp(x) <= _NOISE * np.abs(x).max() or np.ptp(y) <= _NOISE * np.abs(y).max():
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank_with_tolerance(values: np.ndarray, scale: float = 0.0) -> np.ndarray:
    """Rank of every value, 1 for the smallest, with ties shared at the smallest rank.

    Values that agree to ``1e-9`` times the larger of the largest finite value and ``scale`` are ties; missing
    values rank last.  ``scale`` is the size of the quantity the values measure (for the RMSE of effect
    predictors, the largest absolute effect), so that values that are all rounding noise next to that size are
    ties.
    """
    present = values[~np.isnan(values)]
    finite = present[np.isfinite(present)]
    slack = _TIE_TOL * max(float(np.abs(finite).max()) if finite.size else 0.0, scale)
    return np.array([1 + (np.count_nonzero(present < v - slack) if not np.isnan(v) else present.size) for v in values], dtype=int)


def _rmse_ratio_table(
    squared_errors: dict[str, np.ndarray], n_boot: int, rng: np.random.Generator, codes: np.ndarray | None = None
) -> pd.DataFrame:
    """Paired bootstrap intervals for the RMSE ratios of ``ot_weighted`` to two comparators.

    Parameters
    ----------
    squared_errors : dict of ndarray
        Squared prediction error of every case, by method.
    n_boot : int
        Number of resamples; the same resample is used for every method.
    rng : numpy.random.Generator
        Random generator.
    codes : ndarray of int, optional
        Group code ``0, 1, ...`` of every case.  Whole groups are resampled with
        replacement and a resample has the mean squared error of the cases of
        its groups; when ``None`` every case is its own group.

    Returns
    -------
    DataFrame
        One row per available ratio, as described for ``LocoResult.ratios``.
        Resamples in which the ratio is undefined or infinite (a mean squared
        error of zero) are left out of the intervals, which are missing when no
        resample has a finite ratio.
    """
    columns = ["numerator", "denominator", "ratio", "lo80", "hi80", "lo95", "hi95", "share_below_one"]
    rows: dict[str, dict[str, Any]] = {}
    if "ot_weighted" in squared_errors:
        n = squared_errors["ot_weighted"].size
        if codes is None:
            codes = np.arange(n)
        n_groups = int(codes.max()) + 1
        idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
        count = np.bincount(codes, minlength=n_groups).astype(float)[idx].sum(axis=1)
        for other in ("equal", "ot_uniform"):
            if other not in squared_errors:
                continue
            total_w = np.bincount(codes, weights=squared_errors["ot_weighted"], minlength=n_groups)
            total_o = np.bincount(codes, weights=squared_errors[other], minlength=n_groups)
            with np.errstate(divide="ignore", invalid="ignore"):
                boot = np.sqrt(total_w[idx].sum(axis=1) / count) / np.sqrt(total_o[idx].sum(axis=1) / count)
                point = np.sqrt(squared_errors["ot_weighted"].mean() / squared_errors[other].mean())
            valid = boot[np.isfinite(boot)]
            lo95, lo80, hi80, hi95 = np.percentile(valid, [2.5, 10.0, 90.0, 97.5]) if valid.size else (np.nan,) * 4
            rows[f"ot_weighted / {other}"] = {
                "numerator": "ot_weighted",
                "denominator": other,
                "ratio": float(point),
                "lo80": float(lo80),
                "hi80": float(hi80),
                "lo95": float(lo95),
                "hi95": float(hi95),
                "share_below_one": float(np.mean(boot < 1.0 - _TIE_TOL)),
            }
    return pd.DataFrame(list(rows.values()), index=pd.Index(list(rows), name="comparison"), columns=columns)


def loco_validation(
    Z: pd.DataFrame | ArrayLike,
    tau: ArrayLike,
    se: ArrayLike | None,
    w: ArrayLike | Callable[[np.ndarray], ArrayLike],
    clouds: Sequence[pd.DataFrame | ArrayLike] | Mapping[Any, pd.DataFrame | ArrayLike] | None = None,
    eps: float | None = None,
    rho_source: float | None = 1.0,
    methods: Sequence[str] = _LOCO_METHODS,
    a: ArrayLike | None = None,
    seed: int | np.random.Generator = 0,
    n_boot: int = 500,
    groups: ArrayLike | None = None,
    guard: Callable[[], Any] | None = None,
) -> LocoResult:
    """Leave-one-case-out or leave-one-group-out validation of the effect predictors.

    For every case ``i`` the target is the single point ``Z[i]``, or the rows
    of ``clouds[i]`` (the yearly feature vectors of that case, equally
    weighted) when ``clouds`` is given, and the sources are all other cases.
    When ``groups`` is given the sources are all cases outside the group of
    case ``i`` (see :func:`leave_group_out_indices`); the other cases of its
    group, for example of the same economy, are not used.  The target stays the
    case ``i`` alone.  Cases with zero mass in ``a`` are never sources.  The
    predictors are

    ``ot_weighted``
        Transported (barycentric) effect for the target under the Sinkhorn plan
        from the sources, with the cost of the weights ``w``, the source
        marginal relaxed with ``rho_source`` and the target marginal enforced.
    ``ot_uniform``
        The same with uniform feature weights.
    ``equal``
        Unweighted mean effect of the sources.
    ``nn1``, ``nn3``
        Mean effect of the 1 or 3 nearest sources under the weighted distance.
    ``kernel``
        Nadaraya-Watson Gaussian-kernel mean of the sources, with a bandwidth
        equal to the median weighted distance (not squared) between pairs of
        sources.  Distances of at most ``1e-9`` times the largest weighted
        coordinate are rounding noise and count as zero.  When the median is
        zero (weights concentrated on a feature with few values) the bandwidth
        is the median of the positive distances, and when all sources coincide
        it is the largest weighted coordinate.

    For a cloud target the nearest-neighbour and kernel predictors are
    evaluated at each cloud point and averaged with equal weights.  Feature
    weights are rescaled to sum to the number of features as in
    :func:`weighted_sq_cost`.

    The weights ``w`` are either an array, fixed for all folds, or a function
    ``w(train_idx)`` that returns the weights learned from the training cases of
    the fold.  Fixed weights learned from the validated cases themselves carry
    information about the held-out effects into every prediction and make the
    validation optimistic; a function that learns its weights from
    ``train_idx`` only does not.  The function is called once per held-out
    group, in the order in which the groups first appear, with the integer
    positions in ``Z`` of the cases outside that group in increasing order, and
    must return finite, non-negative weights, not all zero, for the columns of
    ``Z`` (a Series is matched to the columns by name).  It is not called when
    none of ``ot_weighted``, ``nn1``, ``nn3`` and ``kernel`` is requested.

    When ``eps`` is ``None`` it is selected in every fold by :func:`select_eps`
    from the sources of the fold together with the rows of the held-out target,
    with the weights of the fold for ``ot_weighted`` and uniform weights for
    ``ot_uniform``.  When the weights of a fold concentrate on a feature that
    takes few values (a 0/1 indicator), more than half of the pairwise
    distances can be zero; :func:`select_eps` then falls back to the median of
    the positive distances, or to 1 when all rows coincide, and
    ``LocoResult.n_eps_fallback`` counts the fold solves in which this happened.
    Weights that are close to, but not exactly, concentrated on such a feature
    (a largest weight of 0.999 or more with small positive weights elsewhere)
    give a small positive default ``eps`` with no fallback.  The scaling
    iterations of such a fold may not reach the tolerance within 5000
    iterations; the fold is then solved by the log-domain rescue of
    :func:`sinkhorn_plan` with the same ``eps``, ``rho_source`` and masses, and
    ``LocoResult.n_log_domain`` counts those folds.  ``LocoResult.n_nonconverged``
    counts only the solves that do not reach the tolerance even after the rescue.

    What the validation measures.  Every fold predicts a case at its own
    features from the other cases (or the cases outside its group), so the
    errors are leave-one-economy-out errors of cases at the features of cases:
    they measure interpolation among the cases.  They do not measure the error
    of a target that lies outside the cases, and the transport cannot predict
    an effect larger than the largest source effect.  Intervals built from
    these errors (:func:`loco_interval`, :func:`loco_predictive_draws`)
    describe a target only inside the support checked by
    :func:`target_support`.

    The paired bootstrap resamples cases with replacement, or whole groups when
    ``groups`` is given, the same cases for every method; a resample has the
    mean squared error of its cases.  The resulting intervals of the RMSE
    ratios are descriptive: they describe the cases at hand and are not a test
    of the route rule.

    The per-case arguments ``tau``, ``se``, ``a`` and ``groups`` are used by
    position, except that a Series whose index is a permutation of the row
    labels of a DataFrame ``Z`` is first put in the order of those labels.

    Parameters
    ----------
    Z : DataFrame or array, shape (n, d)
        Standardised features of the cases.  The index of a DataFrame labels the cases.
    tau : array-like, length n
        Estimated effect of each case, finite.
    se : array-like, length n, or None
        Positive standard error of each estimated effect.
    w : array-like of length d, or callable
        Non-negative feature weights fixed for all folds (weights learned from
        the validated cases make the validation optimistic), or a function
        ``w(train_idx)`` from the integer positions of the training cases of a
        fold to weights learned from those cases only.
    clouds : sequence or mapping of arrays, optional
        Yearly feature vectors of each case, shape ``(m_i, d)``, in the column
        order of ``Z``.  A mapping is keyed by the case labels.
    eps : float, optional
        Entropic regularisation strength of the optimal-transport methods,
        selected in every fold when ``None``.
    rho_source : float or None
        Strength of the penalty on the source marginal; ``None`` enforces it.
    methods : sequence of str
        Predictors to evaluate, a subset of ``ot_weighted``, ``ot_uniform``,
        ``equal``, ``nn1``, ``nn3`` and ``kernel``.
    a : array-like, length n, optional
        Masses of the cases as sources; uniform when ``None``.  They are
        renormalised to sum to one among the sources of each fold.
    seed : int or numpy.random.Generator
        Seed of the bootstrap.
    n_boot : int
        Number of bootstrap resamples of cases, or of groups when ``groups`` is given.
    groups : array-like, length n, optional
        Group label of each case in the row order of ``Z``, for example the
        economy code.  Every fold leaves out all cases with the label of the
        held-out case.  When ``None`` every case is its own group.
    guard : callable, optional
        Function without arguments, called once at the start of the fold of
        each held-out case, ``n`` times in all.  Its return value is ignored and
        an exception it raises ends the validation.

    Returns
    -------
    LocoResult
        Predictions and errors, summary statistics and descriptive bootstrap intervals.
    """
    methods = tuple(methods)
    unknown = [m for m in methods if m not in _LOCO_METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(_LOCO_METHODS)}")
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("methods must be a non-empty list of distinct names")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    _check_guard(guard)
    Zv, cols, row_labels = _as_matrix(Z, "Z", allow_row=False)
    n, d = Zv.shape
    if n < 2:
        raise ValueError("Z needs at least two cases")
    labels = row_labels if row_labels is not None else pd.RangeIndex(n)
    if labels.has_duplicates:
        raise ValueError("case labels must be unique")
    tau_v = _vector(_by_label(tau, row_labels), n, "tau", finite=True)
    se_v = None if se is None else _vector(_by_label(se, row_labels), n, "se", positive=True)
    a_v = _mass_vector(_by_label(a, row_labels), n, "a")
    groups = _by_label(groups, row_labels)
    if groups is None:
        group_labels = list(labels)
        codes = np.arange(n)
        sources = leave_group_out_indices(codes)
    else:
        group_values = np.asarray(groups)
        codes = _group_codes(group_values, n)
        group_labels = group_values.tolist()
        sources = leave_group_out_indices(group_values)
        if any(src.size == 0 for src in sources):
            raise ValueError("groups needs at least two distinct labels")
    if any(a_v[src].sum() <= 0 for src in sources):
        if groups is None:
            raise ValueError("a needs positive mass on at least one other case for every case")
        raise ValueError("a needs positive mass on at least one case outside the group of every case")
    rho_s = _rho(rho_source, "rho_source")
    learn_weights = w if callable(w) else None
    fixed_wt = None if learn_weights is not None else _rescaled_weights(w, d, cols, "w")
    needs_weights = any(m in _WEIGHTED_METHODS for m in methods)
    uniform = np.ones(d)
    fixed_eps = None if eps is None else _check_eps(eps)
    use_ot = [m for m in methods if m in ("ot_weighted", "ot_uniform")]
    if clouds is None:
        cloud_list = [Zv[[i]] for i in range(n)]
    else:
        cloud_list = []
        for item in _per_case(clouds, labels, "clouds"):
            if isinstance(item, pd.DataFrame) and cols is not None:
                item = item[list(cols)]
            arr, _, _ = _as_matrix(item, "clouds")
            if arr.shape[0] < 1 or arr.shape[1] != d:
                raise ValueError(f"each entry of clouds needs at least one row and {d} columns")
            cloud_list.append(arr)
    fold_weights: dict[int, np.ndarray] = {}
    fold_bandwidth: dict[int, float] = {}
    fold_eps: dict[str, list[float]] = {m: [] for m in use_ot}
    pred = {m: np.full(n, np.nan) for m in methods}
    n_used = np.zeros(n, dtype=int)
    n_nonconverged = 0
    n_eps_fallback = 0
    n_log_domain = 0
    for i in range(n):
        if guard is not None:
            guard()
        outside = sources[i]
        src = outside[a_v[outside] > 0]
        n_used[i] = src.size
        code = int(codes[i])
        wt = fixed_wt
        if learn_weights is not None and needs_weights:
            if code not in fold_weights:
                fold_weights[code] = _rescaled_weights(learn_weights(outside.copy()), d, cols, "w(train_idx)")
            wt = fold_weights[code]
        tau_src = tau_v[src]
        target = cloud_list[i]
        b_i = np.full(target.shape[0], 1.0 / target.shape[0])
        a_src = a_v[src] / a_v[src].sum()
        pooled = np.vstack([Zv[src], target]) if fixed_eps is None and use_ot else None
        C_w = _sq_cost(Zv[src], target, wt, ("Z", "clouds")) if wt is not None else None
        for method in methods:
            if method == "equal":
                pred[method][i] = tau_src.mean()
            elif method in ("nn1", "nn3"):
                noise = _DIST_NOISE * _coordinate_size(wt, Zv[src], target)
                pred[method][i] = _loco_nn(C_w, tau_src, 1 if method == "nn1" else 3, noise).mean()
            elif method == "kernel":
                if code not in fold_bandwidth:
                    pair = _sq_cost(Zv[src], Zv[src], wt, ("Z", "Z"))
                    dist = np.sqrt(pair[np.triu_indices(src.size, k=1)]) if src.size > 1 else np.sqrt(C_w.ravel())
                    fold_bandwidth[code] = _kernel_bandwidth(dist, _coordinate_size(wt, Zv[src]))
                pred[method][i] = _loco_kernel(C_w, tau_src, fold_bandwidth[code]).mean()
            else:
                weights_m = wt if method == "ot_weighted" else uniform
                C_m = C_w if method == "ot_weighted" else _sq_cost(Zv[src], target, uniform, ("Z", "clouds"))
                if fixed_eps is not None:
                    eps_m = fixed_eps
                else:
                    eps_m, fell_back = _default_eps(pooled, weights_m)
                    n_eps_fallback += int(fell_back)
                fold_eps[method].append(eps_m)
                pi, converged, _, _, rescued = _solve(a_src, b_i, C_m, eps_m, rho_s, None, 5000, 1e-9)
                n_nonconverged += int(not converged)
                n_log_domain += int(rescued)
                _, effect = _barycentric(pi, tau_src)
                pred[method][i] = float(effect.mean())
    records = []
    for method in methods:
        for i in range(n):
            records.append(
                {
                    "case": labels[i],
                    "method": method,
                    "tau": tau_v[i],
                    "se": np.nan if se_v is None else se_v[i],
                    "prediction": pred[method][i],
                    "error": pred[method][i] - tau_v[i],
                    "group": group_labels[i],
                    "n_sources_used": int(n_used[i]),
                }
            )
    table = pd.DataFrame.from_records(
        records, columns=["case", "method", "tau", "se", "prediction", "error", "group", "n_sources_used"]
    )
    squared = {m: (pred[m] - tau_v) ** 2 for m in methods}
    noise = np.nan if se_v is None else float(np.mean(se_v**2))
    summary = pd.DataFrame(
        {
            "rmse": [float(np.sqrt(squared[m].mean())) for m in methods],
            "mae": [float(np.abs(pred[m] - tau_v).mean()) for m in methods],
            "mean_error": [float((pred[m] - tau_v).mean()) for m in methods],
            "correlation": [_correlation(pred[m], tau_v) for m in methods],
        },
        index=pd.Index(list(methods), name="method"),
    )
    summary["rank"] = _rank_with_tolerance(summary["rmse"].to_numpy(), float(np.abs(tau_v).max()))
    summary["rmse_adj"] = [float(np.sqrt(max(squared[m].mean() - noise, 0.0))) if np.isfinite(noise) else np.nan for m in methods]
    rng = np.random.default_rng(seed)
    ratios = _rmse_ratio_table(squared, int(n_boot), rng, codes)
    ratios.attrs["note"] = _RATIO_NOTE
    eps_by_method = {m: fixed_eps if fixed_eps is not None else float(np.median(fold_eps[m])) for m in use_ot}
    return LocoResult(
        table=table,
        summary=summary,
        ratios=ratios,
        eps=eps_by_method,
        rho_source=rho_s,
        n_boot=int(n_boot),
        n_nonconverged=n_nonconverged,
        ratio_note=_RATIO_NOTE,
        n_eps_fallback=n_eps_fallback,
        n_log_domain=n_log_domain,
    )


def _loco_errors(table: Any) -> np.ndarray:
    """Return the validation errors of a table of one method, one per row.

    Parameters
    ----------
    table : DataFrame
        Rows of ``LocoResult.table`` for one method, with the column ``error``.

    Returns
    -------
    ndarray, shape (len(table),)
        The finite errors (prediction minus observed effect).
    """
    if not isinstance(table, pd.DataFrame) or "error" not in table.columns:
        raise ValueError("table must be a DataFrame with the column 'error'")
    if "method" in table.columns and table["method"].nunique() > 1:
        raise ValueError("table must hold the rows of one method")
    n_cases = len(table)
    if n_cases == 0:
        raise ValueError("table has no rows")
    return _vector(table["error"], n_cases, "table['error']", finite=True)


def _check_level(level: Any) -> float:
    """Return ``level`` as a float strictly between 0 and 1."""
    try:
        value = float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError("level must be a number strictly between 0 and 1") from exc
    if not 0.0 < value < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    return value


def _conformal_rank(n: int, level: float) -> int:
    """Order statistic ``ceil(level * (n + 1))`` of ``n`` scores used by split conformal prediction.

    A product that exceeds an integer by at most ``1e-9`` counts as that integer, so that the
    representation error of ``level`` (``0.55 * 100`` is ``55.00000000000001``) does not change the rank.
    The arithmetic is exact, so the rank does not depend on the rounding of the machine.
    """
    return max(math.ceil(Fraction(level) * (n + 1) - Fraction(_TIE_TOL)), 1)


def _errors_needed(level: float) -> int:
    """Smallest number of scores for which :func:`_conformal_rank` does not exceed the number of scores.

    The rank does not exceed ``n`` exactly when ``n >= (level - 1e-9) / (1 - level)``; the bound is computed
    in exact arithmetic, so the cost does not grow with the answer.
    """
    level = Fraction(level)
    return max(math.ceil((level - Fraction(_TIE_TOL)) / (1 - level)), 1)


def _interval_from_errors(errors: np.ndarray, estimate: float, level: float) -> dict[str, Any]:
    """Conformal interval of :func:`loco_interval` from validated errors, a finite estimate and a valid level."""
    n = int(errors.size)
    rank = _conformal_rank(n, level)
    out: dict[str, Any] = {
        "lower": float("nan"),
        "upper": float("nan"),
        "half_width": float("nan"),
        "n": n,
        "rank": rank,
        "level": level,
        "guaranteed_level": float("nan"),
        "status": "ok",
    }
    if rank > n:
        needed = _errors_needed(level)
        out["status"] = (
            f"No interval exists at level {level!r}: at least {needed} validation errors are needed "
            f"and the table has {n}."
        )
        return out
    half_width = float(np.sort(np.abs(errors))[rank - 1])
    out.update(
        lower=estimate - half_width,
        upper=estimate + half_width,
        half_width=half_width,
        guaranteed_level=rank / (n + 1),
    )
    return out


def loco_interval(table: pd.DataFrame, estimate: float, level: float = 0.90) -> dict[str, Any]:
    """Conformal interval for the effect observed in a new study at the target.

    This is split conformal prediction with the leave-one-out errors as the
    calibration scores.  The ``error`` column of the table of
    :func:`loco_validation` holds the prediction of each case, made without
    that case (or without its group), minus the effect observed for the case.
    With ``n`` such errors the interval is ``estimate -/+ q``, where ``q`` is
    the ``rank``-th smallest absolute error and ``rank = ceil(level * (n +
    1))``.  If the error of a new study at the target is exchangeable with the
    ``n`` validation errors, the interval contains the effect observed in that
    study with probability at least ``rank / (n + 1)``, which is
    ``guaranteed_level``, and exactly ``rank / (n + 1)`` when the absolute
    errors have no ties.  The guarantee holds in finite samples and needs no
    model for the errors.  ``guaranteed_level`` is at least ``level``; it
    exceeds it because ``level * (n + 1)`` is rounded up to an integer.

    The interval is for the effect *observed* in a new study, not for the
    underlying effect, because the errors contain the measurement noise of the
    validated cases.  The estimation error of the transport enters through the
    validation errors, each of which is the error of an estimate built from the
    other cases.

    Three intervals can be produced by this module and each has its own use.
    The bootstrap interval of the transported average effect
    (``bootstrap_transport(...)["draws"]``) describes the sampling uncertainty
    of the transported average effect and is not validated for a new study.
    The predictive draws of :func:`bootstrap_transport` add a normal
    between-case deviation and measurement noise and are not validated either.
    This interval is validated by construction for the observed effect of a new
    study at the target.

    Limits.  The table needs one error per case and is a table for one method;
    every row of the table counts as one validation error.  Cases of one
    economy are dependent, so when an economy has several cases the
    exchangeability assumption holds only approximately and so does the
    guarantee.  Leave-one-out errors are also computed from overlapping sets of
    training cases, so they are exchangeable only approximately.  The
    validation errors are errors of cases predicted at their
    own features (interpolation among the cases).  The interval therefore
    describes a target that resembles the cases, as checked by
    :func:`target_support`, and it does not widen for a target far from the
    cases; the transport cannot predict an effect larger than the largest
    source effect.

    Parameters
    ----------
    table : DataFrame
        Rows of ``LocoResult.table`` for one method, with the column ``error``
        and, optionally, the column ``method``.
    estimate : float
        Finite point estimate of the effect for the target.
    level : float
        Nominal coverage, strictly between 0 and 1.

    Returns
    -------
    dict
        ``lower`` and ``upper`` (the bounds), ``half_width`` (``q``), ``n`` (the
        number of validation errors), ``rank`` (the order statistic used),
        ``level``, ``guaranteed_level`` (``rank / (n + 1)``) and ``status``.
        ``status`` is ``"ok"`` when the interval exists.  It exists when
        ``rank <= n``, which needs at least 9 errors for ``level=0.90``.
        Otherwise the bounds, ``half_width`` and ``guaranteed_level`` are NaN
        and ``status`` is a sentence that gives the number of errors needed.
    """
    errors = _loco_errors(table)
    estimate = float(estimate)
    if not np.isfinite(estimate):
        raise ValueError("estimate must be finite")
    return _interval_from_errors(errors, estimate, _check_level(level))


def loco_predictive_draws(
    table: pd.DataFrame,
    estimate: float,
    groups: ArrayLike | None = None,
    n_draws: int = 2000,
    seed: int | np.random.Generator = 0,
    level: float | None = 0.90,
) -> dict[str, Any]:
    """Predictive draws of the observed effect of an economy that was not in the sample.

    The ``error`` column of the table of :func:`loco_validation` holds the
    prediction of each case made without any case of its own group, minus the
    effect observed for the case.  A draw is the point ``estimate`` for the
    target minus one such error, that is the estimate plus an observed-minus-
    predicted difference.  For each draw the groups are resampled with
    replacement, as many as there are groups, and one error is picked at random
    among the cases of the resampled groups.  The draws therefore carry the
    cost of transporting an effect to an economy that was not in the sample,
    including the dispersion of that cost across groups and its systematic
    bias, and they do not use the uncertainty of the estimate itself.

    Calibration.  The percentiles of the raw draws are essentially the range of
    the validation errors, so their coverage of a new study falls short of the
    nominal one when there are few errors.  With ``level`` given, the interval
    of :func:`loco_interval` is computed from the errors of the table (one per
    row).  When it exists, the draws are rescaled about the estimate by ``scale
    = q / (the level-quantile of the absolute raw draws about the estimate)``,
    where ``q`` is the half width of the interval, so that the symmetric
    interval of the rescaled draws about the estimate at ``level`` equals the
    conformal interval.  When the interval does not exist (too few errors), the
    draws are unchanged and ``calibrated`` is False.  With ``level=None`` no
    interval is computed and the draws are the raw ones.

    Only ``interval`` has a coverage guarantee.  The ``percentiles`` are
    descriptive percentiles of the (rescaled) draws and have no coverage
    guarantee: the 5 and 95 percent points are equal-tail points of the draws,
    not the ends of the symmetric conformal interval, and when the validation
    errors are biased (their mean is far from zero) they can cover less than
    the percentiles of the raw draws.  Statements about coverage of a new
    study must use ``interval``.

    Three intervals can be produced by this module and each has its own use.
    The bootstrap interval of the transported average effect
    (``bootstrap_transport(...)["draws"]``) describes the sampling uncertainty
    of the transported average effect.  The predictive draws of
    :func:`bootstrap_transport` add a normal between-case deviation and
    measurement noise and are not validated.  The conformal interval of this
    function (``interval``) is validated by construction for the observed
    effect of a new study at the target.

    What the draws describe.  The validation errors are leave-one-economy-out
    errors of cases predicted at their own features (interpolation among the
    cases).  The transport cannot predict an effect larger than the largest
    source effect, and the draws and the interval describe the target only
    inside the support checked by :func:`target_support`; they do not widen for
    a target far from the cases.

    Parameters
    ----------
    table : DataFrame
        Rows of ``LocoResult.table`` for one method, with the column ``error``
        and, optionally, the columns ``group`` and ``method``.
    estimate : float
        Finite point estimate of the effect for the target.
    groups : array-like, length len(table), optional
        Group label of each row of ``table``.  Defaults to the column ``group``
        of the table when it has one, and otherwise every row is its own group.
    n_draws : int
        Number of draws.
    seed : int or numpy.random.Generator
        Seed of the random generator.
    level : float or None
        Nominal coverage of the conformal interval and of the calibration,
        strictly between 0 and 1; ``None`` returns the raw draws.

    Returns
    -------
    dict
        ``draws`` (ndarray of length ``n_draws``), ``percentiles`` (Series of the
        5, 25, 50, 75 and 95 percent points of the draws, descriptive and
        without a coverage guarantee), ``estimate``,
        ``n_draws``, ``n_groups`` and ``n_cases``; ``interval`` (the dict of
        :func:`loco_interval`, ``None`` when ``level`` is ``None``), ``scale``
        (the factor applied to the raw draws about the estimate, 1.0 when they
        are unchanged) and ``calibrated`` (bool, whether the draws were
        rescaled to the conformal interval).  The draws are also left unchanged
        when they have no spread about the estimate while the interval has.
    """
    errors = _loco_errors(table)
    n_cases = errors.size
    estimate = float(estimate)
    if not np.isfinite(estimate):
        raise ValueError("estimate must be finite")
    if int(n_draws) < 1:
        raise ValueError("n_draws must be at least 1")
    if level is not None:
        level = _check_level(level)
    if groups is None:
        groups = table["group"] if "group" in table.columns else np.arange(n_cases)
    codes = _group_codes(groups, n_cases)
    n_groups = int(codes.max()) + 1
    members = [np.flatnonzero(codes == g) for g in range(n_groups)]
    rng = np.random.default_rng(seed)
    n_draws = int(n_draws)
    sampled = rng.integers(0, n_groups, size=(n_draws, n_groups))
    position = rng.random(n_draws)
    picked = np.empty(n_draws, dtype=int)
    for k in range(n_draws):
        pool = np.concatenate([members[g] for g in sampled[k]])
        picked[k] = pool[int(position[k] * pool.size)]
    chosen = errors[picked]
    interval: dict[str, Any] | None = None
    scale, calibrated = 1.0, False
    if level is not None:
        interval = _interval_from_errors(errors, estimate, level)
        if interval["status"] == "ok":
            width = interval["half_width"]
            spread = float(np.quantile(np.abs(chosen), level))
            if width <= 0.0:
                scale, calibrated = (0.0 if spread > 0.0 else 1.0), True
            elif spread > _NOISE * width:
                scale, calibrated = width / spread, True
    draws = estimate - scale * chosen
    levels = list(_PERCENTILES)
    return {
        "draws": draws,
        "percentiles": pd.Series(np.percentile(draws, levels), index=pd.Index(levels, name="percentile"), name="loco_draws"),
        "estimate": estimate,
        "n_draws": n_draws,
        "n_groups": n_groups,
        "n_cases": n_cases,
        "interval": interval,
        "scale": float(scale),
        "calibrated": calibrated,
    }


# ----------------------------------------------------------------------------
# Bootstrap of the transported effect
# ----------------------------------------------------------------------------
def bootstrap_transport(
    Zs: pd.DataFrame | ArrayLike,
    Zt: pd.DataFrame | ArrayLike,
    w: ArrayLike,
    tau: ArrayLike,
    se: ArrayLike,
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
    eps: float | None = None,
    rho_source: float | None = 1.0,
    n_boot: int = 1000,
    seed: int | np.random.Generator = 0,
    groups: ArrayLike | None = None,
    guard: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Bootstrap distribution of the transported target effect.

    Each replicate resamples the source cases with replacement, or whole
    groups of cases with replacement when ``groups`` is given (a cluster
    bootstrap, in which every case of a drawn group enters the replicate and a
    group drawn twice enters twice), adds normal measurement noise with
    standard deviation ``se`` to the resampled effects, recomputes the plan
    (source marginal relaxed with ``rho_source``, target marginal enforced) and
    the target effect.  The cost matrix, the target weights and ``eps`` are
    computed once from the original sources and target.

    A predictive draw describes the observed effect of a new case at the
    target.  It adds to the target effect of the replicate two independent
    normal deviates.  The first has the deconvolved between-case standard
    deviation ``tau_b``, with ``tau_b^2 = max(0, V - S)``, where ``V`` is the
    variance of the resampled observed effects (without the added noise) around
    their mean and ``S`` the mean of the squared standard errors, both weighted
    by the source usage ``u`` of the replicate.  The second has the standard
    deviation ``sqrt(S)``, the measurement noise of one case.  The measurement
    noise therefore enters the predictive draw once.

    Three intervals can be produced by this module and each has its own use.
    ``draws`` (with ``percentiles``) is the bootstrap distribution of the
    transported average effect; it describes the sampling uncertainty of that
    average effect and not the effect observed in a new study.
    ``predictive_draws`` come from the normal model above and are not validated
    against held-out cases.  The interval that is validated by construction for
    the effect observed in a new study at the target is the conformal interval
    of :func:`loco_interval`, built from the leave-one-economy-out errors of
    :func:`loco_validation`.

    What the bootstrap describes.  The transport averages source effects, so it
    cannot predict an effect larger than the largest source effect, and the
    resampling reflects only the cases at hand.  The validation folds behind
    :func:`loco_validation` use the leave-one-economy-out errors of cases at
    their own features (interpolation among the cases).  The intervals
    describe the target only inside the support checked by
    :func:`target_support` (effective number of sources, range of the
    features).  When the median pairwise weighted distance of the sources is
    zero and ``eps`` is ``None``, :func:`select_eps` falls back to the median
    of the positive distances, or to 1 when all sources coincide.  When the
    weights are close to, but not exactly, concentrated on a feature with few
    values, ``eps`` is small and positive and the scaling iterations of a
    replicate may not reach the tolerance within 5000 iterations; the replicate
    is then solved by the log-domain rescue of :func:`sinkhorn_plan` with the
    same ``eps``, ``rho_source`` and masses, and ``n_log_domain`` counts those
    solves while ``n_nonconverged`` counts the solves that do not reach the
    tolerance even after the rescue.

    The per-case arguments ``tau``, ``se``, ``a`` and ``groups`` (and ``b`` for
    the target points) are used by position, except that a Series whose index
    is a permutation of the row labels of a DataFrame ``Zs`` (``Zt`` for
    ``b``) is first put in the order of those labels.

    Parameters
    ----------
    Zs : DataFrame or array, shape (n, d)
        Standardised features of the source cases.
    Zt : DataFrame or array, shape (m, d)
        Standardised features of the target points.
    w : array-like, length d
        Non-negative feature weights.
    tau : array-like, length n
        Estimated effect of each source case; finite for every case with positive mass.
    se : array-like, length n
        Standard error of each estimated effect; positive and finite for every
        case with positive mass.
    a : array-like, length n, optional
        Source masses, uniform when ``None``; sources with zero mass are dropped
        and the masses are normalised to sum to one in every replicate.
    b : array-like, length m, optional
        Weights of the target points, uniform when ``None``; normalised to sum to one.
    eps : float, optional
        Entropic regularisation strength.  When ``None`` it is :func:`select_eps` of ``Zs``.
    rho_source : float or None
        Strength of the penalty on the source marginal; ``None`` enforces it.
    n_boot : int
        Number of replicates.
    seed : int or numpy.random.Generator
        Seed of the random generator.
    groups : array-like, length n, optional
        Group label of each source case, for example the economy code, in the
        row order of ``Zs``.  Groups without any source of positive mass are
        ignored and at least two groups must remain.  When every case has its
        own label the result equals that obtained with ``None``.
    guard : callable, optional
        Function without arguments, called before every draw whose number is a
        multiple of 50 (draws are numbered from zero), that is ``ceil(n_boot /
        50)`` times in all.  Its return value is ignored and an exception it
        raises ends the bootstrap.

    Returns
    -------
    dict
        ``draws`` (ndarray of the target effect in each replicate),
        ``percentiles`` (Series of the 5, 25, 50, 75 and 95 percent points of
        ``draws``), ``predictive_draws`` and ``predictive_percentiles`` (the
        same for the predictive draws), ``estimate`` (target effect from the
        original sources), ``between_sd`` (deconvolved between-case standard
        deviation under the original plan), ``measurement_sd`` (root of the
        usage-weighted mean squared standard error under the original plan),
        ``eps``, ``n_boot``, ``n_nonconverged`` (solves of the original sources
        and of the replicates that did not reach the tolerance, also after the
        rescue) and ``n_log_domain`` (solves that the rescue brought to the
        tolerance).
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    _check_guard(guard)
    if not isinstance(Zs, pd.DataFrame) and np.ndim(Zs) != 2:
        raise ValueError("Zs must be a DataFrame or a two-dimensional array")
    A, B, cols, source_labels, target_labels = _pair_matrices(Zs, Zt)
    n, m = A.shape[0], B.shape[0]
    if n == 0 or m == 0:
        raise ValueError("Zs and Zt need at least one row")
    wt = _rescaled_weights(w, A.shape[1], cols, "w")
    tau_v = _vector(_by_label(tau, source_labels), n, "tau")
    se_v = _vector(_by_label(se, source_labels), n, "se")
    codes = None if groups is None else _group_codes(_by_label(groups, source_labels), n)
    a_v = _mass_vector(_by_label(a, source_labels), n, "a")
    keep = a_v > 0
    A, tau_v, se_v, a_v = A[keep], tau_v[keep], se_v[keep], a_v[keep]
    if not np.all(np.isfinite(tau_v)):
        raise ValueError("tau must be finite for every source with positive mass")
    if not (np.all(np.isfinite(se_v)) and np.all(se_v > 0)):
        raise ValueError("se must be positive and finite for every source with positive mass")
    n = A.shape[0]
    if codes is not None:
        codes = pd.factorize(codes[keep])[0]
        n_groups = int(codes.max()) + 1
        if n_groups < 2:
            raise ValueError("groups needs at least two distinct labels among the sources with positive mass")
    b_v = _mass_vector(_by_label(b, target_labels), m, "b")
    b_v = b_v / b_v.sum()
    rho_s = _rho(rho_source, "rho_source")
    C = _sq_cost(A, B, wt)
    eps_v = _check_eps(eps) if eps is not None else _default_eps(A, wt)[0]
    se2 = se_v**2
    rng = np.random.default_rng(seed)
    n_draws = int(n_boot)
    if codes is None:
        resamples = list(rng.integers(0, n, size=(n_draws, n)))
        noise = list(rng.standard_normal((n_draws, n)))
    else:
        members = [np.flatnonzero(codes == g) for g in range(n_groups)]
        drawn = rng.integers(0, n_groups, size=(n_draws, n_groups))
        resamples = [np.concatenate([members[g] for g in row]) for row in drawn]
        lengths = np.array([ii.size for ii in resamples])
        noise = np.split(rng.standard_normal(int(lengths.sum())), np.cumsum(lengths)[:-1])
    predictive_noise = rng.standard_normal((n_draws, 2))

    def one(ii: np.ndarray, tau_r: np.ndarray) -> tuple[float, float, float, bool, bool]:
        """Target effect, between-case variance, measurement variance, convergence and rescue flags for the sources ``ii``.

        ``tau_r`` holds the effects of the sources including any added noise; the spread is measured on the
        observed effects ``tau_v[ii]``.
        """
        a_r = a_v[ii] / a_v[ii].sum()
        pi, converged, _, _, rescued = _solve(a_r, b_v, C[ii], eps_v, rho_s, None, 5000, 1e-9)
        _, effect = _barycentric(pi, tau_r)
        use = b_v > 0
        value = float((b_v[use] * effect[use]).sum())
        usage = pi.sum(axis=1)
        total = usage.sum()
        observed = tau_v[ii]
        centre = float(usage @ observed / total)
        spread = float(usage @ (observed - centre) ** 2 / total)
        measurement = float(usage @ se2[ii] / total)
        return value, max(spread - measurement, 0.0), measurement, converged, rescued

    everyone = np.arange(n)
    estimate, base_between, base_measurement, base_converged, base_rescued = one(everyone, tau_v)
    draws = np.empty(n_draws)
    between = np.empty(n_draws)
    measurement = np.empty(n_draws)
    n_nonconverged = int(not base_converged)
    n_log_domain = int(base_rescued)
    for r in range(n_draws):
        if guard is not None and r % 50 == 0:
            guard()
        ii = resamples[r]
        draws[r], between[r], measurement[r], converged, rescued = one(ii, tau_v[ii] + se_v[ii] * noise[r])
        n_nonconverged += int(not converged)
        n_log_domain += int(rescued)
    predictive = draws + np.sqrt(between) * predictive_noise[:, 0] + np.sqrt(measurement) * predictive_noise[:, 1]
    levels = list(_PERCENTILES)
    return {
        "draws": draws,
        "percentiles": pd.Series(np.percentile(draws, levels), index=pd.Index(levels, name="percentile"), name="draws"),
        "predictive_draws": predictive,
        "predictive_percentiles": pd.Series(
            np.percentile(predictive, levels), index=pd.Index(levels, name="percentile"), name="predictive_draws"
        ),
        "estimate": estimate,
        "between_sd": float(np.sqrt(base_between)),
        "measurement_sd": float(np.sqrt(base_measurement)),
        "eps": eps_v,
        "n_boot": int(n_boot),
        "n_nonconverged": n_nonconverged,
        "n_log_domain": n_log_domain,
    }
