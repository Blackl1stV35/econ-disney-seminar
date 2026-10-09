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

Transported effect
    ``effect_j = sum_i pi_ij tau_i / sum_i pi_ij`` for each target point, and
    the target effect is the ``b``-weighted mean of these values.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
    "loco_predictive_draws",
    "bootstrap_transport",
]

_MAD_TO_SD = 1.4826
_LOCO_METHODS = ("ot_weighted", "ot_uniform", "equal", "nn1", "nn3", "kernel")
_WEIGHTED_METHODS = ("ot_weighted", "nn1", "nn3", "kernel")
_PERCENTILES = (5, 25, 50, 75, 95)
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
    """Validate feature weights and rescale them to sum to ``d``; ``name`` labels the weights in error messages."""
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
    return arr * (d / total)


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

    The total must be positive unless ``allow_zero_total`` is true.
    """
    if x is None:
        return np.full(n, 1.0 / n)
    arr = _vector(x, n, name)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    if arr.sum() <= 0 and not allow_zero_total:
        raise ValueError(f"{name} has zero total mass")
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


def _default_eps(Z: np.ndarray, wt: np.ndarray) -> float:
    """Return :func:`select_eps` of ``Z``, which must be positive."""
    if Z.shape[0] < 2:
        raise ValueError("eps must be given when fewer than two sources are available")
    value = select_eps(Z, wt)
    if not value > 0:
        raise ValueError("the default eps is zero because the sources coincide; pass eps explicitly")
    return value


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
    missing values.  When both inputs are DataFrames the columns of ``Zt`` are
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
        The regularisation strength ``eps``.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie between 0 and 1")
    A, cols, _ = _as_matrix(Z, "Z")
    n = A.shape[0]
    if n < 2:
        raise ValueError("Z needs at least two rows")
    wt = _rescaled_weights(w, A.shape[1], cols)
    C = _sq_cost(A, A, wt, ("Z", "Z"))
    return float(0.1 * np.quantile(C[np.triu_indices(n, k=1)], quantile))


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
        Number of scaling iterations performed.
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


def _solve(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    rho_s: float | None,
    rho_t: float | None,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, bool, int, float]:
    """Solve on the support of ``a`` and ``b``.

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
        Maximum number of iterations.
    tol : float
        Tolerance on the marginal residual relative to the total mass of the plan.

    Returns
    -------
    pi : ndarray, shape (len(a), len(b))
        The plan, zero outside the support.
    converged : bool
        Whether the relative residual fell below ``tol``.
    n_iter : int
        Number of iterations performed.
    err : float
        Marginal residual of the returned plan relative to its total mass.
    """
    ia = np.flatnonzero(a > 0)
    jb = np.flatnonzero(b > 0)
    a_s, b_s = a[ia], b[jb]
    if rho_s is None and rho_t is None:
        b_s = b_s * (a_s.sum() / b_s.sum())
    _, _, pi_s, n_iter, err = _sinkhorn_core(a_s, b_s, C[np.ix_(ia, jb)], eps, rho_s, rho_t, max_iter, tol)
    if not np.isfinite(pi_s).all():
        raise ValueError("the scaling iterations produced non-finite values; increase eps or rho")
    if ia.size == a.size and jb.size == b.size:
        return pi_s, bool(err < tol), n_iter, err
    pi = np.zeros(C.shape)
    pi[np.ix_(ia, jb)] = pi_s
    return pi, bool(err < tol), n_iter, err


def sinkhorn_plan(
    a: ArrayLike,
    b: ArrayLike,
    C: ArrayLike,
    eps: float,
    rho_source: float | None = None,
    rho_target: float | None = None,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> Plan:
    """Entropic optimal transport plan by log-domain Sinkhorn iterations.

    The plan minimises ``<C, pi> + eps KL(pi | a b') + rho_s KL(pi 1 | a) +
    rho_t KL(pi' 1 | b)`` over non-negative matrices, with
    ``KL(p | q) = sum p log(p / q) - p + q``.  A marginal with ``rho = None``
    (or ``inf``) is enforced exactly instead of penalised.  A relaxed marginal
    uses the scaling update with exponent ``rho / (rho + eps)``.  Any
    combination of enforced and relaxed marginals is supported.  Entries of
    ``a`` or ``b`` equal to zero receive no mass.  When both marginals are
    enforced their total masses must agree to a relative 1e-6, and ``b`` is
    rescaled to the total mass of ``a``.

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
        Maximum number of iterations.
    tol : float
        Tolerance on ``Plan.marginal_error``, the marginal residual relative to
        the total mass of the plan.

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
    pi, converged, n_iter, err = _solve(a_arr, b_arr, C_arr, eps, rho_s, rho_t, int(max_iter), float(tol))
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
    distribution.  The p-value is ``(1 + #{null >= statistic}) / (n_perm + 1)``.

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
    p_value = float((1 + np.count_nonzero(null >= statistic)) / (null.size + 1))
    return {"statistic": statistic, "p_value": p_value, "null": null, "n_perm": int(null.size)}


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
        the smallest RMSE) and ``rmse_adj`` (square root of the mean squared
        error less the mean squared standard error, floored at zero).
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
        Number of Sinkhorn solves that did not reach the tolerance.
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


def _loco_nn(C: np.ndarray, tau: np.ndarray, k: int) -> np.ndarray:
    """Mean effect of the ``k`` nearest sources at each target point.

    Parameters
    ----------
    C : ndarray, shape (n, m)
        Weighted squared distances from the sources to the target points.
    tau : ndarray, shape (n,)
        Source effects.
    k : int
        Number of neighbours, reduced to ``n`` when there are fewer sources.

    Returns
    -------
    ndarray, shape (m,)
        Prediction at each target point; ties are broken by source order.
    """
    k = min(k, C.shape[0])
    nearest = np.argsort(C, axis=0, kind="stable")[:k]
    return tau[nearest].mean(axis=0)


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
    """Pearson correlation of two vectors, missing when either is constant."""
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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
                "share_below_one": float(np.mean(boot < 1.0)),
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
        sources.

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
    ``ot_uniform``.

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
            elif method == "nn1":
                pred[method][i] = _loco_nn(C_w, tau_src, 1).mean()
            elif method == "nn3":
                pred[method][i] = _loco_nn(C_w, tau_src, 3).mean()
            elif method == "kernel":
                if code not in fold_bandwidth:
                    pair = _sq_cost(Zv[src], Zv[src], wt, ("Z", "Z"))
                    dist = np.sqrt(pair[np.triu_indices(src.size, k=1)]) if src.size > 1 else np.sqrt(C_w.ravel())
                    fold_bandwidth[code] = float(np.median(dist))
                bandwidth = fold_bandwidth[code]
                pred[method][i] = _loco_kernel(C_w, tau_src, bandwidth if bandwidth > 0 else 1.0).mean()
            else:
                weights_m = wt if method == "ot_weighted" else uniform
                C_m = C_w if method == "ot_weighted" else _sq_cost(Zv[src], target, uniform, ("Z", "clouds"))
                eps_m = fixed_eps if fixed_eps is not None else _default_eps(pooled, weights_m)
                fold_eps[method].append(eps_m)
                pi, converged, _, _ = _solve(a_src, b_i, C_m, eps_m, rho_s, None, 5000, 1e-9)
                n_nonconverged += int(not converged)
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
    summary["rank"] = summary["rmse"].rank(method="min", na_option="bottom").astype(int)
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
    )


def loco_predictive_draws(
    table: pd.DataFrame,
    estimate: float,
    groups: ArrayLike | None = None,
    n_draws: int = 2000,
    seed: int | np.random.Generator = 0,
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

    Returns
    -------
    dict
        ``draws`` (ndarray of length ``n_draws``), ``percentiles`` (Series of the
        5, 25, 50, 75 and 95 percent points of the draws), ``estimate``,
        ``n_draws``, ``n_groups`` and ``n_cases``.
    """
    if not isinstance(table, pd.DataFrame) or "error" not in table.columns:
        raise ValueError("table must be a DataFrame with the column 'error'")
    if "method" in table.columns and table["method"].nunique() > 1:
        raise ValueError("table must hold the rows of one method")
    n_cases = len(table)
    if n_cases == 0:
        raise ValueError("table has no rows")
    errors = _vector(table["error"], n_cases, "table['error']", finite=True)
    estimate = float(estimate)
    if not np.isfinite(estimate):
        raise ValueError("estimate must be finite")
    if int(n_draws) < 1:
        raise ValueError("n_draws must be at least 1")
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
    draws = estimate - errors[picked]
    levels = list(_PERCENTILES)
    return {
        "draws": draws,
        "percentiles": pd.Series(np.percentile(draws, levels), index=pd.Index(levels, name="percentile"), name="loco_draws"),
        "estimate": estimate,
        "n_draws": n_draws,
        "n_groups": n_groups,
        "n_cases": n_cases,
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
        ``eps``, ``n_boot`` and ``n_nonconverged``.
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
    eps_v = _check_eps(eps) if eps is not None else _default_eps(A, wt)
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

    def one(ii: np.ndarray, tau_r: np.ndarray) -> tuple[float, float, float, bool]:
        """Target effect, between-case variance, measurement variance and convergence flag for the sources ``ii``.

        ``tau_r`` holds the effects of the sources including any added noise; the spread is measured on the
        observed effects ``tau_v[ii]``.
        """
        a_r = a_v[ii] / a_v[ii].sum()
        pi, converged, _, _ = _solve(a_r, b_v, C[ii], eps_v, rho_s, None, 5000, 1e-9)
        _, effect = _barycentric(pi, tau_r)
        use = b_v > 0
        value = float((b_v[use] * effect[use]).sum())
        usage = pi.sum(axis=1)
        total = usage.sum()
        observed = tau_v[ii]
        centre = float(usage @ observed / total)
        spread = float(usage @ (observed - centre) ** 2 / total)
        measurement = float(usage @ se2[ii] / total)
        return value, max(spread - measurement, 0.0), measurement, converged

    everyone = np.arange(n)
    estimate, base_between, base_measurement, base_converged = one(everyone, tau_v)
    draws = np.empty(n_draws)
    between = np.empty(n_draws)
    measurement = np.empty(n_draws)
    n_nonconverged = int(not base_converged)
    for r in range(n_draws):
        if guard is not None and r % 50 == 0:
            guard()
        ii = resamples[r]
        draws[r], between[r], measurement[r], converged = one(ii, tau_v[ii] + se_v[ii] * noise[r])
        n_nonconverged += int(not converged)
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
    }
