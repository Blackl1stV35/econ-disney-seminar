"""Synthetic control: weight solvers, nested and direct fits, placebo inference.

The estimator follows Abadie, Diamond and Hainmueller (2010) as implemented in
the Stata command ``synth``.  For a treated unit with predictors ``x1`` (length
``k``) and donors with predictor matrix ``X0`` (``k x J``) the donor weights are

.. math::
    w(v) = \\arg\\min_{w \\in \\Delta_J} (x_1 - X_0 w)' \\mathrm{diag}(v) (x_1 - X_0 w),

with the simplex ``Delta_J = {w >= 0, sum(w) = 1}``, and ``v`` minimises the
pre-treatment mean squared prediction error (MSPE) of the outcome.

Two solvers are provided.

``nested``
    Outer optimisation over ``v`` (softmax parameterisation, analytic gradient by
    implicit differentiation of the inner problem, several starts) with an exact
    active-set quadratic program for ``w(v)``.
``direct``
    The constrained least-squares problem
    ``min_{w in simplex} ||y1_pre - Y0_pre w||^2``.  When every pre-treatment
    outcome is a predictor and the MSPE is evaluated on the same periods, the
    nested objective equals this problem for uniform ``v``, so the direct
    solution is the global optimum of the nested problem.

Both are reported by :func:`synth` together with their pre-treatment RMSPE.
Everything is vectorised over predictors and donors; the placebo routine builds
the donor Gram matrix once and solves one small quadratic program per unit, so
a pool of about one hundred donors runs in well under a second.

Non-unique minimisers
---------------------
The weights are unique only when the pre-treatment outcomes of the donors that
can carry weight determine them. When the pre-treatment outcome of a unit lies in
the convex hull of more than ``T + 1`` donors, with ``T`` pre-treatment periods
in the fit, or when two donors are identical, many weight vectors reach the same
minimum. :func:`qp_simplex` then returns the vector at which its active-set path
stops. The path starts from the donor with the smallest value of
``0.5 * H[j, j] - g[j]`` and adds, one at a time, the donor with the most
negative reduced cost; the donor with the smallest index wins a tie. The result
is a vertex of the set of minimisers, in general with at most ``T + 1`` donors of
positive weight. A working-set system that is singular is solved in the
minimum-norm least-squares sense, but the vector returned is not the minimiser of
smallest norm, and another solver or another tie-breaking rule returns another
minimiser with the same pre-treatment fit. The synthetic outcome in periods that
did not enter the fit, and therefore the effect, depends on this rule. This
affects in particular the placebo effects of donors that the other donors
reproduce exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = [
    "qp_simplex",
    "solve_w_given_v",
    "fit_direct",
    "fit_nested",
    "SCMFit",
    "SCMResult",
    "synth",
    "build_predictors",
    "ridge_augment_weights",
    "PlaceboResult",
    "placebo_in_space",
    "rank_descending",
]


# ----------------------------------------------------------------------------
# Quadratic program on the simplex
# ----------------------------------------------------------------------------
def _solve_kkt(K: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a (possibly singular) KKT system; fall back to least squares."""
    try:
        sol = np.linalg.solve(K, rhs)
        if np.all(np.isfinite(sol)) and np.linalg.norm(K @ sol - rhs) <= 1e-9 * (1.0 + np.linalg.norm(rhs)):
            return sol
    except np.linalg.LinAlgError:
        pass
    return np.linalg.lstsq(K, rhs, rcond=None)[0]


def qp_simplex(
    H: np.ndarray,
    g: np.ndarray,
    ridge: float = 0.0,
    tol: float = 1e-10,
    max_iter: int | None = None,
    return_info: bool = False,
):
    """Minimise ``0.5 w'Hw - g'w`` subject to ``w >= 0`` and ``sum(w) = 1``.

    A primal active-set method: the working set holds the indices allowed to be
    positive; each iteration solves the equality-constrained problem on the
    working set through its KKT system, steps to the first violated bound, and
    adds the index with the most negative reduced cost until the KKT conditions
    hold.  ``H`` must be positive semidefinite.  A singular working-set system
    is solved in the minimum-norm least-squares sense.  When the minimiser is not
    unique the vector returned is the vertex at which the path stops, in general
    with at most ``T + 1`` positive entries for ``T`` rows in the factor of ``H``;
    it is not the minimiser of smallest norm (see the module description).

    Parameters
    ----------
    H : (J, J) array
    g : (J,) array
    ridge : float
        Added to the diagonal of ``H`` (penalty ``0.5 * ridge * ||w||^2``).
    tol : float
        Optimality tolerance on reduced costs, relative to ``max(1, |H|, |g|)``.

    Returns
    -------
    w : (J,) array
    info : dict, only if ``return_info``
        ``support``, ``lam`` (multiplier of the sum constraint), ``iterations``,
        ``kkt_violation``.
    """
    g = np.asarray(g, dtype=float)
    J = g.size
    Hm = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    if ridge:
        Hm = Hm + ridge * np.eye(J)
    scale = max(1.0, float(np.abs(Hm).max()), float(np.abs(g).max()))
    max_iter = 60 * J + 300 if max_iter is None else max_iter
    tol_w = 1e-13
    w = np.zeros(J)
    j0 = int(np.argmin(0.5 * np.diag(Hm) - g))
    w[j0] = 1.0
    S = [j0]
    lam = 0.0
    it = 0
    kkt = np.inf
    for it in range(1, max_iter + 1):
        idx = np.array(S)
        m = idx.size
        K = np.zeros((m + 1, m + 1))
        K[:m, :m] = Hm[np.ix_(idx, idx)]
        K[:m, m] = -1.0
        K[m, :m] = 1.0
        sol = _solve_kkt(K, np.r_[g[idx], 1.0])
        wS, lam = sol[:m], float(sol[m])
        if wS.min() < -tol_w:
            wc = w[idx]
            neg = wS < -tol_w
            alpha = float(np.min(wc[neg] / (wc[neg] - wS[neg])))
            alpha = min(max(alpha, 0.0), 1.0)
            w_idx = wc + alpha * (wS - wc)
            keep = w_idx > tol_w
            if keep.all():
                keep[int(np.argmin(w_idx))] = False
            w[idx] = np.where(keep, w_idx, 0.0)
            S = [int(i) for i in idx[keep]]
            if not S:
                S = [int(idx[np.argmax(w_idx)])]
                w[S[0]] = 1.0
            continue
        w = np.zeros(J)
        w[idx] = np.maximum(wS, 0.0)
        grad = Hm @ w - g
        red = grad - lam
        excluded = np.setdiff1d(np.arange(J), idx, assume_unique=True)
        if excluded.size == 0:
            kkt = float(np.abs(red[idx]).max())
            break
        jmin = excluded[int(np.argmin(red[excluded]))]
        worst = float(red[jmin])
        kkt = max(float(np.abs(red[idx]).max()), max(0.0, -worst))
        if worst >= -tol * scale:
            break
        S.append(int(jmin))
    else:
        # Not converged: fall back to a generic constrained solver.
        res = optimize.minimize(
            lambda x: 0.5 * x @ Hm @ x - g @ x,
            w,
            jac=lambda x: Hm @ x - g,
            bounds=[(0.0, 1.0)] * J,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(J)}],
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 500},
        )
        w = np.maximum(res.x, 0.0)
        w /= w.sum()
    w = np.maximum(w, 0.0)
    w /= w.sum()
    if return_info:
        return w, {"support": np.flatnonzero(w > 0), "lam": lam, "iterations": it, "kkt_violation": kkt}
    return w


def solve_w_given_v(X1: np.ndarray, X0: np.ndarray, v: np.ndarray, ridge: float = 0.0, return_info: bool = False):
    """Inner problem: simplex weights minimising the ``v``-weighted predictor distance."""
    Xv = X0 * v[:, None]
    H = X0.T @ Xv
    g = Xv.T @ X1
    return qp_simplex(H, g, ridge=ridge, return_info=return_info)


# ----------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------
@dataclass
class SCMFit:
    """One solved synthetic control problem.

    ``loss`` is the pre-treatment MSPE ``mean((y1_pre - Y0_pre w)^2)`` and
    ``rmspe`` its square root.  ``v`` is the predictor weight vector (``None``
    for the direct solution).
    """

    method: str
    w: np.ndarray
    v: np.ndarray | None
    loss: float
    rmspe: float
    converged: bool = True
    n_iter: int = 0
    starts: list[dict] = field(default_factory=list)


def _mspe(y1: np.ndarray, Y0: np.ndarray, w: np.ndarray) -> float:
    r = y1 - Y0 @ w
    return float(r @ r / r.size)


def fit_direct(y1_pre: np.ndarray, Y0_pre: np.ndarray, ridge: float = 0.0) -> SCMFit:
    """Constrained least squares ``min ||y1_pre - Y0_pre w||^2`` on the simplex.

    Parameters
    ----------
    y1_pre : (T,) array
        Pre-treatment outcome of the treated unit.
    Y0_pre : (T, J) array
        Pre-treatment outcomes of the ``J`` donors, one column per donor.
    ridge : float, default 0.0
        Penalty on ``||w||^2``.

    Returns
    -------
    SCMFit
        Weights ``w``, pre-treatment MSPE ``loss`` and RMSPE ``rmspe``.

    Notes
    -----
    With a positive ``ridge`` the minimiser is unique. Without it the minimiser is
    not unique when the donors that can carry weight have affinely dependent
    pre-treatment outcomes, for example when ``y1_pre`` lies in the convex hull of
    more than ``T + 1`` donors or when two donors are identical. The weight
    vector returned is then the one at which the active-set path of
    :func:`qp_simplex` stops: a vertex of the set of minimisers, in general with
    at most ``T + 1`` donors of positive weight, reached by starting from the donor
    with the smallest ``0.5 * H[j, j] - g[j]`` and adding the donor with the most
    negative reduced cost, the smallest index winning a tie. It is not the
    minimiser of smallest norm. All minimisers have the same pre-treatment fit and
    differ in the synthetic outcome of later periods, so any effect computed from
    such a fit, in particular the placebo effect of a donor that the other donors
    reproduce exactly, depends on this rule.
    """
    w, info = qp_simplex(Y0_pre.T @ Y0_pre, Y0_pre.T @ y1_pre, ridge=ridge, return_info=True)
    loss = _mspe(y1_pre, Y0_pre, w)
    return SCMFit("direct", w, None, loss, float(np.sqrt(loss)), info["kkt_violation"] < 1e-6, info["iterations"])


def _softmax(theta: np.ndarray) -> np.ndarray:
    z = theta - theta.max()
    e = np.exp(z)
    return e / e.sum()


def _nested_objective(theta, X1, X0, y1, Y0, ridge):
    """Outer loss ``MSPE(w(v))`` and its gradient with respect to ``theta``.

    The gradient follows from implicit differentiation of the KKT system of the
    inner problem on its current support ``S``:
    ``dw_S/dv_k = P x0k (x1_k - x0k' w_S)`` where ``P`` is the leading block of
    the inverse KKT matrix and ``x0k`` the ``k``-th predictor row of the support
    donors.
    """
    v = _softmax(theta)
    w, info = solve_w_given_v(X1, X0, v, ridge, return_info=True)
    r = y1 - Y0 @ w
    T0 = r.size
    loss = float(r @ r / T0)
    S = info["support"]
    m = S.size
    X0S = X0[:, S]
    HS = (X0S.T * v) @ X0S + ridge * np.eye(m)
    K = np.zeros((m + 1, m + 1))
    K[:m, :m] = HS
    K[:m, m] = -1.0
    K[m, :m] = 1.0
    try:
        Kinv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        Kinv = np.linalg.pinv(K)
    P = Kinv[:m, :m]
    rho = X1 - X0S @ w[S]
    D = P @ (X0S.T * rho[None, :])
    dL_dv = -(2.0 / T0) * ((r @ Y0[:, S]) @ D)
    grad = v * (dL_dv - v @ dL_dv)
    return loss, grad


def _regression_start(X1: np.ndarray, X0: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """Start for ``v``: squared coefficients of the donors' mean outcome on the predictors."""
    k, J = X0.shape
    target = Y0.mean(axis=0)
    A = X0.T - X0.T.mean(axis=0)
    beta = np.linalg.lstsq(A, target - target.mean(), rcond=None)[0]
    s = (beta * X0.std(axis=1)) ** 2
    if not np.isfinite(s).all() or s.sum() <= 0:
        return np.full(k, 1.0 / k)
    s = s + 1e-3 * s.sum() / k
    return s / s.sum()


def fit_nested(
    X1: np.ndarray,
    X0: np.ndarray,
    y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    ridge: float = 0.0,
    n_starts: int = 6,
    seed: int = 0,
    maxiter: int = 500,
) -> SCMFit:
    """Nested optimisation of ``diag(V)`` with simplex-constrained ``W``.

    Starts: uniform ``v``, a regression-based ``v`` and ``n_starts - 2`` Dirichlet
    draws (seeded).  Each start runs L-BFGS-B on the softmax parameters with the
    analytic gradient; the start with the smallest pre-treatment MSPE is
    returned and all starts are listed in ``starts``.
    """
    k = X0.shape[0]
    rng = np.random.default_rng(seed)
    thetas = [np.zeros(k)]
    if n_starts > 1:
        thetas.append(np.log(_regression_start(X1, X0, Y0_pre)))
    for _ in range(max(0, n_starts - 2)):
        thetas.append(np.log(rng.dirichlet(np.ones(k))))
    runs = []
    for th0 in thetas:
        res = optimize.minimize(
            _nested_objective,
            th0,
            args=(X1, X0, y1_pre, Y0_pre, ridge),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        v = _softmax(res.x)
        w = solve_w_given_v(X1, X0, v, ridge)
        loss = _mspe(y1_pre, Y0_pre, w)
        runs.append({"loss": loss, "rmspe": float(np.sqrt(loss)), "v": v, "w": w, "nit": int(res.nit), "success": bool(res.success)})
    best = min(runs, key=lambda r: r["loss"])
    return SCMFit(
        "nested",
        best["w"],
        best["v"],
        best["loss"],
        best["rmspe"],
        converged=best["success"],
        n_iter=best["nit"],
        starts=[{"loss": r["loss"], "rmspe": r["rmspe"], "nit": r["nit"]} for r in runs],
    )


# ----------------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------------
def build_predictors(
    y1: np.ndarray,
    Y0: np.ndarray,
    periods: Sequence[Sequence[int]] | None = None,
    z1: np.ndarray | None = None,
    Z0: np.ndarray | None = None,
    scale: str = "sd",
):
    """Assemble ``(X1, X0)`` from outcome averages and optional covariates.

    Parameters
    ----------
    y1, Y0 : arrays ``(T,)`` and ``(T, J)``
        Outcome of the treated unit and of the donors.
    periods : list of index lists
        Each entry defines one predictor: the mean of the outcome over those
        period indices (a single index gives that period's value, as in
        ``receipts_pct_gdp(1999)``).
    z1, Z0 : arrays ``(c,)`` and ``(c, J)``
        Covariates (already averaged over their windows), appended as rows.
    scale : {"sd", "none"}
        ``"sd"`` divides every predictor row by its standard deviation across
        all units, which puts rows on a common scale before ``v`` is optimised.
    """
    rows1, rows0 = [], []
    for p in periods or []:
        p = list(p)
        rows1.append(y1[p].mean())
        rows0.append(Y0[p].mean(axis=0))
    if z1 is not None:
        rows1.extend(np.atleast_1d(z1))
        rows0.extend(np.atleast_2d(Z0))
    X1 = np.asarray(rows1, dtype=float)
    X0 = np.vstack(rows0).astype(float)
    if scale == "sd":
        sd = np.std(np.column_stack([X1, X0]), axis=1, ddof=1)
        sd[sd == 0] = 1.0
        X1, X0 = X1 / sd, X0 / sd[:, None]
    elif scale != "none":
        raise ValueError("scale must be 'sd' or 'none'")
    return X1, X0


# ----------------------------------------------------------------------------
# Ridge augmentation
# ----------------------------------------------------------------------------
def ridge_augment_weights(x1: np.ndarray, X0: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Ridge-augmented weights (Ben-Michael, Feller and Rothstein, 2021).

    ``w_aug = w + Xc' (Xc Xc' + lam I)^{-1} (x1 - X0 w)`` with ``X0`` the
    ``T0 x J`` pre-treatment outcomes of the donors and ``Xc`` its columns
    centred over donors.  The augmented weights sum to one, may be negative, and
    correct the remaining pre-treatment imbalance by ridge regression; as
    ``lam`` grows they return to ``w``.
    """
    Xc = X0 - X0.mean(axis=1, keepdims=True)
    T0 = X0.shape[0]
    adj = np.linalg.solve(Xc @ Xc.T + lam * np.eye(T0), x1 - X0 @ w)
    return w + Xc.T @ adj


# ----------------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------------
@dataclass
class SCMResult:
    """Synthetic control result for one treated unit.

    ``fits`` holds every solver run (``"nested"``, ``"direct"``); ``w`` and
    ``synthetic`` refer to the preferred fit (``primary``).  ``gap`` is the
    treated outcome minus the synthetic outcome for all periods.
    """

    primary: str
    w: np.ndarray
    v: np.ndarray | None
    synthetic: np.ndarray
    gap: np.ndarray
    rmspe_pre: float
    rmspe_post: float
    ratio: float
    mean_post_gap: float
    fits: dict[str, SCMFit]
    w_aug: np.ndarray | None = None
    synthetic_aug: np.ndarray | None = None
    gap_aug: np.ndarray | None = None
    pre: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, f in self.fits.items():
            rows.append({"method": name, "rmspe_pre": f.rmspe, "max_abs_weight_diff_vs_primary": float(np.abs(f.w - self.w).max())})
        return pd.DataFrame(rows)


def synth(
    y1: np.ndarray,
    Y0: np.ndarray,
    pre: np.ndarray,
    X1: np.ndarray | None = None,
    X0: np.ndarray | None = None,
    method: str = "both",
    ridge: float = 0.0,
    n_starts: int = 6,
    seed: int = 0,
    augment_lambda: float | None = None,
) -> SCMResult:
    """Fit a synthetic control for one treated unit.

    Parameters
    ----------
    y1 : (T,) array
        Outcome of the treated unit in all periods.
    Y0 : (T, J) array
        Outcomes of the ``J`` donors.
    pre : (T,) boolean array
        Pre-treatment periods; the MSPE is evaluated on them.
    X1, X0 : arrays ``(k,)`` and ``(k, J)``, optional
        Predictors.  Default: the outcome in every pre-treatment period, which
        is the specification used in the original analysis.
    method : {"nested", "direct", "both"}
        ``"both"`` runs the nested optimisation and the direct solution and
        selects as primary the fit with the smaller MSPE (nested on ties).
    ridge : float
        Penalty on ``||w||^2`` in the inner problem (0 reproduces ``synth``).
    augment_lambda : float, optional
        If given, ridge-augmented weights are also returned.
    """
    pre = np.asarray(pre, dtype=bool)
    y1_pre, Y0_pre = y1[pre], Y0[pre]
    if X1 is None or X0 is None:
        X1, X0 = y1_pre, Y0_pre
    fits: dict[str, SCMFit] = {}
    if method in ("nested", "both"):
        fits["nested"] = fit_nested(X1, X0, y1_pre, Y0_pre, ridge=ridge, n_starts=n_starts, seed=seed)
    if method in ("direct", "both"):
        fits["direct"] = fit_direct(y1_pre, Y0_pre, ridge=ridge)
    if method not in ("nested", "direct", "both"):
        raise ValueError("method must be 'nested', 'direct' or 'both'")
    primary = min(fits, key=lambda n: (fits[n].loss, n != "nested"))
    if "nested" in fits and "direct" in fits and fits["nested"].loss <= fits["direct"].loss + 1e-13:
        primary = "nested"
    best = fits[primary]
    synthetic = Y0 @ best.w
    gap = y1 - synthetic
    post = ~pre
    rmspe_pre = float(np.sqrt(np.mean(gap[pre] ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gap[post] ** 2))) if post.any() else np.nan
    res = SCMResult(
        primary=primary,
        w=best.w,
        v=best.v,
        synthetic=synthetic,
        gap=gap,
        rmspe_pre=rmspe_pre,
        rmspe_post=rmspe_post,
        ratio=rmspe_post / rmspe_pre if rmspe_pre > 0 else np.inf,
        mean_post_gap=float(gap[post].mean()) if post.any() else np.nan,
        fits=fits,
        pre=pre,
    )
    if augment_lambda is not None:
        w_aug = ridge_augment_weights(y1_pre, Y0_pre, best.w, augment_lambda)
        res.w_aug = w_aug
        res.synthetic_aug = Y0 @ w_aug
        res.gap_aug = y1 - res.synthetic_aug
    return res


# ----------------------------------------------------------------------------
# Placebo in space
# ----------------------------------------------------------------------------
def rank_descending(values: np.ndarray) -> np.ndarray:
    """Rank with 1 for the largest value; ties share the smallest rank."""
    values = np.asarray(values, dtype=float)
    return 1 + np.array([(values > v).sum() for v in values])


@dataclass
class PlaceboResult:
    """Placebo-in-space table.

    ``table`` has one row per unit (the treated unit first) with ``pre_rmspe``,
    ``post_rmspe``, ``ratio`` (post/pre), ``mean_post_gap``, ``abs_mean_gap`` and
    the ranks ``rank_abs_mean_gap`` and ``rank_ratio`` (1 = largest).  The
    permutation p-values are ``rank / number_of_units``.
    """

    table: pd.DataFrame
    gaps: np.ndarray
    treated_label: object
    p_value_ratio: float
    p_value_abs_mean_gap: float


def placebo_in_space(
    Y: np.ndarray,
    treated: int,
    pre: np.ndarray,
    labels: Sequence | None = None,
    donors: Sequence[int] | None = None,
    exclude_treated: bool = True,
    ridge: float = 0.0,
    X: np.ndarray | None = None,
    method: str = "direct",
    n_starts: int = 4,
    seed: int = 0,
) -> PlaceboResult:
    """Placebo-in-space inference for a treated unit.

    Parameters
    ----------
    Y : (T, N) array
        Outcomes of all units in all periods.
    treated : int
        Column of the treated unit.
    pre : (T,) boolean array
        Pre-treatment periods.
    donors : sequence of int, optional
        Columns of the donors (default: every column but ``treated``).
    exclude_treated : bool
        True removes the treated unit from the donor pool of every placebo
        (Abadie et al. 2010).  False keeps it in the pool, which reproduces the
        placebo table of the first set of reference values.
    X : (k, N) array, optional
        Predictors for all units (columns).  With ``method="direct"`` the
        predictors are not used.
    method : {"direct", "nested"}
        ``"direct"`` solves the constrained least-squares problem for each
        placebo using one precomputed Gram matrix.

    Returns
    -------
    PlaceboResult

    Notes
    -----
    Each p-value is the rank of the treated unit among all ``1 + len(donors)``
    units divided by that number. No unit is left out because its pre-treatment
    fit is poor, and a unit with a zero pre-treatment RMSPE has an infinite ratio.
    :func:`dtt.effects.estimate_case` ranks the mean gap among the placebos of
    comparable fit only and leaves out the ratios of exact fits.
    """
    Y = np.asarray(Y, dtype=float)
    T, N = Y.shape
    pre = np.asarray(pre, dtype=bool)
    post = ~pre
    donors = [j for j in range(N) if j != treated] if donors is None else list(donors)
    units = [treated] + donors
    labels = list(range(N)) if labels is None else list(labels)
    Ypre = Y[pre]
    G = Ypre.T @ Ypre
    rows, gaps = [], []
    for i in units:
        pool = [j for j in donors if j != i]
        if not exclude_treated and i != treated:
            pool = [treated] + pool
        pool = np.array(pool, dtype=int)
        if method == "direct":
            w = qp_simplex(G[np.ix_(pool, pool)], G[pool, i], ridge=ridge)
        else:
            X1, X0 = X[:, i], X[:, pool]
            w = fit_nested(X1, X0, Ypre[:, i], Ypre[:, pool], ridge=ridge, n_starts=n_starts, seed=seed).w
        gap = Y[:, i] - Y[:, pool] @ w
        gaps.append(gap)
        pre_r = float(np.sqrt(np.mean(gap[pre] ** 2)))
        post_r = float(np.sqrt(np.mean(gap[post] ** 2)))
        mg = float(gap[post].mean())
        rows.append(
            {
                "unit": labels[i],
                "pre_rmspe": pre_r,
                "post_rmspe": post_r,
                "ratio": post_r / pre_r if pre_r > 0 else np.inf,
                "mean_post_gap": mg,
                "abs_mean_gap": abs(mg),
            }
        )
    table = pd.DataFrame(rows)
    table["rank_abs_mean_gap"] = rank_descending(table["abs_mean_gap"].to_numpy())
    table["rank_ratio"] = rank_descending(table["ratio"].to_numpy())
    n = len(table)
    return PlaceboResult(
        table=table,
        gaps=np.vstack(gaps).T,
        treated_label=labels[treated],
        p_value_ratio=float(table.loc[0, "rank_ratio"] / n),
        p_value_abs_mean_gap=float(table.loc[0, "rank_abs_mean_gap"] / n),
    )
