"""Stata-compatible estimation primitives.

This module is a pure NumPy implementation of the estimation conventions of the
Stata commands used in the original analysis.  No Stata installation is needed
and no Stata file is read.  The conventions reproduced are:

* ``regress y X``                       classical OLS (``df_r = N - K``);
* ``regress y X, vce(cluster g)``       cluster-robust OLS, small-sample factor
  ``G/(G-1) * (N-1)/(N-K)``, inference on ``t(G-1)``;
* ``xtreg y X, fe``                     within estimator with the constant
  defined through the grand-mean transformation;
* ``xtreg y X, fe vce(cluster g)``      cluster-robust variance with factor
  ``G/(G-1) * (N-1)/(N-K_w)`` where ``K_w`` counts the regressors that are not
  absorbed (slopes plus the constant), inference on ``t(G-1)``.

The module also holds the closed-form scale factors that relate the different
cluster-robust conventions (see :func:`cluster_variance_scale`) and the
"report formula" standard error used in the ``friend_did`` program.

Conventions
-----------
All cluster-robust variances share one unscaled sandwich,

.. math:: S_0 = (X'X)^{-1} \\Big(\\sum_g X_g' e_g e_g' X_g\\Big) (X'X)^{-1},

and differ only by a deterministic scalar ``c`` that depends on ``N``, ``G`` and
the number of estimated parameters.  Collinear columns are dropped in column
order, as Stata's ``_rmcoll`` does, and are not counted in ``K``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, stats

__all__ = [
    "ttail",
    "invttail",
    "normal",
    "invnormal",
    "t_pvalue",
    "z_pvalue",
    "drop_collinear",
    "factorize",
    "cluster_sum",
    "factor_dummies",
    "small_sample_factor",
    "cluster_variance_scale",
    "se_ratio",
    "report_formula_se",
    "RegResult",
    "XtregResult",
    "regress",
    "xtreg_fe",
]


# ----------------------------------------------------------------------------
# Distribution functions with Stata names
# ----------------------------------------------------------------------------
def ttail(df: float, t):
    """Upper tail ``P(T > t)`` of Student's t with ``df`` degrees of freedom."""
    return stats.t.sf(t, df)


def invttail(df: float, p):
    """Inverse of :func:`ttail`: the ``t`` with upper tail probability ``p``."""
    return stats.t.isf(p, df)


def normal(z):
    """Standard normal cumulative distribution function."""
    return stats.norm.cdf(z)


def invnormal(p):
    """Standard normal quantile function."""
    return stats.norm.ppf(p)


def t_pvalue(t, df: float):
    """Two-sided p-value ``2 * ttail(df, |t|)``."""
    return 2.0 * stats.t.sf(np.abs(t), df)


def z_pvalue(z):
    """Two-sided normal p-value ``2 * normal(-|z|)``."""
    return 2.0 * stats.norm.cdf(-np.abs(z))


# ----------------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------------
def drop_collinear(X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Boolean mask of columns kept after dropping collinear columns in order.

    Columns are processed from left to right with modified Gram-Schmidt (with
    one re-orthogonalisation pass).  A column whose residual norm after
    projecting on the kept columns is at most ``tol`` times its own norm is
    dropped, mirroring the order-dependent behaviour of Stata's ``_rmcoll``.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    keep = np.zeros(k, dtype=bool)
    Q = np.empty((n, 0))
    for j in range(k):
        v = X[:, j]
        nv = np.linalg.norm(v)
        if nv == 0.0:
            continue
        r = v - Q @ (Q.T @ v)
        r = r - Q @ (Q.T @ r)
        nr = np.linalg.norm(r)
        if nr > tol * nv:
            keep[j] = True
            Q = np.column_stack([Q, r / nr])
    return keep


def _ols_qr(X: np.ndarray, y: np.ndarray):
    """OLS through a thin QR factorisation; returns ``b`` and ``(X'X)^{-1}``."""
    Q, R = np.linalg.qr(X, mode="reduced")
    Rinv = linalg.solve_triangular(R, np.eye(R.shape[0]), lower=False)
    b = Rinv @ (Q.T @ y)
    return b, Rinv @ Rinv.T


def factorize(groups) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes ``0..G-1`` (sorted by value) and the unique group labels."""
    codes, uniques = pd.factorize(np.asarray(groups), sort=True)
    return codes.astype(np.int64), np.asarray(uniques)


def cluster_sum(M: np.ndarray, codes: np.ndarray, n_groups: int | None = None) -> np.ndarray:
    """Sum the rows of ``M`` within clusters; returns a ``G x K`` array."""
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    g = int(codes.max()) + 1 if n_groups is None else n_groups
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    sums = np.add.reduceat(M[order], starts, axis=0)
    if sums.shape[0] != g:
        out = np.zeros((g, M.shape[1]))
        out[sorted_codes[starts]] = sums
        return out
    return sums


def factor_dummies(values, prefix: str, base=None) -> pd.DataFrame:
    """Indicator columns for the levels of ``values`` (Stata ``i.var``).

    The base level defaults to the smallest level present in ``values``.  The
    columns are named ``"<prefix>.<level>"`` and returned as float.
    """
    s = pd.Series(np.asarray(values))
    levels = np.sort(s.unique())
    base = levels[0] if base is None else base
    cols = {f"{prefix}.{lev}": (s == lev).to_numpy(dtype=float) for lev in levels if lev != base}
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------
# Small-sample factors and the relations between conventions
# ----------------------------------------------------------------------------
def small_sample_factor(n: int, k: int, g: int, kind: str = "regress") -> float:
    """Scalar multiplying the unscaled cluster sandwich ``S_0``.

    Parameters
    ----------
    n, k, g : int
        Observations, estimated parameters that enter the correction, clusters.
    kind : {"regress", "report", "none"}
        ``"regress"``: ``G/(G-1) * (N-1)/(N-K)`` (Stata ``regress``, ``xtreg, fe``
        and ``boottest`` all use this form; they differ in which ``K`` is passed).
        ``"report"``: ``N/(N-K)``, the factor implied by the ``friend_did``
        rescaling.  ``"none"``: 1.
    """
    if kind == "regress":
        return (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    if kind == "report":
        return n / (n - k)
    if kind == "none":
        return 1.0
    raise ValueError(f"unknown small-sample kind: {kind!r}")


def cluster_variance_scale(convention: str, n: int, g: int, k_full: int, n_absorbed: int | None = None) -> float:
    """Scale ``c`` such that ``Var = c * S_0`` for a named convention.

    ``k_full`` is the number of estimated parameters of the regression with all
    fixed effects entered as dummies (``K = e(N) - e(df_r)`` after ``regress``).
    ``n_absorbed`` is the number of fixed-effect parameters that ``xtreg, fe``
    absorbs, ``G - 1`` for unit effects with the cluster equal to the unit.

    Conventions
    -----------
    ``"regress_dummies"``  ``G/(G-1) * (N-1)/(N-K)``
    ``"xtreg_fe"``         ``G/(G-1) * (N-1)/(N-K+A)`` with ``A = n_absorbed``
    ``"boottest_xtreg_fe"``  as ``xtreg_fe`` with ``K`` reduced by the constant,
                           ``G/(G-1) * (N-1)/(N-K+A+1)``
    ``"report"``           ``N/(N-K)``
    ``"cr0"``              1
    """
    a = (g - 1) if n_absorbed is None else n_absorbed
    if convention == "regress_dummies":
        return small_sample_factor(n, k_full, g, "regress")
    if convention == "xtreg_fe":
        return small_sample_factor(n, k_full - a, g, "regress")
    if convention == "boottest_xtreg_fe":
        return small_sample_factor(n, k_full - a - 1, g, "regress")
    if convention == "report":
        return small_sample_factor(n, k_full, g, "report")
    if convention == "cr0":
        return 1.0
    raise ValueError(f"unknown convention: {convention!r}")


def se_ratio(num: str, den: str, n: int, g: int, k_full: int, n_absorbed: int | None = None) -> float:
    """Ratio ``SE_num / SE_den`` implied by two conventions (closed form).

    Both standard errors are built on the same sandwich ``S_0``, so the ratio
    depends only on ``N``, ``G`` and ``K``.
    """
    c_num = cluster_variance_scale(num, n, g, k_full, n_absorbed)
    c_den = cluster_variance_scale(den, n, g, k_full, n_absorbed)
    return float(np.sqrt(c_num / c_den))


def report_formula_se(se_regress_cluster, n: int, g: int):
    """Rescale a ``regress, vce(cluster)`` standard error as in ``friend_did``.

    ``se_report = se * sqrt(N * (G - 1) / (G * (N - 1)))``.  The product of this
    factor with the ``regress`` small-sample factor equals ``N / (N - K)``.
    """
    return np.asarray(se_regress_cluster) * np.sqrt(n * (g - 1.0) / (g * (n - 1.0)))


# ----------------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------------
@dataclass
class RegResult:
    """Result of :func:`regress` (and base of :class:`XtregResult`).

    Attributes
    ----------
    names, b, V
        Coefficient names, estimates, covariance matrix (small-sample scaled).
    n, k
        Observations and estimated parameters (rank of the design).
    n_clusters
        Number of clusters, or ``None`` for the classical covariance.
    df_r
        Residual degrees of freedom used for t inference: ``G - 1`` with
        clustering, ``N - K`` otherwise.
    ssc
        The small-sample scalar applied to the unscaled sandwich.
    """

    names: list[str]
    b: np.ndarray
    V: np.ndarray
    n: int
    k: int
    n_clusters: int | None
    df_r: int
    resid: np.ndarray
    ssr: float
    tss: float
    r2: float
    root_mse: float
    ssc: float
    vce: str
    omitted: list[str] = field(default_factory=list)
    XtXinv: np.ndarray | None = None
    S0: np.ndarray | None = None

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.V), 0.0))

    def index(self, name: str) -> int:
        return self.names.index(name)

    def coef(self, name: str) -> float:
        return float(self.b[self.index(name)])

    def std_err(self, name: str) -> float:
        return float(self.se[self.index(name)])

    def tstat(self, name: str) -> float:
        return self.coef(name) / self.std_err(name)

    def table(self, level: float = 0.95, dist: str = "t", df: float | None = None) -> pd.DataFrame:
        """Coefficient table with t (or z), p-value and confidence interval."""
        se = self.se
        t = np.divide(self.b, se, out=np.full_like(self.b, np.nan), where=se > 0)
        if dist == "t":
            d = self.df_r if df is None else df
            crit = invttail(d, (1 - level) / 2)
            p = t_pvalue(t, d)
        elif dist == "z":
            crit = invnormal(1 - (1 - level) / 2)
            p = z_pvalue(t)
        else:
            raise ValueError("dist must be 't' or 'z'")
        return pd.DataFrame(
            {"coef": self.b, "se": se, "t": t, "p": p, "ci_lo": self.b - crit * se, "ci_hi": self.b + crit * se},
            index=self.names,
        )


@dataclass
class XtregResult(RegResult):
    """Result of :func:`xtreg_fe`; adds the panel statistics Stata prints."""

    n_groups: int = 0
    r2_within: float = np.nan
    r2_between: float = np.nan
    r2_overall: float = np.nan
    sigma_u: float = np.nan
    sigma_e: float = np.nan
    rho: float = np.nan
    corr_u_xb: float = np.nan
    obs_per_group: tuple[int, float, int] = (0, 0.0, 0)
    F_df1: int = 0
    F_df2: int = 0
    F: float = np.nan
    prob_F: float = np.nan
    fe: np.ndarray | None = None
    sigma_u_variants: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# regress
# ----------------------------------------------------------------------------
def _as_matrix(X, names):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(X.columns) if names is None else list(names)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X, ([f"x{i}" for i in range(X.shape[1])] if names is None else list(names))


def regress(
    y,
    X,
    names: Sequence[str] | None = None,
    cluster=None,
    constant: bool = True,
    ssc: str = "regress",
) -> RegResult:
    """Stata ``regress y X [, vce(cluster c)]``.

    Parameters
    ----------
    y : array-like, shape (N,)
    X : array-like or DataFrame, shape (N, K0)
        Regressors, without the constant.
    cluster : array-like, optional
        Cluster identifiers.  With clustering the covariance is
        ``G/(G-1) * (N-1)/(N-K) * S_0`` and ``df_r = G - 1``.
    constant : bool
        Append a constant column named ``_cons`` (last, as Stata reports it).
    ssc : {"regress", "report"}
        Small-sample scalar for the clustered covariance.  ``"report"`` applies
        ``N/(N-K)`` instead (the ``friend_did`` convention before inference on
        ``t(N-K)``).

    Returns
    -------
    RegResult
        Collinear columns are dropped in order and listed in ``omitted``.
    """
    y = np.asarray(y, dtype=float).ravel()
    Xm, nm = _as_matrix(X, names)
    if constant:
        # The constant is examined first, so that it is retained and a redundant
        # regressor is dropped; it is then reported last, as Stata does.
        keep_all = drop_collinear(np.column_stack([np.ones(len(y)), Xm]))
        keep = np.r_[keep_all[1:], keep_all[0]]
        Xm = np.column_stack([Xm, np.ones(len(y))])
        nm = nm + ["_cons"]
    else:
        keep = drop_collinear(Xm)
    omitted = [n for n, kp in zip(nm, keep) if not kp]
    Xk = Xm[:, keep]
    nk = [n for n, kp in zip(nm, keep) if kp]
    n, k = Xk.shape
    b, XtXinv = _ols_qr(Xk, y)
    resid = y - Xk @ b
    ssr = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum()) if constant else float((y**2).sum())
    r2 = 1.0 - ssr / tss if tss > 0 else np.nan
    root_mse = float(np.sqrt(ssr / (n - k)))
    S0 = None
    if cluster is None:
        V = (ssr / (n - k)) * XtXinv
        n_clusters = None
        df_r = n - k
        c = ssr / (n - k)
        vce = "ols"
    else:
        codes, _ = factorize(cluster)
        g = int(codes.max()) + 1
        scores = cluster_sum(Xk * resid[:, None], codes, g)
        S0 = XtXinv @ (scores.T @ scores) @ XtXinv
        c = small_sample_factor(n, k, g, ssc)
        V = c * S0
        n_clusters = g
        df_r = g - 1
        vce = "cluster"
    return RegResult(
        names=nk,
        b=b,
        V=V,
        n=n,
        k=k,
        n_clusters=n_clusters,
        df_r=df_r,
        resid=resid,
        ssr=ssr,
        tss=tss,
        r2=r2,
        root_mse=root_mse,
        ssc=c,
        vce=vce,
        omitted=omitted,
        XtXinv=XtXinv,
        S0=S0,
    )


# ----------------------------------------------------------------------------
# xtreg, fe
# ----------------------------------------------------------------------------
def _squared_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def xtreg_fe(
    y,
    X,
    group,
    names: Sequence[str] | None = None,
    cluster=None,
) -> XtregResult:
    """Stata ``xtreg y X, fe [vce(cluster c)]``.

    The within estimator is computed on ``y - ybar_g + ybar`` and
    ``X - Xbar_g + Xbar`` with a constant, so that ``_cons = ybar - Xbar b`` as in
    Stata.  Time-invariant regressors are dropped as collinear.

    Covariance
    ----------
    With ``cluster`` given, ``V = c * S_0`` with ``c = G_c/(G_c-1) * (N-1)/(N-K_w)``
    where ``K_w`` is the number of non-absorbed estimated parameters (slopes plus
    the constant) and ``G_c`` the number of clusters; inference uses ``t(G_c-1)``.
    Without ``cluster``, ``V = sigma_e^2 (Z'Z)^{-1}`` with
    ``sigma_e^2 = SSR / (N - G - p)`` for ``p`` slopes.

    Panel statistics
    ----------------
    ``r2_within`` is the ordinary R-squared of the within regression;
    ``r2_between`` and ``r2_overall`` are squared correlations of ``Xb`` with the
    outcome in group means and in observations; ``sigma_e = sqrt(SSR/(N-G-p))``
    (the root mean squared error adjusted for the ``G - 1`` estimated means);
    ``sigma_u`` is the sample standard deviation (divisor ``G - 1``) of the
    estimated group effects ``u_g = ybar_g - xbar_g b - _cons``;
    ``rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)``; ``corr_u_xb`` is the
    correlation over observations of ``u_g`` and the linear prediction ``x b``.
    The attribute ``sigma_u_variants`` lists alternative definitions (standard
    deviation over observations; subtraction of ``sigma_e^2 / Tbar`` with the
    arithmetic or harmonic mean group size) for diagnostic comparison.
    """
    y = np.asarray(y, dtype=float).ravel()
    Xm, nm = _as_matrix(X, names)
    gcodes, _ = factorize(group)
    n_groups = int(gcodes.max()) + 1
    n = len(y)
    counts = np.bincount(gcodes, minlength=n_groups).astype(float)
    ybar_g = np.bincount(gcodes, weights=y, minlength=n_groups) / counts
    Xbar_g = np.column_stack(
        [np.bincount(gcodes, weights=Xm[:, j], minlength=n_groups) / counts for j in range(Xm.shape[1])]
    )
    y_t = y - ybar_g[gcodes] + y.mean()
    Xw = Xm - Xbar_g[gcodes]
    X_t = Xw + Xm.mean(axis=0)
    Z = np.column_stack([X_t, np.ones(n)])
    znames = nm + ["_cons"]
    # Collinearity is judged on the within-transformed regressors: a time-invariant
    # regressor is identically zero after demeaning and is dropped (the constant stays).
    varying = np.linalg.norm(Xw, axis=0) > 1e-10 * np.linalg.norm(Xm, axis=0)
    keep = np.zeros(Z.shape[1], dtype=bool)
    keep[-1] = True
    if varying.any():
        sub = np.flatnonzero(varying)
        keep[sub[drop_collinear(Xw[:, sub])]] = True
    omitted = [a for a, kp in zip(znames, keep) if not kp]
    Zk = Z[:, keep]
    nk = [a for a, kp in zip(znames, keep) if kp]
    k_w = Zk.shape[1]
    p = k_w - 1
    b, ZtZinv = _ols_qr(Zk, y_t)
    resid = y_t - Zk @ b
    ssr = float(resid @ resid)
    df_e = n - n_groups - p
    sigma_e2 = ssr / df_e
    S0 = None
    if cluster is None:
        V = sigma_e2 * ZtZinv
        n_clusters = None
        df_r = df_e
        c = sigma_e2
        vce = "ols"
    else:
        ccodes, _ = factorize(cluster)
        gc = int(ccodes.max()) + 1
        scores = cluster_sum(Zk * resid[:, None], ccodes, gc)
        S0 = ZtZinv @ (scores.T @ scores) @ ZtZinv
        c = small_sample_factor(n, k_w, gc, "regress")
        V = c * S0
        n_clusters = gc
        df_r = gc - 1
        vce = "cluster"

    # Panel statistics.
    kept_pos = np.flatnonzero(keep)
    slope_idx = [i for i, pos in enumerate(kept_pos) if pos < Xm.shape[1]]
    slope_cols = [int(kept_pos[i]) for i in slope_idx]
    beta = b[slope_idx]
    Xk_raw = Xm[:, slope_cols] if slope_cols else np.zeros((n, 0))
    xb = Xk_raw @ beta
    xb_bar_g = np.bincount(gcodes, weights=xb, minlength=n_groups) / counts
    u_g = ybar_g - xb_bar_g
    tss_within = float(((y - ybar_g[gcodes]) ** 2).sum())
    r2_within = 1.0 - ssr / tss_within if tss_within > 0 else np.nan
    r2_between = _squared_corr(xb_bar_g, ybar_g)
    r2_overall = _squared_corr(xb, y)
    corr_u_xb = float(np.corrcoef(u_g[gcodes], xb)[0, 1]) if np.std(xb) > 0 and np.std(u_g) > 0 else np.nan
    var_u = float(np.var(u_g, ddof=1)) if n_groups > 1 else np.nan
    sigma_u = float(np.sqrt(var_u)) if n_groups > 1 else np.nan
    sigma_e = float(np.sqrt(sigma_e2))
    rho = sigma_u**2 / (sigma_u**2 + sigma_e**2) if (sigma_u**2 + sigma_e**2) > 0 else np.nan
    # Alternative definitions of sigma_u, kept for diagnostics when a printed value is compared.
    tbar_arith = n / n_groups
    tbar_harm = n_groups / float(np.sum(1.0 / counts))
    u_obs = u_g[gcodes]
    sigma_u_variants = {
        "sd_groups": sigma_u,
        "sd_obs": float(np.std(u_obs, ddof=1)),
        "adj_arith": float(np.sqrt(max(0.0, var_u - sigma_e2 / tbar_arith))) if n_groups > 1 else np.nan,
        "adj_harm": float(np.sqrt(max(0.0, var_u - sigma_e2 / tbar_harm))) if n_groups > 1 else np.nan,
    }

    # Wald F for the joint significance of the slopes (missing when V is singular).
    f_df2 = (n_clusters - 1) if n_clusters is not None else df_e
    f_df1 = 0
    f_stat = np.nan
    prob_f = np.nan
    if p > 0:
        Vs = V[np.ix_(slope_idx, slope_idx)]
        rank = int(np.linalg.matrix_rank(Vs, tol=1e-9 * max(1.0, np.abs(Vs).max())))
        f_df1 = rank
        if rank == p and f_df2 > 0:
            f_stat = float(beta @ np.linalg.solve(Vs, beta) / p)
            prob_f = float(stats.f.sf(f_stat, p, f_df2))
    sizes = (int(counts.min()), float(counts.mean()), int(counts.max()))
    return XtregResult(
        names=nk,
        b=b,
        V=V,
        n=n,
        k=k_w,
        n_clusters=n_clusters,
        df_r=df_r,
        resid=resid,
        ssr=ssr,
        tss=tss_within,
        r2=r2_within,
        root_mse=sigma_e,
        ssc=c,
        vce=vce,
        omitted=omitted,
        XtXinv=ZtZinv,
        S0=S0,
        n_groups=n_groups,
        r2_within=r2_within,
        r2_between=r2_between,
        r2_overall=r2_overall,
        sigma_u=sigma_u,
        sigma_e=sigma_e,
        rho=rho,
        corr_u_xb=corr_u_xb,
        obs_per_group=sizes,
        F_df1=f_df1,
        F_df2=f_df2,
        F=f_stat,
        prob_F=prob_f,
        fe=u_g,
        sigma_u_variants=sigma_u_variants,
    )
