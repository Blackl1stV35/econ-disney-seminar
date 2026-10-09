"""Difference-in-differences tools reproducing the Stata analysis.

Contents
--------
:func:`friend_did`
    Two-way fixed-effects DiD with the report-formula standard error
    (``regress ... i.unit_id i.year, vce(cluster unit_id)`` followed by the
    rescaling ``sqrt(N (G-1) / (G (N-1)))`` and inference on ``t(N-K)``).
:func:`xtreg_twfe`
    ``xtreg y treated_post i.year, fe vce(cluster unit_id)`` with inference on
    ``t(G-1)``.
:func:`pretrend_test`
    ``regress y t c.t#c.treated i.unit_id, vce(cluster unit_id)`` on the
    pre-treatment sample, with t(G-1) and normal inference.
:func:`wild_cluster_bootstrap`
    Wild cluster bootstrap-t with the null imposed (restricted residuals) and
    Webb six-point, Rademacher or Mammen weights; Monte Carlo or exhaustive
    enumeration; confidence set by test inversion.
:func:`boottest_regress`, :func:`boottest_xtreg_fe`
    Wrappers that apply the small-sample convention of ``boottest`` after
    ``regress`` and after ``xtreg, fe``.
:func:`event_study`
    Leads and lags with binned end points and a base period, estimated by
    ``xtreg, fe vce(cluster)``.

Algebra of the bootstrap
------------------------
Let ``a`` be row ``j`` of ``(X'X)^{-1}``, ``u~`` the residual of the model with
``beta_j = r`` imposed, ``m_g = X_g' u~_g`` and ``s_g = a m_g``.  With cluster
weights ``w`` the bootstrap numerator is ``sum_g w_g s_g`` and the bootstrap
cluster score for cluster ``g`` is ``w_g s_g - sum_h C_gh w_h`` where
``C_gh = a X_g' X_g (X'X)^{-1} m_h``.  Every bootstrap statistic is therefore a
function of the ``G``-vector ``s`` and the ``G x G`` matrix ``C`` only, which
makes ``B`` draws cost ``O(B G^2)``.  Because ``u~(r) = u~(0) - r x~_j``, both
``s`` and ``C`` are affine in ``r``, so the p-value as a function of the null
value, and hence the confidence set, is cheap to evaluate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .stata_compat import (
    RegResult,
    XtregResult,
    cluster_sum,
    factor_dummies,
    factorize,
    invnormal,
    invttail,
    regress,
    report_formula_se,
    small_sample_factor,
    t_pvalue,
    ttail,
    xtreg_fe,
    z_pvalue,
    _ols_qr,
)

__all__ = [
    "DiDResult",
    "friend_did",
    "xtreg_twfe",
    "PretrendResult",
    "pretrend_test",
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
    "boottest_regress",
    "boottest_xtreg_fe",
    "enumerate_weights",
    "draw_weights",
    "EventStudyResult",
    "event_study",
]

Y_DEFAULT = "receipts_pct_gdp"


# ----------------------------------------------------------------------------
# Two-way fixed-effects DiD
# ----------------------------------------------------------------------------
@dataclass
class DiDResult:
    """Result of :func:`friend_did`."""

    label: str
    coef: float
    se: float
    se_cluster_regress: float
    t: float
    df: int
    p: float
    ci_lo: float
    ci_hi: float
    n: int
    n_clusters: int
    k: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _twfe_design(df: pd.DataFrame, treat: str, unit: str, year: str) -> pd.DataFrame:
    return pd.concat(
        [
            df[[treat]].astype(float).reset_index(drop=True),
            factor_dummies(df[unit].to_numpy(), unit).reset_index(drop=True),
            factor_dummies(df[year].to_numpy(), year).reset_index(drop=True),
        ],
        axis=1,
    )


def friend_did(
    df: pd.DataFrame,
    y: str = Y_DEFAULT,
    treat: str = "treated_post",
    unit: str = "unit_id",
    year: str = "year",
    label: str = "",
) -> DiDResult:
    """DiD with the report-formula standard error (Stata program ``friend_did``).

    Steps: ``regress y treat i.unit i.year, vce(cluster unit)``; the clustered
    standard error is multiplied by ``sqrt(N (G-1) / (G (N-1)))``; inference
    uses ``t(N-K)`` with ``K`` the number of estimated parameters of the
    dummy-variable regression.  The result equals the unscaled cluster sandwich
    multiplied by ``N/(N-K)``.
    """
    X = _twfe_design(df, treat, unit, year)
    fit = regress(df[y].to_numpy(), X, cluster=df[unit].to_numpy())
    n, g, k = fit.n, fit.n_clusters, fit.k
    b = fit.coef(treat)
    se = float(report_formula_se(fit.std_err(treat), n, g))
    df_t = n - k
    t = b / se
    crit = float(invttail(df_t, 0.025))
    return DiDResult(
        label=label,
        coef=b,
        se=se,
        se_cluster_regress=fit.std_err(treat),
        t=t,
        df=df_t,
        p=float(t_pvalue(t, df_t)),
        ci_lo=b - crit * se,
        ci_hi=b + crit * se,
        n=n,
        n_clusters=g,
        k=k,
    )


def xtreg_twfe(
    df: pd.DataFrame,
    y: str = Y_DEFAULT,
    treat: str = "treated_post",
    unit: str = "unit_id",
    year: str = "year",
) -> XtregResult:
    """``xtreg y treat i.year, fe vce(cluster unit)`` (inference on ``t(G-1)``)."""
    X = pd.concat(
        [df[[treat]].astype(float).reset_index(drop=True), factor_dummies(df[year].to_numpy(), year).reset_index(drop=True)],
        axis=1,
    )
    return xtreg_fe(df[y].to_numpy(), X, df[unit].to_numpy(), cluster=df[unit].to_numpy())


# ----------------------------------------------------------------------------
# Pre-trend test
# ----------------------------------------------------------------------------
@dataclass
class PretrendResult:
    """Result of :func:`pretrend_test`.

    ``p_t`` and ``ci_t`` use ``t(G-1)`` (Stata ``regress, vce(cluster)``);
    ``z`` and ``p_z`` use the normal approximation of the ``friend_pre`` program.
    """

    coef: float
    se: float
    t: float
    p_t: float
    ci_t: tuple[float, float]
    z: float
    p_z: float
    ci_z: tuple[float, float]
    n: int
    n_clusters: int
    fit: RegResult = field(repr=False, default=None)
    X: pd.DataFrame | None = field(repr=False, default=None)
    y: np.ndarray | None = field(repr=False, default=None)
    cluster: np.ndarray | None = field(repr=False, default=None)


def pretrend_test(
    df_pre: pd.DataFrame,
    y: str = Y_DEFAULT,
    t_col: str = "t",
    treated: str = "treated",
    unit: str = "unit_id",
) -> PretrendResult:
    """Linear differential pre-trend: ``y ~ t + t:treated + i.unit``, clustered by unit.

    The tested coefficient is that of the interaction ``t x treated``.  The
    interaction is invariant to the origin of ``t`` because the unit effects
    absorb level shifts.
    """
    t = df_pre[t_col].to_numpy(dtype=float)
    tr = df_pre[treated].to_numpy(dtype=float)
    X = pd.concat(
        [
            pd.DataFrame({"t": t, "c.t#c.treated": t * tr}),
            factor_dummies(df_pre[unit].to_numpy(), unit),
        ],
        axis=1,
    )
    fit = regress(df_pre[y].to_numpy(), X, cluster=df_pre[unit].to_numpy())
    name = "c.t#c.treated"
    b, se = fit.coef(name), fit.std_err(name)
    g = fit.n_clusters
    tstat = b / se
    crit_t = float(invttail(g - 1, 0.025))
    zcrit = float(invnormal(0.975))
    return PretrendResult(
        coef=b,
        se=se,
        t=tstat,
        p_t=float(t_pvalue(tstat, g - 1)),
        ci_t=(b - crit_t * se, b + crit_t * se),
        z=tstat,
        p_z=float(z_pvalue(tstat)),
        ci_z=(b - zcrit * se, b + zcrit * se),
        n=fit.n,
        n_clusters=g,
        fit=fit,
        X=X,
        y=df_pre[y].to_numpy(dtype=float),
        cluster=df_pre[unit].to_numpy(),
    )


# ----------------------------------------------------------------------------
# Bootstrap weights
# ----------------------------------------------------------------------------
_S5 = np.sqrt(5.0)
_SUPPORTS = {
    # name: (values, probabilities, symmetric)
    "webb": (np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]), np.full(6, 1 / 6), True),
    "rademacher": (np.array([-1.0, 1.0]), np.array([0.5, 0.5]), True),
    "mammen": (
        np.array([(1 - _S5) / 2, (1 + _S5) / 2]),
        np.array([(_S5 + 1) / (2 * _S5), (_S5 - 1) / (2 * _S5)]),
        False,
    ),
}


def draw_weights(kind: str, g: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``reps x g`` array of independent cluster weights of the given type."""
    values, probs, _ = _SUPPORTS[kind]
    return rng.choice(values, size=(reps, g), p=probs)


def enumerate_weights(kind: str, g: int, max_rows: int = 40_000_000) -> tuple[np.ndarray, int]:
    """All weight vectors of a symmetric distribution, up to global sign.

    Returns ``(W, multiplicity)``.  Since the bootstrap statistic depends on the
    weights only through ``|t*|`` and ``t*(-w) = -t*(w)``, vectors with a
    positive first coordinate represent every vector together with its negation;
    ``multiplicity`` is therefore 2.  The full enumeration contains ``m**g``
    vectors for ``m`` support points; ``W`` contains half of them.
    """
    values, _, symmetric = _SUPPORTS[kind]
    if not symmetric:
        raise ValueError("exhaustive enumeration requires a symmetric weight distribution")
    m = len(values)
    total = m**g // 2
    if total > max_rows:
        raise ValueError(f"enumeration needs {total} rows (> max_rows={max_rows})")
    pos = values[values > 0]
    grids = np.meshgrid(pos, *([values] * (g - 1)), indexing="ij")
    W = np.stack([a.reshape(-1) for a in grids], axis=1)
    return W, 2


# ----------------------------------------------------------------------------
# Wild cluster bootstrap-t, null imposed
# ----------------------------------------------------------------------------
@dataclass
class WildBootstrapResult:
    """Result of :func:`wild_cluster_bootstrap`.

    Attributes
    ----------
    coef, se
        Estimate and cluster-robust standard error (small-sample scaled).
    t_stat
        ``(coef - null) / se``.
    p_value
        Symmetric bootstrap p-value ``P(|t*| > |t|)`` (strict inequality).
    ci
        Confidence set for the tested coefficient, obtained by inverting the
        test; ``(nan, nan)`` if not requested.
    reps
        Number of weight vectors represented (``m**G`` when exhaustive).
    exhaustive
        True when every weight vector was enumerated.
    """

    term: str
    null: float
    coef: float
    se: float
    t_stat: float
    df: int
    p_value: float
    ci: tuple[float, float]
    reps: int
    weights: str
    exhaustive: bool
    seed: int | None
    n_clusters: int
    ssc: float
    t_boot: np.ndarray | None = field(default=None, repr=False)
    ci_band: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass
class _WCRContext:
    g: int
    c: float
    b_j: float
    se_j: float
    s0: np.ndarray
    s1: np.ndarray
    C0: np.ndarray
    C1: np.ndarray


def _wcr_context(y: np.ndarray, X: np.ndarray, codes: np.ndarray, j: int, ssc_k: int | None) -> _WCRContext:
    n, k = X.shape
    g = int(codes.max()) + 1
    b, XtXinv = _ols_qr(X, y)
    a = XtXinv[j]
    e = y - X @ b
    scores_j = cluster_sum(X * e[:, None], codes, g) @ a
    k_ssc = k if ssc_k is None else ssc_k
    c = small_sample_factor(n, k_ssc, g, "regress")
    se_j = float(np.sqrt(c * (scores_j**2).sum()))
    Xr = np.delete(X, j, axis=1)
    if Xr.shape[1] > 0:
        Qr, _ = np.linalg.qr(Xr, mode="reduced")
        u0 = y - Qr @ (Qr.T @ y)
        xt = X[:, j] - Qr @ (Qr.T @ X[:, j])
    else:
        u0, xt = y.copy(), X[:, j].copy()
    m0 = cluster_sum(X * u0[:, None], codes, g)
    m1 = cluster_sum(X * xt[:, None], codes, g)
    Q = cluster_sum(X * (X @ a)[:, None], codes, g) @ XtXinv
    return _WCRContext(g=g, c=c, b_j=float(b[j]), se_j=se_j, s0=m0 @ a, s1=m1 @ a, C0=Q @ m0.T, C1=Q @ m1.T)


def _wcr_tstar(ctx: _WCRContext, W: np.ndarray, r: float, out: np.ndarray | None = None) -> np.ndarray:
    s = ctx.s0 - r * ctx.s1
    C = ctx.C0 - r * ctx.C1
    num = W @ s
    E = W * s[None, :] - W @ C.T
    se2 = ctx.c * np.einsum("ij,ij->i", E, E)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / np.sqrt(se2)


def _pvalue(ctx: _WCRContext, W: np.ndarray, r: float, tie_tol: float, chunk: int = 400_000) -> float:
    t_obs = abs((ctx.b_j - r) / ctx.se_j)
    thresh = t_obs * (1.0 + tie_tol)
    count = 0
    for lo in range(0, W.shape[0], chunk):
        ts = _wcr_tstar(ctx, W[lo : lo + chunk], r)
        count += int(np.count_nonzero(np.abs(ts) > thresh))
    return count / W.shape[0]


def _invert_test(ctx: _WCRContext, W: np.ndarray, alpha: float, tie_tol: float) -> tuple[float, float]:
    """Confidence set ``{r: p(r) > alpha}`` as the outermost crossing on each side."""

    def pv(r: float) -> float:
        return _pvalue(ctx, W, r, tie_tol)

    out = []
    scale = ctx.se_j if ctx.se_j > 0 else 1e-3 * (1.0 + abs(ctx.b_j))
    for direction in (-1.0, 1.0):
        step = scale
        r_out = ctx.b_j + direction * step
        n_expand = 0
        while pv(r_out) > alpha and n_expand < 80:
            step *= 2.0
            r_out = r_out + direction * step
            n_expand += 1
        grid = np.linspace(ctx.b_j, r_out, 241)
        ok = np.array([pv(r) > alpha for r in grid])
        idx = np.flatnonzero(ok)
        i_last = int(idx[-1]) if idx.size else 0
        if i_last >= len(grid) - 1:
            out.append(float(grid[-1]))
            continue
        lo_r, hi_r = grid[i_last], grid[i_last + 1]  # p > alpha at lo_r, p <= alpha at hi_r
        for _ in range(100):
            mid = 0.5 * (lo_r + hi_r)
            if pv(mid) > alpha:
                lo_r = mid
            else:
                hi_r = mid
            if abs(hi_r - lo_r) <= 1e-12 * (1.0 + abs(mid)):
                break
        out.append(float(0.5 * (lo_r + hi_r)))
    return out[0], out[1]


def wild_cluster_bootstrap(
    y,
    X,
    cluster,
    term: int | str,
    names: Sequence[str] | None = None,
    null: float = 0.0,
    weights: str = "webb",
    reps: int = 9999,
    seed: int | None = 42,
    exhaustive: bool | None = False,
    ssc_k: int | None = None,
    alpha: float = 0.05,
    ci: bool = True,
    tie_tol: float = 1e-9,
    keep_draws: bool = False,
    band: tuple[int, float] | None = None,
) -> WildBootstrapResult:
    """Wild cluster bootstrap-t with the null imposed (WCR), one-way clustering.

    Parameters
    ----------
    y, X : array-like
        Outcome and the full regressor matrix (include dummies and a constant if
        the model has them).  ``X`` must have full column rank.
    cluster : array-like
        Cluster identifiers.
    term : int or str
        Column of ``X`` under test (index or name).
    null : float
        Null value ``r`` of ``H0: beta_term = r``.
    weights : {"webb", "rademacher", "mammen"}
        Distribution of the cluster-level multipliers.
    reps : int
        Number of Monte Carlo draws.  Ignored when ``exhaustive`` is true.
    exhaustive : bool or None
        True enumerates every weight vector (``6**G`` for Webb, ``2**G`` for
        Rademacher; the enumeration uses the sign symmetry of the statistic).
        ``None`` enumerates when the number of vectors does not exceed ``reps``.
    ssc_k : int, optional
        ``K`` in the small-sample factor ``G/(G-1) (N-1)/(N-K)``.  Defaults to
        the number of columns of ``X`` (the ``regress`` convention).  The factor
        multiplies the variance in the original and in every bootstrap statistic,
        so it changes the reported ``t`` but not the p-value or the confidence set.
    alpha : float
        Test level used for the confidence set.
    tie_tol : float
        Relative tolerance below which ``|t*| = |t|`` counts as a tie, and a tie
        is not an exceedance.  Exact ties occur for the weight vectors that
        reproduce the original sample (all weights equal to one).
    band : (int, float), optional
        ``(reps_ref, n_sigma)``.  Also returns ``ci_band``, the range in which a
        confidence-set end point from a Monte Carlo bootstrap with ``reps_ref``
        draws is expected to fall.  It is obtained by inverting the exact test
        at levels ``alpha -/+ n_sigma * sqrt(alpha (1 - alpha) / reps_ref)``.

    Notes
    -----
    The p-value is ``P*(|t*| > |t|)``.  Strict inequality and the symmetric
    definition follow the ``boottest`` documentation.
    """
    y = np.asarray(y, dtype=float).ravel()
    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if names is None else list(names)
        X = X.to_numpy(dtype=float)
    X = np.asarray(X, dtype=float)
    j = names.index(term) if isinstance(term, str) else int(term)
    label = names[j] if names is not None else str(j)
    codes, _ = factorize(cluster)
    g = int(codes.max()) + 1
    ctx = _wcr_context(y, X, codes, j, ssc_k)

    values, _, symmetric = _SUPPORTS[weights]
    total = len(values) ** g
    if exhaustive is None:
        exhaustive = symmetric and total <= reps
    if exhaustive:
        W, mult = enumerate_weights(weights, g)
        n_rep = total
    else:
        rng = np.random.default_rng(seed)
        W = draw_weights(weights, g, reps, rng)
        n_rep = reps
    t_obs = (ctx.b_j - null) / ctx.se_j
    p = _pvalue(ctx, W, null, tie_tol)
    lo, hi = (np.nan, np.nan)
    if ci:
        lo, hi = _invert_test(ctx, W, alpha, tie_tol)
    t_boot = _wcr_tstar(ctx, W, null) if keep_draws else None
    ci_band = None
    if band is not None and ci:
        reps_ref, n_sigma = band
        half = n_sigma * float(np.sqrt(alpha * (1 - alpha) / reps_ref))
        lo_w, hi_w = _invert_test(ctx, W, alpha + half, tie_tol)  # narrower set
        lo_n, hi_n = _invert_test(ctx, W, max(alpha - half, 1e-6), tie_tol)  # wider set
        ci_band = ((lo_n, lo_w), (hi_w, hi_n))
    return WildBootstrapResult(
        term=label,
        null=null,
        coef=ctx.b_j,
        se=ctx.se_j,
        t_stat=float(t_obs),
        df=g - 1,
        p_value=float(p),
        ci=(float(lo), float(hi)),
        reps=int(n_rep),
        weights=weights,
        exhaustive=bool(exhaustive),
        seed=None if exhaustive else seed,
        n_clusters=g,
        ssc=ctx.c,
        t_boot=t_boot,
        ci_band=ci_band,
    )


def boottest_regress(
    y,
    X: pd.DataFrame,
    cluster,
    term: str,
    constant: bool = True,
    **kwargs,
) -> WildBootstrapResult:
    """``boottest`` after ``regress y X, vce(cluster c)``.

    A constant is appended (as Stata does) and ``K`` is the full column count, so
    that the reported ``t`` equals the ``regress`` cluster t-statistic.
    """
    Xd = X.copy()
    if constant:
        Xd["_cons"] = 1.0
    return wild_cluster_bootstrap(y, Xd, cluster, term, **kwargs)


def boottest_xtreg_fe(
    y,
    X: pd.DataFrame,
    group,
    term: str,
    cluster=None,
    **kwargs,
) -> WildBootstrapResult:
    """``boottest`` after ``xtreg y X, fe vce(cluster c)``.

    The fixed effects are absorbed by within-demeaning the outcome and the
    regressors.  The small-sample factor uses ``K`` equal to the number of
    regressors without the constant, which reproduces the ``t(G-1)`` statistic
    that ``boottest`` reports after ``xtreg, fe``.
    """
    codes, _ = factorize(group)
    ng = int(codes.max()) + 1
    cnt = np.bincount(codes, minlength=ng).astype(float)
    Xm = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float).ravel()
    Xd = Xm - np.column_stack([np.bincount(codes, weights=Xm[:, i], minlength=ng) / cnt for i in range(Xm.shape[1])])[codes]
    yd = yv - (np.bincount(codes, weights=yv, minlength=ng) / cnt)[codes]
    cl = group if cluster is None else cluster
    kwargs.setdefault("ssc_k", Xd.shape[1])
    return wild_cluster_bootstrap(yd, Xd, cl, term, names=list(X.columns), **kwargs)


# ----------------------------------------------------------------------------
# Event study
# ----------------------------------------------------------------------------
@dataclass
class EventStudyResult:
    """Event-study coefficients relative to the base period.

    ``table`` has one row per relative year (including the base row, fixed at
    zero) with columns ``rel, coef, se, t, p, ci_lo, ci_hi``; confidence
    intervals use ``t(G-1)``.  ``fit`` is the underlying :class:`XtregResult`.
    """

    table: pd.DataFrame
    fit: XtregResult = field(repr=False)
    base: int
    lo: int | None
    hi: int | None


def event_study(
    df: pd.DataFrame,
    y: str = Y_DEFAULT,
    unit: str = "unit_id",
    year: str = "year",
    rel: str = "rel_year",
    treated: str = "treated",
    lo: int | None = None,
    hi: int | None = None,
    base: int = -1,
) -> EventStudyResult:
    """Event study with binned end points and a base period.

    The event time is the relative year for the treated unit and ``base`` for all
    other observations, which places the comparison units in the reference
    category.  Event times are clipped to ``[lo, hi]`` (binning the end points),
    year dummies and unit fixed effects are included, and the model is
    estimated by ``xtreg, fe vce(cluster unit)``.
    """
    ev = np.where(df[treated].to_numpy() == 1, df[rel].to_numpy(), base).astype(int)
    if lo is not None:
        ev = np.maximum(ev, lo)
    if hi is not None:
        ev = np.minimum(ev, hi)
    levels = [int(v) for v in np.unique(ev) if v != base]
    ev_cols = pd.DataFrame({f"ev.{lev}": (ev == lev).astype(float) for lev in levels})
    X = pd.concat([ev_cols, factor_dummies(df[year].to_numpy(), year)], axis=1)
    fit = xtreg_fe(df[y].to_numpy(), X, df[unit].to_numpy(), cluster=df[unit].to_numpy())
    tab = fit.table()
    rows = []
    for lev in sorted(levels + [base]):
        if lev == base:
            rows.append(dict(rel=lev, coef=0.0, se=0.0, t=np.nan, p=np.nan, ci_lo=0.0, ci_hi=0.0))
        else:
            r = tab.loc[f"ev.{lev}"]
            rows.append(dict(rel=lev, **{c: float(r[c]) for c in ("coef", "se", "t", "p", "ci_lo", "ci_hi")}))
    return EventStudyResult(table=pd.DataFrame(rows), fit=fit, base=base, lo=lo, hi=hi)
