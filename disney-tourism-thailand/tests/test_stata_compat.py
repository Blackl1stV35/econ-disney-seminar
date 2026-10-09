"""Tests for dtt.stata_compat (Stata-compatible regress and xtreg, fe).

All data in this file are SIMULATED.  The expected values come from closed
forms or from statsmodels, which is an independent implementation.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from dtt import stata_compat as sc


def make_panel(seed=0, g=6, t_lo=10, t_hi=16, p=2):
    """Unbalanced simulated panel with p regressors and unit effects."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(t_lo, t_hi + 1, g)
    unit = np.repeat(np.arange(g), sizes)
    n = unit.size
    X = rng.normal(size=(n, p)) + 0.3 * unit[:, None]
    mu = rng.normal(0, 1.5, g)[unit]
    y = 1.0 + X @ np.arange(1, p + 1) * 0.5 + mu + rng.normal(0, 1.0, n)
    return y, pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), unit


# ----------------------------------------------------------------------------
# Distribution functions
# ----------------------------------------------------------------------------
def test_distribution_functions_match_known_constants():
    assert sc.invttail(5, 0.025) == pytest.approx(2.5705818366, abs=1e-9)
    assert sc.invttail(101, 0.025) == pytest.approx(1.9837, abs=1e-4)
    assert sc.invnormal(0.975) == pytest.approx(1.959963985, abs=1e-9)
    assert sc.normal(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert sc.ttail(5, 2.0) == pytest.approx(stats.t.sf(2.0, 5), rel=1e-14)
    assert sc.t_pvalue(0.0, 7) == pytest.approx(1.0)
    assert sc.t_pvalue(-2.5, 9) == pytest.approx(sc.t_pvalue(2.5, 9))
    assert sc.z_pvalue(1.959963985) == pytest.approx(0.05, abs=1e-9)
    assert sc.t_pvalue(np.array([1.0, 2.0]), 4).shape == (2,)


def test_confidence_limits_of_a_reference_estimate_follow_from_invttail():
    """Reference values, Hong Kong DiD: 5.910137 +/- t(101) * 0.171312 = [5.570300, 6.249974]."""
    crit = sc.invttail(101, 0.025)
    assert 5.910137 - crit * 0.171312 == pytest.approx(5.570300, abs=2e-6)
    assert 5.910137 + crit * 0.171312 == pytest.approx(6.249974, abs=2e-6)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def test_drop_collinear_keeps_the_earlier_column():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=30), rng.normal(size=30)
    assert sc.drop_collinear(np.column_stack([a, b, a + b])).tolist() == [True, True, False]
    assert sc.drop_collinear(np.column_stack([a + b, a, b])).tolist() == [True, True, False]
    assert sc.drop_collinear(np.column_stack([a, a, b])).tolist() == [True, False, True]


def test_drop_collinear_zero_column_and_near_collinearity():
    rng = np.random.default_rng(2)
    a = rng.normal(size=40)
    assert sc.drop_collinear(np.column_stack([a, np.zeros(40)])).tolist() == [True, False]
    near = a + 1e-3 * rng.normal(size=40)
    assert sc.drop_collinear(np.column_stack([a, near])).tolist() == [True, True]
    exact = 3.0 * a
    assert sc.drop_collinear(np.column_stack([a, exact])).tolist() == [True, False]


def test_factorize_sorted_codes_and_labels():
    codes, uniq = sc.factorize(np.array([30, 10, 20, 10, 30]))
    assert uniq.tolist() == [10, 20, 30]
    assert codes.tolist() == [2, 0, 1, 0, 2]
    codes, uniq = sc.factorize(np.array(["b", "a", "b"]))
    assert uniq.tolist() == ["a", "b"] and codes.tolist() == [1, 0, 1]


def test_cluster_sum_matches_groupby():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 5, 100)
    M = rng.normal(size=(100, 3))
    got = sc.cluster_sum(M, codes, 5)
    ref = pd.DataFrame(M).groupby(codes).sum().reindex(range(5), fill_value=0.0).to_numpy()
    assert got == pytest.approx(ref, abs=1e-12)
    assert sc.cluster_sum(M[:, 0], codes, 5).shape == (5, 1)
    sparse = sc.cluster_sum(np.ones(4), np.array([0, 0, 3, 3]), 5)
    assert sparse.ravel().tolist() == [2.0, 0.0, 0.0, 2.0, 0.0]


def test_factor_dummies_naming_and_base_level():
    d = sc.factor_dummies(np.array([2001, 2002, 2003, 2002]), "year")
    assert list(d.columns) == ["year.2002", "year.2003"]
    assert d["year.2002"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert d.dtypes.eq(float).all()
    d2 = sc.factor_dummies(np.array([2001, 2002, 2003, 2002]), "year", base=2002)
    assert list(d2.columns) == ["year.2001", "year.2003"]


# ----------------------------------------------------------------------------
# Small-sample factors and convention relations
# ----------------------------------------------------------------------------
def test_small_sample_factors():
    n, k, g = 129, 28, 6
    assert sc.small_sample_factor(n, k, g, "regress") == pytest.approx(6 / 5 * 128 / 101)
    assert sc.small_sample_factor(n, k, g, "report") == pytest.approx(129 / 101)
    assert sc.small_sample_factor(n, k, g, "none") == 1.0
    with pytest.raises(ValueError):
        sc.small_sample_factor(n, k, g, "other")


def test_cluster_variance_scale_conventions():
    n, g, k = 129, 6, 28
    assert sc.cluster_variance_scale("regress_dummies", n, g, k) == pytest.approx(g / (g - 1) * (n - 1) / (n - k))
    assert sc.cluster_variance_scale("xtreg_fe", n, g, k) == pytest.approx(g / (g - 1) * (n - 1) / (n - k + g - 1))
    assert sc.cluster_variance_scale("boottest_xtreg_fe", n, g, k) == pytest.approx(g / (g - 1) * (n - 1) / (n - k + g))
    assert sc.cluster_variance_scale("report", n, g, k) == pytest.approx(n / (n - k))
    assert sc.cluster_variance_scale("cr0", n, g, k) == 1.0
    assert sc.cluster_variance_scale("xtreg_fe", n, g, k, n_absorbed=2) == pytest.approx(g / (g - 1) * (n - 1) / (n - k + 2))
    with pytest.raises(ValueError):
        sc.cluster_variance_scale("nope", n, g, k)


def test_se_ratio_is_antisymmetric_and_transitive():
    n, g, k = 86, 4, 26
    a = sc.se_ratio("xtreg_fe", "report", n, g, k)
    b = sc.se_ratio("report", "xtreg_fe", n, g, k)
    assert a * b == pytest.approx(1.0)
    c = sc.se_ratio("boottest_xtreg_fe", "xtreg_fe", n, g, k)
    d = sc.se_ratio("boottest_xtreg_fe", "report", n, g, k)
    assert c * a == pytest.approx(d)
    assert sc.se_ratio("report", "report", n, g, k) == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# regress
# ----------------------------------------------------------------------------
def test_regress_classical_matches_statsmodels():
    y, X, _ = make_panel(4)
    fit = sc.regress(y, X)
    ref = sm.OLS(y, sm.add_constant(X.to_numpy())).fit()
    assert fit.names == ["x0", "x1", "_cons"]
    assert fit.b == pytest.approx(np.r_[ref.params[1:], ref.params[0]], rel=1e-10)
    assert fit.se == pytest.approx(np.r_[ref.bse[1:], ref.bse[0]], rel=1e-10)
    assert fit.r2 == pytest.approx(ref.rsquared, rel=1e-10)
    assert fit.root_mse == pytest.approx(np.sqrt(ref.mse_resid), rel=1e-10)
    assert fit.df_r == fit.n - fit.k and fit.n_clusters is None and fit.vce == "ols"


@pytest.mark.parametrize("seed,g", [(5, 4), (6, 7), (7, 12)])
def test_regress_cluster_matches_statsmodels_and_closed_form(seed, g):
    y, X, unit = make_panel(seed, g=g)
    fit = sc.regress(y, X, cluster=unit)
    ref = sm.OLS(y, sm.add_constant(X.to_numpy())).fit(cov_type="cluster", cov_kwds={"groups": unit, "use_correction": True, "df_correction": True})
    assert fit.se == pytest.approx(np.r_[ref.bse[1:], ref.bse[0]], rel=1e-9)
    n, k = fit.n, fit.k
    assert fit.ssc == pytest.approx(g / (g - 1) * (n - 1) / (n - k), rel=1e-14)
    assert fit.df_r == g - 1 and fit.n_clusters == g
    assert fit.V == pytest.approx(fit.ssc * fit.S0, rel=1e-12)


def test_regress_report_convention_scales_by_n_over_n_minus_k():
    y, X, unit = make_panel(8, g=5)
    a = sc.regress(y, X, cluster=unit)
    b = sc.regress(y, X, cluster=unit, ssc="report")
    n, k, g = a.n, a.k, 5
    assert b.ssc == pytest.approx(n / (n - k), rel=1e-14)
    assert b.V == pytest.approx(a.V * (n / (n - k)) / (g / (g - 1) * (n - 1) / (n - k)), rel=1e-12)
    assert b.b == pytest.approx(a.b)


def test_regress_drops_collinear_regressor_and_keeps_constant():
    y, X, unit = make_panel(9)
    X = X.assign(x0_dup=2.0 * X["x0"])
    fit = sc.regress(y, X, cluster=unit)
    assert fit.omitted == ["x0_dup"]
    assert fit.names == ["x0", "x1", "_cons"] and fit.k == 3
    # a complete dummy set together with the constant: a dummy is dropped, not the constant
    D = pd.get_dummies(unit).astype(float)
    D.columns = [f"d{c}" for c in D.columns]
    fit2 = sc.regress(y, pd.concat([X[["x0"]], D], axis=1))
    assert fit2.omitted == ["d5"] and "_cons" in fit2.names


def test_regress_without_constant_uses_uncentered_r2():
    rng = np.random.default_rng(10)
    x = rng.normal(size=50)
    y = 2.0 * x + rng.normal(size=50)
    fit = sc.regress(y, x, constant=False)
    ref = sm.OLS(y, x).fit()
    assert fit.names == ["x0"]
    assert fit.r2 == pytest.approx(ref.rsquared, rel=1e-12)
    assert fit.b[0] == pytest.approx(ref.params[0])


def test_result_table_uses_the_requested_distribution():
    y, X, unit = make_panel(11, g=5)
    fit = sc.regress(y, X, cluster=unit)
    tab_t = fit.table()
    j = fit.index("x0")
    assert tab_t.loc["x0", "t"] == pytest.approx(fit.b[j] / fit.se[j])
    assert tab_t.loc["x0", "p"] == pytest.approx(2 * stats.t.sf(abs(fit.b[j] / fit.se[j]), 4))
    crit = stats.t.isf(0.025, 4)
    assert tab_t.loc["x0", "ci_lo"] == pytest.approx(fit.b[j] - crit * fit.se[j])
    tab_z = fit.table(dist="z")
    assert tab_z.loc["x0", "p"] == pytest.approx(2 * stats.norm.sf(abs(fit.b[j] / fit.se[j])))
    assert tab_z.loc["x0", "ci_hi"] == pytest.approx(fit.b[j] + 1.959963985 * fit.se[j])
    assert fit.table(df=30).loc["x0", "ci_lo"] == pytest.approx(fit.b[j] - stats.t.isf(0.025, 30) * fit.se[j])
    with pytest.raises(ValueError):
        fit.table(dist="chi2")
    assert fit.coef("x1") == fit.b[fit.index("x1")] and fit.tstat("x1") == fit.coef("x1") / fit.std_err("x1")


# ----------------------------------------------------------------------------
# xtreg, fe
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def xt_case():
    y, X, unit = make_panel(12, g=7, p=2)
    return y, X, unit


def _dummy_reference(y, X, unit):
    D = pd.get_dummies(unit, drop_first=True).astype(float).to_numpy()
    Z = sm.add_constant(np.column_stack([X.to_numpy(), D]))
    return sm.OLS(y, Z)


def test_xtreg_slopes_and_classical_se_match_unit_dummy_regression(xt_case):
    y, X, unit = xt_case
    fit = sc.xtreg_fe(y, X, unit)
    ref = _dummy_reference(y, X, unit).fit()
    assert fit.names == ["x0", "x1", "_cons"]
    assert fit.b[:2] == pytest.approx(ref.params[1:3], rel=1e-9)
    assert fit.se[:2] == pytest.approx(ref.bse[1:3], rel=1e-9)
    g, p = 7, 2
    assert fit.df_r == len(y) - g - p
    assert fit.sigma_e == pytest.approx(np.sqrt(ref.ssr / (len(y) - g - p)), rel=1e-10)
    assert fit.n_groups == g and fit.vce == "ols"


def test_xtreg_constant_is_the_grand_mean_intercept(xt_case):
    y, X, unit = xt_case
    fit = sc.xtreg_fe(y, X, unit)
    expected = y.mean() - X.to_numpy().mean(axis=0) @ fit.b[:2]
    assert fit.b[-1] == pytest.approx(expected, rel=1e-10)


def test_xtreg_cluster_variance_uses_stata_small_sample_factor(xt_case):
    y, X, unit = xt_case
    fit = sc.xtreg_fe(y, X, unit, cluster=unit)
    n, g, kw = len(y), 7, 3
    assert fit.k == kw
    assert fit.ssc == pytest.approx(g / (g - 1) * (n - 1) / (n - kw), rel=1e-14)
    assert fit.df_r == g - 1 and fit.n_clusters == g
    # the sandwich equals the dummy-variable regression's, rescaled by the two factors
    ref = _dummy_reference(y, X, unit).fit(cov_type="cluster", cov_kwds={"groups": unit, "use_correction": True, "df_correction": True})
    k_full = ref.params.size
    rescale = (n - k_full) / (n - kw)
    assert fit.se[:2] == pytest.approx(ref.bse[1:3] * np.sqrt(rescale), rel=1e-9)


def test_xtreg_clustering_on_a_coarser_variable(xt_case):
    y, X, unit = xt_case
    coarse = unit // 2
    fit = sc.xtreg_fe(y, X, unit, cluster=coarse)
    assert fit.n_clusters == 4 and fit.df_r == 3
    n, kw = len(y), 3
    assert fit.ssc == pytest.approx(4 / 3 * (n - 1) / (n - kw), rel=1e-14)


def test_xtreg_panel_statistics_follow_documented_definitions(xt_case):
    y, X, unit = xt_case
    fit = sc.xtreg_fe(y, X, unit)
    Xv = X.to_numpy()
    b = fit.b[:2]
    # R-squared within: ordinary R2 of the demeaned regression
    grp = pd.Series(unit)
    yd = y - grp.map(pd.Series(y).groupby(unit).mean()).to_numpy()
    Xd = Xv - pd.DataFrame(Xv).groupby(unit).transform("mean").to_numpy()
    resid = yd - Xd @ b
    assert fit.r2_within == pytest.approx(1 - (resid**2).sum() / (yd**2).sum(), rel=1e-10)
    # group effects, sigma_u (sample sd over groups), rho
    ybar = pd.Series(y).groupby(unit).mean().to_numpy()
    xbar = pd.DataFrame(Xv).groupby(unit).mean().to_numpy()
    u = ybar - xbar @ b
    assert fit.sigma_u == pytest.approx(np.std(u, ddof=1), rel=1e-10)
    assert fit.rho == pytest.approx(fit.sigma_u**2 / (fit.sigma_u**2 + fit.sigma_e**2), rel=1e-12)
    # between and overall R2 are squared correlations
    assert fit.r2_between == pytest.approx(np.corrcoef(xbar @ b, ybar)[0, 1] ** 2, rel=1e-10)
    assert fit.r2_overall == pytest.approx(np.corrcoef(Xv @ b, y)[0, 1] ** 2, rel=1e-10)
    # corr(u_i, Xb) over observations
    assert fit.corr_u_xb == pytest.approx(np.corrcoef(u[pd.factorize(unit, sort=True)[0]], Xv @ b)[0, 1], rel=1e-10)
    sizes = np.bincount(unit)
    assert fit.obs_per_group == (sizes.min(), sizes.mean(), sizes.max())


def test_xtreg_sigma_u_variants_are_reported(xt_case):
    y, X, unit = xt_case
    fit = sc.xtreg_fe(y, X, unit)
    v = fit.sigma_u_variants
    assert set(v) == {"sd_groups", "sd_obs", "adj_arith", "adj_harm"}
    assert v["sd_groups"] == fit.sigma_u
    assert v["adj_arith"] <= v["sd_groups"] and v["adj_harm"] <= v["sd_groups"]


def test_xtreg_drops_time_invariant_regressors_and_keeps_the_constant():
    rng = np.random.default_rng(13)
    g, t = 5, 12
    unit = np.repeat(np.arange(g), t)
    x = rng.normal(size=g * t)
    inv = np.repeat(rng.normal(size=g), t)
    y = 1.0 + 0.5 * x + inv + rng.normal(size=g * t)
    X = pd.DataFrame({"x": x, "inv": inv, "x_twice": 2 * x})
    fit = sc.xtreg_fe(y, X, unit, cluster=unit)
    assert fit.names == ["x", "_cons"]
    assert fit.omitted == ["inv", "x_twice"]
    assert fit.k == 2
    base = sc.xtreg_fe(y, X[["x"]], unit, cluster=unit)
    assert fit.b == pytest.approx(base.b) and fit.se == pytest.approx(base.se)


def test_xtreg_wald_statistic_and_rank_deficiency():
    y, X, unit = make_panel(14, g=7, p=2)
    fit = sc.xtreg_fe(y, X, unit, cluster=unit)
    Vs = fit.V[:2, :2]
    expected_F = fit.b[:2] @ np.linalg.solve(Vs, fit.b[:2]) / 2
    assert fit.F == pytest.approx(expected_F, rel=1e-10)
    assert fit.F_df1 == 2 and fit.F_df2 == 6
    assert fit.prob_F == pytest.approx(stats.f.sf(expected_F, 2, 6), rel=1e-10)
    # more slopes than clusters minus one: the slope covariance is singular, the statistic is missing
    rng = np.random.default_rng(15)
    unit2 = np.repeat(np.arange(3), 15)
    X2 = pd.DataFrame(rng.normal(size=(45, 4)), columns=list("abcd"))
    y2 = rng.normal(size=45)
    fit2 = sc.xtreg_fe(y2, X2, unit2, cluster=unit2)
    assert fit2.F_df1 < 4 and np.isnan(fit2.F)


def test_xtreg_matches_regress_with_unit_dummies_on_the_coefficient_of_interest():
    """Coefficients of xtreg, fe and of the dummy-variable regression are identical."""
    y, X, unit = make_panel(16, g=6)
    D = sc.factor_dummies(unit, "u")
    a = sc.regress(y, pd.concat([X, D], axis=1), cluster=unit)
    b = sc.xtreg_fe(y, X, unit, cluster=unit)
    assert b.b[:2] == pytest.approx(a.b[:2], rel=1e-9)
