"""Tests for dtt.did: DiD, pre-trend test, wild cluster bootstrap and event study.

All data are SIMULATED.  The wild-bootstrap algebra is checked against a brute
force implementation that refits the regression for every weight vector.
"""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dtt import did, simulate
from dtt import stata_compat as sc

Y = "receipts_pct_gdp"


# ----------------------------------------------------------------------------
# friend_did, xtreg_twfe
# ----------------------------------------------------------------------------
def _blocks(df):
    hk = df[df.study_case == "Hong Kong"]
    sh = df[df.study_case == "Shanghai"]
    return hk, hk[hk.unit_id.isin([2, 3, 7, 9])], sh


def test_noiseless_two_way_model_is_recovered_exactly():
    """With y = unit effect + year effect + tau * treated_post, TWFE returns tau to machine precision."""
    df, _ = simulate.simulate_panel(seed=3)
    rng = np.random.default_rng(0)
    ue = dict(zip(sorted(df.unit_id.unique()), rng.normal(0, 2, 9)))
    ye = dict(zip(sorted(df.year.unique()), rng.normal(0, 1, df.year.nunique())))
    for case in ("Hong Kong", "Shanghai"):
        sub = df[df.study_case == case].copy()
        sub[Y] = sub.unit_id.map(ue) + sub.year.map(ye) + 2.5 * sub.treated_post
        r = did.friend_did(sub)
        assert r.coef == pytest.approx(2.5, abs=1e-9)
        x = did.xtreg_twfe(sub)
        assert x.coef("treated_post") == pytest.approx(2.5, abs=1e-9)


def test_friend_did_structure_and_report_formula(sim_parallel):
    df, truth = sim_parallel
    hk, core3, sh = _blocks(df)
    for sub, tau, n_expected, g_expected in ((hk, 5.0, 129, 6), (core3, 5.0, 86, 4), (sh, 1.0, 48, 3)):
        r = did.friend_did(sub, label="x")
        assert (r.n, r.n_clusters) == (n_expected, g_expected)
        assert r.df == r.n - r.k
        assert abs(r.coef - tau) < 0.5  # known effect, parallel trends
        # report formula: variance = S0 * N/(N-K)
        X = did._twfe_design(sub, "treated_post", "unit_id", "year")
        fit = sc.regress(sub[Y].to_numpy(), X, cluster=sub.unit_id.to_numpy())
        j = fit.index("treated_post")
        assert r.se**2 == pytest.approx(fit.S0[j, j] * fit.n / (fit.n - fit.k), rel=1e-12)
        assert r.se_cluster_regress == pytest.approx(fit.std_err("treated_post"))
        assert r.t == pytest.approx(r.coef / r.se)
        assert r.p == pytest.approx(2 * stats.t.sf(abs(r.t), r.df))
        crit = stats.t.isf(0.025, r.df)
        assert (r.ci_lo, r.ci_hi) == pytest.approx((r.coef - crit * r.se, r.coef + crit * r.se))
        assert set(r.as_dict()) >= {"coef", "se", "t", "df", "p", "ci_lo", "ci_hi", "n", "n_clusters", "k"}


def test_xtreg_twfe_is_consistent_with_friend_did(sim_parallel):
    df, _ = sim_parallel
    for sub in _blocks(df):
        r = did.friend_did(sub)
        x = did.xtreg_twfe(sub)
        assert x.coef("treated_post") == pytest.approx(r.coef, rel=1e-9)
        ratio = x.std_err("treated_post") / r.se
        n, g, k = r.n, r.n_clusters, r.k
        assert ratio == pytest.approx(sc.se_ratio("xtreg_fe", "report", n, g, k), rel=1e-9)
        assert x.df_r == g - 1


def test_friend_did_detects_a_false_effect_under_differential_trends(sim_trending):
    """SIMULATED panel with different donor trends: the DiD estimate is far from the true effect."""
    df, _ = sim_trending
    hk, _, _ = _blocks(df)
    assert abs(did.friend_did(hk).coef - 5.0) > 2.0


# ----------------------------------------------------------------------------
# Pre-trend test
# ----------------------------------------------------------------------------
def _pre_sample(df, case, origin):
    sub = df[(df.study_case == case) & (df.post == 0)].copy()
    sub["t"] = sub.year - origin
    return sub


def test_pretrend_matches_manual_regression_and_inference(sim_trending):
    df, _ = sim_trending
    sub = _pre_sample(df, "Hong Kong", 1998)
    r = did.pretrend_test(sub)
    D = sc.factor_dummies(sub.unit_id.to_numpy(), "u")
    X = pd.concat([pd.DataFrame({"t": sub.t.to_numpy(float), "int": sub.t.to_numpy(float) * sub.treated.to_numpy(float)}), D], axis=1)
    fit = sc.regress(sub[Y].to_numpy(), X, cluster=sub.unit_id.to_numpy())
    assert r.coef == pytest.approx(fit.coef("int"), rel=1e-10)
    assert r.se == pytest.approx(fit.std_err("int"), rel=1e-10)
    g = fit.n_clusters
    assert r.p_t == pytest.approx(2 * stats.t.sf(abs(r.t), g - 1))
    assert r.p_z == pytest.approx(2 * stats.norm.sf(abs(r.z)))
    assert r.p_z < r.p_t
    assert r.ci_t[1] - r.ci_t[0] > r.ci_z[1] - r.ci_z[0]
    assert r.n == len(sub) and r.n_clusters == g
    assert r.coef > 0  # the simulated treated unit trends upward relative to the pool


def test_pretrend_interaction_is_invariant_to_the_time_origin(sim_trending):
    df, _ = sim_trending
    a = did.pretrend_test(_pre_sample(df, "Hong Kong", 1998))
    b = did.pretrend_test(_pre_sample(df, "Hong Kong", 2001))
    assert a.coef == pytest.approx(b.coef, rel=1e-9)
    assert a.se == pytest.approx(b.se, rel=1e-8)


def test_pretrend_with_one_treated_cluster_has_a_structural_degeneracy(sim_trending):
    """The treated cluster has zero score, so SE(interaction) equals SE of the control slope."""
    df, _ = sim_trending
    sub = _pre_sample(df, "Hong Kong", 1998)
    r = did.pretrend_test(sub)
    assert r.se == pytest.approx(r.fit.std_err("t"), rel=1e-9)


def test_pretrend_detects_parallel_and_non_parallel_trends(sim_parallel, sim_trending):
    par = did.pretrend_test(_pre_sample(sim_parallel[0], "Hong Kong", 1998))
    trend = did.pretrend_test(_pre_sample(sim_trending[0], "Hong Kong", 1998))
    assert abs(par.coef) < 0.2
    assert abs(trend.coef) > 5 * abs(par.coef)


# ----------------------------------------------------------------------------
# Bootstrap weights
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["webb", "rademacher", "mammen"])
def test_weight_distributions_have_unit_variance_and_zero_mean(kind):
    values, probs, _ = did._SUPPORTS[kind]
    assert (values * probs).sum() == pytest.approx(0.0, abs=1e-12)
    assert (values**2 * probs).sum() == pytest.approx(1.0, abs=1e-12)
    draws = did.draw_weights(kind, 3, 200_000, np.random.default_rng(1))
    assert draws.shape == (200_000, 3)
    assert abs(draws.mean()) < 0.01 and abs(draws.var() - 1.0) < 0.02


def test_webb_support_and_enumeration():
    values, _, _ = did._SUPPORTS["webb"]
    assert values == pytest.approx([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    W, mult = did.enumerate_weights("webb", 4)
    assert mult == 2 and W.shape == (6**4 // 2, 4)
    assert (W[:, 0] > 0).all()
    # every vector or its negation appears exactly once
    full = np.array(list(itertools.product(values, repeat=4)))
    keys_half = {tuple(np.round(r, 9)) for r in W} | {tuple(np.round(-r, 9)) for r in W}
    assert keys_half == {tuple(np.round(r, 9)) for r in full}
    assert len(keys_half) == 6**4
    with pytest.raises(ValueError):
        did.enumerate_weights("mammen", 3)
    with pytest.raises(ValueError):
        did.enumerate_weights("webb", 20, max_rows=1000)


# ----------------------------------------------------------------------------
# Wild cluster bootstrap-t
# ----------------------------------------------------------------------------
def small_problem(seed=0, g=4, per=9, effect=0.8):
    """SIMULATED regression y ~ d + z + const with g clusters; d varies at the observation level."""
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(g), per)
    n = g * per
    d = (rng.random(n) < 0.4).astype(float)
    z = rng.normal(size=n)
    u = rng.normal(0, 1, g)[cluster]
    y = 0.5 + effect * d + 0.3 * z + u + rng.normal(0, 1, n)
    X = pd.DataFrame({"d": d, "z": z, "_cons": 1.0})
    return y, X, cluster


def brute_force_p(y, X, cluster, term, null, weights, ssc_k=None, tie_tol=1e-9, full_support=True):
    """Wild cluster bootstrap-t p-value by explicit refitting (no shortcuts), WCR, symmetric."""
    Xv = X.to_numpy(float)
    j = list(X.columns).index(term)
    n, k = Xv.shape
    codes = pd.factorize(cluster, sort=True)[0]
    g = codes.max() + 1
    kk = k if ssc_k is None else ssc_k
    c = g / (g - 1) * (n - 1) / (n - kk)

    def fit(yv):
        b = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        e = yv - Xv @ b
        A = np.linalg.inv(Xv.T @ Xv)
        S = np.zeros((k, k))
        for h in range(g):
            m = codes == h
            s = Xv[m].T @ e[m]
            S += np.outer(s, s)
        V = c * A @ S @ A
        return b[j], np.sqrt(V[j, j])

    Xr = np.delete(Xv, j, axis=1)
    yr = y - null * Xv[:, j]
    br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    fitted = Xr @ br + null * Xv[:, j]
    u = y - fitted
    b_j, se_j = fit(y)
    t_obs = abs((b_j - null) / se_j)
    values, _, _ = did._SUPPORTS[weights]
    count = total = 0
    for w in itertools.product(values, repeat=g):
        ys = fitted + np.array(w)[codes] * u
        bs, ses = fit(ys)
        total += 1
        if abs((bs - null) / ses) > t_obs * (1 + tie_tol):
            count += 1
    return count / total, (b_j, se_j)


@pytest.mark.parametrize("seed,weights,g", [(0, "webb", 4), (1, "webb", 4), (2, "rademacher", 6), (3, "rademacher", 5)])
def test_exhaustive_p_equals_brute_force_refitting(seed, weights, g):
    y, X, cl = small_problem(seed, g=g)
    r = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, weights=weights, ci=False)
    p_ref, (b, se) = brute_force_p(y, X, cl, "d", 0.0, weights)
    assert r.p_value == pytest.approx(p_ref, abs=1e-12)
    assert r.coef == pytest.approx(b, rel=1e-10) and r.se == pytest.approx(se, rel=1e-10)
    assert r.reps == len(did._SUPPORTS[weights][0]) ** g and r.exhaustive and r.seed is None


def test_exhaustive_p_with_nonzero_null_and_covariate_null():
    y, X, cl = small_problem(4, g=4)
    for null in (0.8, -0.5):
        r = did.wild_cluster_bootstrap(y, X, cl, "d", null=null, exhaustive=True, ci=False)
        p_ref, _ = brute_force_p(y, X, cl, "d", null, "webb")
        assert r.p_value == pytest.approx(p_ref, abs=1e-12)
        assert r.t_stat == pytest.approx((r.coef - null) / r.se)
    r = did.wild_cluster_bootstrap(y, X, cl, "z", exhaustive=True, ci=False)
    assert r.p_value == pytest.approx(brute_force_p(y, X, cl, "z", 0.0, "webb")[0], abs=1e-12)


def test_ssc_k_rescales_t_but_not_the_p_value():
    y, X, cl = small_problem(5, g=4)
    a = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, ci=False)
    b = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, ci=False, ssc_k=2)
    n = len(y)
    assert b.p_value == pytest.approx(a.p_value, abs=1e-12)
    assert b.se**2 / a.se**2 == pytest.approx((n - 3) / (n - 2), rel=1e-12)
    p_ref, _ = brute_force_p(y, X, cl, "d", 0.0, "webb", ssc_k=2)
    assert b.p_value == pytest.approx(p_ref, abs=1e-12)


def test_confidence_set_endpoints_are_where_the_p_value_crosses_alpha():
    y, X, cl = small_problem(6, g=4, effect=1.2)
    r = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, alpha=0.05)
    lo, hi = r.ci
    assert lo < r.coef < hi

    def p_at(null):
        return did.wild_cluster_bootstrap(y, X, cl, "d", null=null, exhaustive=True, ci=False).p_value

    eps = 1e-6 * (hi - lo)
    assert p_at(lo + eps) > 0.05 and p_at(hi - eps) > 0.05
    assert p_at(lo - eps) <= 0.05 and p_at(hi + eps) <= 0.05
    # the point estimate itself is never rejected
    assert p_at(r.coef) > 0.9


def test_confidence_set_matches_brute_force_inversion():
    y, X, cl = small_problem(7, g=4, effect=1.0)
    r = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True)
    lo, hi = r.ci
    eps = 1e-5 * (hi - lo)
    assert brute_force_p(y, X, cl, "d", lo + eps, "webb")[0] > 0.05
    assert brute_force_p(y, X, cl, "d", lo - eps, "webb")[0] <= 0.05
    assert brute_force_p(y, X, cl, "d", hi - eps, "webb")[0] > 0.05
    assert brute_force_p(y, X, cl, "d", hi + eps, "webb")[0] <= 0.05


def test_monte_carlo_is_deterministic_and_close_to_exhaustive():
    y, X, cl = small_problem(8, g=5, per=8)
    ex = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, ci=False)
    a = did.wild_cluster_bootstrap(y, X, cl, "d", reps=30_000, seed=5, ci=False)
    b = did.wild_cluster_bootstrap(y, X, cl, "d", reps=30_000, seed=5, ci=False)
    c = did.wild_cluster_bootstrap(y, X, cl, "d", reps=30_000, seed=6, ci=False)
    assert a.p_value == b.p_value and a.seed == 5 and not a.exhaustive and a.reps == 30_000
    assert a.p_value != c.p_value
    se_mc = np.sqrt(ex.p_value * (1 - ex.p_value) / 30_000)
    assert abs(a.p_value - ex.p_value) < 4 * se_mc + 1e-12
    assert abs(c.p_value - ex.p_value) < 4 * se_mc + 1e-12


def test_exhaustive_none_switches_on_the_number_of_vectors():
    y, X, cl = small_problem(9, g=4)
    assert did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=None, reps=9999, ci=False).exhaustive  # 1296 <= 9999
    assert not did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=None, reps=500, ci=False).exhaustive


def test_keep_draws_and_symmetry_of_the_enumeration():
    y, X, cl = small_problem(10, g=4)
    r = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, keep_draws=True, ci=False)
    assert r.t_boot.shape == (6**4 // 2,)
    t_obs = abs(r.t_stat)
    # the p-value from the retained draws, with the tie rule
    assert np.mean(np.abs(r.t_boot) > t_obs * (1 + 1e-9)) == pytest.approx(r.p_value)


def test_term_by_index_and_by_name_agree_and_dataframe_names_are_used():
    y, X, cl = small_problem(11, g=4)
    a = did.wild_cluster_bootstrap(y, X, cl, "z", exhaustive=True, ci=False)
    b = did.wild_cluster_bootstrap(y, X.to_numpy(), cl, 1, exhaustive=True, ci=False)
    assert a.p_value == b.p_value and a.term == "z"


def test_band_brackets_the_exact_confidence_set():
    y, X, cl = small_problem(12, g=5, per=8, effect=1.0)
    r = did.wild_cluster_bootstrap(y, X, cl, "d", exhaustive=True, band=(9999, 4.0))
    (lo_n, lo_w), (hi_w, hi_n) = r.ci_band
    assert lo_n <= r.ci[0] <= lo_w
    assert hi_w <= r.ci[1] <= hi_n


def test_regress_wrapper_t_equals_cluster_t():
    y, X, cl = small_problem(13, g=5, per=8)
    fit = sc.regress(y, X[["d", "z"]], cluster=cl)
    r = did.boottest_regress(y, X[["d", "z"]], cl, "d", exhaustive=True, ci=False)
    assert r.t_stat == pytest.approx(fit.tstat("d"), rel=1e-10)
    assert r.se == pytest.approx(fit.std_err("d"), rel=1e-10)


def test_xtreg_wrapper_t_and_p_with_unit_effects():
    """boottest after xtreg, fe: K excludes the constant; p equals the dummy-variable bootstrap p."""
    rng = np.random.default_rng(14)
    g, per = 4, 9
    unit = np.repeat(np.arange(g), per)
    n = g * per
    d = ((unit == 0) & (np.tile(np.arange(per), g) >= 5)).astype(float)
    z = rng.normal(size=n)
    y = rng.normal(0, 2, g)[unit] + 1.0 * d + 0.4 * z + rng.normal(0, 0.7, n)
    X = pd.DataFrame({"d": d, "z": z})
    r = did.boottest_xtreg_fe(y, X, unit, "d", exhaustive=True)
    xt = sc.xtreg_fe(y, X, unit, cluster=unit)
    kw = xt.k
    assert r.t_stat == pytest.approx(xt.tstat("d") * np.sqrt((n - kw + 1.0) / (n - kw)), rel=1e-10)
    Xd = pd.concat([X, sc.factor_dummies(unit, "u")], axis=1)
    rd = did.boottest_regress(y, Xd, unit, "d", exhaustive=True)
    assert r.p_value == pytest.approx(rd.p_value, abs=1e-12)
    assert r.ci == pytest.approx(rd.ci, rel=1e-7)


def test_structural_zero_score_of_a_single_treated_cluster_does_not_break_the_bootstrap():
    """With one treated cluster the original SE is positive; the exhaustive p is finite and in [0, 1]."""
    rng = np.random.default_rng(15)
    g, per = 4, 10
    unit = np.repeat(np.arange(g), per)
    year = np.tile(np.arange(per), g)
    d = ((unit == 0) & (year >= 6)).astype(float)
    y = rng.normal(0, 2, g)[unit] + rng.normal(0, 0.5, per)[year] + 3.0 * d + rng.normal(0, 0.3, g * per)
    X = pd.concat([pd.DataFrame({"d": d}), sc.factor_dummies(year, "y")], axis=1)
    r = did.boottest_xtreg_fe(y, X, unit, "d", exhaustive=True)
    assert 0.0 <= r.p_value <= 1.0 and np.isfinite(r.t_stat)
    assert r.ci[0] < r.coef < r.ci[1]


# ----------------------------------------------------------------------------
# Event study
# ----------------------------------------------------------------------------
def test_event_study_recovers_the_effect_path(sim_parallel):
    df, _ = sim_parallel
    hk = df[df.study_case == "Hong Kong"]
    es = did.event_study(hk, lo=-6, hi=10, base=-1)
    tab = es.table.set_index("rel")
    assert tab.loc[-1, ["coef", "se"]].tolist() == [0.0, 0.0]
    assert es.base == -1 and (es.lo, es.hi) == (-6, 10)
    pre = tab.loc[[r for r in tab.index if r < -1], "coef"]
    post = tab.loc[[r for r in tab.index if r >= 0], "coef"]
    assert pre.abs().max() < 0.6  # no effect before the opening (parallel trends)
    assert abs(post.mean() - 5.0) < 0.5
    assert tab.index.min() == -6 and tab.index.max() == 10


def test_event_study_coefficients_equal_those_of_the_dummy_variable_regression(sim_parallel):
    df, _ = sim_parallel
    hk = df[df.study_case == "Hong Kong"].reset_index(drop=True)
    es = did.event_study(hk, lo=-6, hi=10, base=-1)
    ev = np.where(hk.treated == 1, hk.rel_year, -1)
    ev = np.clip(ev, -6, 10)
    levels = [int(v) for v in np.unique(ev) if v != -1]
    cols = pd.DataFrame({f"ev.{lv}": (ev == lv).astype(float) for lv in levels})
    X = pd.concat([cols, sc.factor_dummies(hk.year.to_numpy(), "year"), sc.factor_dummies(hk.unit_id.to_numpy(), "unit")], axis=1)
    fit = sc.regress(hk[Y].to_numpy(), X, cluster=hk.unit_id.to_numpy())
    tab = es.table.set_index("rel")
    for lv in levels:
        assert tab.loc[lv, "coef"] == pytest.approx(fit.coef(f"ev.{lv}"), rel=1e-8, abs=1e-9)


def test_event_study_binning_pools_the_end_points(sim_parallel):
    df, _ = sim_parallel
    hk = df[df.study_case == "Hong Kong"]
    unbinned = did.event_study(hk)
    binned = did.event_study(hk, lo=-3, hi=5, base=-1)
    assert unbinned.table.rel.min() < -3 and binned.table.rel.min() == -3
    assert unbinned.table.rel.max() > 5 and binned.table.rel.max() == 5
    # the binned end point is a weighted pool of the unbinned coefficients, so it lies between their extremes
    u = unbinned.table.set_index("rel").coef
    b = binned.table.set_index("rel").coef
    assert u[u.index >= 5].min() - 1.0 <= b.loc[5] <= u[u.index >= 5].max() + 1.0


def test_event_study_confidence_intervals_use_t_with_g_minus_one(sim_parallel):
    df, _ = sim_parallel
    sh = df[(df.study_case == "Shanghai") & (df.year >= 2008)]
    es = did.event_study(sh)
    row = es.table[es.table.rel == 0].iloc[0]
    crit = stats.t.isf(0.025, 3 - 1)
    assert row.ci_hi - row.coef == pytest.approx(crit * row.se, rel=1e-10)
