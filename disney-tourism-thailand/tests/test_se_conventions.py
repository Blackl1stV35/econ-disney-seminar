"""Closed-form relation between the two clustered standard errors in the reference values.

Two programs of the original analysis report a standard error for the same
coefficient (``treated_post``) on the same sample:

``friend_did``
    ``regress y treated_post i.unit_id i.year, vce(cluster unit_id)`` followed by
    the rescaling ``sqrt(N (G-1) / (G (N-1)))``.  The variance is
    ``S0 * N / (N - K)`` and inference uses ``t(N - K)``.
``xtreg, fe vce(cluster unit_id)``
    The unit effects are absorbed.  The variance is
    ``S0 * G/(G-1) * (N-1) / (N - K + G - 1)`` and inference uses ``t(G - 1)``.

``S0`` is the unscaled cluster sandwich, which is identical in both programs
(Frisch-Waugh).  Hence

    SE_xtreg = SE_report * sqrt( G/(G-1) * (N-1)/(N-K+G-1) * (N-K)/N ),

where ``K`` is the number of estimated parameters of the dummy-variable
regression (``K = N - df`` from the ``friend_did`` output).

The relation is an algebraic identity.  It holds exactly on any data set
(verified below to about 1e-12 on simulated data) and, on the reference values,
up to the print precision of the two standard errors (six and seven decimals).
Only ``N``, ``G`` and ``K`` enter; no data are needed.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from dtt import stata_compat as sc
from dtt.replication import print_halfwidth

CASES = [
    # tag, label, SE printed by friend_did, SE printed by xtreg (as given for the check)
    ("1a", "Hong Kong, all 5 donors", 0.171312, 0.1824719),
    ("1b", "Hong Kong, core 3 donors", 0.214694, 0.2405224),
    ("1c", "Shanghai", 0.276902, 0.3233765),
]


def _value(reference, key):
    return reference["values"][key]["value"]


def _halfwidth(reference, key):
    return print_halfwidth(reference["values"][key]["dp"])


def _reference_quantities(reference, tag):
    n = int(_value(reference, f"log1/friend_did/{tag}/n_obs"))
    g = int(_value(reference, f"log1/friend_did/{tag}/n_clusters"))
    df = int(_value(reference, f"log1/friend_did/{tag}/df"))
    key_rep = f"log1/friend_did/{tag}/se"
    key_xt = f"log2/xtreg_twfe/{tag}/coef/treated_post/se"
    return n, g, n - df, _value(reference, key_rep), _value(reference, key_xt), _halfwidth(reference, key_rep), _halfwidth(reference, key_xt)


def _closed_form_ratio(n, g, k):
    return np.sqrt(g / (g - 1.0) * (n - 1.0) / (n - k + g - 1.0) * (n - k) / n)


@pytest.mark.parametrize("tag,label,se_rep_stated,se_xt_stated", CASES)
def test_printed_values_are_the_stated_ones(reference, tag, label, se_rep_stated, se_xt_stated):
    n, g, k, se_rep, se_xt, _, _ = _reference_quantities(reference, tag)
    assert se_rep == se_rep_stated
    assert se_xt == se_xt_stated


@pytest.mark.parametrize("tag,label,se_rep_stated,se_xt_stated", CASES)
def test_xtreg_se_follows_from_report_se_up_to_print_precision(reference, tag, label, se_rep_stated, se_xt_stated):
    """SE_xtreg = SE_report * c(N, G, K), checked with a bound derived from the printed digits."""
    n, g, k, se_rep, se_xt, hw_rep, hw_xt = _reference_quantities(reference, tag)
    ratio = _closed_form_ratio(n, g, k)
    predicted = se_rep * ratio
    # The prediction carries the rounding error of SE_report (scaled by the ratio);
    # the printed SE_xtreg carries its own.
    bound = ratio * hw_rep + hw_xt
    assert abs(predicted - se_xt) <= bound, (label, predicted, se_xt, bound)
    # The same relation through the library function.
    lib = se_rep * sc.se_ratio("xtreg_fe", "report", n, g, k)
    assert lib == pytest.approx(predicted, rel=1e-13)


@pytest.mark.parametrize("tag,label,se_rep_stated,se_xt_stated", CASES)
def test_absorbed_parameters_are_consistent_with_printed_coefficient_rows(reference, tag, label, se_rep_stated, se_xt_stated):
    """K - (G - 1) equals the number of coefficient rows xtreg prints (slopes plus _cons)."""
    n, g, k, *_ = _reference_quantities(reference, tag)
    prefix = f"log2/xtreg_twfe/{tag}/coef/"
    rows = [key for key in reference["values"] if key.startswith(prefix) and key.endswith("/coef")]
    assert len(rows) == k - (g - 1), (label, len(rows), k, g)


@pytest.mark.parametrize("tag,label,se_rep_stated,se_xt_stated", CASES)
def test_alternative_conventions_are_rejected_by_the_reference_values(reference, tag, label, se_rep_stated, se_xt_stated):
    """The bound is tight enough to discriminate: wrong K or a missing factor fails by orders of magnitude."""
    n, g, k, se_rep, se_xt, hw_rep, hw_xt = _reference_quantities(reference, tag)
    ratio = _closed_form_ratio(n, g, k)
    bound = ratio * hw_rep + hw_xt
    kw = k - (g - 1)
    wrong = {
        "no (N-K)/N factor": se_rep * np.sqrt(g / (g - 1.0) * (n - 1.0) / (n - k + g - 1.0)),
        "K_w counted with the absorbed effects": se_rep * np.sqrt(g / (g - 1.0) * (n - 1.0) / (n - k) * (n - k) / n),
        "constant not counted in K_w": se_rep * np.sqrt(g / (g - 1.0) * (n - 1.0) / (n - kw + 1.0) * (n - k) / n),
        "no G/(G-1)": se_rep * np.sqrt((n - 1.0) / (n - k + g - 1.0) * (n - k) / n),
    }
    for name, value in wrong.items():
        assert abs(value - se_xt) > 20 * bound, (label, name, value, se_xt)


@pytest.mark.parametrize("tag,label,se_rep_stated,se_xt_stated", CASES)
def test_boottest_t_after_xtreg_uses_K_without_the_constant(reference, tag, label, se_rep_stated, se_xt_stated):
    """boottest after xtreg, fe reports t = b / (SE_xtreg * sqrt((N-K_w)/(N-K_w+1)))."""
    n, g, k, se_rep, se_xt, *_ = _reference_quantities(reference, tag)
    kw = k - (g - 1)
    b = _value(reference, f"log2/xtreg_twfe/{tag}/coef/treated_post/coef")
    t_boot = _value(reference, f"log2/xtreg_twfe/{tag}/boottest/t")
    hw_t = _halfwidth(reference, f"log2/xtreg_twfe/{tag}/boottest/t")
    t_pred = b / se_xt * np.sqrt((n - kw + 1.0) / (n - kw))
    # SE_xtreg is printed with 7 decimals: propagate its rounding into t.
    prop = t_pred * _halfwidth(reference, f"log2/xtreg_twfe/{tag}/coef/treated_post/se") / se_xt
    assert abs(t_pred - t_boot) <= hw_t + prop + 1e-12, (label, t_pred, t_boot)
    # Without the adjustment the statistic would equal the xtreg t, which differs.
    assert abs(b / se_xt - t_boot) > 5 * (hw_t + prop)


@pytest.mark.parametrize("key", ["hk", "sh"])
def test_boottest_t_after_regress_equals_regress_t(reference, key):
    """After ``regress, vce(cluster)`` the boottest t equals the regress cluster t."""
    b = _value(reference, f"log2/pretrend/{key}/coef/c.t#c.treated/coef")
    se = _value(reference, f"log2/pretrend/{key}/coef/c.t#c.treated/se")
    t_boot = _value(reference, f"log2/pretrend/{key}/boottest/t")
    hw_t = _halfwidth(reference, f"log2/pretrend/{key}/boottest/t")
    prop = abs(b / se) * _halfwidth(reference, f"log2/pretrend/{key}/coef/c.t#c.treated/se") / se
    assert abs(b / se - t_boot) <= hw_t + prop + 1e-12


def test_report_formula_equals_n_over_n_minus_k_scaling():
    """SE_report = sqrt(S0 N/(N-K)) = SE_regress_cluster * sqrt(N (G-1) / (G (N-1)))."""
    for n, g, k in [(129, 6, 28), (86, 4, 26), (48, 3, 22)]:
        c_regress = sc.small_sample_factor(n, k, g, "regress")
        c_report = sc.small_sample_factor(n, k, g, "report")
        resc = n * (g - 1.0) / (g * (n - 1.0))
        assert c_regress * resc == pytest.approx(c_report, rel=1e-14)
        assert float(sc.report_formula_se(1.0, n, g)) == pytest.approx(np.sqrt(resc), rel=1e-14)


# ----------------------------------------------------------------------------
# Exactness on data (simulated; independent implementation in statsmodels)
# ----------------------------------------------------------------------------
def _random_panel(seed, g, t_min, t_max, heavy=True):
    """SIMULATED unbalanced panel with one treated unit; returns y, treat, unit, year."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = 0
    for u in range(g):
        length = int(rng.integers(t_min, t_max + 1))
        start = int(rng.integers(0, 3))
        for s in range(length):
            rows.append((u, start + s))
    unit = np.array([r[0] for r in rows])
    year = np.array([r[1] for r in rows])
    mu = rng.normal(0, 1.5, g)[unit]
    f = rng.normal(0, 0.5, year.max() + 1)[year]
    treat = ((unit == 0) & (year >= 8)).astype(float)
    scale = (1.0 + rng.uniform(0, 2.0, g))[unit] if heavy else 1.0
    y = mu + f + 0.7 * treat + scale * rng.normal(0, 0.5, len(unit))
    return y, treat, unit, year


def _dummy_design(treat, unit, year):
    return pd.concat(
        [
            pd.DataFrame({"treat": treat}),
            sc.factor_dummies(unit, "u"),
            sc.factor_dummies(year, "y"),
        ],
        axis=1,
    )


@pytest.mark.parametrize("seed,g,t_min,t_max", [(1, 4, 12, 16), (2, 6, 10, 22), (3, 9, 8, 20), (4, 3, 15, 19), (5, 12, 6, 18)])
def test_identity_holds_to_machine_precision_on_simulated_panels(seed, g, t_min, t_max):
    y, treat, unit, year = _random_panel(seed, g, t_min, t_max)
    X = _dummy_design(treat, unit, year)
    fit = sc.regress(y, X, cluster=unit)
    n, k = fit.n, fit.k
    se_rep = float(sc.report_formula_se(fit.std_err("treat"), n, g))
    Xt = pd.concat([pd.DataFrame({"treat": treat}), sc.factor_dummies(year, "y")], axis=1)
    xt = sc.xtreg_fe(y, Xt, unit, cluster=unit)
    se_xt = xt.std_err("treat")
    assert xt.coef("treat") == pytest.approx(fit.coef("treat"), rel=1e-10)
    predicted = se_rep * _closed_form_ratio(n, g, k)
    assert predicted == pytest.approx(se_xt, rel=1e-11)
    # The unscaled sandwich is shared.
    j = fit.index("treat")
    s0_reg = fit.S0[j, j]
    s0_xt = xt.S0[xt.index("treat"), xt.index("treat")]
    assert s0_reg == pytest.approx(s0_xt, rel=1e-9)
    # Report variance = S0 * N/(N-K).
    assert se_rep**2 == pytest.approx(s0_reg * n / (n - k), rel=1e-11)
    # xtreg's K_w and the number of printed coefficients.
    assert xt.k == k - (g - 1)


@pytest.mark.parametrize("seed,g", [(21, 5), (22, 8)])
def test_regress_cluster_se_matches_statsmodels(seed, g):
    """Independent implementation: statsmodels OLS with cluster-robust covariance (CR1 correction)."""
    y, treat, unit, year = _random_panel(seed, g, 10, 20)
    X = _dummy_design(treat, unit, year)
    fit = sc.regress(y, X, cluster=unit)
    ref = sm.OLS(y, sm.add_constant(X.to_numpy(dtype=float))).fit(cov_type="cluster", cov_kwds={"groups": unit, "use_correction": True, "df_correction": True})
    # statsmodels puts the constant first.
    b_ref = np.r_[ref.params[1:], ref.params[0]]
    se_ref = np.r_[ref.bse[1:], ref.bse[0]]
    assert fit.b == pytest.approx(b_ref, rel=1e-8, abs=1e-10)
    assert fit.se == pytest.approx(se_ref, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("seed,g", [(31, 5), (32, 7)])
def test_xtreg_cluster_se_matches_dummy_regression_rescaled(seed, g):
    """xtreg-fe SE = dummy-regression cluster SE * sqrt((N-K)/(N-K+G-1)), computed with statsmodels."""
    y, treat, unit, year = _random_panel(seed, g, 10, 20)
    X = _dummy_design(treat, unit, year)
    ref = sm.OLS(y, sm.add_constant(X.to_numpy(dtype=float))).fit(cov_type="cluster", cov_kwds={"groups": unit, "use_correction": True, "df_correction": True})
    n, k = len(y), X.shape[1] + 1
    se_ref_treat = ref.bse[1]
    Xt = pd.concat([pd.DataFrame({"treat": treat}), sc.factor_dummies(year, "y")], axis=1)
    xt = sc.xtreg_fe(y, Xt, unit, cluster=unit)
    expected = se_ref_treat * np.sqrt((n - k) / (n - k + g - 1.0))
    assert xt.std_err("treat") == pytest.approx(expected, rel=1e-10)
    assert xt.df_r == g - 1


def test_inference_degrees_of_freedom_differ_between_the_two_programs():
    """friend_did uses t(N-K), xtreg uses t(G-1): the same t-statistic gives very different p-values."""
    y, treat, unit, year = _random_panel(7, 6, 15, 22)
    X = _dummy_design(treat, unit, year)
    fit = sc.regress(y, X, cluster=unit)
    assert fit.df_r == 6 - 1
    n, k = fit.n, fit.k
    assert (n - k) > 50
    t = 3.0
    assert sc.t_pvalue(t, n - k) < 0.5 * sc.t_pvalue(t, 5)
