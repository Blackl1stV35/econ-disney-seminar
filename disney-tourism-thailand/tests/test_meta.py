"""Tests for dtt.meta: random-effects, Bayesian, meta-regression and scaling-law routines.

All data are SIMULATED.
"""
from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, linalg, optimize, stats

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dtt import meta  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def simulate_cases(seed, k, mu=0.4, tau=0.3, se_low=0.1, se_high=0.5):
    """Case effects and standard errors from the random-effects model."""
    rng = np.random.default_rng(seed)
    se = rng.uniform(se_low, se_high, k)
    y = mu + rng.normal(0.0, tau, k) + rng.normal(0.0, se)
    return y, se


def _maximise(nll, scale):
    """Minimise a function of tau2 on a grid and refine the best cell."""
    grid = np.concatenate(([0.0], np.geomspace(1e-8 * scale, 50.0 * scale, 250)))
    values = np.array([nll(t) for t in grid])
    j = int(np.argmin(values))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
    if hi > lo:
        res = optimize.minimize_scalar(
            nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13}
        )
        if res.fun < values[j]:
            return float(res.x)
    return float(grid[j])


def contrast_reml_tau2(y, v, design):
    """REML through the likelihood of error contrasts, independent of dtt.meta."""
    basis = linalg.null_space(design.T)
    z = basis.T @ y

    def nll(t):
        cov = basis.T @ np.diag(v + t) @ basis
        return -stats.multivariate_normal.logpdf(z, mean=np.zeros(z.size), cov=cov)

    return _maximise(nll, max(v.mean(), y.var()))


def contrast_reml_tau2_ridge(y, v, z_std, precision):
    """REML with independent N(0, 1 / precision) priors on the standardised slopes."""
    k = y.size
    basis = linalg.null_space(np.ones((1, k)))
    z = basis.T @ y

    def nll(t):
        cov = basis.T @ (np.diag(v + t) + z_std @ z_std.T / precision) @ basis
        return -stats.multivariate_normal.logpdf(z, mean=np.zeros(z.size), cov=cov)

    return _maximise(nll, max(v.mean(), y.var()))


def standardise(X):
    """Columns centred and divided by their population standard deviation."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


# ----------------------------------------------------------------------------
# DerSimonian-Laird
# ----------------------------------------------------------------------------
def test_dl_homogeneous_effects_equal_se_give_zero_tau2_and_the_mean():
    se = np.full(5, 0.5)
    for y in (np.full(5, 1.0), np.array([0.98, 1.0, 1.02, 0.99, 1.01])):
        out = meta.dersimonian_laird(y, se)
        assert out["tau2"] == 0.0
        assert out["mu"] == pytest.approx(np.mean(y), abs=1e-12)
        assert out["se_mu"] == pytest.approx(0.5 / np.sqrt(5), rel=1e-12)
        assert out["I2"] == 0.0
        assert out["df"] == 4


def test_dl_matches_a_hand_computation():
    y = [0.2, 0.8, 0.5, 1.1]
    se = [0.1, 0.2, 0.15, 0.25]
    w = [1.0 / s**2 for s in se]
    sw = sum(w)
    mu_fixed = sum(wi * yi for wi, yi in zip(w, y)) / sw
    q = sum(wi * (yi - mu_fixed) ** 2 for wi, yi in zip(w, y))
    c = sw - sum(wi**2 for wi in w) / sw
    tau2 = max(0.0, (q - 3) / c)
    ws = [1.0 / (s**2 + tau2) for s in se]
    mu = sum(wi * yi for wi, yi in zip(ws, y)) / sum(ws)
    out = meta.dersimonian_laird(y, se)
    assert tau2 > 0.0
    assert out["Q"] == pytest.approx(q, rel=1e-12)
    assert out["tau2"] == pytest.approx(tau2, rel=1e-12)
    assert out["mu"] == pytest.approx(mu, rel=1e-12)
    assert out["se_mu"] == pytest.approx(sum(ws) ** -0.5, rel=1e-12)
    assert out["I2"] == pytest.approx((q - 3) / q, rel=1e-12)
    assert out["df"] == 3


def test_dl_equal_se_closed_form():
    rng = np.random.default_rng(3)
    y = rng.normal(0.5, 0.7, 9)
    s = 0.4
    out = meta.dersimonian_laird(y, np.full(9, s))
    assert out["tau2"] == pytest.approx(max(0.0, np.var(y, ddof=1) - s**2), rel=1e-12)
    assert out["Q"] == pytest.approx(np.sum((y - y.mean()) ** 2) / s**2, rel=1e-12)
    assert out["mu"] == pytest.approx(y.mean(), rel=1e-12)


def test_dl_agrees_with_statsmodels_method_of_moments():
    combine = pytest.importorskip("statsmodels.stats.meta_analysis").combine_effects
    y, se = simulate_cases(seed=101, k=9, tau=0.35)
    ref = combine(y, se**2, method_re="chi2")
    out = meta.dersimonian_laird(y, se)
    assert out["tau2"] > 0.0
    assert out["tau2"] == pytest.approx(ref.tau2, rel=1e-9)
    assert out["mu"] == pytest.approx(ref.mean_effect_re, rel=1e-9)
    assert out["se_mu"] == pytest.approx(ref.sd_eff_w_re, rel=1e-9)
    assert out["Q"] == pytest.approx(ref.q, rel=1e-9)
    assert out["I2"] == pytest.approx(ref.i2, rel=1e-9)


def test_single_case_returns_the_case():
    for fn in (meta.dersimonian_laird, meta.reml):
        out = fn([0.7], [0.2])
        assert out["mu"] == pytest.approx(0.7)
        assert out["se_mu"] == pytest.approx(0.2)
        assert out["tau2"] == 0.0
        assert out["Q"] == 0.0 and out["I2"] == 0.0 and out["df"] == 0


@pytest.mark.parametrize(
    "tau_hat, se",
    [
        ([0.1, 0.2], [0.1]),
        ([0.1, 0.2], [0.1, 0.0]),
        ([0.1, 0.2], [0.1, -0.2]),
        ([0.1, np.nan], [0.1, 0.2]),
        ([0.1, 0.2], [0.1, np.inf]),
        ([[0.1, 0.2]], [[0.1, 0.2]]),
        ([], []),
    ],
)
def test_invalid_inputs_are_rejected(tau_hat, se):
    for fn in (meta.dersimonian_laird, meta.reml):
        with pytest.raises(ValueError):
            fn(tau_hat, se)


# ----------------------------------------------------------------------------
# REML
# ----------------------------------------------------------------------------
def test_reml_equal_se_closed_form():
    rng = np.random.default_rng(8)
    y = rng.normal(0.2, 0.6, 11)
    s = 0.3
    out = meta.reml(y, np.full(11, s))
    tau2 = max(0.0, np.var(y, ddof=1) - s**2)
    assert tau2 > 0.0
    assert out["tau2"] == pytest.approx(tau2, rel=1e-6)
    assert out["mu"] == pytest.approx(y.mean(), rel=1e-9)
    assert out["se_mu"] == pytest.approx(np.sqrt((s**2 + tau2) / 11), rel=1e-6)


def test_reml_homogeneous_effects_equal_se_give_zero_tau2_and_the_mean():
    y = np.array([0.98, 1.0, 1.02, 0.99, 1.01])
    out = meta.reml(y, np.full(5, 0.5))
    assert out["tau2"] == 0.0
    assert out["mu"] == pytest.approx(y.mean(), abs=1e-12)
    assert out["se_mu"] == pytest.approx(0.5 / np.sqrt(5), rel=1e-12)


@pytest.mark.parametrize("seed, tau", [(0, 0.0), (1, 0.1), (2, 0.3), (3, 0.6), (4, 0.3), (5, 0.0)])
def test_reml_maximises_the_restricted_likelihood_of_error_contrasts(seed, tau):
    k = 4 + 3 * seed
    y, se = simulate_cases(seed, k, tau=tau)
    ref = contrast_reml_tau2(y, se**2, np.ones((k, 1)))
    out = meta.reml(y, se)
    assert out["tau2"] == pytest.approx(ref, rel=1e-4, abs=1e-6)
    w = 1.0 / (se**2 + out["tau2"])
    assert out["mu"] == pytest.approx(np.sum(w * y) / np.sum(w), rel=1e-12)


def test_dl_and_reml_coincide_on_balanced_data_and_are_close_when_nearly_balanced():
    rng = np.random.default_rng(12)
    y = rng.normal(0.4, 0.5, 20)
    balanced = np.full(20, 0.3)
    dl, rm = meta.dersimonian_laird(y, balanced), meta.reml(y, balanced)
    assert rm["tau2"] > 0.0
    assert dl["tau2"] == pytest.approx(rm["tau2"], rel=1e-6)
    assert dl["mu"] == pytest.approx(rm["mu"], rel=1e-9)
    assert dl["se_mu"] == pytest.approx(rm["se_mu"], rel=1e-6)
    diffs = []
    for seed in range(30):
        g = np.random.default_rng(seed)
        se = 0.3 * (1.0 + 0.1 * g.uniform(-1.0, 1.0, 12))
        yy = 0.4 + g.normal(0.0, 0.5, 12) + g.normal(0.0, se)
        d, r = meta.dersimonian_laird(yy, se), meta.reml(yy, se)
        diffs.append(abs(d["tau2"] - r["tau2"]))
        assert abs(d["mu"] - r["mu"]) < 0.02
    assert np.median(diffs) < 0.01
    assert max(diffs) < 0.06


# ----------------------------------------------------------------------------
# Hartung-Knapp-Sidik-Jonkman interval
# ----------------------------------------------------------------------------
def test_hksj_equal_se_equals_a_t_scaled_normal_interval():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    se = np.full(5, 0.5)
    tau2 = np.var(y, ddof=1) - 0.25
    se_mu = np.sqrt((0.25 + tau2) / 5)
    t_mean = stats.t.ppf(0.95, 4)
    t_new = stats.t.ppf(0.95, 3)
    ci = meta.hksj_interval(y, se, level=0.9, new_study=False)
    pi = meta.hksj_interval(y, se, level=0.9, new_study=True)
    assert ci["mean"] == pytest.approx(2.0, rel=1e-9)
    assert ci["hi"] - ci["mean"] == pytest.approx(t_mean * se_mu, rel=1e-6)
    assert ci["mean"] - ci["lo"] == pytest.approx(t_mean * se_mu, rel=1e-6)
    assert pi["hi"] - pi["mean"] == pytest.approx(t_new * np.sqrt(se_mu**2 + tau2), rel=1e-6)
    assert pi["mean"] - pi["lo"] == pytest.approx(t_new * np.sqrt(se_mu**2 + tau2), rel=1e-6)


def test_hksj_is_wider_than_the_plain_normal_interval_for_small_k():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    se = np.full(5, 0.5)
    fit = meta.reml(y, se)
    z = stats.norm.ppf(0.95)
    normal_ci = z * fit["se_mu"]
    normal_pi = z * np.sqrt(fit["se_mu"] ** 2 + fit["tau2"])
    ci = meta.hksj_interval(y, se, level=0.9, new_study=False)
    pi = meta.hksj_interval(y, se, level=0.9, new_study=True)
    assert ci["hi"] - ci["lo"] > 2.0 * normal_ci
    assert pi["hi"] - pi["lo"] > 2.0 * normal_pi
    ratio = (ci["hi"] - ci["lo"]) / (2.0 * normal_ci)
    assert ratio == pytest.approx(stats.t.ppf(0.95, 4) / z, rel=1e-6)


def test_hksj_approaches_the_normal_interval_for_large_k():
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 0.8, 200)
    se = np.full(200, 0.4)
    fit = meta.reml(y, se)
    ci = meta.hksj_interval(y, se, level=0.9, new_study=False)
    ratio = (ci["hi"] - ci["lo"]) / (2.0 * stats.norm.ppf(0.95) * fit["se_mu"])
    assert 1.0 < ratio < 1.01


@pytest.mark.parametrize("seed, tau, q_above_one", [(21, 0.4, True), (24, 0.4, False)])
def test_hksj_matches_the_hand_formula_with_unequal_se(seed, tau, q_above_one):
    y, se = simulate_cases(seed=seed, k=8, tau=tau)
    tau2 = meta.reml(y, se)["tau2"]
    assert tau2 > 0.0
    w = 1.0 / (se**2 + tau2)
    mean = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mean) ** 2) / (8 - 1)
    assert bool(q > 1.0) is q_above_one
    var = max(1.0, q) / np.sum(w)
    t_mean = stats.t.ppf(0.5 + 0.8 / 2.0, 7)
    t_new = stats.t.ppf(0.5 + 0.8 / 2.0, 6)
    ci = meta.hksj_interval(y, se, level=0.8, new_study=False)
    pi = meta.hksj_interval(y, se, level=0.8, new_study=True)
    assert ci["mean"] == pytest.approx(mean, rel=1e-9)
    assert ci["hi"] == pytest.approx(mean + t_mean * np.sqrt(var), rel=1e-6)
    assert ci["lo"] == pytest.approx(mean - t_mean * np.sqrt(var), rel=1e-6)
    assert pi["hi"] == pytest.approx(mean + t_new * np.sqrt(var + tau2), rel=1e-6)
    assert pi["lo"] == pytest.approx(mean - t_new * np.sqrt(var + tau2), rel=1e-6)


def test_hksj_prediction_interval_is_wider_and_level_increases_width():
    y, se = simulate_cases(seed=22, k=10, tau=0.4)
    ci = meta.hksj_interval(y, se, level=0.9, new_study=False)
    pi = meta.hksj_interval(y, se, level=0.9, new_study=True)
    assert pi["lo"] < ci["lo"] < ci["mean"] < ci["hi"] < pi["hi"]
    narrow = meta.hksj_interval(y, se, level=0.5)
    wide = meta.hksj_interval(y, se, level=0.99)
    assert narrow["hi"] - narrow["lo"] < pi["hi"] - pi["lo"] < wide["hi"] - wide["lo"]
    assert set(pi) == {"mean", "lo", "hi"}


def test_hksj_input_checks():
    with pytest.raises(ValueError):
        meta.hksj_interval([0.5], [0.1])
    with pytest.raises(ValueError):
        meta.hksj_interval([0.5, 0.6], [0.1, 0.1], level=1.0)
    with pytest.raises(ValueError):
        meta.hksj_interval([0.5, 0.6], [0.1, 0.1], level=0.0)


def test_hksj_identical_effects_give_the_random_effects_width():
    ci = meta.hksj_interval([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], level=0.9, new_study=False)
    half = stats.t.ppf(0.95, 2) * np.sqrt(1.0 / 300.0)
    assert ci["mean"] == pytest.approx(0.5, abs=1e-12)
    assert ci["hi"] - ci["mean"] == pytest.approx(half, rel=1e-9)
    assert ci["mean"] - ci["lo"] == pytest.approx(half, rel=1e-9)
    pi = meta.hksj_interval([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], level=0.9, new_study=True)
    assert pi["hi"] - pi["mean"] == pytest.approx(
        stats.t.ppf(0.95, 1) * np.sqrt(1.0 / 300.0), rel=1e-9
    )
    near = meta.hksj_interval([0.500, 0.501, 0.499, 0.500], [0.1] * 4, level=0.9, new_study=False)
    assert near["hi"] - near["mean"] == pytest.approx(stats.t.ppf(0.95, 3) * 0.05, rel=1e-9)
    unequal = meta.hksj_interval([2.0, 2.0, 2.0], [0.1, 0.2, 0.4], level=0.8, new_study=False)
    precision = 1.0 / 0.1**2 + 1.0 / 0.2**2 + 1.0 / 0.4**2
    assert unequal["hi"] - unequal["mean"] == pytest.approx(
        stats.t.ppf(0.9, 2) * precision**-0.5, rel=1e-9
    )


@pytest.mark.parametrize("seed", range(12))
def test_hksj_interval_is_never_narrower_than_the_random_effects_interval(seed):
    y, se = simulate_cases(seed=300 + seed, k=7, tau=0.0 if seed % 2 == 0 else 0.15)
    fit = meta.reml(y, se)
    w = 1.0 / (se**2 + fit["tau2"])
    q = np.sum(w * (y - fit["mu"]) ** 2) / (y.size - 1)
    plain = stats.t.ppf(0.95, y.size - 1) * fit["se_mu"]
    ci = meta.hksj_interval(y, se, level=0.9, new_study=False)
    assert ci["hi"] - ci["mean"] >= plain * (1.0 - 1e-12)
    assert ci["mean"] - ci["lo"] >= plain * (1.0 - 1e-12)
    if q < 1.0:
        assert ci["hi"] - ci["mean"] == pytest.approx(plain, rel=1e-9)
    else:
        assert ci["hi"] - ci["mean"] == pytest.approx(plain * np.sqrt(q), rel=1e-9)
    pi = meta.hksj_interval(y, se, level=0.9, new_study=True)
    plain_new = stats.t.ppf(0.95, y.size - 2) * np.sqrt(fit["se_mu"] ** 2 + fit["tau2"])
    assert pi["hi"] - pi["mean"] >= plain_new * (1.0 - 1e-12)


def test_hksj_prediction_interval_for_two_cases_is_nan_with_a_warning():
    with pytest.warns(UserWarning, match="degrees of freedom"):
        out = meta.hksj_interval([0.1, 0.5], [0.1, 0.2])
    assert np.isfinite(out["mean"])
    assert np.isnan(out["lo"]) and np.isnan(out["hi"])
    with pytest.warns(RuntimeWarning):
        meta.hksj_interval([0.1, 0.5], [0.1, 0.2], new_study=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ci = meta.hksj_interval([0.1, 0.5], [0.1, 0.2], new_study=False)
        three = meta.hksj_interval([0.1, 0.5, 0.2], [0.1, 0.2, 0.15])
    assert np.isfinite(ci["lo"]) and np.isfinite(ci["hi"])
    assert np.isfinite(three["lo"]) and np.isfinite(three["hi"])


def test_hksj_intervals_cover_the_mean_and_a_new_case_at_a_reasonable_rate():
    rng = np.random.default_rng(404)
    cover_mean, cover_new = [], []
    for _ in range(250):
        k = 15
        se = rng.uniform(0.1, 0.5, k)
        y = 0.4 + rng.normal(0.0, 0.3, k) + rng.normal(0.0, se)
        new = 0.4 + rng.normal(0.0, 0.3)
        ci = meta.hksj_interval(y, se, level=0.9, new_study=False)
        pi = meta.hksj_interval(y, se, level=0.9, new_study=True)
        cover_mean.append(ci["lo"] <= 0.4 <= ci["hi"])
        cover_new.append(pi["lo"] <= new <= pi["hi"])
    assert 0.84 <= np.mean(cover_mean) <= 0.93
    assert 0.82 <= np.mean(cover_new) <= 0.92


# ----------------------------------------------------------------------------
# Bayesian hierarchical model
# ----------------------------------------------------------------------------
def test_bayes_draws_are_reproducible_with_a_seed():
    y, se = simulate_cases(seed=31, k=8)
    a = meta.bayes_normal_hierarchical(y, se, seed=5, n_draws=3000)
    b = meta.bayes_normal_hierarchical(y, se, seed=5, n_draws=3000)
    c = meta.bayes_normal_hierarchical(y, se, seed=6, n_draws=3000)
    for name in ("draws_mu", "draws_tau", "draws_pred", "shrunk_effects"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert not np.array_equal(a.draws_mu, c.draws_mu)
    assert not np.array_equal(a.draws_tau, c.draws_tau)
    assert np.allclose(a.shrunk_effects, c.shrunk_effects)
    pd.testing.assert_frame_equal(a.summary, b.summary)
    d = meta.bayes_normal_hierarchical(y, se, seed=np.random.default_rng(5), n_draws=3000)
    assert np.array_equal(a.draws_mu, d.draws_mu)
    assert np.array_equal(a.draws_pred, d.draws_pred)


def test_bayes_prior_scale_defaults_to_the_sample_sd():
    y, se = simulate_cases(seed=32, k=7)
    default = meta.bayes_normal_hierarchical(y, se, n_draws=100)
    explicit = meta.bayes_normal_hierarchical(y, se, tau_prior_scale=0.25, n_draws=100)
    assert default.tau_prior_scale == pytest.approx(np.std(y, ddof=1), rel=1e-12)
    assert explicit.tau_prior_scale == 0.25


def test_bayes_default_mu_prior_follows_the_scale_of_the_data():
    spread = meta.bayes_normal_hierarchical(
        [0.0, 10.0, 20.0, 30.0], np.ones(4), n_draws=100
    )
    assert spread.mu_prior[0] == 0.0
    assert spread.mu_prior[1] == pytest.approx(
        5.0 * np.std([0.0, 10.0, 20.0, 30.0], ddof=1), rel=1e-12
    )
    noisy = meta.bayes_normal_hierarchical(
        [0.0, 0.1, 0.2], [1.0, 2.0, 30.0], tau_prior_scale=0.1, n_draws=100
    )
    assert noisy.mu_prior == (0.0, 10.0)
    single = meta.bayes_normal_hierarchical([0.7], [0.2], tau_prior_scale=0.5, n_draws=100)
    assert single.mu_prior == (0.0, pytest.approx(1.0, rel=1e-12))
    explicit = meta.bayes_normal_hierarchical(
        [0.0, 10.0, 20.0, 30.0], np.ones(4), mu_prior=(1.5, 0.25), n_draws=100
    )
    assert explicit.mu_prior == (1.5, 0.25)


def test_bayes_default_mu_prior_does_not_pull_effects_in_percent_towards_zero():
    rng = np.random.default_rng(12)
    se = rng.uniform(10.0, 25.0, 12)
    y = 60.0 + rng.normal(0.0, 30.0, 12) + rng.normal(0.0, se)
    fit = meta.reml(y, se)
    out = meta.bayes_normal_hierarchical(y, se, seed=1)
    assert abs(out.summary.loc["mu", "mean"] - fit["mu"]) < 0.25 * fit["se_mu"]
    assert out.summary.loc["predictive", "q50"] == pytest.approx(fit["mu"], abs=0.5 * fit["se_mu"])
    wide = meta.bayes_normal_hierarchical(y, se, mu_prior=(0.0, 1000.0), seed=1)
    assert out.summary.loc["mu", "mean"] == pytest.approx(wide.summary.loc["mu", "mean"], abs=2.0)
    assert out.summary.loc["mu", "mean"] > 40.0


@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_bayes_with_default_priors_is_equivariant_to_the_scale_of_the_effects(factor):
    y, se = simulate_cases(seed=39, k=9, mu=1.0, tau=0.4)
    base = meta.bayes_normal_hierarchical(y, se, seed=3, n_draws=2000)
    scaled = meta.bayes_normal_hierarchical(factor * y, factor * se, seed=3, n_draws=2000)
    assert scaled.mu_prior[1] == pytest.approx(factor * base.mu_prior[1], rel=1e-12)
    assert scaled.tau_prior_scale == pytest.approx(factor * base.tau_prior_scale, rel=1e-12)
    assert np.allclose(scaled.draws_mu, factor * base.draws_mu, rtol=1e-8)
    assert np.allclose(scaled.draws_pred, factor * base.draws_pred, rtol=1e-8)
    assert np.allclose(scaled.shrunk_effects, factor * base.shrunk_effects, rtol=1e-8)


def test_bayes_tau_posterior_equals_the_brute_force_marginal_likelihood():
    y, se = simulate_cases(seed=33, k=6)
    m0, s0 = 0.5, 2.0
    out = meta.bayes_normal_hierarchical(y, se, mu_prior=(m0, s0), n_grid=801, n_draws=100)
    scale = np.std(y, ddof=1)
    grid = out.tau_grid
    assert grid[0] == 0.0 and np.all(np.diff(grid) > 0.0)
    log_post = np.empty(grid.size)
    ones = np.ones(y.size)
    for j, tau in enumerate(grid):
        cov = np.diag(se**2 + tau**2) + s0**2 * np.outer(ones, ones)
        log_post[j] = stats.multivariate_normal.logpdf(y, mean=m0 * ones, cov=cov)
        log_post[j] += -0.5 * (tau / scale) ** 2
    dens = np.exp(log_post - log_post.max())
    dens /= integrate.trapezoid(dens, grid)
    assert integrate.trapezoid(out.tau_density, grid) == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(out.tau_density - dens)) < 1e-7 * dens.max()
    assert out.tau_density[-1] < 1e-6 * out.tau_density.max()


@pytest.mark.parametrize("m0, s0", [(0.0, 10.0), (1.0, 0.25), (-0.5, 1.0)])
def test_bayes_agrees_with_a_two_dimensional_grid_posterior(m0, s0):
    y, se = simulate_cases(seed=34, k=6)
    out = meta.bayes_normal_hierarchical(y, se, mu_prior=(m0, s0), seed=1)
    scale = np.std(y, ddof=1)
    taus = np.linspace(0.0, out.tau_grid[-1], 900)
    mus = np.linspace(y.min() - 4.0, y.max() + 4.0, 900)
    tt, mm = np.meshgrid(taus, mus, indexing="ij")
    logp = -0.5 * (mm - m0) ** 2 / s0**2 - 0.5 * (tt / scale) ** 2
    for yi, si in zip(y, se):
        total = si**2 + tt**2
        logp += -0.5 * np.log(total) - 0.5 * (yi - mm) ** 2 / total
    p = np.exp(logp - logp.max())
    p /= integrate.trapezoid(integrate.trapezoid(p, mus, axis=1), taus)
    p_tau = integrate.trapezoid(p, mus, axis=1)
    p_mu = integrate.trapezoid(p, taus, axis=0)

    def moments(density, x):
        mean = integrate.trapezoid(density * x, x)
        return mean, np.sqrt(integrate.trapezoid(density * x**2, x) - mean**2)

    mu_mean, mu_sd = moments(p_mu, mus)
    tau_mean, tau_sd = moments(p_tau, taus)
    assert out.summary.loc["mu", "mean"] == pytest.approx(mu_mean, abs=0.03 * mu_sd)
    assert out.summary.loc["mu", "sd"] == pytest.approx(mu_sd, rel=0.03)
    assert out.summary.loc["tau", "mean"] == pytest.approx(tau_mean, abs=0.03 * tau_sd)
    assert out.summary.loc["tau", "sd"] == pytest.approx(tau_sd, rel=0.03)
    expected = []
    for yi, si in zip(y, se):
        pull = si**2 / (si**2 + tt**2)
        expected.append(
            integrate.trapezoid(
                integrate.trapezoid(p * ((1.0 - pull) * yi + pull * mm), mus, axis=1), taus
            )
        )
    assert np.allclose(out.shrunk_effects, expected, atol=2e-3)


def test_bayes_posterior_mean_of_mu_is_close_to_reml_with_a_vague_prior_and_many_studies():
    y, se = simulate_cases(seed=35, k=60, mu=0.5, tau=0.3, se_low=0.1, se_high=0.5)
    fit = meta.reml(y, se)
    out = meta.bayes_normal_hierarchical(
        y, se, mu_prior=(0.0, 100.0), tau_prior_scale=1.0, seed=3
    )
    assert abs(out.summary.loc["mu", "mean"] - fit["mu"]) < 0.05 * fit["se_mu"]
    assert out.summary.loc["mu", "sd"] == pytest.approx(fit["se_mu"], rel=0.1)
    assert out.summary.loc["tau", "q50"] == pytest.approx(np.sqrt(fit["tau2"]), rel=0.1)


def test_bayes_summary_predictive_draws_and_probability_are_consistent():
    y, se = simulate_cases(seed=36, k=9, mu=0.4, tau=0.3)
    out = meta.bayes_normal_hierarchical(y, se, seed=2)
    assert list(out.summary.index) == ["mu", "tau", "predictive"]
    assert list(out.summary.columns) == ["mean", "sd", "q5", "q25", "q50", "q75", "q95"]
    draws = {"mu": out.draws_mu, "tau": out.draws_tau, "predictive": out.draws_pred}
    for name, values in draws.items():
        assert values.shape == (20000,)
        assert out.summary.loc[name, "mean"] == pytest.approx(values.mean(), rel=1e-12)
        assert out.summary.loc[name, "sd"] == pytest.approx(values.std(ddof=1), rel=1e-12)
        qs = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        assert np.allclose(out.summary.loc[name, ["q5", "q25", "q50", "q75", "q95"]], qs)
    assert np.all(out.draws_tau >= 0.0)
    assert out.prob_pred_positive == pytest.approx(np.mean(out.draws_pred > 0.0), rel=1e-12)
    assert 0.0 <= out.prob_pred_positive <= 1.0
    q = out.predictive_quantiles([0.05, 0.5, 0.95])
    assert np.allclose(q, out.summary.loc["predictive", ["q5", "q50", "q95"]])
    assert out.predictive_quantiles(0.5) == pytest.approx(out.summary.loc["predictive", "q50"])
    assert out.draws_pred.var() > out.draws_mu.var()
    assert out.predictive_quantiles(0.05) < out.predictive_quantiles(0.95)


def test_bayes_predictive_draws_follow_mu_plus_tau_times_a_normal():
    y, se = simulate_cases(seed=37, k=9)
    out = meta.bayes_normal_hierarchical(y, se, seed=4)
    z = (out.draws_pred - out.draws_mu) / np.where(out.draws_tau > 0, out.draws_tau, np.nan)
    z = z[np.isfinite(z)]
    assert abs(z.mean()) < 0.03
    assert z.std() == pytest.approx(1.0, abs=0.03)
    assert stats.kstest(z, "norm").pvalue > 1e-3


def test_bayes_shrinks_case_effects_towards_the_pooled_mean_by_precision():
    y = np.array([-0.4, 0.0, 0.3, 0.9, 1.4, 2.5])
    se = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 3.0])
    out = meta.bayes_normal_hierarchical(y, se, seed=0)
    pooled = out.summary.loc["mu", "mean"]
    assert np.std(out.shrunk_effects) < np.std(y)
    moved = (out.shrunk_effects - y) / (pooled - y)
    assert np.all((moved > 0.0) & (moved < 1.0))
    assert np.all(moved[:5] < 0.4)
    assert moved[5] > 0.8
    assert moved[5] > np.max(moved[:5]) + 0.5
    precise = meta.bayes_normal_hierarchical(y, np.full(6, 1e-3), seed=0)
    assert np.allclose(precise.shrunk_effects, y, atol=2e-3)


def test_bayes_resolves_the_posterior_for_wide_prior_scales():
    y = np.array([0.2, 0.9, 0.5])
    se = np.array([0.2, 0.3, 0.25])
    a = meta.bayes_normal_hierarchical(y, se, tau_prior_scale=50.0, seed=2)
    b = meta.bayes_normal_hierarchical(y, se, tau_prior_scale=500.0, seed=2)
    assert a.summary.loc["tau", "q50"] == pytest.approx(b.summary.loc["tau", "q50"], rel=0.05)
    assert integrate.trapezoid(a.tau_density, a.tau_grid) == pytest.approx(1.0, abs=1e-12)
    assert a.tau_density[-1] < 1e-6 * a.tau_density.max()


def test_bayes_homogeneous_data_concentrate_tau_near_zero():
    y = 0.5 + np.array([0.001, -0.002, 0.0015, 0.0, -0.001, 0.002])
    out = meta.bayes_normal_hierarchical(y, np.full(6, 0.3), seed=1)
    assert out.summary.loc["tau", "q95"] < 0.01
    assert out.summary.loc["mu", "mean"] == pytest.approx(0.5, abs=0.02)
    assert out.tau_density[0] == out.tau_density.max()


def test_bayes_predictive_interval_has_reasonable_coverage():
    rng = np.random.default_rng(505)
    covered = []
    for i in range(150):
        k = 10
        se = rng.uniform(0.1, 0.5, k)
        y = 0.4 + rng.normal(0.0, 0.3, k) + rng.normal(0.0, se)
        new = 0.4 + rng.normal(0.0, 0.3)
        out = meta.bayes_normal_hierarchical(y, se, n_draws=3000, seed=i)
        lo, hi = out.predictive_quantiles([0.05, 0.95])
        covered.append(lo <= new <= hi)
    assert 0.82 <= np.mean(covered) <= 0.92


def test_bayes_single_case_needs_an_explicit_prior_scale_and_inputs_are_checked():
    with pytest.raises(ValueError):
        meta.bayes_normal_hierarchical([0.7], [0.2])
    out = meta.bayes_normal_hierarchical([0.7], [0.2], tau_prior_scale=0.5, n_draws=500)
    assert out.shrunk_effects.shape == (1,)
    y, se = simulate_cases(seed=38, k=5)
    for kwargs in (
        {"tau_prior_scale": 0.0},
        {"tau_prior_scale": -1.0},
        {"mu_prior": (0.0, 0.0)},
        {"mu_prior": (np.nan, 1.0)},
        {"mu_prior": (0.0,)},
        {"n_grid": 2},
        {"n_draws": 0},
    ):
        with pytest.raises(ValueError):
            meta.bayes_normal_hierarchical(y, se, **kwargs)
    with pytest.raises(ValueError):
        meta.bayes_normal_hierarchical(np.full(4, 0.3), np.full(4, 0.1))


# ----------------------------------------------------------------------------
# Invariance properties
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [1e-4, 1e3])
def test_pooling_and_intervals_are_scale_equivariant(factor):
    y, se = simulate_cases(seed=71, k=14, tau=0.4)
    for fn in (meta.dersimonian_laird, meta.reml):
        base, scaled = fn(y, se), fn(factor * y, factor * se)
        assert scaled["tau2"] == pytest.approx(factor**2 * base["tau2"], rel=1e-6)
        assert scaled["mu"] == pytest.approx(factor * base["mu"], rel=1e-6)
        assert scaled["se_mu"] == pytest.approx(factor * base["se_mu"], rel=1e-6)
        assert scaled["Q"] == pytest.approx(base["Q"], rel=1e-9)
        assert scaled["I2"] == pytest.approx(base["I2"], rel=1e-9)
    base_pi = meta.hksj_interval(y, se)
    scaled_pi = meta.hksj_interval(factor * y, factor * se)
    for key in ("mean", "lo", "hi"):
        assert scaled_pi[key] == pytest.approx(factor * base_pi[key], rel=1e-5)
    base_b = meta.bayes_normal_hierarchical(y, se, seed=3, n_draws=2000)
    scaled_b = meta.bayes_normal_hierarchical(factor * y, factor * se, seed=3, n_draws=2000)
    assert np.allclose(scaled_b.draws_mu, factor * base_b.draws_mu, rtol=1e-8)
    assert np.allclose(scaled_b.draws_pred, factor * base_b.draws_pred, rtol=1e-8)
    assert np.allclose(scaled_b.shrunk_effects, factor * base_b.shrunk_effects, rtol=1e-8)


def test_pooling_is_translation_invariant_and_ignores_the_order_of_cases():
    y, se = simulate_cases(seed=72, k=14, tau=0.4)
    base = meta.reml(y, se)
    shifted = meta.reml(y + 5.0, se)
    assert shifted["tau2"] == pytest.approx(base["tau2"], rel=1e-6)
    assert shifted["mu"] == pytest.approx(base["mu"] + 5.0, rel=1e-9)
    order = np.random.default_rng(0).permutation(14)
    for fn in (meta.dersimonian_laird, meta.reml):
        a, b = fn(y, se), fn(y[order], se[order])
        assert b["tau2"] == pytest.approx(a["tau2"], rel=1e-6, abs=1e-12)
        assert b["mu"] == pytest.approx(a["mu"], rel=1e-9)
    a, b = meta.hksj_interval(y, se), meta.hksj_interval(y[order], se[order])
    assert b["lo"] == pytest.approx(a["lo"], rel=1e-6)


def test_meta_regression_is_invariant_to_affine_changes_of_the_features():
    y, se, X = make_regression_data(seed=73, k=25, slopes=(0.3, -0.2), tau=0.15)
    shift, scale = np.array([5.0, -3.0]), np.array([100.0, 0.01])
    base = meta.meta_regression(y, se, X, x_new=X[0], ridge=0.5)
    moved = meta.meta_regression(y, se, X * scale + shift, x_new=X[0] * scale + shift, ridge=0.5)
    assert moved["tau2"] == pytest.approx(base["tau2"], rel=1e-6)
    assert moved["prediction"]["mean"] == pytest.approx(base["prediction"]["mean"], rel=1e-7)
    assert moved["prediction"]["hi"] == pytest.approx(base["prediction"]["hi"], rel=1e-6)
    assert moved["loo_rmse"] == pytest.approx(base["loo_rmse"], rel=1e-6)
    assert np.allclose(moved["coef"]["estimate"], base["coef"]["estimate"], rtol=1e-6, atol=1e-9)
    assert np.allclose(
        moved["coef_raw"]["estimate"].iloc[1:],
        base["coef_raw"]["estimate"].iloc[1:] / scale,
        rtol=1e-6,
    )


@pytest.mark.parametrize("ridge", [0.5, 50.0])
def test_meta_regression_ridge_penalty_does_not_depend_on_the_units_of_the_features(ridge):
    y, se, X = make_regression_data(seed=75, k=25, slopes=(0.3, -0.2), tau=0.15)
    units = np.array([1e6, 1e-6])
    base = meta.meta_regression(y, se, X, x_new=X[:2], ridge=ridge)
    moved = meta.meta_regression(y, se, X * units, x_new=X[:2] * units, ridge=ridge)
    assert moved["tau2"] == pytest.approx(base["tau2"], rel=1e-6)
    assert np.allclose(moved["coef"]["estimate"], base["coef"]["estimate"], rtol=1e-6, atol=1e-9)
    assert np.allclose(moved["coef"]["se"], base["coef"]["se"], rtol=1e-6)
    assert np.allclose(moved["prediction"]["mean"], base["prediction"]["mean"], rtol=1e-6)
    assert np.allclose(moved["prediction"]["hi"], base["prediction"]["hi"], rtol=1e-6)
    assert moved["loo_rmse"] == pytest.approx(base["loo_rmse"], rel=1e-6)
    raw = moved["coef_raw"]["estimate"].iloc[1:].to_numpy()
    standardised = moved["coef"]["estimate"].iloc[1:].to_numpy()
    assert np.allclose(raw, standardised / (X * units).std(axis=0), rtol=1e-9)
    assert np.allclose(raw, base["coef_raw"]["estimate"].iloc[1:] / units, rtol=1e-6)


def test_meta_regression_is_equivariant_to_the_scale_and_location_of_the_effects():
    y, se, X = make_regression_data(seed=74, k=25, slopes=(0.3, -0.2), tau=0.15)
    base = meta.meta_regression(y, se, X, ridge=0.5)
    scaled = meta.meta_regression(1e3 * y, 1e3 * se, X, ridge=0.5)
    assert scaled["tau2"] == pytest.approx(1e6 * base["tau2"], rel=1e-6)
    assert np.allclose(scaled["coef"]["estimate"], 1e3 * base["coef"]["estimate"], rtol=1e-6)
    assert scaled["loo_rmse"] == pytest.approx(1e3 * base["loo_rmse"], rel=1e-6)
    shifted = meta.meta_regression(y + 7.0, se, X, ridge=0.5)
    assert shifted["tau2"] == pytest.approx(base["tau2"], rel=1e-6)
    assert np.allclose(
        shifted["coef"]["estimate"].iloc[1:], base["coef"]["estimate"].iloc[1:], atol=1e-6
    )
    assert shifted["coef"].loc["intercept", "estimate"] == pytest.approx(
        base["coef"].loc["intercept", "estimate"] + 7.0, rel=1e-7
    )


# ----------------------------------------------------------------------------
# Meta-regression
# ----------------------------------------------------------------------------
def make_regression_data(seed, k, slopes=(0.30, -0.15), intercept=1.0, tau=0.1):
    rng = np.random.default_rng(seed)
    p = len(slopes)
    X = rng.normal(loc=rng.uniform(-3, 8, p), scale=rng.uniform(0.5, 3.0, p), size=(k, p))
    se = rng.uniform(0.05, 0.3, k)
    y = intercept + X @ np.asarray(slopes) + rng.normal(0.0, tau, k) + rng.normal(0.0, se)
    return y, se, X


def test_meta_regression_recovers_a_known_slope():
    y, se, X = make_regression_data(seed=41, k=80, slopes=(0.30, -0.15), tau=0.1)
    out = meta.meta_regression(y, se, X)
    raw = out["coef_raw"]
    assert list(raw.index) == ["intercept", "x0", "x1"]
    assert raw.loc["x0", "estimate"] == pytest.approx(0.30, abs=4 * raw.loc["x0", "se"])
    assert raw.loc["x1", "estimate"] == pytest.approx(-0.15, abs=4 * raw.loc["x1", "se"])
    assert raw.loc["x0", "estimate"] == pytest.approx(0.30, abs=0.03)
    assert raw.loc["x1", "estimate"] == pytest.approx(-0.15, abs=0.03)
    intercept = raw.loc["intercept"]
    assert intercept["estimate"] == pytest.approx(1.0, abs=4 * intercept["se"])
    assert out["tau2"] == pytest.approx(0.01, abs=0.01)
    scale = X.std(axis=0)
    std = out["coef"]
    assert std.loc["x0", "estimate"] == pytest.approx(
        raw.loc["x0", "estimate"] * scale[0], rel=1e-9
    )
    assert std.loc["x1", "se"] == pytest.approx(raw.loc["x1", "se"] * scale[1], rel=1e-9)


def test_meta_regression_with_a_null_slope_does_not_find_one():
    y, se, X = make_regression_data(seed=42, k=60, slopes=(0.0, 0.0), tau=0.1)
    out = meta.meta_regression(y, se, X)
    z = out["coef"]["estimate"] / out["coef"]["se"]
    assert np.all(np.abs(z.loc[["x0", "x1"]]) < 3.5)


def test_meta_regression_coefficients_match_generalised_least_squares_by_hand():
    y, se, X = make_regression_data(seed=43, k=30, slopes=(0.2, 0.1, -0.3), tau=0.15)
    out = meta.meta_regression(y, se, X)
    design = np.column_stack([np.ones(30), standardise(X)])
    w = 1.0 / (se**2 + out["tau2"])
    info = design.T @ (design * w[:, None])
    cov = np.linalg.inv(info)
    beta = cov @ (design.T @ (w * y))
    assert np.allclose(out["coef"]["estimate"], beta, rtol=1e-9, atol=1e-12)
    assert np.allclose(out["coef"]["se"], np.sqrt(np.diag(cov)), rtol=1e-9)
    assert out["df"] == 30 - 4


@pytest.mark.parametrize("seed, k, p", [(44, 14, 1), (45, 20, 2), (46, 25, 3)])
def test_meta_regression_tau2_maximises_the_restricted_likelihood(seed, k, p):
    y, se, X = make_regression_data(seed, k, slopes=tuple(np.linspace(0.3, -0.2, p)), tau=0.25)
    design = np.column_stack([np.ones(k), standardise(X)])
    ref = contrast_reml_tau2(y, se**2, design)
    out = meta.meta_regression(y, se, X, ridge=0.0)
    assert out["tau2"] == pytest.approx(ref, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("ridge", [0.5, 3.0])
def test_meta_regression_ridge_tau2_matches_a_gaussian_prior_on_the_slopes(ridge):
    y, se, X = make_regression_data(seed=47, k=22, slopes=(0.3, -0.2), tau=0.2)
    precision = ridge * np.mean(1.0 / se**2)
    ref = contrast_reml_tau2_ridge(y, se**2, standardise(X), precision)
    out = meta.meta_regression(y, se, X, ridge=ridge)
    assert out["tau2"] == pytest.approx(ref, rel=1e-4, abs=1e-6)
    design = np.column_stack([np.ones(22), standardise(X)])
    w = 1.0 / (se**2 + out["tau2"])
    info = design.T @ (design * w[:, None]) + np.diag([0.0, precision, precision])
    beta = np.linalg.solve(info, design.T @ (w * y))
    assert np.allclose(out["coef"]["estimate"], beta, rtol=1e-9, atol=1e-12)
    assert np.allclose(out["coef"]["se"], np.sqrt(np.diag(np.linalg.inv(info))), rtol=1e-9)


def test_meta_regression_ridge_shrinks_the_slopes_towards_zero():
    y, se, X = make_regression_data(seed=48, k=30, slopes=(0.4, -0.3, 0.2), tau=0.1)
    norms = []
    for ridge in (0.0, 1.0, 20.0, 1e6):
        out = meta.meta_regression(y, se, X, ridge=ridge)
        slopes = out["coef"]["estimate"].drop("intercept").to_numpy()
        norms.append(np.linalg.norm(slopes))
    assert norms[0] > norms[1] > norms[2] > norms[3]
    assert norms[3] < 1e-3 * norms[0]


def test_meta_regression_prediction_and_interval():
    y, se, X = make_regression_data(seed=49, k=40, slopes=(0.3, -0.2), tau=0.1)
    out = meta.meta_regression(y, se, X, x_new=X.mean(axis=0), level=0.9)
    pred = out["prediction"]
    assert set(pred) == {"mean", "lo", "hi", "se_pred"}
    assert pred["mean"] == pytest.approx(out["coef"].loc["intercept", "estimate"], rel=1e-9)
    assert pred["se_pred"] >= np.sqrt(out["tau2"])
    t = stats.t.ppf(0.95, out["df"])
    assert pred["hi"] - pred["mean"] == pytest.approx(t * pred["se_pred"], rel=1e-9)
    assert pred["mean"] - pred["lo"] == pytest.approx(t * pred["se_pred"], rel=1e-9)
    wide = meta.meta_regression(y, se, X, x_new=X.mean(axis=0), level=0.99)["prediction"]
    assert wide["hi"] - wide["lo"] > pred["hi"] - pred["lo"]
    far = meta.meta_regression(y, se, X, x_new=X.mean(axis=0) + 5 * X.std(axis=0))["prediction"]
    assert far["se_pred"] > pred["se_pred"]
    raw = out["coef_raw"]["estimate"].to_numpy()
    x_new = X[3]
    manual = raw[0] + raw[1:] @ x_new
    assert meta.meta_regression(y, se, X, x_new=x_new)["prediction"]["mean"] == pytest.approx(
        manual, rel=1e-9
    )
    assert out["prediction"] is not None
    assert meta.meta_regression(y, se, X)["prediction"] is None


def test_meta_regression_accepts_several_new_cases_and_dataframes():
    y, se, X = make_regression_data(seed=50, k=25, slopes=(0.3, -0.2), tau=0.1)
    frame = pd.DataFrame(X, columns=["size", "gdp"])
    one = meta.meta_regression(y, se, frame, x_new=X[0])
    many = meta.meta_regression(y, se, frame, x_new=X[:3])
    assert list(one["coef"].index) == ["intercept", "size", "gdp"]
    assert isinstance(one["prediction"]["mean"], float)
    assert many["prediction"]["mean"].shape == (3,)
    assert many["prediction"]["mean"][0] == pytest.approx(one["prediction"]["mean"], rel=1e-12)
    assert np.all(many["prediction"]["lo"] < many["prediction"]["hi"])
    series = meta.meta_regression(y, se, pd.Series(X[:, 0], name="size"), x_new=[1.0])
    assert list(series["coef"].index) == ["intercept", "size"]
    scalar = meta.meta_regression(y, se, X[:, 0], x_new=1.0)
    assert isinstance(scalar["prediction"]["mean"], float)
    assert scalar["prediction"]["mean"] == pytest.approx(series["prediction"]["mean"], rel=1e-12)


@pytest.mark.parametrize("new_case", ["series", "mapping", "frame", "frame_with_extras"])
def test_meta_regression_aligns_new_cases_by_name(new_case):
    y, se, X = make_regression_data(seed=56, k=25, slopes=(0.3, -0.2), tau=0.1)
    frame = pd.DataFrame(X, columns=["size", "gdp"])
    row = frame.iloc[3]
    positional = meta.meta_regression(y, se, frame, x_new=X[3], ridge=0.5)["prediction"]
    inputs = {
        "series": row[["gdp", "size"]],
        "mapping": {"gdp": row["gdp"], "size": row["size"]},
        "frame": frame.iloc[[3]][["gdp", "size"]],
        "frame_with_extras": frame.iloc[[3]][["gdp", "size"]].assign(
            label="Thailand", unused=99.0
        ),
    }
    out = meta.meta_regression(y, se, frame, x_new=inputs[new_case], ridge=0.5)["prediction"]
    swapped = meta.meta_regression(y, se, frame, x_new=X[3][::-1], ridge=0.5)["prediction"]
    assert abs(swapped["mean"] - positional["mean"]) > 1e-3
    for key in ("mean", "lo", "hi", "se_pred"):
        assert np.ravel(out[key])[0] == pytest.approx(positional[key], rel=1e-12)
    if new_case in ("series", "mapping"):
        assert all(isinstance(value, float) for value in out.values())
    else:
        assert all(np.shape(value) == (1,) for value in out.values())


def test_meta_regression_aligns_several_new_cases_and_unlabelled_arrays_by_name():
    y, se, X = make_regression_data(seed=57, k=25, slopes=(0.3, -0.2), tau=0.1)
    base = meta.meta_regression(y, se, X, x_new=X[:4])["prediction"]
    reordered = pd.DataFrame(X[:4][:, ::-1], columns=["x1", "x0"])
    out = meta.meta_regression(y, se, X, x_new=reordered)["prediction"]
    assert np.allclose(out["mean"], base["mean"], rtol=1e-12)
    mapping = {"x1": X[:4, 1], "x0": X[:4, 0], "extra": np.zeros(4)}
    again = meta.meta_regression(y, se, X, x_new=mapping)["prediction"]
    assert np.allclose(again["lo"], base["lo"], rtol=1e-12)
    assert again["mean"].shape == (4,)
    positional = meta.meta_regression(y, se, X, x_new=pd.Series(X[0]))["prediction"]
    assert positional["mean"] == pytest.approx(base["mean"][0], rel=1e-12)
    plain_frame = meta.meta_regression(y, se, X, x_new=pd.DataFrame(X[:4]))["prediction"]
    assert np.allclose(plain_frame["mean"], base["mean"], rtol=1e-12)


def test_meta_regression_rejects_new_cases_with_missing_or_repeated_names():
    y, se, X = make_regression_data(seed=58, k=25, slopes=(0.3, -0.2), tau=0.1)
    frame = pd.DataFrame(X, columns=["size", "gdp"])
    for bad in (
        pd.Series({"size": 1.0, "other": 2.0}),
        {"size": 1.0},
        {"size": 1.0, "Gdp": 2.0},
        pd.DataFrame({"size": [1.0], "other": [2.0]}),
    ):
        with pytest.raises(ValueError, match=r"no value for the features \['gdp'\]"):
            meta.meta_regression(y, se, frame, x_new=bad)
    with pytest.raises(ValueError, match="unique"):
        repeated = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["size", "gdp", "gdp"])
        meta.meta_regression(y, se, frame, x_new=repeated)
    with pytest.raises(ValueError, match="unique"):
        meta.meta_regression(y, se, pd.DataFrame(X, columns=["a", "a"]))
    with pytest.raises(ValueError, match="finite"):
        meta.meta_regression(y, se, frame, x_new={"size": np.nan, "gdp": 1.0})


def test_meta_regression_prediction_interval_needs_a_positive_number_of_degrees_of_freedom():
    rng = np.random.default_rng(59)
    X = rng.normal(size=(6, 5))
    y = 0.4 * X[:, 0] + rng.normal(0.0, 0.1, 6)
    se = rng.uniform(0.05, 0.2, 6)
    with pytest.warns(UserWarning, match="degrees of freedom"):
        out = meta.meta_regression(y, se, X, x_new=X[0], ridge=1.0)
    assert out["df"] == 0
    assert np.isfinite(out["prediction"]["mean"]) and np.isfinite(out["prediction"]["se_pred"])
    assert np.isnan(out["prediction"]["lo"]) and np.isnan(out["prediction"]["hi"])
    with pytest.warns(RuntimeWarning):
        many = meta.meta_regression(y[:5], se[:5], X[:5], x_new=X[:2], ridge=1.0)
    assert many["df"] == -1
    assert np.all(np.isnan(many["prediction"]["lo"])) and many["prediction"]["lo"].shape == (2,)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        meta.meta_regression(y, se, X, ridge=1.0)
        enough = meta.meta_regression(y, se, X[:, :4], x_new=X[0, :4], ridge=1.0)
    assert enough["df"] == 1
    assert enough["prediction"]["hi"] - enough["prediction"]["mean"] == pytest.approx(
        stats.t.ppf(0.95, 1) * enough["prediction"]["se_pred"], rel=1e-12
    )


@pytest.mark.parametrize("seed, k, p", [(60, 14, 1), (61, 20, 3)])
def test_meta_regression_prediction_interval_uses_k_minus_p_minus_one_degrees_of_freedom(
    seed, k, p
):
    y, se, X = make_regression_data(seed, k, slopes=tuple(np.linspace(0.3, -0.2, p)), tau=0.1)
    out = meta.meta_regression(y, se, X, x_new=X[0], level=0.8)
    pred = out["prediction"]
    assert out["df"] == k - p - 1
    assert pred["hi"] - pred["mean"] == pytest.approx(
        stats.t.ppf(0.9, k - p - 1) * pred["se_pred"], rel=1e-12
    )


def test_meta_regression_leave_one_out_refits_without_the_case():
    y, se, X = make_regression_data(seed=51, k=18, slopes=(0.3, -0.2), tau=0.15)
    out = meta.meta_regression(y, se, X)
    assert np.isfinite(out["loo_rmse"]) and out["loo_rmse"] > 0.0
    assert out["loo_pred"].shape == (18,)
    assert np.allclose(out["loo_errors"], y - out["loo_pred"])
    assert out["loo_rmse"] == pytest.approx(np.sqrt(np.mean(out["loo_errors"] ** 2)), rel=1e-12)
    for i in (0, 7, 17):
        rows = np.arange(18) != i
        refit = meta.meta_regression(y[rows], se[rows], X[rows], x_new=X[i])
        assert out["loo_pred"][i] == pytest.approx(refit["prediction"]["mean"], rel=1e-9)
        assert out["loo_var"][i] == pytest.approx(se[i] ** 2 + refit["tau2"], rel=1e-9)
    assert np.all(out["loo_var"] >= se**2)


def test_meta_regression_leave_one_out_error_is_smaller_for_informative_features():
    y, se, X = make_regression_data(seed=52, k=40, slopes=(0.6, 0.0), tau=0.05)
    informative = meta.meta_regression(y, se, X)["loo_rmse"]
    noise = np.random.default_rng(1).normal(size=(40, 1))
    uninformative = meta.meta_regression(y, se, noise)["loo_rmse"]
    assert informative < uninformative


def test_meta_regression_constant_columns_get_a_zero_coefficient():
    y, se, X = make_regression_data(seed=53, k=24, slopes=(0.3,), tau=0.1)
    X2 = np.column_stack([X[:, 0], np.full(24, 3.0)])
    out = meta.meta_regression(y, se, X2, x_new=[X[0, 0], 3.0])
    base = meta.meta_regression(y, se, X, x_new=[X[0, 0]])
    assert out["coef"].loc["x1", "estimate"] == 0.0
    assert np.isnan(out["coef"].loc["x1", "se"])
    assert out["coef"].loc["x0", "estimate"] == pytest.approx(base["coef"].loc["x0", "estimate"])
    assert out["prediction"]["mean"] == pytest.approx(base["prediction"]["mean"], rel=1e-9)
    assert out["loo_rmse"] == pytest.approx(base["loo_rmse"], rel=1e-9)


def test_meta_regression_needs_a_ridge_when_there_are_too_many_features():
    rng = np.random.default_rng(54)
    X = rng.normal(size=(8, 12))
    y = 0.5 * X[:, 0] + rng.normal(0.0, 0.1, 8)
    se = rng.uniform(0.05, 0.2, 8)
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, X, ridge=0.0)
    with pytest.warns(UserWarning, match="degrees of freedom"):
        out = meta.meta_regression(y, se, X, x_new=X[:2], ridge=1.0)
    assert out["df"] == 8 - 13
    assert np.isfinite(out["loo_rmse"])
    assert np.isfinite(out["tau2"])
    assert np.all(np.isfinite(out["prediction"]["mean"]))
    assert np.all(np.isnan(out["prediction"]["lo"]))
    square = rng.normal(size=(5, 4))
    with pytest.raises(ValueError):
        meta.meta_regression(rng.normal(size=5), np.full(5, 0.1), square, ridge=0.0)


def test_meta_regression_input_checks():
    y, se, X = make_regression_data(seed=55, k=12, slopes=(0.3,), tau=0.1)
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, X[:-1])
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, X, ridge=-1.0)
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, X, level=1.5)
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, X, x_new=[1.0, 2.0])
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, np.full((12, 1), np.nan))
    collinear = np.column_stack([X[:, 0], 2.0 * X[:, 0] + 1.0])
    with pytest.raises(ValueError):
        meta.meta_regression(y, se, collinear)
    assert np.isfinite(meta.meta_regression(y, se, collinear, ridge=1.0)["loo_rmse"])


def test_meta_regression_reports_a_singular_design_as_a_plain_value_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    se = np.full(20, 0.1)
    y = x + rng.normal(0.0, 0.1, 20)
    nearly = np.column_stack([x, x + 1e-9 * rng.normal(size=20)])
    exactly = np.column_stack([x, x])
    for features in (nearly, exactly):
        with pytest.raises(ValueError, match="singular design") as info:
            meta.meta_regression(y, se, features)
        assert type(info.value) is ValueError
    assert np.isfinite(meta.meta_regression(y, se, nearly, ridge=1.0)["loo_rmse"])
    correlated = np.column_stack([x, x + 1e-3 * rng.normal(size=20)])
    assert np.isfinite(meta.meta_regression(y, se, correlated)["loo_rmse"])
    many = rng.normal(size=(6, 9))
    with pytest.raises(ValueError, match="singular design") as info:
        meta.meta_regression(rng.normal(size=6), np.full(6, 0.1), many)
    assert type(info.value) is ValueError


def test_restricted_likelihood_reports_an_exactly_singular_matrix_as_a_plain_value_error():
    design = np.ones((5, 2))
    with pytest.raises(ValueError, match="singular design") as info:
        meta._neg2_restricted_loglik(0.1, np.arange(5.0), np.full(5, 0.1), design, np.zeros(2))
    assert type(info.value) is ValueError
    penalised = meta._neg2_restricted_loglik(
        0.1, np.arange(5.0), np.full(5, 0.1), design, np.ones(2)
    )
    assert np.isfinite(penalised)
    indefinite = np.array([-3.0, 0.0])
    with pytest.raises(ValueError, match="singular design") as info:
        meta._neg2_restricted_loglik(0.0, np.arange(2.0), np.ones(2), np.eye(2), indefinite)
    assert type(info.value) is ValueError


def test_meta_regression_reports_a_failing_matrix_inversion_as_a_plain_value_error(monkeypatch):
    y, se, X = make_regression_data(seed=56, k=14, slopes=(0.3,), tau=0.1)

    def failing_inverse(matrix):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "inv", failing_inverse)
    with pytest.raises(ValueError, match="singular design") as info:
        meta.meta_regression(y, se, X)
    assert type(info.value) is ValueError


def test_meta_regression_needs_at_least_two_cases():
    with pytest.raises(ValueError, match="two cases"):
        meta.meta_regression([0.3], [0.1], [[1.0]], ridge=1.0)
    with pytest.raises(ValueError, match="two cases"):
        meta.meta_regression([0.3], [0.1], [[1.0]])


def test_meta_regression_names_the_case_whose_removal_leaves_a_singular_design():
    rng = np.random.default_rng(1)
    k = 16
    a = rng.integers(0, 2, k).astype(float)
    b = a.copy()
    b[0] = 1.0 - a[0]
    se = rng.uniform(0.05, 0.2, k)
    y = 0.3 * a + rng.normal(0.0, 0.1, k) + rng.normal(0.0, se)
    with pytest.raises(ValueError, match=r"case 0 leaves a singular design") as info:
        meta.meta_regression(y, se, np.column_stack([a, b]))
    assert type(info.value) is ValueError
    assert np.isfinite(meta.meta_regression(y, se, np.column_stack([a, b]), ridge=0.5)["loo_rmse"])


# ----------------------------------------------------------------------------
# Shrinkage of the ridge slopes and extrapolation
# ----------------------------------------------------------------------------
SHRINKAGE_COLUMNS = ["ridge_slope", "unpenalised_slope", "ratio"]
RESULT_KEYS = {
    "coef",
    "coef_raw",
    "tau2",
    "prediction",
    "loo_pred",
    "loo_errors",
    "loo_var",
    "loo_rmse",
    "df",
}


def costly_target_data(seed, k=30):
    """Cases whose effect rises with the cost share, with a second, unrelated feature."""
    rng = np.random.default_rng(seed)
    cost = rng.uniform(0.3, 1.5, k)
    receipts = rng.uniform(2.0, 8.0, k)
    X = pd.DataFrame({"capex_pct_gdp": cost, "receipts_pct_gdp": receipts})
    se = rng.uniform(0.08, 0.25, k)
    y = 0.1 + 0.8 * cost + rng.normal(0.0, 0.1, k) + rng.normal(0.0, se)
    return y, se, X


def test_meta_regression_keeps_its_keys_and_adds_the_shrinkage_table():
    y, se, X = costly_target_data(1)
    out = meta.meta_regression(y, se, X, x_new={"capex_pct_gdp": 1.0, "receipts_pct_gdp": 5.0})
    assert set(out) == RESULT_KEYS | {"shrinkage"}
    assert isinstance(out["shrinkage"], pd.DataFrame)
    plain = meta.meta_regression(y, se, X, ridge=1.0)
    assert set(plain) == RESULT_KEYS | {"shrinkage"}
    assert plain["prediction"] is None


def test_meta_regression_reports_each_slope_with_and_without_the_penalty():
    y, se, X = costly_target_data(2)
    ridge = meta.meta_regression(y, se, X, ridge=1.0)
    plain = meta.meta_regression(y, se, X, ridge=0.0)
    table = ridge["shrinkage"]
    assert list(table.columns) == SHRINKAGE_COLUMNS
    assert list(table.index) == ["capex_pct_gdp", "receipts_pct_gdp"]
    features = ["capex_pct_gdp", "receipts_pct_gdp"]
    assert table["ridge_slope"].to_numpy() == pytest.approx(
        ridge["coef"].loc[features, "estimate"].to_numpy(), rel=1e-12
    )
    assert table["unpenalised_slope"].to_numpy() == pytest.approx(
        plain["coef"].loc[features, "estimate"].to_numpy(), rel=1e-12
    )
    assert table["ratio"].to_numpy() == pytest.approx(
        (table["ridge_slope"] / table["unpenalised_slope"]).to_numpy(), rel=1e-12
    )
    assert abs(table.loc["capex_pct_gdp", "ridge_slope"]) < abs(
        table.loc["capex_pct_gdp", "unpenalised_slope"]
    )
    assert 0.0 < table.loc["capex_pct_gdp", "ratio"] < 1.0
    for column in SHRINKAGE_COLUMNS:
        assert table[column].dtype == float


def test_meta_regression_shrinkage_is_the_identity_without_a_ridge():
    y, se, X = make_regression_data(seed=71, k=30, slopes=(0.4, -0.3, 0.2), tau=0.1)
    table = meta.meta_regression(y, se, X)["shrinkage"]
    assert list(table.index) == ["x0", "x1", "x2"]
    np.testing.assert_array_equal(table["ridge_slope"], table["unpenalised_slope"])
    assert table["ratio"].to_numpy() == pytest.approx(1.0, rel=1e-14)


def test_meta_regression_shrinkage_ratio_falls_as_the_ridge_grows():
    y, se, X = make_regression_data(seed=72, k=30, slopes=(0.5,), tau=0.1)
    ratios = [
        meta.meta_regression(y, se, X, ridge=ridge)["shrinkage"].loc["x0", "ratio"]
        for ridge in (0.1, 1.0, 10.0, 100.0, 1e6)
    ]
    assert all(0.0 < r < 1.0 for r in ratios)
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] < 1e-3


def test_meta_regression_shrinkage_does_not_change_the_numbers_of_the_fit():
    y, se, X = costly_target_data(3)
    target = {"capex_pct_gdp": 2.0, "receipts_pct_gdp": 5.0}
    out = meta.meta_regression(y, se, X, x_new=target, ridge=1.0)
    names = ["capex_pct_gdp", "receipts_pct_gdp"]
    Z = standardise(X.to_numpy())
    w = 1.0 / (se**2 + out["tau2"])
    design = np.column_stack([np.ones(30), Z])
    info = design.T @ (design * w[:, None]) + np.diag([0.0, 1.0, 1.0]) * np.mean(1.0 / se**2)
    beta = np.linalg.solve(info, design.T @ (w * y))
    assert out["coef"]["estimate"].to_numpy() == pytest.approx(beta, rel=1e-9)
    assert out["shrinkage"].loc[names, "ridge_slope"].to_numpy() == pytest.approx(
        beta[1:], rel=1e-9
    )
    again = meta.meta_regression(y, se, X, x_new=target, ridge=1.0)
    for key in ("coef", "coef_raw"):
        pd.testing.assert_frame_equal(out[key], again[key])
    assert out["prediction"] == again["prediction"]
    np.testing.assert_array_equal(out["loo_pred"], again["loo_pred"])


@pytest.mark.parametrize("seed", range(4))
def test_a_ridge_pulls_down_the_prediction_for_a_costly_target_and_more_so_further_out(seed):
    y, se, X = costly_target_data(seed)
    gaps = []
    for cost in (1.0, 2.0, 3.0):
        target = {"capex_pct_gdp": cost, "receipts_pct_gdp": 5.0}
        shrunk = meta.meta_regression(y, se, X, x_new=target, ridge=1.0)["prediction"]["mean"]
        free = meta.meta_regression(y, se, X, x_new=target, ridge=0.0)["prediction"]["mean"]
        gaps.append(shrunk - free)
    assert gaps[2] < gaps[1] < gaps[0] + 1e-9
    assert gaps[2] < -0.02
    table = meta.meta_regression(y, se, X, ridge=1.0)["shrinkage"]
    assert table.loc["capex_pct_gdp", "ratio"] < 1.0


def test_meta_regression_shrinkage_is_missing_where_the_unpenalised_model_is_singular():
    y, se, X = make_regression_data(seed=73, k=20, slopes=(0.4,), tau=0.1)
    collinear = np.column_stack([X[:, 0], 2.0 * X[:, 0] + 1.0])
    table = meta.meta_regression(y, se, collinear, ridge=1.0)["shrinkage"]
    assert table["ridge_slope"].notna().all()
    assert table["unpenalised_slope"].isna().all() and table["ratio"].isna().all()
    assert list(table.index) == ["x0", "x1"]
    rng = np.random.default_rng(74)
    wide = rng.normal(size=(8, 12))
    narrow = meta.meta_regression(
        0.5 * wide[:, 0] + rng.normal(0.0, 0.1, 8), np.full(8, 0.1), wide, ridge=1.0
    )["shrinkage"]
    assert narrow["ridge_slope"].notna().all() and narrow["unpenalised_slope"].isna().all()
    square = rng.normal(size=(5, 4))
    exactly = meta.meta_regression(rng.normal(size=5), np.full(5, 0.1), square, ridge=1.0)
    assert exactly["shrinkage"]["unpenalised_slope"].isna().all()
    assert exactly["shrinkage"]["ratio"].isna().all()


def test_meta_regression_shrinkage_ratio_is_missing_where_the_unpenalised_slope_is_zero(
    monkeypatch,
):
    y, se, X = make_regression_data(seed=75, k=20, slopes=(0.4, 0.2), tau=0.1)
    real = meta._fit_mixed

    def zero_first_slope(y_, v_, X_, ridge_):
        fit = real(y_, v_, X_, ridge_)
        if ridge_ == 0.0:
            theta = fit.theta.copy()
            theta[1] = 0.0
            return dataclasses.replace(fit, theta=theta)
        return fit

    monkeypatch.setattr(meta, "_fit_mixed", zero_first_slope)
    table = meta.meta_regression(y, se, X, ridge=1.0)["shrinkage"]
    assert table.loc["x0", "unpenalised_slope"] == 0.0 and np.isnan(table.loc["x0", "ratio"])
    assert np.isfinite(table.loc["x1", "ratio"])


def noise_free_effects(k=30, level=1e6):
    """Effects exactly linear in the cost share at a large level, without a receipts effect."""
    rng = np.random.default_rng(81)
    X = pd.DataFrame({"cost": rng.uniform(0.3, 1.5, k), "receipts": rng.uniform(2.0, 8.0, k)})
    se = rng.uniform(0.08, 0.25, k)
    return level + 0.8 * X["cost"].to_numpy(), se, X


@pytest.mark.parametrize("ridge", [0.0, 1.0])
def test_meta_regression_shrinkage_ratio_is_missing_for_a_slope_that_is_rounding_noise(ridge):
    """A feature without an effect gets a slope of rounding noise, which has no ratio."""
    y, se, X = noise_free_effects()
    table = meta.meta_regression(y, se, X, ridge=ridge)["shrinkage"]
    floor = 1e-12 * np.max(np.abs(y))
    assert abs(table.loc["receipts", "unpenalised_slope"]) < floor
    assert np.isnan(table.loc["receipts", "ratio"])
    assert np.isfinite(table.loc["receipts", "ridge_slope"])
    assert abs(table.loc["cost", "unpenalised_slope"]) > 1e3 * floor
    assert np.isfinite(table.loc["cost", "ratio"])
    if ridge == 0.0:
        assert table.loc["cost", "ratio"] == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_array_equal(table["ridge_slope"], table["unpenalised_slope"])
    else:
        assert 0.0 < table.loc["cost", "ratio"] < 1.0


def test_meta_regression_shrinkage_ratio_is_missing_for_every_slope_of_a_constant_effect():
    rng = np.random.default_rng(82)
    X = pd.DataFrame({"a": rng.normal(size=25), "b": rng.uniform(1.0, 4.0, 25)})
    se = rng.uniform(0.1, 0.3, 25)
    for level in (1e6, 5.0, 1e-3):
        for ridge in (0.0, 1.0):
            table = meta.meta_regression(np.full(25, level), se, X, ridge=ridge)["shrinkage"]
            assert table["ratio"].isna().all(), (level, ridge)
            assert table["unpenalised_slope"].abs().max() < 1e-12 * level, (level, ridge)


def test_meta_regression_shrinkage_floor_is_a_multiple_of_the_largest_effect(monkeypatch):
    y, se, X = make_regression_data(seed=75, k=20, slopes=(0.4, 0.2), tau=0.1)
    floor = 1e-12 * float(np.max(np.abs(y)))
    real = meta._fit_mixed

    def with_first_slope(value):
        def patched(y_, v_, X_, ridge_):
            fit = real(y_, v_, X_, ridge_)
            if ridge_ == 0.0:
                theta = fit.theta.copy()
                theta[1] = value
                return dataclasses.replace(fit, theta=theta)
            return fit

        return patched

    cases = ((0.5 * floor, False), (-0.5 * floor, False), (2.0 * floor, True), (-2.0 * floor, True))
    for value, has_ratio in cases:
        monkeypatch.setattr(meta, "_fit_mixed", with_first_slope(value))
        table = meta.meta_regression(y, se, X, ridge=1.0)["shrinkage"]
        assert table.loc["x0", "unpenalised_slope"] == value
        assert bool(np.isfinite(table.loc["x0", "ratio"])) is has_ratio, value
        assert np.isfinite(table.loc["x1", "ratio"])


@pytest.mark.parametrize("factor", [1e-15, 1e-9, 1e-3, 1e3, 1e9])
def test_meta_regression_shrinkage_ratio_does_not_depend_on_the_unit_of_the_effects(factor):
    y, se, X = costly_target_data(2)
    reference = meta.meta_regression(y, se, X, ridge=1.0)["shrinkage"]
    scaled = meta.meta_regression(factor * y, factor * se, X, ridge=1.0)["shrinkage"]
    assert scaled["ratio"].notna().all() and scaled["unpenalised_slope"].notna().all()
    np.testing.assert_allclose(scaled["ratio"], reference["ratio"], rtol=1e-5)
    np.testing.assert_allclose(
        scaled["unpenalised_slope"], factor * reference["unpenalised_slope"], rtol=1e-5
    )


def test_meta_regression_does_not_claim_a_ratio_of_one_for_slopes_that_are_noise():
    doc = " ".join(meta.meta_regression.__doc__.split())
    assert "The ratio is 1 for every feature when ``ridge`` is zero." not in doc
    assert "is rounding noise" in doc and "1e-12`` times the largest absolute effect" in doc
    assert "ratio is 1 for every feature whose slope is not rounding noise and NaN" in doc
    table_doc = " ".join(meta._shrinkage_table.__doc__.split())
    assert "``1e-12`` times the largest absolute effect" in table_doc


def test_meta_regression_shrinkage_covers_the_fitted_features_only():
    y, se, X = make_regression_data(seed=76, k=24, slopes=(0.3,), tau=0.1)
    X3 = pd.DataFrame({"a": X[:, 0], "constant": 3.0, "b": np.sin(np.arange(24.0))})
    out = meta.meta_regression(y, se, X3, ridge=1.0)
    assert list(out["shrinkage"].index) == ["a", "b"]
    assert out["coef"].loc["constant", "estimate"] == 0.0
    unlabelled = meta.meta_regression(y, se, X3.to_numpy(), ridge=1.0)
    assert list(unlabelled["shrinkage"].index) == ["x0", "x2"]
    np.testing.assert_allclose(
        unlabelled["shrinkage"].to_numpy(), out["shrinkage"].to_numpy(), rtol=1e-12
    )


def test_meta_regression_documents_the_shrinkage_and_the_extrapolation():
    doc = " ".join(meta.meta_regression.__doc__.split())
    assert "shrinkage" in doc and "ridge_slope" in doc and "unpenalised_slope" in doc
    assert "outside the range of the cases is an extrapolation of the fitted line" in doc
    assert "extrapolation" in meta.__doc__


# ---- extrapolation -----------------------------------------------------------
@pytest.fixture
def case_features():
    return pd.DataFrame(
        {
            "capex_pct_gdp": [0.4, 0.9, 1.5, 0.7],
            "receipts_pct_gdp": [2.0, 8.0, 5.0, 6.5],
            "balance": [-1.0, 0.5, 2.0, 0.0],
        }
    )


def test_extrapolation_places_a_target_against_the_range_of_the_cases(case_features):
    target = {"capex_pct_gdp": 2.28, "receipts_pct_gdp": 7.81, "balance": -3.0}
    out = meta.extrapolation(case_features, target)
    assert list(out.index) == ["capex_pct_gdp", "receipts_pct_gdp", "balance"]
    assert list(out.columns) == ["x_new", "case_min", "case_max", "position", "ratio_to_max"]
    assert out["position"].tolist() == ["above", "inside", "below"]
    assert out["x_new"].tolist() == [2.28, 7.81, -3.0]
    assert out["case_min"].tolist() == [0.4, 2.0, -1.0]
    assert out["case_max"].tolist() == [1.5, 8.0, 2.0]
    assert out.loc["capex_pct_gdp", "ratio_to_max"] == pytest.approx(2.28 / 1.5, rel=1e-14)
    assert out.loc["receipts_pct_gdp", "ratio_to_max"] == pytest.approx(7.81 / 8.0, rel=1e-14)
    assert np.isnan(out.loc["balance", "ratio_to_max"])


def test_extrapolation_treats_the_ends_of_the_range_as_inside(case_features):
    ends = {"capex_pct_gdp": 0.4, "receipts_pct_gdp": 8.0, "balance": 2.0}
    assert set(meta.extrapolation(case_features, ends)["position"]) == {"inside"}
    noise = {"capex_pct_gdp": 0.4 - 1e-12, "receipts_pct_gdp": 8.0 + 1e-12, "balance": -1.0}
    assert set(meta.extrapolation(case_features, noise)["position"]) == {"inside"}
    just_out = {"capex_pct_gdp": 0.4 - 1e-6, "receipts_pct_gdp": 8.0 + 1e-6, "balance": 2.000001}
    assert meta.extrapolation(case_features, just_out)["position"].tolist() == [
        "below",
        "above",
        "above",
    ]


@pytest.mark.parametrize("factor", [1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e9])
def test_extrapolation_position_does_not_depend_on_the_unit_of_a_feature(factor):
    """A target 1e-12 beyond an end of the range is inside it and one 1e-6 beyond is outside."""
    base = pd.DataFrame(
        {
            "share": [0.4, 0.9, 1.5, 0.7],
            "balance": [-1.0, 0.5, 2.0, 0.0],
            "year": [2000.0, 2010.0, 2019.0, 2005.0],
        }
    )
    cases = base * factor
    scale = np.maximum(cases.min().abs(), cases.max().abs())
    expectations = ((1e-12, "inside", "inside"), (1e-6, "above", "below"))
    for offset, expected_up, expected_down in expectations:
        up = (cases.max() + offset * scale).to_dict()
        down = (cases.min() - offset * scale).to_dict()
        assert meta.extrapolation(cases, up)["position"].tolist() == [expected_up] * 3, offset
        assert meta.extrapolation(cases, down)["position"].tolist() == [expected_down] * 3, offset
    inside = (0.5 * (cases.min() + cases.max())).to_dict()
    assert meta.extrapolation(cases, inside)["position"].tolist() == ["inside"] * 3


def test_extrapolation_tolerance_is_relative_for_features_of_magnitude_below_one():
    X = pd.DataFrame({"tiny": [1e-6, 2e-6, 3e-6]})
    near = meta.extrapolation(X, {"tiny": 3e-6 * (1.0 + 1e-10)})
    assert near["position"].tolist() == ["inside"]
    out = meta.extrapolation(X, {"tiny": 3e-6 * (1.0 + 1e-6)})
    assert out["position"].tolist() == ["above"]
    below = meta.extrapolation(X, {"tiny": 1e-6 * (1.0 - 1e-6)})
    assert below["position"].tolist() == ["below"]
    assert meta.extrapolation(X, {"tiny": 3e-6 * (1.0 + 5e-10)})["position"].tolist() == ["inside"]
    assert meta.extrapolation(X, {"tiny": 3e-6 * (1.0 + 5e-9)})["position"].tolist() == ["above"]


def test_extrapolation_has_no_tolerance_for_a_feature_that_is_zero_in_every_case():
    X = pd.DataFrame({"zeros": [0.0, 0.0, 0.0], "other": [1.0, 2.0, 3.0]})
    expectations = ((0.0, "inside"), (1e-300, "above"), (-1e-300, "below"), (1e-12, "above"))
    for value, expected in expectations:
        out = meta.extrapolation(X, {"zeros": value, "other": 2.0})
        assert out.loc["zeros", "position"] == expected, value
        assert out.loc["other", "position"] == "inside"


def test_extrapolation_documents_its_tolerance_as_scale_free():
    doc = " ".join(meta.extrapolation.__doc__.split())
    assert "times the larger of the absolute smallest and largest case value" in doc
    assert "multiplied by a positive constant" in doc
    assert "max(1," not in doc


def test_extrapolation_ratio_is_defined_for_non_negative_features_with_a_positive_maximum():
    X = pd.DataFrame(
        {"zero_min": [0.0, 1.0, 4.0], "negative": [-1.0, 1.0, 4.0], "zeros": [0.0, 0.0, 0.0]}
    )
    out = meta.extrapolation(X, {"zero_min": 8.0, "negative": 8.0, "zeros": 1.0})
    assert out.loc["zero_min", "ratio_to_max"] == pytest.approx(2.0)
    assert np.isnan(out.loc["negative", "ratio_to_max"])
    assert np.isnan(out.loc["zeros", "ratio_to_max"])
    assert out["position"].tolist() == ["above", "above", "above"]


def test_extrapolation_matches_the_target_by_name_and_accepts_arrays(case_features):
    mapping = {"balance": 0.0, "capex_pct_gdp": 1.0, "receipts_pct_gdp": 5.0, "extra": 99.0}
    reference = meta.extrapolation(case_features, mapping)
    assert reference["position"].tolist() == ["inside"] * 3
    as_series = meta.extrapolation(case_features, pd.Series(mapping))
    pd.testing.assert_frame_equal(as_series, reference)
    as_frame = meta.extrapolation(case_features, pd.DataFrame([mapping]))
    pd.testing.assert_frame_equal(as_frame, reference)
    by_position = meta.extrapolation(case_features, [1.0, 5.0, 0.0])
    pd.testing.assert_frame_equal(by_position, reference)
    arrays = meta.extrapolation(case_features.to_numpy(), [1.0, 5.0, 0.0])
    assert list(arrays.index) == ["x0", "x1", "x2"]
    numeric = ["x_new", "case_min", "case_max", "ratio_to_max"]
    np.testing.assert_allclose(arrays[numeric].to_numpy(), reference[numeric].to_numpy())
    assert arrays["position"].tolist() == reference["position"].tolist()
    single = meta.extrapolation(case_features["capex_pct_gdp"], 2.0)
    assert list(single.index) == ["capex_pct_gdp"] and single["position"].tolist() == ["above"]
    column = meta.extrapolation(np.array([1.0, 2.0, 3.0]), [4.0])
    assert column["position"].tolist() == ["above"]
    assert column.loc["x0", "ratio_to_max"] == pytest.approx(4.0 / 3.0)


def test_extrapolation_agrees_with_the_range_check_of_a_fitted_model():
    y, se, X = costly_target_data(5)
    for cost, expected in ((0.1, "below"), (1.0, "inside"), (2.5, "above")):
        out = meta.extrapolation(X, {"capex_pct_gdp": cost, "receipts_pct_gdp": 5.0})
        assert out.loc["capex_pct_gdp", "position"] == expected
        assert out.loc["receipts_pct_gdp", "position"] == "inside"
    out = meta.extrapolation(X, {"capex_pct_gdp": 2.5, "receipts_pct_gdp": 5.0})
    assert out.loc["capex_pct_gdp", "ratio_to_max"] == pytest.approx(
        2.5 / X["capex_pct_gdp"].max(), rel=1e-14
    )


def test_extrapolation_rejects_invalid_inputs(case_features):
    target = {"capex_pct_gdp": 1.0, "receipts_pct_gdp": 5.0, "balance": 0.0}
    with pytest.raises(ValueError, match="exactly one"):
        meta.extrapolation(case_features, pd.DataFrame([target, target]))
    with pytest.raises(ValueError, match="exactly one"):
        meta.extrapolation(case_features, [[1.0, 5.0, 0.0], [2.0, 5.0, 0.0]])
    with pytest.raises(ValueError, match="balance"):
        meta.extrapolation(case_features, {"capex_pct_gdp": 1.0, "receipts_pct_gdp": 5.0})
    with pytest.raises(ValueError):
        meta.extrapolation(case_features, [1.0, 5.0])
    with pytest.raises(ValueError):
        meta.extrapolation(case_features, dict(target, balance=np.nan))
    with pytest.raises(ValueError):
        meta.extrapolation(case_features, dict(target, balance=np.inf))
    broken = case_features.copy()
    broken.loc[1, "balance"] = np.nan
    with pytest.raises(ValueError):
        meta.extrapolation(broken, target)
    with pytest.raises(ValueError):
        meta.extrapolation(np.float64(3.0), [1.0])
    with pytest.raises(ValueError):
        meta.extrapolation(np.zeros((2, 2, 2)), [1.0])
    with pytest.raises(ValueError):
        meta.extrapolation(np.zeros((3, 0)), [])
    assert "extrapolation" in meta.__all__


# ----------------------------------------------------------------------------
# Scaling law
# ----------------------------------------------------------------------------
def test_scaling_law_recovers_an_exactly_proportional_effect():
    size = np.array([0.1, 0.2, 0.35, 0.5, 0.8, 1.2])
    out = meta.scaling_law(1.7 * size, np.full(6, 0.05), size, 0.6)
    assert out["beta"] == pytest.approx(1.7, rel=1e-10)
    assert out["tau2"] == 0.0
    assert out["loo_rmse"] == pytest.approx(0.0, abs=1e-10)
    assert out["prediction"]["mean"] == pytest.approx(1.7 * 0.6, rel=1e-10)
    assert out["prediction"]["lo"] < out["prediction"]["mean"] < out["prediction"]["hi"]
    assert out["se_beta"] == pytest.approx(0.05 / np.sqrt(np.sum(size**2)), rel=1e-10)


def test_scaling_law_weighted_slope_and_prediction_follow_the_formulas():
    rng = np.random.default_rng(61)
    size = rng.uniform(0.05, 0.6, 14)
    se = rng.uniform(0.02, 0.12, 14)
    y = 2.0 * size + rng.normal(0.0, 0.06, 14) + rng.normal(0.0, se)
    out = meta.scaling_law(y, se, size, 0.3, level=0.8)
    assert out["tau2"] > 0.0
    w = 1.0 / (se**2 + out["tau2"])
    beta = np.sum(w * size * y) / np.sum(w * size**2)
    var_beta = 1.0 / np.sum(w * size**2)
    assert out["beta"] == pytest.approx(beta, rel=1e-9)
    assert out["se_beta"] == pytest.approx(np.sqrt(var_beta), rel=1e-9)
    se_pred = np.sqrt(out["tau2"] + 0.3**2 * var_beta)
    t = stats.t.ppf(0.9, 12)
    pred = out["prediction"]
    assert pred["mean"] == pytest.approx(beta * 0.3, rel=1e-9)
    assert pred["se_pred"] == pytest.approx(se_pred, rel=1e-9)
    assert pred["hi"] == pytest.approx(beta * 0.3 + t * se_pred, rel=1e-9)
    assert pred["lo"] == pytest.approx(beta * 0.3 - t * se_pred, rel=1e-9)
    assert out["df"] == 12


def test_scaling_law_prediction_interval_needs_a_positive_number_of_degrees_of_freedom():
    y = np.array([0.3, 0.9, 0.5])
    se = np.array([0.1, 0.2, 0.1])
    size = np.array([0.2, 0.5, 0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        three = meta.scaling_law(y, se, size, 0.4, level=0.8)
    assert three["df"] == 1
    pred = three["prediction"]
    assert pred["hi"] - pred["mean"] == pytest.approx(stats.t.ppf(0.9, 1) * pred["se_pred"])
    assert pred["mean"] - pred["lo"] == pytest.approx(stats.t.ppf(0.9, 1) * pred["se_pred"])
    with pytest.warns(UserWarning, match="degrees of freedom"):
        two = meta.scaling_law(y[:2], se[:2], size[:2], 0.4)
    assert two["df"] == 0
    assert np.isfinite(two["prediction"]["mean"]) and np.isfinite(two["prediction"]["se_pred"])
    assert np.isnan(two["prediction"]["lo"]) and np.isnan(two["prediction"]["hi"])
    with pytest.warns(RuntimeWarning):
        several = meta.scaling_law(y[:2], se[:2], size[:2], [0.3, 0.4, 0.5])
    assert several["prediction"]["lo"].shape == (3,)
    assert np.all(np.isnan(several["prediction"]["hi"]))


def test_scaling_law_aligns_a_new_size_by_name():
    rng = np.random.default_rng(64)
    size = pd.Series(rng.uniform(0.05, 0.6, 12), name="investment")
    se = rng.uniform(0.02, 0.1, 12)
    y = 2.0 * size.to_numpy() + rng.normal(0.0, 0.05, 12) + rng.normal(0.0, se)
    base = meta.scaling_law(y, se, size, 0.35)["prediction"]
    inputs = [
        {"other": 9.0, "investment": 0.35},
        pd.Series({"gdp": 9.0, "investment": 0.35}),
        pd.DataFrame({"gdp": [9.0], "investment": [0.35]}),
    ]
    for size_new in inputs:
        out = meta.scaling_law(y, se, size, size_new)["prediction"]
        for key in ("mean", "lo", "hi", "se_pred"):
            assert np.ravel(out[key])[0] == pytest.approx(base[key], rel=1e-12)
    for size_new in ({"other": 0.35}, pd.Series({"gdp": 0.35}), pd.DataFrame({"gdp": [0.35]})):
        with pytest.raises(ValueError, match="investment"):
            meta.scaling_law(y, se, size, size_new)
    unnamed = meta.scaling_law(y, se, size.to_numpy(), {"size": 0.35})["prediction"]
    assert unnamed["mean"] == pytest.approx(base["mean"], rel=1e-12)
    with pytest.raises(ValueError, match="size"):
        meta.scaling_law(y, se, size.to_numpy(), {"investment": 0.35})
    plain = meta.scaling_law(y, se, size, [0.35, 0.1])["prediction"]
    assert plain["mean"][0] == pytest.approx(base["mean"], rel=1e-12)
    frame = meta.scaling_law(y, se, size, pd.DataFrame({"investment": [0.35, 0.1]}))["prediction"]
    assert np.allclose(frame["mean"], plain["mean"], rtol=1e-12)


def test_scaling_law_tau2_maximises_the_restricted_likelihood():
    rng = np.random.default_rng(62)
    size = rng.uniform(0.05, 0.6, 16)
    se = rng.uniform(0.02, 0.12, 16)
    y = 2.0 * size + rng.normal(0.0, 0.08, 16) + rng.normal(0.0, se)
    ref = contrast_reml_tau2(y, se**2, size[:, None])
    assert meta.scaling_law(y, se, size, 0.3)["tau2"] == pytest.approx(ref, rel=1e-4, abs=1e-6)


def test_scaling_law_predictions_are_proportional_and_loo_refits_the_slope():
    rng = np.random.default_rng(63)
    size = rng.uniform(0.05, 0.6, 12)
    se = rng.uniform(0.02, 0.1, 12)
    y = 2.0 * size + rng.normal(0.0, 0.05, 12) + rng.normal(0.0, se)
    out = meta.scaling_law(y, se, size, [0.1, 0.2, 0.4])
    means = out["prediction"]["mean"]
    assert means.shape == (3,)
    assert means[1] == pytest.approx(2.0 * means[0], rel=1e-12)
    assert means[2] == pytest.approx(4.0 * means[0], rel=1e-12)
    assert np.all(out["prediction"]["lo"] < means) and np.all(means < out["prediction"]["hi"])
    for i in (0, 5, 11):
        rows = np.arange(12) != i
        refit = meta.scaling_law(y[rows], se[rows], size[rows], size[i])
        assert out["loo_pred"][i] == pytest.approx(refit["prediction"]["mean"], rel=1e-9)
        assert out["loo_var"][i] == pytest.approx(se[i] ** 2 + refit["tau2"], rel=1e-9)
    assert np.allclose(out["loo_errors"], y - out["loo_pred"])
    assert np.isfinite(out["loo_rmse"]) and out["loo_rmse"] > 0.0
    assert out["loo_rmse"] == pytest.approx(np.sqrt(np.mean(out["loo_errors"] ** 2)), rel=1e-12)


def test_scaling_law_input_checks():
    size = np.array([0.1, 0.2, 0.3])
    y = np.array([0.2, 0.4, 0.6])
    se = np.full(3, 0.1)
    with pytest.raises(ValueError):
        meta.scaling_law(y, se, size[:2], 0.2)
    with pytest.raises(ValueError):
        meta.scaling_law(y, se, np.zeros(3), 0.2)
    with pytest.raises(ValueError):
        meta.scaling_law(y[:1], se[:1], size[:1], 0.2)
    with pytest.raises(ValueError):
        meta.scaling_law(y, se, size, np.nan)
    with pytest.raises(ValueError):
        meta.scaling_law(y, se, size, 0.2, level=2.0)
    assert isinstance(meta.scaling_law(y, se, size, 0.2)["prediction"]["mean"], float)


# ----------------------------------------------------------------------------
# Source text
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("path", [Path(meta.__file__), Path(__file__)])
def test_no_line_of_the_module_or_of_this_file_is_longer_than_one_hundred_characters(path):
    text = path.read_text(encoding="utf-8")
    too_long = [
        (number, len(line)) for number, line in enumerate(text.splitlines(), 1) if len(line) > 100
    ]
    assert too_long == [], path.name
    assert "\r" not in text and chr(0x2014) not in text
