"""Tests for dtt.transport: standardisation, weighted cost, Sinkhorn plans, transported effects, distances,
overlap test, leave-one-case-out and leave-one-group-out validation, predictive draws and bootstrap.

All data are SIMULATED inside the tests.  POT (``ot``) and cvxpy are used as independent references and the
tests that need them are skipped when the package is missing.  The solver is also checked against the
first-order optimality conditions of the objective stated in its documentation and against an exact linear program.
"""
import inspect
import itertools
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import null_space
from scipy.spatial.distance import cdist, pdist

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dtt import transport as tr  # noqa: E402

PATTERNS = [(None, None), (1.0, None), (None, 1.0), (1.0, 2.0), (0.2, 0.3)]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def random_problem(seed, n=8, m=9, d=3, shift=0.4, uniform=False):
    """Random sources, target points, feature weights, masses and the cost matrix."""
    rng = np.random.default_rng(seed)
    Zs = rng.normal(size=(n, d))
    Zt = rng.normal(size=(m, d)) + shift
    w = np.array([2.0, 1.0, 0.5, 0.25, 0.1][:d])
    if uniform:
        a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    else:
        a, b = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, m)
        a, b = a / a.sum(), b / b.sum()
    return Zs, Zt, w, a, b, tr.weighted_sq_cost(Zs, Zt, w)


def kl(p, q):
    """Generalised Kullback-Leibler divergence ``sum p log(p / q) - p + q``."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / q), 0.0)
    return float((terms - p + q).sum())


def primal_objective(P, C, a, b, eps, rho_s=None, rho_t=None):
    """Objective of the documented transport problem; an enforced marginal has no penalty term."""
    value = float((C * P).sum()) + eps * kl(P, np.outer(a, b))
    if rho_s is not None:
        value += rho_s * kl(P.sum(axis=1), a)
    if rho_t is not None:
        value += rho_t * kl(P.sum(axis=0), b)
    return value


def kkt_residual(P, C, a, b, eps, rho_s=None, rho_t=None):
    """Largest component of the gradient of the objective along the feasible directions."""
    n, m = P.shape
    G = C + eps * np.log(P / np.outer(a, b))
    if rho_s is not None:
        G = G + rho_s * np.log(P.sum(axis=1) / a)[:, None]
    if rho_t is not None:
        G = G + rho_t * np.log(P.sum(axis=0) / b)[None, :]
    constraints = []
    if rho_s is None:
        for i in range(n):
            row = np.zeros((n, m))
            row[i, :] = 1.0
            constraints.append(row.ravel())
    if rho_t is None:
        for j in range(m):
            row = np.zeros((n, m))
            row[:, j] = 1.0
            constraints.append(row.ravel())
    directions = null_space(np.array(constraints)) if constraints else np.eye(n * m)
    return float(np.abs(directions.T @ G.ravel()).max())


def informative_problem(seed, n=30, d=4, noise=0.1):
    """Cases whose effect depends on feature 0 only (SIMULATED)."""
    rng = np.random.default_rng(seed)
    Z = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{k}" for k in range(d)], index=[f"case{i}" for i in range(n)])
    se = np.full(n, noise)
    tau = 2.0 * Z["f0"].to_numpy() + se * rng.normal(size=n)
    return Z, tau, se


def economy_problem(seed, n_groups=12, per=2, d=4):
    """Cases grouped in economies (SIMULATED): the cases of an economy share one feature vector and one effect, and the effects are unrelated to the features."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_groups, d))
    effect = rng.normal(size=n_groups)
    index = [f"case{i}" for i in range(n_groups * per)]
    Z = pd.DataFrame(np.repeat(base, per, axis=0), columns=[f"f{k}" for k in range(d)], index=index)
    groups = np.repeat([f"E{g:02d}" for g in range(n_groups)], per)
    return Z, np.repeat(effect, per), np.full(n_groups * per, 0.1), groups


def learned_weights(Z, tau, k=2):
    """Feature weights learned from training cases: 1 for the k features most correlated with the effect, else 1e-6."""
    corr = np.abs([np.corrcoef(Z[:, j], tau)[0, 1] for j in range(Z.shape[1])])
    weights = np.full(Z.shape[1], 1e-6)
    weights[np.argsort(corr)[-k:]] = 1.0
    return weights


def reference_predictions(Z, tau, src, i, w, eps_w, eps_u, a=None, rho=1.0, target=None):
    """Predictions of the six methods for case ``i`` from the sources ``src``, written from their definitions."""
    Z = np.asarray(Z, dtype=float)
    d = Z.shape[1]
    target = Z[[i]] if target is None else np.asarray(target, dtype=float)
    mass = np.ones(src.size) if a is None else np.asarray(a, dtype=float)[src]
    mass = mass / mass.sum()
    rescaled = np.asarray(w, dtype=float) * d / np.sum(w)
    C = tr.weighted_sq_cost(Z[src], target, rescaled)
    C_uniform = tr.weighted_sq_cost(Z[src], target, np.ones(d))
    nearest = np.argsort(C, axis=0, kind="stable")
    pair = tr.weighted_sq_cost(Z[src], Z[src], rescaled)[np.triu_indices(src.size, k=1)]
    kernel = np.exp(-C / (2.0 * np.median(np.sqrt(pair)) ** 2))
    b = np.full(target.shape[0], 1.0 / target.shape[0])
    effects = tau[src]
    plan = tr.sinkhorn_plan(mass, b, C, eps_w, rho_source=rho)
    plan_uniform = tr.sinkhorn_plan(mass, b, C_uniform, eps_u, rho_source=rho)
    return {
        "ot_weighted": tr.target_effect(plan, effects, b),
        "ot_uniform": tr.target_effect(plan_uniform, effects, b),
        "equal": effects.mean(),
        "nn1": effects[nearest[0]].mean(),
        "nn3": effects[nearest[:3]].mean(axis=0).mean(),
        "kernel": ((kernel * effects[:, None]).sum(axis=0) / kernel.sum(axis=0)).mean(),
    }


def logo_table(seed=50, n_groups=10, per=2, method="ot_weighted"):
    """Rows of one method of a leave-one-group-out table, and the group labels (SIMULATED economies)."""
    Z, tau, se, groups = economy_problem(seed, n_groups=n_groups, per=per)
    res = tr.loco_validation(Z, tau, se, np.array([1.0, 0.5, 0.3, 0.2]), groups=groups, eps=0.2, n_boot=5)
    return res.table[res.table["method"] == method].reset_index(drop=True), groups


def flat(text):
    """The text with every run of whitespace replaced by one space."""
    return " ".join(text.split())


def spy_on_solve(monkeypatch):
    """Record the cost matrix of every transport problem the module solves and return the list that collects them."""
    recorded = []
    original = tr._solve

    def spy(a, b, C, *args, **kwargs):
        recorded.append(np.array(C))
        return original(a, b, C, *args, **kwargs)

    monkeypatch.setattr(tr, "_solve", spy)
    return recorded


# ----------------------------------------------------------------------------
# robust_standardise
# ----------------------------------------------------------------------------
def test_robust_standardise_uses_median_and_scaled_mad_of_the_reference():
    rng = np.random.default_rng(0)
    scales = np.array([1.0, 10.0, 100.0])
    ref = pd.DataFrame(rng.lognormal(size=(200, 3)) * scales, columns=list("xyz"))
    Xs = pd.DataFrame(rng.lognormal(size=(12, 3)) * scales, columns=list("xyz"), index=[f"s{i}" for i in range(12)])
    Xt = pd.DataFrame(rng.lognormal(size=(5, 3)) * scales, columns=list("xyz"), index=[f"t{i}" for i in range(5)])
    Zs, Zt, scale = tr.robust_standardise(Xs, Xt, reference=ref, clip=None)
    centre = ref.median()
    mad = 1.4826 * (ref - centre).abs().median()
    pd.testing.assert_frame_equal(Zs, (Xs - centre) / mad)
    pd.testing.assert_frame_equal(Zt, (Xt - centre) / mad)
    pd.testing.assert_series_equal(scale, mad, check_names=False)
    Zs0, Zt0, scale0 = tr.robust_standardise(Xs, Xt, clip=None)
    centre0 = Xs.median()
    mad0 = 1.4826 * (Xs - centre0).abs().median()
    pd.testing.assert_frame_equal(Zs0, (Xs - centre0) / mad0)
    pd.testing.assert_frame_equal(Zt0, (Xt - centre0) / mad0)
    pd.testing.assert_series_equal(scale0, mad0, check_names=False)
    Zr, _, _ = tr.robust_standardise(ref, Xt, reference=ref, clip=None)
    assert Zr.median().abs().max() < 1e-12
    assert np.allclose(1.4826 * (Zr - Zr.median()).abs().median(), 1.0)


def test_robust_standardise_zero_scale_column_selection_and_labels():
    Xs = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 100.0], "const": [5.0] * 5, "half": [1.0, 1.0, 1.0, 2.0, 3.0]},
        index=list("pqrst"),
    )
    Xt = pd.DataFrame({"a": [2.0, 3.0], "const": [7.0, 5.0], "half": [4.0, 1.0], "extra": [0.0, 0.0]}, index=["u", "v"])
    Zs, Zt, scale = tr.robust_standardise(Xs, Xt, columns=["half", "a", "const"])
    assert list(Zs.columns) == list(Zt.columns) == ["half", "a", "const"]
    assert list(Zs.index) == list("pqrst") and list(Zt.index) == ["u", "v"]
    assert scale["const"] == 1.0 and scale["half"] == 1.0
    assert Zs["half"].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert Zt["const"].tolist() == [2.0, 0.0]
    assert scale["a"] == pytest.approx(1.4826 * 1.0)
    assert list(scale.index) == ["half", "a", "const"]
    with pytest.raises(KeyError):
        tr.robust_standardise(Xs, Xt, columns=["a", "missing"])
    with pytest.raises(KeyError):
        tr.robust_standardise(Xs, Xt.drop(columns="a"))


def test_robust_standardise_uses_the_scale_of_the_sources_when_the_reference_scale_is_missing_or_zero():
    rng = np.random.default_rng(6)
    Xs = pd.DataFrame(
        {"u": rng.normal(10.0, 3.0, 15), "v": rng.normal(-2.0, 0.5, 15), "w": rng.normal(0.0, 2.0, 15), "x": rng.normal(1.0, 1.0, 15)}
    )
    Xt = pd.DataFrame({"u": [9.0, 12.0], "v": [-2.5, -1.0], "w": [1.0, -3.0], "x": [0.5, 2.0]})
    reference = pd.DataFrame(
        {"u": rng.normal(10.0, 3.0, 40), "v": np.full(40, -2.0), "w": np.full(40, np.nan), "x": rng.normal(1.0, 1.0, 40)}
    )
    Zs, Zt, scale = tr.robust_standardise(Xs, Xt, reference=reference, clip=None)
    pooled_centre = Xs.median()
    pooled_scale = 1.4826 * (Xs - pooled_centre).abs().median()
    reference_centre = reference.median()
    reference_scale = 1.4826 * (reference - reference_centre).abs().median()
    for column in ("u", "x"):
        assert scale[column] == pytest.approx(reference_scale[column])
    assert reference_scale["v"] == 0.0 and np.isnan(reference_scale["w"])
    assert scale["v"] == pytest.approx(pooled_scale["v"]) and scale["v"] != 1.0
    pd.testing.assert_series_equal(Zs["v"], (Xs["v"] + 2.0) / pooled_scale["v"], check_names=False)
    pd.testing.assert_series_equal(Zt["v"], (Xt["v"] + 2.0) / pooled_scale["v"], check_names=False)
    assert scale["w"] == pytest.approx(pooled_scale["w"])
    pd.testing.assert_series_equal(Zs["w"], (Xs["w"] - pooled_centre["w"]) / pooled_scale["w"], check_names=False)
    pd.testing.assert_series_equal(Zt["w"], (Xt["w"] - pooled_centre["w"]) / pooled_scale["w"], check_names=False)
    assert np.isfinite(Zs.to_numpy()).all() and np.isfinite(Zt.to_numpy()).all()


def test_robust_standardise_scale_of_the_reference_is_unusable_when_it_is_negligible_next_to_the_centre():
    Xs = pd.DataFrame({"c": [4.0, 5.0, 6.0, 7.0, 9.0]})
    Xt = pd.DataFrame({"c": [6.0]})
    reference = pd.DataFrame({"c": 5.0 + 1e-14 * np.arange(9)})
    _, _, scale = tr.robust_standardise(Xs, Xt, reference=reference, clip=None)
    assert scale["c"] == pytest.approx(1.4826 * 1.0)
    constant_sources = pd.DataFrame({"c": [4.0] * 6})
    Zs, Zt, scale = tr.robust_standardise(constant_sources, Xt, reference=pd.DataFrame({"c": [5.0] * 10}), clip=None)
    assert scale["c"] == 1.0 and Zs["c"].tolist() == [-1.0] * 6 and Zt["c"].tolist() == [1.0]


def test_robust_standardise_clips_the_standardised_values():
    reference = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    scale = 1.4826 * 2.0
    Xs = pd.DataFrame({"x": [3.0, 3.0 + 7 * scale, 3.0 - 9 * scale, np.nan]}, index=list("pqrs"))
    Xt = pd.DataFrame({"x": [3.0 + 5.5 * scale, 3.0 - 4.5 * scale]}, index=list("tu"))
    Zs, Zt, got = tr.robust_standardise(Xs, Xt, reference=reference)
    assert Zs["x"].iloc[:3].tolist() == pytest.approx([0.0, 5.0, -5.0]) and np.isnan(Zs["x"].iloc[3])
    assert Zt["x"].tolist() == pytest.approx([5.0, -4.5])
    assert got["x"] == pytest.approx(scale)
    Zs2, Zt2, got2 = tr.robust_standardise(Xs, Xt, reference=reference, clip=2.0)
    assert Zs2["x"].iloc[:3].tolist() == pytest.approx([0.0, 2.0, -2.0]) and Zt2["x"].tolist() == pytest.approx([2.0, -2.0])
    assert got2["x"] == got["x"]
    Zs3, Zt3, _ = tr.robust_standardise(Xs, Xt, reference=reference, clip=None)
    assert Zs3["x"].iloc[:3].tolist() == pytest.approx([0.0, 7.0, -9.0]) and Zt3["x"].tolist() == pytest.approx([5.5, -4.5])
    assert inspect.signature(tr.robust_standardise).parameters["clip"].default == 5.0
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="clip"):
            tr.robust_standardise(Xs, Xt, reference=reference, clip=bad)


def test_robust_standardise_names_the_argument_in_its_errors():
    good = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="Xs must be numeric"):
        tr.robust_standardise(pd.DataFrame({"x": ["a", "b", "c"]}), good)
    with pytest.raises(ValueError, match="Xt must not contain infinite"):
        tr.robust_standardise(good, pd.DataFrame({"x": [np.inf]}))
    with pytest.raises(ValueError, match="reference needs at least one row"):
        tr.robust_standardise(good, good, reference=good.iloc[:0])
    with pytest.raises(ValueError, match="columns must not contain duplicates"):
        tr.robust_standardise(good, good, columns=["x", "x"])
    with pytest.raises(TypeError, match="Xs"):
        tr.robust_standardise(good["x"], good)
    with pytest.raises(ValueError, match="Xt"):
        tr.robust_standardise(good, np.ones(3))


def test_robust_standardise_does_not_modify_its_inputs():
    rng = np.random.default_rng(1)
    Xs = pd.DataFrame(rng.normal(size=(6, 2)), columns=["u", "v"])
    Xt = pd.DataFrame(rng.normal(size=(3, 2)), columns=["u", "v"])
    before_s, before_t = Xs.copy(), Xt.copy()
    tr.robust_standardise(Xs, Xt, reference=Xs.iloc[:4])
    pd.testing.assert_frame_equal(Xs, before_s)
    pd.testing.assert_frame_equal(Xt, before_t)


@pytest.mark.parametrize("use_reference", [True, False])
def test_rescaling_the_units_of_one_feature_leaves_cost_and_plan_unchanged(use_reference):
    rng = np.random.default_rng(2)
    cols = ["gdp", "trade", "pop", "tourism"]
    ref = pd.DataFrame(rng.lognormal(size=(300, 4)), columns=cols)
    Xs = pd.DataFrame(rng.lognormal(size=(10, 4)), columns=cols)
    Xt = pd.DataFrame(rng.lognormal(size=(6, 4)), columns=cols)
    w = np.array([3.0, 1.0, 0.5, 2.0])

    def run(factor, offset):
        def rescale(X):
            X = X.copy()
            X["pop"] = X["pop"] * factor + offset
            return X

        kwargs = {"reference": rescale(ref)} if use_reference else {}
        Zs, Zt, _ = tr.robust_standardise(rescale(Xs), rescale(Xt), **kwargs)
        C = tr.weighted_sq_cost(Zs, Zt, w)
        plan = tr.sinkhorn_plan(np.full(10, 0.1), np.full(6, 1 / 6), C, 0.5, rho_source=1.0)
        return Zs, Zt, C, plan.pi

    base = run(1.0, 0.0)
    for factor, offset in [(1000.0, 0.0), (1e-3, 0.0), (250.0, 17.0)]:
        other = run(factor, offset)
        for x, y in zip(base, other):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-9, atol=1e-12)


# ----------------------------------------------------------------------------
# weighted_sq_cost
# ----------------------------------------------------------------------------
def test_weighted_sq_cost_matches_direct_sum_and_uniform_weights_give_euclidean_distance():
    rng = np.random.default_rng(3)
    Zs, Zt = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
    w = np.array([3.0, 1.0, 0.0, 2.0])
    C = tr.weighted_sq_cost(Zs, Zt, w)
    assert C.shape == (7, 5)
    rescaled = w * 4 / w.sum()
    direct = ((Zs[:, None, :] - Zt[None, :, :]) ** 2 * rescaled).sum(axis=2)
    np.testing.assert_allclose(C, direct, rtol=1e-12)
    np.testing.assert_allclose(tr.weighted_sq_cost(Zs, Zt, 7.5 * w), C, rtol=1e-12)
    for level in (1.0, 0.2, 40.0):
        np.testing.assert_allclose(tr.weighted_sq_cost(Zs, Zt, np.full(4, level)), cdist(Zs, Zt, "sqeuclidean"), rtol=1e-12)
    onehot = tr.weighted_sq_cost(Zs, Zt, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(onehot, 4 * (Zs[:, [1]] - Zt[:, [1]].T) ** 2, rtol=1e-12)
    assert np.all(np.diag(tr.weighted_sq_cost(Zs, Zs, w)) == 0.0)


def test_weighted_sq_cost_validates_inputs_and_matches_labels():
    rng = np.random.default_rng(4)
    Zs, Zt = rng.normal(size=(4, 3)), rng.normal(size=(2, 3))
    for bad in ([1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0], [1.0, np.nan, 1.0]):
        with pytest.raises(ValueError):
            tr.weighted_sq_cost(Zs, Zt, bad)
    with pytest.raises(ValueError):
        tr.weighted_sq_cost(Zs, rng.normal(size=(2, 4)), np.ones(3))
    nan_inputs = Zs.copy()
    nan_inputs[0, 1] = np.nan
    with pytest.raises(ValueError):
        tr.weighted_sq_cost(nan_inputs, Zt, np.ones(3))
    assert np.isfinite(tr.weighted_sq_cost(nan_inputs, Zt, [1.0, 0.0, 1.0])).all()
    cols = ["x", "y", "z"]
    dfs, dft = pd.DataFrame(Zs, columns=cols), pd.DataFrame(Zt, columns=cols)
    w = pd.Series([1.0, 2.0, 3.0], index=cols)
    expected = tr.weighted_sq_cost(Zs, Zt, w.to_numpy())
    np.testing.assert_allclose(tr.weighted_sq_cost(dfs, dft[["z", "x", "y"]], w), expected, rtol=1e-12)
    np.testing.assert_allclose(tr.weighted_sq_cost(dfs, dft, w[["y", "z", "x"]]), expected, rtol=1e-12)
    with pytest.raises(KeyError):
        tr.weighted_sq_cost(dfs, dft.drop(columns="y"), w)


def test_weighted_sq_cost_errors_name_the_argument():
    rng = np.random.default_rng(14)
    Zs, Zt = rng.normal(size=(4, 3)), rng.normal(size=(2, 3))
    bad_s, bad_t = Zs.copy(), Zt.copy()
    bad_s[1, 2] = np.nan
    bad_t[0, 0] = np.inf
    cases = [
        ({"Zs": bad_s}, "Zs must be finite"),
        ({"Zt": bad_t}, "Zt must be finite"),
        ({"Zt": Zt[:, :2]}, "Zs has 3 features but Zt has 2"),
        ({"Zs": np.ones((2, 2, 2))}, "Zs must be a DataFrame or a two-dimensional array"),
        ({"Zt": [["a", "b", "c"]]}, "Zt must be numeric"),
        ({"w": [1.0, 1.0]}, "w has 2 entries but there are 3 features"),
        ({"w": [1.0, np.nan, 1.0]}, "w must be finite and non-negative"),
        ({"w": [1.0, -1.0, 1.0]}, "w must be finite and non-negative"),
        ({"w": [0.0, 0.0, 0.0]}, "w sums to zero"),
        ({"w": np.ones((3, 1))}, "w must be one-dimensional"),
    ]
    for override, message in cases:
        args = {"Zs": Zs, "Zt": Zt, "w": np.ones(3)}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.weighted_sq_cost(**args)


def test_a_zero_weight_feature_has_no_influence_on_cost_or_plan():
    rng = np.random.default_rng(5)
    Zs, Zt = rng.normal(size=(9, 4)), rng.normal(size=(7, 4)) + 0.3
    w = np.array([2.0, 1.0, 0.0, 1.0])
    a, b = np.full(9, 1 / 9), np.full(7, 1 / 7)
    base_cost = tr.weighted_sq_cost(Zs, Zt, w)
    Zs2, Zt2 = Zs.copy(), Zt.copy()
    Zs2[:, 2] = rng.normal(scale=50.0, size=9)
    Zt2[:, 2] = rng.normal(scale=50.0, size=7)
    cost2 = tr.weighted_sq_cost(Zs2, Zt2, w)
    assert np.array_equal(base_cost, cost2)
    for rho_s, rho_t in PATTERNS:
        p1 = tr.sinkhorn_plan(a, b, base_cost, 0.5, rho_source=rho_s, rho_target=rho_t)
        p2 = tr.sinkhorn_plan(a, b, cost2, 0.5, rho_source=rho_s, rho_target=rho_t)
        assert np.array_equal(p1.pi, p2.pi)
    assert tr.select_eps(Zs, w) == tr.select_eps(Zs2, w)
    assert tr.overlap_permutation_test(Zs, Zt, w, n_perm=20)["statistic"] == tr.overlap_permutation_test(Zs2, Zt2, w, n_perm=20)["statistic"]


# ----------------------------------------------------------------------------
# sinkhorn_plan
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed,n,m", [(0, 6, 6), (1, 10, 7), (2, 5, 12)])
def test_balanced_sinkhorn_marginals_converge_to_a_and_b(seed, n, m):
    Zs, _, w, a, b, C = random_problem(seed, n, m)
    eps = tr.select_eps(Zs, w)
    plan = tr.sinkhorn_plan(a, b, C, eps, tol=1e-12)
    assert plan.converged and plan.marginal_error < 1e-12
    assert plan.pi.shape == (n, m) and (plan.pi > 0).all()
    assert np.abs(plan.pi.sum(axis=1) - a).sum() < 1e-11
    assert np.abs(plan.pi.sum(axis=0) - b).sum() < 1e-11
    assert plan.mass == pytest.approx(1.0, abs=1e-11)
    assert plan.eps == eps and plan.rho_source is None and plan.rho_target is None
    residual = max(np.abs(plan.pi.sum(axis=1) - a).sum(), np.abs(plan.pi.sum(axis=0) - b).sum())
    assert plan.marginal_error == pytest.approx(residual, abs=1e-12)


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1e3, 1e9])
def test_balanced_convergence_flag_error_and_iterations_do_not_depend_on_the_scale_of_the_masses(scale):
    _, _, _, a, b, C = random_problem(18, 7, 6)
    base = tr.sinkhorn_plan(a, b, C, 0.2)
    other = tr.sinkhorn_plan(a * scale, b * scale, C, 0.2)
    assert base.converged and other.converged and base.n_iter > 20
    assert other.n_iter == base.n_iter
    assert other.marginal_error == pytest.approx(base.marginal_error, rel=1e-6)
    np.testing.assert_allclose(other.pi / scale, base.pi, rtol=1e-9)


@pytest.mark.parametrize("scale", [1e-9, 1e6])
@pytest.mark.parametrize("rho_s,rho_t", [(1.0, None), (0.5, 2.0)])
def test_unbalanced_plan_is_optimal_when_flagged_converged_whatever_the_scale_of_the_masses(scale, rho_s, rho_t):
    rng = np.random.default_rng(19)
    n, m = 5, 4
    C = tr.weighted_sq_cost(rng.normal(size=(n, 2)), rng.normal(size=(m, 2)) + 0.5, [1.0, 1.0])
    a, b = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, m)
    a, b = scale * a / a.sum(), scale * b / b.sum()
    plan = tr.sinkhorn_plan(a, b, C, 0.4, rho_source=rho_s, rho_target=rho_t)
    assert plan.converged and plan.marginal_error < 1e-9 and plan.n_iter > 5
    assert kkt_residual(plan.pi, C, a, b, 0.4, rho_s, rho_t) < 1e-7


def test_balanced_marginal_error_decreases_with_the_iteration_budget():
    _, _, _, a, b, C = random_problem(7, 8, 9)
    errors = [tr.sinkhorn_plan(a, b, C, 0.3, max_iter=k, tol=1e-14).marginal_error for k in (2, 8, 32, 128, 512)]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 1e-8


def test_non_convergence_is_reported_and_iteration_count_is_exact():
    _, _, _, a, b, C = random_problem(8)
    plan = tr.sinkhorn_plan(a, b, C, 0.05, max_iter=3, tol=1e-12)
    assert not plan.converged and plan.n_iter == 3 and plan.marginal_error > 1e-12
    done = tr.sinkhorn_plan(a, b, C, 1.0, max_iter=5000, tol=1e-9)
    assert done.converged and 0 < done.n_iter < 5000 and done.marginal_error < 1e-9


def test_balanced_plan_cost_approaches_the_exact_linprog_cost_as_eps_shrinks():
    n, m = 8, 9
    Zs, Zt, w, a, b, C = random_problem(3, n, m, uniform=True)
    exact = tr.wasserstein2(a, b, Zs, Zt, w) ** 2
    costs = []
    for fraction in (1.0, 0.3, 0.1, 0.03, 0.01):
        eps = fraction * np.median(C)
        plan = tr.sinkhorn_plan(a, b, C, eps)
        assert plan.converged
        cost = float((plan.pi * C).sum())
        assert exact - 1e-9 <= cost <= exact + eps * np.log(min(n, m)) + 1e-9
        costs.append(cost)
    assert all(np.diff(costs) < 0)
    assert costs[-1] - exact < 0.01 * exact


@pytest.mark.parametrize("rho_s,rho_t", PATTERNS)
def test_plan_satisfies_the_first_order_conditions_of_the_documented_objective(rho_s, rho_t):
    rng = np.random.default_rng(11)
    n, m = 4, 5
    Zs, Zt = rng.normal(size=(n, 3)), rng.normal(size=(m, 3)) + 0.5
    C = tr.weighted_sq_cost(Zs, Zt, [2.0, 1.0, 0.5])
    a, b = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, m)
    a, b = a / a.sum(), b / b.sum()
    eps = 0.4
    plan = tr.sinkhorn_plan(a, b, C, eps, rho_source=rho_s, rho_target=rho_t, tol=1e-13)
    assert plan.converged
    assert kkt_residual(plan.pi, C, a, b, eps, rho_s, rho_t) < 1e-9
    base = primal_objective(plan.pi, C, a, b, eps, rho_s, rho_t)
    for _ in range(25):
        E = rng.normal(size=(n, m))
        if rho_s is None:
            E = E - E.mean(axis=1, keepdims=True)
        if rho_t is None:
            E = E - E.mean(axis=0, keepdims=True)
        other = plan.pi + 1e-3 * E / np.abs(E).max() * plan.pi.min()
        assert primal_objective(other, C, a, b, eps, rho_s, rho_t) >= base - 1e-14


@pytest.mark.parametrize("seed", range(3))
def test_balanced_plan_matches_pot_sinkhorn(seed):
    ot = pytest.importorskip("ot")
    Zs, _, w, a, b, C = random_problem(seed, 9, 11)
    eps = tr.select_eps(Zs, w)
    plan = tr.sinkhorn_plan(a, b, C, eps, tol=1e-12)
    reference = ot.sinkhorn(a, b, C, eps, method="sinkhorn_log", numItermax=100000, stopThr=1e-12, warn=False)
    assert np.abs(plan.pi - reference).max() < 1e-6


@pytest.mark.parametrize("rho_s,rho_t", [p for p in PATTERNS if p != (None, None)])
@pytest.mark.parametrize("seed", range(2))
def test_unbalanced_plan_matches_pot_sinkhorn_unbalanced(seed, rho_s, rho_t):
    ot = pytest.importorskip("ot")
    Zs, _, w, a, b, C = random_problem(seed, 9, 11)
    eps = tr.select_eps(Zs, w)
    plan = tr.sinkhorn_plan(a, b, C, eps, rho_source=rho_s, rho_target=rho_t, tol=1e-12)
    reg_m = (np.inf if rho_s is None else rho_s, np.inf if rho_t is None else rho_t)
    reference = ot.unbalanced.sinkhorn_unbalanced(a, b, C, eps, reg_m, numItermax=500000, stopThr=1e-14)
    assert plan.converged
    assert np.abs(plan.pi - reference).max() < 1e-6


@pytest.mark.parametrize("rho_s,rho_t", PATTERNS)
def test_plan_matches_the_cvxpy_solution_of_the_primal_problem(rho_s, rho_t):
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(12)
    n, m = 5, 6
    Zs, Zt = rng.normal(size=(n, 3)), rng.normal(size=(m, 3)) + 0.5
    C = tr.weighted_sq_cost(Zs, Zt, [2.0, 1.0, 0.5])
    a, b = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, m)
    a, b = a / a.sum(), b / b.sum()
    eps = 0.5
    plan = tr.sinkhorn_plan(a, b, C, eps, rho_source=rho_s, rho_target=rho_t, tol=1e-12)
    P = cp.Variable((n, m), nonneg=True)
    objective = cp.sum(cp.multiply(C, P)) + eps * cp.sum(cp.kl_div(P, np.outer(a, b)))
    constraints = []
    if rho_s is None:
        constraints.append(cp.sum(P, axis=1) == a)
    else:
        objective = objective + rho_s * cp.sum(cp.kl_div(cp.sum(P, axis=1), a))
    if rho_t is None:
        constraints.append(cp.sum(P, axis=0) == b)
    else:
        objective = objective + rho_t * cp.sum(cp.kl_div(cp.sum(P, axis=0), b))
    cp.Problem(cp.Minimize(objective), constraints).solve(solver=cp.CLARABEL)
    assert np.abs(plan.pi - P.value).max() < 1e-4
    ours = primal_objective(plan.pi, C, a, b, eps, rho_s, rho_t)
    theirs = primal_objective(np.maximum(P.value, 1e-300), C, a, b, eps, rho_s, rho_t)
    assert ours <= theirs + 1e-7 and theirs - ours < 1e-6


def test_unbalanced_mass_shrinks_when_rho_is_small_and_a_source_is_far():
    Zs = np.array([[0.1, 0.0], [-0.1, 0.1], [0.0, -0.2], [0.2, 0.1], [8.0, 8.0]])
    Zt = np.array([[0.0, 0.0]])
    C = tr.weighted_sq_cost(Zs, Zt, np.ones(2))
    a, b = np.full(5, 0.2), np.array([1.0])
    rhos = [0.05, 0.5, 5.0, 50.0]
    far = [tr.source_usage(tr.sinkhorn_plan(a, b, C, 0.5, rho_source=rho))[4] for rho in rhos]
    assert far[0] < 1e-10 and all(np.diff(far) > 0) and far[-1] < 0.2
    balanced = tr.sinkhorn_plan(a, b, C, 0.5)
    assert tr.source_usage(balanced)[4] == pytest.approx(0.2, abs=1e-9)
    near_share = [tr.source_usage(tr.sinkhorn_plan(a, b, C, 0.5, rho_source=rho))[:4].sum() for rho in rhos]
    assert all(share == pytest.approx(1.0, abs=1e-8) for share in [near_share[i] + far[i] for i in range(4)])
    masses = [tr.sinkhorn_plan(a, b, C, 0.5, rho_source=rho, rho_target=rho).mass for rho in rhos]
    assert all(np.diff(masses) > 0) and masses[-1] < 1.0
    Zs_near = Zs.copy()
    Zs_near[4] = [0.15, -0.1]
    C_near = tr.weighted_sq_cost(Zs_near, Zt, np.ones(2))
    for rho in (0.05, 0.5):
        with_far = tr.sinkhorn_plan(a, b, C, 0.5, rho_source=rho, rho_target=rho).mass
        without_far = tr.sinkhorn_plan(a[:4], b, C[:4], 0.5, rho_source=rho, rho_target=rho).mass
        all_near = tr.sinkhorn_plan(a, b, C_near, 0.5, rho_source=rho, rho_target=rho).mass
        assert with_far == pytest.approx(without_far, abs=1e-8)
        assert with_far < all_near - 0.1 < 1.0
    mass_by_distance = []
    for shift in (1.0, 3.0, 6.0):
        C_shift = tr.weighted_sq_cost(np.array([[shift, 0.0]]), Zt, np.ones(2))
        mass_by_distance.append(tr.sinkhorn_plan(np.array([1.0]), b, C_shift, 0.5, rho_source=1.0, rho_target=1.0).mass)
    assert all(np.diff(mass_by_distance) < 0)


def test_zero_mass_entries_receive_no_mass_and_do_not_change_the_rest():
    _, _, _, a, b, C = random_problem(9, 8, 9)
    a0, b0 = a.copy(), b.copy()
    a0[[1, 4]] = 0.0
    b0[[0, 7]] = 0.0
    a0, b0 = a0 / a0.sum(), b0 / b0.sum()
    for rho_s, rho_t in PATTERNS:
        full = tr.sinkhorn_plan(a0, b0, C, 0.4, rho_source=rho_s, rho_target=rho_t, tol=1e-12)
        assert np.all(full.pi[[1, 4], :] == 0.0) and np.all(full.pi[:, [0, 7]] == 0.0)
        keep_s, keep_t = np.setdiff1d(np.arange(8), [1, 4]), np.setdiff1d(np.arange(9), [0, 7])
        sub = tr.sinkhorn_plan(a0[keep_s], b0[keep_t], C[np.ix_(keep_s, keep_t)], 0.4, rho_source=rho_s, rho_target=rho_t, tol=1e-12)
        np.testing.assert_allclose(full.pi[np.ix_(keep_s, keep_t)], sub.pi, atol=1e-14)


def test_sinkhorn_plan_validates_its_inputs_and_names_the_argument():
    _, _, _, a, b, C = random_problem(10, 5, 6)
    bad = C.copy()
    bad[0, 0] = np.inf
    cases = [
        ({"eps": 0.0}, "eps must be positive and finite"),
        ({"eps": -1.0}, "eps must be positive and finite"),
        ({"eps": np.nan}, "eps must be positive and finite"),
        ({"rho_source": 0.0}, "rho_source must be positive or None"),
        ({"rho_target": -2.0}, "rho_target must be positive or None"),
        ({"a": -a}, "a must be finite and non-negative"),
        ({"a": np.where(np.arange(5) == 1, np.nan, a)}, "a must be finite and non-negative"),
        ({"a": np.zeros(5)}, "a has zero total mass"),
        ({"a": a[:4]}, "a has 4 entries, expected 5"),
        ({"a": np.ones((5, 1))}, "a must be one-dimensional"),
        ({"b": np.zeros(6)}, "b has zero total mass"),
        ({"b": b[:5]}, "b has 5 entries, expected 6"),
        ({"b": b * np.array([1, 1, 1, 1, 1, -1.0])}, "b must be finite and non-negative"),
        ({"C": C[:, :4]}, "b has 6 entries, expected 4"),
        ({"C": C[0]}, "C must be two-dimensional"),
        ({"C": bad}, "C must be finite"),
        ({"C": [["x", "y"]]}, "C must be numeric"),
        ({"b": b * 2.0}, "balanced transport needs a and b with equal total mass"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"tol": 0.0}, "tol positive"),
    ]
    for override, message in cases:
        args = {"a": a, "b": b, "C": C, "eps": 0.5}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.sinkhorn_plan(**args)
    same = tr.sinkhorn_plan(a, b, C, 0.5, rho_source=np.inf, rho_target=np.inf)
    np.testing.assert_array_equal(same.pi, tr.sinkhorn_plan(a, b, C, 0.5).pi)
    assert same.rho_source is None and same.rho_target is None
    assert tr.sinkhorn_plan(a, 3 * b, C, 0.5, rho_source=1.0).converged


def test_plan_labels_come_from_series_and_dataframe_inputs():
    _, _, _, a, b, C = random_problem(13, 4, 3)
    src, tgt = list("wxyz"), ["t0", "t1", "t2"]
    plan = tr.sinkhorn_plan(pd.Series(a, index=src), pd.Series(b, index=tgt), C, 0.5)
    assert list(tr.source_usage(plan).index) == src
    assert list(tr.transported_effects(plan, np.arange(4.0)).index) == tgt
    plan2 = tr.sinkhorn_plan(a, b, pd.DataFrame(C, index=src, columns=tgt), 0.5)
    assert list(tr.source_usage(plan2).index) == src
    assert list(tr.transported_effects(plan2, np.arange(4.0)).index) == tgt
    plan3 = tr.sinkhorn_plan(a, b, C, 0.5)
    assert list(tr.source_usage(plan3).index) == [0, 1, 2, 3]


# ----------------------------------------------------------------------------
# Transported effects, usage and noise
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("rho_s,rho_t", PATTERNS)
def test_barycentric_effect_equals_the_common_value_when_all_sources_share_one_effect(rho_s, rho_t):
    _, _, _, a, b, C = random_problem(14, 7, 5)
    plan = tr.sinkhorn_plan(a, b, C, 0.4, rho_source=rho_s, rho_target=rho_t)
    effects = tr.transported_effects(plan, np.full(7, -2.7))
    np.testing.assert_allclose(effects["effect"], -2.7, rtol=1e-12)
    assert tr.target_effect(plan, np.full(7, -2.7), b) == pytest.approx(-2.7, rel=1e-12)
    assert tr.target_effect(plan, np.full(7, -2.7)) == pytest.approx(-2.7, rel=1e-12)


def test_transported_effects_are_mass_weighted_means_of_the_source_effects():
    _, _, _, a, b, C = random_problem(15, 6, 4)
    plan = tr.sinkhorn_plan(a, b, C, 0.4, rho_source=1.0)
    tau = np.array([1.0, -2.0, 0.5, 3.0, 0.0, 2.0])
    table = tr.transported_effects(plan, tau)
    assert list(table.columns) == ["mass_received", "effect"] and len(table) == 4
    np.testing.assert_allclose(table["mass_received"], plan.pi.sum(axis=0), rtol=1e-12)
    for j in range(4):
        assert table["effect"].iloc[j] == pytest.approx((plan.pi[:, j] * tau).sum() / plan.pi[:, j].sum(), rel=1e-12)
    assert table["effect"].min() >= tau.min() and table["effect"].max() <= tau.max()
    with pytest.raises(ValueError):
        tr.transported_effects(plan, tau[:5])


def test_sources_without_mass_and_target_points_without_mass_are_handled():
    pi = np.array([[0.2, 0.0, 0.1], [0.0, 0.0, 0.0], [0.3, 0.0, 0.4]])
    plan = tr.Plan(pi=pi, converged=True, n_iter=0, marginal_error=0.0, eps=1.0, rho_source=1.0, rho_target=1.0)
    tau = np.array([1.0, np.nan, 5.0])
    table = tr.transported_effects(plan, tau)
    assert table["effect"].iloc[0] == pytest.approx((0.2 * 1.0 + 0.3 * 5.0) / 0.5)
    assert np.isnan(table["effect"].iloc[1]) and table["mass_received"].iloc[1] == 0.0
    assert table["effect"].iloc[2] == pytest.approx((0.1 * 1.0 + 0.4 * 5.0) / 0.5)
    assert tr.target_effect(plan, tau, [1.0, 0.0, 1.0]) == pytest.approx(0.5 * (table["effect"].iloc[0] + table["effect"].iloc[2]))
    assert np.isnan(tr.target_effect(plan, tau, [1.0, 1.0, 1.0]))


def test_effects_and_target_weights_of_the_transported_effect_are_checked():
    pi = np.array([[0.2, 0.0, 0.1], [0.0, 0.0, 0.0], [0.3, 0.0, 0.4]])
    plan = tr.Plan(pi=pi, converged=True, n_iter=0, marginal_error=0.0, eps=1.0, rho_source=1.0, rho_target=1.0)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="tau must be finite for every source that sends mass"):
            tr.transported_effects(plan, [1.0, 0.0, bad])
        with pytest.raises(ValueError, match="tau must be finite for every source that sends mass"):
            tr.target_effect(plan, [bad, 0.0, 1.0])
    ignored = tr.transported_effects(plan, [1.0, np.inf, 5.0])
    assert ignored["effect"].iloc[0] == pytest.approx((0.2 * 1.0 + 0.3 * 5.0) / 0.5)
    with pytest.raises(ValueError, match="tau has 2 entries, expected 3"):
        tr.transported_effects(plan, [1.0, 2.0])
    with pytest.raises(ValueError, match="tau must be one-dimensional"):
        tr.target_effect(plan, np.ones((3, 1)))
    with pytest.raises(ValueError, match="tau must be numeric"):
        tr.transported_effects(plan, ["a", "b", "c"])
    with pytest.raises(ValueError, match="b has 2 entries, expected 3"):
        tr.target_effect(plan, [1.0, 2.0, 3.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="b must be finite and non-negative"):
        tr.target_effect(plan, [1.0, 2.0, 3.0], [1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="b has zero total mass"):
        tr.target_effect(plan, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0])


def test_mixture_and_quantile_errors_name_the_argument():
    pi = np.array([[0.1, 0.2], [0.3, 0.0]])
    plan = tr.Plan(pi=pi, converged=True, n_iter=0, marginal_error=0.0, eps=1.0, rho_source=1.0, rho_target=None)
    with pytest.raises(ValueError, match="draws_by_source has 1 entries, expected 2"):
        tr.transported_mixture(plan, [[1.0]])
    with pytest.raises(ValueError, match="draws_by_source must not contain infinite"):
        tr.transported_mixture(plan, [[1.0], [np.inf]])
    with pytest.raises(ValueError, match="draws_by_source must be numeric"):
        tr.transported_mixture(plan, [["a"], [1.0]])
    with pytest.raises(ValueError, match="probs must be finite and non-negative"):
        tr.weighted_quantile([1.0, 2.0], [1.0, np.nan], 0.5)
    with pytest.raises(ValueError, match="probs must be finite and non-negative"):
        tr.weighted_quantile([1.0, 2.0], [1.0, -1.0], 0.5)
    with pytest.raises(ValueError, match="values must not contain infinite"):
        tr.weighted_quantile([1.0, np.inf], [1.0, 1.0], 0.5)
    with pytest.raises(ValueError, match="values and probs must have the same length"):
        tr.weighted_quantile([1.0, 2.0, 3.0], [1.0, 1.0], 0.5)
    with pytest.raises(ValueError, match="values and probs must be one-dimensional"):
        tr.weighted_quantile(np.ones((2, 2)), np.ones(4), 0.5)
    with pytest.raises(ValueError, match="q must lie between 0 and 1"):
        tr.weighted_quantile([1.0, 2.0], [1.0, 1.0], [0.5, 1.5])
    with pytest.raises(ValueError, match="q must lie between 0 and 1"):
        tr.weighted_quantile([1.0, 2.0], [1.0, 1.0], np.nan)
    with pytest.raises(ValueError, match="values must be numeric"):
        tr.weighted_quantile(["a", "b"], [1.0, 1.0], 0.5)


def test_source_usage_is_the_row_sum_and_effective_sample_size_follows_the_formula():
    _, _, _, a, b, C = random_problem(16, 6, 5)
    plan = tr.sinkhorn_plan(a, b, C, 0.4, rho_source=1.0)
    usage = tr.source_usage(plan)
    assert usage.name == "usage" and len(usage) == 6
    np.testing.assert_allclose(usage.to_numpy(), plan.pi.sum(axis=1))
    assert usage.sum() == pytest.approx(plan.mass)
    w = usage.to_numpy()
    assert tr.effective_sample_size(w) == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert tr.effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    assert tr.effective_sample_size([0.0, 0.0, 5.0, 0.0]) == pytest.approx(1.0)
    assert tr.effective_sample_size([1.0, 3.0]) == pytest.approx(16.0 / 10.0)
    assert tr.effective_sample_size(37.0 * w) == pytest.approx(tr.effective_sample_size(w))
    assert tr.effective_sample_size(np.zeros(3)) == 0.0
    assert tr.effective_sample_size([]) == 0.0
    assert 1.0 <= tr.effective_sample_size(w) <= 6.0
    for level in (1e-200, 1e-320, 1e200):
        assert tr.effective_sample_size([level, 3 * level]) == pytest.approx(1.6)
        assert tr.effective_sample_size([level, level, level]) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        tr.effective_sample_size([1.0, -1.0])
    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        tr.effective_sample_size([1.0, np.nan])
    with pytest.raises(ValueError, match="weights must be numeric"):
        tr.effective_sample_size(["a"])


@pytest.mark.parametrize("rho_s", [None, 1.0, 0.3])
def test_target_effect_is_the_usage_weighted_mean_when_the_target_marginal_is_enforced(rho_s):
    _, _, _, a, b, C = random_problem(17, 8, 6)
    tau = np.random.default_rng(17).normal(size=8)
    plan = tr.sinkhorn_plan(a, b, C, 0.4, rho_source=rho_s, tol=1e-12)
    usage = tr.source_usage(plan).to_numpy()
    assert tr.target_effect(plan, tau, b) == pytest.approx(usage @ tau / usage.sum(), abs=1e-9)
    relaxed = tr.sinkhorn_plan(a, b, C, 0.4, rho_source=rho_s, rho_target=0.3, tol=1e-12)
    usage2 = tr.source_usage(relaxed).to_numpy()
    assert abs(tr.target_effect(relaxed, tau, b) - usage2 @ tau / usage2.sum()) > 1e-6


def test_transported_mixture_weights_follow_source_usage():
    pi = np.array([[0.1, 0.2], [0.3, 0.0], [0.0, 0.0], [0.0, 0.4]])
    plan = tr.Plan(pi=pi, converged=True, n_iter=0, marginal_error=0.0, eps=1.0, rho_source=1.0, rho_target=None)
    draws = [np.array([1.0, 2.0]), np.array([10.0]), np.array([99.0, 98.0]), np.array([5.0, 6.0, 7.0, np.nan])]
    values, probs = tr.transported_mixture(plan, draws)
    assert probs.sum() == pytest.approx(1.0) and np.all(np.diff(values) >= 0)
    assert 99.0 not in values and np.isnan(values).sum() == 0
    expected = {1.0: 0.15, 2.0: 0.15, 10.0: 0.3, 5.0: 0.4 / 3, 6.0: 0.4 / 3, 7.0: 0.4 / 3}
    assert len(values) == len(expected)
    for v, p in zip(values, probs):
        assert p == pytest.approx(expected[v])
    labelled = tr.Plan(pi=pi, converged=True, n_iter=0, marginal_error=0.0, eps=1.0, rho_source=1.0, rho_target=None,
                       source_index=pd.Index(["a", "b", "c", "d"]))
    mapping = dict(zip(["d", "c", "b", "a"], draws[::-1]))
    v2, p2 = tr.transported_mixture(labelled, mapping)
    np.testing.assert_array_equal(v2, values)
    np.testing.assert_allclose(p2, probs)
    with pytest.raises(KeyError):
        tr.transported_mixture(labelled, {"a": draws[0]})
    with pytest.raises(ValueError):
        tr.transported_mixture(plan, draws[:3])
    with pytest.raises(ValueError):
        tr.transported_mixture(plan, [[], [], [], []])


def test_transported_mixture_of_point_masses_reproduces_the_usage_shares():
    rng = np.random.default_rng(18)
    Zs, Zt = rng.normal(size=(6, 2)), np.zeros((1, 2))
    C = tr.weighted_sq_cost(Zs, Zt, np.ones(2))
    plan = tr.sinkhorn_plan(np.full(6, 1 / 6), np.array([1.0]), C, 0.3, rho_source=0.5)
    centres = np.arange(6) * 10.0
    values, probs = tr.transported_mixture(plan, [np.full(5, c) for c in centres])
    share = tr.source_usage(plan).to_numpy()
    share = share / share.sum()
    for c, s in zip(centres, share):
        assert probs[values == c].sum() == pytest.approx(s)
    assert (values * probs).sum() == pytest.approx((share * centres).sum())
    median = tr.weighted_quantile(values, probs, 0.5)
    k = int(np.searchsorted(np.cumsum(share), 0.5))
    assert centres[max(k - 1, 0)] <= median <= centres[min(k + 1, 5)]
    shifted = [np.full(5, c + (3.0 if i == 2 else 0.0)) for i, c in enumerate(centres)]
    v2, p2 = tr.transported_mixture(plan, shifted)
    assert (v2 * p2).sum() - (values * probs).sum() == pytest.approx(3.0 * share[2])


def test_weighted_quantile_matches_hazen_quantile_for_equal_weights_and_interpolates_weights():
    rng = np.random.default_rng(19)
    x = rng.normal(size=37)
    levels = np.array([0.0, 0.03, 0.1, 0.25, 0.5, 0.77, 0.95, 1.0])
    got = tr.weighted_quantile(x, np.ones(37), levels)
    np.testing.assert_allclose(got, np.quantile(x, levels, method="hazen"), rtol=1e-12)
    assert isinstance(tr.weighted_quantile(x, np.ones(37), 0.5), float)
    assert tr.weighted_quantile([0.0, 1.0], [0.25, 0.75], 0.125) == pytest.approx(0.0)
    assert tr.weighted_quantile([0.0, 1.0], [0.25, 0.75], 0.625) == pytest.approx(1.0)
    assert tr.weighted_quantile([0.0, 1.0], [0.25, 0.75], 0.375) == pytest.approx(0.5)
    assert tr.weighted_quantile([0.0, 1.0], [0.25, 0.75], 0.0) == 0.0
    assert tr.weighted_quantile([0.0, 1.0], [0.25, 0.75], 1.0) == 1.0
    assert tr.weighted_quantile([1.0, 0.0], [7.5, 2.5], 0.375) == pytest.approx(0.5)
    assert tr.weighted_quantile([0.0, 5.0, 10.0, np.nan], [1.0, 0.0, 1.0, 1.0], 0.5) == pytest.approx(5.0)
    assert tr.weighted_quantile([4.0], [2.0], 0.3) == 4.0
    assert tr.weighted_quantile([0.0, 1.0, 100.0], [0.01, 0.01, 0.98], 0.5) > 50.0
    grid = tr.weighted_quantile(x, rng.uniform(size=37), np.linspace(0, 1, 21))
    assert np.all(np.diff(grid) >= 0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            tr.weighted_quantile(x, np.ones(37), bad)
    with pytest.raises(ValueError):
        tr.weighted_quantile([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError):
        tr.weighted_quantile([1.0, 2.0], [0.0, 0.0], 0.5)


# ----------------------------------------------------------------------------
# wasserstein2 and sinkhorn_divergence
# ----------------------------------------------------------------------------
def test_wasserstein2_of_identical_clouds_is_zero():
    rng = np.random.default_rng(20)
    Z = rng.normal(size=(9, 3))
    w = np.array([2.0, 1.0, 0.5])
    assert tr.wasserstein2(None, None, Z, Z, w) == pytest.approx(0.0, abs=1e-7)
    a = rng.uniform(0.5, 1.5, 9)
    assert tr.wasserstein2(a, a, Z, Z, w) == pytest.approx(0.0, abs=1e-7)
    assert tr.wasserstein2(a / a.sum(), a / a.sum(), pd.DataFrame(Z), pd.DataFrame(Z.copy()), w) == pytest.approx(0.0, abs=1e-7)


def test_wasserstein2_matches_known_values():
    rng = np.random.default_rng(21)
    x, y = rng.normal(size=11), rng.normal(size=11) + 1.0
    expected = np.sqrt(np.mean((np.sort(x) - np.sort(y)) ** 2))
    assert tr.wasserstein2(None, None, x[:, None], y[:, None], [1.0]) == pytest.approx(expected, rel=1e-8)
    Z = rng.normal(size=(8, 3))
    w = np.array([2.0, 1.0, 0.0])
    delta = np.array([0.7, -0.4, 5.0])
    rescaled = w * 3 / w.sum()
    assert tr.wasserstein2(None, None, Z, Z + delta, w) == pytest.approx(np.sqrt((rescaled * delta**2).sum()), rel=1e-8)
    d = tr.wasserstein2([0.25, 0.75], [1.0], np.array([[0.0], [2.0]]), np.array([[1.0]]), [1.0])
    assert d == pytest.approx(1.0)
    a, b = rng.uniform(0.5, 1.5, 6), rng.uniform(0.5, 1.5, 7)
    A, B = rng.normal(size=(6, 2)), rng.normal(size=(7, 2))
    assert tr.wasserstein2(a, b, A, B, [1.0, 2.0]) == pytest.approx(tr.wasserstein2(b, a, B, A, [1.0, 2.0]), rel=1e-9)


def test_wasserstein2_normalises_both_measures_to_unit_mass():
    rng = np.random.default_rng(34)
    A, B = rng.normal(size=(6, 2)), rng.normal(size=(7, 2)) + 1.0
    w = [1.0, 2.0]
    a, b = rng.uniform(0.5, 1.5, 6), rng.uniform(0.5, 1.5, 7)
    reference = tr.wasserstein2(a / a.sum(), b / b.sum(), A, B, w)
    assert reference > 0.5
    for factor_a, factor_b in [(1.0, 1.0), (2.0, 5.0), (1e-6, 1e6), (37.0, 0.01)]:
        assert tr.wasserstein2(a * factor_a, b * factor_b, A, B, w) == pytest.approx(reference, rel=1e-8)
    assert tr.wasserstein2(np.full(6, 3.0), np.full(7, 0.5), A, B, w) == pytest.approx(tr.wasserstein2(None, None, A, B, w), rel=1e-8)
    assert tr.wasserstein2([1.0, 3.0], [10.0], np.array([[0.0], [2.0]]), np.array([[1.0]]), [1.0]) == pytest.approx(1.0)
    assert tr.wasserstein2([5.0], [0.25], np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]), [1.0, 1.0]) == pytest.approx(5.0)
    assert "normalised to unit mass" in flat(tr.wasserstein2.__doc__)


def test_wasserstein2_is_missing_when_a_measure_has_no_mass():
    rng = np.random.default_rng(35)
    A, B = rng.normal(size=(5, 2)), rng.normal(size=(4, 2))
    w = [1.0, 1.0]
    assert np.isnan(tr.wasserstein2(np.zeros(5), None, A, B, w))
    assert np.isnan(tr.wasserstein2(None, np.zeros(4), A, B, w))
    assert np.isnan(tr.wasserstein2(np.zeros(5), np.zeros(4), A, B, w))
    point_to_point = tr.wasserstein2([0, 1, 0, 0, 0], [0, 0, 1, 0], A, B, w)
    assert point_to_point == pytest.approx(np.sqrt(((A[1] - B[2]) ** 2).sum()))


def test_wasserstein2_errors_name_the_argument():
    rng = np.random.default_rng(36)
    A, B = rng.normal(size=(5, 2)), rng.normal(size=(4, 2))
    bad_a, bad_b = A.copy(), B.copy()
    bad_a[0, 0] = np.nan
    bad_b[1, 1] = np.inf
    cases = [
        ({"Za": bad_a}, "Za must be finite"),
        ({"Zb": bad_b}, "Zb must be finite"),
        ({"Zb": rng.normal(size=(4, 3))}, "Za has 2 features but Zb has 3"),
        ({"w": [1.0, 2.0, 3.0]}, "w has 3 entries but there are 2 features"),
        ({"w": [0.0, 0.0]}, "w sums to zero"),
        ({"a": [1.0, 1.0]}, "a has 2 entries, expected 5"),
        ({"a": [-1.0, 1.0, 1.0, 1.0, 1.0]}, "a must be finite and non-negative"),
        ({"b": [1.0, 1.0]}, "b has 2 entries, expected 4"),
        ({"b": [np.nan, 1.0, 1.0, 1.0]}, "b must be finite and non-negative"),
    ]
    for override, message in cases:
        args = {"a": None, "b": None, "Za": A, "Zb": B, "w": [1.0, 1.0]}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.wasserstein2(**args)


def test_wasserstein2_matches_pot_emd():
    ot = pytest.importorskip("ot")
    Zs, Zt, w, a, b, C = random_problem(22, 9, 12)
    assert tr.wasserstein2(a, b, Zs, Zt, w) ** 2 == pytest.approx(float(ot.emd2(a, b, C)), rel=1e-8)


def test_sinkhorn_divergence_is_non_negative_symmetric_and_near_zero_for_identical_measures():
    rng = np.random.default_rng(23)
    A, B = rng.normal(size=(12, 3)), rng.normal(size=(9, 3)) + 0.8
    w = np.array([2.0, 1.0, 0.5])
    a, b = np.full(12, 1 / 12), rng.uniform(0.5, 1.5, 9)
    b = b / b.sum()
    Cab, Caa, Cbb = tr.weighted_sq_cost(A, B, w), tr.weighted_sq_cost(A, A, w), tr.weighted_sq_cost(B, B, w)
    for eps in (3.0, 1.0, 0.3):
        div = tr.sinkhorn_divergence(a, b, Cab, Caa, Cbb, eps)
        assert div > 0.5
        assert div == pytest.approx(tr.sinkhorn_divergence(b, a, Cab.T, Cbb, Caa, eps), rel=1e-9)
        assert tr.sinkhorn_divergence(a, a, Caa, Caa, Caa, eps) == pytest.approx(0.0, abs=1e-9)
        assert tr.sinkhorn_divergence(b, b, Cbb, Cbb, Cbb, eps) == pytest.approx(0.0, abs=1e-9)
    previous = 0.0
    for shift in (0.05, 0.2, 0.8):
        A2 = A + shift
        C2 = tr.weighted_sq_cost(A, A2, w)
        value = tr.sinkhorn_divergence(a, a, C2, Caa, tr.weighted_sq_cost(A2, A2, w), 1.0)
        assert value > previous >= 0.0
        previous = value
    assert tr.sinkhorn_divergence(5 * a, 3 * b, Cab, Caa, Cbb, 1.0) == pytest.approx(tr.sinkhorn_divergence(a, b, Cab, Caa, Cbb, 1.0), rel=1e-12)


def test_sinkhorn_divergence_equals_the_primal_definition_and_tends_to_squared_wasserstein2():
    rng = np.random.default_rng(24)
    A, B = rng.normal(size=(8, 2)), rng.normal(size=(7, 2)) + 1.0
    w = np.array([1.5, 0.5])
    a, b = np.full(8, 1 / 8), np.full(7, 1 / 7)
    Cab, Caa, Cbb = tr.weighted_sq_cost(A, B, w), tr.weighted_sq_cost(A, A, w), tr.weighted_sq_cost(B, B, w)

    def primal(x, y, C, eps):
        plan = tr.sinkhorn_plan(x, y, C, eps, max_iter=200000, tol=1e-13)
        assert plan.converged
        return primal_objective(plan.pi, C, x, y, eps)

    for eps in (2.0, 1.0):
        reference = primal(a, b, Cab, eps) - 0.5 * primal(a, a, Caa, eps) - 0.5 * primal(b, b, Cbb, eps)
        assert tr.sinkhorn_divergence(a, b, Cab, Caa, Cbb, eps) == pytest.approx(reference, abs=1e-9)
    exact = tr.wasserstein2(a, b, A, B, w) ** 2
    values = [tr.sinkhorn_divergence(a, b, Cab, Caa, Cbb, eps) for eps in (2.0, 0.5, 0.1, 0.03)]
    assert all(np.diff(values) > 0) and values[-1] == pytest.approx(exact, rel=0.03) and values[-1] < exact


def test_sinkhorn_divergence_validates_shapes_and_names_the_argument():
    a, b = np.full(3, 1 / 3), np.full(2, 0.5)
    cases = [
        ((a, b, np.ones((3, 3)), np.ones((3, 3)), np.ones((2, 2)), 1.0), "b has 2 entries, expected 3"),
        ((a, b, np.ones((3, 2)), np.ones((2, 2)), np.ones((2, 2)), 1.0), r"C_aa has shape \(2, 2\), expected \(3, 3\)"),
        ((a, b, np.ones((3, 2)), np.ones((3, 3)), np.ones((3, 3)), 1.0), r"C_bb has shape \(3, 3\), expected \(2, 2\)"),
        ((a, b, np.ones((3, 2)), np.ones((3, 3)), np.ones((2, 2)), 0.0), "eps must be positive and finite"),
        ((a[:2], b, np.ones((3, 2)), np.ones((3, 3)), np.ones((2, 2)), 1.0), "a has 2 entries, expected 3"),
        ((-a, b, np.ones((3, 2)), np.ones((3, 3)), np.ones((2, 2)), 1.0), "a must be finite and non-negative"),
        ((a, b, np.full((3, 2), np.nan), np.ones((3, 3)), np.ones((2, 2)), 1.0), "C_ab must be finite"),
        ((a, b, np.ones((3, 2)), np.full((3, 3), np.inf), np.ones((2, 2)), 1.0), "C_aa must be finite"),
        ((a, b, np.ones(3), np.ones((3, 3)), np.ones((2, 2)), 1.0), "C_ab must be two-dimensional"),
    ]
    for args, message in cases:
        with pytest.raises(ValueError, match=message):
            tr.sinkhorn_divergence(*args)
    asym = np.array([[0.0, 1.0, 4.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    value = tr.sinkhorn_divergence(a, b, np.ones((3, 2)), asym, np.zeros((2, 2)), 1.0)
    assert np.isfinite(value)
    rng = np.random.default_rng(7)
    A, B = rng.normal(size=(6, 2)), rng.normal(size=(5, 2))
    args = (np.full(6, 1 / 6), np.full(5, 0.2), tr.weighted_sq_cost(A, B, [1.0, 1.0]), tr.weighted_sq_cost(A, A, [1.0, 1.0]), tr.weighted_sq_cost(B, B, [1.0, 1.0]))
    with pytest.warns(RuntimeWarning, match="did not reach the tolerance"):
        tr.sinkhorn_divergence(*args, 0.05, max_iter=3)


# ----------------------------------------------------------------------------
# select_eps
# ----------------------------------------------------------------------------
def test_select_eps_is_a_tenth_of_the_quantile_of_pairwise_weighted_distances():
    rng = np.random.default_rng(25)
    Z = rng.normal(size=(15, 4))
    w = np.array([3.0, 1.0, 0.0, 2.0])
    rescaled = w * 4 / w.sum()
    pairs = pdist(Z * np.sqrt(rescaled), "sqeuclidean")
    assert tr.select_eps(Z, w) == pytest.approx(0.1 * np.median(pairs), rel=1e-12)
    for q in (0.1, 0.5, 0.9, 1.0):
        assert tr.select_eps(Z, w, quantile=q) == pytest.approx(0.1 * np.quantile(pairs, q), rel=1e-12)
    assert tr.select_eps(pd.DataFrame(Z), 2.5 * w) == pytest.approx(0.1 * np.median(pairs), rel=1e-12)
    assert tr.select_eps(Z, np.ones(4)) == pytest.approx(0.1 * np.median(pdist(Z, "sqeuclidean")), rel=1e-12)
    bad = Z.copy()
    bad[2, 1] = np.nan
    cases = [
        ((Z[:1], w), {}, "Z needs at least two rows"),
        ((Z, w), {"quantile": 1.5}, "quantile must lie between 0 and 1"),
        ((Z, w), {"quantile": np.nan}, "quantile must lie between 0 and 1"),
        ((bad, w), {}, "Z must be finite"),
        ((Z, w[:3]), {}, "w has 3 entries but there are 4 features"),
        ((Z, [1.0, np.nan, 1.0, 1.0]), {}, "w must be finite and non-negative"),
        ((Z, np.zeros(4)), {}, "w sums to zero"),
        ((Z[:, 0], w), {}, "Z needs at least two rows"),
        ((np.ones((2, 2, 2)), w), {}, "Z must be a DataFrame or a two-dimensional array"),
    ]
    for args, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            tr.select_eps(*args, **kwargs)


# ----------------------------------------------------------------------------
# overlap_permutation_test
# ----------------------------------------------------------------------------
def test_overlap_statistic_is_the_weighted_mean_distance_to_the_nearest_source():
    rng = np.random.default_rng(26)
    Zs, Zt = rng.normal(size=(10, 3)), rng.normal(size=(5, 3)) + 1.0
    w = np.array([2.0, 1.0, 0.5])
    b = np.array([1.0, 2.0, 3.0, 0.0, 4.0])
    C = tr.weighted_sq_cost(Zs, Zt, w)
    res = tr.overlap_permutation_test(Zs, Zt, w, b=b, n_perm=30, seed=1)
    assert res["statistic"] == pytest.approx((b * C.min(axis=0)).sum() / b.sum(), rel=1e-12)
    uniform = tr.overlap_permutation_test(Zs, Zt, w, n_perm=30, seed=1)
    assert uniform["statistic"] == pytest.approx(C.min(axis=0).mean(), rel=1e-12)
    a = np.ones(10)
    a[np.argmin(C, axis=0)] = 0.0
    ignored = tr.overlap_permutation_test(Zs, Zt, w, a=a, n_perm=30, seed=1)
    keep = a > 0
    assert ignored["statistic"] == pytest.approx(C[keep].min(axis=0).mean(), rel=1e-12)
    assert ignored["statistic"] > uniform["statistic"]
    assert set(res) >= {"statistic", "p_value", "null", "n_perm"} and res["null"].shape == (30,) and res["n_perm"] == 30


def test_overlap_permutation_test_gives_a_small_p_value_for_a_target_far_outside_the_sources():
    rng = np.random.default_rng(27)
    Zs = rng.normal(size=(25, 3))
    Zt = rng.normal(size=(6, 3)) + 6.0
    pool = rng.normal(size=(600, 3))
    for kwargs in ({}, {"pool": pool}, {"pool": pd.DataFrame(pool)}):
        res = tr.overlap_permutation_test(Zs, Zt, np.ones(3), n_perm=199, seed=0, **kwargs)
        assert res["p_value"] == pytest.approx(1 / 200)
        assert res["statistic"] > 10 * res["null"].max()


def test_overlap_permutation_test_does_not_reject_a_target_from_the_same_distribution():
    p_none, p_pool = [], []
    for rep in range(40):
        rng = np.random.default_rng(1000 + rep)
        Zs, Zt = rng.normal(size=(25, 3)), rng.normal(size=(6, 3))
        pool = rng.normal(size=(500, 3))
        p_none.append(tr.overlap_permutation_test(Zs, Zt, np.ones(3), n_perm=99, seed=rep)["p_value"])
        p_pool.append(tr.overlap_permutation_test(Zs, Zt, np.ones(3), n_perm=99, seed=rep, pool=pool)["p_value"])
    for p in (np.array(p_none), np.array(p_pool)):
        assert 0.35 < p.mean() < 0.65
        assert np.mean(p <= 0.05) <= 0.15
    rng = np.random.default_rng(5)
    Zs, Zt = rng.normal(size=(30, 3)), rng.normal(size=(5, 3))
    assert tr.overlap_permutation_test(Zs, Zt, np.ones(3), seed=0)["p_value"] > 0.1


def test_overlap_permutation_test_is_reproducible_and_validates_inputs():
    rng = np.random.default_rng(28)
    Zs, Zt = rng.normal(size=(12, 2)), rng.normal(size=(4, 2)) + 1.0
    first = tr.overlap_permutation_test(Zs, Zt, [1.0, 1.0], n_perm=50, seed=3)
    again = tr.overlap_permutation_test(Zs, Zt, [1.0, 1.0], n_perm=50, seed=3)
    other = tr.overlap_permutation_test(Zs, Zt, [1.0, 1.0], n_perm=50, seed=4)
    np.testing.assert_array_equal(first["null"], again["null"])
    assert first["p_value"] == again["p_value"] and not np.array_equal(first["null"], other["null"])
    assert 1 / 51 <= first["p_value"] <= 1.0
    bad_s, bad_t, bad_pool = Zs.copy(), Zt.copy(), rng.normal(size=(20, 2))
    bad_s[0, 0] = np.nan
    bad_t[1, 1] = np.inf
    bad_pool[3, 0] = np.nan
    cases = [
        ({"n_perm": 0}, "n_perm must be at least 1"),
        ({"a": np.zeros(12)}, "a has zero total mass"),
        ({"a": np.ones(11)}, "a has 11 entries, expected 12"),
        ({"a": -np.ones(12)}, "a must be finite and non-negative"),
        ({"b": np.ones(3)}, "b has 3 entries, expected 4"),
        ({"b": np.zeros(4)}, "b has zero total mass"),
        ({"pool": rng.normal(size=(20, 3))}, "pool has 3 features, expected 2"),
        ({"pool": bad_pool}, "pool must be finite"),
        ({"Zs": bad_s}, "Zs must be finite"),
        ({"Zt": bad_t}, "Zt must be finite"),
        ({"Zt": rng.normal(size=(4, 3))}, "Zs has 2 features but Zt has 3"),
        ({"w": [1.0]}, "w has 1 entries but there are 2 features"),
    ]
    for override, message in cases:
        args = {"Zs": Zs, "Zt": Zt, "w": [1.0, 1.0]}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.overlap_permutation_test(**args)
    small = tr.overlap_permutation_test(Zs, Zt, [1.0, 1.0], n_perm=20, seed=0, pool=rng.normal(size=(2, 2)))
    assert np.isfinite(small["p_value"])


# ----------------------------------------------------------------------------
# loco_validation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_loco_weighted_transport_beats_uniform_transport_and_the_equal_mean_when_weights_find_the_driver(seed):
    Z, tau, se = informative_problem(seed)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    res = tr.loco_validation(Z, tau, se, w, n_boot=200, seed=1)
    rmse = res.summary["rmse"]
    assert rmse["ot_weighted"] < rmse["ot_uniform"] < rmse["equal"] * 1.05
    assert rmse["ot_weighted"] < 0.7 * rmse["ot_uniform"] and rmse["ot_weighted"] < 0.5 * rmse["equal"]
    assert res.n_nonconverged == 0
    ratios = res.ratios
    assert set(ratios.index) == {"ot_weighted / equal", "ot_weighted / ot_uniform"}
    assert (ratios["hi95"] < 1.0).all() and (ratios["share_below_one"] > 0.99).all()


def test_loco_summary_table_and_ratio_intervals_are_consistent():
    Z, tau, se = informative_problem(0)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    res = tr.loco_validation(Z, tau, se, w, n_boot=300, seed=2)
    methods = ["ot_weighted", "ot_uniform", "equal", "nn1", "nn3", "kernel"]
    assert list(res.summary.index) == methods
    assert list(res.summary.columns)[:5] == ["rmse", "mae", "mean_error", "correlation", "rank"]
    assert list(res.table.columns) == ["case", "method", "tau", "se", "prediction", "error", "group", "n_sources_used"]
    assert len(res.table) == 6 * len(Z)
    assert set(res.table["case"]) == set(Z.index)
    assert (res.table["group"] == res.table["case"]).all() and (res.table["n_sources_used"] == len(Z) - 1).all()
    np.testing.assert_allclose(res.table["error"], res.table["prediction"] - res.table["tau"])
    predictions, errors = res.predictions(), res.errors()
    assert list(predictions.columns) == methods and list(predictions.index) == list(Z.index)
    np.testing.assert_allclose(errors.to_numpy(), predictions.to_numpy() - tau[:, None], atol=1e-12)
    for method in methods:
        e = errors[method].to_numpy()
        row = res.summary.loc[method]
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean(e**2)))
        assert row["mae"] == pytest.approx(np.mean(np.abs(e)))
        assert row["mean_error"] == pytest.approx(np.mean(e))
        assert row["correlation"] == pytest.approx(np.corrcoef(predictions[method], tau)[0, 1])
        assert row["rmse_adj"] == pytest.approx(np.sqrt(max(np.mean(e**2) - np.mean(se**2), 0.0)))
    assert sorted(res.summary["rank"]) == list(range(1, 7))
    assert res.summary["rank"].idxmin() == res.summary["rmse"].idxmin()
    assert res.summary.loc["equal", "correlation"] == pytest.approx(-1.0)
    ratios = res.ratios
    for label in ratios.index:
        row = ratios.loc[label]
        assert row["lo95"] <= row["lo80"] <= row["hi80"] <= row["hi95"]
        assert row["ratio"] == pytest.approx(res.summary.loc[row["numerator"], "rmse"] / res.summary.loc[row["denominator"], "rmse"])
        assert 0.0 <= row["share_below_one"] <= 1.0
    assert res.n_boot == 300 and res.rho_source == 1.0 and set(res.eps) == {"ot_weighted", "ot_uniform"}
    again = tr.loco_validation(Z, tau, se, w, n_boot=300, seed=2)
    pd.testing.assert_frame_equal(again.ratios, ratios)
    different = tr.loco_validation(Z, tau, se, w, n_boot=300, seed=3)
    assert not np.allclose(different.ratios[["lo95", "hi95"]].to_numpy(), ratios[["lo95", "hi95"]].to_numpy())
    pd.testing.assert_frame_equal(different.summary, res.summary)


def test_loco_predictors_equal_their_definitions():
    rng = np.random.default_rng(30)
    n, d = 8, 3
    Z = pd.DataFrame(rng.normal(size=(n, d)), columns=list("xyz"))
    tau = rng.normal(size=n)
    se = rng.uniform(0.05, 0.2, n)
    w = np.array([2.0, 1.0, 0.0])
    a = rng.uniform(0.5, 1.5, n)
    eps_w, eps_u = 0.37, 0.52
    res = tr.loco_validation(Z, tau, se, w, eps=eps_w, a=a, n_boot=20)
    rescaled = w * d / w.sum()
    for i in range(n):
        src = np.delete(np.arange(n), i)
        C = tr.weighted_sq_cost(Z.iloc[src], Z.iloc[[i]], rescaled)
        dist = np.sqrt(C[:, 0])
        pred = res.predictions().iloc[i]
        assert pred["equal"] == pytest.approx(tau[src].mean())
        assert pred["nn1"] == pytest.approx(tau[src][np.argmin(dist)])
        assert pred["nn3"] == pytest.approx(tau[src][np.argsort(dist)[:3]].mean())
        pair = tr.weighted_sq_cost(Z.iloc[src], Z.iloc[src], rescaled)
        h = np.median(np.sqrt(pair[np.triu_indices(n - 1, k=1)]))
        k = np.exp(-0.5 * dist**2 / h**2)
        assert pred["kernel"] == pytest.approx((k * tau[src]).sum() / k.sum())
        a_src = a[src] / a[src].sum()
        plan = tr.sinkhorn_plan(a_src, [1.0], C, eps_w, rho_source=1.0)
        assert pred["ot_weighted"] == pytest.approx(tr.target_effect(plan, tau[src], [1.0]), abs=1e-9)
        C_u = tr.weighted_sq_cost(Z.iloc[src], Z.iloc[[i]], np.ones(d))
        plan_u = tr.sinkhorn_plan(a_src, [1.0], C_u, eps_w, rho_source=1.0)
        assert pred["ot_uniform"] == pytest.approx(tr.target_effect(plan_u, tau[src], [1.0]), abs=1e-9)
    default = tr.loco_validation(Z, tau, se, w, n_boot=5, methods=("ot_weighted", "ot_uniform"))
    assert default.eps["ot_weighted"] == pytest.approx(tr.select_eps(Z, w))
    assert default.eps["ot_uniform"] == pytest.approx(tr.select_eps(Z, np.ones(d)))
    balanced = tr.loco_validation(Z, tau, se, w, rho_source=None, n_boot=5, methods=("ot_weighted",))
    assert balanced.rho_source is None and np.isfinite(balanced.table["prediction"]).all()


def test_loco_cloud_targets():
    Z, tau, se = informative_problem(1, n=14)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    n = len(Z)
    copies = [np.repeat(Z.to_numpy()[[i]], 4, axis=0) for i in range(n)]
    single = tr.loco_validation(Z, tau, se, w, eps=0.3, n_boot=10)
    cloud = tr.loco_validation(Z, tau, se, w, clouds=copies, eps=0.3, n_boot=10)
    np.testing.assert_allclose(cloud.predictions().to_numpy(), single.predictions().to_numpy(), atol=1e-8)
    rng = np.random.default_rng(31)
    clouds = {label: Z.loc[label].to_numpy() + 0.2 * rng.normal(size=(5, 4)) for label in Z.index}
    res = tr.loco_validation(Z, tau, se, w, clouds=clouds, eps=0.3, n_boot=10)
    i = 3
    src = np.delete(np.arange(n), i)
    target = clouds[Z.index[i]]
    C = tr.weighted_sq_cost(Z.iloc[src], target, w)
    pred = res.predictions().iloc[i]
    assert pred["nn1"] == pytest.approx(tau[src][np.argmin(C, axis=0)].mean())
    assert pred["equal"] == pytest.approx(tau[src].mean())
    plan = tr.sinkhorn_plan(np.full(n - 1, 1 / (n - 1)), np.full(5, 0.2), C, 0.3, rho_source=1.0)
    assert pred["ot_weighted"] == pytest.approx(tr.target_effect(plan, tau[src], np.full(5, 0.2)), abs=1e-9)
    assert res.summary.loc["ot_weighted", "rmse"] < res.summary.loc["equal", "rmse"]
    frames = [pd.DataFrame(clouds[label], columns=Z.columns)[["f3", "f1", "f0", "f2"]] for label in Z.index]
    permuted = tr.loco_validation(Z, tau, se, w, clouds=frames, eps=0.3, n_boot=10)
    np.testing.assert_allclose(permuted.predictions().to_numpy(), res.predictions().to_numpy(), atol=1e-12)
    with pytest.raises(ValueError):
        tr.loco_validation(Z, tau, se, w, clouds=copies[:-1])
    with pytest.raises(KeyError):
        tr.loco_validation(Z, tau, se, w, clouds={"nope": copies[0]})
    with pytest.raises(ValueError):
        tr.loco_validation(Z, tau, se, w, clouds=[c[:, :3] for c in copies])


def test_loco_validation_supports_method_subsets_and_labelled_inputs():
    Z, tau, se = informative_problem(2, n=10)
    w = np.ones(4)
    only = tr.loco_validation(Z, tau, se, w, methods=("nn1", "equal"), n_boot=10)
    assert list(only.summary.index) == ["nn1", "equal"] and only.ratios.empty and only.eps == {}
    one_ratio = tr.loco_validation(Z, tau, None, w, methods=("ot_weighted", "equal"), n_boot=10)
    assert list(one_ratio.ratios.index) == ["ot_weighted / equal"]
    assert one_ratio.table["se"].isna().all() and one_ratio.summary["rmse_adj"].isna().all()
    arrays = tr.loco_validation(Z.to_numpy(), tau, se, w, n_boot=10)
    assert list(arrays.predictions().index) == list(range(10))
    shuffled = tr.loco_validation(Z, tau, se, pd.Series([0.05, 1.0, 0.05, 0.05], index=["f3", "f0", "f2", "f1"]), n_boot=10)
    direct = tr.loco_validation(Z, tau, se, np.array([1.0, 0.05, 0.05, 0.05]), n_boot=10)
    pd.testing.assert_frame_equal(shuffled.summary, direct.summary)


def test_loco_validation_checks_its_arguments_and_names_them():
    Z, tau, se = informative_problem(16, n=12)
    w = np.array([1.0, 0.5, 0.3, 0.2])
    at = np.arange(12)
    bad_features = Z.copy()
    bad_features.iloc[3, 1] = np.nan
    cases = [
        ({"methods": ("equal", "magic")}, "unknown methods"),
        ({"methods": ("equal", "equal")}, "distinct"),
        ({"n_boot": 0}, "n_boot must be at least 1"),
        ({"eps": -1.0}, "eps must be positive and finite"),
        ({"rho_source": 0.0}, "rho_source must be positive or None"),
        ({"tau": tau[:-1]}, "tau has 11 entries, expected 12"),
        ({"tau": np.ones((12, 1))}, "tau must be one-dimensional"),
        ({"tau": np.where(at == 2, np.nan, tau)}, "tau must be finite"),
        ({"tau": np.where(at == 2, np.inf, tau)}, "tau must be finite"),
        ({"se": se[:-1]}, "se has 11 entries, expected 12"),
        ({"se": np.where(at == 0, np.nan, se)}, "se must be finite"),
        ({"se": np.zeros(12)}, "se must be positive"),
        ({"se": -se}, "se must be positive"),
        ({"w": w[:3]}, "w has 3 entries but there are 4 features"),
        ({"w": np.array([1.0, np.nan, 0.3, 0.2])}, "w must be finite and non-negative"),
        ({"w": np.array([1.0, -0.5, 0.3, 0.2])}, "w must be finite and non-negative"),
        ({"w": np.zeros(4)}, "w sums to zero"),
        ({"w": np.ones((4, 1))}, "w must be one-dimensional"),
        ({"a": np.ones(11)}, "a has 11 entries, expected 12"),
        ({"a": -np.ones(12)}, "a must be finite and non-negative"),
        ({"a": np.zeros(12)}, "a has zero total mass"),
        ({"groups": list("aabbccddeef")}, "groups has 11 entries, expected 12"),
        ({"Z": bad_features}, "Z must be finite"),
        ({"Z": Z.to_numpy()[:, 0]}, "Z must be a DataFrame or a two-dimensional array"),
        ({"Z": Z.iloc[:1], "tau": tau[:1], "se": se[:1]}, "Z needs at least two cases"),
        ({"clouds": [np.full((2, 4), np.nan)] + [np.zeros((2, 4))] * 11}, "clouds must be finite"),
        ({"clouds": [np.zeros((2, 3))] * 12}, "each entry of clouds needs at least one row and 4 columns"),
        ({"clouds": [np.zeros((2, 4))] * 11}, "clouds has 11 entries, expected 12"),
    ]
    for override, message in cases:
        args = {"Z": Z, "tau": tau, "se": se, "w": w, "n_boot": 5}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.loco_validation(**args)


def test_loco_prediction_for_a_case_does_not_use_its_own_effect():
    Z, tau, se = informative_problem(3, n=12)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    rng = np.random.default_rng(32)
    clouds = [Z.to_numpy()[i] + 0.2 * rng.normal(size=(3, 4)) for i in range(len(Z))]
    for cloud_arg in (None, clouds):
        base = tr.loco_validation(Z, tau, se, w, clouds=cloud_arg, n_boot=5).predictions()
        for i in (0, 5, 11):
            changed = tau.copy()
            changed[i] += 7.0
            other = tr.loco_validation(Z, changed, se, w, clouds=cloud_arg, n_boot=5).predictions()
            np.testing.assert_allclose(other.iloc[i].to_numpy(), base.iloc[i].to_numpy(), atol=1e-12)
            others = np.delete(np.arange(len(Z)), i)
            assert np.all(np.abs(other.iloc[others]["equal"].to_numpy() - base.iloc[others]["equal"].to_numpy()) > 1e-3)


def test_loco_default_eps_needs_distinct_cases():
    Z = np.zeros((5, 2))
    tau = np.arange(5.0)
    with pytest.raises(ValueError, match="eps"):
        tr.loco_validation(Z, tau, None, [1.0, 1.0], methods=("ot_weighted",), n_boot=5)
    res = tr.loco_validation(Z, tau, None, [1.0, 1.0], eps=0.5, methods=("ot_weighted", "equal"), n_boot=5)
    np.testing.assert_allclose(res.predictions()["ot_weighted"], res.predictions()["equal"], atol=1e-9)
    with pytest.raises(ValueError):
        tr.loco_validation(Z, tau, None, [1.0, 1.0], a=[1.0, 0.0, 0.0, 0.0, 0.0], methods=("equal",), n_boot=5)


# ----------------------------------------------------------------------------
# loco_validation: weights learned in the fold, regularisation per fold, sources without mass
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("use_groups", [False, True])
def test_loco_weight_function_is_called_once_per_held_out_group_with_the_training_positions(use_groups):
    Z, tau, se = informative_problem(9, n=12)
    groups = np.repeat(np.array(list("cadb")), 3) if use_groups else None
    calls = []

    def learn(train_idx):
        calls.append(np.array(train_idx))
        return np.array([1.0, 0.1, 0.1, 0.1])

    tr.loco_validation(Z, tau, se, learn, groups=groups, n_boot=5)
    if use_groups:
        held_out = [np.flatnonzero(groups == label) for label in "cadb"]
        expected = [np.flatnonzero(groups != label) for label in "cadb"]
    else:
        held_out = [np.array([i]) for i in range(12)]
        expected = [np.delete(np.arange(12), i) for i in range(12)]
    assert len(calls) == len(expected)
    for got, want, own in zip(calls, expected, held_out):
        assert got.dtype.kind == "i"
        np.testing.assert_array_equal(got, want)
        assert np.intersect1d(got, own).size == 0


def test_loco_weight_function_receives_a_copy_and_is_skipped_when_no_method_uses_weights():
    Z, tau, se = informative_problem(10, n=10)
    fixed = np.array([1.0, 0.1, 0.1, 0.1])

    def clobber(train_idx):
        train_idx[:] = 0
        return fixed

    changed = tr.loco_validation(Z, tau, se, clobber, n_boot=5)
    reference = tr.loco_validation(Z, tau, se, fixed, n_boot=5)
    pd.testing.assert_frame_equal(changed.table, reference.table, check_exact=True)
    calls = []
    res = tr.loco_validation(Z, tau, se, lambda idx: calls.append(idx), methods=("equal", "ot_uniform"), eps=0.5, n_boot=5)
    assert calls == [] and list(res.summary.index) == ["equal", "ot_uniform"]


@pytest.mark.parametrize("use_groups", [False, True])
def test_loco_with_a_weight_function_equals_the_definitions_with_the_weights_of_each_fold(use_groups):
    rng = np.random.default_rng(38)
    n, d = 16, 6
    Z, tau = rng.normal(size=(n, d)), rng.normal(size=n)
    groups = np.repeat(np.arange(8), 2)

    def learn(train_idx):
        return learned_weights(Z[train_idx], tau[train_idx])

    res = tr.loco_validation(pd.DataFrame(Z), tau, None, learn, groups=groups if use_groups else None, eps=0.35, n_boot=5)
    predictions = res.predictions()
    fold_weights = set()
    for i in range(n):
        src = np.flatnonzero(groups != groups[i]) if use_groups else np.delete(np.arange(n), i)
        w_i = learned_weights(Z[src], tau[src])
        fold_weights.add(tuple(w_i))
        for method, value in reference_predictions(Z, tau, src, i, w_i, 0.35, 0.35).items():
            assert predictions.iloc[i][method] == pytest.approx(value, abs=1e-9), (i, method)
    assert len(fold_weights) > 2


@pytest.mark.parametrize(
    "returned,message",
    [
        (np.array([1.0, np.nan, 1.0, 1.0]), r"w\(train_idx\) must be finite and non-negative"),
        (np.array([1.0, np.inf, 1.0, 1.0]), r"w\(train_idx\) must be finite and non-negative"),
        (np.array([1.0, -1.0, 1.0, 1.0]), r"w\(train_idx\) must be finite and non-negative"),
        (np.zeros(4), r"w\(train_idx\) sums to zero"),
        (np.ones(3), r"w\(train_idx\) has 3 entries but there are 4 features"),
        (np.ones((4, 1)), r"w\(train_idx\) must be one-dimensional"),
        (["a", "b", "c", "d"], r"w\(train_idx\) must be numeric"),
    ],
)
def test_loco_weight_function_must_return_valid_weights(returned, message):
    Z, tau, se = informative_problem(11, n=8)
    with pytest.raises(ValueError, match=message):
        tr.loco_validation(Z, tau, se, lambda train_idx: returned, n_boot=5)


def test_a_constant_weight_function_reproduces_fixed_weights_and_a_series_is_matched_by_name():
    Z, tau, se = informative_problem(42, n=12)
    w = np.array([1.0, 0.4, 0.2, 0.1])
    fixed = tr.loco_validation(Z, tau, se, w, n_boot=30, seed=2)
    named = pd.Series(w[::-1], index=["f3", "f2", "f1", "f0"])
    for learn in (lambda idx: w, lambda idx: named):
        res = tr.loco_validation(Z, tau, se, learn, n_boot=30, seed=2)
        pd.testing.assert_frame_equal(res.table, fixed.table, check_exact=True)
        pd.testing.assert_frame_equal(res.summary, fixed.summary, check_exact=True)
        pd.testing.assert_frame_equal(res.ratios, fixed.ratios, check_exact=True)
        assert res.eps == fixed.eps
    with pytest.raises(KeyError, match="w\\(train_idx\\)"):
        tr.loco_validation(Z, tau, se, lambda idx: named.drop("f1"), n_boot=5)


def test_weights_learned_from_the_validated_cases_make_the_validation_optimistic():
    n, d = 24, 10
    gains = []
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        Z, tau, se = rng.normal(size=(n, d)), rng.normal(size=n), np.full(n, 0.1)
        kwargs = {"methods": ("ot_weighted", "kernel"), "eps": 0.3, "n_boot": 1}
        from_all = tr.loco_validation(Z, tau, se, learned_weights(Z, tau), **kwargs).summary["rmse"]
        in_fold = tr.loco_validation(Z, tau, se, lambda idx: learned_weights(Z[idx], tau[idx]), **kwargs).summary["rmse"]
        gains.append(in_fold["ot_weighted"] - from_all["ot_weighted"])
    assert np.mean(gains) > 0.05 and np.sum(np.array(gains) > 0) >= 8
    assert "optimistic" in tr.loco_validation.__doc__


@pytest.mark.parametrize("variant", ["groups", "clouds", "masses"])
def test_loco_default_eps_comes_from_the_training_sources_and_the_held_out_target_of_each_fold(variant):
    rng = np.random.default_rng(35)
    groups = np.repeat(np.arange(8), 2)
    n, d = groups.size, 3
    Z = rng.normal(size=(n, d))
    Z[groups == 3] += 4.0
    tau = rng.normal(size=n)
    w = np.array([2.0, 1.0, 0.5])
    clouds = [Z[i] + 0.3 * rng.normal(size=(3, d)) for i in range(n)] if variant == "clouds" else None
    a = None
    if variant == "masses":
        a = rng.uniform(0.5, 1.5, n)
        a[[1, 6]] = 0.0
    res = tr.loco_validation(pd.DataFrame(Z), tau, None, w, groups=groups, clouds=clouds, a=a, n_boot=5)
    predictions = res.predictions()
    global_eps = tr.select_eps(Z, w)
    eps_w, eps_u, shift = [], [], []
    for i in range(n):
        outside = np.flatnonzero(groups != groups[i])
        src = outside if a is None else outside[a[outside] > 0]
        target = Z[[i]] if clouds is None else clouds[i]
        pooled = np.vstack([Z[src], target])
        eps_w.append(tr.select_eps(pooled, w))
        eps_u.append(tr.select_eps(pooled, np.ones(d)))
        expected = reference_predictions(Z, tau, src, i, w, eps_w[-1], eps_u[-1], a=a, target=target)
        for method in ("ot_weighted", "ot_uniform"):
            assert predictions.iloc[i][method] == pytest.approx(expected[method], abs=1e-9), (i, method)
        from_all = reference_predictions(Z, tau, src, i, w, global_eps, global_eps, a=a, target=target)
        shift.append(abs(from_all["ot_weighted"] - predictions.iloc[i]["ot_weighted"]))
    assert res.eps["ot_weighted"] == pytest.approx(np.median(eps_w), rel=1e-12)
    assert res.eps["ot_uniform"] == pytest.approx(np.median(eps_u), rel=1e-12)
    assert np.max(np.abs(np.array(eps_w) - global_eps)) > 0.02 * global_eps
    assert max(shift) > 1e-3


def test_loco_default_eps_ignores_the_features_of_the_group_mates_of_the_held_out_case():
    Z, tau, se, groups = economy_problem(36, n_groups=8)
    w = np.array([1.0, 0.5, 0.3, 0.2])
    assert groups[0] == groups[1]
    moved = Z.copy()
    moved.iloc[1] += 6.0
    base = tr.loco_validation(Z, tau, se, w, groups=groups, n_boot=5)
    other = tr.loco_validation(moved, tau, se, w, groups=groups, n_boot=5)
    np.testing.assert_array_equal(other.predictions().loc["case0"].to_numpy(), base.predictions().loc["case0"].to_numpy())
    assert other.eps != base.eps


def test_loco_baselines_and_regularisation_never_use_sources_without_mass():
    rng = np.random.default_rng(37)
    n, d = 14, 3
    Z = rng.normal(size=(n, d))
    tau = rng.normal(size=n)
    w = np.array([2.0, 1.0, 0.5])
    a = np.ones(n)
    a[[2, 9]] = 0.0
    frame = pd.DataFrame(Z, columns=list("xyz"))
    res = tr.loco_validation(frame, tau, None, w, a=a, eps=0.4, n_boot=5)
    predictions = res.predictions()
    one_method = res.table[res.table["method"] == "equal"].reset_index(drop=True)
    for i in range(n):
        src = np.flatnonzero((np.arange(n) != i) & (a > 0))
        for method, value in reference_predictions(Z, tau, src, i, w, 0.4, 0.4, a=a).items():
            assert predictions.iloc[i][method] == pytest.approx(value, abs=1e-9), (i, method)
        assert one_method["n_sources_used"].iloc[i] == src.size
    base = tr.loco_validation(frame, tau, None, w, a=a, n_boot=5).predictions()
    changed_effect = tau.copy()
    changed_effect[2] += 100.0
    effect_changed = tr.loco_validation(frame, changed_effect, None, w, a=a, n_boot=5).predictions()
    np.testing.assert_array_equal(effect_changed.to_numpy(), base.to_numpy())
    moved = frame.copy()
    moved.iloc[2] += 5.0
    features_changed = tr.loco_validation(moved, tau, None, w, a=a, n_boot=5).predictions()
    others = np.delete(np.arange(n), 2)
    np.testing.assert_array_equal(features_changed.iloc[others].to_numpy(), base.iloc[others].to_numpy())
    assert not np.array_equal(features_changed.iloc[2].to_numpy(), base.iloc[2].to_numpy())
    far = tr.loco_validation(frame, tau, None, w, a=a, eps=1e9, n_boot=5, methods=("ot_uniform", "equal")).predictions()
    positive = np.flatnonzero((np.arange(n) != 5) & (a > 0))
    assert far.iloc[5]["equal"] == pytest.approx(tau[positive].mean())
    assert far.iloc[5]["ot_uniform"] == pytest.approx(tau[positive].mean(), abs=1e-6)


# ----------------------------------------------------------------------------
# leave_group_out_indices and grouped validation
# ----------------------------------------------------------------------------
def test_leave_group_out_indices_returns_the_cases_outside_each_group():
    assert "leave_group_out_indices" in tr.__all__
    groups = ["a", "b", "b", "c", "c", "c", "a"]
    out = tr.leave_group_out_indices(groups)
    assert len(out) == 7 and all(isinstance(src, np.ndarray) for src in out)
    expected = {"a": [1, 2, 3, 4, 5], "b": [0, 3, 4, 5, 6], "c": [0, 1, 2, 6]}
    for label, src in zip(groups, out):
        np.testing.assert_array_equal(src, expected[label])
    for i, src in enumerate(tr.leave_group_out_indices(np.arange(5))):
        np.testing.assert_array_equal(src, np.delete(np.arange(5), i))
    assert [src.size for src in tr.leave_group_out_indices(pd.Series([1.0, 1.0, 2.0]))] == [1, 1, 2]
    assert all(src.size == 0 for src in tr.leave_group_out_indices(["x", "x"]))
    assert tr.leave_group_out_indices([]) == []


def test_leave_group_out_indices_rejects_missing_and_multidimensional_labels():
    with pytest.raises(ValueError, match="missing"):
        tr.leave_group_out_indices(["a", None, "b"])
    with pytest.raises(ValueError, match="missing"):
        tr.leave_group_out_indices([1.0, np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        tr.leave_group_out_indices(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        tr.leave_group_out_indices(3)


@pytest.mark.parametrize("variant", ["plain", "clouds_and_masses"])
def test_loco_with_one_case_per_group_reproduces_the_leave_one_case_out_results_exactly(variant):
    Z, tau, se = informative_problem(4, n=16)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    n = len(Z)
    kwargs = {"n_boot": 120, "seed": 5}
    if variant == "clouds_and_masses":
        rng = np.random.default_rng(50)
        kwargs.update(clouds=[Z.to_numpy()[i] + 0.2 * rng.normal(size=(3, 4)) for i in range(n)], a=rng.uniform(0.5, 1.5, n), rho_source=None)
    reference = tr.loco_validation(Z, tau, se, w, **kwargs)
    assert list(reference.table["group"]) == list(Z.index) * 6
    for groups in (np.arange(n), [f"economy{i}" for i in range(n)], list(Z.index)):
        res = tr.loco_validation(Z, tau, se, w, groups=groups, **kwargs)
        assert list(res.table["group"]) == list(groups) * 6
        pd.testing.assert_frame_equal(res.table.drop(columns="group"), reference.table.drop(columns="group"), check_exact=True)
        pd.testing.assert_frame_equal(res.summary, reference.summary, check_exact=True)
        pd.testing.assert_frame_equal(res.ratios, reference.ratios, check_exact=True)
        assert res.eps == reference.eps and res.n_nonconverged == reference.n_nonconverged
    equal = reference.predictions()["equal"].to_numpy()
    np.testing.assert_allclose(equal, [np.delete(tau, i).mean() for i in range(n)], rtol=1e-12)


def test_loco_predictors_with_groups_equal_their_definitions_on_the_cases_outside_the_group():
    rng = np.random.default_rng(33)
    d = 3
    groups = np.array(["a", "a", "b", "c", "c", "c", "d", "e", "e", "f"])
    n = groups.size
    Z = pd.DataFrame(rng.normal(size=(n, d)), columns=list("xyz"))
    tau = rng.normal(size=n)
    se = rng.uniform(0.05, 0.2, n)
    w = np.array([2.0, 1.0, 0.0])
    a = rng.uniform(0.5, 1.5, n)
    eps = 0.37
    res = tr.loco_validation(Z, tau, se, w, eps=eps, a=a, groups=groups, n_boot=20)
    predictions = res.predictions()
    rescaled = w * d / w.sum()
    for i in range(n):
        src = np.flatnonzero(groups != groups[i])
        assert i not in src and not np.any(groups[src] == groups[i])
        C = tr.weighted_sq_cost(Z.iloc[src], Z.iloc[[i]], rescaled)
        dist = np.sqrt(C[:, 0])
        pred = predictions.iloc[i]
        assert pred["equal"] == pytest.approx(tau[src].mean())
        assert pred["nn1"] == pytest.approx(tau[src][np.argmin(dist)])
        assert pred["nn3"] == pytest.approx(tau[src][np.argsort(dist)[:3]].mean())
        pair = tr.weighted_sq_cost(Z.iloc[src], Z.iloc[src], rescaled)
        h = np.median(np.sqrt(pair[np.triu_indices(src.size, k=1)]))
        k = np.exp(-0.5 * dist**2 / h**2)
        assert pred["kernel"] == pytest.approx((k * tau[src]).sum() / k.sum())
        a_src = a[src] / a[src].sum()
        plan = tr.sinkhorn_plan(a_src, [1.0], C, eps, rho_source=1.0)
        assert pred["ot_weighted"] == pytest.approx(tr.target_effect(plan, tau[src], [1.0]), abs=1e-9)
        C_u = tr.weighted_sq_cost(Z.iloc[src], Z.iloc[[i]], np.ones(d))
        plan_u = tr.sinkhorn_plan(a_src, [1.0], C_u, eps, rho_source=1.0)
        assert pred["ot_uniform"] == pytest.approx(tr.target_effect(plan_u, tau[src], [1.0]), abs=1e-9)
    one_method = res.table[res.table["method"] == "equal"].reset_index(drop=True)
    assert list(one_method["group"]) == list(groups)
    assert list(one_method["n_sources_used"]) == [8, 8, 9, 7, 7, 7, 9, 8, 8, 9]
    assert (res.table.groupby("case")["n_sources_used"].nunique() == 1).all()
    assert list(res.table.columns) == ["case", "method", "tau", "se", "prediction", "error", "group", "n_sources_used"]


def test_loco_never_shows_a_case_the_effect_of_its_group_mate():
    Z, tau, se, groups = economy_problem(3, n_groups=8)
    w = np.array([1.0, 0.5, 0.3, 0.2])
    assert groups[0] == groups[1] and groups[2] != groups[1]
    changed = tau.copy()
    changed[1] += 5.0
    kwargs = {"eps": 0.2, "rho_source": 0.3, "n_boot": 5}
    for use_groups in (True, False):
        given = groups if use_groups else None
        base = tr.loco_validation(Z, tau, se, w, groups=given, **kwargs).predictions()
        other = tr.loco_validation(Z, changed, se, w, groups=given, **kwargs).predictions()
        for method in base.columns:
            unchanged = np.array_equal(other.loc["case0", method], base.loc["case0", method])
            assert unchanged == use_groups, method
            if method in ("equal", "kernel", "ot_weighted", "ot_uniform"):
                assert not np.array_equal(other.iloc[2:][method].to_numpy(), base.iloc[2:][method].to_numpy())
    leaky = tr.loco_validation(Z, changed, se, w, **kwargs).predictions()
    assert leaky.loc["case0", "nn1"] == changed[1]


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_leaving_group_mates_in_makes_the_weighted_transport_look_unrealistically_good(seed):
    Z, tau, se, groups = economy_problem(seed)
    w = np.array([1.0, 0.5, 0.3, 0.2])
    kwargs = {"eps": 0.05, "rho_source": 0.1, "n_boot": 300, "seed": 1}
    leaky = tr.loco_validation(Z, tau, se, w, **kwargs)
    clean = tr.loco_validation(Z, tau, se, w, groups=groups, **kwargs)
    leaky_rmse, clean_rmse = leaky.summary["rmse"], clean.summary["rmse"]
    assert leaky_rmse["nn1"] == 0.0
    assert leaky_rmse["ot_weighted"] < 0.2 * leaky_rmse["equal"]
    assert leaky.ratios.loc["ot_weighted / equal", "hi95"] < 0.3
    assert clean_rmse["ot_weighted"] > clean_rmse["equal"] and clean_rmse["nn1"] > clean_rmse["equal"]
    row = clean.ratios.loc["ot_weighted / equal"]
    assert row["ratio"] > 1.0 and row["hi95"] > 1.0 and row["share_below_one"] < 0.5
    assert (leaky.table["n_sources_used"] == 23).all() and (clean.table["n_sources_used"] == 22).all()
    assert leaky.n_nonconverged == 0 and clean.n_nonconverged == 0


def test_loco_ratio_intervals_resample_whole_groups():
    Z, tau, se, groups = economy_problem(11)
    w = np.array([1.0, 0.5, 0.3, 0.2])
    n_boot, seed = 300, 9
    res = tr.loco_validation(Z, tau, se, w, groups=groups, eps=0.05, rho_source=0.1, n_boot=n_boot, seed=seed)
    errors = res.errors()
    codes = pd.factorize(groups)[0]
    n_groups = codes.max() + 1
    drawn = np.random.default_rng(seed).integers(0, n_groups, size=(n_boot, n_groups))
    widths = {}
    for other in ("equal", "ot_uniform"):
        squared_w = errors["ot_weighted"].to_numpy() ** 2
        squared_o = errors[other].to_numpy() ** 2
        boot = []
        for row in drawn:
            cases = np.concatenate([np.flatnonzero(codes == g) for g in row])
            boot.append(np.sqrt(squared_w[cases].mean() / squared_o[cases].mean()))
        boot = np.array(boot)
        expected = np.percentile(boot, [2.5, 10, 90, 97.5])
        got = res.ratios.loc[f"ot_weighted / {other}"]
        np.testing.assert_allclose([got["lo95"], got["lo80"], got["hi80"], got["hi95"]], expected, rtol=1e-10)
        assert got["share_below_one"] == pytest.approx(np.mean(boot < 1.0))
        widths[other] = got["hi95"] - got["lo95"]
    cases_drawn = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    squared_w, squared_e = errors["ot_weighted"].to_numpy() ** 2, errors["equal"].to_numpy() ** 2
    by_case = np.sqrt(squared_w[cases_drawn].mean(axis=1) / squared_e[cases_drawn].mean(axis=1))
    low, high = np.percentile(by_case, [2.5, 97.5])
    assert widths["equal"] > 1.15 * (high - low)


def test_loco_validation_checks_the_groups_argument():
    Z, tau, se = informative_problem(6, n=8)
    w = np.ones(4)
    with pytest.raises(ValueError, match="entries"):
        tr.loco_validation(Z, tau, se, w, groups=list("aabbccd"))
    with pytest.raises(ValueError, match="missing"):
        tr.loco_validation(Z, tau, se, w, groups=["a", "a", "b", "b", "c", "c", None, "d"])
    with pytest.raises(ValueError, match="one-dimensional"):
        tr.loco_validation(Z, tau, se, w, groups=np.zeros((8, 2)))
    with pytest.raises(ValueError, match="two distinct"):
        tr.loco_validation(Z, tau, se, w, groups=["a"] * 8)
    a = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside the group"):
        tr.loco_validation(Z, tau, se, w, a=a, groups=["x", "x", "y", "y", "z", "z", "u", "u"], n_boot=5)
    assert np.isfinite(tr.loco_validation(Z, tau, se, w, a=a, n_boot=5).table["prediction"]).all()


@pytest.mark.parametrize("use_groups", [False, True])
def test_loco_guard_is_called_once_per_held_out_case(use_groups):
    Z, tau, se = informative_problem(7, n=9)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    groups = np.repeat(np.arange(3), 3) if use_groups else None
    calls = []
    res = tr.loco_validation(Z, tau, se, w, groups=groups, n_boot=10, guard=lambda: calls.append(len(calls)))
    assert calls == list(range(9))
    plain = tr.loco_validation(Z, tau, se, w, groups=groups, n_boot=10)
    pd.testing.assert_frame_equal(res.table, plain.table, check_exact=True)
    pd.testing.assert_frame_equal(res.ratios, plain.ratios, check_exact=True)


def test_guard_exceptions_end_the_computation_and_guards_must_be_callable():
    class StopComputation(Exception):
        """Raised by the guards below."""

    rng = np.random.default_rng(63)
    Z, tau, se = informative_problem(8, n=9)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    seen = []

    def stop_at_four():
        seen.append(1)
        if len(seen) == 4:
            raise StopComputation

    with pytest.raises(StopComputation):
        tr.loco_validation(Z, tau, se, w, n_boot=10, guard=stop_at_four)
    assert len(seen) == 4
    seen.clear()

    def stop_at_two():
        seen.append(1)
        if len(seen) == 2:
            raise StopComputation

    Zs, Zt = rng.normal(size=(10, 2)), rng.normal(size=(2, 2))
    with pytest.raises(StopComputation):
        tr.bootstrap_transport(Zs, Zt, [1.0, 1.0], rng.normal(size=10), np.full(10, 0.1), n_boot=500, guard=stop_at_two)
    assert len(seen) == 2
    with pytest.raises(TypeError, match="guard"):
        tr.loco_validation(Z, tau, se, w, guard="not callable")
    with pytest.raises(TypeError, match="guard"):
        tr.bootstrap_transport(Zs, Zt, [1.0, 1.0], np.ones(10), np.ones(10), guard=3)


# ----------------------------------------------------------------------------
# loco_predictive_draws
# ----------------------------------------------------------------------------
def test_loco_predictive_draws_are_the_estimate_minus_resampled_prediction_errors():
    table, _ = logo_table()
    errors = table["error"].to_numpy()
    out = tr.loco_predictive_draws(table, 1.25, n_draws=500, seed=3)
    assert set(out) == {"draws", "percentiles", "estimate", "n_draws", "n_groups", "n_cases"}
    draws = out["draws"]
    assert draws.shape == (500,) and out["n_draws"] == 500 and out["estimate"] == 1.25
    assert out["n_cases"] == 20 and out["n_groups"] == 10
    assert np.isin(np.round(draws, 12), np.round(1.25 - errors, 12)).all()
    assert list(out["percentiles"].index) == [5, 25, 50, 75, 95]
    np.testing.assert_allclose(out["percentiles"].to_numpy(), np.percentile(draws, [5, 25, 50, 75, 95]))
    again = tr.loco_predictive_draws(table, 1.25, n_draws=500, seed=3)
    np.testing.assert_array_equal(again["draws"], draws)
    assert not np.array_equal(tr.loco_predictive_draws(table, 1.25, n_draws=500, seed=4)["draws"], draws)
    assert tr.loco_predictive_draws(table, 1.25, n_draws=7)["draws"].shape == (7,)


def test_loco_predictive_draws_subtract_the_error_because_it_is_prediction_minus_observed():
    table = pd.DataFrame({"error": np.full(6, 2.0)})
    out = tr.loco_predictive_draws(table, 10.0, n_draws=50)
    assert np.all(out["draws"] == 8.0) and np.all(out["percentiles"] == 8.0)
    mixed = pd.DataFrame({"error": [2.0, -3.0]})
    assert set(np.unique(tr.loco_predictive_draws(mixed, 10.0, n_draws=200)["draws"])) == {8.0, 13.0}


def test_loco_predictive_draws_resample_groups_and_then_pick_one_error_from_the_resampled_cases():
    sizes = np.array([1, 2, 5])
    errors = np.repeat([10.0, 20.0, 30.0], sizes)
    labels = np.repeat(["a", "b", "c"], sizes)
    table = pd.DataFrame({"error": errors, "group": labels})
    expected = np.zeros(3)
    for sequence in itertools.product(range(3), repeat=3):
        pool = np.bincount(sequence, minlength=3) * sizes
        expected += pool / pool.sum()
    expected /= 27
    out = tr.loco_predictive_draws(table, 100.0, n_draws=40000, seed=5)
    frequency = np.array([np.mean(out["draws"] == 100.0 - e) for e in (10.0, 20.0, 30.0)])
    np.testing.assert_allclose(frequency, expected, atol=0.012)
    assert abs(frequency[2] - sizes[2] / sizes.sum()) > 0.05 and abs(frequency[0] - 1 / 3) > 0.05
    assert out["n_groups"] == 3 and out["n_cases"] == 8
    by_argument = tr.loco_predictive_draws(table.drop(columns="group"), 100.0, groups=labels, n_draws=40000, seed=5)
    np.testing.assert_array_equal(by_argument["draws"], out["draws"])
    override = tr.loco_predictive_draws(table, 100.0, groups=np.arange(8), n_draws=40000, seed=5)
    assert override["n_groups"] == 8 and not np.array_equal(override["draws"], out["draws"])


def test_loco_predictive_draws_treat_every_row_as_a_group_without_labels_and_summarise_the_errors():
    rng = np.random.default_rng(51)
    errors = rng.normal(0.5, 2.0, 200)
    table = pd.DataFrame({"error": errors})
    out = tr.loco_predictive_draws(table, 3.0, n_draws=20000, seed=1)
    assert out["n_groups"] == 200 and out["n_cases"] == 200
    np.testing.assert_array_equal(out["draws"], tr.loco_predictive_draws(table, 3.0, groups=np.arange(200), n_draws=20000, seed=1)["draws"])
    assert out["draws"].mean() == pytest.approx(3.0 - errors.mean(), abs=0.1)
    assert out["draws"].std() == pytest.approx(errors.std(), rel=0.03)


def test_loco_predictive_draws_cover_the_observed_effect_of_an_economy_that_was_not_in_the_sample():
    covered = []
    for rep in range(40):
        rng = np.random.default_rng(300 + rep)
        n, d = 30, 3
        Z = rng.normal(size=(n + 1, d))
        se = np.full(n + 1, 0.3)
        tau = 1.0 + 1.5 * Z[:, 0] + 0.5 * rng.normal(size=n + 1) + se * rng.normal(size=n + 1)
        w = np.array([1.0, 0.1, 0.1])
        res = tr.loco_validation(Z[:n], tau[:n], se[:n], w, eps=0.3, methods=("ot_weighted",), n_boot=5)
        plan = tr.sinkhorn_plan(np.full(n, 1.0 / n), [1.0], tr.weighted_sq_cost(Z[:n], Z[[n]], w), 0.3, rho_source=1.0)
        out = tr.loco_predictive_draws(res.table, tr.target_effect(plan, tau[:n], [1.0]), n_draws=1000, seed=rep)
        covered.append(out["percentiles"][5] <= tau[n] <= out["percentiles"][95])
    assert 0.75 <= np.mean(covered) <= 0.99


def test_loco_predictive_draws_check_their_arguments_and_name_them():
    table, groups = logo_table()
    two_methods = pd.concat([table, table.assign(method="equal")], ignore_index=True)
    nan_error = table.copy()
    nan_error.loc[3, "error"] = np.nan
    cases = [
        ({"table": np.ones(4)}, "table must be a DataFrame with the column 'error'"),
        ({"table": table.drop(columns="error")}, "table must be a DataFrame with the column 'error'"),
        ({"table": two_methods}, "table must hold the rows of one method"),
        ({"table": table.iloc[:0]}, "table has no rows"),
        ({"table": nan_error}, r"table\['error'\] must be finite"),
        ({"estimate": np.nan}, "estimate must be finite"),
        ({"estimate": np.inf}, "estimate must be finite"),
        ({"n_draws": 0}, "n_draws must be at least 1"),
        ({"groups": groups[:-1]}, "groups has 19 entries, expected 20"),
        ({"groups": [None] * 20}, "groups must not contain missing values"),
        ({"groups": np.zeros((20, 2))}, "groups must be a one-dimensional sequence of labels"),
    ]
    for override, message in cases:
        args = {"table": table, "estimate": 1.0}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.loco_predictive_draws(**args)


def test_loco_predictive_draws_are_documented_as_the_cost_of_transport_to_a_new_economy():
    assert "loco_predictive_draws" in tr.__all__
    doc = flat(tr.loco_predictive_draws.__doc__)
    assert "economy that was not in the sample" in doc and "cost of transporting an effect" in doc


# ----------------------------------------------------------------------------
# Descriptive ratio intervals, labelled inputs, degenerate effects
# ----------------------------------------------------------------------------
def test_ratio_intervals_are_described_as_descriptive_and_not_a_test_of_the_route_rule():
    Z, tau, se = informative_problem(0)
    w = np.array([1.0, 0.05, 0.05, 0.05])
    res = tr.loco_validation(Z, tau, se, w, n_boot=50)
    assert res.ratio_note == res.ratios.attrs["note"]
    assert "descriptive" in res.ratio_note and "not a test of the route rule" in res.ratio_note
    assert "describe the cases at hand" in res.ratio_note
    only = tr.loco_validation(Z, tau, se, w, methods=("nn1", "equal"), n_boot=5)
    assert only.ratios.empty and only.ratios.attrs["note"] == res.ratio_note
    for text in (tr.loco_validation.__doc__, tr.LocoResult.__doc__):
        assert "descriptive" in flat(text) and "not a test of the route rule" in flat(text)


def test_series_arguments_with_permuted_labels_are_put_in_the_order_of_the_cases():
    Z, tau, se = informative_problem(40, n=12)
    groups = np.repeat(np.arange(6), 2)
    a = np.random.default_rng(40).uniform(0.5, 1.5, 12)
    w = np.array([1.0, 0.05, 0.05, 0.05])

    def backwards(values, index):
        return pd.Series(values, index=index).iloc[::-1]

    base = tr.loco_validation(Z, tau, se, w, a=a, groups=groups, n_boot=20, seed=1)
    other = tr.loco_validation(
        Z, backwards(tau, Z.index), backwards(se, Z.index), w, a=backwards(a, Z.index), groups=backwards(groups, Z.index), n_boot=20, seed=1
    )
    pd.testing.assert_frame_equal(other.table, base.table, check_exact=True)
    pd.testing.assert_frame_equal(other.ratios, base.ratios, check_exact=True)
    unrelated = pd.Series(tau, index=range(100, 112))
    pd.testing.assert_frame_equal(
        tr.loco_validation(Z, unrelated, se, w, n_boot=5).table, tr.loco_validation(Z, tau, se, w, n_boot=5).table, check_exact=True
    )
    rng = np.random.default_rng(41)
    n, m = 14, 3
    Zs = pd.DataFrame(rng.normal(size=(n, 2)), index=[f"s{i}" for i in range(n)])
    Zt = pd.DataFrame(rng.normal(size=(m, 2)), index=[f"t{j}" for j in range(m)])
    tau_b, se_b, a_b, b_b = rng.normal(size=n), rng.uniform(0.1, 0.3, n), rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, m)
    groups_b = np.repeat(np.arange(7), 2)
    kwargs = {"w": [1.0, 0.5], "eps": 0.4, "n_boot": 30, "seed": 2}
    plain = tr.bootstrap_transport(Zs, Zt, tau=tau_b, se=se_b, a=a_b, b=b_b, groups=groups_b, **kwargs)
    permuted = tr.bootstrap_transport(
        Zs, Zt, tau=backwards(tau_b, Zs.index), se=backwards(se_b, Zs.index), a=backwards(a_b, Zs.index),
        b=backwards(b_b, Zt.index), groups=backwards(groups_b, Zs.index), **kwargs,
    )
    for key in ("draws", "predictive_draws"):
        np.testing.assert_array_equal(permuted[key], plain[key])


def test_loco_validation_with_identical_effects_runs_without_warnings():
    rng = np.random.default_rng(1)
    Z = pd.DataFrame(rng.normal(size=(12, 3)), columns=list("xyz"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = tr.loco_validation(Z, np.full(12, 1.5), np.full(12, 0.1), [1.0, 0.2, 0.2], n_boot=20)
    assert (res.summary["rmse"] < 1e-12).all() and len(res.ratios) == 2


# ----------------------------------------------------------------------------
# bootstrap_transport
# ----------------------------------------------------------------------------
def test_bootstrap_transport_returns_draws_percentiles_and_predictive_draws():
    rng = np.random.default_rng(40)
    n, m = 14, 4
    Zs, Zt = rng.normal(size=(n, 3)), rng.normal(scale=0.3, size=(m, 3))
    w = np.array([1.0, 0.2, 0.2])
    tau = 1.0 + Zs[:, 0] + 0.2 * rng.normal(size=n)
    se = np.full(n, 0.2)
    res = tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=60, seed=1)
    assert set(res) >= {
        "draws", "percentiles", "predictive_draws", "predictive_percentiles", "estimate", "between_sd", "measurement_sd",
        "eps", "n_boot", "n_nonconverged",
    }
    assert res["draws"].shape == (60,) and res["predictive_draws"].shape == (60,)
    assert list(res["percentiles"].index) == [5, 25, 50, 75, 95]
    assert list(res["predictive_percentiles"].index) == [5, 25, 50, 75, 95]
    np.testing.assert_allclose(res["percentiles"].to_numpy(), np.percentile(res["draws"], [5, 25, 50, 75, 95]))
    assert np.all(np.diff(res["percentiles"].to_numpy()) >= 0)
    assert res["n_boot"] == 60 and res["n_nonconverged"] == 0 and res["eps"] == pytest.approx(tr.select_eps(Zs, w))
    spread = res["percentiles"][95] - res["percentiles"][5]
    assert res["predictive_percentiles"][95] - res["predictive_percentiles"][5] > spread
    assert res["between_sd"] > 0
    plan = tr.sinkhorn_plan(np.full(n, 1 / n), np.full(m, 1 / m), tr.weighted_sq_cost(Zs, Zt, w), res["eps"], rho_source=1.0)
    assert res["estimate"] == pytest.approx(tr.target_effect(plan, tau, np.full(m, 1 / m)), abs=1e-9)
    assert abs(np.mean(res["draws"]) - res["estimate"]) < 0.3
    again = tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=60, seed=1)
    np.testing.assert_array_equal(again["draws"], res["draws"])
    np.testing.assert_array_equal(again["predictive_draws"], res["predictive_draws"])
    other = tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=60, seed=2)
    assert not np.array_equal(other["draws"], res["draws"])


def test_bootstrap_transport_degenerate_cases():
    rng = np.random.default_rng(41)
    Zs, Zt = rng.normal(size=(10, 2)), rng.normal(size=(3, 2))
    res = tr.bootstrap_transport(Zs, Zt, np.ones(2), np.full(10, 1.5), np.full(10, 1e-9), n_boot=20, seed=0)
    np.testing.assert_allclose(res["draws"], 1.5, atol=1e-7)
    np.testing.assert_allclose(res["predictive_draws"], 1.5, atol=1e-6)
    assert res["between_sd"] == 0.0 and res["measurement_sd"] == pytest.approx(1e-9)
    a = rng.uniform(0.5, 1.5, 10)
    noisy = tr.bootstrap_transport(Zs, Zt, np.ones(2), np.full(10, 1.5), np.full(10, 0.5), a=a, b=[1.0, 2.0, 1.0], rho_source=None, n_boot=40, seed=0)
    assert noisy["draws"].std() > 0.05 and abs(noisy["draws"].mean() - 1.5) < 0.3
    assert noisy["n_nonconverged"] == 0


def test_bootstrap_transport_checks_its_arguments_and_names_them():
    rng = np.random.default_rng(17)
    n, m = 10, 3
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
    tau, se = rng.normal(size=n), np.full(n, 0.2)
    at = np.arange(n)
    cases = [
        ({"Zs": np.where(np.eye(n, 2) > 0, np.nan, Zs)}, "Zs must be finite"),
        ({"Zt": np.where(np.eye(m, 2) > 0, np.inf, Zt)}, "Zt must be finite"),
        ({"Zt": rng.normal(size=(m, 3))}, "Zs has 2 features but Zt has 3"),
        ({"Zs": Zs[:, 0]}, "Zs must be a DataFrame or a two-dimensional array"),
        ({"w": [1.0]}, "w has 1 entries but there are 2 features"),
        ({"w": [1.0, np.nan]}, "w must be finite and non-negative"),
        ({"w": [0.0, 0.0]}, "w sums to zero"),
        ({"tau": tau[:-1]}, "tau has 9 entries, expected 10"),
        ({"tau": np.ones((n, 1))}, "tau must be one-dimensional"),
        ({"tau": np.where(at == 4, np.nan, tau)}, "tau must be finite"),
        ({"tau": np.where(at == 4, np.inf, tau)}, "tau must be finite"),
        ({"se": se[:-1]}, "se has 9 entries, expected 10"),
        ({"se": np.zeros(n)}, "se must be positive"),
        ({"se": -se}, "se must be positive"),
        ({"se": np.where(at == 4, np.nan, se)}, "se must be positive and finite"),
        ({"a": np.ones(n - 1)}, "a has 9 entries, expected 10"),
        ({"a": -np.ones(n)}, "a must be finite and non-negative"),
        ({"b": np.ones(m + 1)}, "b has 4 entries, expected 3"),
        ({"b": np.zeros(m)}, "b has zero total mass"),
        ({"groups": np.arange(n - 1)}, "groups has 9 entries, expected 10"),
        ({"groups": [None] * n}, "groups must not contain missing values"),
        ({"eps": 0.0}, "eps must be positive and finite"),
        ({"rho_source": -1.0}, "rho_source must be positive or None"),
        ({"n_boot": 0}, "n_boot must be at least 1"),
    ]
    for override, message in cases:
        args = {"Zs": Zs, "Zt": Zt, "w": [1.0, 1.0], "tau": tau, "se": se, "n_boot": 5}
        args.update(override)
        with pytest.raises(ValueError, match=message):
            tr.bootstrap_transport(**args)


def test_bootstrap_transport_drops_sources_without_mass_and_checks_eps():
    rng = np.random.default_rng(42)
    n, m = 12, 3
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
    tau, se = rng.normal(size=n), rng.uniform(0.05, 0.2, n)
    a = np.ones(n)
    a[[2, 7]] = 0.0
    keep = a > 0
    w = np.array([1.0, 2.0])
    reduced = tr.bootstrap_transport(Zs[keep], Zt, w, tau[keep], se[keep], n_boot=25, seed=3, eps=0.4)
    tau_missing, se_missing, Zs_missing = tau.copy(), se.copy(), Zs.copy()
    tau_missing[[2, 7]], se_missing[[2, 7]], Zs_missing[[2, 7]] = np.nan, np.nan, np.nan
    full = tr.bootstrap_transport(Zs_missing, Zt, w, tau_missing, se_missing, a=a, n_boot=25, seed=3, eps=0.4)
    np.testing.assert_allclose(full["draws"], reduced["draws"], atol=1e-12)
    np.testing.assert_allclose(full["predictive_draws"], reduced["predictive_draws"], atol=1e-12)
    assert full["estimate"] == pytest.approx(reduced["estimate"], abs=1e-12)
    with pytest.raises(ValueError, match="eps"):
        tr.bootstrap_transport(np.zeros((4, 2)), Zt, w, tau[:4], se[:4], n_boot=5)
    with pytest.raises(ValueError, match="eps"):
        tr.bootstrap_transport(Zs, Zt, w, tau, se, a=np.eye(n)[0], n_boot=5)


def test_bootstrap_predictive_draws_add_the_deconvolved_between_case_deviation_and_the_measurement_noise_once():
    rng = np.random.default_rng(71)
    n, m, n_boot, seed, eps = 18, 3, 25, 4, 0.5
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(scale=0.3, size=(m, 2))
    w = np.array([1.0, 0.2])
    se = rng.uniform(0.2, 0.9, n)
    tau = 1.0 + 1.2 * rng.normal(size=n) + se * rng.normal(size=n)
    res = tr.bootstrap_transport(Zs, Zt, w, tau, se, eps=eps, n_boot=n_boot, seed=seed)
    C = tr.weighted_sq_cost(Zs, Zt, w)
    b = np.full(m, 1.0 / m)
    stream = np.random.default_rng(seed)
    picked = stream.integers(0, n, size=(n_boot, n))
    noise = stream.standard_normal((n_boot, n))
    deviates = stream.standard_normal((n_boot, 2))

    def solve(ii):
        plan = tr.sinkhorn_plan(np.full(ii.size, 1.0 / ii.size), b, C[ii], eps, rho_source=1.0)
        usage = plan.pi.sum(axis=1)
        centre = usage @ tau[ii] / usage.sum()
        observed_variance = usage @ (tau[ii] - centre) ** 2 / usage.sum()
        return plan, observed_variance, usage @ se[ii] ** 2 / usage.sum()

    expected = np.empty(n_boot)
    for r in range(n_boot):
        ii = picked[r]
        plan, observed_variance, noise_variance = solve(ii)
        effect = tr.target_effect(plan, tau[ii] + se[ii] * noise[r], b)
        assert res["draws"][r] == pytest.approx(effect, abs=1e-12)
        expected[r] = (
            effect
            + np.sqrt(max(observed_variance - noise_variance, 0.0)) * deviates[r, 0]
            + np.sqrt(noise_variance) * deviates[r, 1]
        )
    np.testing.assert_allclose(res["predictive_draws"], expected, atol=1e-12)
    plan, observed_variance, noise_variance = solve(np.arange(n))
    assert observed_variance > noise_variance > 0
    assert res["between_sd"] == pytest.approx(np.sqrt(observed_variance - noise_variance), rel=1e-12)
    assert res["measurement_sd"] == pytest.approx(np.sqrt(noise_variance), rel=1e-12)
    assert res["estimate"] == pytest.approx(tr.target_effect(plan, tau, b), abs=1e-12)


def test_bootstrap_predictive_dispersion_equals_the_observed_dispersion_of_the_effects():
    rng = np.random.default_rng(70)
    n, m, eps = 60, 3, 0.5
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(scale=0.3, size=(m, 2))
    w = np.array([1.0, 0.2])
    se = np.full(n, 0.8)
    tau = 1.0 + 0.5 * rng.normal(size=n) + se * rng.normal(size=n)
    res = tr.bootstrap_transport(Zs, Zt, w, tau, se, eps=eps, n_boot=1000, seed=3)
    plan = tr.sinkhorn_plan(np.full(n, 1.0 / n), np.full(m, 1.0 / m), tr.weighted_sq_cost(Zs, Zt, w), eps, rho_source=1.0)
    usage = plan.pi.sum(axis=1)
    observed_variance = usage @ (tau - usage @ tau / usage.sum()) ** 2 / usage.sum()
    noise_variance = usage @ se**2 / usage.sum()
    assert res["between_sd"] ** 2 == pytest.approx(observed_variance - noise_variance, rel=1e-9)
    assert res["measurement_sd"] ** 2 == pytest.approx(noise_variance, rel=1e-9)
    added = res["predictive_draws"] - res["draws"]
    assert 0.85 < added.var() / observed_variance < 1.2
    assert added.var() < 0.8 * (observed_variance + noise_variance)


def test_bootstrap_between_case_deviation_is_zero_when_the_measurement_noise_explains_the_dispersion():
    rng = np.random.default_rng(72)
    n = 40
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(scale=0.3, size=(2, 2))
    se = np.full(n, 1.0)
    tau = 2.0 + 0.1 * rng.normal(size=n)
    res = tr.bootstrap_transport(Zs, Zt, np.ones(2), tau, se, eps=0.5, n_boot=800, seed=1)
    assert res["between_sd"] == 0.0 and res["measurement_sd"] == pytest.approx(1.0)
    added = res["predictive_draws"] - res["draws"]
    assert added.std() == pytest.approx(1.0, rel=0.1)
    assert abs(added.mean()) < 0.15


def test_bootstrap_transport_covers_the_true_target_effect_in_a_light_simulation():
    n_rep, n, d, m = 50, 20, 3, 3
    w = np.array([1.0, 0.1, 0.1])
    covered = []
    widths = []
    for rep in range(n_rep):
        rng = np.random.default_rng(5000 + rep)
        Zs = rng.normal(size=(n, d))
        Zt = rng.normal(scale=0.2, size=(m, d))
        truth = 1.0 + Zt[:, 0].mean()
        se = np.full(n, 0.3)
        tau = 1.0 + Zs[:, 0] + se * rng.normal(size=n)
        res = tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=60, seed=rep)
        low, high = res["percentiles"][5], res["percentiles"][95]
        covered.append(low <= truth <= high)
        widths.append(high - low)
    assert np.mean(covered) >= 0.85
    assert np.mean(widths) < 1.5


def test_bootstrap_transport_with_one_case_per_group_equals_the_case_bootstrap():
    rng = np.random.default_rng(43)
    n, m = 14, 4
    Zs, Zt = rng.normal(size=(n, 3)), rng.normal(scale=0.3, size=(m, 3))
    w = np.array([1.0, 0.2, 0.2])
    tau, se = 1.0 + Zs[:, 0] + 0.2 * rng.normal(size=n), np.full(n, 0.2)
    plain = tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=60, seed=1)
    for groups in (np.arange(n), [f"economy{i}" for i in range(n)], rng.permutation(n)):
        res = tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=60, seed=1, groups=groups)
        for key in ("draws", "predictive_draws"):
            np.testing.assert_array_equal(res[key], plain[key])
        pd.testing.assert_series_equal(res["percentiles"], plain["percentiles"], check_exact=True)
        assert res["estimate"] == plain["estimate"] and res["between_sd"] == plain["between_sd"]
        assert res["n_nonconverged"] == plain["n_nonconverged"]


def test_bootstrap_transport_cluster_resamples_contain_whole_groups(monkeypatch):
    rng = np.random.default_rng(60)
    sizes = np.array([1, 2, 3, 4, 2, 3, 1, 2, 4])
    groups = np.repeat(np.arange(sizes.size), sizes)
    n, n_groups, n_boot = groups.size, sizes.size, 200
    Zs, Zt = rng.normal(size=(n, 3)), rng.normal(scale=0.3, size=(3, 3))
    tau, se = rng.normal(size=n), np.full(n, 0.1)
    w = np.array([1.0, 0.5, 0.2])

    def case_counts(solved):
        """How often each case enters each replicate, from the cost matrices of the problems solved."""
        row_of = {row.tobytes(): k for k, row in enumerate(solved[0])}
        return np.array([np.bincount([row_of[r.tobytes()] for r in C], minlength=n) for C in solved[1:]])

    solved = spy_on_solve(monkeypatch)
    tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=n_boot, seed=6, groups=groups)
    assert len(solved) == n_boot + 1 and solved[0].shape[0] == n
    counts = case_counts(solved)
    drawn = np.empty((n_boot, n_groups), dtype=int)
    for g in range(n_groups):
        members = counts[:, groups == g]
        assert np.all(members == members[:, [0]]), "every case of a group enters as often as the group is drawn"
        drawn[:, g] = members[:, 0]
    assert np.all(drawn.sum(axis=1) == n_groups), "as many draws as there are groups"
    assert np.all(counts.sum(axis=1) == drawn @ sizes) and len(set(counts.sum(axis=1))) > 1
    np.testing.assert_allclose(drawn.mean(axis=0), 1.0, atol=0.3)
    assert drawn.max() >= 3 and drawn.min() == 0
    solved.clear()
    tr.bootstrap_transport(Zs, Zt, w, tau, se, n_boot=n_boot, seed=6)
    counts = case_counts(solved)
    assert np.all(counts.sum(axis=1) == n)
    split = [np.ptp(counts[r, groups == g]) > 0 for r in range(n_boot) for g in range(n_groups) if sizes[g] > 1]
    assert np.mean(split) > 0.5


def test_bootstrap_transport_cluster_bootstrap_is_wider_when_the_cases_of_a_group_share_an_effect():
    n_groups, per = 15, 3
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(n_groups, 2))
    Zs = np.repeat(centres, per, axis=0) + 0.01 * rng.normal(size=(n_groups * per, 2))
    tau = np.repeat(rng.normal(size=n_groups), per) + 0.01 * rng.normal(size=n_groups * per)
    se = np.full(n_groups * per, 0.05)
    groups = np.repeat(np.arange(n_groups), per)
    Zt = np.zeros((1, 2))
    plain = tr.bootstrap_transport(Zs, Zt, np.ones(2), tau, se, n_boot=300, seed=2)
    cluster = tr.bootstrap_transport(Zs, Zt, np.ones(2), tau, se, n_boot=300, seed=2, groups=groups)
    assert cluster["estimate"] == plain["estimate"]
    assert cluster["draws"].std() > 1.3 * plain["draws"].std()
    assert cluster["percentiles"][95] - cluster["percentiles"][5] > 1.3 * (plain["percentiles"][95] - plain["percentiles"][5])
    assert cluster["n_nonconverged"] == 0


def test_bootstrap_transport_checks_groups_and_ignores_groups_without_mass():
    rng = np.random.default_rng(61)
    n = 12
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(size=(3, 2))
    tau, se = rng.normal(size=n), rng.uniform(0.05, 0.2, n)
    groups = np.repeat(np.arange(4), 3)
    w = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="entries"):
        tr.bootstrap_transport(Zs, Zt, w, tau, se, groups=groups[:-1], n_boot=5)
    with pytest.raises(ValueError, match="missing"):
        tr.bootstrap_transport(Zs, Zt, w, tau, se, groups=[None, *groups[1:]], n_boot=5)
    with pytest.raises(ValueError, match="one-dimensional"):
        tr.bootstrap_transport(Zs, Zt, w, tau, se, groups=np.zeros((n, 2)), n_boot=5)
    with pytest.raises(ValueError, match="two distinct"):
        tr.bootstrap_transport(Zs, Zt, w, tau, se, groups=np.zeros(n), n_boot=5)
    a = np.where(groups == 2, 0.0, 1.0)
    keep = a > 0
    full = tr.bootstrap_transport(Zs, Zt, w, tau, se, a=a, groups=groups, eps=0.4, n_boot=30, seed=2)
    reduced = tr.bootstrap_transport(Zs[keep], Zt, w, tau[keep], se[keep], groups=groups[keep], eps=0.4, n_boot=30, seed=2)
    np.testing.assert_array_equal(full["draws"], reduced["draws"])
    np.testing.assert_array_equal(full["predictive_draws"], reduced["predictive_draws"])
    with pytest.raises(ValueError, match="two distinct"):
        tr.bootstrap_transport(Zs, Zt, w, tau, se, a=np.where(groups == 0, 1.0, 0.0), groups=groups, eps=0.4, n_boot=5)


@pytest.mark.parametrize("n_boot,expected", [(1, 1), (50, 1), (51, 2), (120, 3)])
def test_bootstrap_transport_guard_is_called_before_every_fiftieth_draw(monkeypatch, n_boot, expected):
    rng = np.random.default_rng(62)
    n = 10
    Zs, Zt = rng.normal(size=(n, 2)), rng.normal(size=(2, 2))
    tau, se = rng.normal(size=n), np.full(n, 0.1)
    solved = spy_on_solve(monkeypatch)
    for groups in (None, np.repeat(np.arange(5), 2)):
        solved.clear()
        solves_so_far = []
        res = tr.bootstrap_transport(
            Zs, Zt, [1.0, 1.0], tau, se, n_boot=n_boot, seed=1, groups=groups, guard=lambda: solves_so_far.append(len(solved))
        )
        assert solves_so_far == [1 + 50 * k for k in range(expected)]
        same = tr.bootstrap_transport(Zs, Zt, [1.0, 1.0], tau, se, n_boot=n_boot, seed=1, groups=groups)
        np.testing.assert_array_equal(res["draws"], same["draws"])


# ----------------------------------------------------------------------------
# Independence from POT
# ----------------------------------------------------------------------------
def test_module_works_without_pot():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "sys.modules['ot'] = None\n"
        "import numpy as np\n"
        "from dtt import transport as tr\n"
        "assert 'ot' not in [k for k, v in sys.modules.items() if v is not None]\n"
        "rng = np.random.default_rng(0)\n"
        "C = tr.weighted_sq_cost(rng.normal(size=(5, 2)), rng.normal(size=(4, 2)), [1.0, 1.0])\n"
        "plan = tr.sinkhorn_plan(np.full(5, 0.2), np.full(4, 0.25), C, 0.5)\n"
        "assert plan.converged\n"
        "assert tr.wasserstein2(None, None, np.zeros((2, 1)), np.ones((2, 1)), [1.0]) == 1.0\n"
    )
    start = time.time()
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert time.time() - start < 60
