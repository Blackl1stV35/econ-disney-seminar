"""Tests for dtt.importance on simulated data generated inside the tests.

No file of the repository is read.  The default configuration is run once, to check the time limit and
the number of guard calls; every other test uses the linear learners, the light configuration or
forests of a few trees.
"""
import ast
import inspect
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dtt import importance  # noqa: E402
from dtt.importance import (  # noqa: E402
    LEARNER_NAMES,
    ImportanceResult,
    StackedEnsemble,
    bootstrap_importance,
    lambda_averaged_importance,
    lambda_grid,
    make_toy_problem,
    verdict_stability,
)
from dtt.thermal import ThermalGuard, ThermalTimeout  # noqa: E402

LINEAR = ("elastic_net", "ridge")


def noise_problem(n=30, d=8, seed=1000):
    """Features and a target that are independent of each other, one case per group."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, d)), columns=[f"x{j}" for j in range(d)])
    y = pd.Series(rng.standard_normal(n), name="effect")
    return X, y, np.arange(n)


def linear_problem(n=80, seed=7):
    """Five standard normal features; the target is 3 x0 + x1 plus noise with standard deviation 0.3."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{j}" for j in range(5)])
    y = pd.Series(3.0 * X["x0"] + 1.0 * X["x1"] + 0.3 * rng.standard_normal(n))
    return X, y, np.arange(n)


def bystander_problem(rho, n=60, seed=0):
    """Target 2 x0 + 2 x1 + noise; ``xc`` has correlation ``rho`` with x0; three noise features."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    columns = {"x0": x0, "x1": x1, "n2": rng.standard_normal(n), "n3": rng.standard_normal(n)}
    columns["xc"] = rho * x0 + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    y = pd.Series(2.0 * x0 + 2.0 * x1 + 0.5 * rng.standard_normal(n), name="effect")
    return pd.DataFrame(columns), y, np.arange(n)


def kfold_by_group(codes, k, rng):
    """Deterministic grouped K-fold with the signature of the random assignment of groups to folds."""
    return list(GroupKFold(n_splits=k).split(np.zeros((len(codes), 1)), groups=codes))


def record_splits(monkeypatch):
    """Record the group codes, fold count and splits of every grouped split the module makes."""
    recorded = []
    original = importance._random_group_splits

    def spy(codes, k, rng):
        splits = original(codes, k, rng)
        recorded.append((np.asarray(codes).copy(), k, splits))
        return splits

    monkeypatch.setattr(importance, "_random_group_splits", spy)
    return recorded


def assert_group_disjoint(codes, splits):
    seen = np.zeros(len(codes), dtype=int)
    for train, test in splits:
        assert set(codes[train]).isdisjoint(set(codes[test]))
        assert len(np.intersect1d(train, test)) == 0
        seen[test] += 1
    assert np.all(seen == 1)


ARRAY_FIELDS = (
    "permutation", "permutation_normalised", "coefficient_path", "selection_frequency", "lambdas", "lambda_weights",
    "cv_mse_by_lambda", "cluster_importance",
)
NUMBER_KEYS = ("cv_r2_stack", "cv_r2_best_single", "cv_r2_null", "perm_p_value", "split_half_correlation")


def assert_same_result(first, second, rtol=0.0, atol=0.0):
    """Importance arrays and diagnostics of two results agree (``nan`` equals ``nan``)."""
    for field in ARRAY_FIELDS:
        npt.assert_allclose(getattr(first, field), getattr(second, field), rtol=rtol, atol=atol, err_msg=field)
    npt.assert_allclose(
        [first.diagnostics[k] for k in NUMBER_KEYS], [second.diagnostics[k] for k in NUMBER_KEYS], rtol=rtol, atol=atol
    )
    assert first.diagnostics["usable"] == second.diagnostics["usable"]
    assert first.diagnostics["reasons"] == second.diagnostics["reasons"]
    npt.assert_array_equal(first.cluster.to_numpy(), second.cluster.to_numpy())


def reordered(X, y, groups, order, relabel=False):
    """The same cases in another row order, optionally with the group labels reversed in sort order."""
    labels = np.asarray(groups)
    if relabel:
        ordered = sorted(set(labels))
        mapping = dict(zip(ordered, reversed(ordered)))
        labels = np.array([mapping[g] for g in labels])
    return X.iloc[order], y.iloc[order], labels[order]


@pytest.fixture(scope="module")
def toy40():
    X, y, groups = make_toy_problem(n=40, d=6, random_state=0)
    return X, y, groups, lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)


@pytest.fixture(scope="module")
def noise30():
    X, y, groups = noise_problem()
    return X, y, groups, lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)


@pytest.fixture
def quick_halves(monkeypatch):
    """Three split-half repetitions in every configuration."""
    monkeypatch.setattr(importance, "_LIGHT_N_HALVES", 3)
    monkeypatch.setattr(importance, "_FULL_N_HALVES", 3)


@pytest.fixture(scope="module")
def toy30():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    res = lambda_averaged_importance(X, y, groups, light=True, compute_diagnostics=False, random_state=1)
    return X, y, groups, res


# ----------------------------------------------------------------------------
# Verdicts on simulated problems
# ----------------------------------------------------------------------------
def test_informative_features_carry_the_largest_importance_and_the_toy_is_usable(toy40):
    X, y, groups, res = toy40
    top_two = set(res.permutation_normalised.sort_values(ascending=False).index[:2])
    assert top_two == {"x0", "x1"}
    d = res.diagnostics
    assert d["usable"] is True and d["reasons"] == []
    assert d["cv_r2_stack"] >= 0.10 and d["perm_p_value"] <= 0.10 and d["split_half_correlation"] >= 0.50
    assert d["n_cases"] == 40 and d["n_features"] == 6
    assert d["cv_r2_null"] < 0.05


def test_result_structure_of_the_toy_problem(toy40):
    X, y, groups, res = toy40
    assert isinstance(res, ImportanceResult)
    assert res.feature_names == list(X.columns)
    for series in (
        res.permutation, res.permutation_normalised, res.coefficient_path, res.selection_frequency, res.cluster,
        res.cluster_importance,
    ):
        assert list(series.index) == res.feature_names
    npt.assert_allclose(res.permutation_normalised.sum(), 1.0)
    assert (res.permutation_normalised >= 0).all()
    clipped = res.permutation.clip(lower=0.0)
    npt.assert_allclose(res.permutation_normalised, clipped / clipped.sum())
    assert res.lambdas.size == 4 and np.all(np.diff(res.lambdas) < 0)
    npt.assert_allclose(res.lambda_weights.sum(), 1.0)
    assert res.lambda_weights.shape == res.cv_mse_by_lambda.shape == res.lambdas.shape
    assert (res.coefficient_path >= 0).all()
    assert ((res.selection_frequency >= 0) & (res.selection_frequency <= 1)).all()
    assert set(res.diagnostics) == {
        "cv_r2_stack", "cv_r2_best_single", "cv_r2_null", "perm_p_value", "split_half_correlation",
        "n_cases", "n_features", "no_signal", "usable", "reasons",
    }
    assert res.no_signal is False and res.diagnostics["no_signal"] is False
    n_perm_used = 9
    p_scaled = res.diagnostics["perm_p_value"] * (n_perm_used + 1)
    assert abs(p_scaled - round(p_scaled)) < 1e-9 and 1 <= round(p_scaled) <= n_perm_used + 1
    sst = float(np.sum((y - y.mean()) ** 2))
    expected_r2 = 1.0 - len(y) * res.cv_mse_by_lambda.min() / sst
    assert res.diagnostics["cv_r2_stack"] == pytest.approx(expected_r2)


def test_pure_noise_is_not_usable_and_every_failed_criterion_is_named(noise30):
    X, y, groups, res = noise30
    d = res.diagnostics
    assert d["usable"] is False
    assert isinstance(d["reasons"], list) and len(d["reasons"]) >= 1
    failed = []
    if not d["cv_r2_stack"] >= 0.10:
        failed.append("cv_r2_stack")
    if not d["perm_p_value"] <= 0.10:
        failed.append("perm_p_value")
    if not d["split_half_correlation"] >= 0.50:
        failed.append("split_half_correlation")
    if not d["n_cases"] >= importance.MIN_CASES:
        failed.append("n_cases")
    assert len(d["reasons"]) == len(failed) >= 1
    for name in failed:
        assert sum(name in reason for reason in d["reasons"]) == 1


def test_a_sample_below_the_minimum_number_of_cases_is_not_usable():
    X, y, groups = make_toy_problem(n=9, random_state=0)
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
    assert res.diagnostics["n_cases"] == 9 and res.diagnostics["usable"] is False
    assert any("n_cases" in reason for reason in res.diagnostics["reasons"])
    scaled = res.diagnostics["perm_p_value"] * 10.0
    assert abs(scaled - round(scaled)) < 1e-9 and 1 <= round(scaled) <= 10


@pytest.mark.parametrize("n, named", [(19, True), (20, False)])
def test_the_minimum_number_of_cases_is_twenty_rows(quick_halves, n, named):
    X, y, groups = make_toy_problem(n=n, d=5, random_state=3)
    assert len(set(groups)) < n
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
    reasons = [reason for reason in res.diagnostics["reasons"] if "n_cases" in reason]
    assert res.diagnostics["n_cases"] == n
    assert bool(reasons) is named
    if named:
        assert reasons == [f"n_cases is 19, below the minimum of {importance.MIN_CASES}"]
        assert res.diagnostics["usable"] is False


def test_the_minimum_counts_cases_and_not_groups():
    diagnostics = {"cv_r2_stack": 0.5, "perm_p_value": 0.05, "split_half_correlation": 0.9, "n_cases": 20}
    assert importance._verdict(diagnostics) == (True, [])
    X, y, groups = make_toy_problem(n=24, random_state=1)
    assert len(set(groups)) < 24 and importance.MIN_CASES < 24
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    assert res.diagnostics["n_cases"] == 24
    assert "split-half criterion" in importance.__doc__ and "economies" in importance.__doc__


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, []),
        ({"cv_r2_stack": 0.0999}, ["cv_r2_stack"]),
        ({"perm_p_value": 0.1001}, ["perm_p_value"]),
        ({"split_half_correlation": 0.4999}, ["split_half_correlation"]),
        ({"split_half_correlation": float("nan")}, ["split_half_correlation"]),
        ({"n_cases": 19}, ["n_cases"]),
        ({"cv_r2_stack": -1.0, "perm_p_value": 0.9, "split_half_correlation": 0.0, "n_cases": 3},
         ["cv_r2_stack", "perm_p_value", "split_half_correlation", "n_cases"]),
        ({"cv_r2_stack": 0.10 - 1e-12}, []),
        ({"perm_p_value": 0.10 + 1e-12}, []),
        ({"split_half_correlation": 0.50 - 1e-12}, []),
        ({"cv_r2_stack": 0.10 - 1e-6}, ["cv_r2_stack"]),
    ],
)
def test_usability_rule_and_boundaries(changes, expected):
    diagnostics = {
        "cv_r2_stack": 0.10, "perm_p_value": 0.10, "split_half_correlation": 0.50, "n_cases": importance.MIN_CASES
    }
    diagnostics.update(changes)
    usable, reasons = importance._verdict(diagnostics)
    assert usable is (expected == [])
    assert len(reasons) == len(expected)
    for name in expected:
        assert sum(name in reason for reason in reasons) == 1


def test_thresholds_of_the_usability_rule():
    assert importance.MIN_CV_R2 == 0.10
    assert importance.MAX_PERM_P_VALUE == 0.10
    assert importance.MIN_SPLIT_HALF_CORRELATION == 0.50
    assert importance.MIN_CASES == 20 and isinstance(importance.MIN_CASES, int)
    assert "MIN_CASES" in importance.__all__
    assert "MIN_SPLIT_HALF_CORRELATION" in importance.__all__
    assert not hasattr(importance, "MIN_SPLIT_HALF_SPEARMAN")
    assert "MIN_SPLIT_HALF_SPEARMAN" not in importance.__all__


# ----------------------------------------------------------------------------
# Diagnostics that are not computed
# ----------------------------------------------------------------------------
def test_without_diagnostics_the_p_value_and_the_correlation_are_nan_and_the_vector_is_not_usable(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    called = []
    for name in ("_label_permutation_p", "_split_half_correlation"):
        monkeypatch.setattr(importance, name, lambda *args, _name=name, **kwargs: called.append(_name))
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    d = res.diagnostics
    assert called == []
    assert np.isnan(d["perm_p_value"]) and np.isnan(d["split_half_correlation"])
    assert d["usable"] is False
    assert len(d["reasons"]) == 1 and "not computed" in d["reasons"][0]
    assert d["cv_r2_stack"] > 0.2 and np.isfinite([d["cv_r2_best_single"], d["cv_r2_null"]]).all()
    assert d["n_cases"] == 24 and d["n_features"] == 8
    assert set(d) == {
        "cv_r2_stack", "cv_r2_best_single", "cv_r2_null", "perm_p_value", "split_half_correlation",
        "n_cases", "n_features", "no_signal", "usable", "reasons",
    }


def test_the_importance_vector_does_not_depend_on_whether_the_diagnostics_are_computed(quick_halves):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    with_diagnostics = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
    without = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    for field in ARRAY_FIELDS:
        npt.assert_array_equal(getattr(with_diagnostics, field), getattr(without, field), err_msg=field)
    for key in ("cv_r2_stack", "cv_r2_best_single", "cv_r2_null"):
        assert with_diagnostics.diagnostics[key] == without.diagnostics[key]


def test_compute_diagnostics_is_true_by_default():
    assert inspect.signature(lambda_averaged_importance).parameters["compute_diagnostics"].default is True


def test_verdict_of_diagnostics_that_were_not_computed():
    diagnostics = {
        "cv_r2_stack": 0.9, "perm_p_value": float("nan"), "split_half_correlation": float("nan"), "n_cases": 40,
    }
    usable, reasons = importance._verdict(diagnostics, computed=False)
    assert usable is False
    assert len(reasons) == 1 and "not computed" in reasons[0] and "compute_diagnostics" in reasons[0]


def test_a_permutation_count_below_nine_cannot_reach_the_p_value_threshold(quick_halves):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    for n_perm in (0, 1, 5, 8):
        with pytest.raises(ValueError, match=r"at least 9 permutations") as info:
            lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=n_perm)
        assert "0.10" in str(info.value) and f"n_perm is {n_perm}" in str(info.value)
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
    assert res.diagnostics["perm_p_value"] == pytest.approx(0.1)
    assert res.diagnostics["perm_p_value"] <= importance.MAX_PERM_P_VALUE
    ignored = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=0, compute_diagnostics=False)
    assert np.isnan(ignored.diagnostics["perm_p_value"])
    band = bootstrap_importance(X, y, groups, n_boot=2, learners=LINEAR, light=True, n_perm=1)
    assert band.shape == (8, 5)


# ----------------------------------------------------------------------------
# Stacked ensemble
# ----------------------------------------------------------------------------
def test_stacking_weights_keep_the_least_squares_scale_and_do_not_sum_to_one():
    rng = np.random.default_rng(24)
    X = pd.DataFrame(rng.standard_normal((30, 5)), columns=list("abcde"))
    y = pd.Series(0.4 * X["a"] + rng.standard_normal(30))
    model = StackedEnsemble(n_estimators=20).fit(X, y)
    weights = model.weights_
    oof = model.oof_predictions_
    assert list(weights.index) == list(LEARNER_NAMES)
    assert list(oof.columns) == list(LEARNER_NAMES) and oof.shape == (30, 4) and oof.index.equals(X.index)
    assert set(model.base_models_) == set(LEARNER_NAMES)
    centre = float(y.mean())
    raw, _ = nnls((oof - centre).to_numpy(), (y - centre).to_numpy())
    npt.assert_allclose(weights.to_numpy(), raw, rtol=1e-8, atol=1e-10)
    assert (weights >= 0).all() and 0.2 < weights.sum() < 0.5
    assert model.target_mean_ == pytest.approx(centre)
    base = np.column_stack([model.base_models_[name].predict(X.to_numpy()) for name in LEARNER_NAMES])
    by_hand = centre + (base - centre) @ weights.to_numpy()
    npt.assert_allclose(model.predict(X), by_hand)
    assert np.abs(model.predict(X) - centre).max() < np.abs(base - centre).max()


def test_stacking_weights_of_a_strong_linear_signal_sum_to_more_than_one():
    X, y, groups = linear_problem(n=60, seed=3)
    model = StackedEnsemble(lam=0.1, learners=LINEAR).fit(X, y, groups)
    assert (model.weights_ >= 0).all()
    assert model.weights_.sum() > 1.05
    assert r2_score(y, model.predict(X)) > 0.95


def test_all_zero_stacking_weights_give_the_mean_of_the_training_target(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=2)
    monkeypatch.setattr(importance, "nnls", lambda A, b: (np.zeros(A.shape[1]), 0.0))
    model = StackedEnsemble(learners=LINEAR).fit(X, y, groups)
    npt.assert_array_equal(model.weights_.to_numpy(), [0.0, 0.0])
    npt.assert_allclose(model.predict(X), np.full(24, y.mean()))
    anti = -(y.to_numpy() - y.mean())[:, None] * np.ones((1, 4))
    monkeypatch.undo()
    npt.assert_array_equal(importance._stack_weights(anti, y.to_numpy()), np.zeros(4))


def test_stack_weights_are_the_nonnegative_least_squares_solution_of_the_centred_problem():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(40) + 5.0
    P = np.column_stack([y + 0.5 * rng.standard_normal(40), 2.0 * rng.standard_normal(40) + 5.0, 3.0 - y])
    weights = importance._stack_weights(P, y)
    raw, _ = nnls(P - y.mean(), y - y.mean())
    npt.assert_allclose(weights, raw)
    assert weights[2] == 0.0 and weights[0] > 0.0 and weights.sum() != pytest.approx(1.0, abs=1e-3)


def test_learner_subsets_and_validation():
    X, y, groups = make_toy_problem(n=24, random_state=3)
    single = StackedEnsemble(learners=("ridge",)).fit(X, y, groups)
    assert single.weights_.shape == (1,) and 0.5 < single.weights_.iloc[0] < 1.5
    mixed = StackedEnsemble(learners=["gradient_boosting", "elastic_net"], n_estimators=20).fit(X, y, groups)
    assert list(mixed.weights_.index) == ["elastic_net", "gradient_boosting"]
    with pytest.raises(ValueError):
        StackedEnsemble(learners=["lasso"]).fit(X, y, groups)
    with pytest.raises(ValueError):
        StackedEnsemble(lam=0.0).fit(X, y, groups)
    with pytest.raises(ValueError):
        StackedEnsemble(l1_ratio=0.0).fit(X, y, groups)
    with pytest.raises(ValueError):
        StackedEnsemble(n_repeats=0).fit(X, y, groups)
    with pytest.raises(ValueError):
        StackedEnsemble().fit(X, y, np.repeat(["a", "b"], 12))


def fitted_linear(model, name):
    """Fitted elastic net or ridge regression inside a fitted stack."""
    return model.base_models_[name].regressor_[-1]


def test_penalty_parametrisation_of_the_linear_learners():
    X, y, groups = make_toy_problem(n=30, random_state=4)
    grid = lambda_grid(X, y)
    weak = StackedEnsemble(lam=grid[-1], learners=LINEAR).fit(X, y, groups)
    strong = StackedEnsemble(lam=grid[2], learners=LINEAR).fit(X, y, groups)
    for name in LINEAR:
        assert np.abs(fitted_linear(strong, name).coef_).sum() < np.abs(fitted_linear(weak, name).coef_).sum()
    assert fitted_linear(weak, "elastic_net").l1_ratio == 0.5
    assert fitted_linear(weak, "elastic_net").alpha == pytest.approx(grid[-1])
    assert fitted_linear(weak, "ridge").alpha == pytest.approx(grid[-1] * 30)
    zero = StackedEnsemble(lam=grid[0], learners=("elastic_net",)).fit(X, y, groups)
    assert np.all(fitted_linear(zero, "elastic_net").coef_ == 0.0)


def test_stacked_ensemble_follows_the_scikit_learn_protocol():
    X, y, groups = make_toy_problem(n=24, random_state=5)
    model = StackedEnsemble(lam=0.2, learners=LINEAR, n_splits=4, random_state=3)
    assert isinstance(model, RegressorMixin)
    assert model.get_params() == {
        "lam": 0.2, "l1_ratio": 0.5, "learners": LINEAR, "n_splits": 4, "n_estimators": 100, "random_state": 3,
        "n_repeats": 1,
    }
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert copy.set_params(lam=0.5).lam == 0.5
    assert model.fit(X, y, groups) is model
    assert model.n_features_in_ == 8 and list(model.feature_names_in_) == list(X.columns)
    assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))
    shuffled = X[list(reversed(X.columns))]
    npt.assert_allclose(model.predict(shuffled), model.predict(X))
    npt.assert_allclose(model.predict(X.to_numpy()), model.predict(X))
    with pytest.raises(ValueError):
        model.predict(X.iloc[:, :5])
    with pytest.raises(ValueError):
        model.predict(X.rename(columns={"x0": "other"}))
    refit = StackedEnsemble(learners=LINEAR).fit(X.to_numpy(), y.to_numpy())
    assert not hasattr(refit, "feature_names_in_") and refit.oof_predictions_.index.equals(pd.RangeIndex(24))
    scores = cross_val_score(StackedEnsemble(learners=LINEAR), X, y, cv=3)
    assert scores.shape == (3,) and np.isfinite(scores).all()
    search = GridSearchCV(StackedEnsemble(learners=LINEAR), {"lam": [0.01, 0.3]}, cv=3).fit(X, y)
    assert search.best_params_["lam"] in (0.01, 0.3)


def test_stacked_ensemble_does_not_depend_on_the_order_of_the_rows_or_the_group_labels():
    X, y, groups = make_toy_problem(n=30, random_state=6)
    order = np.random.default_rng(0).permutation(30)
    X2, y2, groups2 = reordered(X, y, groups, order, relabel=True)
    first = StackedEnsemble(n_estimators=10, n_repeats=2).fit(X, y, groups)
    second = StackedEnsemble(n_estimators=10, n_repeats=2).fit(X2, y2, groups2)
    npt.assert_allclose(second.weights_.to_numpy(), first.weights_.to_numpy(), rtol=1e-9, atol=1e-12)
    npt.assert_allclose(second.oof_predictions_.loc[X.index].to_numpy(), first.oof_predictions_.to_numpy(), rtol=1e-9)
    npt.assert_allclose(second.predict(X), first.predict(X), rtol=1e-9)


def test_stacked_ensemble_assigns_groups_to_folds_at_random_from_its_seed(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=7)
    recorded = record_splits(monkeypatch)
    oof = {}
    for seed in (0, 0, 1):
        oof.setdefault(seed, []).append(StackedEnsemble(lam=0.1, learners=LINEAR, random_state=seed).fit(X, y, groups))
    npt.assert_array_equal(oof[0][0].oof_predictions_.to_numpy(), oof[0][1].oof_predictions_.to_numpy())
    assert not np.allclose(oof[0][0].oof_predictions_.to_numpy(), oof[1][0].oof_predictions_.to_numpy())
    assert len(recorded) == 3
    tests = [[tuple(te) for _, te in splits] for _, _, splits in recorded]
    assert tests[0] == tests[1] != tests[2]
    for codes, k, splits in recorded:
        assert k == 5
        assert_group_disjoint(codes, splits)


def test_stacked_ensemble_averages_out_of_fold_predictions_over_its_repeats(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=8)
    recorded = record_splits(monkeypatch)
    model = StackedEnsemble(lam=0.1, learners=("ridge",), n_repeats=3).fit(X, y, groups)
    assert len(recorded) == 3
    assert len({tuple(tuple(te) for _, te in splits) for _, _, splits in recorded}) == 3
    values, target = X.to_numpy(), y.to_numpy()
    codes = importance._group_codes(groups, 30)
    order, _ = importance._canonical_rows(values, target, codes)
    expected = np.zeros(30)
    for _, _, splits in recorded:
        for train, test in splits:
            fold_model = importance._fit_model("ridge", 0.1, 0.5, 100, 0, values[order][train], target[order][train])
            expected[order[test]] += fold_model.predict(values[order][test]) / 3.0
    npt.assert_allclose(model.oof_predictions_["ridge"].to_numpy(), expected, rtol=1e-9)


# ----------------------------------------------------------------------------
# Grouped folds
# ----------------------------------------------------------------------------
def test_grouped_folds_never_put_a_group_in_both_train_and_test(monkeypatch, quick_halves):
    recorded = record_splits(monkeypatch)
    X, y, groups = make_toy_problem(n=40, random_state=6)
    assert len(set(groups)) < len(groups)
    StackedEnsemble(learners=LINEAR).fit(X, y, groups)
    lambda_averaged_importance(X, y, groups, learners=LINEAR, n_repeats=2, compute_diagnostics=False)
    lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
    assert len(recorded) > 40
    for codes, k, splits in recorded:
        assert len(splits) == k
        assert_group_disjoint(codes, splits)


def test_random_group_splits_balance_the_numbers_of_groups_and_cases():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 5, size=23)
    codes = np.repeat(np.arange(23), sizes)
    for seed in range(12):
        splits = importance._random_group_splits(codes, 5, np.random.default_rng(seed))
        assert_group_disjoint(codes, splits)
        n_groups = [len(set(codes[test])) for _, test in splits]
        assert sum(n_groups) == 23 and max(n_groups) == 5 and min(n_groups) >= 3
        n_cases = [test.size for _, test in splits]
        assert max(n_cases) - min(n_cases) <= sizes.max()


def test_random_group_splits_depend_on_the_generator_only():
    codes = np.repeat(np.arange(20), 2)
    first = importance._random_group_splits(codes, 5, np.random.default_rng(1))
    again = importance._random_group_splits(codes, 5, np.random.default_rng(1))
    other = importance._random_group_splits(codes, 5, np.random.default_rng(2))
    assert [tuple(te) for _, te in first] == [tuple(te) for _, te in again]
    assert [tuple(te) for _, te in first] != [tuple(te) for _, te in other]


def test_make_folds_draws_each_repeat_from_its_own_seeded_stream():
    codes = np.repeat(np.arange(15), 2)
    folds = importance._make_folds(codes, 5, 3, None, 3, 11)
    again = importance._make_folds(codes, 5, 3, None, 3, 11)
    other = importance._make_folds(codes, 5, 3, None, 3, 12)
    assert [(f.repeat, f.index) for f in folds] == [(r, i) for r in range(3) for i in range(5)]
    for a, b in zip(folds, again):
        npt.assert_array_equal(a.test, b.test)
        for (_, a_test), (_, b_test) in zip(a.inner, b.inner):
            npt.assert_array_equal(a_test, b_test)
    partitions = [frozenset(tuple(f.test) for f in folds if f.repeat == r) for r in range(3)]
    assert len(set(partitions)) == 3
    assert [tuple(f.test) for f in folds] != [tuple(f.test) for f in other]
    for f in folds:
        assert set(codes[f.train]).isdisjoint(codes[f.test])
        assert np.array_equal(np.sort(np.concatenate([f.train, f.test])), np.arange(30))
        inner_groups = codes[f.train]
        assert_group_disjoint(inner_groups, f.inner)


def test_make_folds_skips_a_repeat_that_repeats_an_earlier_partition():
    codes = np.repeat(np.arange(4), 2)
    folds = importance._make_folds(codes, 5, 3, None, 5, 0)
    assert len(folds) == 4 and {f.repeat for f in folds} == {0}


def test_fold_count_is_raised_until_every_training_part_keeps_three_groups():
    assert importance._fold_count(24, 5, None) == 5
    assert importance._fold_count(24, 5, 3) == 3
    assert importance._fold_count(3, 5, None) == 3
    assert importance._fold_count(5, 2, None) == 3
    assert importance._fold_count(4, 2, None) == 4
    assert importance._fold_count(6, 3, None) == 3
    with pytest.raises(ValueError):
        importance._fold_count(1, 5, None)
    with pytest.raises(ValueError):
        importance._fold_count(8, 1, None)


def test_few_groups_run_with_more_folds_than_requested():
    X, y, groups = make_toy_problem(n=10, random_state=18)
    five_groups = np.repeat(np.arange(5), 2)
    for light in (False, True):
        res = lambda_averaged_importance(
            X, y, five_groups, n_splits=2, learners=LINEAR, light=light, compute_diagnostics=False
        )
        assert res.diagnostics["n_cases"] == 10 and np.isfinite(res.permutation).all()
    folds = importance._make_folds(np.repeat(np.arange(5), 2), 2, 3, None, 1, 0)
    assert len(folds) == 3
    assert all(len(set(np.repeat(np.arange(5), 2)[f.train])) >= 3 for f in folds)


def test_inner_fold_count_is_clipped_to_three_to_five(monkeypatch):
    recorded = record_splits(monkeypatch)
    X, y, groups = make_toy_problem(n=24, random_state=8)
    for n_splits in (2, 3, 4, 5, 10):
        recorded.clear()
        StackedEnsemble(learners=LINEAR, n_splits=n_splits).fit(X, y, groups)
        assert [k for _, k, _ in recorded] == [min(5, max(3, n_splits))]
    recorded.clear()
    StackedEnsemble(learners=LINEAR).fit(X.iloc[:4], y.iloc[:4], np.array([0, 1, 2, 2]))
    assert [k for _, k, _ in recorded] == [3]


# ----------------------------------------------------------------------------
# Penalty grid
# ----------------------------------------------------------------------------
def test_lambda_grid_is_decreasing_and_its_first_value_zeroes_the_elastic_net():
    X, y, groups = make_toy_problem(n=30, random_state=9)
    grid = lambda_grid(X, y)
    assert grid.shape == (12,) and np.all(np.diff(grid) < 0)
    assert grid[-1] / grid[0] == pytest.approx(1e-3)
    npt.assert_allclose(grid[1:] / grid[:-1], np.full(11, 1e-3 ** (1 / 11)))
    Z = (X.to_numpy() - X.to_numpy().mean(0)) / X.to_numpy().std(0)
    t = (y.to_numpy() - y.mean()) / y.std(ddof=0)
    at_max = ElasticNet(alpha=grid[0], l1_ratio=0.5, max_iter=10000, tol=1e-10).fit(Z, t)
    assert np.all(at_max.coef_ == 0.0)
    below = ElasticNet(alpha=grid[0] * 0.99, l1_ratio=0.5, max_iter=10000, tol=1e-10).fit(Z, t)
    assert np.count_nonzero(below.coef_) >= 1


def test_lambda_grid_does_not_depend_on_the_units_of_the_data():
    X, y, groups = make_toy_problem(n=30, random_state=9)
    scaled = X.copy()
    scaled["x2"] = scaled["x2"] * 1000.0
    npt.assert_allclose(lambda_grid(scaled, 1000.0 * y + 7.0), lambda_grid(X, y), rtol=1e-10)


def test_lambda_grid_options_and_validation():
    X, y, groups = make_toy_problem(n=30, random_state=9)
    short = lambda_grid(X, y, n=4, ratio=1e-2)
    assert short.shape == (4,) and short[-1] / short[0] == pytest.approx(1e-2)
    assert lambda_grid(X, y, n=1).shape == (1,)
    assert lambda_grid(X, y, l1_ratio=1.0)[0] == pytest.approx(lambda_grid(X, y, l1_ratio=0.5)[0] / 2.0)
    npt.assert_allclose(lambda_grid(X.to_numpy(), y.to_numpy()), lambda_grid(X, y))
    with pytest.raises(ValueError):
        lambda_grid(X, y, n=0)
    with pytest.raises(ValueError):
        lambda_grid(X, y, ratio=1.5)
    with pytest.raises(ValueError):
        lambda_grid(X, y, l1_ratio=0.0)
    with pytest.raises(ValueError):
        lambda_grid(X, np.ones(30))


# ----------------------------------------------------------------------------
# Nested stacking and the label permutation statistic
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("light", [True, False], ids=["light", "default"])
def test_stacking_weights_of_every_fold_come_from_the_training_part_only(monkeypatch, light):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    seen = []
    original = importance._stack_weights

    def spy(P, target):
        seen.append((P.shape[0], len(target), np.sort(np.asarray(target))))
        return original(P, target)

    monkeypatch.setattr(importance, "_stack_weights", spy)
    lambda_averaged_importance(
        X, y, groups, learners=LINEAR, light=light, n_repeats=1, cv_repeats=2, compute_diagnostics=False
    )
    assert seen
    pool = set(np.round(y.to_numpy(), 10))
    for rows, length, values in seen:
        assert rows == length and 18 <= rows < len(y)
        assert set(np.round(values, 10)) < pool


def test_the_label_permutation_statistic_is_one_procedure_for_the_observed_and_the_permuted_target(monkeypatch):
    X, y, groups = noise_problem()
    values, target = X.to_numpy(), y.to_numpy()
    statistic, weights = [], []
    original_statistic, original_weights = importance._stack_cv_r2, importance._stack_weights

    def spy_statistic(X_, y_, names, folds, plan, model_seed):
        statistic.append((y_.copy(), folds, plan, names))
        return original_statistic(X_, y_, names, folds, plan, model_seed)

    def spy_weights(P, target_):
        weights.append((P.shape[0], len(target_)))
        return original_weights(P, target_)

    monkeypatch.setattr(importance, "_stack_cv_r2", spy_statistic)
    monkeypatch.setattr(importance, "_stack_weights", spy_weights)
    p = importance._label_permutation_p(values, target, np.arange(30), LINEAR, 5, 9, 1, 0, 2)
    assert len(statistic) == 10
    observed = statistic[0]
    npt.assert_array_equal(observed[0], target)
    for permuted, folds, plan, names in statistic[1:]:
        assert folds is observed[1] and plan is observed[2] is importance._LIGHT_PLAN and names == LINEAR
        npt.assert_array_equal(np.sort(permuted), np.sort(target))
        assert not np.array_equal(permuted, target)
    assert len(weights) == 10 * len(observed[1]) * importance._LIGHT_PLAN.n_lambdas
    assert {rows for rows, length in weights} == {20} and all(rows == length for rows, length in weights)
    assert 0.1 <= p <= 1.0


def test_the_label_permutation_p_value_counts_permuted_statistics_at_least_as_large_as_the_observed(monkeypatch):
    X, y, groups = noise_problem()
    values, target = X.to_numpy(), y.to_numpy()
    sequence = iter([0.5, 0.7, 0.5, 0.2, 0.9, 0.1])
    monkeypatch.setattr(importance, "_stack_cv_r2", lambda *args, **kwargs: next(sequence))
    p = importance._label_permutation_p(values, target, np.arange(30), LINEAR, 5, 5, 1, 0, 2)
    assert p == pytest.approx((1 + 3) / 6)


# ----------------------------------------------------------------------------
# Decisions that rounding noise must not change
# ----------------------------------------------------------------------------
def label_p_with_statistics(monkeypatch, statistics):
    """P-value of the label permutation test for the given observed statistic followed by permuted ones."""
    X, y, groups = noise_problem(n=24)
    queue = iter(statistics)
    monkeypatch.setattr(importance, "_stack_cv_r2", lambda *args, **kwargs: next(queue))
    return importance._label_permutation_p(X.to_numpy(), y.to_numpy(), np.arange(24), LINEAR, 5, len(statistics) - 1, 1, 0, 2)


def test_a_permuted_statistic_that_ties_with_the_observed_one_up_to_rounding_noise_counts_as_an_exceedance(monkeypatch):
    observed = 0.3
    permuted = [0.1] * 4 + [observed + 1e-12, observed - 1e-12, observed + 5e-13, observed - 5e-10] + [0.5] * 3
    assert label_p_with_statistics(monkeypatch, [observed] + permuted) == pytest.approx((1 + 3 + 3 + 1) / 12)
    beyond = [observed - 2e-9, observed - 1e-8, observed - 1e-6, observed - 1e-3]
    assert label_p_with_statistics(monkeypatch, [observed] + beyond + [0.1] * 4) == pytest.approx(1 / 9)


def test_the_tie_tolerance_is_relative_to_the_observed_statistic_when_that_exceeds_one(monkeypatch):
    observed = -5.0
    inside = [observed - 4e-9, observed + 1e-12]
    outside = [observed - 6e-9, observed - 1e-6]
    assert label_p_with_statistics(monkeypatch, [observed] + inside + outside) == pytest.approx(3 / 5)
    assert label_p_with_statistics(monkeypatch, [0.0, -5e-10, -2e-9, 3e-10]) == pytest.approx(3 / 4)


def tie_prone_problem():
    """Twenty-four cases whose target is one for three of them and zero for the others, one case per group.

    Many permuted targets put the same values into the same folds as the observed target, so the permuted
    statistics tie with the observed one.
    """
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.standard_normal((24, 6)), columns=[f"x{j}" for j in range(6)])
    y = pd.Series(np.r_[np.ones(3), np.zeros(21)], name="effect")
    return X, y, np.arange(24)


def perturbed(X, y, seed, scale=1e-13):
    """Copies of the data with noise of ``scale`` times the standard deviation of each column added."""
    rng = np.random.default_rng(seed)
    values = X.to_numpy()
    noisy = values + scale * values.std(axis=0) * rng.standard_normal(values.shape)
    target = y.to_numpy() + scale * y.std() * rng.standard_normal(y.size)
    return pd.DataFrame(noisy, index=X.index, columns=X.columns), pd.Series(target, index=y.index, name=y.name)


def test_the_p_value_of_the_label_permutation_test_does_not_change_with_rounding_noise_in_the_data(monkeypatch):
    X, y, groups = tie_prone_problem()
    codes = importance._group_codes(groups, 24)

    def p_value(features, target):
        return importance._label_permutation_p(features.to_numpy(), target.to_numpy(), codes, LINEAR, 5, 20, 7, 3, 4)

    base = p_value(X, y)
    assert {p_value(*perturbed(X, y, seed)) for seed in range(6)} == {base}
    monkeypatch.setattr(importance, "_TIE_TOLERANCE", 0.0)
    assert len({p_value(*perturbed(X, y, seed)) for seed in range(6)}) > 1


def grouped_noise_problem():
    """Thirty cases of 8 standard normal features and an unrelated standard normal target in 24 groups."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 8)), columns=[f"x{j}" for j in range(8)])
    y = pd.Series(rng.standard_normal(30), name="effect")
    return X, y, np.arange(30) % 24


PERTURBATION_PROBLEMS = {
    "signal": lambda: make_toy_problem(n=30, random_state=5),
    "ties": tie_prone_problem,
    "noise": grouped_noise_problem,
}


@pytest.mark.parametrize(
    "problem, learners, n_noise",
    [
        ("signal", LINEAR, 3),
        ("ties", LINEAR, 3),
        ("signal", None, 2),
        ("noise", None, 2),
    ],
    ids=["signal-linear", "ties-linear", "signal-all-learners", "noise-all-learners"],
)
def test_the_result_does_not_change_with_noise_of_1e_13_in_the_inputs(quick_halves, problem, learners, n_noise):
    X, y, groups = PERTURBATION_PROBLEMS[problem]()
    options = dict(learners=learners, light=True, n_perm=20, random_state=2)
    base = lambda_averaged_importance(X, y, groups, **options)
    for seed in range(n_noise):
        noisy_X, noisy_y = perturbed(X, y, seed)
        other = lambda_averaged_importance(noisy_X, noisy_y, groups, **options)
        assert other.diagnostics["perm_p_value"] == base.diagnostics["perm_p_value"]
        assert other.diagnostics["usable"] is base.diagnostics["usable"]
        assert other.no_signal is base.no_signal
        npt.assert_allclose(other.permutation_normalised, base.permutation_normalised, rtol=0.0, atol=1e-6)
        for key in ("cv_r2_stack", "cv_r2_best_single", "cv_r2_null", "split_half_correlation"):
            assert other.diagnostics[key] == pytest.approx(base.diagnostics[key], abs=1e-6), key
        assert other.summary_frame()["feature"].tolist() == base.summary_frame()["feature"].tolist()


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_tree_learners_give_the_same_predictions_for_targets_that_differ_by_rounding_noise(name):
    rng = np.random.default_rng(0)
    X, y, X_new = rng.standard_normal((24, 8)), rng.standard_normal(24), rng.standard_normal((200, 8))
    base = importance._fit_model(name, None, 0.5, 4, 0, X, y).predict(X_new)
    assert np.ptp(base) > 0.1
    for seed in range(5):
        for level in (1e-15, 1e-13):
            noisy = y * (1.0 + level * np.random.default_rng(100 + seed).standard_normal(24))
            other = importance._fit_model(name, None, 0.5, 4, 0, X, noisy).predict(X_new)
            npt.assert_allclose(other, base, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_tree_learners_follow_a_target_that_is_rescaled_and_shifted(name):
    rng = np.random.default_rng(1)
    X, y, X_new = rng.standard_normal((24, 8)), rng.standard_normal(24), rng.standard_normal((100, 8))
    base = importance._fit_model(name, None, 0.5, 6, 0, X, y).predict(X_new)
    for scale, shift in ((1000.0, 50.0), (1e-9, 3e-9), (1e6, -2e7)):
        other = importance._fit_model(name, None, 0.5, 6, 0, X, scale * y + shift).predict(X_new)
        npt.assert_allclose(other, scale * base + shift, rtol=1e-9, atol=1e-9 * scale)


def test_the_target_of_the_tree_learners_is_centred_scaled_and_rounded_to_six_decimals():
    y = np.array([3.0, 5.0, 9.0, 1.0])
    transformer = importance._RoundedTarget().fit(y)
    assert transformer.mean_ == pytest.approx(4.5) and transformer.scale_ == pytest.approx(y.std())
    rounded = transformer.transform(y)
    npt.assert_array_equal(rounded, np.round((y - 4.5) / y.std(), 6))
    assert importance._TREE_DECIMALS == 6
    npt.assert_allclose(transformer.inverse_transform(rounded), y, atol=1e-6 * y.std())
    noisy = y * (1.0 + 1e-14 * np.random.default_rng(0).standard_normal(4))
    npt.assert_array_equal(importance._RoundedTarget().fit(noisy).transform(noisy), rounded)
    column = transformer.transform(y.reshape(-1, 1))
    assert column.shape == (4, 1) and transformer.inverse_transform(column).shape == (4, 1)


def test_a_constant_target_keeps_the_scale_one_in_the_tree_learners():
    for value in (0.0, 7.0):
        constant = np.full(6, value)
        transformer = importance._RoundedTarget().fit(constant)
        assert transformer.scale_ == 1.0 and transformer.mean_ == value
        npt.assert_array_equal(transformer.transform(constant), np.zeros(6))
        noisy = value * (1.0 + 1e-16 * np.arange(6.0))
        assert importance._RoundedTarget().fit(noisy).scale_ == 1.0
    X = np.random.default_rng(0).standard_normal((12, 3))
    for name in ("random_forest", "gradient_boosting"):
        model = importance._fit_model(name, None, 0.5, 4, 0, X, np.full(12, 2.5))
        npt.assert_allclose(model.predict(X), 2.5)


def test_the_best_penalty_is_the_largest_one_among_errors_that_tie_up_to_rounding_noise():
    best = importance._best_penalty
    grid = np.array([1.0, 0.5, 0.25, 0.125])
    assert best(np.array([3.0, 2.0, 1.0, 1.5]), grid, 4.0) == 2
    tied = np.array([1.0 + 3e-12, 1.0, 1.0 - 4e-12, 1.0 + 1e-12])
    assert best(tied, grid, 1.0) == 0
    assert best(tied[::-1], grid[::-1], 1.0) == 3
    assert best(np.array([1.0 + 5e-9, 1.0, 1.0 + 5e-9, 1.0 + 1e-8]), grid, 1.0) == 1
    assert best(np.array([1.0 + 5e-6, 1.0]), np.array([1.0, 0.5]), 1e4) == 0
    assert best(np.array([1.0 + 5e-6, 1.0]), np.array([1.0, 0.5]), 1.0) == 1
    assert best(np.array([2.0]), np.array([0.3]), 1.0) == 0


def test_the_diagnostics_use_the_best_penalty_of_the_tolerant_rule(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    chosen = []
    original = importance._best_penalty

    def spy(cv_mse, lambdas, variance):
        chosen.append((cv_mse.copy(), lambdas.copy(), variance))
        return original(cv_mse, lambdas, variance)

    monkeypatch.setattr(importance, "_best_penalty", spy)
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    assert len(chosen) == 1
    cv_mse, lambdas, variance = chosen[0]
    npt.assert_array_equal(cv_mse, res.cv_mse_by_lambda)
    npt.assert_array_equal(lambdas, res.lambdas)
    assert variance == pytest.approx(float(np.var(y, ddof=1)))
    sst = float(np.sum((y - y.mean()) ** 2))
    best = int(np.argmin(res.cv_mse_by_lambda))
    assert res.diagnostics["cv_r2_stack"] == pytest.approx(1.0 - len(y) * res.cv_mse_by_lambda[best] / sst)
    monkeypatch.setattr(importance, "_best_penalty", lambda *args: len(args[0]) - 1)
    last = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    assert last.diagnostics["cv_r2_stack"] == pytest.approx(1.0 - len(y) * res.cv_mse_by_lambda[-1] / sst)


def noisy_copies(X, level=1e-14, n=40):
    """Copies of a matrix with noise of the given absolute size added."""
    return [X + level * np.random.default_rng(seed).standard_normal(X.shape) for seed in range(n)]


def test_feature_clusters_do_not_depend_on_rounding_noise_when_two_joins_tie():
    R = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.7], [0.9, 0.7, 1.0]])
    X = with_correlations(R, n=60, seed=1)
    npt.assert_allclose(np.corrcoef(X, rowvar=False), R, atol=1e-12)
    labellings = {tuple(importance._feature_clusters(copy, 0.8)) for copy in noisy_copies(X)}
    assert labellings == {(0, 0, 1)}


def test_a_pair_correlated_exactly_at_the_threshold_is_never_joined():
    X = with_correlations(np.array([[1.0, 0.8], [0.8, 1.0]]), n=60)
    assert {tuple(importance._feature_clusters(copy, 0.8)) for copy in noisy_copies(X)} == {(0, 1)}
    assert {tuple(importance._feature_clusters(copy, 0.79)) for copy in noisy_copies(X)} == {(0, 0)}


def result_with_normalised(names, permutation, normalised):
    """Importance result whose normalised importance is given and need not be the normalised permutation."""
    zeros = np.zeros(len(names))
    values = {
        "permutation": np.asarray(permutation, dtype=float),
        "permutation_normalised": np.asarray(normalised, dtype=float),
        "coefficient_path": zeros,
        "selection_frequency": zeros,
    }
    return ImportanceResult(
        lambdas=np.array([1.0]), lambda_weights=np.array([1.0]), cv_mse_by_lambda=np.array([1.0]),
        feature_names=list(names), **{key: pd.Series(value, index=list(names), name=key) for key, value in values.items()},
    )


def test_the_summary_frame_treats_values_that_agree_up_to_rounding_noise_as_ties():
    names = ["a", "b", "c", "d"]
    near = result_with_normalised(names, [3.0, 2.0, 1.0, 4.0], [0.3, 0.3 + 4e-16, 0.3 - 3e-16, 0.1])
    assert len({0.3, 0.3 + 4e-16, 0.3 - 3e-16}) == 3
    assert near.summary_frame()["feature"].tolist() == ["a", "b", "c", "d"]
    exact = result_with_normalised(names, [2.0, 2.0 + 1e-14, 1.0, 2.0 - 1e-14], [0.25] * 4)
    assert exact.summary_frame()["feature"].tolist() == ["a", "b", "d", "c"]
    apart = result_with_normalised(names, [3.0, 2.0, 1.0, 4.0], [0.3, 0.3 + 1e-6, 0.3 - 1e-6, 0.1])
    assert apart.summary_frame()["feature"].tolist() == ["b", "a", "c", "d"]


def test_elastic_net_coefficients_at_the_noise_level_of_the_target_are_not_selected(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    original = importance._fit_linear_paths

    def with_noise(X_fit, y_fit, lambdas, l1_ratio, names):
        paths = original(X_fit, y_fit, lambdas, l1_ratio, names)
        shape = paths.coef["elastic_net"].shape
        noise = 1e-14 * float(y_fit.std()) * np.random.default_rng(0).standard_normal(shape)
        paths.coef["elastic_net"] = paths.coef["elastic_net"] + noise
        return paths

    monkeypatch.setattr(importance, "_fit_linear_paths", with_noise)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False, n_repeats=2)
    silent = lambda_averaged_importance(X, y, groups, lambdas=[1e3], **options)
    assert (silent.selection_frequency == 0.0).all()
    assert (silent.coefficient_path > 0.0).all() and (silent.coefficient_path < 1e-12 * float(y.std())).all()
    active = lambda_averaged_importance(X, y, groups, lambdas=[1e-3], **options)
    assert (active.selection_frequency[["x0", "x1"]] == 1.0).all()


def test_a_vector_that_is_constant_up_to_rounding_noise_has_no_correlation():
    noisy = np.full(9, 1.0 / 9.0)
    noisy[::2] = np.nextafter(noisy[::2], 1.0)
    assert noisy.std() > 0.0
    other = np.random.default_rng(0).standard_normal(9)
    assert np.isnan(importance._pearson(noisy, other)) and np.isnan(importance._pearson(other, noisy))


def test_the_penalty_grid_ignores_a_column_that_is_constant_up_to_rounding_noise():
    X, y, groups = noise_problem(n=30)
    flat, noisy = X.copy(), X.copy()
    flat["x7"] = 5.0
    noisy["x7"] = 5.0 + np.where(y.to_numpy() > np.median(y), 8.9e-16, 0.0)
    assert noisy["x7"].nunique() == 2 and noisy["x7"].std() > 0.0
    npt.assert_allclose(lambda_grid(noisy, y), lambda_grid(flat, y), rtol=1e-12)
    npt.assert_allclose(lambda_grid(flat, y)[0], lambda_grid(X.drop(columns="x7"), y)[0], rtol=1e-12)


# ----------------------------------------------------------------------------
# Importance estimates
# ----------------------------------------------------------------------------
def test_importance_is_unchanged_when_one_column_is_multiplied_by_1000(toy30):
    X, y, groups, base = toy30
    scaled = X.copy()
    scaled["x1"] = scaled["x1"] * 1000.0
    other = lambda_averaged_importance(scaled, y, groups, light=True, compute_diagnostics=False, random_state=1)
    assert_same_result(base, other, rtol=1e-6, atol=1e-9)


def test_penalised_learners_do_not_depend_on_the_units_of_the_target(quick_halves):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    common = dict(learners=LINEAR, light=True, n_perm=9, random_state=1)
    base = lambda_averaged_importance(X, y, groups, **common)
    other = lambda_averaged_importance(X, 1000.0 * y + 50.0, groups, **common)
    npt.assert_allclose(other.permutation_normalised, base.permutation_normalised, rtol=1e-8, atol=1e-10)
    npt.assert_allclose(other.permutation, 1e6 * base.permutation, rtol=1e-8, atol=1e-10)
    npt.assert_allclose(other.cluster_importance, 1e6 * base.cluster_importance, rtol=1e-8, atol=1e-10)
    npt.assert_allclose(other.coefficient_path, 1000.0 * base.coefficient_path, rtol=1e-8, atol=1e-10)
    npt.assert_allclose(other.selection_frequency, base.selection_frequency)
    npt.assert_allclose(other.lambdas, base.lambdas, rtol=1e-10)
    npt.assert_allclose(other.lambda_weights, base.lambda_weights, rtol=1e-8)
    npt.assert_allclose(other.cv_mse_by_lambda, 1e6 * base.cv_mse_by_lambda, rtol=1e-8)
    for key in ("cv_r2_stack", "cv_r2_best_single", "cv_r2_null", "perm_p_value", "split_half_correlation"):
        assert other.diagnostics[key] == pytest.approx(base.diagnostics[key], rel=1e-8, abs=1e-10)


def test_stack_importance_is_unchanged_when_the_target_is_rescaled_and_shifted(toy30):
    X, y, groups, base = toy30
    other = lambda_averaged_importance(
        X, 1000.0 * y + 50.0, groups, light=True, compute_diagnostics=False, random_state=1
    )
    npt.assert_allclose(other.permutation_normalised, base.permutation_normalised, atol=1e-6)
    npt.assert_allclose(other.permutation, 1e6 * base.permutation, rtol=1e-6)
    npt.assert_allclose(other.lambda_weights, base.lambda_weights, atol=1e-6)
    assert other.diagnostics["cv_r2_stack"] == pytest.approx(base.diagnostics["cv_r2_stack"], abs=1e-6)


def test_identical_random_state_gives_identical_results(toy30):
    X, y, groups, first = toy30
    second = lambda_averaged_importance(X, y, groups, light=True, compute_diagnostics=False, random_state=1)
    assert_same_result(first, second)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    seeded = [lambda_averaged_importance(X, y, groups, random_state=s, **options) for s in (3, 3, 4)]
    assert_same_result(seeded[0], seeded[1])
    assert not np.array_equal(seeded[0].permutation.to_numpy(), seeded[2].permutation.to_numpy())
    numpy_seed = lambda_averaged_importance(X, y, groups, random_state=np.int64(3), **options)
    assert_same_result(seeded[0], numpy_seed)


def test_identical_random_state_gives_identical_diagnostics(quick_halves):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    options = dict(learners=LINEAR, light=True, n_perm=9)
    first = lambda_averaged_importance(X, y, groups, random_state=3, **options)
    second = lambda_averaged_importance(X, y, groups, random_state=3, **options)
    assert_same_result(first, second)
    assert np.isfinite(first.diagnostics["split_half_correlation"])


def test_the_seed_changes_the_assignment_of_groups_to_folds_and_with_it_the_linear_results():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    runs = [
        lambda_averaged_importance(X, y, groups, learners=LINEAR, compute_diagnostics=False, random_state=s)
        for s in range(3)
    ]
    mse = np.array([run.cv_mse_by_lambda for run in runs])
    assert len({tuple(row) for row in np.round(mse, 12)}) == 3
    assert mse.std(axis=0).min() > 0.0


@pytest.mark.parametrize("learners", [LINEAR, None], ids=["linear", "all"])
def test_results_do_not_depend_on_the_order_of_the_rows_or_the_labels_of_the_groups(learners, quick_halves):
    X, y, groups = make_toy_problem(n=30, random_state=9)
    options = dict(learners=learners, light=True, random_state=2, n_perm=9, compute_diagnostics=learners is not None)
    base = lambda_averaged_importance(X, y, groups, **options)
    for k, relabel in ((0, False), (1, True)):
        order = np.random.default_rng(k).permutation(30)
        X2, y2, groups2 = reordered(X, y, groups, order, relabel=relabel)
        assert_same_result(base, lambda_averaged_importance(X2, y2, groups2, **options), rtol=1e-9, atol=1e-12)


def test_results_with_every_case_in_its_own_group_do_not_depend_on_the_order_of_the_rows(quick_halves):
    X, y, groups = noise_problem(n=24, d=5, seed=3)
    options = dict(learners=LINEAR, light=True, random_state=2, n_perm=9)
    base = lambda_averaged_importance(X, y, None, **options)
    order = np.random.default_rng(5).permutation(24)
    other = lambda_averaged_importance(X.iloc[order], y.iloc[order], None, **options)
    assert_same_result(base, other, rtol=1e-9, atol=1e-12)
    assert base.diagnostics["perm_p_value"] == other.diagnostics["perm_p_value"]


def test_cv_metrics_and_importance_are_averaged_over_the_repeats_of_the_fold_assignment(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    values, target = X.to_numpy(), y.to_numpy()
    order, canonical = importance._canonical_rows(values, target, importance._group_codes(groups, 30))
    Xc, yc = values[order], target[order]
    all_folds = importance._make_folds(canonical, 5, 5, None, 3, 77)
    assert {f.repeat for f in all_folds} == {0, 1, 2}
    grid = lambda_grid(Xc, yc, n=3)

    def estimate(folds):
        monkeypatch.setattr(importance, "_make_folds", lambda *args, **kwargs: folds)
        return importance._estimate(
            Xc, yc, canonical, grid, LINEAR, 5, 4, "uniform", 1.0, importance._FULL_PLAN, 0, 5, 77, 3
        )

    pooled = estimate(all_folds)
    singles = [estimate([f for f in all_folds if f.repeat == r]) for r in range(3)]
    assert len({tuple(np.round(s.cv_mse, 12)) for s in singles}) == 3
    npt.assert_allclose(pooled.cv_mse, np.mean([s.cv_mse for s in singles], axis=0), rtol=1e-12)
    npt.assert_allclose(pooled.perm, np.mean([s.perm for s in singles], axis=0), rtol=1e-10, atol=1e-12)
    npt.assert_allclose(pooled.coef, np.mean([s.coef for s in singles], axis=0), rtol=1e-10)
    npt.assert_allclose(pooled.sel, np.mean([s.sel for s in singles], axis=0), rtol=1e-12)


def test_cv_repeats_option_changes_the_estimate_and_is_validated():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    options = dict(learners=LINEAR, compute_diagnostics=False, random_state=1)
    one = lambda_averaged_importance(X, y, groups, cv_repeats=1, **options)
    three = lambda_averaged_importance(X, y, groups, **options)
    assert inspect.signature(lambda_averaged_importance).parameters["cv_repeats"].default == 3
    assert not np.allclose(one.cv_mse_by_lambda, three.cv_mse_by_lambda)
    with pytest.raises(ValueError):
        lambda_averaged_importance(X, y, groups, cv_repeats=0, **options)


def test_the_importance_of_a_penalty_does_not_depend_on_the_other_penalties_of_the_grid():
    X, y, groups = make_toy_problem(n=30, random_state=11)
    grid = lambda_grid(X, y, n=3)
    common = dict(learners=LINEAR, compute_diagnostics=False, n_repeats=3, random_state=4)
    combined = lambda_averaged_importance(X, y, groups, lambdas=grid, weighting="cv", **common)
    singles = [lambda_averaged_importance(X, y, groups, lambdas=[lam], **common) for lam in grid]
    weights = combined.lambda_weights
    assert len(set(np.round(weights, 6))) == 3
    mixture = sum(w * s.permutation for w, s in zip(weights, singles))
    npt.assert_allclose(combined.permutation, mixture, rtol=1e-5)
    mixture = sum(w * s.coefficient_path for w, s in zip(weights, singles))
    npt.assert_allclose(combined.coefficient_path, mixture, rtol=1e-5)
    npt.assert_allclose(combined.selection_frequency, np.mean([s.selection_frequency for s in singles], axis=0))
    npt.assert_allclose(combined.cv_mse_by_lambda, [s.cv_mse_by_lambda[0] for s in singles], rtol=1e-5)


def ols_reference(values, target, folds):
    """Least-squares reference on the given folds.

    Ordinary least squares with an intercept is fitted on the training part of every fold.  For each
    feature the function returns the expected increase of the held-out squared error when the feature
    is permuted within the held-out part, ``2 b^2 Var(x)``, and the absolute coefficient on the scale
    of the training standard deviation of the feature, both averaged over the folds with the numbers
    of held-out cases as weights.
    """
    total = sum(f.test.size for f in folds)
    increase = np.zeros(values.shape[1])
    standardised = np.zeros(values.shape[1])
    for f in folds:
        design = np.column_stack([np.ones(f.train.size), values[f.train]])
        raw = np.linalg.lstsq(design, target[f.train], rcond=None)[0][1:]
        increase += f.test.size / total * 2.0 * raw**2 * values[f.test].var(axis=0)
        standardised += f.test.size / total * np.abs(raw) * values[f.train].std(axis=0)
    return increase, standardised


def test_permutation_importance_and_coefficients_follow_least_squares_for_a_linear_target(monkeypatch):
    recorded = []
    original = importance._make_folds

    def spy(*args, **kwargs):
        recorded.append(original(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(importance, "_make_folds", spy)
    increase_ratios, coefficient_ratios = [], []
    for seed in range(5):
        X, y, groups = linear_problem(seed=seed)
        recorded.clear()
        res = lambda_averaged_importance(
            X, y, groups, lambdas=[0.01], learners=LINEAR, weighting="uniform", n_repeats=40,
            compute_diagnostics=False, random_state=seed,
        )
        order, _ = importance._canonical_rows(X.to_numpy(), y.to_numpy(), np.arange(len(y)))
        increase, standardised = ols_reference(X.to_numpy()[order], y.to_numpy()[order], recorded[0])
        increase_ratios.append(res.permutation.to_numpy()[:2] / increase[:2])
        coefficient_ratios.append(res.coefficient_path.to_numpy()[:2] / standardised[:2])
        assert np.all(np.abs(res.permutation.to_numpy()[2:]) < 0.05)
        assert res.permutation_normalised[["x2", "x3", "x4"]].sum() < 0.01
        assert res.selection_frequency["x0"] == 1.0 and res.selection_frequency["x1"] == 1.0
        assert res.diagnostics["cv_r2_stack"] > 0.95
    npt.assert_allclose(np.mean(increase_ratios, axis=0), [1.0, 1.0], atol=0.05)
    npt.assert_allclose(coefficient_ratios, 1.0, atol=0.03)


def test_coefficient_path_is_weighted_by_the_number_of_held_out_cases():
    X, y, groups = make_toy_problem(n=33, d=4, random_state=3)
    values, target = X.to_numpy(), y.to_numpy()
    order, canonical = importance._canonical_rows(values, target, importance._group_codes(groups, 33))
    Xc, yc = values[order], target[order]
    folds = importance._make_folds(canonical, 5, 5, None, 1, 9)
    sizes = np.array([f.test.size for f in folds])
    assert sizes.min() != sizes.max()
    grid = lambda_grid(Xc, yc, n=2)
    est = importance._estimate(Xc, yc, canonical, grid, LINEAR, 5, 2, "uniform", 1.0, importance._FULL_PLAN, 0, 1, 9, 1)
    expected = np.zeros(4)
    for f, size in zip(folds, sizes):
        fit = importance._fit_linear_paths(Xc[f.train], yc[f.train], grid, 0.5, ["elastic_net"])
        expected += size / sizes.sum() * np.abs(fit.coef["elastic_net"]).mean(axis=1)
    npt.assert_allclose(est.coef, expected, rtol=1e-9)
    per_fold = [
        np.abs(importance._fit_linear_paths(Xc[f.train], yc[f.train], grid, 0.5, ["elastic_net"]).coef["elastic_net"])
        for f in folds
    ]
    equal_weights = np.mean([paths.mean(axis=1) for paths in per_fold], axis=0)
    assert not np.allclose(est.coef, equal_weights, rtol=1e-6)


def test_penalty_weights_follow_the_cross_validated_error():
    X, y, groups = make_toy_problem(n=30, random_state=11)
    common = dict(learners=LINEAR, n_repeats=2, compute_diagnostics=False)
    cv = lambda_averaged_importance(X, y, groups, weighting="cv", temperature=0.5, **common)
    uniform = lambda_averaged_importance(X, y, groups, weighting="uniform", **common)
    cold = lambda_averaged_importance(X, y, groups, weighting="cv", temperature=1e-7, **common)
    assert cv.lambdas.size == 12
    raw = np.exp(-(cv.cv_mse_by_lambda - cv.cv_mse_by_lambda.min()) / (0.5 * np.var(y, ddof=1)))
    npt.assert_allclose(cv.lambda_weights, raw / raw.sum())
    npt.assert_allclose(uniform.lambda_weights, np.full(12, 1.0 / 12.0))
    npt.assert_allclose(uniform.cv_mse_by_lambda, cv.cv_mse_by_lambda)
    assert cold.lambda_weights[np.argmin(cold.cv_mse_by_lambda)] > 0.999
    assert len(set(np.round(cv.lambda_weights, 6))) > 1


@pytest.mark.usefixtures("deterministic_folds")
def test_best_single_learner_r2_is_that_of_the_best_penalty_without_stacking():
    X, y, groups = make_toy_problem(n=30, random_state=12)
    values, target = X.to_numpy(), y.to_numpy()
    grid = lambda_grid(X, y, n=3)
    res = lambda_averaged_importance(
        X, y, groups, lambdas=grid, learners=("ridge",), cv_repeats=1, compute_diagnostics=False
    )
    order, canonical = importance._canonical_rows(values, target, importance._group_codes(groups, 30))
    Xc, yc = values[order], target[order]
    sse = np.zeros(3)
    for train, test in kfold_by_group(canonical, 5, None):
        for k, lam in enumerate(grid):
            fit = importance._fit_model("ridge", lam, 0.5, 100, 0, Xc[train], yc[train])
            sse[k] += np.sum((yc[test] - fit.predict(Xc[test])) ** 2)
    expected = 1.0 - sse.min() / (30.0 * np.var(yc))
    assert res.diagnostics["cv_r2_best_single"] == pytest.approx(expected, rel=1e-6)


@pytest.fixture
def deterministic_folds(monkeypatch):
    """Grouped K-fold of scikit-learn in place of the random assignment of groups to folds."""
    monkeypatch.setattr(importance, "_random_group_splits", kfold_by_group)


def reference_sse(values, target, groups, grid, learners, n_splits, outer_k):
    """Held-out squared error of the stacked ensemble fitted on each training part, per penalty."""
    order, canonical = importance._canonical_rows(values, target, importance._group_codes(groups, len(target)))
    Xc, yc = values[order], target[order]
    sse = np.zeros(len(grid))
    for train, test in kfold_by_group(canonical, outer_k, None):
        for k, lam in enumerate(grid):
            model = StackedEnsemble(
                lam=lam, learners=learners, n_splits=n_splits, n_estimators=importance._FULL_PLAN.n_estimators
            )
            model.fit(Xc[train], yc[train], canonical[train])
            sse[k] += np.sum((yc[test] - model.predict(Xc[test])) ** 2)
    return sse


@pytest.mark.usefixtures("deterministic_folds")
def test_nested_engine_reproduces_stacked_ensemble_fits_with_linear_learners():
    X, y, groups = make_toy_problem(n=30, random_state=13)
    grid = lambda_grid(X, y, n=2)
    res = lambda_averaged_importance(
        X, y, groups, lambdas=grid, learners=LINEAR, n_repeats=2, cv_repeats=1, compute_diagnostics=False
    )
    sse = reference_sse(X.to_numpy(), y.to_numpy(), groups, grid, LINEAR, 5, 5)
    npt.assert_allclose(res.cv_mse_by_lambda, sse / len(y), rtol=1e-6)


@pytest.mark.usefixtures("deterministic_folds")
def test_nested_engine_reproduces_stacked_ensemble_fits_with_boosting(monkeypatch):
    monkeypatch.setattr(importance, "_FULL_PLAN", importance._Plan(12, 20, 3, None, None))
    X, y, groups = make_toy_problem(n=20, random_state=14)
    grid = np.array([0.5, 0.02])
    learners = ("elastic_net", "gradient_boosting")
    res = lambda_averaged_importance(
        X, y, groups, lambdas=grid, learners=learners, n_splits=3, n_repeats=1, cv_repeats=1,
        compute_diagnostics=False,
    )
    sse = reference_sse(X.to_numpy(), y.to_numpy(), groups, grid, learners, 3, 3)
    npt.assert_allclose(res.cv_mse_by_lambda, sse / len(y), rtol=1e-6)


@pytest.mark.usefixtures("deterministic_folds")
def test_weights_of_the_engine_come_from_the_inner_folds_of_the_training_part(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=15)
    values, target = X.to_numpy(), y.to_numpy()
    order, canonical = importance._canonical_rows(values, target, importance._group_codes(groups, 30))
    Xc, yc = values[order], target[order]
    seen = []
    original = importance._stack_weights

    def spy(P, t):
        seen.append((P.copy(), t.copy(), original(P, t)))
        return seen[-1][2]

    monkeypatch.setattr(importance, "_stack_weights", spy)
    grid = lambda_grid(Xc, yc, n=2)
    lambda_averaged_importance(X, y, groups, lambdas=grid, learners=LINEAR, cv_repeats=1, compute_diagnostics=False)
    assert len(seen) == 5 * 2
    outer = kfold_by_group(canonical, 5, None)
    for (P, t, _), (train, _), lam in zip(seen[::2], outer, [grid[0]] * 5):
        npt.assert_array_equal(t, yc[train])
        inner = kfold_by_group(canonical[train], 5, None)
        expected = np.empty((train.size, 2))
        for k, name in enumerate(LINEAR):
            for tr, te in inner:
                fit = importance._fit_model(name, lam, 0.5, 100, 0, Xc[train][tr], yc[train][tr])
                expected[te, k] = fit.predict(Xc[train][te])
        npt.assert_allclose(P, expected, rtol=1e-6, atol=1e-8)


# ----------------------------------------------------------------------------
# Feature clusters
# ----------------------------------------------------------------------------
def with_correlations(R, n=200, seed=0):
    """Columns whose sample correlation matrix is exactly ``R``."""
    Z = np.random.default_rng(seed).standard_normal((n, R.shape[0]))
    q, _ = np.linalg.qr(Z - Z.mean(axis=0))
    return (q * np.sqrt(n)) @ np.linalg.cholesky(R).T


def test_feature_clusters_use_complete_linkage_on_the_absolute_correlation():
    R = np.array([[1.0, 0.95, 0.7], [0.95, 1.0, 0.85], [0.7, 0.85, 1.0]])
    X = with_correlations(R)
    npt.assert_allclose(np.corrcoef(X, rowvar=False), R, atol=1e-12)
    assert importance._feature_clusters(X, 0.8).tolist() == [0, 0, 1]
    assert importance._feature_clusters(X, 0.65).tolist() == [0, 0, 0]
    assert importance._feature_clusters(X, 0.99).tolist() == [0, 1, 2]
    assert importance._feature_clusters(X, None).tolist() == [0, 1, 2]
    assert importance._feature_clusters(X * np.array([1.0, -1.0, 1.0]), 0.8).tolist() == [0, 0, 1]


def test_feature_clusters_label_clusters_in_order_of_appearance_and_leave_constant_columns_alone():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(50)
    X = np.column_stack([np.full(50, 3.0), base, rng.standard_normal(50), -base, 2.0 * base + 1.0])
    assert importance._feature_clusters(X, 0.8).tolist() == [0, 1, 2, 1, 1]
    assert importance._feature_clusters(X[:, :1], 0.8).tolist() == [0]
    assert importance._feature_clusters(X, 0.8).dtype.kind == "i"


def test_joint_permutation_blocks_move_all_members_of_a_cluster_by_one_row_permutation():
    rng = np.random.default_rng(0)
    X_test = rng.standard_normal((6, 4))
    perms = np.argsort(rng.random((4, 2, 6)), axis=2)
    cluster_perms = np.argsort(rng.random((1, 2, 6)), axis=2)
    batch = importance._permuted_batch(X_test, perms, [np.array([0, 2])], cluster_perms)
    assert batch.shape == ((1 + 4 * 2 + 1 * 2) * 6, 4)
    blocks = batch.reshape(11, 6, 4)
    npt.assert_array_equal(blocks[0], X_test)
    for j in range(4):
        for r in range(2):
            expected = X_test.copy()
            expected[:, j] = X_test[perms[j, r], j]
            npt.assert_array_equal(blocks[1 + j * 2 + r], expected)
    for r in range(2):
        expected = X_test.copy()
        expected[:, [0, 2]] = X_test[cluster_perms[0, r]][:, [0, 2]]
        npt.assert_array_equal(blocks[9 + r], expected)
        npt.assert_array_equal(blocks[9 + r][:, [1, 3]], X_test[:, [1, 3]])


def test_a_correlated_bystander_dilutes_the_single_feature_importance_but_not_the_cluster_importance():
    X, y, groups = bystander_problem(0.95)
    res = lambda_averaged_importance(
        X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False, n_repeats=5, random_state=1
    )
    assert res.cluster["x0"] == res.cluster["xc"]
    assert res.cluster.nunique() == 4 and res.cluster.tolist() == [0, 1, 2, 3, 0]
    assert res.cluster_importance["x0"] == res.cluster_importance["xc"]
    assert res.permutation["x0"] < 0.5 * res.permutation["x1"]
    assert res.cluster_importance["x0"] > 1.5 * res.permutation["x0"]
    assert 0.7 * res.permutation["x1"] < res.cluster_importance["x0"] < 1.1 * res.permutation["x1"]
    assert res.cluster_importance["x0"] > res.permutation["x0"] + res.permutation["xc"]
    for name in ("x1", "n2", "n3"):
        assert res.cluster_importance[name] == res.permutation[name]


def test_exact_copies_share_the_single_feature_importance_and_keep_the_cluster_importance():
    rng = np.random.default_rng(5)
    base, other = rng.standard_normal(60), rng.standard_normal(60)
    X = pd.DataFrame({"a": base, "a_copy": base.copy(), "b": other, "c": rng.standard_normal(60)})
    y = pd.Series(2.0 * base + 2.0 * other + 0.5 * rng.standard_normal(60))
    res = lambda_averaged_importance(
        X, y, None, lambdas=[0.01], learners=("ridge",), compute_diagnostics=False, n_repeats=40, random_state=2
    )
    assert res.cluster.tolist() == [0, 0, 1, 2]
    joint = res.cluster_importance["a"]
    assert joint == res.cluster_importance["a_copy"]
    assert 3.0 < joint / res.permutation["a"] < 5.5
    assert 3.0 < joint / res.permutation["a_copy"] < 5.5
    assert joint == pytest.approx(res.permutation["b"], rel=0.3)


def test_without_clustering_every_feature_is_its_own_cluster():
    X, y, groups = bystander_problem(0.95)
    res = lambda_averaged_importance(
        X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False, cluster_threshold=None
    )
    assert res.cluster.tolist() == [0, 1, 2, 3, 4] and res.cluster.name == "cluster"
    npt.assert_array_equal(res.cluster_importance.to_numpy(), res.permutation.to_numpy())
    assert res.cluster_importance.name == "cluster_importance"
    default = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    npt.assert_array_equal(default.permutation.to_numpy(), res.permutation.to_numpy())


def test_cluster_threshold_defaults_to_point_eight_and_is_validated():
    assert inspect.signature(lambda_averaged_importance).parameters["cluster_threshold"].default == 0.8
    X, y, groups = bystander_problem(0.95)
    pair = with_correlations(np.array([[1.0, 0.7], [0.7, 1.0]]), n=60)
    X["x0"], X["xc"] = pair[:, 0], pair[:, 1]
    kwargs = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    assert lambda_averaged_importance(X, y, groups, **kwargs).cluster["xc"] != 0
    assert lambda_averaged_importance(X, y, groups, cluster_threshold=0.75, **kwargs).cluster["xc"] != 0
    loose = lambda_averaged_importance(X, y, groups, cluster_threshold=0.65, **kwargs)
    assert loose.cluster["xc"] == loose.cluster["x0"]
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="cluster_threshold"):
            lambda_averaged_importance(X, y, groups, cluster_threshold=bad, **kwargs)


def test_cluster_columns_appear_in_the_frames_of_the_result():
    X, y, groups = bystander_problem(0.95)
    res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    frame = res.to_frame()
    assert list(frame.columns)[-2:] == ["cluster", "cluster_importance"]
    npt.assert_array_equal(frame["cluster"].to_numpy(), res.cluster.to_numpy())
    summary = res.summary_frame().set_index("feature")
    assert list(summary.columns)[-2:] == ["cluster", "cluster_importance"]
    npt.assert_array_equal(summary.loc[res.feature_names, "cluster_importance"], res.cluster_importance.to_numpy())
    assert res.summary_frame()["cluster"].dtype.kind == "i"


# ----------------------------------------------------------------------------
# Results and tables
# ----------------------------------------------------------------------------
def test_weights_method(toy40):
    X, y, groups, res = toy40
    d = len(res.feature_names)
    base = res.permutation_normalised.to_numpy()
    plain = res.weights()
    assert isinstance(plain, np.ndarray) and plain.shape == (d,)
    npt.assert_allclose(plain, base)
    npt.assert_allclose(res.weights(1.0), np.full(d, 1.0 / d))
    mixed = res.weights(0.3)
    npt.assert_allclose(mixed, 0.7 * base + 0.3 / d)
    for w in (plain, mixed, res.weights(0.05)):
        assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            res.weights(bad)
    table = res.to_frame()
    assert list(table.index) == res.feature_names
    assert list(table.columns) == [
        "permutation", "permutation_normalised", "coefficient_path", "selection_frequency", "cluster",
        "cluster_importance",
    ]


SUMMARY_COLUMNS = [
    "feature", "permutation", "permutation_normalised", "coefficient_path", "selection_frequency", "cluster",
    "cluster_importance",
]


def hand_built(names, permutation, coefficient=None, selection=None):
    """Importance result whose per-feature series hold the given numbers."""
    permutation = np.asarray(permutation, dtype=float)
    zeros = np.zeros(len(names))
    values = {
        "permutation": permutation,
        "permutation_normalised": importance._normalise(permutation),
        "coefficient_path": zeros if coefficient is None else np.asarray(coefficient, dtype=float),
        "selection_frequency": zeros if selection is None else np.asarray(selection, dtype=float),
    }
    return ImportanceResult(
        lambdas=np.array([1.0]), lambda_weights=np.array([1.0]), cv_mse_by_lambda=np.array([1.0]),
        feature_names=list(names),
        **{key: pd.Series(value, index=list(names), name=key) for key, value in values.items()},
    )


def test_a_result_built_without_clusters_has_one_cluster_per_feature():
    res = hand_built(["a", "b", "c"], [3.0, -1.0, 2.0])
    assert res.cluster.tolist() == [0, 1, 2] and list(res.cluster.index) == ["a", "b", "c"]
    npt.assert_array_equal(res.cluster_importance.to_numpy(), [3.0, -1.0, 2.0])


def test_summary_frame_is_a_tidy_table_ordered_by_normalised_importance(toy40):
    X, y, groups, res = toy40
    frame = res.summary_frame()
    assert list(frame.columns) == SUMMARY_COLUMNS
    assert frame.index.equals(pd.RangeIndex(len(res.feature_names)))
    assert sorted(frame["feature"]) == sorted(res.feature_names)
    assert frame["permutation_normalised"].is_monotonic_decreasing
    assert set(frame["feature"].iloc[:2]) == {"x0", "x1"}
    assert frame["feature"].iloc[0] == res.permutation_normalised.idxmax()
    assert all(str(dtype) == "float64" for dtype in frame.drop(columns=["feature", "cluster"]).dtypes)
    pd.testing.assert_frame_equal(frame.set_index("feature").loc[res.feature_names], res.to_frame(), check_names=False)
    before = res.to_frame()
    frame.iloc[:, 1:] = -1.0
    pd.testing.assert_frame_equal(res.to_frame(), before)


@pytest.mark.parametrize(
    "permutation, expected",
    [
        ([0.0, 6.0, 2.0, -1.0], ["b", "c", "a", "d"]),
        ([-3.0, -1.0, -2.0, -4.0], ["b", "c", "a", "d"]),
        ([1.0, 1.0, 1.0, 1.0], ["a", "b", "c", "d"]),
        ([2.0, 0.0, 2.0, 0.0], ["a", "c", "b", "d"]),
    ],
    ids=["distinct", "all negative", "all equal", "equal pairs"],
)
def test_summary_frame_breaks_ties_by_the_permutation_importance_and_then_by_position(permutation, expected):
    coefficient, selection = [0.4, 1.5, 0.9, 0.1], [0.1, 0.9, 0.5, 0.2]
    res = hand_built(["a", "b", "c", "d"], permutation, coefficient, selection)
    frame = res.summary_frame()
    assert list(frame["feature"]) == expected
    position = [res.feature_names.index(name) for name in expected]
    npt.assert_array_equal(frame["permutation"], res.permutation.to_numpy()[position])
    npt.assert_array_equal(frame["permutation_normalised"], res.permutation_normalised.to_numpy()[position])
    npt.assert_array_equal(frame["coefficient_path"], np.array(coefficient)[position])
    npt.assert_array_equal(frame["selection_frequency"], np.array(selection)[position])
    npt.assert_array_equal(frame["cluster"], np.array(position))
    npt.assert_array_equal(frame["cluster_importance"], res.permutation.to_numpy()[position])
    assert frame.index.equals(pd.RangeIndex(4))


def test_normalisation_clips_negatives_and_falls_back_to_uniform():
    npt.assert_allclose(importance._normalise(np.array([-1.0, 2.0, 3.0])), [0.0, 0.4, 0.6])
    npt.assert_allclose(importance._normalise(np.array([-1.0, -2.0, -0.5, -4.0])), np.full(4, 0.25))
    npt.assert_allclose(importance._normalise(np.zeros(3)), np.full(3, 1.0 / 3.0))


# ----------------------------------------------------------------------------
# Importance without signal
# ----------------------------------------------------------------------------
def test_importance_at_the_noise_level_of_the_target_is_no_signal():
    uniform = np.full(5, 0.2)
    noise = np.random.default_rng(0).standard_normal(5) * 1e-17
    for raw in (np.zeros(5), -np.ones(5), noise, np.array([1e-12, -3.0, 0.0, 2e-12, 0.0]), np.full(5, np.nan)):
        vector, silent = importance._normalise_importance(raw, 1.0)
        assert silent is True
        npt.assert_array_equal(vector, uniform)
    vector, silent = importance._normalise_importance(np.array([-1.0, 2.0, 3.0, 0.0, np.nan]), 1.0)
    assert silent is False
    npt.assert_allclose(vector, [0.0, 0.4, 0.6, 0.0, 0.0])
    assert vector.dtype == np.float64 and isinstance(silent, bool)


def test_the_noise_threshold_is_one_billionth_of_the_variance_of_the_target():
    for variance in (1.0, 7.5, 1e-8):
        threshold = 1e-9 * variance
        at = np.array([threshold / 2, threshold / 2, 0.0])
        just_above = at * 1.001
        assert importance._normalise_importance(at, variance)[1] is True
        vector, silent = importance._normalise_importance(just_above, variance)
        assert silent is False
        npt.assert_allclose(vector, [0.5, 0.5, 0.0])


@pytest.mark.parametrize("scale", [1e6, 1e-6])
def test_the_verdict_on_the_noise_level_does_not_depend_on_the_units_of_the_target(scale):
    rng = np.random.default_rng(3)
    for level, expected in ((1e-17, True), (1e-3, False)):
        raw = np.abs(rng.standard_normal(6)) * level
        base = importance._normalise_importance(raw, 1.0)
        scaled = importance._normalise_importance(raw * scale**2, scale**2)
        assert base[1] is scaled[1] is expected
        npt.assert_allclose(base[0], scaled[0], rtol=1e-12)


REAL_CV_CORE = importance._cv_core


def noisy_cv_core(monkeypatch, level):
    """Replace the permutation importance of every fold by ``level`` times the variance of the target as noise."""

    def wrapper(X, y, *args, **kwargs):
        out = REAL_CV_CORE(X, y, *args, **kwargs)
        if out.perm is not None:
            noise = np.random.default_rng(0).standard_normal(out.perm.shape)
            out.perm = level * float(np.var(y, ddof=1)) * noise
        return out

    monkeypatch.setattr(importance, "_cv_core", wrapper)


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e-6])
def test_rounding_noise_in_the_importance_gives_the_uniform_vector_and_the_flag(monkeypatch, scale):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    noisy_cv_core(monkeypatch, 1e-17)
    res = lambda_averaged_importance(X, y * scale, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    assert res.no_signal is True and res.diagnostics["no_signal"] is True
    npt.assert_array_equal(res.permutation_normalised.to_numpy(), np.full(8, 0.125))
    npt.assert_allclose(res.weights(), np.full(8, 0.125))
    noisy_cv_core(monkeypatch, 1e-3)
    res = lambda_averaged_importance(X, y * scale, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    assert res.no_signal is False and res.diagnostics["no_signal"] is False
    assert not np.allclose(res.permutation_normalised, 0.125)
    npt.assert_allclose(res.permutation_normalised.sum(), 1.0)


def no_signal_problem(n=30, d=8, seed=1000):
    """Features and an unrelated target for a stack whose only penalty sets every coefficient to zero."""
    X, y, groups = noise_problem(n=n, d=d, seed=seed)
    return X, y, groups, dict(learners=("elastic_net",), lambdas=[10.0], light=True)


def test_an_unrelated_target_does_not_by_itself_set_the_flag_because_the_stack_still_fits_noise(quick_halves):
    for seed in (1000, 1001):
        X, y, groups = noise_problem(seed=seed)
        res = lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
        assert res.no_signal is False and res.diagnostics["no_signal"] is False
        assert res.diagnostics["usable"] is False and res.diagnostics["reasons"]
        assert not np.allclose(res.permutation_normalised, 0.125)


def test_a_stack_that_cannot_use_any_feature_gives_no_signal():
    X, y, groups, options = no_signal_problem()
    res = lambda_averaged_importance(X, y, groups, compute_diagnostics=False, **options)
    assert res.no_signal is True and res.diagnostics["no_signal"] is True
    npt.assert_array_equal(res.permutation.to_numpy(), np.zeros(8))
    npt.assert_array_equal(res.permutation_normalised.to_numpy(), np.full(8, 0.125))
    assert (res.selection_frequency == 0.0).all()
    assert list(res.summary_frame()["feature"]) == list(X.columns)
    full = lambda_averaged_importance(X, y, groups, n_perm=9, **options)
    assert full.no_signal is True and full.diagnostics["usable"] is False
    assert full.diagnostics["no_signal"] is True


@pytest.mark.parametrize("scale", [1e6, 1e-6])
def test_rescaling_the_target_does_not_change_the_verdict_of_a_problem_without_signal(scale):
    X, y, groups, options = no_signal_problem(seed=1001)
    base = lambda_averaged_importance(X, y, groups, compute_diagnostics=False, **options)
    scaled = lambda_averaged_importance(X, y * scale, groups, compute_diagnostics=False, **options)
    assert base.no_signal is scaled.no_signal is True
    npt.assert_array_equal(base.permutation_normalised.to_numpy(), scaled.permutation_normalised.to_numpy())
    band = bootstrap_importance(X, y * scale, groups, n_boot=4, random_state=2, **options)
    assert (band["n_resamples_informative"] == 0).all() and (band["n_resamples"] == 4).all()


@pytest.mark.parametrize("scale", [1e6, 1e-6])
def test_rescaling_the_target_does_not_change_the_verdict_of_a_problem_with_signal(quick_halves, scale):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    options = dict(learners=LINEAR, light=True, n_perm=9, random_state=1)
    base = lambda_averaged_importance(X, y, groups, **options)
    scaled = lambda_averaged_importance(X, y * scale, groups, **options)
    assert base.no_signal is scaled.no_signal is False
    assert base.diagnostics["usable"] is scaled.diagnostics["usable"]
    assert base.diagnostics["perm_p_value"] == scaled.diagnostics["perm_p_value"]
    npt.assert_allclose(scaled.permutation_normalised, base.permutation_normalised, atol=1e-6)
    npt.assert_allclose(scaled.diagnostics["split_half_correlation"], base.diagnostics["split_half_correlation"], atol=1e-6)
    band = bootstrap_importance(X, y, groups, n_boot=4, random_state=2, learners=LINEAR, light=True)
    band_scaled = bootstrap_importance(X, y * scale, groups, n_boot=4, random_state=2, learners=LINEAR, light=True)
    npt.assert_array_equal(band_scaled[["n_resamples_informative", "n_resamples"]], band[["n_resamples_informative", "n_resamples"]])
    npt.assert_allclose(band_scaled[["median", "p10", "p90"]], band[["median", "p10", "p90"]], atol=1e-6)


def test_the_point_estimate_the_resamples_and_the_halves_use_the_same_rule(monkeypatch, quick_halves):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    calls = []
    original = importance._normalise_importance

    def counted(values, variance):
        calls.append(variance)
        return original(values, variance)

    monkeypatch.setattr(importance, "_normalise_importance", counted)
    lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9, compute_diagnostics=False)
    assert len(calls) == 1 and calls[0] == pytest.approx(float(np.var(y, ddof=1)))
    calls.clear()
    bootstrap_importance(X, y, groups, n_boot=3, learners=LINEAR, light=True)
    assert len(calls) == 3
    calls.clear()
    lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, n_perm=9)
    assert len(calls) == 1 + 2 * 3


def test_a_result_built_with_the_flag_orders_its_summary_by_position():
    names = ["a", "b", "c", "d"]
    noise = np.array([1e-18, 3e-18, -2e-18, 2e-18])
    uniform = np.full(4, 0.25)
    values = {
        "permutation": noise, "permutation_normalised": uniform, "coefficient_path": np.zeros(4),
        "selection_frequency": np.zeros(4),
    }
    res = ImportanceResult(
        lambdas=np.array([1.0]), lambda_weights=np.array([1.0]), cv_mse_by_lambda=np.array([1.0]),
        feature_names=names, no_signal=True,
        **{key: pd.Series(value, index=names, name=key) for key, value in values.items()},
    )
    assert list(res.summary_frame()["feature"]) == names
    res.no_signal = False
    assert list(res.summary_frame()["feature"]) == ["b", "d", "a", "c"]
    assert ImportanceResult.__dataclass_fields__["no_signal"].default is False


# ----------------------------------------------------------------------------
# Split-half correlation
# ----------------------------------------------------------------------------
def test_constant_means_constant_up_to_rounding_noise_relative_to_the_whole():
    rng = np.random.default_rng(0)
    assert importance._is_constant(np.full(6, 3.0)) and importance._is_constant(np.zeros(4))
    assert importance._is_constant(7.0 * (1.0 + 1e-15 * rng.standard_normal(10)))
    assert not importance._is_constant(7.0 * (1.0 + 1e-6 * rng.standard_normal(10)))
    part = 3e-14 * rng.standard_normal(8)
    assert not importance._is_constant(part)
    assert importance._is_constant(part, scale=1.0)
    assert not importance._is_constant(np.array([0.0, 1e-6]), scale=1.0)
    X = np.column_stack([np.full(5, 2.0), np.arange(5.0), np.zeros(5), 5.0 + 1e-16 * rng.standard_normal(5)])
    npt.assert_array_equal(importance._is_constant(X, axis=0), [True, False, True, True])


def test_a_target_that_is_constant_up_to_rounding_noise_is_rejected():
    X, _, groups = make_toy_problem(n=24, random_state=1)
    y = 5.0 * (1.0 + 1e-15 * np.random.default_rng(0).standard_normal(24))
    with pytest.raises(ValueError, match="constant"):
        lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True)
    options = dict(lambdas=[0.5], learners=LINEAR, light=True)
    with pytest.raises(ValueError, match="y is constant"):
        lambda_averaged_importance(X, y, groups, compute_diagnostics=False, **options)
    with pytest.raises(ValueError, match="y is constant"):
        bootstrap_importance(X, y, groups, n_boot=2, **options)
    with pytest.raises(ValueError, match="y is constant"):
        verdict_stability(X, y, groups, seeds=(0,), lambdas=[0.5], learners=LINEAR)
    with pytest.raises(ValueError, match="constant"):
        lambda_grid(X, y)


def test_pearson_helper():
    assert importance._pearson(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
    assert importance._pearson(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert np.isnan(importance._pearson(np.zeros(4), np.arange(4.0)))
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 4, size=12).astype(float), rng.standard_normal(12)
    assert importance._pearson(a, b) == pytest.approx(pearsonr(a, b)[0])
    assert importance._pearson(a, b) != pytest.approx(spearmanr(a, b)[0], abs=1e-3)


def stub_estimates(monkeypatch, vectors, silent=()):
    """Replace the estimate of each half by the given normalised importance vectors, in call order.

    The estimates whose position in call order is in ``silent`` carry no signal.
    """
    queue = iter(vectors)
    calls = []

    def stub(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(perm_norm=np.asarray(next(queue), dtype=float), no_signal=len(calls) - 1 in silent)

    monkeypatch.setattr(importance, "_estimate", stub)
    return calls


def test_split_half_correlation_is_the_median_pearson_correlation_of_the_normalised_vectors(monkeypatch):
    X, y, groups = make_toy_problem(n=16, d=4, random_state=3)
    first, second = np.array([0.7, 0.2, 0.1, 0.0]), np.array([0.4, 0.5, 0.1, 0.0])
    pearson = float(np.corrcoef(first, second)[0, 1])
    assert pearson == pytest.approx(0.6305, abs=1e-4) and spearmanr(first, second)[0] == pytest.approx(0.8)
    constant = np.full(4, 0.25)
    vectors = [first, first, first, second, constant, constant]
    stub_estimates(monkeypatch, vectors)
    value = importance._split_half_correlation(
        X.to_numpy(), y.to_numpy(), np.arange(16), LEARNER_NAMES, 5, 3, "cv", 1.0, 0, 0
    )
    assert value == pytest.approx(pearson)
    stub_estimates(monkeypatch, [first, second] * 2)
    assert importance._split_half_correlation(
        X.to_numpy(), y.to_numpy(), np.arange(16), LEARNER_NAMES, 5, 2, "cv", 1.0, 0, 0
    ) == pytest.approx(pearson)
    stub_estimates(monkeypatch, [first, first, first, -first + 0.3])
    assert importance._split_half_correlation(
        X.to_numpy(), y.to_numpy(), np.arange(16), LEARNER_NAMES, 5, 2, "cv", 1.0, 0, 0
    ) == pytest.approx(0.0)


def test_a_half_without_signal_counts_as_zero_correlation(monkeypatch):
    X, y, groups = make_toy_problem(n=16, d=4, random_state=3)
    first, second = np.array([0.7, 0.2, 0.1, 0.0]), np.array([0.4, 0.5, 0.1, 0.0])
    pearson = float(np.corrcoef(first, second)[0, 1])
    assert pearson == pytest.approx(0.6305, abs=1e-4)
    arguments = (X.to_numpy(), y.to_numpy(), np.arange(16), LEARNER_NAMES, 5, 3, "cv", 1.0, 0, 0)
    for silent, expected in (
        ((), pearson), ({0}, pearson), ({3}, pearson), ({2, 3}, pearson), ({1, 2}, 0.0), ({0, 2, 4}, 0.0),
        ({0, 1, 2, 3, 4, 5}, 0.0),
    ):
        stub_estimates(monkeypatch, [first, second] * 3, silent=silent)
        assert importance._split_half_correlation(*arguments) == pytest.approx(expected), silent


def test_a_half_whose_target_is_rounding_noise_around_zero_is_skipped_like_a_constant_half(monkeypatch):
    rng = np.random.default_rng(4)
    n = 24
    X = rng.standard_normal((n, 4))
    exact = np.r_[np.ones(3), np.zeros(n - 3)]
    noisy = exact + 3e-14 * rng.standard_normal(n) * (exact == 0.0)
    codes = np.arange(n)
    seen = []
    for y in (exact, noisy):
        calls = stub_estimates(monkeypatch, [[0.5, 0.5]] * 120)
        importance._split_half_correlation(X, y, codes, LINEAR, 5, 60, "cv", 1.0, 3, 0)
        seen.append([tuple(call[2]) for call in calls])
    assert 0 < len(seen[0]) < 120
    assert seen[0] == seen[1]


def test_split_half_halves_are_formed_by_group_and_cover_all_groups(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=4)
    codes = importance._group_codes(groups, 30)
    n_groups = len(set(codes))
    calls = stub_estimates(monkeypatch, [[0.5, 0.5]] * 12)
    importance._split_half_correlation(X.to_numpy(), y.to_numpy(), codes, LINEAR, 5, 6, "cv", 1.0, 0, 0)
    assert len(calls) == 12
    for k in range(0, 12, 2):
        left, right = set(calls[k][2]), set(calls[k + 1][2])
        assert left.isdisjoint(right) and left | right == set(codes)
        assert abs(len(left) - len(right)) <= 1 and len(left) + len(right) == n_groups
    assert len({frozenset(calls[k][2]) for k in range(0, 12, 2)}) > 1


def test_split_half_needs_at_least_eight_groups():
    X, y, groups = make_toy_problem(n=7, random_state=15)
    codes = importance._group_codes(np.arange(7), 7)
    calls = []
    value = importance._split_half_correlation(
        X.to_numpy(), y.to_numpy(), codes, LEARNER_NAMES, 5, 3, "cv", 1.0, 0, 0, guard=lambda: calls.append(1)
    )
    assert np.isnan(value) and calls == []


def test_split_half_correlation_is_high_for_a_clear_signal_and_low_for_noise():
    X, y, groups = linear_problem(n=60)
    codes = importance._group_codes(groups, 60)
    signal = importance._split_half_correlation(
        X.to_numpy(), y.to_numpy(), codes, LINEAR, 5, 4, "cv", 1.0, 0, 0
    )
    assert signal > 0.9
    X, y, groups = noise_problem(n=40)
    noise = importance._split_half_correlation(X.to_numpy(), y.to_numpy(), np.arange(40), LINEAR, 5, 4, "cv", 1.0, 0, 0)
    assert noise < 0.5


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------
def spy_public(monkeypatch):
    """Record the groups, target and options of every call of ``lambda_averaged_importance``."""
    calls = []
    original = importance.lambda_averaged_importance

    def spy(X, y, groups, **kwargs):
        calls.append((np.asarray(groups).copy(), np.asarray(y).copy(), kwargs))
        return original(X, y, groups, **kwargs)

    monkeypatch.setattr(importance, "lambda_averaged_importance", spy)
    return calls


BAND_COLUMNS = ["median", "p10", "p90", "n_resamples_informative", "n_resamples"]


def test_bootstrap_importance_band():
    X, y, groups = make_toy_problem(n=40, d=6, random_state=0)
    band = bootstrap_importance(X, y, groups, n_boot=5, random_state=2, light=True)
    again = bootstrap_importance(X, y, groups, n_boot=5, random_state=2, learners=LINEAR, light=True)
    other = bootstrap_importance(X, y, groups, n_boot=5, random_state=2, learners=LINEAR, light=True)
    pd.testing.assert_frame_equal(again, other)
    assert not again.equals(bootstrap_importance(X, y, groups, n_boot=5, random_state=3, learners=LINEAR, light=True))
    assert list(band.columns) == BAND_COLUMNS and list(band.index) == list(X.columns)
    assert ((band["p10"] <= band["median"] + 1e-12) & (band["median"] <= band["p90"] + 1e-12)).all()
    assert ((band[["median", "p10", "p90"]] >= 0) & (band[["median", "p10", "p90"]] <= 1)).all().all()
    assert (band["n_resamples"] == 5).all() and (band["n_resamples_informative"] == 5).all()
    assert set(band["median"].nlargest(2).index) == {"x0", "x1"}
    assert band.loc[["x0", "x1"], "median"].sum() > 0.7


def stub_resamples(monkeypatch, results):
    """Replace the importance of each resample by a given vector and flag, in call order."""
    queue = iter(results)

    def stub(X, y, groups, **kwargs):
        vector, silent = next(queue)
        return SimpleNamespace(permutation_normalised=pd.Series(np.asarray(vector, dtype=float)), no_signal=silent)

    monkeypatch.setattr(importance, "lambda_averaged_importance", stub)


def test_resamples_without_signal_are_left_out_of_the_band_and_counted(monkeypatch):
    X, y, groups = make_toy_problem(n=24, d=4, random_state=1)
    rng = np.random.default_rng(0)
    informative = [(v / v.sum(), False) for v in rng.random((6, 4)) ** 3]
    uniform = (np.full(4, 0.25), True)
    stub_resamples(monkeypatch, informative)
    clean = bootstrap_importance(X, y, groups, n_boot=6, random_state=1)
    mixed_order = informative[:2] + [uniform] + informative[2:5] + [uniform, uniform] + informative[5:] + [uniform]
    stub_resamples(monkeypatch, mixed_order)
    mixed = bootstrap_importance(X, y, groups, n_boot=10, random_state=1)
    bands = ["median", "p10", "p90"]
    pd.testing.assert_frame_equal(mixed[bands], clean[bands], check_exact=True)
    vectors = np.array([vector for vector, _ in informative])
    npt.assert_array_equal(mixed["median"], np.median(vectors, axis=0))
    npt.assert_array_equal(mixed["p10"], np.percentile(vectors, 10, axis=0))
    npt.assert_array_equal(mixed["p90"], np.percentile(vectors, 90, axis=0))
    assert (clean["n_resamples_informative"] == 6).all() and (clean["n_resamples"] == 6).all()
    assert (mixed["n_resamples_informative"] == 6).all() and (mixed["n_resamples"] == 10).all()
    assert mixed["n_resamples"].dtype == np.int64 and mixed["n_resamples_informative"].dtype == np.int64
    with_uniform = np.vstack([vectors] + [uniform[0]] * 4)
    assert not np.allclose(np.median(with_uniform, axis=0), mixed["median"])
    assert list(mixed.columns) == BAND_COLUMNS and mixed.index.name == "feature"


def test_the_band_is_nan_when_no_resample_carries_a_signal(monkeypatch):
    X, y, groups = make_toy_problem(n=24, d=4, random_state=1)
    stub_resamples(monkeypatch, [(np.full(4, 0.25), True)] * 5)
    band = bootstrap_importance(X, y, groups, n_boot=5, random_state=1)
    assert band[["median", "p10", "p90"]].isna().all().all()
    assert (band["n_resamples_informative"] == 0).all() and (band["n_resamples"] == 5).all()
    assert list(band.index) == list(X.columns) and band.shape == (4, 5)


def test_a_problem_without_signal_has_a_band_of_nan_and_the_counts():
    X, y, groups, options = no_signal_problem()
    band = bootstrap_importance(X, y, groups, n_boot=5, random_state=3, **options)
    assert band[["median", "p10", "p90"]].isna().all().all()
    assert (band["n_resamples_informative"] == 0).all() and (band["n_resamples"] == 5).all()


def test_bootstrap_importance_options_and_validation():
    X, y, groups = make_toy_problem(n=24, random_state=1)
    band = bootstrap_importance(X, y, groups, n_boot=3, learners=LINEAR, n_repeats=2, weighting="uniform", n_perm=5)
    assert band.shape == (8, 5)
    with pytest.raises(TypeError):
        bootstrap_importance(X, y, groups, n_boot=2, bogus=1)
    with pytest.raises(ValueError):
        bootstrap_importance(X, y, groups, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_importance(X, y, np.repeat(["a", "b", "c"], 8), n_boot=2)
    with pytest.raises(ValueError):
        bootstrap_importance(X, y, groups, n_boot=2, cv_repeats=0)
    with pytest.raises(ValueError):
        bootstrap_importance(X, y, groups, n_boot=2, weighting="softmax")
    with pytest.raises(ValueError):
        bootstrap_importance(X, y, groups, n_boot=2, random_state=-1)
    with pytest.raises(ValueError):
        bootstrap_importance(X, y, groups, n_boot=2, learners=["lasso"])


def test_group_bootstrap_keeps_repeated_groups_in_one_fold(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=2)
    recorded = record_splits(monkeypatch)
    bootstrap_importance(X, y, groups, n_boot=4, learners=LINEAR, n_repeats=2)
    assert recorded
    for codes, _, splits in recorded:
        assert_group_disjoint(codes, splits)
    assert any(len(codes) != len(np.unique(codes)) for codes, _, _ in recorded)


def test_bootstrap_resamples_are_estimated_without_diagnostics_and_without_clusters(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=2)
    calls = spy_public(monkeypatch)
    heavy = []
    for name in ("_label_permutation_p", "_split_half_correlation"):
        monkeypatch.setattr(importance, name, lambda *args, _name=name, **kwargs: heavy.append(_name))
    band = bootstrap_importance(X, y, groups, n_boot=4, learners=LINEAR, light=True, random_state=3)
    assert band.shape == (8, 5) and len(calls) == 4 and heavy == []
    for _, _, kwargs in calls:
        assert kwargs["compute_diagnostics"] is False and kwargs["cluster_threshold"] is None
        assert kwargs["cv_repeats"] == 1 and kwargs["learners"] == LINEAR and kwargs["light"] is True


def test_bootstrap_redraws_resamples_with_too_few_distinct_groups(monkeypatch):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    four = np.repeat(np.arange(4), 6)
    calls = spy_public(monkeypatch)
    band = bootstrap_importance(X, y, four, n_boot=5, learners=LINEAR, light=True, random_state=1)
    assert band.shape == (8, 5) and len(calls) == 5
    for codes, _, _ in calls:
        assert len(set(codes)) == 4


@pytest.mark.parametrize("seed", range(6))
def test_bootstrap_redraws_resamples_that_need_more_folds_than_the_data(monkeypatch, seed):
    X, y, _ = make_toy_problem(n=30, d=5, random_state=5)
    six = np.repeat(np.arange(6), 5)
    calls = spy_public(monkeypatch)
    band = bootstrap_importance(X, y, six, n_boot=10, random_state=seed, learners=LINEAR, n_splits=3, n_repeats=2)
    assert band.shape == (5, 5) and len(calls) == 10
    assert min(len(set(codes)) for codes, _, _ in calls) >= 5
    assert np.isfinite(band.to_numpy()).all()


def test_bootstrap_redraws_resamples_whose_target_is_constant(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((24, 3)), columns=list("abc"))
    y = np.zeros(24)
    y[20:] = [1.0, 2.0, 3.0, 4.0]
    calls = spy_public(monkeypatch)
    band = bootstrap_importance(
        X, y, np.repeat(np.arange(6), 4), n_boot=5, learners=LINEAR, light=True, n_splits=3, random_state=0
    )
    assert band.shape == (3, 5) and len(calls) == 5
    for codes, target, _ in calls:
        assert np.var(target) > 0.0 and 5 in set(codes)


@pytest.mark.parametrize("seed", [1, 3, 6])
def test_bootstrap_redraws_resamples_whose_target_is_rounding_noise_around_zero(monkeypatch, seed):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((24, 3)), columns=list("abc"))
    y = 3e-14 * rng.standard_normal(24)
    y[20:] = [1.0, 2.0, 3.0, 4.0]
    calls = spy_public(monkeypatch)
    verdicts = []
    original = importance._is_constant

    def spy(values, axis=None, scale=None):
        verdict = original(values, axis=axis, scale=scale)
        if scale is not None:
            verdicts.append(bool(verdict))
        return verdict

    monkeypatch.setattr(importance, "_is_constant", spy)
    band = bootstrap_importance(
        X, y, np.repeat(np.arange(6), 4), n_boot=5, learners=LINEAR, light=True, n_splits=3, random_state=seed
    )
    assert band.shape == (3, 5) and len(calls) == 5
    assert len(verdicts) > 5 and sum(verdicts) >= 1
    for codes, _, _ in calls:
        assert 5 in set(codes)


def test_bootstrap_raises_a_clear_error_when_valid_resamples_cannot_be_drawn(monkeypatch):
    X, y, _ = make_toy_problem(n=24, random_state=3)
    four = np.repeat(np.arange(4), 6)
    monkeypatch.setattr(importance, "_BOOTSTRAP_ATTEMPTS_PER_RESAMPLE", 1)
    with pytest.raises(ValueError, match=r"of 60 valid bootstrap resamples could be drawn in 60 attempts") as info:
        bootstrap_importance(X, y, four, n_boot=60, learners=LINEAR, light=True)
    assert "at least 4 distinct groups" in str(info.value)
    assert importance._BOOTSTRAP_ATTEMPTS_PER_RESAMPLE == 1


def test_bootstrap_needs_at_least_four_groups_in_the_data():
    X, y, _ = make_toy_problem(n=24, random_state=3)
    with pytest.raises(ValueError, match="at least 4 distinct groups"):
        bootstrap_importance(X, y, np.repeat(np.arange(3), 8), n_boot=2, learners=LINEAR)


# ----------------------------------------------------------------------------
# Stability of the verdict across seeds
# ----------------------------------------------------------------------------
STABILITY_COLUMNS = ["seed", "cv_r2_stack", "perm_p_value", "split_half_correlation", "n_cases", "usable"]


def test_verdict_stability_has_the_documented_signature():
    parameters = inspect.signature(verdict_stability).parameters
    assert list(parameters)[:6] == ["X", "y", "groups", "seeds", "light", "n_perm"]
    assert parameters["seeds"].default == (0, 1, 2, 3, 4)
    assert parameters["light"].default is True and parameters["n_perm"].default == 20
    assert parameters["guard"].default is None
    assert "verdict_stability" in importance.__all__


def test_verdict_stability_returns_one_row_per_seed_and_the_share_of_usable_seeds(quick_halves):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    frame, summary = verdict_stability(X, y, groups, seeds=(3, 1, 2), learners=LINEAR)
    assert list(frame.columns) == STABILITY_COLUMNS and frame.index.equals(pd.RangeIndex(3))
    assert frame["seed"].tolist() == [3, 1, 2] and (frame["n_cases"] == 30).all()
    assert [str(dtype) for dtype in frame.dtypes] == ["int64", "float64", "float64", "float64", "int64", "bool"]
    assert summary == {
        "share_usable": float(frame["usable"].mean()), "n_usable": int(frame["usable"].sum()), "n_seeds": 3
    }
    scaled = frame["perm_p_value"] * 21.0
    assert np.allclose(scaled, scaled.round(), atol=1e-9) and (frame["perm_p_value"] >= 1.0 / 21.0 - 1e-12).all()


def test_each_row_of_the_verdict_stability_is_the_diagnostics_of_the_importance_of_its_seed(quick_halves):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    frame, _ = verdict_stability(X, y, groups, seeds=(7, 0), learners=LINEAR)
    for row in frame.itertuples():
        for cluster_threshold in (None, 0.8):
            res = lambda_averaged_importance(
                X, y, groups, random_state=row.seed, light=True, n_perm=20, learners=LINEAR,
                cluster_threshold=cluster_threshold,
            )
            d = res.diagnostics
            assert (row.cv_r2_stack, row.perm_p_value, row.split_half_correlation, row.n_cases, row.usable) == (
                d["cv_r2_stack"], d["perm_p_value"], d["split_half_correlation"], d["n_cases"], d["usable"]
            )
    assert frame["cv_r2_stack"].nunique() == 2


def test_verdict_stability_is_deterministic_and_changes_nothing_else(quick_halves, restore_global_rng):
    X, y, groups = noise_problem(n=30)
    X_before, y_before = X.copy(), y.copy()
    np.random.seed(123)
    first, first_summary = verdict_stability(X, y, groups, seeds=range(4), learners=LINEAR)
    np.random.seed(456)
    second, second_summary = verdict_stability(X, y, groups, seeds=range(4), learners=LINEAR)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first_summary == second_summary
    pd.testing.assert_frame_equal(X, X_before, check_exact=True)
    pd.testing.assert_series_equal(y, y_before, check_exact=True)
    assert first["cv_r2_stack"].nunique() == 4


def test_the_share_of_usable_seeds_is_the_mean_of_the_verdicts(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    flags = iter([True, False, True, True, False])
    calls = []

    def stub(*args, **kwargs):
        calls.append(args)
        diagnostics = {
            "cv_r2_stack": 0.2, "perm_p_value": 0.05, "split_half_correlation": 0.6, "n_cases": 30,
            "usable": next(flags),
        }
        return SimpleNamespace(diagnostics=diagnostics)

    monkeypatch.setattr(importance, "_importance", stub)
    frame, summary = verdict_stability(X, y, groups)
    assert frame["seed"].tolist() == [0, 1, 2, 3, 4]
    assert frame["usable"].tolist() == [True, False, True, True, False]
    assert summary == {"share_usable": 0.6, "n_usable": 3, "n_seeds": 5}
    assert [call[8] for call in calls] == [0, 1, 2, 3, 4]
    for call in calls:
        assert call[11] is True and call[12] == 20 and call[13] is None
        assert call[14] is None and call[15] is True


def test_verdict_stability_passes_its_options_to_the_importance(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    calls = []

    def stub(*args, **kwargs):
        calls.append(args)
        diagnostics = {
            "cv_r2_stack": 0.0, "perm_p_value": 1.0, "split_half_correlation": 0.0, "n_cases": 30, "usable": False
        }
        return SimpleNamespace(diagnostics=diagnostics)

    monkeypatch.setattr(importance, "_importance", stub)
    guard = lambda: None  # noqa: E731
    frame, summary = verdict_stability(
        X, y, groups, seeds=[np.int64(5)], light=False, n_perm=33, guard=guard, n_splits=4, n_repeats=7,
        weighting="uniform", temperature=2.0, learners=["ridge"], cv_repeats=2, lambdas=[0.1, 0.2],
    )
    assert summary == {"share_usable": 0.0, "n_usable": 0, "n_seeds": 1} and frame["seed"].tolist() == [5]
    (args,) = calls
    assert args[4] == ("ridge",) and list(args[5]) == [0.1, 0.2] and args[6] == 4 and args[7] == 7
    assert args[8] == 5 and args[9] == "uniform" and args[10] == 2.0 and args[11] is False and args[12] == 33
    assert args[13] is guard and args[14] is None and args[15] is True and args[16] == 2


def test_a_single_integer_is_one_seed(quick_halves):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    for seed in (4, np.int64(4)):
        frame, summary = verdict_stability(X, y, groups, seeds=seed, learners=LINEAR)
        assert frame["seed"].tolist() == [4] and summary["n_seeds"] == 1


def test_more_than_twenty_permutations_are_not_used_in_the_light_configuration(monkeypatch, quick_halves):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    sizes = []
    original = importance._label_permutation_p

    def spy(X_, y_, codes, names, n_splits, n_perm, *args, **kwargs):
        sizes.append(n_perm)
        return original(X_, y_, codes, names, n_splits, n_perm, *args, **kwargs)

    monkeypatch.setattr(importance, "_label_permutation_p", spy)
    verdict_stability(X, y, groups, seeds=(0,), n_perm=30, learners=LINEAR)
    verdict_stability(X, y, groups, seeds=(0,), n_perm=12, learners=LINEAR)
    assert sizes == [20, 12]


@pytest.mark.parametrize(
    "seeds",
    [(), [], (0, 0), (1, 2, 1), (-1,), (2**32,), (1.5,), ("a",), (None,), (0, None), 2.5, object()],
    ids=repr,
)
def test_invalid_seeds_raise_a_value_error(seeds):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    with pytest.raises(ValueError):
        verdict_stability(X, y, groups, seeds=seeds, learners=LINEAR)


@pytest.mark.parametrize("seeds", [(1.5,), ("a",), "01", (-1,), (2**32,), (0, 2.5), (None,), 2.5, (), (3, 3)], ids=repr)
def test_the_error_for_an_invalid_seed_names_the_seeds_argument(seeds):
    X, y, groups = make_toy_problem(n=24, random_state=3)
    with pytest.raises(ValueError, match="seeds") as info:
        verdict_stability(X, y, groups, seeds=seeds, learners=LINEAR)
    assert "random_state" not in str(info.value)


def test_verdict_stability_validates_its_inputs_and_options():
    X, y, groups = make_toy_problem(n=24, random_state=3)
    with pytest.raises(ValueError, match="at least 9 permutations"):
        verdict_stability(X, y, groups, n_perm=8)
    with pytest.raises(TypeError, match="guard"):
        verdict_stability(X, y, groups, guard=3)
    with pytest.raises(ValueError, match="weighting"):
        verdict_stability(X, y, groups, weighting="softmax")
    with pytest.raises(ValueError, match="at least 4 distinct groups"):
        verdict_stability(X, y, np.repeat(np.arange(3), 8))
    with pytest.raises(ValueError, match="constant"):
        verdict_stability(X, np.ones(24), groups)
    with pytest.raises(ValueError, match="learners"):
        verdict_stability(X, y, groups, learners=["lasso"])
    with pytest.raises(TypeError):
        verdict_stability(X, y, groups, bogus=1)


@pytest.mark.usefixtures("quick_halves")
def test_verdict_stability_calls_the_guard_21_times_per_seed(staged):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    verdict_stability(X, y, groups, seeds=(0, 1), learners=LINEAR, guard=staged)
    assert staged.stages == (["folds"] * 18 + ["label"] * 2 + ["halves"]) * 2


@pytest.mark.usefixtures("restore_global_rng", "quick_halves")
def test_the_guard_does_not_change_the_verdict_stability():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    plain, plain_summary = verdict_stability(X, y, groups, seeds=(0, 1), learners=LINEAR)
    calls = []
    guarded, guarded_summary = verdict_stability(
        X, y, groups, seeds=(0, 1), learners=LINEAR, guard=consuming_guard(calls)
    )
    assert len(calls) == 42
    pd.testing.assert_frame_equal(guarded, plain, check_exact=True)
    assert guarded_summary == plain_summary


@pytest.mark.usefixtures("quick_halves")
@pytest.mark.parametrize("n", [1, 21, 22, 40])
def test_an_exception_raised_by_the_guard_ends_the_verdict_stability_at_once(n):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    calls = []
    with pytest.raises(StopRun):
        verdict_stability(X, y, groups, seeds=(0, 1), learners=LINEAR, guard=stop_at(n, calls))
    assert len(calls) == n


# ----------------------------------------------------------------------------
# Simulated problem
# ----------------------------------------------------------------------------
def test_toy_problem_shapes_and_reproducibility():
    X, y, groups = make_toy_problem()
    assert X.shape == (30, 8) and list(X.columns) == [f"x{j}" for j in range(8)]
    assert isinstance(y, pd.Series) and y.shape == (30,) and y.index.equals(X.index)
    assert len(groups) == 30 and 3 <= len(set(groups)) < 30
    X2, y2, groups2 = make_toy_problem()
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)
    npt.assert_array_equal(groups, groups2)
    X3, y3, _ = make_toy_problem(random_state=1)
    assert not np.allclose(X.to_numpy(), X3.to_numpy())
    X4, y4, g4 = make_toy_problem(n=12, d=6, informative=(2, 4), noise=0.0, random_state=3)
    assert X4.shape == (12, 6) and len(g4) == 12
    npt.assert_allclose(y4, 2 * X4["x2"] + 2 * X4["x4"] + X4["x2"] * X4["x4"])


def test_toy_problem_follows_its_documented_structure():
    X, y, groups = make_toy_problem(n=6000, d=8, noise=0.5, random_state=0)
    values = X.to_numpy()
    design = np.column_stack([np.ones(6000), values[:, 0], values[:, 1], values[:, 0] * values[:, 1]])
    coef, *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
    npt.assert_allclose(coef, [0.0, 2.0, 2.0, 1.0], atol=0.05)
    residual = y.to_numpy() - design @ coef
    assert residual.std() == pytest.approx(0.5, rel=0.05)
    corr = np.corrcoef(values, rowvar=False)
    assert corr[0, 2] == pytest.approx(0.5, abs=0.04) and corr[0, 3] == pytest.approx(0.3, abs=0.04)
    assert corr[0, 4] == pytest.approx(0.2, abs=0.04)
    assert max(abs(corr[0, 5]), abs(corr[0, 6]), abs(corr[0, 7]), abs(corr[1, 2]), abs(corr[1, 5])) < 0.05
    for j in range(2, 8):
        assert abs(np.corrcoef(residual, values[:, j])[0, 1]) < 0.05


def test_toy_problem_validation():
    invalid = (
        {"n": 3}, {"d": 1}, {"informative": (0, 8)}, {"informative": (1, 1)}, {"informative": ()}, {"noise": -1.0},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            make_toy_problem(**kwargs)


# ----------------------------------------------------------------------------
# Inputs and options
# ----------------------------------------------------------------------------
def test_arrays_lists_and_missing_groups_are_accepted():
    X, y, groups = make_toy_problem(n=20, random_state=16)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    res = lambda_averaged_importance(X.to_numpy(), list(y), list(groups), **options)
    assert res.feature_names == [f"x{j}" for j in range(8)]
    alone = lambda_averaged_importance(X, y, None, **options)
    assert alone.diagnostics["n_cases"] == 20


def test_invalid_inputs_raise_value_errors():
    X, y, groups = make_toy_problem(n=20, random_state=17)
    bad_X = X.copy()
    bad_X.iloc[2, 3] = np.nan
    bad_y = y.copy()
    bad_y.iloc[4] = np.inf
    cases = [
        dict(X=bad_X, y=y, groups=groups),
        dict(X=X, y=bad_y, groups=groups),
        dict(X=X, y=y.iloc[:19], groups=groups),
        dict(X=X, y=y, groups=groups[:19]),
        dict(X=X, y=pd.Series(np.ones(20)), groups=groups),
        dict(X=X, y=y, groups=np.repeat(["a", "b", "c"], [7, 7, 6])),
        dict(X=X, y=y, groups=np.array([None] + ["g"] * 19, dtype=object)),
        dict(X=X, y=y, groups=groups, n_splits=1),
        dict(X=X, y=y, groups=groups, n_repeats=0),
        dict(X=X, y=y, groups=groups, cv_repeats=0),
        dict(X=X, y=y, groups=groups, weighting="softmax"),
        dict(X=X, y=y, groups=groups, temperature=0.0),
        dict(X=X, y=y, groups=groups, n_perm=0),
        dict(X=X, y=y, groups=groups, n_perm=8),
        dict(X=X, y=y, groups=groups, cluster_threshold=1.0),
        dict(X=X, y=y, groups=groups, learners=["lasso"]),
        dict(X=X, y=y, groups=groups, lambdas=[0.1, -0.2]),
        dict(X=X, y=y, groups=groups, lambdas=[]),
        dict(X=X, y=y, groups=groups, random_state=-1),
        dict(X=X, y=y, groups=groups, random_state=2**32),
        dict(X=X, y=y, groups=groups, random_state=1.5),
        dict(X=X, y=y, groups=groups, random_state="seed"),
    ]
    for kwargs in cases:
        with pytest.raises(ValueError):
            lambda_averaged_importance(learners=kwargs.pop("learners", LINEAR), light=True, **kwargs)


def test_y_and_groups_are_aligned_to_the_index_of_x():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    labels = pd.Series(groups, index=X.index)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False, random_state=1)
    base = lambda_averaged_importance(X, y, labels, **options)
    order = np.random.default_rng(2).permutation(24)
    assert not X.index[order].equals(X.index)
    aligned = lambda_averaged_importance(X, y.iloc[order], labels.iloc[order], **options)
    assert_same_result(base, aligned)
    mixed = lambda_averaged_importance(X, y.iloc[order], groups, **options)
    assert_same_result(base, mixed)
    positional = lambda_averaged_importance(X.to_numpy(), y.iloc[order], labels.iloc[order], **options)
    assert not np.allclose(positional.permutation.to_numpy(), base.permutation.to_numpy())
    positional = lambda_averaged_importance(X.to_numpy(), y.iloc[order].to_numpy(), labels.iloc[order], **options)
    assert not np.allclose(positional.permutation.to_numpy(), base.permutation.to_numpy())


def test_an_index_that_differs_from_the_index_of_x_is_an_error():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    renamed = pd.Series(y.to_numpy(), index=[f"other_{i}" for i in range(24)])
    with pytest.raises(ValueError, match="index of y differs from the index of X"):
        lambda_averaged_importance(X, renamed, groups, **options)
    partly = pd.Series(y.to_numpy(), index=list(X.index[:23]) + ["extra"])
    with pytest.raises(ValueError, match="index of y differs from the index of X"):
        lambda_averaged_importance(X, partly, groups, **options)
    with pytest.raises(ValueError, match="index of groups differs from the index of X"):
        lambda_averaged_importance(X, y, pd.Series(groups, index=renamed.index), **options)
    with pytest.raises(ValueError, match="index of y differs from the index of X"):
        bootstrap_importance(X, renamed, groups, n_boot=2, learners=LINEAR, light=True)
    repeated = X.copy()
    repeated.index = ["a", "a"] + list(X.index[2:])
    series = pd.Series(y.to_numpy(), index=repeated.index)
    with pytest.raises(ValueError, match="repeated labels"):
        lambda_averaged_importance(repeated, series.iloc[::-1], groups, **options)
    res = lambda_averaged_importance(repeated, series, groups, **options)
    assert res.diagnostics["n_cases"] == 24


def test_a_length_mismatch_is_an_error():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    with pytest.raises(ValueError, match="y has 20 rows but X has 24"):
        lambda_averaged_importance(X, y.iloc[:20], groups, **options)
    with pytest.raises(ValueError, match="y"):
        lambda_averaged_importance(X, y.to_numpy()[:20], groups, **options)
    with pytest.raises(ValueError, match="groups"):
        lambda_averaged_importance(X, y, groups[:20], **options)
    with pytest.raises(ValueError, match="groups has 30 rows but X has 24"):
        lambda_averaged_importance(X, y, pd.Series(np.arange(30)), **options)
    with pytest.raises(ValueError, match="y"):
        lambda_averaged_importance(X.to_numpy(), np.ones((24, 2)), groups, **options)
    with pytest.raises(ValueError):
        lambda_grid(X, y.iloc[:20])


def test_missing_and_infinite_values_are_errors():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    for bad in (np.nan, np.inf, -np.inf):
        frame = X.copy()
        frame.iloc[3, 2] = bad
        with pytest.raises(ValueError, match="X contains missing or infinite values"):
            lambda_averaged_importance(frame, y, groups, **options)
        target = y.copy()
        target.iloc[5] = bad
        with pytest.raises(ValueError, match="y contains missing or infinite values"):
            lambda_averaged_importance(X, target, groups, **options)
        with pytest.raises(ValueError, match="y contains missing or infinite values"):
            lambda_grid(X, target)
        with pytest.raises(ValueError, match="y contains missing or infinite values"):
            StackedEnsemble(learners=LINEAR).fit(X, target, groups)
        with pytest.raises(ValueError, match="X contains missing or infinite values"):
            bootstrap_importance(frame, y, groups, n_boot=2, learners=LINEAR, light=True)


def test_values_whose_variance_overflows_are_errors():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    with pytest.raises(ValueError, match="y has values so large that its variance overflows"):
        lambda_averaged_importance(X, y * 1e160, groups, **options)
    with pytest.raises(ValueError, match="X has values so large that its variance overflows"):
        lambda_averaged_importance(X * 1e160, y, groups, **options)
    res = lambda_averaged_importance(X, y * 1e100, groups, **options)
    assert np.isfinite(res.permutation_normalised).all()


def test_values_whose_variance_underflows_are_errors():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    small_X = X.copy()
    small_X["x2"] = small_X["x2"] * 1e-200
    small_y = y * 1e-200
    denormal_y = y * 1e-160
    with pytest.raises(ValueError, match="y has values so small that its variance underflows"):
        lambda_averaged_importance(X, denormal_y, groups, **options)
    for call in (
        lambda: lambda_averaged_importance(X, small_y, groups, **options),
        lambda: lambda_averaged_importance(X, small_y.to_numpy(), groups, learners=None, light=True),
        lambda: lambda_grid(X, small_y),
        lambda: bootstrap_importance(X, small_y, groups, n_boot=2, learners=LINEAR),
        lambda: verdict_stability(X, small_y, groups, seeds=(0,), learners=LINEAR),
        lambda: StackedEnsemble().fit(X, small_y, groups),
    ):
        with pytest.raises(ValueError, match="y has values so small that its variance underflows"):
            call()
    for call in (
        lambda: lambda_averaged_importance(small_X, y, groups, **options),
        lambda: lambda_grid(small_X, y),
        lambda: StackedEnsemble().fit(small_X, y, groups),
    ):
        with pytest.raises(ValueError, match="X has values so small that its variance underflows"):
            call()
    base = lambda_averaged_importance(X, y, groups, **options)
    res = lambda_averaged_importance(X, y * 1e-100, groups, **options)
    npt.assert_allclose(res.permutation_normalised, base.permutation_normalised, atol=1e-8)


def test_a_column_or_target_that_is_constant_is_not_a_scale_error():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    flat = X.copy()
    flat["x2"] = 1e-200
    flat["x3"] = 0.0
    res = lambda_averaged_importance(flat, y, groups, learners=LINEAR, light=True, compute_diagnostics=False)
    assert np.isfinite(res.permutation_normalised).all()
    with pytest.raises(ValueError, match="y is constant"):
        lambda_averaged_importance(X, np.full(24, 1e-200), groups, learners=LINEAR, light=True)


def test_exact_duplicates_of_x_with_another_target_or_group_raise_a_warning():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    doubled_X = pd.concat([X, X], ignore_index=True)
    doubled_y = pd.concat([y, y + 1.0], ignore_index=True)
    twin_groups = np.tile(np.arange(24), 2)
    with pytest.warns(UserWarning, match=r"48 rows of X are exact duplicates of other rows"):
        lambda_averaged_importance(doubled_X, doubled_y, twin_groups, **options)
    with pytest.warns(UserWarning, match=r"48 rows of X are exact duplicates of other rows"):
        lambda_averaged_importance(doubled_X, doubled_y, None, **options)
    same_target = pd.concat([y, y], ignore_index=True)
    with pytest.warns(UserWarning, match=r"48 rows of X are exact duplicates of other rows"):
        lambda_averaged_importance(doubled_X, same_target, None, **options)
    with pytest.warns(UserWarning, match=r"exact duplicates"):
        bootstrap_importance(doubled_X, doubled_y, twin_groups, n_boot=2, learners=LINEAR, light=True)


def test_twins_in_one_group_with_one_target_and_distinct_rows_raise_no_warning():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    options = dict(learners=LINEAR, light=True, compute_diagnostics=False)
    doubled_X = pd.concat([X, X], ignore_index=True)
    same_target = pd.concat([y, y], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lambda_averaged_importance(doubled_X, same_target, np.tile(np.arange(24), 2), **options)
        lambda_averaged_importance(X, y, groups, **options)
        lambda_averaged_importance(X, y, None, **options)
        bootstrap_importance(X, y, groups, n_boot=2, learners=LINEAR, light=True)


def test_the_warning_for_duplicates_is_issued_once_by_the_bootstrap():
    X, y, groups = make_toy_problem(n=24, random_state=6)
    doubled_X = pd.concat([X, X], ignore_index=True)
    doubled_y = pd.concat([y, y + 1.0], ignore_index=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bootstrap_importance(
            doubled_X, doubled_y, np.tile(np.arange(24), 2), n_boot=3, learners=LINEAR, light=True
        )
    messages = [str(w.message) for w in caught if "exact duplicates" in str(w.message)]
    assert len(messages) == 1


# ----------------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------------
class GuardLog:
    """Zero-argument callable that records the stage of the computation it is called from."""

    def __init__(self):
        self.stage = "outside"
        self.stages = []

    def __call__(self):
        self.stages.append(self.stage)


def tagged(function, tag, log):
    """Wrap an engine function so that guard calls made while it runs carry ``tag``."""

    def wrapper(*args, **kwargs):
        outer, log.stage = log.stage, tag
        try:
            return function(*args, **kwargs)
        finally:
            log.stage = outer

    return wrapper


@pytest.fixture
def staged(monkeypatch):
    """Guard log that tags every call with the stage: folds, label or halves."""
    log = GuardLog()
    for name, tag in (("_cv_core", "folds"), ("_label_permutation_p", "label"), ("_split_half_correlation", "halves")):
        monkeypatch.setattr(importance, name, tagged(getattr(importance, name), tag, log))
    return log


@pytest.fixture
def restore_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


class StopRun(Exception):
    """Raised by guards that end a run."""


def stop_at(n, calls):
    """Guard that records its calls and raises ``StopRun`` at the ``n``-th call."""

    def guard():
        calls.append(1)
        if len(calls) == n:
            raise StopRun

    return guard


def consuming_guard(calls):
    """Guard that records its calls and draws from the global NumPy generator."""

    def guard():
        calls.append(1)
        np.random.random(5)

    return guard


def assert_identical(first, second):
    assert first.feature_names == second.feature_names
    assert_same_result(first, second)


def counting(events, label, function):
    """Wrap ``function`` so that every call appends ``label`` to ``events`` first."""

    def wrapper(*args, **kwargs):
        events.append(label)
        return function(*args, **kwargs)

    return wrapper


def traced(monkeypatch, name, label, events):
    """Log ``label`` for every call of ``importance.<name>`` and return the list that keeps its results."""
    original = getattr(importance, name)
    results = []

    def wrapper(*args, **kwargs):
        events.append(label)
        results.append(original(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(importance, name, wrapper)
    return results


class FakeClock:
    """Clock and sleep function that advance together without waiting."""

    def __init__(self):
        self.now = 0.0
        self.pauses = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.pauses.append(seconds)
        self.now += seconds


def readings(*values, then):
    """Temperature reader that returns the given values once and ``then`` afterwards."""
    queue = list(values)
    return lambda: queue.pop(0) if queue else then


def test_guard_is_an_optional_argument_that_defaults_to_none():
    for function in (lambda_averaged_importance, bootstrap_importance):
        assert inspect.signature(function).parameters["guard"].default is None


@pytest.mark.parametrize("bad", [1, 2.5, "check", (), {}], ids=repr)
def test_guard_must_be_none_or_callable(bad):
    X, y, groups = make_toy_problem(n=12, d=3, random_state=1)
    with pytest.raises(TypeError, match="guard"):
        lambda_averaged_importance(X, y, groups, light=True, guard=bad)
    with pytest.raises(TypeError, match="guard"):
        bootstrap_importance(X, y, groups, n_boot=2, guard=bad)


@pytest.mark.parametrize(
    "light, expected",
    [
        (True, ["folds"] * 18 + ["label"] * 2 + ["halves"]),
        (False, ["folds"] * 60 + ["label"] * 3 + ["halves"] * 2),
    ],
    ids=["light", "default"],
)
def test_guard_is_called_at_the_scheduled_points_of_each_stage(staged, light, expected):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    lambda_averaged_importance(X, y, groups, learners=LINEAR, light=light, n_repeats=2, n_perm=25, guard=staged)
    assert staged.stages == expected


@pytest.mark.parametrize(
    "n_lambdas, per_fold",
    [(1, "GGP"), (4, "GGPPPP"), (5, "GGPPPPGP"), (9, "GGPPPPGPPPPGP"), (12, "GGPPPPGPPPPGPPPP")],
)
def test_guard_is_called_before_each_fold_and_before_every_fourth_penalty(monkeypatch, n_lambdas, per_fold):
    events = []
    monkeypatch.setattr(importance, "_stack_weights", counting(events, "P", importance._stack_weights))
    X, y, groups = make_toy_problem(n=30, random_state=5)
    lambda_averaged_importance(
        X, y, groups, lambdas=lambda_grid(X, y, n=n_lambdas), learners=LINEAR, n_splits=3, n_repeats=1,
        cv_repeats=1, compute_diagnostics=False, guard=lambda: events.append("G"),
    )
    assert "".join(events) == per_fold * 3


def test_guard_is_called_before_each_fold_of_each_repeat_of_the_fold_assignment(monkeypatch):
    events = []
    monkeypatch.setattr(importance, "_stack_weights", counting(events, "P", importance._stack_weights))
    X, y, groups = make_toy_problem(n=30, random_state=5)
    lambda_averaged_importance(
        X, y, groups, lambdas=lambda_grid(X, y, n=2), learners=LINEAR, n_splits=3, n_repeats=1, cv_repeats=3,
        compute_diagnostics=False, guard=lambda: events.append("G"),
    )
    assert "".join(events) == "GGPP" * 9


def test_guard_is_called_every_tenth_iteration_of_the_label_permutation_loop(monkeypatch):
    events = []
    statistics = traced(monkeypatch, "_stack_cv_r2", "R", events)
    X, y, groups = noise_problem()
    arguments = (X.to_numpy(), y.to_numpy(), np.arange(30), LINEAR, 5, 12, 0, 0, 1)
    plain = importance._label_permutation_p(*arguments)
    plain_statistics = list(statistics)
    events.clear()
    statistics.clear()
    guarded = importance._label_permutation_p(*arguments, guard=lambda: events.append("G"))
    assert "".join(events) == "R" + "G" + "R" * 10 + "G" + "R" * 2
    assert statistics == plain_statistics and guarded == plain
    assert len(set(statistics)) == len(statistics)


def test_guard_is_called_every_tenth_iteration_of_the_split_half_loop(monkeypatch):
    events = []
    estimates = traced(monkeypatch, "_estimate", "E", events)
    X, y, groups = make_toy_problem(n=30, random_state=5)
    codes = importance._group_codes(groups, len(groups))
    arguments = (X.to_numpy(), y.to_numpy(), codes, LINEAR, 5, 12, "cv", 1.0, 0, 0)
    plain = importance._split_half_correlation(*arguments)
    plain_vectors = np.array([estimate.perm for estimate in estimates])
    events.clear()
    estimates.clear()
    guarded = importance._split_half_correlation(*arguments, guard=lambda: events.append("G"))
    assert "".join(events) == "G" + "EE" * 10 + "G" + "EE" * 2
    npt.assert_array_equal(np.array([estimate.perm for estimate in estimates]), plain_vectors)
    assert guarded == plain


@pytest.mark.usefixtures("restore_global_rng")
def test_guard_does_not_change_the_importance_of_the_stack():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    options = dict(light=True, n_repeats=3, random_state=1, compute_diagnostics=False)
    plain = lambda_averaged_importance(X, y, groups, **options)
    calls = []
    guarded = lambda_averaged_importance(X, y, groups, guard=consuming_guard(calls), **options)
    assert len(calls) == 18
    assert_identical(plain, guarded)


@pytest.mark.usefixtures("restore_global_rng", "quick_halves")
@pytest.mark.parametrize("light, n_calls", [(True, 21), (False, 64)], ids=["light", "default"])
def test_guard_does_not_change_any_diagnostic(light, n_calls):
    X, y, groups = noise_problem()
    options = dict(learners=LINEAR, light=light, n_repeats=2, n_perm=21, random_state=2)
    plain = lambda_averaged_importance(X, y, groups, **options)
    calls = []
    guarded = lambda_averaged_importance(X, y, groups, guard=consuming_guard(calls), **options)
    assert len(calls) == n_calls
    assert_identical(plain, guarded)


@pytest.mark.usefixtures("quick_halves")
@pytest.mark.parametrize("n", [1, 3, 17, 18, 19, 20])
def test_an_exception_raised_by_the_guard_ends_the_run_at_once(n):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    calls = []
    with pytest.raises(StopRun):
        lambda_averaged_importance(
            X, y, groups, learners=LINEAR, light=True, n_perm=9, guard=stop_at(n, calls)
        )
    assert len(calls) == n


@pytest.mark.usefixtures("restore_global_rng")
def test_bootstrap_guard_is_called_once_per_resample_before_its_estimate(monkeypatch):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    options = dict(n_boot=4, random_state=3, learners=LINEAR, light=True, n_repeats=2)
    plain = bootstrap_importance(X, y, groups, **options)
    events = []
    monkeypatch.setattr(importance, "_estimate", counting(events, "estimate", importance._estimate))

    def guard():
        events.append("guard")
        np.random.random(5)

    guarded = bootstrap_importance(X, y, groups, guard=guard, **options)
    assert events == ["guard", "estimate"] * 4
    pd.testing.assert_frame_equal(guarded, plain, check_exact=True)


def test_an_exception_raised_by_the_guard_ends_the_bootstrap_at_once():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    calls = []
    with pytest.raises(StopRun):
        bootstrap_importance(
            X, y, groups, n_boot=5, learners=LINEAR, light=True, n_repeats=2, guard=stop_at(2, calls)
        )
    assert len(calls) == 2


def test_a_thermal_guard_serves_as_guard_and_waits_while_the_machine_is_hot(quick_halves):
    X, y, groups = make_toy_problem(n=30, random_state=5)
    options = dict(learners=LINEAR, light=True, n_perm=9, random_state=1)
    plain = lambda_averaged_importance(X, y, groups, **options)
    clock = FakeClock()
    guard = ThermalGuard(
        read_temp=readings(91.0, 80.0, 70.0, then=55.0), read_cpu=lambda: 10.0, sleep=clock.sleep, clock=clock,
        poll_seconds=5.0, max_wait_seconds=60.0,
    )
    guarded = lambda_averaged_importance(X, y, groups, guard=guard, **options)
    summary = guard.summary()
    assert summary["checks"] == 20 and summary["waits"] == 1 and summary["max_temperature_c"] == 91.0
    assert clock.pauses == [5.0, 5.0]
    assert_identical(plain, guarded)


def test_a_thermal_timeout_raised_by_the_guard_reaches_the_caller():
    X, y, groups = make_toy_problem(n=30, random_state=5)
    clock = FakeClock()
    guard = ThermalGuard(
        read_temp=lambda: 99.0, read_cpu=lambda: 10.0, sleep=clock.sleep, clock=clock,
        poll_seconds=5.0, max_wait_seconds=10.0,
    )
    with pytest.raises(ThermalTimeout):
        lambda_averaged_importance(X, y, groups, learners=LINEAR, light=True, guard=guard)
    assert guard.summary()["checks"] == 1
    with pytest.raises(ThermalTimeout):
        bootstrap_importance(X, y, groups, n_boot=3, learners=LINEAR, light=True, guard=guard)
    assert guard.summary()["checks"] == 2


# ----------------------------------------------------------------------------
# Time limit and guard calls of the default configuration
# ----------------------------------------------------------------------------
def test_default_configuration_finishes_within_a_minute_and_calls_the_guard_72_times(staged):
    X, y, groups = make_toy_problem(n=30, d=8, random_state=0)
    started = time.process_time()
    res = lambda_averaged_importance(X, y, groups, guard=staged)
    elapsed = time.process_time() - started
    assert elapsed < 60.0
    assert staged.stages == ["folds"] * 60 + ["label"] * 10 + ["halves"] * 2
    assert res.lambdas.size == 12 and np.all(np.diff(res.lambdas) < 0)
    npt.assert_allclose(res.lambda_weights.sum(), 1.0)
    assert np.isfinite(res.cv_mse_by_lambda).all()
    assert isinstance(res.diagnostics["usable"], bool)
    p_scaled = res.diagnostics["perm_p_value"] * 101.0
    assert abs(p_scaled - round(p_scaled)) < 1e-9
    assert np.isfinite(res.diagnostics["split_half_correlation"])
    assert res.cluster.tolist() == list(range(8))


# ----------------------------------------------------------------------------
# Hygiene of the module
# ----------------------------------------------------------------------------
def unused_parameters(module):
    """Parameters of the functions of a module that are never read, as ``(function, parameter)`` pairs."""
    tree = ast.parse(inspect.getsource(module))
    unused = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            names = [a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]
            names += [a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None]
            read = {
                n.id for statement in node.body for n in ast.walk(statement)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            unused += [(node.name, name) for name in names if name not in read and name not in ("self", "cls")]
    return unused


def test_every_parameter_of_every_function_of_the_module_is_read():
    assert unused_parameters(importance) == []
