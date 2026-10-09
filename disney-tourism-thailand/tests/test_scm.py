"""Tests for dtt.scm: simplex QP, direct and nested synthetic control, ridge augmentation, placebos.

All data are SIMULATED.  Reference solutions come from cvxpy (interior point,
CLARABEL) and from first-order optimality conditions checked independently.
"""
import time

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cvxpy")

from dtt import scm, simulate  # noqa: E402


def cvx_solve(prob):
    """Solve with OSQP at tight tolerances (polishing on); fall back to SCS if not optimal."""
    prob.solve(solver=cp.OSQP, eps_abs=1e-12, eps_rel=1e-12, max_iter=500_000, polishing=True, verbose=False)
    if prob.status != "optimal":
        prob.solve(solver=cp.SCS, eps=1e-12, max_iters=500_000, verbose=False)
    assert prob.status in ("optimal", "optimal_inaccurate")
    return prob


def cvx_ls(A, b):
    """Reference solution of min ||b - A w||^2 on the simplex; returns (w, sum of squares)."""
    w = cp.Variable(A.shape[1])
    prob = cvx_solve(cp.Problem(cp.Minimize(cp.sum_squares(b - A @ w)), [w >= 0, cp.sum(w) == 1]))
    wv = np.maximum(w.value, 0.0)
    return wv / wv.sum(), prob.value


def cvx_simplex(H, g, ridge=0.0):
    """Reference solution of min 0.5 w'(H + ridge I)w - g'w on the simplex (H positive semidefinite)."""
    J = len(g)
    Hm = 0.5 * (H + H.T) + ridge * np.eye(J)
    lam, Q = np.linalg.eigh(Hm)
    keep = lam > 1e-12 * lam.max()
    B = np.sqrt(lam[keep])[:, None] * Q[:, keep].T  # B'B = Hm
    c = np.linalg.lstsq(B.T, g, rcond=None)[0]  # B'c = g
    w = cp.Variable(J)
    cvx_solve(cp.Problem(cp.Minimize(0.5 * cp.sum_squares(B @ w - c)), [w >= 0, cp.sum(w) == 1]))
    wv = np.maximum(w.value, 0.0)
    wv = wv / wv.sum()
    return wv, 0.5 * wv @ Hm @ wv - g @ wv


def objective(H, g, w, ridge=0.0):
    Hm = 0.5 * (H + H.T) + ridge * np.eye(len(g))
    return 0.5 * w @ Hm @ w - g @ w


def random_qp(seed, J, rank=None, scale=1.0):
    rng = np.random.default_rng(seed)
    r = J if rank is None else rank
    A = rng.normal(size=(max(r, 1) + 3, J)) * scale
    b = rng.normal(size=A.shape[0]) * scale
    return A.T @ A, A.T @ b, A, b


def kkt_residuals(H, g, w, ridge=0.0, tol=1e-7):
    """Violations of primal feasibility and of the KKT conditions for the simplex QP."""
    Hm = 0.5 * (H + H.T) + ridge * np.eye(len(g))
    grad = Hm @ w - g
    S = w > 1e-9
    lam = grad[S].mean()
    return {
        "neg": max(0.0, -w.min()),
        "sum": abs(w.sum() - 1.0),
        "stationarity": np.abs(grad[S] - lam).max(),
        "dual": max(0.0, (lam - grad[~S]).max()) if (~S).any() else 0.0,
    }


# ----------------------------------------------------------------------------
# qp_simplex
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed,J", [(0, 2), (1, 5), (2, 12), (3, 30), (4, 96)])
def test_qp_simplex_satisfies_kkt_and_matches_cvxpy(seed, J):
    H, g, _, _ = random_qp(seed, J)
    w, info = scm.qp_simplex(H, g, return_info=True)
    res = kkt_residuals(H, g, w)
    scale = max(1.0, np.abs(H).max(), np.abs(g).max())
    assert res["neg"] == 0.0 and res["sum"] < 1e-12
    assert res["stationarity"] < 1e-7 * scale and res["dual"] < 1e-7 * scale
    w_cvx, f_cvx = cvx_simplex(H, g)
    # the exact solver is at least as good as the reference and, since H is positive definite, agrees with it
    assert objective(H, g, w) <= f_cvx + 1e-9 * scale
    assert w == pytest.approx(w_cvx, abs=1e-6)
    assert set(info) == {"support", "lam", "iterations", "kkt_violation"}
    assert info["kkt_violation"] < 1e-7 * scale


def test_qp_simplex_known_solutions():
    # interior unconstrained optimum on the simplex is returned as is
    w_true = np.array([0.2, 0.5, 0.3])
    assert scm.qp_simplex(np.eye(3), w_true + 0.0) == pytest.approx(w_true, abs=1e-12)
    # a corner: the best single donor
    H = np.diag([1.0, 1.0, 1.0])
    g = np.array([5.0, 0.0, 0.0])
    assert scm.qp_simplex(H, g) == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    # one donor
    assert scm.qp_simplex(np.array([[2.0]]), np.array([1.0])).tolist() == [1.0]


def test_qp_simplex_handles_duplicated_donors_and_singular_matrices():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(8, 3))
    A = np.column_stack([a, a[:, 0], a[:, 1]])  # two exact duplicates
    b = a @ np.array([0.5, 0.3, 0.2]) + 0.01 * rng.normal(size=8)
    H, g = A.T @ A, A.T @ b
    w = scm.qp_simplex(H, g)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0, abs=1e-12)
    _, f_cvx = cvx_simplex(H, g)
    assert objective(H, g, w) <= f_cvx + 1e-9
    # more donors than periods (rank deficient): the minimum value is still attained
    H2, g2, _, _ = random_qp(6, 40, rank=5)
    w2 = scm.qp_simplex(H2, g2)
    _, f2 = cvx_simplex(H2, g2)
    assert objective(H2, g2, w2) <= f2 + 1e-9 * max(1, np.abs(H2).max())


def exact_fit_problem(seed, T=6, J=20, n_post=4):
    """Donors whose pre-period outcomes contain the treated unit's pre-period outcome in their convex hull.

    Returns the treated and donor outcomes of the pre-period and of the post-period.
    """
    rng = np.random.default_rng(seed)
    Y0 = 3.0 + 0.3 * rng.normal(size=(T + n_post, J)).cumsum(axis=0)
    w_true = rng.dirichlet(np.ones(J))
    y_pre = Y0[:T] @ w_true
    return y_pre, Y0[:T], Y0[T:]


def min_norm_minimiser(y, Y0):
    """Weights of smallest Euclidean norm among the simplex weights that reproduce ``y`` exactly (cvxpy)."""
    w = cp.Variable(Y0.shape[1])
    cvx_solve(cp.Problem(cp.Minimize(cp.sum_squares(w)), [Y0 @ w == y, w >= 0, cp.sum(w) == 1]))
    wv = np.maximum(w.value, 0.0)
    return wv / wv.sum()


@pytest.mark.parametrize("seed", range(6))
def test_a_non_unique_exact_fit_returns_a_vertex_of_the_set_of_minimisers(seed):
    y, Y0, _ = exact_fit_problem(seed)
    T, J = Y0.shape
    fit = scm.fit_direct(y, Y0)
    assert fit.loss < 1e-20 and fit.w.min() >= 0.0 and fit.w.sum() == pytest.approx(1.0, abs=1e-12)
    support = np.flatnonzero(fit.w > 1e-12)
    assert support.size == T + 1 < J
    # the equality constraints restricted to the support have independent columns: a vertex
    constraints = np.vstack([Y0[:, support], np.ones(support.size)])
    assert np.linalg.matrix_rank(constraints) == support.size


@pytest.mark.parametrize("seed", range(3))
def test_the_solver_does_not_return_the_minimiser_of_smallest_norm(seed):
    y, Y0, post = exact_fit_problem(seed)
    fit = scm.fit_direct(y, Y0)
    other = min_norm_minimiser(y, Y0)
    # both reproduce the pre-period exactly, the solver's vector has the larger norm and another effect
    assert np.abs(Y0 @ other - y).max() < 1e-6 and fit.loss < 1e-20
    assert fit.w @ fit.w > other @ other + 1e-3
    assert np.abs(post @ fit.w - post @ other).max() > 1e-3


def test_identical_donors_leave_all_weight_to_the_smallest_index():
    rng = np.random.default_rng(3)
    base = 3.0 + 0.3 * rng.normal(size=7).cumsum()
    other = 3.0 + 0.3 * rng.normal(size=7).cumsum()
    assert scm.fit_direct(base, np.column_stack([base, base, base])).w.tolist() == [1.0, 0.0, 0.0]
    assert scm.fit_direct(base, np.column_stack([other, base, base, base])).w.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_the_minimiser_is_unique_with_a_positive_ridge_penalty():
    y, Y0, _ = exact_fit_problem(1)
    first = scm.fit_direct(y, Y0, ridge=0.5)
    permuted = np.random.default_rng(0).permutation(Y0.shape[1])
    second = scm.fit_direct(y, Y0[:, permuted], ridge=0.5)
    assert second.w[np.argsort(permuted)] == pytest.approx(first.w, abs=1e-10)
    assert (first.w > 1e-6).sum() > Y0.shape[0] + 1


def test_qp_simplex_ridge_equals_adding_to_the_diagonal():
    H, g, _, _ = random_qp(7, 8)
    for ridge in (0.5, 5.0):
        a = scm.qp_simplex(H, g, ridge=ridge)
        b = scm.qp_simplex(H + ridge * np.eye(8), g)
        assert a == pytest.approx(b, abs=1e-12)
        res = kkt_residuals(H, g, a, ridge=ridge)
        assert res["stationarity"] < 1e-7 and res["dual"] < 1e-7


def test_qp_simplex_solutions_are_sparse_in_the_typical_case():
    H, g, _, _ = random_qp(8, 60)
    w = scm.qp_simplex(H, g)
    assert (w > 0).sum() < 60  # simplex constraints bind


# ----------------------------------------------------------------------------
# Direct and nested fits
# ----------------------------------------------------------------------------
def block_problem(seed=0, T0=7, T=20, J=5, noise=0.05):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 0.3, (T, 2)), axis=0)
    L = rng.normal(size=(J, 2))
    mu = rng.normal(3, 1, J)
    Y0 = mu + f @ L.T + rng.normal(0, noise, (T, J))
    w = rng.dirichlet(np.ones(J))
    y1 = Y0 @ w + rng.normal(0, noise / 2, T)
    pre = np.arange(T) < T0
    return y1, Y0, pre, w


def test_fit_direct_matches_cvxpy_solution():
    y1, Y0, pre, _ = block_problem(1)
    f = scm.fit_direct(y1[pre], Y0[pre])
    w_ref, ssq = cvx_ls(Y0[pre], y1[pre])
    assert f.method == "direct" and f.v is None and f.converged
    assert f.w == pytest.approx(w_ref, abs=1e-8)
    assert f.loss == pytest.approx(ssq / pre.sum(), rel=1e-9)
    assert f.rmspe == pytest.approx(np.sqrt(f.loss))
    assert f.w.sum() == pytest.approx(1.0) and f.w.min() >= 0


def test_nested_with_pre_period_predictors_reaches_the_direct_optimum():
    """With every pre-period outcome as a predictor, uniform V is optimal for the nested problem."""
    for seed in range(4):
        y1, Y0, pre, _ = block_problem(seed, J=6)
        d = scm.fit_direct(y1[pre], Y0[pre])
        n = scm.fit_nested(y1[pre], Y0[pre], y1[pre], Y0[pre], n_starts=6, seed=0)
        assert n.loss == pytest.approx(d.loss, rel=1e-8, abs=1e-12)
        assert n.w == pytest.approx(d.w, abs=1e-5)
        # the nested objective at uniform v equals the direct loss
        loss_uniform, _ = scm._nested_objective(np.zeros(pre.sum()), y1[pre], Y0[pre], y1[pre], Y0[pre], 0.0)
        assert loss_uniform == pytest.approx(d.loss, rel=1e-8)


def test_nested_gradient_matches_finite_differences():
    """Analytic gradient (implicit differentiation) against central differences.

    More predictors than donors, so that the inner solution is unique and varies with v.
    """
    rng = np.random.default_rng(3)
    T0, J, k = 9, 4, 7
    Y0 = rng.normal(size=(T0, J)) + 3
    y1 = Y0 @ np.array([0.4, 0.3, 0.2, 0.1]) + 0.2 * rng.normal(size=T0)
    X0 = rng.normal(size=(k, J))
    X1 = X0 @ np.array([0.5, 0.2, 0.2, 0.1]) + 0.3 * rng.normal(size=k)
    checked = 0
    for ridge in (0.0, 0.3):
        for _ in range(3):
            theta = rng.normal(size=k)
            f0, g0 = scm._nested_objective(theta, X1, X0, y1, Y0, ridge)
            assert np.abs(g0).max() > 1e-4  # a non-trivial gradient
            num = np.zeros(k)
            h = 1e-6
            for i in range(k):
                e = np.zeros(k)
                e[i] = h
                fp, _ = scm._nested_objective(theta + e, X1, X0, y1, Y0, ridge)
                fm, _ = scm._nested_objective(theta - e, X1, X0, y1, Y0, ridge)
                num[i] = (fp - fm) / (2 * h)
            assert g0 == pytest.approx(num, rel=1e-4, abs=1e-7)
            checked += 1
    assert checked == 6


def test_nested_with_covariate_predictors_is_bounded_below_by_the_direct_optimum():
    rng = np.random.default_rng(4)
    y1, Y0, pre, _ = block_problem(2, J=6)
    X0 = np.vstack([Y0[pre][[0, 3, 6]], rng.normal(size=(2, 6))])  # three outcome lags plus two covariates
    X1 = np.r_[y1[pre][[0, 3, 6]], rng.normal(size=2)]
    n = scm.fit_nested(X1, X0, y1[pre], Y0[pre], n_starts=6, seed=0)
    d = scm.fit_direct(y1[pre], Y0[pre])
    assert n.loss >= d.loss - 1e-12
    assert n.v.sum() == pytest.approx(1.0) and n.v.min() >= 0
    assert len(n.starts) == 6
    assert n.loss <= min(s["loss"] for s in n.starts) + 1e-15
    # determinism
    n2 = scm.fit_nested(X1, X0, y1[pre], Y0[pre], n_starts=6, seed=0)
    assert n2.loss == n.loss and np.array_equal(n2.w, n.w)
    # inner weights are the simplex QP solution for the returned v
    w_chk = scm.solve_w_given_v(X1, X0, n.v)
    assert w_chk == pytest.approx(n.w, abs=1e-10)


def test_solve_w_given_v_matches_cvxpy():
    rng = np.random.default_rng(6)
    k, J = 6, 7
    X0, X1 = rng.normal(size=(k, J)), rng.normal(size=k)
    v = rng.dirichlet(np.ones(k))
    w = scm.solve_w_given_v(X1, X0, v)
    w_ref, _ = cvx_ls(np.sqrt(v)[:, None] * X0, np.sqrt(v) * X1)
    assert w == pytest.approx(w_ref, abs=1e-8)


# ----------------------------------------------------------------------------
# build_predictors
# ----------------------------------------------------------------------------
def test_build_predictors_means_scaling_and_covariates():
    rng = np.random.default_rng(7)
    T, J = 10, 4
    y1, Y0 = rng.normal(size=T), rng.normal(size=(T, J))
    X1, X0 = scm.build_predictors(y1, Y0, periods=[[0], [1, 2], [3, 4, 5]], scale="none")
    assert X1 == pytest.approx([y1[0], y1[1:3].mean(), y1[3:6].mean()])
    assert X0[1] == pytest.approx(Y0[1:3].mean(axis=0))
    z1, Z0 = np.array([2.0]), rng.normal(size=(1, J))
    X1c, X0c = scm.build_predictors(y1, Y0, periods=[[0, 1]], z1=z1, Z0=Z0, scale="sd")
    full1 = np.r_[y1[:2].mean(), 2.0]
    full0 = np.vstack([Y0[:2].mean(axis=0), Z0])
    sd = np.std(np.column_stack([full1, full0]), axis=1, ddof=1)
    assert X1c == pytest.approx(full1 / sd) and X0c == pytest.approx(full0 / sd[:, None])
    with pytest.raises(ValueError):
        scm.build_predictors(y1, Y0, periods=[[0]], scale="bad")


# ----------------------------------------------------------------------------
# Ridge augmentation
# ----------------------------------------------------------------------------
def test_ridge_augmentation_algebra():
    rng = np.random.default_rng(8)
    T0, J = 6, 9
    X0 = rng.normal(size=(T0, J)) + 2
    w = rng.dirichlet(np.ones(J))
    x1 = X0 @ w + 0.3 * rng.normal(size=T0)
    lam = 0.7
    aug = scm.ridge_augment_weights(x1, X0, w, lam)
    assert aug.sum() == pytest.approx(1.0, abs=1e-12)
    # primal ridge form: w + (Xc'Xc + lam I)^{-1} Xc' (x1 - X0 w)
    Xc = X0 - X0.mean(axis=1, keepdims=True)
    primal = w + np.linalg.solve(Xc.T @ Xc + lam * np.eye(J), Xc.T @ (x1 - X0 @ w))
    assert aug == pytest.approx(primal, abs=1e-10)
    # large lambda returns the original weights; small lambda balances the pre-period
    assert scm.ridge_augment_weights(x1, X0, w, 1e12) == pytest.approx(w, abs=1e-9)
    tiny = scm.ridge_augment_weights(x1, X0, w, 1e-10)
    assert X0 @ tiny == pytest.approx(x1, abs=1e-6)
    # augmentation reduces the pre-treatment imbalance
    assert np.linalg.norm(x1 - X0 @ aug) < np.linalg.norm(x1 - X0 @ w)


# ----------------------------------------------------------------------------
# synth
# ----------------------------------------------------------------------------
def test_synth_recovers_known_weights_and_effect_without_noise():
    rng = np.random.default_rng(9)
    T, T0, J = 22, 14, 5
    f = np.cumsum(rng.normal(0, 0.4, (T, 3)), axis=0)
    Y0 = 3 + f @ rng.normal(size=(J, 3)).T + rng.normal(0, 0.3, (T, J)).cumsum(axis=0) * 0.2
    w_true = np.array([0.0, 0.5, 0.3, 0.0, 0.2])
    pre = np.arange(T) < T0
    y1 = Y0 @ w_true + 2.0 * (~pre)
    r = scm.synth(y1, Y0, pre, method="both")
    assert r.w == pytest.approx(w_true, abs=1e-6)
    assert r.rmspe_pre < 1e-6
    assert r.mean_post_gap == pytest.approx(2.0, abs=1e-6)
    assert r.gap[pre] == pytest.approx(0.0, abs=1e-6)
    assert set(r.fits) == {"nested", "direct"} and r.primary == "nested"
    assert r.synthetic == pytest.approx(Y0 @ r.w)


def test_synth_summary_statistics_are_consistent():
    y1, Y0, pre, _ = block_problem(10, J=5)
    r = scm.synth(y1, Y0, pre, method="both", n_starts=4, seed=1)
    gap = y1 - Y0 @ r.w
    assert r.gap == pytest.approx(gap)
    assert r.rmspe_pre == pytest.approx(np.sqrt(np.mean(gap[pre] ** 2)))
    assert r.rmspe_post == pytest.approx(np.sqrt(np.mean(gap[~pre] ** 2)))
    assert r.ratio == pytest.approx(r.rmspe_post / r.rmspe_pre)
    assert r.mean_post_gap == pytest.approx(gap[~pre].mean())
    s = r.summary()
    assert set(s.method) == {"nested", "direct"}
    assert s["rmspe_pre"].min() == pytest.approx(r.rmspe_pre, rel=1e-9)


def test_synth_single_method_and_invalid_method():
    y1, Y0, pre, _ = block_problem(11, J=4)
    assert set(scm.synth(y1, Y0, pre, method="direct").fits) == {"direct"}
    assert set(scm.synth(y1, Y0, pre, method="nested").fits) == {"nested"}
    with pytest.raises(ValueError):
        scm.synth(y1, Y0, pre, method="other")


def test_synth_with_covariate_predictors_and_ridge_augmentation():
    rng = np.random.default_rng(12)
    y1, Y0, pre, _ = block_problem(12, J=6)
    z0 = rng.normal(size=(2, 6))
    X1, X0 = scm.build_predictors(y1, Y0, periods=[[0, 1, 2], [3, 4, 5, 6]], z1=z0 @ np.full(6, 1 / 6), Z0=z0)
    r = scm.synth(y1, Y0, pre, X1=X1, X0=X0, method="both", augment_lambda=0.5)
    assert r.w_aug is not None and r.w_aug.sum() == pytest.approx(1.0)
    assert r.gap_aug == pytest.approx(y1 - Y0 @ r.w_aug)
    assert np.linalg.norm(r.gap_aug[pre]) <= np.linalg.norm(r.gap[pre]) + 1e-12
    # the direct fit lower-bounds the pre-treatment MSPE of any simplex weights
    assert r.fits["direct"].loss <= r.fits["nested"].loss + 1e-12


def test_synth_matches_independent_cvxpy_solution_on_simulated_hong_kong_block():
    df, _ = simulate.simulate_panel(seed=21, trend_spread=0.5)
    hk = df[(df.study_case == "Hong Kong") & df.year.between(1999, 2018)]
    W = hk.pivot(index="year", columns="unit_id", values="receipts_pct_gdp")[[2, 3, 4, 7, 8, 9]]
    pre = W.index.values < 2005
    y1, Y0 = W[2].to_numpy(), W[[3, 4, 7, 8, 9]].to_numpy()
    r = scm.synth(y1, Y0, pre, method="both")
    w_ref, _ = cvx_ls(Y0[pre], y1[pre])
    assert r.w == pytest.approx(w_ref, abs=1e-6)
    assert r.fits["direct"].w == pytest.approx(w_ref, abs=1e-6)
    assert abs(r.mean_post_gap - 5.0) < 0.6  # known effect of the simulated panel


# ----------------------------------------------------------------------------
# Placebo in space
# ----------------------------------------------------------------------------
def test_rank_descending_with_ties():
    assert scm.rank_descending(np.array([3.0, 1.0, 2.0])).tolist() == [1, 3, 2]
    assert scm.rank_descending(np.array([2.0, 2.0, 1.0, 3.0])).tolist() == [2, 2, 4, 1]


def reference_placebo(Y, treated, pre, exclude_treated):
    """Independent implementation with cvxpy: one constrained regression per unit."""
    T, N = Y.shape
    rows = []
    for i in [treated] + [j for j in range(N) if j != treated]:
        if i == treated:
            pool = [j for j in range(N) if j != treated]
        elif exclude_treated:
            pool = [j for j in range(N) if j not in (i, treated)]
        else:
            pool = [j for j in range(N) if j != i]
        wv, _ = cvx_ls(Y[pre][:, pool], Y[pre, i])
        gap = Y[:, i] - Y[:, pool] @ wv
        rows.append((i, np.sqrt(np.mean(gap[pre] ** 2)), np.sqrt(np.mean(gap[~pre] ** 2)), gap[~pre].mean()))
    return pd.DataFrame(rows, columns=["unit", "pre", "post", "mean_gap"])


@pytest.mark.parametrize("exclude_treated", [True, False])
def test_placebo_table_matches_independent_solutions(exclude_treated):
    d = simulate.simulate_factor_panel(n_donors=7, T=20, T0=12, n_active=3, seed=3)
    Y, pre = d["Y"], d["pre"]
    pl = scm.placebo_in_space(Y, 0, pre, exclude_treated=exclude_treated)
    ref = reference_placebo(Y, 0, pre, exclude_treated)
    t = pl.table
    assert t.unit.tolist() == ref.unit.tolist()
    assert t.pre_rmspe.to_numpy() == pytest.approx(ref.pre.to_numpy(), rel=1e-6, abs=1e-8)
    assert t.post_rmspe.to_numpy() == pytest.approx(ref.post.to_numpy(), rel=1e-6, abs=1e-8)
    assert t.mean_post_gap.to_numpy() == pytest.approx(ref.mean_gap.to_numpy(), rel=1e-6, abs=1e-7)


def test_placebo_ranks_and_p_values():
    d = simulate.simulate_factor_panel(n_donors=15, T=20, T0=12, n_active=3, tau=3.0, seed=4)
    pl = scm.placebo_in_space(d["Y"], 0, d["pre"], labels=[f"u{i}" for i in range(16)])
    t = pl.table
    assert len(t) == 16 and t.unit.iloc[0] == "u0" and pl.treated_label == "u0"
    assert t.rank_ratio.iloc[0] == 1 and t.rank_abs_mean_gap.iloc[0] == 1  # true effect of 3 is the largest
    assert pl.p_value_ratio == pytest.approx(1 / 16) and pl.p_value_abs_mean_gap == pytest.approx(1 / 16)
    assert t.ratio.iloc[0] == pytest.approx(t.post_rmspe.iloc[0] / t.pre_rmspe.iloc[0])
    assert t.abs_mean_gap.to_numpy() == pytest.approx(np.abs(t.mean_post_gap.to_numpy()))
    assert pl.gaps.shape == (20, 16)
    assert scm.rank_descending(t.ratio.to_numpy()).tolist() == t.rank_ratio.tolist()


def test_placebo_p_values_rank_every_unit_including_those_that_fit_badly():
    d = simulate.simulate_factor_panel(n_donors=15, T=20, T0=12, n_active=3, tau=3.0, seed=4)
    pre = d["pre"]
    clean = scm.placebo_in_space(d["Y"], 0, pre)
    assert clean.table.rank_abs_mean_gap.iloc[0] == 1 and clean.p_value_abs_mean_gap == pytest.approx(1 / 16)
    Y = d["Y"].copy()
    Y[:, 7] += 50.0
    pl = scm.placebo_in_space(Y, 0, pre)
    t = pl.table
    assert len(t) == 16
    assert t.pre_rmspe.iloc[7] > 10.0 * np.median(np.delete(t.pre_rmspe.to_numpy(), 7))
    assert t.rank_abs_mean_gap.iloc[7] == 1 and t.rank_abs_mean_gap.iloc[0] == 2
    assert pl.p_value_abs_mean_gap == pytest.approx(2 / 16)
    assert "all" in scm.placebo_in_space.__doc__ and "comparable" in scm.placebo_in_space.__doc__


def test_placebo_without_effect_gives_a_large_p_value():
    d = simulate.simulate_factor_panel(n_donors=15, T=20, T0=12, n_active=3, tau=0.0, seed=5)
    pl = scm.placebo_in_space(d["Y"], 0, d["pre"])
    assert pl.p_value_ratio > 0.1


def test_placebo_donor_subset_and_pool_logic():
    d = simulate.simulate_factor_panel(n_donors=8, T=20, T0=12, n_active=3, seed=6)
    Y, pre = d["Y"], d["pre"]
    donors = [1, 2, 3, 4, 5]
    pl = scm.placebo_in_space(Y, 0, pre, donors=donors)
    assert pl.table.unit.tolist() == [0] + donors
    a = scm.placebo_in_space(Y, 0, pre, exclude_treated=True).table.set_index("unit")
    b = scm.placebo_in_space(Y, 0, pre, exclude_treated=False).table.set_index("unit")
    # the treated unit's own fit does not depend on the pool convention
    assert a.loc[0, "pre_rmspe"] == pytest.approx(b.loc[0, "pre_rmspe"])
    # a placebo may use the treated unit as a donor only when it is kept in the pool
    assert (b.loc[1:, "pre_rmspe"] <= a.loc[1:, "pre_rmspe"] + 1e-9).all()
    assert (b.loc[1:, "pre_rmspe"] < a.loc[1:, "pre_rmspe"] - 1e-6).any()


def test_placebo_nested_method_uses_predictors():
    d = simulate.simulate_factor_panel(n_donors=6, T=18, T0=10, n_active=3, seed=7)
    Y, pre = d["Y"], d["pre"]
    X = Y[pre]  # outcomes in the pre-period as predictors (units in columns)
    a = scm.placebo_in_space(Y, 0, pre, method="direct")
    b = scm.placebo_in_space(Y, 0, pre, method="nested", X=X, n_starts=3)
    assert b.table.pre_rmspe.to_numpy() == pytest.approx(a.table.pre_rmspe.to_numpy(), rel=1e-5, abs=1e-7)


def test_ninety_six_economy_pool_is_fast_and_exact():
    d = simulate.simulate_factor_panel(n_donors=96, T=22, T0=14, seed=8)
    Y, pre = d["Y"], d["pre"]
    t0 = time.perf_counter()
    r = scm.synth(Y[:, 0], Y[:, 1:], pre, method="both", seed=0)
    pl = scm.placebo_in_space(Y, 0, pre)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert len(pl.table) == 97
    assert abs(r.mean_post_gap - d["tau"]) < 0.3
    assert r.rmspe_pre < 0.1
    # exactness check on a subset of placebo units against the interior-point reference
    ref = reference_placebo_subset(Y, pre, [0, 5, 40, 96])
    t = pl.table.set_index("unit")
    for i, (pre_r, post_r) in ref.items():
        assert t.loc[i, "pre_rmspe"] <= pre_r + 1e-9
        assert t.loc[i, "pre_rmspe"] == pytest.approx(pre_r, rel=1e-6, abs=1e-8)
        assert t.loc[i, "post_rmspe"] == pytest.approx(post_r, rel=1e-4, abs=1e-6)


def reference_placebo_subset(Y, pre, units):
    out = {}
    N = Y.shape[1]
    for i in units:
        pool = [j for j in range(1, N) if j != i]
        wv, _ = cvx_ls(Y[pre][:, pool], Y[pre, i])
        gap = Y[:, i] - Y[:, pool] @ wv
        out[i] = (np.sqrt(np.mean(gap[pre] ** 2)), np.sqrt(np.mean(gap[~pre] ** 2)))
    return out
