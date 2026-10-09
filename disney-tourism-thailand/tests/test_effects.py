"""Tests for dtt.effects: synthetic-control effects, placebo inference and the batch driver.

All data are SIMULATED by dtt.simulate_global.  Reference values are obtained by calling the
estimators of dtt.scm and dtt.did directly on arrays built by hand from the simulated panel.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dtt import scm  # noqa: E402
from dtt.cases import (  # noqa: E402
    FEATURE_COLUMNS,
    POST_HORIZON,
    build_episodes,
    case_features,
    check_feasibility,
    donor_pool,
    load_cases,
)
from dtt.did import friend_did, xtreg_twfe  # noqa: E402
from dtt.effects import (  # noqa: E402
    EXACT_FIT_RELATIVE_TOLERANCE,
    EXACT_FIT_TOLERANCE,
    GUARD_INTERVAL,
    MIN_COMPARABLE_PLACEBOS,
    PLACEBO_FIT_FACTOR,
    CaseEffect,
    _comparable_placebos,
    _identification_range,
    _ratio,
    _sample_sd,
    estimate_all,
    estimate_case,
    rank_p_value,
    write_effects,
)
from dtt.panel import build_global_panel  # noqa: E402
from dtt.simulate_global import simulate_global_world  # noqa: E402

FIRST, LAST = 1995, 2019
EFFECT_COLUMNS = [
    "case_id", "iso3", "economy_group", "opening_year", "n_pre", "n_post", "window_end", "post_horizon_used",
    "n_donors", "method", "att_pp", "att_rel_pct", "att_pp_min", "att_pp_max", "att_pp_lp_min", "att_pp_lp_max",
    "rmspe_pre", "rmspe_post", "rmspe_ratio", "p_gap_rank", "p_ratio_rank", "placebo_sd_pp", "placebo_sd_rel_pct",
    "n_placebos", "n_placebos_comparable", "n_placebos_exact", "fit_exact", "fit_tolerance", "pre_fit_ok",
    "did_att_pp",
]
PLACEBO_COLUMNS = ["case_id", "donor_iso3", "placebo_att_pp", "placebo_att_rel_pct", "placebo_rmspe_pre", "comparable"]
RESULT_ARRAYS = (
    "years", "actual", "synthetic", "gap", "placebo_att_pp", "placebo_att_rel_pct", "placebo_rmspe_pre", "placebo_gaps",
    "placebo_comparable",
)
RESULT_NUMBERS = (
    "n_pre", "n_post", "window_end", "post_horizon_used", "att_pp", "att_rel_pct", "rmspe_pre", "rmspe_post",
    "rmspe_ratio", "p_gap_rank", "p_ratio_rank", "did_att_pp", "placebo_sd_pp", "placebo_sd_rel_pct", "fit_tolerance",
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return simulate_global_world(tmp_path_factory.mktemp("world"), seed=0)


@pytest.fixture(scope="module")
def panel(world) -> pd.DataFrame:
    return build_global_panel(world.raw_dir)


@pytest.fixture(scope="module")
def cases(world) -> pd.DataFrame:
    return load_cases(world.cases_path)


@pytest.fixture(scope="module")
def fitted(panel, cases):
    return estimate_all(panel, cases)


@pytest.fixture(scope="module")
def mid(panel, cases) -> pd.Series:
    """A feasible case with complete outcome data and a mid-sample opening year."""
    feas = check_feasibility(cases, panel)
    full = feas.loc[feas["feasible"] & (feas["n_pre"] == feas["opening_year"] - FIRST)]
    pick = full.iloc[(full["opening_year"] - full["opening_year"].median()).abs().argsort(kind="stable")].iloc[0]["case_id"]
    return cases.loc[cases["case_id"] == pick].iloc[0]


@pytest.fixture(scope="module")
def cluster_world(tmp_path_factory):
    return simulate_global_world(tmp_path_factory.mktemp("cluster"), seed=0, cluster_economies=True)


@pytest.fixture(scope="module")
def cluster_panel(cluster_world) -> pd.DataFrame:
    return build_global_panel(cluster_world.raw_dir)


@pytest.fixture(scope="module")
def cluster_cases(cluster_world) -> pd.DataFrame:
    return load_cases(cluster_world.cases_path)


@pytest.fixture(scope="module")
def episodes(cluster_cases) -> pd.DataFrame:
    return build_episodes(cluster_cases)


@pytest.fixture(scope="module")
def split_episodes(cluster_cases) -> pd.DataFrame:
    """Episode table in which every opening is its own episode."""
    return build_episodes(cluster_cases, merge_gap=1)


@pytest.fixture(scope="module")
def episode_fit(cluster_panel, episodes):
    return estimate_all(cluster_panel, episodes)


@pytest.fixture(scope="module")
def split_fit(cluster_panel, split_episodes, cluster_cases):
    return estimate_all(cluster_panel, split_episodes, catalogue=cluster_cases)


def window_end_of(case, horizon: int | None = POST_HORIZON) -> int:
    """Last year of the post-opening window, written out directly."""
    end = LAST
    if horizon is not None:
        end = min(end, int(case["opening_year"]) + horizon - 1)
    following = case.get("next_start_year", np.nan)
    if not pd.isna(following):
        end = min(end, int(following) - 1)
    return end


def hand_arrays(panel: pd.DataFrame, case, donors: list[str], horizon: int | None = POST_HORIZON):
    """Outcome arrays of the case and its donors over the window, on the years with an observed outcome of the case."""
    end = window_end_of(case, horizon)
    wide = panel.loc[panel["year"].between(FIRST, end)].pivot(index="year", columns="iso3", values="receipts_pct_gdp")
    y = wide[case["iso3"]].to_numpy(float)
    valid = np.isfinite(y)
    pre = (wide.index.to_numpy() < case["opening_year"])[valid]
    return wide, y[valid], wide[donors].to_numpy(float)[valid], pre, valid


def ratio_ranks_are_defined(table: pd.DataFrame, tolerance: float) -> bool:
    """True when no unit of the placebo table has a pre-opening RMSPE within ``tolerance``, where the ratio is undefined."""
    return bool(table["pre_rmspe"].min() > tolerance)


def assert_same_effect(a: CaseEffect, b: CaseEffect) -> None:
    """Every array and number of two results is identical, undefined values counting as equal."""
    for name in RESULT_ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.weights.to_numpy(), b.weights.to_numpy())
    for name in RESULT_NUMBERS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def rank_p(statistic: float, others) -> float:
    """Rank p-value written out directly: (1 + defined others at least as large) / (1 + defined others)."""
    defined = [float(v) for v in others if not np.isnan(v)]
    return (1 + sum(v >= statistic for v in defined)) / (1 + len(defined))


def placebo_ratios(effect: CaseEffect) -> np.ndarray:
    """Post-to-pre RMSPE ratio of every placebo from the stored gaps; NaN where the pre-opening RMSPE is within the fit tolerance."""
    pre = (effect.years < effect.opening_year) & np.isfinite(effect.actual)
    post = effect.years >= effect.opening_year
    rmspe_pre = np.sqrt(np.mean(effect.placebo_gaps[pre] ** 2, axis=0))
    rmspe_post = np.sqrt(np.mean(effect.placebo_gaps[post] ** 2, axis=0))
    safe = np.where(rmspe_pre > 0.0, rmspe_pre, 1.0)
    return np.where(rmspe_pre <= effect.fit_tolerance, np.nan, rmspe_post / safe)


def comparable_by_hand(placebo_pre: np.ndarray, factor: float | None = PLACEBO_FIT_FACTOR) -> np.ndarray:
    """Comparable placebos: pre-opening RMSPE within ``factor`` times the median (at least the exact-fit tolerance), or among the five best (ties stay)."""
    if factor is None:
        return np.ones(placebo_pre.size, dtype=bool)
    keep = placebo_pre <= factor * max(float(np.median(placebo_pre)), EXACT_FIT_TOLERANCE)
    return keep | (placebo_pre <= np.sort(placebo_pre)[min(placebo_pre.size, 5) - 1])


def toy_panel(donor_values: np.ndarray, treated_values: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Panel with donors D00, D01, ... and the treated economy TRT over the years FIRST to LAST, and the donor codes."""
    years = np.arange(FIRST, LAST + 1)
    rows = [(f"D{j:02d}", int(y), float(v)) for j in range(donor_values.shape[0]) for y, v in zip(years, donor_values[j])]
    rows += [("TRT", int(y), float(v)) for y, v in zip(years, treated_values)]
    codes = [f"D{j:02d}" for j in range(donor_values.shape[0])]
    return pd.DataFrame(rows, columns=["iso3", "year", "receipts_pct_gdp"]), codes


def random_walks(n: int, seed: int) -> np.ndarray:
    """Donor outcome paths around three percent of GDP, one row per donor."""
    rng = np.random.default_rng(seed)
    return 3.0 + 0.1 * rng.normal(size=(n, LAST - FIRST + 1)).cumsum(axis=1)


def planted_panel(panel: pd.DataFrame, case, donors: list[str], weights: np.ndarray, shift: float) -> pd.DataFrame:
    """Panel in which the case economy follows a fixed weighted average of the donors plus ``shift`` from its opening year on."""
    wide = panel.loc[panel["year"].between(FIRST, LAST)].pivot(index="year", columns="iso3", values="receipts_pct_gdp")
    planted = pd.Series(wide[donors].to_numpy() @ weights + np.where(wide.index >= int(case["opening_year"]), shift, 0.0), index=wide.index)
    altered = panel.copy()
    own = altered["iso3"] == case["iso3"]
    altered.loc[own, "receipts_pct_gdp"] = altered.loc[own, "year"].map(planted).to_numpy()
    return altered


def exact_fit_range(y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray, tolerance: float = 0.0) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Smallest and largest post-opening mean gap over the simplex weights that reproduce ``y1[pre]`` within ``tolerance``.

    A tolerance of zero imposes equality in every pre-opening year. Also returns the equality constraints
    (matrix and right-hand side) of the weight set with zero tolerance.
    """
    n_donors = Y0.shape[1]
    level = Y0[~pre].mean(axis=0)
    ones = np.ones((1, n_donors))
    A = np.vstack([Y0[pre], ones])
    b = np.r_[y1[pre], 1.0]
    if tolerance == 0.0:
        constraints = {"A_eq": A, "b_eq": b}
    else:
        constraints = {
            "A_ub": np.vstack([Y0[pre], -Y0[pre]]),
            "b_ub": np.r_[y1[pre] + tolerance, tolerance - y1[pre]],
            "A_eq": ones,
            "b_eq": np.ones(1),
        }
    low = linprog(level, bounds=(0, None), method="highs", **constraints)
    high = linprog(-level, bounds=(0, None), method="highs", **constraints)
    assert low.status == 0 and high.status == 0
    observed = y1[~pre].mean()
    return observed + high.fun, observed - low.fun, A, b


# ----------------------------------------------------------------------------
# estimate_case: direct method against dtt.scm called by hand
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [POST_HORIZON, None])
def test_direct_method_reproduces_fit_direct_for_every_feasible_case(panel, cases, horizon):
    effects, _, results = estimate_all(panel, cases, post_horizon=horizon)
    assert len(results) == len(effects) >= 14
    for row in cases.to_dict("records"):
        if row["case_id"] not in results:
            continue
        donors = donor_pool(panel, row, cases)
        got = estimate_case(panel, row, donors, post_horizon=horizon)
        wide, y1, Y0, pre, valid = hand_arrays(panel, pd.Series(row), donors, horizon)
        fit = scm.fit_direct(y1[pre], Y0[pre])
        np.testing.assert_allclose(got.weights.to_numpy(), fit.w, atol=1e-12, rtol=0)
        assert got.weights.index.tolist() == donors
        synthetic = Y0 @ fit.w
        gap = y1 - synthetic
        post = ~pre
        assert got.n_pre == pre.sum() and got.n_post == post.sum() and got.n_donors == len(donors)
        assert got.window_end == window_end_of(row, horizon) == wide.index[-1]
        assert got.post_horizon_used == got.window_end - row["opening_year"] + 1
        np.testing.assert_allclose(got.synthetic[valid], synthetic, atol=1e-12)
        np.testing.assert_allclose(got.gap[valid], gap, atol=1e-12)
        assert got.att_pp == pytest.approx(gap[post].mean(), abs=1e-12)
        assert got.att_rel_pct == pytest.approx(100.0 * gap[post].mean() / synthetic[post].mean(), abs=1e-10)
        assert got.rmspe_pre == pytest.approx(fit.rmspe, abs=1e-12)
        assert got.rmspe_post == pytest.approx(np.sqrt(np.mean(gap[post] ** 2)), abs=1e-12)
        if got.fit_exact:
            assert np.isnan(got.rmspe_ratio)
        else:
            assert got.rmspe_ratio == pytest.approx(got.rmspe_post / got.rmspe_pre, rel=1e-9)
        assert got.method == "direct" and got.outcome == "receipts_pct_gdp"
        # the batch driver returns the same numbers
        assert results[row["case_id"]].att_pp == got.att_pp


def test_weights_are_on_the_simplex_and_arrays_cover_the_window(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    effect = estimate_case(panel, mid, donors)
    assert isinstance(effect, CaseEffect)
    assert (effect.weights >= -1e-12).all() and effect.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert effect.weights.name == "weight" and effect.weights.index.name == "donor_iso3"
    end = int(mid["opening_year"]) + POST_HORIZON - 1
    assert FIRST < end < LAST and effect.window_end == end
    assert effect.years.tolist() == list(range(FIRST, end + 1))
    for name in ("actual", "synthetic", "gap"):
        assert getattr(effect, name).shape == (end - FIRST + 1,)
    np.testing.assert_allclose(effect.gap, effect.actual - effect.synthetic, atol=1e-14)
    post = effect.years >= mid["opening_year"]
    assert post.sum() == POST_HORIZON and effect.att_pp == pytest.approx(effect.gap[post].mean())
    assert (effect.case_id, effect.iso3, effect.opening_year) == (mid["case_id"], mid["iso3"], int(mid["opening_year"]))
    assert effect.placebo_att_pp.shape == effect.placebo_att_rel_pct.shape == effect.placebo_rmspe_pre.shape == (len(donors),)
    assert effect.placebo_gaps.shape == (end - FIRST + 1, len(donors))


def test_treated_years_without_data_are_skipped_in_the_fit_and_the_effect(panel, cases):
    feas = check_feasibility(cases, panel)
    partial = feas.loc[feas["feasible"] & (feas["n_pre"] < feas["opening_year"] - FIRST)]
    assert len(partial) == 1
    row = cases.loc[cases["case_id"] == partial.iloc[0]["case_id"]].iloc[0]
    donors = donor_pool(panel, row, cases)
    effect = estimate_case(panel, row, donors)
    assert effect.n_pre == int(row["opening_year"]) - FIRST - 2
    assert np.isnan(effect.actual[:2]).all() and np.isnan(effect.gap[:2]).all() and np.isfinite(effect.synthetic).all()
    assert np.isnan(effect.placebo_gaps[:2]).all() and np.isfinite(effect.placebo_gaps[2:]).all()
    wide, y1, Y0, pre, valid = hand_arrays(panel, row, donors)
    assert valid.sum() == len(wide) - 2
    np.testing.assert_allclose(effect.weights.to_numpy(), scm.fit_direct(y1[pre], Y0[pre]).w, atol=1e-12)


# ----------------------------------------------------------------------------
# Post-opening window
# ----------------------------------------------------------------------------
def test_default_window_runs_five_years_from_the_opening_year(fitted):
    effects, _, results = fitted
    assert POST_HORIZON == 5
    assert (effects["window_end"] == np.minimum(effects["opening_year"] + 4, LAST)).all()
    assert (effects["post_horizon_used"] == effects["window_end"] - effects["opening_year"] + 1).all()
    assert (effects["n_post"] == effects["post_horizon_used"]).all()
    assert effects["post_horizon_used"].max() == 5 and (effects["post_horizon_used"] <= 5).all()
    for row in effects.itertuples():
        effect = results[row.case_id]
        assert effect.window_end == row.window_end and effect.post_horizon_used == row.post_horizon_used
        assert effect.years[-1] == row.window_end and effect.n_post == row.n_post


def test_post_years_without_an_outcome_shorten_n_post_but_not_the_window(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    opening = int(mid["opening_year"])
    holed = panel.copy()
    holed.loc[(holed["iso3"] == mid["iso3"]) & (holed["year"] == opening + 2), "receipts_pct_gdp"] = np.nan
    effect = estimate_case(holed, mid, donors)
    assert (effect.n_post, effect.post_horizon_used, effect.window_end) == (POST_HORIZON - 1, POST_HORIZON, opening + POST_HORIZON - 1)
    assert np.isnan(effect.actual[effect.years == opening + 2]).all() and np.isfinite(effect.synthetic).all()
    wide, y1, Y0, pre, valid = hand_arrays(holed, mid, donors)
    gap = y1 - Y0 @ scm.fit_direct(y1[pre], Y0[pre]).w
    assert (~pre).sum() == POST_HORIZON - 1 and effect.att_pp == pytest.approx(gap[~pre].mean(), abs=1e-12)
    row = check_feasibility(pd.DataFrame([mid]), holed, min_donors=1).iloc[0]
    assert (row["n_post"], row["window_end"]) == (POST_HORIZON - 1, opening + POST_HORIZON - 1)
    effects, _, _ = estimate_all(holed, pd.DataFrame([mid]), min_donors=1)
    assert effects.loc[0, ["n_post", "post_horizon_used"]].tolist() == [POST_HORIZON - 1, POST_HORIZON]


def test_window_is_cut_at_the_last_year_when_the_horizon_runs_past_it(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    late = {**dict(mid), "opening_year": LAST - 1}
    effect = estimate_case(panel, late, donors)
    assert (effect.window_end, effect.n_post, effect.post_horizon_used) == (LAST, 2, 2)
    assert effect.years[-1] == LAST
    wide, y1, Y0, pre, valid = hand_arrays(panel, late, donors)
    gap = y1 - Y0 @ scm.fit_direct(y1[pre], Y0[pre]).w
    assert effect.att_pp == pytest.approx(gap[~pre].mean(), abs=1e-12)
    opening = int(mid["opening_year"])
    shorter = estimate_case(panel, mid, donors, last_year=opening + 1)
    assert (shorter.window_end, shorter.n_post, shorter.post_horizon_used) == (opening + 1, 2, 2)
    assert shorter.years.tolist() == list(range(FIRST, opening + 2))


def test_horizon_none_removes_the_cap_and_the_window_runs_to_the_last_year(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    opening = int(mid["opening_year"])
    unlimited = estimate_case(panel, mid, donors, post_horizon=None)
    assert unlimited.window_end == LAST and unlimited.post_horizon_used == unlimited.n_post == LAST - opening + 1
    assert unlimited.years.tolist() == list(range(FIRST, LAST + 1))
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors, None)
    gap = y1 - Y0 @ scm.fit_direct(y1[pre], Y0[pre]).w
    assert unlimited.att_pp == pytest.approx(gap[~pre].mean(), abs=1e-12)
    assert unlimited.rmspe_post == pytest.approx(np.sqrt(np.mean(gap[~pre] ** 2)), abs=1e-12)
    for horizon in (LAST - opening + 1, LAST - opening + 7, 10_000):
        assert_same_effect(estimate_case(panel, mid, donors, post_horizon=horizon), unlimited)
    default = estimate_case(panel, mid, donors)
    assert default.n_post == POST_HORIZON < unlimited.n_post and default.att_pp != unlimited.att_pp


def test_horizon_accepts_numpy_integers_and_whole_floats(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:12]
    reference = estimate_case(panel, mid, donors, post_horizon=3)
    assert reference.n_post == 3
    for horizon in (np.int64(3), 3.0):
        assert_same_effect(estimate_case(panel, mid, donors, post_horizon=horizon), reference)


@pytest.mark.parametrize("bad", [0, -3, 2.5, "5", True])
def test_invalid_post_horizon_is_rejected_by_every_entry_point(panel, cases, mid, bad):
    donors = donor_pool(panel, mid, cases)[:12]
    with pytest.raises(ValueError, match="post_horizon"):
        estimate_case(panel, mid, donors, post_horizon=bad)
    with pytest.raises(ValueError, match="post_horizon"):
        check_feasibility(cases, panel, post_horizon=bad)
    with pytest.raises(ValueError, match="post_horizon"):
        estimate_all(panel, cases, post_horizon=bad)


def test_next_start_year_caps_the_window_before_the_horizon_does(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    opening = int(mid["opening_year"])
    for follow, end in [(opening + 2, opening + 1), (opening + 3, opening + 2), (opening + 5, opening + 4), (opening + 9, opening + 4)]:
        case = {**dict(mid), "next_start_year": float(follow)}
        effect = estimate_case(panel, case, donors)
        assert window_end_of(case) == end == effect.window_end
        assert effect.n_post == effect.post_horizon_used == end - opening + 1
        assert effect.years[-1] == end and effect.placebo_gaps.shape[0] == end - FIRST + 1
        wide, y1, Y0, pre, valid = hand_arrays(panel, case, donors)
        fit = scm.fit_direct(y1[pre], Y0[pre])
        gap = y1 - Y0 @ fit.w
        np.testing.assert_allclose(effect.weights.to_numpy(), fit.w, atol=1e-12)
        assert effect.att_pp == pytest.approx(gap[~pre].mean(), abs=1e-12)
        assert effect.rmspe_post == pytest.approx(np.sqrt(np.mean(gap[~pre] ** 2)), abs=1e-12)
    one_year = estimate_case(panel, {**dict(mid), "next_start_year": opening + 1}, donors)
    assert (one_year.n_post, one_year.window_end) == (1, opening) and 0.0 < one_year.p_gap_rank <= 1.0


def test_next_start_year_still_caps_the_window_without_a_horizon(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:15]
    opening = int(mid["opening_year"])
    capped = estimate_case(panel, {**dict(mid), "next_start_year": opening + 7}, donors, post_horizon=None)
    assert capped.window_end == opening + 6 and capped.n_post == capped.post_horizon_used == 7
    assert_same_effect(capped, estimate_case(panel, {**dict(mid), "next_start_year": opening + 7}, donors, post_horizon=7))


def test_missing_next_start_year_leaves_the_window_alone(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:12]
    reference = estimate_case(panel, mid, donors)
    for missing in (None, np.nan, pd.NA):
        assert_same_effect(estimate_case(panel, {**dict(mid), "next_start_year": missing}, donors), reference)
    row = mid.copy()
    row["next_start_year"] = np.nan
    assert_same_effect(estimate_case(panel, row, donors), reference)


def test_next_start_year_must_be_after_the_opening(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:12]
    opening = int(mid["opening_year"])
    for follow in (opening, opening - 2):
        with pytest.raises(ValueError, match="next_start_year"):
            estimate_case(panel, {**dict(mid), "next_start_year": follow}, donors)


@pytest.mark.parametrize(
    "horizon,follow_offset",
    [(POST_HORIZON, None), (2, None), (None, None), (POST_HORIZON, 3), (None, 4)],
)
def test_placebo_statistics_use_the_window_of_the_case(panel, cases, mid, horizon, follow_offset):
    donors = donor_pool(panel, mid, cases)[:15]
    case = dict(mid)
    if follow_offset is not None:
        case["next_start_year"] = int(mid["opening_year"]) + follow_offset
    effect = estimate_case(panel, case, donors, post_horizon=horizon, placebo_fit_factor=None)
    wide, y1, Y0, pre, valid = hand_arrays(panel, case, donors, horizon)
    assert effect.window_end == wide.index[-1] and effect.n_post == (~pre).sum()
    Y = np.column_stack([y1, Y0])
    ref = scm.placebo_in_space(Y, treated=0, pre=pre, donors=list(range(1, len(donors) + 1)), method="direct")
    np.testing.assert_allclose(effect.placebo_att_pp, ref.table["mean_post_gap"].to_numpy()[1:], atol=1e-8)
    np.testing.assert_allclose(effect.placebo_rmspe_pre, ref.table["pre_rmspe"].to_numpy()[1:], atol=1e-8)
    assert effect.p_gap_rank == pytest.approx(ref.p_value_abs_mean_gap)
    assert ratio_ranks_are_defined(ref.table, effect.fit_tolerance) and effect.p_ratio_rank == pytest.approx(ref.p_value_ratio)
    assert effect.placebo_gaps.shape == (wide.shape[0], len(donors))
    for k in (0, 7, len(donors) - 1):
        pool = [j for j in range(len(donors)) if j != k]
        w = scm.fit_direct(Y0[pre][:, k], Y0[pre][:, pool]).w
        synth = Y0[:, pool] @ w
        np.testing.assert_allclose(effect.placebo_gaps[valid, k], Y0[:, k] - synth, atol=1e-8)
        assert effect.placebo_att_rel_pct[k] == pytest.approx(100.0 * (Y0[:, k] - synth)[~pre].mean() / synth[~pre].mean(), abs=1e-6)


def test_nothing_after_the_window_enters_any_statistic(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    base = estimate_case(panel, mid, donors)
    late = panel["year"] > base.window_end
    assert late.any()
    noisy = panel.copy()
    noisy.loc[late, "receipts_pct_gdp"] += np.random.default_rng(11).normal(0.0, 25.0, size=int(late.sum()))
    assert_same_effect(estimate_case(noisy, mid, donors), base)
    holes = panel.copy()
    holes.loc[late & holes["iso3"].isin(donors), "receipts_pct_gdp"] = np.nan
    assert_same_effect(estimate_case(holes, mid, donors), base)


def test_donor_gaps_inside_the_window_are_rejected_and_gaps_after_it_are_not(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    end = window_end_of(mid)
    inside = panel.copy()
    inside.loc[(inside["iso3"] == donors[3]) & (inside["year"] == end), "receipts_pct_gdp"] = np.nan
    with pytest.raises(ValueError, match=f"missing outcome years in the window up to {end}"):
        estimate_case(inside, mid, donors)
    after = panel.copy()
    after.loc[(after["iso3"] == donors[3]) & (after["year"] == end + 1), "receipts_pct_gdp"] = np.nan
    assert_same_effect(estimate_case(after, mid, donors), estimate_case(panel, mid, donors))
    with pytest.raises(ValueError, match="missing outcome years"):
        estimate_case(after, mid, donors, post_horizon=None)


# ----------------------------------------------------------------------------
# Recovery of the simulated truth
# ----------------------------------------------------------------------------
def test_recovered_effects_correlate_with_the_true_effects(world, fitted):
    effects = fitted[0].merge(world.true_effects, on="case_id", validate="one_to_one")
    assert len(effects) >= 14
    err = effects["att_pp"] - effects["true_att_pp"]
    assert np.corrcoef(effects["att_pp"], effects["true_att_pp"])[0, 1] > 0.8
    assert abs(err.mean()) < 0.5 * effects["placebo_sd_pp"].mean()
    assert err.abs().mean() < effects["placebo_sd_pp"].mean()
    slope = np.polyfit(effects["true_att_pp"], effects["att_pp"], 1)[0]
    assert 0.8 < slope < 1.2
    assert np.corrcoef(effects["att_rel_pct"], effects["true_att_rel_pct"])[0, 1] > 0.8


def test_estimated_effects_recover_the_generating_coefficient_of_capex(world, panel, cases, fitted):
    feats = case_features(cases, panel).set_index("case_id")
    effects = fitted[0].set_index("case_id")
    x = feats.loc[effects.index, "capex_pct_gdp"].to_numpy()
    slope, intercept = np.polyfit(x, effects["att_pp"].to_numpy(), 1)
    assert slope == pytest.approx(world.coefficients["b1"], abs=0.25)
    assert intercept == pytest.approx(world.coefficients["b0"], abs=0.35)
    others = [c for c in FEATURE_COLUMNS if c not in ("capex_pct_gdp", "is_integrated_resort")]
    corr = feats.loc[effects.index, others].corrwith(effects["att_pp"]).abs()
    assert abs(np.corrcoef(x, effects["att_pp"])[0, 1]) > corr.max() + 0.2


def test_no_effect_is_found_in_economies_that_never_opened_anything(panel, cases, fitted):
    rng = np.random.default_rng(5)
    never = sorted(set(panel["iso3"]) - set(cases["iso3"]))
    wide = panel.loc[panel["year"].between(FIRST, LAST)].pivot(index="year", columns="iso3", values="receipts_pct_gdp")
    complete = [c for c in never if wide[c].notna().all()]
    chosen = list(rng.choice(complete, size=10, replace=False))
    rejections, sizes = 0, []
    for iso3 in chosen:
        pool = [c for c in complete if c != iso3]
        fake = {"case_id": f"fake_{iso3}", "iso3": iso3, "opening_year": int(rng.integers(2002, 2012))}
        effect = estimate_case(panel, fake, pool)
        rejections += int(effect.p_gap_rank < 0.05)
        sizes.append(abs(effect.att_pp))
    assert rejections <= 2
    assert np.mean(sizes) < 0.5 * fitted[0]["att_pp"].abs().mean()


# ----------------------------------------------------------------------------
# Placebo inference
# ----------------------------------------------------------------------------
def test_placebo_p_values_lie_in_the_open_unit_interval_and_are_ranks_over_the_placebos_they_use(fitted):
    effects = fitted[0]
    defined = effects.loc[~effects["fit_exact"]]
    assert len(defined) >= 10 and effects["p_ratio_rank"].isna().equals(effects["fit_exact"])
    for column, n_units in (
        ("p_gap_rank", effects["n_placebos_comparable"] + 1),
        ("p_ratio_rank", effects["n_donors"] - effects["n_placebos_exact"] + 1),
    ):
        p = effects[column].to_numpy()
        n_units = n_units.to_numpy()
        use = ~np.isnan(p)
        assert ((p[use] > 0.0) & (p[use] <= 1.0)).all()
        ranks = p[use] * n_units[use]
        np.testing.assert_allclose(ranks, np.round(ranks), atol=1e-9)
        assert (np.round(ranks) >= 1).all() and (np.round(ranks) <= n_units[use]).all()
    assert (effects["p_gap_rank"] < 0.2).sum() >= 8


def test_placebo_distribution_agrees_with_dtt_scm_placebo_in_space(panel, cases, fitted):
    for row in cases.to_dict("records"):
        if row["case_id"] not in fitted[2]:
            continue
        effect = fitted[2][row["case_id"]]
        donors = effect.weights.index.tolist()
        wide, y1, Y0, pre, valid = hand_arrays(panel, pd.Series(row), donors)
        Y = np.column_stack([y1, Y0])
        ref = scm.placebo_in_space(Y, treated=0, pre=pre, donors=list(range(1, len(donors) + 1)), method="direct")
        table = ref.table
        np.testing.assert_allclose(effect.placebo_att_pp, table["mean_post_gap"].to_numpy()[1:], atol=1e-8)
        np.testing.assert_allclose(effect.placebo_rmspe_pre, table["pre_rmspe"].to_numpy()[1:], atol=1e-8)
        comparable = comparable_by_hand(table["pre_rmspe"].to_numpy()[1:])
        np.testing.assert_array_equal(effect.placebo_comparable, comparable)
        assert effect.p_gap_rank == pytest.approx(rank_p(abs(effect.att_pp), table["abs_mean_gap"].to_numpy()[1:][comparable]))
        if ratio_ranks_are_defined(table, effect.fit_tolerance):
            assert effect.p_ratio_rank == pytest.approx(ref.p_value_ratio)
            assert 0.0 < effect.p_ratio_rank <= 1.0
        assert effect.att_pp == pytest.approx(table["mean_post_gap"].iloc[0], abs=1e-8)


def test_placebo_relative_effects_use_each_donors_synthetic_post_level(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:12]
    effect = estimate_case(panel, mid, donors)
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors)
    post = ~pre
    for k in range(len(donors)):
        pool = [j for j in range(len(donors)) if j != k]
        w = scm.fit_direct(Y0[pre][:, k], Y0[pre][:, pool]).w
        synth = Y0[:, pool] @ w
        gap = Y0[:, k] - synth
        assert effect.placebo_att_pp[k] == pytest.approx(gap[post].mean(), abs=1e-8)
        assert effect.placebo_att_rel_pct[k] == pytest.approx(100.0 * gap[post].mean() / synth[post].mean(), abs=1e-6)
        assert effect.placebo_rmspe_pre[k] == pytest.approx(np.sqrt(np.mean(gap[pre] ** 2)), abs=1e-8)


def test_a_planted_effect_is_recovered_exactly_when_the_pre_period_is_matched_exactly(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:4]
    opening = int(mid["opening_year"])
    weights = np.array([0.5, 0.3, 0.2, 0.0])
    wide = panel.loc[panel["year"].between(FIRST, LAST)].pivot(index="year", columns="iso3", values="receipts_pct_gdp")
    planted = pd.Series(wide[donors].to_numpy() @ weights + np.where(wide.index >= opening, 1.5, 0.0), index=wide.index)
    altered = panel.copy()
    own = altered["iso3"] == mid["iso3"]
    altered.loc[own, "receipts_pct_gdp"] = altered.loc[own, "year"].map(planted).to_numpy()
    effect = estimate_case(altered, mid, donors)
    np.testing.assert_allclose(effect.weights.to_numpy(), weights, atol=1e-6)
    assert effect.rmspe_pre < 1e-6 and effect.n_post == POST_HORIZON
    assert effect.att_pp == pytest.approx(1.5, abs=1e-6) and effect.rmspe_post == pytest.approx(1.5, abs=1e-6)
    np.testing.assert_allclose(effect.gap[effect.years >= opening], 1.5, atol=1e-6)
    assert effect.fit_exact and np.isnan(effect.rmspe_ratio) and np.isnan(effect.p_ratio_rank)
    assert 1 / (1 + effect.n_placebos_comparable) <= effect.p_gap_rank <= 1.0
    assert abs(effect.did_att_pp - 1.5) > 1e-3


# ----------------------------------------------------------------------------
# Other weight methods
# ----------------------------------------------------------------------------
def test_ridge_method_augments_the_direct_weights(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors)
    base = scm.fit_direct(y1[pre], Y0[pre]).w
    effect = estimate_case(panel, mid, donors, method="ridge", ridge=0.5)
    expected = scm.ridge_augment_weights(y1[pre], Y0[pre], base, 0.5)
    np.testing.assert_allclose(effect.weights.to_numpy(), expected, atol=1e-12)
    assert effect.weights.sum() == pytest.approx(1.0, abs=1e-9) and (effect.weights < -1e-9).any()
    assert effect.method == "ridge"
    gap = y1 - Y0 @ expected
    assert effect.att_pp == pytest.approx(gap[~pre].mean(), abs=1e-12)
    assert effect.rmspe_pre <= scm.fit_direct(y1[pre], Y0[pre]).rmspe + 1e-9
    heavy = estimate_case(panel, mid, donors, method="ridge", ridge=1e9)
    np.testing.assert_allclose(heavy.weights.to_numpy(), base, atol=1e-6)
    assert np.all(np.isfinite(effect.placebo_att_pp)) and 0.0 < effect.p_gap_rank <= 1.0


def test_ridge_placebos_use_the_same_augmentation(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:10]
    effect = estimate_case(panel, mid, donors, method="ridge", ridge=0.7)
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors)
    Ypre = Y0[pre]
    for k in (0, 4, 9):
        pool = [j for j in range(len(donors)) if j != k]
        w = scm.ridge_augment_weights(Ypre[:, k], Ypre[:, pool], scm.fit_direct(Ypre[:, k], Ypre[:, pool]).w, 0.7)
        gap = Y0[:, k] - Y0[:, pool] @ w
        assert effect.placebo_att_pp[k] == pytest.approx(gap[~pre].mean(), abs=1e-8)


def test_nested_method_matches_fit_nested_and_placebo_in_space(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:8]
    effect = estimate_case(panel, mid, donors, method="nested", seed=3, placebo_fit_factor=None)
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors)
    fit = scm.fit_nested(y1[pre], Y0[pre], y1[pre], Y0[pre], n_starts=4, seed=3)
    np.testing.assert_allclose(effect.weights.to_numpy(), fit.w, atol=1e-12)
    assert effect.method == "nested" and effect.rmspe_pre == pytest.approx(fit.rmspe, abs=1e-12)
    Y = np.column_stack([y1, Y0])
    ref = scm.placebo_in_space(Y, treated=0, pre=pre, donors=list(range(1, 9)), method="nested", X=Y[pre], n_starts=4, seed=3)
    np.testing.assert_allclose(effect.placebo_att_pp, ref.table["mean_post_gap"].to_numpy()[1:], atol=1e-9)
    assert effect.p_gap_rank == pytest.approx(ref.p_value_abs_mean_gap)
    assert ratio_ranks_are_defined(ref.table, effect.fit_tolerance) and effect.p_ratio_rank == pytest.approx(ref.p_value_ratio)
    direct = estimate_case(panel, mid, donors)
    assert effect.rmspe_pre <= direct.rmspe_pre + 1e-6


# ----------------------------------------------------------------------------
# Difference in differences
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [POST_HORIZON, None])
def test_did_estimate_is_the_double_difference_and_equals_two_way_fixed_effects(panel, cases, mid, horizon):
    donors = donor_pool(panel, mid, cases)
    effect = estimate_case(panel, mid, donors, post_horizon=horizon)
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors, horizon)
    mean_donor = Y0.mean(axis=1)
    by_hand = (y1[~pre].mean() - y1[pre].mean()) - (mean_donor[~pre].mean() - mean_donor[pre].mean())
    assert effect.did_att_pp == pytest.approx(by_hand, abs=1e-12)

    years = wide.index.to_numpy()[valid]
    frames = [pd.DataFrame({"unit_id": 0, "year": years, "y": y1, "treated_post": (~pre).astype(float)})]
    for j in range(len(donors)):
        frames.append(pd.DataFrame({"unit_id": j + 1, "year": years, "y": Y0[:, j], "treated_post": 0.0}))
    long = pd.concat(frames, ignore_index=True)
    assert friend_did(long, y="y").coef == pytest.approx(effect.did_att_pp, abs=1e-9)
    assert xtreg_twfe(long, y="y").coef("treated_post") == pytest.approx(effect.did_att_pp, abs=1e-9)


# ----------------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------------
def test_estimate_all_calls_the_guard_once_per_case_and_once_every_ten_donor_fits(panel, cases):
    calls = []
    effects, _, results = estimate_all(panel, cases, guard=lambda: calls.append(1))
    expected = len(effects) + int((effects["n_donors"] // GUARD_INTERVAL).sum())
    assert GUARD_INTERVAL == 10
    assert len(calls) == expected
    assert (effects["n_donors"] >= 20).all() and expected > len(effects)
    assert set(results) == set(effects["case_id"])


@pytest.mark.parametrize("n_donors,expected", [(2, 0), (9, 0), (10, 1), (19, 1), (20, 2), (25, 2)])
def test_estimate_case_calls_the_guard_after_every_tenth_donor_fit(panel, cases, mid, n_donors, expected):
    calls = []
    estimate_case(panel, mid, donor_pool(panel, mid, cases)[:n_donors], guard=lambda: calls.append(1))
    assert len(calls) == expected


def test_guard_is_called_before_the_first_case_and_may_interrupt_the_run(panel, cases):
    class Stop(Exception):
        pass

    state = {"n": 0}

    def guard():
        state["n"] += 1
        if state["n"] == 1:
            raise Stop

    with pytest.raises(Stop):
        estimate_all(panel, cases, guard=guard)
    assert state["n"] == 1


def test_guard_does_not_change_the_results(panel, cases, fitted):
    effects, placebos, _ = estimate_all(panel, cases, guard=lambda: None)
    pd.testing.assert_frame_equal(effects, fitted[0])
    pd.testing.assert_frame_equal(placebos, fitted[1])


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def test_effects_table_layout_and_definitions(panel, cases, fitted):
    effects, placebos, results = fitted
    assert list(effects.columns) == EFFECT_COLUMNS
    feas = check_feasibility(cases, panel)
    assert effects["case_id"].tolist() == feas.loc[feas["feasible"], "case_id"].tolist()
    assert effects["case_id"].is_unique and set(effects["method"]) == {"direct"}
    for column in ("opening_year", "n_pre", "n_post", "window_end", "post_horizon_used", "n_donors"):
        assert effects[column].dtype == np.int64
    assert effects["pre_fit_ok"].dtype == bool
    merged = effects.merge(feas, on="case_id", suffixes=("", "_f"))
    for column in ("n_pre", "n_post", "window_end", "n_donors"):
        assert (merged[column] == merged[f"{column}_f"]).all()
    assert (effects["post_horizon_used"] == effects["window_end"] - effects["opening_year"] + 1).all()
    for row in effects.itertuples():
        effect = results[row.case_id]
        assert row.att_pp == effect.att_pp and row.did_att_pp == effect.did_att_pp
        np.testing.assert_equal(row.rmspe_ratio, effect.rmspe_ratio)
        np.testing.assert_equal(row.p_ratio_rank, effect.p_ratio_rank)
        comparable = effect.placebo_comparable
        assert row.placebo_sd_pp == effect.placebo_sd_pp == pytest.approx(np.std(effect.placebo_att_pp[comparable], ddof=1))
        assert row.placebo_sd_rel_pct == effect.placebo_sd_rel_pct == pytest.approx(np.std(effect.placebo_att_rel_pct[comparable], ddof=1))
        assert (row.n_placebos, row.n_placebos_comparable) == (row.n_donors, int(comparable.sum()))
        assert row.n_placebos_exact == int((effect.placebo_rmspe_pre <= row.fit_tolerance).sum())
        assert row.fit_tolerance == effect.fit_tolerance
        within_factor = effect.rmspe_pre <= 2.0 * np.median(effect.placebo_rmspe_pre)
        assert row.pre_fit_ok == (within_factor and not (row.fit_exact and row.n_pre < row.n_donors))
    assert effects["pre_fit_ok"].any()
    for column in ("n_placebos", "n_placebos_comparable", "n_placebos_exact"):
        assert effects[column].dtype == np.int64
    assert (effects["n_placebos_comparable"] >= np.minimum(effects["n_donors"], 5)).all()
    assert (effects["n_placebos_comparable"] < effects["n_placebos"]).any()


def test_economy_group_repeats_the_economy_code_for_clustering(fitted):
    effects = fitted[0]
    assert effects["economy_group"].tolist() == effects["iso3"].tolist()
    assert effects.columns.get_loc("economy_group") == effects.columns.get_loc("iso3") + 1


def test_pre_fit_flag_marks_a_case_with_a_poor_synthetic_control(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    wide, y1, Y0, pre, valid = hand_arrays(panel, mid, donors)
    outlier = dict(mid)
    stranger = panel.copy()
    stranger.loc[stranger["iso3"] == mid["iso3"], "receipts_pct_gdp"] += 40.0 + 3.0 * np.sin(np.arange((stranger["iso3"] == mid["iso3"]).sum()))
    effects, _, results = estimate_all(stranger, pd.DataFrame([outlier]), feasibility=pd.DataFrame({"case_id": [mid["case_id"]], "feasible": [True]}))
    assert not effects.loc[0, "pre_fit_ok"]
    assert results[mid["case_id"]].rmspe_pre > 2.0 * np.median(results[mid["case_id"]].placebo_rmspe_pre)


def test_placebo_table_has_one_row_per_case_and_donor(fitted):
    effects, placebos, results = fitted
    assert list(placebos.columns) == PLACEBO_COLUMNS
    assert len(placebos) == int(effects["n_donors"].sum())
    for case_id, effect in results.items():
        sub = placebos.loc[placebos["case_id"] == case_id]
        assert sub["donor_iso3"].tolist() == effect.weights.index.tolist()
        np.testing.assert_allclose(sub["placebo_att_pp"], effect.placebo_att_pp)
        np.testing.assert_allclose(sub["placebo_att_rel_pct"], effect.placebo_att_rel_pct)
        np.testing.assert_allclose(sub["placebo_rmspe_pre"], effect.placebo_rmspe_pre)
        assert sub["comparable"].tolist() == effect.placebo_comparable.tolist()
    assert placebos[["placebo_att_pp", "placebo_rmspe_pre"]].notna().all().all()
    assert placebos["comparable"].dtype == bool and placebos["comparable"].any() and not placebos["comparable"].all()


def test_estimates_are_deterministic_and_leave_the_inputs_unchanged(panel, cases, fitted):
    panel_copy, cases_copy = panel.copy(), cases.copy()
    again = estimate_all(panel, cases)
    pd.testing.assert_frame_equal(again[0], fitted[0])
    pd.testing.assert_frame_equal(again[1], fitted[1])
    pd.testing.assert_frame_equal(panel, panel_copy)
    pd.testing.assert_frame_equal(cases, cases_copy)


# ----------------------------------------------------------------------------
# estimate_all options
# ----------------------------------------------------------------------------
def test_feasibility_argument_selects_the_cases(panel, cases, fitted):
    feas = check_feasibility(cases, panel)
    keep = fitted[0]["case_id"].tolist()[:3]
    narrowed = feas.assign(feasible=feas["case_id"].isin(keep))
    effects, placebos, results = estimate_all(panel, cases, feasibility=narrowed)
    assert effects["case_id"].tolist() == keep and set(results) == set(keep) and set(placebos["case_id"]) == set(keep)
    pd.testing.assert_frame_equal(effects, fitted[0].iloc[:3].reset_index(drop=True))
    with pytest.raises(ValueError, match="feasible"):
        estimate_all(panel, cases, feasibility=feas.drop(columns=["feasible"]))


def test_keyword_arguments_reach_the_feasibility_check_and_the_estimates(panel, cases, fitted):
    cut = int(np.median(fitted[0]["n_pre"]))
    effects, _, _ = estimate_all(panel, cases, min_pre=cut)
    assert 0 < len(effects) and set(effects["case_id"]) < set(fitted[0]["case_id"])
    assert (effects["n_pre"] >= cut).all()
    cut = int(fitted[0]["n_donors"].max())
    only, _, _ = estimate_all(panel, cases, min_donors=cut)
    assert 0 < len(only) < len(fitted[0]) and (only["n_donors"] >= cut).all()
    shifted, _, _ = estimate_all(panel, cases, first_year=2000, last_year=2015, min_pre=5, min_post=2, min_donors=15)
    assert (shifted["n_pre"] + shifted["n_post"] <= 16).all() and (shifted["window_end"] <= 2015).all()
    ridge, _, res = estimate_all(panel, cases.iloc[1:4], method="ridge", ridge=1.0)
    assert set(ridge["method"]) == {"ridge"} and all(r.method == "ridge" for r in res.values())
    with pytest.raises(TypeError, match="unexpected keyword"):
        estimate_all(panel, cases, bogus=1)


def test_estimate_all_passes_the_horizon_and_the_minimum_post_years_on(panel, cases, fitted):
    short, _, results = estimate_all(panel, cases, post_horizon=3)
    assert set(short["case_id"]) == set(fitted[0]["case_id"])
    assert (short["window_end"] == np.minimum(short["opening_year"] + 2, LAST)).all()
    assert (short["n_post"] == short["post_horizon_used"]).all() and (short["post_horizon_used"] <= 3).all()
    for row in short.itertuples():
        single = estimate_case(panel, cases.loc[cases["case_id"] == row.case_id].iloc[0], results[row.case_id].weights.index.tolist(), post_horizon=3)
        assert single.att_pp == row.att_pp and single.window_end == row.window_end
    assert estimate_all(panel, cases, post_horizon=3, min_post=4)[0].empty
    assert estimate_all(panel, cases, post_horizon=2)[0].empty
    two, _, _ = estimate_all(panel, cases, post_horizon=2, min_post=2)
    assert set(two["case_id"]) == set(fitted[0]["case_id"]) and (two["n_post"] == 2).all()
    unlimited, _, _ = estimate_all(panel, cases, post_horizon=None)
    assert (unlimited["window_end"] == LAST).all()
    assert (unlimited["post_horizon_used"] == LAST - unlimited["opening_year"] + 1).all()
    strict, _, _ = estimate_all(panel, cases, min_post=POST_HORIZON)
    assert set(strict["case_id"]) <= set(fitted[0]["case_id"]) and (strict["n_post"] == POST_HORIZON).all()


def test_catalogue_argument_keeps_the_donor_exclusions_of_the_full_catalogue(panel, cases, fitted):
    subset = cases.iloc[2:8]
    alone, _, _ = estimate_all(panel, subset)
    kept, placebos, results = estimate_all(panel, subset, catalogue=cases)
    expected = fitted[0].loc[fitted[0]["case_id"].isin(subset["case_id"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, expected)
    assert alone["n_donors"].sum() > kept["n_donors"].sum()
    for case_id, effect in results.items():
        assert effect.weights.index.tolist() == donor_pool(panel, case_id, cases)


def test_estimate_all_validates_the_catalogue(panel, cases):
    with pytest.raises(ValueError, match="unique"):
        estimate_all(panel, pd.concat([cases, cases.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="opening_year"):
        estimate_all(panel, cases.drop(columns=["opening_year"]))


def test_empty_result_when_no_case_is_feasible(panel, cases):
    effects, placebos, results = estimate_all(panel, cases, min_donors=10_000)
    assert effects.empty and list(effects.columns) == EFFECT_COLUMNS
    assert placebos.empty and list(placebos.columns) == PLACEBO_COLUMNS and placebos["comparable"].dtype == bool
    assert results == {}


def test_write_effects_creates_both_files(fitted, tmp_path):
    effects, placebos, _ = fitted
    effects_path, placebos_path = write_effects(effects, placebos, tmp_path / "out" / "tables")
    assert effects_path.name == "case_effects.csv" and placebos_path.name == "case_placebos.csv"
    back = pd.read_csv(effects_path)
    assert list(back.columns) == EFFECT_COLUMNS and len(back) == len(effects)
    text = ["case_id", "iso3", "economy_group", "method"]
    pd.testing.assert_frame_equal(back.drop(columns=text), effects.drop(columns=text), rtol=1e-12)
    for column in text:
        assert back[column].tolist() == effects[column].tolist()
    back_p = pd.read_csv(placebos_path)
    assert list(back_p.columns) == PLACEBO_COLUMNS and len(back_p) == len(placebos)
    np.testing.assert_allclose(back_p["placebo_att_pp"], placebos["placebo_att_pp"], rtol=1e-12)
    assert back_p["comparable"].dtype == bool and back_p["comparable"].tolist() == placebos["comparable"].tolist()


# ----------------------------------------------------------------------------
# Treatment episodes
# ----------------------------------------------------------------------------
def banned_economies(catalogue: pd.DataFrame, start: int, buffer: int = 5) -> set[str]:
    """Economies with a catalogue opening from the buffer before the start year to the last year."""
    near = catalogue["opening_year"].between(start - buffer, LAST)
    return set(catalogue.loc[near, "iso3"])


def test_each_episode_gets_one_estimate_with_the_window_from_its_start_year(cluster_world, episodes, episode_fit):
    effects, placebos, results = episode_fit
    assert list(effects.columns) == EFFECT_COLUMNS and list(placebos.columns) == PLACEBO_COLUMNS
    infeasible = set(cluster_world.infeasible_case_ids)
    estimable = [not set(members.split(";")) <= infeasible for members in episodes["member_case_ids"]]
    assert effects["case_id"].tolist() == episodes.loc[estimable, "case_id"].tolist()
    assert effects["case_id"].str.endswith("_ep").all() and set(results) == set(effects["case_id"])
    assert effects["iso3"].is_unique and (effects["economy_group"] == effects["iso3"]).all()
    assert (effects["window_end"] == np.minimum(effects["opening_year"] + 4, LAST)).all()
    assert (effects["n_post"] == effects["post_horizon_used"]).all()
    assert placebos["case_id"].isin(effects["case_id"]).all() and len(placebos) == int(effects["n_donors"].sum())


def test_episode_donors_exclude_every_opening_of_the_catalogue(cluster_cases, episodes, episode_fit):
    _, _, results = episode_fit
    sharper = 0
    for row in episodes.itertuples():
        if row.case_id not in results:
            continue
        donors = set(results[row.case_id].weights.index)
        banned = banned_economies(cluster_cases, row.opening_year) | {row.iso3}
        assert donors.isdisjoint(banned)
        sharper += len(banned_economies(episodes, row.opening_year) | {row.iso3}) < len(banned)
    assert sharper >= 1


def test_raw_cases_with_a_later_opening_of_their_economy_in_the_window_are_not_estimated(cluster_world, cluster_panel, cluster_cases):
    effects, placebos, results = estimate_all(cluster_panel, cluster_cases)
    followed = {"C05", "C09"}
    built_to_fail = set(cluster_world.infeasible_case_ids)
    assert followed.isdisjoint(built_to_fail)
    assert set(effects["case_id"]) == set(cluster_cases["case_id"]) - followed - built_to_fail
    assert set(results) == set(effects["case_id"]) and set(placebos["case_id"]) == set(effects["case_id"])
    last_members = cluster_cases.groupby("iso3")["case_id"].last()
    assert set(last_members.loc[["SA37", "SA00"]]) <= set(effects["case_id"])
    assert (effects["window_end"] == np.minimum(effects["opening_year"] + 4, LAST)).all()


def test_episode_estimates_are_the_same_with_the_catalogue_passed_explicitly(cluster_panel, cluster_cases, episodes, episode_fit):
    again = estimate_all(cluster_panel, episodes, catalogue=cluster_cases)
    pd.testing.assert_frame_equal(again[0], episode_fit[0])
    pd.testing.assert_frame_equal(again[1], episode_fit[1])


def test_extra_episode_columns_are_optional(cluster_panel, episodes, episode_fit):
    plain = episodes[["case_id", "iso3", "opening_year"]]
    effects, _, _ = estimate_all(cluster_panel, plain)
    assert effects["case_id"].tolist() == episode_fit[0]["case_id"].tolist()
    assert effects["window_end"].tolist() == episode_fit[0]["window_end"].tolist()
    marked = episodes.assign(next_start_year=np.nan)
    pd.testing.assert_frame_equal(estimate_all(cluster_panel, marked)[0], episode_fit[0])


def test_episode_effects_recover_the_level_shift_over_the_window(cluster_world, episodes, episode_fit):
    effects = episode_fit[0].copy()
    effects["truth"] = [cluster_world.window_effect(r.iso3, r.opening_year, r.window_end) for r in effects.itertuples()]
    err = effects["att_pp"] - effects["truth"]
    assert len(effects) >= 10
    assert np.corrcoef(effects["att_pp"], effects["truth"])[0, 1] > 0.8
    assert err.abs().mean() < effects["placebo_sd_pp"].mean()
    assert abs(err.mean()) < effects["placebo_sd_pp"].mean()
    shared = effects["case_id"].isin(episodes.loc[episodes["n_openings"] > 1, "case_id"])
    assert shared.sum() >= 1 and (err[shared].abs() < 3.0 * effects.loc[shared, "placebo_sd_pp"]).all()


def test_split_episodes_stop_the_window_before_the_next_episode(split_episodes, split_fit):
    effects, _, results = split_fit
    following = effects["case_id"].map(split_episodes.set_index("case_id")["next_start_year"])
    expected = [
        min(start + 4, LAST) if pd.isna(nxt) else min(start + 4, LAST, int(nxt) - 1)
        for start, nxt in zip(effects["opening_year"], following)
    ]
    assert effects["window_end"].tolist() == expected
    binding = following.notna() & (following - effects["opening_year"] < POST_HORIZON)
    assert binding.sum() >= 1
    assert (effects.loc[binding, "window_end"] == following[binding] - 1).all()
    assert (effects.loc[binding, "post_horizon_used"] < POST_HORIZON).all()
    for row in effects.itertuples():
        effect = results[row.case_id]
        assert effect.placebo_gaps.shape[0] == row.window_end - FIRST + 1 and effect.years[-1] == row.window_end
        assert effect.n_post == row.n_post == row.post_horizon_used


def test_split_episode_statistics_match_a_hand_computation_on_the_capped_window(cluster_panel, split_episodes, split_fit):
    effects, _, results = split_fit
    capped = [r for r in split_episodes.itertuples() if r.case_id in results and not pd.isna(r.next_start_year) and r.next_start_year - r.opening_year < POST_HORIZON]
    assert capped
    for row in capped:
        series = split_episodes.loc[split_episodes["case_id"] == row.case_id].iloc[0]
        effect = results[row.case_id]
        donors = effect.weights.index.tolist()
        wide, y1, Y0, pre, valid = hand_arrays(cluster_panel, series, donors)
        fit = scm.fit_direct(y1[pre], Y0[pre])
        gap = y1 - Y0 @ fit.w
        assert wide.index[-1] == row.next_start_year - 1 == effect.window_end
        assert effect.att_pp == pytest.approx(gap[~pre].mean(), abs=1e-12)
        ref = scm.placebo_in_space(np.column_stack([y1, Y0]), treated=0, pre=pre, donors=list(range(1, len(donors) + 1)), method="direct")
        np.testing.assert_allclose(effect.placebo_att_pp, ref.table["mean_post_gap"].to_numpy()[1:], atol=1e-8)
        comparable = comparable_by_hand(ref.table["pre_rmspe"].to_numpy()[1:])
        np.testing.assert_array_equal(effect.placebo_comparable, comparable)
        assert effect.p_gap_rank == pytest.approx(rank_p(abs(effect.att_pp), ref.table["abs_mean_gap"].to_numpy()[1:][comparable]))


def test_a_window_that_ignores_the_next_episode_picks_up_its_level_shift(cluster_world, cluster_panel, split_episodes, split_fit):
    effects, _, results = split_fit
    sd = effects.set_index("case_id")["placebo_sd_pp"]
    first_of_economy = dict(zip(split_episodes["case_id"], split_episodes["iso3"].ne(split_episodes["iso3"].shift())))
    checked = 0
    for row in split_episodes.itertuples():
        if row.case_id not in results or pd.isna(row.next_start_year):
            continue
        capped = results[row.case_id]
        series = split_episodes.loc[split_episodes["case_id"] == row.case_id].iloc[0]
        open_ended = estimate_case(cluster_panel, series.drop("next_start_year"), capped.weights.index.tolist(), post_horizon=None)
        assert open_ended.window_end == LAST and capped.window_end < LAST
        truth_capped = cluster_world.window_effect(row.iso3, row.opening_year, capped.window_end)
        truth_open = cluster_world.window_effect(row.iso3, row.opening_year, LAST)
        assert truth_open - truth_capped > 0.1 and open_ended.att_pp > capped.att_pp
        assert abs((open_ended.att_pp - capped.att_pp) - (truth_open - truth_capped)) < 2.0 * sd[row.case_id]
        if first_of_economy[row.case_id]:
            assert abs(capped.att_pp - truth_capped) < 3.0 * sd[row.case_id]
        checked += 1
    assert checked >= 2


def test_guard_calls_per_episode_follow_the_same_formula(cluster_panel, episodes):
    calls = []
    effects, _, _ = estimate_all(cluster_panel, episodes, guard=lambda: calls.append(1))
    assert len(calls) == len(effects) + int((effects["n_donors"] // GUARD_INTERVAL).sum())


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
def test_estimate_case_rejects_invalid_arguments(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    with pytest.raises(ValueError, match="method"):
        estimate_case(panel, mid, donors, method="magic")
    with pytest.raises(ValueError, match="ridge"):
        estimate_case(panel, mid, donors, method="ridge", ridge=0.0)
    with pytest.raises(ValueError, match="at least two"):
        estimate_case(panel, mid, donors[:1])
    with pytest.raises(ValueError, match="unique"):
        estimate_case(panel, mid, [donors[0], donors[0], donors[1]])
    with pytest.raises(ValueError, match="economy of the case"):
        estimate_case(panel, mid, [mid["iso3"], *donors[:5]])
    with pytest.raises(ValueError, match="not in the panel"):
        estimate_case(panel, mid, ["ZZZ", *donors[:5]])
    with pytest.raises(ValueError, match="not in the panel"):
        estimate_case(panel, {**dict(mid), "iso3": "ZZZ"}, donors)


def test_estimate_case_rejects_donors_with_missing_years_and_cases_without_data(panel, cases, mid):
    wide = panel.loc[panel["year"].between(FIRST, LAST)].pivot(index="year", columns="iso3", values="receipts_pct_gdp")
    gappy = [c for c in wide.columns if wide[c].isna().any() and c != mid["iso3"]]
    assert gappy
    with pytest.raises(ValueError, match="missing outcome years"):
        estimate_case(panel, mid, [*donor_pool(panel, mid, cases)[:5], gappy[0]], post_horizon=None)
    donors = donor_pool(panel, mid, cases)
    too_early = {**dict(mid), "opening_year": FIRST + 1}
    with pytest.raises(ValueError, match="pre-opening"):
        estimate_case(panel, too_early, donors)
    too_late = {**dict(mid), "opening_year": LAST + 1}
    with pytest.raises(ValueError, match="post-opening"):
        estimate_case(panel, too_late, donors)


# ----------------------------------------------------------------------------
# Comparable placebos
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rmspe,factor,expected",
    [
        ([1.0] * 7 + [10.0], 2.0, [True] * 7 + [False]),
        ([1.0] * 7 + [2.0, 2.5], 2.0, [True] * 8 + [False]),
        ([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 4.0, 6.0, 8.0], 0.1, [True, True, False, True, False, True, True, False, False]),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 8.0, 9.0], 0.1, [True] * 7 + [False] * 2),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 5.0 * (1 + 1.0e-12), 5.0 * (1 - 1.0e-12), 8.0, 9.0], 0.1, [True] * 7 + [False] * 2),
        ([1.0, 2.0, 3.0, 4.0, 5.0 * (1 - 1.0e-12), 5.0, 5.0 * (1 + 1.0e-6), 8.0, 9.0], 0.1, [True] * 6 + [False] * 3),
        ([3.0, 1.0, 2.0], 1.0e-9, [True] * 3),
        ([3.0, 1.0], 1.0e-9, [True] * 2),
        ([1.0e-13] * 10 + [5.0e-7, 1.5e-6, 3.0e-6], 2.0, [True] * 12 + [False]),
        ([0.0] * 6, 2.0, [True] * 6),
        ([1.0, 2.0, 40.0, 3.0, 2.0, 1.5], 1.0e9, [True] * 6),
    ],
)
def test_comparable_placebos_are_within_the_factor_of_the_median_or_among_the_five_best(rmspe, factor, expected):
    got = _comparable_placebos(np.array(rmspe), factor)
    assert got.dtype == bool and got.tolist() == expected


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("factor", [0.05, 0.8, 2.0, 5.0])
def test_comparable_placebos_agree_with_the_rule_written_out_by_hand(seed, factor):
    rng = np.random.default_rng(seed)
    values = np.exp(rng.normal(size=int(rng.integers(2, 60))))
    if seed % 2:
        values[: min(values.size, 3)] = 1.0e-13 * rng.uniform(size=min(values.size, 3))
    np.testing.assert_array_equal(_comparable_placebos(values, factor), comparable_by_hand(values, factor))


def test_comparable_placebos_are_all_placebos_without_a_factor():
    values = np.array([0.0, 1.0, 1.0e3, 7.0, np.nan_to_num(np.inf)])
    assert _comparable_placebos(values, None).tolist() == [True] * 5


@pytest.mark.parametrize("factor", [0.01, 0.5, 1.0, 2.0, 3.0, 10.0])
def test_the_number_of_comparable_placebos_never_falls_below_five_and_grows_with_the_factor(factor):
    rng = np.random.default_rng(12)
    values = np.exp(rng.normal(size=40))
    smaller = _comparable_placebos(values, factor)
    larger = _comparable_placebos(values, 1.5 * factor)
    assert smaller.sum() >= 5 and larger.sum() >= smaller.sum() and (larger | ~smaller).all()
    assert set(np.flatnonzero(smaller)) >= set(np.argsort(values)[:5])


def test_the_sample_sd_uses_the_finite_entries_only():
    assert _sample_sd(np.array([1.0, np.nan, 3.0, np.inf])) == pytest.approx(np.sqrt(2.0))
    assert np.isnan(_sample_sd(np.array([1.0, np.nan]))) and np.isnan(_sample_sd(np.array([])))
    assert _sample_sd(np.array([2.0, 2.0, 2.0])) == 0.0


def test_a_donor_with_an_outlying_level_does_not_inflate_the_placebo_statistics(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    base = estimate_case(panel, mid, donors)
    k = 10
    outlying = panel.copy()
    own = outlying["iso3"] == donors[k]
    outlying.loc[own, "receipts_pct_gdp"] += 40.0
    effect = estimate_case(outlying, mid, donors)
    everything = estimate_case(outlying, mid, donors, placebo_fit_factor=None)
    others = np.delete(effect.placebo_rmspe_pre, k)
    assert effect.placebo_rmspe_pre[k] > 100.0 * np.median(others) and not effect.placebo_comparable[k]
    assert abs(effect.placebo_att_pp[k]) > 10.0 * abs(effect.att_pp)
    # the comparable placebos follow the rule, and the case itself is hardly changed by a donor far above the others
    np.testing.assert_array_equal(effect.placebo_comparable, comparable_by_hand(effect.placebo_rmspe_pre))
    assert effect.n_placebos_comparable == int(effect.placebo_comparable.sum()) < len(donors)
    assert abs(effect.n_placebos_comparable - base.n_placebos_comparable) <= 3
    assert effect.att_pp == pytest.approx(base.att_pp, abs=0.05) and effect.weights.iloc[k] < 1.0e-6
    # the standard deviation and the p-value use the comparable placebos
    comparable = effect.placebo_comparable
    assert effect.placebo_sd_pp == pytest.approx(np.std(effect.placebo_att_pp[comparable], ddof=1), rel=1e-12)
    assert effect.placebo_sd_pp == pytest.approx(base.placebo_sd_pp, rel=0.25) and effect.placebo_sd_pp < 0.5
    assert effect.p_gap_rank == rank_p(abs(effect.att_pp), np.abs(effect.placebo_att_pp[comparable]))
    # the standard deviation over all placebos is dominated by the outlier
    assert everything.placebo_sd_pp == pytest.approx(np.std(everything.placebo_att_pp, ddof=1), rel=1e-12)
    assert everything.placebo_sd_pp > 20.0 * effect.placebo_sd_pp
    assert everything.p_gap_rank == rank_p(abs(everything.att_pp), np.abs(everything.placebo_att_pp))
    assert everything.p_gap_rank > effect.p_gap_rank


def test_a_placebo_fit_factor_of_none_keeps_every_placebo_and_changes_nothing_else(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    default = estimate_case(panel, mid, donors)
    everything = estimate_case(panel, mid, donors, placebo_fit_factor=None)
    assert default.placebo_fit_factor == PLACEBO_FIT_FACTOR and everything.placebo_fit_factor is None
    assert everything.placebo_comparable.all() and everything.placebo_comparable.dtype == bool
    assert everything.n_placebos_comparable == everything.n_placebos == len(donors) > default.n_placebos_comparable
    assert everything.placebo_sd_pp == pytest.approx(np.std(everything.placebo_att_pp, ddof=1), rel=1e-12)
    finite = everything.placebo_att_rel_pct[np.isfinite(everything.placebo_att_rel_pct)]
    assert everything.placebo_sd_rel_pct == pytest.approx(np.std(finite, ddof=1), rel=1e-12)
    assert everything.p_gap_rank == rank_p(abs(everything.att_pp), np.abs(everything.placebo_att_pp))
    for name in ("placebo_att_pp", "placebo_att_rel_pct", "placebo_rmspe_pre", "placebo_gaps", "gap", "synthetic"):
        np.testing.assert_array_equal(getattr(everything, name), getattr(default, name))
    for name in ("att_pp", "att_rel_pct", "rmspe_pre", "rmspe_post", "rmspe_ratio", "p_ratio_rank", "fit_tolerance", "did_att_pp"):
        np.testing.assert_array_equal(getattr(everything, name), getattr(default, name))
    assert everything.fit_exact == default.fit_exact and everything.pre_fit_ok == default.pre_fit_ok


def test_a_tiny_factor_keeps_the_five_best_fitting_placebos(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    effect = estimate_case(panel, mid, donors, placebo_fit_factor=1.0e-9)
    assert effect.n_placebos_comparable == 5
    assert set(np.flatnonzero(effect.placebo_comparable)) == set(np.argsort(effect.placebo_rmspe_pre)[:5])
    assert effect.placebo_sd_pp == pytest.approx(np.std(effect.placebo_att_pp[effect.placebo_comparable], ddof=1), rel=1e-12)
    assert effect.p_gap_rank == rank_p(abs(effect.att_pp), np.abs(effect.placebo_att_pp[effect.placebo_comparable]))
    assert 1 / 6 <= effect.p_gap_rank <= 1.0
    wide = estimate_case(panel, mid, donors, placebo_fit_factor=1.0e9)
    assert wide.placebo_comparable.all()


@pytest.mark.parametrize("n_donors", [2, 3, 4])
def test_fewer_than_five_donors_are_all_comparable(n_donors):
    donors = random_walks(n_donors, seed=n_donors)
    panel, codes = toy_panel(donors, random_walks(1, seed=50)[0])
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": 2008}, codes, placebo_fit_factor=1.0e-9)
    assert effect.placebo_comparable.tolist() == [True] * n_donors and effect.n_placebos_comparable == n_donors
    assert effect.p_gap_rank == rank_p(abs(effect.att_pp), np.abs(effect.placebo_att_pp))


def test_the_comparable_placebos_of_a_case_with_exact_placebos_keep_all_exact_ones():
    donors, factors, rng = factor_donors(4, n_factors=1)
    treated = random_walks(1, seed=7)[0]
    panel, codes = toy_panel(donors, treated)
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": OPENING}, codes)
    exact = effect.placebo_rmspe_pre <= EXACT_FIT_TOLERANCE
    assert exact.sum() == 12 and effect.placebo_comparable[exact].all()
    assert effect.n_placebos_comparable >= 12 and effect.n_placebos_comparable == int(comparable_by_hand(effect.placebo_rmspe_pre).sum())


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, True, "2"])
def test_placebo_fit_factor_must_be_none_or_a_positive_finite_number(panel, cases, mid, bad):
    with pytest.raises(ValueError, match="placebo_fit_factor"):
        estimate_case(panel, mid, donor_pool(panel, mid, cases)[:5], placebo_fit_factor=bad)
    with pytest.raises(ValueError, match="placebo_fit_factor"):
        estimate_all(panel, cases.iloc[:0], placebo_fit_factor=bad)


def test_placebo_fit_factor_accepts_whole_numbers(panel, cases, mid):
    effect = estimate_case(panel, mid, donor_pool(panel, mid, cases)[:8], placebo_fit_factor=3)
    assert effect.placebo_fit_factor == 3.0 and isinstance(effect.placebo_fit_factor, float)
    assert estimate_all(panel, cases.iloc[:0], placebo_fit_factor=None)[0].empty


def test_estimate_all_passes_the_placebo_fit_factor_on(panel, cases, fitted):
    default = fitted[0]
    everything, placebos, results = estimate_all(panel, cases, placebo_fit_factor=None)
    assert all(effect.placebo_fit_factor is None for effect in results.values())
    assert (everything["n_placebos_comparable"] == everything["n_placebos"]).all() and placebos["comparable"].all()
    changing = ["placebo_sd_pp", "placebo_sd_rel_pct", "p_gap_rank", "n_placebos_comparable"]
    pd.testing.assert_frame_equal(everything.drop(columns=changing), default.drop(columns=changing))
    assert (everything["n_placebos_comparable"] > default["n_placebos_comparable"]).any()
    strict, strict_placebos, strict_results = estimate_all(panel, cases, placebo_fit_factor=0.5)
    assert all(effect.placebo_fit_factor == 0.5 for effect in strict_results.values())
    assert (strict["n_placebos_comparable"] >= np.minimum(strict["n_donors"], 5)).all()
    assert (strict["n_placebos_comparable"] <= default["n_placebos_comparable"]).all()
    assert (strict["n_placebos_comparable"] < default["n_placebos_comparable"]).any()
    assert strict_placebos["comparable"].sum() == strict["n_placebos_comparable"].sum()
    for row in strict.itertuples():
        effect = strict_results[row.case_id]
        assert row.placebo_sd_pp == pytest.approx(np.std(effect.placebo_att_pp[effect.placebo_comparable], ddof=1), rel=1e-12)


# ----------------------------------------------------------------------------
# Exact fits: non-unique weights and the range of the effect
# ----------------------------------------------------------------------------
OPENING = 2003


def factor_donors(seed: int, n_donors: int = 14, n_factors: int = 3, step: float = 0.25):
    """Donor paths driven by random-walk factors, one row per donor, and the factor paths.

    With ``n_factors`` factors the pre-opening paths of the donors span a set of dimension
    ``n_factors``, so that a treated path can lie at a known distance from their convex hull.
    """
    rng = np.random.default_rng(seed)
    factors = step * rng.normal(size=(LAST - FIRST + 1, n_factors)).cumsum(axis=0)
    return (3.0 + factors @ rng.normal(size=(n_factors, n_donors))).T, factors, rng


def near_exact_world(seed: int = 9, misfit: float = 0.004, shift: float = 1.0):
    """Panel, donor codes and case whose pre-opening fit has a largest absolute gap of exactly ``misfit``.

    The case is a weighted mean of fourteen donors plus ``shift`` from the opening year on. A gap that no
    weighted mean of the donors can remove is added to the eight pre-opening years. The weights of the fit are
    not unique, and for the default seed the typical placebo misfit is about 0.1.
    """
    donors, factors, rng = factor_donors(seed)
    pre = np.arange(FIRST, LAST + 1) < OPENING
    treated = rng.dirichlet(np.full(donors.shape[0], 2.0)) @ donors
    basis, _ = np.linalg.qr(factors[pre], mode="complete")
    stray = basis[:, factors.shape[1]:] @ rng.normal(size=basis.shape[1] - factors.shape[1])
    treated[pre] += misfit * stray / np.abs(stray).max()
    treated[~pre] += shift
    panel, codes = toy_panel(donors, treated)
    return panel, codes, {"case_id": "T", "iso3": "TRT", "opening_year": OPENING}


def test_exact_fit_with_more_donors_than_years_gives_a_range_and_fails_the_pre_fit_rule(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    weights = np.zeros(len(donors))
    weights[:3] = [0.5, 0.3, 0.2]
    altered = planted_panel(panel, mid, donors, weights, 1.5)
    effect = estimate_case(altered, mid, donors)
    assert effect.n_pre < effect.n_donors and effect.fit_exact and effect.rmspe_pre < EXACT_FIT_TOLERANCE
    assert effect.rmspe_pre <= 2.0 * np.median(effect.placebo_rmspe_pre)
    assert not effect.pre_fit_ok
    assert effect.att_pp_min <= effect.att_pp <= effect.att_pp_max
    assert effect.att_pp_min < 1.5 < effect.att_pp_max
    assert effect.att_pp_max - effect.att_pp_min > 0.02
    wide, y1, Y0, pre, valid = hand_arrays(altered, mid, donors)
    low, high, A, b = exact_fit_range(y1, Y0, pre, effect.fit_tolerance)
    assert effect.att_pp_min == pytest.approx(low, abs=1e-5) and effect.att_pp_max == pytest.approx(high, abs=1e-5)
    exactly_low, exactly_high, _, _ = exact_fit_range(y1, Y0, pre)
    assert low < exactly_low < exactly_high < high
    rng = np.random.default_rng(7)
    level = Y0[~pre].mean(axis=0)
    for _ in range(30):
        vertex = linprog(rng.normal(size=len(donors)), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        assert vertex.status == 0
        gap = y1[~pre].mean() - level @ vertex.x
        assert effect.att_pp_min - 1e-5 <= gap <= effect.att_pp_max + 1e-5


def test_exact_fit_in_the_simulated_world_is_reported_in_the_effects_table(panel, cases, fitted):
    effects, _, results = fitted
    assert effects["fit_exact"].dtype == bool and effects["fit_exact"].any() and not effects["fit_exact"].all()
    for row in effects.itertuples():
        effect = results[row.case_id]
        pre = effect.years < row.opening_year
        worst = np.nanmax(np.abs(effect.gap[pre]))
        assert row.fit_exact == (worst <= row.fit_tolerance) == effect.fit_exact
        assert row.fit_tolerance == max(EXACT_FIT_TOLERANCE, EXACT_FIT_RELATIVE_TOLERANCE * np.median(effect.placebo_rmspe_pre))
        if not row.fit_exact:
            assert row.att_pp_min == row.att_pp_max == row.att_pp
            assert np.isnan(row.att_pp_lp_min) and np.isnan(row.att_pp_lp_max)
            continue
        assert row.n_pre < row.n_donors and not row.pre_fit_ok
        assert row.att_pp_min <= row.att_pp <= row.att_pp_max and row.att_pp_max - row.att_pp_min > 0.05
        case = cases.loc[cases["case_id"] == row.case_id].iloc[0]
        wide, y1, Y0, pre_mask, valid = hand_arrays(panel, case, effect.weights.index.tolist())
        low, high, _, _ = exact_fit_range(y1, Y0, pre_mask, row.fit_tolerance)
        assert row.att_pp_min == pytest.approx(low, abs=1e-5) and row.att_pp_max == pytest.approx(high, abs=1e-5)


def test_exact_fit_with_unique_weights_has_a_degenerate_range_under_the_absolute_tolerance_and_passes_the_pre_fit_rule(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:4]
    altered = planted_panel(panel, mid, donors, np.array([0.5, 0.3, 0.2, 0.0]), 1.5)
    effect = estimate_case(altered, mid, donors, exact_fit_relative_tolerance=0.0)
    assert effect.n_pre >= effect.n_donors and effect.fit_exact and effect.fit_tolerance == EXACT_FIT_TOLERANCE
    assert effect.att_pp == pytest.approx(1.5, abs=1e-6)
    assert effect.att_pp_min == pytest.approx(1.5, abs=1e-5) and effect.att_pp_max == pytest.approx(1.5, abs=1e-5)
    assert effect.att_pp_min <= effect.att_pp <= effect.att_pp_max
    assert effect.pre_fit_ok
    relaxed = estimate_case(altered, mid, donors)
    assert relaxed.fit_exact and relaxed.pre_fit_ok and relaxed.fit_tolerance > 100 * EXACT_FIT_TOLERANCE
    assert relaxed.att_pp_min < 1.5 - 1e-3 and relaxed.att_pp_max > 1.5 + 1e-3
    assert relaxed.att_pp == effect.att_pp


def test_the_range_is_widened_to_contain_the_effect_when_the_weights_are_not_negative(panel, cases, mid, monkeypatch):
    import dtt.effects as effects_module

    donors = donor_pool(panel, mid, cases)[:4]
    altered = planted_panel(panel, mid, donors, np.array([0.5, 0.3, 0.2, 0.0]), 1.5)
    exact = estimate_case(altered, mid, donors)
    assert exact.fit_exact and exact.weights.min() >= 0.0
    for low, high in [(0.5, 0.7), (-0.7, -0.5), (-0.1, 0.1)]:
        monkeypatch.setattr(
            effects_module, "_identification_range", lambda y1, Y0, pre, tolerance, a=low, b=high: (exact.att_pp + a, exact.att_pp + b)
        )
        widened = estimate_case(altered, mid, donors)
        assert widened.att_pp_min == min(exact.att_pp + low, exact.att_pp)
        assert widened.att_pp_max == max(exact.att_pp + high, exact.att_pp)
        assert widened.att_pp_min <= widened.att_pp <= widened.att_pp_max
        assert (widened.att_pp_lp_min, widened.att_pp_lp_max) == (exact.att_pp + low, exact.att_pp + high)
    monkeypatch.setattr(effects_module, "_identification_range", lambda y1, Y0, pre, tolerance: (np.nan, np.nan))
    unsolved = estimate_case(altered, mid, donors)
    assert np.isnan(unsolved.att_pp_min) and np.isnan(unsolved.att_pp_max) and unsolved.fit_exact
    assert np.isnan(unsolved.att_pp_lp_min) and np.isnan(unsolved.att_pp_lp_max)


def test_the_range_is_not_widened_when_the_weights_can_be_negative(monkeypatch):
    import dtt.effects as effects_module

    panel, codes = toy_panel(random_walks(14, 0), random_walks(1, 100)[0])
    case = {"case_id": "T", "iso3": "TRT", "opening_year": OPENING}
    reference = estimate_case(panel, case, codes, method="ridge", ridge=1.0e-8)
    assert reference.fit_exact and reference.weights.min() < -0.1
    bounds = (reference.att_pp + 1.0, reference.att_pp + 2.0)
    monkeypatch.setattr(effects_module, "_identification_range", lambda y1, Y0, pre, tolerance: bounds)
    effect = estimate_case(panel, case, codes, method="ridge", ridge=1.0e-8)
    assert effect.att_pp == reference.att_pp and effect.att_pp < bounds[0]
    assert (effect.att_pp_min, effect.att_pp_max) == bounds == (effect.att_pp_lp_min, effect.att_pp_lp_max)


def test_fit_exact_switches_at_the_tolerance(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)[:4]
    weights = np.array([0.5, 0.3, 0.2, 0.0])
    opening = int(mid["opening_year"])
    for relative, tolerance in ((0.0, EXACT_FIT_TOLERANCE), (EXACT_FIT_RELATIVE_TOLERANCE, None)):
        flags = {}
        for scale in (0.2, 20.0):
            altered = planted_panel(panel, mid, donors, weights, 1.5)
            if tolerance is None:
                tolerance = estimate_case(altered, mid, donors).fit_tolerance
            own = (altered["iso3"] == mid["iso3"]) & (altered["year"] == opening - 3)
            altered.loc[own, "receipts_pct_gdp"] += scale * tolerance
            effect = estimate_case(altered, mid, donors, exact_fit_relative_tolerance=relative)
            flags[scale] = effect.fit_exact
            assert effect.fit_tolerance == pytest.approx(tolerance, rel=1e-12)
            if not effect.fit_exact:
                assert effect.att_pp_min == effect.att_pp_max == effect.att_pp
        assert flags == {0.2: True, 20.0: False}


def test_estimate_all_reports_the_exact_fit_columns_for_a_planted_economy(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    weights = np.zeros(len(donors))
    weights[:3] = [0.5, 0.3, 0.2]
    altered = planted_panel(panel, mid, donors, weights, 1.5)
    effects, _, results = estimate_all(altered, pd.DataFrame([mid]), feasibility=pd.DataFrame({"case_id": [mid["case_id"]], "feasible": [True]}))
    row = effects.iloc[0]
    effect = results[mid["case_id"]]
    assert row["fit_exact"] and not row["pre_fit_ok"]
    assert (row["att_pp_min"], row["att_pp_max"]) == (effect.att_pp_min, effect.att_pp_max)
    assert (row["att_pp_lp_min"], row["att_pp_lp_max"]) == (effect.att_pp_lp_min, effect.att_pp_lp_max)
    assert row["att_pp_min"] < row["att_pp"] < row["att_pp_max"]
    assert row["fit_tolerance"] == effect.fit_tolerance > EXACT_FIT_TOLERANCE


# ----------------------------------------------------------------------------
# The tolerance of an exact fit scales with the misfit of the placebos
# ----------------------------------------------------------------------------
def test_default_tolerances_are_the_documented_constants():
    assert (PLACEBO_FIT_FACTOR, EXACT_FIT_RELATIVE_TOLERANCE, MIN_COMPARABLE_PLACEBOS, EXACT_FIT_TOLERANCE) == (2.0, 0.05, 5, 1.0e-6)
    panel, codes, case = near_exact_world()
    effect = estimate_case(panel, case, codes)
    assert effect.placebo_fit_factor == PLACEBO_FIT_FACTOR and effect.exact_fit_relative_tolerance == EXACT_FIT_RELATIVE_TOLERANCE


def test_a_fit_within_a_share_of_the_placebo_misfit_is_exact_and_the_absolute_rule_is_recoverable():
    panel, codes, case = near_exact_world()
    default = estimate_case(panel, case, codes)
    absolute = estimate_case(panel, case, codes, exact_fit_relative_tolerance=0.0)
    typical = float(np.median(default.placebo_rmspe_pre))
    worst = float(np.nanmax(np.abs(default.gap[default.years < OPENING])))
    assert worst == pytest.approx(0.004, abs=1e-9) and 0.08 < typical < 0.15 and default.n_pre < default.n_donors
    assert default.fit_tolerance == pytest.approx(EXACT_FIT_RELATIVE_TOLERANCE * typical, rel=1e-12) and worst < default.fit_tolerance
    assert default.fit_exact and not default.pre_fit_ok
    assert np.isfinite(default.att_pp_lp_min) and default.att_pp_lp_min < default.att_pp < default.att_pp_lp_max
    wide, y1, Y0, pre, valid = hand_arrays(panel, case, codes)
    low, high, _, _ = exact_fit_range(y1, Y0, pre, default.fit_tolerance)
    assert default.att_pp_min == pytest.approx(low, abs=1e-5) and default.att_pp_max == pytest.approx(high, abs=1e-5)
    assert default.att_pp_max - default.att_pp_min > 0.01
    # the absolute rule treats the same fit as not exact: the case stays in the primary sample and has no range
    assert absolute.fit_tolerance == EXACT_FIT_TOLERANCE and not absolute.fit_exact and absolute.pre_fit_ok
    assert absolute.att_pp_min == absolute.att_pp_max == absolute.att_pp == default.att_pp
    assert np.isnan(absolute.att_pp_lp_min) and np.isnan(absolute.att_pp_lp_max)
    assert absolute.rmspe_pre == default.rmspe_pre and np.isfinite(absolute.rmspe_ratio) and np.isnan(default.rmspe_ratio)
    assert np.isnan(default.p_ratio_rank) and 0.0 < absolute.p_ratio_rank <= 1.0


@pytest.mark.parametrize("relative", [0.0, 0.01, 0.05, 0.5, 10.0])
def test_fit_tolerance_is_the_larger_of_the_absolute_floor_and_a_share_of_the_median_placebo_misfit(relative):
    panel, codes, case = near_exact_world()
    effect = estimate_case(panel, case, codes, exact_fit_relative_tolerance=relative)
    median = float(np.median(effect.placebo_rmspe_pre))
    assert effect.fit_tolerance == max(EXACT_FIT_TOLERANCE, relative * median)
    assert effect.exact_fit_relative_tolerance == relative
    worst = float(np.nanmax(np.abs(effect.gap[effect.years < OPENING])))
    assert effect.fit_exact == (worst <= effect.fit_tolerance)
    assert effect.n_placebos_exact == int((effect.placebo_rmspe_pre <= effect.fit_tolerance).sum())


def test_the_tolerance_keeps_the_absolute_floor_when_nearly_every_placebo_fits_exactly():
    donors, factors, rng = factor_donors(4, n_factors=1)
    treated = rng.dirichlet(np.full(14, 2.0)) @ donors + np.where(np.arange(FIRST, LAST + 1) >= OPENING, 1.0, 0.0)
    panel, codes = toy_panel(donors, treated)
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": OPENING}, codes)
    median = float(np.median(effect.placebo_rmspe_pre))
    assert median < 1.0e-12 and effect.fit_tolerance == EXACT_FIT_TOLERANCE
    assert effect.n_placebos_exact == 12 and effect.fit_exact and not effect.pre_fit_ok
    # placebos at rounding-error level are all comparable, although several lie above twice the median
    assert (effect.placebo_rmspe_pre > 2.0 * median).sum() > 2
    assert effect.placebo_comparable[effect.placebo_rmspe_pre <= EXACT_FIT_TOLERANCE].all()
    assert effect.n_placebos_comparable == 12


def test_relaxing_the_tolerance_widens_the_identified_range():
    panel, codes, case = near_exact_world(misfit=0.0)
    tight = estimate_case(panel, case, codes, exact_fit_relative_tolerance=0.0)
    relaxed = estimate_case(panel, case, codes)
    assert tight.fit_exact and relaxed.fit_exact and tight.fit_tolerance < relaxed.fit_tolerance
    assert relaxed.att_pp_lp_min < tight.att_pp_lp_min - 1.0e-3 and relaxed.att_pp_lp_max > tight.att_pp_lp_max + 1.0e-3
    assert tight.att_pp == relaxed.att_pp
    wide, y1, Y0, pre, valid = hand_arrays(panel, case, codes)
    lows, highs = [], []
    for tolerance in (EXACT_FIT_TOLERANCE, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2):
        low, high = _identification_range(y1, Y0, pre, tolerance)
        lows.append(low)
        highs.append(high)
    assert np.all(np.diff(lows) <= 1.0e-9) and np.all(np.diff(highs) >= -1.0e-9)
    assert lows[-1] < lows[0] - 0.01 and highs[-1] > highs[0] + 0.01


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf, True, "0.05", None])
def test_exact_fit_relative_tolerance_must_be_a_non_negative_finite_number(panel, cases, mid, bad):
    with pytest.raises(ValueError, match="exact_fit_relative_tolerance"):
        estimate_case(panel, mid, donor_pool(panel, mid, cases)[:5], exact_fit_relative_tolerance=bad)
    with pytest.raises(ValueError, match="exact_fit_relative_tolerance"):
        estimate_all(panel, cases.iloc[:0], exact_fit_relative_tolerance=bad)


def test_estimate_all_passes_the_exact_fit_tolerance_on(panel, cases, fitted):
    absolute, _, results = estimate_all(panel, cases, exact_fit_relative_tolerance=0.0)
    assert (absolute["fit_tolerance"] == EXACT_FIT_TOLERANCE).all()
    assert all(effect.exact_fit_relative_tolerance == 0.0 for effect in results.values())
    for row in absolute.itertuples():
        effect = results[row.case_id]
        assert row.fit_exact == (np.nanmax(np.abs(effect.gap[effect.years < row.opening_year])) <= EXACT_FIT_TOLERANCE)
    wide = estimate_all(panel, cases, exact_fit_relative_tolerance=10.0)[0]
    default = fitted[0]
    assert wide["fit_exact"].sum() > default["fit_exact"].sum() >= absolute["fit_exact"].sum()
    assert (wide["fit_tolerance"] > default["fit_tolerance"]).all() and (default["fit_tolerance"] > EXACT_FIT_TOLERANCE).all()


# ----------------------------------------------------------------------------
# Raw bounds of the identified range
# ----------------------------------------------------------------------------
def hand_problem():
    """Three donors, two pre-opening years and one post-opening year with a known range of the effect.

    The weights that reproduce the pre-opening outcomes put ``w3`` on the third donor and ``0.5 - 0.5 w3`` on each
    of the others, so that the post-opening level is ``4 + 6 w3`` for ``w3`` in [0, 1]; the observed post-opening
    outcome is 6, and the effect lies in [-4, 2]. With a tolerance of 0.1 the upper end becomes 2.2, because the
    first two weights may then sum to 1 with a split of 0.6 and 0.4.
    """
    Y0 = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [3.0, 5.0, 10.0]])
    y1 = np.array([0.5, 0.5, 6.0])
    return y1, Y0, np.array([True, True, False])


def test_the_helper_returns_the_bounds_of_the_two_linear_programs():
    y1, Y0, pre = hand_problem()
    low, high = _identification_range(y1, Y0, pre, 1.0e-9)
    assert (low, high) == pytest.approx((-4.0, 2.0), abs=1e-6)
    # weights inside the set give effects inside the bounds, the ends are reached by the vertices
    for w3 in (0.0, 0.25, 0.5, 1.0):
        w = np.array([0.5 - 0.5 * w3, 0.5 - 0.5 * w3, w3])
        effect = 6.0 - Y0[2] @ w
        assert low - 1.0e-8 <= effect <= high + 1.0e-8
    wider_low, wider_high = _identification_range(y1, Y0, pre, 0.1)
    assert wider_low == pytest.approx(low, abs=1e-6) and wider_high == pytest.approx(2.2, abs=1e-6)
    reference = exact_fit_range(y1, Y0, pre, 0.1)
    assert (wider_low, wider_high) == pytest.approx(reference[:2], abs=1e-9)


def test_the_helper_does_not_widen_the_bounds_and_is_undefined_without_a_solution():
    y1, Y0, pre = hand_problem()
    low, high = _identification_range(y1, Y0, pre, 1.0e-9)
    shifted = y1.copy()
    shifted[:2] = [3.0, 3.0]
    nan_low, nan_high = _identification_range(shifted, Y0, pre, 1.0e-9)
    assert np.isnan(nan_low) and np.isnan(nan_high)
    assert (low, high) != (6.0 - 6.0, 6.0 - 6.0)


def test_a_wrong_objective_in_the_helper_is_caught_by_the_check_against_the_effect(monkeypatch):
    import dtt.effects as effects_module

    y1, Y0, pre = hand_problem()
    w = np.array([0.25, 0.25, 0.5])
    effect = 6.0 - Y0[2] @ w
    low, high = _identification_range(y1, Y0, pre, 1.0e-9)
    assert low - 1.0e-8 <= effect <= high + 1.0e-8

    def opposite_objective(c, *args, **kwargs):
        return linprog(-c, *args, **kwargs)

    monkeypatch.setattr(effects_module.optimize, "linprog", opposite_objective)
    wrong_low, wrong_high = _identification_range(y1, Y0, pre, 1.0e-9)
    assert not (wrong_low - 1.0e-8 <= effect <= wrong_high + 1.0e-8)


def test_raw_bounds_contain_the_effect_of_every_exact_fit_with_non_negative_weights(fitted):
    effects, _, results = fitted
    checked = 0
    for row in effects.loc[effects["fit_exact"]].itertuples():
        effect = results[row.case_id]
        assert effect.weights.min() >= -1.0e-12
        assert effect.att_pp_lp_min - 1.0e-8 <= effect.att_pp <= effect.att_pp_lp_max + 1.0e-8
        assert (row.att_pp_lp_min, row.att_pp_lp_max) == (effect.att_pp_lp_min, effect.att_pp_lp_max)
        assert effect.att_pp_min == min(effect.att_pp_lp_min, effect.att_pp)
        assert effect.att_pp_max == max(effect.att_pp_lp_max, effect.att_pp)
        checked += 1
    assert checked >= 2
    for row in effects.loc[~effects["fit_exact"]].itertuples():
        assert np.isnan(row.att_pp_lp_min) and np.isnan(row.att_pp_lp_max)


@pytest.mark.parametrize("seed", [3, 9, 11])
def test_raw_bounds_contain_the_effect_in_the_factor_worlds(seed):
    panel, codes, case = near_exact_world(seed)
    effect = estimate_case(panel, case, codes)
    assert effect.fit_exact and effect.weights.min() >= -1.0e-12
    assert effect.att_pp_lp_min - 1.0e-8 <= effect.att_pp <= effect.att_pp_lp_max + 1.0e-8
    assert (effect.att_pp_min, effect.att_pp_max) == (effect.att_pp_lp_min, effect.att_pp_lp_max)


# ----------------------------------------------------------------------------
# Ratios and rank p-values
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "numerator,denominator,expected",
    [(0.0, 0.0, "nan"), (2.0, 0.0, "inf"), (0.0, 2.0, 0.0), (3.0, 2.0, 1.5), (np.nan, 1.0, "nan"), (1.0, np.nan, "nan")],
)
def test_ratio_of_two_rmspe_values(numerator, denominator, expected):
    got = _ratio(numerator, denominator)
    if expected == "nan":
        assert np.isnan(got)
    elif expected == "inf":
        assert got == np.inf
    else:
        assert got == expected


def test_a_case_that_equals_a_donor_without_an_effect_has_an_undefined_ratio():
    donors = random_walks(20, seed=1)
    panel, codes = toy_panel(donors, donors[0])
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": 2008}, codes)
    assert effect.rmspe_pre == 0.0 and effect.rmspe_post == 0.0 and effect.att_pp == 0.0
    assert np.isnan(effect.rmspe_ratio) and np.isnan(effect.p_ratio_rank)
    assert effect.p_gap_rank == 1.0
    table, _, _ = estimate_all(panel, pd.DataFrame([{"case_id": "T", "iso3": "TRT", "opening_year": 2008}]), feasibility=pd.DataFrame({"case_id": ["T"], "feasible": [True]}))
    assert np.isnan(table.loc[0, "rmspe_ratio"]) and np.isnan(table.loc[0, "p_ratio_rank"])


def test_a_case_that_equals_a_donor_with_an_effect_has_an_exact_fit_and_no_ratio_statistics():
    donors = random_walks(20, seed=1)
    treated = donors[0] + np.where(np.arange(FIRST, LAST + 1) >= 2008, 0.5, 0.0)
    panel, codes = toy_panel(donors, treated)
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": 2008}, codes)
    assert effect.rmspe_pre == 0.0 and effect.fit_exact and effect.att_pp == pytest.approx(0.5, abs=1e-12)
    assert np.isnan(effect.rmspe_ratio) and np.isnan(effect.p_ratio_rank)
    assert effect.p_gap_rank == rank_p(0.5, np.abs(effect.placebo_att_pp[effect.placebo_comparable]))


def test_rank_p_value_counts_a_tied_placebo_as_at_least_as_extreme():
    assert rank_p_value(3.0, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(3 / 5)
    assert rank_p_value(5.0, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1 / 5)
    assert rank_p_value(0.5, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert rank_p_value(0.0, [0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert rank_p_value(1.0, [1.0] * 5 + [0.5] * 5) == pytest.approx(6 / 11)
    assert rank_p_value(np.inf, [np.inf, np.inf, 1.0]) == pytest.approx(3 / 4)
    assert rank_p_value(2.0, np.array([[1.0, 2.0], [3.0, 0.0]])) == pytest.approx(3 / 5)


def test_rank_p_value_leaves_out_undefined_placebos_and_is_undefined_for_an_undefined_statistic():
    assert np.isnan(rank_p_value(np.nan, [1.0, 2.0]))
    assert rank_p_value(2.0, [np.nan, 1.0, 3.0]) == pytest.approx(2 / 3)
    assert rank_p_value(2.0, [np.nan, np.nan]) == 1.0
    assert rank_p_value(2.0, []) == 1.0


def test_rank_p_value_without_ties_is_the_rank_over_the_number_of_units():
    rng = np.random.default_rng(3)
    for _ in range(20):
        values = rng.normal(size=11)
        expected = scm.rank_descending(values)[0] / 11
        assert rank_p_value(values[0], values[1:]) == pytest.approx(expected)


def test_reported_p_values_are_the_rank_p_values_of_the_reported_statistics(fitted):
    effects, _, results = fitted
    assert effects["fit_exact"].any() and not effects["fit_exact"].all()
    for row in effects.itertuples():
        effect = results[row.case_id]
        p_gap = rank_p(abs(effect.att_pp), np.abs(effect.placebo_att_pp[effect.placebo_comparable]))
        assert row.p_gap_rank == pytest.approx(p_gap, abs=1e-12)
        if row.fit_exact:
            assert np.isnan(row.rmspe_ratio) and np.isnan(row.p_ratio_rank)
            continue
        assert row.rmspe_ratio == pytest.approx(effect.rmspe_post / effect.rmspe_pre, rel=1e-12)
        assert row.p_ratio_rank == pytest.approx(rank_p(row.rmspe_ratio, placebo_ratios(effect)), abs=1e-12)


# ----------------------------------------------------------------------------
# Ratio statistics of exact fits and exact placebos, and non-unique placebo fits
# ----------------------------------------------------------------------------
def one_factor_world(seed: int = 3, post_noise: float = 0.3):
    """Panel, donor codes and case with donors on a line before the opening and noisy donors after it.

    Fourteen donors follow one common factor before the opening, so that the placebo of each of the twelve donors
    inside the range of the loadings fits exactly with many equivalent weight vectors. After the opening each donor
    carries independent noise. The case is an unrelated random walk, so that its own fit is poor.
    """
    donors, factors, rng = factor_donors(seed, n_factors=1)
    post = np.arange(FIRST, LAST + 1) >= OPENING
    donors = donors + np.where(post, post_noise * rng.normal(size=donors.shape), 0.0)
    panel, codes = toy_panel(donors, random_walks(1, seed + 20)[0])
    return panel, codes, {"case_id": "T", "iso3": "TRT", "opening_year": OPENING}


def raw_placebo_ratios(effect: CaseEffect) -> np.ndarray:
    """Post-to-pre RMSPE ratio of every placebo with no exclusion, from the stored gaps."""
    pre = (effect.years < effect.opening_year) & np.isfinite(effect.actual)
    post = effect.years >= effect.opening_year
    rmspe_pre = np.sqrt(np.mean(effect.placebo_gaps[pre] ** 2, axis=0))
    rmspe_post = np.sqrt(np.mean(effect.placebo_gaps[post] ** 2, axis=0))
    return np.array([_ratio(a, b) for a, b in zip(rmspe_post, rmspe_pre)])


def test_placebos_with_an_exact_fit_are_left_out_of_the_ratio_rank():
    panel, codes, case = one_factor_world()
    effect = estimate_case(panel, case, codes)
    exact = effect.placebo_rmspe_pre <= effect.fit_tolerance
    assert not effect.fit_exact and exact.sum() == effect.n_placebos_exact == 12
    ratios = placebo_ratios(effect)
    assert np.isnan(ratios[exact]).all() and np.isfinite(ratios[~exact]).all() and (~exact).sum() == 2
    expected = rank_p(effect.rmspe_ratio, ratios[~exact])
    assert effect.p_ratio_rank == pytest.approx(expected, abs=1e-12) and expected in (1 / 3, 2 / 3, 1.0)
    # the ratios of the exact placebos are enormous and would decide the rank if they were kept
    raw = raw_placebo_ratios(effect)
    assert raw[exact].min() > 1.0e9
    assert rank_p(effect.rmspe_ratio, raw) > 0.8 > effect.p_ratio_rank
    assert effect.p_gap_rank == rank_p(abs(effect.att_pp), np.abs(effect.placebo_att_pp[effect.placebo_comparable]))
    table, _, _ = estimate_all(panel, pd.DataFrame([case]), feasibility=pd.DataFrame({"case_id": ["T"], "feasible": [True]}))
    assert table.loc[0, "p_ratio_rank"] == effect.p_ratio_rank and table.loc[0, "n_placebos_exact"] == 12


def test_placebos_within_the_relative_tolerance_are_left_out_of_the_ratio_rank(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    absolute = estimate_case(panel, mid, donors, exact_fit_relative_tolerance=0.0)
    relaxed = estimate_case(panel, mid, donors, exact_fit_relative_tolerance=0.15)
    assert not absolute.fit_exact and not relaxed.fit_exact and relaxed.rmspe_ratio == absolute.rmspe_ratio
    assert absolute.n_placebos_exact == 0 and (relaxed.placebo_rmspe_pre > EXACT_FIT_TOLERANCE).all()
    assert 0 < relaxed.n_placebos_exact < relaxed.n_placebos
    kept = ~(relaxed.placebo_rmspe_pre <= relaxed.fit_tolerance)
    assert relaxed.p_ratio_rank == pytest.approx(rank_p(relaxed.rmspe_ratio, raw_placebo_ratios(relaxed)[kept]), abs=1e-12)
    assert absolute.p_ratio_rank == pytest.approx(rank_p(absolute.rmspe_ratio, raw_placebo_ratios(absolute)), abs=1e-12)
    assert relaxed.p_ratio_rank != absolute.p_ratio_rank


def test_exact_cases_in_the_simulated_world_have_no_ratio_statistics(fitted):
    effects, _, results = fitted
    exact = effects["fit_exact"]
    assert exact.any() and not exact.all()
    assert effects.loc[exact, ["rmspe_ratio", "p_ratio_rank"]].isna().all().all()
    assert effects.loc[~exact, ["rmspe_ratio", "p_ratio_rank"]].notna().all().all()
    for row in effects.loc[exact].itertuples():
        effect = results[row.case_id]
        assert np.isnan(effect.rmspe_ratio) and np.isnan(effect.p_ratio_rank)
        assert effect.rmspe_pre <= row.fit_tolerance and np.isfinite(effect.rmspe_post) and np.isfinite(effect.p_gap_rank)


def test_the_ratio_of_an_exact_fit_is_not_reported_although_it_would_be_huge():
    panel, codes, case = near_exact_world(misfit=0.0)
    effect = estimate_case(panel, case, codes, exact_fit_relative_tolerance=0.0)
    assert effect.fit_exact and effect.rmspe_pre < 1.0e-9 < effect.rmspe_post
    assert effect.rmspe_post / max(effect.rmspe_pre, 1.0e-300) > 1.0e8
    assert np.isnan(effect.rmspe_ratio) and np.isnan(effect.p_ratio_rank)


def test_placebo_effects_of_non_unique_exact_fits_follow_fit_direct_and_depend_on_its_rule():
    panel, codes, case = one_factor_world()
    effect = estimate_case(panel, case, codes)
    wide = panel.pivot(index="year", columns="iso3", values="receipts_pct_gdp")
    wide = wide.loc[wide.index <= effect.window_end]
    Y0 = wide[codes].to_numpy()
    pre = wide.index.to_numpy() < OPENING
    exact = np.flatnonzero(effect.placebo_rmspe_pre <= effect.fit_tolerance)
    assert exact.size == effect.n_placebos_exact == 12 and pre.sum() < Y0.shape[1] - 1
    differences = []
    for k in exact:
        pool = np.delete(np.arange(Y0.shape[1]), k)
        w = scm.fit_direct(Y0[pre][:, k], Y0[pre][:, pool]).w
        stored = Y0[:, k] - Y0[:, pool] @ w
        assert effect.placebo_att_pp[k] == pytest.approx(stored[~pre].mean(), abs=1e-9)
        other = linprog(
            -np.arange(pool.size, dtype=float),
            A_eq=np.vstack([Y0[pre][:, pool], np.ones(pool.size)]),
            b_eq=np.r_[Y0[pre][:, k], 1.0],
            bounds=(0, None),
            method="highs",
        )
        assert other.status == 0
        alternative = Y0[:, k] - Y0[:, pool] @ other.x
        assert np.abs(alternative[pre]).max() < 1.0e-8
        differences.append(abs(alternative[~pre].mean() - stored[~pre].mean()))
    assert max(differences) > 0.05


def test_the_rule_for_non_unique_minimisers_is_stated_where_the_brief_asks_for_it():
    import dtt.effects as effects_module

    for doc in (scm.__doc__, scm.fit_direct.__doc__, effects_module.__doc__, estimate_case.__doc__):
        flat = " ".join(doc.split())
        assert "unique" in flat and "vertex" in flat and "smallest norm" in flat
    assert "fit_direct" in estimate_all.__doc__ and "n_placebos_exact" in estimate_all.__doc__
    assert "n_placebos_exact" in effects_module.__doc__ and "n_placebos_exact" in estimate_case.__doc__


# ----------------------------------------------------------------------------
# Relative effects
# ----------------------------------------------------------------------------
def planted_levels(post_level: float, seed: int = 4, n_donors: int = 12, opening: int = 2008):
    """Donors that fall to ``post_level`` from the opening year on, and a treated economy that is their mean plus one."""
    rng = np.random.default_rng(seed)
    years = np.arange(FIRST, LAST + 1)
    post = years >= opening
    donors = 4.0 + 0.2 * rng.normal(size=(n_donors, years.size)).cumsum(axis=1)
    donors[:, post] = post_level + 0.05 * rng.normal(size=(n_donors, int(post.sum())))
    treated = donors[:3].mean(axis=0) + np.where(post, 1.0, 0.0)
    return toy_panel(donors, treated)


def test_relative_effect_is_missing_when_the_synthetic_post_level_is_negative():
    panel, codes = planted_levels(-3.0)
    case = {"case_id": "T", "iso3": "TRT", "opening_year": 2008}
    effect = estimate_case(panel, case, codes)
    assert effect.synthetic[effect.years >= 2008].mean() < -2.5
    assert effect.att_pp == pytest.approx(1.0, abs=1e-5)
    assert np.isnan(effect.att_rel_pct)
    assert np.isnan(effect.placebo_att_rel_pct).all() and np.isfinite(effect.placebo_att_pp).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table, _, _ = estimate_all(panel, pd.DataFrame([case]), feasibility=pd.DataFrame({"case_id": ["T"], "feasible": [True]}))
    assert np.isnan(table.loc[0, "att_rel_pct"]) and np.isnan(table.loc[0, "placebo_sd_rel_pct"])
    assert np.isfinite(table.loc[0, "placebo_sd_pp"])


def test_relative_effect_is_missing_when_the_synthetic_post_level_is_zero():
    donors = random_walks(12, seed=2)
    post = np.arange(FIRST, LAST + 1) >= 2008
    donors[:, post] = 0.0
    treated = donors[:3].mean(axis=0) + np.where(post, 0.5, 0.0)
    panel, codes = toy_panel(donors, treated)
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": 2008}, codes)
    assert effect.att_pp == pytest.approx(0.5, abs=1e-6) and np.isnan(effect.att_rel_pct)


def test_relative_effect_is_the_effect_over_the_positive_synthetic_post_level():
    panel, codes = planted_levels(3.0)
    effect = estimate_case(panel, {"case_id": "T", "iso3": "TRT", "opening_year": 2008}, codes)
    level = effect.synthetic[effect.years >= 2008].mean()
    assert level > 2.5 and effect.att_rel_pct == pytest.approx(100.0 * effect.att_pp / level, rel=1e-12)
    assert np.isfinite(effect.placebo_att_rel_pct).all()


# ----------------------------------------------------------------------------
# The pre-opening fit rule and its factor
# ----------------------------------------------------------------------------
def test_pre_fit_factor_sets_the_rule_and_defaults_to_two(panel, cases, mid):
    donors = donor_pool(panel, mid, cases)
    default = estimate_case(panel, mid, donors)
    assert default.pre_fit_factor == 2.0 and not default.fit_exact
    ratio = default.rmspe_pre / np.median(default.placebo_rmspe_pre)
    assert 0.0 < ratio
    assert default.pre_fit_ok == (ratio <= 2.0)
    just_above = estimate_case(panel, mid, donors, pre_fit_factor=1.01 * ratio)
    just_below = estimate_case(panel, mid, donors, pre_fit_factor=0.99 * ratio)
    assert just_above.pre_fit_ok and not just_below.pre_fit_ok
    assert just_above.pre_fit_factor == 1.01 * ratio
    assert_same_effect(just_above, default)


def test_estimate_all_passes_the_pre_fit_factor_on(panel, cases, fitted):
    effects = fitted[0]
    lenient, _, _ = estimate_all(panel, cases, pre_fit_factor=1.0e9)
    strict, _, _ = estimate_all(panel, cases, pre_fit_factor=1.0e-9)
    unique = ~(effects["fit_exact"] & (effects["n_pre"] < effects["n_donors"]))
    assert lenient["pre_fit_ok"].tolist() == unique.tolist()
    assert not strict["pre_fit_ok"].any()
    assert effects["pre_fit_ok"].any() and not effects["pre_fit_ok"].all()
    pd.testing.assert_frame_equal(effects.drop(columns="pre_fit_ok"), lenient.drop(columns="pre_fit_ok"))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, True, "2", None])
def test_pre_fit_factor_must_be_a_positive_finite_number(panel, cases, mid, bad):
    with pytest.raises(ValueError, match="pre_fit_factor"):
        estimate_case(panel, mid, donor_pool(panel, mid, cases)[:5], pre_fit_factor=bad)
    with pytest.raises(ValueError, match="pre_fit_factor"):
        estimate_all(panel, cases.iloc[:0], pre_fit_factor=bad)
