"""Tests for dtt.simulate (SIMULATED validation panels)."""
import numpy as np
import pandas as pd
import pytest

from dtt import did, replication, simulate


def test_panel_has_the_documented_structure(sim_parallel):
    df, truth = sim_parallel
    assert list(df.columns) == ["unit_id", "year", "receipts_pct_gdp", "treated", "post", "treated_post", "study_case", "rel_year"]
    assert len(df) == 177
    assert replication.validate_panel(df) == []
    hk, sh = df[df.study_case == "Hong Kong"], df[df.study_case == "Shanghai"]
    assert (len(hk), len(sh)) == (129, 48)
    assert sorted(hk.unit_id.unique()) == [2, 3, 4, 7, 8, 9] and sorted(sh.unit_id.unique()) == [1, 5, 6]
    for case, units in simulate.STRUCTURE.items():
        for u, (a, b) in units.items():
            yrs = df.loc[df.unit_id == u, "year"]
            assert (yrs.min(), yrs.max(), len(yrs)) == (a, b, b - a + 1)
    assert len(sh[sh.year >= 2008]) == 36
    assert df.duplicated(["unit_id", "year"]).sum() == 0
    assert df.receipts_pct_gdp.notna().all()


def test_treatment_variables_follow_the_design(sim_parallel):
    df, _ = sim_parallel
    t0 = {"Hong Kong": 2005, "Shanghai": 2016}
    for case, y0 in t0.items():
        blk = df[df.study_case == case]
        assert (blk.rel_year == blk.year - y0).all()
        assert (blk.post == (blk.year >= y0)).all()
        assert blk[blk.treated == 1].unit_id.nunique() == 1
    assert sorted(df[df.treated == 1].unit_id.unique()) == [2, 5]
    assert (df.treated_post == df.treated * df.post).all()
    assert df.treated_post.sum() == (2019 - 2005 + 1) + (2019 - 2016 + 1)


def test_generation_is_deterministic_and_seed_dependent():
    a, ta = simulate.simulate_panel(seed=5)
    b, tb = simulate.simulate_panel(seed=5)
    c, _ = simulate.simulate_panel(seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.receipts_pct_gdp, c.receipts_pct_gdp)
    assert ta == tb
    assert ta.note.startswith("SIMULATED")


def test_known_effect_and_weights_are_stored_in_the_truth():
    _, truth = simulate.simulate_panel(seed=1, tau_hk=3.0, tau_sh=0.5)
    assert truth.tau == {"Hong Kong": 3.0, "Shanghai": 0.5}
    for case, w in truth.weights.items():
        assert sum(w.values()) == pytest.approx(1.0)
    assert truth.weights["Hong Kong"][4] == 0.5


def test_effect_size_enters_the_treated_outcome_one_for_one():
    a, _ = simulate.simulate_panel(seed=2, tau_hk=0.0)
    b, _ = simulate.simulate_panel(seed=2, tau_hk=4.0)
    diff = (b.receipts_pct_gdp - a.receipts_pct_gdp).to_numpy()
    expect = 4.0 * ((a.unit_id == 2) & (a.year >= 2005)).to_numpy()
    assert diff == pytest.approx(expect, abs=1e-12)


def test_trend_spread_creates_a_pretrend_in_the_treated_unit_only():
    par, _ = simulate.simulate_panel(seed=7, trend_spread=0.0)
    trd, _ = simulate.simulate_panel(seed=7, trend_spread=0.5)

    def pre_interaction(df):
        sub = df[(df.study_case == "Hong Kong") & (df.post == 0)].assign(t=lambda d: d.year - 1998)
        return did.pretrend_test(sub).coef

    # Slope of the treated unit minus the average donor slope, from the data-generating process:
    # donor slope directions {3: -0.5, 4: 1.0, 7: 1.5, 8: -0.8, 9: 0.2}, weights {4: 0.5, 7: 0.3, 9: 0.2}.
    spread = 0.5
    expected = spread * ((0.5 * 1.0 + 0.3 * 1.5 + 0.2 * 0.2) - np.mean([-0.5, 1.0, 1.5, -0.8, 0.2]))
    assert expected == pytest.approx(0.355)
    assert abs(pre_interaction(par)) < 0.15
    assert abs(pre_interaction(trd) - expected) < 0.1


def test_noise_parameter_scales_the_idiosyncratic_variation():
    quiet, _ = simulate.simulate_panel(seed=8, noise=0.01)
    loud, _ = simulate.simulate_panel(seed=8, noise=0.5)
    pre = lambda d: d[(d.unit_id == 3)].receipts_pct_gdp.diff().std()
    assert pre(loud) > 2 * pre(quiet)


def test_factor_panel_shapes_and_convexity():
    d = simulate.simulate_factor_panel(n_donors=96, T=22, T0=14, n_active=4, noise=0.0, seed=3)
    Y, w = d["Y"], d["weights"]
    assert Y.shape == (22, 97) and d["pre"].sum() == 14 and d["T0"] == 14
    assert w.shape == (96,) and w.sum() == pytest.approx(1.0) and (w > 0).sum() == 4
    # without noise the treated pre-period path is exactly the weighted donor path; post adds tau
    assert Y[:14, 0] == pytest.approx(Y[:14, 1:] @ w, abs=1e-12)
    assert Y[14:, 0] - Y[14:, 1:] @ w == pytest.approx(np.full(8, d["tau"]), abs=1e-12)


def test_factor_panel_is_deterministic():
    a = simulate.simulate_factor_panel(seed=4)
    b = simulate.simulate_factor_panel(seed=4)
    c = simulate.simulate_factor_panel(seed=5)
    assert np.array_equal(a["Y"], b["Y"]) and not np.array_equal(a["Y"], c["Y"])
