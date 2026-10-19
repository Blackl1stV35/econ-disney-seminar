"""Tests of the scenario script (``scripts/appraisal_scenarios.py``): the cash cost, break-even capital cost, the hotel and national checks."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("appraisal_scenarios", ROOT / "scripts" / "appraisal_scenarios.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


@pytest.fixture(scope="module")
def tables():
    return sc.load()


def test_cash_opex_share_is_read_from_the_hong_kong_accounts(tables):
    sourced, _ = tables
    assert sc.hk_cash_opex_share(sourced) == pytest.approx(6705.0 / 8694.0, rel=1e-12)


def test_break_even_capex_gives_zero_operator_npv_under_the_cash_cost(tables):
    sourced, assump = tables
    cash = sc.with_assumption(assump, "opex_share", sc.hk_cash_opex_share(sourced))
    for onsite in (0.0, sc.ONSITE_THB):
        c = sc.break_even_capex(sourced, cash, onsite)
        assert c is not None
        assert sc.operator_npv(sc.with_capex(sourced, c), cash, onsite) == pytest.approx(0.0, abs=1e-6)


def test_break_even_capex_falls_as_the_operating_cost_share_rises(tables):
    sourced, assump = tables
    curve = [sc.break_even_capex(sourced, sc.with_assumption(assump, "opex_share", s), 0.0) for s in (0.0, 0.4, 0.6, 0.9383)]
    assert all(b < a for a, b in zip(curve, curve[1:]))


def test_invariance_break_even_matches_the_root_of_the_operator_npv(tables):
    sourced, assump = tables
    cash = sc.with_assumption(assump, "opex_share", sc.hk_cash_opex_share(sourced))
    res = sc.run(sourced, assump)
    root = sc.break_even_capex(sourced, cash, 0.0)
    assert res["invariance"]["breakeven_capex_bn"] == pytest.approx(root, rel=1e-6)


def test_extra_hotel_revenue_at_the_solved_level_gives_zero_operator_npv(tables):
    sourced, assump = tables
    cash = sc.with_assumption(assump, "opex_share", sc.hk_cash_opex_share(sourced))
    extra = sc.bisect(lambda r: sc.operator_npv_with_extra_revenue(sourced, cash, r), 0.0, 500.0)
    assert extra is not None
    assert sc.operator_npv_with_extra_revenue(sourced, cash, extra) == pytest.approx(0.0, abs=1e-6)


def test_rooms_needed_falls_as_the_room_rate_rises():
    low = sc.rooms_needed(10.0, 2000.0, 0.655)
    high = sc.rooms_needed(10.0, 5000.0, 0.655)
    assert high < low
    assert low == pytest.approx(10.0e9 / (2000.0 * 365.0 * 0.655), rel=1e-12)


def test_the_confirmed_base_is_not_changed_by_the_scenarios(tables):
    sourced, assump = tables
    before = sc.ap.labelled_case(sourced, assump, sc.CASE)["operator_npv_bn"]
    sc.run(sourced, assump)
    after = sc.ap.labelled_case(sourced, assump, sc.CASE)["operator_npv_bn"]
    assert after == pytest.approx(before, rel=1e-12)
