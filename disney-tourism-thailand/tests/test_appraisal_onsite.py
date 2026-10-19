"""Tests of the labelled case with on-site spending counted as operator revenue (:func:`dtt.appraisal.onsite_revenue_case`)."""

from pathlib import Path

import numpy as np
import pytest

from dtt import appraisal as ap

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "appraisal" / "inputs.csv"
ASSUMPTIONS = ROOT / "data" / "appraisal" / "assumptions.csv"


@pytest.fixture(scope="module")
def tables():
    return ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS)


@pytest.mark.parametrize("case", ap.CASES)
def test_zero_onsite_income_reproduces_the_labelled_case(tables, case):
    sourced, assumptions = tables
    base = ap.labelled_case(sourced, assumptions, case)
    onsite = ap.onsite_revenue_case(sourced, assumptions, case, 0.0)
    assert onsite["operator_npv_bn"] == pytest.approx(base["operator_npv_bn"], rel=1e-12)
    assert onsite["national_npv_bn"] == pytest.approx(base["national_npv_bn"], rel=1e-12)
    assert onsite["fiscal_npv_bn"] == pytest.approx(base["fiscal_npv_bn"], rel=1e-12)
    assert onsite["revenue_bn_per_year"] == pytest.approx(base["revenue_bn_per_year"], rel=1e-12)


def test_onsite_revenue_is_operator_visits_times_the_income_per_visit(tables):
    sourced, assumptions = tables
    inp = ap.case_inputs(sourced, assumptions, "base_case")
    visits = ap.case_annual(inp)["operator_visits_m"]
    out = ap.onsite_revenue_case(sourced, assumptions, "base_case", 1862.0)
    assert out["onsite_revenue_bn_per_year"] == pytest.approx(visits * 1862.0 / 1000.0, rel=1e-12)
    base = ap.labelled_case(sourced, assumptions, "base_case")
    assert out["revenue_bn_per_year"] - base["revenue_bn_per_year"] == pytest.approx(out["onsite_revenue_bn_per_year"], rel=1e-12)


def test_national_ledger_does_not_change_with_onsite_income(tables):
    sourced, assumptions = tables
    base = ap.labelled_case(sourced, assumptions, "base_case")
    for value in (500.0, 1862.0, 10000.0):
        out = ap.onsite_revenue_case(sourced, assumptions, "base_case", value)
        assert out["national_npv_bn"] == pytest.approx(base["national_npv_bn"], rel=1e-12)


def test_operator_npv_gain_is_the_margin_on_onsite_income_over_the_operating_years(tables):
    sourced, assumptions = tables
    inp = ap.case_inputs(sourced, assumptions, "base_case")
    annual = ap.case_annual(inp)
    base = ap.labelled_case(sourced, assumptions, "base_case")
    value = 1862.0
    out = ap.onsite_revenue_case(sourced, assumptions, "base_case", value)
    onsite_bn = annual["operator_visits_m"] * value / 1000.0
    build = int(inp["construction_years"])
    total = build + int(inp["project_life_years"])
    t = np.arange(total)
    # the on-site income is earned in the operating years only, net of the same operating-cost share
    margin = np.where(t >= build, (1.0 - inp["opex_share"]) * onsite_bn, 0.0)
    expected_gain = ap.npv(margin, inp["wacc_financial"])
    assert out["operator_npv_bn"] - base["operator_npv_bn"] == pytest.approx(expected_gain, rel=1e-9)


def test_onsite_income_raises_the_operator_npv_monotonically(tables):
    sourced, assumptions = tables
    values = [0.0, 1000.0, 1862.0, 5000.0]
    npvs = [ap.onsite_revenue_case(sourced, assumptions, "base_case", v)["operator_npv_bn"] for v in values]
    assert all(b > a for a, b in zip(npvs, npvs[1:]))


def test_negative_onsite_income_is_refused(tables):
    sourced, assumptions = tables
    with pytest.raises(ValueError, match="onsite_thb_per_visit"):
        ap.onsite_revenue_case(sourced, assumptions, "base_case", -1.0)
