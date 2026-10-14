"""Tests of dtt.appraisal: discounting, the ledgers, break-even and the input table.

The values in this file are synthetic test fixtures. They are not Thai estimates.
"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from dtt import appraisal as ap

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "appraisal" / "inputs.csv"


def test_discount_factors_start_at_one_and_decline():
    factors = ap.discount_factors(0.1, 4)
    npt.assert_allclose(factors, [1.0, 1 / 1.1, 1 / 1.1**2, 1 / 1.1**3])


def test_discount_factors_reject_invalid_inputs():
    with pytest.raises(ValueError):
        ap.discount_factors(-1.0, 3)
    with pytest.raises(ValueError):
        ap.discount_factors(0.05, 0)


def test_npv_at_zero_rate_is_the_sum_of_flows():
    assert ap.npv([-100.0, 40.0, 70.0], 0.0) == pytest.approx(10.0)


def test_npv_of_a_single_flow_at_time_zero_is_that_flow():
    assert ap.npv([-25.0], 0.2) == pytest.approx(-25.0)


def test_irr_recovers_the_rate_of_a_known_flow():
    # -100 today and 121 in two years gives an IRR of exactly 10 percent.
    assert ap.irr([-100.0, 0.0, 121.0]) == pytest.approx(0.10, abs=1e-8)


def test_irr_is_nan_without_a_sign_change():
    assert np.isnan(ap.irr([100.0, 50.0]))


def test_incremental_visitors_reject_shares_outside_unit_interval():
    with pytest.raises(ValueError):
        ap.incremental_visitors([10.0], 1.5)
    with pytest.raises(ValueError):
        ap.incremental_visitors([10.0], -0.1)


def test_net_spend_converts_units_and_applies_displacement():
    # 2 million visits at THB 10,000 each is THB 20 billion gross; a quarter of it is displaced.
    gross = ap.net_tourism_spend([2.0], 10_000.0, 0.0)
    assert gross[0] == pytest.approx(20.0)
    assert ap.net_tourism_spend([2.0], 10_000.0, 0.25)[0] == pytest.approx(15.0)


def test_full_displacement_removes_the_spending():
    assert ap.net_tourism_spend([3.0], 5_000.0, 1.0)[0] == pytest.approx(0.0)


def test_displacement_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        ap.net_tourism_spend([1.0], 1_000.0, 1.2)


def test_operator_flows_subtract_costs_and_capex():
    flows = ap.operator_flows(revenue=[30.0, 40.0], opex=[10.0, 12.0], capex=[50.0, 0.0])
    npt.assert_allclose(flows, [-30.0, 28.0])


def test_national_flows_rise_with_domestic_retention():
    net = np.array([10.0, 10.0])
    low = ap.national_flows(net, 0.2, [5.0, 0.0], [1.0, 1.0], 0.9)
    high = ap.national_flows(net, 0.8, [5.0, 0.0], [1.0, 1.0], 0.9)
    assert np.all(high > low)


def test_national_flows_reject_retention_outside_unit_interval():
    with pytest.raises(ValueError):
        ap.national_flows([1.0], 1.5, [0.0], [0.0], 1.0)


def test_break_even_scale_finds_the_sign_change():
    scale = ap.break_even_scale(lambda s: 100.0 * s - 50.0)
    assert scale == pytest.approx(0.5, abs=1e-6)


def test_break_even_scale_raises_without_a_sign_change():
    with pytest.raises(ValueError):
        ap.break_even_scale(lambda s: 1.0 + s)


def test_require_values_names_every_pending_parameter():
    table = pd.DataFrame(
        {"value": [1.0, np.nan, np.nan], "unit": ["x", "y", "z"], "source": ["a", "b", "c"], "status": ["sourced", "pending", "pending"]},
        index=pd.Index(["known", "gap_one", "gap_two"], name="parameter"),
    )
    with pytest.raises(ValueError, match="gap_one, gap_two"):
        ap.require_values(table, ["known", "gap_one", "gap_two"])
    assert ap.require_values(table, ["known"]) == {"known": 1.0}


def test_input_table_has_a_source_for_every_row():
    table = ap.load_inputs(INPUTS)
    assert table["source"].notna().all() and (table["source"].str.len() > 0).all()


def test_blank_values_are_marked_pending():
    table = ap.load_inputs(INPUTS)
    blank = table["value"].isna()
    assert blank.any()
    assert table.loc[blank, "status"].str.startswith("pending").all()


def test_adb_standard_discount_rate_is_recorded_as_sourced():
    table = ap.load_inputs(INPUTS)
    assert table.loc["discount_rate_eirr_standard", "value"] == pytest.approx(0.09)
    assert table.loc["discount_rate_eirr_standard", "status"].startswith("sourced")


def test_base_case_cannot_run_while_inputs_are_pending():
    table = ap.load_inputs(INPUTS)
    with pytest.raises(ValueError, match="discount_rate_national_economic"):
        ap.require_values(table, ["discount_rate_national_economic"])
