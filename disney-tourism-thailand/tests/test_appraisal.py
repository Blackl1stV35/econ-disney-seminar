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
    with pytest.raises(ValueError, match="attendance_million_per_year, incremental_share"):
        ap.require_values(table, ["attendance_million_per_year", "incremental_share"])


def test_nesdc_social_discount_rate_matches_the_guide():
    table = ap.load_inputs(INPUTS)
    assert table.loc["discount_rate_social_nesdc", "value"] == pytest.approx(ap.SOCIAL_DISCOUNT_RATE_NESDC)
    low, high = ap.SOCIAL_DISCOUNT_RATE_RANGE
    assert low <= ap.SOCIAL_DISCOUNT_RATE_NESDC <= high
    assert table.loc["discount_rate_social_nesdc", "status"].startswith("sourced")


def test_standard_conversion_factor_matches_the_guide():
    table = ap.load_inputs(INPUTS)
    assert table.loc["conversion_factor_standard", "value"] == pytest.approx(ap.STANDARD_CONVERSION_FACTOR)


def test_sensitivity_cases_are_five_ten_and_twenty_percent():
    assert ap.SENSITIVITY_PERCENTS == (5, 10, 20)


def test_project_life_and_financial_rate_stay_pending_until_the_team_states_them():
    table = ap.load_inputs(INPUTS)
    for name in ("project_life_years", "wacc_financial"):
        assert pd.isna(table.loc[name, "value"])
        with pytest.raises(ValueError, match=name):
            ap.require_values(table, [name])


def test_infrastructure_headroom_is_never_negative():
    # Synthetic fixture: capacities and loads are invented for this test.
    headroom = ap.infrastructure_headroom([5.0, 3.0, 0.0, 8.0], [2.0, 4.0, 1.0, 8.0])
    npt.assert_allclose(headroom, [3.0, 0.0, 0.0, 0.0])
    assert np.all(headroom >= 0)


def test_planned_addition_raises_the_headroom_before_the_floor_applies():
    # Synthetic fixture: 4 million visits of capacity with 3 million in use.
    assert ap.infrastructure_headroom(4.0, 3.0, planned_addition=2.5) == pytest.approx(3.5)
    assert ap.infrastructure_headroom(4.0, 9.0, planned_addition=1.0) == pytest.approx(0.0)


def test_headroom_rejects_negative_capacity_or_use():
    with pytest.raises(ValueError):
        ap.infrastructure_headroom(-1.0, 0.0)
    with pytest.raises(ValueError):
        ap.infrastructure_headroom(1.0, -0.5)


def test_capacity_cap_never_exceeds_demand_or_headroom():
    # Synthetic fixture: demand and headroom are invented for this test.
    demand = np.array([0.5, 2.0, 3.0, 0.0])
    headroom = np.array([1.0, 1.0, 4.0, 2.0])
    capped = ap.capacity_constrained_visits(demand, headroom)
    npt.assert_allclose(capped, [0.5, 1.0, 3.0, 0.0])
    assert np.all(capped <= demand) and np.all(capped <= headroom)


def test_capacity_cap_rejects_negative_demand_or_headroom():
    with pytest.raises(ValueError):
        ap.capacity_constrained_visits([-1.0], [2.0])
    with pytest.raises(ValueError):
        ap.capacity_constrained_visits([1.0], [-2.0])


def test_accommodation_cap_matches_a_hand_calculation():
    # Synthetic fixture: 1,000 rooms at 50 percent occupancy over a 365-day year,
    # 2 guests per room and stays of 2 nights.
    # Room nights: 1000 * 0.5 * 365 = 182,500. Guest nights: 365,000. Visits: 182,500, or 0.1825 million.
    npt.assert_allclose(ap.accommodation_visit_cap([1000.0, 2000.0], 0.5, 2.0, 2.0), [0.1825, 0.365])


def test_accommodation_cap_rejects_occupancy_outside_unit_interval_and_zero_nights():
    with pytest.raises(ValueError):
        ap.accommodation_visit_cap(100.0, 1.2, 2.0, 2.0)
    with pytest.raises(ValueError):
        ap.accommodation_visit_cap(100.0, -0.1, 2.0, 2.0)
    with pytest.raises(ValueError):
        ap.accommodation_visit_cap(100.0, 0.5, 0.0, 2.0)


def test_capped_visits_respect_the_accommodation_limit():
    # Synthetic fixture: the cap of 0.1825 million visits comes from the hand calculation above.
    cap = ap.accommodation_visit_cap(1000.0, 0.5, 2.0, 2.0)
    capped = ap.capacity_constrained_visits([0.05, 0.5], cap)
    npt.assert_allclose(capped, [0.05, 0.1825])


def test_sourced_rows_carry_a_value_and_a_sourced_status():
    table = ap.load_inputs(INPUTS)
    sourced = table["status"].str.startswith("sourced")
    assert sourced.any()
    assert table.loc[sourced, "value"].notna().all()


def test_sourced_thai_figures_match_their_cited_values():
    table = ap.load_inputs(INPUTS)
    assert table.loc["hotel_guesthouse_rooms_2022", "value"] == 956372


def test_secondary_relay_figures_stay_pending_until_the_primary_source_is_fetched():
    table = ap.load_inputs(INPUTS)
    relay_rows = [
        "utapao_passengers_oct2023_sep2024",
        "suvarnabhumi_passengers_2019",
        "don_mueang_design_capacity_passengers_per_year",
        "aot_six_airports_passengers_fy2023",
        "chonburi_tourists_jan_nov_2024",
        "chonburi_foreign_visitors_jan_nov_2024",
        "tat_spend_per_trip_all_nationalities_2023",
        "tat_spend_per_trip_middle_eastern_2023",
        "tat_spend_per_day_uae_2023",
        "mots_spend_per_trip_all_tourists_2019",
        "mots_spend_per_day_all_tourists_2019",
        "intl_avg_length_of_stay_2016_2017_days",
        "intl_arrivals_2025",
        "intl_tourism_receipts_2025_mots",
    ]
    for name in relay_rows:
        assert pd.isna(table.loc[name, "value"]), name
        assert table.loc[name, "status"].startswith("pending"), name


def test_accommodation_base_case_stays_pending_without_a_tat_occupancy_figure():
    table = ap.load_inputs(INPUTS)
    assert pd.isna(table.loc["chonburi_occupancy_2024_tat", "value"])
    with pytest.raises(ValueError, match="chonburi_occupancy_2024_tat"):
        ap.require_values(table, ["chonburi_occupancy_2024_tat"])


def test_airport_passengers_convert_to_round_trip_visits():
    # 3.7 million passenger movements, two per round-trip visit, give 1.85 million visits.
    npt.assert_allclose(ap.visits_from_passengers(3_700_000.0, 2.0), 1.85)


def test_passenger_conversion_rejects_negative_passengers_and_zero_movements():
    with pytest.raises(ValueError):
        ap.visits_from_passengers([-1.0], 2.0)
    with pytest.raises(ValueError):
        ap.visits_from_passengers([1.0], 0.0)


def test_utapao_capacity_is_halved_before_it_enters_the_headroom():
    # The capacity row is in passenger movements, so the headroom must use the converted value.
    table = ap.load_inputs(INPUTS)
    capacity_passengers = float(table.loc["utapao_terminal_capacity_passengers_per_year", "value"])
    visits = ap.visits_from_passengers(capacity_passengers, 2.0)
    npt.assert_allclose(ap.infrastructure_headroom(visits, 0.0), 1.85)


def test_capped_net_spend_limits_visits_to_headroom():
    # Synthetic fixture: 10 million visits at a share of 0.1 give 1.0 million incremental visits.
    # The headroom of 0.4 million binds: 0.4 million visits at THB 10,000 each give THB 4 billion.
    capped, spend = ap.capped_net_spend(10.0, 0.1, [0.4, 5.0], 10_000.0, 0.0)
    npt.assert_allclose(capped, [0.4, 1.0])
    npt.assert_allclose(spend, [4.0, 10.0])


def test_capped_net_spend_never_exceeds_the_uncapped_spend():
    uncapped = ap.net_tourism_spend(ap.incremental_visitors(10.0, 0.1), 10_000.0, 0.25)
    _, spend = ap.capped_net_spend(10.0, 0.1, [0.4, 5.0], 10_000.0, 0.25)
    assert np.all(spend <= uncapped + 1e-12)


def test_capped_net_spend_rejects_negative_headroom():
    with pytest.raises(ValueError):
        ap.capped_net_spend(10.0, 0.1, [-1.0], 10_000.0, 0.0)
