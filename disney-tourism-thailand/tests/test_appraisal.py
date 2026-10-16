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
ASSUMPTIONS = ROOT / "data" / "appraisal" / "assumptions.csv"
CBRE_ROWS = ("pattaya_hotel_keys_h2_2025",)
PROXY_ROWS = ("pattaya_hotel_keys_h2_2024", "pattaya_occupancy_2024_cbre")

# Synthetic fixture values: every number below is invented for these tests and is not an estimate.
SYNTHETIC_GRID = {
    "project_life_years": 3,
    "construction_years": 2,
    "wacc_financial": 0.10,
    "attendance_million_per_year": 5.0,
    "international_share": 0.5,
    "incremental_share": 0.5,
    "displacement_share": 0.25,
    "spend_per_visit_thb": 10_000.0,
    "domestic_retention_share": 0.5,
    "tax_share_of_spend": 0.1,
    "opex_share": 0.5,
    "ticket_revenue_per_visit_thb": 2_000.0,
    "nights_per_stay": 2.0,
    "guests_per_room": 2.0,
    "movements_per_visit": 2.0,
    "current_airport_use_passengers_per_year": 1_000_000.0,
    "planned_airport_addition_passengers_per_year": 0.0,
    "public_cost_after_construction_thb_bn_per_year": 0.5,
    "rooms_serving_site": 10_000.0,
    "occupancy_serving_site": 0.8,
    "occupancy_ceiling_serving_site": 1.0,
}


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
    # The refusal rule is unchanged. The rows used here are still blank, so the rule applies to them.
    table = ap.load_inputs(INPUTS)
    with pytest.raises(ValueError, match="tat_spend_per_night_by_nationality_2024_2025, intl_first_visit_share_2024"):
        ap.require_values(table, ["tat_spend_per_night_by_nationality_2024_2025", "intl_first_visit_share_2024"])


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


def test_project_life_and_financial_rate_are_team_decisions_and_not_sourced():
    # The leader confirmed the team recommendations on 14 October 2026. They are labelled decisions, not sourced values.
    table = ap.load_inputs(INPUTS)
    expected = {"project_life_years": 20.0, "wacc_financial": 0.08}
    for name, value in expected.items():
        assert table.loc[name, "value"] == pytest.approx(value), name
        assert table.loc[name, "status"].startswith("team decision"), name
        assert not table.loc[name, "status"].startswith("sourced"), name
        assert ap.require_values(table, [name])[name] == pytest.approx(value)


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


def test_secondary_relay_figures_stay_unconfirmed_until_the_primary_source_is_fetched():
    table = ap.load_inputs(INPUTS)
    # These rows still rest on a relay (a news or aggregator page, or a Ministry figure read through a relay) and have no primary value.
    # The AOT, U-Tapao, Don Mueang, MoTS and national statistics rows were resolved from primary documents and are sourced now.
    relay_rows = [
        "chonburi_tourists_jan_nov_2024",
        "chonburi_foreign_visitors_jan_nov_2024",
        "tat_spend_per_trip_middle_eastern_2023",
        "tat_spend_per_day_uae_2023",
    ]
    # The candidate values stay in the table, flagged unconfirmed, so none of them counts as sourced.
    for name in relay_rows:
        assert pd.notna(table.loc[name, "value"]), name
        assert table.loc[name, "status"].startswith("unconfirmed"), name


def test_tat_spend_per_trip_is_a_secondary_sourced_value_that_the_model_does_not_use():
    table = ap.load_inputs(INPUTS)
    name = "tat_spend_per_trip_all_nationalities_2023"
    assert table.loc[name, "status"].startswith("sourced (secondary")
    assert table.loc[name, "value"] == pytest.approx(50900.0)
    assert name not in ap.GRID_NAMES and name not in ap.SOURCED_NAMES


def test_chon_buri_occupancy_is_a_sourced_value_that_the_base_case_can_use():
    table = ap.load_inputs(INPUTS)
    assert table.loc["chonburi_occupancy_2024_tat", "status"].startswith("sourced (National Statistical Office")
    assert table.loc["chonburi_occupancy_2024_tat", "value"] == pytest.approx(80.42)
    assert ap.require_values(table, ["chonburi_occupancy_2024_tat"])["chonburi_occupancy_2024_tat"] == pytest.approx(80.42)


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


def _synthetic_sourced(**overrides):
    """Return an input table of synthetic values. The values are invented for these tests."""
    values = {
        "utapao_terminal_capacity_passengers_per_year": 4_000_000.0,
        "capex_complex_thb_bn": 10.0,
        "discount_rate_social_nesdc": 0.05,
        "conversion_factor_standard": 0.7,
    }
    values.update(overrides)
    names = list(values)
    return pd.DataFrame(
        {
            "value": [values[n] for n in names],
            "unit": ["synthetic"] * len(names),
            "source": ["synthetic fixture"] * len(names),
            "status": ["sourced (synthetic fixture)"] * len(names),
        },
        index=pd.Index(names, name="parameter"),
    )


def _synthetic_assumptions(**overrides):
    """Return an assumptions table with the same synthetic value in all three cases."""
    values = {**SYNTHETIC_GRID, **overrides}
    names = list(values)
    return pd.DataFrame(
        {
            "low_case": [float(values[n]) for n in names],
            "base_case": [float(values[n]) for n in names],
            "high_case": [float(values[n]) for n in names],
            "unit": ["synthetic"] * len(names),
            "basis": ["synthetic fixture, not an estimate"] * len(names),
            "status": ["assumption (synthetic fixture; not sourced)"] * len(names),
        },
        index=pd.Index(names, name="parameter"),
    )


def test_synthetic_case_matches_a_hand_calculation():
    # Synthetic fixture. Free rooms: 10,000 * (1 - 0.8) = 2,000, the ceiling being 1.0. Accommodation cap: 2,000 * 365 * 2 / 2 / 1e6 = 0.73 million visits.
    # Headroom: 4,000,000 - 1,000,000 = 3,000,000 passengers. Airport cap: 3,000,000 / 2 / 1e6 = 1.5 million visits. Accommodation binds.
    # Physical cap: the smaller of 4,000,000 / 2 / 1e6 = 2.0 and 10,000 * 365 * 2 / 2 / 1e6 = 3.65 million visits, so 2.0 binds.
    # International visits: 5 * 0.5 = 2.5 million. To the complex: min(2.5, 2.0) = 2.0 million. Lost to the operator: 0.5 million.
    # Incremental demand: 2.0 * 0.5 = 1.0 million, capped to the headroom of 0.73. Above the headroom: 0.27 million.
    # Net spend: 0.73 * 10,000 / 1000 * (1 - 0.25) = 5.475 THB bn per year.
    # Operator visits: 5 - 0.5 = 4.5 million. Revenue: 4.5 * 2,000 / 1000 = 9.0 THB bn. Opex: 0.5 * 9.0 = 4.5.
    # National operating cost: the whole operating cost, 4.5 THB bn.
    # Construction 2 years and life 3 years make 5 years. Capex 10 / 2 = 5 in each construction year.
    sourced = _synthetic_sourced()
    assumptions = _synthetic_assumptions()
    inp = ap.case_inputs(sourced, assumptions, "base_case")
    annual = ap.case_annual(inp)
    npt.assert_allclose([annual["accom_cap_m"], annual["airport_cap_m"], annual["infra_cap_m"]], [0.73, 1.5, 0.73])
    assert annual["binding_constraint"] == "accommodation"
    npt.assert_allclose(annual["physical_cap_m"], 2.0)
    npt.assert_allclose(
        [annual["intl_m"], annual["intl_to_complex_m"], annual["incr_demand_m"], annual["incr_m"], annual["excluded_m"], annual["headroom_excess_m"]],
        [2.5, 2.0, 1.0, 0.73, 0.5, 0.27],
    )
    npt.assert_allclose(annual["net_spend_bn_per_year"], 5.475)
    npt.assert_allclose(annual["operator_visits_m"], 4.5)
    npt.assert_allclose(annual["revenue_bn_per_year"], 9.0)
    npt.assert_allclose(annual["opex_bn_per_year"], 4.5)
    npt.assert_allclose(annual["national_opex_bn_per_year"], 4.5)

    flows = ap.case_flows(inp, annual)
    for name in ("capex", "revenue", "opex", "national_opex", "net_spend", "public", "zeros"):
        assert len(flows[name]) == 5, name
    npt.assert_allclose(flows["capex"], [5.0, 5.0, 0.0, 0.0, 0.0])
    npt.assert_allclose(flows["revenue"], [0.0, 0.0, 9.0, 9.0, 9.0])
    npt.assert_allclose(flows["opex"], [0.0, 0.0, 4.5, 4.5, 4.5])
    npt.assert_allclose(flows["national_opex"], [0.0, 0.0, 4.5, 4.5, 4.5])
    npt.assert_allclose(flows["net_spend"], [0.0, 0.0, 5.475, 5.475, 5.475])
    npt.assert_allclose(flows["public"], [0.0, 0.0, 0.5, 0.5, 0.5])

    result = ap.labelled_case(sourced, assumptions, "base_case")
    # Operator flows: -5, -5, then revenue less opex of 9.0 - 4.5 = 4.5 in each operating year, discounted at 10 percent term by term.
    expected_operator = -5.0 - 5.0 / 1.1 + 4.5 * (1 / 1.1**2 + 1 / 1.1**3 + 1 / 1.1**4)
    assert result["operator_npv_bn"] == pytest.approx(expected_operator, rel=1e-12)
    # National flows: 0.5 * net spend - 0.7 * (capex + national operating cost), discounted at 5 percent.
    national = [-3.5, -3.5] + [0.5 * 5.475 - 0.7 * 4.5] * 3
    expected_national = sum(f / 1.05**t for t, f in enumerate(national))
    assert result["national_npv_bn"] == pytest.approx(expected_national, rel=1e-12)
    # Fiscal flows without the capital cost: 0.1 * net spend - public cost, discounted at 10 percent.
    fiscal = [0.0, 0.0] + [0.1 * 5.475 - 0.5] * 3
    expected_fiscal_excl = sum(f / 1.1**t for t, f in enumerate(fiscal))
    assert result["fiscal_npv_excl_public_capex_bn"] == pytest.approx(expected_fiscal_excl, rel=1e-12)
    # The primary fiscal ledger charges the complex capital cost in the construction years.
    capex = [5.0, 5.0, 0.0, 0.0, 0.0]
    expected_fiscal = sum((f - c) / 1.1**t for t, (f, c) in enumerate(zip(fiscal, capex)))
    assert result["fiscal_npv_bn"] == pytest.approx(expected_fiscal, rel=1e-12)
    assert result["case"] == "base_case" and result["assumption_driven"] is True


def test_synthetic_operator_irr_is_a_root_and_zero_rate_npv_is_the_sum():
    inp = ap.case_inputs(_synthetic_sourced(), _synthetic_assumptions(), "base_case")
    flows = ap.case_flows(inp, ap.case_annual(inp))
    operator = ap.operator_flows(flows["revenue"], flows["opex"], flows["capex"])
    rate = ap.irr(operator)
    assert np.isfinite(rate)
    assert ap.npv(operator, rate) == pytest.approx(0.0, abs=1e-6)
    assert ap.npv(operator, 0.0) == pytest.approx(float(np.sum(operator)))


def test_synthetic_binding_constraint_is_the_smaller_cap():
    # Accommodation cap 0.73 against airport cap 1.5: accommodation binds.
    default = ap.case_annual(ap.case_inputs(_synthetic_sourced(), _synthetic_assumptions(), "base_case"))
    assert default["binding_constraint"] == "accommodation"
    assert default["infra_cap_m"] == pytest.approx(0.73)
    # Airport capacity of 2 million passengers gives 0.5 million visits of headroom, so the airport cap of 0.5 binds.
    airport = ap.case_annual(
        ap.case_inputs(_synthetic_sourced(utapao_terminal_capacity_passengers_per_year=2_000_000.0), _synthetic_assumptions(), "base_case")
    )
    assert airport["binding_constraint"] == "airport"
    assert airport["infra_cap_m"] == pytest.approx(0.5)
    assert airport["incr_m"] == pytest.approx(0.5)
    assert airport["excluded_m"] == pytest.approx(2.5 - 1.0)
    assert airport["headroom_excess_m"] == pytest.approx(0.0)
    # A tie gives the airport. Synthetic fixture: both caps are 1.825 million visits.
    tie = ap.case_annual(
        ap.case_inputs(
            _synthetic_sourced(utapao_terminal_capacity_passengers_per_year=3_650_000.0),
            _synthetic_assumptions(current_airport_use_passengers_per_year=0.0, nights_per_stay=1.0, guests_per_room=1.0, occupancy_serving_site=0.5),
            "base_case",
        )
    )
    assert tie["airport_cap_m"] == pytest.approx(1.825) and tie["accom_cap_m"] == pytest.approx(1.825)
    assert tie["binding_constraint"] == "airport"


def test_pending_sourced_value_stops_a_labelled_case():
    with pytest.raises(ValueError, match="utapao_terminal_capacity_passengers_per_year"):
        ap.case_inputs(_synthetic_sourced(utapao_terminal_capacity_passengers_per_year=np.nan), _synthetic_assumptions(), "base_case")


def test_pending_proposal_value_stops_a_labelled_case():
    with pytest.raises(ValueError, match="capex_complex_thb_bn"):
        ap.case_inputs(_synthetic_sourced(capex_complex_thb_bn=np.nan), _synthetic_assumptions(), "base_case")


def test_grid_value_sourced_in_the_input_table_is_refused():
    with pytest.raises(ValueError, match="must not also be a grid value"):
        ap.case_inputs(_synthetic_sourced(project_life_years=3.0), _synthetic_assumptions(), "base_case")


def test_unknown_case_and_non_whole_years_are_refused():
    with pytest.raises(ValueError):
        ap.case_inputs(_synthetic_sourced(), _synthetic_assumptions(), "mid_case")
    with pytest.raises(ValueError, match="whole number of years"):
        ap.case_inputs(_synthetic_sourced(), _synthetic_assumptions(construction_years=2.5), "base_case")


def test_load_assumptions_reads_the_three_cases_as_floats():
    table = ap.load_assumptions(ASSUMPTIONS)
    assert table.index.name == "parameter"
    for case in ap.CASES:
        assert table[case].dtype == np.float64
        assert np.isfinite(table[case]).all()


def test_real_labelled_cases_give_three_finite_rows():
    df = ap.run_labelled_cases(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS))
    assert list(df["case"]) == list(ap.CASES)
    numeric = [
        "accom_cap_m",
        "airport_cap_m",
        "infra_cap_m",
        "intl_m",
        "incr_demand_m",
        "incr_m",
        "excluded_m",
        "net_spend_bn_per_year",
        "revenue_bn_per_year",
        "operator_visits_m",
        "operator_npv_bn",
        "national_npv_bn",
        "fiscal_npv_bn",
        "fiscal_npv_excl_public_capex_bn",
        "physical_cap_m",
        "intl_to_complex_m",
        "headroom_excess_m",
    ]
    assert np.isfinite(df[numeric].to_numpy()).all()


def test_real_low_case_has_no_irr_because_its_operating_flows_are_negative():
    sourced, assumptions = ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS)
    df = ap.run_labelled_cases(sourced, assumptions)
    assert np.isnan(df.loc[df["case"] == "low_case", "operator_irr"].iloc[0])
    flows = ap.case_flows(ap.case_inputs(sourced, assumptions, "low_case"), ap.case_annual(ap.case_inputs(sourced, assumptions, "low_case")))
    operating = flows["revenue"] > 0
    assert np.all(flows["revenue"][operating] - flows["opex"][operating] < 0)


def test_real_cases_are_assumption_driven_and_list_their_labels():
    df = ap.run_labelled_cases(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS))
    assert df["assumption_driven"].all()
    for labels in df["labels"]:
        assert len(labels) == len(ap.SOURCED_NAMES) + len(ap.PROPOSAL_NAMES) + len(ap.GRID_NAMES)
        assert any(label.startswith("sourced:") for label in labels)
        assert any(label.startswith("assumption:") for label in labels)
        assert any(label.startswith("generalized:") for label in labels)
        assert any(label.startswith("other:") for label in labels)


def test_real_binding_constraint_is_the_smaller_of_the_two_caps():
    df = ap.run_labelled_cases(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS))
    for _, row in df.iterrows():
        expected = "airport" if row["airport_cap_m"] <= row["accom_cap_m"] else "accommodation"
        assert row["binding_constraint"] == expected, row["case"]
        assert row["infra_cap_m"] == pytest.approx(min(row["airport_cap_m"], row["accom_cap_m"]))
        assert row["incr_m"] <= row["incr_demand_m"] + 1e-12
        assert row["incr_m"] <= row["infra_cap_m"] + 1e-12
        assert row["intl_to_complex_m"] + row["excluded_m"] == pytest.approx(row["intl_m"])
        assert row["intl_to_complex_m"] <= row["physical_cap_m"] + 1e-12
        assert row["headroom_excess_m"] == pytest.approx(row["incr_demand_m"] - row["incr_m"])
        assert row["incr_demand_m"] <= row["intl_m"] + 1e-12


def test_real_operator_npv_runs_from_pessimistic_to_optimistic():
    df = ap.run_labelled_cases(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS))
    low, base, high = df["operator_npv_bn"]
    assert low < base < high


def test_real_fiscal_primary_is_below_its_variant_and_the_high_case_beats_the_base():
    # The low case is not the worst fiscal case. Its higher discount rate and longer build reduce the present value of the capital cost,
    # so its primary fiscal NPV can exceed the base. The test checks that the capital cost lowers every case and that the high case beats the base.
    df = ap.run_labelled_cases(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS))
    assert (df["fiscal_npv_bn"] < df["fiscal_npv_excl_public_capex_bn"]).all()
    base = df.loc[df["case"] == "base_case", "fiscal_npv_bn"].iloc[0]
    high = df.loc[df["case"] == "high_case", "fiscal_npv_bn"].iloc[0]
    assert base < high


def test_real_national_npv_is_negative_in_the_base_case_because_the_whole_operating_cost_is_charged():
    # The national ledger charges the whole operating cost and credits only the net spending of the incremental visits.
    df = ap.run_labelled_cases(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS))
    assert np.isfinite(df["national_npv_bn"]).all()
    base = df.loc[df["case"] == "base_case", "national_npv_bn"].iloc[0]
    assert base < 0


def test_real_cbre_rows_read_as_sourced_with_their_values():
    table = ap.load_inputs(INPUTS)
    for name in CBRE_ROWS:
        status = table.loc[name, "status"]
        assert status.startswith("sourced (CBRE"), name
        assert "broker" in status, name
        assert pd.notna(table.loc[name, "value"]), name
    for name in PROXY_ROWS:
        assert str(table.loc[name, "status"]).startswith("generalized (CBRE"), name


def test_every_assumption_row_is_labelled_as_an_assumption_or_a_generalization():
    table = ap.load_assumptions(ASSUMPTIONS)
    for name, status in table["status"].items():
        assert status.startswith("assumption") or status.startswith("generalized"), name


def test_no_grid_row_is_marked_sourced():
    table = ap.load_assumptions(ASSUMPTIONS)
    for name, status in table["status"].items():
        assert not status.startswith("sourced"), name


def test_assumption_table_covers_exactly_the_grid_parameters():
    table = ap.load_assumptions(ASSUMPTIONS)
    assert set(table.index) == set(ap.GRID_NAMES)
    assert table[list(ap.CASES)].notna().all().all()


def test_grid_parameters_are_not_sourced_in_the_input_table():
    inputs = ap.load_inputs(INPUTS)
    for name in ap.load_assumptions(ASSUMPTIONS).index:
        if name in inputs.index:
            assert not str(inputs.loc[name, "status"]).startswith("sourced"), name


def test_physical_cap_limits_all_international_visits_and_headroom_limits_the_incremental_ones():
    # The physical cap applies to all international visits. Spending uses the incremental share of the capped visits,
    # and the infrastructure headroom caps those incremental visits.
    sourced, assumptions = ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS)
    for case in ap.CASES:
        inp = ap.case_inputs(sourced, assumptions, case)
        annual = ap.case_annual(inp)
        intl = inp["attendance_million_per_year"] * inp["international_share"]
        to_complex = min(intl, annual["physical_cap_m"])
        assert annual["intl_to_complex_m"] == pytest.approx(to_complex)
        assert annual["excluded_m"] == pytest.approx(intl - to_complex)
        assert annual["operator_visits_m"] == pytest.approx(inp["attendance_million_per_year"] - annual["excluded_m"])
        demand = to_complex * inp["incremental_share"]
        assert annual["incr_demand_m"] == pytest.approx(demand)
        assert annual["incr_m"] == pytest.approx(min(demand, annual["infra_cap_m"]))
        expected_spend = annual["incr_m"] * inp["spend_per_visit_thb"] / 1000.0 * (1.0 - inp["displacement_share"])
        assert annual["net_spend_bn_per_year"] == pytest.approx(expected_spend)


def test_base_case_net_spend_counts_only_the_capped_incremental_visits():
    # Base case: free rooms 31,581 * (0.85 - 0.8042) = 1,446.4, so the accommodation cap is 1,446.4 * 365 * 2.0 / 10.66 / 1e6 = 0.0990506 million visits.
    # The physical cap is the airport total of 3,700,000 / 2 / 1e6 = 1.85 million, so the international visits 4.425 million are capped to 1.85 million.
    # Incremental demand: 1.85 * 0.30 = 0.555 million, capped to 0.0990506 million. Net spend: 0.0990506 * 10,000 / 1000 * (1 - 0.3) = 0.693354 THB bn per year.
    sourced, assumptions = ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS)
    annual = ap.case_annual(ap.case_inputs(sourced, assumptions, "base_case"))
    assert annual["binding_constraint"] == "accommodation"
    assert annual["physical_cap_m"] == pytest.approx(1.85, abs=1e-9)
    assert annual["incr_m"] == pytest.approx(0.0990506, abs=1e-6)
    assert annual["net_spend_bn_per_year"] == pytest.approx(0.693354, abs=5e-5)


def test_national_ledger_carries_the_whole_operating_cost_so_it_falls_when_visits_above_the_cap_rise():
    # Synthetic fixture: the physical and accommodation caps bind at both attendances, so the incremental visits and the net spending
    # are the same. The operator visits and the operating cost rise, and the national ledger carries that cost.
    low = ap.labelled_case(_synthetic_sourced(), _synthetic_assumptions(), "base_case")
    high = ap.labelled_case(_synthetic_sourced(), _synthetic_assumptions(attendance_million_per_year=8.0), "base_case")
    assert high["incr_m"] == pytest.approx(low["incr_m"])
    assert high["net_spend_bn_per_year"] == pytest.approx(low["net_spend_bn_per_year"])
    assert high["operator_visits_m"] > low["operator_visits_m"]
    assert high["national_npv_bn"] < low["national_npv_bn"]


def test_national_operating_cost_is_the_whole_operating_cost_of_the_complex():
    annual = ap.case_annual(ap.case_inputs(_synthetic_sourced(), _synthetic_assumptions(), "base_case"))
    assert annual["national_opex_bn_per_year"] == pytest.approx(annual["opex_bn_per_year"])
    assert annual["national_opex_bn_per_year"] == pytest.approx(4.5)


def test_primary_fiscal_ledger_includes_the_capital_cost_and_the_variant_leaves_it_out():
    # Synthetic fixture: capex of 5 in each of two construction years, discounted at the 10 percent financial rate.
    result = ap.labelled_case(_synthetic_sourced(), _synthetic_assumptions(), "base_case")
    assert result["fiscal_npv_excl_public_capex_bn"] - result["fiscal_npv_bn"] == pytest.approx(5.0 + 5.0 / 1.1)


def test_proposal_capital_cost_is_labelled_other_and_not_sourced():
    sourced = ap.load_inputs(INPUTS)
    assert "capex_complex_thb_bn" in ap.PROPOSAL_NAMES
    assert "capex_complex_thb_bn" not in ap.SOURCED_NAMES
    assert sourced.loc["capex_complex_thb_bn", "status"].startswith("from proposal")
    result = ap.labelled_case(sourced, ap.load_assumptions(ASSUMPTIONS), "base_case")
    assert any(label.startswith("other: capex_complex_thb_bn") for label in result["labels"])
    assert result["assumption_driven"] is True


def test_room_stock_is_the_pattaya_proxy_and_occupancy_is_the_chon_buri_figure():
    inputs = ap.load_inputs(INPUTS)
    assumptions = ap.load_assumptions(ASSUMPTIONS)
    for name in ("rooms_serving_site", "occupancy_serving_site"):
        assert assumptions.loc[name, "status"].startswith("generalized ("), name
        assert name not in ap.SOURCED_NAMES and name not in ap.PROPOSAL_NAMES
    assert assumptions.loc["rooms_serving_site", "base_case"] == pytest.approx(inputs.loc["pattaya_hotel_keys_h2_2024", "value"])
    assert assumptions.loc["occupancy_serving_site", "base_case"] == pytest.approx(inputs.loc["chonburi_occupancy_2024_tat", "value"] / 100.0)


def test_not_fetched_rows_are_not_listed_as_plain_sourced_values():
    table = ap.load_inputs(INPUTS)
    for name, status in table["status"].items():
        text = str(status)
        if "not fetched" in text and text.startswith("sourced"):
            assert "search" in text, name


def test_input_tables_carry_no_process_notes_or_local_paths():
    for path in (INPUTS, ASSUMPTIONS):
        text = path.read_text(encoding="utf-8")
        for phrase in ("accepted by the leader", "Searches attempted", "copy read locally", "Downloads", "D:\\", "No value entered", "no value entered", "so no value", " lanes", "in this session"):
            assert phrase not in text, (path.name, phrase)


def test_nights_per_stay_basis_is_an_unsourced_placeholder():
    basis = ap.load_assumptions(ASSUMPTIONS).loc["nights_per_stay", "basis"]
    assert "unsourced placeholder" in basis
    assert "302" not in basis


def test_physical_cap_is_the_smaller_of_total_airport_and_total_room_stock():
    # Airport total: 3,700,000 / 2 / 1e6 = 1.85 million visits. Room stock at full occupancy: 31,581 * 365 * 2 / 10.66 / 1e6 = 2.163 million visits.
    assert ap.physical_visit_cap(3_700_000.0, 2.0, 31_581.0, 10.66, 2.0) == pytest.approx(1.85)
    # A small room stock binds instead: 1,000 * 365 * 2 / 2 / 1e6 = 0.365 million visits.
    assert ap.physical_visit_cap(3_700_000.0, 2.0, 1_000.0, 2.0, 2.0) == pytest.approx(0.365)
    with pytest.raises(ValueError):
        ap.physical_visit_cap(3_700_000.0, 0.0, 31_581.0, 10.66, 2.0)


def test_occupancy_ceiling_sets_the_free_room_stock_and_no_free_rooms_give_no_headroom():
    # Free rooms: 10,000 * (0.85 - 0.70) = 1,500. Accommodation cap: 1,500 * 365 * 2 / 2 / 1e6 = 0.5475 million visits.
    sourced = _synthetic_sourced()
    annual = ap.case_annual(ap.case_inputs(sourced, _synthetic_assumptions(occupancy_serving_site=0.70, occupancy_ceiling_serving_site=0.85), "base_case"))
    assert annual["accom_cap_m"] == pytest.approx(0.5475)
    # Occupancy above the ceiling leaves no free room, so the accommodation headroom and the incremental visits are zero.
    full = ap.case_annual(ap.case_inputs(sourced, _synthetic_assumptions(occupancy_serving_site=0.9, occupancy_ceiling_serving_site=0.85), "base_case"))
    assert full["accom_cap_m"] == pytest.approx(0.0)
    assert full["incr_m"] == pytest.approx(0.0)


def test_case_flows_scale_each_factor_and_the_delay_moves_only_the_operations():
    inp = ap.case_inputs(_synthetic_sourced(), _synthetic_assumptions(), "base_case")
    annual = ap.case_annual(inp)
    flows = ap.case_flows(inp, annual, capex_factor=1.1, opex_factor=0.9, revenue_factor=0.8, delay_years=1)
    # Capital spending keeps its timing; the complex opens one year later, so the operating flows start in year 3.
    npt.assert_allclose(flows["capex"], [5.5, 5.5, 0.0, 0.0, 0.0, 0.0])
    npt.assert_allclose(flows["revenue"], [0.0, 0.0, 0.0, 9.0 * 0.8, 9.0 * 0.8, 9.0 * 0.8])
    npt.assert_allclose(flows["opex"], [0.0, 0.0, 0.0, 4.5 * 0.9, 4.5 * 0.9, 4.5 * 0.9])
    npt.assert_allclose(flows["national_opex"], [0.0, 0.0, 0.0, 4.5 * 0.9, 4.5 * 0.9, 4.5 * 0.9])
    with pytest.raises(ValueError):
        ap.case_flows(inp, annual, capex_factor=-0.1)
    with pytest.raises(ValueError):
        ap.case_flows(inp, annual, delay_years=0.5)


def test_sensitivity_table_has_the_nesdc_cases_and_its_base_row_matches_the_labelled_case():
    sourced, assumptions = ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS)
    table = ap.sensitivity_table(sourced, assumptions, "base_case")
    # Base, then 5, 10 and 20 percent up and down for three single factors, two two-factor pairs for each percentage, and one delay.
    assert len(table) == 1 + 3 * 6 + 2 * 3 + 1
    base = table.iloc[0]
    labelled = ap.labelled_case(sourced, assumptions, "base_case")
    assert base["sensitivity"] == "base"
    assert base["operator_npv_bn"] == pytest.approx(labelled["operator_npv_bn"], rel=1e-12)
    assert base["national_npv_bn"] == pytest.approx(labelled["national_npv_bn"], rel=1e-12)
    assert base["fiscal_npv_bn"] == pytest.approx(labelled["fiscal_npv_bn"], rel=1e-12)
    assert base["fiscal_npv_excl_public_capex_bn"] == pytest.approx(labelled["fiscal_npv_excl_public_capex_bn"], rel=1e-12)


def test_sensitivity_directions_move_the_npvs_the_right_way():
    table = ap.sensitivity_table(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS), "base_case").set_index("sensitivity")
    base = table.loc["base"]
    assert table.loc["construction cost up 20 percent", "fiscal_npv_bn"] < base["fiscal_npv_bn"]
    assert table.loc["revenue up 20 percent", "operator_npv_bn"] > base["operator_npv_bn"]
    assert table.loc["revenue down 20 percent", "operator_npv_bn"] < base["operator_npv_bn"]
    assert table.loc["operating and maintenance cost up 20 percent", "operator_npv_bn"] < base["operator_npv_bn"]
    assert table.loc["one-year delay in opening", "operator_npv_bn"] < base["operator_npv_bn"]


def test_two_factor_cases_pair_construction_cost_with_revenue_in_both_directions():
    table = ap.sensitivity_table(ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS), "base_case").set_index("sensitivity")
    adverse = table.loc["two factors: construction cost up 10 percent and revenue down 10 percent"]
    favourable = table.loc["two factors: construction cost down 10 percent and revenue up 10 percent"]
    assert adverse["operator_npv_bn"] < table.loc["base", "operator_npv_bn"] < favourable["operator_npv_bn"]
    assert adverse["fiscal_npv_bn"] < table.loc["base", "fiscal_npv_bn"] < favourable["fiscal_npv_bn"]


def test_input_tables_have_no_byte_order_mark_and_use_lf_line_endings():
    for path in (INPUTS, ASSUMPTIONS):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), path.name
        assert b"\r\n" not in raw, path.name


def test_lotte_world_adult_price_has_one_row_and_the_2026_price_is_sourced():
    table = ap.load_inputs(INPUTS)
    lotte = [name for name in table.index if "lotte_world" in name and "ticket" in name]
    assert lotte == ["lotte_world_adult_ticket_2026_krw"]
    assert table.loc["lotte_world_adult_ticket_2026_krw", "value"] == pytest.approx(67000.0)


def test_wacc_low_case_is_a_financial_rate_and_not_the_adb_economic_hurdle():
    assumptions = ap.load_assumptions(ASSUMPTIONS)
    assert assumptions.loc["wacc_financial", "low_case"] != pytest.approx(0.09)
    assert "discount_rate_eirr_standard" in assumptions.loc["wacc_financial", "basis"]


def test_occupancy_ceiling_is_an_assumed_grid_row_that_stays_above_the_occupancy():
    assumptions = ap.load_assumptions(ASSUMPTIONS)
    row = assumptions.loc["occupancy_ceiling_serving_site"]
    assert row["low_case"] <= row["base_case"] <= row["high_case"]
    assert row["base_case"] >= assumptions.loc["occupancy_serving_site", "base_case"]
    assert row["status"].startswith("assumption")
