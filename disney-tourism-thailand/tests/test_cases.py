"""Tests for dtt.cases: catalogue schema, features, imputation, donor pools and feasibility.

All data are SIMULATED by dtt.simulate_global or built by hand inside the tests.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dtt.cases import (  # noqa: E402
    CASE_COLUMNS,
    CATALOGUE_COLUMNS,
    CATEGORIES,
    EPISODE_COLUMNS,
    EPISODE_FIRST_YEAR,
    EPISODE_LAST_FEASIBLE_YEAR,
    FEATURE_COLUMNS,
    INTEGRATED_CATEGORIES,
    INVESTMENT_BASES,
    POST_HORIZON,
    build_episodes,
    case_features,
    check_feasibility,
    donor_pool,
    economy_feature_cloud,
    impute_features,
    load_cases,
    outcome_matrix,
    post_window_end,
    resolve_case,
    resolve_next_start,
)
from dtt.panel import build_global_panel  # noqa: E402
from dtt.simulate_global import simulate_global_world  # noqa: E402

LEVELS = ["log_gdp_pc", "log_pop", "receipts_pct_gdp", "log_receipts_per_arrival", "arrivals_per_capita", "air_pax_per_capita"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return simulate_global_world(tmp_path_factory.mktemp("world"), seed=2)


@pytest.fixture(scope="module")
def panel(world) -> pd.DataFrame:
    return build_global_panel(world.raw_dir)


@pytest.fixture(scope="module")
def cases(world) -> pd.DataFrame:
    return load_cases(world.cases_path)


@pytest.fixture(scope="module")
def plain_economy(panel, cases) -> str:
    """An economy that has no case and complete data in every indicator."""
    ok = panel.groupby("iso3")[["receipts_usd", "arrivals", "gdp_usd", "pop", "air_pax"]].apply(lambda g: bool(g.notna().all().all()))
    return sorted(c for c in ok.index[ok] if c not in set(cases["iso3"]))[0]


@pytest.fixture(scope="module")
def mid(panel, cases) -> pd.Series:
    """The feasible case with complete outcome data whose opening year is closest to the median."""
    feas = check_feasibility(cases, panel)
    full = feas.loc[feas["feasible"] & (feas["n_pre"] == feas["opening_year"] - 1995)]
    median = full["opening_year"].median()
    pick = full.iloc[(full["opening_year"] - median).abs().argsort(kind="stable")].iloc[0]["case_id"]
    return cases.loc[cases["case_id"] == pick].iloc[0]


def edited_catalogue(world, target: Path, edit) -> Path:
    """Copy the simulated catalogue to ``target`` after applying ``edit`` to the all-text frame."""
    raw = pd.read_csv(world.cases_path, dtype=str, keep_default_na=False)
    edit(raw)
    raw.to_csv(target, index=False)
    return target


# ----------------------------------------------------------------------------
# load_cases
# ----------------------------------------------------------------------------
def test_feature_columns_are_the_nine_documented_features_in_order():
    assert FEATURE_COLUMNS == (
        "log_gdp_pc",
        "log_pop",
        "receipts_pct_gdp",
        "log_receipts_per_arrival",
        "arrivals_per_capita",
        "air_pax_per_capita",
        "receipts_growth_pre",
        "capex_pct_gdp",
        "is_integrated_resort",
    )
    assert isinstance(FEATURE_COLUMNS, tuple) and "capex_first_pct_gdp" not in FEATURE_COLUMNS
    assert len(CASE_COLUMNS) == 21 and len(set(CASE_COLUMNS)) == 21
    assert set(INTEGRATED_CATEGORIES) < set(CATEGORIES)


def test_load_cases_returns_the_schema_with_numeric_types(cases):
    assert tuple(cases.columns) == CATALOGUE_COLUMNS
    assert CATALOGUE_COLUMNS == (*CASE_COLUMNS, "investment_usd_bn_used", "investment_basis", "price_year", "price_year_assumed")
    assert cases["opening_year"].dtype == np.int64
    assert cases["investment_usd_bn_used"].dtype == np.float64 and cases["investment_basis"].isin(INVESTMENT_BASES).all()
    for column in ("investment_usd_bn_nominal", "investment_year_basis", "first_year_attendance_m", "price_year"):
        assert cases[column].dtype == np.float64, column
    assert str(cases["price_year_assumed"].dtype) == "boolean"
    assert cases["price_year"].isna().all() and cases["price_year_assumed"].isna().all()
    assert str(cases["in_wdi_panel"].dtype) == "boolean" and str(cases["panel_feasible"].dtype) == "boolean"
    assert cases["source_url_2"].isna().all()
    assert cases["case_id"].is_unique
    assert set(cases["category"]) <= set(CATEGORIES)


@pytest.mark.parametrize(
    "edit,pattern",
    [
        (lambda df: df.drop(columns=["notes"], inplace=True), "missing"),
        (lambda df: df.insert(0, "extra", "x"), "unexpected"),
        (lambda df: df.__setitem__("case_id", ["C01"] * len(df)), "unique"),
        (lambda df: df.__setitem__("opening_year", ["2005.5"] + list(df["opening_year"][1:])), "whole numbers"),
        (lambda df: df.__setitem__("opening_year", ["abc"] + list(df["opening_year"][1:])), "numeric"),
        (lambda df: df.__setitem__("opening_year", [""] + list(df["opening_year"][1:])), "empty"),
        (lambda df: df.__setitem__("investment_usd_bn_nominal", ["1,5"] + list(df["investment_usd_bn_nominal"][1:])), "numeric"),
        (lambda df: df.__setitem__("investment_usd_bn_nominal", ["-1"] + list(df["investment_usd_bn_nominal"][1:])), "at least"),
        (lambda df: df.__setitem__("investment_year_basis", ["2004.5"] + list(df["investment_year_basis"][1:])), "whole numbers"),
        (lambda df: df.__setitem__("first_year_attendance_m", ["n/a"] + list(df["first_year_attendance_m"][1:])), "numeric"),
        (lambda df: df.__setitem__("category", ["zoo"] + list(df["category"][1:])), "category"),
        (lambda df: df.__setitem__("case_id", [""] + list(df["case_id"][1:])), "empty"),
        (lambda df: df.__setitem__("iso3", [""] + list(df["iso3"][1:])), "empty"),
        (lambda df: df.__setitem__("panel_feasible", ["maybe"] + list(df["panel_feasible"][1:])), "yes/no"),
    ],
)
def test_schema_violations_raise_value_error(world, tmp_path, edit, pattern):
    path = edited_catalogue(world, tmp_path / "cases.csv", edit)
    with pytest.raises(ValueError, match=pattern):
        load_cases(path)


def test_blank_numbers_and_flag_spellings_are_accepted(world, tmp_path):
    def edit(df):
        df.loc[0, "investment_usd_bn_nominal"] = ""
        df.loc[1, "first_year_attendance_m"] = " "
        df.loc[2, "investment_year_basis"] = ""
        df["in_wdi_panel"] = ["TRUE", "false", "1", "0", "Yes", "no", "y", "n"] + ["yes"] * (len(df) - 8)
        df.loc[3, "panel_feasible"] = ""

    cs = load_cases(edited_catalogue(world, tmp_path / "cases.csv", edit))
    assert cs.loc[0, "investment_usd_bn_nominal"] != cs.loc[0, "investment_usd_bn_nominal"]
    assert np.isnan(cs.loc[1, "first_year_attendance_m"]) and np.isnan(cs.loc[2, "investment_year_basis"])
    assert cs["in_wdi_panel"].tolist()[:8] == [True, False, True, False, True, False, True, False]
    assert cs["panel_feasible"].isna().tolist()[3] and cs["panel_feasible"].notna().sum() == len(cs) - 1


def test_text_valued_other_openings_column_is_kept_as_text(world, tmp_path):
    cs = load_cases(edited_catalogue(world, tmp_path / "cases.csv", lambda df: df.__setitem__(
        "other_openings_same_economy_within_5y", ["C02;C03"] + [""] * (len(df) - 1))))
    assert cs.loc[0, "other_openings_same_economy_within_5y"] == "C02;C03"
    assert cs["other_openings_same_economy_within_5y"].iloc[1:].isna().all()


def test_fully_quoted_file_with_letter_flags_and_a_byte_order_mark_loads_like_the_plain_file(world, tmp_path):
    raw = pd.read_csv(world.cases_path, dtype=str, keep_default_na=False)
    for column in ("in_wdi_panel", "panel_feasible"):
        raw[column] = raw[column].map({"yes": "Y", "no": "N"})
    assert set(raw["panel_feasible"]) == {"Y", "N"}
    target = tmp_path / "quoted.csv"
    raw.to_csv(target, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
    assert target.read_bytes().startswith(b"\xef\xbb\xbf" + b'"case_id"')
    pd.testing.assert_frame_equal(load_cases(target), load_cases(world.cases_path))


def test_column_order_in_the_file_does_not_matter(world, tmp_path):
    path = edited_catalogue(world, tmp_path / "cases.csv", lambda df: None)
    shuffled = pd.read_csv(path, dtype=str, keep_default_na=False)[list(CASE_COLUMNS)[::-1]]
    shuffled.to_csv(tmp_path / "reversed.csv", index=False)
    pd.testing.assert_frame_equal(load_cases(tmp_path / "reversed.csv"), load_cases(path))


def test_missing_catalogue_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "absent.csv")


# ----------------------------------------------------------------------------
# resolve_case
# ----------------------------------------------------------------------------
def test_resolve_case_accepts_identifiers_rows_and_mappings(cases):
    row = cases.iloc[3]
    expected = (row["case_id"], row["iso3"], int(row["opening_year"]))
    assert resolve_case(row["case_id"], cases) == expected
    assert resolve_case(row) == expected
    assert resolve_case(row.to_dict()) == expected
    assert resolve_case({"iso3": "ZZZ", "opening_year": 2003.0}) == ("", "ZZZ", 2003)


def test_resolve_case_rejects_unknown_identifiers_and_incomplete_rows(cases):
    with pytest.raises(ValueError, match="matches 0"):
        resolve_case("nope", cases)
    with pytest.raises(ValueError, match="catalogue"):
        resolve_case("C01")
    with pytest.raises(ValueError, match="opening_year"):
        resolve_case({"iso3": "ZZZ"})


# ----------------------------------------------------------------------------
# case_features
# ----------------------------------------------------------------------------
def by_hand(panel: pd.DataFrame, row, window: int = 3) -> dict:
    """Features of one case computed with plain pandas and numpy, independently of the module."""
    sub = panel.loc[panel["iso3"] == row.iso3].set_index("year")
    o = int(row.opening_year)
    win = sub.loc[o - window : o - 1]
    out = {c: (win[c].mean() if win[c].notna().any() else np.nan) for c in LEVELS}
    last5 = sub.loc[o - 5 : o - 1, "log_receipts"].dropna()
    out["receipts_growth_pre"] = float(np.polyfit(last5.index.to_numpy(float), last5.to_numpy(), 1)[0]) if len(last5) >= 3 else np.nan
    gdp = sub["gdp_usd"].get(o - 1, np.nan)
    out["capex_pct_gdp"] = 100.0 * row.investment_usd_bn_nominal * 1e9 / gdp if np.isfinite(gdp) else np.nan
    out["is_integrated_resort"] = 1.0 if row.category in ("integrated_resort", "destination_resort") else 0.0
    return out


def assert_features_equal(got: pd.DataFrame, expected_rows: list[dict]) -> None:
    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(got[col].to_numpy(float), [r[col] for r in expected_rows], rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=col)


@pytest.mark.parametrize("window", [1, 3, 5])
def test_case_features_match_a_hand_computation(panel, cases, window):
    feats = case_features(cases, panel, window=window)
    assert list(feats.columns) == ["case_id", "iso3", "opening_year", *FEATURE_COLUMNS, "capex_first_pct_gdp"]
    assert feats["case_id"].tolist() == cases["case_id"].tolist() and len(feats) == len(cases)
    assert feats["opening_year"].dtype == np.int64
    assert_features_equal(feats, [by_hand(panel, r, window) for r in cases.itertuples()])


def test_default_window_is_three_years(panel, cases):
    pd.testing.assert_frame_equal(case_features(cases, panel), case_features(cases, panel, window=3))


def test_features_use_only_years_before_the_opening(panel, cases):
    original = case_features(cases, panel)
    corrupted = panel.copy()
    sources = ["receipts_usd", "arrivals", "gdp_usd", "pop", "air_pax", "receipts_pct_gdp", "log_receipts", "log_gdp_pc",
               "log_pop", "log_gdp", "gdp_per_capita_usd", "arrivals_per_capita", "receipts_per_arrival_usd",
               "log_receipts_per_arrival", "air_pax_per_capita"]
    for r in cases.itertuples():
        mask = (corrupted["iso3"] == r.iso3) & (corrupted["year"] >= r.opening_year)
        corrupted.loc[mask, sources] = corrupted.loc[mask, sources] * 13.0 + 5.0
        late = mask & (corrupted["year"] % 2 == 0)
        corrupted.loc[late, sources] = np.nan
    assert not corrupted.equals(panel)
    pd.testing.assert_frame_equal(case_features(cases, panel), original)
    pd.testing.assert_frame_equal(case_features(cases, corrupted), original, check_exact=True)
    for window in (1, 5):
        pd.testing.assert_frame_equal(case_features(cases, corrupted, window=window), case_features(cases, panel, window=window), check_exact=True)


def test_changing_the_last_pre_opening_year_changes_the_features(panel, cases, mid):
    row = mid
    altered = panel.copy()
    mask = (altered["iso3"] == row["iso3"]) & (altered["year"] == row["opening_year"] - 1)
    altered.loc[mask, ["log_gdp_pc", "log_pop", "receipts_pct_gdp", "log_receipts", "gdp_usd"]] *= 1.5
    before = case_features(cases, panel).set_index("case_id").loc[row["case_id"]]
    after = case_features(cases, altered).set_index("case_id").loc[row["case_id"]]
    for col in ("log_gdp_pc", "log_pop", "receipts_pct_gdp", "receipts_growth_pre", "capex_pct_gdp"):
        assert before[col] != after[col], col
    assert before["is_integrated_resort"] == after["is_integrated_resort"]


def test_inputs_are_not_modified(panel, cases):
    panel_copy, cases_copy = panel.copy(), cases.copy()
    case_features(cases, panel)
    pd.testing.assert_frame_equal(panel, panel_copy)
    pd.testing.assert_frame_equal(cases, cases_copy)


def test_growth_slope_needs_three_years_of_receipts(panel, cases, mid):
    base = cases.loc[cases["case_id"] == mid["case_id"]].copy()
    results = {}
    for opening in (1996, 1997, 1998, 1999):
        shifted = base.assign(opening_year=opening)
        results[opening] = case_features(shifted, panel).iloc[0]["receipts_growth_pre"]
    assert np.isnan(results[1996]) and np.isnan(results[1997])
    assert np.isfinite(results[1998]) and np.isfinite(results[1999])
    iso3 = base.iloc[0]["iso3"]
    sub = panel.loc[(panel["iso3"] == iso3) & panel["year"].between(1995, 1997)].dropna(subset=["log_receipts"])
    assert results[1998] == pytest.approx(np.polyfit(sub["year"].to_numpy(float), sub["log_receipts"].to_numpy(), 1)[0])


def test_partial_windows_use_the_available_years_and_empty_windows_are_missing(panel, cases, mid):
    base = cases.loc[cases["case_id"] == mid["case_id"]].copy()
    early = case_features(base.assign(opening_year=1996), panel).iloc[0]
    row = panel.loc[(panel["iso3"] == base.iloc[0]["iso3"]) & (panel["year"] == 1995)].iloc[0]
    assert early["log_pop"] == pytest.approx(row["log_pop"])
    none = case_features(base.assign(opening_year=1995), panel).iloc[0]
    assert none[LEVELS + ["receipts_growth_pre", "capex_pct_gdp"]].isna().all()
    assert none["is_integrated_resort"] in (0.0, 1.0)


def test_economy_outside_the_panel_gets_missing_panel_features(panel, cases):
    extra = cases.iloc[[2]].copy().assign(case_id="Z1", iso3="ZZZ", category="integrated_resort")
    feats = case_features(extra, panel).iloc[0]
    assert feats[[c for c in FEATURE_COLUMNS if c != "is_integrated_resort"]].isna().all()
    assert feats["is_integrated_resort"] == 1.0


def test_capex_is_investment_over_gdp_of_the_year_before_the_opening(panel, cases):
    feats = case_features(cases, panel).set_index("case_id")
    for r in cases.head(6).itertuples():
        gdp = panel.loc[(panel["iso3"] == r.iso3) & (panel["year"] == r.opening_year - 1), "gdp_usd"].iloc[0]
        assert feats.loc[r.case_id, "capex_pct_gdp"] == pytest.approx(100.0 * r.investment_usd_bn_nominal * 1e9 / gdp)
    no_investment = cases.iloc[[4]].copy().assign(investment_usd_bn_nominal=np.nan, investment_usd_bn_used=np.nan)
    assert np.isnan(case_features(no_investment, panel).iloc[0]["capex_pct_gdp"])


def test_capex_uses_the_used_investment_and_falls_back_to_the_reported_one(panel, cases):
    row = cases.iloc[[4]].copy()
    base = case_features(row, panel).iloc[0]["capex_pct_gdp"]
    doubled = row.assign(investment_usd_bn_used=2.0 * row["investment_usd_bn_nominal"])
    assert case_features(doubled, panel).iloc[0]["capex_pct_gdp"] == pytest.approx(2.0 * base)
    converted = row.assign(investment_usd_bn_nominal=np.nan, investment_usd_bn_used=3.0 * row["investment_usd_bn_nominal"])
    assert case_features(converted, panel).iloc[0]["capex_pct_gdp"] == pytest.approx(3.0 * base)
    only_reported = row.drop(columns=["investment_usd_bn_used", "investment_basis"])
    assert case_features(only_reported, panel).iloc[0]["capex_pct_gdp"] == pytest.approx(base)
    with pytest.raises(ValueError, match="investment_usd_bn_used"):
        case_features(only_reported.drop(columns=["investment_usd_bn_nominal"]), panel)


def test_integrated_resort_flag_follows_the_category(panel, cases):
    rows = []
    for category in CATEGORIES:
        rows.append(cases.iloc[[1]].copy().assign(category=category, case_id=f"X_{category}"))
    feats = case_features(pd.concat(rows, ignore_index=True), panel).set_index("case_id")["is_integrated_resort"]
    for category in CATEGORIES:
        assert feats[f"X_{category}"] == (1.0 if category in INTEGRATED_CATEGORIES else 0.0)


def test_case_features_input_validation(panel, cases):
    with pytest.raises(ValueError, match="window"):
        case_features(cases, panel, window=0)
    with pytest.raises(ValueError, match="category"):
        case_features(cases.drop(columns=["category"]), panel)
    with pytest.raises(ValueError, match="log_receipts"):
        case_features(cases, panel.drop(columns=["log_receipts"]))
    with pytest.raises(ValueError, match="duplicated"):
        case_features(cases, pd.concat([panel, panel.iloc[:1]], ignore_index=True))


# ----------------------------------------------------------------------------
# economy_feature_cloud
# ----------------------------------------------------------------------------
def test_cloud_has_one_row_per_year_with_the_feature_columns(panel, plain_economy):
    years = range(2000, 2011)
    cloud = economy_feature_cloud(panel, plain_economy, years)
    assert list(cloud.columns) == ["iso3", "year", *FEATURE_COLUMNS]
    assert cloud["year"].tolist() == list(years) and (cloud["iso3"] == plain_economy).all()
    sub = panel.loc[panel["iso3"] == plain_economy].set_index("year")
    for col in LEVELS:
        np.testing.assert_allclose(cloud[col], sub.loc[2000:2010, col].to_numpy(), equal_nan=True)
    assert cloud["capex_pct_gdp"].isna().all()
    assert (cloud["is_integrated_resort"] == 0.0).all()


def test_cloud_growth_is_the_slope_over_the_five_years_ending_in_the_row_year(panel, plain_economy):
    cloud = economy_feature_cloud(panel, plain_economy, [2003, 2010]).set_index("year")
    sub = panel.loc[panel["iso3"] == plain_economy].set_index("year")["log_receipts"]
    for year in (2003, 2010):
        window = sub.loc[year - 4 : year].dropna()
        assert cloud.loc[year, "receipts_growth_pre"] == pytest.approx(np.polyfit(window.index.to_numpy(float), window.to_numpy(), 1)[0])


def test_cloud_capex_is_relative_to_gdp_of_the_row_year_and_flag_is_passed_through(panel, plain_economy):
    cloud = economy_feature_cloud(panel, plain_economy, [2005, 2006, 2007], capex_usd_bn=2.0, is_integrated_resort=1)
    gdp = panel.loc[(panel["iso3"] == plain_economy) & panel["year"].between(2005, 2007), "gdp_usd"].to_numpy()
    np.testing.assert_allclose(cloud["capex_pct_gdp"], 100.0 * 2.0e9 / gdp)
    assert (cloud["is_integrated_resort"] == 1.0).all()
    assert economy_feature_cloud(panel, plain_economy, [2005], capex_usd_bn=np.nan)["capex_pct_gdp"].isna().all()


def test_a_cost_in_dollars_gives_a_share_that_changes_from_year_to_year_and_a_constant_share_does_not(panel, plain_economy):
    years = list(range(2000, 2011))
    in_dollars = economy_feature_cloud(panel, plain_economy, years, capex_usd_bn=2.0)["capex_pct_gdp"].to_numpy()
    assert in_dollars.max() > 1.2 * in_dollars.min()
    constant = economy_feature_cloud(panel, plain_economy, years, capex_pct_gdp=1.75)
    assert (constant["capex_pct_gdp"] == 1.75).all() and constant["capex_pct_gdp"].dtype == np.float64
    reference = economy_feature_cloud(panel, plain_economy, years)
    others = [c for c in constant.columns if c != "capex_pct_gdp"]
    pd.testing.assert_frame_equal(constant[others], reference[others])
    assert list(constant.columns) == ["iso3", "year", *FEATURE_COLUMNS]
    flagged = economy_feature_cloud(panel, plain_economy, years, capex_pct_gdp=0.4, is_integrated_resort=1)
    assert (flagged["is_integrated_resort"] == 1.0).all() and (flagged["capex_pct_gdp"] == 0.4).all()


def test_a_constant_share_of_zero_is_a_known_share_and_a_missing_share_stays_missing(panel, plain_economy):
    assert (economy_feature_cloud(panel, plain_economy, [2003, 2004], capex_pct_gdp=0.0)["capex_pct_gdp"] == 0.0).all()
    assert economy_feature_cloud(panel, plain_economy, [2003, 2004], capex_pct_gdp=np.nan)["capex_pct_gdp"].isna().all()
    assert economy_feature_cloud(panel, plain_economy, [2003, 2004], capex_pct_gdp=None)["capex_pct_gdp"].isna().all()


def test_a_constant_share_equals_the_share_of_the_case_in_every_year_of_the_cloud(panel, cases):
    feats = case_features(cases, panel, window=1).set_index("case_id")
    for r in cases.head(5).itertuples():
        share_of_case = feats.loc[r.case_id, "capex_pct_gdp"]
        cloud = economy_feature_cloud(panel, r.iso3, range(2000, 2011), capex_pct_gdp=share_of_case)
        assert (cloud["capex_pct_gdp"] == share_of_case).all()
        in_dollars = economy_feature_cloud(panel, r.iso3, [r.opening_year - 1], capex_usd_bn=r.investment_usd_bn_nominal)
        assert in_dollars.iloc[0]["capex_pct_gdp"] == pytest.approx(share_of_case, rel=1e-12)


@pytest.mark.parametrize("usd,pct", [(2.0, 1.0), (0.0, 0.0), (np.nan, 1.0), (2.0, np.nan)])
def test_a_cost_in_dollars_and_a_constant_share_exclude_each_other(panel, plain_economy, usd, pct):
    with pytest.raises(ValueError, match="mutually exclusive") as info:
        economy_feature_cloud(panel, plain_economy, [2005], capex_usd_bn=usd, capex_pct_gdp=pct)
    assert "capex_usd_bn" in str(info.value) and "capex_pct_gdp" in str(info.value)


def test_the_fourth_positional_argument_of_the_cloud_is_the_cost_in_dollars(panel, plain_economy):
    positional = economy_feature_cloud(panel, plain_economy, [2005, 2006], 2.0, 1)
    keyword = economy_feature_cloud(panel, plain_economy, [2005, 2006], capex_usd_bn=2.0, is_integrated_resort=1)
    pd.testing.assert_frame_equal(positional, keyword)


def test_cloud_row_equals_case_features_of_an_opening_in_the_next_year_with_a_one_year_window(panel, cases):
    feats = case_features(cases, panel, window=1).set_index("case_id")
    for r in cases.itertuples():
        flag = 1 if r.category in INTEGRATED_CATEGORIES else 0
        cloud = economy_feature_cloud(panel, r.iso3, [r.opening_year - 1], capex_usd_bn=r.investment_usd_bn_nominal, is_integrated_resort=flag)
        np.testing.assert_allclose(cloud.iloc[0][list(FEATURE_COLUMNS)].to_numpy(float), feats.loc[r.case_id, list(FEATURE_COLUMNS)].to_numpy(float), rtol=1e-12, equal_nan=True)


def test_cloud_years_outside_the_panel_are_missing_and_unknown_economy_raises(panel, plain_economy):
    cloud = economy_feature_cloud(panel, plain_economy, [1990, 2030])
    assert cloud[LEVELS + ["receipts_growth_pre"]].isna().all().all()
    with pytest.raises(ValueError, match="not in the panel"):
        economy_feature_cloud(panel, "ZZZ", [2000])


# ----------------------------------------------------------------------------
# impute_features
# ----------------------------------------------------------------------------
def feature_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 8
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    frame.insert(0, "case_id", [f"K{i}" for i in range(n)])
    frame.insert(1, "iso3", [f"E{i}" for i in range(n)])
    frame.insert(2, "opening_year", 2000 + np.arange(n))
    frame.loc[[1, 5], "log_pop"] = np.nan  # 2 of 8: share 0.25, kept
    frame.loc[[1, 2, 4], "receipts_pct_gdp"] = np.nan  # 3 of 8: share 0.375, dropped
    frame.loc[[3], "air_pax_per_capita"] = np.nan  # 1 of 8, imputed
    frame.loc[[6], "capex_pct_gdp"] = np.nan
    return frame


def one_gap_frame(column: str = "log_pop", row: int = 2, n: int = 6) -> pd.DataFrame:
    """Complete features of ``n`` cases that open in 2000, 2001, ...; only ``column`` of case ``row`` is missing."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    frame.insert(0, "case_id", [f"K{i}" for i in range(n)])
    frame.insert(1, "iso3", [f"E{i}" for i in range(n)])
    frame.insert(2, "opening_year", 2000 + np.arange(n))
    frame.loc[row, column] = np.nan
    return frame


def reference_frame(first: int = 1995, last: int = 2009) -> pd.DataFrame:
    """Economy-year feature values that rise by one per year; feature number ``j`` starts at ``10 * (j + 1)``."""
    years = np.arange(first, last + 1)
    frame = pd.DataFrame({"iso3": "ZZZ", "year": years})
    for j, name in enumerate(FEATURE_COLUMNS):
        frame[name] = 10.0 * (j + 1) + (years - first)
    return frame


def expected_fill(features: pd.DataFrame, column: str, k: int, reference: pd.DataFrame | None) -> float:
    """Fill value of one missing cell with plain pandas.

    The median of the finite reference values of years before the opening year of
    case ``k``, else the median of the finite values of the cases whose opening
    year is not later; NaN when neither exists.
    """
    opening = features.loc[k, "opening_year"]
    if reference is not None:
        earlier = reference.loc[(reference["year"] < opening) & reference[column].notna(), column]
        if len(earlier):
            return float(earlier.median())
    others = features.loc[(features["opening_year"] <= opening) & features[column].notna(), column]
    return float(others.median()) if len(others) else float("nan")


def test_features_over_the_threshold_are_dropped_and_the_others_imputed_from_earlier_cases():
    frame = feature_frame()
    imputed, report = impute_features(frame)
    assert "receipts_pct_gdp" not in imputed.columns
    assert list(imputed.columns) == [c for c in frame.columns if c != "receipts_pct_gdp"]
    kept = [c for c in FEATURE_COLUMNS if c != "receipts_pct_gdp"]
    assert not imputed[kept].isna().any().any()
    fills = {}
    for col in ("log_pop", "air_pax_per_capita", "capex_pct_gdp"):
        for k in frame.index[frame[col].isna()]:
            fills[(col, k)] = imputed.loc[k, col]
            assert imputed.loc[k, col] == pytest.approx(expected_fill(frame, col, k, None), rel=1e-14), (col, k)
    assert fills[("log_pop", 1)] == frame.loc[0, "log_pop"]
    assert fills[("log_pop", 5)] == pytest.approx(frame.loc[[0, 2, 3, 4], "log_pop"].median(), rel=1e-14)
    assert fills[("log_pop", 5)] != frame["log_pop"].median()
    complete = frame.drop(columns=["receipts_pct_gdp"]).dropna()
    pd.testing.assert_frame_equal(imputed.loc[complete.index], complete)
    assert imputed[["case_id", "iso3", "opening_year"]].equals(frame[["case_id", "iso3", "opening_year"]])

    assert list(report.columns) == ["feature", "n_missing", "share_missing", "dropped", "reason", "fill_value", "fill_source"]
    assert report["feature"].tolist() == list(FEATURE_COLUMNS)
    assert report.loc[report["dropped"], "feature"].tolist() == ["receipts_pct_gdp"]
    row = report.set_index("feature")
    assert row.loc["receipts_pct_gdp", "n_missing"] == 3 and row.loc["receipts_pct_gdp", "share_missing"] == pytest.approx(0.375)
    assert "exceeds" in row.loc["receipts_pct_gdp", "reason"] and row.loc["log_pop", "reason"] == ""
    assert row.loc["log_pop", "fill_value"] == pytest.approx((fills[("log_pop", 1)] + fills[("log_pop", 5)]) / 2.0)
    assert row.loc["log_pop", "fill_source"] == "own" and row.loc["log_gdp_pc", "fill_source"] == "none"
    assert np.isnan(row.loc["log_gdp_pc", "fill_value"]) and np.isnan(row.loc["receipts_pct_gdp", "fill_value"])
    assert row.loc["receipts_pct_gdp", "fill_source"] == "none"


def test_a_fill_value_uses_no_case_that_opens_later():
    frame = feature_frame()
    base, _ = impute_features(frame)
    for col in ("log_pop", "air_pax_per_capita", "capex_pct_gdp"):
        for k in frame.index[frame[col].isna()]:
            altered = frame.copy()
            later = altered["opening_year"] > altered.loc[k, "opening_year"]
            assert later.any()
            altered.loc[later, col] = altered.loc[later, col] * 50.0 + 7.0
            got, _ = impute_features(altered)
            assert got.loc[k, col] == base.loc[k, col], (col, k)
    first_year_ties = frame.assign(opening_year=2000)
    tied, _ = impute_features(first_year_ties)
    assert tied.loc[1, "log_pop"] == pytest.approx(frame["log_pop"].median(), rel=1e-14)


def test_threshold_is_strict_and_adjustable():
    frame = feature_frame()
    _, strict = impute_features(frame, max_missing_share=0.0)
    assert set(strict.loc[strict["dropped"], "feature"]) == {"log_pop", "receipts_pct_gdp", "air_pax_per_capita", "capex_pct_gdp"}
    kept, loose = impute_features(frame, max_missing_share=0.4)
    assert not loose["dropped"].any() and not kept[list(FEATURE_COLUMNS)].isna().any().any()
    at_limit, report = impute_features(frame, max_missing_share=0.25)
    assert "log_pop" in at_limit.columns and not report.set_index("feature").loc["log_pop", "dropped"]
    with pytest.raises(ValueError, match="max_missing_share"):
        impute_features(frame, max_missing_share=1.5)


def test_missing_values_take_the_median_of_the_reference_years_before_their_own_opening():
    frame = feature_frame()
    reference = reference_frame()
    imputed, report = impute_features(frame, reference=reference)
    # log_pop starts at 20, air_pax_per_capita at 60 and capex_pct_gdp at 80 and each rises by one per year.
    assert imputed.loc[1, "log_pop"] == 22.5  # years 1995 to 2000 for the opening in 2001
    assert imputed.loc[5, "log_pop"] == 24.5  # years 1995 to 2004 for the opening in 2005
    assert imputed.loc[3, "air_pax_per_capita"] == 63.5  # years 1995 to 2002
    assert imputed.loc[6, "capex_pct_gdp"] == 85.0  # years 1995 to 2005
    for col in ("log_pop", "air_pax_per_capita", "capex_pct_gdp"):
        for k in frame.index[frame[col].isna()]:
            assert imputed.loc[k, col] == pytest.approx(expected_fill(frame, col, k, reference), rel=1e-14)
            assert imputed.loc[k, col] != reference[col].median()
    row = report.set_index("feature")
    assert row.loc["log_pop", "fill_source"] == "reference" and row.loc["log_pop", "fill_value"] == pytest.approx(23.5)
    assert row.loc["receipts_pct_gdp", "dropped"] and "receipts_pct_gdp" not in imputed.columns
    pd.testing.assert_series_equal(imputed["log_gdp_pc"], frame["log_gdp_pc"])


def test_reference_rows_of_the_opening_year_and_later_years_are_never_used():
    frame = one_gap_frame("log_pop", row=2)
    reference = pd.DataFrame({"year": [1999, 2001, 2002, 2003], "log_pop": [5.0, 7.0, 1000.0, 2000.0]})
    imputed, report = impute_features(frame, reference=reference)
    assert frame.loc[2, "opening_year"] == 2002 and imputed.loc[2, "log_pop"] == 6.0
    assert report.set_index("feature").loc["log_pop", "fill_source"] == "reference"
    inclusive = reference.loc[reference["year"] <= 2002, "log_pop"].median()
    assert inclusive == 7.0 and reference["log_pop"].median() == 503.5


def test_changing_the_reference_from_the_opening_year_on_does_not_change_the_fill():
    frame = feature_frame()
    reference = reference_frame()
    base, _ = impute_features(frame, reference=reference)
    for col in ("log_pop", "air_pax_per_capita", "capex_pct_gdp"):
        for k in frame.index[frame[col].isna()]:
            altered = reference.copy()
            late = altered["year"] >= frame.loc[k, "opening_year"]
            assert late.any() and not late.all()
            altered.loc[late, col] = altered.loc[late, col] * 100.0 - 3.0
            got, _ = impute_features(frame, reference=altered)
            assert got.loc[k, col] == base.loc[k, col], (col, k)


def test_cases_without_an_earlier_reference_year_use_the_earlier_cases_and_the_source_is_mixed():
    frame = feature_frame()
    reference = reference_frame(first=2003)
    imputed, report = impute_features(frame, reference=reference)
    row = report.set_index("feature")
    assert imputed.loc[1, "log_pop"] == frame.loc[0, "log_pop"]  # no reference year before 2001
    assert imputed.loc[5, "log_pop"] == 20.5  # reference years 2003 and 2004
    assert row.loc["log_pop", "fill_source"] == "mixed"
    assert row.loc["log_pop", "fill_value"] == pytest.approx((frame.loc[0, "log_pop"] + 20.5) / 2.0)
    assert imputed.loc[3, "air_pax_per_capita"] == pytest.approx(frame.loc[[0, 1, 2], "air_pax_per_capita"].median(), rel=1e-14)
    assert row.loc["air_pax_per_capita", "fill_source"] == "own"
    assert imputed.loc[6, "capex_pct_gdp"] == 81.0 and row.loc["capex_pct_gdp", "fill_source"] == "reference"


def test_reference_without_finite_values_falls_back_to_the_earlier_cases():
    frame = feature_frame()
    reference = pd.DataFrame({"year": np.arange(1995, 2000), **{c: np.full(5, np.nan) for c in FEATURE_COLUMNS}})
    imputed, report = impute_features(frame, reference=reference)
    own, _ = impute_features(frame)
    pd.testing.assert_frame_equal(imputed, own)
    assert report.set_index("feature").loc["log_pop", "fill_source"] == "own"


def test_a_feature_that_cannot_be_filled_from_before_the_opening_is_dropped():
    frame = one_gap_frame("log_pop", row=0)
    imputed, report = impute_features(frame)
    row = report.set_index("feature").loc["log_pop"]
    assert row["dropped"] and "log_pop" not in imputed.columns and row["n_missing"] == 1
    assert "no value from years before the opening" in row["reason"] and row["fill_source"] == "none" and np.isnan(row["fill_value"])
    assert not report.set_index("feature").drop(index="log_pop")["dropped"].any()
    late_reference = pd.DataFrame({"year": [2000, 2001], "log_pop": [3.0, 4.0]})
    imputed, report = impute_features(frame, reference=late_reference)
    assert "log_pop" not in imputed.columns and report.set_index("feature").loc["log_pop", "dropped"]
    early_reference = pd.DataFrame({"year": [1998, 1999, 2000], "log_pop": [3.0, 4.0, 90.0]})
    imputed, report = impute_features(frame, reference=early_reference)
    assert imputed.loc[0, "log_pop"] == 3.5 and report.set_index("feature").loc["log_pop", "fill_source"] == "reference"


def test_imputation_needs_the_opening_year_and_the_reference_year_only_when_values_are_missing():
    frame = one_gap_frame("log_pop", row=3)
    with pytest.raises(ValueError, match="opening_year"):
        impute_features(frame.drop(columns=["opening_year"]))
    with pytest.raises(ValueError, match="year"):
        impute_features(frame, reference=reference_frame().drop(columns=["year"]))
    with pytest.raises(ValueError, match="log_pop"):
        impute_features(frame, reference=reference_frame().drop(columns=["log_pop"]))
    complete = frame.copy()
    complete["log_pop"] = complete["log_pop"].fillna(0.0)
    unchanged, report = impute_features(complete.drop(columns=["opening_year"]), reference=pd.DataFrame({"x": [1.0]}))
    pd.testing.assert_frame_equal(unchanged, complete.drop(columns=["opening_year"]))
    assert not report["dropped"].any() and (report["fill_source"] == "none").all() and report["n_missing"].sum() == 0


def test_all_missing_feature_is_dropped_even_when_the_threshold_is_one():
    frame = feature_frame()
    frame["log_pop"] = np.nan
    imputed, report = impute_features(frame, max_missing_share=1.0)
    row = report.set_index("feature").loc["log_pop"]
    assert row["dropped"] and "log_pop" not in imputed.columns and "no value from years before the opening" in row["reason"]
    filled, report = impute_features(frame, reference=reference_frame(), max_missing_share=1.0)
    assert not report.set_index("feature").loc["log_pop", "dropped"]
    assert filled["log_pop"].tolist() == [20.0 + (opening - 1996) / 2.0 for opening in range(2000, 2008)]


def test_the_first_opening_investment_column_is_not_a_feature_and_passes_through_imputation_unchanged():
    frame = one_gap_frame("log_pop", row=3)
    frame["capex_first_pct_gdp"] = [0.5, np.nan, 1.5, np.nan, 2.5, 3.5]
    out, report = impute_features(frame)
    assert "capex_first_pct_gdp" not in set(report["feature"])
    np.testing.assert_array_equal(out["capex_first_pct_gdp"].to_numpy(), frame["capex_first_pct_gdp"].to_numpy())
    assert out.columns.tolist() == frame.columns.tolist() and out["log_pop"].notna().all()


def test_impute_adds_no_indicator_columns_and_keeps_the_input_intact():
    frame = feature_frame()
    snapshot = frame.copy()
    reference = reference_frame()
    reference_snapshot = reference.copy()
    imputed, _ = impute_features(frame, reference=reference)
    assert set(imputed.columns) <= set(frame.columns)
    pd.testing.assert_frame_equal(frame, snapshot)
    pd.testing.assert_frame_equal(reference, reference_snapshot)
    with pytest.raises(ValueError, match="FEATURE_COLUMNS"):
        impute_features(frame[["case_id", "iso3"]])


def test_imputation_of_simulated_case_features_uses_only_earlier_reference_years(panel, cases):
    feats = case_features(cases, panel)
    assert feats[list(FEATURE_COLUMNS)].isna().any().any()
    stacked = pd.concat(
        [economy_feature_cloud(panel, iso, range(1995, 2020)) for iso in sorted(panel["iso3"].unique())[:20]], ignore_index=True
    )
    imputed, report = impute_features(feats, reference=stacked, max_missing_share=0.5)
    row = report.set_index("feature")
    for name in FEATURE_COLUMNS:
        cells = list(feats.index[feats[name].isna()])
        fills = [expected_fill(feats, name, k, stacked) for k in cells]
        if row.loc[name, "dropped"]:
            assert name not in imputed.columns
            assert np.isnan(fills).any() or len(cells) / len(feats) > 0.5, name
            continue
        for k, fill in zip(cells, fills):
            assert imputed.loc[k, name] == pytest.approx(fill, rel=1e-12), (name, k)
        observed = feats[name].notna()
        pd.testing.assert_series_equal(imputed.loc[observed, name], feats.loc[observed, name])
        assert row.loc[name, "n_missing"] == len(cells)
    kept = [c for c in FEATURE_COLUMNS if c in imputed.columns]
    assert not imputed[kept].isna().any().any() and len(kept) >= 7
    assert report["n_missing"].sum() > 0
    earliest = feats.loc[feats["opening_year"].idxmin()]
    assert earliest["opening_year"] == 1997 and np.isnan(earliest["receipts_growth_pre"])
    assert row.loc["receipts_growth_pre", "dropped"] and "no value from years before the opening" in row.loc["receipts_growth_pre", "reason"]


# ----------------------------------------------------------------------------
# outcome_matrix, donor_pool, check_feasibility
# ----------------------------------------------------------------------------
def brute_force_donors(panel, cases, row, outcome="receipts_pct_gdp", first=1995, last=2019, buffer=5) -> list[str]:
    found = []
    for iso3, sub in panel.groupby("iso3"):
        if iso3 == row["iso3"]:
            continue
        series = sub.set_index("year")[outcome].reindex(range(first, last + 1))
        if series.isna().any():
            continue
        openings = cases.loc[cases["iso3"] == iso3, "opening_year"]
        if ((openings >= row["opening_year"] - buffer) & (openings <= last)).any():
            continue
        found.append(iso3)
    return sorted(found)


def test_outcome_matrix_layout(panel, plain_economy):
    wide = outcome_matrix(panel, "receipts_pct_gdp", 1995, 2019)
    assert wide.shape == (25, panel["iso3"].nunique())
    assert wide.index.tolist() == list(range(1995, 2020))
    expected = panel.loc[(panel["iso3"] == plain_economy) & (panel["year"] == 2003), "receipts_pct_gdp"].iloc[0]
    assert wide.loc[2003, plain_economy] == expected
    padded = outcome_matrix(panel, "receipts_pct_gdp", 1990, 2030)
    assert padded.shape[0] == 41 and padded.loc[1990].isna().all() and padded.loc[2030].isna().all()
    with pytest.raises(ValueError, match="year range"):
        outcome_matrix(panel, "receipts_pct_gdp", 2019, 1995)


def test_donor_pools_match_a_brute_force_search_for_every_case(panel, cases):
    for row in cases.to_dict("records"):
        pool = donor_pool(panel, row, cases)
        assert pool == brute_force_donors(panel, cases, row), row["case_id"]
        assert isinstance(pool, list) and all(isinstance(c, str) for c in pool)
        assert pool == sorted(set(pool)) and row["iso3"] not in pool


def test_donor_pool_excludes_own_economy_incomplete_series_and_nearby_openings(panel, cases, mid):
    wide = outcome_matrix(panel, "receipts_pct_gdp", 1995, 2019)
    incomplete = set(wide.columns[wide.isna().any()])
    assert len(incomplete) >= 3
    case = mid
    pool = set(donor_pool(panel, case, cases))
    assert pool.isdisjoint(incomplete) and case["iso3"] not in pool
    window = cases.loc[(cases["opening_year"] >= case["opening_year"] - 5) & (cases["opening_year"] <= 2019)]
    assert len(window) >= 3 and pool.isdisjoint(set(window["iso3"]))
    far = cases.loc[(cases["opening_year"] < case["opening_year"] - 5)]
    assert len(far) >= 1
    complete_far = [c for c in far["iso3"] if c not in incomplete]
    assert complete_far and set(complete_far) <= pool


def test_donor_pool_depends_on_the_focal_opening_year(panel, cases):
    feas = check_feasibility(cases, panel)
    ok = cases.loc[cases["case_id"].isin(feas.loc[feas["feasible"], "case_id"])]
    early = ok.sort_values("opening_year", kind="stable").iloc[0]
    late = ok.sort_values("opening_year", kind="stable").iloc[-1]
    c01 = cases.loc[cases["case_id"] == "C01"].iloc[0]
    assert c01["opening_year"] == 1997 and early["opening_year"] <= 2002 and late["opening_year"] >= 2003
    assert c01["iso3"] not in donor_pool(panel, early, cases)
    assert c01["iso3"] in donor_pool(panel, late, cases)
    far_future = cases.loc[cases["case_id"] == "C16"].iloc[0]
    assert far_future["opening_year"] > 2019 and far_future["iso3"] in donor_pool(panel, early, cases)


def test_donor_pool_parameters(panel, cases, mid):
    case = mid
    base = donor_pool(panel, case, cases)
    no_buffer = donor_pool(panel, case, cases, buffer=0)
    assert set(base) < set(no_buffer)
    assert no_buffer == brute_force_donors(panel, cases, case, buffer=0)
    assert donor_pool(panel, case, cases, last_year=2010, first_year=2000) == brute_force_donors(panel, cases, case, first=2000, last=2010)
    assert donor_pool(panel, case, cases, outcome="log_receipts") == brute_force_donors(panel, cases, case, outcome="log_receipts")


def test_donor_pool_accepts_identifier_row_and_mapping(panel, cases):
    row = cases.iloc[6]
    expected = donor_pool(panel, row, cases)
    assert donor_pool(panel, row["case_id"], cases) == expected
    assert donor_pool(panel, {"iso3": row["iso3"], "opening_year": row["opening_year"]}, cases) == expected
    with pytest.raises(ValueError, match="matches 0"):
        donor_pool(panel, "missing-case", cases)
    with pytest.raises(ValueError, match="opening_year"):
        donor_pool(panel, row, cases.drop(columns=["opening_year"]))


def test_check_feasibility_counts_years_and_donors_like_a_brute_force_count(panel, cases):
    feas = check_feasibility(cases, panel)
    assert list(feas.columns) == ["case_id", "iso3", "opening_year", "n_pre", "n_post", "window_end", "n_donors", "feasible", "reason"]
    assert feas["case_id"].tolist() == cases["case_id"].tolist()
    assert feas["feasible"].dtype == bool and feas["window_end"].dtype == np.int64
    for row in cases.to_dict("records"):
        sub = panel.loc[(panel["iso3"] == row["iso3"]) & panel["year"].between(1995, 2019) & panel["receipts_pct_gdp"].notna()]
        got = feas.loc[feas["case_id"] == row["case_id"]].iloc[0]
        end = min(row["opening_year"] + 4, 2019)
        assert got["window_end"] == end
        assert got["n_pre"] == (sub["year"] < row["opening_year"]).sum()
        assert got["n_post"] == ((sub["year"] >= row["opening_year"]) & (sub["year"] <= end)).sum()
        assert got["n_donors"] == len(brute_force_donors(panel, cases, row))
        assert got["feasible"] == (got["n_pre"] >= 5 and got["n_post"] >= 3 and got["n_donors"] >= 15)


def test_feasibility_reasons_name_every_failed_condition(panel, cases, mid):
    feas = check_feasibility(cases, panel).set_index("case_id")
    assert feas.loc["C01", "reason"] == "n_pre 2 below minimum 5" and not feas.loc["C01", "feasible"]
    assert feas.loc["C16", "reason"] == "n_post 0 below minimum 3" and not feas.loc["C16", "feasible"]
    assert feas.loc[mid["case_id"], "reason"] == "" and feas.loc[mid["case_id"], "feasible"]
    got = feas.loc[mid["case_id"]]
    strict = check_feasibility(
        cases, panel, min_pre=got["n_pre"] + 1, min_post=got["n_post"], min_donors=got["n_donors"] + 1
    ).set_index("case_id")
    expected = f"n_pre {got['n_pre']} below minimum {got['n_pre'] + 1}; n_donors {got['n_donors']} below minimum {got['n_donors'] + 1}"
    assert strict.loc[mid["case_id"], "reason"] == expected and not strict.loc[mid["case_id"], "feasible"]
    both = check_feasibility(cases, panel, min_post=got["n_post"] + 1).set_index("case_id")
    assert both.loc[mid["case_id"], "reason"] == f"n_post {got['n_post']} below minimum {got['n_post'] + 1}"
    assert not strict["feasible"].all()
    relaxed = check_feasibility(cases, panel, min_pre=2, min_post=0, min_donors=1)
    assert relaxed["feasible"].all() and (relaxed["reason"] == "").all()


def with_extra_openings(cases: pd.DataFrame, row, offsets) -> pd.DataFrame:
    """The catalogue plus one more case in the economy of ``row`` for every offset in years from its opening year."""
    extra = [
        cases.loc[cases["case_id"] == row["case_id"]].assign(case_id=f"X{n}", opening_year=int(row["opening_year"]) + offset)
        for n, offset in enumerate(offsets)
    ]
    return pd.concat([cases, *extra], ignore_index=True)


@pytest.mark.parametrize("offset,inside", [(-5, False), (-1, False), (0, True), (1, True), (4, True), (5, False), (9, False)])
def test_another_opening_of_the_economy_counts_when_it_lies_in_the_post_window(panel, cases, mid, offset, inside):
    opening = int(mid["opening_year"])
    feas = check_feasibility(with_extra_openings(cases, mid, [offset]), panel).set_index("case_id")
    got = feas.loc[mid["case_id"]]
    assert got["window_end"] == opening + 4
    if inside:
        assert not got["feasible"]
        assert got["reason"] == f"other_openings 1 in post window (years {opening + offset})"
    else:
        assert got["feasible"] and got["reason"] == ""


def test_other_openings_are_counted_with_their_years_and_listed_between_n_post_and_n_donors(panel, cases, mid):
    opening = int(mid["opening_year"])
    table = with_extra_openings(cases, mid, [3, 1, 3, 0])
    got = check_feasibility(table, panel).set_index("case_id").loc[mid["case_id"]]
    assert got["reason"] == f"other_openings 4 in post window (years {opening}, {opening + 1}, {opening + 3}, {opening + 3})"
    assert got["n_pre"] == opening - 1995 and got["n_post"] == 5
    strict = check_feasibility(
        table, panel, min_pre=got["n_pre"] + 1, min_post=got["n_post"] + 1, min_donors=got["n_donors"] + 1
    ).set_index("case_id")
    parts = strict.loc[mid["case_id"], "reason"].split("; ")
    assert [part.split(" ")[0] for part in parts] == ["n_pre", "n_post", "other_openings", "n_donors"]
    assert parts[2] == f"other_openings 4 in post window (years {opening}, {opening + 1}, {opening + 3}, {opening + 3})"


def test_the_extra_case_sees_the_focal_opening_as_an_other_opening_when_it_opens_earlier(panel, cases, mid):
    opening = int(mid["opening_year"])
    feas = check_feasibility(with_extra_openings(cases, mid, [-2, 6]), panel).set_index("case_id")
    assert feas.loc["X0", "reason"] == f"other_openings 1 in post window (years {opening})"
    assert feas.loc["X1", "reason"] == "" and feas.loc[mid["case_id"], "reason"] == ""
    assert not feas.loc["X0", "feasible"] and feas.loc["X1", "feasible"] and feas.loc[mid["case_id"], "feasible"]


def test_other_openings_follow_the_window_set_by_the_horizon_and_the_last_year(panel, cases, mid):
    opening = int(mid["opening_year"])
    assert opening + 7 <= 2019
    table = with_extra_openings(cases, mid, [3, 7])
    focal = mid["case_id"]

    def row(**kwargs):
        return check_feasibility(table, panel, **kwargs).set_index("case_id").loc[focal]

    assert row(post_horizon=3)["feasible"] and row(post_horizon=3)["window_end"] == opening + 2
    assert row(post_horizon=4)["reason"] == f"other_openings 1 in post window (years {opening + 3})"
    assert row(post_horizon=8)["reason"] == f"other_openings 2 in post window (years {opening + 3}, {opening + 7})"
    assert row(post_horizon=None)["reason"] == f"other_openings 2 in post window (years {opening + 3}, {opening + 7})"
    assert row(last_year=opening + 2)["feasible"] and row(last_year=opening + 2)["window_end"] == opening + 2
    assert row(last_year=opening + 3)["reason"] == f"other_openings 1 in post window (years {opening + 3})"


def test_the_catalogue_flag_does_not_replace_the_check_for_other_openings(panel, cases, mid):
    table = with_extra_openings(cases, mid, [2])
    assert bool(table.loc[table["case_id"] == mid["case_id"], "panel_feasible"].iloc[0])
    assert not check_feasibility(table, panel).set_index("case_id").loc[mid["case_id"], "feasible"]


def test_raw_cases_of_a_shared_economy_fail_on_other_openings_exactly_when_one_follows_within_the_window(cluster_panel, cluster_cases):
    feas = check_feasibility(cluster_cases, cluster_panel).set_index("case_id")
    flagged = []
    for row in cluster_cases.itertuples():
        end = min(row.opening_year + 4, 2019)
        later = sorted(
            int(y) for c, i, y in zip(cluster_cases["case_id"], cluster_cases["iso3"], cluster_cases["opening_year"])
            if i == row.iso3 and c != row.case_id and row.opening_year <= y <= end
        )
        reasons = feas.loc[row.case_id, "reason"].split("; ")
        if later:
            flagged.append(row.case_id)
            assert f"other_openings {len(later)} in post window (years {', '.join(str(y) for y in later)})" in reasons, row.case_id
            assert not feas.loc[row.case_id, "feasible"]
        else:
            assert not any(part.startswith("other_openings") for part in reasons), row.case_id
    assert flagged == ["C05", "C09"]


def test_the_members_of_an_episode_are_not_other_openings_of_the_episode(cluster_panel, cluster_cases, episodes):
    for feas in (check_feasibility(episodes, cluster_panel), check_feasibility(episodes, cluster_panel, catalogue=cluster_cases)):
        assert not feas["reason"].str.contains("other_openings").any()
        assert feas["feasible"].sum() == 11


def test_another_episode_in_the_window_is_an_other_opening_only_when_the_window_is_not_cut_at_its_start(cluster_panel, cluster_cases):
    split = build_episodes(cluster_cases, merge_gap=2)
    assert len(split) == 16
    capped = check_feasibility(split, cluster_panel, catalogue=cluster_cases)
    assert not capped["reason"].str.contains("other_openings").any()
    uncapped = check_feasibility(split.drop(columns=["next_start_year"]), cluster_panel, catalogue=cluster_cases)
    opening = cluster_cases.groupby("iso3")["opening_year"].apply(list)
    expected = {}
    for row in uncapped.itertuples():
        years = sorted(y for y in opening[row.iso3] if row.opening_year <= y <= row.window_end)
        if row.opening_year in years:
            years.remove(row.opening_year)
        if years:
            expected[row.case_id] = f"other_openings {len(years)} in post window (years {', '.join(str(y) for y in years)})"
    assert len(expected) == 2
    for case_id, text in expected.items():
        assert text in uncapped.set_index("case_id").loc[case_id, "reason"].split("; ")
    assert set(uncapped.loc[uncapped["reason"].str.contains("other_openings"), "case_id"]) == set(expected)


def test_feasibility_window_and_buffer_parameters(panel, cases, mid):
    narrow = check_feasibility(cases, panel, first_year=2000, last_year=2015).set_index("case_id")
    base = check_feasibility(cases, panel).set_index("case_id")
    opening = int(mid["opening_year"])
    assert 2000 < opening <= 2014
    assert narrow.loc[mid["case_id"], "n_pre"] == opening - 2000
    assert narrow.loc[mid["case_id"], "n_post"] == min(POST_HORIZON, 2015 - opening + 1)
    assert narrow.loc[mid["case_id"], "window_end"] == min(opening + POST_HORIZON - 1, 2015)
    assert (narrow["n_pre"] <= base["n_pre"]).all()
    assert narrow.loc["C01", "n_pre"] == 0 and "n_pre 0" in narrow.loc["C01", "reason"]
    no_buffer = check_feasibility(cases, panel, buffer=0)
    assert (no_buffer["n_donors"].to_numpy() >= base["n_donors"].to_numpy()).all()


def test_economy_outside_the_panel_and_empty_outcome_are_infeasible(panel, cases, mid):
    extra = cases.iloc[[4]].copy().assign(case_id="Z1", iso3="ZZZ")
    feas = check_feasibility(pd.concat([cases, extra], ignore_index=True), panel).set_index("case_id")
    assert not feas.loc["Z1", "feasible"] and "economy not in panel" in feas.loc["Z1", "reason"]
    assert feas.loc["Z1", "n_pre"] == 0 and feas.loc["Z1", "n_post"] == 0

    blank = panel.copy()
    blank.loc[blank["iso3"] == mid["iso3"], "receipts_pct_gdp"] = np.nan
    feas = check_feasibility(cases, blank).set_index("case_id")
    row = feas.loc[mid["case_id"]]
    assert not row["feasible"] and row["n_pre"] == 0 and "n_pre 0" in row["reason"]

    late = cases.iloc[[4]].copy().assign(case_id="Z2", opening_year=2030)
    feas = check_feasibility(late, panel).set_index("case_id")
    assert feas.loc["Z2", "n_post"] == 0 and "n_post 0" in feas.loc["Z2", "reason"]


def test_check_feasibility_input_validation(panel, cases):
    with pytest.raises(ValueError, match="case_id"):
        check_feasibility(cases.drop(columns=["case_id"]), panel)
    with pytest.raises(ValueError, match="year range"):
        check_feasibility(cases, panel, first_year=2019, last_year=1995)
    with pytest.raises(ValueError, match="receipts_nonexistent"):
        check_feasibility(cases, panel, outcome="receipts_nonexistent")


# ----------------------------------------------------------------------------
# Investment conversion file
# ----------------------------------------------------------------------------
FX_COLUMNS = [
    "case_id", "currency", "local_amount_bn", "price_year", "price_year_assumed", "fx_lcu_per_usd", "fx_source_url",
    "investment_usd_bn_fxconv", "figure_rule", "n_figures_considered", "notes",
]


def write_fx(path: Path, rows: list[dict]) -> Path:
    """Write a conversion file with the documented columns; keys left out of a row stay empty."""
    pd.DataFrame(rows).reindex(columns=FX_COLUMNS).to_csv(path, index=False)
    return path


def test_without_an_investment_file_the_used_investment_is_the_reported_one(world, tmp_path):
    def edit(df):
        df.loc[[0, 3], ["investment_usd_bn_nominal", "investment_year_basis"]] = ""

    cs = load_cases(edited_catalogue(world, tmp_path / "cases.csv", edit))
    pd.testing.assert_series_equal(cs["investment_usd_bn_used"], cs["investment_usd_bn_nominal"], check_names=False)
    expected = ["missing" if np.isnan(v) else "reported" for v in cs["investment_usd_bn_nominal"]]
    assert cs["investment_basis"].tolist() == expected
    assert expected[0] == "missing" and expected[3] == "missing" and expected[1] == "reported"


def test_investment_file_fills_missing_costs_and_never_replaces_reported_ones(world, tmp_path):
    def edit(df):
        df.loc[[0, 1, 2], "investment_usd_bn_nominal"] = ""
        df.loc[4, "investment_usd_bn_nominal"] = "0"

    path = edited_catalogue(world, tmp_path / "cases.csv", edit)
    plain = load_cases(path)
    ids = plain["case_id"].tolist()
    fx = write_fx(
        tmp_path / "fx.csv",
        [
            {"case_id": ids[0], "currency": "EUR", "local_amount_bn": 1.0, "price_year": 2000, "price_year_assumed": "no",
             "fx_lcu_per_usd": 0.9, "investment_usd_bn_fxconv": 1.11, "figure_rule": "single", "n_figures_considered": 1},
            {"case_id": ids[1], "investment_usd_bn_fxconv": 2.5},
            {"case_id": ids[2]},
            {"case_id": ids[4], "investment_usd_bn_fxconv": 5.0},
            {"case_id": ids[5], "investment_usd_bn_fxconv": 99.0},
            {"case_id": "NOT_A_CASE", "investment_usd_bn_fxconv": 7.0},
        ],
    )
    joined = load_cases(path, investment_fx_path=fx)
    assert list(joined.columns) == list(CATALOGUE_COLUMNS) and joined["case_id"].tolist() == ids
    pd.testing.assert_frame_equal(joined[list(CASE_COLUMNS)], plain[list(CASE_COLUMNS)])
    used, basis = joined["investment_usd_bn_used"], joined["investment_basis"]
    assert used.iloc[0] == 1.11 and basis.iloc[0] == "fx_converted"
    assert used.iloc[1] == 2.5 and basis.iloc[1] == "fx_converted"
    assert np.isnan(used.iloc[2]) and basis.iloc[2] == "missing"
    assert used.iloc[4] == 0.0 and basis.iloc[4] == "reported"
    assert used.iloc[5] == plain["investment_usd_bn_nominal"].iloc[5] != 99.0 and basis.iloc[5] == "reported"
    rest = [i for i in range(len(ids)) if i > 2]
    pd.testing.assert_series_equal(used.iloc[rest], plain["investment_usd_bn_used"].iloc[rest])
    assert (basis.iloc[rest] == "reported").all() and set(basis) <= set(INVESTMENT_BASES)
    assert joined["investment_usd_bn_nominal"].iloc[:3].isna().all()


def test_investment_file_with_no_matching_rows_changes_nothing(world, tmp_path):
    fx = write_fx(tmp_path / "fx.csv", [{"case_id": "OTHER", "investment_usd_bn_fxconv": 3.0}])
    pd.testing.assert_frame_equal(load_cases(world.cases_path, investment_fx_path=fx), load_cases(world.cases_path))
    header_only = write_fx(tmp_path / "empty_fx.csv", [])
    pd.testing.assert_frame_equal(load_cases(world.cases_path, investment_fx_path=header_only), load_cases(world.cases_path))


def test_investment_file_with_extra_columns_and_a_byte_order_mark_is_read(world, tmp_path):
    def edit(df):
        df.loc[0, "investment_usd_bn_nominal"] = ""

    path = edited_catalogue(world, tmp_path / "cases.csv", edit)
    first = load_cases(path)["case_id"].iloc[0]
    fx = tmp_path / "fx.csv"
    fx.write_bytes(b"\xef\xbb\xbf" + f"case_id,investment_usd_bn_fxconv,something_else\n{first},4.25,x\n".encode())
    joined = load_cases(path, investment_fx_path=fx)
    assert joined["investment_usd_bn_used"].iloc[0] == 4.25 and joined["investment_basis"].iloc[0] == "fx_converted"
    assert joined["price_year"].isna().all() and joined["price_year_assumed"].isna().all()
    assert joined["price_year"].dtype == np.float64 and str(joined["price_year_assumed"].dtype) == "boolean"


def test_price_year_and_its_assumed_flag_are_kept_from_the_conversion_file(world, tmp_path):
    def edit(df):
        df.loc[[0, 1, 2], "investment_usd_bn_nominal"] = ""

    path = edited_catalogue(world, tmp_path / "cases.csv", edit)
    ids = load_cases(path)["case_id"].tolist()
    fx = write_fx(
        tmp_path / "fx.csv",
        [
            {"case_id": ids[0], "price_year": 2000, "price_year_assumed": "no", "investment_usd_bn_fxconv": 1.11},
            {"case_id": ids[1], "price_year": 2004, "price_year_assumed": "Yes", "investment_usd_bn_fxconv": 2.5},
            {"case_id": ids[2], "investment_usd_bn_fxconv": 3.5},
            {"case_id": ids[5], "price_year": 1999, "price_year_assumed": "n", "investment_usd_bn_fxconv": 99.0},
            {"case_id": ids[6], "price_year": 2010, "price_year_assumed": "y"},
            {"case_id": "NOT_A_CASE", "price_year": 1990, "price_year_assumed": "yes", "investment_usd_bn_fxconv": 7.0},
        ],
    )
    joined = load_cases(path, investment_fx_path=fx)
    assert list(joined.columns) == list(CATALOGUE_COLUMNS)
    assert joined["price_year"].dtype == np.float64 and str(joined["price_year_assumed"].dtype) == "boolean"
    assert joined["price_year"].iloc[[0, 1]].tolist() == [2000.0, 2004.0]
    assert joined["price_year_assumed"].iloc[[0, 1]].tolist() == [False, True]
    assert joined["investment_usd_bn_used"].iloc[[0, 1]].tolist() == [1.11, 2.5]
    assert np.isnan(joined["price_year"].iloc[2]) and pd.isna(joined["price_year_assumed"].iloc[2])
    assert joined["investment_basis"].iloc[2] == "fx_converted"
    assert joined["investment_basis"].iloc[5] == "reported" and joined["price_year"].iloc[5] == 1999.0
    assert joined["price_year_assumed"].iloc[5] == False  # noqa: E712
    assert joined["price_year"].iloc[6] == 2010.0 and joined["price_year_assumed"].iloc[6] == True  # noqa: E712
    assert joined["investment_basis"].iloc[6] == "reported"
    rest = [i for i in range(len(ids)) if i not in (0, 1, 2, 5, 6)]
    assert joined["price_year"].iloc[rest].isna().all() and joined["price_year_assumed"].iloc[rest].isna().all()


@pytest.mark.parametrize(
    "column,value,pattern",
    [("price_year", "2000.5", "whole numbers"), ("price_year", "abc", "numeric"), ("price_year_assumed", "maybe", "yes/no")],
)
def test_invalid_price_year_values_raise_value_error_naming_the_file(world, tmp_path, column, value, pattern):
    frame = pd.DataFrame(
        {
            "case_id": ["K0", "K1"],
            "investment_usd_bn_fxconv": ["1.5", "2.5"],
            "price_year": ["2000", "2001"],
            "price_year_assumed": ["no", "yes"],
        }
    )
    frame.loc[0, column] = value
    path = tmp_path / "conversion.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=pattern) as info:
        load_cases(world.cases_path, investment_fx_path=path)
    assert "conversion.csv" in str(info.value) and column in str(info.value)


def test_blank_price_year_cells_stay_missing(world, tmp_path):
    ids = load_cases(world.cases_path)["case_id"].tolist()
    fx = write_fx(
        tmp_path / "fx.csv",
        [
            {"case_id": ids[0], "price_year": 2001, "investment_usd_bn_fxconv": 1.0},
            {"case_id": ids[1], "price_year_assumed": "no", "investment_usd_bn_fxconv": 1.0},
        ],
    )
    joined = load_cases(world.cases_path, investment_fx_path=fx)
    assert joined["price_year"].iloc[0] == 2001.0 and pd.isna(joined["price_year_assumed"].iloc[0])
    assert np.isnan(joined["price_year"].iloc[1]) and joined["price_year_assumed"].iloc[1] == False  # noqa: E712


@pytest.mark.parametrize(
    "edit,pattern",
    [
        (lambda df: df.drop(columns=["investment_usd_bn_fxconv"], inplace=True), "required column"),
        (lambda df: df.drop(columns=["case_id"], inplace=True), "required column"),
        (lambda df: df.__setitem__("case_id", ["K0"] * len(df)), "unique"),
        (lambda df: df.__setitem__("case_id", [""] + list(df["case_id"][1:])), "empty"),
        (lambda df: df.__setitem__("investment_usd_bn_fxconv", ["abc"] + list(df["investment_usd_bn_fxconv"][1:])), "numeric"),
        (lambda df: df.__setitem__("investment_usd_bn_fxconv", ["-2"] + list(df["investment_usd_bn_fxconv"][1:])), "at least"),
        (lambda df: df.__setitem__("investment_usd_bn_fxconv", ["inf"] + list(df["investment_usd_bn_fxconv"][1:])), "finite"),
    ],
)
def test_invalid_investment_files_raise_value_error_naming_the_file(world, tmp_path, edit, pattern):
    frame = pd.DataFrame({"case_id": [f"K{i}" for i in range(4)], "investment_usd_bn_fxconv": ["1.5", "2.5", "", "4"]})
    edit(frame)
    path = tmp_path / "conversion.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=pattern) as info:
        load_cases(world.cases_path, investment_fx_path=path)
    assert "conversion.csv" in str(info.value)


def test_missing_or_unreadable_investment_file_raises(world, tmp_path):
    with pytest.raises(FileNotFoundError, match="conversion"):
        load_cases(world.cases_path, investment_fx_path=tmp_path / "absent.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        load_cases(world.cases_path, investment_fx_path=empty)


# ----------------------------------------------------------------------------
# Post-opening window
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "opening,last,horizon,following,expected",
    [
        (2005, 2019, 5, None, 2009),
        (2005, 2019, 5, np.nan, 2009),
        (2005, 2019, 5, 2008, 2007),
        (2005, 2019, 5, 2010, 2009),
        (2005, 2019, 5, 2011, 2009),
        (2005, 2019, None, None, 2019),
        (2005, 2019, None, 2012, 2011),
        (2017, 2019, 5, None, 2019),
        (2021, 2019, 5, None, 2019),
        (2005, 2019, 1, None, 2005),
        (2005, 2019, 20, None, 2019),
        (2005, 2019, 5.0, 2008.0, 2007),
    ],
)
def test_post_window_ends_at_the_earliest_of_horizon_last_year_and_next_start(opening, last, horizon, following, expected):
    assert post_window_end(opening, last, horizon, following) == expected
    assert isinstance(post_window_end(opening, last, horizon, following), int)


def test_default_horizon_is_five_years():
    assert POST_HORIZON == 5 and post_window_end(2000, 2019) == 2004


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True, "5"])
def test_post_window_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError, match="post_horizon"):
        post_window_end(2005, 2019, horizon)


def test_post_window_rejects_a_next_start_that_is_not_after_the_opening():
    for following in (2005, 2004):
        with pytest.raises(ValueError, match="next_start_year"):
            post_window_end(2005, 2019, 5, following)


def test_resolve_next_start_reads_rows_and_mappings():
    assert resolve_next_start(pd.Series({"next_start_year": 2011.0})) == 2011
    assert resolve_next_start({"next_start_year": 2011}) == 2011
    assert resolve_next_start(pd.Series({"next_start_year": np.nan})) is None
    assert resolve_next_start({"next_start_year": None}) is None
    assert resolve_next_start({"iso3": "X"}) is None
    assert resolve_next_start(pd.Series({"iso3": "X"})) is None


# ----------------------------------------------------------------------------
# build_episodes on hand-made tables
# ----------------------------------------------------------------------------
HAND_DEFAULTS = {
    "operator": "Operator", "category": "theme_park", "location": "City", "opening_date": np.nan,
    "investment_usd_bn_nominal": np.nan, "investment_year_basis": np.nan, "investment_source_note": np.nan,
    "first_year_attendance_m": np.nan, "concurrent_confounds": np.nan, "in_wdi_panel": True, "panel_feasible": True,
    "other_openings_same_economy_within_5y": np.nan, "evidence_quality": "A", "source_url_1": "https://example.org/1",
    "source_url_2": np.nan, "notes": np.nan, "price_year": np.nan, "price_year_assumed": pd.NA,
}


def hand_cases(rows: list[dict]) -> pd.DataFrame:
    """A small catalogue; each row needs case_id, iso3 and opening_year, other columns have defaults."""
    records = []
    for r in rows:
        rec = {"attraction": f"Attraction {r['case_id']}", "economy": r["iso3"], **HAND_DEFAULTS, **r}
        rec.setdefault("investment_usd_bn_used", rec["investment_usd_bn_nominal"])
        rec.setdefault("investment_basis", "missing" if pd.isna(rec["investment_usd_bn_used"]) else "reported")
        records.append(rec)
    frame = pd.DataFrame(records)
    for flag in ("in_wdi_panel", "panel_feasible", "price_year_assumed"):
        frame[flag] = pd.array(frame[flag], dtype="boolean")
    frame["price_year"] = frame["price_year"].astype(float)
    return frame[list(CATALOGUE_COLUMNS)]


def one_economy(years, iso3="MAC", **per_row) -> pd.DataFrame:
    """One opening per year in ``years``; ``per_row`` maps a column name to a list with one entry per opening."""
    rows = []
    for i, y in enumerate(years):
        rows.append({"case_id": f"{iso3}_{y}_{i}", "iso3": iso3, "opening_year": y, **{k: v[i] for k, v in per_row.items()}})
    return hand_cases(rows)


MACAU_YEARS = [2004, 2008, 2011, 2014, 2018]


def test_a_chain_of_five_openings_spanning_fourteen_years_becomes_one_episode():
    assert MACAU_YEARS[-1] - MACAU_YEARS[0] == 14 and max(np.diff(MACAU_YEARS)) <= 5
    eps = build_episodes(one_economy(MACAU_YEARS))
    assert len(eps) == 1
    row = eps.iloc[0]
    assert row["case_id"] == "MAC_2004_ep" and row["opening_year"] == 2004 and row["n_openings"] == 5
    assert row["member_case_ids"] == "MAC_2004_0;MAC_2008_1;MAC_2011_2;MAC_2014_3;MAC_2018_4"
    assert row["member_opening_years"] == "2004;2008;2011;2014;2018"
    assert np.isnan(row["next_start_year"])
    assert row["attraction"] == ";".join(f"Attraction MAC_{y}_{i}" for i, y in enumerate(MACAU_YEARS))


def test_a_gap_is_measured_from_the_previous_opening_and_exactly_the_merge_gap_still_merges():
    five = build_episodes(one_economy([2005, 2010]))
    six = build_episodes(one_economy([2005, 2011]))
    assert len(five) == 1 and five.iloc[0]["n_openings"] == 2
    assert len(six) == 2 and six["opening_year"].tolist() == [2005, 2011] and six["n_openings"].tolist() == [1, 1]
    assert six["next_start_year"].tolist()[0] == 2011 and np.isnan(six["next_start_year"].tolist()[1])
    assert len(build_episodes(one_economy([2000, 2005, 2010, 2015]))) == 1
    split = build_episodes(one_economy([2000, 2005, 2011]))
    assert split["member_opening_years"].tolist() == ["2000;2005", "2011"]


def test_merge_gap_argument_changes_the_chains():
    years = [2005, 2008, 2008, 2013]
    assert len(build_episodes(one_economy(years), merge_gap=5)) == 1
    assert build_episodes(one_economy(years), merge_gap=4)["member_opening_years"].tolist() == ["2005;2008;2008", "2013"]
    assert build_episodes(one_economy(years), merge_gap=2)["member_opening_years"].tolist() == ["2005", "2008;2008", "2013"]
    assert build_episodes(one_economy(years), merge_gap=0)["member_opening_years"].tolist() == ["2005", "2008;2008", "2013"]
    assert len(build_episodes(one_economy([2005, 2011]), merge_gap=6)) == 1
    default = build_episodes(one_economy(years))
    pd.testing.assert_frame_equal(default, build_episodes(one_economy(years), 5))


def test_openings_of_different_economies_are_never_merged():
    cases = hand_cases(
        [
            {"case_id": "A1", "iso3": "AAA", "opening_year": 2005},
            {"case_id": "B1", "iso3": "BBB", "opening_year": 2005},
            {"case_id": "A2", "iso3": "AAA", "opening_year": 2007},
        ]
    )
    eps = build_episodes(cases)
    assert eps["case_id"].tolist() == ["AAA_2005_ep", "BBB_2005_ep"]
    assert eps["member_case_ids"].tolist() == ["A1;A2", "B1"]
    assert eps["next_start_year"].isna().all()


def test_investment_sum_ignores_missing_values_and_the_known_share_counts_members():
    eps = build_episodes(one_economy([2005, 2007, 2009, 2011], investment_usd_bn_nominal=[1.0, np.nan, 2.5, np.nan]))
    row = eps.iloc[0]
    assert row["investment_usd_bn_used"] == pytest.approx(3.5) and row["investment_usd_bn_nominal"] == pytest.approx(3.5)
    assert row["investment_share_known"] == pytest.approx(0.5) and row["investment_basis"] == "reported"
    none = build_episodes(one_economy([2005, 2007], investment_usd_bn_nominal=[np.nan, np.nan])).iloc[0]
    assert np.isnan(none["investment_usd_bn_used"]) and np.isnan(none["investment_usd_bn_nominal"])
    assert none["investment_share_known"] == 0.0 and none["investment_basis"] == "missing"
    single = build_episodes(one_economy([2005], investment_usd_bn_nominal=[0.0])).iloc[0]
    assert single["investment_usd_bn_used"] == 0.0 and single["investment_share_known"] == 1.0


def test_converted_figures_enter_the_used_sum_but_not_the_reported_sum():
    cases = one_economy(
        [2005, 2007, 2009],
        investment_usd_bn_nominal=[1.0, np.nan, np.nan],
        investment_usd_bn_used=[1.0, 2.0, np.nan],
        investment_basis=["reported", "fx_converted", "missing"],
    )
    row = build_episodes(cases).iloc[0]
    assert row["investment_usd_bn_nominal"] == pytest.approx(1.0) and row["investment_usd_bn_used"] == pytest.approx(3.0)
    assert row["investment_basis"] == "reported;fx_converted" and row["investment_share_known"] == pytest.approx(2 / 3)


def test_investment_falls_back_to_the_reported_column_when_the_used_columns_are_absent():
    cases = one_economy([2005, 2007], investment_usd_bn_nominal=[1.5, np.nan]).drop(columns=["investment_usd_bn_used", "investment_basis"])
    row = build_episodes(cases).iloc[0]
    assert row["investment_usd_bn_used"] == pytest.approx(1.5) and row["investment_basis"] == "reported"


def test_category_is_that_of_the_member_with_the_largest_known_investment():
    categories = ["theme_park", "integrated_resort", "mega_attraction", "destination_resort"]
    row = build_episodes(one_economy([2005, 2006, 2007, 2008], category=categories, investment_usd_bn_nominal=[1.0, 4.0, 2.0, 4.0])).iloc[0]
    assert row["category"] == "integrated_resort"
    unknown = build_episodes(one_economy([2005, 2006, 2007], category=categories[:3], investment_usd_bn_nominal=[np.nan] * 3)).iloc[0]
    assert unknown["category"] == "theme_park"
    partial = build_episodes(one_economy([2005, 2006, 2007], category=categories[:3], investment_usd_bn_nominal=[np.nan, np.nan, 0.5])).iloc[0]
    assert partial["category"] == "mega_attraction"


def test_next_start_year_of_economies_with_several_episodes():
    cases = hand_cases(
        [{"case_id": f"A{i}", "iso3": "AAA", "opening_year": y} for i, y in enumerate([2000, 2003, 2010, 2012, 2020])]
        + [{"case_id": "B0", "iso3": "BBB", "opening_year": 2001}]
    )
    eps = build_episodes(cases).set_index("case_id")
    assert eps.index.tolist() == ["AAA_2000_ep", "AAA_2010_ep", "AAA_2020_ep", "BBB_2001_ep"]
    assert eps.loc["AAA_2000_ep", "next_start_year"] == 2010 and eps.loc["AAA_2010_ep", "next_start_year"] == 2020
    assert np.isnan(eps.loc["AAA_2020_ep", "next_start_year"]) and np.isnan(eps.loc["BBB_2001_ep", "next_start_year"])
    assert eps["n_openings"].tolist() == [2, 2, 1, 1]
    assert eps["next_start_year"].dtype == np.float64


def test_members_are_ordered_by_year_and_case_id_and_the_earliest_date_of_the_start_year_is_kept():
    dated = one_economy([2005, 2005, 2005], opening_date=["2005-09", "2005-03-15", "2005-03-02"])
    row = build_episodes(dated).iloc[0]
    assert row["opening_date"] == "2005-03-02"
    assert row["member_case_ids"] == "MAC_2005_0;MAC_2005_1;MAC_2005_2"
    mixed = one_economy([2005, 2005], opening_date=[np.nan, "2005-12-01"])
    row = build_episodes(mixed).iloc[0]
    assert row["opening_date"] == "2005-12-01" and row["member_case_ids"] == "MAC_2005_0;MAC_2005_1"
    undated = build_episodes(one_economy([2005, 2006])).iloc[0]
    assert pd.isna(undated["opening_date"]) and undated["member_case_ids"] == "MAC_2005_0;MAC_2006_1"
    assert build_episodes(one_economy([2005, 2006], opening_date=["2005-11-30", "2006-01-02"])).iloc[0]["opening_date"] == "2005-11-30"
    assert pd.isna(build_episodes(one_economy([2005, 2006], opening_date=[np.nan, "2006-01-02"])).iloc[0]["opening_date"])
    reversed_ids = hand_cases([{"case_id": "b", "iso3": "AAA", "opening_year": 2005}, {"case_id": "a", "iso3": "AAA", "opening_year": 2005}])
    assert build_episodes(reversed_ids).iloc[0]["member_case_ids"] == "a;b"


def test_integrated_resort_indicator_is_one_if_any_member_is_a_resort():
    for category, expected in [("integrated_resort", 1), ("destination_resort", 1), ("theme_park", 0), ("multi_park_resort", 0), ("mega_attraction", 0)]:
        row = build_episodes(one_economy([2005, 2007], category=["theme_park", category])).iloc[0]
        assert row["is_integrated_resort_any"] == expected, category
    assert build_episodes(one_economy([2005, 2007])).iloc[0]["is_integrated_resort_any"] == 0


@pytest.mark.parametrize(
    "years,flags,expected",
    [
        ([2000], [True], True),
        ([1999], [True], False),
        ([2017], [True], True),
        ([2018], [True], False),
        ([2019], [True], False),
        ([2005, 2007], [False, False], False),
        ([2005, 2007], [False, True], True),
        ([2016, 2018], [False, True], True),
        ([2017, 2019], [True, False], True),
        ([2018, 2019], [True, True], False),
        ([1998, 2002], [True, True], False),
        ([2005, 2007], [None, True], True),
        ([2005, 2007], [None, False], False),
        ([2005, 2007], [None, None], None),
    ],
)
def test_panel_feasible_needs_a_flagged_member_and_a_start_year_from_2000_to_2017(years, flags, expected):
    eps = build_episodes(one_economy(years, panel_feasible=flags))
    row = eps.iloc[0]
    assert str(eps["panel_feasible"].dtype) == "boolean"
    if expected is None:
        assert pd.isna(row["panel_feasible"])
    else:
        assert row["panel_feasible"] == expected


def test_the_feasible_start_years_are_the_documented_constants():
    assert (EPISODE_FIRST_YEAR, EPISODE_LAST_FEASIBLE_YEAR) == (2000, 2017)
    for year in range(1995, 2025):
        flag = build_episodes(one_economy([year], panel_feasible=[True])).iloc[0]["panel_feasible"]
        assert bool(flag) == (2000 <= year <= 2017), year


def test_evidence_grade_is_the_lowest_grade_of_the_members():
    for grades, expected in [(["A", "C", "B"], "C"), (["A", "B"], "B"), (["A", "A"], "A"), (["A", np.nan], "A"), (["B", "high"], "high")]:
        row = build_episodes(one_economy([2005 + i for i in range(len(grades))], evidence_quality=grades)).iloc[0]
        assert row["evidence_quality"] == expected, grades
    assert pd.isna(build_episodes(one_economy([2005, 2006], evidence_quality=[np.nan, np.nan])).iloc[0]["evidence_quality"])


def test_text_columns_of_an_episode():
    cases = one_economy(
        [2005, 2007, 2009],
        operator=["X", "X", "Y"],
        location=["Town", "Town", "Port"],
        concurrent_confounds=["a; b", "b; c", np.nan],
        investment_source_note=["note one", "note one", "note two"],
        notes=[np.nan, "remark", np.nan],
        source_url_1=[np.nan, "https://example.org/b", "https://example.org/c"],
        source_url_2=[np.nan, np.nan, "https://example.org/z"],
        first_year_attendance_m=[2.0, np.nan, 3.5],
        in_wdi_panel=[True, True, True],
    )
    row = build_episodes(cases).iloc[0]
    assert row["operator"] == "X;Y" and row["location"] == "Town;Port"
    assert row["concurrent_confounds"] == "a; b; c"
    assert row["investment_source_note"] == "note one | note two" and row["notes"] == "remark"
    assert row["source_url_1"] == "https://example.org/b" and row["source_url_2"] == "https://example.org/z"
    assert row["first_year_attendance_m"] == pytest.approx(5.5)
    assert row["economy"] == "MAC" and row["iso3"] == "MAC" and bool(row["in_wdi_panel"]) is True
    assert pd.isna(row["other_openings_same_economy_within_5y"])
    blank = build_episodes(one_economy([2005])).iloc[0]
    assert pd.isna(blank["concurrent_confounds"]) and pd.isna(blank["notes"]) and pd.isna(blank["first_year_attendance_m"])


def test_price_year_is_kept_only_when_the_known_figures_share_it():
    same = build_episodes(one_economy([2005, 2007], investment_usd_bn_nominal=[1.0, 2.0], investment_year_basis=[2004.0, 2004.0])).iloc[0]
    differ = build_episodes(one_economy([2005, 2007], investment_usd_bn_nominal=[1.0, 2.0], investment_year_basis=[2004.0, 2006.0])).iloc[0]
    assert same["investment_year_basis"] == 2004.0 and np.isnan(differ["investment_year_basis"])


def test_the_conversion_price_year_of_an_episode_is_the_common_value_and_the_assumed_flag_is_true_if_any_is():
    def episode(years, flags):
        count = len(years)
        return build_episodes(one_economy([2005 + 2 * i for i in range(count)], price_year=years, price_year_assumed=flags))

    row = episode([2000.0, 2000.0, np.nan], [False, False, None]).iloc[0]
    assert row["price_year"] == 2000.0 and row["price_year_assumed"] == False  # noqa: E712
    row = episode([2000.0, 2001.0], [False, True]).iloc[0]
    assert np.isnan(row["price_year"]) and row["price_year_assumed"] == True  # noqa: E712
    row = episode([np.nan, 1999.0], [None, True]).iloc[0]
    assert row["price_year"] == 1999.0 and row["price_year_assumed"] == True  # noqa: E712
    row = episode([np.nan, np.nan], [None, None]).iloc[0]
    assert np.isnan(row["price_year"]) and pd.isna(row["price_year_assumed"])
    table = episode([2000.0, 2000.0], [False, False])
    assert table["price_year"].dtype == np.float64 and str(table["price_year_assumed"].dtype) == "boolean"
    bare = build_episodes(pd.DataFrame({"case_id": ["a"], "iso3": ["XXA"], "opening_year": [2005]}))
    assert np.isnan(bare.iloc[0]["price_year"]) and pd.isna(bare.iloc[0]["price_year_assumed"])


def test_member_investments_are_listed_in_member_order_with_an_empty_entry_for_an_unknown_figure():
    row = build_episodes(one_economy([2005, 2006, 2007, 2008], investment_usd_bn_nominal=[1.0, np.nan, 2.5, np.nan])).iloc[0]
    assert row["member_investment_usd_bn"] == "1.0;;2.5;"
    assert build_episodes(one_economy([2005], investment_usd_bn_nominal=[np.nan])).iloc[0]["member_investment_usd_bn"] == ""
    assert build_episodes(one_economy([2005, 2006], investment_usd_bn_nominal=[0.1, 1e-05])).iloc[0]["member_investment_usd_bn"] == "0.1;1e-05"
    converted = one_economy(
        [2005, 2007],
        investment_usd_bn_nominal=[1.0, np.nan],
        investment_usd_bn_used=[1.0, 2.0],
        investment_basis=["reported", "fx_converted"],
    )
    assert build_episodes(converted).iloc[0]["member_investment_usd_bn"] == "1.0;2.0"
    shuffled = hand_cases(
        [
            {"case_id": "late", "iso3": "AAA", "opening_year": 2007, "investment_usd_bn_nominal": 2.0},
            {"case_id": "early", "iso3": "AAA", "opening_year": 2005, "investment_usd_bn_nominal": 1.0},
        ]
    )
    listed = build_episodes(shuffled).iloc[0]
    assert listed["member_case_ids"] == "early;late" and listed["member_investment_usd_bn"] == "1.0;2.0"


def test_openings_before_2000_are_listed_one_per_row_and_never_merged():
    flags = [True] * 5
    eps = build_episodes(one_economy([1996, 1998, 1999, 2001, 2003], panel_feasible=flags))
    assert eps["case_id"].tolist() == ["MAC_1996_ep", "MAC_1998_ep", "MAC_1999_ep", "MAC_2001_ep"]
    assert eps["n_openings"].tolist() == [1, 1, 1, 2]
    assert eps["member_case_ids"].tolist() == ["MAC_1996_0", "MAC_1998_1", "MAC_1999_2", "MAC_2001_3;MAC_2003_4"]
    assert eps["member_opening_years"].tolist() == ["1996", "1998", "1999", "2001;2003"]
    assert eps["panel_feasible"].tolist() == [False, False, False, True]
    assert eps["next_start_year"].iloc[:3].tolist() == [1998.0, 1999.0, 2001.0] and np.isnan(eps["next_start_year"].iloc[3])
    assert eps["opening_year"].tolist() == [1996, 1998, 1999, 2001]


def test_an_opening_in_1999_is_not_merged_with_one_in_2000_but_2000_and_2001_are():
    assert build_episodes(one_economy([1999, 2000]))["opening_year"].tolist() == [1999, 2000]
    merged = build_episodes(one_economy([2000, 2001]))
    assert merged["opening_year"].tolist() == [2000] and merged["n_openings"].tolist() == [2]


def test_a_large_merge_gap_still_leaves_the_early_openings_alone():
    eps = build_episodes(one_economy([1990, 1995, 2000, 2010]), merge_gap=100)
    assert eps["member_opening_years"].tolist() == ["1990", "1995", "2000;2010"]
    assert eps["n_openings"].tolist() == [1, 1, 2] and eps["panel_feasible"].tolist() == [False, False, True]


def test_unmerged_openings_of_one_year_before_2000_get_numbered_identifiers_and_the_later_start_as_next_start():
    eps = build_episodes(one_economy([1998, 1998, 2000]))
    assert eps["case_id"].tolist() == ["MAC_1998_ep", "MAC_1998_ep2", "MAC_2000_ep"]
    assert eps["case_id"].is_unique and eps["n_openings"].tolist() == [1, 1, 1]
    assert eps["member_case_ids"].tolist() == ["MAC_1998_0", "MAC_1998_1", "MAC_2000_2"]
    assert eps["next_start_year"].iloc[:2].tolist() == [2000.0, 2000.0] and np.isnan(eps["next_start_year"].iloc[2])
    triple = build_episodes(one_economy([1997, 1997, 1997]))
    assert triple["case_id"].tolist() == ["MAC_1997_ep", "MAC_1997_ep2", "MAC_1997_ep3"]


def test_episodes_are_the_same_for_every_order_of_the_input_rows():
    cases = hand_cases(
        [
            {"case_id": "k2", "iso3": "AAA", "opening_year": 2001, "investment_usd_bn_nominal": 2.0, "opening_date": "2001-05-01"},
            {"case_id": "k1", "iso3": "AAA", "opening_year": 2001, "investment_usd_bn_nominal": 1.0, "opening_date": "2001-02-01"},
            {"case_id": "k3", "iso3": "AAA", "opening_year": 2004, "investment_usd_bn_nominal": 4.0},
            {"case_id": "j2", "iso3": "AAA", "opening_year": 1998},
            {"case_id": "j1", "iso3": "AAA", "opening_year": 1998},
            {"case_id": "m1", "iso3": "BBB", "opening_year": 2001},
        ]
    )
    reference = build_episodes(cases)
    assert reference["case_id"].tolist() == ["AAA_1998_ep", "AAA_1998_ep2", "AAA_2001_ep", "BBB_2001_ep"]
    assert reference["member_case_ids"].tolist() == ["j1", "j2", "k1;k2;k3", "m1"]
    assert reference["member_investment_usd_bn"].tolist() == ["", "", "1.0;2.0;4.0", ""]
    assert reference["opening_date"].iloc[2] == "2001-02-01"
    rng = np.random.default_rng(7)
    orders = [list(range(len(cases)))[::-1]] + [list(rng.permutation(len(cases))) for _ in range(150)]
    for order in orders:
        shuffled = cases.iloc[order]
        pd.testing.assert_frame_equal(build_episodes(shuffled), reference)
    pd.testing.assert_frame_equal(build_episodes(cases.iloc[orders[0]].reset_index(drop=True)), reference)
    assert len({tuple(order) for order in orders}) > 100


def test_episode_table_layout_and_dtypes():
    cases = hand_cases(
        [{"case_id": f"B{i}", "iso3": "BBB", "opening_year": y, "investment_usd_bn_nominal": 1.0} for i, y in enumerate([2012, 2001])]
        + [{"case_id": f"A{i}", "iso3": "AAA", "opening_year": y} for i, y in enumerate([2003, 2004])]
    )
    eps = build_episodes(cases)
    assert tuple(eps.columns) == EPISODE_COLUMNS and EPISODE_COLUMNS[:21] == CASE_COLUMNS
    assert eps["case_id"].is_unique and eps["case_id"].tolist() == ["AAA_2003_ep", "BBB_2001_ep", "BBB_2012_ep"]
    assert eps["opening_year"].dtype == np.int64 and eps["n_openings"].dtype == np.int64 and eps["is_integrated_resort_any"].dtype == np.int64
    for name in ("investment_usd_bn_used", "investment_usd_bn_nominal", "investment_share_known", "next_start_year", "first_year_attendance_m"):
        assert eps[name].dtype == np.float64, name
    assert str(eps["in_wdi_panel"].dtype) == "boolean" and str(eps["panel_feasible"].dtype) == "boolean"
    assert eps.index.equals(pd.RangeIndex(3))


def test_input_order_does_not_matter_and_the_input_is_not_modified():
    cases = hand_cases(
        [{"case_id": f"{iso}{i}", "iso3": iso, "opening_year": y, "investment_usd_bn_nominal": 0.5 + i}
         for iso, years in [("AAA", [2000, 2004, 2012]), ("BBB", [2003, 2005]), ("CCC", [2010])] for i, y in enumerate(years)]
    )
    snapshot = cases.copy()
    reference = build_episodes(cases)
    shuffled = cases.sample(frac=1.0, random_state=np.random.RandomState(4)).reset_index(drop=True)
    pd.testing.assert_frame_equal(build_episodes(shuffled), reference)
    pd.testing.assert_frame_equal(cases, snapshot)


def test_only_the_three_required_columns_are_needed():
    bare = pd.DataFrame({"case_id": ["a", "b", "c"], "iso3": ["XXA", "XXA", "XXB"], "opening_year": [2001, 2004, 2001]})
    eps = build_episodes(bare)
    assert tuple(eps.columns) == EPISODE_COLUMNS and eps["case_id"].tolist() == ["XXA_2001_ep", "XXB_2001_ep"]
    assert eps["n_openings"].tolist() == [2, 1] and eps["investment_usd_bn_used"].isna().all()
    assert eps["investment_basis"].tolist() == ["missing", "missing"] and eps["is_integrated_resort_any"].tolist() == [0, 0]
    assert len(build_episodes(bare.iloc[0:0])) == 0


@pytest.mark.parametrize("merge_gap", [-1, 2.5, True, "3", None])
def test_invalid_merge_gap_raises(merge_gap):
    with pytest.raises(ValueError, match="merge_gap"):
        build_episodes(one_economy([2005]), merge_gap=merge_gap)


def test_build_episodes_requires_identifier_economy_and_year():
    for name in ("case_id", "iso3", "opening_year"):
        with pytest.raises(ValueError, match=name):
            build_episodes(one_economy([2005]).drop(columns=[name]))


def test_every_case_of_the_simulated_catalogue_is_its_own_episode(cases):
    eps = build_episodes(cases)
    assert len(eps) == len(cases) and (eps["n_openings"] == 1).all() and eps["next_start_year"].isna().all()
    merged = eps.merge(cases, left_on="member_case_ids", right_on="case_id", suffixes=("", "_c"))
    assert len(merged) == len(cases)
    assert (merged["opening_year"] == merged["opening_year_c"]).all() and (merged["iso3"] == merged["iso3_c"]).all()
    np.testing.assert_allclose(merged["investment_usd_bn_used"], merged["investment_usd_bn_used_c"])
    assert (merged["category"] == merged["category_c"]).all() and (merged["attraction"] == merged["attraction_c"]).all()
    assert merged["is_integrated_resort_any"].tolist() == [int(c in INTEGRATED_CATEGORIES) for c in merged["category_c"]]


# ----------------------------------------------------------------------------
# Episodes with a panel: a synthetic test world in which economies host several cases
# ----------------------------------------------------------------------------
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


def test_clustered_catalogue_collapses_to_thirteen_episodes(cluster_cases, episodes):
    sizes = cluster_cases.groupby("iso3").size()
    assert len(cluster_cases) == 16 and sorted(sizes[sizes > 1]) == [2, 3]
    assert len(episodes) == 13 and episodes["n_openings"].sum() == 16
    assert sorted(episodes["n_openings"][episodes["n_openings"] > 1]) == [2, 3]
    assert episodes["case_id"].is_unique and episodes["next_start_year"].isna().all()
    for row in episodes.itertuples():
        members = cluster_cases.set_index("case_id").loc[row.member_case_ids.split(";")]
        assert row.opening_year == members["opening_year"].min() and (members["iso3"] == row.iso3).all()
        assert row.member_opening_years == ";".join(str(y) for y in sorted(members["opening_year"]))
        assert row.investment_usd_bn_used == pytest.approx(members["investment_usd_bn_used"].sum())
        assert members["category"].iloc[int(np.argmax(members["investment_usd_bn_used"].to_numpy()))] == row.category


def test_shorter_merge_gap_splits_the_clusters_and_sets_next_start_years(cluster_cases):
    eps = build_episodes(cluster_cases, merge_gap=2)
    assert len(eps) > 13
    for iso3, sub in eps.groupby("iso3"):
        assert sub["opening_year"].is_monotonic_increasing
        assert sub["next_start_year"].iloc[:-1].tolist() == sub["opening_year"].iloc[1:].astype(float).tolist()
        assert np.isnan(sub["next_start_year"].iloc[-1])
    assert eps["next_start_year"].notna().any()


def test_episode_feasibility_uses_the_shortened_window_and_every_catalogue_opening(cluster_panel, cluster_cases, episodes):
    feas = check_feasibility(episodes, cluster_panel, catalogue=cluster_cases)
    assert feas["case_id"].tolist() == episodes["case_id"].tolist()
    for row in episodes.to_dict("records"):
        sub = cluster_panel.loc[(cluster_panel["iso3"] == row["iso3"]) & cluster_panel["year"].between(1995, 2019) & cluster_panel["receipts_pct_gdp"].notna()]
        got = feas.loc[feas["case_id"] == row["case_id"]].iloc[0]
        end = min(row["opening_year"] + 4, 2019)
        assert got["window_end"] == end
        assert got["n_pre"] == (sub["year"] < row["opening_year"]).sum()
        assert got["n_post"] == ((sub["year"] >= row["opening_year"]) & (sub["year"] <= end)).sum()
        assert got["n_donors"] == len(brute_force_donors(cluster_panel, cluster_cases, row))
    assert int(feas["feasible"].sum()) == 11
    pd.testing.assert_frame_equal(check_feasibility(episodes, cluster_panel), feas)


def test_next_start_year_shortens_the_window_in_feasibility(cluster_panel, cluster_cases):
    eps = build_episodes(cluster_cases, merge_gap=2)
    feas = check_feasibility(eps, cluster_panel, catalogue=cluster_cases)
    assert (feas["window_end"] < feas["opening_year"] + 4).any()
    for row in eps.to_dict("records"):
        expected = min(row["opening_year"] + 4, 2019)
        if not np.isnan(row["next_start_year"]):
            expected = min(expected, int(row["next_start_year"]) - 1)
        assert feas.loc[feas["case_id"] == row["case_id"], "window_end"].iloc[0] == expected
    no_cap = check_feasibility(eps.drop(columns=["next_start_year"]), cluster_panel, catalogue=cluster_cases)
    assert (no_cap["window_end"] >= feas["window_end"]).all() and (no_cap["window_end"] > feas["window_end"]).any()
    assert (no_cap["n_post"] >= feas["n_post"]).all()


def test_donor_pools_of_episodes_exclude_every_opening_of_the_catalogue(cluster_panel, cluster_cases, episodes):
    naive = episodes.drop(columns=["member_opening_years"])
    naive_pool_larger = 0
    for row in episodes.to_dict("records"):
        expected = brute_force_donors(cluster_panel, cluster_cases, row)
        assert donor_pool(cluster_panel, row, episodes, catalogue=cluster_cases) == expected
        assert donor_pool(cluster_panel, row, episodes) == expected
        assert donor_pool(cluster_panel, row["case_id"], episodes) == expected
        start_years_only = donor_pool(cluster_panel, row, naive)
        assert set(expected) <= set(start_years_only)
        naive_pool_larger += int(len(start_years_only) > len(expected))
    assert naive_pool_larger >= 1


def test_catalogue_argument_overrides_the_openings_listed_in_the_episode_table(cluster_panel, cluster_cases, episodes):
    row = episodes.loc[episodes["n_openings"] == 1].iloc[-1]
    no_other = cluster_cases.iloc[0:0]
    assert donor_pool(cluster_panel, row, episodes, catalogue=no_other) == sorted(
        set(outcome_matrix(cluster_panel, "receipts_pct_gdp", 1995, 2019).dropna(axis=1).columns) - {row["iso3"]}
    )
    with pytest.raises(ValueError, match="catalogue lack"):
        donor_pool(cluster_panel, row, episodes, catalogue=cluster_cases.drop(columns=["iso3"]))
    with pytest.raises(ValueError, match="member_opening_years"):
        donor_pool(cluster_panel, row, episodes.assign(member_opening_years="2004;x"))


def episode_capex_by_hand(panel: pd.DataFrame, row, catalogue: pd.DataFrame, last_year: int = 2019, horizon: int = 5) -> float:
    """Capex of one episode with plain pandas.

    The members that open in the post window count, each as a percentage of the
    GDP of the year before its own opening.
    """
    gdp = panel.loc[panel["iso3"] == row.iso3].set_index("year")["gdp_usd"]
    members = catalogue.set_index("case_id").loc[row.member_case_ids.split(";")]
    start = int(row.opening_year)
    end = min(start + horizon - 1, last_year)
    if not np.isnan(row.next_start_year):
        end = min(end, int(row.next_start_year) - 1)
    shares = []
    for member in members.itertuples():
        if member.opening_year > max(end, start) or np.isnan(member.investment_usd_bn_used):
            continue
        share = 100.0 * member.investment_usd_bn_used * 1e9 / gdp.get(member.opening_year - 1, np.nan)
        if not np.isfinite(share):
            return float("nan")
        shares.append(share)
    return float(sum(shares)) if shares else float("nan")


def episode_by_hand(panel: pd.DataFrame, row, catalogue: pd.DataFrame, window: int = 3) -> dict:
    """Features of one episode from plain pandas, using the member-level investment and the resort indicator."""
    out = by_hand(panel, row, window)
    out["capex_pct_gdp"] = episode_capex_by_hand(panel, row, catalogue)
    out["is_integrated_resort"] = float(row.is_integrated_resort_any)
    return out


@pytest.mark.parametrize("window", [1, 3])
def test_episode_features_match_a_hand_computation(cluster_panel, cluster_cases, episodes, window):
    feats = case_features(episodes, cluster_panel, window=window)
    assert feats["case_id"].tolist() == episodes["case_id"].tolist()
    assert_features_equal(feats, [episode_by_hand(cluster_panel, r, cluster_cases, window) for r in episodes.itertuples()])
    shared = episodes.loc[episodes["n_openings"] > 1]
    assert len(shared) == 2 and feats.set_index("case_id").loc[shared["case_id"], "capex_pct_gdp"].notna().all()


def test_episode_capex_in_the_simulated_world_differs_from_the_whole_investment_over_the_start_gdp(cluster_panel, cluster_cases, episodes):
    feats = case_features(episodes, cluster_panel).set_index("case_id")
    gdp = cluster_panel.set_index(["iso3", "year"])["gdp_usd"]
    for row in episodes.loc[episodes["n_openings"] > 1].itertuples():
        whole = 100.0 * row.investment_usd_bn_used * 1e9 / gdp.loc[(row.iso3, row.opening_year - 1)]
        got = feats.loc[row.case_id, "capex_pct_gdp"]
        assert got == pytest.approx(episode_capex_by_hand(cluster_panel, row, cluster_cases), rel=1e-12)
        assert abs(got - whole) > 1e-3 * whole, row.case_id
    members = cluster_cases.set_index("case_id")
    two = episodes.loc[episodes["n_openings"] == 2].iloc[0]
    first, second = two["member_case_ids"].split(";")
    assert members.loc[second, "opening_year"] - members.loc[first, "opening_year"] == 5
    expected = 100.0 * members.loc[first, "investment_usd_bn_used"] * 1e9 / gdp.loc[(two["iso3"], two["opening_year"] - 1)]
    assert feats.loc[two["case_id"], "capex_pct_gdp"] == pytest.approx(expected, rel=1e-12)


def test_episode_features_equal_case_features_for_single_opening_episodes(cluster_panel, cluster_cases, episodes):
    single = episodes.loc[episodes["n_openings"] == 1]
    by_episode = case_features(single, cluster_panel)
    by_case = case_features(cluster_cases, cluster_panel).set_index("case_id").loc[single["member_case_ids"]]
    np.testing.assert_allclose(by_episode[list(FEATURE_COLUMNS)].to_numpy(float), by_case[list(FEATURE_COLUMNS)].to_numpy(float), rtol=1e-12, equal_nan=True)


def test_episode_features_use_only_years_before_the_start_year_except_the_gdp_before_a_later_opening_in_the_window(
    cluster_panel, cluster_cases, episodes
):
    original = case_features(episodes, cluster_panel)
    sources = ["receipts_usd", "arrivals", "gdp_usd", "pop", "air_pax", "receipts_pct_gdp", "log_receipts", "log_gdp_pc",
               "log_pop", "log_gdp", "gdp_per_capita_usd", "arrivals_per_capita", "receipts_per_arrival_usd",
               "log_receipts_per_arrival", "air_pax_per_capita"]
    corrupted = cluster_panel.copy()
    for r in episodes.itertuples():
        mask = (corrupted["iso3"] == r.iso3) & (corrupted["year"] >= r.opening_year)
        corrupted.loc[mask, sources] = corrupted.loc[mask, sources] * 11.0 + 3.0
        corrupted.loc[mask & (corrupted["year"] % 2 == 1), sources] = np.nan
    assert not corrupted.equals(cluster_panel)
    others = ["case_id", "iso3", "opening_year", *[c for c in FEATURE_COLUMNS if c != "capex_pct_gdp"]]
    for window in (1, 3, 5):
        got = case_features(episodes, corrupted, window=window)
        pd.testing.assert_frame_equal(got[others], case_features(episodes, cluster_panel, window=window)[others], check_exact=True)
    after = case_features(episodes, corrupted).set_index("case_id")["capex_pct_gdp"]
    before = original.set_index("case_id")["capex_pct_gdp"]
    changed = 0
    for r in episodes.itertuples():
        by_hand_value = episode_capex_by_hand(corrupted, r, cluster_cases)
        assert after[r.case_id] == pytest.approx(by_hand_value, rel=1e-12, nan_ok=True), r.case_id
        end = min(r.opening_year + 4, 2019)
        later_in_window = any(r.opening_year < y <= end for y in map(int, r.member_opening_years.split(";")))
        if later_in_window:
            changed += 1
            assert not before[r.case_id] == after[r.case_id], r.case_id
        else:
            assert before[r.case_id] == after[r.case_id], r.case_id
    assert changed == 1
    shared = episodes.loc[episodes["n_openings"] > 1].iloc[0]
    altered = cluster_panel.copy()
    mask = (altered["iso3"] == shared["iso3"]) & (altered["year"] == shared["opening_year"] - 1)
    altered.loc[mask, ["log_gdp_pc", "receipts_pct_gdp", "log_receipts", "gdp_usd"]] *= 1.5
    after_change = case_features(episodes, altered).set_index("case_id").loc[shared["case_id"]]
    row_before = original.set_index("case_id").loc[shared["case_id"]]
    assert row_before["log_gdp_pc"] != after_change["log_gdp_pc"] and row_before["capex_pct_gdp"] != after_change["capex_pct_gdp"]


def test_episode_resort_indicator_comes_from_the_extra_column(cluster_panel, episodes):
    row = episodes.iloc[[0]].copy()
    flagged = row.assign(is_integrated_resort_any=1, category="theme_park")
    plain = row.assign(is_integrated_resort_any=0, category="integrated_resort")
    assert case_features(flagged, cluster_panel).iloc[0]["is_integrated_resort"] == 1.0
    assert case_features(plain, cluster_panel).iloc[0]["is_integrated_resort"] == 0.0
    dropped = plain.drop(columns=["is_integrated_resort_any"])
    assert case_features(dropped, cluster_panel).iloc[0]["is_integrated_resort"] == 1.0


# ----------------------------------------------------------------------------
# Episode capex on a hand-made panel
# ----------------------------------------------------------------------------
def hand_panel(iso3: str = "MAC", first: int = 1995, last: int = 2022) -> pd.DataFrame:
    """Panel of one economy whose GDP in year ``t`` is 100 billion times ``t - 1994``; the other columns are constant."""
    years = np.arange(first, last + 1)
    frame = pd.DataFrame({"iso3": iso3, "year": years, "gdp_usd": 100.0e9 * (years - 1994)})
    for name in LEVELS:
        frame[name] = 1.0
    frame["log_receipts"] = np.log(years - 1990.0)
    return frame


def gdp_of(year: int) -> float:
    """GDP of the hand-made panel in ``year``."""
    return 100.0e9 * (year - 1994)


def episode_capex(openings, investments, panel=None, **kwargs) -> float:
    """Capex of the single episode that ``build_episodes`` forms from openings of the economy MAC."""
    table = build_episodes(one_economy(openings, investment_usd_bn_nominal=investments))
    assert len(table) == 1
    return float(case_features(table, hand_panel() if panel is None else panel, **kwargs).iloc[0]["capex_pct_gdp"])


def share(investment: float, year: int) -> float:
    """Investment in billions of dollars as a percentage of GDP of ``year``."""
    return 100.0 * investment * 1.0e9 / gdp_of(year)


def test_episode_capex_sums_the_openings_of_the_window_each_over_the_gdp_before_its_own_opening():
    got = episode_capex([2004, 2008, 2011], [1.0, 2.0, 4.0])
    assert got == pytest.approx(share(1.0, 2003) + share(2.0, 2007), rel=1e-13)
    assert got == pytest.approx(0.1111111111111111 + 0.15384615384615385, rel=1e-12)
    whole_over_start = share(7.0, 2003)
    assert whole_over_start == pytest.approx(0.7777777777777778) and abs(got - whole_over_start) > 0.5


def test_a_member_in_the_last_year_of_the_window_counts_and_one_year_later_does_not():
    assert episode_capex([2004, 2008], [1.0, 2.0]) == pytest.approx(share(1.0, 2003) + share(2.0, 2007), rel=1e-13)
    assert episode_capex([2004, 2009], [1.0, 2.0], post_horizon=5) == pytest.approx(share(1.0, 2003), rel=1e-13)
    assert episode_capex([2004, 2009], [1.0, 2.0], post_horizon=6) == pytest.approx(share(1.0, 2003) + share(2.0, 2008), rel=1e-13)


@pytest.mark.parametrize(
    "kwargs,counted",
    [
        ({"post_horizon": 1}, [(1.0, 2003)]),
        ({"post_horizon": 3}, [(1.0, 2003)]),
        ({"post_horizon": 5}, [(1.0, 2003), (2.0, 2007)]),
        ({"post_horizon": 8}, [(1.0, 2003), (2.0, 2007), (4.0, 2010)]),
        ({"post_horizon": None}, [(1.0, 2003), (2.0, 2007), (4.0, 2010)]),
        ({"post_horizon": None, "last_year": 2010}, [(1.0, 2003), (2.0, 2007)]),
        ({"last_year": 2007}, [(1.0, 2003)]),
        ({"last_year": 2008}, [(1.0, 2003), (2.0, 2007)]),
    ],
)
def test_the_horizon_and_the_last_year_decide_which_openings_enter_the_capex(kwargs, counted):
    got = episode_capex([2004, 2008, 2011], [1.0, 2.0, 4.0], **kwargs)
    assert got == pytest.approx(sum(share(inv, year) for inv, year in counted), rel=1e-13)


def test_the_next_start_year_of_an_episode_ends_the_window_for_the_capex():
    table = build_episodes(one_economy([2004, 2008, 2011], investment_usd_bn_nominal=[1.0, 2.0, 4.0]))
    panel = hand_panel()
    for following, counted in [(2008.0, [(1.0, 2003)]), (2009.0, [(1.0, 2003), (2.0, 2007)]), (2012.0, [(1.0, 2003), (2.0, 2007)]), (np.nan, [(1.0, 2003), (2.0, 2007)])]:
        got = case_features(table.assign(next_start_year=following), panel).iloc[0]["capex_pct_gdp"]
        assert got == pytest.approx(sum(share(inv, year) for inv, year in counted), rel=1e-13), following
    uncapped = case_features(table.assign(next_start_year=np.nan), panel, post_horizon=None).iloc[0]["capex_pct_gdp"]
    assert uncapped == pytest.approx(share(1.0, 2003) + share(2.0, 2007) + share(4.0, 2010), rel=1e-13)
    capped = case_features(table.assign(next_start_year=2011.0), panel, post_horizon=None).iloc[0]["capex_pct_gdp"]
    assert capped == pytest.approx(share(1.0, 2003) + share(2.0, 2007), rel=1e-13)


def test_the_first_opening_always_counts_even_when_it_lies_after_the_last_year():
    got = episode_capex([2021, 2022], [3.0, 5.0], last_year=2019)
    assert got == pytest.approx(share(3.0, 2020), rel=1e-13)
    assert episode_capex([2021], [3.0], last_year=2019, post_horizon=None) == pytest.approx(share(3.0, 2020), rel=1e-13)


def test_members_without_a_known_investment_are_skipped_and_zero_is_a_known_investment():
    assert episode_capex([2004, 2006, 2008], [1.0, np.nan, 4.0]) == pytest.approx(share(1.0, 2003) + share(4.0, 2007), rel=1e-13)
    assert episode_capex([2004, 2006], [np.nan, 2.0]) == pytest.approx(share(2.0, 2005), rel=1e-13)
    assert np.isnan(episode_capex([2004, 2006], [np.nan, np.nan]))
    assert episode_capex([2004], [0.0]) == 0.0
    assert episode_capex([2004, 2006], [0.0, 2.0]) == pytest.approx(share(2.0, 2005), rel=1e-13)


def test_a_counted_member_without_gdp_makes_the_capex_missing_and_an_uncounted_one_does_not():
    panel = hand_panel()
    panel.loc[panel["year"] == 2007, "gdp_usd"] = np.nan
    assert np.isnan(episode_capex([2004, 2008], [1.0, 2.0], panel=panel))
    assert episode_capex([2004, 2008], [1.0, 2.0], panel=panel, post_horizon=4) == pytest.approx(share(1.0, 2003), rel=1e-13)
    assert episode_capex([2004, 2008], [1.0, np.nan], panel=panel) == pytest.approx(share(1.0, 2003), rel=1e-13)
    zero = hand_panel()
    zero.loc[zero["year"] == 2003, "gdp_usd"] = 0.0
    assert np.isnan(episode_capex([2004, 2008], [1.0, 2.0], panel=zero))
    assert np.isnan(episode_capex([2004], [1.0], panel=hand_panel("OTHER")))


def test_a_table_without_the_member_columns_is_read_as_single_openings():
    table = build_episodes(one_economy([2004, 2008], investment_usd_bn_nominal=[1.0, 2.0]))
    panel = hand_panel()
    plain = table.drop(columns=["member_opening_years", "member_investment_usd_bn"])
    assert case_features(plain, panel).iloc[0]["capex_pct_gdp"] == pytest.approx(share(3.0, 2003), rel=1e-13)
    assert case_features(table.drop(columns=["member_investment_usd_bn"]), panel).iloc[0]["capex_pct_gdp"] == pytest.approx(share(3.0, 2003), rel=1e-13)
    catalogue = one_economy([2004, 2008], investment_usd_bn_nominal=[1.0, 2.0])
    got = case_features(catalogue, panel).set_index("case_id")["capex_pct_gdp"]
    assert got.tolist() == pytest.approx([share(1.0, 2003), share(2.0, 2007)], rel=1e-13)


def test_a_malformed_member_investment_list_is_rejected():
    table = build_episodes(one_economy([2004, 2008, 2011], investment_usd_bn_nominal=[1.0, 2.0, 4.0]))
    panel = hand_panel()
    with pytest.raises(ValueError, match="member_investment_usd_bn"):
        case_features(table.assign(member_investment_usd_bn="1.0;2.0"), panel)
    with pytest.raises(ValueError, match="member_investment_usd_bn"):
        case_features(table.assign(member_investment_usd_bn="1.0;x;4.0"), panel)
    with pytest.raises(ValueError, match="member_opening_years"):
        case_features(table.assign(member_opening_years="2004;x;2011"), panel)
    with pytest.raises(ValueError, match="post_horizon"):
        case_features(table, panel, post_horizon=0)


def test_episode_capex_survives_a_round_trip_through_a_csv_file(tmp_path):
    cases = hand_cases(
        [
            {"case_id": "m1", "iso3": "MAC", "opening_year": 2004, "investment_usd_bn_nominal": 1.0},
            {"case_id": "m2", "iso3": "MAC", "opening_year": 2006, "investment_usd_bn_nominal": np.nan},
            {"case_id": "m3", "iso3": "MAC", "opening_year": 2008, "investment_usd_bn_nominal": 2.5},
            {"case_id": "x1", "iso3": "XXA", "opening_year": 2005, "investment_usd_bn_nominal": np.nan},
            {"case_id": "x2", "iso3": "XXB", "opening_year": 2005, "investment_usd_bn_nominal": 3.0},
            {"case_id": "x3", "iso3": "XXB", "opening_year": 2020, "investment_usd_bn_nominal": 0.0},
        ]
    )
    table = build_episodes(cases)
    assert table["member_investment_usd_bn"].tolist() == ["1.0;;2.5", "", "3.0", "0.0"]
    panel = pd.concat([hand_panel("MAC"), hand_panel("XXA"), hand_panel("XXB")], ignore_index=True)
    direct = case_features(table, panel)
    assert direct["capex_pct_gdp"].iloc[0] == pytest.approx(share(1.0, 2003) + share(2.5, 2007), rel=1e-13)
    assert np.isnan(direct["capex_pct_gdp"].iloc[1]) and direct["capex_pct_gdp"].iloc[2] == pytest.approx(share(3.0, 2004), rel=1e-13)
    assert direct["capex_pct_gdp"].iloc[3] == 0.0
    path = tmp_path / "episodes.csv"
    table.to_csv(path, index=False)
    read = pd.read_csv(path)
    pd.testing.assert_frame_equal(case_features(read, panel), direct)


# ----------------------------------------------------------------------------
# Investment of the first opening
# ----------------------------------------------------------------------------
def episode_row(openings, investments, panel=None, **kwargs) -> pd.Series:
    """Feature row of the single episode that ``build_episodes`` forms from openings of the economy MAC."""
    table = build_episodes(one_economy(openings, investment_usd_bn_nominal=investments))
    assert len(table) == 1
    return case_features(table, hand_panel() if panel is None else panel, **kwargs).iloc[0]


def test_a_merged_episode_with_a_later_cost_has_a_smaller_first_opening_share_than_the_total():
    row = episode_row([2004, 2008, 2011], [1.0, 2.0, 4.0])
    assert row["capex_first_pct_gdp"] == pytest.approx(share(1.0, 2003), rel=1e-13)
    assert row["capex_pct_gdp"] == pytest.approx(share(1.0, 2003) + share(2.0, 2007), rel=1e-13)
    assert row["capex_pct_gdp"] > 2.0 * row["capex_first_pct_gdp"]
    assert list(case_features(build_episodes(one_economy([2004, 2008], investment_usd_bn_nominal=[1.0, 2.0])), hand_panel()).columns)[-1] == "capex_first_pct_gdp"


def test_an_episode_whose_first_cost_is_unknown_has_no_first_opening_share_although_the_total_uses_its_later_member():
    row = episode_row([2004, 2008], [np.nan, 2.0])
    assert np.isnan(row["capex_first_pct_gdp"])
    assert row["capex_pct_gdp"] == pytest.approx(share(2.0, 2007), rel=1e-13)
    both_unknown = episode_row([2004, 2008], [np.nan, np.nan])
    assert np.isnan(both_unknown["capex_first_pct_gdp"]) and np.isnan(both_unknown["capex_pct_gdp"])


def test_a_known_first_cost_with_an_unknown_later_cost_gives_the_same_first_opening_share_and_total():
    row = episode_row([2004, 2008], [1.5, np.nan])
    assert row["capex_first_pct_gdp"] == row["capex_pct_gdp"] == pytest.approx(share(1.5, 2003), rel=1e-13)


def test_a_zero_first_cost_is_known_and_a_single_opening_has_the_same_share_in_both_columns():
    row = episode_row([2004, 2008], [0.0, 2.0])
    assert row["capex_first_pct_gdp"] == 0.0 and row["capex_pct_gdp"] == pytest.approx(share(2.0, 2007), rel=1e-13)
    single = episode_row([2006], [3.0])
    assert single["capex_first_pct_gdp"] == single["capex_pct_gdp"] == pytest.approx(share(3.0, 2005), rel=1e-13)
    unknown = episode_row([2006], [np.nan])
    assert np.isnan(unknown["capex_first_pct_gdp"]) and np.isnan(unknown["capex_pct_gdp"])


def test_openings_in_the_first_year_of_an_episode_all_count_as_the_first_opening():
    row = episode_row([2004, 2004, 2007], [1.0, 2.0, 4.0])
    assert row["capex_first_pct_gdp"] == pytest.approx(share(3.0, 2003), rel=1e-13)
    assert row["capex_pct_gdp"] == pytest.approx(share(3.0, 2003) + share(4.0, 2006), rel=1e-13)
    only_same_year = episode_row([2004, 2004], [1.0, 2.0])
    assert only_same_year["capex_first_pct_gdp"] == only_same_year["capex_pct_gdp"] == pytest.approx(share(3.0, 2003), rel=1e-13)
    one_known = episode_row([2004, 2004, 2007], [np.nan, 2.0, 4.0])
    assert one_known["capex_first_pct_gdp"] == pytest.approx(share(2.0, 2003), rel=1e-13)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"post_horizon": 1}, {"post_horizon": None}, {"last_year": 2005}, {"last_year": 2003}, {"window": 1}, {"window": 5}],
)
def test_the_first_opening_share_does_not_depend_on_the_post_window_or_the_feature_window(kwargs):
    row = episode_row([2004, 2008, 2011], [1.0, 2.0, 4.0], **kwargs)
    assert row["capex_first_pct_gdp"] == pytest.approx(share(1.0, 2003), rel=1e-13)


def test_the_first_opening_share_ignores_the_next_episode():
    table = build_episodes(one_economy([2004, 2008, 2011], investment_usd_bn_nominal=[1.0, 2.0, 4.0]))
    for following in (2005.0, 2008.0, 2012.0, np.nan):
        got = case_features(table.assign(next_start_year=following), hand_panel()).iloc[0]
        assert got["capex_first_pct_gdp"] == pytest.approx(share(1.0, 2003), rel=1e-13), following


def test_the_first_opening_share_uses_no_panel_year_from_the_opening_year_on():
    clean = episode_row([2004, 2008], [1.0, 2.0])
    corrupted = hand_panel()
    late = corrupted["year"] >= 2004
    corrupted.loc[late, "gdp_usd"] = np.nan
    corrupted.loc[late, "log_receipts"] = 99.0
    got = episode_row([2004, 2008], [1.0, 2.0], panel=corrupted)
    assert got["capex_first_pct_gdp"] == clean["capex_first_pct_gdp"] == pytest.approx(share(1.0, 2003), rel=1e-13)
    assert np.isnan(got["capex_pct_gdp"]) and clean["capex_pct_gdp"] > clean["capex_first_pct_gdp"]
    assert corrupted.loc[corrupted["year"] == 2007, "gdp_usd"].isna().all()
    changed = hand_panel()
    changed.loc[changed["year"] == 2007, "gdp_usd"] *= 3.0
    shifted = episode_row([2004, 2008], [1.0, 2.0], panel=changed)
    assert shifted["capex_first_pct_gdp"] == clean["capex_first_pct_gdp"] and shifted["capex_pct_gdp"] != clean["capex_pct_gdp"]


def test_the_first_opening_share_is_missing_without_a_positive_gdp_before_the_opening_or_outside_the_panel():
    zero = hand_panel()
    zero.loc[zero["year"] == 2003, "gdp_usd"] = 0.0
    assert np.isnan(episode_row([2004, 2008], [1.0, 2.0], panel=zero)["capex_first_pct_gdp"])
    absent = hand_panel()
    absent = absent.loc[absent["year"] != 2003]
    assert np.isnan(episode_row([2004, 2008], [1.0, 2.0], panel=absent)["capex_first_pct_gdp"])
    assert np.isnan(episode_row([2004], [1.0], panel=hand_panel("OTHER"))["capex_first_pct_gdp"])
    later_missing = hand_panel()
    later_missing.loc[later_missing["year"] == 2007, "gdp_usd"] = np.nan
    row = episode_row([2004, 2008], [1.0, 2.0], panel=later_missing)
    assert row["capex_first_pct_gdp"] == pytest.approx(share(1.0, 2003), rel=1e-13) and np.isnan(row["capex_pct_gdp"])


def test_a_table_without_the_member_columns_has_a_first_opening_share_equal_to_the_total():
    table = build_episodes(one_economy([2004, 2008], investment_usd_bn_nominal=[1.0, 2.0]))
    panel = hand_panel()
    plain = table.drop(columns=["member_opening_years", "member_investment_usd_bn"])
    got = case_features(plain, panel).iloc[0]
    assert got["capex_first_pct_gdp"] == got["capex_pct_gdp"] == pytest.approx(share(3.0, 2003), rel=1e-13)
    catalogue = one_economy([2004, 2008], investment_usd_bn_nominal=[1.0, 2.0])
    rows = case_features(catalogue, panel)
    np.testing.assert_array_equal(rows["capex_first_pct_gdp"].to_numpy(), rows["capex_pct_gdp"].to_numpy())
    assert rows["capex_first_pct_gdp"].tolist() == pytest.approx([share(1.0, 2003), share(2.0, 2007)], rel=1e-13)


def test_the_first_opening_share_equals_the_total_for_every_case_of_a_catalogue(panel, cases):
    feats = case_features(cases, panel)
    np.testing.assert_array_equal(feats["capex_first_pct_gdp"].to_numpy(), feats["capex_pct_gdp"].to_numpy())
    assert feats["capex_first_pct_gdp"].notna().any()
    outside = cases.iloc[[2]].copy().assign(case_id="Z1", iso3="ZZZ")
    assert np.isnan(case_features(outside, panel).iloc[0]["capex_first_pct_gdp"])


def test_episodes_of_the_simulated_world_have_the_share_of_their_first_member_alone(cluster_panel, cluster_cases, episodes):
    feats = case_features(episodes, cluster_panel).set_index("case_id")
    members = cluster_cases.set_index("case_id")
    gdp = cluster_panel.set_index(["iso3", "year"])["gdp_usd"]
    differs = 0
    for row in episodes.itertuples():
        first = members.loc[row.member_case_ids.split(";")[0]]
        expected = 100.0 * first["investment_usd_bn_used"] * 1e9 / gdp.loc[(row.iso3, row.opening_year - 1)]
        assert feats.loc[row.case_id, "capex_first_pct_gdp"] == pytest.approx(expected, rel=1e-12), row.case_id
        total = feats.loc[row.case_id, "capex_pct_gdp"]
        if row.n_openings == 1:
            assert feats.loc[row.case_id, "capex_first_pct_gdp"] == total
        elif total != feats.loc[row.case_id, "capex_first_pct_gdp"]:
            differs += 1
            assert total > feats.loc[row.case_id, "capex_first_pct_gdp"]
    assert differs == 1


def test_the_first_opening_share_of_an_episode_survives_a_round_trip_through_a_csv_file(tmp_path):
    cases = hand_cases(
        [
            {"case_id": "m1", "iso3": "MAC", "opening_year": 2004, "investment_usd_bn_nominal": np.nan},
            {"case_id": "m2", "iso3": "MAC", "opening_year": 2006, "investment_usd_bn_nominal": 2.0},
            {"case_id": "n1", "iso3": "NAC", "opening_year": 2004, "investment_usd_bn_nominal": 1.0},
            {"case_id": "n2", "iso3": "NAC", "opening_year": 2007, "investment_usd_bn_nominal": 2.5},
        ]
    )
    table = build_episodes(cases)
    panel = pd.concat([hand_panel("MAC"), hand_panel("NAC")], ignore_index=True)
    direct = case_features(table, panel).set_index("iso3")
    assert np.isnan(direct.loc["MAC", "capex_first_pct_gdp"]) and direct.loc["MAC", "capex_pct_gdp"] == pytest.approx(share(2.0, 2005), rel=1e-13)
    assert direct.loc["NAC", "capex_first_pct_gdp"] == pytest.approx(share(1.0, 2003), rel=1e-13)
    path = tmp_path / "episodes.csv"
    table.to_csv(path, index=False)
    pd.testing.assert_frame_equal(case_features(pd.read_csv(path), panel).set_index("iso3"), direct)
