"""Tests for dtt.impact: baseline loader, effect scaling, detection and route decision.

All inputs are hand-made or drawn from seeded random number generators.
"""
from __future__ import annotations

import inspect
import itertools
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy import stats

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dtt import impact, meta  # noqa: E402

EM_DASH = chr(0x2014)

BASELINE_COLUMNS = (
    "gdp_usd_bn",
    "receipts_usd_bn",
    "receipts_pct_gdp",
    "arrivals_million",
    "fx_thb_per_usd",
)
MAIN_VARIABLES = (
    "gdp_current_usd_bn",
    "intl_tourism_receipts_usd_bn",
    "receipts_pct_gdp",
    "intl_arrivals_million",
    "fx_thb_per_usd",
)


# ----------------------------------------------------------------------------
# Hand-made inputs
# ----------------------------------------------------------------------------
def long_rows(year, gdp, receipts, pct, arrivals, fx):
    """Rows of the long baseline file for one year, plus ignored variables."""
    return [
        (year, "gdp_current_usd_bn", gdp),
        (year, "intl_tourism_receipts_usd_bn", receipts),
        (year, "receipts_pct_gdp", pct),
        (year, "intl_arrivals_million", arrivals),
        (year, "fx_thb_per_usd", fx),
        (year, "intl_tourism_receipts_thb_bn", receipts * fx),
        (year, "gdp_current_thb_bn", gdp * fx),
        (year, "alt_receipts_imf_usd_bn", receipts + 1000.0),
        (year, "alt_fx_bot_thb_per_usd", fx + 1000.0),
    ]


def write_long_file(path, rows):
    frame = pd.DataFrame(rows, columns=["year", "variable", "value"])
    frame["unit"] = "unit"
    frame["source"] = "hand-made"
    frame["url"] = "https://example.org/a|https://example.org/b"
    frame["note"] = "ADOPTED, with a comma"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def baseline():
    """Hand-made baseline with two years."""
    return pd.DataFrame(
        {
            "gdp_usd_bn": [500.0, 600.0],
            "receipts_usd_bn": [60.0, 45.0],
            "receipts_pct_gdp": [12.0, 7.5],
            "arrivals_million": [40.0, 35.0],
            "fx_thb_per_usd": [30.0, 35.0],
        },
        index=pd.Index([2019, 2024], name="year"),
    )


# ----------------------------------------------------------------------------
# Baseline loader
# ----------------------------------------------------------------------------
def test_loader_returns_one_row_per_year_with_the_main_variables_only(tmp_path):
    rows = long_rows(2024, 600.0, 45.0, 7.5, 35.0, 35.0)
    rows += long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    path = write_long_file(tmp_path / "baseline.csv", rows)
    out = impact.load_thailand_baseline(path)
    assert tuple(out.columns) == BASELINE_COLUMNS
    assert list(out.index) == [2019, 2024]
    assert out.index.name == "year"
    assert out.index.dtype.kind == "i"
    assert all(dtype == np.float64 for dtype in out.dtypes)
    assert out.loc[2019].to_dict() == {
        "gdp_usd_bn": 500.0,
        "receipts_usd_bn": 60.0,
        "receipts_pct_gdp": 12.0,
        "arrivals_million": 40.0,
        "fx_thb_per_usd": 30.0,
    }
    assert out.loc[2024, "receipts_usd_bn"] == 45.0
    assert out.loc[2024, "fx_thb_per_usd"] == 35.0
    assert out.columns.name is None
    assert not out.isna().any().any()


def test_loader_accepts_a_path_string_and_a_header_with_extra_columns(tmp_path):
    path = write_long_file(tmp_path / "b.csv", long_rows(2020, 1.0, 2.0, 3.0, 4.0, 5.0))
    out = impact.load_thailand_baseline(str(path))
    assert out.shape == (1, 5)
    assert out.loc[2020, "arrivals_million"] == 4.0


def test_loader_reads_files_with_a_byte_order_mark(tmp_path):
    plain = write_long_file(tmp_path / "plain.csv", long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0))
    marked = tmp_path / "marked.csv"
    marked.write_bytes(b"\xef\xbb\xbf" + plain.read_bytes())
    pd.testing.assert_frame_equal(
        impact.load_thailand_baseline(marked), impact.load_thailand_baseline(plain)
    )


def test_loader_leaves_a_missing_value_as_nan(tmp_path):
    rows = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    rows += [r for r in long_rows(2020, 510.0, 12.0, 2.4, 6.0, 31.0) if r[1] != "fx_thb_per_usd"]
    out = impact.load_thailand_baseline(write_long_file(tmp_path / "b.csv", rows))
    assert list(out.index) == [2019, 2020]
    assert np.isnan(out.loc[2020, "fx_thb_per_usd"])
    assert out.loc[2020, "gdp_usd_bn"] == 510.0


def test_loader_rejects_malformed_files(tmp_path):
    good = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    without_variable = [r for r in good if r[1] != "intl_arrivals_million"]
    with pytest.raises(ValueError, match="intl_arrivals_million"):
        impact.load_thailand_baseline(write_long_file(tmp_path / "a.csv", without_variable))
    only_alt = [r for r in good if r[1].startswith("alt_")]
    with pytest.raises(ValueError):
        impact.load_thailand_baseline(write_long_file(tmp_path / "b.csv", only_alt))
    duplicated = good + [(2019, "gdp_current_usd_bn", 501.0)]
    with pytest.raises(ValueError, match="more than once"):
        impact.load_thailand_baseline(write_long_file(tmp_path / "c.csv", duplicated))
    pd.DataFrame({"year": [2019], "value": [1.0]}).to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(ValueError, match="variable"):
        impact.load_thailand_baseline(tmp_path / "d.csv")
    fractional = [(2019.5, v, x) for (_, v, x) in good]
    with pytest.raises(ValueError, match="whole numbers"):
        impact.load_thailand_baseline(write_long_file(tmp_path / "e.csv", fractional))
    text_value = [(y, v, "44.8 (P)" if v == "gdp_current_usd_bn" else x) for (y, v, x) in good]
    with pytest.raises(ValueError):
        impact.load_thailand_baseline(write_long_file(tmp_path / "f.csv", text_value))


def test_loader_ignores_alternative_copies_of_the_main_variables(tmp_path):
    good = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    plain = impact.load_thailand_baseline(write_long_file(tmp_path / "plain.csv", good))
    alternatives = [(2019, "alt_" + name, -1.0) for name in MAIN_VARIABLES]
    alternatives += [(2018, "alt_" + name, "n/a") for name in MAIN_VARIABLES]
    alternatives += [(2019, name + "_alt", 0.0) for name in MAIN_VARIABLES]
    path = write_long_file(tmp_path / "alt.csv", good + alternatives)
    with_alternatives = impact.load_thailand_baseline(path)
    pd.testing.assert_frame_equal(with_alternatives, plain)
    assert list(with_alternatives.index) == [2019]
    assert with_alternatives.loc[2019, "gdp_usd_bn"] == 500.0


@pytest.mark.parametrize("variable", MAIN_VARIABLES)
@pytest.mark.parametrize("bad", [0.0, -3.0, np.inf, -np.inf])
def test_loader_rejects_values_that_are_not_finite_and_positive(tmp_path, variable, bad):
    good = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    rows = [(y, v, bad if v == variable else x) for (y, v, x) in good]
    with pytest.raises(ValueError, match=variable) as info:
        impact.load_thailand_baseline(write_long_file(tmp_path / "bad.csv", rows))
    assert "finite and positive" in str(info.value)


def test_loader_checks_the_values_of_every_year(tmp_path):
    rows = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    rows += [(2015, "gdp_current_usd_bn", -1.0)]
    with pytest.raises(ValueError, match="gdp_current_usd_bn in 2015"):
        impact.load_thailand_baseline(write_long_file(tmp_path / "b.csv", rows))


def test_loader_rejects_a_variable_without_any_value(tmp_path):
    for year in (2019, 2020):
        good = long_rows(year, 500.0, 60.0, 12.0, 40.0, 30.0)
        rows = [(y, v, np.nan if v == "intl_arrivals_million" else x) for (y, v, x) in good]
        with pytest.raises(ValueError, match="no values for the variables.*intl_arrivals_million"):
            impact.load_thailand_baseline(write_long_file(tmp_path / f"b{year}.csv", rows))


def test_loader_requires_the_years_it_is_asked_for_to_be_complete(tmp_path):
    rows = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    rows += [r for r in long_rows(2024, 600.0, 45.0, 7.5, 35.0, 35.0) if r[1] != "receipts_pct_gdp"]
    path = write_long_file(tmp_path / "gap.csv", rows)
    with pytest.raises(ValueError, match=r"receipts_pct_gdp.*2024"):
        impact.load_thailand_baseline(path)
    with pytest.raises(ValueError, match=r"receipts_pct_gdp.*2024"):
        impact.load_thailand_baseline(path, years=[2019, 2024])
    only_2019 = impact.load_thailand_baseline(path, years=2019)
    assert list(only_2019.index) == [2019, 2024]
    assert np.isnan(only_2019.loc[2024, "receipts_pct_gdp"])
    assert only_2019.loc[2019, "receipts_pct_gdp"] == 12.0
    with pytest.raises(ValueError, match="2030"):
        impact.load_thailand_baseline(path, years=[2019, 2030])
    with pytest.raises(ValueError, match="2018"):
        impact.load_thailand_baseline(path, years=2018)


def test_loader_applies_the_default_years_only_when_the_file_contains_them(tmp_path):
    only_2020 = write_long_file(tmp_path / "a.csv", long_rows(2020, 1.0, 2.0, 3.0, 4.0, 5.0))
    assert list(impact.load_thailand_baseline(only_2020).index) == [2020]
    with pytest.raises(ValueError, match="2019"):
        impact.load_thailand_baseline(only_2020, years=2019)
    incomplete_2019 = [
        r for r in long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0) if r[1] != "fx_thb_per_usd"
    ]
    path = write_long_file(
        tmp_path / "b.csv", incomplete_2019 + long_rows(2020, 1.0, 2.0, 3.0, 4.0, 5.0)
    )
    with pytest.raises(ValueError, match=r"fx_thb_per_usd.*2019"):
        impact.load_thailand_baseline(path)
    assert list(impact.load_thailand_baseline(path, years=2020).index) == [2019, 2020]


def test_loader_accepts_years_as_whole_numbers_and_rejects_other_values(tmp_path):
    rows = long_rows(2019, 500.0, 60.0, 12.0, 40.0, 30.0)
    rows += long_rows(2024, 600.0, 45.0, 7.5, 35.0, 35.0)
    path = write_long_file(tmp_path / "b.csv", rows)
    default = impact.load_thailand_baseline(path)
    for years in (2019.0, np.int64(2019), (2019, 2024), [2019.0, 2024.0], np.array([2019, 2024])):
        pd.testing.assert_frame_equal(impact.load_thailand_baseline(path, years=years), default)
    for years in ([], (), "2019", b"2019", 2019.5, True, np.True_, [2019, "2024"], [2019, None]):
        with pytest.raises(ValueError, match="years must"):
            impact.load_thailand_baseline(path, years=years)
    for years in ([np.nan], [np.inf], [True, 2019], [False]):
        with pytest.raises(ValueError, match="years must"):
            impact.load_thailand_baseline(path, years=years)


# ----------------------------------------------------------------------------
# scale_effect
# ----------------------------------------------------------------------------
def test_scale_effect_matches_hand_computed_numbers(baseline):
    out = impact.scale_effect(0.5, 20.0, baseline.loc[2019])
    assert out["absolute_usd_bn"] == pytest.approx(0.5 / 100 * 500.0)
    assert out["absolute_usd_bn"] == pytest.approx(2.5)
    assert out["absolute_thb_bn"] == pytest.approx(75.0)
    assert out["relative_usd_bn"] == pytest.approx(20.0 / 100 * 60.0)
    assert out["relative_usd_bn"] == pytest.approx(12.0)
    assert out["relative_thb_bn"] == pytest.approx(360.0)
    assert out["ratio_relative_to_absolute"] == pytest.approx(4.8)
    other = impact.scale_effect(0.5, 20.0, baseline.loc[2024])
    assert other["absolute_usd_bn"] == pytest.approx(3.0)
    assert other["absolute_thb_bn"] == pytest.approx(105.0)
    assert other["relative_usd_bn"] == pytest.approx(9.0)
    assert other["relative_thb_bn"] == pytest.approx(315.0)
    assert other["ratio_relative_to_absolute"] == pytest.approx(3.0)


def test_scale_effect_accepts_a_plain_mapping_and_negative_effects():
    row = {"gdp_usd_bn": 400.0, "receipts_usd_bn": 50.0, "fx_thb_per_usd": 32.0}
    out = impact.scale_effect(-0.25, -10.0, row)
    assert out["absolute_usd_bn"] == pytest.approx(-1.0)
    assert out["absolute_thb_bn"] == pytest.approx(-32.0)
    assert out["relative_usd_bn"] == pytest.approx(-5.0)
    assert out["relative_thb_bn"] == pytest.approx(-160.0)
    assert out["ratio_relative_to_absolute"] == pytest.approx(5.0)
    assert isinstance(out["absolute_usd_bn"], float)


def test_scale_effect_allows_either_effect_to_be_missing(baseline):
    row = baseline.loc[2019]
    only_abs = impact.scale_effect(0.5, None, row)
    assert only_abs["absolute_usd_bn"] == pytest.approx(2.5)
    assert only_abs["relative_usd_bn"] is None and only_abs["relative_thb_bn"] is None
    assert only_abs["ratio_relative_to_absolute"] is None
    only_rel = impact.scale_effect(None, 20.0, row)
    assert only_rel["relative_thb_bn"] == pytest.approx(360.0)
    assert only_rel["absolute_usd_bn"] is None and only_rel["absolute_thb_bn"] is None
    assert only_rel["ratio_relative_to_absolute"] is None
    neither = impact.scale_effect(None, None, row)
    assert set(neither) == set(only_abs)
    assert all(value is None for value in neither.values())


def test_scale_effect_works_elementwise_on_draws_and_guards_a_zero_absolute_effect(baseline):
    row = baseline.loc[2019]
    out = impact.scale_effect(np.array([0.0, 0.5, 1.0]), np.array([10.0, 20.0, 40.0]), row)
    assert np.allclose(out["absolute_usd_bn"], [0.0, 2.5, 5.0])
    assert np.allclose(out["relative_thb_bn"], [180.0, 360.0, 720.0])
    ratio = out["ratio_relative_to_absolute"]
    assert np.isnan(ratio[0])
    assert np.allclose(ratio[1:], [4.8, 4.8])
    zero = impact.scale_effect(0.0, 5.0, row)
    assert np.isnan(zero["ratio_relative_to_absolute"])


# ----------------------------------------------------------------------------
# impact_table
# ----------------------------------------------------------------------------
def test_impact_table_matches_hand_computed_quantiles_and_shares(baseline):
    draws_pp = np.linspace(0.0, 1.0, 101)
    draws_rel = np.linspace(0.0, 40.0, 101)
    out = impact.impact_table(draws_pp, draws_rel, baseline, route="ambient")
    assert list(out.columns) == [
        "route",
        "baseline_year",
        "scaling",
        "q5",
        "q25",
        "q50",
        "q75",
        "q95",
        "median_share_receipts_pct",
        "median_share_gdp_pct",
    ]
    assert len(out) == 4
    assert set(out["route"]) == {"ambient"}
    expected = {
        (2019, "absolute"): ([0.25, 1.25, 2.5, 3.75, 4.75], 2.5 / 60.0 * 100, 2.5 / 500.0 * 100),
        (2019, "relative"): ([1.2, 6.0, 12.0, 18.0, 22.8], 12.0 / 60.0 * 100, 12.0 / 500.0 * 100),
        (2024, "absolute"): ([0.3, 1.5, 3.0, 4.5, 5.7], 3.0 / 45.0 * 100, 3.0 / 600.0 * 100),
        (2024, "relative"): ([0.9, 4.5, 9.0, 13.5, 17.1], 9.0 / 45.0 * 100, 9.0 / 600.0 * 100),
    }
    for _, row in out.iterrows():
        quantiles, share_receipts, share_gdp = expected[(row["baseline_year"], row["scaling"])]
        assert [row[c] for c in ("q5", "q25", "q50", "q75", "q95")] == pytest.approx(quantiles)
        assert row["median_share_receipts_pct"] == pytest.approx(share_receipts)
        assert row["median_share_gdp_pct"] == pytest.approx(share_gdp)
    assert out["baseline_year"].tolist() == [2019, 2019, 2024, 2024]
    assert out["scaling"].tolist() == ["absolute", "relative", "absolute", "relative"]
    assert out.index.tolist() == [0, 1, 2, 3]


def test_impact_table_options(baseline):
    draws = np.linspace(-1.0, 3.0, 401)
    custom = impact.impact_table(
        draws, None, baseline, years=2019, route="x", probs=(0.025, 0.5, 0.975)
    )
    assert list(custom.columns[3:6]) == ["q2.5", "q50", "q97.5"]
    assert custom["scaling"].tolist() == ["absolute"]
    assert custom["baseline_year"].tolist() == [2019]
    row = custom.iloc[0]
    assert row["q2.5"] == pytest.approx(np.quantile(draws, 0.025) / 100 * 500.0)
    assert row["q97.5"] == pytest.approx(np.quantile(draws, 0.975) / 100 * 500.0)
    assert row["median_share_gdp_pct"] == pytest.approx(np.median(draws) / 100 * 100.0)
    rel_only = impact.impact_table(None, draws, baseline, years=[2024], probs=[0.1, 0.9])
    assert rel_only["scaling"].tolist() == ["relative"]
    assert list(rel_only.columns[3:5]) == ["q10", "q90"]
    assert rel_only.iloc[0]["q90"] == pytest.approx(np.quantile(draws, 0.9) / 100 * 45.0)
    # the median shares do not depend on the requested probabilities
    assert rel_only.iloc[0]["median_share_receipts_pct"] == pytest.approx(np.median(draws))
    with_year_column = baseline.reset_index()
    again = impact.impact_table(draws, None, with_year_column, years=2019, probs=(0.5,))
    assert again.iloc[0]["q50"] == pytest.approx(np.median(draws) / 100 * 500.0)


def test_impact_table_rejects_bad_arguments(baseline):
    draws = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError):
        impact.impact_table(None, None, baseline)
    with pytest.raises(ValueError, match="2030"):
        impact.impact_table(draws, None, baseline, years=(2019, 2030))
    with pytest.raises(ValueError):
        impact.impact_table(draws, None, baseline, probs=(0.5, 1.5))
    with pytest.raises(ValueError):
        impact.impact_table(draws, None, baseline, probs=(0.5, 0.5))
    with pytest.raises(ValueError):
        impact.impact_table(draws, None, baseline, probs=())
    with pytest.raises(ValueError):
        impact.impact_table(np.array([0.1, np.nan]), None, baseline)
    with pytest.raises(ValueError):
        impact.impact_table(draws.reshape(1, -1), None, baseline)
    for column in ("gdp_usd_bn", "receipts_usd_bn"):
        with pytest.raises(ValueError, match=column):
            impact.impact_table(draws, draws, baseline.drop(columns=column))


def test_impact_table_needs_only_gdp_and_receipts_in_the_baseline(baseline):
    draws = np.linspace(-1.0, 3.0, 101)
    full = impact.impact_table(draws, 40.0 * draws, baseline, route="x")
    minimal = baseline[["gdp_usd_bn", "receipts_usd_bn"]]
    again = impact.impact_table(draws, 40.0 * draws, minimal, route="x")
    pd.testing.assert_frame_equal(again, full)
    unused_missing = baseline.copy()
    unused_missing["fx_thb_per_usd"] = np.nan
    unused_missing["arrivals_million"] = -1.0
    pd.testing.assert_frame_equal(
        impact.impact_table(draws, 40.0 * draws, unused_missing, route="x"), full
    )


@pytest.mark.parametrize("column", ["gdp_usd_bn", "receipts_usd_bn"])
@pytest.mark.parametrize("bad", [np.nan, 0.0, -5.0, np.inf])
def test_impact_table_rejects_a_baseline_that_is_not_finite_and_positive(baseline, column, bad):
    draws = np.linspace(0.0, 1.0, 11)
    broken = baseline.copy()
    broken.loc[2024, column] = bad
    with pytest.raises(ValueError, match=rf"{column} in 2024 must be finite and positive"):
        impact.impact_table(draws, None, broken)
    with pytest.raises(ValueError, match=rf"{column} in 2024"):
        impact.impact_table(None, draws, broken, years=(2019, 2024))
    unused_year = impact.impact_table(draws, draws, broken, years=2019)
    assert np.all(np.isfinite(unused_year.select_dtypes("number").to_numpy()))


def test_impact_table_rejects_repeated_baseline_years(baseline):
    draws = np.linspace(0.0, 1.0, 11)
    repeated = pd.concat([baseline, baseline.loc[[2019]]])
    with pytest.raises(ValueError, match="2019") as info:
        impact.impact_table(draws, None, repeated)
    assert not isinstance(info.value, TypeError)
    with pytest.raises(ValueError, match="2019"):
        impact.impact_table(draws, None, repeated.reset_index())
    distinct = impact.impact_table(draws, None, baseline.reset_index())
    assert distinct["baseline_year"].tolist() == [2019, 2024]


def test_impact_table_reads_years_as_whole_numbers(baseline):
    draws = np.linspace(0.0, 1.0, 11)
    expected = impact.impact_table(draws, draws, baseline, years=(2019, 2024))
    for years in ((2019.0, 2024.0), np.array([2019, 2024]), [np.int64(2019), np.int64(2024)]):
        out = impact.impact_table(draws, draws, baseline, years=years)
        pd.testing.assert_frame_equal(out, expected)
        assert out["baseline_year"].dtype.kind == "i"
    single = impact.impact_table(draws, None, baseline, years=np.float64(2024.0))
    assert single["baseline_year"].tolist() == [2024]
    for years in ([], "2019", 2019.5, [2019, "2024"], True, np.True_, [np.nan], [False]):
        with pytest.raises(ValueError, match="years must"):
            impact.impact_table(draws, None, baseline, years=years)
    as_text = baseline.copy()
    as_text.index = pd.Index(["2019", "2024"], name="year")
    pd.testing.assert_frame_equal(impact.impact_table(draws, draws, as_text), expected)
    for labels in (["2019", "latest"], [2019.5, 2024.0]):
        broken = baseline.copy()
        broken.index = pd.Index(labels, name="year")
        with pytest.raises(ValueError, match="whole-number years"):
            impact.impact_table(draws, None, broken)


def test_impact_table_median_share_follows_the_median_of_skewed_draws(baseline):
    draws = np.array([1.0, 1.0, 2.0, 3.0, 13.0])
    assert np.mean(draws) == 4.0 and np.median(draws) == 2.0
    out = impact.impact_table(draws, 10.0 * draws, baseline)
    cell = out[(out.baseline_year == 2019) & (out.scaling == "absolute")].iloc[0]
    assert cell["q50"] == pytest.approx(2.0 / 100 * 500.0)
    assert cell["median_share_receipts_pct"] == pytest.approx(10.0 / 60.0 * 100)
    assert cell["median_share_gdp_pct"] == pytest.approx(10.0 / 500.0 * 100)
    cell = out[(out.baseline_year == 2024) & (out.scaling == "relative")].iloc[0]
    assert cell["q50"] == pytest.approx(20.0 / 100 * 45.0)
    assert cell["median_share_receipts_pct"] == pytest.approx(9.0 / 45.0 * 100)
    assert cell["median_share_gdp_pct"] == pytest.approx(9.0 / 600.0 * 100)
    for _, row in out.iterrows():
        year = row["baseline_year"]
        receipts, gdp = baseline.loc[year, "receipts_usd_bn"], baseline.loc[year, "gdp_usd_bn"]
        assert row["median_share_receipts_pct"] == pytest.approx(row["q50"] / receipts * 100)
        assert row["median_share_gdp_pct"] == pytest.approx(row["q50"] / gdp * 100)


def test_impact_table_takes_posterior_draws_from_the_bayesian_meta_analysis(baseline):
    rng = np.random.default_rng(7)
    se = rng.uniform(0.1, 0.4, 10)
    y = 0.3 + rng.normal(0.0, 0.2, 10) + rng.normal(0.0, se)
    post = meta.bayes_normal_hierarchical(y, se, n_draws=4000, seed=1)
    out = impact.impact_table(post.draws_pred, 40.0 * post.draws_pred, baseline, route="meta")
    quantile_columns = ["q5", "q25", "q50", "q75", "q95"]
    assert (out[quantile_columns].diff(axis=1).iloc[:, 1:] > 0).all().all()
    absolute_2019 = out[(out.baseline_year == 2019) & (out.scaling == "absolute")].iloc[0]
    assert absolute_2019["q50"] == pytest.approx(np.median(post.draws_pred) / 100 * 500.0)
    relative_2024 = out[(out.baseline_year == 2024) & (out.scaling == "relative")].iloc[0]
    assert relative_2024["q50"] == pytest.approx(np.median(40.0 * post.draws_pred) / 100 * 45.0)


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def test_minimum_detectable_effect_for_a_standard_normal_null():
    null = np.random.default_rng(0).standard_normal(400_000)
    out = impact.minimum_detectable_effect(null)
    assert set(out) == {"threshold", "mde", "mde_normal"}
    assert all(type(value) is float for value in out.values())
    assert out["threshold"] == pytest.approx(1.96, abs=0.02)
    assert out["mde"] == pytest.approx(2.80, abs=0.03)
    assert out["mde"] - out["threshold"] == pytest.approx(0.8416, abs=0.005)
    assert out["mde_normal"] == pytest.approx(2.80, abs=0.03)
    assert out["mde_normal"] == pytest.approx(out["mde"], abs=0.01)


@pytest.mark.parametrize("centre", [0.3, -0.3])
def test_minimum_detectable_effect_follows_the_definition(centre):
    null = np.random.default_rng(1).normal(centre, 2.0, 5000)
    out = impact.minimum_detectable_effect(null, alpha=0.1, power=0.9)
    threshold = np.quantile(np.abs(null), 0.9)
    upper, lower = np.quantile(null, 0.9), -np.quantile(null, 0.1)
    assert (upper > lower) == (centre > 0.0)
    assert out["threshold"] == pytest.approx(threshold, rel=1e-12)
    assert out["mde"] == pytest.approx(threshold + max(upper, lower), rel=1e-12)
    z_power = stats.norm.ppf(0.9)
    assert z_power == pytest.approx(1.2815515655446004, rel=1e-12)
    assert out["mde_normal"] == pytest.approx(threshold + z_power * np.std(null, ddof=1), rel=1e-12)
    scaled = impact.minimum_detectable_effect(2.5 * null, alpha=0.1, power=0.9)
    for key in ("threshold", "mde", "mde_normal"):
        assert scaled[key] == pytest.approx(2.5 * out[key], rel=1e-12)
    strict = impact.minimum_detectable_effect(null, alpha=0.01)
    loose = impact.minimum_detectable_effect(null, alpha=0.2)
    assert strict["threshold"] > loose["threshold"]
    high = impact.minimum_detectable_effect(null, power=0.95)
    low = impact.minimum_detectable_effect(null, power=0.6)
    assert high["mde"] > low["mde"] and high["mde_normal"] > low["mde_normal"]


def test_minimum_detectable_effect_is_unchanged_by_reflecting_the_null():
    null = np.random.default_rng(5).exponential(1.0, 4000) - 0.3
    out = impact.minimum_detectable_effect(null, alpha=0.1, power=0.85)
    mirrored = impact.minimum_detectable_effect(-null, alpha=0.1, power=0.85)
    assert mirrored["threshold"] == pytest.approx(out["threshold"], rel=1e-12)
    assert mirrored["mde"] == pytest.approx(out["mde"], rel=1e-12)
    assert mirrored["mde_normal"] == pytest.approx(out["mde_normal"], rel=1e-12)
    order = np.random.default_rng(6).permutation(null.size)
    shuffled = impact.minimum_detectable_effect(null[order], alpha=0.1, power=0.85)
    assert shuffled == pytest.approx(out, rel=1e-12)


def test_minimum_detectable_effect_for_a_symmetric_null_is_the_threshold_plus_its_power_quantile():
    half = np.random.default_rng(7).gamma(2.0, 1.0, 3000)
    null = np.concatenate([half, -half])
    out = impact.minimum_detectable_effect(null, alpha=0.1, power=0.75)
    assert out["threshold"] == pytest.approx(np.quantile(half, 0.9), rel=1e-3)
    assert out["mde"] == pytest.approx(out["threshold"] + np.quantile(null, 0.75), rel=1e-12)
    assert impact.minimum_detectable_effect(null, power=0.5)["mde"] == pytest.approx(
        impact.minimum_detectable_effect(null, power=0.5)["threshold"], rel=1e-12
    )
    shifted = impact.minimum_detectable_effect(null + 0.4, power=0.5)
    assert shifted["mde"] == pytest.approx(shifted["threshold"] + 0.4, rel=1e-9)


NULL_SHAPES = ("biased_normal", "laplace", "skewed", "student_t3", "uniform")


def draw_null(shape, rng, n):
    """Draws of a null distribution that is not a centred normal."""
    if shape == "student_t3":
        return rng.standard_t(3, n)
    if shape == "uniform":
        return rng.uniform(-1.0, 1.0, n)
    if shape == "biased_normal":
        return rng.normal(1.5, 1.0, n)
    if shape == "skewed":
        return rng.exponential(1.0, n) - 0.4
    if shape == "laplace":
        return rng.laplace(0.0, 1.0, n)
    raise ValueError(shape)


@pytest.mark.parametrize("shape", NULL_SHAPES)
@pytest.mark.parametrize("alpha, power", [(0.05, 0.8), (0.2, 0.9)])
def test_minimum_detectable_effect_reaches_the_power_on_the_empirical_null(shape, alpha, power):
    n = 100_000
    null = draw_null(shape, np.random.default_rng(11), n)
    cut = impact.minimum_detectable_effect(null, alpha=alpha, power=power)
    assert impact.detection_probability(null, cut["threshold"]) == pytest.approx(alpha, abs=2e-4)
    reached = [
        impact.detection_probability(sign * cut["mde"], cut["threshold"], null_draws=null)
        for sign in (1.0, -1.0)
    ]
    assert min(reached) >= power - 2.0 / n
    assert min(reached) <= power + 0.02
    below = [
        impact.detection_probability(sign * 0.9 * cut["mde"], cut["threshold"], null_draws=null)
        for sign in (1.0, -1.0)
    ]
    assert min(below) < power - 0.01


@pytest.mark.parametrize(
    "shape, direction", [("student_t3", "over"), ("uniform", "under"), ("biased_normal", "under")]
)
def test_the_normal_formula_misses_the_power_when_the_null_is_not_normal_and_centred(
    shape, direction
):
    n = 100_000
    null = draw_null(shape, np.random.default_rng(12), n)
    cut = impact.minimum_detectable_effect(null)
    reached = min(
        impact.detection_probability(sign * cut["mde_normal"], cut["threshold"], null_draws=null)
        for sign in (1.0, -1.0)
    )
    if direction == "over":
        assert reached > 0.85
        assert cut["mde_normal"] > cut["mde"]
    else:
        assert reached < 0.77
        assert cut["mde_normal"] < cut["mde"]


def test_minimum_detectable_effect_rejects_bad_arguments():
    null = np.random.default_rng(2).standard_normal(100)
    for kwargs in ({"alpha": 0.0}, {"alpha": 1.0}, {"power": 0.0}, {"power": 1.0}):
        with pytest.raises(ValueError):
            impact.minimum_detectable_effect(null, **kwargs)
    with pytest.raises(ValueError):
        impact.minimum_detectable_effect([0.3])
    with pytest.raises(ValueError):
        impact.minimum_detectable_effect([0.3, np.nan, 0.1])
    with pytest.raises(ValueError):
        impact.minimum_detectable_effect([0.3, np.inf, 0.1])
    with pytest.raises(ValueError):
        impact.minimum_detectable_effect(null.reshape(10, 10))
    two = impact.minimum_detectable_effect([-1.0, 2.0])
    reach = max(np.quantile([-1.0, 2.0], 0.8), -np.quantile([-1.0, 2.0], 0.2))
    assert two["threshold"] == pytest.approx(np.quantile([1.0, 2.0], 0.95))
    assert two["mde"] == pytest.approx(two["threshold"] + reach)


def test_detection_probability_is_a_share_between_zero_and_one():
    draws = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    assert impact.detection_probability(draws, 1.5) == pytest.approx(0.5)
    assert impact.detection_probability(draws, 1.0) == pytest.approx(3.0 / 6.0)
    assert impact.detection_probability(draws, 0.0) == pytest.approx(5.0 / 6.0)
    assert impact.detection_probability(draws, 5.0) == 0.0
    assert impact.detection_probability(draws, 100.0) == 0.0
    assert impact.detection_probability(draws, -1.0) == 1.0
    assert impact.detection_probability(-draws, 1.5) == impact.detection_probability(draws, 1.5)
    rng = np.random.default_rng(3)
    sample = rng.normal(0.5, 1.5, 2000)
    shares = [impact.detection_probability(sample, t) for t in np.linspace(0.0, 8.0, 17)]
    assert all(0.0 <= s <= 1.0 for s in shares)
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    with pytest.raises(ValueError):
        impact.detection_probability([], 1.0)
    with pytest.raises(ValueError):
        impact.detection_probability(draws, np.nan)
    with pytest.raises(ValueError):
        impact.detection_probability([1.0, np.nan], 1.0)
    with pytest.raises(ValueError):
        impact.detection_probability(draws.reshape(2, 3), 1.0)
    assert impact.detection_probability(draws, np.inf) == 0.0
    assert impact.detection_probability(draws, -np.inf) == 1.0


def test_detection_probability_accepts_a_single_effect():
    assert impact.detection_probability(2.0, 1.5) == 1.0
    assert impact.detection_probability(-2.0, 1.5) == 1.0
    assert impact.detection_probability(1.0, 1.5) == 0.0
    assert impact.detection_probability(2.0, 1.5, null_draws=[-1.0, 0.0, 1.0]) == pytest.approx(
        2.0 / 3.0
    )
    assert type(impact.detection_probability(np.float64(2.0), 1.5)) is float


def brute_force_detection(effects, threshold, null):
    effects = np.atleast_1d(np.asarray(effects, dtype=float))
    null = np.asarray(null, dtype=float)
    return float(np.mean(np.abs(effects[:, None] + null[None, :]) > threshold))


@pytest.mark.parametrize("seed", range(6))
def test_detection_probability_with_a_null_counts_every_pair_of_effect_and_null_draw(seed):
    rng = np.random.default_rng(seed)
    effects = rng.integers(-4, 5, 17).astype(float) / 2.0
    null = rng.integers(-3, 4, 23).astype(float) / 2.0
    for threshold in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 8.0):
        expected = brute_force_detection(effects, threshold, null)
        got = impact.detection_probability(effects, threshold, null_draws=null)
        assert got == pytest.approx(expected, abs=1e-12)
    continuous_effects, continuous_null = rng.normal(0.4, 1.0, 40), rng.standard_t(3, 60)
    for threshold in (0.3, 1.2, 2.5):
        expected = brute_force_detection(continuous_effects, threshold, continuous_null)
        got = impact.detection_probability(continuous_effects, threshold, continuous_null)
        assert got == pytest.approx(expected, abs=1e-12)


def test_detection_probability_with_a_null_has_the_expected_limits():
    null = np.random.default_rng(8).standard_normal(500)
    effects = np.random.default_rng(9).normal(1.0, 2.0, 300)
    assert impact.detection_probability(effects, 2.0, null_draws=np.zeros(5)) == pytest.approx(
        impact.detection_probability(effects, 2.0)
    )
    assert impact.detection_probability(0.0, 1.96, null_draws=null) == pytest.approx(
        impact.detection_probability(null, 1.96)
    )
    assert impact.detection_probability(effects, -0.5, null_draws=null) == 1.0
    assert impact.detection_probability(effects, 1e9, null_draws=null) == 0.0
    assert impact.detection_probability(-effects, 2.0, null_draws=-null) == pytest.approx(
        impact.detection_probability(effects, 2.0, null_draws=null)
    )
    shares = [impact.detection_probability(1.5, t, null_draws=null) for t in np.linspace(0, 6, 25)]
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    assert 0.0 <= min(shares) and max(shares) <= 1.0
    for bad in ([], [np.nan, 1.0], [[1.0, 2.0]], [np.inf]):
        with pytest.raises(ValueError, match="null_draws"):
            impact.detection_probability(effects, 1.0, null_draws=bad)


def test_a_null_effect_is_detected_at_the_size_and_the_mde_at_the_target_power():
    rng = np.random.default_rng(4)
    null = rng.standard_normal(200_000)
    cut = impact.minimum_detectable_effect(null, alpha=0.05, power=0.8)
    assert impact.detection_probability(null, cut["threshold"]) == pytest.approx(0.05, abs=0.002)
    effect = cut["mde"] + rng.standard_normal(200_000)
    assert impact.detection_probability(effect, cut["threshold"]) == pytest.approx(0.8, abs=0.01)
    exact = impact.detection_probability(cut["mde"], cut["threshold"], null_draws=null)
    assert 0.8 - 1e-5 <= exact <= 0.8 + 0.004




# ----------------------------------------------------------------------------
# Scale of the null
# ----------------------------------------------------------------------------
def skewed_null(seed, n=4000):
    """Placebo-like draws that are skewed and not centred at zero."""
    rng = np.random.default_rng(seed)
    return 0.3 * (rng.gamma(3.0, 1.0, n) - 3.0) + 0.05


def test_rescale_null_pp_with_equal_shares_leaves_the_draws_unchanged():
    null = skewed_null(0)
    result = impact.rescale_null_pp(null, 5.7, 5.7)
    assert isinstance(result, tuple) and len(result) == 2
    draws, factor = result
    assert factor == 1.0 and type(factor) is float
    assert np.array_equal(draws, null)
    assert draws is not null and not np.shares_memory(draws, null)
    draws[0] = 1e6
    assert null[0] != 1e6
    thresholds = impact.minimum_detectable_effect(null)
    assert impact.minimum_detectable_effect(impact.rescale_null_pp(null, 3.0, 3.0)[0]) == thresholds


def test_rescale_null_pp_multiplies_the_draws_by_the_ratio_of_the_shares():
    null = skewed_null(1)
    draws, factor = impact.rescale_null_pp(null, 5.7, 7.81)
    assert factor == pytest.approx(7.81 / 5.7, rel=1e-15)
    assert np.allclose(draws, null * (7.81 / 5.7), rtol=1e-14, atol=0.0)
    assert draws.shape == null.shape and draws.dtype == np.float64
    smaller, shrink = impact.rescale_null_pp(null, 7.81, 5.7)
    assert shrink == pytest.approx(5.7 / 7.81, rel=1e-15)
    assert np.abs(smaller).max() < np.abs(null).max()
    back, inverse = impact.rescale_null_pp(draws, 7.81, 5.7)
    assert inverse * factor == pytest.approx(1.0, rel=1e-14)
    assert np.allclose(back, null, rtol=1e-12, atol=1e-15)
    from_list, same = impact.rescale_null_pp([0.5, -1.0, 2.0], 2.0, 3.0)
    assert same == 1.5 and from_list.tolist() == [0.75, -1.5, 3.0]
    exact, ratio = impact.rescale_null_pp(np.array([1.0, -2.0]), np.float32(2.0), np.int64(4))
    assert ratio == 2.0 and exact.tolist() == [2.0, -4.0]


@pytest.mark.parametrize("target", [3.0, 5.7, 7.81, 12.0])
def test_scaling_a_null_by_k_scales_the_minimum_detectable_effect_by_k(target):
    null = skewed_null(2)
    base = impact.minimum_detectable_effect(null)
    scaled_null, factor = impact.rescale_null_pp(null, 5.7, target)
    assert factor == pytest.approx(target / 5.7, rel=1e-15)
    got = impact.minimum_detectable_effect(scaled_null)
    for key in ("threshold", "mde", "mde_normal"):
        assert got[key] == pytest.approx(factor * base[key], rel=1e-12)
    effect = np.random.default_rng(3).normal(0.9, 0.25, 300)
    before = impact.detection_probability(effect, base["threshold"], null_draws=null)
    after = impact.detection_probability(factor * effect, got["threshold"], null_draws=scaled_null)
    assert after == pytest.approx(before, abs=1e-4)


def test_a_target_with_a_larger_receipts_share_needs_a_larger_effect_in_percentage_points():
    # a placebo spread that grows with the share makes the same effect harder to detect where the
    # share is larger
    rng = np.random.default_rng(4)
    null = rng.normal(0.0, 0.3, 6000)
    effect = rng.normal(1.0, 0.1, 500)
    mde, probability = [], []
    for target in (4.0, 5.7, 7.81, 11.0):
        scaled, _ = impact.rescale_null_pp(null, 5.7, target)
        cut = impact.minimum_detectable_effect(scaled)
        mde.append(cut["mde"])
        probability.append(
            impact.detection_probability(effect, cut["threshold"], null_draws=scaled)
        )
    assert all(a < b for a, b in zip(mde, mde[1:]))
    assert all(a > b for a, b in zip(probability, probability[1:]))
    assert probability[0] > 0.99 and probability[-1] < 0.5
    assert mde[2] / mde[1] == pytest.approx(7.81 / 5.7, rel=1e-12)


@pytest.mark.parametrize("name", ["source_receipts_share", "target_receipts_share"])
def test_rescale_null_pp_rejects_shares_that_are_not_positive_and_finite(name):
    null = skewed_null(5, n=50)
    invalid = (0.0, -1.0, np.nan, np.inf, -np.inf, True, np.True_, "5.7", None, [5.7])
    for bad in (*invalid, np.array(5.0)):
        kwargs = {"source_receipts_share": 5.7, "target_receipts_share": 7.81, name: bad}
        with pytest.raises(ValueError, match=name):
            impact.rescale_null_pp(null, **kwargs)
    ok = {"source_receipts_share": 5.7, "target_receipts_share": 7.81, name: 1e-9}
    assert np.isfinite(impact.rescale_null_pp(null, **ok)[1])


def test_rescale_null_pp_rejects_draws_that_are_not_a_finite_vector():
    for bad in ([], [0.1, np.nan], [0.1, np.inf], [[0.1, 0.2]], 0.3, None, ["a", "b"]):
        with pytest.raises((ValueError, TypeError)):
            impact.rescale_null_pp(bad, 5.7, 7.81)
    with pytest.raises(ValueError, match="null_pp"):
        impact.rescale_null_pp([], 5.7, 7.81)
    with pytest.raises(ValueError, match="null_pp"):
        impact.rescale_null_pp([0.1, np.nan], 5.7, 7.81)


def test_source_receipts_share_is_the_usage_weighted_mean():
    shares = [2.0, 4.0, 10.0]
    assert impact.source_receipts_share(shares, [1.0, 1.0, 2.0]) == pytest.approx(6.5, rel=1e-15)
    assert impact.source_receipts_share(shares, [1.0, 1.0, 1.0]) == pytest.approx(16.0 / 3.0)
    assert impact.source_receipts_share(shares, [0.0, 3.0, 0.0]) == 4.0
    assert impact.source_receipts_share(shares, [0.0, 1e-300, 0.0]) == pytest.approx(4.0)
    assert type(impact.source_receipts_share(shares, [1, 1, 2])) is float
    base = impact.source_receipts_share(shares, [0.2, 0.3, 0.5])
    assert base == pytest.approx(0.4 + 1.2 + 5.0, rel=1e-15)
    for factor in (1e-8, 7.0, 1e6):
        scaled = impact.source_receipts_share(shares, factor * np.array([0.2, 0.3, 0.5]))
        assert scaled == pytest.approx(base, rel=1e-12)
    rng = np.random.default_rng(6)
    values = rng.uniform(0.5, 15.0, 40)
    weights = rng.dirichlet(np.ones(40))
    order = rng.permutation(40)
    mean = impact.source_receipts_share(values, weights)
    assert mean == pytest.approx(float(np.average(values, weights=weights)), rel=1e-12)
    permuted = impact.source_receipts_share(values[order], weights[order])
    assert mean == pytest.approx(permuted, rel=1e-12)
    assert values.min() <= mean <= values.max()
    mass = np.where(weights > np.quantile(weights, 0.5), weights, 0.0)
    used = mass > 0
    assert values[used].min() <= impact.source_receipts_share(values, mass) <= values[used].max()


def test_source_receipts_share_weights_the_sources_that_carry_the_mass():
    # most of the mass sits on sources with a small share; a few with a large share carry little
    shares = np.array([4.0, 5.0, 6.0, 20.0, 25.0])
    usage = np.array([0.35, 0.35, 0.25, 0.03, 0.02])
    weighted = impact.source_receipts_share(shares, usage)
    by_hand = 4.0 * 0.35 + 5.0 * 0.35 + 6.0 * 0.25 + 20.0 * 0.03 + 25.0 * 0.02
    assert weighted == pytest.approx(by_hand)
    assert weighted < float(shares.mean())
    draws, factor = impact.rescale_null_pp(skewed_null(7, n=100), weighted, 7.81)
    assert factor == pytest.approx(7.81 / weighted, rel=1e-15)
    assert factor > 1.0


def test_source_receipts_share_accepts_series_and_checks_their_alignment():
    index = pd.Index(["a", "b", "c"], name="case")
    shares = pd.Series([2.0, 4.0, 10.0], index=index)
    usage = pd.Series([1.0, 1.0, 2.0], index=index)
    assert impact.source_receipts_share(shares, usage) == pytest.approx(6.5)
    assert impact.source_receipts_share(shares, usage.to_numpy()) == pytest.approx(6.5)
    assert impact.source_receipts_share(shares.to_numpy(), usage) == pytest.approx(6.5)
    shuffled = usage.loc[["c", "a", "b"]]
    with pytest.raises(ValueError, match="same index"):
        impact.source_receipts_share(shares, shuffled)
    other = pd.Series(usage.to_numpy(), index=["a", "b", "z"])
    with pytest.raises(ValueError, match="same index"):
        impact.source_receipts_share(shares, other)
    aligned = impact.source_receipts_share(shares, shuffled.reindex(shares.index))
    assert aligned == pytest.approx(6.5)


def test_source_receipts_share_rejects_invalid_inputs():
    shares = [2.0, 4.0, 10.0]
    usage = [1.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="same length"):
        impact.source_receipts_share(shares, usage[:2])
    for bad in ([], [[2.0, 4.0, 10.0]], [2.0, np.nan, 10.0], [2.0, np.inf, 10.0]):
        with pytest.raises(ValueError, match="shares"):
            impact.source_receipts_share(bad, usage)
    for bad in ([2.0, 0.0, 10.0], [2.0, -1.0, 10.0]):
        with pytest.raises(ValueError, match="shares must be positive"):
            impact.source_receipts_share(bad, usage)
    for bad in ([1.0, np.nan, 2.0], [1.0, np.inf, 2.0], [[1.0, 1.0, 2.0]], []):
        with pytest.raises(ValueError, match="usage"):
            impact.source_receipts_share(shares, bad)
    with pytest.raises(ValueError, match="usage must be non-negative"):
        impact.source_receipts_share(shares, [1.0, -0.1, 2.0])
    with pytest.raises(ValueError, match="positive sum"):
        impact.source_receipts_share(shares, [0.0, 0.0, 0.0])
    assert impact.source_receipts_share(shares, [0.0, 0.0, 1e-12]) == 10.0


def test_the_module_header_states_which_null_belongs_to_which_scaling():
    header = " ".join(impact.__doc__.split())
    assert "null of the relative scaling" in header and "scale free" in header
    assert "null of the absolute scaling" in header
    assert "rescale_null_pp" in header and "source_receipts_share" in header
    assert "rescale_null_pp" in impact.__all__ and "source_receipts_share" in impact.__all__
    assert "relative scaling" in impact.rescale_null_pp.__doc__
    assert EM_DASH not in impact.__doc__


# ----------------------------------------------------------------------------
# decide_route
# ----------------------------------------------------------------------------
SUPPORTED = {"supported": True, "ess": 6.2, "reasons": []}
UNSUPPORTED = {
    "supported": False,
    "ess": 1.97,
    "reasons": [
        "The cost share of the target is 14.8 robust standard deviations outside the range "
        "of the sources."
    ],
}
CRITERIA = ("importance_usable", "loco_rmse", "enough_cases", "target_supported")


def loco_table(rmse_weighted, rmse_equal, rmse_uniform, extra=None):
    values = {"ot_weighted": rmse_weighted, "equal": rmse_equal, "ot_uniform": rmse_uniform}
    values.update(extra or {})
    return pd.DataFrame({"rmse": list(values.values())}, index=list(values))


def decide(diagnostics, summary, n_cases, **kwargs):
    """Apply the route rule to a target that the support check accepts, unless told otherwise."""
    kwargs.setdefault("support", SUPPORTED)
    return impact.decide_route(diagnostics, summary, n_cases, **kwargs)


@pytest.mark.parametrize(
    "usable, beats_equal, beats_uniform, enough, supported",
    list(itertools.product([True, False], repeat=5)),
)
def test_decide_route_truth_table(usable, beats_equal, beats_uniform, enough, supported):
    rmse_equal = 1.0 if beats_equal else 0.8
    rmse_uniform = 1.0 if beats_uniform else 0.8
    summary = loco_table(0.9, rmse_equal, rmse_uniform)
    n_cases = 25 if enough else 19
    decision = impact.decide_route(
        {"usable": usable}, summary, n_cases, support=SUPPORTED if supported else UNSUPPORTED
    )
    rmse_ok = beats_equal and beats_uniform
    assert decision.passed == {
        "importance_usable": usable,
        "loco_rmse": rmse_ok,
        "enough_cases": enough,
        "target_supported": supported,
    }
    assert list(decision.passed) == list(CRITERIA)
    assert all(type(v) is bool for v in decision.passed.values())
    expected = "ot_importance" if (usable and rmse_ok and enough and supported) else "ambient"
    assert decision.primary == expected
    assert expected in decision.details
    assert ("All four criteria are met" in decision.details) == (expected == "ot_importance")
    assert ("At least one criterion is not met" in decision.details) == (expected == "ambient")


def test_decide_route_signature_and_default_rules():
    parameters = inspect.signature(impact.decide_route).parameters
    assert list(parameters) == [
        "importance_diagnostics",
        "loco_summary",
        "n_cases",
        "rules",
        "support",
        "ambient_rmse",
    ]
    assert all(parameters[name].default is None for name in ("rules", "support", "ambient_rmse"))
    assert list(inspect.signature(impact.RouteDecision).parameters) == [
        "primary",
        "passed",
        "details",
        "info",
    ]


def test_decide_route_boundaries_follow_at_most_and_at_least():
    summary = loco_table(0.95, 1.0, 1.0)
    on_the_line = decide({"usable": True}, summary, 20)
    assert on_the_line.primary == "ot_importance"
    assert on_the_line.passed == dict.fromkeys(CRITERIA, True)
    above = decide({"usable": True}, loco_table(0.9501, 1.0, 1.0), 20)
    assert above.primary == "ambient" and above.passed["loco_rmse"] is False
    above_uniform = decide({"usable": True}, loco_table(0.95, 1.0, 0.99), 20)
    assert above_uniform.primary == "ambient"
    few = decide({"usable": True}, summary, 19)
    assert few.primary == "ambient" and few.passed["enough_cases"] is False
    worse = decide({"usable": True}, loco_table(1.2, 1.0, 1.0), 40)
    assert worse.primary == "ambient"


def test_decide_route_ratio_limits_tolerate_rounding_noise_and_nothing_larger():
    for noise in (-1e-9, -1e-12, 0.0, 1e-14, 1e-12, 5e-10):
        weighted = 0.95 * (1.0 + noise)
        for summary in (loco_table(weighted, 1.0, 5.0), loco_table(weighted, 5.0, 1.0)):
            assert decide({"usable": True}, summary, 20).passed["loco_rmse"] is True, noise
    for noise in (1e-6, 1e-4):
        weighted = 0.95 * (1.0 + noise)
        for summary in (loco_table(weighted, 1.0, 5.0), loco_table(weighted, 5.0, 1.0)):
            assert decide({"usable": True}, summary, 20).passed["loco_rmse"] is False, noise
    # a ratio computed on another math kernel differs in the last digits only
    base = loco_table(0.3515, 0.37, 0.4)
    nudged = base * (1.0 + 4e-16)
    assert decide({"usable": True}, base, 20).passed == decide({"usable": True}, nudged, 20).passed


def test_decide_route_ignores_other_methods_and_uses_the_named_ones():
    summary = loco_table(0.5, 1.0, 1.0, extra={"ot_other": 0.1, "ambient": 0.2})
    assert decide({"usable": True}, summary, 20).primary == "ot_importance"
    shuffled = summary.iloc[::-1]
    assert decide({"usable": True}, shuffled, 20).primary == "ot_importance"
    swapped = loco_table(1.0, 0.5, 0.5, extra={"ot_other": 0.1})
    assert decide({"usable": True}, swapped, 20).primary == "ambient"
    assert decide({"usable": True}, summary, 20).info == {}


def test_decide_route_default_thresholds_and_overrides():
    assert dict(impact.DEFAULT_ROUTE_RULES) == {
        "rmse_ratio_equal": 0.95,
        "rmse_ratio_uniform": 0.95,
        "min_cases": 20,
    }
    with pytest.raises(TypeError):
        impact.DEFAULT_ROUTE_RULES["min_cases"] = 3
    summary = loco_table(0.97, 1.0, 1.0)
    assert decide({"usable": True}, summary, 25).primary == "ambient"
    relaxed = decide(
        {"usable": True}, summary, 25, rules={"rmse_ratio_equal": 1.0, "rmse_ratio_uniform": 1.0}
    )
    assert relaxed.primary == "ot_importance"
    only_equal = decide({"usable": True}, summary, 25, rules={"rmse_ratio_equal": 1.0})
    assert only_equal.primary == "ambient"
    good = loco_table(0.5, 1.0, 1.0)
    small = decide({"usable": True}, good, 6, rules={"min_cases": 5})
    assert small.primary == "ot_importance"
    strict = decide({"usable": True}, good, 25, rules={"min_cases": 30})
    assert strict.primary == "ambient" and strict.passed["enough_cases"] is False
    assert decide({"usable": True}, good, 30, rules={"min_cases": 30}).primary == "ot_importance"
    tight = decide(
        {"usable": True}, loco_table(0.9, 1.0, 1.0), 25, rules={"rmse_ratio_uniform": 0.8}
    )
    assert tight.primary == "ambient"
    # an empty override leaves the defaults unchanged
    unchanged = decide({"usable": True}, loco_table(0.9, 1.0, 1.0), 20, rules={})
    assert unchanged.primary == "ot_importance"
    assert decide({"usable": True}, loco_table(0.9, 1.0, 1.0), 19, rules={}).primary == "ambient"
    with pytest.raises(ValueError, match="min_case"):
        decide({"usable": True}, summary, 25, rules={"min_case": 5})
    with pytest.raises(ValueError):
        decide({"usable": True}, summary, 25, rules={"min_cases": float("nan")})
    with pytest.raises(ValueError):
        decide({"usable": True}, summary, 25, rules={"rmse_ratio_equal": -0.1})
    # the support requirement has no threshold that a rule could relax
    with pytest.raises(ValueError, match="target_supported"):
        decide({"usable": True}, summary, 25, rules={"target_supported": 0.0})


@pytest.mark.parametrize("name", ["rmse_ratio_equal", "rmse_ratio_uniform"])
def test_decide_route_accepts_rmse_ratio_thresholds_in_the_interval_zero_to_one_and_a_half(name):
    summary = loco_table(0.5, 1.0, 1.0)
    for value in (1e-9, 0.3, 0.95, 1.0, 1.5, np.float64(1.2), np.float32(0.5)):
        decision = decide({"usable": True}, summary, 25, rules={name: value})
        assert decision.passed["loco_rmse"] is (0.5 <= float(value))
    for value in (0.0, -0.5, 1.5000001, 2.0, np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match=name):
            decide({"usable": True}, summary, 25, rules={name: value})
    for value in (True, "0.9", None, [0.9], 1 + 0j):
        with pytest.raises(ValueError, match=name):
            decide({"usable": True}, summary, 25, rules={name: value})


def test_decide_route_accepts_a_non_negative_finite_minimum_number_of_cases():
    summary = loco_table(0.5, 1.0, 1.0)
    zero = decide({"usable": True}, summary, 0, rules={"min_cases": 0})
    assert zero.primary == "ot_importance" and zero.passed["enough_cases"] is True
    fractional = decide({"usable": True}, summary, 2, rules={"min_cases": 2.5})
    assert fractional.passed["enough_cases"] is False
    assert "at least 2.5 are required" in fractional.details
    for value in (-1, -0.001, np.inf, np.nan, True, "10", None):
        with pytest.raises(ValueError, match="min_cases"):
            decide({"usable": True}, summary, 25, rules={"min_cases": value})
    for bad_rules in ([("min_cases", 5)], 5, "min_cases", (10,)):
        with pytest.raises(ValueError, match="mapping"):
            decide({"usable": True}, summary, 25, rules=bad_rules)


def test_decide_route_reads_the_usable_flag_strictly():
    summary = loco_table(0.5, 1.0, 1.0)
    for flag in (True, np.True_):
        decision = decide({"usable": flag}, summary, 25)
        assert decision.primary == "ot_importance"
        assert decision.passed["importance_usable"] is True
    for flag in (False, np.False_):
        decision = decide({"usable": flag}, summary, 25)
        assert decision.primary == "ambient"
        assert decision.passed["importance_usable"] is False
    for not_boolean in (1, 0, 1.0, "yes", "True", None, [True], np.int64(1), np.float64(1.0)):
        with pytest.raises(ValueError, match="usable"):
            decide({"usable": not_boolean}, summary, 25)
    extra_keys = {"usable": True, "reason": "ok", "n_folds": 5}
    assert decide(extra_keys, summary, 25).primary == "ot_importance"
    series = pd.Series({"usable": True, "reason": "ok"})
    assert decide(series, summary, 25).primary == "ot_importance"


def test_decide_route_rejects_diagnostics_without_a_usable_entry():
    summary = loco_table(0.5, 1.0, 1.0)
    for diagnostics in (None, {}, {"other": True}, {"Usable": True}, [True], True, "usable", 1):
        with pytest.raises(ValueError, match="usable"):
            decide(diagnostics, summary, 25)
    with pytest.raises(ValueError, match="usable"):
        decide(pd.DataFrame({"usable": [True]}), summary, 25)


def test_decide_route_rejects_a_summary_without_valid_rmse_values():
    diagnostics = {"usable": True}
    complete = loco_table(0.5, 1.0, 1.0)
    for method in ("ot_weighted", "equal", "ot_uniform"):
        with pytest.raises(ValueError, match=method):
            decide(diagnostics, complete.drop(index=method), 20)
    broken = {
        "nan": np.nan,
        "inf": np.inf,
        "negative infinity": -np.inf,
        "zero": 0.0,
        "negative": -0.2,
    }
    for position, method in enumerate(("ot_weighted", "equal", "ot_uniform")):
        for label, value in broken.items():
            values = [0.5, 1.0, 1.0]
            values[position] = value
            with pytest.raises(ValueError, match=method):
                decide(diagnostics, loco_table(*values), 20)
    with pytest.raises(ValueError):
        decide(diagnostics, loco_table(0.0, 0.0, 0.0), 20)
    with pytest.raises(ValueError):
        decide(diagnostics, loco_table(np.inf, np.inf, np.inf), 20)
    for text in ("0.5", None, True):
        textual = loco_table(0.5, 1.0, 1.0).astype({"rmse": object})
        textual.loc["ot_weighted", "rmse"] = text
        with pytest.raises(ValueError, match="ot_weighted"):
            decide(diagnostics, textual, 20)
    float32 = complete.astype({"rmse": np.float32})
    assert decide(diagnostics, float32, 20).primary == "ot_importance"


def test_decide_route_rejects_a_summary_that_is_not_indexed_by_method_name():
    diagnostics = {"usable": True}
    complete = loco_table(0.5, 1.0, 1.0)
    in_a_column = complete.reset_index(names="method")
    with pytest.raises(ValueError, match="ot_weighted"):
        decide(diagnostics, in_a_column, 20)
    for index in (["OT_weighted", "equal", "ot_uniform"], ["ot_weighted ", "equal", "ot_uniform"]):
        mislabelled = complete.copy()
        mislabelled.index = index
        with pytest.raises(ValueError, match="ot_weighted"):
            decide(diagnostics, mislabelled, 20)
    with pytest.raises(ValueError, match="rmse"):
        decide(diagnostics, complete.rename(columns={"rmse": "mse"}), 20)
    duplicated = pd.DataFrame(
        {"rmse": [0.5, 0.6, 1.0, 1.0]}, index=["ot_weighted", "ot_weighted", "equal", "ot_uniform"]
    )
    with pytest.raises(ValueError, match="one row per method"):
        decide(diagnostics, duplicated, 25)
    for not_a_frame in (None, complete["rmse"], complete.to_dict(), complete.to_numpy(), "rmse"):
        with pytest.raises(ValueError, match="DataFrame"):
            decide(diagnostics, not_a_frame, 20)


def test_decide_route_requires_a_non_negative_integer_number_of_cases():
    summary = loco_table(0.5, 1.0, 1.0)
    diagnostics = {"usable": True}
    not_counts = (25.0, 25.5, np.float64(25.0), True, np.True_, None, "25", [25], -1, -10, np.nan)
    for n_cases in not_counts:
        with pytest.raises(ValueError, match="n_cases"):
            decide(diagnostics, summary, n_cases)
    for n_cases in (20, np.int64(20), np.int32(25), np.uint8(30), 10**6):
        decision = decide(diagnostics, summary, n_cases)
        assert decision.primary == "ot_importance"
    nothing = decide(diagnostics, summary, 0)
    assert nothing.primary == "ambient" and nothing.passed["enough_cases"] is False
    assert "There are 0 cases" in nothing.details
    assert "There are 25 cases" in decide(diagnostics, summary, np.int64(25)).details


def test_decide_route_details_report_each_criterion_in_plain_text():
    summary = loco_table(0.4, 0.5, 0.8)
    decision = decide({"usable": True}, summary, 25)
    text = decision.details
    assert isinstance(text, str) and text.endswith(".")
    assert "0.4" in text and "0.5" in text and "0.8" in text
    assert "0.800 times the RMSE of equal" in text
    assert "0.500 times the RMSE of ot_uniform" in text
    assert "25 cases" in text and "at least 20" in text
    assert "primary route is ot_importance" in text
    failing = impact.decide_route(
        {"usable": False}, loco_table(1.0, 1.0, 1.0), 4, support=UNSUPPORTED
    )
    assert "primary route is ambient" in failing.details
    assert "not met" in failing.details
    for decision in (decision, failing):
        assert EM_DASH not in decision.details


def test_decide_route_names_the_economy_that_is_left_out():
    decision = decide({"usable": True}, loco_table(0.4, 0.5, 0.8), 25)
    assert "The leave-one-economy-out RMSE of ot_weighted is 0.4" in decision.details
    assert "leave-one-economy-out" in impact.decide_route.__doc__
    for module in (impact, meta):
        for name, member in vars(module).items():
            if inspect.isfunction(member) or inspect.isclass(member):
                assert "leave-one-case-out" not in (member.__doc__ or ""), name
    assert "leave-one-case-out" not in decision.details
    assert "leave-one-group-out" in impact._loco_rmse.__doc__
    assert "leave-one-group-out" not in decision.details


def test_decide_route_documents_that_cases_are_episodes_and_economies_are_independent():
    doc = " ".join(impact.decide_route.__doc__.split())
    assert "counts episodes" in doc
    assert "economies are the independent units" in doc
    assert "leaves out whole economies" in doc
    assert "target_supported" in doc and "four criteria" in doc


def test_route_decision_is_immutable_and_info_defaults_to_an_empty_dict():
    decision = decide({"usable": True}, loco_table(0.5, 1.0, 1.0), 25)
    with pytest.raises(AttributeError):
        decision.primary = "ambient"
    with pytest.raises(AttributeError):
        decision.info = {}
    plain = impact.RouteDecision(primary="ambient", passed={}, details="text")
    other = impact.RouteDecision("ambient", {}, "text")
    assert plain.info == {} and other.info == {}
    assert plain.info is not other.info
    assert impact.RouteDecision("ambient", {}, "text", {"best_ambient": "A1"}).info == {
        "best_ambient": "A1"
    }


# ----------------------------------------------------------------------------
# decide_route: the support criterion
# ----------------------------------------------------------------------------
def test_decide_route_requires_the_support_result():
    summary = loco_table(0.5, 1.0, 1.0)
    message = "support is required: pass the result of transport.target_support"
    with pytest.raises(ValueError) as info:
        impact.decide_route({"usable": True}, summary, 25)
    assert str(info.value) == message
    with pytest.raises(ValueError, match="support is required"):
        impact.decide_route({"usable": True}, summary, 25, support=None)
    with pytest.raises(ValueError, match="support is required"):
        impact.decide_route(
            {"usable": True}, summary, 25, rules={"min_cases": 5}, ambient_rmse={"A1": 0.4}
        )
    # a three-criterion call with a decision that is already ambient is refused as well
    with pytest.raises(ValueError, match="support is required"):
        impact.decide_route({"usable": False}, loco_table(2.0, 1.0, 1.0), 3)


def test_decide_route_reads_the_support_result_strictly():
    summary = loco_table(0.5, 1.0, 1.0)
    for flag in (True, np.True_):
        decision = decide({"usable": True}, summary, 25, support={"supported": flag})
        assert decision.primary == "ot_importance"
        assert decision.passed["target_supported"] is True
    for flag in (False, np.False_):
        decision = decide({"usable": True}, summary, 25, support={"supported": flag})
        assert decision.primary == "ambient"
        assert decision.passed["target_supported"] is False
    for not_boolean in (1, 0, 1.0, "yes", "True", None, [True], np.int64(1), np.float64(1.0)):
        with pytest.raises(ValueError, match="supported"):
            decide({"usable": True}, summary, 25, support={"supported": not_boolean})
    for not_a_mapping in ([True], True, False, "supported", 1, pd.Series({"supported": True})):
        with pytest.raises(ValueError, match="mapping"):
            decide({"usable": True}, summary, 25, support=not_a_mapping)
    for lacking in ({}, {"ess": 3.0}, {"Supported": True}, {"ok": True, "reasons": []}):
        with pytest.raises(ValueError, match="supported"):
            decide({"usable": True}, summary, 25, support=lacking)
    wider = {"supported": True, "ess": 4.0, "reasons": [], "max_source_share": 0.3, "n_sources": 12}
    assert decide({"usable": True}, summary, 25, support=wider).primary == "ot_importance"
    frozen = MappingProxyType({"supported": True, "ess": 4.0})
    assert decide({"usable": True}, summary, 25, support=frozen).primary == "ot_importance"


def test_decide_route_support_is_a_fourth_criterion_that_the_others_cannot_replace():
    # the scenario of a target far outside the cost range of the cases: the validation and the
    # case count are fine, the support check is not
    summary = loco_table(0.36, 0.45, 0.42, extra={"nn1": 0.40, "nn3": 0.38})
    outside = decide({"usable": True}, summary, 60, support=UNSUPPORTED)
    assert outside.passed == {
        "importance_usable": True,
        "loco_rmse": True,
        "enough_cases": True,
        "target_supported": False,
    }
    assert outside.primary == "ambient"
    inside = decide({"usable": True}, summary, 60, support=SUPPORTED)
    assert inside.primary == "ot_importance"
    assert outside.passed["loco_rmse"] == inside.passed["loco_rmse"]
    # a supported target does not rescue a failing validation
    assert decide({"usable": True}, loco_table(0.6, 0.5, 0.5), 60).primary == "ambient"


def test_decide_route_states_the_support_result_with_ess_and_reasons():
    summary = loco_table(0.5, 1.0, 1.0)
    reasons = ["The first reason.", "The second reason has no full stop"]
    text = decide(
        {"usable": True}, summary, 25, support={"supported": False, "ess": 1.97, "reasons": reasons}
    ).details
    assert "target is not supported by the sources (effective number of sources 1.97)." in text
    assert "Reasons reported: The first reason. The second reason has no full stop." in text
    assert "target-support criterion is not met" in text
    good = decide(
        {"usable": True}, summary, 25, support={"supported": True, "ess": 5.31234, "reasons": []}
    ).details
    assert "target is supported by the sources (effective number of sources 5.31)." in good
    assert "Reasons reported" not in good
    assert "target-support criterion is met" in good
    bare = decide({"usable": True}, summary, 25, support={"supported": True}).details
    assert "effective number" not in bare and "Reasons reported" not in bare
    assert "target is supported by the sources." in bare
    single = decide(
        {"usable": True}, summary, 25, support={"supported": False, "reasons": "Only one reason"}
    ).details
    assert "Reasons reported: Only one reason." in single
    as_tuple = decide(
        {"usable": True}, summary, 25, support={"supported": False, "reasons": ("A.", "B.")}
    ).details
    assert "Reasons reported: A. B." in as_tuple
    no_reasons = decide(
        {"usable": True}, summary, 25, support={"supported": False, "reasons": [" ", ""]}
    ).details
    assert "Reasons reported" not in no_reasons
    undefined = decide(
        {"usable": True},
        summary,
        25,
        support={"supported": False, "ess": float("nan"), "reasons": ["The plan carries no mass."]},
    )
    assert "effective number of sources nan" in undefined.details
    assert undefined.primary == "ambient"
    nothing = decide({"usable": True}, summary, 25, support={"supported": True, "reasons": None})
    assert nothing.primary == "ot_importance" and "Reasons reported" not in nothing.details
    assert EM_DASH not in text and EM_DASH not in good


def test_decide_route_rejects_malformed_ess_and_reasons():
    summary = loco_table(0.5, 1.0, 1.0)
    for ess in ("3", True, np.True_, [3.0], np.array([1.0]), {"a": 1}):
        with pytest.raises(ValueError, match="ess"):
            decide({"usable": True}, summary, 25, support={"supported": True, "ess": ess})
    for ess in (3, np.float32(2.5), np.int64(4)):
        decide({"usable": True}, summary, 25, support={"supported": True, "ess": ess})
    for reasons in (3, 2.5, True, {"a": "b"}, ["fine", 3], [None], (1, 2), b"bytes"):
        with pytest.raises(ValueError, match="reasons"):
            decide({"usable": True}, summary, 25, support={"supported": False, "reasons": reasons})


# ----------------------------------------------------------------------------
# decide_route: comparisons that are reported and not used
# ----------------------------------------------------------------------------
def test_decide_route_reports_the_ratios_to_the_best_ambient_and_neighbour_predictors():
    # a run in which the ambient meta-regression and the nearest-neighbour predictor were better
    # than the weighted transport, and the rule still chose the transport
    summary = loco_table(
        0.365, 0.45, 0.42, extra={"nn1": 0.31, "nn3": 0.276, "kernel": 0.30}
    )
    ambient = {"A1": 0.45, "A2": 0.40, "A3": 0.323}
    decision = decide({"usable": True}, summary, 30, ambient_rmse=ambient)
    assert decision.primary == "ot_importance"
    assert set(decision.info) == {
        "rmse_ratio_to_best_ambient",
        "best_ambient",
        "rmse_ratio_to_best_neighbour",
        "best_neighbour",
    }
    assert decision.info["best_ambient"] == "A3"
    assert decision.info["rmse_ratio_to_best_ambient"] == pytest.approx(0.365 / 0.323, rel=1e-12)
    assert decision.info["best_neighbour"] == "nn3"
    assert decision.info["rmse_ratio_to_best_neighbour"] == pytest.approx(0.365 / 0.276, rel=1e-12)
    assert type(decision.info["rmse_ratio_to_best_ambient"]) is float
    assert type(decision.info["rmse_ratio_to_best_neighbour"]) is float
    text = decision.details
    assert "1.130 times the smallest RMSE among the ambient estimators" in text
    assert "which is A3 with 0.323" in text
    assert "1.322 times the smallest RMSE among the nearest-neighbour predictors" in text
    assert "nn3 with 0.276" in text
    assert text.count("reported and is not used by the rule") == 2
    assert text.index("not used by the rule") < text.index("All four criteria are met")
    assert EM_DASH not in text


def test_decide_route_ratios_are_reported_and_never_used_by_the_rule():
    summary = loco_table(0.365, 0.45, 0.42, extra={"nn1": 0.31, "nn3": 0.276})
    plain = decide({"usable": True}, summary, 30)
    assert plain.info["best_neighbour"] == "nn3" and "best_ambient" not in plain.info
    for ambient in ({"A1": 0.01}, {"A1": 100.0}, {"A1": 0.365}, {"x": 0.2, "y": 0.3}):
        decision = decide({"usable": True}, summary, 30, ambient_rmse=ambient)
        assert decision.primary == plain.primary == "ot_importance"
        assert decision.passed == plain.passed
    failing = decide({"usable": True}, loco_table(0.6, 0.5, 0.5, extra={"nn3": 0.9}), 30)
    assert failing.primary == "ambient"
    assert failing.info["rmse_ratio_to_best_neighbour"] == pytest.approx(0.6 / 0.9)
    better = decide({"usable": True}, loco_table(0.6, 0.5, 0.5), 30, ambient_rmse={"A1": 0.9})
    assert better.info["rmse_ratio_to_best_ambient"] == pytest.approx(0.6 / 0.9)
    assert better.primary == "ambient"


def test_decide_route_info_holds_only_what_its_inputs_allow():
    base = loco_table(0.5, 1.0, 1.0)
    assert decide({"usable": True}, base, 25).info == {}
    assert "reported and is not used" not in decide({"usable": True}, base, 25).details
    only_ambient = decide({"usable": True}, base, 25, ambient_rmse={"A1": 0.4})
    assert set(only_ambient.info) == {"rmse_ratio_to_best_ambient", "best_ambient"}
    assert only_ambient.info["best_ambient"] == "A1"
    assert only_ambient.info["rmse_ratio_to_best_ambient"] == pytest.approx(1.25)
    for present in ("nn1", "nn3"):
        summary = loco_table(0.5, 1.0, 1.0, extra={present: 0.25})
        one = decide({"usable": True}, summary, 25)
        assert one.info["best_neighbour"] == present
        assert one.info["rmse_ratio_to_best_neighbour"] == pytest.approx(2.0)
        assert set(one.info) == {"rmse_ratio_to_best_neighbour", "best_neighbour"}
    both = loco_table(0.5, 1.0, 1.0, extra={"nn1": 0.25, "nn3": 0.5})
    assert decide({"usable": True}, both, 25).info["best_neighbour"] == "nn1"
    swapped = loco_table(0.5, 1.0, 1.0, extra={"nn1": 0.5, "nn3": 0.25})
    assert decide({"usable": True}, swapped, 25).info["best_neighbour"] == "nn3"


def test_decide_route_info_resolves_ties_by_order_and_accepts_a_series():
    base = loco_table(0.5, 1.0, 1.0)
    tied = {"A2": 0.4, "A1": 0.4, "A3": 0.4 * (1.0 + 1e-13)}
    assert decide({"usable": True}, base, 25, ambient_rmse=tied).info["best_ambient"] == "A2"
    reordered = {"A3": 0.4 * (1.0 + 1e-13), "A1": 0.4, "A2": 0.4}
    assert decide({"usable": True}, base, 25, ambient_rmse=reordered).info["best_ambient"] == "A3"
    distinct = {"A2": 0.4 * (1.0 + 1e-6), "A1": 0.4}
    assert decide({"usable": True}, base, 25, ambient_rmse=distinct).info["best_ambient"] == "A1"
    as_series = pd.Series({"A1": 0.45, "A2": 0.40, "A3": 0.323}, name="rmse")
    from_series = decide({"usable": True}, base, 25, ambient_rmse=as_series)
    from_dict = decide({"usable": True}, base, 25, ambient_rmse=dict(as_series))
    assert from_series.info == from_dict.info and from_series.details == from_dict.details
    scalar_types = {"A1": np.float32(0.5), "A2": np.int64(1)}
    mixed = decide({"usable": True}, base, 25, ambient_rmse=scalar_types)
    assert mixed.info["best_ambient"] == "A1"


def test_decide_route_rejects_malformed_ambient_rmse_and_neighbour_rows():
    base = loco_table(0.5, 1.0, 1.0)
    for not_a_mapping in ([0.3, 0.4], 0.3, "A1", np.array([0.3]), [("A1", 0.3)]):
        with pytest.raises(ValueError, match="ambient_rmse"):
            decide({"usable": True}, base, 25, ambient_rmse=not_a_mapping)
    with pytest.raises(ValueError, match="at least one estimator"):
        decide({"usable": True}, base, 25, ambient_rmse={})
    with pytest.raises(ValueError, match="strings"):
        decide({"usable": True}, base, 25, ambient_rmse={1: 0.3})
    for bad in (0.0, -0.1, np.nan, np.inf, True, "0.3", None, [0.3]):
        with pytest.raises(ValueError, match="rmse of A2"):
            decide({"usable": True}, base, 25, ambient_rmse={"A1": 0.3, "A2": bad})
    duplicated = pd.Series([0.3, 0.4], index=["A1", "A1"])
    with pytest.raises(ValueError, match="more than one entry"):
        decide({"usable": True}, base, 25, ambient_rmse=duplicated)
    for name in ("nn1", "nn3"):
        for bad in (np.nan, 0.0, -0.2, np.inf):
            with pytest.raises(ValueError, match=f"rmse of {name}"):
                decide({"usable": True}, loco_table(0.5, 1.0, 1.0, extra={name: bad}), 25)
    other = loco_table(0.5, 1.0, 1.0, extra={"kernel": np.nan})
    assert decide({"usable": True}, other, 25).primary == "ot_importance"


# ----------------------------------------------------------------------------
# route_comparison
# ----------------------------------------------------------------------------
def test_route_comparison_stacks_the_tables_and_flags_the_primary_route(baseline):
    pp = np.linspace(0.0, 1.0, 101)
    tables = {
        "ot_importance": impact.impact_table(pp, 40.0 * pp, baseline, route="ignored"),
        "ambient": impact.impact_table(0.5 * pp, None, baseline, route="also ignored"),
    }
    before = {name: table.copy() for name, table in tables.items()}
    decision = decide({"usable": True}, loco_table(0.5, 1.0, 1.0), 25)
    out = impact.route_comparison(tables, decision)
    assert decision.primary == "ot_importance"
    assert len(out) == len(tables["ot_importance"]) + len(tables["ambient"]) == 6
    assert out["route"].tolist() == ["ot_importance"] * 4 + ["ambient"] * 2
    assert out["primary"].dtype == bool
    assert out["primary"].tolist() == [True] * 4 + [False] * 2
    assert list(out.columns)[0] == "route" and list(out.columns)[-1] == "primary"
    assert out.index.tolist() == list(range(6))
    ambient_rows = out[out["route"] == "ambient"]
    assert ambient_rows["q50"].tolist() == pytest.approx([1.25, 1.5])
    for name, table in tables.items():
        pd.testing.assert_frame_equal(table, before[name])
    swapped = impact.route_comparison(
        tables, decide({"usable": False}, loco_table(0.5, 1.0, 1.0), 25)
    )
    assert swapped["primary"].tolist() == [False] * 4 + [True] * 2
    unsupported = impact.route_comparison(
        tables, decide({"usable": True}, loco_table(0.5, 1.0, 1.0), 25, support=UNSUPPORTED)
    )
    assert unsupported["primary"].tolist() == [False] * 4 + [True] * 2


def test_route_comparison_supports_additional_routes_and_checks_the_primary_exists(baseline):
    pp = np.linspace(0.0, 1.0, 101)
    table = impact.impact_table(pp, None, baseline)
    ambient_decision = decide({"usable": False}, loco_table(0.5, 1.0, 1.0), 25)
    every_route = {"meta": table, "ambient": table, "ot_importance": table}
    out = impact.route_comparison(every_route, ambient_decision)
    assert out.groupby("route")["primary"].all().to_dict() == {
        "ambient": True,
        "meta": False,
        "ot_importance": False,
    }
    primary_decision = decide({"usable": True}, loco_table(0.5, 1.0, 1.0), 25)
    with pytest.raises(ValueError, match="ot_importance"):
        impact.route_comparison({"ambient": table}, primary_decision)
    with pytest.raises(ValueError):
        impact.route_comparison({}, ambient_decision)
    only_primary = impact.route_comparison({"ambient": table}, ambient_decision)
    assert only_primary["primary"].all()


# ----------------------------------------------------------------------------
# select_ambient
# ----------------------------------------------------------------------------
def notebook_rule(rmse, baseline="A1", ratio=0.95):
    """Selection exactly as the selection cell of notebook 05 writes it."""
    names = list(rmse)
    ratios = {name: rmse[name] / rmse[baseline] for name in names}
    qualifying = [name for name in names if name != baseline and ratios[name] <= ratio]
    if not qualifying:
        return baseline, qualifying
    return min(qualifying, key=lambda name: rmse[name]), qualifying


def test_select_ambient_signature_and_defaults():
    parameters = inspect.signature(impact.select_ambient).parameters
    assert list(parameters) == ["rmse", "baseline", "ratio"]
    assert parameters["baseline"].default == "A1"
    assert parameters["ratio"].default == 0.95


@pytest.mark.parametrize("seed", range(40))
def test_select_ambient_reproduces_the_notebook_rule_on_random_errors(seed):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 1.0)
    rmse = {
        "A1": base,
        "A2": base * rng.uniform(0.85, 1.05),
        "A3": base * rng.uniform(0.85, 1.05),
    }
    for ratio in (0.9, 0.95, 0.97):
        result = impact.select_ambient(rmse, ratio=ratio)
        selected, qualifying = notebook_rule(rmse, ratio=ratio)
        assert result["selected"] == selected
        assert result["qualifying"] == qualifying


def test_select_ambient_breaks_exact_ties_by_the_order_of_the_mapping():
    tied = {"A1": 1.0, "A2": 0.9, "A3": 0.9}
    result = impact.select_ambient(tied)
    assert result["selected"] == "A2" and result["qualifying"] == ["A2", "A3"]
    assert result["runner_up"] == "A3" and result["margin"] == 0.0
    assert result["close"] is True
    reordered = {"A1": 1.0, "A3": 0.9, "A2": 0.9}
    assert impact.select_ambient(reordered)["selected"] == "A3"


def test_select_ambient_limit_is_inclusive_and_tolerates_rounding_noise():
    assert impact.select_ambient({"A1": 1.0, "A2": 0.95, "A3": 0.99})["selected"] == "A2"
    assert impact.select_ambient({"A1": 1.0, "A2": 0.9501, "A3": 0.99})["selected"] == "A1"
    noisy = {"A1": 0.3515, "A2": 0.95 * 0.3515 * (1.0 + 4e-16), "A3": 0.4}
    assert impact.select_ambient(noisy)["selected"] == "A2"
    just_outside = {"A1": 0.3515, "A2": 0.95 * 0.3515 * (1.0 + 1e-6), "A3": 0.4}
    assert impact.select_ambient(just_outside)["selected"] == "A1"


def test_select_ambient_ratio_moves_the_qualification_limit():
    rmse = {"A1": 1.0, "A2": 0.92, "A3": 0.99}
    assert impact.select_ambient(rmse)["selected"] == "A2"
    assert impact.select_ambient(rmse, ratio=0.9)["selected"] == "A1"
    assert impact.select_ambient(rmse, ratio=1.0)["qualifying"] == ["A2", "A3"]
    assert impact.select_ambient({"A1": 1.0, "A2": 1.0}, ratio=1.0)["selected"] == "A2"


def test_select_ambient_falls_back_to_the_baseline_when_nothing_qualifies():
    result = impact.select_ambient({"A1": 1.0, "A2": 0.97, "A3": 1.2})
    assert result["selected"] == "A1" and result["qualifying"] == []
    assert result["runner_up"] == "A2"
    assert result["margin"] == pytest.approx(-0.03, rel=1e-12)
    assert result["close"] is True
    far = impact.select_ambient({"A1": 1.0, "A2": 1.3, "A3": 1.2})
    assert far["selected"] == "A1" and far["qualifying"] == []
    assert far["runner_up"] == "A3" and far["margin"] == pytest.approx(0.2, rel=1e-12)
    assert far["close"] is False


def test_select_ambient_reports_the_margin_the_runner_up_and_the_ratios():
    result = impact.select_ambient({"A1": 1.0, "A2": 0.9, "A3": 0.96})
    assert result["selected"] == "A2" and result["qualifying"] == ["A2"]
    assert result["runner_up"] == "A3"
    assert result["margin"] == pytest.approx(0.06 / 0.9, rel=1e-12)
    assert result["ratios"] == pytest.approx({"A1": 1.0, "A2": 0.9, "A3": 0.96})
    assert result["close"] is True
    assert isinstance(result["margin"], float) and isinstance(result["close"], bool)


def test_select_ambient_is_close_when_the_runner_up_is_within_five_percent():
    by_margin = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.52})
    assert by_margin["selected"] == "A2" and by_margin["close"] is True
    assert by_margin["margin"] == pytest.approx(0.04, rel=1e-12)
    apart = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.8})
    assert apart["selected"] == "A2" and apart["close"] is False


def test_select_ambient_is_close_when_an_estimator_sits_near_the_limit():
    near_limit = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.94})
    assert near_limit["selected"] == "A2" and near_limit["close"] is True
    edge = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.95 * 1.05})
    assert edge["close"] is True
    outside = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.95 * 1.06})
    assert outside["close"] is False


def test_select_ambient_with_only_the_baseline():
    assert impact.select_ambient({"A1": 0.4}) == {
        "selected": "A1",
        "qualifying": [],
        "ratios": {"A1": 1.0},
        "margin": 0.0,
        "runner_up": None,
        "close": False,
    }


def test_select_ambient_accepts_another_baseline_and_a_series():
    rmse = pd.Series({"A2": 0.5, "A1": 0.6, "A3": 0.7})
    result = impact.select_ambient(rmse, baseline="A2")
    assert result["selected"] == "A2" and result["qualifying"] == []
    assert result["runner_up"] == "A1" and result["margin"] == pytest.approx(0.2, rel=1e-12)
    assert impact.select_ambient(pd.Series({"A1": 1.0, "A2": 0.9}))["selected"] == "A2"


def test_select_ambient_does_not_change_its_input():
    rmse = pd.Series({"A1": 1.0, "A2": 0.9, "A3": 0.96})
    before = rmse.copy()
    impact.select_ambient(rmse)
    pd.testing.assert_series_equal(rmse, before)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"A1": np.nan, "A2": 0.9},
        {"A1": 0.0, "A2": 0.9},
        {"A1": 1.0, "A2": -0.9},
        {"A1": 1.0, "A2": np.inf},
        {"A1": 1.0, 2: 0.9},
        {"A1": True, "A2": 0.9},
        {"A1": 1.0, "A2": "0.9"},
    ],
)
def test_select_ambient_rejects_invalid_rmse_values(bad):
    with pytest.raises(ValueError):
        impact.select_ambient(bad)


def test_select_ambient_rejects_a_missing_baseline_a_bad_ratio_and_a_non_mapping():
    with pytest.raises(ValueError, match="baseline"):
        impact.select_ambient({"A2": 0.9, "A3": 0.8})
    with pytest.raises(ValueError, match="baseline"):
        impact.select_ambient({"A1": 1.0, "A2": 0.9}, baseline="A4")
    with pytest.raises(ValueError, match="baseline"):
        impact.select_ambient({"A1": 1.0}, baseline=1)
    for bad_ratio in (0.0, -0.95, 1.2, np.nan, np.inf, True, "0.95", None):
        with pytest.raises(ValueError, match="ratio"):
            impact.select_ambient({"A1": 1.0, "A2": 0.9}, ratio=bad_ratio)
    for not_mapping in ([1.0, 0.9], 0.9, None, np.array([1.0, 0.9])):
        with pytest.raises(ValueError, match="mapping"):
            impact.select_ambient(not_mapping)


def test_select_ambient_documents_the_sign_of_the_margin_and_the_tolerance():
    doc = " ".join(impact.select_ambient.__doc__.split())
    assert "negative when the baseline is selected" in doc
    assert "relative tolerance of 1e-9" in doc
    assert "select_ambient" in impact.__all__
    assert EM_DASH not in impact.select_ambient.__doc__


# ----------------------------------------------------------------------------
# Thailand on the basis of the panel
# ----------------------------------------------------------------------------
def panel_baseline(years, receipts):
    """Baseline frame in the layout of load_thailand_baseline, with the given receipts."""
    years = list(years)
    receipts = np.asarray(receipts, dtype=float)
    gdp = np.linspace(400.0, 550.0, len(years))
    return pd.DataFrame(
        {
            "gdp_usd_bn": gdp,
            "receipts_usd_bn": receipts,
            "receipts_pct_gdp": 100.0 * receipts / gdp,
            "arrivals_million": np.linspace(30.0, 40.0, len(years)),
            "fx_thb_per_usd": np.linspace(33.0, 36.0, len(years)),
        },
        index=pd.Index(years, name="year"),
    )


def test_to_panel_basis_applies_a_constant_ratio_to_the_years_without_panel_values():
    years = list(range(2014, 2025))
    base = panel_baseline(years, np.linspace(20.0, 30.0, len(years)))
    panel = (1.08 * base["receipts_usd_bn"]).loc[2014:2019]
    out = impact.to_panel_basis(base, panel)
    on_panel = out.loc[2014:2019]
    assert (on_panel["receipts_basis"] == "panel").all()
    assert on_panel["basis_ratio"].isna().all()
    np.testing.assert_allclose(on_panel["receipts_usd_bn"], panel.to_numpy(), rtol=1e-14, atol=0.0)
    off_panel = out.loc[2020:2024]
    assert (off_panel["receipts_basis"] == "ratio").all()
    np.testing.assert_allclose(off_panel["basis_ratio"], 1.08, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        off_panel["receipts_usd_bn"],
        1.08 * base.loc[2020:2024, "receipts_usd_bn"],
        rtol=1e-14,
        atol=0.0,
    )
    np.testing.assert_array_equal(out["receipts_usd_bn_baseline"], base["receipts_usd_bn"])


def test_to_panel_basis_uses_the_mean_of_the_yearly_ratios_and_the_panel_where_it_has_a_value():
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    base = panel_baseline(years, [10.0, 10.0, 10.0, 10.0, 40.0, 40.0])
    panel = pd.Series([10.0, 10.0, 10.0, 10.0, 60.0], index=years[:5])
    out = impact.to_panel_basis(base, panel)
    yearly = np.array([1.0, 1.0, 1.0, 1.0, 1.5])
    assert out.loc[2020, "basis_ratio"] == pytest.approx(yearly.mean(), rel=1e-15)
    assert out.loc[2020, "receipts_usd_bn"] == pytest.approx(40.0 * 1.1, rel=1e-15)
    assert out.loc[2019, "receipts_usd_bn"] == 60.0 and out.loc[2019, "receipts_basis"] == "panel"
    assert np.isnan(out.loc[2019, "basis_ratio"])
    assert out.loc[2015, "receipts_usd_bn"] == 10.0


def test_to_panel_basis_uses_only_the_ratio_years_that_have_both_values():
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    base = panel_baseline(years, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    panel = pd.Series({2015: 11.0, 2016: 12.0, 2019: 13.0})
    out = impact.to_panel_basis(base, panel, ratio_years=[2015, 2016, 2019, 2030])
    assert out.loc[2020, "basis_ratio"] == pytest.approx((1.1 + 1.2 + 1.3) / 3.0, rel=1e-15)
    only_2019 = impact.to_panel_basis(base, panel, ratio_years=[2019, 2017])
    assert only_2019.loc[2020, "basis_ratio"] == pytest.approx(1.3, rel=1e-15)
    default = impact.to_panel_basis(base, panel)
    assert default.loc[2020, "basis_ratio"] == pytest.approx((1.1 + 1.2 + 1.3) / 3.0, rel=1e-15)


def test_to_panel_basis_keeps_the_index_gdp_and_the_other_columns_and_recomputes_the_share():
    years = list(range(2015, 2021))
    base = panel_baseline(years, np.linspace(20.0, 30.0, len(years)))
    snapshot = base.copy()
    out = impact.to_panel_basis(base, base["receipts_usd_bn"] * 1.2)
    pd.testing.assert_frame_equal(base, snapshot)
    assert list(out.index) == years and out.index.name == "year"
    for column in ("gdp_usd_bn", "arrivals_million", "fx_thb_per_usd"):
        pd.testing.assert_series_equal(out[column], base[column])
    np.testing.assert_allclose(
        out["receipts_pct_gdp"],
        100.0 * out["receipts_usd_bn"] / out["gdp_usd_bn"],
        rtol=1e-14,
        atol=0.0,
    )
    assert list(out.columns) == [
        *base.columns,
        "receipts_usd_bn_baseline",
        "receipts_basis",
        "basis_ratio",
    ]


def test_to_panel_basis_gives_a_missing_share_for_a_missing_gdp():
    base = panel_baseline([2015, 2016, 2017], [10.0, 10.0, np.nan])
    base.loc[2016, "gdp_usd_bn"] = np.nan
    out = impact.to_panel_basis(base, pd.Series({2015: 12.0}))
    assert np.isnan(out.loc[2016, "receipts_pct_gdp"])
    assert np.isfinite(out.loc[2015, "receipts_pct_gdp"])
    assert np.isnan(out.loc[2017, "receipts_usd_bn"])


def test_to_panel_basis_raises_when_no_ratio_year_has_both_values():
    base = panel_baseline(range(2020, 2025), np.linspace(20.0, 30.0, 5))
    with pytest.raises(ValueError, match="ratio"):
        impact.to_panel_basis(base, pd.Series({2015: 11.0, 2016: 12.0}))
    early = panel_baseline(range(2010, 2016), np.linspace(20.0, 30.0, 6))
    with pytest.raises(ValueError, match="ratio"):
        impact.to_panel_basis(early, pd.Series({2000: 11.0, 2001: 12.0}))
    with pytest.raises(ValueError, match="ratio"):
        impact.to_panel_basis(early, pd.Series({2015: 11.0}), ratio_years=[2014, 2030])


def test_to_panel_basis_accepts_a_mapping_and_rejects_invalid_inputs():
    base = panel_baseline([2015, 2016, 2020], [10.0, 10.0, 10.0])
    as_map = impact.to_panel_basis(base, {2015: 12.0, 2016: 11.0, 2020: 13.0})
    assert as_map.loc[2020, "receipts_basis"] == "panel"
    assert as_map.loc[2015, "receipts_usd_bn"] == 12.0
    with pytest.raises(ValueError, match="Series or a mapping"):
        impact.to_panel_basis(base, [12.0, 11.0])
    with pytest.raises(ValueError, match="positive"):
        impact.to_panel_basis(base, pd.Series({2015: -1.0}))
    with pytest.raises(ValueError, match="positive"):
        impact.to_panel_basis(base, pd.Series({2015: np.inf}))
    with pytest.raises(ValueError, match="whole-number"):
        impact.to_panel_basis(base, pd.Series({2015.5: 11.0}))
    with pytest.raises(ValueError, match="more than one value"):
        impact.to_panel_basis(base, pd.Series([11.0, 12.0], index=[2015, 2015]))
    with pytest.raises(ValueError, match="whole numbers"):
        impact.to_panel_basis(base, pd.Series({2015: 11.0}), ratio_years=[2015.5])
    with pytest.raises(ValueError, match="receipts_usd_bn"):
        impact.to_panel_basis(base.drop(columns="receipts_usd_bn"), pd.Series({2015: 11.0}))
    with pytest.raises(ValueError, match="receipts_basis"):
        impact.to_panel_basis(base.assign(receipts_basis="x"), pd.Series({2015: 11.0}))


def test_to_panel_basis_is_documented_and_exported():
    assert "to_panel_basis" in impact.__all__
    doc = " ".join(impact.to_panel_basis.__doc__.split())
    assert "mean of the yearly ratios" in doc and "receipts_basis" in doc
    assert EM_DASH not in impact.to_panel_basis.__doc__
