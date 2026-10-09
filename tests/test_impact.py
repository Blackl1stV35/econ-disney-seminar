"""Tests for dtt.impact: baseline loader, effect scaling, detection and route decision.

All inputs are hand-made or SIMULATED.
"""
from __future__ import annotations

import inspect
import itertools
import sys
from pathlib import Path

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
# decide_route
# ----------------------------------------------------------------------------
def loco_table(rmse_weighted, rmse_equal, rmse_uniform, extra=None):
    values = {"ot_weighted": rmse_weighted, "equal": rmse_equal, "ot_uniform": rmse_uniform}
    values.update(extra or {})
    return pd.DataFrame({"rmse": list(values.values())}, index=list(values))


@pytest.mark.parametrize(
    "usable, beats_equal, beats_uniform, enough",
    list(itertools.product([True, False], repeat=4)),
)
def test_decide_route_truth_table(usable, beats_equal, beats_uniform, enough):
    rmse_equal = 1.0 if beats_equal else 0.8
    rmse_uniform = 1.0 if beats_uniform else 0.8
    summary = loco_table(0.9, rmse_equal, rmse_uniform)
    n_cases = 12 if enough else 9
    decision = impact.decide_route({"usable": usable}, summary, n_cases)
    rmse_ok = beats_equal and beats_uniform
    assert decision.passed == {
        "importance_usable": usable,
        "loco_rmse": rmse_ok,
        "enough_cases": enough,
    }
    assert all(type(v) is bool for v in decision.passed.values())
    expected = "ot_importance" if (usable and rmse_ok and enough) else "ambient"
    assert decision.primary == expected
    assert expected in decision.details


def test_decide_route_boundaries_follow_at_most_and_at_least():
    summary = loco_table(0.95, 1.0, 1.0)
    on_the_line = impact.decide_route({"usable": True}, summary, 10)
    assert on_the_line.primary == "ot_importance"
    assert on_the_line.passed == {
        "importance_usable": True,
        "loco_rmse": True,
        "enough_cases": True,
    }
    above = impact.decide_route({"usable": True}, loco_table(0.9501, 1.0, 1.0), 10)
    assert above.primary == "ambient" and above.passed["loco_rmse"] is False
    above_uniform = impact.decide_route({"usable": True}, loco_table(0.95, 1.0, 0.99), 10)
    assert above_uniform.primary == "ambient"
    few = impact.decide_route({"usable": True}, summary, 9)
    assert few.primary == "ambient" and few.passed["enough_cases"] is False
    worse = impact.decide_route({"usable": True}, loco_table(1.2, 1.0, 1.0), 40)
    assert worse.primary == "ambient"


def test_decide_route_ignores_other_methods_and_uses_the_named_ones():
    summary = loco_table(0.5, 1.0, 1.0, extra={"ot_other": 0.1, "ambient": 0.2})
    assert impact.decide_route({"usable": True}, summary, 20).primary == "ot_importance"
    shuffled = summary.iloc[::-1]
    assert impact.decide_route({"usable": True}, shuffled, 20).primary == "ot_importance"
    swapped = loco_table(1.0, 0.5, 0.5, extra={"ot_other": 0.1})
    assert impact.decide_route({"usable": True}, swapped, 20).primary == "ambient"


def test_decide_route_default_thresholds_and_overrides():
    assert dict(impact.DEFAULT_ROUTE_RULES) == {
        "rmse_ratio_equal": 0.95,
        "rmse_ratio_uniform": 0.95,
        "min_cases": 10,
    }
    with pytest.raises(TypeError):
        impact.DEFAULT_ROUTE_RULES["min_cases"] = 3
    summary = loco_table(0.97, 1.0, 1.0)
    assert impact.decide_route({"usable": True}, summary, 12).primary == "ambient"
    relaxed = impact.decide_route(
        {"usable": True}, summary, 12, rules={"rmse_ratio_equal": 1.0, "rmse_ratio_uniform": 1.0}
    )
    assert relaxed.primary == "ot_importance"
    only_equal = impact.decide_route({"usable": True}, summary, 12, rules={"rmse_ratio_equal": 1.0})
    assert only_equal.primary == "ambient"
    good = loco_table(0.5, 1.0, 1.0)
    small = impact.decide_route({"usable": True}, good, 6, rules={"min_cases": 5})
    assert small.primary == "ot_importance"
    strict = impact.decide_route({"usable": True}, good, 12, rules={"min_cases": 15})
    assert strict.primary == "ambient"
    tight = impact.decide_route(
        {"usable": True}, loco_table(0.9, 1.0, 1.0), 12, rules={"rmse_ratio_uniform": 0.8}
    )
    assert tight.primary == "ambient"
    # an empty override leaves the defaults unchanged
    unchanged = impact.decide_route({"usable": True}, loco_table(0.9, 1.0, 1.0), 10, rules={})
    assert unchanged.primary == "ot_importance"
    with pytest.raises(ValueError, match="min_case"):
        impact.decide_route({"usable": True}, summary, 12, rules={"min_case": 5})
    with pytest.raises(ValueError):
        impact.decide_route({"usable": True}, summary, 12, rules={"min_cases": float("nan")})
    with pytest.raises(ValueError):
        impact.decide_route({"usable": True}, summary, 12, rules={"rmse_ratio_equal": -0.1})


@pytest.mark.parametrize("name", ["rmse_ratio_equal", "rmse_ratio_uniform"])
def test_decide_route_accepts_rmse_ratio_thresholds_in_the_interval_zero_to_one_and_a_half(name):
    summary = loco_table(0.5, 1.0, 1.0)
    for value in (1e-9, 0.3, 0.95, 1.0, 1.5, np.float64(1.2), np.float32(0.5)):
        decision = impact.decide_route({"usable": True}, summary, 12, rules={name: value})
        assert decision.passed["loco_rmse"] is (0.5 <= float(value))
    for value in (0.0, -0.5, 1.5000001, 2.0, np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match=name):
            impact.decide_route({"usable": True}, summary, 12, rules={name: value})
    for value in (True, "0.9", None, [0.9], 1 + 0j):
        with pytest.raises(ValueError, match=name):
            impact.decide_route({"usable": True}, summary, 12, rules={name: value})


def test_decide_route_accepts_a_non_negative_finite_minimum_number_of_cases():
    summary = loco_table(0.5, 1.0, 1.0)
    zero = impact.decide_route({"usable": True}, summary, 0, rules={"min_cases": 0})
    assert zero.primary == "ot_importance" and zero.passed["enough_cases"] is True
    fractional = impact.decide_route({"usable": True}, summary, 2, rules={"min_cases": 2.5})
    assert fractional.passed["enough_cases"] is False
    assert "at least 2.5 are required" in fractional.details
    for value in (-1, -0.001, np.inf, np.nan, True, "10", None):
        with pytest.raises(ValueError, match="min_cases"):
            impact.decide_route({"usable": True}, summary, 12, rules={"min_cases": value})
    for bad_rules in ([("min_cases", 5)], 5, "min_cases", (10,)):
        with pytest.raises(ValueError, match="mapping"):
            impact.decide_route({"usable": True}, summary, 12, rules=bad_rules)


def test_decide_route_reads_the_usable_flag_strictly():
    summary = loco_table(0.5, 1.0, 1.0)
    for flag in (True, np.True_):
        decision = impact.decide_route({"usable": flag}, summary, 12)
        assert decision.primary == "ot_importance"
        assert decision.passed["importance_usable"] is True
    for flag in (False, np.False_):
        decision = impact.decide_route({"usable": flag}, summary, 12)
        assert decision.primary == "ambient"
        assert decision.passed["importance_usable"] is False
    for not_boolean in (1, 0, 1.0, "yes", "True", None, [True], np.int64(1), np.float64(1.0)):
        with pytest.raises(ValueError, match="usable"):
            impact.decide_route({"usable": not_boolean}, summary, 12)
    extra_keys = {"usable": True, "reason": "ok", "n_folds": 5}
    assert impact.decide_route(extra_keys, summary, 12).primary == "ot_importance"
    series = pd.Series({"usable": True, "reason": "ok"})
    assert impact.decide_route(series, summary, 12).primary == "ot_importance"


def test_decide_route_rejects_diagnostics_without_a_usable_entry():
    summary = loco_table(0.5, 1.0, 1.0)
    for diagnostics in (None, {}, {"other": True}, {"Usable": True}, [True], True, "usable", 1):
        with pytest.raises(ValueError, match="usable"):
            impact.decide_route(diagnostics, summary, 12)
    with pytest.raises(ValueError, match="usable"):
        impact.decide_route(pd.DataFrame({"usable": [True]}), summary, 12)


def test_decide_route_rejects_a_summary_without_valid_rmse_values():
    diagnostics = {"usable": True}
    complete = loco_table(0.5, 1.0, 1.0)
    for method in ("ot_weighted", "equal", "ot_uniform"):
        with pytest.raises(ValueError, match=method):
            impact.decide_route(diagnostics, complete.drop(index=method), 20)
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
                impact.decide_route(diagnostics, loco_table(*values), 20)
    with pytest.raises(ValueError):
        impact.decide_route(diagnostics, loco_table(0.0, 0.0, 0.0), 20)
    with pytest.raises(ValueError):
        impact.decide_route(diagnostics, loco_table(np.inf, np.inf, np.inf), 20)
    for text in ("0.5", None, True):
        textual = loco_table(0.5, 1.0, 1.0).astype({"rmse": object})
        textual.loc["ot_weighted", "rmse"] = text
        with pytest.raises(ValueError, match="ot_weighted"):
            impact.decide_route(diagnostics, textual, 20)
    float32 = complete.astype({"rmse": np.float32})
    assert impact.decide_route(diagnostics, float32, 20).primary == "ot_importance"


def test_decide_route_rejects_a_summary_that_is_not_indexed_by_method_name():
    diagnostics = {"usable": True}
    complete = loco_table(0.5, 1.0, 1.0)
    in_a_column = complete.reset_index(names="method")
    with pytest.raises(ValueError, match="ot_weighted"):
        impact.decide_route(diagnostics, in_a_column, 20)
    for index in (["OT_weighted", "equal", "ot_uniform"], ["ot_weighted ", "equal", "ot_uniform"]):
        mislabelled = complete.copy()
        mislabelled.index = index
        with pytest.raises(ValueError, match="ot_weighted"):
            impact.decide_route(diagnostics, mislabelled, 20)
    with pytest.raises(ValueError, match="rmse"):
        impact.decide_route(diagnostics, complete.rename(columns={"rmse": "mse"}), 20)
    duplicated = pd.DataFrame(
        {"rmse": [0.5, 0.6, 1.0, 1.0]}, index=["ot_weighted", "ot_weighted", "equal", "ot_uniform"]
    )
    with pytest.raises(ValueError, match="one row per method"):
        impact.decide_route(diagnostics, duplicated, 12)
    for not_a_frame in (None, complete["rmse"], complete.to_dict(), complete.to_numpy(), "rmse"):
        with pytest.raises(ValueError, match="DataFrame"):
            impact.decide_route(diagnostics, not_a_frame, 20)


def test_decide_route_requires_a_non_negative_integer_number_of_cases():
    summary = loco_table(0.5, 1.0, 1.0)
    diagnostics = {"usable": True}
    not_counts = (12.0, 12.5, np.float64(12.0), True, np.True_, None, "12", [12], -1, -10, np.nan)
    for n_cases in not_counts:
        with pytest.raises(ValueError, match="n_cases"):
            impact.decide_route(diagnostics, summary, n_cases)
    for n_cases in (12, np.int64(12), np.int32(12), np.uint8(12), 10**6):
        decision = impact.decide_route(diagnostics, summary, n_cases)
        assert decision.primary == "ot_importance"
    nothing = impact.decide_route(diagnostics, summary, 0)
    assert nothing.primary == "ambient" and nothing.passed["enough_cases"] is False
    assert "There are 0 cases" in nothing.details
    assert "There are 12 cases" in impact.decide_route(diagnostics, summary, np.int64(12)).details


def test_decide_route_details_report_each_criterion_in_plain_text():
    summary = loco_table(0.4, 0.5, 0.8)
    decision = impact.decide_route({"usable": True}, summary, 12)
    text = decision.details
    assert isinstance(text, str) and text.endswith(".")
    assert "0.4" in text and "0.5" in text and "0.8" in text
    assert "0.800 times the RMSE of equal" in text
    assert "0.500 times the RMSE of ot_uniform" in text
    assert "12 cases" in text and "at least 10" in text
    assert "primary route is ot_importance" in text
    failing = impact.decide_route({"usable": False}, loco_table(1.0, 1.0, 1.0), 4)
    assert "primary route is ambient" in failing.details
    assert "not met" in failing.details
    for decision in (decision, failing):
        assert EM_DASH not in decision.details


def test_decide_route_names_the_economy_that_is_left_out():
    decision = impact.decide_route({"usable": True}, loco_table(0.4, 0.5, 0.8), 12)
    assert "The leave-one-economy-out RMSE of ot_weighted is 0.4" in decision.details
    assert "leave-one-economy-out" in impact.decide_route.__doc__
    for module in (impact, meta):
        for name, member in vars(module).items():
            if inspect.isfunction(member) or inspect.isclass(member):
                assert "leave-one-case-out" not in (member.__doc__ or ""), name
    assert "leave-one-case-out" not in decision.details
    assert "leave-one-group-out" in impact._loco_rmse.__doc__
    assert "leave-one-group-out" not in decision.details


def test_route_decision_is_immutable():
    decision = impact.decide_route({"usable": True}, loco_table(0.5, 1.0, 1.0), 12)
    with pytest.raises(AttributeError):
        decision.primary = "ambient"


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
    decision = impact.decide_route({"usable": True}, loco_table(0.5, 1.0, 1.0), 12)
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
        tables, impact.decide_route({"usable": False}, loco_table(0.5, 1.0, 1.0), 12)
    )
    assert swapped["primary"].tolist() == [False] * 4 + [True] * 2


def test_route_comparison_supports_additional_routes_and_checks_the_primary_exists(baseline):
    pp = np.linspace(0.0, 1.0, 101)
    table = impact.impact_table(pp, None, baseline)
    ambient_decision = impact.decide_route({"usable": False}, loco_table(0.5, 1.0, 1.0), 12)
    every_route = {"meta": table, "ambient": table, "ot_importance": table}
    out = impact.route_comparison(every_route, ambient_decision)
    assert out.groupby("route")["primary"].all().to_dict() == {
        "ambient": True,
        "meta": False,
        "ot_importance": False,
    }
    primary_decision = impact.decide_route({"usable": True}, loco_table(0.5, 1.0, 1.0), 12)
    with pytest.raises(ValueError, match="ot_importance"):
        impact.route_comparison({"ambient": table}, primary_decision)
    with pytest.raises(ValueError):
        impact.route_comparison({}, ambient_decision)
    only_primary = impact.route_comparison({"ambient": table}, ambient_decision)
    assert only_primary["primary"].all()
