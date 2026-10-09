"""Tests for dtt.impact: baseline, effect scaling, detection, route decision and selection.

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
# Thailand on the basis of the panel
# ----------------------------------------------------------------------------
PANEL_RATIOS = {2015: 1.076, 2016: 1.080, 2017: 1.083, 2018: 1.086, 2019: 1.089}


@pytest.fixture
def long_baseline():
    """Hand-made baseline for 2015 to 2024 on the balance of payments basis."""
    years = list(range(2015, 2025))
    gdp = np.array([400.0 + 10.0 * (year - 2015) for year in years])
    receipts = np.array([50.0 + 2.0 * (year - 2015) for year in years])
    return pd.DataFrame(
        {
            "gdp_usd_bn": gdp,
            "receipts_usd_bn": receipts,
            "receipts_pct_gdp": 100.0 * receipts / gdp,
            "arrivals_million": np.linspace(30.0, 40.0, len(years)),
            "fx_thb_per_usd": np.linspace(33.0, 35.0, len(years)),
        },
        index=pd.Index(years, name="year"),
    )


def panel_from(baseline, ratios, years=None):
    """Panel receipts equal to the baseline times a yearly ratio."""
    years = list(ratios) if years is None else years
    return pd.Series(
        {year: baseline.loc[year, "receipts_usd_bn"] * ratios[year] for year in years},
        name="receipts",
    )


def test_to_panel_basis_applies_a_constant_ratio_to_the_years_without_a_panel_value(long_baseline):
    ratios = {year: 1.08 for year in range(2015, 2022)}
    panel = panel_from(long_baseline, ratios)
    out = impact.to_panel_basis(long_baseline, panel)
    gdp = long_baseline["gdp_usd_bn"]
    base = long_baseline["receipts_usd_bn"]
    expected = base * 1.08
    assert out["receipts_usd_bn"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-12)
    assert out["receipts_pct_gdp"].to_numpy() == pytest.approx(
        (100.0 * expected / gdp).to_numpy(), rel=1e-12
    )
    assert out["receipts_usd_bn_baseline"].to_numpy() == pytest.approx(base.to_numpy())
    assert out["receipts_basis"].tolist() == ["panel"] * 7 + ["ratio"] * 3
    assert np.isnan(out["basis_ratio"].to_numpy()[:7]).all()
    assert out["basis_ratio"].to_numpy()[7:] == pytest.approx(1.08, rel=1e-12)


def test_to_panel_basis_uses_the_panel_value_where_the_panel_has_one(long_baseline):
    panel = pd.Series({2019: 61.0, 2020: 20.0, 2024: 90.0})
    out = impact.to_panel_basis(long_baseline, panel, ratio_years=[2019])
    assert out.loc[2019, "receipts_usd_bn"] == 61.0
    assert out.loc[2020, "receipts_usd_bn"] == 20.0
    assert out.loc[2024, "receipts_usd_bn"] == 90.0
    ratio = 61.0 / long_baseline.loc[2019, "receipts_usd_bn"]
    assert out.loc[2016, "receipts_usd_bn"] == pytest.approx(
        long_baseline.loc[2016, "receipts_usd_bn"] * ratio, rel=1e-12
    )
    assert out.loc[2016, "basis_ratio"] == pytest.approx(ratio, rel=1e-12)
    assert out.loc[2016, "receipts_basis"] == "ratio"
    assert out.loc[2019, "receipts_basis"] == "panel" and np.isnan(out.loc[2019, "basis_ratio"])
    shares = 100.0 * out["receipts_usd_bn"] / out["gdp_usd_bn"]
    assert out["receipts_pct_gdp"].to_numpy() == pytest.approx(shares.to_numpy(), rel=1e-12)


def test_to_panel_basis_averages_the_yearly_ratios_over_the_ratio_years(long_baseline):
    panel = panel_from(long_baseline, PANEL_RATIOS)
    out = impact.to_panel_basis(long_baseline, panel)
    mean_ratio = float(np.mean(list(PANEL_RATIOS.values())))
    assert mean_ratio == pytest.approx(1.0828)
    later = out.loc[2022:2024]
    assert later["basis_ratio"].to_numpy() == pytest.approx(mean_ratio, rel=1e-12)
    assert later["receipts_usd_bn"].to_numpy() == pytest.approx(
        long_baseline.loc[2022:2024, "receipts_usd_bn"].to_numpy() * mean_ratio, rel=1e-12
    )
    restricted = impact.to_panel_basis(long_baseline, panel, ratio_years=(2018, 2019))
    assert restricted.loc[2024, "basis_ratio"] == pytest.approx(1.0875, rel=1e-12)
    from_range = impact.to_panel_basis(long_baseline, panel, ratio_years=range(2015, 2020))
    pd.testing.assert_frame_equal(from_range, out)
    from_array = impact.to_panel_basis(long_baseline, panel, ratio_years=np.arange(2015, 2020))
    pd.testing.assert_frame_equal(from_array, out)
    repeated = impact.to_panel_basis(long_baseline, panel, ratio_years=[2018, 2018, 2019])
    pd.testing.assert_frame_equal(repeated, restricted)


def test_to_panel_basis_ignores_ratio_years_without_both_values(long_baseline):
    panel = pd.Series({2017: 54.0 * 1.1, 2018: 56.0 * 1.2, 2019: 58.0 * 1.3})
    one_sided = impact.to_panel_basis(long_baseline, panel, ratio_years=[2010, 2017, 2023])
    assert one_sided.loc[2024, "basis_ratio"] == pytest.approx(1.1, rel=1e-12)
    gap = panel.copy()
    gap[2017] = np.nan
    missing = impact.to_panel_basis(long_baseline, gap, ratio_years=[2017, 2018])
    assert missing.loc[2024, "basis_ratio"] == pytest.approx(1.2, rel=1e-12)
    assert missing.loc[2017, "receipts_basis"] == "ratio"
    holes = long_baseline.copy()
    holes.loc[2018, "receipts_usd_bn"] = np.nan
    skipped = impact.to_panel_basis(holes, panel, ratio_years=[2017, 2018])
    assert skipped.loc[2024, "basis_ratio"] == pytest.approx(1.1, rel=1e-12)
    assert skipped.loc[2018, "receipts_basis"] == "panel"
    assert np.isnan(skipped.loc[2018, "receipts_usd_bn_baseline"])


def test_to_panel_basis_needs_a_year_with_both_values(long_baseline):
    panel = pd.Series({2020: 30.0, 2021: 31.0})
    with pytest.raises(ValueError, match="ratio_years"):
        impact.to_panel_basis(long_baseline, panel)
    with pytest.raises(ValueError, match="ratio_years"):
        impact.to_panel_basis(long_baseline, panel, ratio_years=[2015, 2016])
    alone = impact.to_panel_basis(long_baseline, panel, ratio_years=[2020])
    assert alone.loc[2024, "basis_ratio"] == pytest.approx(0.5, rel=1e-12)
    nothing = pd.Series({2015: np.nan, 2016: np.nan})
    with pytest.raises(ValueError, match="ratio_years"):
        impact.to_panel_basis(long_baseline, nothing, ratio_years=[2015, 2016])
    with pytest.raises(ValueError, match="ratio_years"):
        impact.to_panel_basis(long_baseline, pd.Series({2018: 60.0}), ratio_years=[2030])


def test_to_panel_basis_keeps_gdp_other_columns_and_the_index(long_baseline):
    before = long_baseline.copy()
    panel = panel_from(long_baseline, PANEL_RATIOS)
    out = impact.to_panel_basis(long_baseline, panel)
    pd.testing.assert_frame_equal(long_baseline, before)
    assert out is not long_baseline
    pd.testing.assert_index_equal(out.index, long_baseline.index)
    for column in ("gdp_usd_bn", "arrivals_million", "fx_thb_per_usd"):
        pd.testing.assert_series_equal(out[column], long_baseline[column])
    assert list(out.columns) == [
        *long_baseline.columns,
        "receipts_usd_bn_baseline",
        "receipts_basis",
        "basis_ratio",
    ]
    assert out["basis_ratio"].dtype == float


def test_to_panel_basis_accepts_a_year_column_a_mapping_and_float_years(long_baseline):
    panel = panel_from(long_baseline, PANEL_RATIOS)
    reference = impact.to_panel_basis(long_baseline, panel)
    as_column = long_baseline.reset_index()
    out = impact.to_panel_basis(as_column, panel)
    assert out["year"].tolist() == long_baseline.index.tolist()
    assert out["receipts_usd_bn"].to_numpy() == pytest.approx(
        reference["receipts_usd_bn"].to_numpy()
    )
    from_mapping = impact.to_panel_basis(long_baseline, dict(panel))
    pd.testing.assert_frame_equal(from_mapping, reference)
    floats = panel.copy()
    floats.index = floats.index.astype(float)
    pd.testing.assert_frame_equal(impact.to_panel_basis(long_baseline, floats), reference)
    shuffled = panel.iloc[::-1]
    pd.testing.assert_frame_equal(impact.to_panel_basis(long_baseline, shuffled), reference)


def test_to_panel_basis_ignores_panel_years_outside_the_baseline(long_baseline):
    panel = panel_from(long_baseline, PANEL_RATIOS)
    panel[1990] = 12.0
    panel[2031] = 99.0
    out = impact.to_panel_basis(long_baseline, panel)
    reference = impact.to_panel_basis(long_baseline, panel_from(long_baseline, PANEL_RATIOS))
    pd.testing.assert_frame_equal(out, reference)


def test_to_panel_basis_keeps_a_missing_baseline_value_missing(long_baseline):
    holes = long_baseline.copy()
    holes.loc[2023, "receipts_usd_bn"] = np.nan
    panel = panel_from(long_baseline, PANEL_RATIOS)
    out = impact.to_panel_basis(holes, panel)
    assert np.isnan(out.loc[2023, "receipts_usd_bn"])
    assert np.isnan(out.loc[2023, "receipts_pct_gdp"])
    assert out.loc[2023, "receipts_basis"] == "ratio"
    assert np.isfinite(out.loc[2022, "receipts_usd_bn"])


def test_to_panel_basis_rejects_bad_inputs(long_baseline):
    panel = panel_from(long_baseline, PANEL_RATIOS)
    for not_a_series in (None, [60.0, 61.0], 60.0, "panel", np.array([60.0])):
        with pytest.raises(ValueError, match="panel_receipts_usd_bn"):
            impact.to_panel_basis(long_baseline, not_a_series)
    for bad in (0.0, -3.0, np.inf):
        broken = panel.copy()
        broken[2016] = bad
        with pytest.raises(ValueError, match="panel_receipts_usd_bn"):
            impact.to_panel_basis(long_baseline, broken)
    repeated = pd.Series([60.0, 61.0], index=[2018, 2018])
    with pytest.raises(ValueError, match="more than one"):
        impact.to_panel_basis(long_baseline, repeated, ratio_years=[2018])
    fractional = pd.Series([60.0], index=[2018.5])
    with pytest.raises(ValueError, match="whole-number"):
        impact.to_panel_basis(long_baseline, fractional)
    with pytest.raises(ValueError, match="whole-number"):
        impact.to_panel_basis(long_baseline, pd.Series([60.0], index=["x"]))
    for column in ("gdp_usd_bn", "receipts_usd_bn"):
        with pytest.raises(ValueError, match=column):
            impact.to_panel_basis(long_baseline.drop(columns=column), panel)
    for bad in (0.0, -1.0, np.inf):
        broken_baseline = long_baseline.copy()
        broken_baseline.loc[2016, "receipts_usd_bn"] = bad
        with pytest.raises(ValueError, match="receipts_usd_bn"):
            impact.to_panel_basis(broken_baseline, panel)
    converted = impact.to_panel_basis(long_baseline, panel)
    with pytest.raises(ValueError, match="already has"):
        impact.to_panel_basis(converted, panel)
    for years in ([], [2015.5], ["2015"], [True]):
        with pytest.raises(ValueError):
            impact.to_panel_basis(long_baseline, panel, ratio_years=years)
    twice = pd.concat([long_baseline, long_baseline.iloc[:1]])
    with pytest.raises(ValueError, match="more than one row"):
        impact.to_panel_basis(twice, panel)


def test_to_panel_basis_works_on_the_output_of_the_loader(tmp_path):
    rows = []
    for year in (2015, 2019, 2024):
        rows += long_rows(year, 400.0 + year - 2015, 50.0 + year - 2015, 12.0, 30.0, 33.0)
    path = write_long_file(tmp_path / "baseline.csv", rows)
    loaded = impact.load_thailand_baseline(path, years=[2015, 2019, 2024])
    panel = pd.Series({2015: 54.0, 2019: 58.32})
    out = impact.to_panel_basis(loaded, panel)
    assert out.loc[2015, "receipts_usd_bn"] == 54.0
    assert out.loc[2024, "basis_ratio"] == pytest.approx(1.08, rel=1e-12)
    assert out.loc[2024, "receipts_usd_bn"] == pytest.approx(59.0 * 1.08, rel=1e-12)
    assert out.loc[2024, "receipts_pct_gdp"] == pytest.approx(100.0 * 59.0 * 1.08 / 409.0)
    table = impact.impact_table(np.full(5, 0.5), np.full(5, 10.0), out, years=[2019, 2024])
    relative = table[table["scaling"] == "relative"].set_index("baseline_year")
    assert relative.loc[2024, "q50"] == pytest.approx(0.10 * 59.0 * 1.08, rel=1e-12)


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
# Placebo null on the scale of the target
# ----------------------------------------------------------------------------
def test_rescale_null_pp_with_a_factor_of_one_leaves_the_draws_unchanged():
    null = np.random.default_rng(11).standard_normal(500)
    draws, factor = impact.rescale_null_pp(null, 5.7, 5.7)
    assert factor == 1.0 and isinstance(factor, float)
    assert isinstance(draws, np.ndarray)
    np.testing.assert_array_equal(draws, null)
    assert draws is not null
    draws[0] = 99.0
    assert null[0] != 99.0


def test_rescale_null_pp_multiplies_by_the_ratio_of_target_to_source_share():
    null = np.array([-1.0, -0.5, 0.0, 0.25, 2.0])
    draws, factor = impact.rescale_null_pp(null, 5.7, 7.81)
    assert factor == pytest.approx(7.81 / 5.7, rel=1e-14)
    np.testing.assert_allclose(draws, null * 7.81 / 5.7, rtol=1e-14)
    smaller, shrink = impact.rescale_null_pp(null, 8.0, 2.0)
    assert shrink == 0.25
    np.testing.assert_allclose(smaller, 0.25 * null, rtol=1e-15)
    result = impact.rescale_null_pp(null, 5.7, 7.81)
    assert isinstance(result, tuple) and len(result) == 2
    np.testing.assert_array_equal(null, [-1.0, -0.5, 0.0, 0.25, 2.0])


@pytest.mark.parametrize("scale", [0.25, 1.0, 1.37, 3.0])
def test_scaling_a_null_by_k_scales_the_minimum_detectable_effect_by_k(scale):
    rng = np.random.default_rng(12)
    null = rng.standard_t(5, 4000) * 0.3 + 0.05
    base = impact.minimum_detectable_effect(null)
    draws, factor = impact.rescale_null_pp(null, 4.0, 4.0 * scale)
    assert factor == pytest.approx(scale, rel=1e-14)
    scaled = impact.minimum_detectable_effect(draws)
    for key in ("threshold", "mde", "mde_normal"):
        assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12)


def test_a_null_on_the_scale_of_the_target_changes_the_detection_probability():
    rng = np.random.default_rng(13)
    null = rng.normal(0.0, 0.25, 6000)
    effect = np.full(400, 0.6)
    limits = impact.minimum_detectable_effect(null)
    plain = impact.detection_probability(effect, limits["threshold"], null_draws=null)
    wide, factor = impact.rescale_null_pp(null, 5.7, 7.81)
    wide_limits = impact.minimum_detectable_effect(wide)
    assert factor > 1.0 and wide_limits["mde"] > limits["mde"]
    scaled = impact.detection_probability(effect, wide_limits["threshold"], null_draws=wide)
    assert scaled < plain
    shrunk, _ = impact.rescale_null_pp(null, 7.81, 5.7)
    narrow_limits = impact.minimum_detectable_effect(shrunk)
    narrow = impact.detection_probability(effect, narrow_limits["threshold"], null_draws=shrunk)
    assert narrow > plain


def test_rescale_null_pp_rejects_invalid_shares_and_draws():
    null = np.linspace(-1.0, 1.0, 11)
    bad_shares = (0.0, -1.0, np.nan, np.inf, -np.inf, True, "5.7", None, [5.7])
    for bad in bad_shares:
        with pytest.raises(ValueError, match="source_receipts_share"):
            impact.rescale_null_pp(null, bad, 7.81)
        with pytest.raises(ValueError, match="target_receipts_share"):
            impact.rescale_null_pp(null, 5.7, bad)
    for bad_draws in ([], [[1.0, 2.0]], [1.0, np.nan], [np.inf, 0.0], 3.0):
        with pytest.raises(ValueError, match="null_pp"):
            impact.rescale_null_pp(bad_draws, 5.7, 7.81)
    draws, factor = impact.rescale_null_pp(null, np.float32(2.0), np.int64(3))
    assert factor == pytest.approx(1.5)
    np.testing.assert_allclose(draws, 1.5 * null)


def test_source_receipts_share_is_the_usage_weighted_mean():
    shares = [4.0, 6.0, 10.0]
    assert impact.source_receipts_share(shares, [1.0, 1.0, 2.0]) == pytest.approx(7.5)
    assert impact.source_receipts_share(shares, [0.2, 0.2, 0.4]) == pytest.approx(7.5)
    assert impact.source_receipts_share(shares, [3.0, 0.0, 0.0]) == pytest.approx(4.0)
    assert impact.source_receipts_share(shares, [1.0, 1.0, 1.0]) == pytest.approx(20.0 / 3.0)
    mass = impact.source_receipts_share([5.0, 5.0, 5.0], [0.3, 0.3, 0.4])
    assert mass == pytest.approx(5.0, rel=1e-14)
    assert isinstance(impact.source_receipts_share(shares, [1.0, 1.0, 2.0]), float)
    assert impact.source_receipts_share(np.array(shares), np.array([1, 1, 2])) == pytest.approx(7.5)


def test_source_receipts_share_follows_the_sources_that_carry_the_mass():
    shares = pd.Series({"CHN": 1.1, "MYS": 5.7, "PRT": 15.0, "GRC": 14.0})
    usage = pd.Series({"MYS": 0.7, "CHN": 0.05, "GRC": 0.05, "PRT": 0.2})
    value = impact.source_receipts_share(shares, usage)
    assert value == pytest.approx(0.055 + 3.99 + 0.7 + 3.0)
    shuffled = usage.loc[["PRT", "GRC", "CHN", "MYS"]]
    assert impact.source_receipts_share(shares, shuffled) == pytest.approx(value, rel=1e-14)
    positional = impact.source_receipts_share(shares.to_numpy(), usage.to_numpy())
    assert positional != pytest.approx(value)
    with pytest.raises(ValueError, match="same sources"):
        impact.source_receipts_share(shares, usage.drop("CHN"))
    with pytest.raises(ValueError, match="same sources"):
        impact.source_receipts_share(shares, usage.rename({"CHN": "CAN"}))
    duplicated = pd.Series([1.0, 2.0], index=["a", "a"])
    with pytest.raises(ValueError, match="one entry per source"):
        impact.source_receipts_share(duplicated, duplicated)


def test_source_receipts_share_ignores_sources_without_usage():
    shares = [4.0, np.nan, 10.0]
    assert impact.source_receipts_share(shares, [1.0, 0.0, 1.0]) == pytest.approx(7.0)
    with pytest.raises(ValueError, match="in use"):
        impact.source_receipts_share(shares, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="in use"):
        impact.source_receipts_share([4.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="in use"):
        impact.source_receipts_share([4.0, -2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="in use"):
        impact.source_receipts_share([4.0, np.inf], [1.0, 1.0])


def test_source_receipts_share_rejects_invalid_usage_and_shapes():
    shares = [4.0, 6.0]
    for usage in ([0.0, 0.0], [-1.0, 2.0], [np.nan, 1.0], [np.inf, 1.0]):
        with pytest.raises(ValueError, match="usage"):
            impact.source_receipts_share(shares, usage)
    for bad in ([1.0], [1.0, 1.0, 1.0], [[1.0, 1.0]], 1.0):
        with pytest.raises(ValueError):
            impact.source_receipts_share(shares, bad)
    with pytest.raises(ValueError):
        impact.source_receipts_share([], [])
    with pytest.raises(ValueError):
        impact.source_receipts_share([[4.0, 6.0]], [[1.0, 1.0]])


def test_the_rescaled_null_gives_the_minimum_detectable_effect_of_the_target_scale():
    rng = np.random.default_rng(14)
    shares = np.array([5.0, 6.0, 8.0])
    usage = np.array([0.5, 0.3, 0.2])
    source = impact.source_receipts_share(shares, usage)
    assert source == pytest.approx(5.9)
    placebo = rng.normal(0.0, 0.2, 5000)
    draws, factor = impact.rescale_null_pp(placebo, source, 7.81)
    assert factor == pytest.approx(7.81 / 5.9, rel=1e-14)
    mde_target = impact.minimum_detectable_effect(draws)["mde"]
    mde_source = impact.minimum_detectable_effect(placebo)["mde"]
    assert mde_target / mde_source == pytest.approx(7.81 / 5.9, rel=1e-12)


# ----------------------------------------------------------------------------
# decide_route
# ----------------------------------------------------------------------------
SUPPORTED = {"supported": True, "ess": 4.2, "reasons": []}
UNSUPPORTED = {
    "supported": False,
    "ess": 1.97,
    "reasons": ["cost share is far outside the sources"],
}
CRITERIA = ("importance_usable", "loco_rmse", "enough_cases", "target_supported")
MANY = 25


def loco_table(rmse_weighted, rmse_equal, rmse_uniform, extra=None):
    values = {"ot_weighted": rmse_weighted, "equal": rmse_equal, "ot_uniform": rmse_uniform}
    values.update(extra or {})
    return pd.DataFrame({"rmse": list(values.values())}, index=list(values))


def decide(summary=None, n_cases=MANY, usable=True, support=SUPPORTED, **kwargs):
    """Call decide_route with a passing default for every input."""
    if summary is None:
        summary = loco_table(0.5, 1.0, 1.0)
    return impact.decide_route({"usable": usable}, summary, n_cases, support=support, **kwargs)


@pytest.mark.parametrize(
    "usable, rmse_ok, enough, supported", list(itertools.product([True, False], repeat=4))
)
def test_decide_route_truth_table(usable, rmse_ok, enough, supported):
    summary = loco_table(0.9, 1.0 if rmse_ok else 0.8, 1.0)
    n_cases = 20 if enough else 19
    support = SUPPORTED if supported else UNSUPPORTED
    decision = impact.decide_route({"usable": usable}, summary, n_cases, support=support)
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
    assert f"the primary route is {expected}." in decision.details
    if expected == "ot_importance":
        assert "All four criteria are met" in decision.details
    else:
        assert "At least one criterion is not met" in decision.details


@pytest.mark.parametrize(
    "beats_equal, beats_uniform", list(itertools.product([True, False], repeat=2))
)
def test_decide_route_rmse_criterion_needs_both_comparisons(beats_equal, beats_uniform):
    summary = loco_table(0.9, 1.0 if beats_equal else 0.8, 1.0 if beats_uniform else 0.8)
    decision = decide(summary)
    assert decision.passed["loco_rmse"] is (beats_equal and beats_uniform)
    assert decision.primary == ("ot_importance" if beats_equal and beats_uniform else "ambient")


def test_decide_route_boundaries_follow_at_most_and_at_least():
    summary = loco_table(0.95, 1.0, 1.0)
    on_the_line = decide(summary, 20)
    assert on_the_line.primary == "ot_importance"
    assert on_the_line.passed == dict.fromkeys(CRITERIA, True)
    above = decide(loco_table(0.9501, 1.0, 1.0), 20)
    assert above.primary == "ambient" and above.passed["loco_rmse"] is False
    above_uniform = decide(loco_table(0.95, 1.0, 0.99), 20)
    assert above_uniform.primary == "ambient"
    few = decide(summary, 19)
    assert few.primary == "ambient" and few.passed["enough_cases"] is False
    worse = decide(loco_table(1.2, 1.0, 1.0), 40)
    assert worse.primary == "ambient"


def test_decide_route_compares_rmse_values_with_a_tolerance_for_rounding_noise():
    noise = 0.95 * (1.0 + 1e-14)
    assert decide(loco_table(noise, 1.0, 1.0)).passed["loco_rmse"] is True
    assert decide(loco_table(0.95 * (1.0 + 1e-9), 1.0, 1.0)).passed["loco_rmse"] is False


def test_decide_route_ignores_other_methods_and_uses_the_named_ones():
    summary = loco_table(0.5, 1.0, 1.0, extra={"ot_other": 0.1, "ambient": 0.2})
    assert decide(summary).primary == "ot_importance"
    shuffled = summary.iloc[::-1]
    assert decide(shuffled).primary == "ot_importance"
    swapped = loco_table(1.0, 0.5, 0.5, extra={"ot_other": 0.1})
    assert decide(swapped).primary == "ambient"


def test_decide_route_default_thresholds_and_overrides():
    assert dict(impact.DEFAULT_ROUTE_RULES) == {
        "rmse_ratio_equal": 0.95,
        "rmse_ratio_uniform": 0.95,
        "min_cases": 20,
    }
    with pytest.raises(TypeError):
        impact.DEFAULT_ROUTE_RULES["min_cases"] = 3
    summary = loco_table(0.97, 1.0, 1.0)
    assert decide(summary, 25).primary == "ambient"
    relaxed = decide(summary, 25, rules={"rmse_ratio_equal": 1.0, "rmse_ratio_uniform": 1.0})
    assert relaxed.primary == "ot_importance"
    only_equal = decide(summary, 25, rules={"rmse_ratio_equal": 1.0})
    assert only_equal.primary == "ambient"
    good = loco_table(0.5, 1.0, 1.0)
    assert decide(good, 12).primary == "ambient"
    small = decide(good, 6, rules={"min_cases": 5})
    assert small.primary == "ot_importance"
    strict = decide(good, 25, rules={"min_cases": 30})
    assert strict.primary == "ambient"
    ten = decide(good, 10, rules={"min_cases": 10})
    assert ten.primary == "ot_importance"
    tight = decide(loco_table(0.9, 1.0, 1.0), 25, rules={"rmse_ratio_uniform": 0.8})
    assert tight.primary == "ambient"
    # an empty override leaves the defaults unchanged
    unchanged = decide(loco_table(0.9, 1.0, 1.0), 20, rules={})
    assert unchanged.primary == "ot_importance"
    assert decide(loco_table(0.9, 1.0, 1.0), 19, rules={}).primary == "ambient"
    with pytest.raises(ValueError, match="min_case"):
        decide(summary, 25, rules={"min_case": 5})
    with pytest.raises(ValueError):
        decide(summary, 25, rules={"min_cases": float("nan")})
    with pytest.raises(ValueError):
        decide(summary, 25, rules={"rmse_ratio_equal": -0.1})


@pytest.mark.parametrize("name", ["rmse_ratio_equal", "rmse_ratio_uniform"])
def test_decide_route_accepts_rmse_ratio_thresholds_in_the_interval_zero_to_one_and_a_half(name):
    summary = loco_table(0.5, 1.0, 1.0)
    for value in (1e-9, 0.3, 0.95, 1.0, 1.5, np.float64(1.2), np.float32(0.5)):
        decision = decide(summary, rules={name: value})
        assert decision.passed["loco_rmse"] is (0.5 <= float(value))
    for value in (0.0, -0.5, 1.5000001, 2.0, np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match=name):
            decide(summary, rules={name: value})
    for value in (True, "0.9", None, [0.9], 1 + 0j):
        with pytest.raises(ValueError, match=name):
            decide(summary, rules={name: value})


def test_decide_route_accepts_a_non_negative_finite_minimum_number_of_cases():
    summary = loco_table(0.5, 1.0, 1.0)
    zero = decide(summary, 0, rules={"min_cases": 0})
    assert zero.primary == "ot_importance" and zero.passed["enough_cases"] is True
    fractional = decide(summary, 2, rules={"min_cases": 2.5})
    assert fractional.passed["enough_cases"] is False
    assert "at least 2.5 are required" in fractional.details
    for value in (-1, -0.001, np.inf, np.nan, True, "10", None):
        with pytest.raises(ValueError, match="min_cases"):
            decide(summary, rules={"min_cases": value})
    for bad_rules in ([("min_cases", 5)], 5, "min_cases", (10,)):
        with pytest.raises(ValueError, match="mapping"):
            decide(summary, rules=bad_rules)


def test_decide_route_reads_the_usable_flag_strictly():
    summary = loco_table(0.5, 1.0, 1.0)
    for flag in (True, np.True_):
        decision = decide(summary, usable=flag)
        assert decision.primary == "ot_importance"
        assert decision.passed["importance_usable"] is True
    for flag in (False, np.False_):
        decision = decide(summary, usable=flag)
        assert decision.primary == "ambient"
        assert decision.passed["importance_usable"] is False
    for not_boolean in (1, 0, 1.0, "yes", "True", None, [True], np.int64(1), np.float64(1.0)):
        with pytest.raises(ValueError, match="usable"):
            decide(summary, usable=not_boolean)
    extra_keys = {"usable": True, "reason": "ok", "n_folds": 5}
    decision = impact.decide_route(extra_keys, summary, MANY, support=SUPPORTED)
    assert decision.primary == "ot_importance"
    series = pd.Series({"usable": True, "reason": "ok"})
    decision = impact.decide_route(series, summary, MANY, support=SUPPORTED)
    assert decision.primary == "ot_importance"


def test_decide_route_rejects_diagnostics_without_a_usable_entry():
    summary = loco_table(0.5, 1.0, 1.0)
    for diagnostics in (None, {}, {"other": True}, {"Usable": True}, [True], True, "usable", 1):
        with pytest.raises(ValueError, match="usable"):
            impact.decide_route(diagnostics, summary, MANY, support=SUPPORTED)
    with pytest.raises(ValueError, match="usable"):
        impact.decide_route(pd.DataFrame({"usable": [True]}), summary, MANY, support=SUPPORTED)


def test_decide_route_rejects_a_summary_without_valid_rmse_values():
    complete = loco_table(0.5, 1.0, 1.0)
    for method in ("ot_weighted", "equal", "ot_uniform"):
        with pytest.raises(ValueError, match=method):
            decide(complete.drop(index=method))
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
                decide(loco_table(*values))
    with pytest.raises(ValueError):
        decide(loco_table(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        decide(loco_table(np.inf, np.inf, np.inf))
    for text in ("0.5", None, True):
        textual = loco_table(0.5, 1.0, 1.0).astype({"rmse": object})
        textual.loc["ot_weighted", "rmse"] = text
        with pytest.raises(ValueError, match="ot_weighted"):
            decide(textual)
    float32 = complete.astype({"rmse": np.float32})
    assert decide(float32).primary == "ot_importance"


def test_decide_route_rejects_a_summary_that_is_not_indexed_by_method_name():
    complete = loco_table(0.5, 1.0, 1.0)
    in_a_column = complete.reset_index(names="method")
    with pytest.raises(ValueError, match="ot_weighted"):
        decide(in_a_column)
    for index in (["OT_weighted", "equal", "ot_uniform"], ["ot_weighted ", "equal", "ot_uniform"]):
        mislabelled = complete.copy()
        mislabelled.index = index
        with pytest.raises(ValueError, match="ot_weighted"):
            decide(mislabelled)
    with pytest.raises(ValueError, match="rmse"):
        decide(complete.rename(columns={"rmse": "mse"}))
    duplicated = pd.DataFrame(
        {"rmse": [0.5, 0.6, 1.0, 1.0]}, index=["ot_weighted", "ot_weighted", "equal", "ot_uniform"]
    )
    with pytest.raises(ValueError, match="one row per method"):
        decide(duplicated)
    for not_a_frame in (None, complete["rmse"], complete.to_dict(), complete.to_numpy(), "rmse"):
        with pytest.raises(ValueError, match="DataFrame"):
            impact.decide_route({"usable": True}, not_a_frame, MANY, support=SUPPORTED)


def test_decide_route_requires_a_non_negative_integer_number_of_cases():
    summary = loco_table(0.5, 1.0, 1.0)
    not_counts = (25.0, 25.5, np.float64(25.0), True, np.True_, None, "25", [25], -1, -10, np.nan)
    for n_cases in not_counts:
        with pytest.raises(ValueError, match="n_cases"):
            decide(summary, n_cases)
    for n_cases in (25, np.int64(25), np.int32(25), np.uint8(25), 10**6):
        assert decide(summary, n_cases).primary == "ot_importance"
    nothing = decide(summary, 0)
    assert nothing.primary == "ambient" and nothing.passed["enough_cases"] is False
    assert "There are 0 cases" in nothing.details
    assert "There are 25 cases" in decide(summary, np.int64(25)).details


def test_decide_route_details_report_each_criterion_in_plain_text():
    summary = loco_table(0.4, 0.5, 0.8)
    decision = decide(summary, 25)
    text = decision.details
    assert isinstance(text, str) and text.endswith(".")
    assert "0.4" in text and "0.5" in text and "0.8" in text
    assert "0.800 times the RMSE of equal" in text
    assert "0.500 times the RMSE of ot_uniform" in text
    assert "25 cases" in text and "at least 20" in text
    assert "support criterion is met" in text
    assert "primary route is ot_importance" in text
    failing = decide(loco_table(1.0, 1.0, 1.0), 4, usable=False, support=UNSUPPORTED)
    assert "primary route is ambient" in failing.details
    assert "not met" in failing.details
    for item in (decision, failing):
        assert EM_DASH not in item.details


def test_decide_route_names_the_economy_that_is_left_out():
    decision = decide(loco_table(0.4, 0.5, 0.8))
    assert "The leave-one-economy-out RMSE of ot_weighted is 0.4" in decision.details
    assert "leave-one-economy-out" in impact.decide_route.__doc__
    for module in (impact, meta):
        for name, member in vars(module).items():
            if inspect.isfunction(member) or inspect.isclass(member):
                assert "leave-one-case-out" not in (member.__doc__ or ""), name
    assert "leave-one-case-out" not in decision.details
    assert "leave-one-group-out" in impact._loco_rmse.__doc__
    assert "leave-one-group-out" not in decision.details


def test_decide_route_documents_that_economies_are_the_independent_units():
    doc = " ".join(impact.decide_route.__doc__.split())
    assert "counts episodes" in doc
    assert "independent units are the economies" in doc
    assert "at least 20" in doc
    assert "about 20 episodes" in doc


# ---- the support criterion ---------------------------------------------------
def test_decide_route_requires_the_support_result():
    summary = loco_table(0.5, 1.0, 1.0)
    with pytest.raises(ValueError, match="support is required: pass the result of"):
        impact.decide_route({"usable": True}, summary, MANY)
    with pytest.raises(ValueError, match="transport.target_support"):
        impact.decide_route({"usable": True}, summary, MANY, support=None)
    with pytest.raises(ValueError, match="support is required"):
        impact.decide_route({"usable": True}, summary, MANY, rules={"min_cases": 5})
    signature = inspect.signature(impact.decide_route)
    assert list(signature.parameters) == [
        "importance_diagnostics",
        "loco_summary",
        "n_cases",
        "rules",
        "support",
        "ambient_rmse",
    ]
    assert signature.parameters["support"].default is None
    assert signature.parameters["ambient_rmse"].default is None


def test_decide_route_support_decides_the_route_when_the_other_criteria_hold():
    summary = loco_table(0.4, 1.0, 1.0)
    supported = decide(summary, 40)
    unsupported = decide(summary, 40, support=UNSUPPORTED)
    assert supported.primary == "ot_importance" and unsupported.primary == "ambient"
    assert unsupported.passed == {
        "importance_usable": True,
        "loco_rmse": True,
        "enough_cases": True,
        "target_supported": False,
    }
    only_flag = decide(summary, 40, support={"supported": True})
    assert only_flag.primary == "ot_importance"
    assert "effective number" not in only_flag.details
    assert "support criterion is met" in only_flag.details


def test_decide_route_states_the_support_result_with_ess_and_reasons():
    reasons = ["the cost share is 14.8 robust standard deviations out.", "ess is low"]
    decision = decide(support={"supported": False, "ess": 1.97, "reasons": reasons})
    text = decision.details
    assert "(effective number of sources 1.97)" in text
    assert "the cost share is 14.8 robust standard deviations out; ess is low" in text
    assert ".;" not in text and ".," not in text and ".." not in text
    assert "support criterion is not met" in text
    supported = decide(support={"supported": True, "ess": 3.456}).details
    assert "(effective number of sources 3.46)" in supported
    assert "support criterion is met" in supported
    bare = decide(support={"supported": False}).details
    assert "is supported, so the support criterion is not met." in bare
    assert "effective number" not in bare
    single = decide(support={"supported": False, "reasons": "outside the reach"}).details
    assert "supported: outside the reach, so the support" in single
    blank = decide(support={"supported": False, "reasons": ["", "  "]}).details
    assert "supported, so the support criterion is not met" in blank
    assert EM_DASH not in text


def test_decide_route_reads_the_support_result_strictly():
    for flag in (np.True_, np.False_):
        assert decide(support={"supported": flag}).passed["target_supported"] is bool(flag)
    for bad in (1, 0, 1.0, "yes", "True", None, [True], np.int64(1)):
        with pytest.raises(ValueError, match="supported"):
            decide(support={"supported": bad})
    for not_a_mapping in ([True], True, "supported", 1, (True,), np.array([True])):
        with pytest.raises(ValueError, match="supported"):
            decide(support=not_a_mapping)
    for incomplete in ({}, {"ess": 3.0}, {"Supported": True}, {"reasons": []}):
        with pytest.raises(ValueError, match="supported"):
            decide(support=incomplete)
    for bad in ("3.0", True, [3.0], 1 + 0j):
        with pytest.raises(ValueError, match="ess"):
            decide(support={"supported": True, "ess": bad})
    for bad in (5, [1], [None], {"a": "b"}, b"abc", [["x"]]):
        with pytest.raises(ValueError, match="reasons"):
            decide(support={"supported": False, "reasons": bad})
    accepted = (
        {"supported": True, "ess": None, "reasons": None},
        {"supported": True, "ess": np.float64(2.5), "reasons": ("a", "b")},
        {"supported": True, "ess": 3, "other": object()},
        {"supported": True, "ess": float("nan")},
        pd.Series({"supported": True, "ess": 2.0, "reasons": ["x"]}),
    )
    for support in accepted:
        assert decide(support=support).passed["target_supported"] is True


# ---- the comparisons that are reported and not used --------------------------
def test_decide_route_reports_the_ambient_and_neighbour_comparisons_without_using_them():
    summary = loco_table(0.365, 0.5, 0.6, extra={"nn1": 0.35, "nn3": 0.276})
    ambient = {"A1": 0.40, "A2": 0.35, "A3": 0.323}
    decision = decide(summary, ambient_rmse=ambient)
    info = decision.info
    assert list(info) == [
        "rmse_ratio_to_best_ambient",
        "best_ambient",
        "rmse_ratio_to_best_neighbour",
        "best_neighbour",
    ]
    assert info["best_ambient"] == "A3"
    assert info["rmse_ratio_to_best_ambient"] == pytest.approx(0.365 / 0.323, rel=1e-14)
    assert info["best_neighbour"] == "nn3"
    assert info["rmse_ratio_to_best_neighbour"] == pytest.approx(0.365 / 0.276, rel=1e-14)
    assert type(info["rmse_ratio_to_best_ambient"]) is float
    text = decision.details
    assert "1.130 times the RMSE of the best ambient estimator A3 (0.323)" in text
    assert "1.322 times the RMSE of the best neighbour predictor nn3 (0.276)" in text
    assert text.count("this is reported and is not used by the rule") == 2
    # the rule does not read them: the primary route and the criteria are the same
    plain = decide(summary)
    assert plain.primary == decision.primary == "ot_importance"
    assert plain.passed == decision.passed
    worse = decide(summary, ambient_rmse={"A3": 0.01})
    assert worse.primary == "ot_importance" and worse.passed == decision.passed
    assert set(plain.info) == {"rmse_ratio_to_best_neighbour", "best_neighbour"}
    assert "best ambient" not in plain.details


def test_decide_route_info_depends_on_what_is_available():
    plain = decide()
    assert plain.info == {}
    assert "ambient estimator" not in plain.details and "neighbour" not in plain.details
    only_ambient = decide(ambient_rmse={"A1": 0.4, "A2": 0.25})
    assert only_ambient.info == {
        "rmse_ratio_to_best_ambient": pytest.approx(0.5 / 0.25),
        "best_ambient": "A2",
    }
    assert "neighbour" not in only_ambient.details
    summary = loco_table(0.5, 1.0, 1.0, extra={"nn1": 0.2})
    only_neighbour = decide(summary)
    assert only_neighbour.info == {
        "rmse_ratio_to_best_neighbour": pytest.approx(0.5 / 0.2),
        "best_neighbour": "nn1",
    }
    assert "ambient estimator" not in only_neighbour.details
    empty = decide(ambient_rmse={})
    assert empty.info == {}


def test_decide_route_ambient_entries_named_nn1_or_nn3_are_neighbour_predictors():
    summary = loco_table(0.5, 1.0, 1.0, extra={"nn1": 0.4, "nn3": 0.3})
    decision = decide(summary, ambient_rmse={"A1": 0.45, "nn3": 0.5, "nn1": 0.25})
    assert decision.info["best_ambient"] == "A1"
    assert decision.info["rmse_ratio_to_best_ambient"] == pytest.approx(0.5 / 0.45)
    assert decision.info["best_neighbour"] == "nn1"
    assert decision.info["rmse_ratio_to_best_neighbour"] == pytest.approx(0.5 / 0.25)
    nn_only = decide(ambient_rmse={"nn3": 0.2})
    assert "best_ambient" not in nn_only.info and nn_only.info["best_neighbour"] == "nn3"


def test_decide_route_reports_the_first_of_equal_best_values_and_ratios_below_one():
    decision = decide(ambient_rmse={"A2": 0.4, "A3": 0.4, "A1": 0.7})
    assert decision.info["best_ambient"] == "A2"
    tie = decide(loco_table(0.5, 1.0, 1.0, extra={"nn1": 0.25, "nn3": 0.25}))
    assert tie.info["best_neighbour"] == "nn1"
    ahead = decide(ambient_rmse=pd.Series({"A1": 0.8, "A3": 1.0}))
    assert ahead.info["best_ambient"] == "A1"
    assert ahead.info["rmse_ratio_to_best_ambient"] == pytest.approx(0.5 / 0.8)
    assert "0.625 times the RMSE of the best ambient estimator A1" in ahead.details


def test_decide_route_rejects_invalid_ambient_or_neighbour_rmse():
    for not_a_mapping in ([("A1", 0.3)], 0.3, "A1", (0.3,), np.array([0.3])):
        with pytest.raises(ValueError, match="ambient_rmse"):
            decide(ambient_rmse=not_a_mapping)
    for bad in (np.nan, np.inf, 0.0, -0.1, True, "0.3", None, [0.3]):
        with pytest.raises(ValueError, match="A2"):
            decide(ambient_rmse={"A1": 0.4, "A2": bad})
    with pytest.raises(ValueError, match="strings"):
        decide(ambient_rmse={1: 0.3})
    for bad in (np.nan, 0.0, -1.0, np.inf):
        for method in ("nn1", "nn3"):
            with pytest.raises(ValueError, match=method):
                decide(loco_table(0.5, 1.0, 1.0, extra={method: bad}))
        with pytest.raises(ValueError, match="nn3"):
            decide(ambient_rmse={"nn3": bad})


def test_route_decision_info_defaults_to_an_empty_dict_of_its_own():
    first = impact.RouteDecision("ambient", {"importance_usable": False}, "text")
    second = impact.RouteDecision("ambient", {"importance_usable": False}, "text")
    assert first.info == {} and second.info == {}
    assert first.info is not second.info
    explicit = impact.RouteDecision("ambient", {}, "text", info={"best_ambient": "A1"})
    assert explicit.info == {"best_ambient": "A1"}
    with pytest.raises(AttributeError):
        explicit.info = {}


def test_route_decision_is_immutable():
    decision = decide()
    with pytest.raises(AttributeError):
        decision.primary = "ambient"


def test_the_route_rule_declines_a_target_the_transport_cannot_reach():
    """Three criteria hold but the target lies far outside the sources, with a low ESS."""
    summary = loco_table(0.311, 0.45, 0.50, extra={"nn1": 0.35, "nn3": 0.276})
    support = {
        "supported": False,
        "ess": 1.97,
        "reasons": ["the cost share is 14.8 robust standard deviations from the sources"],
    }
    decision = decide(summary, 32, support=support, ambient_rmse={"A1": 0.4, "A3": 0.323})
    assert decision.primary == "ambient"
    assert decision.passed["importance_usable"] and decision.passed["loco_rmse"]
    assert decision.passed["enough_cases"] and not decision.passed["target_supported"]
    assert "14.8 robust standard deviations" in decision.details


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
    decision = decide()
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
    swapped = impact.route_comparison(tables, decide(usable=False))
    assert swapped["primary"].tolist() == [False] * 4 + [True] * 2
    unsupported = impact.route_comparison(tables, decide(support=UNSUPPORTED))
    assert unsupported["primary"].tolist() == [False] * 4 + [True] * 2


def test_route_comparison_supports_additional_routes_and_checks_the_primary_exists(baseline):
    pp = np.linspace(0.0, 1.0, 101)
    table = impact.impact_table(pp, None, baseline)
    ambient_decision = decide(usable=False)
    every_route = {"meta": table, "ambient": table, "ot_importance": table}
    out = impact.route_comparison(every_route, ambient_decision)
    assert out.groupby("route")["primary"].all().to_dict() == {
        "ambient": True,
        "meta": False,
        "ot_importance": False,
    }
    primary_decision = decide()
    with pytest.raises(ValueError, match="ot_importance"):
        impact.route_comparison({"ambient": table}, primary_decision)
    with pytest.raises(ValueError):
        impact.route_comparison({}, ambient_decision)
    only_primary = impact.route_comparison({"ambient": table}, ambient_decision)
    assert only_primary["primary"].all()


# ----------------------------------------------------------------------------
# select_ambient
# ----------------------------------------------------------------------------
SELECTION_KEYS = {"selected", "qualifying", "ratios", "margin", "runner_up", "close"}


def selection_of_the_notebook(values, ratio=0.95):
    """The selection rule as written in the leave-one-economy-out table."""
    loeo = pd.DataFrame({"rmse": list(values.values())}, index=list(values))
    loeo["ratio_to_A1"] = loeo["rmse"] / loeo.loc["A1", "rmse"]
    loeo["qualifies"] = (loeo.index != "A1") & (loeo["ratio_to_A1"] <= ratio)
    qualifying = loeo[loeo["qualifies"]]
    selected = "A1" if qualifying.empty else str(qualifying["rmse"].idxmin())
    return selected, qualifying.index.tolist(), loeo["ratio_to_A1"].to_dict()


def test_select_ambient_reproduces_the_rule_of_the_notebook_on_random_values():
    rng = np.random.default_rng(15)
    counts = {"A1": 0, "A2": 0, "A3": 0}
    for _ in range(1500):
        a1 = rng.uniform(0.2, 0.6)
        values = {
            "A1": a1,
            "A2": a1 * rng.uniform(0.7, 1.2),
            "A3": a1 * rng.uniform(0.7, 1.2),
        }
        selected, qualifying, ratios = selection_of_the_notebook(values)
        out = impact.select_ambient(values)
        assert out["selected"] == selected
        assert out["qualifying"] == qualifying
        assert out["ratios"] == pytest.approx(ratios, rel=1e-14)
        counts[selected] += 1
    assert min(counts.values()) > 100


def test_select_ambient_edge_cases_of_the_current_rule():
    # the ratio on the line qualifies
    on_the_line = impact.select_ambient({"A1": 2.0, "A2": 1.9, "A3": 2.5})
    assert on_the_line["selected"] == "A2" and on_the_line["qualifying"] == ["A2"]
    assert selection_of_the_notebook({"A1": 2.0, "A2": 1.9, "A3": 2.5})[0] == "A2"
    just_above = impact.select_ambient({"A1": 2.0, "A2": 1.91, "A3": 2.5})
    assert just_above["selected"] == "A1" and just_above["qualifying"] == []
    # ties between qualifiers go to the one listed first
    tie = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.5})
    assert tie["selected"] == "A2" and tie["qualifying"] == ["A2", "A3"]
    assert selection_of_the_notebook({"A1": 1.0, "A2": 0.5, "A3": 0.5})[0] == "A2"
    flipped = impact.select_ambient({"A1": 1.0, "A3": 0.5, "A2": 0.5})
    assert flipped["selected"] == "A3" and flipped["qualifying"] == ["A3", "A2"]
    # the smaller RMSE wins when both qualify
    both = impact.select_ambient({"A1": 1.0, "A2": 0.9, "A3": 0.8})
    assert both["selected"] == "A3" and both["qualifying"] == ["A2", "A3"]
    only_third = impact.select_ambient({"A1": 1.0, "A2": 0.96, "A3": 0.94})
    assert only_third["selected"] == "A3" and only_third["qualifying"] == ["A3"]
    # no qualifier leaves the baseline, even if another estimator has a smaller RMSE
    none = impact.select_ambient({"A1": 1.0, "A2": 0.97, "A3": 1.4})
    assert none["selected"] == "A1" and none["qualifying"] == []
    # the baseline alone
    alone = impact.select_ambient({"A1": 0.7})
    assert alone["selected"] == "A1" and alone["qualifying"] == []
    # a baseline that is missing
    with pytest.raises(ValueError, match="A1"):
        impact.select_ambient({"A2": 0.5, "A3": 0.6})
    with pytest.raises(ValueError, match="Z"):
        impact.select_ambient({"A1": 0.5, "A2": 0.6}, baseline="Z")


def test_select_ambient_compares_with_a_tolerance_for_rounding_noise():
    noisy = impact.select_ambient({"A1": 1.0, "A2": 0.95 * (1.0 + 1e-14)})
    assert noisy["selected"] == "A2"
    clear = impact.select_ambient({"A1": 1.0, "A2": 0.95 * (1.0 + 1e-9)})
    assert clear["selected"] == "A1"
    near_tie = impact.select_ambient({"A1": 1.0, "A2": 0.8, "A3": 0.8 * (1.0 - 1e-14)})
    assert near_tie["selected"] == "A2"
    apart = impact.select_ambient({"A1": 1.0, "A2": 0.8, "A3": 0.8 * (1.0 - 1e-9)})
    assert apart["selected"] == "A3"


def test_select_ambient_reports_margin_runner_up_and_ratios():
    out = impact.select_ambient({"A1": 1.0, "A2": 0.80, "A3": 0.90})
    assert set(out) == SELECTION_KEYS
    assert out["selected"] == "A2" and out["qualifying"] == ["A2", "A3"]
    assert out["runner_up"] == "A3"
    assert out["margin"] == pytest.approx(0.9 / 0.8 - 1.0, rel=1e-12)
    assert out["ratios"] == pytest.approx({"A1": 1.0, "A2": 0.8, "A3": 0.9})
    assert list(out["ratios"]) == ["A1", "A2", "A3"]
    assert out["close"] is False
    assert isinstance(out["selected"], str) and isinstance(out["margin"], float)
    assert isinstance(out["qualifying"], list)
    other = impact.select_ambient({"A1": 1.0, "A2": 0.85, "A3": 0.70})
    assert other["selected"] == "A3" and other["runner_up"] == "A2"
    assert other["margin"] == pytest.approx(0.85 / 0.70 - 1.0, rel=1e-12)
    # the baseline is the runner up when it has the second smallest RMSE
    third = impact.select_ambient({"A1": 1.0, "A2": 0.60, "A3": 1.3})
    assert third["selected"] == "A2" and third["runner_up"] == "A1"
    assert third["margin"] == pytest.approx(1.0 / 0.6 - 1.0, rel=1e-12)


def test_select_ambient_margin_is_zero_without_competitors_and_negative_when_the_baseline_stays():
    alone = impact.select_ambient({"A1": 0.7})
    assert alone["margin"] == 0.0 and alone["runner_up"] is None
    assert alone["close"] is False and alone["ratios"] == {"A1": 1.0}
    stays = impact.select_ambient({"A1": 1.0, "A2": 0.97, "A3": 1.4})
    assert stays["selected"] == "A1" and stays["runner_up"] == "A2"
    assert stays["margin"] == pytest.approx(-0.03, rel=1e-9)
    worse = impact.select_ambient({"A1": 0.5, "A2": 0.9, "A3": 0.8})
    assert worse["selected"] == "A1" and worse["runner_up"] == "A3"
    assert worse["margin"] == pytest.approx(0.6, rel=1e-12)
    tied = impact.select_ambient({"A1": 1.0, "A2": 0.5, "A3": 0.5})
    assert tied["runner_up"] == "A3" and tied["margin"] == 0.0 and tied["close"] is True


def test_select_ambient_flags_a_close_selection():
    # the runner up is within 5 percent of the selected RMSE
    near_runner = impact.select_ambient({"A1": 1.0, "A2": 0.80, "A3": 0.82})
    assert near_runner["close"] is True
    far_runner = impact.select_ambient({"A1": 1.0, "A2": 0.80, "A3": 0.86})
    assert far_runner["close"] is False
    # an estimator within 5 percent of the qualification threshold
    near_limit = impact.select_ambient({"A1": 1.0, "A2": 0.70, "A3": 0.99})
    assert near_limit["selected"] == "A2" and near_limit["close"] is True
    just_inside = impact.select_ambient({"A1": 1.0, "A2": 0.60, "A3": 0.9026})
    assert just_inside["close"] is True
    just_outside = impact.select_ambient({"A1": 1.0, "A2": 0.60, "A3": 0.9024})
    assert just_outside["close"] is False
    below = impact.select_ambient({"A1": 1.0, "A2": 0.60, "A3": 1.0 * 0.95 * 1.049})
    assert below["close"] is True
    # the baseline is not an estimator that is compared with the threshold
    assert impact.select_ambient({"A1": 0.95, "A2": 2.0, "A3": 3.0})["close"] is False
    # the baseline kept with a rival just above the limit
    kept = impact.select_ambient({"A1": 1.0, "A2": 0.96, "A3": 1.5})
    assert kept["selected"] == "A1" and kept["close"] is True
    wide = impact.select_ambient({"A1": 1.0, "A2": 0.80, "A3": 0.82}, close_within=0.01)
    assert wide["close"] is False
    assert impact.select_ambient({"A1": 1.0, "A2": 0.7, "A3": 0.9}, close_within=0.2)["close"]


def test_select_ambient_takes_a_baseline_a_ratio_and_a_series():
    values = {"base": 1.0, "x": 0.6, "y": 0.5}
    out = impact.select_ambient(values, baseline="base")
    assert out["selected"] == "y" and out["qualifying"] == ["x", "y"]
    strict = impact.select_ambient(values, baseline="base", ratio=0.55)
    assert strict["selected"] == "y" and strict["qualifying"] == ["y"]
    nothing = impact.select_ambient(values, baseline="base", ratio=0.4)
    assert nothing["selected"] == "base"
    relaxed = impact.select_ambient({"A1": 1.0, "A2": 1.02}, ratio=1.05)
    assert relaxed["selected"] == "A2"
    from_series = impact.select_ambient(pd.Series({"A1": 0.4, "A2": 0.3, "A3": 0.35}))
    assert from_series["selected"] == "A2"
    from_numpy = impact.select_ambient({"A1": np.float64(0.4), "A2": np.float32(0.3), "A3": 5})
    assert from_numpy["selected"] == "A2"
    assert list(inspect.signature(impact.select_ambient).parameters)[:3] == [
        "rmse",
        "baseline",
        "ratio",
    ]


def test_select_ambient_rejects_invalid_inputs():
    good = {"A1": 0.4, "A2": 0.3}
    for not_a_mapping in (None, [("A1", 0.4)], 0.4, "A1", np.array([0.4, 0.3])):
        with pytest.raises(ValueError, match="rmse"):
            impact.select_ambient(not_a_mapping)
    with pytest.raises(ValueError, match="rmse"):
        impact.select_ambient({})
    for bad in (np.nan, np.inf, 0.0, -0.3, True, "0.3", None):
        with pytest.raises(ValueError, match="A2"):
            impact.select_ambient({"A1": 0.4, "A2": bad})
        with pytest.raises(ValueError, match="A1"):
            impact.select_ambient({"A1": bad, "A2": 0.3})
    with pytest.raises(ValueError, match="strings"):
        impact.select_ambient({1: 0.4, "A1": 0.3})
    for bad_baseline in (None, 1, "a1", ["A1"]):
        with pytest.raises(ValueError, match="baseline"):
            impact.select_ambient(good, baseline=bad_baseline)
    for bad_ratio in (0.0, -0.5, 1.6, np.nan, np.inf, True, "0.95", None):
        with pytest.raises(ValueError, match="ratio"):
            impact.select_ambient(good, ratio=bad_ratio)
    for bad_close in (-0.1, np.nan, np.inf, True, "0.05", None):
        with pytest.raises(ValueError, match="close_within"):
            impact.select_ambient(good, close_within=bad_close)


# ----------------------------------------------------------------------------
# Source text
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("module", [impact, meta])
def test_the_modules_have_no_em_dash_and_no_placeholder_text(module):
    text = Path(module.__file__).read_text(encoding="utf-8")
    assert EM_DASH not in text
    assert "TO" + "DO" not in text and "\r" not in text
