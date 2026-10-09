"""Tests for dtt.panel: loading and validation of raw files, panel construction, coverage, storage.

All data are simulated by dtt.simulate_global. Single cells, rows and files of a simulated
world are then edited to create the situations under test.
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

from dtt.panel import (  # noqa: E402
    DERIVED_COLUMNS,
    GROUPS,
    INDICATORS,
    OUTCOME_PLAUSIBLE_MAX,
    PANEL_COLUMNS,
    RANGE_ROWS_LISTED,
    RAW_COLUMNS,
    build_global_panel,
    check_value_ranges,
    coverage_report,
    load_wdi,
    read_panel,
    wdi_filename,
    write_panel,
)
from dtt.simulate_global import simulate_global_world  # noqa: E402

N_ECON = 60
FIRST_YEAR = 1995
LAST_YEAR = 2024
N_YEARS = LAST_YEAR - FIRST_YEAR + 1
SMALL_ECON = 24
SMALL_CASES = 3
CODE_OF = {variable: code for code, variable in INDICATORS.items()}
BASE = {"receipts_usd": 1.0e9, "arrivals": 2.0e6, "gdp_usd": 5.0e10, "pop": 1.0e7, "air_pax": 3.0e6}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return simulate_global_world(tmp_path_factory.mktemp("world"), n_econ=N_ECON, seed=1)


@pytest.fixture(scope="module")
def panel(world) -> pd.DataFrame:
    return build_global_panel(world.raw_dir)


@pytest.fixture
def small(tmp_path) -> Path:
    """A fresh 24-economy simulated world; the economies SA00 to SA05 form group G1, SA06 to SA11 group G2, and so on."""
    return simulate_global_world(tmp_path / "raw", n_econ=SMALL_ECON, n_cases=SMALL_CASES, seed=1).raw_dir


# ----------------------------------------------------------------------------
# Helpers that edit the files of a simulated world
# ----------------------------------------------------------------------------
def set_cells(raw: Path, changes: dict) -> None:
    """Overwrite single cells; ``changes`` maps (variable, iso3, year) to a number or None (empty cell)."""
    frames: dict[Path, pd.DataFrame] = {}
    for (variable, iso3, year), value in changes.items():
        hits = 0
        for group in GROUPS:
            path = raw / wdi_filename(group, CODE_OF[variable])
            if path not in frames:
                frames[path] = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame = frames[path]
            mask = (frame["iso3"] == iso3) & (frame["year"] == str(year))
            if mask.any():
                frame.loc[mask, "value"] = "" if value is None else repr(float(value))
                hits += 1
        assert hits == 1, (variable, iso3, year)
    for path, frame in frames.items():
        frame.to_csv(path, index=False)


def row_changes(iso3: str, year: int, **special: float | None) -> dict:
    """Cell changes that give one economy-year the base values, except those in ``special``."""
    values = {**BASE, **special}
    return {(variable, iso3, year): value for variable, value in values.items()}


def series_changes(variable: str, iso3: str, missing_years=()) -> dict:
    """Cell changes that fill a whole series with numbers, leaving ``missing_years`` empty."""
    return {(variable, iso3, y): (None if y in missing_years else 1.0e6 + y) for y in range(FIRST_YEAR, LAST_YEAR + 1)}


def drop_row(raw: Path, iso3: str, year: int) -> None:
    """Remove the rows of one economy-year from every indicator file."""
    for code in INDICATORS:
        for group in GROUPS:
            path = raw / wdi_filename(group, code)
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            keep = ~((frame["iso3"] == iso3) & (frame["year"] == str(year)))
            if not keep.all():
                frame.loc[keep].to_csv(path, index=False)


def drop_economy(raw: Path, iso3: str) -> None:
    """Remove every row of one economy from every indicator file."""
    for code in INDICATORS:
        for group in GROUPS:
            path = raw / wdi_filename(group, code)
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.loc[frame["iso3"] != iso3].to_csv(path, index=False)


def edit_file(path: Path, fn) -> None:
    """Rewrite a text file line by line with ``fn``, which maps a list of lines to a list of lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(fn(lines)) + "\n", encoding="utf-8")


def edit_metadata(raw: Path, fn) -> None:
    """Rewrite the metadata file with ``fn``, which edits a string-valued data frame in place."""
    path = raw / "country_metadata.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    fn(frame)
    frame.to_csv(path, index=False)


# ----------------------------------------------------------------------------
# Panel shape, sorting and derived columns
# ----------------------------------------------------------------------------
def test_panel_has_one_row_per_economy_and_year_in_the_documented_columns(panel):
    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert panel.shape == (N_ECON * N_YEARS, len(PANEL_COLUMNS))
    assert not panel.duplicated(["iso3", "year"]).any()
    assert panel["year"].dtype == np.int64
    assert panel["year"].min() == FIRST_YEAR and panel["year"].max() == LAST_YEAR
    counts = panel.groupby("iso3")["year"].agg(["size", "min", "max"])
    assert (counts["size"] == N_YEARS).all() and (counts["min"] == FIRST_YEAR).all() and (counts["max"] == LAST_YEAR).all()
    for column in list(RAW_COLUMNS) + list(DERIVED_COLUMNS):
        assert panel[column].dtype == np.float64, column


def test_panel_is_sorted_by_iso3_and_year(panel):
    expected = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(panel, expected)
    assert panel.index.equals(pd.RangeIndex(len(panel)))


def test_raw_columns_equal_the_file_contents(world, panel):
    keyed = panel.set_index(["iso3", "year"])
    for group in GROUPS:
        for code, variable in INDICATORS.items():
            with open(world.raw_dir / wdi_filename(group, code), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows[:: max(1, len(rows) // 40)]:
                got = keyed.loc[(row["iso3"], int(row["year"])), variable]
                if row["value"] == "":
                    assert np.isnan(got)
                else:
                    assert got == float(row["value"])


def test_metadata_columns_are_merged(world, panel):
    meta = pd.read_csv(world.paths["metadata"]).set_index("iso3")
    first = panel.groupby("iso3").first()
    assert (first["name"] == meta["name"]).all()
    assert (first["region"] == meta["region"]).all()
    assert (first["income_level"] == meta["income_level"]).all()
    assert panel[["name", "region", "income_level"]].notna().all().all()


def test_derived_columns_follow_their_definitions(panel):
    r, a, g, p, x = (panel[c].to_numpy() for c in ("receipts_usd", "arrivals", "gdp_usd", "pop", "air_pax"))
    close = dict(rtol=1e-12, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(panel["receipts_pct_gdp"], 100.0 * r / g, **close)
    np.testing.assert_allclose(panel["log_receipts"], np.log(r), **close)
    np.testing.assert_allclose(panel["gdp_per_capita_usd"], g / p, **close)
    np.testing.assert_allclose(panel["log_gdp_pc"], np.log(g / p), **close)
    np.testing.assert_allclose(panel["log_pop"], np.log(p), **close)
    np.testing.assert_allclose(panel["log_gdp"], np.log(g), **close)
    np.testing.assert_allclose(panel["arrivals_per_capita"], a / p, **close)
    np.testing.assert_allclose(panel["receipts_per_arrival_usd"], r / a, **close)
    np.testing.assert_allclose(panel["log_receipts_per_arrival"], np.log(r / a), **close)
    np.testing.assert_allclose(panel["air_pax_per_capita"], x / p, **close)
    np.testing.assert_allclose(panel["log_gdp"], panel["log_gdp_pc"] + panel["log_pop"], rtol=1e-12, equal_nan=True)


def test_derived_columns_are_missing_exactly_where_inputs_are_missing(panel):
    assert panel["receipts_pct_gdp"].isna().equals(panel["receipts_usd"].isna() | panel["gdp_usd"].isna())
    assert panel["arrivals_per_capita"].isna().equals(panel["arrivals"].isna() | panel["pop"].isna())
    assert panel["air_pax_per_capita"].isna().equals(panel["air_pax"].isna() | panel["pop"].isna())
    assert panel["log_receipts_per_arrival"].isna().equals(panel["receipts_usd"].isna() | panel["arrivals"].isna())
    assert panel["receipts_pct_gdp"].isna().any() and panel["air_pax_per_capita"].isna().any()
    assert not np.isinf(panel[list(DERIVED_COLUMNS)].to_numpy()).any()


def test_derived_values_by_hand_for_one_economy_year(small):
    set_cells(small, row_changes("SA00", 2000))
    pan = build_global_panel(small)
    meta = pd.read_csv(small / "country_metadata.csv").set_index("iso3")
    row = pan.loc[(pan["iso3"] == "SA00") & (pan["year"] == 2000)].iloc[0]
    assert row["name"] == meta.loc["SA00", "name"] and row["region"] == meta.loc["SA00", "region"]
    assert row["income_level"] == meta.loc["SA00", "income_level"]
    assert row["receipts_pct_gdp"] == pytest.approx(2.0)
    assert row["gdp_per_capita_usd"] == pytest.approx(5000.0)
    assert row["log_gdp_pc"] == pytest.approx(np.log(5000.0))
    assert row["arrivals_per_capita"] == pytest.approx(0.2)
    assert row["receipts_per_arrival_usd"] == pytest.approx(500.0)
    assert row["log_receipts_per_arrival"] == pytest.approx(np.log(500.0))
    assert row["air_pax_per_capita"] == pytest.approx(0.3)
    assert row["log_receipts"] == pytest.approx(np.log(1.0e9))
    assert row["log_pop"] == pytest.approx(np.log(1.0e7)) and row["log_gdp"] == pytest.approx(np.log(5.0e10))


def test_division_by_zero_and_missing_denominators_give_nan_not_inf(small):
    changes: dict = {}
    changes.update(row_changes("SA00", 2001, gdp_usd=0.0))
    changes.update(row_changes("SA01", 2002, pop=0.0))
    changes.update(row_changes("SA06", 2003, arrivals=0.0))
    changes.update(row_changes("SA07", 2001, receipts_usd=0.0))
    changes.update(row_changes("SA12", 2002, arrivals=-5.0))
    changes.update(row_changes("SA13", 2003, gdp_usd=None))
    changes.update(row_changes("SA18", 2004, pop=None))
    changes.update(row_changes("SA19", 2000, arrivals=None))
    set_cells(small, changes)
    pan = build_global_panel(small).set_index(["iso3", "year"])

    def is_nan(key: tuple, column: str) -> bool:
        return bool(np.isnan(pan.loc[key, column]))

    key = ("SA00", 2001)
    assert is_nan(key, "receipts_pct_gdp") and is_nan(key, "log_gdp") and is_nan(key, "log_gdp_pc")
    assert pan.loc[key, "gdp_per_capita_usd"] == 0.0

    key = ("SA01", 2002)
    for column in ("gdp_per_capita_usd", "log_gdp_pc", "arrivals_per_capita", "air_pax_per_capita", "log_pop"):
        assert is_nan(key, column), column
    assert not is_nan(key, "receipts_pct_gdp")

    key = ("SA06", 2003)
    assert is_nan(key, "receipts_per_arrival_usd") and is_nan(key, "log_receipts_per_arrival")
    assert pan.loc[key, "arrivals_per_capita"] == 0.0

    key = ("SA07", 2001)
    assert pan.loc[key, "receipts_pct_gdp"] == 0.0 and pan.loc[key, "receipts_per_arrival_usd"] == 0.0
    assert is_nan(key, "log_receipts") and is_nan(key, "log_receipts_per_arrival")

    key = ("SA12", 2002)
    assert pan.loc[key, "receipts_per_arrival_usd"] < 0.0 and is_nan(key, "log_receipts_per_arrival")

    key = ("SA13", 2003)
    for column in ("receipts_pct_gdp", "gdp_per_capita_usd", "log_gdp_pc", "log_gdp"):
        assert is_nan(key, column), column

    key = ("SA18", 2004)
    for column in ("gdp_per_capita_usd", "arrivals_per_capita", "air_pax_per_capita", "log_pop", "log_gdp_pc"):
        assert is_nan(key, column), column

    key = ("SA19", 2000)
    assert is_nan(key, "receipts_per_arrival_usd") and is_nan(key, "arrivals_per_capita")
    assert not np.isinf(pan[list(DERIVED_COLUMNS)].to_numpy()).any()


def test_missing_rows_are_filled_into_a_complete_grid(small):
    drop_row(small, "SA02", 2002)
    drop_row(small, "SA23", LAST_YEAR)
    pan = build_global_panel(small)
    assert len(pan) == SMALL_ECON * N_YEARS
    for iso3, year in (("SA02", 2002), ("SA23", LAST_YEAR)):
        row = pan.loc[(pan["iso3"] == iso3) & (pan["year"] == year)].iloc[0]
        assert row[list(RAW_COLUMNS)].isna().all() and pd.notna(row["name"])


def test_economy_absent_from_one_indicator_file_still_appears(small):
    edit_file(small / wdi_filename("G1", "IS.AIR.PSGR"), lambda lines: [ln for ln in lines if not ln.startswith("SA00,")])
    pan = build_global_panel(small)
    sub = pan.loc[pan["iso3"] == "SA00"]
    assert len(sub) == N_YEARS
    assert sub["air_pax"].isna().all() and sub["air_pax_per_capita"].isna().all()
    assert sub["gdp_usd"].notna().all() and sub["pop"].notna().all()


def test_year_range_arguments(small):
    narrow = build_global_panel(small, first_year=2001, last_year=2003)
    assert narrow["year"].min() == 2001 and narrow["year"].max() == 2003 and len(narrow) == SMALL_ECON * 3
    wide = build_global_panel(small, first_year=1990, last_year=2030)
    assert len(wide) == SMALL_ECON * 41
    outside = ~wide["year"].between(FIRST_YEAR, LAST_YEAR)
    assert outside.sum() == SMALL_ECON * 11
    assert wide.loc[outside, list(RAW_COLUMNS)].isna().all().all()
    default = build_global_panel(small)
    assert default["year"].min() == FIRST_YEAR and default["year"].max() == LAST_YEAR
    pd.testing.assert_frame_equal(wide.loc[~outside].reset_index(drop=True), default)
    with pytest.raises(ValueError, match="year range"):
        build_global_panel(small, first_year=2004, last_year=2001)


def test_files_with_a_byte_order_mark_are_read(small):
    reference = build_global_panel(small)
    for name in (wdi_filename("G2", "SP.POP.TOTL"), "country_metadata.csv"):
        path = small / name
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    pd.testing.assert_frame_equal(build_global_panel(small), reference)


@pytest.mark.parametrize("token", ["NA", "..", "NaN", "null", "   "])
def test_common_missing_value_tokens_are_read_as_missing(small, token):
    path = small / wdi_filename("G1", "SP.POP.TOTL")
    edit_file(path, lambda lines: [ln if not ln.startswith("SA00,2002,") else f"SA00,2002,{token}" for ln in lines])
    long = load_wdi(small)
    cell = long.loc[(long["variable"] == "pop") & (long["iso3"] == "SA00") & (long["year"] == 2002), "value"]
    assert len(cell) == 1 and np.isnan(cell.iloc[0])


# ----------------------------------------------------------------------------
# load_wdi: layout and validation
# ----------------------------------------------------------------------------
def test_load_wdi_long_format(world):
    long = load_wdi(world.raw_dir)
    assert list(long.columns) == ["iso3", "year", "group", "indicator", "variable", "value"]
    assert len(long) == 20 * (N_ECON // 4) * N_YEARS
    assert set(long["group"]) == set(GROUPS)
    assert set(long["indicator"]) == set(INDICATORS)
    assert dict(zip(long["indicator"], long["variable"])) == dict(INDICATORS)
    assert long["year"].dtype == np.int64 and long["value"].dtype == np.float64
    assert not long.duplicated(["variable", "iso3", "year"]).any()
    assert long["value"].isna().any()


def test_load_wdi_accepts_string_and_path_arguments(small):
    pd.testing.assert_frame_equal(load_wdi(str(small)), load_wdi(small))


def test_missing_directory_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_wdi(tmp_path / "does_not_exist")


def test_missing_file_is_named_in_the_error(small):
    (small / "wdi_G3_SP.POP.TOTL.csv").unlink()
    (small / "wdi_G1_IS.AIR.PSGR.csv").unlink()
    with pytest.raises(ValueError, match="missing") as info:
        load_wdi(small)
    assert "wdi_G3_SP.POP.TOTL.csv" in str(info.value) and "wdi_G1_IS.AIR.PSGR.csv" in str(info.value)
    with pytest.raises(ValueError, match="missing"):
        build_global_panel(small)


def test_missing_column_raises_value_error(small):
    path = small / "wdi_G2_NY.GDP.MKTP.CD.csv"
    edit_file(path, lambda lines: ["iso3,year,val", *lines[1:]])
    with pytest.raises(ValueError, match="value") as info:
        load_wdi(small)
    assert "wdi_G2_NY.GDP.MKTP.CD.csv" in str(info.value)


def test_duplicated_iso3_year_raises_value_error(small):
    path = small / "wdi_G4_ST.INT.ARVL.csv"
    edit_file(path, lambda lines: [*lines, lines[1]])
    with pytest.raises(ValueError, match="duplicated") as info:
        load_wdi(small)
    assert "wdi_G4_ST.INT.ARVL.csv" in str(info.value)


def test_economy_listed_in_two_groups_raises_value_error(small):
    source = (small / wdi_filename("G1", "ST.INT.RCPT.CD")).read_text(encoding="utf-8").splitlines()
    rows = [ln for ln in source if ln.startswith("SA00,")]
    edit_file(small / wdi_filename("G2", "ST.INT.RCPT.CD"), lambda lines: [*lines, *rows])
    with pytest.raises(ValueError, match="more than one group") as info:
        load_wdi(small)
    assert "SA00" in str(info.value)


@pytest.mark.parametrize("group,code", [("G3", "SP.POP.TOTL"), ("G1", "ST.INT.RCPT.CD"), ("G4", "IS.AIR.PSGR")])
def test_a_file_with_a_header_and_no_data_rows_raises_value_error_naming_the_file(small, group, code):
    name = wdi_filename(group, code)
    edit_file(small / name, lambda lines: lines[:1])
    for call in (load_wdi, build_global_panel):
        with pytest.raises(ValueError, match="no data rows") as info:
            call(small)
        assert name in str(info.value)


def test_a_header_only_population_file_is_rejected_and_does_not_leave_a_group_without_population(small):
    reference = build_global_panel(small)
    assert reference["pop"].notna().groupby(reference["iso3"]).any().all()
    name = wdi_filename("G2", "SP.POP.TOTL")
    path = small / name
    edit_file(path, lambda lines: lines[:1])
    with pytest.raises(ValueError, match="no data rows") as info:
        build_global_panel(small)
    assert name in str(info.value)


def test_a_file_whose_values_are_all_empty_still_has_data_rows(small):
    path = small / wdi_filename("G2", "SP.POP.TOTL")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["value"] = ""
    frame.to_csv(path, index=False)
    panel = build_global_panel(small)
    assert panel.loc[panel["iso3"].between("SA06", "SA11"), "pop"].isna().all()
    assert panel.loc[panel["iso3"] == "SA00", "pop"].notna().all()


def test_an_economy_in_two_groups_with_disjoint_years_raises_value_error_listing_it(small):
    rows = ["SA00,2030,5.0", "SA00,2031,6.0"]
    edit_file(small / wdi_filename("G2", "ST.INT.RCPT.CD"), lambda lines: [*lines, *rows])
    with pytest.raises(ValueError, match="more than one group") as info:
        load_wdi(small)
    assert "1 economy code(s)" in str(info.value) and "SA00 (G1, G2)" in str(info.value)
    with pytest.raises(ValueError, match="more than one group"):
        build_global_panel(small)


def test_an_economy_in_two_groups_through_different_indicators_is_listed_with_all_its_groups(small):
    edit_file(small / wdi_filename("G3", "ST.INT.ARVL"), lambda lines: [*lines, "SA07,2001,3.0"])
    edit_file(small / wdi_filename("G4", "IS.AIR.PSGR"), lambda lines: [*lines, "SA07,2002,3.0", "SA01,2002,4.0"])
    with pytest.raises(ValueError, match="more than one group") as info:
        load_wdi(small)
    message = str(info.value)
    assert "2 economy code(s)" in message and "SA01 (G1, G4)" in message and "SA07 (G2, G3, G4)" in message


def test_economies_in_one_group_each_pass_the_group_check(small):
    long = load_wdi(small)
    assert long.groupby("iso3")["group"].nunique().eq(1).all()
    assert set(long["iso3"]) == {f"SA{i:02d}" for i in range(SMALL_ECON)}


@pytest.mark.parametrize(
    "bad_row,pattern",
    [
        ("SA00,2001,abc", "numeric"),
        ("SA00,2001,inf", "finite"),
        ("SA00,2001.5,10", "year"),
        ("SA00,year,10", "year"),
        (",2001,10", "iso3"),
    ],
)
def test_unparseable_cells_raise_value_error(small, bad_row, pattern):
    path = small / "wdi_G1_ST.INT.RCPT.CD.csv"
    edit_file(path, lambda lines: [lines[0], bad_row, *[ln for ln in lines[1:] if not ln.startswith("SA00,2001,")]])
    with pytest.raises(ValueError, match=pattern):
        load_wdi(small)


def test_unreadable_file_raises_value_error(small):
    (small / "wdi_G1_ST.INT.RCPT.CD.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        load_wdi(small)


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def test_missing_metadata_file_raises_value_error(small):
    (small / "country_metadata.csv").unlink()
    with pytest.raises(ValueError, match="metadata"):
        build_global_panel(small)


def test_metadata_without_a_panel_economy_raises_value_error(small):
    edit_file(small / "country_metadata.csv", lambda lines: [ln for ln in lines if not ln.startswith("SA23,")])
    with pytest.raises(ValueError, match="SA23"):
        build_global_panel(small)


def test_metadata_with_missing_column_or_duplicate_code_raises_value_error(small):
    path = small / "country_metadata.csv"
    original = path.read_text(encoding="utf-8")
    edit_file(path, lambda lines: [lines[0].replace("income_level", "income"), *lines[1:]])
    with pytest.raises(ValueError, match="income_level"):
        build_global_panel(small)
    path.write_text(original, encoding="utf-8")
    edit_file(path, lambda lines: [*lines, lines[1]])
    with pytest.raises(ValueError, match="duplicated"):
        build_global_panel(small)


def test_metadata_rows_of_economies_without_data_raise_value_error_listing_the_codes(small):
    edit_file(small / "country_metadata.csv", lambda lines: [*lines, "ZZZ,Not in panel,Region 9,Low income,,0,0", "YYY,Also absent,Region 9,Low income,,0,0"])
    with pytest.raises(ValueError, match="differ") as info:
        build_global_panel(small)
    message = str(info.value)
    assert "country_metadata.csv" in message
    assert "2 missing from the files ['YYY', 'ZZZ']" in message and "0 unexpected" in message


def test_an_economy_in_the_metadata_but_in_no_indicator_file_is_not_dropped_silently(small):
    drop_economy(small, "SA05")
    with pytest.raises(ValueError, match="missing from the files") as info:
        build_global_panel(small)
    assert "1 missing from the files ['SA05']" in str(info.value) and "0 unexpected" in str(info.value)
    edit_file(small / "country_metadata.csv", lambda lines: [ln for ln in lines if not ln.startswith("SA05,")])
    panel = build_global_panel(small)
    assert "SA05" not in set(panel["iso3"]) and panel["iso3"].nunique() == SMALL_ECON - 1


def test_the_error_lists_both_the_missing_and_the_unexpected_economies(small):
    edit_file(small / "country_metadata.csv", lambda lines: [ln for ln in lines if not ln.startswith(("SA23,", "SA22,"))] + ["ZZZ,Not in panel,Region 9,Low income,,0,0"])
    with pytest.raises(ValueError, match="differ") as info:
        build_global_panel(small)
    message = str(info.value)
    assert "1 missing from the files ['ZZZ']" in message and "2 unexpected, not in the metadata ['SA22', 'SA23']" in message


def test_matching_economy_sets_build_the_panel(small):
    panel = build_global_panel(small)
    meta = pd.read_csv(small / "country_metadata.csv", dtype=str, keep_default_na=False)
    assert set(panel["iso3"]) == set(meta["iso3"]) and panel["iso3"].nunique() == SMALL_ECON


# ----------------------------------------------------------------------------
# Plausible ranges of the values
# ----------------------------------------------------------------------------
def test_value_ranges_accept_a_panel_with_missing_values_and_leave_it_unchanged(panel):
    assert panel.isna().any().any() and panel["receipts_pct_gdp"].max() < OUTCOME_PLAUSIBLE_MAX
    before = panel.copy()
    assert check_value_ranges(panel) is None
    pd.testing.assert_frame_equal(panel, before)
    assert (OUTCOME_PLAUSIBLE_MAX, RANGE_ROWS_LISTED) == (300.0, 10)


def test_value_ranges_stop_on_a_negative_raw_value_and_name_economy_year_indicator_and_value(small):
    set_cells(small, {("arrivals", "SA03", 2004): -12.5})
    panel = build_global_panel(small)
    with pytest.raises(ValueError, match="plausible") as info:
        check_value_ranges(panel)
    message = str(info.value)
    assert message.startswith("1 value(s) outside the plausible ranges") and "the first 1 as (economy, year, indicator, value)" in message
    assert message.endswith("(SA03, 2004, arrivals, -12.5)")


def test_value_ranges_stop_on_an_outcome_above_the_maximum_and_name_the_row(small):
    set_cells(small, row_changes("SA08", 2010, receipts_usd=2.0e11))
    panel = build_global_panel(small)
    assert panel.loc[(panel["iso3"] == "SA08") & (panel["year"] == 2010), "receipts_pct_gdp"].iloc[0] == 400.0
    with pytest.raises(ValueError, match="receipts_pct_gdp above 300") as info:
        check_value_ranges(panel)
    assert str(info.value).endswith("(SA08, 2010, receipts_pct_gdp, 400)")
    assert check_value_ranges(panel, outcome_max=400.0) is None
    with pytest.raises(ValueError, match="1 value"):
        check_value_ranges(panel, outcome_max=399.0)


def test_value_ranges_list_the_first_ten_rows_and_the_total_number(small):
    changes = {("arrivals", f"SA{i:02d}", 2001 + (i % 3)): -(i + 1.0) for i in range(14)}
    changes[("receipts_usd", "SA20", 2005)] = -4.0
    changes.update(row_changes("SA21", 2007, receipts_usd=1.6e12))
    set_cells(small, changes)
    panel = build_global_panel(small)
    with pytest.raises(ValueError, match="plausible") as info:
        check_value_ranges(panel)
    message = str(info.value)
    assert message.startswith("16 value(s) outside the plausible ranges") and "the first 10 as" in message
    listed = message.split(": ")[-1].split("; ")
    assert len(listed) == RANGE_ROWS_LISTED
    expected = [f"(SA{i:02d}, {2001 + (i % 3)}, arrivals, {-(i + 1.0):g})" for i in range(10)]
    assert listed == expected
    for left_out in ("SA10", "SA13", "SA20", "SA21"):
        assert left_out not in message


def test_value_ranges_list_the_rows_in_the_order_of_economy_year_and_indicator(small):
    set_cells(small, {("pop", "SA02", 2003): -5.0, ("air_pax", "SA02", 2003): -6.0, ("arrivals", "SA02", 2001): -7.0, ("gdp_usd", "SA01", 2009): -8.0})
    panel = build_global_panel(small)
    with pytest.raises(ValueError) as info:
        check_value_ranges(panel)
    listed = str(info.value).split(": ")[-1].split("; ")
    assert listed == ["(SA01, 2009, gdp_usd, -8)", "(SA02, 2001, arrivals, -7)", "(SA02, 2003, pop, -5)", "(SA02, 2003, air_pax, -6)"]


def test_value_ranges_treat_zero_receipts_as_plausible_and_zero_gdp_or_population_as_not(small):
    set_cells(small, {("receipts_usd", "SA01", 2003): 0.0, ("arrivals", "SA01", 2003): 0.0, ("air_pax", "SA01", 2003): 0.0})
    assert check_value_ranges(build_global_panel(small)) is None
    set_cells(small, {("gdp_usd", "SA01", 2004): 0.0, ("pop", "SA02", 2005): 0.0})
    with pytest.raises(ValueError, match="2 value") as info:
        check_value_ranges(build_global_panel(small))
    assert "(SA01, 2004, gdp_usd, 0)" in str(info.value) and "(SA02, 2005, pop, 0)" in str(info.value)


def test_value_ranges_ignore_missing_values_and_can_use_another_outcome_column(small):
    set_cells(small, {("gdp_usd", "SA00", 2001): None, ("pop", "SA00", 2002): None})
    panel = build_global_panel(small)
    assert check_value_ranges(panel) is None
    top = float(panel["arrivals_per_capita"].max())
    assert check_value_ranges(panel, outcome="arrivals_per_capita", outcome_max=top) is None
    with pytest.raises(ValueError, match="arrivals_per_capita above") as info:
        check_value_ranges(panel, outcome="arrivals_per_capita", outcome_max=0.5 * top)
    assert ", arrivals_per_capita, " in str(info.value)


def test_value_ranges_reject_a_panel_without_the_needed_columns(panel):
    with pytest.raises(ValueError, match="receipts_pct_gdp"):
        check_value_ranges(panel.drop(columns=["receipts_pct_gdp"]))
    with pytest.raises(ValueError, match="nonexistent"):
        check_value_ranges(panel, outcome="nonexistent")


# ----------------------------------------------------------------------------
# Coverage report
# ----------------------------------------------------------------------------
def test_coverage_report_matches_a_brute_force_count(panel):
    report = coverage_report(panel)
    assert list(report.columns) == ["iso3", "indicator", "n_years", "first_year", "last_year", "n_holes", "n_negative"]
    assert len(report) == N_ECON * len(RAW_COLUMNS)
    assert report["iso3"].tolist() == [c for c in sorted(panel["iso3"].unique()) for _ in RAW_COLUMNS]
    assert report["indicator"].tolist() == list(RAW_COLUMNS) * N_ECON
    for iso3, sub in panel.groupby("iso3"):
        for variable in RAW_COLUMNS:
            years = sub.loc[sub[variable].notna(), "year"]
            row = report.loc[(report["iso3"] == iso3) & (report["indicator"] == variable)].iloc[0]
            assert row["n_years"] == len(years)
            if len(years):
                assert row["first_year"] == years.min() and row["last_year"] == years.max()
                assert row["n_holes"] == years.max() - years.min() + 1 - len(years)
            else:
                assert row["n_holes"] == 0
            assert row["n_negative"] == int((sub[variable] < 0).sum())
    assert (report["n_years"] <= N_YEARS).all()
    assert (report["n_years"] < N_YEARS).any()
    assert (report["n_holes"] > 0).any() and (report["n_negative"] == 0).all()
    assert report["n_holes"].dtype == np.int64 and report["n_negative"].dtype == np.int64


def test_coverage_report_in_a_hand_edited_world(small):
    changes: dict = {}
    changes.update(series_changes("arrivals", "SA00", missing_years=range(FIRST_YEAR, LAST_YEAR + 1)))
    changes.update(series_changes("receipts_usd", "SA01", missing_years=(1995, 1996)))
    changes.update(series_changes("gdp_usd", "SA02", missing_years=(2003,)))
    changes.update(series_changes("pop", "SA03"))
    changes.update(series_changes("air_pax", "SA04", missing_years=(1995, 1996, 1997, 2024)))
    set_cells(small, changes)
    report = coverage_report(build_global_panel(small)).set_index(["iso3", "indicator"])
    assert report.loc[("SA00", "arrivals"), "n_years"] == 0
    assert pd.isna(report.loc[("SA00", "arrivals"), "first_year"]) and pd.isna(report.loc[("SA00", "arrivals"), "last_year"])
    assert tuple(report.loc[("SA01", "receipts_usd"), ["n_years", "first_year", "last_year"]]) == (28, 1997, 2024)
    assert tuple(report.loc[("SA02", "gdp_usd"), ["n_years", "first_year", "last_year"]]) == (29, 1995, 2024)
    assert tuple(report.loc[("SA03", "pop"), ["n_years", "first_year", "last_year"]]) == (30, 1995, 2024)
    assert tuple(report.loc[("SA04", "air_pax"), ["n_years", "first_year", "last_year"]]) == (26, 1998, 2023)
    assert str(report["first_year"].dtype) == "Int64"
    assert report.loc[("SA00", "arrivals"), "n_holes"] == 0
    assert report.loc[("SA01", "receipts_usd"), "n_holes"] == 0
    assert report.loc[("SA02", "gdp_usd"), "n_holes"] == 1
    assert report.loc[("SA03", "pop"), "n_holes"] == 0
    assert report.loc[("SA04", "air_pax"), "n_holes"] == 0


def test_years_missing_inside_the_observed_span_are_counted_as_holes(small):
    changes = series_changes("pop", "SA05", missing_years=(2000, 2001, 2010))
    changes.update(series_changes("gdp_usd", "SA06"))
    set_cells(small, changes)
    drop_row(small, "SA06", 2005)
    drop_row(small, "SA06", LAST_YEAR)
    report = coverage_report(build_global_panel(small, first_year=FIRST_YEAR, last_year=LAST_YEAR)).set_index(["iso3", "indicator"])
    assert tuple(report.loc[("SA05", "pop"), ["n_years", "first_year", "last_year", "n_holes"]]) == (27, 1995, 2024, 3)
    assert tuple(report.loc[("SA06", "gdp_usd"), ["n_years", "first_year", "last_year", "n_holes"]]) == (28, 1995, 2023, 1)


def test_a_stray_year_far_outside_the_range_shows_up_as_a_large_number_of_holes(small):
    set_cells(small, series_changes("receipts_usd", "SA00"))
    path = small / wdi_filename("G1", "ST.INT.RCPT.CD")
    edit_file(path, lambda lines: [*lines, "SA00,2204,5.0"])
    panel = build_global_panel(small)
    assert panel["year"].max() == 2204
    row = coverage_report(panel).set_index(["iso3", "indicator"]).loc[("SA00", "receipts_usd")]
    assert (row["n_years"], row["first_year"], row["last_year"]) == (N_YEARS + 1, FIRST_YEAR, 2204)
    assert row["n_holes"] == 2204 - FIRST_YEAR + 1 - (N_YEARS + 1)


def test_negative_values_are_kept_and_counted_per_economy_and_indicator(small):
    set_cells(small, {("gdp_usd", "SA00", 2001): -5.0, ("gdp_usd", "SA00", 2002): -7.0, ("arrivals", "SA12", 2003): -1.0, ("receipts_usd", "SA01", 2003): 0.0})
    panel = build_global_panel(small)
    assert panel.loc[(panel["iso3"] == "SA00") & (panel["year"] == 2001), "gdp_usd"].iloc[0] == -5.0
    report = coverage_report(panel).set_index(["iso3", "indicator"])
    assert report.loc[("SA00", "gdp_usd"), "n_negative"] == 2
    assert report.loc[("SA12", "arrivals"), "n_negative"] == 1
    assert report.loc[("SA01", "receipts_usd"), "n_negative"] == 0
    assert int(report["n_negative"].sum()) == 3


def test_coverage_report_rejects_a_panel_with_repeated_economy_years(panel):
    repeated = pd.concat([panel, panel.iloc[[5]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated") as info:
        coverage_report(repeated)
    assert f"({panel.iloc[5]['iso3']}, {panel.iloc[5]['year']})" in str(info.value)


def test_coverage_report_accepts_derived_indicators_and_rejects_unknown_columns(panel):
    report = coverage_report(panel, indicators=["receipts_pct_gdp", "log_receipts"])
    assert len(report) == N_ECON * 2 and set(report["indicator"]) == {"receipts_pct_gdp", "log_receipts"}
    with pytest.raises(ValueError, match="nonexistent"):
        coverage_report(panel, indicators=["nonexistent"])


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------
def test_write_and_read_panel_round_trip_exactly(panel, tmp_path):
    target = write_panel(panel, tmp_path / "deep" / "dir" / "panel.csv")
    assert target.is_file()
    back = read_panel(target)
    pd.testing.assert_frame_equal(back, panel, check_exact=True)
    assert back["year"].dtype == np.int64
    assert back["receipts_pct_gdp"].isna().sum() == panel["receipts_pct_gdp"].isna().sum()


def test_round_trip_keeps_names_with_commas_and_missing_text(small, tmp_path):
    def edit(frame: pd.DataFrame) -> None:
        frame.loc[frame["iso3"] == "SA01", "name"] = 'Republic of SA01, "The"'
        frame.loc[frame["iso3"] == "SA02", "region"] = ""

    edit_metadata(small, edit)
    pan = build_global_panel(small)
    assert pan.loc[pan["iso3"] == "SA02", "region"].isna().all()
    back = read_panel(write_panel(pan, tmp_path / "small.csv"))
    pd.testing.assert_frame_equal(back, pan, check_exact=True)
    assert (back.loc[back["iso3"] == "SA01", "name"] == 'Republic of SA01, "The"').all()
    assert back.loc[back["iso3"] == "SA02", "region"].isna().all()


def test_read_panel_rejects_repeated_economy_years_and_names_them(panel, tmp_path):
    path = write_panel(panel, tmp_path / "panel.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[7]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated") as info:
        read_panel(path)
    assert f"({panel.iloc[6]['iso3']}, {panel.iloc[6]['year']})" in str(info.value) and "panel.csv" in str(info.value)


def test_read_panel_keeps_missing_rows_and_the_coverage_report_counts_them_as_holes(small, tmp_path):
    set_cells(small, series_changes("pop", "SA03"))
    pan = build_global_panel(small)
    assert coverage_report(pan).set_index(["iso3", "indicator"]).loc[("SA03", "pop"), "n_holes"] == 0
    thinned = pan.loc[~((pan["iso3"] == "SA03") & pan["year"].isin([2005, 2006]))]
    back = read_panel(write_panel(thinned, tmp_path / "thin.csv"))
    assert len(back) == len(pan) - 2
    row = coverage_report(back).set_index(["iso3", "indicator"]).loc[("SA03", "pop")]
    assert (row["n_years"], row["n_holes"]) == (N_YEARS - 2, 2)


def test_read_panel_rejects_a_file_without_panel_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"iso3": ["SA00"], "year": [2000]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        read_panel(path)
