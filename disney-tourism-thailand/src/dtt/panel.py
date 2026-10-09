"""Global country-year panel of international tourism indicators.

The raw inputs are the World Bank indicator files ``wdi_{group}_{indicator}.csv``
(four groups G1 to G4 and five indicators, twenty files in total, each with the
columns ``iso3``, ``year`` and ``value``) and the file ``country_metadata.csv``.
The functions in this module validate and read the files, assemble one row per
economy and year with derived ratios and logarithms, summarise the data coverage
of every economy and indicator (including years missing inside the observed
span and negative values), and write and read the assembled panel. A panel with
two rows for the same economy and year is rejected wherever it is built, read or
summarised.

Checks on the raw files
-----------------------
Every indicator file must hold at least one data row, so that a file with only a
header is an error and not an indicator that is missing for a whole group. An
economy belongs to exactly one group: it may not appear in the files of two
groups, whichever indicators and years the rows cover. The economies of the
indicator files must be those of ``country_metadata.csv``, no more and no fewer.
Each failure raises a :class:`ValueError` that names the file or lists the
economies. Implausible values are not errors of the loader; the function
:func:`check_value_ranges` stops on negative raw values and on an outcome above
its plausible maximum, and names the first offending rows and their number.

Raw indicators
--------------
``receipts_usd``
    International tourism receipts, current US dollars (``ST.INT.RCPT.CD``).
``arrivals``
    International tourism arrivals, persons (``ST.INT.ARVL``).
``gdp_usd``
    Gross domestic product, current US dollars (``NY.GDP.MKTP.CD``).
``pop``
    Population, persons (``SP.POP.TOTL``).
``air_pax``
    Air passengers carried (``IS.AIR.PSGR``).
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GROUPS",
    "INDICATORS",
    "OUTCOME_PLAUSIBLE_MAX",
    "RANGE_ROWS_LISTED",
    "RAW_COLUMNS",
    "DERIVED_COLUMNS",
    "PANEL_COLUMNS",
    "wdi_filename",
    "load_wdi",
    "build_global_panel",
    "coverage_report",
    "check_value_ranges",
    "write_panel",
    "read_panel",
]

#: Group labels of the raw files; together the groups partition the economies.
GROUPS: tuple[str, ...] = ("G1", "G2", "G3", "G4")

#: World Bank indicator code mapped to the column name used in the panel.
INDICATORS: Mapping[str, str] = MappingProxyType(
    {
        "ST.INT.RCPT.CD": "receipts_usd",
        "ST.INT.ARVL": "arrivals",
        "NY.GDP.MKTP.CD": "gdp_usd",
        "SP.POP.TOTL": "pop",
        "IS.AIR.PSGR": "air_pax",
    }
)

RAW_COLUMNS: tuple[str, ...] = tuple(INDICATORS.values())

DERIVED_COLUMNS: tuple[str, ...] = (
    "receipts_pct_gdp",
    "log_receipts",
    "gdp_per_capita_usd",
    "log_gdp_pc",
    "log_pop",
    "log_gdp",
    "arrivals_per_capita",
    "receipts_per_arrival_usd",
    "log_receipts_per_arrival",
    "air_pax_per_capita",
)

_ID_COLUMNS: tuple[str, ...] = ("iso3", "year", "name", "region", "income_level")
PANEL_COLUMNS: tuple[str, ...] = _ID_COLUMNS + RAW_COLUMNS + DERIVED_COLUMNS

#: Largest plausible value of the outcome, in percent of GDP, used by :func:`check_value_ranges`.
OUTCOME_PLAUSIBLE_MAX: float = 300.0

#: Number of offending rows that :func:`check_value_ranges` lists in its error message.
RANGE_ROWS_LISTED: int = 10

_METADATA_FILE = "country_metadata.csv"
_METADATA_REQUIRED: tuple[str, ...] = ("iso3", "name", "region", "income_level")
_NA_TOKENS: tuple[str, ...] = ("", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "..")
_TEXT_COLUMNS: tuple[str, ...] = ("iso3", "name", "region", "income_level")


# ----------------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------------
def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise ratio that is NaN wherever an input is missing or the denominator is zero."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    np.divide(num, den, out=out, where=ok)
    return out


def _safe_log(values: np.ndarray) -> np.ndarray:
    """Element-wise natural logarithm that is NaN wherever the input is missing or not positive."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0.0)
    np.log(x, out=out, where=ok)
    return out


def _describe_rows(mask: np.ndarray | pd.Series, frame: pd.DataFrame, limit: int = 3) -> str:
    """Return a short text listing the first rows of ``frame`` selected by ``mask``."""
    positions = np.flatnonzero(np.asarray(mask))
    shown = [f"line {int(p) + 2} ({frame.iloc[p].to_dict()})" for p in positions[:limit]]
    extra = "" if positions.size <= limit else f" and {positions.size - limit} more"
    return "; ".join(shown) + extra


def _require_unique_keys(panel: pd.DataFrame, source: str) -> None:
    """Raise ValueError if two rows of the panel share an economy code and a year."""
    repeated = panel.duplicated(subset=["iso3", "year"], keep=False).to_numpy()
    if repeated.any():
        pairs = panel.loc[repeated, ["iso3", "year"]].drop_duplicates().head(5)
        listed = ", ".join(f"({r.iso3}, {r.year})" for r in pairs.itertuples())
        raise ValueError(f"{source}: duplicated (iso3, year) rows for {listed}")


# ----------------------------------------------------------------------------
# Raw files
# ----------------------------------------------------------------------------
def wdi_filename(group: str, indicator: str) -> str:
    """Return the file name of the raw indicator file of one group.

    Parameters
    ----------
    group : str
        Group label, one of :data:`GROUPS`.
    indicator : str
        World Bank indicator code, a key of :data:`INDICATORS`.

    Returns
    -------
    str
        The name ``wdi_{group}_{indicator}.csv``.
    """
    return f"wdi_{group}_{indicator}.csv"


def _read_indicator_file(path: Path) -> pd.DataFrame:
    """Read and validate one raw indicator file; return columns iso3, year (int) and value (float)."""
    name = path.name
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name}: the file cannot be read as csv ({exc})") from exc
    absent = [c for c in ("iso3", "year", "value") if c not in raw.columns]
    if absent:
        raise ValueError(f"{name}: required column(s) {absent} are missing; the file has columns {list(raw.columns)}")
    raw = raw[["iso3", "year", "value"]].copy()
    if raw.empty:
        raise ValueError(f"{name}: the file has a header but no data rows")

    iso3 = raw["iso3"].str.strip()
    blank = (iso3 == "").to_numpy()
    if blank.any():
        raise ValueError(f"{name}: empty iso3 code at {_describe_rows(blank, raw)}")

    year_num = pd.to_numeric(raw["year"].str.strip(), errors="coerce").to_numpy(dtype=float)
    bad_year = ~np.isfinite(year_num) | (year_num != np.floor(year_num))
    if bad_year.any():
        raise ValueError(f"{name}: year is not an integer at {_describe_rows(bad_year, raw)}")

    text = raw["value"].str.strip()
    is_na = text.isin(_NA_TOKENS).to_numpy()
    value = pd.to_numeric(text.where(~text.isin(_NA_TOKENS)), errors="coerce").to_numpy(dtype=float)
    unreadable = ~is_na & np.isnan(value)
    if unreadable.any():
        raise ValueError(f"{name}: value is not numeric at {_describe_rows(unreadable, raw)}")
    infinite = np.isinf(value)
    if infinite.any():
        raise ValueError(f"{name}: value is not finite at {_describe_rows(infinite, raw)}")

    out = pd.DataFrame({"iso3": iso3.to_numpy(dtype=object), "year": year_num.astype(np.int64), "value": value})
    out["iso3"] = out["iso3"].astype(str)
    duplicated = out.duplicated(subset=["iso3", "year"], keep=False).to_numpy()
    if duplicated.any():
        raise ValueError(f"{name}: duplicated (iso3, year) pairs at {_describe_rows(duplicated, raw)}")
    return out


def load_wdi(raw_dir: str | Path) -> pd.DataFrame:
    """Read and validate the twenty raw indicator files.

    The directory must hold ``wdi_{group}_{indicator}.csv`` for the four groups
    and the five indicators. Every file must have the columns ``iso3``, ``year``
    and ``value``, at least one data row, an integer year, a numeric or empty
    value, and no duplicated ``(iso3, year)`` pair. An economy belongs to exactly
    one group: it may not appear in the files of two groups, for any indicator
    and any years.

    Parameters
    ----------
    raw_dir : str or pathlib.Path
        Directory containing the raw files.

    Returns
    -------
    pandas.DataFrame
        One row per file row with columns ``iso3``, ``year`` (int), ``group``,
        ``indicator`` (World Bank code), ``variable`` (panel column name) and
        ``value`` (float, NaN where the file is empty), sorted by variable, iso3
        and year.

    Raises
    ------
    ValueError
        If the directory or any file is missing, a file lacks a required column
        or has no data rows (the message names the file), a year or value cannot
        be parsed, a duplicated key is found, or an economy appears in more than
        one group (the message lists the economies and their groups).
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise ValueError(f"raw data directory not found: {root}")
    pairs = [(g, code) for g in GROUPS for code in INDICATORS]
    absent = [wdi_filename(g, code) for g, code in pairs if not (root / wdi_filename(g, code)).is_file()]
    if absent:
        raise ValueError(f"{len(absent)} of {len(pairs)} indicator file(s) missing in {root}: {absent}")

    frames = []
    for group, code in pairs:
        frame = _read_indicator_file(root / wdi_filename(group, code))
        frame["group"] = group
        frame["indicator"] = code
        frame["variable"] = INDICATORS[code]
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)

    memberships = long[["iso3", "group"]].drop_duplicates().groupby("iso3")["group"].agg(lambda s: sorted(s))
    several = memberships[memberships.map(len) > 1]
    if not several.empty:
        listed = "; ".join(f"{iso3} ({', '.join(groups)})" for iso3, groups in several.items())
        raise ValueError(f"{len(several)} economy code(s) appear in more than one group: {listed}")
    order = {name: i for i, name in enumerate(RAW_COLUMNS)}
    long["_order"] = long["variable"].map(order).astype(int)
    long = long.sort_values(["_order", "iso3", "year"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    return long[["iso3", "year", "group", "indicator", "variable", "value"]]


def _read_metadata(path: Path) -> pd.DataFrame:
    """Read and validate the economy metadata file; return iso3, name, region and income_level."""
    if not path.is_file():
        raise ValueError(f"economy metadata file not found: {path}")
    try:
        meta = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name}: the file cannot be read as csv ({exc})") from exc
    absent = [c for c in _METADATA_REQUIRED if c not in meta.columns]
    if absent:
        raise ValueError(f"{path.name}: required column(s) {absent} are missing; the file has columns {list(meta.columns)}")
    meta = meta[list(_METADATA_REQUIRED)].copy()
    for column in _METADATA_REQUIRED:
        meta[column] = meta[column].str.strip()
    dup = meta["iso3"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{path.name}: duplicated iso3 code(s) {sorted(set(meta.loc[dup, 'iso3']))[:5]}")
    for column in ("name", "region", "income_level"):
        meta[column] = meta[column].where(meta[column] != "", np.nan)
    return meta.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------
def _add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    """Append the derived columns to a panel that holds the five raw indicators."""
    receipts = panel["receipts_usd"].to_numpy(dtype=float)
    arrivals = panel["arrivals"].to_numpy(dtype=float)
    gdp = panel["gdp_usd"].to_numpy(dtype=float)
    pop = panel["pop"].to_numpy(dtype=float)
    air = panel["air_pax"].to_numpy(dtype=float)

    gdp_pc = _safe_divide(gdp, pop)
    per_arrival = _safe_divide(receipts, arrivals)
    out = panel.copy()
    out["receipts_pct_gdp"] = 100.0 * _safe_divide(receipts, gdp)
    out["log_receipts"] = _safe_log(receipts)
    out["gdp_per_capita_usd"] = gdp_pc
    out["log_gdp_pc"] = _safe_log(gdp_pc)
    out["log_pop"] = _safe_log(pop)
    out["log_gdp"] = _safe_log(gdp)
    out["arrivals_per_capita"] = _safe_divide(arrivals, pop)
    out["receipts_per_arrival_usd"] = per_arrival
    out["log_receipts_per_arrival"] = _safe_log(per_arrival)
    out["air_pax_per_capita"] = _safe_divide(air, pop)
    return out


def build_global_panel(
    raw_dir: str | Path,
    first_year: int | None = None,
    last_year: int | None = None,
) -> pd.DataFrame:
    """Assemble the country-year panel from the raw World Bank files.

    The panel has one row for every combination of economy and year in the
    requested range, whether or not the files contain data for it. Ratios whose
    denominator is zero or missing are NaN, and logarithms of values that are
    missing or not positive are NaN.

    Parameters
    ----------
    raw_dir : str or pathlib.Path
        Directory with the twenty indicator files and ``country_metadata.csv``.
        The economies of the indicator files and those of the metadata file must
        be the same set.
    first_year, last_year : int, optional
        Range of years of the panel. By default the smallest and largest year
        found in the files are used (1995 and 2024 for the World Bank files).

    Returns
    -------
    pandas.DataFrame
        Columns ``iso3``, ``year``, ``name``, ``region``, ``income_level``, the
        raw indicators ``receipts_usd``, ``arrivals``, ``gdp_usd``, ``pop``,
        ``air_pax`` and the derived columns ``receipts_pct_gdp``
        (100 * receipts_usd / gdp_usd), ``log_receipts``, ``gdp_per_capita_usd``,
        ``log_gdp_pc``, ``log_pop``, ``log_gdp``, ``arrivals_per_capita``,
        ``receipts_per_arrival_usd``, ``log_receipts_per_arrival`` and
        ``air_pax_per_capita``. Rows are sorted by ``iso3`` and ``year``.

    Raises
    ------
    ValueError
        If the raw files fail validation (see :func:`load_wdi`), including a
        repeated economy and year, a file without data rows and an economy in two
        groups, the metadata file is invalid, the economies of the indicator files
        differ from those of the metadata file (the message lists the codes that
        are missing from the files and the codes that are unexpected), or the
        year range is empty. Years missing inside the observed span of an
        indicator and negative values are not errors; see :func:`coverage_report`
        and :func:`check_value_ranges`.
    """
    root = Path(raw_dir)
    long = load_wdi(root)
    meta = _read_metadata(root / _METADATA_FILE)

    y0 = int(long["year"].min()) if first_year is None else int(first_year)
    y1 = int(long["year"].max()) if last_year is None else int(last_year)
    if y1 < y0:
        raise ValueError(f"empty year range: first_year={y0}, last_year={y1}")

    economies = sorted(long["iso3"].unique())
    missing = sorted(set(meta["iso3"]) - set(economies))
    unexpected = sorted(set(economies) - set(meta["iso3"]))
    if missing or unexpected:
        raise ValueError(
            f"the economies of the indicator files differ from those of {_METADATA_FILE}: "
            f"{len(missing)} missing from the files {missing}; {len(unexpected)} unexpected, not in the metadata {unexpected}"
        )

    wide = long.pivot(index=["iso3", "year"], columns="variable", values="value").reindex(columns=list(RAW_COLUMNS))
    grid = pd.MultiIndex.from_product([economies, range(y0, y1 + 1)], names=["iso3", "year"])
    wide = wide.reindex(grid).astype(float)
    panel = wide.reset_index()
    panel["year"] = panel["year"].astype(np.int64)
    panel = panel.merge(meta, on="iso3", how="left", validate="many_to_one")
    panel = _add_derived(panel)
    panel = panel[list(PANEL_COLUMNS)].sort_values(["iso3", "year"], kind="mergesort").reset_index(drop=True)
    _require_unique_keys(panel, "assembled panel")
    return panel


def coverage_report(panel: pd.DataFrame, indicators: Sequence[str] | None = None) -> pd.DataFrame:
    """Summarise the data coverage of every economy and indicator.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with the columns ``iso3``, ``year`` and the requested indicators.
    indicators : sequence of str, optional
        Panel columns to report. Default: the five raw indicators.

    Returns
    -------
    pandas.DataFrame
        One row per economy and indicator, sorted by economy and then in the
        order of ``indicators``, with columns ``iso3``, ``indicator``,
        ``n_years`` (number of years with a non-missing value), ``first_year``
        and ``last_year`` (first and last year with a value, missing when
        ``n_years`` is zero; nullable integers), ``n_holes`` (number of years
        between ``first_year`` and ``last_year`` without a value, whether the row
        of the year is absent or its value is empty; zero when ``n_years`` is zero)
        and ``n_negative`` (number of years with a negative value).

    Raises
    ------
    ValueError
        If a required column is absent from the panel or two rows share an
        economy code and a year.
    """
    columns = list(RAW_COLUMNS) if indicators is None else list(indicators)
    absent = [c for c in ["iso3", "year", *columns] if c not in panel.columns]
    if absent:
        raise ValueError(f"panel lacks column(s) {absent}")
    _require_unique_keys(panel, "panel")
    economies = sorted(panel["iso3"].unique())
    long = panel.melt(id_vars=["iso3", "year"], value_vars=columns, var_name="indicator", value_name="value")
    value = long["value"].to_numpy(dtype=float)
    present = long.loc[np.isfinite(value)]
    summary = present.groupby(["iso3", "indicator"])["year"].agg(["size", "min", "max"])
    negative = long.loc[np.isfinite(value) & (value < 0.0)].groupby(["iso3", "indicator"]).size()
    full = pd.MultiIndex.from_product([economies, columns], names=["iso3", "indicator"])
    summary = summary.reindex(full)
    n_years = summary["size"].fillna(0).astype(np.int64).to_numpy()
    span = (summary["max"] - summary["min"] + 1).fillna(0).astype(np.int64).to_numpy()
    out = pd.DataFrame(
        {
            "n_years": n_years,
            "first_year": pd.array(summary["min"].to_numpy(), dtype="Int64"),
            "last_year": pd.array(summary["max"].to_numpy(), dtype="Int64"),
            "n_holes": span - n_years,
            "n_negative": negative.reindex(full).fillna(0).astype(np.int64).to_numpy(),
        },
        index=full,
    )
    return out.reset_index()


def _range_violations(panel: pd.DataFrame, outcome: str, outcome_max: float) -> pd.DataFrame:
    """Rows of the panel with a value outside its plausible range, one row per offending value.

    The columns are ``iso3``, ``year``, ``indicator`` and ``value``. The rows are
    sorted by economy and year, and within a year by indicator in the order of
    :data:`RAW_COLUMNS` followed by the outcome.
    """
    positive_only = ("gdp_usd", "pop")
    frames = []
    for rank, column in enumerate([*RAW_COLUMNS, outcome]):
        values = panel[column].to_numpy(dtype=float)
        if column == outcome:
            offending = np.isfinite(values) & (values > outcome_max)
        elif column in positive_only:
            offending = np.isfinite(values) & (values <= 0.0)
        else:
            offending = np.isfinite(values) & (values < 0.0)
        if offending.any():
            found = panel.loc[offending, ["iso3", "year"]].copy()
            found["indicator"] = column
            found["value"] = values[offending]
            found["_order"] = rank
            frames.append(found)
    columns = ["iso3", "year", "indicator", "value"]
    if not frames:
        return pd.DataFrame({"iso3": [], "year": [], "indicator": [], "value": []})[columns]
    found = pd.concat(frames, ignore_index=True)
    found = found.sort_values(["iso3", "year", "_order"], kind="mergesort").reset_index(drop=True)
    return found[columns]


def check_value_ranges(
    panel: pd.DataFrame,
    outcome: str = "receipts_pct_gdp",
    outcome_max: float = OUTCOME_PLAUSIBLE_MAX,
) -> None:
    """Stop when the panel holds a negative raw value or an implausibly large outcome.

    A value is offending when ``receipts_usd``, ``arrivals`` or ``air_pax`` is
    negative, ``gdp_usd`` or ``pop`` is not positive, or the outcome exceeds
    ``outcome_max``. Missing values are never offending. The panel is not changed.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel from :func:`build_global_panel` with the columns ``iso3``, ``year``,
        the five raw indicators and the outcome column.
    outcome : str, default "receipts_pct_gdp"
        Panel column holding the outcome, in percent of GDP.
    outcome_max : float, default 300.0
        Largest plausible value of the outcome.

    Raises
    ------
    ValueError
        If a required column is absent from the panel, or if any value is
        offending. The message gives the total number of offending values and
        lists the first :data:`RANGE_ROWS_LISTED` of them in the order of economy,
        year and indicator, each as ``(economy, year, indicator, value)``.
    """
    absent = [c for c in ["iso3", "year", *RAW_COLUMNS, outcome] if c not in panel.columns]
    if absent:
        raise ValueError(f"panel lacks column(s) {absent}")
    found = _range_violations(panel, outcome, float(outcome_max))
    if found.empty:
        return
    shown = found.head(RANGE_ROWS_LISTED)
    listed = "; ".join(f"({r.iso3}, {int(r.year)}, {r.indicator}, {format(r.value, '.10g')})" for r in shown.itertuples())
    raise ValueError(
        f"{len(found)} value(s) outside the plausible ranges (a negative raw value, a GDP or population that is not positive, "
        f"or {outcome} above {outcome_max:g}); the first {len(shown)} as (economy, year, indicator, value): {listed}"
    )


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------
def write_panel(panel: pd.DataFrame, path: str | Path) -> Path:
    """Write the panel to a csv file, creating the parent directory if needed.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel produced by :func:`build_global_panel`.
    path : str or pathlib.Path
        Destination file.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(target, index=False)
    return target


def read_panel(path: str | Path) -> pd.DataFrame:
    """Read a panel written by :func:`write_panel`.

    Parameters
    ----------
    path : str or pathlib.Path
        Csv file with the panel columns.

    Returns
    -------
    pandas.DataFrame
        The panel with text identifiers, integer years and float data, sorted by
        ``iso3`` and ``year``. Floating point values are restored exactly.

    Raises
    ------
    ValueError
        If a panel column is missing from the file or two rows share an economy
        code and a year. Missing years and negative values are not errors; see
        :func:`coverage_report`.
    """
    source = Path(path)
    header = pd.read_csv(source, nrows=0)
    absent = [c for c in PANEL_COLUMNS if c not in header.columns]
    if absent:
        raise ValueError(f"{source.name}: panel column(s) {absent} are missing")
    dtypes: dict[str, object] = {c: str for c in _TEXT_COLUMNS}
    dtypes["year"] = np.int64
    panel = pd.read_csv(source, dtype=dtypes, float_precision="round_trip")
    for column in list(RAW_COLUMNS) + list(DERIVED_COLUMNS):
        panel[column] = panel[column].astype(float)
    panel = panel[list(PANEL_COLUMNS)]
    _require_unique_keys(panel, source.name)
    return panel.sort_values(["iso3", "year"], kind="mergesort").reset_index(drop=True)
