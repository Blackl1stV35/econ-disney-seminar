"""Case catalogue: loading, treatment episodes, feasibility rules, donor pools and case features.

A case is the opening of one attraction in one economy. The catalogue lists the
cases with their opening year and investment. This module reads and validates
the catalogue, merges the openings of one economy that follow each other closely
into treatment episodes (:func:`build_episodes`), decides which cases or episodes
can be analysed with a synthetic control on the global panel, builds the donor
pool of each, and describes the economy at the time of the opening with a fixed
list of numeric features (:data:`FEATURE_COLUMNS`) and the investment of the
first opening (``capex_first_pct_gdp``).

The functions :func:`case_features`, :func:`donor_pool` and
:func:`check_feasibility` accept either the catalogue or the episode table. The
columns that only the episode table has (``next_start_year``,
``member_opening_years``, ``member_investment_usd_bn``,
``is_integrated_resort_any``) are used when present and ignored otherwise.

The features that describe the economy (the levels and the growth slope) are
computed from panel years before the opening year only, so they carry no
information about the effect of the opening. The investment feature
``capex_pct_gdp`` is different for an episode that merges several openings. It is
the total investment of the openings that fall in the post-opening window of the
episode, each divided by the GDP of the year before its own opening. For such an
episode it measures the dose of the whole episode, which is known only after the
later openings and can respond to how the first opening went. The column
``capex_first_pct_gdp``, added after the columns of :data:`FEATURE_COLUMNS`, is
the version that is known before the first opening: the investment of the first
opening alone, divided by the GDP of the year before it. For a single opening the
two coincide. Missing features are imputed from years before the opening year of
the case only (:func:`impute_features`).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CASE_COLUMNS",
    "CATEGORIES",
    "INTEGRATED_CATEGORIES",
    "INVESTMENT_BASES",
    "CATALOGUE_COLUMNS",
    "EPISODE_COLUMNS",
    "FEATURE_COLUMNS",
    "USD_PER_BN",
    "POST_HORIZON",
    "EPISODE_FIRST_YEAR",
    "EPISODE_LAST_FEASIBLE_YEAR",
    "load_cases",
    "resolve_case",
    "resolve_next_start",
    "post_window_end",
    "build_episodes",
    "case_features",
    "economy_feature_cloud",
    "impute_features",
    "outcome_matrix",
    "donor_pool",
    "check_feasibility",
]

#: Columns of the cases catalogue, in file order.
CASE_COLUMNS: tuple[str, ...] = (
    "case_id",
    "attraction",
    "operator",
    "category",
    "iso3",
    "economy",
    "location",
    "opening_date",
    "opening_year",
    "investment_usd_bn_nominal",
    "investment_year_basis",
    "investment_source_note",
    "first_year_attendance_m",
    "concurrent_confounds",
    "in_wdi_panel",
    "panel_feasible",
    "other_openings_same_economy_within_5y",
    "evidence_quality",
    "source_url_1",
    "source_url_2",
    "notes",
)

#: Allowed values of the catalogue column ``category``.
CATEGORIES: tuple[str, ...] = (
    "theme_park",
    "multi_park_resort",
    "integrated_resort",
    "destination_resort",
    "mega_attraction",
)

#: Categories for which the feature ``is_integrated_resort`` equals one.
INTEGRATED_CATEGORIES: tuple[str, ...] = ("integrated_resort", "destination_resort")

#: Allowed values of the column ``investment_basis``.
INVESTMENT_BASES: tuple[str, ...] = ("reported", "fx_converted", "missing")

#: Columns of the table returned by :func:`load_cases`: the catalogue columns, the investment in US dollars that is used and its basis, and the price year of the conversion file with its assumed flag.
CATALOGUE_COLUMNS: tuple[str, ...] = (
    *CASE_COLUMNS,
    "investment_usd_bn_used",
    "investment_basis",
    "price_year",
    "price_year_assumed",
)

#: Columns of the table returned by :func:`build_episodes`.
EPISODE_COLUMNS: tuple[str, ...] = (
    *CATALOGUE_COLUMNS,
    "n_openings",
    "member_case_ids",
    "member_opening_years",
    "member_investment_usd_bn",
    "investment_share_known",
    "next_start_year",
    "is_integrated_resort_any",
)

#: Numeric features that describe an economy at the time of an opening, in order.
FEATURE_COLUMNS: tuple[str, ...] = (
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

#: Number of US dollars in one billion.
USD_PER_BN = 1.0e9

#: Default largest number of post-opening years of an estimation window.
POST_HORIZON = 5

#: First opening year that :func:`build_episodes` merges into episodes and flags as feasible; earlier openings are listed one per row.
EPISODE_FIRST_YEAR = 2000

#: Last start year of an episode that :func:`build_episodes` flags as feasible.
EPISODE_LAST_FEASIBLE_YEAR = 2017

_LEVEL_FEATURES: tuple[str, ...] = (
    "log_gdp_pc",
    "log_pop",
    "receipts_pct_gdp",
    "log_receipts_per_arrival",
    "arrivals_per_capita",
    "air_pax_per_capita",
)
_GROWTH_YEARS = 5
_MIN_GROWTH_POINTS = 3
_PANEL_FEATURE_SOURCES: tuple[str, ...] = (*_LEVEL_FEATURES, "log_receipts", "gdp_usd")
_TEXT_COLUMNS: tuple[str, ...] = (
    "case_id",
    "attraction",
    "operator",
    "category",
    "iso3",
    "economy",
    "location",
    "opening_date",
    "investment_source_note",
    "concurrent_confounds",
    "evidence_quality",
    "source_url_1",
    "source_url_2",
    "notes",
)
_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y"})
_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n"})


# ----------------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------------
def _lines(mask: np.ndarray | pd.Series, index: pd.Index, limit: int = 5) -> list[int]:
    """File line numbers (header is line 1) of the first rows selected by ``mask``."""
    positions = np.flatnonzero(np.asarray(mask))[:limit]
    return [int(index[p]) + 2 for p in positions]


def _numeric_column(
    frame: pd.DataFrame,
    name: str,
    *,
    integer: bool = False,
    required: bool = False,
    minimum: float | None = None,
) -> pd.Series:
    """Parse a text column of the catalogue as float, with blanks as NaN, or raise ValueError."""
    text = frame[name].str.strip()
    blank = (text == "").to_numpy()
    values = pd.to_numeric(text.where(text != ""), errors="coerce").to_numpy(dtype=float)
    unreadable = ~blank & np.isnan(values)
    if unreadable.any():
        raise ValueError(f"column '{name}' must be numeric; bad value(s) {list(text[unreadable].head(3))} on line(s) {_lines(unreadable, frame.index)}")
    if np.isinf(values).any():
        raise ValueError(f"column '{name}' must be finite; line(s) {_lines(np.isinf(values), frame.index)}")
    if required and blank.any():
        raise ValueError(f"column '{name}' may not be empty; line(s) {_lines(blank, frame.index)}")
    if integer:
        fractional = ~np.isnan(values) & (values != np.floor(np.nan_to_num(values)))
        if fractional.any():
            raise ValueError(f"column '{name}' must hold whole numbers; line(s) {_lines(fractional, frame.index)}")
    if minimum is not None:
        below = ~np.isnan(values) & (np.nan_to_num(values, nan=minimum) < minimum)
        if below.any():
            raise ValueError(f"column '{name}' must be at least {minimum}; line(s) {_lines(below, frame.index)}")
    return pd.Series(values, index=frame.index, name=name)


def _flag_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Parse a yes/no style text column as nullable boolean, with blanks as missing."""
    text = frame[name].str.strip().str.lower()
    blank = (text == "").to_numpy()
    is_true = text.isin(_TRUE_TOKENS).to_numpy()
    is_false = text.isin(_FALSE_TOKENS).to_numpy()
    unknown = ~blank & ~is_true & ~is_false
    if unknown.any():
        raise ValueError(f"column '{name}' must hold yes/no values; bad value(s) {list(text[unknown].head(3))} on line(s) {_lines(unknown, frame.index)}")
    out = pd.array(np.where(blank, None, is_true), dtype="boolean")
    return pd.Series(out, index=frame.index, name=name)


def _read_investment_fx(path: str | Path) -> pd.DataFrame:
    """Read the file of converted investment costs.

    Returns a frame indexed by ``case_id`` with the converted cost
    (``investment_usd_bn_fxconv``, float), the price year (``price_year``, float)
    and its assumed flag (``price_year_assumed``, nullable boolean); the last two
    are missing throughout when the file lacks the column.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"investment conversion file not found: {source}")
    try:
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{source.name}: the file cannot be read as csv ({exc})") from exc
    absent = [c for c in ("case_id", "investment_usd_bn_fxconv") if c not in raw.columns]
    if absent:
        raise ValueError(f"{source.name}: required column(s) {absent} are missing; the file has columns {list(raw.columns)}")
    ids = raw["case_id"].str.strip()
    blank = (ids == "").to_numpy()
    if blank.any():
        raise ValueError(f"{source.name}: column 'case_id' may not be empty; line(s) {_lines(blank, raw.index)}")
    repeated = ids.duplicated(keep=False)
    if repeated.any():
        raise ValueError(f"{source.name}: column 'case_id' must be unique; repeated value(s) {sorted(set(ids[repeated]))[:5]}")
    index = pd.Index(ids.to_numpy(), name="case_id")
    try:
        values = _numeric_column(raw, "investment_usd_bn_fxconv", minimum=0.0)
        years = _numeric_column(raw, "price_year", integer=True) if "price_year" in raw.columns else None
        assumed = _flag_column(raw, "price_year_assumed") if "price_year_assumed" in raw.columns else None
    except ValueError as exc:
        raise ValueError(f"{source.name}: {exc}") from exc
    return pd.DataFrame(
        {
            "investment_usd_bn_fxconv": values.to_numpy(dtype=float),
            "price_year": np.full(len(raw), np.nan) if years is None else years.to_numpy(dtype=float),
            "price_year_assumed": pd.array([pd.NA] * len(raw), dtype="boolean") if assumed is None else assumed.array,
        },
        index=index,
    )


def load_cases(path: str | Path, investment_fx_path: str | Path | None = None) -> pd.DataFrame:
    """Read the cases catalogue and validate its schema.

    The file must have exactly the columns listed in :data:`CASE_COLUMNS`. The
    case identifier must be non-empty and unique, the economy code non-empty, the
    category one of :data:`CATEGORIES`, the opening year a whole number, and the
    investment, investment year and first-year attendance numeric (blank means
    missing). The columns ``in_wdi_panel`` and ``panel_feasible`` accept yes/no,
    true/false, y/n and 1/0 values in any letter case.

    The investment in US dollars that the analysis uses is the reported figure
    ``investment_usd_bn_nominal`` where the catalogue has one. Where it is blank
    and the optional conversion file has a figure for the case, the converted
    figure is used instead.

    Parameters
    ----------
    path : str or pathlib.Path
        Csv file of the catalogue.
    investment_fx_path : str or pathlib.Path, optional
        Csv file of investment costs converted to US dollars. It needs the
        columns ``case_id`` and ``investment_usd_bn_fxconv`` (billions of US
        dollars, blank when no conversion exists). The optional columns
        ``price_year`` (whole number, blank when unknown) and
        ``price_year_assumed`` (yes/no flag, blank when unknown) are kept; further
        columns are ignored. The file is joined to the catalogue on ``case_id``;
        rows for cases that are not in the catalogue are ignored.

    Returns
    -------
    pandas.DataFrame
        The catalogue with the columns of :data:`CATALOGUE_COLUMNS` in that
        order: the 21 columns of :data:`CASE_COLUMNS` (text columns as strings
        with blank as missing, ``opening_year`` as int64, the numeric columns as
        float64, the two flags as nullable boolean and
        ``other_openings_same_economy_within_5y`` as float64 when all its values
        are numeric and as text otherwise) followed by
        ``investment_usd_bn_used`` (float64; the reported figure if present, else
        the converted figure, else missing), ``investment_basis`` (``"reported"``,
        ``"fx_converted"`` or ``"missing"``), ``price_year`` (float64) and
        ``price_year_assumed`` (nullable boolean). The last two are the values of
        the conversion file for the case, whichever figure is used, and are
        missing for a case without a row in the file, with a blank value there,
        or without a conversion file. Without a conversion file the basis is
        ``"reported"`` or ``"missing"``.

    Raises
    ------
    FileNotFoundError
        If the catalogue or the conversion file does not exist.
    ValueError
        If a column is missing or unexpected, a value cannot be parsed, a
        case identifier is empty or duplicated, the category is unknown, or the
        conversion file lacks a required column, repeats a case, has a value
        that is not a non-negative number, a price year that is not a whole
        number or a flag that is not yes/no.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"cases catalogue not found: {source}")
    try:
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{source.name}: the file cannot be read as csv ({exc})") from exc

    missing = [c for c in CASE_COLUMNS if c not in raw.columns]
    unexpected = [c for c in raw.columns if c not in CASE_COLUMNS]
    if missing or unexpected:
        raise ValueError(f"{source.name}: column mismatch; missing {missing}, unexpected {unexpected}")
    frame = raw[list(CASE_COLUMNS)].copy()

    out = pd.DataFrame(index=frame.index)
    for column in _TEXT_COLUMNS:
        text = frame[column].str.strip()
        out[column] = text.where(text != "", np.nan)
    for column in ("case_id", "iso3"):
        if out[column].isna().any():
            raise ValueError(f"column '{column}' may not be empty; line(s) {_lines(out[column].isna(), frame.index)}")
    duplicated = out["case_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"column 'case_id' must be unique; repeated value(s) {sorted(set(out.loc[duplicated, 'case_id']))[:5]}")
    bad_category = ~out["category"].isin(CATEGORIES)
    if bad_category.any():
        raise ValueError(f"column 'category' must be one of {list(CATEGORIES)}; bad value(s) {list(out.loc[bad_category, 'category'].head(3))}")

    out["opening_year"] = _numeric_column(frame, "opening_year", integer=True, required=True).astype(np.int64)
    out["investment_usd_bn_nominal"] = _numeric_column(frame, "investment_usd_bn_nominal", minimum=0.0)
    out["investment_year_basis"] = _numeric_column(frame, "investment_year_basis", integer=True)
    out["first_year_attendance_m"] = _numeric_column(frame, "first_year_attendance_m", minimum=0.0)
    out["in_wdi_panel"] = _flag_column(frame, "in_wdi_panel")
    out["panel_feasible"] = _flag_column(frame, "panel_feasible")

    other = frame["other_openings_same_economy_within_5y"].str.strip()
    try:
        out["other_openings_same_economy_within_5y"] = _numeric_column(
            frame, "other_openings_same_economy_within_5y", integer=True, minimum=0.0
        )
    except ValueError:
        out["other_openings_same_economy_within_5y"] = other.where(other != "", np.nan)

    reported = out["investment_usd_bn_nominal"].to_numpy(dtype=float)
    converted = np.full(len(out), np.nan)
    out["price_year"] = np.full(len(out), np.nan)
    out["price_year_assumed"] = pd.Series(pd.array([pd.NA] * len(out), dtype="boolean"), index=out.index)
    if investment_fx_path is not None:
        joined = _read_investment_fx(investment_fx_path).reindex(out["case_id"])
        converted = joined["investment_usd_bn_fxconv"].to_numpy(dtype=float)
        out["price_year"] = joined["price_year"].to_numpy(dtype=float)
        out["price_year_assumed"] = pd.Series(joined["price_year_assumed"].array, index=out.index)
    has_reported = np.isfinite(reported)
    has_converted = ~has_reported & np.isfinite(converted)
    out["investment_usd_bn_used"] = np.where(has_reported, reported, np.where(has_converted, converted, np.nan))
    out["investment_basis"] = pd.Series(
        np.where(has_reported, "reported", np.where(has_converted, "fx_converted", "missing")),
        index=out.index,
        dtype="str",
    )
    return out[list(CATALOGUE_COLUMNS)].reset_index(drop=True)


def resolve_case(case: str | pd.Series | Mapping[str, Any], cases: pd.DataFrame | None = None) -> tuple[str, str, int]:
    """Return the identifier, economy code and opening year of a case.

    Parameters
    ----------
    case : str, pandas.Series or mapping
        Either a case identifier, which is looked up in ``cases``, or a row-like
        object with the keys ``iso3`` and ``opening_year`` (and optionally
        ``case_id``).
    cases : pandas.DataFrame, optional
        Catalogue used to look up a case identifier.

    Returns
    -------
    tuple of (str, str, int)
        ``(case_id, iso3, opening_year)``; ``case_id`` is an empty string when a
        row-like object has no such key.

    Raises
    ------
    ValueError
        If an identifier is not found in the catalogue (or no catalogue is given)
        or a row-like object lacks a required key.
    """
    if isinstance(case, str):
        if cases is None or "case_id" not in cases.columns:
            raise ValueError("a case identifier can only be resolved against a catalogue with a case_id column")
        hit = cases.loc[cases["case_id"] == case]
        if len(hit) != 1:
            raise ValueError(f"case_id '{case}' matches {len(hit)} catalogue rows, expected exactly one")
        row = hit.iloc[0]
    else:
        row = case
    try:
        iso3 = str(row["iso3"])
        year = int(row["opening_year"])
    except KeyError as exc:
        raise ValueError(f"case needs the keys 'iso3' and 'opening_year'; missing {exc}") from exc
    case_id = str(row["case_id"]) if "case_id" in row else ""
    return case_id, iso3, year


# ----------------------------------------------------------------------------
# Treatment episodes
# ----------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    """True for None and for scalar missing values."""
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def _join_text(values: Iterable[Any], separator: str = ";", distinct: bool = False) -> Any:
    """Join the non-missing entries of ``values``; NaN when none is left."""
    parts: list[str] = []
    for value in values:
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text and not (distinct and text in parts):
            parts.append(text)
    return separator.join(parts) if parts else np.nan


def _union_events(values: Iterable[Any]) -> Any:
    """Combine semicolon-separated event lists into one list without repeated events; NaN when empty."""
    events: list[str] = []
    for value in values:
        if _is_missing(value):
            continue
        for part in str(value).split(";"):
            text = part.strip()
            if text and text not in events:
                events.append(text)
    return "; ".join(events) if events else np.nan


def _worst_grade(values: Iterable[Any]) -> Any:
    """The lowest evidence grade (A is best, then B, then C; any other value ranks below C); NaN when none is given."""
    grades = [str(v).strip() for v in values if not _is_missing(v) and str(v).strip()]
    if not grades:
        return np.nan
    order = ("A", "B", "C")
    return max(grades, key=lambda g: (order.index(g.upper()) if g.upper() in order else len(order), g))


def _any_flag(values: Iterable[Any]) -> Any:
    """True if any known flag is true, False if all known flags are false, missing if none is known."""
    known = [bool(v) for v in values if not _is_missing(v)]
    return pd.NA if not known else any(known)


def build_episodes(cases: pd.DataFrame, merge_gap: int = 5) -> pd.DataFrame:
    """Merge the openings of each economy that follow each other closely into treatment episodes.

    Within an economy the openings are sorted by opening year and case
    identifier, so the result does not depend on the order of the rows. From
    :data:`EPISODE_FIRST_YEAR` on, an opening whose year is at most ``merge_gap``
    years after the previous opening of the same economy joins the episode of that
    opening, so a chain of openings can span more than ``merge_gap`` years. Any
    larger gap starts a new episode. Openings before :data:`EPISODE_FIRST_YEAR`
    are never merged, neither with each other nor with later openings: each is
    listed as an episode of its own that is not flagged as feasible. The table
    has one row per episode, sorted by economy code, start year and case
    identifier of the first member.

    Parameters
    ----------
    cases : pandas.DataFrame
        Catalogue from :func:`load_cases` with at least the columns ``case_id``,
        ``iso3`` and ``opening_year``. Other catalogue columns are used when
        present. The investment is read from ``investment_usd_bn_used`` and
        ``investment_basis`` and, when those columns are absent, from
        ``investment_usd_bn_nominal``.
    merge_gap : int, default 5
        Largest number of years between two consecutive openings of one economy
        that still belong to one episode; zero merges only openings in the same
        year.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`EPISODE_COLUMNS`: the 21 catalogue columns, the used
        investment, its basis and the price year, and the episode columns listed
        here.

        ``case_id`` is ``{iso3}_{start year}_ep``; the second and later unmerged
        openings of one economy in the same year before
        :data:`EPISODE_FIRST_YEAR` carry the number 2, 3, ... after ``ep``.
        ``opening_year`` is that of the earliest opening and ``opening_date`` the
        earliest date given among the members that open in that year (missing when
        none is given). ``attraction`` lists the attractions of the members
        separated by semicolons, and ``operator`` and ``location`` list their
        distinct values. ``category`` is the category of the member with the
        largest known investment, or of the first member when no investment is
        known. ``investment_usd_bn_nominal``, ``investment_usd_bn_used`` and
        ``first_year_attendance_m`` are the sums of the known member values
        (missing when none is known). ``investment_year_basis`` and ``price_year``
        are the common value of the members that have one, missing when there is
        none or the values differ. ``price_year_assumed`` is true if any member
        has an assumed price year, false if all known flags are false and missing
        if none is known. ``investment_basis`` lists the distinct bases of the
        known member figures separated by semicolons (``"missing"`` when no figure
        is known). ``concurrent_confounds`` combines the event lists of the
        members without repeats. ``investment_source_note`` and ``notes`` join the
        distinct member texts with a vertical bar. ``in_wdi_panel`` is true if any
        member is in the panel. ``panel_feasible`` is true if any member is
        flagged feasible and the start year lies in
        :data:`EPISODE_FIRST_YEAR` to :data:`EPISODE_LAST_FEASIBLE_YEAR`, false
        otherwise, and missing when no member has a flag. ``evidence_quality`` is
        the lowest grade of the members (A, then B, then C). ``source_url_1`` and
        ``source_url_2`` are the first member values that are present.
        ``other_openings_same_economy_within_5y`` is missing.

        The extra columns are ``n_openings`` (int), ``member_case_ids``,
        ``member_opening_years`` and ``member_investment_usd_bn`` (semicolon-separated
        lists in member order; an entry of the last list is empty when the member
        has no known investment), ``investment_share_known`` (share of the members
        with a known investment), ``next_start_year`` (the first later start year
        of an episode of the economy, NaN for its last episode) and
        ``is_integrated_resort_any`` (int; one if any member has a category in
        :data:`INTEGRATED_CATEGORIES`).

    Raises
    ------
    ValueError
        If ``merge_gap`` is not a non-negative whole number or a required column
        is missing.
    """
    if isinstance(merge_gap, (bool, np.bool_)):
        raise ValueError("merge_gap must be a non-negative whole number")
    try:
        gap = int(merge_gap)
        valid_gap = gap == merge_gap and gap >= 0
    except (TypeError, ValueError):
        valid_gap = False
    if not valid_gap:
        raise ValueError(f"merge_gap must be a non-negative whole number, got {merge_gap!r}")
    absent = [c for c in ("case_id", "iso3", "opening_year") if c not in cases.columns]
    if absent:
        raise ValueError(f"cases lack column(s) {absent}")

    n = len(cases)
    column = {c: (cases[c].tolist() if c in cases.columns else [np.nan] * n) for c in CASE_COLUMNS}
    price_year = cases["price_year"].tolist() if "price_year" in cases.columns else [np.nan] * n
    price_assumed = cases["price_year_assumed"].tolist() if "price_year_assumed" in cases.columns else [pd.NA] * n
    iso3 = [str(v) for v in column["iso3"]]
    identifier = [str(v) for v in column["case_id"]]
    year = [int(v) for v in column["opening_year"]]
    nominal = np.array([np.nan if _is_missing(v) else float(v) for v in column["investment_usd_bn_nominal"]])
    if "investment_usd_bn_used" in cases.columns:
        used = pd.to_numeric(cases["investment_usd_bn_used"], errors="coerce").to_numpy(dtype=float)
    else:
        used = nominal.copy()
    basis = ["reported" if np.isfinite(u) else "missing" for u in used]
    if "investment_basis" in cases.columns:
        basis = [b if not _is_missing(b) and str(b).strip() else default for b, default in zip(cases["investment_basis"].tolist(), basis)]

    order = sorted(range(n), key=lambda i: (iso3[i], year[i], identifier[i], i))
    episodes: list[list[int]] = []
    for i in order:
        tail = episodes[-1][-1] if episodes else None
        merges = (
            tail is not None
            and iso3[tail] == iso3[i]
            and year[tail] >= EPISODE_FIRST_YEAR
            and year[i] >= EPISODE_FIRST_YEAR
            and year[i] - year[tail] <= gap
        )
        if merges:
            episodes[-1].append(i)
        else:
            episodes.append([i])

    rows: list[dict[str, Any]] = []
    issued: dict[str, int] = {}
    for e, members in enumerate(episodes):
        first = members[0]
        start = year[first]
        known = np.isfinite(used[members])
        if known.any():
            pick = members[int(np.argmax(np.where(known, used[members], -np.inf)))]
        else:
            pick = first
        reported = nominal[members][np.isfinite(nominal[members])]
        price_years = {column["investment_year_basis"][i] for i in members if not _is_missing(column["investment_year_basis"][i])}
        conversion_years = {price_year[i] for i in members if not _is_missing(price_year[i])}
        attendance = [float(column["first_year_attendance_m"][i]) for i in members if not _is_missing(column["first_year_attendance_m"][i])]
        flags = _any_flag(column["panel_feasible"][i] for i in members)
        feasible = pd.NA if flags is pd.NA else bool(flags) and EPISODE_FIRST_YEAR <= start <= EPISODE_LAST_FEASIBLE_YEAR
        dates = [str(column["opening_date"][i]).strip() for i in members if year[i] == start and not _is_missing(column["opening_date"][i])]
        following = next(
            (episodes[k][0] for k in range(e + 1, len(episodes)) if iso3[episodes[k][0]] == iso3[first] and year[episodes[k][0]] > start),
            None,
        )
        base_id = f"{iso3[first]}_{start}_ep"
        issued[base_id] = issued.get(base_id, 0) + 1
        rows.append(
            {
                "case_id": base_id if issued[base_id] == 1 else f"{base_id}{issued[base_id]}",
                "attraction": _join_text(column["attraction"][i] for i in members),
                "operator": _join_text((column["operator"][i] for i in members), distinct=True),
                "category": column["category"][pick],
                "iso3": iso3[first],
                "economy": column["economy"][first],
                "location": _join_text((column["location"][i] for i in members), distinct=True),
                "opening_date": min(dates) if dates else np.nan,
                "opening_year": start,
                "investment_usd_bn_nominal": float(reported.sum()) if reported.size else np.nan,
                "investment_year_basis": float(next(iter(price_years))) if len(price_years) == 1 else np.nan,
                "investment_source_note": _join_text((column["investment_source_note"][i] for i in members), " | ", distinct=True),
                "first_year_attendance_m": float(np.sum(attendance)) if attendance else np.nan,
                "concurrent_confounds": _union_events(column["concurrent_confounds"][i] for i in members),
                "in_wdi_panel": _any_flag(column["in_wdi_panel"][i] for i in members),
                "panel_feasible": feasible,
                "other_openings_same_economy_within_5y": np.nan,
                "evidence_quality": _worst_grade(column["evidence_quality"][i] for i in members),
                "source_url_1": next((column["source_url_1"][i] for i in members if not _is_missing(column["source_url_1"][i])), np.nan),
                "source_url_2": next((column["source_url_2"][i] for i in members if not _is_missing(column["source_url_2"][i])), np.nan),
                "notes": _join_text((column["notes"][i] for i in members), " | ", distinct=True),
                "investment_usd_bn_used": float(used[members][known].sum()) if known.any() else np.nan,
                "investment_basis": _join_text((basis[i] for i, k in zip(members, known) if k), distinct=True) if known.any() else "missing",
                "price_year": float(next(iter(conversion_years))) if len(conversion_years) == 1 else np.nan,
                "price_year_assumed": _any_flag(price_assumed[i] for i in members),
                "n_openings": len(members),
                "member_case_ids": ";".join(identifier[i] for i in members),
                "member_opening_years": ";".join(str(year[i]) for i in members),
                "member_investment_usd_bn": ";".join(repr(float(used[i])) if np.isfinite(used[i]) else "" for i in members),
                "investment_share_known": float(known.mean()),
                "next_start_year": np.nan if following is None else float(year[following]),
                "is_integrated_resort_any": int(any(column["category"][i] in INTEGRATED_CATEGORIES for i in members)),
            }
        )

    out = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    text_columns = [c for c in _TEXT_COLUMNS if c in out.columns] + [
        "other_openings_same_economy_within_5y",
        "investment_basis",
        "member_case_ids",
        "member_opening_years",
        "member_investment_usd_bn",
    ]
    for name in text_columns:
        out[name] = pd.Series(out[name].tolist(), index=out.index, dtype="str")
    for name in ("in_wdi_panel", "panel_feasible", "price_year_assumed"):
        out[name] = pd.Series(pd.array(out[name].tolist(), dtype="boolean"), index=out.index)
    for name in ("opening_year", "n_openings", "is_integrated_resort_any"):
        out[name] = out[name].astype(np.int64)
    for name in (
        "investment_usd_bn_nominal",
        "investment_year_basis",
        "first_year_attendance_m",
        "investment_usd_bn_used",
        "price_year",
        "investment_share_known",
        "next_start_year",
    ):
        out[name] = out[name].astype(float)
    return out


def resolve_next_start(case: pd.Series | Mapping[str, Any]) -> int | None:
    """Return the start year of the next episode of the economy of a case, if there is one.

    Parameters
    ----------
    case : pandas.Series or mapping
        Row-like object; the year is read from the key ``next_start_year``.

    Returns
    -------
    int or None
        The year, or None when the key is absent or its value is missing.
    """
    try:
        value = case["next_start_year"]
    except (KeyError, IndexError):
        return None
    if value is None or pd.isna(value):
        return None
    return int(value)


def _validate_horizon(post_horizon: int | None) -> int | None:
    """Return the post horizon as an int or None; raise ValueError unless it is None or a positive whole number."""
    if post_horizon is None:
        return None
    valid = False
    horizon = 0
    if not isinstance(post_horizon, (bool, np.bool_)):
        try:
            horizon = int(post_horizon)
            valid = horizon == post_horizon and horizon >= 1
        except (TypeError, ValueError):
            valid = False
    if not valid:
        raise ValueError(f"post_horizon must be None or a positive whole number, got {post_horizon!r}")
    return horizon


def post_window_end(
    opening_year: int,
    last_year: int,
    post_horizon: int | None = POST_HORIZON,
    next_start_year: float | None = None,
) -> int:
    """Return the last year of the post-opening window of a case or episode.

    The window starts in ``opening_year`` and ends at the earliest of
    ``opening_year + post_horizon - 1``, ``last_year`` and the year before
    ``next_start_year``. A horizon of None and a missing next start year each
    remove their own cap.

    Parameters
    ----------
    opening_year : int
        First year of the window.
    last_year : int
        End of the estimation window.
    post_horizon : int, optional
        Largest number of years from the opening year on; None for no cap.
    next_start_year : float, optional
        Start year of the next episode of the same economy; None or NaN for none.

    Returns
    -------
    int
        The last year of the window. It is smaller than ``opening_year`` when
        ``opening_year`` is after ``last_year``.

    Raises
    ------
    ValueError
        If ``post_horizon`` is neither None nor a positive whole number, or
        ``next_start_year`` is not after ``opening_year``.
    """
    horizon = _validate_horizon(post_horizon)
    opening = int(opening_year)
    end = int(last_year)
    if horizon is not None:
        end = min(end, opening + horizon - 1)
    if next_start_year is not None and not pd.isna(next_start_year):
        following = int(next_start_year)
        if following <= opening:
            raise ValueError(f"next_start_year {following} must be later than opening_year {opening}")
        end = min(end, following - 1)
    return end


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def _check_panel(panel: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ValueError if the panel lacks a column or repeats an (iso3, year) pair."""
    absent = [c for c in ["iso3", "year", *columns] if c not in panel.columns]
    if absent:
        raise ValueError(f"panel lacks column(s) {absent}")
    if panel.duplicated(subset=["iso3", "year"]).any():
        raise ValueError("panel has duplicated (iso3, year) rows")


def _economy_frames(panel: pd.DataFrame, columns: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split the panel into year-indexed frames, one per economy."""
    return {
        str(iso3): group.set_index("year")[list(columns)].sort_index()
        for iso3, group in panel.groupby("iso3", sort=True)
    }


def _window_mean(frame: pd.DataFrame, column: str, first: int, stop: int) -> float:
    """Mean of the finite values of ``column`` in the years ``first`` to ``stop - 1``."""
    values = frame[column].reindex(range(first, stop)).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _slope(years: np.ndarray, values: np.ndarray) -> float:
    """OLS slope of ``values`` on ``years`` over the finite pairs; NaN with too few points."""
    ok = np.isfinite(years) & np.isfinite(values)
    if ok.sum() < _MIN_GROWTH_POINTS:
        return float("nan")
    x = years[ok].astype(float)
    y = values[ok]
    xc = x - x.mean()
    denominator = float(xc @ xc)
    if denominator == 0.0:
        return float("nan")
    return float(xc @ (y - y.mean()) / denominator)


def _trailing_slope(frame: pd.DataFrame, last_year: int) -> float:
    """Slope of log receipts on year over the five years ending in ``last_year``."""
    years = np.arange(last_year - _GROWTH_YEARS + 1, last_year + 1)
    values = frame["log_receipts"].reindex(years).to_numpy(dtype=float)
    return _slope(years, values)


def _capex_pct_gdp(investment_usd_bn: float, gdp_usd: float) -> float:
    """Investment as a percentage of GDP; NaN when either input is missing or GDP is not positive."""
    if investment_usd_bn is None or gdp_usd is None:
        return float("nan")
    inv = float(investment_usd_bn)
    gdp = float(gdp_usd)
    if not (np.isfinite(inv) and np.isfinite(gdp)) or gdp <= 0.0:
        return float("nan")
    return 100.0 * inv * USD_PER_BN / gdp


def _value_at(frame: pd.DataFrame, column: str, year: int) -> float:
    """Value of ``column`` in ``year``, NaN when the year is absent."""
    return float(frame[column].reindex([year]).iloc[0])


def _investment_list(text: Any, count: int) -> list[float]:
    """Parse a semicolon-separated list of ``count`` investments; empty entries and a missing list give NaN."""
    if _is_missing(text):
        return [float("nan")] * count
    parts = str(text).split(";")
    if len(parts) != count:
        raise ValueError(
            f"member_investment_usd_bn must list one investment for each of the {count} openings of member_opening_years, got {text!r}"
        )
    values = []
    for part in parts:
        token = part.strip()
        try:
            values.append(float(token) if token else float("nan"))
        except ValueError as exc:
            raise ValueError(f"member_investment_usd_bn must be a semicolon-separated list of numbers, got {text!r}") from exc
    return values


def _episode_capex(frame: pd.DataFrame, members: Sequence[tuple[int, float]]) -> float:
    """Sum of the investments of the openings as percentages of GDP in the year before each opening.

    Openings without a known investment are skipped. The result is NaN when no
    opening has a known investment or when one that has lacks a positive GDP.
    """
    shares = []
    for year, investment in members:
        if not np.isfinite(investment):
            continue
        share = _capex_pct_gdp(investment, _value_at(frame, "gdp_usd", year - 1))
        if np.isnan(share):
            return float("nan")
        shares.append(share)
    return float(sum(shares)) if shares else float("nan")


def case_features(
    cases: pd.DataFrame,
    panel: pd.DataFrame,
    window: int = 3,
    last_year: int = 2019,
    post_horizon: int | None = POST_HORIZON,
) -> pd.DataFrame:
    """Describe the economy of every case or episode before its opening, and its investment.

    The level features use only panel years from ``opening_year - window`` to
    ``opening_year - 1`` and the growth slope the five years ending in
    ``opening_year - 1``. Levels are means of the non-missing yearly values in the
    window. A case whose economy is not in the panel has missing values for every
    panel-based feature. These features are known before the opening.

    The investment of an episode with several openings enters in two ways.
    ``capex_pct_gdp`` is the total over the openings that fall in the
    post-opening window of the episode of each investment as a percentage of GDP
    in the year before its own opening. The window runs from the opening year to
    the earliest of ``opening_year + post_horizon - 1``, ``last_year`` and the year
    before ``next_start_year`` (see :func:`post_window_end`); the first opening
    always counts. It is the dose of the whole episode, and for an episode with
    later openings it is not known before the first opening: it includes the costs
    of the openings that follow, which are known only later and can respond to how
    the first opening went. ``capex_first_pct_gdp`` is the version that is known
    before the first opening: the investment of the first opening alone as a
    percentage of GDP in the year before it. Openings in the first year of the
    episode all count as the first opening, since none of them follows another.
    Later openings are read from the columns ``member_opening_years`` and
    ``member_investment_usd_bn`` of the episode table; without them every row is a
    single opening, for which ``capex_first_pct_gdp`` equals ``capex_pct_gdp``.

    Parameters
    ----------
    cases : pandas.DataFrame
        Catalogue from :func:`load_cases` or episode table from
        :func:`build_episodes`, with the columns ``case_id``, ``iso3`` and
        ``opening_year``. The investment is read from ``investment_usd_bn_used``
        and, when that column is absent, from ``investment_usd_bn_nominal``. The
        resort indicator is read from ``is_integrated_resort_any`` and, when that
        column is absent, from ``category``.
    panel : pandas.DataFrame
        Panel from :func:`dtt.panel.build_global_panel`.
    window : int, default 3
        Number of pre-opening years averaged for the level features.
    last_year : int, default 2019
        End of the estimation window, which limits the post-opening window of an
        episode.
    post_horizon : int, optional, default 5
        Largest number of post-opening years, counted from the opening year; None
        for no cap.

    Returns
    -------
    pandas.DataFrame
        One row per case with ``case_id``, ``iso3``, ``opening_year``, the
        columns of :data:`FEATURE_COLUMNS` and, after them, ``capex_first_pct_gdp``.
        ``receipts_growth_pre`` is the OLS slope of ``log_receipts`` on year (at
        least three non-missing years required). ``capex_pct_gdp`` is 100 times the
        investment in current US dollars divided by GDP in the year before the
        opening; for an episode it is the sum described above over the openings
        with a known investment, and it is missing when none has one or when one
        that has lacks a positive GDP. ``capex_first_pct_gdp`` is the same
        quantity for the first opening of the episode alone (the openings of its
        first year when several share it): it equals ``capex_pct_gdp`` for a single
        opening and is missing when the investment of the first opening is unknown,
        even if a later opening has a known investment. ``is_integrated_resort`` is
        one for the categories in :data:`INTEGRATED_CATEGORIES` (or when
        ``is_integrated_resort_any`` is one) and zero otherwise.

    Raises
    ------
    ValueError
        If ``window`` is below one, ``post_horizon`` is neither None nor a positive
        whole number, a member list of an episode is malformed, or an input lacks
        a required column.
    """
    if int(window) < 1:
        raise ValueError("window must be at least 1")
    _validate_horizon(post_horizon)
    absent = [c for c in ("case_id", "iso3", "opening_year") if c not in cases.columns]
    if absent:
        raise ValueError(f"cases lack column(s) {absent}")
    invest_column = next((c for c in ("investment_usd_bn_used", "investment_usd_bn_nominal") if c in cases.columns), None)
    if invest_column is None:
        raise ValueError("cases need the column 'investment_usd_bn_used' or 'investment_usd_bn_nominal'")
    if "is_integrated_resort_any" in cases.columns:
        resort = pd.to_numeric(cases["is_integrated_resort_any"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif "category" in cases.columns:
        resort = cases["category"].isin(INTEGRATED_CATEGORIES).to_numpy(dtype=float)
    else:
        raise ValueError("cases need the column 'category' or 'is_integrated_resort_any'")
    _check_panel(panel, _PANEL_FEATURE_SOURCES)
    frames = _economy_frames(panel, _PANEL_FEATURE_SOURCES)

    investment = pd.to_numeric(cases[invest_column], errors="coerce").to_numpy(dtype=float)
    member_years = cases["member_opening_years"].tolist() if "member_opening_years" in cases.columns else None
    member_investments = cases["member_investment_usd_bn"].tolist() if "member_investment_usd_bn" in cases.columns else None
    next_starts = pd.to_numeric(cases["next_start_year"], errors="coerce").to_numpy(dtype=float) if "next_start_year" in cases.columns else None
    rows = []
    for i, (case_id, iso3_raw, opening_raw) in enumerate(zip(cases["case_id"], cases["iso3"], cases["opening_year"])):
        iso3 = str(iso3_raw)
        opening = int(opening_raw)
        row: dict[str, Any] = {"case_id": case_id, "iso3": iso3, "opening_year": opening}
        frame = frames.get(iso3)
        for name in _LEVEL_FEATURES:
            row[name] = float("nan") if frame is None else _window_mean(frame, name, opening - int(window), opening)
        row["receipts_growth_pre"] = float("nan") if frame is None else _trailing_slope(frame, opening - 1)
        openings = [(opening, float(investment[i]))]
        if member_years is not None and member_investments is not None:
            listed = _year_list(member_years[i])
            if listed:
                openings = list(zip(listed, _investment_list(member_investments[i], len(listed))))
        end = post_window_end(opening, int(last_year), post_horizon, None if next_starts is None else next_starts[i])
        counted = [(year, value) for year, value in openings if year <= max(end, opening)]
        first_opening_year = min(year for year, _ in openings)
        first = [(year, value) for year, value in openings if year == first_opening_year]
        row["capex_pct_gdp"] = float("nan") if frame is None else _episode_capex(frame, counted)
        row["is_integrated_resort"] = float(resort[i])
        row["capex_first_pct_gdp"] = float("nan") if frame is None else _episode_capex(frame, first)
        rows.append(row)
    out = pd.DataFrame(rows, columns=["case_id", "iso3", "opening_year", *FEATURE_COLUMNS, "capex_first_pct_gdp"])
    out["opening_year"] = out["opening_year"].astype(np.int64)
    return out


def economy_feature_cloud(
    panel: pd.DataFrame,
    iso3: str,
    years: Iterable[int],
    capex_usd_bn: float | None = None,
    is_integrated_resort: float = 0,
    capex_pct_gdp: float | None = None,
) -> pd.DataFrame:
    """Feature vectors of one economy in a set of years.

    The row of year ``y`` uses a window of one year: levels are the panel values
    of year ``y`` and the growth slope is the OLS slope of ``log_receipts`` over
    the five years ending in ``y``. The row of year ``y`` is therefore the
    description of the economy that :func:`case_features` would give to an opening
    in year ``y + 1`` with a window of one year.

    The feature ``capex_pct_gdp`` can be supplied in two ways, which exclude each
    other. With ``capex_usd_bn`` (billions of current US dollars) it is 100 times
    that cost divided by the GDP of year ``y``, so the share of the cost in GDP
    differs from row to row with the growth of GDP. With ``capex_pct_gdp`` it is
    the same share, in percent, in every row, which is the cloud of a case whose
    cost share is held constant. Without either the feature is missing.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel from :func:`dtt.panel.build_global_panel`.
    iso3 : str
        Economy code.
    years : iterable of int
        Years for which a row is returned.
    capex_usd_bn : float, optional
        Investment in billions of current US dollars, the quantity of the
        catalogue column ``investment_usd_bn_used``. The share of GDP in the
        column ``capex_pct_gdp`` then varies with the GDP of each year.
    is_integrated_resort : float, default 0
        Value of the feature of the same name in every row.
    capex_pct_gdp : float, optional
        Investment as a constant percentage of GDP, used as ``capex_pct_gdp`` in
        every row. Cannot be combined with ``capex_usd_bn``.

    Returns
    -------
    pandas.DataFrame
        One row per year with ``iso3``, ``year`` and the columns of
        :data:`FEATURE_COLUMNS`.

    Raises
    ------
    ValueError
        If both ``capex_usd_bn`` and ``capex_pct_gdp`` are given, the economy is
        not in the panel or the panel lacks a column.
    """
    if capex_usd_bn is not None and capex_pct_gdp is not None:
        raise ValueError("capex_usd_bn and capex_pct_gdp are mutually exclusive: give a cost in US dollars or a constant share of GDP, not both")
    _check_panel(panel, _PANEL_FEATURE_SOURCES)
    sub = panel.loc[panel["iso3"] == iso3]
    if sub.empty:
        raise ValueError(f"economy '{iso3}' is not in the panel")
    frame = sub.set_index("year")[list(_PANEL_FEATURE_SOURCES)].sort_index()
    rows = []
    for year in years:
        y = int(year)
        row: dict[str, Any] = {"iso3": iso3, "year": y}
        for name in _LEVEL_FEATURES:
            row[name] = _value_at(frame, name, y)
        row["receipts_growth_pre"] = _trailing_slope(frame, y)
        if capex_pct_gdp is None:
            row["capex_pct_gdp"] = _capex_pct_gdp(capex_usd_bn, _value_at(frame, "gdp_usd", y))
        else:
            row["capex_pct_gdp"] = float(capex_pct_gdp)
        row["is_integrated_resort"] = float(is_integrated_resort)
        rows.append(row)
    out = pd.DataFrame(rows, columns=["iso3", "year", *FEATURE_COLUMNS])
    out["year"] = out["year"].astype(np.int64)
    return out


def impute_features(
    features: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    max_missing_share: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop features with too many missing values and impute the rest by medians of earlier years.

    Only the columns of ``features`` that belong to :data:`FEATURE_COLUMNS` are
    treated; all other columns are returned unchanged. A feature whose share of
    missing values in ``features`` exceeds ``max_missing_share`` is dropped. A
    missing value of a case is replaced by the median of the same feature over
    the rows of ``reference`` whose year is before the opening year of that case.
    When ``reference`` is None or has no finite value for the feature in those
    years, the median of the feature over the other cases of ``features`` whose
    opening year is not later than that of the case is used. The fill value is
    therefore chosen separately for every case and uses no year from the opening
    year of the case onward. A feature is dropped when a missing value of one of
    its cases cannot be filled this way. No indicator columns are added.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature table, for example from :func:`case_features`. It needs the column
        ``opening_year`` when a retained feature has missing values.
    reference : pandas.DataFrame, optional
        Economy-year feature values from the panel, for example stacked outputs
        of :func:`economy_feature_cloud`; it needs the column ``year`` and a column
        for every feature that has missing values and is not dropped. A row of year
        ``y`` describes the economy at the end of year ``y``.
    max_missing_share : float, default 0.25
        Largest tolerated share of missing values of a feature, in [0, 1].

    Returns
    -------
    imputed : pandas.DataFrame
        Copy of ``features`` without the dropped features and without missing
        values in the retained ones.
    report : pandas.DataFrame
        One row per feature in ``features`` with ``feature``, ``n_missing``,
        ``share_missing``, ``dropped`` (bool), ``reason`` (text for a dropped
        feature, otherwise empty), ``fill_value`` (the mean of the values imputed,
        missing when nothing was imputed or the feature was dropped) and
        ``fill_source`` (``"reference"`` or ``"own"`` when all imputed values come
        from that source, ``"mixed"`` when both are used and ``"none"`` when
        nothing was imputed or the feature was dropped).

    Raises
    ------
    ValueError
        If ``max_missing_share`` is outside [0, 1], ``features`` has no feature
        column, ``features`` lacks ``opening_year`` or ``reference`` lacks ``year``
        or a feature column although values must be imputed.
    """
    if not 0.0 <= float(max_missing_share) <= 1.0:
        raise ValueError("max_missing_share must lie in [0, 1]")
    names = [c for c in features.columns if c in FEATURE_COLUMNS]
    if not names:
        raise ValueError("features has none of the columns in FEATURE_COLUMNS")

    out = features.copy()
    rows = []
    drop: list[str] = []
    openings: np.ndarray | None = None
    for name in names:
        values = features[name].to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        n_missing = int(missing.sum())
        share = n_missing / values.size if values.size else 0.0
        fill = float("nan")
        source = "none"
        reason = ""
        dropped = False
        if share > max_missing_share:
            dropped = True
            reason = f"missing share {share:.3f} exceeds {float(max_missing_share):.3f}"
        elif n_missing > 0:
            if "opening_year" not in features.columns:
                raise ValueError("features needs the column 'opening_year' to impute missing values")
            if openings is None:
                openings = pd.to_numeric(features["opening_year"], errors="coerce").to_numpy(dtype=float)
            ref_years = ref_values = None
            if reference is not None:
                for column in ("year", name):
                    if column not in reference.columns:
                        raise ValueError(f"reference lacks the column '{column}'")
                ref_years = pd.to_numeric(reference["year"], errors="coerce").to_numpy(dtype=float)
                ref_values = reference[name].to_numpy(dtype=float)
            filled = np.full(values.size, np.nan)
            sources: list[str] = []
            for k in np.flatnonzero(missing):
                pool = np.empty(0)
                if ref_values is not None:
                    pool = ref_values[np.isfinite(ref_values) & (ref_years < openings[k])]
                label = "reference"
                if not pool.size:
                    pool = values[~missing & (openings <= openings[k])]
                    label = "own"
                if pool.size:
                    filled[k] = float(np.median(pool))
                    sources.append(label)
            unfilled = int(np.isnan(filled[missing]).sum())
            if unfilled:
                dropped = True
                reason = f"no value from years before the opening available to impute {unfilled} case(s)"
            else:
                out.loc[missing, name] = filled[missing]
                fill = float(np.mean(filled[missing]))
                source = sources[0] if len(set(sources)) == 1 else "mixed"
        if dropped:
            drop.append(name)
        rows.append(
            {
                "feature": name,
                "n_missing": n_missing,
                "share_missing": share,
                "dropped": dropped,
                "reason": reason,
                "fill_value": fill,
                "fill_source": source,
            }
        )
    out = out.drop(columns=drop)
    report = pd.DataFrame(
        rows,
        columns=["feature", "n_missing", "share_missing", "dropped", "reason", "fill_value", "fill_source"],
    )
    return out, report


# ----------------------------------------------------------------------------
# Donor pools and feasibility
# ----------------------------------------------------------------------------
def outcome_matrix(panel: pd.DataFrame, outcome: str, first_year: int, last_year: int) -> pd.DataFrame:
    """Arrange the outcome as a years by economies matrix.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with the columns ``iso3``, ``year`` and ``outcome``.
    outcome : str
        Panel column holding the outcome.
    first_year, last_year : int
        First and last row of the matrix.

    Returns
    -------
    pandas.DataFrame
        One row per year from ``first_year`` to ``last_year`` and one column per
        economy; years absent from the panel and non-finite values are NaN.

    Raises
    ------
    ValueError
        If the year range is empty or the panel lacks a column or repeats a key.
    """
    if int(last_year) < int(first_year):
        raise ValueError(f"empty year range: first_year={first_year}, last_year={last_year}")
    _check_panel(panel, [outcome])
    sub = panel.loc[(panel["year"] >= first_year) & (panel["year"] <= last_year), ["iso3", "year", outcome]]
    wide = sub.pivot(index="year", columns="iso3", values=outcome)
    wide = wide.reindex(range(int(first_year), int(last_year) + 1)).astype(float)
    wide = wide.where(np.isfinite(wide))
    wide.columns.name = None
    return wide


def _year_list(text: Any) -> list[int]:
    """Parse a semicolon-separated list of years; an empty or missing value gives an empty list."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return []
    try:
        return [int(part) for part in str(text).split(";") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"member_opening_years must be a semicolon-separated list of years, got {text!r}") from exc


def _exclusion_openings(cases: pd.DataFrame, catalogue: pd.DataFrame | None) -> pd.DataFrame:
    """Economy codes and years of the openings that exclude an economy from donor pools.

    These are the rows of ``catalogue`` when it is given. Otherwise they are the
    rows of ``cases``, where an episode table contributes every opening listed in
    ``member_opening_years`` instead of its start year alone.
    """
    source = cases if catalogue is None else catalogue
    absent = [c for c in ("iso3", "opening_year") if c not in source.columns]
    if absent:
        name = "cases" if catalogue is None else "catalogue"
        raise ValueError(f"{name} lack column(s) {absent}")
    iso3 = source["iso3"].astype(str).to_numpy()
    years = source["opening_year"].to_numpy()
    if catalogue is None and "member_opening_years" in cases.columns:
        pairs = []
        for code, start, members in zip(iso3, years, cases["member_opening_years"]):
            listed = _year_list(members) or [int(start)]
            pairs.extend((str(code), year) for year in listed)
        return pd.DataFrame(pairs, columns=["iso3", "opening_year"])
    return pd.DataFrame({"iso3": iso3, "opening_year": years.astype(np.int64)})


def _donors_from_wide(
    wide: pd.DataFrame,
    iso3: str,
    opening_year: int,
    openings: pd.DataFrame,
    last_year: int,
    buffer: int,
) -> list[str]:
    """Donor codes for one case given the outcome matrix and the table of exclusion openings."""
    complete = wide.notna().all(axis=0)
    candidates = set(complete.index[complete.to_numpy()])
    near = (openings["opening_year"] >= opening_year - buffer) & (openings["opening_year"] <= last_year)
    excluded = {str(c) for c in openings.loc[near, "iso3"]} | {iso3}
    return sorted(candidates - excluded)


def donor_pool(
    panel: pd.DataFrame,
    case: str | pd.Series | Mapping[str, Any],
    cases: pd.DataFrame,
    outcome: str = "receipts_pct_gdp",
    first_year: int = 1995,
    last_year: int = 2019,
    buffer: int = 5,
    catalogue: pd.DataFrame | None = None,
) -> list[str]:
    """List the economies that can serve as donors for a case or episode.

    A donor has a complete, finite outcome series from ``first_year`` to
    ``last_year``, is not the economy of the case, and has no catalogue opening
    (of any case) from ``opening_year - buffer`` to ``last_year``, where
    ``opening_year`` belongs to the case. The openings that exclude an economy
    are all openings of the catalogue, not only the start years of episodes: they
    are the rows of ``catalogue`` when it is given, and otherwise the openings
    listed in the column ``member_opening_years`` of an episode table, or the
    rows of ``cases`` for a catalogue.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel from :func:`dtt.panel.build_global_panel`.
    case : str, pandas.Series or mapping
        Case identifier (looked up in ``cases``) or a row with ``iso3`` and
        ``opening_year``.
    cases : pandas.DataFrame
        Catalogue or episode table, with the columns ``iso3`` and ``opening_year``.
    outcome : str, default "receipts_pct_gdp"
        Panel column holding the outcome.
    first_year, last_year : int
        Years over which the outcome must be complete.
    buffer : int, default 5
        Years before the opening from which other openings exclude an economy.
    catalogue : pandas.DataFrame, optional
        Full catalogue with the columns ``iso3`` and ``opening_year``, whose rows
        are the openings that exclude economies.

    Returns
    -------
    list of str
        Sorted economy codes.
    """
    openings = _exclusion_openings(cases, catalogue)
    _, iso3, opening = resolve_case(case, cases)
    wide = outcome_matrix(panel, outcome, first_year, last_year)
    return _donors_from_wide(wide, iso3, opening, openings, int(last_year), int(buffer))


def check_feasibility(
    cases: pd.DataFrame,
    panel: pd.DataFrame,
    outcome: str = "receipts_pct_gdp",
    first_year: int = 1995,
    last_year: int = 2019,
    min_pre: int = 5,
    min_post: int = 3,
    min_donors: int = 15,
    buffer: int = 5,
    post_horizon: int | None = POST_HORIZON,
    catalogue: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Decide for every case or episode whether a synthetic control can be estimated.

    A case is feasible when its economy is in the panel, the outcome is observed
    in at least ``min_pre`` years before the opening and at least ``min_post``
    years of the post-opening window (counting years from ``first_year`` on), the
    donor pool has at least ``min_donors`` economies, and no other opening of the
    economy lies in the post-opening window. The post-opening window runs from the
    opening year to the earliest of ``opening_year + post_horizon - 1``,
    ``last_year`` and the year before ``next_start_year`` when the table has that
    column (see :func:`post_window_end`). The other openings are the rows of
    ``catalogue`` when it is given and otherwise the rows of ``cases``; the
    openings listed in ``member_opening_years`` of an episode belong to the
    episode itself and do not count.

    Parameters
    ----------
    cases : pandas.DataFrame
        Catalogue or episode table with the columns ``case_id``, ``iso3`` and
        ``opening_year``, and optionally ``next_start_year``.
    panel : pandas.DataFrame
        Panel from :func:`dtt.panel.build_global_panel`.
    outcome : str, default "receipts_pct_gdp"
        Panel column holding the outcome.
    first_year, last_year : int
        Estimation window.
    min_pre, min_post, min_donors : int
        Smallest numbers of pre-opening years, post-opening years and donors.
    buffer : int, default 5
        Buffer of :func:`donor_pool`.
    post_horizon : int, optional, default 5
        Largest number of post-opening years, counted from the opening year;
        None for no cap.
    catalogue : pandas.DataFrame, optional
        Full catalogue whose openings exclude economies from the donor pools; see
        :func:`donor_pool`.

    Returns
    -------
    pandas.DataFrame
        One row per case with ``case_id``, ``iso3``, ``opening_year``, ``n_pre``
        (years with data from ``first_year`` to the year before the opening),
        ``n_post`` (years with data in the post-opening window), ``window_end``
        (last year of that window; before ``opening_year`` when the opening is
        after ``last_year``), ``n_donors``, ``feasible`` (bool) and ``reason``: an
        empty string for a feasible case and otherwise the failed conditions
        separated by semicolons, each starting with ``n_pre``, ``n_post``,
        ``other_openings`` (followed by the number of other openings in the window
        and their years), ``n_donors`` or ``economy not in panel``.

    Raises
    ------
    ValueError
        If a required column is missing, the year range is empty, or
        ``post_horizon`` is neither None nor a positive whole number.
    """
    absent = [c for c in ("case_id", "iso3", "opening_year") if c not in cases.columns]
    if absent:
        raise ValueError(f"cases lack column(s) {absent}")
    _validate_horizon(post_horizon)
    openings = _exclusion_openings(cases, catalogue)
    wide = outcome_matrix(panel, outcome, first_year, last_year)
    next_starts = cases["next_start_year"].to_numpy(dtype=float) if "next_start_year" in cases.columns else None
    members = cases["member_opening_years"].tolist() if "member_opening_years" in cases.columns else None
    rows = []
    for i, (case_id, iso3_raw, opening_raw) in enumerate(zip(cases["case_id"], cases["iso3"], cases["opening_year"])):
        iso3 = str(iso3_raw)
        opening = int(opening_raw)
        end = post_window_end(opening, int(last_year), post_horizon, None if next_starts is None else next_starts[i])
        reasons: list[str] = []
        if iso3 in wide.columns:
            series = wide[iso3]
            n_pre = int(series.loc[series.index < opening].notna().sum())
            n_post = int(series.loc[(series.index >= opening) & (series.index <= end)].notna().sum())
            if n_pre < min_pre:
                reasons.append(f"n_pre {n_pre} below minimum {min_pre}")
            if n_post < min_post:
                reasons.append(f"n_post {n_post} below minimum {min_post}")
        else:
            n_pre = n_post = 0
            reasons.append("economy not in panel")
        own = (_year_list(members[i]) if members is not None else []) or [opening]
        inside = Counter(int(y) for y in openings.loc[openings["iso3"] == iso3, "opening_year"] if opening <= y <= end)
        others = sorted((inside - Counter(y for y in own if opening <= y <= end)).elements())
        if others:
            reasons.append(f"other_openings {len(others)} in post window (years {', '.join(str(y) for y in others)})")
        n_donors = len(_donors_from_wide(wide, iso3, opening, openings, int(last_year), int(buffer)))
        if n_donors < min_donors:
            reasons.append(f"n_donors {n_donors} below minimum {min_donors}")
        rows.append(
            {
                "case_id": case_id,
                "iso3": iso3,
                "opening_year": opening,
                "n_pre": n_pre,
                "n_post": n_post,
                "window_end": end,
                "n_donors": n_donors,
                "feasible": not reasons,
                "reason": "; ".join(reasons),
            }
        )
    out = pd.DataFrame(
        rows,
        columns=["case_id", "iso3", "opening_year", "n_pre", "n_post", "window_end", "n_donors", "feasible", "reason"],
    )
    out["opening_year"] = out["opening_year"].astype(np.int64)
    out["window_end"] = out["window_end"].astype(np.int64)
    return out
