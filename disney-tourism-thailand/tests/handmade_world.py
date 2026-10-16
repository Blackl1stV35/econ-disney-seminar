"""Hand-built world in the file format of the raw World Bank data, for the tests of the panel, the cases and the effects.

:func:`build_handmade_world` writes, into a directory, the twenty indicator
files ``wdi_{group}_{indicator}.csv``, the file ``country_metadata.csv`` and a
cases catalogue ``cases_catalogue.csv`` with the 21-column schema of
:data:`dtt.cases.CASE_COLUMNS`. The economies are called ``SA00`` to ``SA23``
and form the groups G1 (``SA00`` to ``SA05``), G2 (``SA06`` to ``SA11``), G3
(``SA12`` to ``SA17``) and G4 (``SA18`` to ``SA23``). Every value is a literal or
a closed-form arithmetic expression of the economy index and the year index.
No random numbers are drawn, so the world is the same on every run.

Untreated outcome
-----------------
The untreated outcome ``receipts_pct_gdp`` of economy ``i`` in year ``t`` is

``y0[i, t] = a[i] + b[i] * f[t] + c[i] * g[t]``

with two common factors, the linear trend ``f[t] = 0.04 * k`` and the periodic
pattern ``g[t] = 0.5 * ((k mod 5) - 2)``, where ``k = t - 1995``. The loadings
``a``, ``b`` and ``c`` are closed-form functions of the economy index: with
``s = -1 + 0.1 * index`` they are ``3 + s``, ``0.7 + s**2`` and ``0.3 + s**3``.
These points lie on a curve in three dimensions, so no never-treated economy
lies in the convex hull of the others, and the placebo fits of the never-treated
economies are inexact. An economy that hosts a case has the loadings of a convex
combination of never-treated economies, so its untreated outcome is an exact
convex combination of their outcomes in every year; a synthetic control built
from them therefore reproduces its untreated path and recovers its level shift
exactly. The economy ``SA16`` of the default world carries in addition a
residual of plus or minus 0.03 points in its pre-opening years, so that its
synthetic control does not fit exactly.

Treatment
---------
From its opening year a case economy receives the level shift

``tau = B0 + B1 * capex_pct_gdp + B2 * receipts_pct_gdp_pre``

in percentage points of GDP, with ``B0 = 0.30``, ``B1 = 0.85`` and ``B2 = 0``.
Here ``capex_pct_gdp`` is the catalogue investment as a percentage of GDP in the
year before the opening, and ``receipts_pct_gdp_pre`` is the mean outcome in the
three years before the opening, as :func:`dtt.cases.case_features` defines them
for a single opening. The shifts of the openings of one economy add up from each
opening year on. Outside the treated economies the outcome has no level shift,
so the true effect of a case is its own shift.

Cases
-----
The default world holds six cases and the cluster world eight. Their identifiers,
economies, opening years and investments are listed in ``_DEFAULT_CASES`` and
``_CLUSTER_CASES``. In the default world ``C01`` (economy ``SA03``, 1997) opens
too early for the pre-opening rule, ``C06`` (economy ``SA22``, 2024) opens after
the last year of the analysis window, ``C02`` (``SA08``, 2001), ``C03``
(``SA14``, 2008) and ``C05`` (``SA16``, 2016) are feasible with complete outcome
years, and ``C04`` (``SA20``, 2013) is feasible although its outcome is missing
in 1995 and 1996. In the cluster world ``SA08`` hosts three openings (2001, 2004
and 2007) and ``SA14`` hosts two (2008 and 2013); ``C01`` and ``C08`` are built
to fail as in the default world, and ``C07`` (``SA16``, 2016) is a single feasible
opening.

Missing values
--------------
Three economies have gaps in the outcome series: ``SA05`` in 2003, ``SA17`` in
1999 and 2010, and ``SA20`` in 1995 and 1996. The other indicators have a few
scattered gaps, and the air passenger series of ``SA08`` is missing for its first
twelve years. The economy ``SA00`` has no gap in any indicator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dtt.cases import CASE_COLUMNS, CATEGORIES
from dtt.panel import GROUPS, INDICATORS, wdi_filename

__all__ = ["HandmadeWorld", "build_handmade_world", "CASES_FILE", "N_ECON", "FIRST_YEAR", "LAST_YEAR"]

#: Number of economies, named ``SA00`` to ``SA23``.
N_ECON = 24

#: First and last year of the panel.
FIRST_YEAR = 1995
LAST_YEAR = 2024

#: Last year of the default estimation window; relative effects are measured up to it.
ANALYSIS_LAST_YEAR = 2019

#: Name of the cases catalogue written into the world directory.
CASES_FILE = "cases_catalogue.csv"

_METADATA_FILE = "country_metadata.csv"
_B0 = 0.30
_B1 = 0.85
_B2 = 0.0
_REGIONS = (
    "East Asia & Pacific",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East & North Africa",
    "North America",
    "South Asia",
    "Sub-Saharan Africa",
)
_INCOME_LEVELS = ("High income", "Upper middle income", "Lower middle income", "Low income")

# Case specifications: (case_id, economy index, opening year, category, investment in USD bn, first-year attendance in millions).
_DEFAULT_CASES: tuple[tuple[str, int, int, str, float, float], ...] = (
    ("C01", 3, 1997, "theme_park", 0.25, 2.10),
    ("C02", 8, 2001, "integrated_resort", 0.60, 5.40),
    ("C03", 14, 2008, "multi_park_resort", 0.90, 7.20),
    ("C04", 20, 2013, "destination_resort", 0.40, 3.30),
    ("C05", 16, 2016, "mega_attraction", 1.10, 8.80),
    ("C06", 22, 2024, "theme_park", 0.30, 1.90),
)
_CLUSTER_CASES: tuple[tuple[str, int, int, str, float, float], ...] = (
    ("C01", 3, 1997, "theme_park", 0.25, 2.10),
    ("C02", 8, 2001, "integrated_resort", 0.60, 5.40),
    ("C03", 8, 2004, "multi_park_resort", 0.35, 4.00),
    ("C04", 8, 2007, "destination_resort", 0.45, 3.80),
    ("C05", 14, 2008, "mega_attraction", 0.90, 7.20),
    ("C06", 14, 2013, "theme_park", 0.50, 6.10),
    ("C07", 16, 2016, "multi_park_resort", 1.10, 8.80),
    ("C08", 22, 2024, "theme_park", 0.30, 1.90),
)
_DEFAULT_FAILING = ("C01", "C06")
_CLUSTER_FAILING = ("C01", "C08")

# Never-treated, complete economies whose convex combinations give the loadings of the treated economies.
_MIX: dict[int, tuple[tuple[int, float], ...]] = {
    3: ((0, 0.5), (1, 0.3), (2, 0.2)),
    8: ((4, 0.4), (6, 0.35), (7, 0.25)),
    14: ((9, 0.5), (10, 0.3), (11, 0.2)),
    16: ((12, 0.45), (13, 0.35), (15, 0.2)),
    20: ((18, 0.6), (19, 0.4)),
    22: ((21, 0.5), (23, 0.3), (0, 0.2)),
}
_RESIDUAL_ECON = 16
_RESIDUAL_BEFORE_YEAR = 2016
_RESIDUAL_SIZE = 0.03

# Missing values as (raw variable, economy index, years without a value).
_GAPS: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("receipts_usd", 5, (2003,)),
    ("receipts_usd", 17, (1999, 2010)),
    ("receipts_usd", 20, (1995, 1996)),
    ("arrivals", 2, (1995, 1996, 1997)),
    ("arrivals", 11, (1995, 1996, 1997)),
    ("arrivals", 9, (2012,)),
    ("arrivals", 19, (2005,)),
    ("air_pax", 8, tuple(range(1995, 2007))),
    ("air_pax", 4, (2005,)),
    ("air_pax", 13, (2015,)),
)


@dataclass(eq=False)
class HandmadeWorld:
    """A hand-built world written to disk, with the level shifts of its cases.

    Attributes
    ----------
    directory : pathlib.Path
        Directory holding the raw files and the cases catalogue.
    paths : dict of str to pathlib.Path
        Every written file, keyed by file name, plus the keys ``raw_dir``,
        ``cases`` and ``metadata``.
    true_effects : pandas.DataFrame
        One row per case with ``case_id``, ``true_att_pp`` (the level shift of the
        case in percentage points of GDP) and ``true_att_rel_pct`` (the shift as a
        percentage of the mean untreated outcome from the opening year to the end
        of the analysis window).
    drivers : pandas.DataFrame
        One row per case with ``case_id``, ``iso3``, ``opening_year``,
        ``investment_usd_bn_nominal``, ``capex_pct_gdp``, ``receipts_pct_gdp_pre``
        and ``effect_noise``; the level shift equals
        ``b0 + b1 * capex_pct_gdp + b2 * receipts_pct_gdp_pre``.
    coefficients : dict of str to float
        Coefficients ``b0``, ``b1``, ``b2`` and ``effect_noise_sd`` of the level-shift rule.
    counterfactual : pandas.DataFrame
        Untreated outcome ``receipts_pct_gdp_cf`` for every economy and year.
    first_year, last_year, n_econ : int
        Years of the panel and number of economies.
    infeasible_case_ids : list of str
        Cases built to fail the default feasibility rules.
    cluster_economies : bool
        Whether several cases share an economy.
    """

    directory: Path
    paths: dict[str, Path]
    true_effects: pd.DataFrame
    drivers: pd.DataFrame
    coefficients: dict[str, float]
    counterfactual: pd.DataFrame
    first_year: int
    last_year: int
    n_econ: int
    infeasible_case_ids: list[str]
    cluster_economies: bool = False

    @property
    def raw_dir(self) -> Path:
        """Directory with the raw indicator files."""
        return self.directory

    @property
    def cases_path(self) -> Path:
        """Path of the cases catalogue."""
        return self.paths["cases"]

    def window_effect(self, iso3: str, first_year: int, last_year: int) -> float:
        """Return the mean level shift of one economy over a range of years.

        The level shift of a year is the sum of ``true_att_pp`` over the cases of
        the economy that opened in that year or earlier.

        Parameters
        ----------
        iso3 : str
            Economy code.
        first_year, last_year : int
            First and last year of the range, both included.

        Returns
        -------
        float
            The mean shift over the range; zero for an economy without cases.

        Raises
        ------
        ValueError
            If ``last_year`` is before ``first_year``.
        """
        if int(last_year) < int(first_year):
            raise ValueError(f"empty year range: first_year={first_year}, last_year={last_year}")
        years = np.arange(int(first_year), int(last_year) + 1)
        merged = self.drivers.loc[self.drivers["iso3"] == iso3, ["case_id", "opening_year"]].merge(
            self.true_effects[["case_id", "true_att_pp"]], on="case_id", validate="one_to_one"
        )
        shift = np.zeros(years.size)
        for opening, effect in zip(merged["opening_year"], merged["true_att_pp"]):
            shift += np.where(years >= int(opening), float(effect), 0.0)
        return float(shift.mean())


def _own_loadings(index: int) -> np.ndarray:
    """Loadings on the constant, the trend and the periodic factor of an economy with the given index.

    The loadings of the economies lie on a curve in three dimensions, so that no
    economy of the index set lies in the convex hull of the others; the treated
    economies are convex combinations of other economies (see ``_MIX``).
    """
    s = -1.0 + 0.1 * index
    return np.array([3.0 + s, 0.7 + s**2, 0.3 + s**3])


def _set_gaps(data: dict[str, np.ndarray]) -> None:
    """Blank the cells listed in ``_GAPS``."""
    for variable, econ, years in _GAPS:
        for year in years:
            data[variable][econ, year - FIRST_YEAR] = np.nan


def build_handmade_world(directory: str | Path, cluster_economies: bool = False) -> HandmadeWorld:
    """Write the hand-built world in the format of the raw files and return its known effects.

    Parameters
    ----------
    directory : str or pathlib.Path
        Output directory, created if needed. It receives the twenty files
        ``wdi_{group}_{indicator}.csv``, ``country_metadata.csv`` and
        ``cases_catalogue.csv``.
    cluster_economies : bool, default False
        If true, the cluster world is written: economy ``SA08`` hosts three
        openings and economy ``SA14`` hosts two, with openings a few years apart.
        Otherwise the default world is written, with one case per treated economy.

    Returns
    -------
    HandmadeWorld
        Paths of the written files, the level shifts of the cases, the drivers of
        the shifts, the coefficients of the rule and the untreated outcome.

    Raises
    ------
    ValueError
        If the output directory cannot be created as a directory.
    """
    specs = _CLUSTER_CASES if cluster_economies else _DEFAULT_CASES
    failing = _CLUSTER_FAILING if cluster_economies else _DEFAULT_FAILING
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = LAST_YEAR - FIRST_YEAR + 1
    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    k = np.arange(T, dtype=float)
    econ_index = np.arange(N_ECON, dtype=float)
    codes = np.array([f"SA{i:02d}" for i in range(N_ECON)])

    # ---- untreated outcome: two common factors with loadings ------------------------------------
    own = np.array([_own_loadings(i) for i in range(N_ECON)])
    loadings = own.copy()
    for econ in sorted({spec[1] for spec in specs}):
        loadings[econ] = sum(weight * own[donor] for donor, weight in _MIX[econ])
    trend = 0.04 * k
    periodic = 0.5 * ((np.arange(T) % 5) - 2.0)
    y0 = loadings[:, [0]] + loadings[:, [1]] * trend[None, :] + loadings[:, [2]] * periodic[None, :]
    if not cluster_economies:
        residual = np.where(k < _RESIDUAL_BEFORE_YEAR - FIRST_YEAR, _RESIDUAL_SIZE * np.where(k % 2 == 0, 1.0, -1.0), 0.0)
        y0[_RESIDUAL_ECON] += residual

    # ---- size variables -------------------------------------------------------------------------
    gdp = np.round((1.0e10 + 4.0e9 * econ_index)[:, None] * (1.04 + 0.002 * econ_index)[:, None] ** k[None, :])
    pop = np.round((4.0e5 + 6.0e6 * econ_index)[:, None] * (1.005 + 0.0004 * econ_index)[:, None] ** k[None, :])
    per_arrival = (420.0 + 35.0 * econ_index)[:, None] * 1.02 ** k[None, :]
    air_rate = (0.05 + 0.01 * (np.arange(N_ECON) % 7))[:, None]
    air = np.round(pop * air_rate * (1.0 + 0.01 * k[None, :]))

    # ---- cases: investment, level shift, and the outcome with the shifts ------------------------
    y_obs = y0.copy()
    rows = []
    for case_id, econ, opening, category, investment, attendance in specs:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        col = opening - FIRST_YEAR
        capex = 100.0 * investment * 1.0e9 / gdp[econ, col - 1]
        pre_cols = [c for c in range(col - 3, col) if c >= 0]
        pre_level = float(np.mean(y_obs[econ, pre_cols]))
        tau = _B0 + _B1 * capex + _B2 * pre_level
        y_obs[econ, col:] += tau
        rel_end = ANALYSIS_LAST_YEAR if opening <= ANALYSIS_LAST_YEAR else LAST_YEAR
        rel_cols = list(range(col, rel_end - FIRST_YEAR + 1))
        rows.append(
            {
                "case_id": case_id,
                "econ": econ,
                "category": category,
                "attendance": attendance,
                "iso3": str(codes[econ]),
                "opening_year": opening,
                "investment_usd_bn_nominal": investment,
                "capex_pct_gdp": capex,
                "receipts_pct_gdp_pre": pre_level,
                "effect_noise": 0.0,
                "true_att_pp": tau,
                "true_att_rel_pct": 100.0 * tau / float(np.mean(y0[econ, rel_cols])),
            }
        )
    drivers_all = pd.DataFrame(rows)

    # ---- raw indicators and missing values ------------------------------------------------------
    receipts = y_obs / 100.0 * gdp
    data = {
        "receipts_usd": np.round(receipts),
        "arrivals": np.round(receipts / per_arrival),
        "gdp_usd": gdp,
        "pop": pop,
        "air_pax": air,
    }
    _set_gaps(data)

    # ---- write the files ------------------------------------------------------------------------
    paths: dict[str, Path] = {}
    for group, members in zip(GROUPS, np.array_split(np.arange(N_ECON), len(GROUPS))):
        for code, variable in INDICATORS.items():
            frame = pd.DataFrame(
                {
                    "iso3": np.repeat(codes[members], T),
                    "year": np.tile(years, members.size),
                    "value": data[variable][members].reshape(-1),
                }
            )
            target = out_dir / wdi_filename(group, code)
            frame.to_csv(target, index=False)
            paths[target.name] = target

    metadata = pd.DataFrame(
        {
            "iso3": codes,
            "name": [f"Fixture economy {c}" for c in codes],
            "region": [_REGIONS[i % len(_REGIONS)] for i in range(N_ECON)],
            "income_level": [_INCOME_LEVELS[(3 * i + 1) % len(_INCOME_LEVELS)] for i in range(N_ECON)],
            "capital": [f"Fixture capital {c}" for c in codes],
            "longitude": [round(-170.0 + 340.0 * i / 23.0, 3) for i in range(N_ECON)],
            "latitude": [round(-45.0 + 105.0 * ((7 * i) % 24) / 23.0, 3) for i in range(N_ECON)],
        }
    )
    metadata_path = out_dir / _METADATA_FILE
    metadata.to_csv(metadata_path, index=False)
    paths[metadata_path.name] = metadata_path

    names = dict(zip(metadata["iso3"], metadata["name"]))
    if cluster_economies:
        overlap = [
            ";".join(
                str(other_id)
                for other_id, other_iso3, other_year in zip(drivers_all["case_id"], drivers_all["iso3"], drivers_all["opening_year"])
                if other_id != case_id and other_iso3 == iso3 and abs(int(other_year) - int(year)) <= 5
            )
            for case_id, iso3, year in zip(drivers_all["case_id"], drivers_all["iso3"], drivers_all["opening_year"])
        ]
    else:
        overlap = [0] * len(drivers_all)
    catalogue = pd.DataFrame(
        {
            "case_id": drivers_all["case_id"],
            "attraction": [f"Fixture attraction {k + 1:02d}" for k in range(len(rows))],
            "operator": [f"Fixture operator {'ABCDE'[k % 5]}" for k in range(len(rows))],
            "category": [row["category"] for row in rows],
            "iso3": drivers_all["iso3"],
            "economy": [names[c] for c in drivers_all["iso3"]],
            "location": [f"Fixture city {c}" for c in drivers_all["iso3"]],
            "opening_date": [
                f"{int(o)}-{(5 * k) % 12 + 1:02d}-{(7 * k) % 28 + 1:02d}"
                for k, o in enumerate(drivers_all["opening_year"])
            ],
            "opening_year": drivers_all["opening_year"],
            "investment_usd_bn_nominal": drivers_all["investment_usd_bn_nominal"],
            "investment_year_basis": drivers_all["opening_year"] - 1,
            "investment_source_note": "Fixture figure",
            "first_year_attendance_m": [row["attendance"] for row in rows],
            "concurrent_confounds": "none",
            "in_wdi_panel": "yes",
            "panel_feasible": ["no" if case_id in failing else "yes" for case_id in drivers_all["case_id"]],
            "other_openings_same_economy_within_5y": overlap,
            "evidence_quality": "high",
            "source_url_1": [f"https://example.org/fixture/{c}" for c in drivers_all["case_id"]],
            "source_url_2": "",
            "notes": "Fixture case",
        }
    )[list(CASE_COLUMNS)]
    cases_path = out_dir / CASES_FILE
    catalogue.to_csv(cases_path, index=False)
    paths[cases_path.name] = cases_path
    paths["raw_dir"] = out_dir
    paths["cases"] = cases_path
    paths["metadata"] = metadata_path

    counterfactual = pd.DataFrame(
        {
            "iso3": np.repeat(codes, T),
            "year": np.tile(years, N_ECON),
            "receipts_pct_gdp_cf": y0.reshape(-1),
        }
    )
    return HandmadeWorld(
        directory=out_dir,
        paths=paths,
        true_effects=drivers_all[["case_id", "true_att_pp", "true_att_rel_pct"]].copy(),
        drivers=drivers_all[
            ["case_id", "iso3", "opening_year", "investment_usd_bn_nominal", "capex_pct_gdp", "receipts_pct_gdp_pre", "effect_noise"]
        ].copy(),
        coefficients={"b0": _B0, "b1": _B1, "b2": _B2, "effect_noise_sd": 0.0},
        counterfactual=counterfactual,
        first_year=FIRST_YEAR,
        last_year=LAST_YEAR,
        n_econ=N_ECON,
        infeasible_case_ids=[str(c) for c in drivers_all.loc[drivers_all["case_id"].isin(failing), "case_id"]],
        cluster_economies=bool(cluster_economies),
    )
