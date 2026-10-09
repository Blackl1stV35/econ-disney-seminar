"""Synthetic test worlds in the file format of the raw World Bank data.

:func:`simulate_global_world` writes, into a directory, the twenty indicator
files ``wdi_{group}_{indicator}.csv``, the file ``country_metadata.csv`` and a
cases catalogue ``cases_catalogue.csv`` with the 21-column schema of
:data:`dtt.cases.CASE_COLUMNS`. The economies are called ``SA00``, ``SA01`` and so
on. The output is synthetic test data.

Data-generating process
-----------------------
The untreated outcome ``receipts_pct_gdp`` of economy ``e`` in year ``t`` is

``y[e, t] = mu[e] + lambda[e] . f[t] + u[e, t]``

with three latent factors ``f`` (a trending AR(1) process, an AR(1) process and a
random walk), economy-specific loadings, and AR(1) noise ``u``. The loadings of
every case economy are a convex combination of the loadings of five other
economies. From its opening year a case economy receives the level shift

``tau = b0 + b1 * capex_pct_gdp + b2 * receipts_pct_gdp_pre + noise``

in percentage points of GDP, with ``b1`` non-zero and ``b2`` zero. Here
``capex_pct_gdp`` is the catalogue investment as a percentage of GDP in the year
before the opening and ``receipts_pct_gdp_pre`` is the mean outcome in the three
years before the opening, exactly as :func:`dtt.cases.case_features` defines them
for a single opening. For a treatment episode that merges several openings, the
column ``capex_first_pct_gdp`` of :func:`dtt.cases.case_features` is the driver
``capex_pct_gdp`` of its first opening.

Design of the cases
-------------------
By default every case sits in its own economy. For ``n_cases`` of at least eight,
the catalogue contains one case that opens two years after the first year of the
panel (two pre-opening years) and, from sixteen cases on, one case that opens in
the last year of the panel (no post-opening year in the default estimation
window). All other cases satisfy the default feasibility rules of
:func:`dtt.cases.check_feasibility` as long as every case sits in its own
economy. One feasible case has its first two outcome years missing, a few other
economies have gaps in the outcome series, and the other indicators have
scattered missing values.

With ``cluster_economies=True`` five of the feasible cases without missing outcome
years are moved into two shared economies: one economy hosts three openings, with
gaps of two to four years, and another hosts two openings, with a gap of three to
five years. The level shifts of the openings of one economy add up from each
opening year on, and the outcome of the economy rises in steps. The column
``other_openings_same_economy_within_5y`` of the catalogue then lists the other
cases of the economy that open within five years, separated by semicolons. The
column ``panel_feasible`` is "no" only for the cases built to fail. The openings
of a shared economy form one episode in :func:`dtt.cases.build_episodes`, and
these episodes satisfy the feasibility rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cases import CASE_COLUMNS, CATEGORIES
from .panel import GROUPS, INDICATORS, wdi_filename

__all__ = ["SimWorld", "simulate_global_world", "ANALYSIS_LAST_YEAR", "CASES_FILE"]

#: Last year of the default estimation window; relative effects are measured up to it.
ANALYSIS_LAST_YEAR = 2019

#: Name of the cases catalogue written into the world directory.
CASES_FILE = "cases_catalogue.csv"

_METADATA_FILE = "country_metadata.csv"
_B0 = 0.30
_B1 = 0.85
_B2 = 0.0
_EFFECT_NOISE_SD = 0.20
_N_FACTORS = 3
_N_ACTIVE = 5
_OUTCOME_FLOOR = 0.1
_MIN_PRE_FEASIBLE = 5
_MIN_POST_FEASIBLE = 3
_CLUSTER_SIZES = (3, 2)
_CLUSTER_GAPS = ((2, 4), (3, 5))
_CLUSTER_MAX_SPAN = 8
_MIN_CLUSTER_CASES = 8
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


@dataclass(eq=False)
class SimWorld:
    """A synthetic test world written to disk, with its known truth.

    Attributes
    ----------
    directory : pathlib.Path
        Directory holding the raw files and the cases catalogue.
    paths : dict of str to pathlib.Path
        Every written file, keyed by file name, plus the keys ``raw_dir``,
        ``cases`` and ``metadata``.
    true_effects : pandas.DataFrame
        One row per case with ``case_id``, ``true_att_pp`` (true average effect on
        ``receipts_pct_gdp`` in percentage points of GDP) and ``true_att_rel_pct``
        (the effect as a percentage of the mean untreated outcome from the
        opening year to the end of the analysis window).
    drivers : pandas.DataFrame
        One row per case with ``case_id``, ``iso3``, ``opening_year``,
        ``investment_usd_bn_nominal``, ``capex_pct_gdp``, ``receipts_pct_gdp_pre``
        and ``effect_noise``; the true effect equals
        ``b0 + b1 * capex_pct_gdp + b2 * receipts_pct_gdp_pre + effect_noise``.
    coefficients : dict of str to float
        Generating coefficients ``b0``, ``b1``, ``b2`` and ``effect_noise_sd``.
    counterfactual : pandas.DataFrame
        Untreated outcome ``receipts_pct_gdp_cf`` for every economy and year.
    seed, n_econ, first_year, last_year : int
        Arguments of the generator.
    infeasible_case_ids : list of str
        Cases built to fail the default feasibility rules.
    cluster_economies : bool
        Whether several cases share an economy. ``true_att_pp`` is always the level
        shift of the case's own opening; :meth:`window_effect` gives the true mean
        shift of an economy over a range of years when it has several openings.
    """

    directory: Path
    paths: dict[str, Path]
    true_effects: pd.DataFrame
    drivers: pd.DataFrame
    coefficients: dict[str, float]
    counterfactual: pd.DataFrame
    seed: int
    n_econ: int
    first_year: int
    last_year: int
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
        """Return the true mean level shift of one economy over a range of years.

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


def _ar1(rng: np.random.Generator, n: int, length: int, rho: float, innovation_sd: float) -> np.ndarray:
    """Stationary AR(1) paths, an array of shape (n, length)."""
    eps = rng.normal(0.0, innovation_sd, size=(n, length))
    out = np.empty((n, length))
    out[:, 0] = eps[:, 0] / np.sqrt(1.0 - rho**2)
    for j in range(1, length):
        out[:, j] = rho * out[:, j - 1] + eps[:, j]
    return out


def _economy_codes(n_econ: int) -> np.ndarray:
    """Codes SA00, SA01, ... with a width that fits ``n_econ``."""
    width = max(2, len(str(n_econ - 1)))
    return np.array([f"SA{i:0{width}d}" for i in range(n_econ)])


def _draw_opening_years(
    rng: np.random.Generator,
    n_cases: int,
    first_year: int,
    last_year: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Opening years of the cases, which of them are built infeasible, and the partial-data case.

    Returns the opening years in chronological order, a boolean array marking the
    cases built to fail the feasibility rules, and the position of the feasible
    case whose first outcome years are set to missing (``None`` for tiny worlds).
    """
    low = first_year + _MIN_PRE_FEASIBLE
    high = min(last_year, ANALYSIS_LAST_YEAR) - _MIN_POST_FEASIBLE
    n_bad = min(2, n_cases // 8)
    n_good = n_cases - n_bad
    low_partial = min(high, first_year + _MIN_PRE_FEASIBLE + 3)
    good = rng.integers(low, high + 1, size=n_good)
    partial = None
    if n_good >= 4:
        partial = int(rng.integers(0, n_good))
        good[partial] = rng.integers(low_partial, high + 1)
    years = list(good)
    bad_years = [first_year + 2, last_year][:n_bad]
    years.extend(bad_years)
    infeasible = np.array([False] * n_good + [True] * n_bad)
    years_arr = np.asarray(years, dtype=int)
    partial_marker = np.zeros(n_cases, dtype=bool)
    if partial is not None:
        partial_marker[partial] = True
    order = np.argsort(years_arr, kind="stable")
    years_sorted = years_arr[order]
    infeasible_sorted = infeasible[order]
    partial_sorted = partial_marker[order]
    partial_pos = int(np.flatnonzero(partial_sorted)[0]) if partial_sorted.any() else None
    return years_sorted, infeasible_sorted, partial_pos


def _cluster_openings(
    rng: np.random.Generator,
    opening: np.ndarray,
    infeasible: np.ndarray,
    partial_pos: int | None,
    first_year: int,
    last_year: int,
) -> tuple[np.ndarray, np.ndarray, int | None, np.ndarray]:
    """Move groups of feasible cases into shared economies with openings a few years apart.

    Returns the opening years in chronological order, the flags of the cases built
    to be infeasible, the position of the partial-data case and, for every case,
    the number of its economy slot; cases with the same slot share an economy.
    """
    n = opening.size
    low = first_year + _MIN_PRE_FEASIBLE
    high = min(last_year, ANALYSIS_LAST_YEAR) - _MIN_POST_FEASIBLE
    movable = np.flatnonzero(~infeasible)
    if partial_pos is not None:
        movable = movable[movable != partial_pos]
    chosen = rng.choice(movable, size=sum(_CLUSTER_SIZES), replace=False)
    years = opening.copy()
    group = np.arange(n)
    start = 0
    for size, (gap_low, gap_high) in zip(_CLUSTER_SIZES, _CLUSTER_GAPS):
        members = chosen[start : start + size]
        start += size
        gaps = rng.integers(gap_low, gap_high + 1, size=size - 1)
        base = int(rng.integers(low, high - int(gaps.sum()) + 1))
        years[members] = base + np.concatenate([[0], np.cumsum(gaps)])
        group[members] = members[0]
    order = np.argsort(years, kind="stable")
    new_partial = None if partial_pos is None else int(np.flatnonzero(order == partial_pos)[0])
    return years[order], infeasible[order], new_partial, pd.factorize(group[order])[0]


def simulate_global_world(
    directory: str | Path,
    n_econ: int = 60,
    first_year: int = 1995,
    last_year: int = 2024,
    n_cases: int = 16,
    seed: int = 0,
    cluster_economies: bool = False,
) -> SimWorld:
    """Write a synthetic test world in the format of the raw files and return its truth.

    Parameters
    ----------
    directory : str or pathlib.Path
        Output directory, created if needed. It receives the twenty files
        ``wdi_{group}_{indicator}.csv``, ``country_metadata.csv`` and
        ``cases_catalogue.csv``.
    n_econ : int, default 60
        Number of economies, named ``SA00``, ``SA01`` and so on, split into the
        four groups G1 to G4 in consecutive blocks. At least ``n_cases + 20``.
    first_year, last_year : int
        Years of the panel.
    n_cases : int, default 16
        Number of cases, each in a different economy unless ``cluster_economies``
        is true.
    seed : int, default 0
        Seed of the random generator.
    cluster_economies : bool, default False
        If true, five of the cases are placed in two shared economies, three in
        one and two in the other, with openings a few years apart (see the module
        description).

    Returns
    -------
    SimWorld
        Paths of the written files, the true effects, the drivers of the effects,
        the generating coefficients and the untreated outcome. At least 14 of the
        16 default cases satisfy the default feasibility rules of
        :func:`dtt.cases.check_feasibility`. With shared economies the cases that
        are not built to fail are feasible as the episodes that
        :func:`dtt.cases.build_episodes` forms with its default merge gap; as
        single cases, those that another opening of their economy follows within
        the post-opening window fail the rule on other openings.

    Raises
    ------
    ValueError
        If ``n_cases`` is below one, ``n_econ`` is below ``n_cases + 20`` or the
        year range is too short for cases with five pre-opening years and three
        post-opening years before 2020. With ``cluster_economies`` also if
        ``n_cases`` is below eight or the year range leaves less than eight years
        between the earliest and the latest feasible opening year.
    """
    n_econ = int(n_econ)
    n_cases = int(n_cases)
    first_year = int(first_year)
    last_year = int(last_year)
    cluster_economies = bool(cluster_economies)
    if n_cases < 1:
        raise ValueError("n_cases must be at least 1")
    if n_econ < n_cases + 20:
        raise ValueError("n_econ must be at least n_cases + 20")
    if min(last_year, ANALYSIS_LAST_YEAR) - _MIN_POST_FEASIBLE < first_year + _MIN_PRE_FEASIBLE:
        raise ValueError("the year range is too short for cases with five pre-opening and three post-opening years")
    if cluster_economies:
        if n_cases < _MIN_CLUSTER_CASES:
            raise ValueError(f"cluster_economies needs at least {_MIN_CLUSTER_CASES} cases")
        if min(last_year, ANALYSIS_LAST_YEAR) - _MIN_POST_FEASIBLE - (first_year + _MIN_PRE_FEASIBLE) < _CLUSTER_MAX_SPAN:
            raise ValueError("the year range is too short for cluster_economies")
    n_shared = sum(_CLUSTER_SIZES) - len(_CLUSTER_SIZES) if cluster_economies else 0

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    codes = _economy_codes(n_econ)
    years = np.arange(first_year, last_year + 1)
    T = years.size
    t = np.arange(T, dtype=float)

    # ---- cases: economies, opening years, drivers ------------------------------------------
    case_pos = np.sort(rng.choice(n_econ, size=n_cases - n_shared, replace=False))
    is_case = np.zeros(n_econ, dtype=bool)
    is_case[case_pos] = True
    other_pos = np.flatnonzero(~is_case)
    opening, infeasible, partial_pos = _draw_opening_years(rng, n_cases, first_year, last_year)
    slot = np.arange(n_cases)
    if cluster_economies:
        opening, infeasible, partial_pos, slot = _cluster_openings(rng, opening, infeasible, partial_pos, first_year, last_year)
    case_econ = rng.permutation(case_pos)[slot]
    capex_target = np.exp(rng.uniform(np.log(0.1), np.log(3.0), size=n_cases))
    effect_noise = rng.normal(0.0, _EFFECT_NOISE_SD, size=n_cases)

    # ---- untreated outcome: three-factor model with AR(1) noise ----------------------------
    factors = np.empty((T, _N_FACTORS))
    factors[:, 0] = 0.03 * t + _ar1(rng, 1, T, 0.85, 0.10)[0]
    factors[:, 1] = _ar1(rng, 1, T, 0.70, 0.22)[0]
    factors[:, 2] = np.cumsum(rng.normal(0.0, 0.10, size=T))
    mu = rng.uniform(2.5, 7.5, size=n_econ)
    loadings = rng.normal(0.0, 0.55, size=(n_econ, _N_FACTORS))
    for econ in dict.fromkeys(int(e) for e in case_econ):
        active = rng.choice(other_pos, size=_N_ACTIVE, replace=False)
        omega = rng.dirichlet(np.full(_N_ACTIVE, 2.0))
        mu[econ] = omega @ mu[active]
        loadings[econ] = omega @ loadings[active]
    noise = _ar1(rng, n_econ, T, 0.5, 0.10)
    y_cf = np.maximum(mu[:, None] + loadings @ factors.T + noise, _OUTCOME_FLOOR)

    # ---- size variables ---------------------------------------------------------------------
    log_gdp0 = rng.uniform(np.log(5e9), np.log(3e12), size=n_econ)
    gdp = np.exp(
        log_gdp0[:, None]
        + rng.uniform(0.02, 0.09, size=n_econ)[:, None] * t[None, :]
        + _ar1(rng, n_econ, T, 0.6, 0.03)
    )
    log_pop0 = rng.uniform(np.log(4e5), np.log(1.5e8), size=n_econ)
    pop = np.exp(log_pop0[:, None] + rng.uniform(0.002, 0.02, size=n_econ)[:, None] * t[None, :])
    air_rate0 = np.exp(rng.uniform(np.log(0.03), np.log(2.0), size=n_econ))
    air_pax = pop * air_rate0[:, None] * np.exp(0.04 * t[None, :] + _ar1(rng, n_econ, T, 0.6, 0.05))
    per_arrival0 = np.exp(rng.uniform(np.log(350.0), np.log(2200.0), size=n_econ))
    per_arrival = per_arrival0[:, None] * np.exp(0.01 * t[None, :] + _ar1(rng, n_econ, T, 0.6, 0.04))
    gdp_int = np.round(gdp)

    # ---- effects and investment -------------------------------------------------------------
    col = {int(y): i for i, y in enumerate(years)}
    est_last = min(last_year, ANALYSIS_LAST_YEAR)
    rows_driver = []
    y_obs = y_cf.copy()
    for k in range(n_cases):
        e = int(case_econ[k])
        o = int(opening[k])
        inv = round(float(capex_target[k] / 100.0 * gdp_int[e, col[o - 1]] / 1.0e9), 4)
        capex = 100.0 * inv * 1.0e9 / gdp_int[e, col[o - 1]]
        window = [col[y] for y in range(o - 3, o) if y in col]
        pre_level = float(np.mean(y_obs[e, window]))
        tau = _B0 + _B1 * capex + _B2 * pre_level + float(effect_noise[k])
        y_obs[e, col[o]:] += tau
        rel_end = est_last if o <= est_last else last_year
        rel_cols = [col[y] for y in range(o, rel_end + 1)]
        rows_driver.append(
            {
                "case_id": f"C{k + 1:02d}",
                "iso3": str(codes[e]),
                "opening_year": o,
                "investment_usd_bn_nominal": inv,
                "capex_pct_gdp": capex,
                "receipts_pct_gdp_pre": pre_level,
                "effect_noise": float(effect_noise[k]),
                "true_att_pp": tau,
                "true_att_rel_pct": 100.0 * tau / float(np.mean(y_cf[e, rel_cols])),
            }
        )
    drivers_all = pd.DataFrame(rows_driver)

    # ---- raw indicators and missing values --------------------------------------------------
    receipts = y_obs / 100.0 * gdp_int
    arrivals = receipts / per_arrival
    data = {
        "receipts_usd": np.round(receipts),
        "arrivals": np.round(arrivals),
        "gdp_usd": gdp_int,
        "pop": np.round(pop),
        "air_pax": np.round(air_pax),
    }
    for name in ("arrivals", "air_pax"):
        data[name][rng.random(size=(n_econ, T)) < 0.03] = np.nan
    early = rng.choice(n_econ, size=max(1, n_econ // 10), replace=False)
    for e in early:
        data["arrivals"][e, : int(rng.integers(2, 6))] = np.nan
    n_late_air = min(3, max(1, n_cases // 4))
    earliest = np.argsort(opening, kind="stable")[: max(1, n_cases // 2)]
    for k in rng.choice(earliest, size=n_late_air, replace=False):
        data["air_pax"][case_econ[k], : min(12, T - 1)] = np.nan
    n_gap = max(0, min(n_econ // 15, n_econ - n_cases - 18))
    gap_econ = rng.choice(other_pos, size=n_gap, replace=False) if n_gap else np.array([], dtype=int)
    gap_window = max(1, min(last_year, ANALYSIS_LAST_YEAR) - first_year + 1)
    for e in gap_econ:
        n_missing = int(rng.integers(1, 4))
        data["receipts_usd"][e, rng.choice(gap_window, size=n_missing, replace=False)] = np.nan
    if partial_pos is not None:
        data["receipts_usd"][int(case_econ[partial_pos]), :2] = np.nan

    # ---- write the files --------------------------------------------------------------------
    paths: dict[str, Path] = {}
    for group, members in zip(GROUPS, np.array_split(np.arange(n_econ), len(GROUPS))):
        for code, variable in INDICATORS.items():
            flat = np.round(data[variable][members]).reshape(-1)
            frame = pd.DataFrame(
                {
                    "iso3": np.repeat(codes[members], T),
                    "year": np.tile(years, members.size),
                    "value": pd.Series(flat).astype("Int64"),
                }
            )
            target = out_dir / wdi_filename(group, code)
            frame.to_csv(target, index=False)
            paths[target.name] = target

    metadata = pd.DataFrame(
        {
            "iso3": codes,
            "name": [f"Simulated economy {c}" for c in codes],
            "region": [_REGIONS[i % len(_REGIONS)] for i in range(n_econ)],
            "income_level": [_INCOME_LEVELS[(3 * i + 1) % len(_INCOME_LEVELS)] for i in range(n_econ)],
            "capital": [f"Capital {c}" for c in codes],
            "longitude": np.round(rng.uniform(-170.0, 170.0, size=n_econ), 3),
            "latitude": np.round(rng.uniform(-45.0, 60.0, size=n_econ), 3),
        }
    )
    metadata_path = out_dir / _METADATA_FILE
    metadata.to_csv(metadata_path, index=False)
    paths[metadata_path.name] = metadata_path

    names = dict(zip(metadata["iso3"], metadata["name"]))
    overlap: Any = 0
    if cluster_economies:
        overlap = [
            ";".join(
                str(other_id)
                for other_id, other_iso3, other_year in zip(drivers_all["case_id"], drivers_all["iso3"], drivers_all["opening_year"])
                if other_id != case_id and other_iso3 == iso3 and abs(int(other_year) - int(year)) <= 5
            )
            for case_id, iso3, year in zip(drivers_all["case_id"], drivers_all["iso3"], drivers_all["opening_year"])
        ]
    catalogue = pd.DataFrame(
        {
            "case_id": drivers_all["case_id"],
            "attraction": [f"Simulated attraction {k + 1:02d}" for k in range(n_cases)],
            "operator": [f"Simulated operator {'ABCDE'[k % 5]}" for k in range(n_cases)],
            "category": rng.choice(np.array(CATEGORIES), size=n_cases),
            "iso3": drivers_all["iso3"],
            "economy": [names[c] for c in drivers_all["iso3"]],
            "location": [f"City of {c}" for c in drivers_all["iso3"]],
            "opening_date": [
                f"{int(o)}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}" for o in drivers_all["opening_year"]
            ],
            "opening_year": drivers_all["opening_year"],
            "investment_usd_bn_nominal": drivers_all["investment_usd_bn_nominal"],
            "investment_year_basis": drivers_all["opening_year"] - 1,
            "investment_source_note": "Simulated figure",
            "first_year_attendance_m": np.round(rng.uniform(1.5, 12.0, size=n_cases), 2),
            "concurrent_confounds": "none",
            "in_wdi_panel": "yes",
            "panel_feasible": np.where(infeasible, "no", "yes"),
            "other_openings_same_economy_within_5y": overlap,
            "evidence_quality": "high",
            "source_url_1": [f"https://example.org/simulated/{c}" for c in drivers_all["case_id"]],
            "source_url_2": "",
            "notes": "Simulated case",
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
            "year": np.tile(years, n_econ),
            "receipts_pct_gdp_cf": y_cf.reshape(-1),
        }
    )
    return SimWorld(
        directory=out_dir,
        paths=paths,
        true_effects=drivers_all[["case_id", "true_att_pp", "true_att_rel_pct"]].copy(),
        drivers=drivers_all[
            ["case_id", "iso3", "opening_year", "investment_usd_bn_nominal", "capex_pct_gdp", "receipts_pct_gdp_pre", "effect_noise"]
        ].copy(),
        coefficients={"b0": _B0, "b1": _B1, "b2": _B2, "effect_noise_sd": _EFFECT_NOISE_SD},
        counterfactual=counterfactual,
        seed=int(seed),
        n_econ=n_econ,
        first_year=first_year,
        last_year=last_year,
        infeasible_case_ids=[str(c) for c in drivers_all.loc[infeasible, "case_id"]],
        cluster_economies=cluster_economies,
    )
